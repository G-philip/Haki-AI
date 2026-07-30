'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import Sidebar from '../components/Sidebar'
import ChatArea from '../components/ChatArea'
import ChatInput from '../components/ChatInput'
import Toast from '../components/Toast'
import Navbar from '../components/Navbar'
import MockLogin from '../components/MockLogin'
import ScrollDebugOverlay from '../components/ScrollDebugOverlay'

const API = 'http://localhost:8000'
const UNAVAILABLE_TEXT = "Haki AI is currently unavailable. Please try again shortly."

// Q&A mode is now auto-selected (see ChatInput's sessionId effect), so
// the backend's welcome message telling the user to "pick an option"
// is no longer accurate — this strips that sentence out client-side.
// The real fix belongs in the backend's welcome_message copy itself;
// this is a stopgap until that's updated there.
const MODE_INSTRUCTION_RE = /\s*Pick an option below[^.]*\.\s*/i
const cleanWelcomeMessage = (text) => (text || '').replace(MODE_INSTRUCTION_RE, ' ').trim()

export default function Home() {
  const [userType, setUserType] = useState(null)
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [stage, setStage] = useState('greeting')
  const [docType, setDocType] = useState(null)
  const [collectedFields, setCollectedFields] = useState({})
  const [documentText, setDocumentText] = useState(null)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [history, setHistory] = useState({})
  const [loading, setLoading] = useState(false)
  const [thinking, setThinking] = useState(false)
  const [viewingSavedChat, setViewingSavedChat] = useState(null)
  const [needsMode, setNeedsMode] = useState(true)
  const [mode, setMode] = useState(null)
  const [toastMessage, setToastMessage] = useState(null)
  const [toastVariant, setToastVariant] = useState('error')

  const showErrorToast = (msg) => {
    setToastVariant('error')
    setToastMessage(msg)
  }
  const showInfoToast = (msg) => {
    setToastVariant('info')
    setToastMessage(msg)
  }
  const showSuccessToast = (msg) => {
    setToastVariant('success')
    setToastMessage(msg)
  }
  const chatEndRef = useRef(null)
  const chatInputWrapperRef = useRef(null)
  // Bumped by the ResizeObserver below whenever ChatInput's rendered
  // height changes (textarea growing as the user types, the mode
  // dropdown opening/closing). This is the actual fix for the
  // "scrollbar doesn't reach the bottom, text gets cut off" bug: the
  // previous scroll-to-bottom effect only re-ran on `messages`/
  // `thinking` changes, so if ChatInput grew AFTER that scroll already
  // ran (e.g. the user types a long message and the textarea expands),
  // the last chat message could end up sitting behind the now-taller
  // input with no further scroll triggered.
  const [inputHeightTick, setInputHeightTick] = useState(0)

  const refreshHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/history`)
      if (!res.ok) throw new Error(`History fetch failed: ${res.status}`)
      const data = await res.json()
      setHistory(data.chats || {})
    } catch (err) {
      console.error('History error:', err)
      showErrorToast("Couldn't load chat history. Please try again.")
    }
  }, [])

  const sessionInitedForUserTypeRef = useRef(null)

  useEffect(() => {
    if (!userType) return  // wait for mock login before starting a session
    // Guards against this effect firing twice for the same userType --
    // e.g. React Strict Mode's deliberate double-invoke of effects in
    // development, which was creating two sessions back to back (visible
    // as two NEW_SESSION log lines seconds apart). Same pattern
    // ChatInput.jsx already uses for its own once-per-session effect.
    if (sessionInitedForUserTypeRef.current === userType) return
    sessionInitedForUserTypeRef.current = userType

    async function init() {
      // new-session and history are fully independent — history isn't
      // scoped to a session_id at all. refreshHistory() is fired but
      // NOT awaited here: it has its own try/catch + error toast, and
      // gates its own state (the sidebar list) whenever it resolves.
      // Session readiness (sessionId/messages/needsMode, which is what
      // the chat input's disabled state depends on) only waits on the
      // new-session call itself, so a slow /api/history response can no
      // longer hold up being able to type a question — the two used to
      // be joined by Promise.all, which meant chat readiness waited for
      // whichever of the two was slower even though it never actually
      // depended on history at all.
      refreshHistory()

      try {
        const res = await fetch(`${API}/api/new-session`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_type: userType }),
        })
        const data = await res.json()
        setSessionId(data.session_id)
        setMessages([{ role: 'assistant', content: cleanWelcomeMessage(data.welcome_message) }])
        // 'normal' users have exactly one allowed mode, so there's no
        // real choice for them to make — skip the dropdown entirely.
        setNeedsMode(userType !== 'normal')
      } catch (err) {
        console.error('Init error:', err)
        showErrorToast("Couldn't start a session. Please refresh the page.")
      }
    }
    init()
  }, [userType, refreshHistory])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, thinking, inputHeightTick])

  // Watches ChatInput's actual rendered height and triggers a re-scroll
  // whenever it changes, so growing/shrinking the input never leaves the
  // last message hidden behind it.
  useEffect(() => {
    const el = chatInputWrapperRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => {
      setInputHeightTick(t => t + 1)
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  // sendMessage handles three cases:
  // 1. Normal send — push a user bubble, persist both sides as usual.
  // 2. Mode selection (needsMode is true) — same as normal send; the
  //    backend's MessageHandler._handle_mode_selection reads the literal
  //    'question'/'document' string. Once it succeeds we clear needsMode.
  // 3. Retry (isRetry: true) — do NOT push a new user bubble (the
  //    original is still there), and tell the backend via is_retry so a
  //    second failure isn't persisted — see main.py's transient_failure.
  const sendMessage = async (message, { isRetry = false } = {}) => {
    if (!sessionId || loading) return

    // Mode is selected via ChatInput's dropdown sending the literal
    // string 'question' or 'document' (see ChatInput.jsx's
    // handleModeChange). Since the dropdown is now always visible for
    // lawyer/admin accounts (not just before the first choice), this is
    // detected by the message content itself, not by a "haven't chosen
    // yet" state flag.
    const isModeSelection = !isRetry && (message === 'question' || message === 'document')

    // Mode-switch messages are never shown as chat bubbles at all — not
    // even briefly. The backend excludes them from persisted history
    // (see main.py's mode_notice handling), so optimistically adding one
    // here would cause a visible flash (bubble appears, then disappears
    // once the backend's response overwrites local state with history
    // that doesn't include it).
    if (!isRetry && !isModeSelection) {
      setMessages(prev => [...prev, { role: 'user', content: message }])
    }
    setThinking(true)
    setLoading(true)

    try {
      const res = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message, is_retry: isRetry }),
      })
      const data = await res.json()

      if (data.transient_failure) {
        // Retry failed again — nothing was persisted on the backend.
        // Show a toast, leave the chat history exactly as it was
        // (still showing the original "unavailable" bubble with its
        // retry icon, so the user can try again).
        setThinking(false)
        setLoading(false)
        showErrorToast(UNAVAILABLE_TEXT)
        return
      }

      if (data.mode_notice) {
        // Mode-switch confirmation/denial/nudge — the backend already
        // excluded both the user's literal 'question'/'document'
        // message and this text from persisted history (see main.py's
        // chat() route), so `data.messages` here reflects the chat
        // WITHOUT that turn. Show the notice as a brief info toast
        // instead of a chat bubble.
        setMessages(data.messages || [])
        setStage(data.stage || 'greeting')
        setDocType(data.doc_type || null)
        setCollectedFields(data.collected_fields || {})
        setDocumentText(data.document_text || null)
        setThinking(false)
        setLoading(false)
        showInfoToast(data.mode_notice)
        if (isModeSelection) {
          setMode(message === 'question' ? 'qa' : message === 'document' ? 'document' : null)
        }
        return
      }

      setMessages(data.messages || [])
      setStage(data.stage || 'greeting')
      setDocType(data.doc_type || null)
      setCollectedFields(data.collected_fields || {})
      setDocumentText(data.document_text || null)
      setThinking(false)

      // NOTE: mode is now only updated above, in the mode_notice branch
      // — a successful mode switch always comes back as a mode_notice,
      // so this path no longer needs to check isModeSelection itself.

      if (data.has_document) {
        await refreshHistory()
      }
    } catch (err) {
      console.error('Chat error:', err)
      // All user-visible errors now go through the toast (which stays
      // open until manually dismissed — see Toast.jsx), never as a
      // saved chat bubble, for consistency with how transient_failure
      // is handled above.
      showErrorToast("Couldn't reach the server. Please try again.")
      setThinking(false)
    }
    setLoading(false)
  }

  // Retry: find the user message that immediately precedes this assistant
  // reply and re-send it with is_retry=true.
  // Resends the user message at `index` directly. (Previously this
  // walked BACKWARD from an assistant message's index to find the
  // preceding user message — that made sense when the retry icon lived
  // on the assistant's failed reply. Now the icon lives directly on the
  // user's own bubble, so the index passed in already IS the user
  // message to resend; walking backward from it would incorrectly
  // search messages before it.)
  const retryMessage = async (userMessageIndex) => {
    const target = messages[userMessageIndex]
    if (!target || target.role !== 'user') return
    await sendMessage(target.content, { isRetry: true })
  }

  // Sends an edited message. Mirrors sendMessage's response handling
  // (transient_failure -> toast, otherwise sync full state) but posts
  // is_edit/edit_index instead of is_retry, and never pushes a new user
  // bubble locally (editMessage already updated it above).
  const sendEdit = async (newText, index) => {
    if (!sessionId || loading) return
    setThinking(true)
    setLoading(true)

    try {
      const res = await fetch(`${API}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: newText,
          is_edit: true,
          edit_index: index,
        }),
      })
      const data = await res.json()

      if (data.transient_failure) {
        setThinking(false)
        setLoading(false)
        showErrorToast(UNAVAILABLE_TEXT)
        return
      }

      if (data.mode_notice) {
        // Rare edge case: editing a message resulted in a literal
        // 'question'/'document' mode-switch. Same treatment as in
        // sendMessage — info toast, not a chat bubble.
        setMessages(data.messages || [])
        setStage(data.stage || 'greeting')
        setDocType(data.doc_type || null)
        setCollectedFields(data.collected_fields || {})
        setDocumentText(data.document_text || null)
        setThinking(false)
        setLoading(false)
        showInfoToast(data.mode_notice)
        if (newText === 'question' || newText === 'document') {
          setMode(newText === 'question' ? 'qa' : 'document')
        }
        return
      }

      setMessages(data.messages || [])
      setStage(data.stage || 'greeting')
      setDocType(data.doc_type || null)
      setCollectedFields(data.collected_fields || {})
      setDocumentText(data.document_text || null)
      setThinking(false)

      if (data.has_document) {
        await refreshHistory()
      }
    } catch (err) {
      console.error('Edit error:', err)
      showErrorToast("Couldn't reach the server. Please try again.")
      setThinking(false)
    }
    setLoading(false)
  }

  // Edit: replace the user message's text at `index` locally, then send
  // it to the backend with is_edit+edit_index so ChatManager truncates
  // its OWN stored history at that point and replaces it with the
  // revised text — is_retry's semantics ("the old text is already
  // correctly stored, just don't re-add it") don't apply here, since the
  // backend's stored text is the OLD text, not the new one.
  const editMessage = async (index, newText) => {
    if (loading) return
    setMessages(prev => {
      const next = [...prev]
      next[index] = { ...next[index], content: newText }
      return next.slice(0, index + 1)
    })
    await sendEdit(newText, index)
  }

  const newChat = async () => {
    try {
      const res = await fetch(`${API}/api/new-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_type: userType }),
      })
      const data = await res.json()
      setSessionId(data.session_id)
      setMessages([{ role: 'assistant', content: cleanWelcomeMessage(data.welcome_message) }])
      setStage('greeting')
      setDocType(null)
      setCollectedFields({})
      setDocumentText(null)
      setViewingSavedChat(null)
      setThinking(false)
      setNeedsMode(userType !== 'normal')
      setMode(null)
      await refreshHistory()
    } catch (err) {
      console.error('New chat error:', err)
      showErrorToast("Couldn't start a new chat. Please try again.")
    }
  }

  const loadSavedChat = async (chatId) => {
    try {
      const newSessionRes = await fetch(`${API}/api/new-session`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_type: userType }),
      })
      const newSessionData = await newSessionRes.json()
      const newSessionId = newSessionData.session_id

      await fetch(`${API}/api/load-chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: newSessionId, chat_id: chatId }),
      })

      const chatRes = await fetch(`${API}/api/history/${chatId}`)
      const chatData = await chatRes.json()

      setSessionId(newSessionId)
      setMessages(chatData.messages || [])
      setDocType(chatData.doc_type || null)
      setCollectedFields(chatData.collected_fields || {})
      setDocumentText(null)
      setStage(chatData.stage || 'greeting')
      setViewingSavedChat(null)
      setThinking(false)
      setNeedsMode(userType !== 'normal')

      await refreshHistory()
    } catch (err) {
      console.error('Load chat error:', err)
      showErrorToast("Couldn't load that conversation. Please try again.")
    }
  }

  // Fixed: this used to read `history[chatId]` directly and pull
  // `.messages` off of it. But `history` is populated by
  // refreshHistory() -> GET /api/history, which (going by what Sidebar
  // actually reads off each entry — just `title` and `date`) only
  // returns lightweight summaries, not full message arrays. So
  // `chatData.messages` was always undefined, silently falling back to
  // `[]`, and the chat area rendered empty even though the click was
  // registering correctly. Now it fetches the same per-chat detail
  // endpoint loadSavedChat already uses, just without spinning up a new
  // live session (viewing a saved chat shouldn't create one).
  const viewSavedChat = async (chatId) => {
    try {
      const res = await fetch(`${API}/api/history/${chatId}`)
      if (!res.ok) throw new Error(`Failed to load chat ${chatId}: ${res.status}`)
      const chatData = await res.json()

      setViewingSavedChat(chatId)
      setMessages(chatData.messages || [])
      setDocType(chatData.doc_type || null)
      setCollectedFields(chatData.collected_fields || {})
      setDocumentText(null)
      setStage('viewing')
      setSessionId(null)
    } catch (err) {
      console.error('View chat error:', err)
      showErrorToast("Couldn't load that conversation. Please try again.")
    }
  }

  const downloadDoc = async (format) => {
    if (!sessionId) return
    try {
      const res = await fetch(`${API}/api/download`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, format }),
      })
      if (res.ok) {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `document.${format}`
        a.click()
        URL.revokeObjectURL(url)
      } else {
        // NOTE: previously this branch did nothing at all on failure —
        // no toast, no log, the download just silently didn't happen.
        console.error('Download failed with status:', res.status)
        showErrorToast(`Couldn't download the ${format.toUpperCase()} file. Please try again.`)
      }
    } catch (err) {
      console.error('Download error:', err)
      showErrorToast(`Couldn't download the ${format.toUpperCase()} file. Please try again.`)
    }
  }

  return (
    <div className="flex h-screen bg-white">
      {!userType ? (
        <MockLogin onLogin={setUserType} />
      ) : (
        <>
          <Sidebar
            collapsed={sidebarCollapsed}
            setCollapsed={setSidebarCollapsed}
            history={history}
            newChat={newChat}
            loadSavedChat={loadSavedChat}
            viewSavedChat={viewSavedChat}
            refreshHistory={refreshHistory}
            showSuccessToast={showSuccessToast}
          />

          <div className="flex-1 flex flex-col min-w-0 min-h-0">
            {/* Navbar + ChatArea now share ONE scroll container (min-h-0
                is required here, not decorative — without it, a flex
                child's default min-height:auto can stop it from actually
                shrinking to fit the available space, which is the classic
                cause of a scrollbox reporting a shorter height than the
                space it visually has). ChatInput stays OUTSIDE this div,
                as a separate flex-shrink-0 sibling below it, so it's
                pinned at the bottom and never scrolls away. */}
            <div className="chat-scroll flex-1 min-h-0 overflow-y-auto">
              <Navbar mode={mode} userType={userType} userName="Guest" />

              <ChatArea
                messages={messages}
                documentText={documentText}
                docType={docType}
                collectedFields={collectedFields}
                downloadDoc={downloadDoc}
                chatEndRef={chatEndRef}
                stage={stage}
                newChat={newChat}
                thinking={thinking}
                loading={loading}
                onRetry={retryMessage}
                onEdit={editMessage}
              />
            </div>

            <div ref={chatInputWrapperRef}>
              <ChatInput
                onSend={sendMessage}
                disabled={loading}
                needsMode={needsMode}
                sessionId={sessionId}
                mode={mode}
              />
            </div>
          </div>

          <Toast message={toastMessage} variant={toastVariant} onDismiss={() => setToastMessage(null)} />
        </>
      )}
      {/* <ScrollDebugOverlay /> */}
    </div>
  )
}
