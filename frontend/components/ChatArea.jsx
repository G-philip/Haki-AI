'use client'

import { useState } from 'react'

const UNAVAILABLE_TEXT = "Haki AI is currently unavailable. Please try again shortly."

export default function ChatArea({ messages, documentText, docType, collectedFields, downloadDoc, chatEndRef, stage, newChat, thinking, loading, onRetry, onEdit }) {
  const safeMessages = Array.isArray(messages) ? messages : [];
  const [editingIndex, setEditingIndex] = useState(null)
  const [editText, setEditText] = useState('')

  const isUnavailableMessage = (content) =>
    typeof content === 'string' && content.trim() === UNAVAILABLE_TEXT

  // IMPORTANT: visibleMessages is a FILTERED view for display only. The
  // indices used by startEdit/onEdit/onRetry must refer to positions in
  // the REAL `messages` array (what page.jsx actually holds and sends to
  // the backend), not positions within this filtered array — those two
  // only coincide if nothing was ever filtered out. If even one message
  // earlier in history was filtered (e.g. a stale "unavailable" message
  // from before that stopped being persisted), every index after it
  // would be off by however many messages were filtered, causing edits/
  // retries to operate on the wrong message — this was the actual cause
  // of "edit saves as a new question instead of editing": the wrong
  // index got truncated/replaced on the backend.
  //
  // Fix: keep each visible message paired with its REAL index from the
  // original array, and use that real index everywhere, not the
  // position from .map() over the filtered array.
  const visibleMessages = safeMessages
    .map((m, realIndex) => ({ ...m, __realIndex: realIndex }))
    .filter(m => !isUnavailableMessage(m.content))

  const startEdit = (index, currentText) => {
    setEditingIndex(index)
    setEditText(currentText)
  }

  const cancelEdit = () => {
    setEditingIndex(null)
    setEditText('')
  }

  const submitEdit = (index) => {
    const text = editText.trim()
    setEditingIndex(null)
    setEditText('')
    if (text && onEdit) {
      onEdit(index, text)
    }
  }

  return (
    <>
      <style>{`
        .chat-scroll::-webkit-scrollbar { width: 1px; }
        .chat-scroll::-webkit-scrollbar-track { background: transparent; }
        .chat-scroll::-webkit-scrollbar-thumb {
          background-color: rgba(0, 0, 0, 0.2);
          border-radius: 9999px;
        }
        .chat-scroll::-webkit-scrollbar-thumb:hover {
          background-color: rgba(0, 0, 0, 0.32);
        }
        .chat-scroll {
          scrollbar-width: thin;
          scrollbar-color: rgba(0, 0, 0, 0.2) transparent;
        }
      `}</style>

      {/*
        No longer owns its own overflow/scroll wrapper — that moved up
        to page.js, which now wraps Navbar + this content in ONE shared
        scrolling container (see page.js). That's what "same container
        as chat area and chat input" and "float"/full-height scrollbar
        actually required: Navbar needs to be a true DOM sibling INSIDE
        the same scrolling ancestor for its `sticky` positioning to
        pin it while scrolling, and for the scrollbar itself to span
        the full height (navbar included) rather than a shorter box
        confined to below a separately-positioned navbar.
      */}
      <div className="max-w-[720px] mx-auto px-6 py-8 pb-4">
          {visibleMessages.map((msg) => {
            const realIndex = msg.__realIndex
            const isUser = msg.role === 'user'
            const isEditing = editingIndex === realIndex

            if (isUser) {
              return (
                <div key={realIndex} className="group flex flex-col items-end my-3">
                  {isEditing ? (
                    <div className="w-[90%] flex flex-col gap-2">
                      <textarea
                        value={editText}
                        onChange={(e) => setEditText(e.target.value)}
                        autoFocus
                        rows={Math.min(6, Math.max(2, Math.ceil(editText.length / 60)))}
                        className="w-full border border-teal-300 rounded-2xl px-5 py-3.5 text-[14px] leading-relaxed bg-white outline-none focus:border-teal-500 resize-none"
                      />
                      <div className="flex justify-end gap-2">
                        <button
                          onClick={cancelEdit}
                          className="px-3 py-1.5 text-[12px] font-medium text-gray-500 hover:bg-gray-100 rounded-lg transition"
                        >
                          Cancel
                        </button>
                        <button
                          onClick={() => submitEdit(realIndex)}
                          className="px-3 py-1.5 text-[12px] font-medium text-white bg-teal-600 hover:bg-teal-700 rounded-lg transition"
                        >
                          Save & resend
                        </button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="bg-[#0d9488] text-white px-5 py-3.5 rounded-2xl rounded-br-md max-w-[90%] text-[14px] leading-relaxed whitespace-pre-wrap break-words">
                        {msg.content}
                      </div>

                      {/* Edit + retry icons — every user message, hidden
                          until hover via group/group-hover. Disabled
                          while a request is in flight: previously these
                          stayed clickable-looking during loading, so a
                          click could land in the window where
                          sendMessage's own `loading` guard silently
                          no-ops it — no error, no toast, no animation,
                          just nothing happening. Disabling them here
                          prevents that click from landing at all. */}
                      <div className="flex items-center gap-1.5 mt-1.5 opacity-0 group-hover:opacity-100 transition">
                        <button
                          onClick={() => !loading && startEdit(realIndex, msg.content)}
                          disabled={loading}
                          title={loading ? 'Please wait…' : 'Edit and resend'}
                          className="w-7 h-7 flex items-center justify-center rounded-full border border-gray-200 dark:border-gray-600 text-gray-400 hover:text-teal-600 hover:border-teal-300 hover:bg-teal-50 dark:hover:bg-teal-900 transition disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-gray-400 disabled:hover:border-gray-200"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                          </svg>
                        </button>
                        <button
                          onClick={() => !loading && onRetry && onRetry(realIndex)}
                          disabled={loading}
                          title={loading ? 'Please wait…' : 'Resend'}
                          className="w-7 h-7 flex items-center justify-center rounded-full border border-gray-200 dark:border-gray-600 text-gray-400 hover:text-teal-600 hover:border-teal-300 hover:bg-teal-50 dark:hover:bg-teal-900 transition disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-gray-400 disabled:hover:border-gray-200"
                        >
                          <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="23 4 23 10 17 10"></polyline>
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path>
                          </svg>
                        </button>
                      </div>
                    </>
                  )}
                </div>
              )
            }

            return (
              <div key={realIndex} className="my-2">
                <div className="text-gray-601 font-normal py-2 text-[15px] leading-relaxed whitespace-pre-wrap break-words">
                  {msg.content}
                </div>
              </div>
            )
          })}

          {thinking && (
            <div className="py-2 my-2">
              <div className="flex items-center gap-2">
                <span className="text-[13px] text-teal-600">Thinking</span>
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 bg-teal-700 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-2 h-2 bg-teal-700 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-2 h-2 bg-teal-700 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
                {/* <span className="text-[13px] text-gray-500">Thinking</span> */}
              </div>
            </div>
          )}

          {documentText && (
            <div className="mt-6">
              <h3 className="text-base font-normal text-gray-700 mb-3">Generated Document</h3>
              <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6 document-box text-gray-900 dark:text-gray-100">
                {documentText}
              </div>

              <div className="flex gap-3 mt-4">
                <button onClick={() => downloadDoc('docx')} className="px-4 py-2 border border-gray-200 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-gray-300 transition">
                  Download DOCX
                </button>
                <button onClick={() => downloadDoc('pdf')} className="px-4 py-2 border border-gray-200 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 hover:border-gray-300 transition">
                  Download PDF
                </button>
              </div>
            </div>
          )}

          {stage === 'done' && (
            <div className="mt-6 text-center">
              <button onClick={newChat} className="px-5 py-2.5 border border-gray-200 dark:border-gray-600 rounded-lg text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition">
                Start new document
              </button>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>
    </>
  )
}
