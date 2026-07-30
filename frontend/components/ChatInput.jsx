'use client'

import { useState, useEffect, useRef } from 'react'

export default function ChatInput({ onSend, disabled, needsMode, sessionId, mode }) {
  const [input, setInput] = useState('')
  const [showModeDropdown, setShowModeDropdown] = useState(false)
  const inputRef = useRef(null)
  const modeDropdownRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    if (!disabled) {
      inputRef.current?.focus()
    }
  }, [disabled])

  useEffect(() => {
    function handleClickOutside(e) {
      if (modeDropdownRef.current && !modeDropdownRef.current.contains(e.target)) {
        setShowModeDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Auto-select Q&A mode for every new session instead of making the
  // user pick it from the dropdown each time. Tracks which sessionId it
  // already auto-selected for in a ref (not state) so this fires
  // exactly once per session — the effect's dependency array includes
  // `onSend`, whose reference changes on every parent re-render, but the
  // ref guard (checked synchronously, set before calling onSend) means
  // that alone can't trigger a duplicate send. It re-fires only when
  // `sessionId` itself changes to a new, not-yet-handled value — which
  // happens on initial load and on every "New chat" click, but not on
  // ordinary re-renders, and not while viewing a saved chat (sessionId
  // is null then). The dropdown stays available so document mode can
  // still be chosen manually within a session.
  const autoSelectedSessionRef = useRef(null)
  useEffect(() => {
    if (needsMode && sessionId && autoSelectedSessionRef.current !== sessionId) {
      autoSelectedSessionRef.current = sessionId
      onSend('question')
    }
  }, [needsMode, sessionId, onSend])

  const handleSubmit = (e) => {
    e.preventDefault()
    if (!input.trim() || disabled) return
    onSend(input.trim())
    setInput('')
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }

  const handleModeSelect = (value) => {
    onSend(value)
    setShowModeDropdown(false)
  }

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      onSend(`Uploaded: ${file.name}`)
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const getRows = () => {
    const chars = input.length
    if (chars < 60) return 1
    if (chars < 120) return 2
    if (chars < 200) return 3
    if (chars < 300) return 4
    return 5
  }

  // Shown on the dropdown trigger itself so the current mode is always
  // visible, not just at selection time. Falls back to a neutral label
  // for the brief window before the auto-select effect's response comes
  // back and `mode` is still null.
  const MODE_LABELS = {
    qa: 'Veritas',
    document: 'Dictum',
  }
  const modeLabel = MODE_LABELS[mode] || 'Choose mode'

  return (
    // NOTE: previously `fixed bottom-0 left-0 right-0` — that takes the
    // element OUT of normal layout flow and anchors it to the full
    // browser viewport, ignoring the actual parent column it's rendered
    // in (the flex column to the right of the sidebar). That's why
    // matching ChatArea's max-w/px values alone didn't fix the
    // alignment: both were centering correctly, but within two
    // DIFFERENT reference widths — ChatArea within (viewport - sidebar),
    // ChatInput within the full viewport.
    //
    // This is now a normal flow child of the same flex column ChatArea
    // lives in (see page.jsx: both are direct children of the same
    // `flex-1 flex flex-col` div). Since ChatArea has `flex-1` and
    // expands to fill available height, plain flow already pushes this
    // element to the bottom of that column with no special positioning
    // needed — and it now shares ChatArea's exact available width
    // automatically, with no sidebar-width special-casing required.
    // max-w-[720px] below MUST stay in sync with ChatArea's container.
    <div className="w-full bg-white pt-2 flex-shrink-0">
      <div className="max-w-[720px] mx-auto px-6">
        <div className="pb-4">
          <form onSubmit={handleSubmit}>
            <div className="relative border-2 border-teal-500 rounded-2xl bg-white shadow-lg focus-within:shadow-xl transition">

              {/* Textarea — needsMode controls ONLY whether the mode
                  dropdown control renders (see below); it must never
                  disable typing itself. Previously `disabled={disabled
                  || needsMode}` blocked the whole textarea for the
                  entire session on any account where the dropdown is
                  permanently visible (lawyer/admin), which is exactly
                  the "chat input still disabled" bug. */}
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={disabled ? 'Document generated. Start a new document.' : 'Describe your legal need...'}
                disabled={disabled}
                rows={getRows()}
                className="w-full px-4 pt-3 pb-12 text-[14px] bg-transparent border-none outline-none text-teal-600 placeholder-teal-600 resize-none min-h-[56px]"
                autoFocus
              />

              {/* Bottom row: upload (left) | mode + send (right) */}
              <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between">

                {/* Left: Upload document button */}
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  className="w-8 h-8 flex items-center justify-center rounded-full border bg-teal-500 text-white/100 hover:bg-gray-100 hover:border-gray-300 hover:text-gray-700 transition"
                  title="Upload document"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="5" x2="12" y2="19"></line>
                    <line x1="5" y1="12" x2="19" y2="12"></line>
                  </svg>
                </button>

                {/* Hidden file input */}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.doc,.docx,.txt"
                  onChange={handleFileSelect}
                  className="hidden"
                />

                {/* Right: Mode dropdown + Send */}
                <div className="flex items-center gap-2">
                  {needsMode && (
                    <div ref={modeDropdownRef} className="relative">
                      <button
                        type="button"
                        onClick={() => setShowModeDropdown(!showModeDropdown)}
                        className="text-[12px] font-medium text-gray-500 px-2 py-1 rounded-lg hover:bg-gray-100 hover:text-gray-700 transition whitespace-nowrap"
                      >
                        {modeLabel} ▾
                      </button>

                      {showModeDropdown && (
                        <div className="absolute bottom-full right-0 mb-2 w-44 bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden z-20">
                          <button
                            type="button"
                            onClick={() => handleModeSelect('question')}
                            className="w-full text-left px-4 py-2.5 text-[13px] text-gray-700 hover:bg-gray-50 transition"
                          >
                            Veritas
                            {/* 💬 Ask a question */}
                          </button>
                          <button
                            type="button"
                            onClick={() => handleModeSelect('document')}
                            className="w-full text-left px-4 py-2.5 text-[13px] text-gray-700 hover:bg-gray-50 transition"
                          >
                            Dictum
                            {/* 📄 Draft a document */}
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  <button
                    type="submit"
                    disabled={disabled || !input.trim()}
                    className="w-9 h-9 flex items-center justify-center bg-teal-600 text-white rounded-xl hover:bg-teal-300 disabled:opacity-40 flex-shrink-0 transition"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <line x1="12" y1="19" x2="12" y2="5"></line>
                      <polyline points="5 12 12 5 19 12"></polyline>
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          </form>

          {/* Disclaimer */}
          <p className="text-center text-[11px] text-gray-300 mt-2.5 px-4">
            Haki AI provides legal information, not legal advice. Always consult a licensed Kenyan advocate.
          </p>
        </div>
      </div>
    </div>
  )
}
