'use client'

export default function Sidebar({
  collapsed,
  setCollapsed,
  history,
  newChat,
  loadSavedChat,
  viewSavedChat,
  refreshHistory,
  onChatHistory,   // optional: called when "Chat History" nav button is clicked
  onCaseAnalysis,  // optional: called when "Case Analysis" nav button is clicked
  showSuccessToast, // optional: called with a message after a successful delete
}) {
  const formatDate = (dateStr) => {
    try {
      const d = new Date(dateStr)
      const now = new Date()
      const diff = Math.floor((now - d) / (1000 * 60 * 60 * 24))
      if (diff === 0) return 'Today'
      if (diff === 1) return 'Yesterday'
      if (diff < 7) return `${diff} days ago`
      if (diff < 14) return '1 week ago'
      if (diff < 30) return `${Math.floor(diff / 7)} weeks ago`
      return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    } catch { return '' }
  }

  const grouped = {}
  Object.entries(history || {})
    .sort((a, b) => new Date(b[1].date) - new Date(a[1].date))
    .forEach(([id, chat]) => {
      const label = formatDate(chat.date)
      if (!grouped[label]) grouped[label] = []
      grouped[label].push({ id, ...chat })
    })

  // Fixed: previously this had no error handling at all, so a failed
  // request (wrong port, backend down, non-2xx response) failed silently
  // and refreshHistory() was called unconditionally even when the prop
  // wasn't passed, throwing before you'd ever see what went wrong.
  const handleDelete = async (e, chatId) => {
    e.stopPropagation()
    try {
      const res = await fetch(`http://localhost:8000/api/history/${chatId}`, {
        method: 'DELETE',
      })
      if (!res.ok) {
        const body = await res.text().catch(() => '')
        console.error(`Failed to delete chat ${chatId}: ${res.status} ${res.statusText}`, body)
        return
      }
      if (typeof refreshHistory === 'function') {
        await refreshHistory()
      }
      if (typeof showSuccessToast === 'function') {
        showSuccessToast('Chat deleted.')
      }
    } catch (err) {
      console.error(`Error deleting chat ${chatId}:`, err)
    }
  }

  const handleView = (chatId) => {
    if (viewSavedChat) {
      viewSavedChat(chatId)
    }
  }

  // title (not just label) matters here now: in the collapsed rail these
  // render icon-only, so the title attribute becomes the only indication
  // of what each button does — a native tooltip on hover.
  const navButtons = [
    {
      key: 'new-chat',
      label: 'New chat',
      onClick: newChat,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <line x1="12" y1="5" x2="12" y2="19"></line>
          <line x1="5" y1="12" x2="19" y2="12"></line>
        </svg>
      ),
    },
    {
      key: 'chat-history',
      label: 'Chat history',
      onClick: onChatHistory,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="9"></circle>
          <polyline points="12 7 12 12 15 14.5"></polyline>
        </svg>
      ),
    },
    {
      key: 'case-analysis',
      label: 'Case analysis',
      onClick: onCaseAnalysis,
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M12 3v18"></path>
          <path d="M5 7l-3 6a3 3 0 0 0 6 0l-3-6z"></path>
          <path d="M19 7l-3 6a3 3 0 0 0 6 0l-3-6z"></path>
          <path d="M4 7h16"></path>
          <path d="M9 21h6"></path>
        </svg>
      ),
    },
  ]

  // Collapse toggle icon — a "panel" glyph, same idea as Claude's own
  // sidebar toggle (a rectangle with a divider, rather than a hamburger
  // or arrow), so the same icon reads correctly as "collapse" AND
  // "expand" without needing to swap glyphs based on state.
  const PanelToggleIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="16" rx="2"></rect>
      <line x1="10" y1="4" x2="10" y2="20"></line>
    </svg>
  )

  // Collapsed: an icon-only rail, not a fully hidden sidebar (previously
  // this returned a small floating "☰" button elsewhere on the page
  // instead, per explicit instruction to match Claude's own collapsed
  // behavior — the sidebar stays present as a narrow strip rather than
  // disappearing). Recents/history and the version footer don't fit
  // meaningfully in this width, so they're the two things intentionally
  // omitted here — everything else (new chat, chat history, case
  // analysis) stays available, icon-only with a native tooltip.
  if (collapsed) {
    return (
      <div className="w-14 h-full bg-[#0d9488] flex flex-col items-center py-4 gap-5 flex-shrink-0">
        <button
          onClick={() => setCollapsed(false)}
          title="Expand sidebar"
          className="text-white/90 hover:text-white transition"
        >
          <PanelToggleIcon />
        </button>

        <div className="w-6 h-px bg-white/20" />

        {navButtons.map((btn) => (
          <button
            key={btn.key}
            onClick={btn.onClick}
            title={btn.label}
            className="text-white/85 hover:text-white transition"
          >
            {btn.icon}
          </button>
        ))}
      </div>
    )
  }

  return (
    <>
      <style>{`
        .sidebar-scroll::-webkit-scrollbar { width: 1px; }
        .sidebar-scroll::-webkit-scrollbar-track { background: transparent; }
        .sidebar-scroll::-webkit-scrollbar-thumb {
          background-color: rgba(255, 255, 255, 0.35);
          border-radius: 9999px;
        }
        .sidebar-scroll::-webkit-scrollbar-thumb:hover {
          background-color: rgba(255, 255, 255, 0.55);
        }
        .sidebar-scroll {
          scrollbar-width: thin;
          scrollbar-color: rgba(255, 255, 255, 0.35) transparent;
        }
      `}</style>
      <div className="w-[270px] bg-[#0d9488] rrounded-r-lg flex flex-col h-full flex-shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between" style={{ padding: '12px 17px 17px 18px' }}>
        <span style={{ fontFamily: "'Lato', sans-serif", fontSize: '20px', letterSpacing: '1.5px', color: '#ffffff' }}>
          Haki
        </span>
        <button
          onClick={() => setCollapsed(true)}
          title="Collapse sidebar"
          className="w-7 h-7 flex items-center justify-center rounded-md text-white/85 hover:text-white hover:bg-white/10 transition"
        >
          <PanelToggleIcon />
        </button>
      </div>

      {/* Nav button list */}
      <div className="flex flex-col gap-1 px-4">
        {navButtons.map((btn) => (
          <button
            key={btn.key}
            onClick={btn.onClick}
            className="flex items-center gap-2.5 px-3 py-2 rounded-lg text-[14px] font-medium text-white/90 hover:bg-white/10 transition text-left"
          >
            <span className="flex-shrink-0">{btn.icon}</span>
            {btn.label}
          </button>
        ))}
      </div>

      <div className="h-3" />

      {/* Recents */}
      <div className="sidebar-scroll flex-1 overflow-y-auto px-3 pt-0 pb-0 sidebar-history">
        <p className="text-[12px] text-white/60 font-medium pt-1 pb-1 px-2">Recents</p>
        {Object.keys(grouped).length === 0 ? (
          <p className="text-[14px] text-white/60 py-2 px-2">No conversations yet</p>
        ) : (
          Object.entries(grouped).map(([label, items]) => (
            <div key={label}>
              <p className="text-[12px] text-white/60 font-medium pt-3 pb-1">{label}</p>
              {items.map((item) => (
                <div key={item.id} className="flex items-center group">
                  {/* Title — click to view */}
                  <button
                    onClick={() => handleView(item.id)}
                    className="flex-1 text-left px-2 py-1.5 -mx-2 rounded-md text-[14px] text-white/90 hover:bg-white/15 truncate"
                  >
                    {item.title || 'Untitled'}
                  </button>

                  {/* Delete icon */}
                  <button
                    onClick={(e) => handleDelete(e, item.id)}
                    className="opacity-0 group-hover:opacity-100 ml-1 p-1 rounded hover:bg-red-400/30 text-white/50 hover:text-white transition flex-shrink-0"
                    title="Delete this draft"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                  </button>
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      {/* Footer — app version number, carried over from the original
          sidebar design; only shown in the expanded state (see the
          collapsed-rail branch above for why it's omitted there). */}
      <div className="px-3 pt-3 pb-2.5 text-[12px] text-white/50 text-center flex-shrink-0">
        Legal Assistive Tool v1.0
      </div>
      </div>
    </>
  )
}
