'use client'

import { useState, useEffect, useRef } from 'react'

const API = 'http://localhost:8000'

const MODE_LABELS = {
  qa: 'Veritas',
  document: 'Dictum',
}

export default function Navbar({ mode, userName = 'Guest', onSettings, onEditProfile, onChangePassword }) {
  const [menuOpen, setMenuOpen] = useState(false)
  const [connected, setConnected] = useState(false)
  const menuRef = useRef(null)

  useEffect(() => {
    function handleClickOutside(e) {
      if (menuRef.current && !menuRef.current.contains(e.target)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Poll /api/health to know when the backend is actually reachable.
  // Starts disconnected (grey, animated) and flips to connected (green,
  // still animated — a soft pulse rather than a static dot) once a
  // health check succeeds. Keeps polling afterward in case the backend
  // goes down later in the session.
  //
  // Hardening against log-flood: previously polled every 5s with no
  // overlap guard and no regard for tab visibility. Multiple open tabs
  // (or React Strict Mode's deliberate double-mount in dev) each run
  // their own independent interval, and since none of that is wrong on
  // its own, the fix is making each individual poller lighter — longer
  // interval, skip starting a new check while one is still in flight
  // (avoids piling up requests if the backend is slow), and pause
  // entirely while the tab is in the background (most "why are there so
  // many health checks" cases are just background tabs nobody's
  // looking at, still dutifully polling).
  useEffect(() => {
    let cancelled = false
    let inFlight = false

    async function checkHealth() {
      if (inFlight || document.visibilityState !== 'visible') return
      inFlight = true
      try {
        const res = await fetch(`${API}/api/health`)
        if (!cancelled) setConnected(res.ok)
      } catch {
        if (!cancelled) setConnected(false)
      } finally {
        inFlight = false
      }
    }

    checkHealth()
    const interval = setInterval(checkHealth, 15000)

    function handleVisibilityChange() {
      if (document.visibilityState === 'visible') checkHealth()
    }
    document.addEventListener('visibilitychange', handleVisibilityChange)

    return () => {
      cancelled = true
      clearInterval(interval)
      document.removeEventListener('visibilitychange', handleVisibilityChange)
    }
  }, [])

  // Always shows a mode label — defaults to Q&A (the default selection)
  // even before any real mode/backend connection is established, per
  // explicit instruction, rather than showing nothing until a choice is
  // made or the backend responds.
  const modeLabel = MODE_LABELS[mode] || MODE_LABELS.qa

  return (
    <>
      <style>{`
        @keyframes statusPulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.35; }
        }
        .status-dot {
          animation: statusPulse 1.6s ease-in-out infinite;
        }
      `}</style>

      {/* sticky + z-10: this now lives INSIDE the same scrolling
          container as the chat messages (see page.js) rather than
          being a separate block above ChatArea's own scrollbox — sticky
          only pins correctly when it shares a scrolling ancestor with
          the content scrolling past it. bg-white is required here, not
          just decorative: without an opaque background, message content
          scrolling underneath would show through as it passes beneath
          this pinned bar. No top/bottom border, per explicit instruction. */}
      <div className="sticky top-0 z-10 w-full bg-white flex items-center justify-between px-6 py-3 flex-shrink-0">
        {/* Left: connection status + model/mode label. No userType
            ("Lawyer") label shown anymore. */}
        <div className="flex items-center gap-2 text-[13px] text-gray-600">
          <span className="text-gray-600">Model: {modeLabel}</span>
          <span
            className={`status-dot inline-block w-2 h-2 rounded-full ${
              connected ? 'bg-green-500' : 'bg-gray-400'
            }`}
            title={connected ? 'Connected' : 'Connecting…'}
          />
          {/* <span className="text-gray-600">Model: {modeLabel}</span> */}
          {/* <span className="text-gray-300">•</span> */}
          <span className="px-2 py-0.5 bg-teal-500 text-white rounded-full text-[11px] font-medium">
            Free plan
          </span>
        </div>

        {/* Right: user menu */}
        <div ref={menuRef} className="relative">
          <button
            onClick={() => setMenuOpen(!menuOpen)}
            className="flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-gray-50 transition"
          >
            <span className="w-7 h-7 flex items-center justify-center rounded-full bg-teal-500 text-white text-[12px] font-semibold">
              {userName.charAt(0).toUpperCase()}
            </span>
            <span className="text-[13px] text-gray-700 font-medium">{userName}</span>
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-400">
              <polyline points="6 9 12 15 18 9"></polyline>
            </svg>
          </button>

          {menuOpen && (
            <div className="absolute top-full right-0 mt-2 w-52 bg-white border border-gray-200 rounded-xl shadow-lg overflow-hidden z-50">
              <button
                onClick={() => { setMenuOpen(false); onSettings?.() }}
                className="w-full text-left px-4 py-2.5 text-[13px] text-gray-700 hover:bg-teal-500 hover:text-white transition flex items-center gap-2"
              >
                ⚙️ Settings
              </button>
              <button
                onClick={() => { setMenuOpen(false); onEditProfile?.() }}
                className="w-full text-left px-4 py-2.5 text-[13px] text-gray-700 hover:bg-teal-500 hover:text-white transition flex items-center gap-2"
              >
                👤 Edit profile
              </button>
              <button
                onClick={() => { setMenuOpen(false); onChangePassword?.() }}
                className="w-full text-left px-4 py-2.5 text-[13px] text-gray-700 hover:bg-teal-500 hover:text-white transition flex items-center gap-2"
              >
                🔒 Change password
              </button>
            </div>
          )}
        </div>
      </div>
    </>
  )
}
