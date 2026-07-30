'use client'

import { useEffect } from 'react'

const VARIANTS = {
  error: {
    border: 'border-red-200',
    iconStroke: '#dc2626',
    icon: (
      <>
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="8" x2="12" y2="12"></line>
        <line x1="12" y1="16" x2="12.01" y2="16"></line>
      </>
    ),
  },
  info: {
    border: 'border-teal-200',
    iconStroke: '#0d9488',
    icon: (
      <>
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="12" y1="16" x2="12" y2="12"></line>
        <line x1="12" y1="8" x2="12.01" y2="8"></line>
      </>
    ),
  },
  success: {
    border: 'border-green-200',
    iconStroke: '#16a34a',
    icon: (
      <>
        <circle cx="12" cy="12" r="10"></circle>
        <polyline points="8 12 11 15 16 9"></polyline>
      </>
    ),
  },
}

const AUTO_DISMISS_MS = 3000

export default function Toast({ message, onDismiss, variant = 'error' }) {
  // Error toasts stay until manually dismissed (no timer at all) since
  // they need action. Info and success toasts are routine confirmations
  // the user doesn't need to act on, so they auto-dismiss after a few
  // seconds — but can still be dismissed early via the ✕.
  const autoDismiss = variant === 'info' || variant === 'success'

  useEffect(() => {
    if (!message || !autoDismiss) return
    const timer = setTimeout(onDismiss, AUTO_DISMISS_MS)
    return () => clearTimeout(timer)
  }, [message, autoDismiss, onDismiss])

  if (!message) return null

  const style = VARIANTS[variant] || VARIANTS.error

  return (
    <div className="fixed top-4 right-4 z-[100] max-w-sm">
      <div className={`flex items-start gap-3 bg-white border ${style.border} shadow-lg rounded-xl px-4 py-3`}>
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={style.iconStroke} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 mt-0.5">
          {style.icon}
        </svg>
        <p className="text-[13px] text-gray-700 leading-snug">{message}</p>
        <button onClick={onDismiss} className="flex-shrink-0 text-gray-400 hover:text-gray-600 ml-1">
          ✕
        </button>
      </div>
    </div>
  )
}
