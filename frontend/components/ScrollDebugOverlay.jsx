'use client'

import { useEffect, useState } from 'react'

// TEMPORARY diagnostic component. Drop <ScrollDebugOverlay /> anywhere in
// page.js (e.g. right before the closing </div> of the root return) to
// see live measurements while you scroll/resize. Remove it once the
// scroll-height question is settled -- it's not meant to ship.
//
// What it measures, and why each one matters:
//   - viewportHeight        window.innerHeight -- the real, total space
//                            available on screen.
//   - scrollContainerRect   the .chat-scroll div's actual bounding box
//                            (top/bottom/height) -- this is the element
//                            that owns overflow-y-auto in the current
//                            layout (Navbar + ChatArea live inside it).
//   - scrollContainerClientHeight / scrollHeight
//                            clientHeight = visible height of the
//                            scrollable box. scrollHeight = total height
//                            of everything INSIDE it (messages). If
//                            scrollHeight > clientHeight, scrolling is
//                            possible; how much bigger tells you how far
//                            you can scroll.
//   - chatInputRect          the chat input wrapper's own bounding box --
//                            this is what currently sits OUTSIDE the
//                            scroll container as a separate flex sibling.
//   - gapBelowScrollContainer  scrollContainerRect.bottom vs
//                            chatInputRect.top -- if this is ~0, the
//                            scroll area's bottom edge and the input's
//                            top edge meet exactly (expected, by design,
//                            with the current sibling layout). If it's
//                            NOT ~0, something else is actually wrong
//                            (an unaccounted-for gap) rather than this
//                            being the expected "input reserves its own
//                            space" behavior.
export default function ScrollDebugOverlay() {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    function measure() {
      const scrollEl = document.querySelector('.chat-scroll')
      // The chat input wrapper doesn't currently have a stable selector
      // of its own -- querying the last flex-shrink-0 sibling after the
      // scroll container is a reasonable temporary hook for THIS
      // diagnostic; not something to rely on long-term.
      const mainColumn = scrollEl?.parentElement
      const chatInputEl = mainColumn
        ? Array.from(mainColumn.children).find(
            (el) => el !== scrollEl && el.tagName === 'DIV'
          )
        : null

      if (!scrollEl) {
        setStats({ error: 'Could not find .chat-scroll element on the page.' })
        return
      }

      const scrollRect = scrollEl.getBoundingClientRect()
      const inputRect = chatInputEl ? chatInputEl.getBoundingClientRect() : null

      setStats({
        viewportHeight: window.innerHeight,
        scrollContainer: {
          top: Math.round(scrollRect.top),
          bottom: Math.round(scrollRect.bottom),
          height: Math.round(scrollRect.height),
          clientHeight: scrollEl.clientHeight,
          scrollHeight: scrollEl.scrollHeight,
          isScrollable: scrollEl.scrollHeight > scrollEl.clientHeight,
        },
        chatInput: inputRect
          ? {
              top: Math.round(inputRect.top),
              bottom: Math.round(inputRect.bottom),
              height: Math.round(inputRect.height),
            }
          : null,
        gapBelowScrollContainer: inputRect
          ? Math.round(inputRect.top - scrollRect.bottom)
          : null,
        // If the scroll container's bottom + input's height doesn't add
        // up to the full viewport height, minus whatever's ABOVE the
        // scroll container (nothing, in the current layout -- it starts
        // at the very top), that's the real "is it spanning the whole
        // screen" answer.
        totalAccountedHeight: inputRect
          ? Math.round(scrollRect.height + inputRect.height)
          : null,
        unaccountedHeight: inputRect
          ? Math.round(window.innerHeight - (scrollRect.height + inputRect.height))
          : null,
      })
    }

    measure()
    window.addEventListener('resize', measure)
    const interval = setInterval(measure, 500) // catches scroll-driven height changes too
    return () => {
      window.removeEventListener('resize', measure)
      clearInterval(interval)
    }
  }, [])

  if (!stats) return null

  return (
    <div
      style={{
        position: 'fixed',
        bottom: 12,
        right: 12,
        zIndex: 9999,
        background: 'rgba(0,0,0,0.88)',
        color: '#0f0',
        fontFamily: 'monospace',
        fontSize: 11,
        lineHeight: 1.5,
        padding: '10px 12px',
        borderRadius: 8,
        maxWidth: 340,
        whiteSpace: 'pre-wrap',
        pointerEvents: 'none',
      }}
    >
      {stats.error ? (
        stats.error
      ) : (
        <>
          viewportHeight: {stats.viewportHeight}px{'\n'}
          {'\n'}
          scrollContainer (.chat-scroll){'\n'}
          {'  '}top: {stats.scrollContainer.top}px{'\n'}
          {'  '}bottom: {stats.scrollContainer.bottom}px{'\n'}
          {'  '}height: {stats.scrollContainer.height}px{'\n'}
          {'  '}clientHeight: {stats.scrollContainer.clientHeight}px{'\n'}
          {'  '}scrollHeight: {stats.scrollContainer.scrollHeight}px{'\n'}
          {'  '}isScrollable: {String(stats.scrollContainer.isScrollable)}{'\n'}
          {'\n'}
          chatInput (best-guess sibling){'\n'}
          {stats.chatInput ? (
            <>
              {'  '}top: {stats.chatInput.top}px{'\n'}
              {'  '}bottom: {stats.chatInput.bottom}px{'\n'}
              {'  '}height: {stats.chatInput.height}px{'\n'}
            </>
          ) : (
            '  not found\n'
          )}
          {'\n'}
          gap between scroll bottom & input top: {stats.gapBelowScrollContainer}px{'\n'}
          {'  '}(should be ~0 if input is meant to sit flush below the scroll area){'\n'}
          {'\n'}
          unaccounted viewport height: {stats.unaccountedHeight}px{'\n'}
          {'  '}(viewport − (scrollContainer.height + chatInput.height)){'\n'}
          {'  '}(non-zero here = something ABOVE the scroll container, or a{'\n'}
          {'  '}sizing bug, is eating space neither element accounts for)
        </>
      )}
    </div>
  )
}
