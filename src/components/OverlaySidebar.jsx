/**
 * OverlaySidebar — Right-side sidebar with vertical tab buttons.
 * Tab toggles the sidebar open/closed.
 * Shift+Tab toggles full-screen overlay mode for detailed viewing.
 * Escape closes either mode.
 */

import { useState, useEffect, useCallback } from 'react'

const SIDEBAR_WIDTH = 360

const S = {
  wrapper: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    zIndex: 20,
    display: 'flex',
    pointerEvents: 'none',
  },
  tabStrip: {
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
    padding: '8px 0',
    pointerEvents: 'auto',
  },
  tabBtn: {
    background: 'rgba(7,7,15,0.85)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRight: 'none',
    borderRadius: '6px 0 0 6px',
    color: 'rgba(255,255,255,0.7)',
    cursor: 'pointer',
    padding: '8px 6px',
    fontSize: 9,
    fontFamily: "Inter, sans-serif",
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
    transition: 'all 0.15s ease',
    textAlign: 'center',
    lineHeight: 1.3,
    minWidth: 40,
    backdropFilter: 'blur(12px)',
  },
  tabBtnActive: {
    background: 'rgba(7,7,15,0.92)',
    color: '#e2b96f',
    borderColor: 'rgba(226,185,111,0.3)',
  },
  panel: {
    width: SIDEBAR_WIDTH,
    background: 'rgba(7,7,15,0.92)',
    borderLeft: '1px solid rgba(255,255,255,0.07)',
    display: 'flex',
    flexDirection: 'column',
    pointerEvents: 'auto',
    backdropFilter: 'blur(12px)',
    overflow: 'hidden',
  },
  header: {
    padding: '8px 12px',
    borderBottom: '1px solid rgba(255,255,255,0.07)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: '0.07em',
    color: 'rgba(255,255,255,0.75)',
    flexShrink: 0,
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    color: 'rgba(255,255,255,0.7)',
    cursor: 'pointer',
    fontSize: 14,
    padding: '2px 4px',
    lineHeight: 1,
  },
  content: {
    flex: 1,
    overflowY: 'auto',
    overflowX: 'hidden',
    minHeight: 0,
  },
  // Full-screen overlay
  fullscreenBackdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    zIndex: 30,
    background: 'rgba(7,7,15,0.95)',
    backdropFilter: 'blur(20px)',
    display: 'flex',
    flexDirection: 'column',
    pointerEvents: 'auto',
  },
  fullscreenHeader: {
    padding: '12px 20px',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    flexShrink: 0,
  },
  fullscreenTabs: {
    display: 'flex',
    gap: 0,
  },
  fullscreenTabBtn: {
    background: 'transparent',
    border: 'none',
    borderBottom: '2px solid transparent',
    color: 'rgba(255,255,255,0.6)',
    cursor: 'pointer',
    padding: '8px 16px',
    fontSize: 11,
    fontFamily: "Inter, sans-serif",
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    transition: 'all 0.15s ease',
  },
  fullscreenTabBtnActive: {
    color: '#e2b96f',
    borderBottomColor: '#e2b96f',
  },
  fullscreenContent: {
    flex: 1,
    overflowY: 'auto',
    overflowX: 'hidden',
    minHeight: 0,
    padding: 20,
  },
  fullscreenCloseBtn: {
    background: 'rgba(255,255,255,0.06)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 6,
    color: 'rgba(255,255,255,0.7)',
    cursor: 'pointer',
    fontSize: 11,
    fontFamily: "Inter, sans-serif",
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    padding: '6px 14px',
    transition: 'all 0.15s ease',
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  hint: {
    fontSize: 9,
    color: 'rgba(255,255,255,0.4)',
    letterSpacing: '0.06em',
    marginLeft: 12,
  },
}

const scrollbarCSS = `
.overlay-sidebar-scroll::-webkit-scrollbar { width: 4px; }
.overlay-sidebar-scroll::-webkit-scrollbar-track { background: transparent; }
.overlay-sidebar-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
.overlay-sidebar-scroll::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
`

/**
 * @param {Object} props
 * @param {{ id: string, icon: string, label: string, content: React.ReactNode, fullscreenContent?: React.ReactNode }[]} props.tabs
 * @param {boolean} props.open
 * @param {function} props.onToggle
 * @param {string} props.activeTab
 * @param {function} props.onTabChange
 * @param {boolean} props.fullscreen
 * @param {function} props.onFullscreenToggle
 */
export default function OverlaySidebar({ tabs, open, onToggle, activeTab, onTabChange, fullscreen, onFullscreenToggle }) {
  const activeTabDef = tabs.find(t => t.id === activeTab)

  // Keyboard handler
  useEffect(() => {
    const handler = (e) => {
      // Don't capture in inputs, textareas, selects, or contenteditable
      const tag = document.activeElement?.tagName?.toLowerCase()
      if (tag === 'input' || tag === 'textarea' || tag === 'select' || document.activeElement?.contentEditable === 'true') {
        return
      }
      if (e.key === 'Tab' && e.shiftKey) {
        e.preventDefault()
        onFullscreenToggle()
      } else if (e.key === 'Tab') {
        e.preventDefault()
        if (fullscreen) {
          // Shift+Tab opened fullscreen, plain Tab downgrades to sidebar
          onFullscreenToggle()
          if (!open) onToggle()
        } else {
          onToggle()
        }
      }
      if (e.key === 'Escape') {
        if (fullscreen) {
          e.preventDefault()
          onFullscreenToggle()
        } else if (open) {
          e.preventDefault()
          onToggle()
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, fullscreen, onToggle, onFullscreenToggle])

  if (!tabs.length) return null

  // Full-screen overlay mode
  if (fullscreen) {
    return (
      <>
        <style>{scrollbarCSS}</style>
        <div style={S.fullscreenBackdrop}>
          <div style={S.fullscreenHeader}>
            <div style={S.fullscreenTabs}>
              {tabs.map(tab => (
                <button
                  key={tab.id}
                  style={{
                    ...S.fullscreenTabBtn,
                    ...(activeTab === tab.id ? S.fullscreenTabBtnActive : {}),
                  }}
                  onClick={() => onTabChange(tab.id)}
                >
                  {tab.icon} {tab.label}
                </button>
              ))}
            </div>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <span style={S.hint}>Shift+Tab or Esc to close</span>
              <button style={S.fullscreenCloseBtn} onClick={onFullscreenToggle}>
                ✕ Close
              </button>
            </div>
          </div>
          <div style={S.fullscreenContent} className="overlay-sidebar-scroll">
            {activeTabDef && (activeTabDef.fullscreenContent || activeTabDef.content)}
          </div>
        </div>
      </>
    )
  }

  // Normal sidebar mode
  return (
    <>
      <style>{scrollbarCSS}</style>
      <div style={S.wrapper}>
        {/* Tab strip — always visible */}
        <div style={S.tabStrip}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              style={{
                ...S.tabBtn,
                ...(open && activeTab === tab.id ? S.tabBtnActive : {}),
              }}
              onClick={() => {
                if (open && activeTab === tab.id) {
                  onToggle() // close if clicking active tab
                } else {
                  onTabChange(tab.id)
                  if (!open) onToggle()
                }
              }}
              title={tab.label}
            >
              <div style={{ fontSize: 14, marginBottom: 2 }}>{tab.icon}</div>
              <div>{tab.label}</div>
            </button>
          ))}
        </div>

        {/* Panel content — only when open */}
        {open && activeTabDef && (
          <div style={S.panel}>
            <div style={S.header}>
              <span>{activeTabDef.icon} {activeTabDef.label}</span>
              <button style={S.closeBtn} onClick={onToggle}>✕</button>
            </div>
            <div style={S.content} className="overlay-sidebar-scroll">
              {activeTabDef.content}
            </div>
          </div>
        )}
      </div>
    </>
  )
}

export { SIDEBAR_WIDTH }
