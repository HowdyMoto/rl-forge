/**
 * EnvSelector — Dropdown environment selector for the header bar.
 * Shows current env name, dropdown with all envs + "Build New..." option.
 */

import { useState, useRef, useEffect } from 'react'

const S = {
  wrapper: {
    position: 'relative',
    display: 'inline-block',
  },
  trigger: {
    background: 'rgba(255,255,255,0.03)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 6,
    color: '#e2b96f',
    cursor: 'pointer',
    fontFamily: "Inter, sans-serif",
    fontSize: 12,
    letterSpacing: '0.04em',
    padding: '6px 14px',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    transition: 'all 0.15s ease',
    minWidth: 180,
  },
  triggerDisabled: {
    opacity: 0.4,
    cursor: 'not-allowed',
  },
  chevron: {
    fontSize: 8,
    color: 'rgba(255,255,255,0.75)',
    marginLeft: 'auto',
  },
  dropdown: {
    position: 'absolute',
    top: '100%',
    left: 0,
    marginTop: 4,
    background: 'rgba(7,7,15,0.95)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 8,
    backdropFilter: 'blur(16px)',
    overflow: 'hidden',
    zIndex: 100,
    minWidth: 280,
    maxHeight: 420,
    overflowY: 'auto',
    boxShadow: '0 12px 40px rgba(0,0,0,0.5)',
  },
  item: {
    padding: '8px 14px',
    cursor: 'pointer',
    borderBottom: '1px solid rgba(255,255,255,0.04)',
    transition: 'background 0.1s',
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  itemLabel: {
    fontSize: 12,
    fontFamily: "Inter, sans-serif",
    letterSpacing: '0.04em',
    textTransform: 'uppercase',
  },
  itemDesc: {
    fontSize: 10,
    color: 'rgba(255,255,255,0.75)',
  },
  buildItem: {
    padding: '10px 14px',
    cursor: 'pointer',
    transition: 'background 0.1s',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    borderTop: '1px solid rgba(255,255,255,0.08)',
  },
}

export default function EnvSelector({ envConfigs, currentEnv, onSelect, disabled, onBuildNew }) {
  const [open, setOpen] = useState(false)
  const wrapperRef = useRef(null)

  // Close on click outside
  useEffect(() => {
    if (!open) return
    const handler = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [open])

  // Close on Escape
  useEffect(() => {
    if (!open) return
    const handler = (e) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [open])

  const currentConfig = envConfigs[currentEnv]

  return (
    <div style={S.wrapper} ref={wrapperRef}>
      <button
        style={{
          ...S.trigger,
          ...(disabled ? S.triggerDisabled : {}),
        }}
        onClick={() => !disabled && setOpen(!open)}
        disabled={disabled}
      >
        <span>{currentConfig?.label || currentEnv}</span>
        <span style={S.chevron}>▼</span>
      </button>

      {open && (
        <div style={S.dropdown}>
          {Object.entries(envConfigs).map(([key, cfg]) => (
            <div
              key={key}
              style={{
                ...S.item,
                background: key === currentEnv ? 'rgba(226,185,111,0.08)' : 'transparent',
                color: key === currentEnv ? '#e2b96f' : 'rgba(255,255,255,0.75)',
              }}
              onMouseEnter={e => {
                if (key !== currentEnv) e.currentTarget.style.background = 'rgba(255,255,255,0.04)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = key === currentEnv ? 'rgba(226,185,111,0.08)' : 'transparent'
              }}
              onClick={() => {
                onSelect(key)
                setOpen(false)
              }}
            >
              <span style={S.itemLabel}>{cfg.label}</span>
              <span style={S.itemDesc}>{cfg.desc}</span>
            </div>
          ))}

          {/* Build New */}
          <div
            style={{
              ...S.buildItem,
              color: 'rgba(255,255,255,0.75)',
            }}
            onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            onClick={() => {
              onBuildNew()
              setOpen(false)
            }}
          >
            <span style={{ fontSize: 14 }}>+</span>
            <div>
              <div style={{ fontSize: 12, fontFamily: "Inter, sans-serif", letterSpacing: '0.04em', textTransform: 'uppercase' }}>
                Build New...
              </div>
              <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.75)' }}>
                Open creature builder in terrain mode
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
