import { useState, useRef, useId } from 'react'

export default function Tooltip({ text }) {
  const [pos, setPos] = useState(null)
  const iconRef = useRef(null)
  const tooltipId = useId()

  const show = () => {
    const rect = iconRef.current?.getBoundingClientRect()
    if (rect) setPos({ x: rect.left + rect.width / 2, y: rect.top })
  }
  const hide = () => setPos(null)

  const onKeyDown = (e) => {
    if (e.key === 'Escape') hide()
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      pos ? hide() : show()
    }
  }

  const paragraphs = typeof text === 'string' ? text.split('\n\n') : [text]

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center' }}>
      <span
        ref={iconRef}
        role="button"
        tabIndex={0}
        aria-label="More information"
        aria-describedby={pos ? tooltipId : undefined}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onKeyDown={onKeyDown}
        style={{
          cursor: 'help',
          color: 'rgba(255,255,255,0.2)',
          fontSize: 9,
          marginLeft: 5,
          lineHeight: 1,
          userSelect: 'none',
          fontStyle: 'normal',
          outline: 'none',
          borderRadius: 2,
        }}
      >
        ⓘ
      </span>
      {pos && (
        <div
          id={tooltipId}
          role="tooltip"
          style={{
            position: 'fixed',
            left: Math.min(Math.max(pos.x - 125, 8), window.innerWidth - 266),
            top: pos.y - 10,
            transform: 'translateY(-100%)',
            width: 250,
            background: 'rgba(10,10,20,0.97)',
            border: '1px solid rgba(255,255,255,0.1)',
            borderRadius: 6,
            padding: '9px 11px',
            fontSize: 10,
            lineHeight: 1.6,
            color: 'rgba(255,255,255,0.65)',
            fontFamily: '"DM Mono", monospace',
            letterSpacing: '0.02em',
            textTransform: 'none',
            zIndex: 9999,
            pointerEvents: 'none',
            boxShadow: '0 4px 24px rgba(0,0,0,0.6)',
          }}
        >
          {paragraphs.map((p, i) => (
            <span key={i}>
              {i > 0 && (
                <span style={{
                  display: 'block',
                  height: 1,
                  background: 'rgba(255,255,255,0.08)',
                  margin: '7px 0',
                }} />
              )}
              <span style={{ display: 'block', color: i === 0 ? 'rgba(255,255,255,0.65)' : 'rgba(255,255,255,0.4)' }}>
                {p}
              </span>
            </span>
          ))}
        </div>
      )}
    </span>
  )
}
