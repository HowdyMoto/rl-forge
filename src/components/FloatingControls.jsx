/**
 * FloatingControls — Always-visible minimal training controls
 * floating at the bottom-right of the canvas.
 */

const S = {
  wrapper: {
    position: 'absolute',
    bottom: 16,
    right: 16,
    zIndex: 15,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    background: 'rgba(7,7,15,0.85)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 8,
    padding: '6px 10px',
    backdropFilter: 'blur(12px)',
    pointerEvents: 'auto',
  },
  btn: {
    border: 'none',
    borderRadius: 5,
    cursor: 'pointer',
    fontFamily: "Inter, sans-serif",
    fontSize: 11,
    fontWeight: 500,
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    transition: 'all 0.15s ease',
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    padding: '6px 12px',
  },
  statusDot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    display: 'inline-block',
  },
  status: {
    fontSize: 9,
    color: 'rgba(255,255,255,0.65)',
    letterSpacing: '0.06em',
    textTransform: 'uppercase',
    display: 'flex',
    alignItems: 'center',
    gap: 4,
  },
}

export default function FloatingControls({
  trainingState,
  trainLabel,
  onTrain,
  onPause,
  onStop,
  sidebarOpen,
  sidebarWidth = 0,
}) {
  const isIdle = trainingState === 'idle'
  const isRunning = trainingState === 'running'
  const isPaused = trainingState === 'paused'

  const dotColor = isRunning ? '#4ade80' : isPaused ? '#e2b96f' : 'rgba(255,255,255,0.2)'
  const statusText = isRunning ? 'training' : isPaused ? 'paused' : 'idle'

  return (
    <div style={{
      ...S.wrapper,
      right: sidebarOpen ? sidebarWidth + 56 : 16,
      transition: 'right 0.2s ease',
    }}>
      {/* Status indicator */}
      <div style={S.status}>
        <span style={{ ...S.statusDot, background: dotColor, boxShadow: isRunning ? '0 0 6px rgba(74,222,128,0.5)' : 'none' }} />
        {statusText}
      </div>

      <div style={{ width: 1, height: 18, background: 'rgba(255,255,255,0.1)', margin: '0 2px' }} />

      {/* Train button */}
      {isIdle && (
        <button
          style={{
            ...S.btn,
            background: '#e2b96f',
            color: '#0a0a14',
          }}
          onClick={onTrain}
        >
          ▶ Train {trainLabel}
        </button>
      )}

      {/* Pause/Resume */}
      {!isIdle && (
        <button
          style={{
            ...S.btn,
            background: 'rgba(255,255,255,0.03)',
            color: 'rgba(255,255,255,0.75)',
            border: '1px solid rgba(255,255,255,0.07)',
          }}
          onClick={onPause}
        >
          {isPaused ? '▶ Resume' : '⏸ Pause'}
        </button>
      )}

      {/* Stop */}
      {!isIdle && (
        <button
          style={{
            ...S.btn,
            background: 'rgba(224,90,90,0.12)',
            color: '#e05a5a',
            border: '1px solid rgba(224,90,90,0.2)',
          }}
          onClick={onStop}
        >
          ■ Stop
        </button>
      )}
    </div>
  )
}
