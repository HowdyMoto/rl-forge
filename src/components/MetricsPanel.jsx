export default function MetricsPanel({ metrics, episodes, status, backend }) {
  const latest = metrics[metrics.length - 1]

  const stat = (label, value, unit = '') => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
        {label}
      </span>
      <span style={{ fontSize: 16, color: '#e2b96f', fontVariantNumeric: 'tabular-nums' }}>
        {value}<span style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', marginLeft: 2 }}>{unit}</span>
      </span>
    </div>
  )

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(3, 1fr)',
      gap: '12px 20px',
      fontFamily: '"DM Mono", monospace',
    }}>
      {stat('Updates', latest?.update ?? '—')}
      {stat('Episodes', episodes)}
      {stat('Steps', latest?.totalSteps != null
        ? (latest.totalSteps >= 1000 ? (latest.totalSteps / 1000).toFixed(1) + 'k' : latest.totalSteps)
        : '—')}
      {stat('Mean Reward', latest?.meanReward20 != null ? latest.meanReward20.toFixed(1) : '—')}
      {stat('Policy Loss', latest?.policyLoss != null ? latest.policyLoss.toFixed(3) : '—')}
      {stat('Entropy', latest?.entropy != null ? latest.entropy.toFixed(3) : '—')}

      <div style={{ gridColumn: '1 / -1', marginTop: 4, display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{
          width: 6, height: 6, borderRadius: '50%',
          background: status === 'Training started' || status === 'Resumed'
            ? '#4ade80'
            : status === 'Paused' ? '#facc15' : 'rgba(255,255,255,0.2)',
          boxShadow: status === 'Training started' || status === 'Resumed'
            ? '0 0 6px #4ade80' : 'none'
        }} />
        <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)' }}>
          {status || 'idle'} {backend ? `· ${backend}` : ''}
        </span>
      </div>
    </div>
  )
}
