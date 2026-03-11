import Tooltip from './Tooltip.jsx'

const STAT_TOOLTIPS = {
  Updates: 'Policy gradient updates performed since training started. Each update processes one full rollout buffer of experience.',
  Episodes: 'Completed episodes — each a full trajectory from environment reset to terminal state. Not directly tied to update count; one buffer may span multiple episodes.',
  Steps: 'Total environment timesteps simulated across all episodes. The fundamental measure of data consumed by training.',
  'Mean Reward': 'Rolling mean episode return over the last 20 episodes. The primary training signal — smoother than per-episode reward and what determines whether the environment is "solved".\n\nRising: agent is improving. Plateaued: may need more time or hyperparameter tuning. Crossing the dashed target line means solved.',
  'Policy Loss': 'The clipped PPO surrogate objective (negated for minimization). Watch the trend, not the absolute value.\n\nNear zero: the clip bound is active — policy is not changing much per update. Erratic spikes: learning rate may be too high.',
  Entropy: 'Shannon entropy of the current action distribution — how random the policy\'s decisions are.\n\nHigher: more exploratory. Lower: more deterministic. Gradual decay during training is healthy; sudden early collapse can signal getting stuck in a local optimum.',
}

export default function MetricsPanel({ metrics, episodes, status, backend }) {
  const latest = metrics[metrics.length - 1]

  const stat = (label, value, unit = '') => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.55)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center' }}>
        {label}
        {STAT_TOOLTIPS[label] && <Tooltip text={STAT_TOOLTIPS[label]} />}
      </span>
      <span style={{ fontSize: 18, color: '#e2b96f', fontVariantNumeric: 'tabular-nums' }}>
        {value}<span style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', marginLeft: 2 }}>{unit}</span>
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
        <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.55)' }}>
          {status || 'idle'} {backend ? `· ${backend}` : ''}
        </span>
      </div>
    </div>
  )
}
