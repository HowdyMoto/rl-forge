import { useState, useEffect } from 'react'
import { DEFAULT_PPO_CONFIG } from '../rl/ppo.js'
import Tooltip from './Tooltip.jsx'

const PARAM_META = [
  {
    key: 'learningRate', label: 'Learning Rate',
    min: 1e-5, max: 1e-2, step: 1e-5, fmt: v => v.toExponential(1),
    tooltip: 'Step size for the Adam optimizer. Controls how aggressively weights update each gradient step. 3e-4 is a reliable PPO default.\n\nHigher: faster updates but risks instability or divergence. Lower: more stable but slower convergence.',
  },
  {
    key: 'clipEpsilon', label: 'Clip ε',
    min: 0.05, max: 0.4, step: 0.01, fmt: v => v.toFixed(2),
    tooltip: "PPO's trust-region constraint. The probability ratio π(a|s)/π_old(a|s) is clipped to [1−ε, 1+ε], keeping the updated policy close to the one that collected the data.\n\nHigher: allows larger policy updates per step, riskier. Lower: more conservative updates, slower but more stable.",
  },
  {
    key: 'gamma', label: 'γ Discount',
    min: 0.9, max: 0.999, step: 0.001, fmt: v => v.toFixed(3),
    tooltip: 'Exponential discount on future rewards. At γ=0.99, a reward 100 steps away is worth ~37% of its face value.\n\nHigher: longer planning horizon, more weight on distant rewards. Lower: myopic — may train faster on short tasks but ignores long-term consequences.',
  },
  {
    key: 'lambda', label: 'λ GAE',
    min: 0.8, max: 1.0, step: 0.01, fmt: v => v.toFixed(2),
    tooltip: 'Generalized Advantage Estimation interpolation between one-step TD and full Monte Carlo returns. λ≈0.95 is the standard sweet spot.\n\nHigher (toward 1): less bias, more variance — closer to Monte Carlo. Lower (toward 0): less variance, more bias — closer to one-step TD.',
  },
  {
    key: 'entropyCoef', label: 'Entropy Coef',
    min: 0, max: 0.1, step: 0.001, fmt: v => v.toFixed(3),
    tooltip: 'Weight of an entropy bonus in the loss. Encourages the policy to stay stochastic and exploratory rather than committing prematurely to one action.\n\nHigher: more exploration, slower convergence. Lower: faster convergence to a strategy, but risks getting stuck in a local optimum.',
  },
  {
    key: 'stepsPerUpdate', label: 'Steps/Update',
    min: 128, max: 2048, step: 128, fmt: v => v.toString(),
    tooltip: 'Number of environment steps collected before each policy gradient update — the rollout buffer size.\n\nHigher: more stable gradient estimates, updates less often. Lower: more frequent updates but with noisier advantage estimates.',
  },
  {
    key: 'numEpochs', label: 'Epochs',
    min: 1, max: 20, step: 1, fmt: v => v.toString(),
    tooltip: "Gradient passes over each collected rollout buffer per update. PPO's clipping limits damage at high values, but instability can still emerge.\n\nHigher: better sample efficiency, risk of overfitting to the stale batch. Lower: safer but wastes collected data.",
  },
]

export default function HyperParams({ onChange, disabled, overrides }) {
  const [params, setParams] = useState({ ...DEFAULT_PPO_CONFIG, ...overrides })

  // When environment changes, apply its overrides on top of defaults
  const [prevOverrides, setPrevOverrides] = useState(overrides)
  if (overrides !== prevOverrides) {
    setPrevOverrides(overrides)
    setParams({ ...DEFAULT_PPO_CONFIG, ...overrides })
  }

  // Notify parent after render, not during
  useEffect(() => {
    onChange?.({ ...DEFAULT_PPO_CONFIG, ...overrides })
  }, [overrides])

  const update = (key, value) => {
    const next = { ...params, [key]: value }
    setParams(next)
    onChange?.(next)
  }

  return (
    <div style={{ fontFamily: '"DM Mono", monospace', display: 'flex', flexDirection: 'column', gap: 10 }}>
      {PARAM_META.map(({ key, label, min, max, step, fmt, tooltip }) => (
        <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.6)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center' }}>
              {label}
              <Tooltip text={tooltip} />
            </span>
            <span style={{ fontSize: 13, color: '#e2b96f' }}>{fmt(params[key])}</span>
          </div>
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={params[key]}
            disabled={disabled}
            onChange={e => update(key, parseFloat(e.target.value))}
            style={{
              width: '100%',
              accentColor: '#e2b96f',
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.4 : 1,
            }}
          />
        </div>
      ))}
    </div>
  )
}
