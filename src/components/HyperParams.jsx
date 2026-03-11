import { useState } from 'react'
import { DEFAULT_PPO_CONFIG } from '../rl/ppo.js'

const PARAM_META = [
  { key: 'learningRate', label: 'Learning Rate', min: 1e-5, max: 1e-2, step: 1e-5, fmt: v => v.toExponential(1) },
  { key: 'clipEpsilon', label: 'Clip ε', min: 0.05, max: 0.4, step: 0.01, fmt: v => v.toFixed(2) },
  { key: 'gamma', label: 'γ Discount', min: 0.9, max: 0.999, step: 0.001, fmt: v => v.toFixed(3) },
  { key: 'lambda', label: 'λ GAE', min: 0.8, max: 1.0, step: 0.01, fmt: v => v.toFixed(2) },
  { key: 'entropyCoef', label: 'Entropy Coef', min: 0, max: 0.1, step: 0.001, fmt: v => v.toFixed(3) },
  { key: 'stepsPerUpdate', label: 'Steps/Update', min: 128, max: 2048, step: 128, fmt: v => v.toString() },
  { key: 'numEpochs', label: 'Epochs', min: 1, max: 20, step: 1, fmt: v => v.toString() },
]

export default function HyperParams({ onChange, disabled }) {
  const [params, setParams] = useState(DEFAULT_PPO_CONFIG)

  const update = (key, value) => {
    const next = { ...params, [key]: value }
    setParams(next)
    onChange?.(next)
  }

  return (
    <div style={{ fontFamily: '"DM Mono", monospace', display: 'flex', flexDirection: 'column', gap: 10 }}>
      {PARAM_META.map(({ key, label, min, max, step, fmt }) => (
        <div key={key} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              {label}
            </span>
            <span style={{ fontSize: 12, color: '#e2b96f' }}>{fmt(params[key])}</span>
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
