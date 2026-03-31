import { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, ReferenceLine, Brush,
} from 'recharts'
import InfoTooltip from './Tooltip.jsx'

// Distinct colors for reward components
const COMPONENT_COLORS = {
  'Forward Vel': '#4ade80',
  'Alive Bonus': '#60a5fa',
  'Tip Height': '#c084fc',
  'Angle Penalty': '#f87171',
  'Position Penalty': '#fb923c',
  'Ctrl Cost': '#fbbf24',
  'Termination': '#ef4444',
}

const fallbackColors = ['#67e8f9', '#a78bfa', '#f472b6', '#34d399', '#fcd34d', '#818cf8']

const BreakdownTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'rgba(10,10,20,0.95)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 5,
      padding: '6px 10px',
      fontSize: 10,
      fontFamily: 'Inter, sans-serif',
    }}>
      <div style={{ color: 'rgba(255,255,255,0.65)', marginBottom: 4 }}>update {label}</div>
      {payload
        .filter(p => p.value != null)
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .map(p => (
          <div key={p.dataKey} style={{ color: p.color, display: 'flex', justifyContent: 'space-between', gap: 12 }}>
            <span>{p.name}</span>
            <span>{p.value >= 0 ? '+' : ''}{p.value.toFixed(3)}</span>
          </div>
        ))}
    </div>
  )
}

export default function RewardBreakdownChart({ metrics }) {
  // Extract all component keys that appear in any metric
  const { data, componentKeys } = useMemo(() => {
    const keys = new Set()
    const d = metrics.map(m => {
      const point = { update: m.update }
      if (m.rewardBreakdown) {
        for (const [label, value] of Object.entries(m.rewardBreakdown)) {
          keys.add(label)
          point[label] = +value.toFixed(4)
        }
      }
      return point
    })
    return { data: d, componentKeys: Array.from(keys) }
  }, [metrics])

  if (data.length === 0 || componentKeys.length === 0) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', color: 'rgba(255,255,255,0.12)',
        fontFamily: 'Inter, sans-serif', fontSize: 10,
      }}>
        waiting for reward data...
      </div>
    )
  }

  // Compute y-axis domain across all components
  const allValues = data.flatMap(d => componentKeys.map(k => d[k]).filter(v => v != null))
  const min = Math.min(0, ...allValues)
  const max = Math.max(0, ...allValues)
  const pad = (max - min) * 0.1 || 0.5

  let colorIdx = 0

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div style={{
        position: 'absolute', top: 6, left: 10, zIndex: 1,
        fontSize: 9, color: 'rgba(255,255,255,0.55)',
        fontFamily: 'Inter, sans-serif',
        letterSpacing: '0.06em', textTransform: 'uppercase',
        display: 'flex', alignItems: 'center',
      }}>
        Reward Components
        <InfoTooltip text={'Per-update average of each reward component (sampled from env[0]). Shows how the balance of incentives shifts as the agent learns.\n\nGreen lines = positive incentives. Red/orange lines = penalties. Use this to diagnose reward hacking — e.g., if alive bonus stays high but forward velocity stays flat, the agent may be standing still to survive.'} />
      </div>
      <div style={{ padding: '18px 2px 2px 2px', height: '100%', boxSizing: 'border-box' }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, bottom: 2, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
            <XAxis
              dataKey="update"
              tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 9, fontFamily: 'Inter, sans-serif' }}
              tickLine={false}
              axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
            />
            <YAxis
              tick={{ fill: 'rgba(255,255,255,0.55)', fontSize: 9, fontFamily: 'Inter, sans-serif' }}
              tickLine={false}
              axisLine={false}
              domain={[min - pad, max + pad]}
              width={40}
              tickFormatter={v => v.toFixed(Math.abs(v) < 1 ? 2 : 1)}
            />
            <Tooltip content={<BreakdownTooltip />} />
            <ReferenceLine y={0} stroke="rgba(255,255,255,0.08)" />
            <Brush
              dataKey="update"
              height={18}
              stroke="rgba(255,255,255,0.15)"
              fill="#0a0a14"
              travellerWidth={8}
              tickFormatter={() => ''}
            />
            {componentKeys.map(key => {
              const color = COMPONENT_COLORS[key] || fallbackColors[colorIdx++ % fallbackColors.length]
              return (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  name={key}
                  stroke={color}
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{ r: 2, fill: color }}
                  isAnimationActive={false}
                  connectNulls
                />
              )
            })}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
