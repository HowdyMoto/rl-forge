import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'rgba(10,10,20,0.95)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 6,
      padding: '8px 12px',
      fontSize: 11,
      fontFamily: '"DM Mono", monospace',
      color: '#aaa'
    }}>
      <div style={{ color: '#e2b96f', marginBottom: 2 }}>update {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
        </div>
      ))}
    </div>
  )
}

export default function RewardChart({ metrics, solvedThreshold = 500 }) {
  const data = useMemo(() => metrics.map((m, i) => ({
    update: m.update,
    reward: +m.meanReward20.toFixed(2),
    entropy: +(m.entropy * 10).toFixed(3),
  })), [metrics])

  const maxReward = Math.max(...data.map(d => d.reward), 1)
  const solved = maxReward >= 450

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {data.length === 0 ? (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          height: '100%', color: 'rgba(255,255,255,0.15)',
          fontFamily: '"DM Mono", monospace', fontSize: 12
        }}>
          waiting for data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <XAxis
              dataKey="update"
              tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10, fontFamily: '"DM Mono", monospace' }}
              tickLine={false}
              axisLine={{ stroke: 'rgba(255,255,255,0.08)' }}
            />
            <YAxis
              tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10, fontFamily: '"DM Mono", monospace' }}
              tickLine={false}
              axisLine={false}
              domain={[0, Math.max(solvedThreshold * 1.1, maxReward * 1.1)]}
              width={36}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine
              y={solvedThreshold}
              stroke="rgba(100,255,150,0.2)"
              strokeDasharray="4 4"
              label={{ value: 'target', fill: 'rgba(100,255,150,0.4)', fontSize: 10, fontFamily: '"DM Mono", monospace' }}
            />
            <Line
              type="monotone"
              dataKey="reward"
              name="mean reward (20ep)"
              stroke="#e2b96f"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 3, fill: '#e2b96f' }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
