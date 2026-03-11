import { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine, CartesianGrid,
} from 'recharts'

const CHART_DEFS = [
  { key: 'meanReward20', label: 'Mean Reward (20 ep)', color: '#e2b96f', refLine: 'solvedThreshold' },
  { key: 'policyLoss', label: 'Policy Loss', color: '#7c9bf5' },
  { key: 'valueLoss', label: 'Value Loss', color: '#e06888' },
  { key: 'entropy', label: 'Entropy', color: '#5ed8a5' },
]

const MiniTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'rgba(10,10,20,0.95)',
      border: '1px solid rgba(255,255,255,0.1)',
      borderRadius: 5,
      padding: '6px 10px',
      fontSize: 10,
      fontFamily: '"DM Mono", monospace',
    }}>
      <div style={{ color: 'rgba(255,255,255,0.35)', marginBottom: 2 }}>update {label}</div>
      {payload.map(p => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {typeof p.value === 'number' ? p.value.toFixed(4) : p.value}
        </div>
      ))}
    </div>
  )
}

function SmallChart({ data, dataKey, label, color, solvedThreshold }) {
  if (data.length === 0) {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100%', color: 'rgba(255,255,255,0.12)',
        fontFamily: '"DM Mono", monospace', fontSize: 10,
      }}>
        waiting...
      </div>
    )
  }

  const values = data.map(d => d[dataKey]).filter(v => v != null)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const pad = (max - min) * 0.1 || 1
  const domainMin = dataKey === 'meanReward20' ? Math.min(0, min - pad) : min - pad
  const domainMax = solvedThreshold
    ? Math.max(max + pad, solvedThreshold * 1.1)
    : max + pad

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 4, right: 8, bottom: 2, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
        <XAxis
          dataKey="update"
          tick={{ fill: 'rgba(255,255,255,0.2)', fontSize: 9, fontFamily: '"DM Mono", monospace' }}
          tickLine={false}
          axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
        />
        <YAxis
          tick={{ fill: 'rgba(255,255,255,0.2)', fontSize: 9, fontFamily: '"DM Mono", monospace' }}
          tickLine={false}
          axisLine={false}
          domain={[domainMin, domainMax]}
          width={40}
          tickFormatter={v => Math.abs(v) >= 1000 ? (v / 1000).toFixed(0) + 'k' : v.toFixed(Math.abs(v) < 1 ? 3 : 1)}
        />
        <Tooltip content={<MiniTooltip />} />
        {solvedThreshold && (
          <ReferenceLine
            y={solvedThreshold}
            stroke="rgba(100,255,150,0.2)"
            strokeDasharray="4 4"
            label={{ value: 'target', fill: 'rgba(100,255,150,0.3)', fontSize: 9, fontFamily: '"DM Mono", monospace' }}
          />
        )}
        <Line
          type="monotone"
          dataKey={dataKey}
          name={label}
          stroke={color}
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 2, fill: color }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}

export default function TrainingCharts({ metrics, solvedThreshold }) {
  const data = useMemo(() => metrics.map(m => ({
    update: m.update,
    meanReward20: m.meanReward20,
    policyLoss: m.policyLoss,
    valueLoss: m.valueLoss,
    entropy: m.entropy,
  })), [metrics])

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      gridTemplateRows: '1fr 1fr',
      gap: 1,
      height: '100%',
      background: 'rgba(255,255,255,0.04)',
    }}>
      {CHART_DEFS.map(({ key, label, color, refLine }) => (
        <div key={key} style={{ background: '#07070f', position: 'relative', minHeight: 0 }}>
          <div style={{
            position: 'absolute', top: 6, left: 10, zIndex: 1,
            fontSize: 9, color: 'rgba(255,255,255,0.25)',
            fontFamily: '"DM Mono", monospace',
            letterSpacing: '0.06em', textTransform: 'uppercase',
          }}>
            {label}
          </div>
          <div style={{ padding: '18px 2px 2px 2px', height: '100%', boxSizing: 'border-box' }}>
            <SmallChart
              data={data}
              dataKey={key}
              label={label}
              color={color}
              solvedThreshold={refLine ? solvedThreshold : null}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
