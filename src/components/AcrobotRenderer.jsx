import { useEffect, useRef, useState } from 'react'
import { ACROBOT } from '../env/characters/acrobot.js'

const COLORS = {
  bg: '#07070f',
  ground: '#0f0f1e',
  groundLine: '#1e2040',
  gridLine: 'rgba(255,255,255,0.025)',
  torso:  { fill: '#ffffff', stroke: 'rgba(255,255,255,0.3)', glow: 'rgba(255,255,255,0.4)' },
  link1:  { fill: '#6fa8e2', stroke: 'rgba(255,255,255,0.15)' },
  link2:  { fill: '#4a8fd4', stroke: 'rgba(255,255,255,0.12)' },
  joint:  '#ff9966',
  contact: '#4ade80',
  pivot:  'rgba(255,255,255,0.6)',
  shadow: 'rgba(0,0,0,0.5)',
}

const SCALE = 130  // pixels per meter

function drawCapsule(ctx, cx, cy, angle, halfHeight, radius, colors) {
  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(-angle)

  const h = halfHeight * SCALE
  const r = radius * SCALE

  ctx.fillStyle = colors.fill
  ctx.strokeStyle = colors.stroke
  ctx.lineWidth = 1

  ctx.beginPath()
  ctx.roundRect(-r, -h - r, r * 2, (h + r) * 2, r)
  ctx.fill()
  ctx.stroke()

  ctx.restore()
}

function drawBox(ctx, cx, cy, angle, hw, hh, colors) {
  ctx.save()
  ctx.translate(cx, cy)
  ctx.rotate(-angle)

  const pw = hw * SCALE
  const ph = hh * SCALE

  ctx.fillStyle = colors.fill
  ctx.strokeStyle = colors.stroke
  ctx.lineWidth = 1

  ctx.shadowColor = colors.glow || 'transparent'
  ctx.shadowBlur = colors.glow ? 12 : 0

  ctx.beginPath()
  ctx.roundRect(-pw, -ph, pw * 2, ph * 2, 3)
  ctx.fill()
  ctx.stroke()

  ctx.shadowBlur = 0
  ctx.restore()
}

function drawJointDot(ctx, sx, sy) {
  ctx.fillStyle = COLORS.joint
  ctx.shadowColor = COLORS.joint
  ctx.shadowBlur = 8
  ctx.beginPath()
  ctx.arc(sx, sy, 4, 0, Math.PI * 2)
  ctx.fill()
  ctx.shadowBlur = 0
}

export default function AcrobotRenderer({ snapshot, episodeReward, episodeSteps }) {
  const canvasRef = useRef(null)
  const [canvasSize, setCanvasSize] = useState({ w: 500, h: 300 })

  // Keep canvas pixel dimensions in sync with its display size
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        if (width > 0 && height > 0) {
          canvas.width = Math.round(width)
          canvas.height = Math.round(height)
          setCanvasSize({ w: Math.round(width), h: Math.round(height) })
        }
      }
    })
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    // Anchor position determines the viewport center — no camera follow
    const camX = 0

    // Place the anchor pivot near 40% from the top so the pendulum has room to swing
    const PIVOT_SCREEN_Y = Math.round(H * 0.35)
    const ANCHOR_WORLD_Y = 2.0  // matches anchor spawnY

    // World -> canvas transform (pivot-centered)
    const ws = (wx, wy) => ({
      sx: W / 2 + (wx - camX) * SCALE,
      sy: PIVOT_SCREEN_Y + (ANCHOR_WORLD_Y - wy) * SCALE,
    })

    ctx.clearRect(0, 0, W, H)

    // Background
    ctx.fillStyle = COLORS.bg
    ctx.fillRect(0, 0, W, H)

    // Grid (world-aligned)
    ctx.strokeStyle = COLORS.gridLine
    ctx.lineWidth = 1
    const gridStep = 1.0
    for (let wx = -3; wx <= 3; wx += gridStep) {
      const { sx } = ws(wx, 0)
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, H); ctx.stroke()
    }
    for (let wy = -1; wy <= 5; wy += gridStep) {
      const { sy } = ws(0, wy)
      ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(W, sy); ctx.stroke()
    }

    if (!snapshot) {
      ctx.fillStyle = 'rgba(255,255,255,0.1)'
      ctx.font = '500 13px "DM Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText('Press TRAIN to begin', W / 2, H / 2)
      return
    }

    const def = ACROBOT

    // Draw pivot indicator (small cross at the anchor point)
    const anchor = snapshot.torso
    if (anchor) {
      const { sx, sy } = ws(anchor.x, anchor.y)
      ctx.strokeStyle = COLORS.pivot
      ctx.lineWidth = 2
      const armLen = 8
      ctx.beginPath()
      ctx.moveTo(sx - armLen, sy); ctx.lineTo(sx + armLen, sy)
      ctx.moveTo(sx, sy - armLen); ctx.lineTo(sx, sy + armLen)
      ctx.stroke()

      // Small filled circle at pivot center
      ctx.fillStyle = COLORS.pivot
      ctx.beginPath()
      ctx.arc(sx, sy, 3, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw bodies
    for (const bodyDef of def.bodies) {
      const t = snapshot[bodyDef.id]
      if (!t) continue
      const { sx, sy } = ws(t.x, t.y)
      const colors = COLORS[bodyDef.id] || COLORS.link2

      if (bodyDef.shape === 'box') {
        drawBox(ctx, sx, sy, t.angle, bodyDef.w / 2, bodyDef.h / 2, colors)
      } else if (bodyDef.shape === 'capsule') {
        const halfCyl = Math.max(0.001, (bodyDef.length - 2 * bodyDef.radius) / 2)
        drawCapsule(ctx, sx, sy, t.angle, halfCyl, bodyDef.radius, colors)
      }
    }

    // Draw joint positions
    for (const jointDef of def.joints) {
      const tA = snapshot[jointDef.bodyA]
      if (!tA) continue

      const [ax, ay] = jointDef.anchorA
      const cos = Math.cos(tA.angle)
      const sin = Math.sin(tA.angle)
      const jwx = tA.x + ax * cos - ay * sin
      const jwy = tA.y + ax * sin + ay * cos

      const { sx, sy } = ws(jwx, jwy)
      drawJointDot(ctx, sx, sy)
    }

    // Draw a faint "goal line" at the height threshold (minY = 1.5)
    {
      const { sy: goalY } = ws(0, 1.5)
      ctx.strokeStyle = 'rgba(255,100,100,0.15)'
      ctx.lineWidth = 1
      ctx.setLineDash([6, 6])
      ctx.beginPath()
      ctx.moveTo(0, goalY)
      ctx.lineTo(W, goalY)
      ctx.stroke()
      ctx.setLineDash([])

      ctx.fillStyle = 'rgba(255,100,100,0.2)'
      ctx.font = '500 9px "DM Mono", monospace'
      ctx.textAlign = 'right'
      ctx.fillText('min height', W - 8, goalY - 4)
    }

    // HUD
    ctx.font = '500 10px "DM Mono", monospace'
    ctx.textAlign = 'left'
    ctx.fillStyle = 'rgba(255,255,255,0.35)'
    if (snapshot.link2) {
      const h = snapshot.link2.y.toFixed(2)
      ctx.fillText(`tip h ${h}m`, 12, H - 28)
    }
    ctx.fillText(`t ${episodeSteps ?? 0}`, 12, H - 12)

    if (episodeReward !== undefined) {
      ctx.textAlign = 'right'
      ctx.fillStyle = episodeReward > 0 ? 'rgba(74,222,128,0.5)' : 'rgba(224,90,90,0.5)'
      ctx.fillText(`r ${(episodeReward || 0).toFixed(1)}`, W - 12, H - 12)
    }

  }, [snapshot, episodeReward, episodeSteps, canvasSize])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  )
}
