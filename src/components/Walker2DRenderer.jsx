import { useEffect, useRef, useState } from 'react'
import { WALKER2D } from '../env/characters/walker2d.js'

const COLORS = {
  bg: '#07070f',
  ground: '#0f0f1e',
  groundLine: '#1e2040',
  gridLine: 'rgba(255,255,255,0.025)',
  torso: { fill: '#e2b96f', stroke: 'rgba(255,255,255,0.15)', glow: 'rgba(226,185,111,0.3)' },
  // Left leg: blue-tinted gold
  left_thigh: { fill: '#a0906f', stroke: 'rgba(255,255,255,0.1)' },
  left_shin:  { fill: '#908060', stroke: 'rgba(255,255,255,0.08)' },
  // Right leg: original warm gold
  right_thigh: { fill: '#c8a05a', stroke: 'rgba(255,255,255,0.1)' },
  right_shin:  { fill: '#b08840', stroke: 'rgba(255,255,255,0.08)' },
  joint: '#ff9966',
  contact: '#4ade80',
  velocity: 'rgba(226,185,111,0.5)',
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

export default function Walker2DRenderer({ snapshot, episodeReward, episodeSteps }) {
  const canvasRef = useRef(null)
  const cameraXRef = useRef(0)
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

    // Dynamic ground position: 80% down the canvas
    const GSY = Math.round(H * 0.8)

    // World -> canvas transform (uses current canvas size)
    const ws = (wx, wy, camX = 0) => ({
      sx: W / 2 + (wx - camX) * SCALE,
      sy: GSY - wy * SCALE,
    })

    ctx.clearRect(0, 0, W, H)

    // Background
    ctx.fillStyle = COLORS.bg
    ctx.fillRect(0, 0, W, H)

    // Smooth camera follow torso x
    if (snapshot?.torso) {
      const targetCamX = snapshot.torso.x - 0.5
      cameraXRef.current += (targetCamX - cameraXRef.current) * 0.08
    }
    const camX = cameraXRef.current

    // Grid (world-aligned)
    ctx.strokeStyle = COLORS.gridLine
    ctx.lineWidth = 1
    const gridStep = 1.0
    const startX = Math.floor(camX - 3) * gridStep
    for (let wx = startX; wx < camX + 5; wx += gridStep) {
      const { sx } = ws(wx, 0, camX)
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, H); ctx.stroke()
    }
    for (let wy = 0; wy < 4; wy += gridStep) {
      const { sy } = ws(0, wy, camX)
      ctx.beginPath(); ctx.moveTo(0, sy); ctx.lineTo(W, sy); ctx.stroke()
    }

    // Ground
    const groundGrad = ctx.createLinearGradient(0, GSY, 0, H)
    groundGrad.addColorStop(0, '#131330')
    groundGrad.addColorStop(1, '#070714')
    ctx.fillStyle = groundGrad
    ctx.fillRect(0, GSY, W, H - GSY)

    ctx.strokeStyle = COLORS.groundLine
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, GSY)
    ctx.lineTo(W, GSY)
    ctx.stroke()

    // Ground tick marks
    ctx.strokeStyle = 'rgba(255,255,255,0.06)'
    ctx.lineWidth = 1
    for (let wx = startX; wx < camX + 5; wx += 0.5) {
      const { sx } = ws(wx, 0, camX)
      ctx.beginPath()
      ctx.moveTo(sx, GSY)
      ctx.lineTo(sx, GSY + 5)
      ctx.stroke()
    }

    if (!snapshot) {
      ctx.fillStyle = 'rgba(255,255,255,0.1)'
      ctx.font = '500 13px "DM Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText('Press TRAIN to begin', W / 2, H / 2)
      return
    }

    const def = WALKER2D

    // Draw shadows first
    for (const bodyDef of def.bodies) {
      const t = snapshot[bodyDef.id]
      if (!t) continue
      const { sx, sy: groundY } = ws(t.x, 0, camX)
      const { sy } = ws(t.x, t.y, camX)
      const shadowW = bodyDef.shape === 'box' ? bodyDef.w * SCALE * 0.7 : bodyDef.radius * SCALE * 2.5
      const shadowH = 6
      const opacity = Math.max(0, 0.4 - Math.abs(sy - groundY) * 0.003)
      ctx.fillStyle = `rgba(0,0,0,${opacity})`
      ctx.beginPath()
      ctx.ellipse(sx, groundY + 3, shadowW, shadowH, 0, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw bodies (left leg first so right leg renders on top)
    for (const bodyDef of def.bodies) {
      const t = snapshot[bodyDef.id]
      if (!t) continue
      const { sx, sy } = ws(t.x, t.y, camX)
      const colors = COLORS[bodyDef.id] || COLORS.right_shin

      if (bodyDef.shape === 'box') {
        drawBox(ctx, sx, sy, t.angle, bodyDef.w / 2, bodyDef.h / 2, colors)
      } else if (bodyDef.shape === 'capsule') {
        const halfCyl = Math.max(0.001, (bodyDef.length - 2 * bodyDef.radius) / 2)
        drawCapsule(ctx, sx, sy, t.angle, halfCyl, bodyDef.radius, colors)
      }
    }

    // Draw joint positions
    for (const jointDef of def.joints) {
      const bodyADef = def.bodies.find(b => b.id === jointDef.bodyA)
      const tA = snapshot[jointDef.bodyA]
      if (!tA || !bodyADef) continue

      const [ax, ay] = jointDef.anchorA
      const cos = Math.cos(tA.angle)
      const sin = Math.sin(tA.angle)
      const wx = tA.x + ax * cos - ay * sin
      const wy = tA.y + ax * sin + ay * cos

      const { sx, sy } = ws(wx, wy, camX)
      drawJointDot(ctx, sx, sy)
    }

    // Foot contact indicators
    const contacts = snapshot._footContacts || {}
    for (const [bodyId, inContact] of Object.entries(contacts)) {
      if (inContact && snapshot[bodyId]) {
        const { sx } = ws(snapshot[bodyId].x, 0.05, camX)
        ctx.fillStyle = COLORS.contact
        ctx.shadowColor = COLORS.contact
        ctx.shadowBlur = 10
        ctx.beginPath()
        ctx.arc(sx, GSY - 2, 5, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0
      }
    }

    // HUD
    ctx.font = '500 10px "DM Mono", monospace'
    ctx.textAlign = 'left'
    ctx.fillStyle = 'rgba(255,255,255,0.35)'
    if (snapshot.torso) {
      const h = snapshot.torso.y.toFixed(2)
      ctx.fillText(`h ${h}m`, 12, H - 28)
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
