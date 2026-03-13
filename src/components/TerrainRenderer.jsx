/**
 * TerrainRenderer — Universal canvas renderer for any creature on procedural terrain.
 *
 * Draws:
 *   - Procedural terrain (from segment data in the render snapshot)
 *   - Any creature body (reads body shapes from the character definition)
 *   - Joints, foot contacts, HUD with distance/height/reward
 *   - Camera follows the creature with smooth tracking
 */

import { useEffect, useRef, useState } from 'react'

const SCALE = 130  // pixels per meter

const COLORS = {
  bg: '#07070f',
  terrain: {
    fill: '#131330',
    stroke: '#2a2a5a',
    accent: '#1e2040',
  },
  body: {
    torso: { fill: '#e2b96f', stroke: 'rgba(255,255,255,0.15)', glow: 'rgba(226,185,111,0.3)' },
    limb: { fill: '#c8a05a', stroke: 'rgba(255,255,255,0.1)' },
    foot: { fill: '#b08840', stroke: 'rgba(255,255,255,0.08)' },
    default: { fill: '#a08050', stroke: 'rgba(255,255,255,0.1)' },
  },
  joint: '#ff9966',
  contact: '#4ade80',
  distance: '#e2b96f',
}

function getBodyColor(bodyDef) {
  if (bodyDef.id === 'torso') return COLORS.body.torso
  if (bodyDef.isFootBody) return COLORS.body.foot
  // Alternate colors for left/right differentiation
  if (bodyDef.id.includes('left')) return { fill: '#a0906f', stroke: 'rgba(255,255,255,0.1)' }
  if (bodyDef.id.includes('right')) return { fill: '#c8a05a', stroke: 'rgba(255,255,255,0.1)' }
  if (bodyDef.id === 'head') return { fill: '#d4a85c', stroke: 'rgba(255,255,255,0.12)', glow: 'rgba(226,185,111,0.2)' }
  if (bodyDef.id === 'tail') return { fill: '#9a7840', stroke: 'rgba(255,255,255,0.08)' }
  return COLORS.body.default
}

function drawShape(ctx, sx, sy, angle, bodyDef, colors) {
  ctx.save()
  ctx.translate(sx, sy)
  ctx.rotate(-angle)

  ctx.fillStyle = colors.fill
  ctx.strokeStyle = colors.stroke
  ctx.lineWidth = 1

  if (colors.glow) {
    ctx.shadowColor = colors.glow
    ctx.shadowBlur = 12
  }

  if (bodyDef.shape === 'box') {
    const pw = (bodyDef.w / 2) * SCALE
    const ph = (bodyDef.h / 2) * SCALE
    ctx.beginPath()
    ctx.roundRect(-pw, -ph, pw * 2, ph * 2, 3)
    ctx.fill()
    ctx.stroke()
  } else if (bodyDef.shape === 'capsule') {
    const r = (bodyDef.radius || 0.04) * SCALE
    const halfH = Math.max(0.001, ((bodyDef.length || 0.3) - 2 * (bodyDef.radius || 0.04)) / 2) * SCALE
    ctx.beginPath()
    ctx.roundRect(-r, -halfH - r, r * 2, (halfH + r) * 2, r)
    ctx.fill()
    ctx.stroke()
  } else if (bodyDef.shape === 'ball') {
    const r = (bodyDef.radius || 0.05) * SCALE
    ctx.beginPath()
    ctx.arc(0, 0, r, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
  }

  ctx.shadowBlur = 0
  ctx.restore()
}

export default function TerrainRenderer({ snapshot, charDef, episodeReward, episodeSteps }) {
  const canvasRef = useRef(null)
  const cameraXRef = useRef(0)
  const cameraYRef = useRef(0)
  const [canvasSize, setCanvasSize] = useState({ w: 500, h: 300 })

  // Canvas resize observer
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
    if (!canvas || !charDef) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = COLORS.bg
    ctx.fillRect(0, 0, W, H)

    // Find torso for camera
    const forwardBodyId = charDef.forwardBody || 'torso'
    const torsoSnap = snapshot?.[forwardBodyId]

    // Smooth camera follow
    if (torsoSnap) {
      const targetCamX = torsoSnap.x - 0.5
      const targetCamY = torsoSnap.y - 1.0
      cameraXRef.current += (targetCamX - cameraXRef.current) * 0.08
      cameraYRef.current += (targetCamY - cameraYRef.current) * 0.05
    }
    const camX = cameraXRef.current
    const camY = cameraYRef.current

    // World → screen transform (camera-relative)
    const GSY = Math.round(H * 0.75)
    const ws = (wx, wy) => ({
      sx: W / 2 + (wx - camX) * SCALE,
      sy: GSY - (wy - camY) * SCALE,
    })

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.02)'
    ctx.lineWidth = 1
    const startX = Math.floor(camX - 5)
    for (let wx = startX; wx < camX + 8; wx += 1.0) {
      const { sx } = ws(wx, 0)
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, H); ctx.stroke()
    }

    // Draw terrain
    const terrain = snapshot?._terrain
    if (terrain?.segments) {
      // Fill terrain body
      ctx.fillStyle = COLORS.terrain.fill
      ctx.strokeStyle = COLORS.terrain.stroke
      ctx.lineWidth = 2

      // Draw each segment as a filled polygon reaching down to the bottom
      for (const seg of terrain.segments) {
        const p1 = ws(seg.x1, seg.y1)
        const p2 = ws(seg.x2, seg.y2)

        // Skip segments far off screen
        if (p1.sx > W + 50 && p2.sx > W + 50) continue
        if (p1.sx < -50 && p2.sx < -50) continue

        const bottom = ws(0, camY - 3).sy

        // Filled terrain body
        ctx.beginPath()
        ctx.moveTo(p1.sx, p1.sy)
        ctx.lineTo(p2.sx, p2.sy)
        ctx.lineTo(p2.sx, bottom)
        ctx.lineTo(p1.sx, bottom)
        ctx.closePath()
        ctx.fill()

        // Terrain surface line
        ctx.beginPath()
        ctx.moveTo(p1.sx, p1.sy)
        ctx.lineTo(p2.sx, p2.sy)
        ctx.stroke()
      }

      // Surface highlights
      ctx.strokeStyle = 'rgba(42,42,90,0.5)'
      ctx.lineWidth = 1
      for (const seg of terrain.segments) {
        const p1 = ws(seg.x1, seg.y1)
        const p2 = ws(seg.x2, seg.y2)
        if (p1.sx > W + 50 || p2.sx < -50) continue
        ctx.beginPath()
        ctx.moveTo(p1.sx, p1.sy - 1)
        ctx.lineTo(p2.sx, p2.sy - 1)
        ctx.stroke()
      }
    }

    if (!snapshot) {
      ctx.fillStyle = 'rgba(255,255,255,0.1)'
      ctx.font = '500 13px "DM Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText('Build a creature, then press TRAIN', W / 2, H / 2)
      return
    }

    // Draw shadows
    for (const bodyDef of charDef.bodies) {
      if (bodyDef.fixed) continue
      const t = snapshot[bodyDef.id]
      if (!t) continue
      const { sx } = ws(t.x, t.y)
      // Find ground height at body's x
      const groundY = terrain?.segments
        ? (() => {
            for (const seg of terrain.segments) {
              if (t.x >= seg.x1 && t.x <= seg.x2) {
                const tFrac = Math.abs(seg.x2 - seg.x1) < 0.001 ? 0 : (t.x - seg.x1) / (seg.x2 - seg.x1)
                return seg.y1 + tFrac * (seg.y2 - seg.y1)
              }
            }
            return 0
          })()
        : 0
      const groundScreen = ws(t.x, groundY)
      const bodyScreen = ws(t.x, t.y)
      const shadowW = bodyDef.shape === 'box' ? (bodyDef.w || 0.2) * SCALE * 0.7 : (bodyDef.radius || 0.04) * SCALE * 2.5
      const opacity = Math.max(0, 0.4 - Math.abs(bodyScreen.sy - groundScreen.sy) * 0.003)
      ctx.fillStyle = `rgba(0,0,0,${opacity})`
      ctx.beginPath()
      ctx.ellipse(sx, groundScreen.sy + 3, shadowW, 5, 0, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw bodies
    for (const bodyDef of charDef.bodies) {
      if (bodyDef.fixed) continue
      const t = snapshot[bodyDef.id]
      if (!t) continue
      const { sx, sy } = ws(t.x, t.y)
      drawShape(ctx, sx, sy, t.angle, bodyDef, getBodyColor(bodyDef))
    }

    // Draw joints
    for (const jointDef of charDef.joints) {
      const tA = snapshot[jointDef.bodyA]
      if (!tA) continue
      const [ax, ay] = jointDef.anchorA
      const cos = Math.cos(tA.angle)
      const sin = Math.sin(tA.angle)
      const wx = tA.x + ax * cos - ay * sin
      const wy = tA.y + ax * sin + ay * cos
      const { sx, sy } = ws(wx, wy)

      ctx.fillStyle = COLORS.joint
      ctx.shadowColor = COLORS.joint
      ctx.shadowBlur = 8
      ctx.beginPath()
      ctx.arc(sx, sy, 3, 0, Math.PI * 2)
      ctx.fill()
      ctx.shadowBlur = 0
    }

    // Foot contacts
    const contacts = snapshot._footContacts || {}
    for (const [bodyId, inContact] of Object.entries(contacts)) {
      if (inContact && snapshot[bodyId]) {
        const { sx, sy } = ws(snapshot[bodyId].x, snapshot[bodyId].y)
        ctx.fillStyle = COLORS.contact
        ctx.shadowColor = COLORS.contact
        ctx.shadowBlur = 10
        ctx.beginPath()
        ctx.arc(sx, sy + 8, 4, 0, Math.PI * 2)
        ctx.fill()
        ctx.shadowBlur = 0
      }
    }

    // HUD
    ctx.font = '500 10px "DM Mono", monospace'
    ctx.textAlign = 'left'
    ctx.fillStyle = 'rgba(255,255,255,0.35)'

    if (torsoSnap) {
      ctx.fillText(`dist ${torsoSnap.x.toFixed(1)}m`, 12, H - 42)
      ctx.fillText(`best ${(snapshot._maxDistance || 0).toFixed(1)}m`, 12, H - 28)
    }
    ctx.fillText(`t ${episodeSteps ?? 0}`, 12, H - 12)

    if (episodeReward !== undefined) {
      ctx.textAlign = 'right'
      ctx.fillStyle = episodeReward > 0 ? 'rgba(74,222,128,0.5)' : 'rgba(224,90,90,0.5)'
      ctx.fillText(`r ${(episodeReward || 0).toFixed(1)}`, W - 12, H - 12)
    }

    // Distance marker
    if (torsoSnap) {
      ctx.textAlign = 'right'
      ctx.fillStyle = 'var(--gold)'
      ctx.font = '600 11px "DM Mono", monospace'
      ctx.fillStyle = '#e2b96f'
      ctx.fillText(`${torsoSnap.x.toFixed(1)}m`, W - 12, H - 28)
    }

  }, [snapshot, charDef, episodeReward, episodeSteps, canvasSize])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  )
}
