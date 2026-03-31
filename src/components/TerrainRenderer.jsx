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

export default function TerrainRenderer({ snapshot, charDef, episodeReward, episodeSteps, resetEvent, onDebugMouse }) {
  const canvasRef = useRef(null)
  const cameraXRef = useRef(0)
  const cameraYRef = useRef(0)
  const [canvasSize, setCanvasSize] = useState({ w: 500, h: 300 })

  // Drag state for debug mode (refs so mouse handlers don't cause re-renders)
  const dragRef = useRef({ active: false, bodyId: null, wx: 0, wy: 0, prevWx: 0, prevWy: 0, prevTime: 0 })
  const snapshotRef = useRef(null)
  snapshotRef.current = snapshot

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

  // Mouse interaction for debug mode drag-and-fling
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !onDebugMouse || !charDef) return

    // Screen → world coordinate transform (inverse of ws)
    const sw = (sx, sy) => {
      const W = canvas.width
      const H = canvas.height
      const GSY = Math.round(H * 0.75)
      return {
        wx: cameraXRef.current + (sx - W / 2) / SCALE,
        wy: cameraYRef.current + (GSY - sy) / SCALE,
      }
    }

    // Find closest body to a world position
    const findBody = (wx, wy) => {
      const snap = snapshotRef.current
      if (!snap) return null
      let bestId = null
      let bestDist = 0.3  // max grab distance in meters
      for (const bodyDef of charDef.bodies) {
        if (bodyDef.fixed) continue
        const t = snap[bodyDef.id]
        if (!t) continue
        const dx = t.x - wx
        const dy = t.y - wy
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < bestDist) {
          bestDist = dist
          bestId = bodyDef.id
        }
      }
      return bestId
    }

    const getCanvasXY = (e) => {
      const rect = canvas.getBoundingClientRect()
      return {
        sx: (e.clientX - rect.left) * (canvas.width / rect.width),
        sy: (e.clientY - rect.top) * (canvas.height / rect.height),
      }
    }

    const onMouseDown = (e) => {
      const { sx, sy } = getCanvasXY(e)
      const { wx, wy } = sw(sx, sy)
      const bodyId = findBody(wx, wy)
      if (bodyId) {
        dragRef.current = { active: true, bodyId, wx, wy, prevWx: wx, prevWy: wy, prevTime: performance.now() }
        onDebugMouse({ type: 'grab', bodyId, wx, wy })
        canvas.style.cursor = 'grabbing'
      }
    }

    const onMouseMove = (e) => {
      if (!dragRef.current.active) {
        // Hover cursor
        const { sx, sy } = getCanvasXY(e)
        const { wx, wy } = sw(sx, sy)
        canvas.style.cursor = findBody(wx, wy) ? 'grab' : 'default'
        return
      }
      const { sx, sy } = getCanvasXY(e)
      const { wx, wy } = sw(sx, sy)
      const now = performance.now()
      dragRef.current.prevWx = dragRef.current.wx
      dragRef.current.prevWy = dragRef.current.wy
      dragRef.current.prevTime = now
      dragRef.current.wx = wx
      dragRef.current.wy = wy
      onDebugMouse({ type: 'drag', bodyId: dragRef.current.bodyId, wx, wy })
    }

    const onMouseUp = () => {
      if (!dragRef.current.active) return
      const d = dragRef.current
      const dt = Math.max(1, performance.now() - d.prevTime) / 1000
      const vx = (d.wx - d.prevWx) / dt
      const vy = (d.wy - d.prevWy) / dt
      // Clamp fling velocity so it doesn't go crazy
      const maxV = 15
      const clampedVx = Math.max(-maxV, Math.min(maxV, vx))
      const clampedVy = Math.max(-maxV, Math.min(maxV, vy))
      onDebugMouse({ type: 'release', bodyId: d.bodyId, vx: clampedVx, vy: clampedVy })
      dragRef.current = { active: false, bodyId: null, wx: 0, wy: 0, prevWx: 0, prevWy: 0, prevTime: 0 }
      canvas.style.cursor = 'default'
    }

    canvas.addEventListener('mousedown', onMouseDown)
    canvas.addEventListener('mousemove', onMouseMove)
    canvas.addEventListener('mouseup', onMouseUp)
    canvas.addEventListener('mouseleave', onMouseUp)

    return () => {
      canvas.removeEventListener('mousedown', onMouseDown)
      canvas.removeEventListener('mousemove', onMouseMove)
      canvas.removeEventListener('mouseup', onMouseUp)
      canvas.removeEventListener('mouseleave', onMouseUp)
    }
  }, [onDebugMouse, charDef])

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
      ctx.font = '500 13px Inter, sans-serif'
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
    ctx.font = '500 10px Inter, sans-serif'
    ctx.textAlign = 'left'
    ctx.fillStyle = 'rgba(255,255,255,0.65)'

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
      ctx.font = '600 11px Inter, sans-serif'
      ctx.fillStyle = '#e2b96f'
      ctx.fillText(`${torsoSnap.x.toFixed(1)}m`, W - 12, H - 28)
    }

    // ── Debug overlay ──────────────────────────────────────────────────────
    const debug = snapshot?._debug
    if (debug && charDef) {
      // Body labels
      ctx.font = '500 9px Inter, sans-serif'
      ctx.textAlign = 'center'
      for (const bl of (debug.bodyLabels || [])) {
        const t = snapshot[bl.id]
        if (!t) continue
        const { sx, sy } = ws(t.x, t.y)
        ctx.fillStyle = 'rgba(255,255,255,0.75)'
        ctx.fillText(bl.id, sx, sy - 18)
        ctx.fillStyle = 'rgba(255,255,255,0.55)'
        ctx.fillText(`${bl.mass}kg`, sx, sy - 8)
      }

      // Joint limit arcs + angle indicators
      for (const jDef of charDef.joints) {
        const jDebug = debug.joints?.[jDef.id]
        if (!jDebug) continue
        const tA = snapshot[jDef.bodyA]
        if (!tA) continue

        // Joint world position from bodyA anchor
        const [ax, ay] = jDef.anchorA
        const cos = Math.cos(tA.angle)
        const sin = Math.sin(tA.angle)
        const wx = tA.x + ax * cos - ay * sin
        const wy = tA.y + ax * sin + ay * cos
        const { sx, sy } = ws(wx, wy)

        const arcRadius = 28
        const isActive = debug.activeJoint === jDef.id

        // Reference angle = bodyA's world angle (arcs are relative to parent)
        const refAngle = tA.angle

        // Draw limit arc (screen coords: y is flipped, so negate angles)
        ctx.beginPath()
        ctx.arc(sx, sy, arcRadius,
          -(refAngle + jDebug.upper),
          -(refAngle + jDebug.lower),
          false
        )
        ctx.strokeStyle = isActive ? 'rgba(226,185,111,0.5)' : 'rgba(255,255,255,0.15)'
        ctx.lineWidth = isActive ? 2.5 : 1.5
        ctx.stroke()

        // Draw limit tick marks at ends
        for (const limitAngle of [jDebug.lower, jDebug.upper]) {
          const screenAngle = -(refAngle + limitAngle)
          const innerR = arcRadius - 4
          const outerR = arcRadius + 4
          ctx.beginPath()
          ctx.moveTo(sx + innerR * Math.cos(screenAngle), sy + innerR * Math.sin(screenAngle))
          ctx.lineTo(sx + outerR * Math.cos(screenAngle), sy + outerR * Math.sin(screenAngle))
          ctx.strokeStyle = isActive ? 'rgba(226,185,111,0.6)' : 'rgba(255,255,255,0.55)'
          ctx.lineWidth = 1.5
          ctx.stroke()
        }

        // Current angle indicator (line from center)
        const currentScreenAngle = -(refAngle + jDebug.angle)
        ctx.beginPath()
        ctx.moveTo(sx, sy)
        ctx.lineTo(
          sx + (arcRadius + 6) * Math.cos(currentScreenAngle),
          sy + (arcRadius + 6) * Math.sin(currentScreenAngle)
        )
        // Color based on how close to limits (green=center, red=near limit)
        const range = jDebug.upper - jDebug.lower
        const normalized = range > 0 ? (jDebug.angle - jDebug.lower) / range : 0.5
        const nearLimit = Math.min(normalized, 1 - normalized) * 2  // 0=at limit, 1=center
        const r = Math.round(255 * (1 - nearLimit))
        const g = Math.round(200 * nearLimit)
        ctx.strokeStyle = isActive
          ? `rgba(${r},${g},80,0.9)`
          : `rgba(${r},${g},80,0.5)`
        ctx.lineWidth = isActive ? 2.5 : 1.5
        ctx.stroke()

        // Angle readout
        const angleDeg = (jDebug.angle * 180 / Math.PI).toFixed(0)
        const lowerDeg = (jDebug.lower * 180 / Math.PI).toFixed(0)
        const upperDeg = (jDebug.upper * 180 / Math.PI).toFixed(0)
        ctx.font = '500 8px Inter, sans-serif'
        ctx.textAlign = 'left'
        ctx.fillStyle = isActive ? 'rgba(226,185,111,0.8)' : 'rgba(255,255,255,0.6)'
        ctx.fillText(`${angleDeg}° [${lowerDeg},${upperDeg}]`, sx + arcRadius + 10, sy - 2)
        if (isActive) {
          ctx.fillStyle = 'rgba(226,185,111,0.5)'
          ctx.fillText(`τmax ${jDebug.maxTorque}`, sx + arcRadius + 10, sy + 9)
          ctx.fillText(`kp${jDebug.kp} kd${jDebug.kd}`, sx + arcRadius + 10, sy + 20)
        }

        // Torque direction arrow when actively testing
        if (isActive && debug.direction !== 0) {
          const arrowAngle = currentScreenAngle + (debug.direction > 0 ? -0.4 : 0.4)
          const arrowR = arcRadius + 14
          const tipX = sx + arrowR * Math.cos(arrowAngle)
          const tipY = sy + arrowR * Math.sin(arrowAngle)

          ctx.beginPath()
          ctx.arc(sx, sy, arcRadius + 10, currentScreenAngle,
            currentScreenAngle + (debug.direction > 0 ? -0.5 : 0.5), debug.direction > 0)
          ctx.strokeStyle = debug.direction > 0 ? 'rgba(74,222,128,0.7)' : 'rgba(224,90,90,0.7)'
          ctx.lineWidth = 2.5
          ctx.stroke()

          // Arrowhead
          ctx.beginPath()
          ctx.arc(tipX, tipY, 3, 0, Math.PI * 2)
          ctx.fillStyle = debug.direction > 0 ? 'rgba(74,222,128,0.8)' : 'rgba(224,90,90,0.8)'
          ctx.fill()
        }
      }

      // Drag spring line
      const drag = dragRef.current
      if (drag.active && snapshot[drag.bodyId]) {
        const bodySnap = snapshot[drag.bodyId]
        const bodyScreen = ws(bodySnap.x, bodySnap.y)
        const mouseScreen = ws(drag.wx, drag.wy)

        // Spring line
        ctx.beginPath()
        ctx.moveTo(bodyScreen.sx, bodyScreen.sy)
        ctx.lineTo(mouseScreen.sx, mouseScreen.sy)
        ctx.strokeStyle = 'rgba(226,185,111,0.6)'
        ctx.lineWidth = 2
        ctx.setLineDash([4, 4])
        ctx.stroke()
        ctx.setLineDash([])

        // Mouse cursor dot
        ctx.beginPath()
        ctx.arc(mouseScreen.sx, mouseScreen.sy, 6, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(226,185,111,0.4)'
        ctx.fill()
        ctx.strokeStyle = 'rgba(226,185,111,0.7)'
        ctx.lineWidth = 1.5
        ctx.stroke()

        // Grabbed body highlight
        ctx.beginPath()
        ctx.arc(bodyScreen.sx, bodyScreen.sy, 10, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(226,185,111,0.5)'
        ctx.lineWidth = 2
        ctx.stroke()
      }

      // Debug mode label
      ctx.font = '600 11px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillStyle = 'rgba(226,185,111,0.6)'
      ctx.fillText('PHYSICS DEBUG', W / 2, 20)
      const debugLabel = drag.active ? `dragging: ${drag.bodyId}`
        : debug.activeJoint ? `testing: ${debug.activeJoint}`
        : 'click body to drag'
      ctx.font = '500 10px Inter, sans-serif'
      ctx.fillStyle = 'rgba(226,185,111,0.4)'
      ctx.fillText(debugLabel, W / 2, 34)
    }

    // Episode reset banner (fades out over 2s)
    if (resetEvent) {
      const elapsed = performance.now() - resetEvent.time
      const BANNER_DURATION = 2000
      if (elapsed < BANNER_DURATION) {
        const alpha = Math.max(0, 1 - elapsed / BANNER_DURATION)
        const isTimeout = resetEvent.reason === 'timeout'
        const reasonLabels = {
          angle: 'pole angle exceeded',
          position: 'cart position exceeded',
          fell: 'agent fell',
          timeout: 'episode complete',
        }
        const label = isTimeout ? 'EPISODE COMPLETE' : 'RESET'
        const detail = reasonLabels[resetEvent.reason] || resetEvent.reason
        const color = isTimeout ? [74, 222, 128] : [224, 90, 90]
        const [cr, cg, cb] = color

        // Background pill
        ctx.fillStyle = `rgba(${cr}, ${cg}, ${cb}, ${alpha * 0.15})`
        const pillW = 260, pillH = 44, pillX = (W - pillW) / 2, pillY = 50
        ctx.beginPath()
        ctx.roundRect(pillX, pillY, pillW, pillH, 8)
        ctx.fill()
        ctx.strokeStyle = `rgba(${cr}, ${cg}, ${cb}, ${alpha * 0.4})`
        ctx.lineWidth = 1
        ctx.stroke()

        // Main label
        ctx.textAlign = 'center'
        ctx.font = '600 12px Inter, sans-serif'
        ctx.fillStyle = `rgba(${cr}, ${cg}, ${cb}, ${alpha})`
        ctx.fillText(`${label} \u2014 ${detail}`, W / 2, pillY + 18)

        // Episode summary
        ctx.font = '500 10px Inter, sans-serif'
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha * 0.6})`
        ctx.fillText(`r ${resetEvent.reward.toFixed(1)}  \u00b7  ${resetEvent.steps} steps`, W / 2, pillY + 34)
      }
    }

  }, [snapshot, charDef, episodeReward, episodeSteps, canvasSize, resetEvent])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  )
}
