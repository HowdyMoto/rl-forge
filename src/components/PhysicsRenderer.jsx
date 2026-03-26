/**
 * PhysicsRenderer — Universal canvas renderer for any Rapier2D body.
 *
 * Renders any body topology from a render snapshot produced by UnifiedRapierEnv.
 * Handles terrain and flat ground, debug overlays, mouse interaction.
 *
 * The charDef is read from snapshot._charDef, so no external body definition needed.
 */

import { useEffect, useRef, useState } from 'react'

const SCALE = 130  // pixels per meter

const COLORS = {
  bg: '#07070f',
  ground: { fill: '#131330', stroke: '#2a2a5a' },
  terrain: { fill: '#131330', stroke: '#2a2a5a' },
  body: {
    torso: { fill: '#e2b96f', stroke: 'rgba(255,255,255,0.15)', glow: 'rgba(226,185,111,0.3)' },
    limb: { fill: '#c8a05a', stroke: 'rgba(255,255,255,0.1)' },
    foot: { fill: '#b08840', stroke: 'rgba(255,255,255,0.08)' },
    fixed: { fill: 'rgba(255,255,255,0.1)', stroke: 'rgba(255,255,255,0.05)' },
    default: { fill: '#a08050', stroke: 'rgba(255,255,255,0.1)' },
  },
  joint: '#ff9966',
  prismatic: '#66aaff',
  contact: '#4ade80',
}

function getBodyColor(bodyDef) {
  if (bodyDef.fixed) return COLORS.body.fixed
  if (bodyDef.id === 'torso' || bodyDef.id === 'cart') return COLORS.body.torso
  if (bodyDef.isFootBody) return COLORS.body.foot
  if (bodyDef.id.includes('left')) return { fill: '#a0906f', stroke: 'rgba(255,255,255,0.1)' }
  if (bodyDef.id.includes('right')) return { fill: '#c8a05a', stroke: 'rgba(255,255,255,0.1)' }
  if (bodyDef.id === 'pole') return { fill: '#c8a05a', stroke: 'rgba(255,255,255,0.12)' }
  if (bodyDef.id === 'head') return { fill: '#d4a85c', stroke: 'rgba(255,255,255,0.12)', glow: 'rgba(226,185,111,0.2)' }
  if (bodyDef.id === 'tail') return { fill: '#9a7840', stroke: 'rgba(255,255,255,0.08)' }
  return COLORS.body.default
}

function drawShape(ctx, sx, sy, angle, bodyDef, colors, scale) {
  const S = scale || DEFAULT_SCALE
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
    const pw = (bodyDef.w / 2) * S
    const ph = (bodyDef.h / 2) * S
    ctx.beginPath()
    ctx.roundRect(-pw, -ph, pw * 2, ph * 2, 3)
    ctx.fill()
    ctx.stroke()
  } else if (bodyDef.shape === 'capsule') {
    const r = (bodyDef.radius || 0.04) * S
    const halfH = Math.max(0.001, ((bodyDef.length || 0.3) - 2 * (bodyDef.radius || 0.04)) / 2) * S
    ctx.beginPath()
    ctx.roundRect(-r, -halfH - r, r * 2, (halfH + r) * 2, r)
    ctx.fill()
    ctx.stroke()
  } else if (bodyDef.shape === 'ball') {
    const r = (bodyDef.radius || 0.05) * S
    ctx.beginPath()
    ctx.arc(0, 0, r, 0, Math.PI * 2)
    ctx.fill()
    ctx.stroke()
  }

  ctx.shadowBlur = 0
  ctx.restore()
}

const DEFAULT_SCALE = 130
const MIN_SCALE = 40
const MAX_SCALE = 500
const ZOOM_SPEED = 1.1

export default function PhysicsRenderer({ snapshot, episodeReward, episodeSteps, onDebugMouse, autoFollow = true, highlightBodyId = null, onBodyClick = null }) {
  const canvasRef = useRef(null)
  const cameraXRef = useRef(0)
  const cameraYRef = useRef(0)
  const scaleRef = useRef(DEFAULT_SCALE)
  const [canvasSize, setCanvasSize] = useState({ w: 500, h: 300 })

  // Pan state (right-click or middle-click drag)
  const panRef = useRef({ active: false, startSx: 0, startSy: 0, startCamX: 0, startCamY: 0 })
  const userPannedRef = useRef(false) // disable auto-follow after manual pan

  // Contact splat particles: { id, x, y, time, particles: [{dx, dy, speed, angle}] }
  const splatRef = useRef([])
  const prevContactsRef = useRef({})

  // Drag state for debug mode
  const dragRef = useRef({ active: false, bodyId: null, wx: 0, wy: 0, prevWx: 0, prevWy: 0, prevTime: 0 })
  // Click-to-select tracking (distinguish click from drag)
  const clickStartRef = useRef({ sx: 0, sy: 0 })
  const onBodyClickRef = useRef(onBodyClick)
  onBodyClickRef.current = onBodyClick
  const snapshotRef = useRef(null)
  snapshotRef.current = snapshot

  const charDef = snapshot?._charDef

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

  // Zoom + pan + drag interaction
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const screenToWorld = (sx, sy) => {
      const W = canvas.width
      const H = canvas.height
      const GSY = Math.round(H * 0.75)
      const s = scaleRef.current
      return {
        wx: cameraXRef.current + (sx - W / 2) / s,
        wy: cameraYRef.current + (GSY - sy) / s,
      }
    }

    const findBody = (wx, wy) => {
      const snap = snapshotRef.current
      if (!snap?._charDef) return null
      let bestId = null
      let bestDist = 0.3

      // Check character bodies
      for (const bodyDef of snap._charDef.bodies) {
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

      // Check props
      const props = snap._debug?.props
      if (props) {
        for (const prop of props) {
          const dx = prop.x - wx
          const dy = prop.y - wy
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < bestDist) {
            bestDist = dist
            bestId = prop.id
          }
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

    // ── Scroll wheel: zoom ──
    const onWheel = (e) => {
      e.preventDefault()
      const { sx, sy } = getCanvasXY(e)
      const worldBefore = screenToWorld(sx, sy)

      // Zoom in or out
      const factor = e.deltaY < 0 ? ZOOM_SPEED : 1 / ZOOM_SPEED
      scaleRef.current = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scaleRef.current * factor))

      // Adjust camera so the world point under the cursor stays fixed
      const W = canvas.width
      const H = canvas.height
      const GSY = Math.round(H * 0.75)
      const s = scaleRef.current
      cameraXRef.current = worldBefore.wx - (sx - W / 2) / s
      cameraYRef.current = worldBefore.wy - (GSY - sy) / s
      userPannedRef.current = true
    }

    // ── Mouse down: right/middle = pan, left = drag body ──
    const onMouseDown = (e) => {
      const { sx, sy } = getCanvasXY(e)
      clickStartRef.current = { sx, sy }

      // Right-click or middle-click: pan
      if (e.button === 1 || e.button === 2) {
        e.preventDefault()
        panRef.current = { active: true, startSx: sx, startSy: sy, startCamX: cameraXRef.current, startCamY: cameraYRef.current }
        canvas.style.cursor = 'move'
        return
      }

      // Left-click: grab body (if debug mode)
      if (onDebugMouse) {
        const { wx, wy } = screenToWorld(sx, sy)
        const bodyId = findBody(wx, wy)
        if (bodyId) {
          dragRef.current = { active: true, bodyId, wx, wy, prevWx: wx, prevWy: wy, prevTime: performance.now() }
          onDebugMouse({ type: 'grab', bodyId, wx, wy })
          canvas.style.cursor = 'grabbing'
        }
      }
    }

    const onMouseMove = (e) => {
      const { sx, sy } = getCanvasXY(e)

      // Pan
      if (panRef.current.active) {
        const s = scaleRef.current
        cameraXRef.current = panRef.current.startCamX - (sx - panRef.current.startSx) / s
        cameraYRef.current = panRef.current.startCamY + (sy - panRef.current.startSy) / s
        userPannedRef.current = true
        return
      }

      // Drag body
      if (dragRef.current.active) {
        const { wx, wy } = screenToWorld(sx, sy)
        const now = performance.now()
        dragRef.current.prevWx = dragRef.current.wx
        dragRef.current.prevWy = dragRef.current.wy
        dragRef.current.prevTime = now
        dragRef.current.wx = wx
        dragRef.current.wy = wy
        if (onDebugMouse) onDebugMouse({ type: 'drag', bodyId: dragRef.current.bodyId, wx, wy })
        return
      }

      // Hover cursor
      if (onDebugMouse) {
        const { wx, wy } = screenToWorld(sx, sy)
        canvas.style.cursor = findBody(wx, wy) ? 'grab' : 'default'
      }
    }

    const onMouseUp = (e) => {
      // Pan release
      if (panRef.current.active) {
        panRef.current.active = false
        canvas.style.cursor = 'default'
        return
      }

      // Body drag release
      if (dragRef.current.active && onDebugMouse) {
        const d = dragRef.current
        const dt = Math.max(1, performance.now() - d.prevTime) / 1000
        const vx = (d.wx - d.prevWx) / dt
        const vy = (d.wy - d.prevWy) / dt
        const maxV = 15
        onDebugMouse({ type: 'release', bodyId: d.bodyId, vx: Math.max(-maxV, Math.min(maxV, vx)), vy: Math.max(-maxV, Math.min(maxV, vy)) })
        dragRef.current = { active: false, bodyId: null, wx: 0, wy: 0, prevWx: 0, prevWy: 0, prevTime: 0 }
        canvas.style.cursor = 'default'
        return
      }

      // Click-to-select (left button, no drag)
      if (e.button === 0 && onBodyClickRef.current) {
        const { sx, sy } = getCanvasXY(e)
        const dx = sx - clickStartRef.current.sx
        const dy = sy - clickStartRef.current.sy
        if (dx * dx + dy * dy < 25) { // less than 5px movement = click
          const { wx, wy } = screenToWorld(sx, sy)
          const bodyId = findBody(wx, wy)
          onBodyClickRef.current(bodyId) // null clears selection
        }
      }
    }

    const onContextMenu = (e) => e.preventDefault()

    // Double-click: reset zoom and re-enable auto-follow
    const onDblClick = () => {
      scaleRef.current = DEFAULT_SCALE
      userPannedRef.current = false
    }

    canvas.addEventListener('wheel', onWheel, { passive: false })
    canvas.addEventListener('mousedown', onMouseDown)
    canvas.addEventListener('mousemove', onMouseMove)
    canvas.addEventListener('mouseup', onMouseUp)
    canvas.addEventListener('mouseleave', onMouseUp)
    canvas.addEventListener('contextmenu', onContextMenu)
    canvas.addEventListener('dblclick', onDblClick)

    return () => {
      canvas.removeEventListener('wheel', onWheel)
      canvas.removeEventListener('mousedown', onMouseDown)
      canvas.removeEventListener('mousemove', onMouseMove)
      canvas.removeEventListener('mouseup', onMouseUp)
      canvas.removeEventListener('mouseleave', onMouseUp)
      canvas.removeEventListener('contextmenu', onContextMenu)
      canvas.removeEventListener('dblclick', onDblClick)
    }
  }, [onDebugMouse, charDef])

  // Main render loop
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    ctx.clearRect(0, 0, W, H)
    ctx.fillStyle = COLORS.bg
    ctx.fillRect(0, 0, W, H)

    if (!charDef) {
      ctx.fillStyle = 'rgba(255,255,255,0.1)'
      ctx.font = '500 13px "DM Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText('Waiting for simulation...', W / 2, H / 2)
      return
    }

    // Current zoom level
    const S = scaleRef.current

    // Find forward body for camera
    const forwardBodyId = charDef.forwardBody || 'torso'
    const torsoSnap = snapshot?.[forwardBodyId]

    // Smooth camera follow (disabled if user has panned/zoomed manually)
    if (torsoSnap && autoFollow && !userPannedRef.current) {
      const targetCamX = torsoSnap.x - 0.5
      const targetCamY = torsoSnap.y - 1.0
      cameraXRef.current += (targetCamX - cameraXRef.current) * 0.08
      cameraYRef.current += (targetCamY - cameraYRef.current) * 0.05
    }
    const camX = cameraXRef.current
    const camY = cameraYRef.current

    // World → screen transform (using dynamic scale)
    const GSY = Math.round(H * 0.75)
    const ws = (wx, wy) => ({
      sx: W / 2 + (wx - camX) * S,
      sy: GSY - (wy - camY) * S,
    })

    // Grid
    ctx.strokeStyle = 'rgba(255,255,255,0.02)'
    ctx.lineWidth = 1
    const startX = Math.floor(camX - 5)
    for (let wx = startX; wx < camX + 8; wx += 1.0) {
      const { sx } = ws(wx, 0)
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, H); ctx.stroke()
    }

    // Draw terrain or flat ground
    const terrain = snapshot?._terrain
    if (terrain?.segments) {
      // Terrain segments
      ctx.fillStyle = COLORS.terrain.fill
      ctx.strokeStyle = COLORS.terrain.stroke
      ctx.lineWidth = 2

      for (const seg of terrain.segments) {
        const p1 = ws(seg.x1, seg.y1)
        const p2 = ws(seg.x2, seg.y2)
        if (p1.sx > W + 50 && p2.sx > W + 50) continue
        if (p1.sx < -50 && p2.sx < -50) continue

        const bottom = ws(0, camY - 3).sy
        ctx.beginPath()
        ctx.moveTo(p1.sx, p1.sy)
        ctx.lineTo(p2.sx, p2.sy)
        ctx.lineTo(p2.sx, bottom)
        ctx.lineTo(p1.sx, bottom)
        ctx.closePath()
        ctx.fill()

        ctx.beginPath()
        ctx.moveTo(p1.sx, p1.sy)
        ctx.lineTo(p2.sx, p2.sy)
        ctx.stroke()
      }
    } else {
      // Flat ground
      const groundY = charDef.ground?.y ?? 0
      const gScreen = ws(-10, groundY)
      const gScreen2 = ws(10, groundY)
      const bottom = H

      ctx.fillStyle = COLORS.ground.fill
      ctx.beginPath()
      ctx.moveTo(0, gScreen.sy)
      ctx.lineTo(W, gScreen2.sy)
      ctx.lineTo(W, bottom)
      ctx.lineTo(0, bottom)
      ctx.closePath()
      ctx.fill()

      ctx.strokeStyle = COLORS.ground.stroke
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(0, gScreen.sy)
      ctx.lineTo(W, gScreen.sy)
      ctx.stroke()

      // Ground tick marks – makes forward/backward motion obvious
      const tickSpacing = 0.5          // world-units apart
      const tickFirst = Math.floor((camX - W / S / 2) / tickSpacing) * tickSpacing
      const tickLast  = camX + W / S / 2 + tickSpacing
      for (let wx = tickFirst; wx <= tickLast; wx += tickSpacing) {
        const { sx } = ws(wx, groundY)
        const isMajor = Math.abs(wx - Math.round(wx)) < 0.01
        const tickH = isMajor ? 12 : 6
        ctx.strokeStyle = isMajor ? 'rgba(255,255,255,0.25)' : 'rgba(255,255,255,0.12)'
        ctx.lineWidth = isMajor ? 1.5 : 1
        ctx.beginPath()
        ctx.moveTo(sx, gScreen.sy)
        ctx.lineTo(sx, gScreen.sy + tickH)
        ctx.stroke()
        // Number label on major ticks
        if (isMajor) {
          ctx.fillStyle = 'rgba(255,255,255,0.2)'
          ctx.font = '10px monospace'
          ctx.textAlign = 'center'
          ctx.fillText(`${Math.round(wx)}m`, sx, gScreen.sy + tickH + 11)
        }
      }
    }

    if (!snapshot) return

    // Draw shadows
    for (const bodyDef of charDef.bodies) {
      if (bodyDef.fixed) continue
      const t = snapshot[bodyDef.id]
      if (!t) continue
      const { sx } = ws(t.x, t.y)

      let groundYAtX = charDef.ground?.y ?? 0
      if (terrain?.segments) {
        for (const seg of terrain.segments) {
          if (t.x >= seg.x1 && t.x <= seg.x2) {
            const frac = Math.abs(seg.x2 - seg.x1) < 0.001 ? 0 : (t.x - seg.x1) / (seg.x2 - seg.x1)
            groundYAtX = seg.y1 + frac * (seg.y2 - seg.y1)
            break
          }
        }
      }

      const groundScreen = ws(t.x, groundYAtX)
      const bodyScreen = ws(t.x, t.y)
      const shadowW = bodyDef.shape === 'box' ? (bodyDef.w || 0.2) * S * 0.7 : (bodyDef.radius || 0.04) * S * 2.5
      const opacity = Math.max(0, 0.4 - Math.abs(bodyScreen.sy - groundScreen.sy) * 0.003)
      ctx.fillStyle = `rgba(0,0,0,${opacity})`
      ctx.beginPath()
      ctx.ellipse(sx, groundScreen.sy + 3, shadowW, 5, 0, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw props (debug mode obstacles)
    const debugProps = snapshot._debug?.props
    if (debugProps) {
      for (const prop of debugProps) {
        const { sx, sy } = ws(prop.x, prop.y)
        const propColors = { fill: '#4a5568', stroke: 'rgba(255,255,255,0.12)' }
        drawShape(ctx, sx, sy, prop.angle, prop, propColors, S)
      }
    }

    // Draw bodies
    for (const bodyDef of charDef.bodies) {
      if (bodyDef.fixed) continue
      const t = snapshot[bodyDef.id]
      if (!t) continue
      const { sx, sy } = ws(t.x, t.y)
      drawShape(ctx, sx, sy, t.angle, bodyDef, getBodyColor(bodyDef), S)

      // Highlight selected body
      if (highlightBodyId && bodyDef.id === highlightBodyId) {
        ctx.save()
        ctx.translate(sx, sy)
        ctx.rotate(-t.angle)
        ctx.strokeStyle = '#e2b96f'
        ctx.lineWidth = 2
        ctx.shadowColor = 'rgba(226,185,111,0.6)'
        ctx.shadowBlur = 10
        if (bodyDef.shape === 'box') {
          const pw = (bodyDef.w / 2) * S + 3
          const ph = (bodyDef.h / 2) * S + 3
          ctx.beginPath()
          ctx.roundRect(-pw, -ph, pw * 2, ph * 2, 5)
          ctx.stroke()
        } else if (bodyDef.shape === 'capsule') {
          const r = ((bodyDef.radius || 0.04) * S) + 3
          const halfH = Math.max(0.001, ((bodyDef.length || 0.3) - 2 * (bodyDef.radius || 0.04)) / 2) * S
          ctx.beginPath()
          ctx.roundRect(-r, -halfH - r, r * 2, (halfH + r) * 2, r)
          ctx.stroke()
        } else if (bodyDef.shape === 'ball') {
          const r = ((bodyDef.radius || 0.05) * S) + 3
          ctx.beginPath()
          ctx.arc(0, 0, r, 0, Math.PI * 2)
          ctx.stroke()
        }
        ctx.shadowBlur = 0
        ctx.restore()
      }
    }

    // Draw joints
    const jointDebug = snapshot._joints || {}
    for (const jointDef of charDef.joints) {
      if (jointDef.type === 'prismatic') {
        // Draw prismatic joint as a rail line
        const tA = snapshot[jointDef.bodyA]
        const tB = snapshot[jointDef.bodyB]
        if (!tA || !tB) continue
        const axis = jointDef.axis || [1, 0]
        const lo = jointDef.lowerLimit ?? -2
        const hi = jointDef.upperLimit ?? 2
        const p1 = ws(tA.x + axis[0] * lo, tA.y + axis[1] * lo)
        const p2 = ws(tA.x + axis[0] * hi, tA.y + axis[1] * hi)
        ctx.beginPath()
        ctx.moveTo(p1.sx, p1.sy)
        ctx.lineTo(p2.sx, p2.sy)
        ctx.strokeStyle = 'rgba(100,170,255,0.3)'
        ctx.lineWidth = 3
        ctx.stroke()
        // Rail ends
        for (const p of [p1, p2]) {
          ctx.beginPath()
          ctx.arc(p.sx, p.sy, 4, 0, Math.PI * 2)
          ctx.fillStyle = 'rgba(100,170,255,0.4)'
          ctx.fill()
        }
      } else {
        // Revolute joint dot
        const tA = snapshot[jointDef.bodyA]
        if (!tA) continue
        const [ax, ay] = jointDef.anchorA || [0, 0]
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
    }

    // Foot contact splats — flash + particles on new contact events
    const contacts = snapshot._footContacts || {}
    const now = performance.now()
    const prevContacts = prevContactsRef.current
    const splats = splatRef.current
    const SPLAT_DURATION = 250 // ms

    // Detect new contacts (was off, now on) and spawn splats
    for (const [bodyId, contact] of Object.entries(contacts)) {
      const isOn = contact && typeof contact === 'object'
      const wasOn = prevContacts[bodyId] || false
      if (isOn && !wasOn) {
        splats.push({ x: contact.x, y: contact.y, time: now })
      }
      prevContactsRef.current[bodyId] = isOn
    }

    // Remove expired splats
    while (splats.length > 0 && now - splats[0].time > SPLAT_DURATION) {
      splats.shift()
    }

    // Draw splats: expanding ring behind a central dot
    for (const splat of splats) {
      const elapsed = now - splat.time
      const t = Math.min(1, elapsed / SPLAT_DURATION)
      const alpha = Math.max(0, 1 - t)
      const { sx: cx, sy: cy } = ws(splat.x, splat.y)

      // Expanding ring (behind the dot)
      const ringRadius = 3 + 18 * t
      ctx.strokeStyle = `rgba(74, 222, 128, ${alpha * 0.5})`
      ctx.lineWidth = 2 * (1 - t)
      ctx.beginPath()
      ctx.arc(cx, cy, ringRadius, 0, Math.PI * 2)
      ctx.stroke()

      // Central dot (bright, fades with ring)
      ctx.fillStyle = `rgba(74, 222, 128, ${alpha})`
      ctx.shadowColor = `rgba(74, 222, 128, ${alpha * 0.8})`
      ctx.shadowBlur = 8 * alpha
      ctx.beginPath()
      ctx.arc(cx, cy, 3 * (1 - t * 0.5), 0, Math.PI * 2)
      ctx.fill()
      ctx.shadowBlur = 0
    }

    // ── Debug overlay ──────────────────────────────────────────────────────
    const debug = snapshot?._debug
    if (debug) {
      // Determine which bodies are "relevant" (connected to active joint or being dragged)
      const activeJointDef = debug.activeJoint ? charDef.joints.find(j => j.id === debug.activeJoint) : null
      const relevantBodies = new Set()
      if (activeJointDef) {
        relevantBodies.add(activeJointDef.bodyA)
        relevantBodies.add(activeJointDef.bodyB)
      }
      if (dragRef.current.active) {
        relevantBodies.add(dragRef.current.bodyId)
      }

      // Body labels — only for relevant bodies
      if (relevantBodies.size > 0) {
        ctx.font = '500 9px "DM Mono", monospace'
        ctx.textAlign = 'center'
        for (const bl of (snapshot._bodyLabels || debug.bodyLabels || [])) {
          if (!relevantBodies.has(bl.id)) continue
          if (charDef.bodies.find(b => b.id === bl.id)?.fixed) continue
          const t = snapshot[bl.id]
          if (!t) continue
          const { sx, sy } = ws(t.x, t.y)
          ctx.fillStyle = 'rgba(255,255,255,0.5)'
          ctx.fillText(bl.id, sx, sy - 18)
          ctx.fillStyle = 'rgba(255,255,255,0.25)'
          ctx.fillText(`${bl.mass}kg`, sx, sy - 8)
        }
      }

      // Joint limit arcs + angle indicators — only for active joint
      for (const jDef of charDef.joints) {
        const isActive = debug.activeJoint === jDef.id
        if (!isActive) continue  // skip non-active joints
        const jDebug = (debug.joints || snapshot._joints)?.[jDef.id]
        if (!jDebug || jDef.type === 'prismatic') continue
        const tA = snapshot[jDef.bodyA]
        if (!tA) continue

        const [ax, ay] = jDef.anchorA || [0, 0]
        const cos = Math.cos(tA.angle)
        const sin = Math.sin(tA.angle)
        const wx = tA.x + ax * cos - ay * sin
        const wy = tA.y + ax * sin + ay * cos
        const { sx, sy } = ws(wx, wy)

        const arcRadius = 28
        const refAngle = tA.angle

        // Limit arc
        ctx.beginPath()
        ctx.arc(sx, sy, arcRadius, -(refAngle + jDebug.upper), -(refAngle + jDebug.lower), false)
        ctx.strokeStyle = 'rgba(226,185,111,0.5)'
        ctx.lineWidth = 2.5
        ctx.stroke()

        // Limit tick marks
        for (const limitAngle of [jDebug.lower, jDebug.upper]) {
          const screenAngle = -(refAngle + limitAngle)
          ctx.beginPath()
          ctx.moveTo(sx + (arcRadius - 4) * Math.cos(screenAngle), sy + (arcRadius - 4) * Math.sin(screenAngle))
          ctx.lineTo(sx + (arcRadius + 4) * Math.cos(screenAngle), sy + (arcRadius + 4) * Math.sin(screenAngle))
          ctx.strokeStyle = 'rgba(226,185,111,0.6)'
          ctx.lineWidth = 1.5
          ctx.stroke()
        }

        // Current angle indicator
        const currentScreenAngle = -(refAngle + jDebug.angle)
        ctx.beginPath()
        ctx.moveTo(sx, sy)
        ctx.lineTo(sx + (arcRadius + 6) * Math.cos(currentScreenAngle), sy + (arcRadius + 6) * Math.sin(currentScreenAngle))
        const range = jDebug.upper - jDebug.lower
        const normalized = range > 0 ? (jDebug.angle - jDebug.lower) / range : 0.5
        const nearLimit = Math.min(normalized, 1 - normalized) * 2
        const r = Math.round(255 * (1 - nearLimit))
        const g = Math.round(200 * nearLimit)
        ctx.strokeStyle = `rgba(${r},${g},80,0.9)`
        ctx.lineWidth = 2.5
        ctx.stroke()

        // Angle readout
        const angleDeg = (jDebug.angle * 180 / Math.PI).toFixed(0)
        const lowerDeg = (jDebug.lower * 180 / Math.PI).toFixed(0)
        const upperDeg = (jDebug.upper * 180 / Math.PI).toFixed(0)
        ctx.font = '500 8px "DM Mono", monospace'
        ctx.textAlign = 'left'
        ctx.fillStyle = 'rgba(226,185,111,0.8)'
        ctx.fillText(`${jDef.id}: ${angleDeg}° [${lowerDeg},${upperDeg}]`, sx + arcRadius + 10, sy - 2)
        ctx.fillStyle = 'rgba(226,185,111,0.5)'
        ctx.fillText(`τmax ${jDebug.maxTorque}`, sx + arcRadius + 10, sy + 9)

        // Torque direction arrow
        if (debug.direction !== 0) {
          const arrowAngle = currentScreenAngle + (debug.direction > 0 ? -0.4 : 0.4)
          const tipX = sx + (arcRadius + 14) * Math.cos(arrowAngle)
          const tipY = sy + (arcRadius + 14) * Math.sin(arrowAngle)
          ctx.beginPath()
          ctx.arc(sx, sy, arcRadius + 10, currentScreenAngle,
            currentScreenAngle + (debug.direction > 0 ? -0.5 : 0.5), debug.direction > 0)
          ctx.strokeStyle = debug.direction > 0 ? 'rgba(74,222,128,0.7)' : 'rgba(224,90,90,0.7)'
          ctx.lineWidth = 2.5
          ctx.stroke()
          ctx.beginPath()
          ctx.arc(tipX, tipY, 3, 0, Math.PI * 2)
          ctx.fillStyle = debug.direction > 0 ? 'rgba(74,222,128,0.8)' : 'rgba(224,90,90,0.8)'
          ctx.fill()
        }
      }

      // Drag spring line — from grab point on body to mouse cursor
      const grabInfo = debug.grab
      if (grabInfo) {
        const grabScreen = ws(grabInfo.x, grabInfo.y)
        const targetScreen = ws(grabInfo.targetX, grabInfo.targetY)

        // Spring line (dashed)
        ctx.beginPath()
        ctx.moveTo(grabScreen.sx, grabScreen.sy)
        ctx.lineTo(targetScreen.sx, targetScreen.sy)
        ctx.strokeStyle = 'rgba(226,185,111,0.6)'
        ctx.lineWidth = 2
        ctx.setLineDash([4, 4])
        ctx.stroke()
        ctx.setLineDash([])

        // Grab point on body (small filled dot)
        ctx.beginPath()
        ctx.arc(grabScreen.sx, grabScreen.sy, 4, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(226,185,111,0.8)'
        ctx.fill()

        // Mouse cursor target (ring)
        ctx.beginPath()
        ctx.arc(targetScreen.sx, targetScreen.sy, 6, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(226,185,111,0.3)'
        ctx.fill()
        ctx.strokeStyle = 'rgba(226,185,111,0.7)'
        ctx.lineWidth = 1.5
        ctx.stroke()
      } else if (dragRef.current.active && snapshot[dragRef.current.bodyId]) {
        // Fallback for non-debug mode
        const bodySnap = snapshot[dragRef.current.bodyId]
        const bodyScreen = ws(bodySnap.x, bodySnap.y)
        const mouseScreen = ws(dragRef.current.wx, dragRef.current.wy)

        ctx.beginPath()
        ctx.moveTo(bodyScreen.sx, bodyScreen.sy)
        ctx.lineTo(mouseScreen.sx, mouseScreen.sy)
        ctx.strokeStyle = 'rgba(226,185,111,0.6)'
        ctx.lineWidth = 2
        ctx.setLineDash([4, 4])
        ctx.stroke()
        ctx.setLineDash([])
      }

      // Debug mode label
      ctx.font = '600 11px "DM Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillStyle = 'rgba(226,185,111,0.6)'
      ctx.fillText('PHYSICS DEBUG', W / 2, 20)
      const debugLabel = dragRef.current.active ? `dragging: ${dragRef.current.bodyId}`
        : debug.activeJoint ? `testing: ${debug.activeJoint}`
        : 'click body to drag · scroll to zoom · right-drag to pan'
      ctx.font = '500 10px "DM Mono", monospace'
      ctx.fillStyle = 'rgba(226,185,111,0.4)'
      ctx.fillText(debugLabel, W / 2, 34)
    }

    // HUD
    ctx.font = '500 10px "DM Mono", monospace'
    ctx.textAlign = 'left'
    ctx.fillStyle = 'rgba(255,255,255,0.35)'

    if (torsoSnap) {
      ctx.fillText(`pos ${torsoSnap.x.toFixed(2)}m`, 12, H - 42)
      if (snapshot._maxDistance !== undefined) {
        ctx.fillText(`best ${snapshot._maxDistance.toFixed(1)}m`, 12, H - 28)
      }
    }
    ctx.fillText(`t ${episodeSteps ?? 0}`, 12, H - 12)

    if (episodeReward !== undefined) {
      ctx.textAlign = 'right'
      ctx.fillStyle = episodeReward > 0 ? 'rgba(74,222,128,0.5)' : 'rgba(224,90,90,0.5)'
      ctx.fillText(`r ${(episodeReward || 0).toFixed(1)}`, W - 12, H - 12)
    }

    // Red Light / Green Light indicator
    if (snapshot._extra?.lightGreen !== undefined) {
      const isGreen = snapshot._extra.lightGreen
      const color = isGreen ? '#4ade80' : '#f87171'
      const label = isGreen ? 'GO!' : 'STOP!'
      // Large colored circle in top-center
      const cx = W / 2, cy = 30, r = 16
      ctx.beginPath()
      ctx.arc(cx, cy, r, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
      ctx.strokeStyle = 'rgba(0,0,0,0.4)'
      ctx.lineWidth = 2
      ctx.stroke()
      // Label below
      ctx.font = 'bold 12px "DM Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillStyle = color
      ctx.fillText(label, cx, cy + r + 14)
    }

    // Zoom indicator (when not default)
    const zoomPct = Math.round(S / DEFAULT_SCALE * 100)
    if (zoomPct !== 100) {
      ctx.textAlign = 'right'
      ctx.fillStyle = 'rgba(255,255,255,0.3)'
      ctx.font = '500 9px "DM Mono", monospace'
      ctx.fillText(`${zoomPct}%`, W - 12, 16)
    }

  }, [snapshot, episodeReward, episodeSteps, canvasSize, autoFollow, highlightBodyId])

  return (
    <canvas
      ref={canvasRef}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  )
}
