/**
 * CreatureBuilder — Visual body editor for user-created RL agents.
 *
 * Users add body parts (torso, limbs) and joints to construct a creature.
 * The builder generates a character definition JSON compatible with TerrainRapierEnv.
 *
 * Design: Canvas-based drag-and-drop with a parts palette and property inspector.
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import { TERRAIN_SAMPLES } from '../env/terrainEnv.js'

// ─── Preset Templates ─────────────────────────────────────────────────────────

const PRESETS = {
  biped: {
    label: 'Biped',
    desc: '2 legs, balanced walker',
    parts: [
      { id: 'torso', type: 'torso', shape: 'box', x: 0, y: 1.25, w: 0.25, h: 0.35, mass: 4.0, angle: 0 },
      { id: 'left_thigh', type: 'limb', shape: 'capsule', x: 0.06, y: 0.83, radius: 0.055, length: 0.38, mass: 1.2, angle: 0 },
      { id: 'left_shin', type: 'foot', shape: 'capsule', x: 0.06, y: 0.45, radius: 0.045, length: 0.35, mass: 0.5, angle: 0 },
      { id: 'right_thigh', type: 'limb', shape: 'capsule', x: -0.06, y: 0.83, radius: 0.055, length: 0.38, mass: 1.2, angle: 0 },
      { id: 'right_shin', type: 'foot', shape: 'capsule', x: -0.06, y: 0.45, radius: 0.045, length: 0.35, mass: 0.5, angle: 0 },
    ],
    connections: [
      { from: 'torso', to: 'left_thigh', anchorA: [0.06, -0.175], anchorB: [0.0, 0.19], limits: [-0.9, 0.9] },
      { from: 'left_thigh', to: 'left_shin', anchorA: [0.0, -0.19], anchorB: [0.0, 0.175], limits: [-1.5, 0.05] },
      { from: 'torso', to: 'right_thigh', anchorA: [-0.06, -0.175], anchorB: [0.0, 0.19], limits: [-0.9, 0.9] },
      { from: 'right_thigh', to: 'right_shin', anchorA: [0.0, -0.19], anchorB: [0.0, 0.175], limits: [-1.5, 0.05] },
    ],
  },
  quadruped: {
    label: 'Quadruped',
    desc: '4 legs, dog-like',
    parts: [
      { id: 'torso', type: 'torso', shape: 'box', x: 0, y: 0.9, w: 0.55, h: 0.2, mass: 5.0, angle: 0 },
      { id: 'front_left_thigh', type: 'limb', shape: 'capsule', x: 0.22, y: 0.6, radius: 0.04, length: 0.28, mass: 0.6, angle: 0 },
      { id: 'front_left_shin', type: 'foot', shape: 'capsule', x: 0.22, y: 0.3, radius: 0.035, length: 0.25, mass: 0.3, angle: 0 },
      { id: 'front_right_thigh', type: 'limb', shape: 'capsule', x: -0.22, y: 0.6, radius: 0.04, length: 0.28, mass: 0.6, angle: 0 },
      { id: 'front_right_shin', type: 'foot', shape: 'capsule', x: -0.22, y: 0.3, radius: 0.035, length: 0.25, mass: 0.3, angle: 0 },
      { id: 'back_left_thigh', type: 'limb', shape: 'capsule', x: 0.22, y: 0.6, radius: 0.04, length: 0.28, mass: 0.6, angle: 0 },
      { id: 'back_left_shin', type: 'foot', shape: 'capsule', x: 0.22, y: 0.3, radius: 0.035, length: 0.25, mass: 0.3, angle: 0 },
      { id: 'back_right_thigh', type: 'limb', shape: 'capsule', x: -0.22, y: 0.6, radius: 0.04, length: 0.28, mass: 0.6, angle: 0 },
      { id: 'back_right_shin', type: 'foot', shape: 'capsule', x: -0.22, y: 0.3, radius: 0.035, length: 0.25, mass: 0.3, angle: 0 },
    ],
    connections: [
      { from: 'torso', to: 'front_left_thigh', anchorA: [0.22, -0.1], anchorB: [0.0, 0.14], limits: [-0.8, 0.8] },
      { from: 'front_left_thigh', to: 'front_left_shin', anchorA: [0.0, -0.14], anchorB: [0.0, 0.125], limits: [-1.2, 0.05] },
      { from: 'torso', to: 'front_right_thigh', anchorA: [-0.22, -0.1], anchorB: [0.0, 0.14], limits: [-0.8, 0.8] },
      { from: 'front_right_thigh', to: 'front_right_shin', anchorA: [0.0, -0.14], anchorB: [0.0, 0.125], limits: [-1.2, 0.05] },
      { from: 'torso', to: 'back_left_thigh', anchorA: [0.22, -0.1], anchorB: [0.0, 0.14], limits: [-0.8, 0.8] },
      { from: 'back_left_thigh', to: 'back_left_shin', anchorA: [0.0, -0.14], anchorB: [0.0, 0.125], limits: [-1.2, 0.05] },
      { from: 'torso', to: 'back_right_thigh', anchorA: [-0.22, -0.1], anchorB: [0.0, 0.14], limits: [-0.8, 0.8] },
      { from: 'back_right_thigh', to: 'back_right_shin', anchorA: [0.0, -0.14], anchorB: [0.0, 0.125], limits: [-1.2, 0.05] },
    ],
  },
  hopper: {
    label: 'Hopper',
    desc: '1 leg, pogo stick',
    parts: [
      { id: 'torso', type: 'torso', shape: 'box', x: 0, y: 1.35, w: 0.25, h: 0.45, mass: 3.5, angle: 0 },
      { id: 'thigh', type: 'limb', shape: 'capsule', x: 0, y: 0.87, radius: 0.05, length: 0.35, mass: 0.9, angle: 0 },
      { id: 'shin', type: 'foot', shape: 'capsule', x: 0, y: 0.47, radius: 0.04, length: 0.30, mass: 0.4, angle: 0 },
    ],
    connections: [
      { from: 'torso', to: 'thigh', anchorA: [0.0, -0.225], anchorB: [0.0, 0.175], limits: [-0.7, 0.7] },
      { from: 'thigh', to: 'shin', anchorA: [0.0, -0.175], anchorB: [0.0, 0.15], limits: [-1.4, 0.0] },
    ],
  },
  trex: {
    label: 'T-Rex',
    desc: 'Big body, tiny arms, 2 legs',
    parts: [
      { id: 'torso', type: 'torso', shape: 'box', x: 0, y: 1.4, w: 0.4, h: 0.5, mass: 8.0, angle: 0 },
      { id: 'head', type: 'limb', shape: 'box', x: 0.15, y: 1.8, w: 0.2, h: 0.15, mass: 1.5, angle: 0 },
      { id: 'tail', type: 'limb', shape: 'capsule', x: -0.35, y: 1.3, radius: 0.04, length: 0.5, mass: 1.0, angle: 1.5 },
      { id: 'left_thigh', type: 'limb', shape: 'capsule', x: 0.05, y: 0.85, radius: 0.07, length: 0.45, mass: 2.0, angle: 0 },
      { id: 'left_shin', type: 'foot', shape: 'capsule', x: 0.05, y: 0.4, radius: 0.05, length: 0.4, mass: 0.8, angle: 0 },
      { id: 'right_thigh', type: 'limb', shape: 'capsule', x: -0.05, y: 0.85, radius: 0.07, length: 0.45, mass: 2.0, angle: 0 },
      { id: 'right_shin', type: 'foot', shape: 'capsule', x: -0.05, y: 0.4, radius: 0.05, length: 0.4, mass: 0.8, angle: 0 },
    ],
    connections: [
      { from: 'torso', to: 'head', anchorA: [0.15, 0.2], anchorB: [-0.1, -0.05], limits: [-0.3, 0.3] },
      { from: 'torso', to: 'tail', anchorA: [-0.2, 0.0], anchorB: [0.0, 0.25], limits: [-0.5, 0.5] },
      { from: 'torso', to: 'left_thigh', anchorA: [0.05, -0.25], anchorB: [0.0, 0.225], limits: [-0.8, 0.8] },
      { from: 'left_thigh', to: 'left_shin', anchorA: [0.0, -0.225], anchorB: [0.0, 0.2], limits: [-1.5, 0.05] },
      { from: 'torso', to: 'right_thigh', anchorA: [-0.05, -0.25], anchorB: [0.0, 0.225], limits: [-0.8, 0.8] },
      { from: 'right_thigh', to: 'right_shin', anchorA: [0.0, -0.225], anchorB: [0.0, 0.2], limits: [-1.5, 0.05] },
    ],
  },
}

// ─── Convert builder state to character definition ────────────────────────────

function builderToCharDef(parts, connections, name = 'custom') {
  const bodies = parts.map(p => {
    const body = {
      id: p.id,
      shape: p.shape,
      mass: p.mass,
      friction: p.type === 'foot' ? 0.9 : 0.3,
      restitution: 0.0,
      spawnX: p.x,
      spawnY: p.y,
      spawnAngle: p.angle || 0,
    }

    if (p.shape === 'box') {
      body.w = p.w
      body.h = p.h
    } else if (p.shape === 'capsule') {
      body.radius = p.radius
      body.length = p.length
    } else if (p.shape === 'ball') {
      body.radius = p.radius
    }

    if (p.type === 'torso') {
      body.minY = 0.4
      body.maxAngle = 0.7
      body.terminateOnContact = true
    }

    if (p.type === 'foot') {
      body.isFootBody = true
    }

    return body
  })

  const joints = connections.map((c, i) => ({
    id: `joint_${i}`,
    bodyA: c.from,
    bodyB: c.to,
    anchorA: c.anchorA,
    anchorB: c.anchorB,
    lowerLimit: c.limits[0],
    upperLimit: c.limits[1],
    maxTorque: 300.0,
    maxVelocity: 12.0,
    kp: 300,
    kd: 30,
    stiffness: 0,
    damping: 5.0,
  }))

  const numFeet = bodies.filter(b => b.isFootBody).length

  // obsSize = 5(torso) + joints*2 + feet + terrain height samples
  const obsSize = 5 + joints.length * 2 + numFeet + TERRAIN_SAMPLES
  // actionSize = number of actuated joints (policy outputs target angles)
  const actionSize = joints.length

  return {
    name,
    gravityScale: 1.0,
    ground: { y: 0.0, friction: 0.8, restitution: 0.1 },
    bodies,
    joints,
    forwardBody: 'torso',
    obsSize,
    actionSize,
    defaultReward: {
      forwardVelWeight: 1.5,
      aliveBonusWeight: 1.0,
      ctrlCostWeight: 0.001,
      terminationPenalty: 50.0,
    },
  }
}

// ─── Canvas Preview ───────────────────────────────────────────────────────────

const PREVIEW_SCALE = 100
const PART_COLORS = {
  torso: '#e2b96f',
  limb: '#c8a05a',
  foot: '#b08840',
}

function drawPreview(ctx, W, H, parts, connections, selectedId) {
  ctx.clearRect(0, 0, W, H)
  ctx.fillStyle = '#07070f'
  ctx.fillRect(0, 0, W, H)

  const GSY = H * 0.82

  // Ground
  const groundGrad = ctx.createLinearGradient(0, GSY, 0, H)
  groundGrad.addColorStop(0, '#131330')
  groundGrad.addColorStop(1, '#070714')
  ctx.fillStyle = groundGrad
  ctx.fillRect(0, GSY, W, H - GSY)

  ctx.strokeStyle = '#1e2040'
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(0, GSY)
  ctx.lineTo(W, GSY)
  ctx.stroke()

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.025)'
  ctx.lineWidth = 1
  for (let x = 0; x < W; x += PREVIEW_SCALE) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
  }
  for (let y = 0; y < H; y += PREVIEW_SCALE) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
  }

  const toScreen = (wx, wy) => ({
    sx: W / 2 + wx * PREVIEW_SCALE,
    sy: GSY - wy * PREVIEW_SCALE,
  })

  // Draw connections
  ctx.strokeStyle = 'rgba(255,153,102,0.3)'
  ctx.lineWidth = 2
  for (const conn of connections) {
    const fromPart = parts.find(p => p.id === conn.from)
    const toPart = parts.find(p => p.id === conn.to)
    if (!fromPart || !toPart) continue
    const from = toScreen(fromPart.x, fromPart.y)
    const to = toScreen(toPart.x, toPart.y)
    ctx.beginPath()
    ctx.moveTo(from.sx, from.sy)
    ctx.lineTo(to.sx, to.sy)
    ctx.stroke()
  }

  // Draw parts
  for (const part of parts) {
    const { sx, sy } = toScreen(part.x, part.y)
    const color = PART_COLORS[part.type] || PART_COLORS.limb
    const isSelected = part.id === selectedId

    ctx.save()
    ctx.translate(sx, sy)
    ctx.rotate(-(part.angle || 0))

    if (isSelected) {
      ctx.shadowColor = '#e2b96f'
      ctx.shadowBlur = 15
    }

    ctx.fillStyle = color
    ctx.strokeStyle = isSelected ? '#fff' : 'rgba(255,255,255,0.15)'
    ctx.lineWidth = isSelected ? 2 : 1

    if (part.shape === 'box') {
      const pw = (part.w || 0.2) * PREVIEW_SCALE
      const ph = (part.h || 0.2) * PREVIEW_SCALE
      ctx.beginPath()
      ctx.roundRect(-pw / 2, -ph / 2, pw, ph, 3)
      ctx.fill()
      ctx.stroke()
    } else if (part.shape === 'capsule') {
      const r = (part.radius || 0.04) * PREVIEW_SCALE
      const halfH = Math.max(0.001, ((part.length || 0.3) - 2 * (part.radius || 0.04)) / 2) * PREVIEW_SCALE
      ctx.beginPath()
      ctx.roundRect(-r, -halfH - r, r * 2, (halfH + r) * 2, r)
      ctx.fill()
      ctx.stroke()
    } else if (part.shape === 'ball') {
      const r = (part.radius || 0.05) * PREVIEW_SCALE
      ctx.beginPath()
      ctx.arc(0, 0, r, 0, Math.PI * 2)
      ctx.fill()
      ctx.stroke()
    }

    ctx.shadowBlur = 0
    ctx.restore()

    // Label
    ctx.fillStyle = isSelected ? '#fff' : 'rgba(255,255,255,0.4)'
    ctx.font = '9px "DM Mono", monospace'
    ctx.textAlign = 'center'
    ctx.fillText(part.id, sx, sy - ((part.h || part.length || 0.2) * PREVIEW_SCALE / 2 + 10))
  }
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function CreatureBuilder({ onCreatureChange, disabled }) {
  const [parts, setParts] = useState(PRESETS.biped.parts)
  const [connections, setConnections] = useState(PRESETS.biped.connections)
  const [selectedPreset, setSelectedPreset] = useState('biped')
  const [selectedPart, setSelectedPart] = useState(null)
  const [creatureName, setCreatureName] = useState('my-creature')
  const canvasRef = useRef(null)
  const [canvasSize, setCanvasSize] = useState({ w: 400, h: 280 })

  // Notify parent of character definition changes
  useEffect(() => {
    const charDef = builderToCharDef(parts, connections, creatureName)
    onCreatureChange?.(charDef)
  }, [parts, connections, creatureName])

  // Canvas sizing
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

  // Draw preview
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    drawPreview(ctx, canvas.width, canvas.height, parts, connections, selectedPart)
  }, [parts, connections, selectedPart, canvasSize])

  // Click to select parts
  const handleCanvasClick = useCallback((e) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const mx = e.clientX - rect.left
    const my = e.clientY - rect.top
    const GSY = canvas.height * 0.82

    const toScreen = (wx, wy) => ({
      sx: canvas.width / 2 + wx * PREVIEW_SCALE,
      sy: GSY - wy * PREVIEW_SCALE,
    })

    let found = null
    for (const part of parts) {
      const { sx, sy } = toScreen(part.x, part.y)
      const size = Math.max((part.w || part.length || part.radius * 2 || 0.2) * PREVIEW_SCALE, 15)
      if (Math.abs(mx - sx) < size && Math.abs(my - sy) < size) {
        found = part.id
      }
    }
    setSelectedPart(found)
  }, [parts])

  const loadPreset = (presetKey) => {
    const preset = PRESETS[presetKey]
    setParts([...preset.parts])
    setConnections([...preset.connections])
    setSelectedPreset(presetKey)
    setSelectedPart(null)
    setCreatureName(presetKey)
  }

  const updatePart = (id, key, value) => {
    setParts(prev => prev.map(p => p.id === id ? { ...p, [key]: value } : p))
  }

  const selectedPartData = parts.find(p => p.id === selectedPart)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, height: '100%' }}>
      {/* Preset selector */}
      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
        {Object.entries(PRESETS).map(([key, preset]) => (
          <button
            key={key}
            onClick={() => loadPreset(key)}
            disabled={disabled}
            style={{
              flex: '1 1 auto',
              padding: '5px 8px',
              background: selectedPreset === key ? 'var(--gold-dim)' : 'var(--surface)',
              border: `1px solid ${selectedPreset === key ? 'var(--gold-border)' : 'var(--border)'}`,
              borderRadius: 5,
              color: selectedPreset === key ? 'var(--gold)' : 'var(--text-dim)',
              fontFamily: '"DM Mono", monospace',
              fontSize: 10,
              cursor: disabled ? 'not-allowed' : 'pointer',
              textAlign: 'center',
            }}
          >
            <div style={{ fontWeight: 500 }}>{preset.label}</div>
            <div style={{ fontSize: 8, opacity: 0.7 }}>{preset.desc}</div>
          </button>
        ))}
      </div>

      {/* Canvas preview */}
      <div style={{
        flex: 1,
        minHeight: 180,
        border: '1px solid var(--border)',
        borderRadius: 6,
        overflow: 'hidden',
      }}>
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          style={{ width: '100%', height: '100%', display: 'block', cursor: disabled ? 'default' : 'crosshair' }}
        />
      </div>

      {/* Part inspector */}
      {selectedPartData && !disabled && (
        <div style={{
          padding: '8px 10px',
          background: 'var(--surface)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          fontSize: 10,
          fontFamily: '"DM Mono", monospace',
        }}>
          <div style={{ color: 'var(--gold)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            {selectedPartData.id} ({selectedPartData.type})
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 12px' }}>
            <label style={{ color: 'var(--text-dim)' }}>
              Mass
              <input type="range" min="0.1" max="10" step="0.1"
                value={selectedPartData.mass}
                onChange={e => updatePart(selectedPart, 'mass', parseFloat(e.target.value))}
                style={{ width: '100%', accentColor: '#e2b96f' }}
              />
              <span style={{ color: '#e2b96f' }}>{selectedPartData.mass.toFixed(1)}kg</span>
            </label>
            {selectedPartData.shape === 'box' && (
              <>
                <label style={{ color: 'var(--text-dim)' }}>
                  Width
                  <input type="range" min="0.1" max="0.8" step="0.01"
                    value={selectedPartData.w}
                    onChange={e => updatePart(selectedPart, 'w', parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#e2b96f' }}
                  />
                  <span style={{ color: '#e2b96f' }}>{selectedPartData.w.toFixed(2)}m</span>
                </label>
                <label style={{ color: 'var(--text-dim)' }}>
                  Height
                  <input type="range" min="0.1" max="0.8" step="0.01"
                    value={selectedPartData.h}
                    onChange={e => updatePart(selectedPart, 'h', parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#e2b96f' }}
                  />
                  <span style={{ color: '#e2b96f' }}>{selectedPartData.h.toFixed(2)}m</span>
                </label>
              </>
            )}
            {selectedPartData.shape === 'capsule' && (
              <>
                <label style={{ color: 'var(--text-dim)' }}>
                  Length
                  <input type="range" min="0.1" max="0.8" step="0.01"
                    value={selectedPartData.length}
                    onChange={e => updatePart(selectedPart, 'length', parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#e2b96f' }}
                  />
                  <span style={{ color: '#e2b96f' }}>{selectedPartData.length.toFixed(2)}m</span>
                </label>
                <label style={{ color: 'var(--text-dim)' }}>
                  Radius
                  <input type="range" min="0.02" max="0.15" step="0.005"
                    value={selectedPartData.radius}
                    onChange={e => updatePart(selectedPart, 'radius', parseFloat(e.target.value))}
                    style={{ width: '100%', accentColor: '#e2b96f' }}
                  />
                  <span style={{ color: '#e2b96f' }}>{selectedPartData.radius.toFixed(3)}m</span>
                </label>
              </>
            )}
          </div>
        </div>
      )}

      {/* Name input */}
      <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
        <span style={{ fontSize: 10, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap' }}>
          Name
        </span>
        <input
          type="text"
          value={creatureName}
          onChange={e => setCreatureName(e.target.value.replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 20))}
          disabled={disabled}
          maxLength={20}
          style={{
            flex: 1,
            background: 'var(--surface)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            color: 'var(--gold)',
            fontFamily: '"DM Mono", monospace',
            fontSize: 11,
            padding: '4px 8px',
            outline: 'none',
          }}
        />
      </div>

      {/* Stats */}
      <div style={{
        display: 'flex', gap: 12, fontSize: 9,
        color: 'rgba(255,255,255,0.4)', fontFamily: '"DM Mono", monospace',
      }}>
        <span>{parts.length} bodies</span>
        <span>{connections.length} joints</span>
        <span>{parts.filter(p => p.type === 'foot').length} feet</span>
        <span>{parts.reduce((s, p) => s + p.mass, 0).toFixed(1)}kg total</span>
      </div>
    </div>
  )
}

export { builderToCharDef, PRESETS }
