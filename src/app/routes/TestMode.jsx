/**
 * TestMode — Physics Lab
 *
 * Comprehensive body verification without any ML.
 * Users can:
 *   - Load a body from MJCF file, JSON, or the creature builder
 *   - Drop it into a flat-ground physics sandbox
 *   - Test each joint: range of motion, strength, damping
 *   - Grab and drag bodies with the mouse
 *   - Measure gravity response, friction, restitution
 *   - Run automated verification checks
 *   - View real-time measurements overlay (velocities, forces, angles)
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import PhysicsRenderer from '../../components/PhysicsRenderer.jsx'
import SceneHierarchy from '../../components/SceneHierarchy.jsx'
import CreatureBuilder from '../../components/CreatureBuilder.jsx'
import { computeDerivedFields } from '../../formats/bodyDef.js'
import { BIPED } from '../../env/characters/biped.js'

// ─── Verification Tests ────────────────────────────────────────────────────

const TESTS = [
  {
    id: 'gravity',
    label: 'Gravity Drop',
    desc: 'Drop the body from spawn height. It should fall and settle on the ground.',
    auto: true,
  },
  {
    id: 'joint_range',
    label: 'Joint Range',
    desc: 'Each joint should move within its defined limits and stop at the boundaries.',
    auto: true,
  },
  {
    id: 'joint_strength',
    label: 'Joint Strength',
    desc: 'Apply max torque to each joint. The body should respond proportionally.',
    auto: true,
  },
  {
    id: 'friction',
    label: 'Friction',
    desc: 'Slide the body along the ground. It should decelerate based on friction.',
    auto: false,
  },
  {
    id: 'restitution',
    label: 'Restitution (Bounce)',
    desc: 'Drop the body. It should bounce according to its restitution coefficient.',
    auto: false,
  },
  {
    id: 'mass',
    label: 'Mass Distribution',
    desc: 'Check that body parts have the expected masses and the body balances as expected.',
    auto: true,
  },
  {
    id: 'ragdoll',
    label: 'Ragdoll',
    desc: 'Free interaction mode. Grab, drag, and throw body parts to verify overall physics.',
    auto: false,
  },
]

export default function TestMode() {
  const workerRef = useRef(null)
  const [charDef, setCharDef] = useState(null)
  const [isRunning, setIsRunning] = useState(false)
  const [snapshot, setSnapshot] = useState(null)
  const [selectedTest, setSelectedTest] = useState('ragdoll')
  const [testResults, setTestResults] = useState({})
  const [selectedJoint, setSelectedJoint] = useState(null)
  const [torqueDirection, setTorqueDirection] = useState(0)
  const [measurements, setMeasurements] = useState(null)
  const [showOverlay, setShowOverlay] = useState(true)
  const [pinned, setPinned] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [sceneSelectedId, setSceneSelectedId] = useState(null)
  const [sceneSelectedType, setSceneSelectedType] = useState(null)

  // Initialize with default biped
  useEffect(() => {
    setCharDef(BIPED)
  }, [])

  const initWorker = useCallback(() => {
    if (workerRef.current) workerRef.current.terminate()
    const worker = new Worker(
      new URL('../../rl/trainWorker.js', import.meta.url),
      { type: 'module' }
    )
    worker.onmessage = (e) => {
      const msg = e.data
      if (msg.type === 'RENDER_SNAPSHOT') {
        setSnapshot(msg.snapshot)
        // Extract measurements from snapshot
        if (msg.snapshot) {
          const m = {}
          const joints = msg.snapshot._joints || {}
          m.joints = joints
          m.bodies = {}
          const charDefData = msg.snapshot._charDef
          if (charDefData) {
            for (const bDef of charDefData.bodies) {
              const bs = msg.snapshot[bDef.id]
              if (bs) {
                m.bodies[bDef.id] = {
                  x: bs.x,
                  y: bs.y,
                  angle: bs.angle,
                  mass: bDef.mass,
                  shape: bDef.shape,
                }
              }
            }
          }
          m.footContacts = msg.snapshot._footContacts || {}
          m.obs = msg.snapshot._obs || null
          m.rewardBreakdown = msg.snapshot._rewardBreakdown || null
          setMeasurements(m)
          if (msg.snapshot._debug?.pinned !== undefined) {
            setPinned(msg.snapshot._debug.pinned)
          }
        }
      }
    }
    workerRef.current = worker
    return worker
  }, [])

  const startDebug = useCallback((def) => {
    if (!def) return
    const worker = initWorker()
    setIsRunning(true)
    setSnapshot(null)
    setTestResults({})
    setSelectedJoint(null)
    setTorqueDirection(0)
    worker.postMessage({ type: 'PHYSICS_DEBUG', config: { charDef: def } })
  }, [initWorker])

  const stopDebug = useCallback(() => {
    workerRef.current?.postMessage({ type: 'STOP' })
    setIsRunning(false)
    setSnapshot(null)
  }, [])

  const handleReset = useCallback(() => {
    workerRef.current?.postMessage({ type: 'DEBUG_RESET' })
    setPinned(false)
  }, [])

  const handlePinTorso = useCallback(() => {
    workerRef.current?.postMessage({ type: 'DEBUG_PIN_TORSO' })
  }, [])

  const handleSceneSelect = useCallback((id, type) => {
    setSceneSelectedId(id)
    setSceneSelectedType(type)
  }, [])

  const handleScenePropertyChange = useCallback((msg) => {
    workerRef.current?.postMessage({ type: msg.type, config: msg.config })
  }, [])

  const handleBodyClick = useCallback((bodyId) => {
    if (bodyId) {
      setSceneSelectedId(bodyId)
      setSceneSelectedType('body')
    } else {
      setSceneSelectedId(null)
      setSceneSelectedType(null)
    }
  }, [])

  // Joint control
  const handleJointSelect = (jointId) => {
    const next = selectedJoint === jointId ? null : jointId
    setSelectedJoint(next)
    setTorqueDirection(0)
    workerRef.current?.postMessage({ type: 'DEBUG_CONTROL', config: { joint: next, direction: 0 } })
  }

  // Torque drag state
  const torqueDragRef = useRef({ active: false, startX: 0, sign: 0 })

  const handleTorque = (dir) => {
    setTorqueDirection(dir)
    workerRef.current?.postMessage({ type: 'DEBUG_CONTROL', config: { joint: selectedJoint, direction: dir } })
  }

  const handleTorqueDragStart = (e, sign) => {
    e.preventDefault()
    torqueDragRef.current = { active: true, startX: e.clientX, sign }
    handleTorque(sign * 0.5) // start at 50%

    const onMove = (me) => {
      if (!torqueDragRef.current.active) return
      const dx = me.clientX - torqueDragRef.current.startX
      // Drag right = increase magnitude, drag left = decrease magnitude
      // 100px right from start = 100%, 100px left = 0%
      const magnitude = Math.min(1, Math.max(0, 0.5 + dx / 200))
      const dir = torqueDragRef.current.sign * magnitude
      setTorqueDirection(dir)
      workerRef.current?.postMessage({ type: 'DEBUG_CONTROL', config: { joint: selectedJoint, direction: dir } })
    }

    const onUp = () => {
      torqueDragRef.current.active = false
      handleTorque(0)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }

  // Mouse interaction for grab/drag/release
  const handleDebugMouse = useCallback((msg) => {
    if (!workerRef.current) return
    switch (msg.type) {
      case 'grab':
        workerRef.current.postMessage({ type: 'DEBUG_GRAB', config: { bodyId: msg.bodyId, wx: msg.wx, wy: msg.wy } })
        break
      case 'drag':
        workerRef.current.postMessage({ type: 'DEBUG_DRAG', config: { bodyId: msg.bodyId, wx: msg.wx, wy: msg.wy } })
        break
      case 'release':
        workerRef.current.postMessage({ type: 'DEBUG_RELEASE', config: { bodyId: msg.bodyId, vx: msg.vx, vy: msg.vy } })
        break
    }
  }, [])

  // Creature builder change
  const handleCreatureChange = (def) => {
    setCharDef(def)
  }

  // Derived info
  const derived = charDef ? computeDerivedFields(charDef) : null
  const totalMass = charDef?.bodies?.reduce((s, b) => s + (b.mass || 0), 0) || 0

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

      {/* Top bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        {charDef && (
          <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.6)', fontFamily: 'Inter, sans-serif' }}>
            {charDef.name || 'unnamed'} · {charDef.bodies?.length || 0} bodies · {charDef.joints?.length || 0} joints · {totalMass.toFixed(1)}kg
          </span>
        )}
      </div>

      {/* Main layout */}
      <div style={{ display: 'grid', gridTemplateColumns: expanded ? '1fr' : '1fr 320px', gap: 16 }}>

        {/* Left: Renderer + measurements */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Sim viewport */}
          <div className="panel">
            <div className="panel-header">
              <span>⚙ physics lab</span>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="btn btn-ghost"
                  onClick={() => setExpanded(!expanded)}
                  style={{ padding: '3px 8px', fontSize: 9 }}
                  title={expanded ? 'Collapse viewport' : 'Expand viewport'}
                >
                  {expanded ? '◱ collapse' : '◳ expand'}
                </button>
                <button
                  className="btn btn-ghost"
                  onClick={() => setShowOverlay(!showOverlay)}
                  style={{ padding: '3px 8px', fontSize: 9, color: showOverlay ? 'var(--gold)' : undefined }}
                >
                  {showOverlay ? '◉ overlay' : '○ overlay'}
                </button>
                {isRunning ? (
                  <>
                    <button
                      className="btn btn-ghost"
                      onClick={handlePinTorso}
                      style={{
                        padding: '3px 8px', fontSize: 9,
                        color: pinned ? 'var(--gold)' : undefined,
                        borderColor: pinned ? 'var(--gold-border)' : undefined,
                      }}
                      title="Pin/unpin the root body in the air so limbs dangle freely"
                    >
                      {pinned ? '◉ pinned' : '○ pin torso'}
                    </button>
                    <button className="btn btn-ghost" onClick={handleReset} style={{ padding: '3px 8px', fontSize: 9 }}>
                      ↺ reset
                    </button>
                    <button className="btn btn-danger" onClick={stopDebug} style={{ padding: '3px 8px', fontSize: 9 }}>
                      ■ stop
                    </button>
                  </>
                ) : (
                  <button
                    className="btn btn-primary"
                    onClick={() => startDebug(charDef)}
                    disabled={!charDef}
                    style={{ padding: '3px 12px', fontSize: 9 }}
                  >
                    ▶ start
                  </button>
                )}
              </div>
            </div>
            <div style={{ height: expanded ? 'calc(100vh - 200px)' : 400, position: 'relative' }}>
              {isRunning && snapshot ? (
                <>
                  <PhysicsRenderer
                    snapshot={snapshot}
                    episodeReward={0}
                    episodeSteps={0}
                    onDebugMouse={handleDebugMouse}
                    autoFollow={false}
                    highlightBodyId={sceneSelectedType === 'body' ? sceneSelectedId : null}
                    onBodyClick={handleBodyClick}
                  />
                  <SceneHierarchy
                    snapshot={snapshot}
                    selectedId={sceneSelectedId}
                    selectedType={sceneSelectedType}
                    onSelect={handleSceneSelect}
                    onPropertyChange={handleScenePropertyChange}
                    editable={true}
                  />
                </>
              ) : (
                <div style={{
                  height: '100%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--text-dim)',
                  fontSize: 12,
                  flexDirection: 'column',
                  gap: 8,
                }}>
                  <span style={{ fontSize: 24 }}>⚙</span>
                  {charDef
                    ? 'Press Start to drop the body into the physics sandbox'
                    : 'Import a body or use the creature builder below'
                  }
                </div>
              )}
            </div>
          </div>

          {/* Measurements panel */}
          {isRunning && measurements && showOverlay && (
            <div className="panel">
              <div className="panel-header">◉ measurements</div>
              <div style={{ padding: '10px 14px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>

                {/* Body states */}
                <div>
                  <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.65)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>
                    Bodies
                  </div>
                  {Object.entries(measurements.bodies).map(([id, b]) => (
                    <div key={id} style={{ fontSize: 10, color: 'rgba(255,255,255,0.6)', fontFamily: 'Inter, sans-serif', lineHeight: 1.8 }}>
                      <span style={{ color: 'var(--gold)' }}>{id}</span>
                      {' '}x:{b.x.toFixed(2)} y:{b.y.toFixed(2)} θ:{(b.angle * 180 / Math.PI).toFixed(1)}° m:{b.mass}kg
                    </div>
                  ))}
                </div>

                {/* Joint states */}
                <div>
                  <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.65)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 6 }}>
                    Joints
                  </div>
                  {Object.entries(measurements.joints).map(([id, j]) => (
                    <div key={id} style={{ fontSize: 10, color: 'rgba(255,255,255,0.6)', fontFamily: 'Inter, sans-serif', lineHeight: 1.8 }}>
                      <span style={{ color: j.type === 'prismatic' ? '#66aaff' : '#ff9966' }}>{id}</span>
                      {j.type === 'prismatic' ? (
                        <>{' '}pos:{j.translation?.toFixed(3)}m [{j.lower?.toFixed(2)}, {j.upper?.toFixed(2)}]</>
                      ) : (
                        <>{' '}{(j.angle * 180 / Math.PI).toFixed(1)}° [{(j.lower * 180 / Math.PI).toFixed(0)}°, {(j.upper * 180 / Math.PI).toFixed(0)}°]
                          {' '}ω:{j.angVel?.toFixed(1)}
                        </>
                      )}
                    </div>
                  ))}

                  {/* Foot contacts */}
                  {Object.keys(measurements.footContacts).length > 0 && (
                    <div style={{ marginTop: 6 }}>
                      <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.65)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 4 }}>
                        Contacts
                      </div>
                      {Object.entries(measurements.footContacts).map(([id, contact]) => (
                        <span key={id} style={{
                          display: 'inline-block',
                          padding: '2px 6px',
                          marginRight: 4,
                          fontSize: 9,
                          fontFamily: 'Inter, sans-serif',
                          background: contact ? 'rgba(74,222,128,0.12)' : 'rgba(255,255,255,0.03)',
                          color: contact ? 'var(--green)' : 'rgba(255,255,255,0.55)',
                          border: `1px solid ${contact ? 'rgba(74,222,128,0.2)' : 'rgba(255,255,255,0.06)'}`,
                          borderRadius: 4,
                        }}>
                          {id}: {contact ? 'ON' : 'off'}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Observation inspector (live) */}
          {isRunning && measurements?.obs && showOverlay && (
            <div className="panel">
              <div className="panel-header">◎ observation vector ({measurements.obs.length}D)</div>
              <div style={{ padding: '8px 14px', fontFamily: 'Inter, sans-serif', fontSize: 10, lineHeight: 1.8 }}>
                {(() => {
                  const obs = measurements.obs
                  const labels = []
                  const useSinCos = charDef?.sinCosAngles ?? false
                  // Base: [height, angle, vx, vy, angvel] or [height, sinθ, cosθ, vx, vy, angvel]
                  labels.push({ i: 0, label: 'height', desc: 'torso.y' })
                  if (useSinCos) {
                    labels.push({ i: 1, label: 'sin(θ)', desc: 'sin(torso.rotation)' })
                    labels.push({ i: 2, label: 'cos(θ)', desc: 'cos(torso.rotation)' })
                    labels.push({ i: 3, label: 'vx', desc: 'torso.linvel.x' })
                    labels.push({ i: 4, label: 'vy', desc: 'torso.linvel.y' })
                    labels.push({ i: 5, label: 'angVel', desc: 'torso.angvel' })
                  } else {
                    labels.push({ i: 1, label: 'angle', desc: 'torso.rotation' })
                    labels.push({ i: 2, label: 'vx', desc: 'torso.linvel.x' })
                    labels.push({ i: 3, label: 'vy', desc: 'torso.linvel.y' })
                    labels.push({ i: 4, label: 'angVel', desc: 'torso.angvel' })
                  }
                  let idx = useSinCos ? 6 : 5
                  // Joint angles/velocities
                  for (const j of (charDef?.joints || [])) {
                    if (useSinCos && j.type !== 'prismatic') {
                      labels.push({ i: idx, label: `sin(${j.id})`, desc: `sin(${j.bodyB}.rot - ${j.bodyA}.rot)` })
                      labels.push({ i: idx + 1, label: `cos(${j.id})`, desc: `cos(relative angle)` })
                      labels.push({ i: idx + 2, label: `${j.id}_angVel`, desc: 'relative angvel' })
                      idx += 3
                    } else {
                      labels.push({ i: idx, label: `${j.id}_angle`, desc: `${j.bodyB}.rot - ${j.bodyA}.rot` })
                      labels.push({ i: idx + 1, label: `${j.id}_angVel`, desc: 'relative angvel' })
                      idx += 2
                    }
                  }
                  // Foot contacts
                  for (const b of (charDef?.bodies || [])) {
                    if (b.isFootBody) {
                      labels.push({ i: idx, label: `${b.id}_contact`, desc: 'ground contact' })
                      idx++
                    }
                  }
                  // Any remaining (terrain, etc.)
                  while (idx < obs.length) {
                    labels.push({ i: idx, label: `terrain[${idx - labels.length + (labels.length - idx)}]`, desc: 'heightfield' })
                    idx++
                  }

                  return labels.map(({ i, label, desc }) => (
                    <div key={i} style={{ display: 'flex', gap: 8, color: 'rgba(255,255,255,0.55)' }}>
                      <span style={{ color: 'rgba(255,255,255,0.55)', width: 20, textAlign: 'right' }}>[{i}]</span>
                      <span style={{ color: 'var(--gold)', width: 90, overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</span>
                      <span style={{ width: 70, textAlign: 'right' }}>
                        {obs[i] !== undefined ? obs[i].toFixed(4) : '—'}
                      </span>
                      <span style={{ color: 'rgba(255,255,255,0.55)' }}>{desc}</span>
                    </div>
                  ))
                })()}
              </div>
            </div>
          )}

          {/* Reward breakdown (live) */}
          {isRunning && measurements?.rewardBreakdown && showOverlay && (
            <div className="panel">
              <div className="panel-header">◎ reward breakdown (per step)</div>
              <div style={{ padding: '10px 14px', fontFamily: 'Inter, sans-serif', fontSize: 10 }}>
                {measurements.rewardBreakdown.components.map((c, i) => {
                  const maxBar = 1.5 // scale: 1.5 maps to full bar width
                  const barPct = Math.min(100, Math.abs(c.value) / maxBar * 100)
                  const isPositive = c.value >= 0
                  return (
                    <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                      <span style={{ width: 100, color: 'rgba(255,255,255,0.75)', flexShrink: 0 }}>{c.label}</span>
                      <div style={{
                        flex: 1,
                        height: 14,
                        background: 'rgba(255,255,255,0.03)',
                        borderRadius: 3,
                        position: 'relative',
                        overflow: 'hidden',
                      }}>
                        <div style={{
                          position: 'absolute',
                          [isPositive ? 'left' : 'right']: 0,
                          top: 0,
                          height: '100%',
                          width: `${barPct}%`,
                          background: isPositive ? 'rgba(74, 222, 128, 0.25)' : 'rgba(248, 113, 113, 0.25)',
                          borderRadius: 3,
                          transition: 'width 0.1s',
                        }} />
                      </div>
                      <span style={{
                        width: 60,
                        textAlign: 'right',
                        flexShrink: 0,
                        color: isPositive ? '#4ade80' : '#f87171',
                        fontVariantNumeric: 'tabular-nums',
                      }}>
                        {isPositive ? '+' : ''}{c.value.toFixed(3)}
                      </span>
                    </div>
                  )
                })}
                {/* Total */}
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 8,
                  marginTop: 8, paddingTop: 8,
                  borderTop: '1px solid rgba(255,255,255,0.06)',
                }}>
                  <span style={{ width: 100, color: 'rgba(255,255,255,0.7)', fontWeight: 600, flexShrink: 0 }}>Total</span>
                  <div style={{ flex: 1 }} />
                  <span style={{
                    width: 60,
                    textAlign: 'right',
                    flexShrink: 0,
                    color: measurements.rewardBreakdown.total >= 0 ? '#4ade80' : '#f87171',
                    fontWeight: 600,
                    fontSize: 12,
                    fontVariantNumeric: 'tabular-nums',
                  }}>
                    {measurements.rewardBreakdown.total >= 0 ? '+' : ''}{measurements.rewardBreakdown.total.toFixed(3)}
                  </span>
                </div>
                {/* Health status */}
                <div style={{ marginTop: 6, fontSize: 9, color: 'rgba(255,255,255,0.6)' }}>
                  {measurements.rewardBreakdown.healthy
                    ? <span style={{ color: '#4ade80' }}>healthy</span>
                    : <span style={{ color: '#f87171' }}>terminated</span>
                  }
                  {measurements.rewardBreakdown.done && <span> · done</span>}
                </div>
              </div>
            </div>
          )}

          {/* Body summary */}
          {charDef && !isRunning && (
            <div className="panel">
              <div className="panel-header">◫ body summary · {charDef.name || 'unnamed'}</div>
              <div style={{ padding: '12px 14px' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                  <Stat label="Bodies" value={charDef.bodies?.length || 0} />
                  <Stat label="Joints" value={charDef.joints?.length || 0} />
                  <Stat label="Total Mass" value={`${totalMass.toFixed(1)}kg`} />
                  <Stat label="Obs / Act" value={derived ? `${derived.obsSize} / ${derived.actionSize}` : '—'} />
                </div>

                {/* Bodies table */}
                <div style={{ marginTop: 12 }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10, fontFamily: 'Inter, sans-serif' }}>
                    <thead>
                      <tr style={{ color: 'rgba(255,255,255,0.65)', textAlign: 'left', borderBottom: '1px solid var(--border)' }}>
                        <th style={{ padding: '4px 6px' }}>id</th>
                        <th style={{ padding: '4px 6px' }}>shape</th>
                        <th style={{ padding: '4px 6px' }}>size</th>
                        <th style={{ padding: '4px 6px' }}>mass</th>
                        <th style={{ padding: '4px 6px' }}>friction</th>
                        <th style={{ padding: '4px 6px' }}>spawn</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(charDef.bodies || []).map(b => (
                        <tr key={b.id} style={{ color: 'rgba(255,255,255,0.55)', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                          <td style={{ padding: '4px 6px', color: b.fixed ? 'rgba(255,255,255,0.55)' : 'var(--gold)' }}>
                            {b.id}{b.fixed ? ' (fixed)' : ''}{b.isFootBody ? ' (foot)' : ''}
                          </td>
                          <td style={{ padding: '4px 6px' }}>{b.shape}</td>
                          <td style={{ padding: '4px 6px' }}>
                            {b.shape === 'box' ? `${b.w}×${b.h}` : b.shape === 'capsule' ? `r${b.radius} l${b.length}` : `r${b.radius}`}
                          </td>
                          <td style={{ padding: '4px 6px' }}>{b.mass}kg</td>
                          <td style={{ padding: '4px 6px' }}>{b.friction}</td>
                          <td style={{ padding: '4px 6px' }}>({b.spawnX}, {b.spawnY})</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Joints table */}
                {charDef.joints?.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10, fontFamily: 'Inter, sans-serif' }}>
                      <thead>
                        <tr style={{ color: 'rgba(255,255,255,0.65)', textAlign: 'left', borderBottom: '1px solid var(--border)' }}>
                          <th style={{ padding: '4px 6px' }}>id</th>
                          <th style={{ padding: '4px 6px' }}>type</th>
                          <th style={{ padding: '4px 6px' }}>bodies</th>
                          <th style={{ padding: '4px 6px' }}>range</th>
                          <th style={{ padding: '4px 6px' }}>torque</th>
                          <th style={{ padding: '4px 6px' }}>damping</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(charDef.joints || []).map(j => (
                          <tr key={j.id} style={{ color: 'rgba(255,255,255,0.55)', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                            <td style={{ padding: '4px 6px', color: j.type === 'prismatic' ? '#66aaff' : '#ff9966' }}>
                              {j.id}
                            </td>
                            <td style={{ padding: '4px 6px' }}>{j.type || 'revolute'}</td>
                            <td style={{ padding: '4px 6px' }}>{j.bodyA} → {j.bodyB}</td>
                            <td style={{ padding: '4px 6px' }}>
                              {j.type === 'prismatic'
                                ? `${j.lowerLimit?.toFixed(1)}m .. ${j.upperLimit?.toFixed(1)}m`
                                : `${(j.lowerLimit * 180 / Math.PI).toFixed(0)}° .. ${(j.upperLimit * 180 / Math.PI).toFixed(0)}°`
                              }
                            </td>
                            <td style={{ padding: '4px 6px' }}>{j.maxTorque || 0}</td>
                            <td style={{ padding: '4px 6px' }}>{j.damping ?? '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Right column (hidden when expanded) */}
        {!expanded && <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

          {/* Tests checklist */}
          <div className="panel">
            <div className="panel-header">⚡ verification tests</div>
            <div style={{ padding: '8px 0' }}>
              {TESTS.map(test => {
                const result = testResults[test.id]
                return (
                  <button
                    key={test.id}
                    onClick={() => setSelectedTest(selectedTest === test.id ? null : test.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: 8,
                      width: '100%',
                      padding: '8px 14px',
                      background: selectedTest === test.id ? 'rgba(255,255,255,0.03)' : 'transparent',
                      border: 'none',
                      borderLeft: `2px solid ${selectedTest === test.id ? 'var(--gold)' : 'transparent'}`,
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <span style={{
                      fontSize: 11,
                      color: result === 'pass' ? 'var(--green)' : result === 'fail' ? 'var(--red)' : 'rgba(255,255,255,0.55)',
                      width: 16,
                      flexShrink: 0,
                    }}>
                      {result === 'pass' ? '✓' : result === 'fail' ? '✗' : '○'}
                    </span>
                    <div>
                      <div style={{
                        fontSize: 11,
                        fontFamily: 'Inter, sans-serif',
                        color: selectedTest === test.id ? '#fff' : 'rgba(255,255,255,0.6)',
                        letterSpacing: '0.03em',
                      }}>
                        {test.label}
                      </div>
                      {selectedTest === test.id && (
                        <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.65)', marginTop: 3, lineHeight: 1.4 }}>
                          {test.desc}
                        </div>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Joint tester (when running) */}
          {isRunning && charDef?.joints?.length > 0 && (
            <div className="panel">
              <div className="panel-header">
                <span>◎ joint control</span>
                <button className="btn btn-ghost" onClick={handleReset} style={{ padding: '3px 8px', fontSize: 9 }}>
                  reset pose
                </button>
              </div>
              <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                {/* Joint selector */}
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {charDef.joints.map(j => (
                    <button
                      key={j.id}
                      onClick={() => handleJointSelect(j.id)}
                      style={{
                        flex: '1 1 auto',
                        padding: '5px 8px',
                        background: selectedJoint === j.id ? 'var(--gold-dim)' : 'var(--surface)',
                        border: `1px solid ${selectedJoint === j.id ? 'var(--gold-border)' : 'var(--border)'}`,
                        borderRadius: 5,
                        color: selectedJoint === j.id ? 'var(--gold)' : 'var(--text-dim)',
                        fontFamily: 'Inter, sans-serif',
                        fontSize: 10,
                        cursor: 'pointer',
                      }}
                    >
                      {j.id}
                      {(j.maxTorque ?? 0) === 0 && (
                        <span style={{ fontSize: 7, opacity: 0.4, marginLeft: 4, fontStyle: 'italic' }}>passive</span>
                      )}
                      <span style={{ fontSize: 8, opacity: 0.5, marginLeft: 3 }}>
                        {(j.type || 'revolute') === 'prismatic' ? '↔' : '↻'}
                      </span>
                    </button>
                  ))}
                </div>

                {/* Torque/force controls */}
                {selectedJoint && (
                  <>
                    {(() => {
                      const isPrismatic = charDef.joints.find(j => j.id === selectedJoint)?.type === 'prismatic'
                      const label = isPrismatic ? 'force' : 'torque'
                      const absMag = Math.abs(torqueDirection)
                      const pct = Math.round(absMag * 100)
                      const isActive = torqueDirection !== 0

                      return (
                        <>
                          <div style={{ display: 'flex', gap: 6 }}>
                            <button
                              className="btn"
                              onMouseDown={(e) => handleTorqueDragStart(e, -1)}
                              style={{
                                flex: 1, justifyContent: 'center', padding: '8px',
                                background: torqueDirection < 0 ? 'rgba(224,90,90,0.15)' : 'var(--surface)',
                                color: torqueDirection < 0 ? 'var(--red)' : 'var(--text-dim)',
                                border: `1px solid ${torqueDirection < 0 ? 'rgba(224,90,90,0.3)' : 'var(--border)'}`,
                                cursor: 'ew-resize',
                                userSelect: 'none',
                              }}
                            >
                              ← {label} −
                            </button>
                            <button
                              className="btn"
                              onMouseDown={(e) => handleTorqueDragStart(e, 1)}
                              style={{
                                flex: 1, justifyContent: 'center', padding: '8px',
                                background: torqueDirection > 0 ? 'rgba(74,222,128,0.12)' : 'var(--surface)',
                                color: torqueDirection > 0 ? 'var(--green)' : 'var(--text-dim)',
                                border: `1px solid ${torqueDirection > 0 ? 'rgba(74,222,128,0.2)' : 'var(--border)'}`,
                                cursor: 'ew-resize',
                                userSelect: 'none',
                              }}
                            >
                              {label} + →
                            </button>
                          </div>

                          {/* Torque magnitude indicator */}
                          {isActive && (
                            <div style={{
                              padding: '4px 8px',
                              background: 'rgba(255,255,255,0.03)',
                              borderRadius: 5,
                              border: '1px solid rgba(255,255,255,0.06)',
                              textAlign: 'center',
                            }}>
                              <div style={{
                                height: 4,
                                borderRadius: 2,
                                background: 'rgba(255,255,255,0.06)',
                                overflow: 'hidden',
                                marginBottom: 4,
                              }}>
                                <div style={{
                                  width: `${pct}%`,
                                  height: '100%',
                                  borderRadius: 2,
                                  background: torqueDirection > 0 ? 'var(--green)' : 'var(--red)',
                                  transition: 'width 0.05s',
                                }} />
                              </div>
                              <span style={{
                                fontSize: 9,
                                fontFamily: 'Inter, sans-serif',
                                color: torqueDirection > 0 ? 'var(--green)' : 'var(--red)',
                              }}>
                                {torqueDirection > 0 ? '+' : ''}{(torqueDirection * 100).toFixed(0)}% {label}
                              </span>
                              <span style={{ fontSize: 8, color: 'rgba(255,255,255,0.55)', marginLeft: 6 }}>
                                drag left/right to adjust
                              </span>
                            </div>
                          )}
                        </>
                      )
                    })()}


                    {/* Selected joint details */}
                    {measurements?.joints?.[selectedJoint] && (() => {
                      const j = measurements.joints[selectedJoint]
                      const jDef = charDef.joints.find(jd => jd.id === selectedJoint)
                      return (
                        <div style={{
                          padding: '8px 10px',
                          background: 'rgba(255,255,255,0.02)',
                          borderRadius: 6,
                          fontSize: 10,
                          fontFamily: 'Inter, sans-serif',
                          color: 'rgba(255,255,255,0.75)',
                          lineHeight: 1.8,
                        }}>
                          {(jDef?.maxTorque ?? 0) === 0 && (
                            <div style={{
                              color: 'rgba(255,255,255,0.65)',
                              fontStyle: 'italic',
                              marginBottom: 4,
                              padding: '3px 6px',
                              background: 'rgba(255,255,255,0.03)',
                              borderRadius: 4,
                              border: '1px solid rgba(255,255,255,0.06)',
                            }}>
                              passive joint — unactuated during training. Test torque applied for verification only.
                            </div>
                          )}
                          {j.type === 'prismatic' ? (
                            <>
                              <div>position: <span style={{ color: '#66aaff' }}>{j.translation?.toFixed(3)}m</span></div>
                              <div>range: [{j.lower?.toFixed(2)}, {j.upper?.toFixed(2)}]m</div>
                              <div>max force: {jDef?.maxTorque || 0}N {(jDef?.maxTorque ?? 0) === 0 && '(passive)'}</div>
                            </>
                          ) : (
                            <>
                              <div>angle: <span style={{ color: '#ff9966' }}>{(j.angle * 180 / Math.PI).toFixed(1)}°</span></div>
                              <div>range: [{(j.lower * 180 / Math.PI).toFixed(0)}°, {(j.upper * 180 / Math.PI).toFixed(0)}°]</div>
                              <div>angular vel: {j.angVel?.toFixed(2)} rad/s</div>
                              <div>max torque: {jDef?.maxTorque || 0}Nm {(jDef?.maxTorque ?? 0) === 0 && '(passive)'}</div>
                            </>
                          )}
                          <div>damping: {jDef?.damping ?? 0}</div>
                          {jDef?.kp && <div>PD: kp={jDef.kp} kd={jDef.kd}</div>}
                        </div>
                      )
                    })()}
                  </>
                )}

                <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.55)', lineHeight: 1.5 }}>
                  Select a joint then hold torque buttons. You can also grab and drag body parts with the mouse.
                </div>
              </div>
            </div>
          )}

          {/* Creature builder */}
          <div className="panel" style={{ flex: '1 1 250px' }}>
            <div className="panel-header">
              <span>◧ creature builder</span>
            </div>
            <div style={{ padding: '10px 12px' }}>
              <CreatureBuilder
                onCreatureChange={handleCreatureChange}
                disabled={false}
              />
            </div>
          </div>
        </div>}
      </div>
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.65)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 3 }}>
        {label}
      </div>
      <div style={{ fontSize: 14, fontFamily: 'Inter, sans-serif', color: '#fff', fontWeight: 500 }}>
        {value}
      </div>
    </div>
  )
}
