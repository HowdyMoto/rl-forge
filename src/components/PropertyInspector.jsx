/**
 * PropertyInspector — Editable property grid for a selected scene object.
 *
 * Shows read-only labels for definition properties (shape, mass, limits)
 * and editable number inputs for runtime properties (position, velocity, motor target).
 *
 * When a field is focused (user editing), incoming snapshot updates are suppressed
 * for that field to prevent fighting with live data.
 */

import { useState, useRef, useCallback, useEffect } from 'react'

const S = {
  container: {
    borderTop: '1px solid rgba(255,255,255,0.07)',
    padding: '8px 10px',
    fontSize: 11,
    fontFamily: "'DM Mono', monospace",
  },
  header: {
    fontSize: 10,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: 'rgba(255,255,255,0.45)',
    marginBottom: 6,
    display: 'flex',
    alignItems: 'center',
    gap: 6,
  },
  typeBadge: {
    fontSize: 9,
    padding: '1px 5px',
    borderRadius: 3,
  },
  section: {
    marginBottom: 8,
  },
  sectionLabel: {
    fontSize: 9,
    color: 'rgba(255,255,255,0.3)',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    marginBottom: 3,
  },
  row: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: '2px 0',
    gap: 8,
  },
  label: {
    color: 'rgba(255,255,255,0.5)',
    fontSize: 10,
    flexShrink: 0,
    minWidth: 60,
  },
  value: {
    color: '#e0e0e8',
    fontSize: 11,
    textAlign: 'right',
  },
  input: {
    background: 'rgba(255,255,255,0.06)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 4,
    color: '#e0e0e8',
    fontSize: 10,
    fontFamily: "'DM Mono', monospace",
    padding: '2px 6px',
    width: 72,
    textAlign: 'right',
    outline: 'none',
    transition: 'border-color 0.15s ease',
  },
  inputDisabled: {
    opacity: 0.4,
    cursor: 'not-allowed',
  },
}

function fmtNum(n, decimals = 3) {
  return typeof n === 'number' ? n.toFixed(decimals) : '--'
}

function fmtDeg(rad) {
  return typeof rad === 'number' ? (rad * 180 / Math.PI).toFixed(1) + '\u00B0' : '--'
}

function ReadOnlyRow({ label, value }) {
  return (
    <div style={S.row}>
      <span style={S.label}>{label}</span>
      <span style={S.value}>{value}</span>
    </div>
  )
}

function EditableRow({ label, value, field, disabled, onCommit, editingRef }) {
  const [localVal, setLocalVal] = useState('')
  const [editing, setEditing] = useState(false)

  // Sync from snapshot when not editing
  useEffect(() => {
    if (!editing) {
      setLocalVal(typeof value === 'number' ? value.toFixed(3) : '')
    }
  }, [value, editing])

  const handleFocus = () => {
    setEditing(true)
    editingRef.current = field
    setLocalVal(typeof value === 'number' ? value.toFixed(3) : '')
  }

  const handleBlur = () => {
    setEditing(false)
    editingRef.current = null
    const num = parseFloat(localVal)
    if (!isNaN(num) && num !== value) {
      onCommit(field, num)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.target.blur()
    } else if (e.key === 'Escape') {
      setLocalVal(typeof value === 'number' ? value.toFixed(3) : '')
      setEditing(false)
      editingRef.current = null
    }
  }

  return (
    <div style={S.row}>
      <span style={S.label}>{label}</span>
      <input
        type="text"
        style={{ ...S.input, ...(disabled ? S.inputDisabled : {}), borderColor: editing ? '#e2b96f' : 'rgba(255,255,255,0.1)' }}
        value={editing ? localVal : (typeof value === 'number' ? value.toFixed(3) : '')}
        onChange={e => setLocalVal(e.target.value)}
        onFocus={handleFocus}
        onBlur={handleBlur}
        onKeyDown={handleKeyDown}
        disabled={disabled}
      />
    </div>
  )
}


export default function PropertyInspector({ selectedId, selectedType, charDef, snapshot, onPropertyChange, editable }) {
  const editingRef = useRef(null)

  // All hooks must be called unconditionally (Rules of Hooks)
  const commitBody = useCallback((field, value) => {
    if (!onPropertyChange || !selectedId) return
    const snap = snapshot?.[selectedId]
    if (field === 'x' || field === 'y') {
      const x = field === 'x' ? value : snap?.x ?? 0
      const y = field === 'y' ? value : snap?.y ?? 0
      onPropertyChange({ type: 'SCENE_SET_BODY_POS', config: { bodyId: selectedId, x, y } })
    } else if (field === 'angle') {
      onPropertyChange({ type: 'SCENE_SET_BODY_ANGLE', config: { bodyId: selectedId, angle: value } })
    } else if (field === 'vx' || field === 'vy') {
      const vx = field === 'vx' ? value : snap?.vx ?? 0
      const vy = field === 'vy' ? value : snap?.vy ?? 0
      onPropertyChange({ type: 'SCENE_SET_BODY_LINVEL', config: { bodyId: selectedId, vx, vy } })
    } else if (field === 'angvel') {
      onPropertyChange({ type: 'SCENE_SET_BODY_ANGVEL', config: { bodyId: selectedId, angvel: value } })
    }
  }, [onPropertyChange, selectedId, snapshot])

  const commitJoint = useCallback((field, value) => {
    if (!onPropertyChange || !selectedId) return
    if (field === 'motorTarget') {
      onPropertyChange({ type: 'SCENE_SET_MOTOR_TARGET', config: { jointId: selectedId, target: value } })
    }
  }, [onPropertyChange, selectedId])

  if (!selectedId || !charDef) return null

  // ── Ground ──────────────────────────────────────────────────────────
  if (selectedType === 'ground') {
    const g = charDef.ground || {}
    return (
      <div style={S.container}>
        <div style={S.header}>
          <span>Ground</span>
          <span style={{ ...S.typeBadge, background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.5)' }}>static</span>
        </div>
        <ReadOnlyRow label="y" value={fmtNum(g.y, 2)} />
        <ReadOnlyRow label="friction" value={fmtNum(g.friction, 2)} />
        <ReadOnlyRow label="restitution" value={fmtNum(g.restitution, 2)} />
      </div>
    )
  }

  // ── Body ────────────────────────────────────────────────────────────
  if (selectedType === 'body') {
    const bodyDef = charDef.bodies?.find(b => b.id === selectedId)
    if (!bodyDef) return null
    const snap = snapshot?.[selectedId]
    const disabled = !editable

    const shapeDesc = bodyDef.shape === 'box' ? `box ${bodyDef.w} \u00D7 ${bodyDef.h}`
      : bodyDef.shape === 'capsule' ? `capsule r=${bodyDef.radius || 0.04} l=${bodyDef.length || 0.3}`
      : bodyDef.shape === 'ball' ? `ball r=${bodyDef.radius || 0.05}`
      : bodyDef.shape

    return (
      <div style={S.container}>
        <div style={S.header}>
          <span>{selectedId}</span>
          <span style={{ ...S.typeBadge, background: 'rgba(226,185,111,0.12)', color: '#e2b96f' }}>body</span>
          {bodyDef.fixed && <span style={{ ...S.typeBadge, background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.4)' }}>fixed</span>}
        </div>

        <div style={S.section}>
          <div style={S.sectionLabel}>Position & Rotation</div>
          <EditableRow label="x" value={snap?.x} field="x" disabled={disabled} onCommit={commitBody} editingRef={editingRef} />
          <EditableRow label="y" value={snap?.y} field="y" disabled={disabled} onCommit={commitBody} editingRef={editingRef} />
          <EditableRow label="angle" value={snap?.angle} field="angle" disabled={disabled} onCommit={commitBody} editingRef={editingRef} />
        </div>

        <div style={S.section}>
          <div style={S.sectionLabel}>Velocity</div>
          <EditableRow label="vx" value={snap?.vx} field="vx" disabled={disabled} onCommit={commitBody} editingRef={editingRef} />
          <EditableRow label="vy" value={snap?.vy} field="vy" disabled={disabled} onCommit={commitBody} editingRef={editingRef} />
          <EditableRow label="angvel" value={snap?.angvel} field="angvel" disabled={disabled} onCommit={commitBody} editingRef={editingRef} />
        </div>

        <div style={S.section}>
          <div style={S.sectionLabel}>Definition (read-only)</div>
          <ReadOnlyRow label="shape" value={shapeDesc} />
          <ReadOnlyRow label="mass" value={`${bodyDef.mass} kg`} />
          <ReadOnlyRow label="friction" value={fmtNum(bodyDef.friction, 2)} />
          <ReadOnlyRow label="restitution" value={fmtNum(bodyDef.restitution, 2)} />
          {bodyDef.minY != null && <ReadOnlyRow label="minY" value={fmtNum(bodyDef.minY, 2)} />}
          {bodyDef.maxAngle != null && <ReadOnlyRow label="maxAngle" value={fmtDeg(bodyDef.maxAngle)} />}
        </div>
      </div>
    )
  }

  // ── Joint ───────────────────────────────────────────────────────────
  if (selectedType === 'joint') {
    const jointDef = charDef.joints?.find(j => j.id === selectedId)
    if (!jointDef) return null
    const jSnap = snapshot?._joints?.[selectedId]
    const disabled = !editable

    return (
      <div style={S.container}>
        <div style={S.header}>
          <span>{selectedId}</span>
          <span style={{
            ...S.typeBadge,
            background: jointDef.type === 'revolute' ? 'rgba(255,153,102,0.12)' : 'rgba(102,170,255,0.12)',
            color: jointDef.type === 'revolute' ? '#ff9966' : '#66aaff',
          }}>
            {jointDef.type}
          </span>
        </div>

        <div style={S.section}>
          <div style={S.sectionLabel}>Live State</div>
          <ReadOnlyRow label="angle" value={jSnap ? fmtDeg(jSnap.angle) : '--'} />
          <ReadOnlyRow label="angVel" value={jSnap ? fmtNum(jSnap.angVel, 2) : '--'} />
        </div>

        {(jointDef.maxTorque > 0) && (
          <div style={S.section}>
            <div style={S.sectionLabel}>Motor</div>
            <EditableRow label="target" value={0} field="motorTarget" disabled={disabled} onCommit={commitJoint} editingRef={editingRef} />
          </div>
        )}

        <div style={S.section}>
          <div style={S.sectionLabel}>Definition (read-only)</div>
          <ReadOnlyRow label="bodyA" value={jointDef.bodyA} />
          <ReadOnlyRow label="bodyB" value={jointDef.bodyB} />
          <ReadOnlyRow label="lower" value={fmtDeg(jointDef.lowerLimit)} />
          <ReadOnlyRow label="upper" value={fmtDeg(jointDef.upperLimit)} />
          <ReadOnlyRow label="maxTorque" value={`${jointDef.maxTorque} N`} />
          {jointDef.controlMode && <ReadOnlyRow label="control" value={jointDef.controlMode} />}
          {jointDef.kp != null && <ReadOnlyRow label="kp" value={fmtNum(jointDef.kp, 0)} />}
          {jointDef.kd != null && <ReadOnlyRow label="kd" value={fmtNum(jointDef.kd, 0)} />}
        </div>
      </div>
    )
  }

  // ── Foot sensor (read-only) ─────────────────────────────────────────
  if (selectedType === 'sensor') {
    const bodyDef = charDef.bodies?.find(b => b.id === selectedId)
    const contact = snapshot?._footContacts?.[selectedId]
    const on = !!contact
    return (
      <div style={S.container}>
        <div style={S.header}>
          <span>{selectedId}</span>
          <span style={{ ...S.typeBadge, background: on ? 'rgba(74,222,128,0.12)' : 'rgba(255,255,255,0.08)', color: on ? '#4ade80' : 'rgba(255,255,255,0.4)' }}>
            {on ? 'contact' : 'no contact'}
          </span>
        </div>
        <ReadOnlyRow label="state" value={on ? 'ON' : 'off'} />
        {on && contact.x != null && (
          <>
            <ReadOnlyRow label="contact x" value={fmtNum(contact.x, 2)} />
            <ReadOnlyRow label="contact y" value={fmtNum(contact.y, 2)} />
          </>
        )}
        {bodyDef && <ReadOnlyRow label="body" value={bodyDef.id} />}
      </div>
    )
  }

  return null
}
