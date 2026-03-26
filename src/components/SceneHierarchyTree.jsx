/**
 * SceneHierarchyTree — Collapsible tree view of physics scene objects.
 *
 * Groups: Ground, Bodies, Joints, Foot Sensors.
 * Each leaf node is click-to-select. Live values (position, angle, contact)
 * update from the snapshot every frame.
 */

import { useState, memo, useCallback } from 'react'

const S = {
  group: {
    marginBottom: 2,
  },
  groupHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '4px 8px',
    cursor: 'pointer',
    fontSize: 10,
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
    color: 'rgba(255,255,255,0.45)',
    userSelect: 'none',
    borderRadius: 4,
  },
  chevron: {
    fontSize: 8,
    width: 12,
    textAlign: 'center',
    transition: 'transform 0.15s ease',
    color: 'rgba(255,255,255,0.3)',
  },
  node: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '3px 8px 3px 26px',
    cursor: 'pointer',
    fontSize: 11,
    fontFamily: "'DM Mono', monospace",
    borderRadius: 4,
    transition: 'background 0.1s ease',
    userSelect: 'none',
    whiteSpace: 'nowrap',
    overflow: 'hidden',
  },
  nodeId: {
    color: '#e0e0e8',
    flexShrink: 0,
  },
  nodeDetail: {
    color: 'rgba(255,255,255,0.35)',
    fontSize: 10,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
  },
  badge: {
    fontSize: 9,
    padding: '1px 4px',
    borderRadius: 3,
    flexShrink: 0,
  },
  count: {
    marginLeft: 'auto',
    fontSize: 9,
    color: 'rgba(255,255,255,0.25)',
  },
}

function fmtNum(n, decimals = 2) {
  return typeof n === 'number' ? n.toFixed(decimals) : '--'
}

function fmtDeg(rad) {
  return typeof rad === 'number' ? (rad * 180 / Math.PI).toFixed(1) + '\u00B0' : '--'
}

function shapeLabel(bodyDef) {
  if (bodyDef.shape === 'box') return `box ${bodyDef.w}\u00D7${bodyDef.h}`
  if (bodyDef.shape === 'capsule') return `cap r${bodyDef.radius || 0.04}`
  if (bodyDef.shape === 'ball') return `ball r${bodyDef.radius || 0.05}`
  return bodyDef.shape || '?'
}

// ── Group (collapsible) ──────────────────────────────────────────────────

function TreeGroup({ label, count, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div style={S.group}>
      <div
        style={S.groupHeader}
        onClick={() => setOpen(o => !o)}
        onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
      >
        <span style={{ ...S.chevron, transform: open ? 'rotate(90deg)' : 'rotate(0deg)' }}>&#9654;</span>
        <span>{label}</span>
        {count != null && <span style={S.count}>{count}</span>}
      </div>
      {open && children}
    </div>
  )
}

// ── Leaf node ────────────────────────────────────────────────────────────

const TreeNode = memo(function TreeNode({ id, type, selected, onSelect, children }) {
  const isSelected = selected
  const bg = isSelected ? 'rgba(226,185,111,0.15)' : 'transparent'
  const border = isSelected ? '1px solid rgba(226,185,111,0.25)' : '1px solid transparent'
  return (
    <div
      style={{ ...S.node, background: bg, border }}
      onClick={() => onSelect(id, type)}
      onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = 'rgba(255,255,255,0.04)' }}
      onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = 'transparent' }}
    >
      {children}
    </div>
  )
})

// ── Main tree ────────────────────────────────────────────────────────────

export default function SceneHierarchyTree({ charDef, snapshot, selectedId, selectedType, onSelect }) {
  if (!charDef) return <div style={{ padding: 12, color: 'rgba(255,255,255,0.3)', fontSize: 11 }}>No scene loaded</div>

  const bodies = charDef.bodies || []
  const joints = charDef.joints || []
  const ground = charDef.ground
  const footBodies = bodies.filter(b => b.isFootBody)
  const footContacts = snapshot?._footContacts || {}
  const jointData = snapshot?._joints || {}

  return (
    <div style={{ padding: '4px 0' }}>
      {/* Ground */}
      {ground && (
        <TreeGroup label="Ground" defaultOpen={false}>
          <TreeNode
            id="__ground__"
            type="ground"
            selected={selectedId === '__ground__' && selectedType === 'ground'}
            onSelect={onSelect}
          >
            <span style={S.nodeId}>ground</span>
            <span style={S.nodeDetail}>
              y={fmtNum(ground.y, 1)} f={fmtNum(ground.friction, 1)} r={fmtNum(ground.restitution, 1)}
            </span>
          </TreeNode>
        </TreeGroup>
      )}

      {/* Bodies */}
      <TreeGroup label="Bodies" count={bodies.length}>
        {bodies.map(b => {
          const snap = snapshot?.[b.id]
          return (
            <TreeNode
              key={b.id}
              id={b.id}
              type="body"
              selected={selectedId === b.id && selectedType === 'body'}
              onSelect={onSelect}
            >
              <span style={S.nodeId}>{b.id}</span>
              {b.fixed && (
                <span style={{ ...S.badge, background: 'rgba(255,255,255,0.08)', color: 'rgba(255,255,255,0.4)' }}>
                  fixed
                </span>
              )}
              {b.isFootBody && (
                <span style={{ ...S.badge, background: 'rgba(74,222,128,0.12)', color: '#4ade80' }}>
                  foot
                </span>
              )}
              <span style={S.nodeDetail}>
                [{shapeLabel(b)}, {b.mass}kg]
                {snap ? ` (${fmtNum(snap.x, 1)}, ${fmtNum(snap.y, 1)})` : ''}
              </span>
            </TreeNode>
          )
        })}
      </TreeGroup>

      {/* Joints */}
      {joints.length > 0 && (
        <TreeGroup label="Joints" count={joints.length}>
          {joints.map(j => {
            const jSnap = jointData[j.id]
            return (
              <TreeNode
                key={j.id}
                id={j.id}
                type="joint"
                selected={selectedId === j.id && selectedType === 'joint'}
                onSelect={onSelect}
              >
                <span style={S.nodeId}>{j.id}</span>
                <span style={{ ...S.badge, background: j.type === 'revolute' ? 'rgba(255,153,102,0.12)' : 'rgba(102,170,255,0.12)', color: j.type === 'revolute' ? '#ff9966' : '#66aaff' }}>
                  {j.type === 'revolute' ? 'rev' : 'pri'}
                </span>
                <span style={S.nodeDetail}>
                  {j.bodyA}{'\u2192'}{j.bodyB}
                  {jSnap ? ` ${fmtDeg(jSnap.angle)}` : ''}
                </span>
              </TreeNode>
            )
          })}
        </TreeGroup>
      )}

      {/* Foot Sensors */}
      {footBodies.length > 0 && (
        <TreeGroup label="Foot Sensors" count={footBodies.length} defaultOpen={false}>
          {footBodies.map(b => {
            const contact = footContacts[b.id]
            const on = !!contact
            return (
              <TreeNode
                key={b.id}
                id={b.id}
                type="sensor"
                selected={selectedId === b.id && selectedType === 'sensor'}
                onSelect={onSelect}
              >
                <span style={S.nodeId}>{b.id}</span>
                <span style={{
                  ...S.badge,
                  background: on ? 'rgba(74,222,128,0.15)' : 'rgba(255,255,255,0.05)',
                  color: on ? '#4ade80' : 'rgba(255,255,255,0.3)',
                }}>
                  {on ? 'ON' : 'off'}
                </span>
                {on && contact.x != null && (
                  <span style={S.nodeDetail}>
                    ({fmtNum(contact.x, 1)}, {fmtNum(contact.y, 1)})
                  </span>
                )}
              </TreeNode>
            )
          })}
        </TreeGroup>
      )}
    </div>
  )
}
