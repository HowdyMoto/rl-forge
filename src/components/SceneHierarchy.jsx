/**
 * SceneHierarchy — Collapsible left-side drawer showing the physics scene tree
 * and a property inspector for the selected object.
 *
 * Works in all modes by receiving snapshot + charDef as props.
 */

import { useState, useCallback } from 'react'
import SceneHierarchyTree from './SceneHierarchyTree.jsx'
import PropertyInspector from './PropertyInspector.jsx'

const PANEL_WIDTH = 280

const S = {
  wrapper: {
    position: 'absolute',
    top: 0,
    left: 0,
    bottom: 0,
    zIndex: 20,
    display: 'flex',
    pointerEvents: 'none',
  },
  panel: {
    width: PANEL_WIDTH,
    background: 'rgba(7,7,15,0.92)',
    borderRight: '1px solid rgba(255,255,255,0.07)',
    display: 'flex',
    flexDirection: 'column',
    pointerEvents: 'auto',
    backdropFilter: 'blur(12px)',
    overflow: 'hidden',
  },
  header: {
    padding: '8px 12px',
    borderBottom: '1px solid rgba(255,255,255,0.07)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    fontSize: 11,
    textTransform: 'uppercase',
    letterSpacing: '0.07em',
    color: 'rgba(255,255,255,0.5)',
    flexShrink: 0,
  },
  closeBtn: {
    background: 'none',
    border: 'none',
    color: 'rgba(255,255,255,0.4)',
    cursor: 'pointer',
    fontSize: 14,
    padding: '2px 4px',
    lineHeight: 1,
  },
  treeArea: {
    flex: 1,
    overflowY: 'auto',
    overflowX: 'hidden',
    minHeight: 0,
  },
  inspectorArea: {
    flexShrink: 0,
    maxHeight: '50%',
    overflowY: 'auto',
  },
  toggleBtn: {
    position: 'absolute',
    top: 8,
    left: 8,
    zIndex: 19,
    background: 'rgba(7,7,15,0.85)',
    border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 6,
    color: 'rgba(255,255,255,0.5)',
    cursor: 'pointer',
    padding: '6px 8px',
    fontSize: 11,
    fontFamily: "'DM Mono', monospace",
    letterSpacing: '0.05em',
    transition: 'all 0.15s ease',
    pointerEvents: 'auto',
    display: 'flex',
    alignItems: 'center',
    gap: 5,
  },
  sceneName: {
    color: '#e2b96f',
    fontSize: 11,
  },
}

// Custom scrollbar styling (injected once)
const scrollbarCSS = `
.scene-hierarchy-scroll::-webkit-scrollbar { width: 4px; }
.scene-hierarchy-scroll::-webkit-scrollbar-track { background: transparent; }
.scene-hierarchy-scroll::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
.scene-hierarchy-scroll::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }
`

export default function SceneHierarchy({ snapshot, selectedId, selectedType, onSelect, onPropertyChange, editable = false }) {
  const [open, setOpen] = useState(false)

  const charDef = snapshot?._charDef

  const handleSelect = useCallback((id, type) => {
    onSelect(id, type)
  }, [onSelect])

  const handleClearSelection = useCallback(() => {
    onSelect(null, null)
  }, [onSelect])

  // Toggle button (always visible when there's a snapshot)
  if (!snapshot) return null

  return (
    <>
      <style>{scrollbarCSS}</style>

      {/* Toggle button */}
      {!open && (
        <button
          style={S.toggleBtn}
          onClick={() => setOpen(true)}
          onMouseEnter={e => {
            e.currentTarget.style.borderColor = 'rgba(226,185,111,0.3)'
            e.currentTarget.style.color = 'rgba(255,255,255,0.7)'
          }}
          onMouseLeave={e => {
            e.currentTarget.style.borderColor = 'rgba(255,255,255,0.1)'
            e.currentTarget.style.color = 'rgba(255,255,255,0.5)'
          }}
        >
          <span style={{ fontSize: 13 }}>&#9776;</span>
          <span>Scene</span>
        </button>
      )}

      {/* Panel */}
      {open && (
        <div style={S.wrapper}>
          <div style={S.panel}>
            <div style={S.header}>
              <span>
                Scene{charDef?.name ? ': ' : ''}
                {charDef?.name && <span style={S.sceneName}>{charDef.name}</span>}
              </span>
              <button style={S.closeBtn} onClick={() => setOpen(false)} title="Close">&times;</button>
            </div>

            <div className="scene-hierarchy-scroll" style={S.treeArea}>
              <SceneHierarchyTree
                charDef={charDef}
                snapshot={snapshot}
                selectedId={selectedId}
                selectedType={selectedType}
                onSelect={handleSelect}
              />
            </div>

            {selectedId && (
              <div className="scene-hierarchy-scroll" style={S.inspectorArea}>
                <PropertyInspector
                  selectedId={selectedId}
                  selectedType={selectedType}
                  charDef={charDef}
                  snapshot={snapshot}
                  onPropertyChange={onPropertyChange}
                  editable={editable}
                />
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}
