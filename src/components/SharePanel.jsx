/**
 * SharePanel — Export and share creature designs + trained policies.
 *
 * Generates a shareable JSON blob containing:
 *   - Creature body definition
 *   - Trained policy weights (if available)
 *   - Training metadata (best distance, reward, etc.)
 *
 * Users can:
 *   - Download the creature+policy as a .json file
 *   - Copy a base64-encoded URL parameter for sharing via link
 *   - Import a shared creature from file or pasted JSON
 */

import { useState, useRef } from 'react'

/**
 * Create a shareable package from creature def + policy weights.
 */
function createSharePackage(charDef, weights, metadata = {}) {
  return {
    version: 1,
    creature: charDef,
    weights: weights || null,
    metadata: {
      ...metadata,
      sharedAt: new Date().toISOString(),
    },
  }
}

/**
 * Compress a share package to a base64 URL-safe string.
 * Uses a simple JSON → base64 encoding (no compression library needed).
 */
function encodeShareString(pkg) {
  const json = JSON.stringify(pkg)
  // Use btoa for base64 encoding — works in browsers
  return btoa(encodeURIComponent(json))
}

function decodeShareString(str) {
  const json = decodeURIComponent(atob(str))
  return JSON.parse(json)
}

export default function SharePanel({ charDef, exportUrl, bestDistance, bestReward, onImportCreature }) {
  const [shareUrl, setShareUrl] = useState(null)
  const [copyStatus, setCopyStatus] = useState(null)
  const [importError, setImportError] = useState(null)
  const fileInputRef = useRef(null)

  const handleExportCreature = () => {
    const pkg = createSharePackage(charDef, null, {
      bestDistance: bestDistance || 0,
      bestReward: bestReward || 0,
    })
    const blob = new Blob([JSON.stringify(pkg, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${charDef?.name || 'creature'}_creature.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleCopyLink = () => {
    const pkg = createSharePackage(charDef, null, {
      bestDistance: bestDistance || 0,
      bestReward: bestReward || 0,
    })
    try {
      // Only encode creature def (no weights — too large for URL)
      const creatureOnly = { version: 1, creature: charDef, weights: null, metadata: pkg.metadata }
      const encoded = encodeShareString(creatureOnly)

      // Build share URL
      const baseUrl = window.location.origin + window.location.pathname
      const fullUrl = `${baseUrl}?creature=${encoded}`

      navigator.clipboard.writeText(fullUrl).then(() => {
        setCopyStatus('copied')
        setTimeout(() => setCopyStatus(null), 2000)
      }).catch(() => {
        setCopyStatus('failed')
        setTimeout(() => setCopyStatus(null), 2000)
      })
    } catch {
      setCopyStatus('too-large')
      setTimeout(() => setCopyStatus(null), 2000)
    }
  }

  const handleImportFile = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''
    setImportError(null)

    const reader = new FileReader()
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result)
        if (data.creature) {
          onImportCreature?.(data)
        } else if (data.bodies && data.joints) {
          // Direct charDef format
          onImportCreature?.({ creature: data })
        } else {
          setImportError('Invalid creature file')
        }
      } catch {
        setImportError('Could not parse file')
      }
    }
    reader.readAsText(file)
  }

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 6,
      fontFamily: '"DM Mono", monospace',
    }}>
      {/* Export creature design */}
      <button
        className="btn btn-ghost"
        onClick={handleExportCreature}
        disabled={!charDef}
        style={{ width: '100%', justifyContent: 'center', fontSize: 10 }}
      >
        ↓ Save Creature
      </button>

      {/* Copy share link */}
      <button
        className="btn btn-ghost"
        onClick={handleCopyLink}
        disabled={!charDef}
        style={{ width: '100%', justifyContent: 'center', fontSize: 10 }}
      >
        {copyStatus === 'copied' ? '✓ Link Copied' :
         copyStatus === 'failed' ? 'Copy Failed' :
         copyStatus === 'too-large' ? 'Creature too complex for URL' :
         '⎘ Copy Share Link'}
      </button>

      {/* Import creature */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleImportFile}
        style={{ display: 'none' }}
      />
      <button
        className="btn btn-ghost"
        onClick={() => fileInputRef.current?.click()}
        style={{ width: '100%', justifyContent: 'center', fontSize: 10 }}
      >
        ↑ Load Creature
      </button>

      {importError && (
        <div style={{
          padding: '4px 8px', borderRadius: 4,
          background: 'rgba(224,90,90,0.1)', border: '1px solid rgba(224,90,90,0.2)',
          color: 'var(--red)', fontSize: 9, textAlign: 'center',
        }}>
          {importError}
        </div>
      )}

      {/* Stats */}
      {charDef && (
        <div style={{
          fontSize: 9, color: 'rgba(255,255,255,0.3)',
          display: 'flex', flexDirection: 'column', gap: 2, marginTop: 2,
        }}>
          <div>creature: {charDef.name || 'unnamed'}</div>
          <div>{charDef.bodies?.length || 0} bodies, {charDef.joints?.length || 0} joints</div>
          {bestDistance > 0 && <div>best: {bestDistance.toFixed(1)}m</div>}
        </div>
      )}
    </div>
  )
}

export { createSharePackage, encodeShareString, decodeShareString }
