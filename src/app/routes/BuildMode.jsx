export default function BuildMode() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div className="panel" style={{ padding: 40, textAlign: 'center' }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>◧</div>
        <h2 style={{
          fontFamily: 'Inter, sans-serif',
          fontWeight: 600,
          fontSize: 18,
          color: '#fff',
          marginBottom: 8,
        }}>
          Build Mode
        </h2>
        <p style={{ color: 'var(--text-dim)', fontSize: 12, maxWidth: 400, margin: '0 auto', lineHeight: 1.6 }}>
          Assemble 2D bodies using MJCF format. Design joints, set physics properties,
          and author worlds for your agents to traverse.
        </p>
        <div style={{
          marginTop: 20,
          padding: '10px 16px',
          background: 'var(--gold-dim)',
          border: '1px solid var(--gold-border)',
          borderRadius: 8,
          color: 'var(--gold)',
          fontSize: 11,
          display: 'inline-block',
        }}>
          Coming in Phase 1 &amp; 5
        </div>
      </div>
    </div>
  )
}
