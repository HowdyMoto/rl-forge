// Shared CSS styles used across all modes
export const globalCSS = `
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #07070f;
    color: #e0e0e8;
    font-family: Inter, sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }

  :root {
    --gold: #e2b96f;
    --gold-dim: rgba(226,185,111,0.15);
    --gold-border: rgba(226,185,111,0.2);
    --surface: rgba(255,255,255,0.03);
    --surface-hover: rgba(255,255,255,0.05);
    --border: rgba(255,255,255,0.07);
    --text-dim: rgba(255,255,255,0.75);
    --red: #e05a5a;
    --green: #4ade80;
  }

  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }

  .panel-header {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text-dim);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .btn {
    border: none;
    border-radius: 6px;
    cursor: pointer;
    font-family: Inter, sans-serif;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    transition: all 0.15s ease;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
  }

  .btn-primary {
    background: var(--gold);
    color: #0a0a14;
  }
  .btn-primary:hover:not(:disabled) { background: #f0cc88; transform: translateY(-1px); }
  .btn-primary:active:not(:disabled) { transform: translateY(0); }

  .btn-ghost {
    background: var(--surface);
    color: var(--text-dim);
    border: 1px solid var(--border);
  }
  .btn-ghost:hover:not(:disabled) {
    background: var(--surface-hover);
    color: rgba(255,255,255,0.85);
    border-color: rgba(255,255,255,0.2);
  }

  .btn-danger {
    background: rgba(224,90,90,0.12);
    color: var(--red);
    border: 1px solid rgba(224,90,90,0.2);
  }
  .btn-danger:hover:not(:disabled) { background: rgba(224,90,90,0.2); }

  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 9px;
    border-radius: 20px;
    font-size: 11px;
    letter-spacing: 0.04em;
  }

  input[type=range] {
    -webkit-appearance: none;
    height: 3px;
    border-radius: 2px;
    background: rgba(255,255,255,0.1);
    outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--gold);
    cursor: pointer;
    box-shadow: 0 0 6px rgba(226,185,111,0.4);
  }
  input[type=range]:disabled::-webkit-slider-thumb {
    background: rgba(255,255,255,0.2);
    box-shadow: none;
  }

  [role="button"][aria-label="More information"]:focus-visible {
    outline: 1px solid rgba(226,185,111,0.6);
    outline-offset: 2px;
    border-radius: 2px;
    color: rgba(255,255,255,0.75) !important;
  }

  body::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 1000;
    opacity: 0.6;
  }

  /* Mode nav tabs */
  .mode-nav {
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
  }
  .mode-tab {
    padding: 10px 20px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-dim);
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: all 0.15s;
    text-decoration: none;
  }
  .mode-tab:hover {
    color: rgba(255,255,255,0.9);
    background: var(--surface-hover);
  }
  .mode-tab.active {
    color: var(--gold);
    border-bottom-color: var(--gold);
  }
`
