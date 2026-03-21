import { useState, useEffect, useRef, useCallback } from 'react'
import CartPoleRenderer from './components/CartPoleRenderer.jsx'
import HopperRenderer from './components/HopperRenderer.jsx'
import Walker2DRenderer from './components/Walker2DRenderer.jsx'
import AcrobotRenderer from './components/AcrobotRenderer.jsx'
import TerrainRenderer from './components/TerrainRenderer.jsx'
import CreatureBuilder from './components/CreatureBuilder.jsx'
import SharePanel from './components/SharePanel.jsx'
import RewardChart from './components/RewardChart.jsx'
import TrainingCharts from './components/TrainingCharts.jsx'
import MetricsPanel from './components/MetricsPanel.jsx'
import HyperParams from './components/HyperParams.jsx'
import Tooltip from './components/Tooltip.jsx'
import { DEFAULT_PPO_CONFIG } from './rl/ppo.js'
import { decodeShareString } from './components/SharePanel.jsx'

const ENV_CONFIGS = {
  terrain: {
    label: 'Terrain',
    desc: 'Build & train creatures on terrain',
    solvedThreshold: 500,
    ppoOverrides: { hiddenSizes: [64, 64] },
    tooltip: 'Terrain platformer with PD-controlled creatures. Build a body in the creature builder, then train it to traverse procedurally generated terrain. Uses target joint angles (PD control) for smooth, natural-looking motion. Inspired by Peng et al. TerrainRL.',
  },
  cartpole: {
    label: 'CartPole',
    desc: 'Inverted pendulum · 4 obs · 1 action',
    solvedThreshold: 450,
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 512 },
    tooltip: 'Classic control benchmark. The agent applies horizontal force to a cart to keep an inverted pendulum upright. 4D observation (position, velocity, pole angle, angular velocity), 1 continuous action. Solved at mean reward ≥ 450.',
  },
  hopper: {
    label: 'Hopper',
    desc: 'Monopod hopper · 10 obs · 2 actions',
    solvedThreshold: 1500,
    ppoOverrides: { hiddenSizes: [64, 64] },
    tooltip: 'Physics-based locomotion (Rapier2D). A single-legged robot learns to hop forward without falling. 10D observation, 2 continuous joint torques. More challenging than CartPole — contact dynamics, sparse-ish reward, harder exploration. Solved at mean reward ≥ 1500.',
  },
  walker2d: {
    label: 'Walker2D',
    desc: 'Bipedal walker · 15 obs · 4 actions',
    solvedThreshold: 1500,
    ppoOverrides: { hiddenSizes: [64, 64] },
    tooltip: 'Bipedal locomotion (Rapier2D). A two-legged robot learns to walk forward without falling. 15D observation, 4 continuous joint torques (hip and knee per leg). Harder than Hopper — must coordinate both legs symmetrically. Solved at mean reward ≥ 1500.',
  },
  acrobot: {
    label: 'Acrobot',
    desc: 'Double pendulum · 10 obs · 2 actions',
    solvedThreshold: 500,
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 2048, numEpochs: 10 },
    tooltip: 'Swing-up double pendulum (Rapier2D). Two links hang from a fixed pivot — only the elbow joint is actuated. The agent must pump energy to swing the tip above the pivot. 10D observation, 1 effective action. Solved at mean reward ≥ 500.',
  },
}

const TOOLTIPS = {
  rewardCurve: 'Rolling 20-episode mean reward plotted over training updates. The primary visual signal for whether the agent is improving. The dashed line is the solved threshold for the current environment.',
  trainingMetrics: 'Key scalars logged after each policy update: reward trend, losses, and entropy. Useful for diagnosing training problems — e.g., flat reward with rising entropy often indicates poor exploration.',
  hyperparameters: 'The knobs that control PPO training dynamics. Locked during a run; set them before hitting Train. Defaults are reasonable starting points — most experiments only need learning rate tuning.',
  network: 'The neural network architecture for this environment. Actor and Critic are separate MLPs with the same hidden width but independent weights.',
  actor: 'The policy network. Takes an observation vector and outputs action parameters — for terrain mode, target joint angles driven by PD controllers.',
  critic: 'The value network V(s). Estimates expected future return from the current state. Used only during training to compute advantage estimates.',
  obs: 'Observation vector — body state, joint angles/velocities, foot contacts, and terrain heightfield.',
  act: 'Action output from the actor. For terrain mode: target joint angles (PD controllers convert these to smooth torques).',
  solved: 'The rolling mean reward over the last 20 episodes has exceeded the benchmark threshold.',
  avgPill: 'Current rolling mean reward (last 20 episodes) vs. the solved threshold for this environment.',
  creatureBuilder: 'Design your creature body. Pick a preset or customize body part sizes and masses. The creature will learn to traverse terrain using PD-controlled joints.',
  share: 'Save your creature design as a file, copy a share link, or load a friend\'s creature.',
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const css = `
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=Syne:wght@400;500;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #07070f;
    color: #e0e0e8;
    font-family: 'DM Mono', monospace;
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
    --text-dim: rgba(255,255,255,0.5);
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
    font-family: 'DM Mono', monospace;
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
    color: rgba(255,255,255,0.6);
    border-color: rgba(255,255,255,0.15);
  }

  .btn-danger {
    background: rgba(224,90,90,0.12);
    color: var(--red);
    border: 1px solid rgba(224,90,90,0.2);
  }
  .btn-danger:hover:not(:disabled) { background: rgba(224,90,90,0.2); }

  .btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .env-tab {
    flex: 1;
    padding: 7px 10px;
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    color: var(--text-dim);
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    cursor: pointer;
    transition: all 0.15s;
    text-align: center;
  }
  .env-tab.active {
    color: var(--gold);
    border-bottom-color: var(--gold);
    background: var(--gold-dim);
  }
  .env-tab:hover:not(.active):not(:disabled) {
    color: rgba(255,255,255,0.5);
    background: var(--surface-hover);
  }
  .env-tab:disabled { opacity: 0.35; cursor: not-allowed; }

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
    color: rgba(255,255,255,0.5) !important;
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
`

const TRAINING_STATES = { IDLE: 'idle', RUNNING: 'running', PAUSED: 'paused' }

export default function App() {
  const workerRef = useRef(null)
  const [envType, setEnvType] = useState('terrain')
  const [trainingState, setTrainingState] = useState(TRAINING_STATES.IDLE)

  // Render states
  const [cartpoleState, setCartpoleState] = useState(null)
  const [hopperSnapshot, setHopperSnapshot] = useState(null)

  const [episodeReward, setEpisodeReward] = useState(0)
  const [episodeSteps, setEpisodeSteps] = useState(0)
  const [metrics, setMetrics] = useState([])
  const [episodes, setEpisodes] = useState(0)
  const [status, setStatus] = useState('idle')
  const [backend, setBackend] = useState('')
  const [ppoConfig, setPpoConfig] = useState(DEFAULT_PPO_CONFIG)
  const [exportUrl, setExportUrl] = useState(null)
  const [selectedTab, setSelectedTab] = useState('sim')
  const fileInputRef = useRef(null)

  // Creature builder state
  const [customCharDef, setCustomCharDef] = useState(null)
  const [bestDistance, setBestDistance] = useState(0)

  // Physics debug mode
  const [debugMode, setDebugMode] = useState(false)
  const [debugJoint, setDebugJoint] = useState(null)
  const [debugDirection, setDebugDirection] = useState(0)

  const isRunning = trainingState === TRAINING_STATES.RUNNING
  const isTerrainMode = envType === 'terrain'

  // Check for shared creature in URL on load
  useEffect(() => {
    try {
      const params = new URLSearchParams(window.location.search)
      const creatureParam = params.get('creature')
      if (creatureParam) {
        const pkg = decodeShareString(creatureParam)
        if (pkg.creature) {
          setCustomCharDef(pkg.creature)
          setEnvType('terrain')
          // Clean URL
          window.history.replaceState({}, '', window.location.pathname)
        }
      }
    } catch { /* ignore invalid share links */ }
  }, [])

  const initWorker = useCallback(() => {
    if (workerRef.current) workerRef.current.terminate()
    const worker = new Worker(
      new URL('./rl/trainWorker.js', import.meta.url),
      { type: 'module' }
    )
    worker.onmessage = (e) => {
      const msg = e.data
      switch (msg.type) {
        case 'RENDER_FRAME':
          setCartpoleState(msg.state)
          setEpisodeReward(msg.episodeReward)
          setEpisodeSteps(msg.episodeSteps)
          break
        case 'RENDER_SNAPSHOT':
          setHopperSnapshot(msg.snapshot)
          setEpisodeReward(msg.episodeReward)
          setEpisodeSteps(msg.episodeSteps)
          // Track best distance for terrain mode
          if (msg.snapshot?._maxDistance) {
            setBestDistance(prev => Math.max(prev, msg.snapshot._maxDistance))
          }
          break
        case 'METRICS':
          setMetrics(prev => [...prev.slice(-300), msg.data])
          break
        case 'EPISODE':
          setEpisodes(msg.data.episode + 1)
          break
        case 'STATUS':
          setStatus(msg.msg)
          break
        case 'BACKEND':
          setBackend(msg.deviceName ? `${msg.backend} · ${msg.deviceName}` : msg.backend)
          break
        case 'EXPORT_URL': {
          setExportUrl(msg.url)
          // Auto-trigger download
          const a = document.createElement('a')
          a.href = msg.url
          a.download = 'policy_weights.json'
          a.click()
          break
        }
      }
    }
    workerRef.current = worker
    return worker
  }, [])

  useEffect(() => {
    initWorker()
    return () => workerRef.current?.terminate()
  }, [])

  const handleTrain = () => {
    const worker = initWorker()
    setMetrics([])
    setEpisodes(0)
    setCartpoleState(null)
    setHopperSnapshot(null)
    setExportUrl(null)
    setBestDistance(0)
    setTrainingState(TRAINING_STATES.RUNNING)

    const envCfg = ENV_CONFIGS[envType]
    const config = {
      envType,
      ppoConfig: { ...ppoConfig, ...envCfg.ppoOverrides },
      maxUpdates: 3000,
    }

    // Pass custom character definition for terrain mode
    if (isTerrainMode && customCharDef) {
      config.charDef = customCharDef
    }

    worker.postMessage({ type: 'START', config })
  }

  const handlePause = () => {
    if (isRunning) {
      workerRef.current?.postMessage({ type: 'PAUSE' })
      setTrainingState(TRAINING_STATES.PAUSED)
    } else {
      workerRef.current?.postMessage({ type: 'RESUME' })
      setTrainingState(TRAINING_STATES.RUNNING)
    }
  }

  const handleStop = () => {
    workerRef.current?.postMessage({ type: 'STOP' })
    setTrainingState(TRAINING_STATES.IDLE)
    setDebugMode(false)
    setDebugJoint(null)
    setDebugDirection(0)
  }

  const handlePhysicsDebug = () => {
    const worker = initWorker()
    setMetrics([])
    setEpisodes(0)
    setCartpoleState(null)
    setHopperSnapshot(null)
    setExportUrl(null)
    setDebugMode(true)
    setDebugJoint(null)
    setDebugDirection(0)
    setTrainingState(TRAINING_STATES.RUNNING)
    setSelectedTab('sim')

    const config = { charDef: customCharDef }
    worker.postMessage({ type: 'PHYSICS_DEBUG', config })
  }

  const handleDebugJointChange = (jointId) => {
    setDebugJoint(jointId)
    setDebugDirection(0)
    workerRef.current?.postMessage({ type: 'DEBUG_CONTROL', config: { joint: jointId, direction: 0 } })
  }

  const handleDebugDirection = (dir) => {
    setDebugDirection(dir)
    workerRef.current?.postMessage({ type: 'DEBUG_CONTROL', config: { joint: debugJoint, direction: dir } })
  }

  const handleDebugReset = () => {
    workerRef.current?.postMessage({ type: 'DEBUG_RESET' })
  }

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

  const handleImportPlay = (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    e.target.value = ''
    const reader = new FileReader()
    reader.onload = () => {
      const weights = JSON.parse(reader.result)
      const worker = initWorker()
      setMetrics([])
      setEpisodes(0)
      setCartpoleState(null)
      setHopperSnapshot(null)
      setExportUrl(null)
      setTrainingState(TRAINING_STATES.RUNNING)
      setSelectedTab('sim')

      const envCfg = ENV_CONFIGS[envType]
      const config = {
        envType,
        ppoConfig: { ...ppoConfig, ...envCfg.ppoOverrides },
        weights,
      }
      if (isTerrainMode && customCharDef) {
        config.charDef = customCharDef
      }
      worker.postMessage({ type: 'PLAYBACK', config })
    }
    reader.readAsText(file)
  }

  const handleImportCreature = (pkg) => {
    if (pkg.creature) {
      setCustomCharDef(pkg.creature)
      setEnvType('terrain')
    }
  }

  const latestReward = metrics[metrics.length - 1]?.meanReward20 ?? 0
  const solvedThreshold = ENV_CONFIGS[envType].solvedThreshold
  const isSolved = latestReward >= solvedThreshold

  // Renderer selection
  const renderSnapshotProps = { snapshot: hopperSnapshot, episodeReward, episodeSteps }
  let simContent
  if (envType === 'terrain') {
    simContent = <TerrainRenderer {...renderSnapshotProps} charDef={customCharDef} onDebugMouse={debugMode ? handleDebugMouse : undefined} />
  } else if (envType === 'cartpole') {
    simContent = <CartPoleRenderer state={cartpoleState} episodeReward={episodeReward} episodeSteps={episodeSteps} />
  } else if (envType === 'hopper') {
    simContent = <HopperRenderer {...renderSnapshotProps} />
  } else if (envType === 'walker2d') {
    simContent = <Walker2DRenderer {...renderSnapshotProps} />
  } else if (envType === 'acrobot') {
    simContent = <AcrobotRenderer {...renderSnapshotProps} />
  }

  // Network description
  const getNetworkDesc = () => {
    if (isTerrainMode && customCharDef) {
      const obs = customCharDef.obsSize || '?'
      const act = customCharDef.actionSize || '?'
      return {
        actor: `${obs} → 128 → 128 → ${act}`,
        obs: `[h, θ, vx, vy, ω, joints, contacts, terrain]`,
        actions: `[${act} target angles → PD control]`,
      }
    }
    const descs = {
      cartpole: { actor: '4 → 64 → 64 → 1', obs: '[x, ẋ, θ, θ̇]', actions: 'force' },
      hopper: { actor: '10 → 128 → 128 → 2', obs: '[y, θ, vx, vy, ω, hip∠, hip·ω, knee∠, knee·ω, contact]', actions: '[τ_hip, τ_knee]' },
      walker2d: { actor: '15 → 128 → 128 → 4', obs: '[y, θ, vx, vy, ω, 4×(j∠, j·ω), Lcontact, Rcontact]', actions: '[τ_lhip, τ_lknee, τ_rhip, τ_rknee]' },
      acrobot: { actor: '10 → 64 → 64 → 2', obs: '[y, θ, vx, vy, ω, shoulder∠, shoulder·ω, elbow∠, elbow·ω, contact]', actions: '[—, τ_elbow]' },
    }
    return descs[envType] || descs.cartpole
  }
  const networkDesc = getNetworkDesc()

  const trainLabel = isTerrainMode ? (customCharDef?.name || 'Creature') : ENV_CONFIGS[envType].label

  return (
    <>
      <style>{css}</style>
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        padding: '20px 24px',
        gap: 16,
        maxWidth: 1280,
        margin: '0 auto',
      }}>

        {/* Header */}
        <header style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          paddingBottom: 16,
          borderBottom: '1px solid var(--border)',
        }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12 }}>
            <h1 style={{
              fontFamily: 'Syne, sans-serif',
              fontWeight: 800,
              fontSize: 22,
              letterSpacing: '-0.02em',
              color: '#fff',
            }}>
              RL<span style={{ color: 'var(--gold)' }}>Forge</span>
            </h1>
            <span style={{ fontSize: 10, color: 'var(--text-dim)', letterSpacing: '0.08em' }}>
              MILESTONE 3 · TERRAIN
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            {isTerrainMode && bestDistance > 0 && (
              <span className="pill" style={{ background: 'var(--gold-dim)', color: 'var(--gold)', border: '1px solid var(--gold-border)' }}>
                best: {bestDistance.toFixed(1)}m
              </span>
            )}
            {isSolved && (
              <span className="pill" style={{ background: 'rgba(74,222,128,0.12)', color: 'var(--green)', border: '1px solid rgba(74,222,128,0.2)', display: 'inline-flex', alignItems: 'center' }}>
                ✓ solved
                <Tooltip text={TOOLTIPS.solved} />
              </span>
            )}
            {metrics.length > 0 && (
              <span className="pill" style={{ background: 'var(--gold-dim)', color: 'var(--gold)', border: '1px solid var(--gold-border)', display: 'inline-flex', alignItems: 'center' }}>
                {latestReward.toFixed(0)} / {solvedThreshold} avg
                <Tooltip text={TOOLTIPS.avgPill} />
              </span>
            )}
          </div>
        </header>

        {/* Main layout */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 16, flex: 1 }}>

          {/* Left column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Env selector + sim/chart tabs */}
            <div className="panel">
              <div style={{ display: 'flex', borderBottom: '1px solid var(--border)' }}>
                {Object.entries(ENV_CONFIGS).map(([key, cfg]) => (
                  <button
                    key={key}
                    className={`env-tab ${envType === key ? 'active' : ''}`}
                    onClick={() => { setEnvType(key); setMetrics([]); setEpisodes(0); setBestDistance(0) }}
                    disabled={isRunning || trainingState === TRAINING_STATES.PAUSED}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2 }}>
                      {cfg.label}
                      <Tooltip text={cfg.tooltip} />
                    </span>
                    <span style={{ fontSize: 10, display: 'block', marginTop: 2, opacity: 0.7 }}>
                      {cfg.desc}
                    </span>
                  </button>
                ))}

                {/* Sim/Chart toggle on right */}
                <div style={{ marginLeft: 'auto', display: 'flex', borderLeft: '1px solid var(--border)' }}>
                  {['sim', 'chart'].map(tab => (
                    <button
                      key={tab}
                      onClick={() => setSelectedTab(tab)}
                      style={{
                        background: selectedTab === tab ? 'rgba(255,255,255,0.04)' : 'transparent',
                        border: 'none',
                        color: selectedTab === tab ? 'rgba(255,255,255,0.6)' : 'var(--text-dim)',
                        fontFamily: '"DM Mono", monospace',
                        fontSize: 9,
                        letterSpacing: '0.1em',
                        textTransform: 'uppercase',
                        padding: '0 14px',
                        cursor: 'pointer',
                      }}
                    >
                      {tab === 'sim' ? '⬡ sim' : '◈ chart'}
                    </button>
                  ))}
                </div>
              </div>

              <div style={{ height: selectedTab === 'chart' ? 440 : 300 }}>
                {selectedTab === 'sim' ? simContent : (
                  <TrainingCharts metrics={metrics} solvedThreshold={solvedThreshold} />
                )}
              </div>
            </div>

            {/* Always-visible reward chart when sim tab is active */}
            {selectedTab === 'sim' && (
              <div className="panel" style={{ flex: '1 1 160px' }}>
                <div className="panel-header">
                  <span style={{ display: 'flex', alignItems: 'center' }}>
                    ◈ reward curve
                    <Tooltip text={TOOLTIPS.rewardCurve} />
                  </span>
                  <span>target: {solvedThreshold}</span>
                </div>
                <div style={{ height: 150, padding: '8px 4px' }}>
                  <RewardChart metrics={metrics} solvedThreshold={solvedThreshold} />
                </div>
              </div>
            )}

            {/* Metrics */}
            <div className="panel">
              <div className="panel-header">
                <span style={{ display: 'flex', alignItems: 'center' }}>
                  ◉ training metrics
                  <Tooltip text={TOOLTIPS.trainingMetrics} />
                </span>
              </div>
              <div style={{ padding: '14px 16px' }}>
                <MetricsPanel metrics={metrics} episodes={episodes} status={status} backend={backend} />
              </div>
            </div>
          </div>

          {/* Right column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Controls */}
            <div className="panel">
              <div className="panel-header">◎ controls</div>
              <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
                <button
                  className="btn btn-primary"
                  onClick={handleTrain}
                  disabled={isRunning || trainingState === TRAINING_STATES.PAUSED}
                  style={{ width: '100%', justifyContent: 'center', padding: '10px 14px', fontSize: 12 }}
                >
                  ▶ Train {trainLabel}
                </button>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <button
                    className="btn btn-ghost"
                    onClick={handlePause}
                    disabled={trainingState === TRAINING_STATES.IDLE}
                    style={{ justifyContent: 'center' }}
                  >
                    {trainingState === TRAINING_STATES.PAUSED ? '▶ Resume' : '⏸ Pause'}
                  </button>
                  <button
                    className="btn btn-danger"
                    onClick={handleStop}
                    disabled={trainingState === TRAINING_STATES.IDLE}
                    style={{ justifyContent: 'center' }}
                  >
                    ■ Stop
                  </button>
                </div>

                <div style={{ height: 1, background: 'var(--border)', margin: '4px 0' }} />

                <button
                  className="btn btn-ghost"
                  onClick={() => {
                    setExportUrl(null)
                    workerRef.current?.postMessage({ type: 'EXPORT' })
                  }}
                  disabled={metrics.length === 0}
                  style={{ width: '100%', justifyContent: 'center' }}
                >
                  {exportUrl ? '✓ Export Ready — Click Again to Re-export' : '↓ Export Weights'}
                </button>

                {exportUrl && (
                  <a href={exportUrl} download={`${customCharDef?.name || envType}_policy.json`}
                    onClick={() => {/* auto-download triggered */}}
                    style={{
                      display: 'block', textAlign: 'center', padding: '8px',
                      background: 'rgba(74,222,128,0.12)', border: '1px solid rgba(74,222,128,0.25)',
                      borderRadius: 6, color: 'var(--green)', fontSize: 11,
                      fontFamily: '"DM Mono", monospace', letterSpacing: '0.06em', textDecoration: 'none',
                      fontWeight: 500,
                    }}
                  >
                    ↓ Download {customCharDef?.name || envType}_policy.json
                  </a>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".json"
                  onChange={handleImportPlay}
                  style={{ display: 'none' }}
                />
                <button
                  className="btn btn-ghost"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isRunning || trainingState === TRAINING_STATES.PAUSED}
                  style={{ width: '100%', justifyContent: 'center' }}
                >
                  ↑ Import &amp; Play
                </button>

                {isTerrainMode && (
                  <>
                    <div style={{ height: 1, background: 'var(--border)', margin: '4px 0' }} />
                    <button
                      className="btn btn-ghost"
                      onClick={debugMode ? handleStop : handlePhysicsDebug}
                      disabled={isRunning && !debugMode}
                      style={{
                        width: '100%', justifyContent: 'center',
                        color: debugMode ? 'var(--gold)' : undefined,
                        borderColor: debugMode ? 'var(--gold-border)' : undefined,
                      }}
                    >
                      {debugMode ? '■ Stop Debug' : '⚙ Physics Debug'}
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Debug controls (visible when debug mode is active) */}
            {debugMode && isTerrainMode && customCharDef && (
              <div className="panel">
                <div className="panel-header">
                  <span>⚙ joint tester</span>
                  <button
                    className="btn btn-ghost"
                    onClick={handleDebugReset}
                    style={{ padding: '3px 8px', fontSize: 9 }}
                  >
                    reset pose
                  </button>
                </div>
                <div style={{ padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {/* Joint selector */}
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                    {(customCharDef.joints || []).map(j => (
                      <button
                        key={j.id}
                        onClick={() => handleDebugJointChange(debugJoint === j.id ? null : j.id)}
                        style={{
                          flex: '1 1 auto',
                          padding: '5px 8px',
                          background: debugJoint === j.id ? 'var(--gold-dim)' : 'var(--surface)',
                          border: `1px solid ${debugJoint === j.id ? 'var(--gold-border)' : 'var(--border)'}`,
                          borderRadius: 5,
                          color: debugJoint === j.id ? 'var(--gold)' : 'var(--text-dim)',
                          fontFamily: '"DM Mono", monospace',
                          fontSize: 10,
                          cursor: 'pointer',
                          letterSpacing: '0.03em',
                        }}
                      >
                        {j.id}
                      </button>
                    ))}
                  </div>

                  {/* Torque direction controls */}
                  {debugJoint && (
                    <div style={{ display: 'flex', gap: 6 }}>
                      <button
                        className="btn"
                        onMouseDown={() => handleDebugDirection(-1)}
                        onMouseUp={() => handleDebugDirection(0)}
                        onMouseLeave={() => handleDebugDirection(0)}
                        style={{
                          flex: 1, justifyContent: 'center', padding: '8px',
                          background: debugDirection === -1 ? 'rgba(224,90,90,0.15)' : 'var(--surface)',
                          color: debugDirection === -1 ? 'var(--red)' : 'var(--text-dim)',
                          border: `1px solid ${debugDirection === -1 ? 'rgba(224,90,90,0.3)' : 'var(--border)'}`,
                        }}
                      >
                        ← torque −
                      </button>
                      <button
                        className="btn"
                        onMouseDown={() => handleDebugDirection(1)}
                        onMouseUp={() => handleDebugDirection(0)}
                        onMouseLeave={() => handleDebugDirection(0)}
                        style={{
                          flex: 1, justifyContent: 'center', padding: '8px',
                          background: debugDirection === 1 ? 'rgba(74,222,128,0.12)' : 'var(--surface)',
                          color: debugDirection === 1 ? 'var(--green)' : 'var(--text-dim)',
                          border: `1px solid ${debugDirection === 1 ? 'rgba(74,222,128,0.2)' : 'var(--border)'}`,
                        }}
                      >
                        torque + →
                      </button>
                    </div>
                  )}

                  <div style={{ fontSize: 9, color: 'rgba(255,255,255,0.25)', lineHeight: 1.5 }}>
                    Select a joint, then hold torque buttons to test. Arcs show joint limits, line shows current angle.
                  </div>
                </div>
              </div>
            )}

            {/* Creature Builder (terrain mode only) */}
            {isTerrainMode && (
              <div className="panel" style={{ flex: '2 1 280px' }}>
                <div className="panel-header">
                  <span style={{ display: 'flex', alignItems: 'center' }}>
                    ◧ creature builder
                    <Tooltip text={TOOLTIPS.creatureBuilder} />
                  </span>
                  <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.05em' }}>
                    {isRunning ? 'LOCKED' : 'EDIT'}
                  </span>
                </div>
                <div style={{ padding: '10px 12px' }}>
                  <CreatureBuilder
                    onCreatureChange={setCustomCharDef}
                    disabled={isRunning || trainingState === TRAINING_STATES.PAUSED}
                  />
                </div>
              </div>
            )}

            {/* Share panel (terrain mode only) */}
            {isTerrainMode && (
              <div className="panel">
                <div className="panel-header">
                  <span style={{ display: 'flex', alignItems: 'center' }}>
                    ⎘ share
                    <Tooltip text={TOOLTIPS.share} />
                  </span>
                </div>
                <div style={{ padding: '10px 12px' }}>
                  <SharePanel
                    charDef={customCharDef}
                    exportUrl={exportUrl}
                    bestDistance={bestDistance}
                    bestReward={latestReward}
                    onImportCreature={handleImportCreature}
                  />
                </div>
              </div>
            )}

            {/* Hyperparameters */}
            {!isTerrainMode && (
              <div className="panel" style={{ flex: 1 }}>
                <div className="panel-header">
                  <span style={{ display: 'flex', alignItems: 'center' }}>
                    ◧ hyperparameters
                    <Tooltip text={TOOLTIPS.hyperparameters} />
                  </span>
                  <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.05em' }}>
                    {isRunning ? 'LOCKED' : 'EDIT BEFORE TRAIN'}
                  </span>
                </div>
                <div style={{ padding: '14px 16px', overflowY: 'auto' }}>
                  <HyperParams onChange={setPpoConfig} disabled={isRunning} />
                </div>
              </div>
            )}

            {/* Network info */}
            <div className="panel">
              <div className="panel-header">
                <span style={{ display: 'flex', alignItems: 'center' }}>
                  ◫ network · {trainLabel}
                  <Tooltip text={TOOLTIPS.network} />
                </span>
              </div>
              <div style={{ padding: '12px 14px', fontSize: 11, color: 'rgba(255,255,255,0.7)', lineHeight: 1.8, fontFamily: '"DM Mono", monospace' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  Actor: {networkDesc.actor}
                  <Tooltip text={TOOLTIPS.actor} />
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  Critic: same width → 1
                  <Tooltip text={TOOLTIPS.critic} />
                </div>
                <div style={{ marginTop: 6, color: 'rgba(255,255,255,0.45)', fontSize: 10, lineHeight: 1.6, display: 'flex', alignItems: 'center' }}>
                  obs: {networkDesc.obs}
                  <Tooltip text={TOOLTIPS.obs} />
                </div>
                <div style={{ color: 'rgba(255,255,255,0.45)', fontSize: 10, display: 'flex', alignItems: 'center' }}>
                  act: {networkDesc.actions}
                  <Tooltip text={TOOLTIPS.act} />
                </div>
                {isTerrainMode && (
                  <div style={{ color: 'rgba(74,222,128,0.5)', fontSize: 9, marginTop: 4 }}>
                    PD control: kp=300 kd=30 · 240Hz physics · 30Hz policy
                  </div>
                )}
              </div>
            </div>

          </div>
        </div>

        {/* Footer */}
        <footer style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          paddingTop: 12, borderTop: '1px solid var(--border)',
          fontSize: 9, color: 'rgba(255,255,255,0.15)', letterSpacing: '0.08em',
        }}>
          <span>RLFORGE · M3 Terrain + Creature Builder</span>
          <span>PD controllers · procedural terrain · share with friends</span>
        </footer>

      </div>
    </>
  )
}
