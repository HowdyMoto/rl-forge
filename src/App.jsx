import { useState, useEffect, useRef, useCallback } from 'react'
import CartPoleRenderer from './components/CartPoleRenderer.jsx'
import HopperRenderer from './components/HopperRenderer.jsx'
import Walker2DRenderer from './components/Walker2DRenderer.jsx'
import AcrobotRenderer from './components/AcrobotRenderer.jsx'
import RewardChart from './components/RewardChart.jsx'
import TrainingCharts from './components/TrainingCharts.jsx'
import MetricsPanel from './components/MetricsPanel.jsx'
import HyperParams from './components/HyperParams.jsx'
import Tooltip from './components/Tooltip.jsx'
import { DEFAULT_PPO_CONFIG } from './rl/ppo.js'

const ENV_CONFIGS = {
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
    ppoOverrides: { hiddenSizes: [128, 128], stepsPerUpdate: 1024, numEpochs: 10 },
    tooltip: 'Physics-based locomotion (Rapier2D). A single-legged robot learns to hop forward without falling. 10D observation, 2 continuous joint torques. More challenging than CartPole — contact dynamics, sparse-ish reward, harder exploration. Solved at mean reward ≥ 1500.',
  },
  walker2d: {
    label: 'Walker2D',
    desc: 'Bipedal walker · 14 obs · 4 actions',
    solvedThreshold: 1500,
    ppoOverrides: { hiddenSizes: [128, 128], stepsPerUpdate: 2048, numEpochs: 10 },
    tooltip: 'Bipedal locomotion (Rapier2D). A two-legged robot learns to walk forward without falling. 14D observation, 4 continuous joint torques (hip and knee per leg). Harder than Hopper — must coordinate both legs symmetrically. Solved at mean reward ≥ 1500.',
  },
  acrobot: {
    label: 'Acrobot',
    desc: 'Double pendulum · 10 obs · 2 actions',
    solvedThreshold: 800,
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 512, numEpochs: 10 },
    tooltip: 'Inverted double pendulum balance (Rapier2D). Two links hang from a fixed pivot — only the elbow joint is actuated. The agent must keep both links balanced above a height threshold. 10D observation, 1 effective action. Solved at mean reward ≥ 800.',
  },
}

const TOOLTIPS = {
  rewardCurve: 'Rolling 20-episode mean reward plotted over training updates. The primary visual signal for whether the agent is improving. The dashed line is the solved threshold for the current environment.',
  trainingMetrics: 'Key scalars logged after each policy update: reward trend, losses, and entropy. Useful for diagnosing training problems — e.g., flat reward with rising entropy often indicates poor exploration.',
  hyperparameters: 'The knobs that control PPO training dynamics. Locked during a run; set them before hitting Train. Defaults are reasonable starting points — most experiments only need learning rate tuning.',
  network: 'The neural network architecture for this environment. Actor and Critic are separate MLPs with the same hidden width but independent weights.',
  actor: 'The policy network π(a|s). Takes an observation vector and outputs action parameters — for continuous actions, the mean (and log-std) of a Gaussian. During training, actions are sampled from this distribution.',
  critic: 'The value network V(s). Estimates expected future return from the current state. Used only during training to compute advantage estimates A = R − V(s). Not used at inference time.',
  obs: 'Observation vector — the raw state representation fed to both the actor and critic at each timestep.',
  act: 'Action output from the actor. For CartPole: scalar horizontal force. For Hopper/Walker2D: continuous joint torques.',
  solved: 'The rolling mean reward over the last 20 episodes has exceeded the benchmark threshold. The agent has learned a reliably successful policy.',
  avgPill: 'Current rolling mean reward (last 20 episodes) vs. the solved threshold for this environment.',
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
    --text-dim: rgba(255,255,255,0.3);
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
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
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
    font-size: 10px;
    letter-spacing: 0.08em;
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
    padding: 2px 8px;
    border-radius: 20px;
    font-size: 9px;
    letter-spacing: 0.06em;
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
  const [envType, setEnvType] = useState('cartpole')
  const [trainingState, setTrainingState] = useState(TRAINING_STATES.IDLE)

  // CartPole render state
  const [cartpoleState, setCartpoleState] = useState(null)
  // Hopper render state
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

  const isRunning = trainingState === TRAINING_STATES.RUNNING

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
          break
        case 'METRICS':
          setMetrics(prev => [...prev.slice(-300), msg.data])
          break
        case 'EPISODE':
          setEpisodes(msg.data.episode + 1)
          break
        case 'STATUS':
          setStatus(msg.msg)
          if (msg.msg.startsWith('TF:')) setBackend(msg.msg.replace('TF: ', ''))
          break
        case 'EXPORT_URL':
          setExportUrl(msg.url)
          break
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
    setTrainingState(TRAINING_STATES.RUNNING)

    const envCfg = ENV_CONFIGS[envType]
    worker.postMessage({
      type: 'START',
      config: {
        envType,
        ppoConfig: { ...ppoConfig, ...envCfg.ppoOverrides },
        maxUpdates: 3000,
      }
    })
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
  }

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
      worker.postMessage({
        type: 'PLAYBACK',
        config: {
          envType,
          ppoConfig: { ...ppoConfig, ...envCfg.ppoOverrides },
          weights,
        }
      })
    }
    reader.readAsText(file)
  }

  const latestReward = metrics[metrics.length - 1]?.meanReward20 ?? 0
  const solvedThreshold = ENV_CONFIGS[envType].solvedThreshold
  const isSolved = latestReward >= solvedThreshold

  const renderSnapshotProps = { snapshot: hopperSnapshot, episodeReward, episodeSteps }
  const RENDERERS = {
    cartpole: <CartPoleRenderer state={cartpoleState} episodeReward={episodeReward} episodeSteps={episodeSteps} />,
    hopper: <HopperRenderer {...renderSnapshotProps} />,
    walker2d: <Walker2DRenderer {...renderSnapshotProps} />,
    acrobot: <AcrobotRenderer {...renderSnapshotProps} />,
  }
  const simContent = RENDERERS[envType]

  const NETWORK_DESCS = {
    cartpole: { actor: '4 → 64 → 64 → 1', obs: '[x, ẋ, θ, θ̇]', actions: 'force' },
    hopper: { actor: '10 → 128 → 128 → 2', obs: '[y, θ, vx, vy, ω, hip∠, hip·ω, knee∠, knee·ω, contact]', actions: '[τ_hip, τ_knee]' },
    walker2d: { actor: '14 → 128 → 128 → 4', obs: '[y, θ, vx, vy, ω, 4×(j∠, j·ω), contact]', actions: '[τ_lhip, τ_lknee, τ_rhip, τ_rknee]' },
    acrobot: { actor: '10 → 64 → 64 → 2', obs: '[y, θ, vx, vy, ω, shoulder∠, shoulder·ω, elbow∠, elbow·ω, contact]', actions: '[0, τ_elbow]' },
  }
  const networkDesc = NETWORK_DESCS[envType]

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
              MILESTONE 2 · PHYSICS
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
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
                    onClick={() => { setEnvType(key); setMetrics([]); setEpisodes(0) }}
                    disabled={isRunning || trainingState === TRAINING_STATES.PAUSED}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 2 }}>
                      {cfg.label}
                      <Tooltip text={cfg.tooltip} />
                    </span>
                    <span style={{ fontSize: 8, display: 'block', marginTop: 1, opacity: 0.6 }}>
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
                  ▶ Train {ENV_CONFIGS[envType].label}
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
                  onClick={() => workerRef.current?.postMessage({ type: 'EXPORT' })}
                  disabled={metrics.length === 0}
                  style={{ width: '100%', justifyContent: 'center' }}
                >
                  ↓ Export Weights
                </button>

                {exportUrl && (
                  <a href={exportUrl} download={`${envType}_policy.json`}
                    style={{
                      display: 'block', textAlign: 'center', padding: '6px',
                      background: 'rgba(74,222,128,0.08)', border: '1px solid rgba(74,222,128,0.2)',
                      borderRadius: 6, color: 'var(--green)', fontSize: 10,
                      fontFamily: '"DM Mono", monospace', letterSpacing: '0.06em', textDecoration: 'none',
                    }}
                  >
                    ↓ {envType}_policy.json ready
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
              </div>
            </div>

            {/* Hyperparameters */}
            <div className="panel" style={{ flex: 1 }}>
              <div className="panel-header">
                <span style={{ display: 'flex', alignItems: 'center' }}>
                  ◧ hyperparameters
                  <Tooltip text={TOOLTIPS.hyperparameters} />
                </span>
                <span style={{ fontSize: 8, color: 'rgba(255,255,255,0.2)' }}>
                  {isRunning ? 'LOCKED' : 'EDIT BEFORE TRAIN'}
                </span>
              </div>
              <div style={{ padding: '14px 16px', overflowY: 'auto' }}>
                <HyperParams onChange={setPpoConfig} disabled={isRunning} />
              </div>
            </div>

            {/* Network info */}
            <div className="panel">
              <div className="panel-header">
                <span style={{ display: 'flex', alignItems: 'center' }}>
                  ◫ network · {ENV_CONFIGS[envType].label}
                  <Tooltip text={TOOLTIPS.network} />
                </span>
              </div>
              <div style={{ padding: '12px 14px', fontSize: 10, color: 'var(--text-dim)', lineHeight: 1.8, fontFamily: '"DM Mono", monospace' }}>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  Actor: {networkDesc.actor}
                  <Tooltip text={TOOLTIPS.actor} />
                </div>
                <div style={{ display: 'flex', alignItems: 'center' }}>
                  Critic: same width → 1
                  <Tooltip text={TOOLTIPS.critic} />
                </div>
                <div style={{ marginTop: 6, color: 'rgba(255,255,255,0.15)', fontSize: 9, lineHeight: 1.6, display: 'flex', alignItems: 'center' }}>
                  obs: {networkDesc.obs}
                  <Tooltip text={TOOLTIPS.obs} />
                </div>
                <div style={{ color: 'rgba(255,255,255,0.15)', fontSize: 9, display: 'flex', alignItems: 'center' }}>
                  act: {networkDesc.actions}
                  <Tooltip text={TOOLTIPS.act} />
                </div>
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
          <span>RLFORGE · M2 Rapier Physics</span>
          <span>next: character authoring canvas · reward function builder</span>
        </footer>

      </div>
    </>
  )
}
