import { useState, useEffect, useRef, useCallback } from 'react'
import PhysicsRenderer from '../../components/PhysicsRenderer.jsx'
import TerrainRenderer from '../../components/TerrainRenderer.jsx'
import CreatureBuilder from '../../components/CreatureBuilder.jsx'
import SharePanel from '../../components/SharePanel.jsx'
import RewardChart from '../../components/RewardChart.jsx'
import TrainingCharts from '../../components/TrainingCharts.jsx'
import MetricsPanel from '../../components/MetricsPanel.jsx'
import HyperParams from '../../components/HyperParams.jsx'
import SceneHierarchy from '../../components/SceneHierarchy.jsx'
import Tooltip from '../../components/Tooltip.jsx'
import OverlaySidebar, { SIDEBAR_WIDTH } from '../../components/OverlaySidebar.jsx'
import FloatingControls from '../../components/FloatingControls.jsx'
import { DEFAULT_PPO_CONFIG } from '../../rl/ppo.js'
import { decodeShareString } from '../../components/SharePanel.jsx'

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
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 2048, gamma: 0.95, normalizeRewards: false },
    tooltip: 'Classic control benchmark. The agent applies horizontal force to a cart to keep an inverted pendulum upright. 4D observation (position, velocity, pole angle, angular velocity), 1 continuous action. Solved at mean reward ≥ 450.',
  },
  pendulum: {
    label: 'Pendulum',
    desc: 'Swing-up balance · 3 obs · 1 action',
    solvedThreshold: -200,
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 2048, normalizeRewards: false },
    tooltip: 'Classic Gym Pendulum-v1. A single link hangs from a fixed pivot under gravity. Apply torque to swing it up and balance inverted. 3D observation (cos θ, sin θ, angular velocity), 1 continuous action. Dense reward penalizes angle, velocity, and torque. Best possible: 0 per step.',
  },
  hopper: {
    label: 'Hopper',
    desc: 'Monopod hopper · 12 obs · 3 actions',
    solvedThreshold: 1500,
    ppoOverrides: { hiddenSizes: [64, 64], entropyCoef: 0.02, stepsPerUpdate: 2048 },
    tooltip: 'Physics-based locomotion (Rapier2D). A single-legged robot learns to hop forward without falling. 12D observation, 3 continuous joint targets (PD control). More challenging than CartPole — contact dynamics, sparse-ish reward, harder exploration. Solved at mean reward ≥ 1500.',
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
    desc: 'Double pendulum · 10 obs · 1 action',
    solvedThreshold: 500,
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 2048, numEpochs: 10 },
    tooltip: 'Swing-up double pendulum (Rapier2D). Two links hang from a fixed pivot — only the elbow joint is actuated. The agent must pump energy to swing the tip above the pivot. 10D observation, 1 action. Solved at mean reward ≥ 500.',
  },
  'acrobot-damped': {
    label: 'Acrobot (damped)',
    desc: 'Double pendulum with friction · 10 obs · 1 action',
    solvedThreshold: 500,
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 2048, numEpochs: 10 },
    tooltip: 'Same as Acrobot but with joint damping (friction=2.0). The pendulum loses energy and settles, making the swing-up task easier to learn. Compare training curves with the frictionless version.',
  },
  'spinner-constant': {
    label: 'Spinner (Constant)',
    desc: 'Match target speed · 9 obs · 1 action',
    solvedThreshold: 0.9,
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 2048 },
    tooltip: 'Rotate a body at exactly 1 revolution per 3 seconds (≈2.09 rad/s). Rewarded for matching the target angular velocity. Tests precision control — the agent must find and maintain a specific speed.',
  },
  'spinner-max': {
    label: 'Spinner (Max)',
    desc: 'Spin as fast as possible · 9 obs · 1 action',
    solvedThreshold: 2.0,
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 2048 },
    tooltip: 'Spin a body as fast as possible against damping. Reward scales linearly with angular velocity. The simplest possible RL task — should converge quickly to max-torque policy.',
  },
  'red-light-green-light': {
    label: 'Red Light Green Light',
    desc: 'Stop & go game · 10 obs · 1 action',
    solvedThreshold: 1.0,
    ppoOverrides: { hiddenSizes: [64, 64], stepsPerUpdate: 4096, numEpochs: 10 },
    tooltip: 'A signal alternates between green (spin!) and red (stop!). The agent must spin during green phases and brake quickly when the light turns red. Tests reactive control and temporal reasoning.',
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

const TRAINING_STATES = { IDLE: 'idle', RUNNING: 'running', PAUSED: 'paused' }

export default function TrainMode({ envType, setEnvType, onTrainingStateChange, onRegisterBuildNew }) {
  const workerRef = useRef(null)
  const [trainingState, setTrainingState] = useState(TRAINING_STATES.IDLE)

  // Render state (unified: all envs use snapshots now)
  const [hopperSnapshot, setHopperSnapshot] = useState(null)

  const [episodeReward, setEpisodeReward] = useState(0)
  const [episodeSteps, setEpisodeSteps] = useState(0)
  const [metrics, setMetrics] = useState([])
  const [episodes, setEpisodes] = useState(0)
  const [status, setStatus] = useState('idle')
  const [backend, setBackend] = useState('')
  const [ppoConfig, setPpoConfig] = useState(DEFAULT_PPO_CONFIG)
  const [numEnvs, setNumEnvs] = useState(8)
  const [exportUrl, setExportUrl] = useState(null)
  const [playbackDiag, setPlaybackDiag] = useState(null)
  const [resetEvent, setResetEvent] = useState(null)
  const fileInputRef = useRef(null)

  // Sidebar overlay state
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarFullscreen, setSidebarFullscreen] = useState(false)
  const [activeTab, setActiveTab] = useState('charts')

  // Creature builder state
  const [customCharDef, setCustomCharDef] = useState(null)
  const [bestDistance, setBestDistance] = useState(0)

  // Physics debug mode
  const [debugMode, setDebugMode] = useState(false)
  const [debugJoint, setDebugJoint] = useState(null)
  const [debugDirection, setDebugDirection] = useState(0)

  // Scene hierarchy selection
  const [sceneSelectedId, setSceneSelectedId] = useState(null)
  const [sceneSelectedType, setSceneSelectedType] = useState(null)

  const isRunning = trainingState === TRAINING_STATES.RUNNING
  const isTerrainMode = envType === 'terrain'

  // Notify App about training state for disabling env selector
  useEffect(() => {
    onTrainingStateChange?.(trainingState !== TRAINING_STATES.IDLE)
  }, [trainingState, onTrainingStateChange])

  // Register buildNew callback so header EnvSelector can trigger it
  useEffect(() => {
    onRegisterBuildNew?.(() => {
      setEnvType('terrain')
      setMetrics([])
      setEpisodes(0)
      setBestDistance(0)
      setActiveTab('build')
      setSidebarOpen(true)
    })
  }, [onRegisterBuildNew, setEnvType])

  // Reset metrics when env changes
  const prevEnvRef = useRef(envType)
  useEffect(() => {
    if (prevEnvRef.current !== envType) {
      setMetrics([])
      setEpisodes(0)
      setBestDistance(0)
      prevEnvRef.current = envType
    }
  }, [envType])

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
          window.history.replaceState({}, '', window.location.pathname)
        }
      }
    } catch { /* ignore invalid share links */ }
  }, [])

  const initWorker = useCallback(() => {
    if (workerRef.current) workerRef.current.terminate()
    const worker = new Worker(
      new URL('../../rl/trainWorker.js', import.meta.url),
      { type: 'module' }
    )
    worker.onmessage = (e) => {
      const msg = e.data
      switch (msg.type) {
        case 'RENDER_SNAPSHOT':
          setHopperSnapshot(msg.snapshot)
          setEpisodeReward(msg.episodeReward)
          setEpisodeSteps(msg.episodeSteps)
          if (msg.snapshot?._maxDistance) {
            setBestDistance(prev => Math.max(prev, msg.snapshot._maxDistance))
          }
          if (msg.resetReason) {
            setResetEvent({ reason: msg.resetReason, reward: msg.episodeReward, steps: msg.episodeSteps, time: performance.now() })
          }
          break
        case 'METRICS':
          setMetrics(prev => [...prev, msg.data])
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
          const a = document.createElement('a')
          a.href = msg.url
          a.download = 'policy_weights.json'
          a.click()
          break
        }
        case 'PLAYBACK_DIAGNOSTIC':
          setPlaybackDiag(msg.data)
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
    setPlaybackDiag(null)
    setHopperSnapshot(null)
    setExportUrl(null)
    setBestDistance(0)
    setTrainingState(TRAINING_STATES.RUNNING)

    const envCfg = ENV_CONFIGS[envType]
    const config = {
      envType,
      ppoConfig: { ...ppoConfig, ...envCfg.ppoOverrides },
      maxUpdates: 10000,
      numEnvs,
    }

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

    setHopperSnapshot(null)
    setExportUrl(null)
    setDebugMode(true)
    setDebugJoint(null)
    setDebugDirection(0)
    setTrainingState(TRAINING_STATES.RUNNING)


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
  
      setHopperSnapshot(null)
      setExportUrl(null)
      setTrainingState(TRAINING_STATES.RUNNING)
  

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

  // Renderer: universal PhysicsRenderer for all environments
  // TerrainRenderer kept for terrain mode (has terrain-specific drawing)
  const renderSnapshotProps = { snapshot: hopperSnapshot, episodeReward, episodeSteps, resetEvent }
  const sceneHighlightId = sceneSelectedType === 'body' ? sceneSelectedId : null
  const canEditScene = trainingState !== TRAINING_STATES.RUNNING
  let simContent
  if (envType === 'terrain') {
    simContent = <TerrainRenderer {...renderSnapshotProps} charDef={customCharDef} onDebugMouse={debugMode ? handleDebugMouse : undefined} />
  } else {
    simContent = <PhysicsRenderer {...renderSnapshotProps} onDebugMouse={debugMode ? handleDebugMouse : undefined} highlightBodyId={sceneHighlightId} onBodyClick={handleBodyClick} />
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
      cartpole: { actor: '9 → 64 → 64 → 1', obs: '[cart.y, cart.θ, cart.vx, cart.vy, cart.ω, rail_pos, rail_vel, pole.θ, pole.ω]', actions: '[rail_force]' },
      hopper: { actor: '10 → 128 → 128 → 2', obs: '[y, θ, vx, vy, ω, hip∠, hip·ω, knee∠, knee·ω, contact]', actions: '[τ_hip, τ_knee]' },
      walker2d: { actor: '15 → 128 → 128 → 4', obs: '[y, θ, vx, vy, ω, 4×(j∠, j·ω), Lcontact, Rcontact]', actions: '[τ_lhip, τ_lknee, τ_rhip, τ_rknee]' },
      acrobot: { actor: '10 → 64 → 64 → 1', obs: '[y, θ, vx, vy, ω, shoulder∠, shoulder·ω, elbow∠, elbow·ω, contact]', actions: '[τ_elbow]' },
      'acrobot-damped': { actor: '10 → 64 → 64 → 1', obs: '[y, θ, vx, vy, ω, shoulder∠, shoulder·ω, elbow∠, elbow·ω, contact]', actions: '[τ_elbow]' },
      'spinner-constant': { actor: '9 → 64 → 64 → 1', obs: '[y, sinθ, cosθ, vx, vy, ω, sinJ, cosJ, j·ω]', actions: '[τ_spin]' },
      'spinner-max': { actor: '9 → 64 → 64 → 1', obs: '[y, sinθ, cosθ, vx, vy, ω, sinJ, cosJ, j·ω]', actions: '[τ_spin]' },
      'red-light-green-light': { actor: '10 → 64 → 64 → 1', obs: '[y, sinθ, cosθ, vx, vy, ω, sinJ, cosJ, j·ω, light]', actions: '[τ_spin]' },
    }
    return descs[envType] || descs.cartpole
  }
  const networkDesc = getNetworkDesc()

  const trainLabel = isTerrainMode ? (customCharDef?.name || 'Creature') : ENV_CONFIGS[envType].label

  const handleToggleSidebar = useCallback(() => {
    setSidebarOpen(prev => !prev)
  }, [])

  const handleToggleFullscreen = useCallback(() => {
    setSidebarFullscreen(prev => !prev)
  }, [])

  // Build sidebar tabs based on current env mode
  const sidebarTabs = []

  sidebarTabs.push({
    id: 'charts',
    icon: '◈',
    label: 'Charts',
    content: (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        <div style={{ height: 400 }}>
          <TrainingCharts metrics={metrics} solvedThreshold={solvedThreshold} />
        </div>
        <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', padding: '8px 4px' }}>
          <div style={{ padding: '4px 10px', fontSize: 10, color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Reward Curve · target: {solvedThreshold}
          </div>
          <div style={{ height: 150 }}>
            <RewardChart metrics={metrics} solvedThreshold={solvedThreshold} />
          </div>
        </div>
      </div>
    ),
    fullscreenContent: (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 24, height: '100%' }}>
        <div style={{ flex: 2, minHeight: 0 }}>
          <TrainingCharts metrics={metrics} solvedThreshold={solvedThreshold} />
        </div>
        <div style={{ flex: 1, minHeight: 0, borderTop: '1px solid rgba(255,255,255,0.07)', paddingTop: 16 }}>
          <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>
            Reward Curve · target: {solvedThreshold}
          </div>
          <div style={{ height: 'calc(100% - 28px)' }}>
            <RewardChart metrics={metrics} solvedThreshold={solvedThreshold} />
          </div>
        </div>
      </div>
    ),
  })

  sidebarTabs.push({
    id: 'metrics',
    icon: '◉',
    label: 'Metrics',
    content: (
      <div style={{ padding: '14px 16px' }}>
        <MetricsPanel metrics={metrics} episodes={episodes} status={status} backend={backend} />
      </div>
    ),
  })

  if (!isTerrainMode) {
    sidebarTabs.push({
      id: 'params',
      icon: '⚙',
      label: 'Params',
      content: (
        <div style={{ padding: '14px 16px', overflowY: 'auto' }}>
          <HyperParams onChange={setPpoConfig} disabled={isRunning} overrides={ENV_CONFIGS[envType]?.ppoOverrides} />
          <div style={{ marginTop: 14, borderTop: '1px solid rgba(255,255,255,0.08)', paddingTop: 12, fontFamily: 'Inter, sans-serif' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.6)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center' }}>
                Parallel Envs
                <Tooltip text={'Number of independent environment copies stepped each frame. More envs = more experience per policy update, better gradient estimates, and amortized GPU overhead.\n\nHigher values use more CPU (physics) and memory (one Rapier world per env). Watch the physics timing in the metrics bar — if it dominates, you\'ve hit the single-thread ceiling.'} />
              </span>
              <span style={{ fontSize: 13, color: '#e2b96f' }}>{numEnvs}</span>
            </div>
            <input
              type="range"
              min={1} max={64} step={1}
              value={numEnvs}
              disabled={isRunning}
              onChange={e => setNumEnvs(parseInt(e.target.value))}
              style={{
                width: '100%',
                accentColor: '#e2b96f',
                cursor: isRunning ? 'not-allowed' : 'pointer',
                opacity: isRunning ? 0.4 : 1,
                marginTop: 4,
              }}
            />
          </div>
        </div>
      ),
    })
  }

  if (isTerrainMode) {
    sidebarTabs.push({
      id: 'build',
      icon: '◧',
      label: 'Build',
      content: (
        <div style={{ padding: '10px 12px' }}>
          <CreatureBuilder
            onCreatureChange={setCustomCharDef}
            disabled={isRunning || trainingState === TRAINING_STATES.PAUSED}
          />
        </div>
      ),
    })

    sidebarTabs.push({
      id: 'share',
      icon: '⎘',
      label: 'Share',
      content: (
        <div style={{ padding: '10px 12px' }}>
          <SharePanel
            charDef={customCharDef}
            exportUrl={exportUrl}
            bestDistance={bestDistance}
            bestReward={latestReward}
            onImportCreature={handleImportCreature}
          />
        </div>
      ),
    })
  }

  sidebarTabs.push({
    id: 'network',
    icon: '◫',
    label: 'Net',
    content: (
      <div style={{ padding: '12px 14px', fontSize: 11, color: 'rgba(255,255,255,0.7)', lineHeight: 1.8, fontFamily: 'Inter, sans-serif' }}>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          Actor: {networkDesc.actor}
          <Tooltip text={TOOLTIPS.actor} />
        </div>
        <div style={{ display: 'flex', alignItems: 'center' }}>
          Critic: same width → 1
          <Tooltip text={TOOLTIPS.critic} />
        </div>
        <div style={{ marginTop: 6, color: 'rgba(255,255,255,0.7)', fontSize: 10, lineHeight: 1.6, display: 'flex', alignItems: 'center' }}>
          obs: {networkDesc.obs}
          <Tooltip text={TOOLTIPS.obs} />
        </div>
        <div style={{ color: 'rgba(255,255,255,0.7)', fontSize: 10, display: 'flex', alignItems: 'center' }}>
          act: {networkDesc.actions}
          <Tooltip text={TOOLTIPS.act} />
        </div>
        {isTerrainMode && (
          <div style={{ color: 'rgba(74,222,128,0.5)', fontSize: 9, marginTop: 4 }}>
            PD control: kp=300 kd=30 · 240Hz physics · 30Hz policy
          </div>
        )}
      </div>
    ),
  })

  sidebarTabs.push({
    id: 'export',
    icon: '↕',
    label: 'Export',
    content: (
      <div style={{ padding: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
        <button
          className="btn btn-ghost"
          onClick={() => {
            setExportUrl(null)
            workerRef.current?.postMessage({ type: 'EXPORT' })
          }}
          disabled={metrics.length === 0}
          style={{ width: '100%', justifyContent: 'center' }}
        >
          {exportUrl ? '✓ Export Ready — Click Again' : '↓ Export Weights'}
        </button>

        {exportUrl && (
          <a href={exportUrl} download={`${customCharDef?.name || envType}_policy.json`}
            style={{
              display: 'block', textAlign: 'center', padding: '8px',
              background: 'rgba(74,222,128,0.12)', border: '1px solid rgba(74,222,128,0.25)',
              borderRadius: 6, color: 'var(--green)', fontSize: 11,
              fontFamily: 'Inter, sans-serif', letterSpacing: '0.06em', textDecoration: 'none',
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

        {/* Playback diagnostic */}
        {playbackDiag && (
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', marginTop: 4, paddingTop: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Playback Diagnostic</span>
              <button
                className="btn btn-ghost"
                onClick={() => {
                  navigator.clipboard.writeText(JSON.stringify(playbackDiag, null, 2))
                  alert('Diagnostic copied to clipboard — paste it to Claude for analysis')
                }}
                style={{ padding: '3px 8px', fontSize: 9 }}
              >
                copy
              </button>
            </div>
            <div style={{ fontSize: 10, fontFamily: 'Inter, sans-serif', color: 'rgba(255,255,255,0.75)' }}>
              <div>{playbackDiag.envType} · {playbackDiag.obsSize}D obs · {playbackDiag.actionSize}D act · {playbackDiag.steps.length} steps</div>
              <div style={{ marginTop: 4 }}>
                ep1 reward: {playbackDiag.steps.reduce((s, d) => s + d.reward, 0).toFixed(1)}
                {' · '}
                {playbackDiag.steps[playbackDiag.steps.length - 1]?.done ? 'terminated' : 'ongoing'}
              </div>
            </div>
          </div>
        )}

        {/* Debug joint tester */}
        {debugMode && isTerrainMode && customCharDef && (
          <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', marginTop: 4, paddingTop: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.7)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Joint Tester</span>
              <button className="btn btn-ghost" onClick={handleDebugReset} style={{ padding: '3px 8px', fontSize: 9 }}>reset pose</button>
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
              {(customCharDef.joints || []).map(j => (
                <button
                  key={j.id}
                  onClick={() => handleDebugJointChange(debugJoint === j.id ? null : j.id)}
                  style={{
                    flex: '1 1 auto', padding: '5px 8px',
                    background: debugJoint === j.id ? 'var(--gold-dim)' : 'var(--surface)',
                    border: `1px solid ${debugJoint === j.id ? 'var(--gold-border)' : 'var(--border)'}`,
                    borderRadius: 5, color: debugJoint === j.id ? 'var(--gold)' : 'var(--text-dim)',
                    fontFamily: 'Inter, sans-serif', fontSize: 10, cursor: 'pointer', letterSpacing: '0.03em',
                  }}
                >
                  {j.id}
                </button>
              ))}
            </div>
            {debugJoint && (
              <div style={{ display: 'flex', gap: 6, marginTop: 6 }}>
                <button className="btn"
                  onMouseDown={() => handleDebugDirection(-1)} onMouseUp={() => handleDebugDirection(0)} onMouseLeave={() => handleDebugDirection(0)}
                  style={{
                    flex: 1, justifyContent: 'center', padding: '8px',
                    background: debugDirection === -1 ? 'rgba(224,90,90,0.15)' : 'var(--surface)',
                    color: debugDirection === -1 ? 'var(--red)' : 'var(--text-dim)',
                    border: `1px solid ${debugDirection === -1 ? 'rgba(224,90,90,0.3)' : 'var(--border)'}`,
                  }}
                >← torque −</button>
                <button className="btn"
                  onMouseDown={() => handleDebugDirection(1)} onMouseUp={() => handleDebugDirection(0)} onMouseLeave={() => handleDebugDirection(0)}
                  style={{
                    flex: 1, justifyContent: 'center', padding: '8px',
                    background: debugDirection === 1 ? 'rgba(74,222,128,0.12)' : 'var(--surface)',
                    color: debugDirection === 1 ? 'var(--green)' : 'var(--text-dim)',
                    border: `1px solid ${debugDirection === 1 ? 'rgba(74,222,128,0.2)' : 'var(--border)'}`,
                  }}
                >torque + →</button>
              </div>
            )}
          </div>
        )}
      </div>
    ),
  })

  // If active tab no longer exists (e.g., switched from terrain to non-terrain), fall back
  if (!sidebarTabs.find(t => t.id === activeTab)) {
    // Don't call setActiveTab in render — use effect or just override
  }
  const effectiveTab = sidebarTabs.find(t => t.id === activeTab) ? activeTab : sidebarTabs[0]?.id

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, position: 'relative', minHeight: 0 }}>

      {/* Status pills — floating top-right of canvas */}
      <div style={{
        position: 'absolute', top: 8, right: sidebarOpen ? SIDEBAR_WIDTH + 56 : 56,
        zIndex: 15, display: 'flex', alignItems: 'center', gap: 8,
        transition: 'right 0.2s ease', pointerEvents: 'none',
      }}>
        {isTerrainMode && bestDistance > 0 && (
          <span className="pill" style={{ background: 'var(--gold-dim)', color: 'var(--gold)', border: '1px solid var(--gold-border)', pointerEvents: 'auto' }}>
            best: {bestDistance.toFixed(1)}m
          </span>
        )}
        {isSolved && (
          <span className="pill" style={{ background: 'rgba(74,222,128,0.12)', color: 'var(--green)', border: '1px solid rgba(74,222,128,0.2)', display: 'inline-flex', alignItems: 'center', pointerEvents: 'auto' }}>
            ✓ solved
            <Tooltip text={TOOLTIPS.solved} />
          </span>
        )}
        {metrics.length > 0 && (
          <span className="pill" style={{ background: 'var(--gold-dim)', color: 'var(--gold)', border: '1px solid var(--gold-border)', display: 'inline-flex', alignItems: 'center', pointerEvents: 'auto' }}>
            {latestReward.toFixed(0)} / {solvedThreshold} avg
            <Tooltip text={TOOLTIPS.avgPill} />
          </span>
        )}
      </div>

      {/* Canvas — fills all available space */}
      <div style={{ flex: 1, position: 'relative', minHeight: 0, borderRadius: 10, overflow: 'hidden' }}>
        {simContent}

        {/* Scene hierarchy overlay (left side) */}
        {hopperSnapshot && (
          <SceneHierarchy
            snapshot={hopperSnapshot}
            selectedId={sceneSelectedId}
            selectedType={sceneSelectedType}
            onSelect={handleSceneSelect}
            onPropertyChange={handleScenePropertyChange}
            editable={canEditScene}
          />
        )}

        {/* Floating train/pause/stop controls */}
        <FloatingControls
          trainingState={trainingState}
          trainLabel={trainLabel}
          onTrain={handleTrain}
          onPause={handlePause}
          onStop={handleStop}
          sidebarOpen={sidebarOpen}
          sidebarWidth={SIDEBAR_WIDTH}
        />

        {/* Right sidebar overlay */}
        <OverlaySidebar
          tabs={sidebarTabs}
          open={sidebarOpen}
          onToggle={handleToggleSidebar}
          activeTab={effectiveTab}
          onTabChange={setActiveTab}
          fullscreen={sidebarFullscreen}
          onFullscreenToggle={handleToggleFullscreen}
        />
      </div>
    </div>
  )
}

export { ENV_CONFIGS }
