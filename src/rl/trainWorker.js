/**
 * Training Worker (v2)
 *
 * Supports two environment backends:
 *   'cartpole' → CartPoleEnv (pure JS, no WASM)
 *   'hopper'   → RapierEnv (Rapier2D WASM, initialized here in the worker)
 *
 * Rapier WASM is initialized once at worker startup when env type is 'hopper'.
 * All physics runs entirely in this worker — the main thread only receives
 * lightweight render snapshots and metric updates.
 *
 * Message protocol:
 *   IN  { type: 'START', config: { envType, ppoConfig, maxUpdates } }
 *   IN  { type: 'PLAYBACK', config: { envType, ppoConfig, weights } }
 *   IN  { type: 'PAUSE' | 'RESUME' | 'STOP' | 'EXPORT' }
 *   OUT { type: 'RENDER_SNAPSHOT', snapshot, episodeReward, episodeSteps }
 *   OUT { type: 'RENDER_FRAME', state, episodeReward, episodeSteps }
 *   OUT { type: 'METRICS', data }
 *   OUT { type: 'EPISODE', data }
 *   OUT { type: 'STATUS', msg }
 *   OUT { type: 'EXPORT_URL', url }
 */

import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgpu'
import { CartPoleEnv } from '../env/cartpole.js'
import { VecEnv } from '../env/vecEnv.js'
import { PPOAgent } from './ppo.js'

let paused = false
let stopped = false
let agent = null
let env = null

/** Minimum storage buffers per shader stage that tfjs-backend-webgpu needs. */
const WEBGPU_MIN_STORAGE_BUFFERS = 16

/**
 * Probe the GPU adapter to check whether WebGPU is viable.
 * Returns true only if the device exposes enough storage buffers.
 */
/** Probe the GPU adapter; returns { viable, deviceName } */
async function probeWebGPU() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return { viable: false, deviceName: '' }
  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return { viable: false, deviceName: '' }
    const maxBuffers = adapter.limits.maxStorageBuffersPerShaderStage
    const viable = maxBuffers >= WEBGPU_MIN_STORAGE_BUFFERS
    let deviceName = ''
    try {
      const info = await adapter.requestAdapterInfo()
      deviceName = info.device || info.description || ''
    } catch { /* adapter info not available */ }
    return { viable, deviceName }
  } catch {
    return { viable: false, deviceName: '' }
  }
}

async function initTFBackend() {
  const gpu = await probeWebGPU()
  const backends = gpu.viable ? ['webgpu', 'webgl', 'cpu'] : ['webgl', 'cpu']
  for (const b of backends) {
    try {
      await tf.setBackend(b)
      await tf.ready()
      return { backend: b, deviceName: gpu.deviceName }
    } catch { continue }
  }
  return { backend: 'cpu', deviceName: '' }
}

/** Character def loaders for Rapier-based environments */
const RAPIER_ENVS = {
  hopper: () => import('../env/characters/hopper.js').then(m => m.HOPPER),
  walker2d: () => import('../env/characters/walker2d.js').then(m => m.WALKER2D),
  acrobot: () => import('../env/characters/acrobot.js').then(m => m.ACROBOT),
}

async function initEnv(envType) {
  if (envType === 'cartpole') {
    return new CartPoleEnv()
  }

  if (RAPIER_ENVS[envType]) {
    await import('@dimforge/rapier2d')
    postMessage({ type: 'STATUS', msg: 'Rapier WASM ready' })

    const { RapierEnv } = await import('../env/rapierEnv.js')
    const charDef = await RAPIER_ENVS[envType]()
    return new RapierEnv(charDef)
  }

  throw new Error(`Unknown env: ${envType}`)
}

/**
 * Create a VecEnv with N copies of a Rapier environment.
 * WASM must already be initialized before calling this.
 */
async function initVecEnv(envType, numEnvs) {
  const { RapierEnv } = await import('../env/rapierEnv.js')
  const charDef = await RAPIER_ENVS[envType]()
  return new VecEnv(() => new RapierEnv(charDef), numEnvs)
}

/** Default number of parallel envs for vectorized training */
const DEFAULT_NUM_ENVS = 8

async function runTraining(config) {
  const { envType = 'cartpole', ppoConfig = {}, maxUpdates = 3000, numEnvs = DEFAULT_NUM_ENVS } = config

  const { backend: tfBackend, deviceName } = await initTFBackend()
  postMessage({ type: 'BACKEND', backend: tfBackend, deviceName })

  const useVecEnv = RAPIER_ENVS[envType] != null

  if (useVecEnv) {
    // Initialize Rapier WASM once, then create vectorized envs
    await import('@dimforge/rapier2d')
    postMessage({ type: 'STATUS', msg: 'Rapier WASM ready' })
    env = await initVecEnv(envType, numEnvs)
    postMessage({ type: 'STATUS', msg: `${envType} ready · ${numEnvs} envs` })
  } else {
    env = await initEnv(envType)
    postMessage({ type: 'STATUS', msg: `${envType} ready` })
  }

  const obsSize = useVecEnv ? env.observationSize : env.observationSize
  const actSize = useVecEnv ? env.actionSize : env.actionSize
  agent = new PPOAgent(obsSize, actSize, ppoConfig)

  if (useVecEnv) {
    await runVectorizedLoop(envType, maxUpdates, numEnvs)
  } else {
    await runSingleEnvLoop(envType, maxUpdates)
  }

  env.dispose?.()
  postMessage({ type: 'STATUS', msg: stopped ? 'Stopped' : 'Training complete' })
}

/**
 * Vectorized training loop for Rapier-based envs.
 * N envs step in lockstep, one batched GPU forward pass per step.
 */
async function runVectorizedLoop(envType, maxUpdates, numEnvs) {
  let totalSteps = 0
  let totalEpisodes = 0
  let updateCount = 0
  let episodeRewards = []
  let lastRenderTime = 0
  const RENDER_INTERVAL_MS = 16

  // Per-env episode tracking
  const epRewards = new Float32Array(numEnvs)
  const epSteps = new Uint32Array(numEnvs)

  let obsArray = env.resetAll()  // [N][obsSize]
  postMessage({ type: 'STATUS', msg: 'Training started' })

  // Timing accumulators
  let t_inference = 0
  let t_physics = 0
  let t_update = 0
  let t_rolloutStart = performance.now()

  while (!stopped && updateCount < maxUpdates) {
    while (paused && !stopped) await new Promise(r => setTimeout(r, 100))
    if (stopped) break

    // One batched GPU forward pass for all N observations
    const t0 = performance.now()
    const { actions, values, logProbs } = await agent.actBatch(obsArray)
    const t1 = performance.now()
    t_inference += t1 - t0

    // Step all N envs (sequential physics, but amortized GPU cost)
    const { obs: nextObsArray, rewards, dones } = env.stepAll(actions)
    const t2 = performance.now()
    t_physics += t2 - t1

    // Store N transitions at once
    agent.storeTransitions(obsArray, actions, rewards, dones, values, logProbs)
    totalSteps += numEnvs

    // Per-env episode tracking
    for (let i = 0; i < numEnvs; i++) {
      epRewards[i] += rewards[i]
      epSteps[i]++
      if (dones[i]) {
        episodeRewards.push(epRewards[i])
        postMessage({ type: 'EPISODE', data: { episode: totalEpisodes, reward: epRewards[i], steps: epSteps[i], totalSteps } })
        epRewards[i] = 0
        epSteps[i] = 0
        totalEpisodes++
      }
    }

    // Render from env[0] at ~60fps
    const now = performance.now()
    if (now - lastRenderTime >= RENDER_INTERVAL_MS) {
      lastRenderTime = now
      postMessage({
        type: 'RENDER_SNAPSHOT',
        snapshot: env.getRenderSnapshot(),
        episodeReward: epRewards[0],
        episodeSteps: epSteps[0],
      })
    }

    obsArray = nextObsArray

    if (agent.bufferFull) {
      // For GAE bootstrap, use env[0]'s current obs
      const tUpdateStart = performance.now()
      const metrics = await agent.update(obsArray[0])
      const tUpdateEnd = performance.now()
      t_update = tUpdateEnd - tUpdateStart
      updateCount++

      const t_rolloutTotal = tUpdateStart - t_rolloutStart
      const totalMs = tUpdateEnd - t_rolloutStart
      const stepsInBuffer = agent.cfg.stepsPerUpdate
      const stepsPerSec = Math.round(stepsInBuffer / (totalMs / 1000))
      const timing = {
        rolloutMs: Math.round(t_rolloutTotal),
        inferenceMs: Math.round(t_inference),
        physicsMs: Math.round(t_physics),
        updateMs: Math.round(t_update),
        totalMs: Math.round(totalMs),
        stepsPerSec,
        numEnvs,
      }
      console.log(`[perf] update ${updateCount} (${numEnvs} envs): rollout ${timing.rolloutMs}ms (inference ${timing.inferenceMs}ms + physics ${timing.physicsMs}ms) | PPO update ${timing.updateMs}ms | ${stepsPerSec} steps/s`)

      t_inference = 0
      t_physics = 0
      t_rolloutStart = performance.now()

      const recent = episodeRewards.slice(-20)
      const meanReward = recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : 0

      postMessage({
        type: 'METRICS',
        data: { update: updateCount, totalSteps, totalEpisodes, meanReward20: meanReward, ...metrics, timing, tensorMemory: tf.memory().numTensors }
      })

      await new Promise(r => setTimeout(r, 0))
    }
  }
}

/**
 * Single-env training loop (CartPole and other non-vectorized envs).
 * Preserved from the original implementation.
 */
async function runSingleEnvLoop(envType, maxUpdates) {
  let totalSteps = 0
  let totalEpisodes = 0
  let updateCount = 0
  let episodeReward = 0
  let episodeSteps = 0
  let episodeRewards = []
  let lastRenderTime = 0
  const RENDER_INTERVAL_MS = 16

  let obs = env.reset()
  postMessage({ type: 'STATUS', msg: 'Training started' })

  let t_inference = 0
  let t_physics = 0
  let t_update = 0
  let t_rolloutStart = performance.now()

  while (!stopped && updateCount < maxUpdates) {
    while (paused && !stopped) await new Promise(r => setTimeout(r, 100))
    if (stopped) break

    const isMultiAction = env.actionSize > 1
    let actionForEnv, storedAction, value, logProb

    const t0 = performance.now()
    if (isMultiAction) {
      const result = await agent.actMulti(obs)
      actionForEnv = result.actions
      storedAction = Array.from(result.actions)
      value = result.value
      logProb = result.logProb
    } else {
      const result = await agent.act(obs)
      actionForEnv = result.action
      storedAction = result.action
      value = result.value
      logProb = result.logProb
    }
    const t1 = performance.now()
    t_inference += t1 - t0

    const { obs: nextObs, reward, done } = env.step(actionForEnv)
    const t2 = performance.now()
    t_physics += t2 - t1

    agent.storeTransition(obs, storedAction, reward, done, value, logProb)
    episodeReward += reward
    episodeSteps++
    totalSteps++

    const now = performance.now()
    if (now - lastRenderTime >= RENDER_INTERVAL_MS) {
      lastRenderTime = now
      if (envType === 'cartpole') {
        postMessage({
          type: 'RENDER_FRAME',
          state: env.state,
          episodeReward,
          episodeSteps,
        })
      } else {
        postMessage({
          type: 'RENDER_SNAPSHOT',
          snapshot: env.getRenderSnapshot(),
          episodeReward,
          episodeSteps,
        })
      }
    }

    obs = nextObs

    if (done) {
      episodeRewards.push(episodeReward)
      postMessage({ type: 'EPISODE', data: { episode: totalEpisodes, reward: episodeReward, steps: episodeSteps, totalSteps } })
      episodeReward = 0
      episodeSteps = 0
      totalEpisodes++
      obs = env.reset()
    }

    if (agent.bufferFull) {
      const tUpdateStart = performance.now()
      const metrics = await agent.update(obs)
      const tUpdateEnd = performance.now()
      t_update = tUpdateEnd - tUpdateStart
      updateCount++

      const t_rolloutTotal = tUpdateStart - t_rolloutStart
      const totalMs = tUpdateEnd - t_rolloutStart
      const stepsPerSec = Math.round(agent.cfg.stepsPerUpdate / (totalMs / 1000))
      const timing = {
        rolloutMs: Math.round(t_rolloutTotal),
        inferenceMs: Math.round(t_inference),
        physicsMs: Math.round(t_physics),
        updateMs: Math.round(t_update),
        totalMs: Math.round(totalMs),
        stepsPerSec,
      }
      console.log(`[perf] update ${updateCount}: rollout ${timing.rolloutMs}ms (inference ${timing.inferenceMs}ms + physics ${timing.physicsMs}ms + other ${Math.round(t_rolloutTotal - t_inference - t_physics)}ms) | PPO update ${timing.updateMs}ms | total ${timing.totalMs}ms`)

      t_inference = 0
      t_physics = 0
      t_rolloutStart = performance.now()

      const recent = episodeRewards.slice(-20)
      const meanReward = recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : 0

      postMessage({
        type: 'METRICS',
        data: { update: updateCount, totalSteps, totalEpisodes, meanReward20: meanReward, ...metrics, timing, tensorMemory: tf.memory().numTensors }
      })

      await new Promise(r => setTimeout(r, 0))
    }
  }
}

async function runPlayback(config) {
  const { envType = 'cartpole', ppoConfig = {}, weights } = config

  const { backend: tfBackend, deviceName } = await initTFBackend()
  postMessage({ type: 'BACKEND', backend: tfBackend, deviceName })

  env = await initEnv(envType)
  agent = new PPOAgent(env.observationSize, env.actionSize, ppoConfig)
  agent.importWeights(weights)
  postMessage({ type: 'STATUS', msg: 'Playback' })

  let obs = env.reset()
  let episodeReward = 0
  let episodeSteps = 0
  let totalEpisodes = 0

  while (!stopped) {
    while (paused && !stopped) await new Promise(r => setTimeout(r, 100))
    if (stopped) break

    const action = env.actionSize > 1
      ? agent.actDeterministicMulti(obs)
      : agent.actDeterministic(obs)

    const { obs: nextObs, reward, done } = env.step(action)
    episodeReward += reward
    episodeSteps++

    if (envType === 'cartpole') {
      postMessage({
        type: 'RENDER_FRAME',
        state: env.state,
        episodeReward,
        episodeSteps,
      })
    } else {
      postMessage({
        type: 'RENDER_SNAPSHOT',
        snapshot: env.getRenderSnapshot(),
        episodeReward,
        episodeSteps,
      })
    }

    obs = nextObs

    if (done) {
      postMessage({ type: 'EPISODE', data: { episode: totalEpisodes, reward: episodeReward, steps: episodeSteps, totalSteps: 0 } })
      episodeReward = 0
      episodeSteps = 0
      totalEpisodes++
      obs = env.reset()
    }

    // ~60fps pacing
    await new Promise(r => setTimeout(r, 16))
  }

  env.dispose?.()
  postMessage({ type: 'STATUS', msg: 'Playback stopped' })
}

self.onmessage = async ({ data: { type, config } }) => {
  switch (type) {
    case 'START':
      stopped = false; paused = false
      runTraining(config)
      break
    case 'PLAYBACK':
      stopped = false; paused = false
      runPlayback(config)
      break
    case 'PAUSE':
      paused = true; postMessage({ type: 'STATUS', msg: 'Paused' }); break
    case 'RESUME':
      paused = false; postMessage({ type: 'STATUS', msg: 'Resumed' }); break
    case 'STOP':
      stopped = true; break
    case 'EXPORT':
      if (agent) postMessage({ type: 'EXPORT_URL', url: await agent.exportModel() })
      break
  }
}
