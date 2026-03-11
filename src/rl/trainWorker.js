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
async function isWebGPUViable() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false
  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return false
    const maxBuffers = adapter.limits.maxStorageBuffersPerShaderStage
    return maxBuffers >= WEBGPU_MIN_STORAGE_BUFFERS
  } catch {
    return false
  }
}

async function initTFBackend() {
  const webgpuOk = await isWebGPUViable()
  const backends = webgpuOk ? ['webgpu', 'webgl', 'cpu'] : ['webgl', 'cpu']
  for (const b of backends) {
    try {
      await tf.setBackend(b)
      await tf.ready()
      return b
    } catch { continue }
  }
  return 'cpu'
}

async function initEnv(envType) {
  if (envType === 'cartpole') {
    return new CartPoleEnv()
  }

  // All Rapier-based environments
  const RAPIER_ENVS = {
    hopper: () => import('../env/characters/hopper.js').then(m => m.HOPPER),
    walker2d: () => import('../env/characters/walker2d.js').then(m => m.WALKER2D),
    acrobot: () => import('../env/characters/acrobot.js').then(m => m.ACROBOT),
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

async function runTraining(config) {
  const { envType = 'cartpole', ppoConfig = {}, maxUpdates = 3000 } = config

  const tfBackend = await initTFBackend()
  postMessage({ type: 'STATUS', msg: `TF: ${tfBackend}` })

  env = await initEnv(envType)
  postMessage({ type: 'STATUS', msg: `${envType} ready` })

  agent = new PPOAgent(env.observationSize, env.actionSize, ppoConfig)

  let totalSteps = 0
  let totalEpisodes = 0
  let updateCount = 0
  let episodeReward = 0
  let episodeSteps = 0
  let episodeRewards = []
  let lastRenderTime = 0
  const RENDER_INTERVAL_MS = 16  // ~60fps time-based throttle

  let obs = env.reset()
  postMessage({ type: 'STATUS', msg: 'Training started' })

  while (!stopped && updateCount < maxUpdates) {
    while (paused && !stopped) await new Promise(r => setTimeout(r, 100))
    if (stopped) break

    // Multi-action envs use actMulti; scalar (cartpole) uses act
    const isMultiAction = env.actionSize > 1
    let actionForEnv, storedAction, value, logProb

    if (isMultiAction) {
      const result = await agent.actMulti(obs)
      actionForEnv = result.actions    // Float32Array
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

    const { obs: nextObs, reward, done } = env.step(actionForEnv)

    agent.storeTransition(obs, storedAction, reward, done, value, logProb)
    episodeReward += reward
    episodeSteps++
    totalSteps++

    // Time-based render throttle (~60fps) instead of every N steps
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
      const metrics = await agent.update(obs, envType === 'hopper')
      updateCount++

      const recent = episodeRewards.slice(-20)
      const meanReward = recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : 0

      postMessage({
        type: 'METRICS',
        data: { update: updateCount, totalSteps, totalEpisodes, meanReward20: meanReward, ...metrics, tensorMemory: tf.memory().numTensors }
      })

      await new Promise(r => setTimeout(r, 0))
    }
  }

  env.dispose?.()
  postMessage({ type: 'STATUS', msg: stopped ? 'Stopped' : 'Training complete' })
}

async function runPlayback(config) {
  const { envType = 'cartpole', ppoConfig = {}, weights } = config

  const tfBackend = await initTFBackend()
  postMessage({ type: 'STATUS', msg: `TF: ${tfBackend}` })

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
