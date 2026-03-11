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

  if (envType === 'hopper') {
    // Must initialize Rapier WASM explicitly in worker context.
    // Dynamic import ensures WASM isn't loaded for CartPole sessions.
    const rapierModule = await import('@dimforge/rapier2d')
    await rapierModule.default()
    postMessage({ type: 'STATUS', msg: 'Rapier WASM ready' })

    const { RapierEnv } = await import('../env/rapierEnv.js')
    const { HOPPER } = await import('../env/characters/hopper.js')
    return new RapierEnv(HOPPER)
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
  let frameTimer = 0
  const RENDER_EVERY = 3  // send render data every N steps

  let obs = env.reset()
  postMessage({ type: 'STATUS', msg: 'Training started' })

  while (!stopped && updateCount < maxUpdates) {
    while (paused && !stopped) await new Promise(r => setTimeout(r, 100))
    if (stopped) break

    // For multi-action envs (hopper), use actMulti; for scalar (cartpole), use act
    let actionForEnv, storedAction, value, logProb

    if (envType === 'hopper') {
      const result = agent.actMulti(obs)
      actionForEnv = result.actions    // Float32Array
      storedAction = Array.from(result.actions)
      value = result.value
      logProb = result.logProb
    } else {
      const result = agent.act(obs)
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
    frameTimer++

    // Render frame
    if (frameTimer >= RENDER_EVERY) {
      frameTimer = 0
      if (envType === 'hopper') {
        postMessage({
          type: 'RENDER_SNAPSHOT',
          snapshot: env.getRenderSnapshot(),
          episodeReward,
          episodeSteps,
        })
      } else {
        postMessage({
          type: 'RENDER_FRAME',
          state: env.state,
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

    const action = envType === 'hopper'
      ? agent.actDeterministicMulti(obs)
      : agent.actDeterministic(obs)

    const { obs: nextObs, reward, done } = env.step(action)
    episodeReward += reward
    episodeSteps++

    if (envType === 'hopper') {
      postMessage({
        type: 'RENDER_SNAPSHOT',
        snapshot: env.getRenderSnapshot(),
        episodeReward,
        episodeSteps,
      })
    } else {
      postMessage({
        type: 'RENDER_FRAME',
        state: env.state,
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
