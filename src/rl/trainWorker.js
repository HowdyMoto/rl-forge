/**
 * Training Worker (v3 — Unified Rapier)
 *
 * All environments now use UnifiedRapierEnv backed by Rapier2D WASM.
 * No more separate CartPole JS physics or split code paths.
 *
 * Message protocol:
 *   IN  { type: 'START', config }
 *   IN  { type: 'PLAYBACK', config }
 *   IN  { type: 'PHYSICS_DEBUG', config }
 *   IN  { type: 'PAUSE' | 'RESUME' | 'STOP' | 'EXPORT' }
 *   IN  { type: 'DEBUG_CONTROL' | 'DEBUG_GRAB' | 'DEBUG_DRAG' | 'DEBUG_RELEASE' | 'DEBUG_RESET' }
 *   OUT { type: 'RENDER_SNAPSHOT' | 'METRICS' | 'EPISODE' | 'STATUS' | 'BACKEND' | 'EXPORT_URL' }
 */

import * as tf from '@tensorflow/tfjs'
// This import auto-registers a 'webgpu' backend, but its factory omits
// maxStorageBuffersPerShaderStage from requiredLimits. We re-register
// in registerWebGPUBackend() with the missing limit included.
import { WebGPUBackend } from '@tensorflow/tfjs-backend-webgpu'
import { VecEnv } from '../env/vecEnv.js'
import { PPOAgent, RewardNormalizer } from './ppo.js'

let paused = false
let stopped = false
let agent = null
let env = null
let debugActiveJoint = null
let debugDirection = 0
let debugGrab = null
let debugProps = []
let debugPinned = false
let debugNeedsReset = false
let RAPIER = null

// ─── Character definitions ──────────────────────────────────────────────────

const CHAR_LOADERS = {
  cartpole: () => import('../env/characters/cartpole.js').then(m => m.CARTPOLE),
  hopper: () => import('../env/characters/hopper.js').then(m => m.HOPPER),
  walker2d: () => import('../env/characters/walker2d.js').then(m => m.WALKER2D),
  acrobot: () => import('../env/characters/acrobot.js').then(m => m.ACROBOT),
  'acrobot-damped': () => import('../env/characters/acrobot_damped.js').then(m => m.ACROBOT_DAMPED),
  terrain: () => import('../env/characters/biped.js').then(m => m.BIPED),
  'spinner-constant': () => import('../env/characters/spinner_constant.js').then(m => m.SPINNER_CONSTANT),
  'spinner-max': () => import('../env/characters/spinner_max.js').then(m => m.SPINNER_MAX),
  'red-light-green-light': () => import('../env/characters/red_light_green_light.js').then(m => m.RED_LIGHT_GREEN_LIGHT),
}

/**
 * Get the env options for a given envType.
 */
function getEnvOpts(envType, charDef) {
  // Terrain mode: PD control + terrain generation
  if (envType === 'terrain') {
    return {
      controlMode: 'pd',
      terrain: true,
      gravity: -5.5,
      physicsHz: 240,
      policyHz: 30,
      difficulty: charDef?.difficulty ?? 0.3,
      terrainLength: 50,
      maxSteps: 1000,
    }
  }

  // Spinner envs: velocity control, zero gravity
  if (envType === 'spinner-constant' || envType === 'spinner-max' || envType === 'red-light-green-light') {
    return {
      controlMode: 'velocity',
      physicsHz: 240,
      policyHz: 30,
      gravity: 0,
      maxSteps: 1000,
    }
  }

  // CartPole: higher physics rate for stability
  if (envType === 'cartpole') {
    return {
      controlMode: 'velocity',
      physicsHz: 240,
      policyHz: 50,
      gravity: -9.8,
      maxSteps: 500,
    }
  }

  // Standard Rapier envs (Hopper, Walker2D, Acrobot)
  return {
    controlMode: 'pd',
    physicsHz: 120,
    policyHz: 30,
    maxSteps: 1000,
  }
}

// ─── TF Backend ──────────────────────────────────────────────────────────────

const WEBGPU_MIN_STORAGE_BUFFERS = 16

/**
 * Register a custom 'webgpu' backend that requests maxStorageBuffersPerShaderStage
 * from the adapter. The stock TF.js registration omits this limit, causing
 * pipeline creation failures on adapters that support >8 but default to 8.
 */
async function registerWebGPUBackend() {
  if (typeof navigator === 'undefined' || !navigator.gpu) return { viable: false, deviceName: '' }
  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) return { viable: false, deviceName: '' }

    const adapterLimits = adapter.limits
    if (adapterLimits.maxStorageBuffersPerShaderStage < WEBGPU_MIN_STORAGE_BUFFERS) {
      return { viable: false, deviceName: '' }
    }

    let deviceName = ''
    try {
      const info = 'info' in adapter ? adapter.info : await adapter.requestAdapterInfo()
      deviceName = info.device || info.description || ''
    } catch { /* */ }

    // Remove the stock registration (imported side-effect) so we can re-register.
    // Must use findBackendFactory — findBackend only returns instantiated backends,
    // which is null before first use, so our removal would be skipped.
    if (tf.findBackendFactory('webgpu') != null) tf.removeBackend('webgpu')

    tf.registerBackend('webgpu', async () => {
      const requiredFeatures = []
      if (adapter.features.has('timestamp-query')) requiredFeatures.push('timestamp-query')
      if (adapter.features.has('bgra8unorm-storage')) requiredFeatures.push('bgra8unorm-storage')

      const device = await adapter.requestDevice({
        requiredFeatures,
        requiredLimits: {
          maxComputeWorkgroupStorageSize:    adapterLimits.maxComputeWorkgroupStorageSize,
          maxComputeWorkgroupsPerDimension:  adapterLimits.maxComputeWorkgroupsPerDimension,
          maxStorageBufferBindingSize:        adapterLimits.maxStorageBufferBindingSize,
          maxBufferSize:                      adapterLimits.maxBufferSize,
          maxComputeWorkgroupSizeX:           adapterLimits.maxComputeWorkgroupSizeX,
          maxComputeInvocationsPerWorkgroup:  adapterLimits.maxComputeInvocationsPerWorkgroup,
          // ── This is the missing limit that causes the pipeline errors ──
          maxStorageBuffersPerShaderStage:    adapterLimits.maxStorageBuffersPerShaderStage,
        },
      })
      const adapterInfo = 'info' in adapter ? adapter.info
        : 'requestAdapterInfo' in adapter ? await adapter.requestAdapterInfo()
        : undefined
      return new WebGPUBackend(device, adapterInfo)
    }, 3 /* priority */)

    return { viable: true, deviceName }
  } catch {
    return { viable: false, deviceName: '' }
  }
}

async function initTFBackend() {
  const gpu = await registerWebGPUBackend()
  const backends = gpu.viable ? ['webgpu', 'webgl', 'cpu'] : ['webgl', 'cpu']
  for (const b of backends) {
    try {
      await tf.setBackend(b)
      await tf.ready()
      // Smoke-test: run a small matmul to verify the backend works
      const a = tf.ones([2, 2])
      const c = tf.matMul(a, a)
      await c.data()          // force GPU execution
      tf.dispose([a, c])
      return { backend: b, deviceName: gpu.deviceName }
    } catch {
      try { tf.disposeVariables() } catch { /* */ }
      continue
    }
  }
  return { backend: 'cpu', deviceName: '' }
}

// ─── Environment Creation ────────────────────────────────────────────────────

async function initRapier() {
  if (!RAPIER) {
    RAPIER = await import('@dimforge/rapier2d-compat')
    // Fetch and pass WASM bytes directly — the compat package's default URL
    // resolution is broken in bundled builds
    const wasmModule = await import(
      '@dimforge/rapier2d-compat/rapier_wasm2d_bg.wasm?url'
    )
    await RAPIER.init(wasmModule.default)
    postMessage({ type: 'STATUS', msg: 'Rapier WASM ready' })
  }
}

async function createEnv(envType, customCharDef = null) {
  await initRapier()
  const { UnifiedRapierEnv } = await import('../env/unifiedEnv.js')

  const charDef = customCharDef || await CHAR_LOADERS[envType]()
  const opts = getEnvOpts(envType, charDef)

  return new UnifiedRapierEnv(charDef, opts)
}

async function createVecEnv(envType, numEnvs, customCharDef = null) {
  await initRapier()
  const { UnifiedRapierEnv } = await import('../env/unifiedEnv.js')

  const charDef = customCharDef || await CHAR_LOADERS[envType]()
  const opts = getEnvOpts(envType, charDef)

  return new VecEnv(() => new UnifiedRapierEnv(charDef, opts), numEnvs)
}

// ─── Training ────────────────────────────────────────────────────────────────

const DEFAULT_NUM_ENVS = 8

async function runTraining(config) {
  const { envType = 'cartpole', ppoConfig = {}, maxUpdates = 3000, numEnvs = DEFAULT_NUM_ENVS, charDef = null } = config

  const { backend: tfBackend, deviceName } = await initTFBackend()
  postMessage({ type: 'BACKEND', backend: tfBackend, deviceName })

  env = await createVecEnv(envType, numEnvs, charDef)
  postMessage({ type: 'STATUS', msg: `${charDef?.name || envType} ready · ${numEnvs} envs` })

  const obsSize = env.observationSize
  const actSize = env.actionSize
  agent = new PPOAgent(obsSize, actSize, { ...ppoConfig, maxUpdates })

  await runVectorizedLoop(maxUpdates, numEnvs)

  env.dispose?.()
  postMessage({ type: 'STATUS', msg: stopped ? 'Stopped' : 'Training complete' })
}

async function runVectorizedLoop(maxUpdates, numEnvs) {
  let totalSteps = 0
  let totalEpisodes = 0
  let updateCount = 0
  let episodeRewards = []
  let lastRenderTime = 0
  const RENDER_INTERVAL_MS = 16

  const rewardNorm = new RewardNormalizer(numEnvs, agent.cfg.gamma)
  const epRewards = new Float32Array(numEnvs)
  const epSteps = new Uint32Array(numEnvs)

  let rawObsArray = env.resetAll()
  agent.updateObsStats(rawObsArray)
  let obsArray = agent.normalizeObsBatch(rawObsArray)

  postMessage({ type: 'STATUS', msg: 'Training started' })

  let t_inference = 0
  let t_physics = 0
  let t_update = 0
  let t_rolloutStart = performance.now()

  // Reward component accumulator (sampled from env[0] each step)
  let rewardCompSums = {}
  let rewardCompCount = 0

  while (!stopped && updateCount < maxUpdates) {
    while (paused && !stopped) await new Promise(r => setTimeout(r, 100))
    if (stopped) break

    const t0 = performance.now()
    const { actions, values, logProbs } = await agent.actBatch(obsArray)
    const t1 = performance.now()
    t_inference += t1 - t0

    const { obs: nextRawObsArray, rewards: rawRewards, dones } = env.stepAll(actions)
    const t2 = performance.now()
    t_physics += t2 - t1

    // Sample reward breakdown from env[0] for visualization
    const rb = env.getRewardBreakdown()
    if (rb?.components) {
      for (const c of rb.components) {
        rewardCompSums[c.label] = (rewardCompSums[c.label] ?? 0) + c.value
      }
      rewardCompCount++
    }

    agent.updateObsStats(nextRawObsArray)
    const nextObsArray = agent.normalizeObsBatch(nextRawObsArray)
    const rewards = rewardNorm.normalize(rawRewards, dones)

    agent.storeTransitions(obsArray, actions, rewards, dones, values, logProbs)
    totalSteps += numEnvs

    for (let i = 0; i < numEnvs; i++) {
      epRewards[i] += rawRewards[i]
      epSteps[i]++
      if (dones[i]) {
        episodeRewards.push(epRewards[i])
        postMessage({ type: 'EPISODE', data: { episode: totalEpisodes, reward: epRewards[i], steps: epSteps[i], totalSteps } })
        epRewards[i] = 0
        epSteps[i] = 0
        totalEpisodes++
      }
    }

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

    rawObsArray = nextRawObsArray
    obsArray = nextObsArray

    if (agent.bufferFull) {
      const tUpdateStart = performance.now()
      const metrics = await agent.update(obsArray, numEnvs)
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
        numEnvs,
      }

      t_inference = 0
      t_physics = 0
      t_rolloutStart = performance.now()

      const recent = episodeRewards.slice(-20)
      const meanReward = recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : 0

      // Average reward component breakdown over this rollout
      const rewardBreakdown = {}
      if (rewardCompCount > 0) {
        for (const [label, sum] of Object.entries(rewardCompSums)) {
          rewardBreakdown[label] = sum / rewardCompCount
        }
      }
      rewardCompSums = {}
      rewardCompCount = 0

      postMessage({
        type: 'METRICS',
        data: { update: updateCount, totalSteps, totalEpisodes, meanReward20: meanReward, ...metrics, timing, tensorMemory: tf.memory().numTensors, rewardBreakdown }
      })

      await new Promise(r => setTimeout(r, 0))
    }
  }
}

// ─── Playback ────────────────────────────────────────────────────────────────

async function runPlayback(config) {
  const { envType = 'cartpole', ppoConfig = {}, weights, charDef = null } = config

  const { backend: tfBackend, deviceName } = await initTFBackend()
  postMessage({ type: 'BACKEND', backend: tfBackend, deviceName })

  env = await createEnv(envType, charDef)
  agent = new PPOAgent(env.observationSize, env.actionSize, ppoConfig)
  agent.importWeights(weights)
  postMessage({ type: 'STATUS', msg: 'Playback' })

  let rawObs = env.reset()
  let obs = agent.normalizeObs(rawObs)
  let episodeReward = 0
  let episodeSteps = 0
  let totalEpisodes = 0

  // Diagnostic: capture first N steps for debugging
  const DIAG_STEPS = 60
  const diagLog = []

  while (!stopped) {
    while (paused && !stopped) await new Promise(r => setTimeout(r, 100))
    if (stopped) break

    const action = env.actionSize > 1
      ? await agent.actDeterministicMulti(obs)
      : await agent.actDeterministic(obs)

    const { obs: nextRawObs, reward, done, info } = env.step(action)
    episodeReward += reward
    episodeSteps++

    // Capture diagnostic data for first steps of first episode
    if (totalEpisodes === 0 && diagLog.length < DIAG_STEPS) {
      const rb = env.computeRewardBreakdown?.()
      diagLog.push({
        step: episodeSteps,
        rawObs: rawObs.map(v => +v.toFixed(4)),
        normObs: obs.map(v => +v.toFixed(4)),
        action: (Array.isArray(action) || ArrayBuffer.isView(action)) ? Array.from(action).map(v => +v.toFixed(4)) : +action.toFixed(4),
        reward: +reward.toFixed(4),
        done,
        info: info ? { healthy: info.healthy, stepCount: info.stepCount } : undefined,
        rewardComponents: rb?.components?.map(c => ({ label: c.label, value: +c.value.toFixed(4) })),
        bodies: (() => {
          const snap = env.getRenderSnapshot()
          const b = {}
          for (const [id, s] of Object.entries(snap)) {
            if (id.startsWith('_')) continue
            b[id] = { x: +s.x.toFixed(3), y: +s.y.toFixed(3), angle: +(s.angle * 180 / Math.PI).toFixed(1) }
          }
          return b
        })(),
      })
      if (diagLog.length === DIAG_STEPS || done) {
        postMessage({ type: 'PLAYBACK_DIAGNOSTIC', data: { envType, obsSize: env.observationSize, actionSize: env.actionSize, steps: diagLog } })
      }
    }

    postMessage({
      type: 'RENDER_SNAPSHOT',
      snapshot: env.getRenderSnapshot(),
      episodeReward,
      episodeSteps,
      resetReason: done ? (info?.reason || 'fell') : null,
    })

    rawObs = nextRawObs
    obs = agent.normalizeObs(rawObs)

    if (done) {
      // Send diagnostic if episode ended early
      if (totalEpisodes === 0 && diagLog.length > 0 && diagLog.length < DIAG_STEPS) {
        postMessage({ type: 'PLAYBACK_DIAGNOSTIC', data: { envType, obsSize: env.observationSize, actionSize: env.actionSize, steps: diagLog } })
      }
      postMessage({ type: 'EPISODE', data: { episode: totalEpisodes, reward: episodeReward, steps: episodeSteps, totalSteps: 0 } })
      // Brief pause so user can register the reset banner
      await new Promise(r => setTimeout(r, 500))
      episodeReward = 0
      episodeSteps = 0
      totalEpisodes++
      rawObs = env.reset()
      obs = agent.normalizeObs(rawObs)
    }

    await new Promise(r => setTimeout(r, 16))
  }

  env.dispose?.()
  postMessage({ type: 'STATUS', msg: 'Playback stopped' })
}

// ─── Physics Debug ───────────────────────────────────────────────────────────

async function runPhysicsDebug(config) {
  const { charDef = null, envType = 'terrain' } = config

  await initRapier()
  const { UnifiedRapierEnv } = await import('../env/unifiedEnv.js')

  const finalCharDef = charDef || await CHAR_LOADERS[envType]()

  // Flat ground, no terrain, no auto-termination
  env = new UnifiedRapierEnv(finalCharDef, {
    controlMode: 'velocity',
    terrain: false,
    gravity: -9.81,
    physicsHz: 240,
    policyHz: 30,
    maxSteps: 999999,
  })

  env.reset()

  // Props helper — creates test objects in the current world
  // Must be called after every reset since reset rebuilds the world
  function createProps() {
    const p = []
    const dynamicBodies = finalCharDef.bodies.filter(b => !b.fixed)
    const minSpawnY = Math.min(...dynamicBodies.map(b => b.spawnY ?? 1.0))
    const maxSpawnY = Math.max(...dynamicBodies.map(b => b.spawnY ?? 1.0))
    const legHeight = maxSpawnY - minSpawnY
    const boxH = Math.max(0.15, legHeight * 0.66)
    const boxW = boxH * 0.8
    const groundY = finalCharDef.ground?.y ?? 0

    const boxRb = env.world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(1.5, groundY + boxH / 2 + 0.01)
    )
    const boxCollider = RAPIER.ColliderDesc
      .cuboid(boxW / 2, boxH / 2)
      .setFriction(0.6)
      .setRestitution(0.2)
      .setDensity(500)
    env.world.createCollider(boxCollider, boxRb)
    p.push({ id: '_prop_box', rb: boxRb, shape: 'box', w: boxW, h: boxH })
    debugProps = p
    return p
  }

  let props = createProps()

  postMessage({ type: 'STATUS', msg: 'Physics debug · ragdoll' })

  const SUBSTEPS = 8

  while (!stopped) {
    while (paused && !stopped) await new Promise(r => setTimeout(r, 100))
    if (stopped) break

    // Handle reset (must happen in this scope so props can be recreated)
    if (debugNeedsReset) {
      debugNeedsReset = false
      env.reset()
      props = createProps()
      postMessage({ type: 'STATUS', msg: 'Physics debug · reset' })
    }

    // Apply debug torque as IMPULSE (one-shot, doesn't persist across steps)
    if (debugActiveJoint && debugDirection !== 0) {
      const jDef = finalCharDef.joints.find(j => j.id === debugActiveJoint)
      if (jDef) {
        const bodyA = env.bodies[jDef.bodyA]
        const bodyB = env.bodies[jDef.bodyB]
        if (bodyA && bodyB) {
          if (jDef.type === 'prismatic') {
            const axis = jDef.axis || [1, 0]
            const impulse = debugDirection * Math.max(jDef.maxTorque ?? 10, 2) * 0.02
            bodyB.applyImpulse({ x: axis[0] * impulse, y: axis[1] * impulse }, true)
          } else {
            const mass = bodyB.mass()
            const armLen = 0.25
            const gravityTorque = mass * 9.81 * armLen
            const impulse = debugDirection * gravityTorque * 0.24
            bodyB.applyTorqueImpulse(impulse, true)
            if (!bodyA.isFixed()) bodyA.applyTorqueImpulse(-impulse * 0.3, true)
          }
        }
      }
    }

    for (let sub = 0; sub < SUBSTEPS; sub++) {

      // Mouse grab: move body toward cursor via velocity targeting
      if (debugGrab) {
        let rb = env.bodies[debugGrab.bodyId]
        if (!rb) {
          const prop = debugProps.find(p => p.id === debugGrab.bodyId)
          if (prop) rb = prop.rb
        }
        if (rb) {
          // Compute where the local grab point currently is in world space
          const pos = rb.translation()
          const rot = rb.rotation()
          const cos = Math.cos(rot)
          const sin = Math.sin(rot)
          const grabWorldX = pos.x + debugGrab.localX * cos - debugGrab.localY * sin
          const grabWorldY = pos.y + debugGrab.localX * sin + debugGrab.localY * cos

          // Desired velocity = direction to cursor, proportional to distance
          const dx = debugGrab.wx - grabWorldX
          const dy = debugGrab.wy - grabWorldY
          const dist = Math.sqrt(dx * dx + dy * dy)

          const maxSpeed = 8.0
          const gain = 15.0  // how aggressively it tracks (per second)
          let vx = dx * gain
          let vy = dy * gain

          // Clamp velocity
          const speed = Math.sqrt(vx * vx + vy * vy)
          if (speed > maxSpeed) {
            vx = vx / speed * maxSpeed
            vy = vy / speed * maxSpeed
          }

          // Blend toward desired velocity (don't slam it — smooth transition)
          const curVel = rb.linvel()
          const blend = 0.3
          rb.setLinvel({
            x: curVel.x + (vx - curVel.x) * blend,
            y: curVel.y + (vy - curVel.y) * blend,
          }, true)

          // Dampen angular velocity so it doesn't spin wildly
          const angvel = rb.angvel()
          rb.setAngvel(angvel * 0.85, true)
        }
      }

      env.world.step()
    }

    // Update foot contacts
    for (const [bodyId, handle] of Object.entries(env.footSensorHandles)) {
      env._footContacts[bodyId] = false
      env.world.narrowPhase.intersectionPairsWith(handle, () => {
        env._footContacts[bodyId] = true
      })
    }

    const snapshot = env.getRenderSnapshot()

    // Compute grab point world position for rendering
    let grabWorldPoint = null
    if (debugGrab) {
      let rb = env.bodies[debugGrab.bodyId]
      if (!rb) {
        const prop = debugProps.find(p => p.id === debugGrab.bodyId)
        if (prop) rb = prop.rb
      }
      if (rb) {
        const pos = rb.translation()
        const rot = rb.rotation()
        const cos = Math.cos(rot)
        const sin = Math.sin(rot)
        grabWorldPoint = {
          x: pos.x + debugGrab.localX * cos - debugGrab.localY * sin,
          y: pos.y + debugGrab.localX * sin + debugGrab.localY * cos,
          targetX: debugGrab.wx,
          targetY: debugGrab.wy,
        }
      }
    }

    // Add prop positions to snapshot
    const propSnapshots = []
    for (const prop of props) {
      const pos = prop.rb.translation()
      propSnapshots.push({
        id: prop.id,
        x: pos.x,
        y: pos.y,
        angle: prop.rb.rotation(),
        shape: prop.shape,
        w: prop.w,
        h: prop.h,
      })
    }

    snapshot._obs = env._getObs()
    snapshot._rewardBreakdown = env.computeRewardBreakdown()

    snapshot._debug = {
      joints: snapshot._joints,
      activeJoint: debugActiveJoint,
      direction: debugDirection,
      bodyLabels: snapshot._bodyLabels,
      grab: grabWorldPoint,
      props: propSnapshots,
      pinned: debugPinned,
    }

    postMessage({
      type: 'RENDER_SNAPSHOT',
      snapshot,
      episodeReward: 0,
      episodeSteps: 0,
    })

    await new Promise(r => setTimeout(r, 16))
  }

  env.dispose?.()
  postMessage({ type: 'STATUS', msg: 'Debug stopped' })
}

// ─── Message Handler ─────────────────────────────────────────────────────────

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
    case 'PHYSICS_DEBUG':
      stopped = false; paused = false
      debugActiveJoint = null; debugDirection = 0
      runPhysicsDebug(config)
      break
    case 'DEBUG_CONTROL':
      debugActiveJoint = config.joint ?? null
      debugDirection = config.direction ?? 0
      break
    case 'DEBUG_GRAB': {
      // Look up in env bodies or props
      let grabRb = env?.bodies[config.bodyId]
      if (!grabRb) {
        const prop = debugProps.find(p => p.id === config.bodyId)
        if (prop) grabRb = prop.rb
      }
      if (grabRb) {
        // Compute local-space grab offset (world click -> body local frame)
        const pos = grabRb.translation()
        const rot = grabRb.rotation()
        const cos = Math.cos(-rot)
        const sin = Math.sin(-rot)
        const relX = config.wx - pos.x
        const relY = config.wy - pos.y
        const localX = relX * cos - relY * sin
        const localY = relX * sin + relY * cos
        debugGrab = { bodyId: config.bodyId, wx: config.wx, wy: config.wy, localX, localY }
      }
      break
    }
    case 'DEBUG_DRAG':
      if (debugGrab) {
        debugGrab.wx = config.wx
        debugGrab.wy = config.wy
      }
      break
    case 'DEBUG_RELEASE': {
      if (debugGrab) {
        let relRb = env?.bodies[debugGrab.bodyId]
        if (!relRb) {
          const prop = debugProps.find(p => p.id === debugGrab.bodyId)
          if (prop) relRb = prop.rb
        }
        if (relRb) {
          // no cleanup needed — body was kept fully dynamic
        }
      }
      debugGrab = null
      break
    }
    case 'DEBUG_RESET':
      debugGrab = null
      debugPinned = false
      debugNeedsReset = true
      break
    case 'DEBUG_PIN_TORSO': {
      if (!env || !RAPIER) break
      const charDef = env.def
      const forwardId = charDef.forwardBody || charDef.bodies?.find(b => b.id === 'torso')?.id || charDef.bodies?.[0]?.id
      const forwardDef = charDef.bodies?.find(b => b.id === forwardId)
      const rb = env.bodies[forwardId]
      if (!rb) break
      debugPinned = !debugPinned
      if (debugPinned) {
        // Lift to spawn position + extra clearance, zero out velocity, then fix
        const spawnY = forwardDef?.spawnY ?? 1.5
        const liftY = Math.max(spawnY, 1.5)
        rb.setBodyType(RAPIER.RigidBodyType.Dynamic, true)
        rb.setTranslation({ x: rb.translation().x, y: liftY }, true)
        rb.setLinvel({ x: 0, y: 0 }, true)
        rb.setAngvel(0, true)
        rb.setRotation(0, true)
        // Also reset child body velocities so limbs dangle cleanly
        for (const bDef of charDef.bodies) {
          if (bDef.id === forwardId || bDef.fixed) continue
          const childRb = env.bodies[bDef.id]
          if (childRb) {
            childRb.setLinvel({ x: 0, y: 0 }, true)
            childRb.setAngvel(0, true)
          }
        }
        rb.setBodyType(RAPIER.RigidBodyType.Fixed, true)
        postMessage({ type: 'STATUS', msg: 'Physics debug · torso pinned' })
      } else {
        rb.setBodyType(RAPIER.RigidBodyType.Dynamic, true)
        postMessage({ type: 'STATUS', msg: 'Physics debug · torso released' })
      }
      break
    }

    // ── Scene Hierarchy property editing ──────────────────────────────────
    case 'SCENE_SET_BODY_POS': {
      const targetEnv = env?.envs ? env.envs[0] : env
      const bodyRb = targetEnv?.bodies[config.bodyId]
      if (bodyRb) bodyRb.setTranslation({ x: config.x, y: config.y }, true)
      break
    }
    case 'SCENE_SET_BODY_ANGLE': {
      const targetEnv = env?.envs ? env.envs[0] : env
      const bodyRb = targetEnv?.bodies[config.bodyId]
      if (bodyRb) bodyRb.setRotation(config.angle, true)
      break
    }
    case 'SCENE_SET_BODY_LINVEL': {
      const targetEnv = env?.envs ? env.envs[0] : env
      const bodyRb = targetEnv?.bodies[config.bodyId]
      if (bodyRb) bodyRb.setLinvel({ x: config.vx, y: config.vy }, true)
      break
    }
    case 'SCENE_SET_BODY_ANGVEL': {
      const targetEnv = env?.envs ? env.envs[0] : env
      const bodyRb = targetEnv?.bodies[config.bodyId]
      if (bodyRb) bodyRb.setAngvel(config.angvel, true)
      break
    }
    case 'SCENE_SET_MOTOR_TARGET': {
      const targetEnv = env?.envs ? env.envs[0] : env
      const joint = targetEnv?.joints[config.jointId]
      const jDef = targetEnv?.def.joints?.find(j => j.id === config.jointId)
      if (joint && jDef) {
        joint.configureMotorVelocity(config.target, jDef.maxTorque || 100)
      }
      break
    }
  }
}
