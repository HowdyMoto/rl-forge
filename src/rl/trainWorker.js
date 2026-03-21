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
import '@tensorflow/tfjs-backend-webgpu'
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
  terrain: () => import('../env/characters/biped.js').then(m => m.BIPED),
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
    controlMode: 'velocity',
    physicsHz: 120,
    policyHz: 30,
    maxSteps: 1000,
  }
}

// ─── TF Backend ──────────────────────────────────────────────────────────────

const WEBGPU_MIN_STORAGE_BUFFERS = 16

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
    } catch { /* */ }
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

// ─── Environment Creation ────────────────────────────────────────────────────

async function initRapier() {
  if (!RAPIER) {
    RAPIER = await import('@dimforge/rapier2d')
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
      const metrics = await agent.update(obsArray[0])
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

      postMessage({
        type: 'METRICS',
        data: { update: updateCount, totalSteps, totalEpisodes, meanReward20: meanReward, ...metrics, timing, tensorMemory: tf.memory().numTensors }
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

  while (!stopped) {
    while (paused && !stopped) await new Promise(r => setTimeout(r, 100))
    if (stopped) break

    const action = env.actionSize > 1
      ? agent.actDeterministicMulti(obs)
      : agent.actDeterministic(obs)

    const { obs: nextRawObs, reward, done } = env.step(action)
    episodeReward += reward
    episodeSteps++

    postMessage({
      type: 'RENDER_SNAPSHOT',
      snapshot: env.getRenderSnapshot(),
      episodeReward,
      episodeSteps,
    })

    rawObs = nextRawObs
    obs = agent.normalizeObs(rawObs)

    if (done) {
      postMessage({ type: 'EPISODE', data: { episode: totalEpisodes, reward: episodeReward, steps: episodeSteps, totalSteps: 0 } })
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

    for (let sub = 0; sub < SUBSTEPS; sub++) {
      // Joint torque/force testing
      if (debugActiveJoint && debugDirection !== 0) {
        const jDef = finalCharDef.joints.find(j => j.id === debugActiveJoint)
        if (jDef) {
          const bodyA = env.bodies[jDef.bodyA]
          const bodyB = env.bodies[jDef.bodyB]
          if (bodyA && bodyB) {
            if (jDef.type === 'prismatic') {
              // Prismatic: apply force along the joint axis
              const axis = jDef.axis || [1, 0]
              const force = debugDirection * (jDef.maxTorque ?? 10) * 0.5
              bodyB.addForce({ x: axis[0] * force, y: axis[1] * force }, true)
            } else {
              // Revolute: apply torque
              const torque = debugDirection * (jDef.maxTorque ?? 300) * 0.5
              bodyB.addTorque(torque, true)
              bodyA.addTorque(-torque, true)
            }
          }
        }
      }

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
  }
}
