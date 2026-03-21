/**
 * TerrainRapierEnv
 *
 * A physics environment inspired by Peng et al. "Terrain-Adaptive Locomotion
 * Skills Using Deep Reinforcement Learning" (2016):
 *
 *   - PD target-angle control: policy outputs target joint angles in [-1,1],
 *     mapped to joint limits via addTorque() with manual PD each substep.
 *   - Dense terrain perception: 40 height samples from behind to well ahead
 *   - Linear velocity reward + alive bonus + control cost penalty
 *   - High-frequency physics (240Hz) with 30Hz policy for stable PD tracking
 *   - Procedural terrain (gaps, ramps, stairs, bumps)
 */

import RAPIER from '@dimforge/rapier2d'
import { generateTerrain, sampleHeightfield } from './terrain.js'

// Physics at 240Hz, policy at 30Hz → 8 substeps per policy step
const PHYSICS_DT = 1 / 240
const POLICY_FREQ = 30
const SUBSTEPS_PER_POLICY = Math.round((1 / POLICY_FREQ) / PHYSICS_DT)  // = 8

const GRAVITY = -5.5  // lower than Earth for natural on-screen feel at this render scale

// Default PD gains (overridden per-joint in character def)
// These are torque-domain gains (Nm/rad). At ~1 rad error, PD output ≈ maxTorque.
// Too high → bang-bang saturation → jerky/unstable. Too low → limp joints.
const DEFAULT_KP = 300
const DEFAULT_KD = 30

// Terrain perception — denser sampling and longer look-ahead than before
export const TERRAIN_SAMPLES = 40   // Peng uses 200; 40 is practical for browser PPO
const LOOK_AHEAD = 8.0       // meters ahead (Peng: 10m)
const LOOK_BEHIND = 1.0      // meters behind (Peng: 0.5m)

export class TerrainRapierEnv {
  /**
   * @param {Object} characterDef - Creature body definition (same format as hopper.js etc, with optional PD params)
   * @param {Object} [opts] - Options
   * @param {number} [opts.terrainSeed] - Terrain generation seed
   * @param {number} [opts.difficulty] - Terrain difficulty 0.0-1.0
   * @param {number} [opts.terrainLength] - Total terrain length in meters
   * @param {number} [opts.maxSteps] - Max steps per episode
   */
  constructor(characterDef, opts = {}) {
    this.def = characterDef
    this.terrainSeed = opts.terrainSeed ?? Math.floor(Math.random() * 100000)
    this.difficulty = opts.difficulty ?? 0.3
    this.terrainLength = opts.terrainLength ?? 50
    this.maxSteps = opts.maxSteps ?? 1000

    this.world = null
    this.bodies = {}
    this.joints = {}
    this.footSensorHandles = {}
    this._footContacts = {}
    this.stepCount = 0
    this._prevTorsoX = 0
    this._maxTorsoX = 0   // track furthest distance for scoring

    // PD controller state
    this._targetAngles = null    // current target joint angles from policy

    // Terrain
    this.terrain = null
    this.terrainBodies = []

    // Count actuated joints for action space sizing
    this._actuatedJoints = characterDef.joints.filter(j => (j.maxTorque ?? 0) > 0)
    // Observation size: base body obs + joint obs + foot contacts + terrain heightfield
    const numJoints = characterDef.joints.length
    const numFeet = characterDef.bodies.filter(b => b.isFootBody).length
    this._baseObsSize = 5 + numJoints * 2 + numFeet  // same as RapierEnv
    this._obsSize = this._baseObsSize + TERRAIN_SAMPLES
    this._actSize = this._actuatedJoints.length  // target angles for actuated joints
  }

  _buildWorld() {
    if (this.world) this.world.free()

    this.world = new RAPIER.World({ x: 0.0, y: GRAVITY })
    this.world.timestep = PHYSICS_DT
    this.bodies = {}
    this.joints = {}
    this.footSensorHandles = {}
    this.terrainBodies = []

    // Generate terrain (use curriculum difficulty if available)
    this.terrain = generateTerrain(this.terrainSeed, this.terrainLength, this._effectiveDifficulty ?? this.difficulty)

    // Build terrain colliders from segments
    for (const seg of this.terrain.segments) {
      const dx = seg.x2 - seg.x1
      const dy = seg.y2 - seg.y1
      const length = Math.sqrt(dx * dx + dy * dy)
      if (length < 0.001) continue

      const cx = (seg.x1 + seg.x2) / 2
      const cy = (seg.y1 + seg.y2) / 2
      const angle = Math.atan2(dy, dx)

      const body = this.world.createRigidBody(
        RAPIER.RigidBodyDesc.fixed().setTranslation(cx, cy).setRotation(angle)
      )
      const collider = RAPIER.ColliderDesc
        .cuboid(length / 2, 0.05)
        .setFriction(0.8)
        .setRestitution(0.1)
      this.world.createCollider(collider, body)
      this.terrainBodies.push(body)
    }

    // Build character bodies
    const def = this.def
    for (const bodyDef of def.bodies) {
      const rbDesc = (bodyDef.fixed
        ? RAPIER.RigidBodyDesc.fixed()
        : RAPIER.RigidBodyDesc.dynamic()
      )
        .setTranslation(bodyDef.spawnX, bodyDef.spawnY)
        .setRotation(bodyDef.spawnAngle ?? 0)

      const rb = this.world.createRigidBody(rbDesc)

      let colliderDesc
      if (bodyDef.shape === 'box') {
        colliderDesc = RAPIER.ColliderDesc.cuboid(bodyDef.w / 2, bodyDef.h / 2)
      } else if (bodyDef.shape === 'capsule') {
        const halfHeight = Math.max(0.001, (bodyDef.length - 2 * bodyDef.radius) / 2)
        colliderDesc = RAPIER.ColliderDesc.capsule(halfHeight, bodyDef.radius)
      } else if (bodyDef.shape === 'ball') {
        colliderDesc = RAPIER.ColliderDesc.ball(bodyDef.radius)
      } else {
        throw new Error(`Unknown shape: ${bodyDef.shape}`)
      }

      colliderDesc
        .setFriction(bodyDef.friction ?? 0.5)
        .setRestitution(bodyDef.restitution ?? 0.0)
        .setDensity(bodyDef.mass / this._bodyVolume(bodyDef))

      // Foot sensor
      if (bodyDef.isFootBody) {
        const sensorRadius = bodyDef.radius ?? (bodyDef.w ? bodyDef.w / 2 : 0.04)
        const sensorY = bodyDef.shape === 'capsule' ? -(bodyDef.length / 2) : -(bodyDef.h ? bodyDef.h / 2 : sensorRadius)
        const sensorDesc = RAPIER.ColliderDesc
          .cuboid(sensorRadius, 0.02)
          .setTranslation(0, sensorY)
          .setSensor(true)
          .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS)
        const sensorCollider = this.world.createCollider(sensorDesc, rb)
        this.footSensorHandles[bodyDef.id] = sensorCollider.handle
      }

      this.world.createCollider(colliderDesc, rb)
      this.bodies[bodyDef.id] = rb
    }

    // Build joints
    for (const jointDef of def.joints) {
      const bodyA = this.bodies[jointDef.bodyA]
      const bodyB = this.bodies[jointDef.bodyB]
      if (!bodyA || !bodyB) continue

      const [axA, ayA] = jointDef.anchorA
      const [axB, ayB] = jointDef.anchorB

      const jointData = RAPIER.JointData.revolute(
        { x: axA, y: ayA },
        { x: axB, y: ayB }
      )

      jointData.limitsEnabled = true
      jointData.limits = [jointDef.lowerLimit, jointDef.upperLimit]

      const joint = this.world.createImpulseJoint(jointData, bodyA, bodyB, true)

      // Passive damping resists joint motion (overridden per-step by PD motor)
      if ((jointDef.damping ?? 0) > 0) {
        joint.configureMotorVelocity(0.0, jointDef.damping)
      }

      this.joints[jointDef.id] = joint
    }
  }

  _bodyVolume(bodyDef) {
    if (bodyDef.shape === 'box') return (bodyDef.w ?? 0.1) * (bodyDef.h ?? 0.1) * 0.1
    if (bodyDef.shape === 'capsule') {
      const r = bodyDef.radius ?? 0.05
      return Math.PI * r * r * (bodyDef.length ?? 0.3)
    }
    if (bodyDef.shape === 'ball') {
      const r = bodyDef.radius ?? 0.05
      return (4 / 3) * Math.PI * r * r * r
    }
    return 1.0
  }

  reset() {
    // Keep the same terrain seed for the first N episodes so the agent can
    // learn basic locomotion on consistent ground, then start randomizing.
    this._episodeCount = (this._episodeCount ?? 0) + 1
    if (this._episodeCount > 50) {
      this.terrainSeed = Math.floor(Math.random() * 100000)
    }
    // Curriculum: start with very low difficulty, ramp up over episodes
    this._effectiveDifficulty = Math.min(this.difficulty, this._episodeCount / 100)
    this._buildWorld()
    this.stepCount = 0
    this._footContacts = {}

    // Random perturbation — forces the policy to learn recovery and handle
    // varied starting poses, improving robustness to external perturbations
    const def = this.def
    for (const bodyDef of def.bodies) {
      if (bodyDef.fixed) continue
      const rb = this.bodies[bodyDef.id]
      rb.setTranslation({
        x: bodyDef.spawnX + (Math.random() - 0.5) * 0.15,
        y: bodyDef.spawnY + (Math.random() - 0.5) * 0.1,
      }, true)
      rb.setRotation((bodyDef.spawnAngle ?? 0) + (Math.random() - 0.5) * 0.4, true)
      rb.setLinvel({ x: (Math.random() - 0.5) * 0.5, y: (Math.random() - 0.5) * 0.3 }, true)
      rb.setAngvel((Math.random() - 0.5) * 0.5, true)
    }

    // Initialize PD targets to current joint angles (rest pose)
    // Cache per-joint PD parameters (constant per episode)
    this._maxTorques = this._actuatedJoints.map(j => j.maxTorque ?? 300)
    this._kps = this._actuatedJoints.map(j => j.kp ?? DEFAULT_KP)
    this._kds = this._actuatedJoints.map(j => j.kd ?? DEFAULT_KD)
    this._bodyAs = this._actuatedJoints.map(j => this.bodies[j.bodyA])
    this._bodyBs = this._actuatedJoints.map(j => this.bodies[j.bodyB])

    this._targetAngles = new Float32Array(this._actuatedJoints.length)
    for (let i = 0; i < this._actuatedJoints.length; i++) {
      const jDef = this._actuatedJoints[i]
      const bodyA = this.bodies[jDef.bodyA]
      const bodyB = this.bodies[jDef.bodyB]
      if (bodyA && bodyB) {
        this._targetAngles[i] = bodyB.rotation() - bodyA.rotation()
      }
    }

    const torso = this.bodies[def.forwardBody || 'torso']
    this._prevTorsoX = torso ? torso.translation().x : 0
    this._maxTorsoX = this._prevTorsoX

    return this._getObs()
  }

  /**
   * Step the environment.
   *
   * Peng-style PD target-angle control:
   *   actions[i] ∈ [-1, 1] → target angle within joint limits
   *   Manual PD: τ = kp·(target - θ) - kd·ω, applied via addTorque() each substep
   */
  step(actions) {
    const def = this.def
    // Normalize to array (scalar possible for single-joint creatures)
    const acts = typeof actions === 'number' ? [actions] : actions
    const clampedActions = acts.map(a => Math.max(-1, Math.min(1, a)))

    // Peng-style: actions are offsets from rest pose, not raw joint positions.
    // action=0 → rest angle (standing), ±1 → full deviation toward joint limits.
    for (let i = 0; i < this._actuatedJoints.length; i++) {
      const jDef = this._actuatedJoints[i]
      const lo = jDef.lowerLimit
      const hi = jDef.upperLimit
      const rest = jDef.restAngle ?? 0  // standing pose angle
      const a = clampedActions[i]
      // Asymmetric mapping: positive action → toward upper limit, negative → toward lower limit
      const target = a >= 0
        ? rest + a * (hi - rest)
        : rest + a * (rest - lo)
      this._targetAngles[i] = Math.max(lo, Math.min(hi, target))
    }

    // Run physics substeps, applying PD torques each substep for stability
    for (let sub = 0; sub < SUBSTEPS_PER_POLICY; sub++) {
      for (let i = 0; i < this._actuatedJoints.length; i++) {
        const bodyA = this._bodyAs[i]
        const bodyB = this._bodyBs[i]

        const currentAngle = bodyB.rotation() - bodyA.rotation()
        const currentAngVel = bodyB.angvel() - bodyA.angvel()

        // PD: τ = kp·(target - current) - kd·velocity
        let torque = this._kps[i] * (this._targetAngles[i] - currentAngle) - this._kds[i] * currentAngVel
        const maxT = this._maxTorques[i]
        torque = Math.max(-maxT, Math.min(maxT, torque))

        // Apply equal and opposite torques (Newton's third law)
        bodyB.addTorque(torque, true)
        bodyA.addTorque(-torque, true)
      }
      this.world.step()
    }

    this.stepCount++

    // Update foot contacts
    for (const [bodyId, handle] of Object.entries(this.footSensorHandles)) {
      this._footContacts[bodyId] = false
      this.world.narrowPhase.intersectionPairsWith(handle, () => {
        this._footContacts[bodyId] = true
      })
    }

    // Get torso state
    const forwardBodyId = def.forwardBody || 'torso'
    const torso = this.bodies[forwardBodyId]
    const torsoPos = torso.translation()
    const torsoRot = torso.rotation()

    // Forward velocity from position delta
    const agentDt = SUBSTEPS_PER_POLICY * PHYSICS_DT
    const forwardVel = (torsoPos.x - this._prevTorsoX) / agentDt
    this._prevTorsoX = torsoPos.x
    this._maxTorsoX = Math.max(this._maxTorsoX, torsoPos.x)

    // Health / early termination
    const torsoDef = def.bodies.find(b => b.id === forwardBodyId) || def.bodies[0]
    const minY = torsoDef.minY ?? 0.3
    const maxAngle = torsoDef.maxAngle ?? 0.6

    const groundAtTorso = this._getGroundHeight(torsoPos.x)
    const heightAboveGround = torsoPos.y - groundAtTorso

    const healthy = heightAboveGround >= minY && Math.abs(torsoRot) <= maxAngle
    const done = !healthy || this.stepCount >= this.maxSteps
    const timedOut = this.stepCount >= this.maxSteps

    // ── Reward ──
    // Linear forward velocity (always gives gradient) + alive bonus + control cost
    const r = def.defaultReward ?? {}
    let reward = 0

    if (healthy || timedOut) {
      // Forward velocity reward — the primary training signal.
      // Must dominate alive bonus so moving forward beats standing still.
      reward += (r.forwardVelWeight ?? 3.0) * forwardVel

      // Alive bonus — small tiebreaker, NOT the primary signal.
      // Too high creates "stand still" local minimum (standing still = max alive bonus).
      reward += (r.aliveBonusWeight ?? 0.1)

      // Control cost: penalize large target angle changes (smooth motion)
      const ctrlCost = clampedActions.reduce((s, a) => s + a * a, 0)
      reward -= (r.ctrlCostWeight ?? 0.001) * ctrlCost
    }

    // Termination penalty (default 0 — alive bonus already incentivizes survival)
    if (done && !timedOut) {
      reward -= (r.terminationPenalty ?? 0.0)
    }

    const obs = this._getObs()

    return {
      obs,
      reward,
      done,
      info: {
        forwardVel,
        healthy,
        stepCount: this.stepCount,
        distance: torsoPos.x,
        maxDistance: this._maxTorsoX,
        heightAboveGround,
      }
    }
  }

  _getGroundHeight(x) {
    if (!this.terrain) return 0
    // Simple linear search (terrain segments are roughly ordered by x)
    for (const seg of this.terrain.segments) {
      if (x >= seg.x1 && x <= seg.x2) {
        if (Math.abs(seg.x2 - seg.x1) < 0.001) return seg.y1
        const t = (x - seg.x1) / (seg.x2 - seg.x1)
        return seg.y1 + t * (seg.y2 - seg.y1)
      }
    }
    if (this.terrain.segments.length > 0) {
      if (x < this.terrain.segments[0].x1) return this.terrain.segments[0].y1
      return this.terrain.segments[this.terrain.segments.length - 1].y2
    }
    return 0
  }

  _getObs() {
    const def = this.def
    const forwardBodyId = def.forwardBody || 'torso'
    const torso = this.bodies[forwardBodyId]
    if (!torso) return new Array(this._obsSize).fill(0)

    const torsoPos = torso.translation()
    const torsoVel = torso.linvel()
    const torsoAngle = torso.rotation()
    const torsoAngVel = torso.angvel()

    // Ground-relative height
    const groundH = this._getGroundHeight(torsoPos.x)

    const obs = [
      torsoPos.y - groundH,   // height above ground (terrain-relative)
      torsoAngle,              // tilt
      torsoVel.x,              // forward velocity
      torsoVel.y,              // vertical velocity
      torsoAngVel,             // angular velocity
    ]

    // Joint angles and velocities
    for (const jointDef of def.joints) {
      const bodyA = this.bodies[jointDef.bodyA]
      const bodyB = this.bodies[jointDef.bodyB]
      if (!bodyA || !bodyB) {
        obs.push(0, 0)
        continue
      }
      obs.push(bodyB.rotation() - bodyA.rotation())
      obs.push(bodyB.angvel() - bodyA.angvel())
    }

    // Foot contacts
    for (const bodyDef of def.bodies) {
      if (bodyDef.isFootBody) {
        obs.push(this._footContacts[bodyDef.id] ? 1.0 : 0.0)
      }
    }

    // Terrain heightfield perception
    const heightfield = sampleHeightfield(
      this.terrain?.segments || [],
      torsoPos.x,
      torsoPos.y,
      TERRAIN_SAMPLES,
      LOOK_AHEAD,
      LOOK_BEHIND
    )
    for (let i = 0; i < TERRAIN_SAMPLES; i++) {
      obs.push(heightfield[i])
    }

    return obs
  }

  getRenderSnapshot() {
    const snapshot = {}
    for (const [id, rb] of Object.entries(this.bodies)) {
      snapshot[id] = {
        x: rb.translation().x,
        y: rb.translation().y,
        angle: rb.rotation(),
      }
    }
    snapshot._footContacts = { ...this._footContacts }
    snapshot._terrain = this.terrain
    snapshot._maxDistance = this._maxTorsoX
    return snapshot
  }

  get observationSize() { return this._obsSize }
  get actionSize() { return this._actSize }

  dispose() {
    if (this.world) {
      this.world.free()
      this.world = null
    }
  }
}
