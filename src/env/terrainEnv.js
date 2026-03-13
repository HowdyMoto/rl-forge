/**
 * TerrainRapierEnv
 *
 * A physics environment with:
 *   - PD controllers (agent outputs target joint angles, not raw torques)
 *   - Procedural terrain (gaps, ramps, stairs, bumps)
 *   - Terrain perception (heightfield sampling ahead of agent)
 *   - Peng-style reward (forward velocity + alive bonus + strict early termination)
 *   - Frame skip with PD tracking for smooth motion
 *   - Support for arbitrary user-built creature definitions
 *
 * Inspired by Peng et al. "Terrain-Adaptive Locomotion Skills Using Deep RL"
 */

import RAPIER from '@dimforge/rapier2d'
import { generateTerrain, sampleHeightfield } from './terrain.js'

// Physics runs at 120Hz internally, policy queries at 30Hz (4 substeps per policy step)
const PHYSICS_DT = 1 / 120
const POLICY_FREQ = 30         // Hz — how often the policy picks new target angles
const SUBSTEPS_PER_POLICY = Math.round((1 / POLICY_FREQ) / PHYSICS_DT)  // = 4

const GRAVITY = -9.81

// PD controller gains (can be overridden per-joint in the character def)
const DEFAULT_KP = 300    // proportional gain
const DEFAULT_KD = 30     // derivative gain

// Terrain perception
const TERRAIN_SAMPLES = 10   // number of height samples in observation
const LOOK_AHEAD = 3.0       // meters ahead
const LOOK_BEHIND = 0.5      // meters behind

// Ground contact collision group for early termination bodies
const TERMINATE_ON_CONTACT_GROUP = 0x0002

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
    this._jointOrder = []         // ordered joint IDs for action mapping

    // Terrain
    this.terrain = null
    this.terrainBodies = []

    // Count actuated joints for action space sizing
    this._actuatedJoints = characterDef.joints.filter(j => (j.maxTorque ?? 0) > 0)
    this._jointOrder = this._actuatedJoints.map(j => j.id)

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
    this.bodies = {}
    this.joints = {}
    this.footSensorHandles = {}
    this.terrainBodies = []

    // Generate terrain
    this.terrain = generateTerrain(this.terrainSeed, this.terrainLength, this.difficulty)

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

      // Set initial motor with damping
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
    // New terrain each episode with different seed
    this.terrainSeed = Math.floor(Math.random() * 100000)
    this._buildWorld()
    this.stepCount = 0
    this._footContacts = {}

    // Small random perturbation
    const def = this.def
    for (const bodyDef of def.bodies) {
      if (bodyDef.fixed) continue
      const rb = this.bodies[bodyDef.id]
      rb.setTranslation({
        x: bodyDef.spawnX + (Math.random() - 0.5) * 0.01,
        y: bodyDef.spawnY + (Math.random() - 0.5) * 0.01,
      }, true)
      rb.setRotation((bodyDef.spawnAngle ?? 0) + (Math.random() - 0.5) * 0.02, true)
      rb.setLinvel({ x: (Math.random() - 0.5) * 0.01, y: 0 }, true)
      rb.setAngvel((Math.random() - 0.5) * 0.01, true)
    }

    // Initialize PD targets to current joint angles (rest pose)
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
   * Actions are target joint angles in [-1, 1], mapped to [lowerLimit, upperLimit].
   */
  step(actions) {
    const def = this.def
    const clampedActions = Array.isArray(actions)
      ? actions.map(a => Math.max(-1, Math.min(1, a)))
      : [Math.max(-1, Math.min(1, actions))]

    // Map actions from [-1,1] to target joint angles within limits
    for (let i = 0; i < this._actuatedJoints.length; i++) {
      const jDef = this._actuatedJoints[i]
      const lower = jDef.lowerLimit
      const upper = jDef.upperLimit
      // Map [-1, 1] → [lower, upper]
      this._targetAngles[i] = lower + (clampedActions[i] + 1) * 0.5 * (upper - lower)
    }

    // Run physics substeps with PD controller
    let totalControlCost = 0
    for (let sub = 0; sub < SUBSTEPS_PER_POLICY; sub++) {
      // Apply PD torques to each actuated joint
      for (let i = 0; i < this._actuatedJoints.length; i++) {
        const jDef = this._actuatedJoints[i]
        const joint = this.joints[jDef.id]
        if (!joint) continue

        const bodyA = this.bodies[jDef.bodyA]
        const bodyB = this.bodies[jDef.bodyB]
        if (!bodyA || !bodyB) continue

        // Current joint state
        const currentAngle = bodyB.rotation() - bodyA.rotation()
        const currentAngVel = bodyB.angvel() - bodyA.angvel()

        // PD control: torque = kp * (target - current) - kd * velocity
        const kp = jDef.kp ?? DEFAULT_KP
        const kd = jDef.kd ?? DEFAULT_KD
        const targetAngle = this._targetAngles[i]

        let torque = kp * (targetAngle - currentAngle) - kd * currentAngVel

        // Clamp torque to joint limits
        const maxT = jDef.maxTorque ?? 200
        torque = Math.max(-maxT, Math.min(maxT, torque))

        totalControlCost += torque * torque / (maxT * maxT)

        // Apply torque via motor API — set target velocity proportional to error
        // with high stiffness to approximate direct torque
        const targetVel = torque / (kd > 0 ? kd : 30)
        joint.configureMotorVelocity(targetVel, Math.abs(torque) + 1.0)
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

    // Health / early termination checks
    const torsoDef = def.bodies.find(b => b.id === forwardBodyId) || def.bodies[0]
    const minY = torsoDef.minY ?? 0.3
    const maxAngle = torsoDef.maxAngle ?? 0.6

    // Check if torso is above ground (terrain-relative)
    const groundAtTorso = this._getGroundHeight(torsoPos.x)
    const heightAboveGround = torsoPos.y - groundAtTorso

    const healthy = heightAboveGround >= minY && Math.abs(torsoRot) <= maxAngle
    const done = !healthy || this.stepCount >= this.maxSteps
    const timedOut = this.stepCount >= this.maxSteps

    // Peng-style reward
    const r = def.defaultReward || {}
    let reward = 0
    if (healthy || timedOut) {
      // Forward velocity reward
      reward += (r.forwardVelWeight ?? 1.0) * forwardVel
      // Alive bonus
      reward += (r.aliveBonusWeight ?? 0.5)
      // Control cost (normalized by number of actuated joints)
      const ctrlWeight = r.ctrlCostWeight ?? 0.001
      reward -= ctrlWeight * (totalControlCost / Math.max(1, this._actuatedJoints.length * SUBSTEPS_PER_POLICY))
    }
    // Termination penalty (strict — encourages not falling)
    if (done && !timedOut) {
      reward -= (r.terminationPenalty ?? 50.0)
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
