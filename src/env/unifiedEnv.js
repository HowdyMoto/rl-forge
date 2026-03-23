/**
 * UnifiedRapierEnv
 *
 * Single environment class for all Rapier2D-based physics environments.
 * Replaces the previous RapierEnv + TerrainRapierEnv split.
 *
 * Supports:
 *   - Revolute joints (hinges) and prismatic joints (sliders)
 *   - Control modes: 'velocity' (motor velocity), 'pd' (PD target angles), 'force' (direct force)
 *   - Optional procedural terrain with heightfield perception
 *   - Configurable reward via defaultReward in the character definition
 *   - Any body topology (hopper, walker, cartpole, custom creatures)
 *
 * IMPORTANT: Rapier WASM must be initialized before constructing this class.
 */

import RAPIER from '@dimforge/rapier2d'
import { computeDerivedFields, TERRAIN_OBSERVATION_SAMPLES } from '../formats/bodyDef.js'
import { generateTerrain, sampleHeightfield } from './terrain.js'

// ─── Physics constants ─────────────────────────────────────────────────────

// Default physics rates (can be overridden in options)
const DEFAULT_PHYSICS_HZ = 240
const DEFAULT_POLICY_HZ = 30
const DEFAULT_GRAVITY = -9.81

// Default PD gains (used when joint specifies controlMode:'pd' but no kp/kd)
const DEFAULT_KP = 300
const DEFAULT_KD = 30

export class UnifiedRapierEnv {
  /**
   * @param {Object} characterDef - Body definition (from bodyDef.js or characters/*.js)
   * @param {Object} [opts]
   * @param {string} [opts.controlMode] - 'velocity' | 'pd' | 'force' (default: auto-detect from joint defs)
   * @param {number} [opts.physicsHz=240] - Physics simulation frequency
   * @param {number} [opts.policyHz=30] - Policy decision frequency
   * @param {number} [opts.gravity] - Gravity (negative = down, default -9.81)
   * @param {boolean} [opts.terrain=false] - Enable procedural terrain
   * @param {number} [opts.terrainSeed] - Terrain generation seed
   * @param {number} [opts.difficulty=0.3] - Terrain difficulty 0.0-1.0
   * @param {number} [opts.terrainLength=50] - Terrain length in meters
   * @param {number} [opts.maxSteps=1000] - Max steps per episode
   */
  constructor(characterDef, opts = {}) {
    this.def = characterDef

    // Physics timing
    this.physicsHz = opts.physicsHz ?? DEFAULT_PHYSICS_HZ
    this.policyHz = opts.policyHz ?? DEFAULT_POLICY_HZ
    this.physicsDt = 1 / this.physicsHz
    this.substeps = Math.round(this.physicsHz / this.policyHz)
    this.agentDt = this.substeps * this.physicsDt
    this.gravity = opts.gravity ?? DEFAULT_GRAVITY
    this.maxSteps = opts.maxSteps ?? 1000

    // Control mode: auto-detect from joint definitions
    this.controlMode = opts.controlMode ?? this._detectControlMode()

    // Terrain options
    this.useTerrain = opts.terrain ?? false
    this.terrainSeed = opts.terrainSeed ?? Math.floor(Math.random() * 100000)
    this.difficulty = opts.difficulty ?? 0.3
    this.terrainLength = opts.terrainLength ?? 50

    // Derived fields
    const derived = computeDerivedFields(characterDef, { terrain: this.useTerrain })
    this._obsSize = derived.obsSize
    this._actSize = derived.actionSize
    this._forwardBody = derived.forwardBody

    // Runtime state
    this.world = null
    this.bodies = {}
    this.joints = {}
    this.footSensorHandles = {}
    this._footContacts = {}
    this.stepCount = 0
    this._prevTorsoX = 0
    this._maxTorsoX = 0
    this._episodeCount = 0

    // Terrain
    this.terrain = null
    this.terrainBodies = []

    // PD controller caches (populated on reset)
    this._pdCache = null
    this._targetAngles = null
  }

  /**
   * Auto-detect control mode from joint definitions.
   * If any joint has kp/kd, use 'pd'. Otherwise 'velocity'.
   */
  _detectControlMode() {
    const joints = this.def.joints || []
    // Check for prismatic joints with force control (CartPole)
    const hasPrismatic = joints.some(j => j.type === 'prismatic')
    if (hasPrismatic) return 'velocity'  // prismatic joints use velocity motors

    // Check for PD gains
    const hasPD = joints.some(j => j.kp !== undefined || j.kd !== undefined)
    if (hasPD) return 'pd'

    return 'velocity'
  }

  // ─── World Construction ────────────────────────────────────────────────────

  _buildWorld() {
    if (this.world) this.world.free()

    this.world = new RAPIER.World({ x: 0.0, y: this.gravity })
    this.world.timestep = this.physicsDt
    this.bodies = {}
    this.joints = {}
    this.footSensorHandles = {}
    this.terrainBodies = []

    const def = this.def

    // Ground or terrain
    if (this.useTerrain) {
      this._buildTerrain()
    } else {
      this._buildGround()
    }

    // Character bodies
    for (const bodyDef of def.bodies) {
      // Compute damping for this body from connected joints
      let bodyLinDamping = 0
      let bodyAngDamping = 0
      if (!bodyDef.fixed) {
        for (const jDef of def.joints) {
          if (jDef.bodyB === bodyDef.id && (jDef.damping ?? 0) > 0) {
            bodyLinDamping = Math.max(bodyLinDamping, jDef.damping)
            bodyAngDamping = Math.max(bodyAngDamping, jDef.damping)
          }
        }
      }

      const rbDesc = bodyDef.fixed
        ? RAPIER.RigidBodyDesc.fixed()
        : RAPIER.RigidBodyDesc.dynamic()
            .setCanSleep(false)
            .setLinearDamping(bodyLinDamping)
            .setAngularDamping(bodyAngDamping)
      rbDesc.setTranslation(bodyDef.spawnX ?? 0, bodyDef.spawnY ?? 1.0)
      rbDesc.setRotation(bodyDef.spawnAngle ?? 0)

      const rb = this.world.createRigidBody(rbDesc)

      // Collider shape
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
        .setFriction(bodyDef.friction ?? 0.3)
        .setRestitution(bodyDef.restitution ?? 0.15)
        .setMass(bodyDef.mass ?? 1.0)

      // Foot sensor
      if (bodyDef.isFootBody) {
        const sensorRadius = bodyDef.radius ?? (bodyDef.w ? bodyDef.w / 2 : 0.04)
        const sensorY = bodyDef.shape === 'capsule'
          ? -(bodyDef.length / 2)
          : -(bodyDef.h ? bodyDef.h / 2 : sensorRadius)
        const sensorDesc = RAPIER.ColliderDesc
          .cuboid(sensorRadius, 0.02)
          .setTranslation(0, sensorY)
          .setSensor(true)
          .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS)
        const sensorCollider = this.world.createCollider(sensorDesc, rb)
        this.footSensorHandles[bodyDef.id] = sensorCollider.handle
      }

      // Skip collider for pure pivot bodies (e.g., prismatic anchor)
      if (!bodyDef.noCollider) {
        this.world.createCollider(colliderDesc, rb)
      }
      this.bodies[bodyDef.id] = rb
    }

    // Joints
    for (const jointDef of def.joints) {
      const bodyA = this.bodies[jointDef.bodyA]
      const bodyB = this.bodies[jointDef.bodyB]
      if (!bodyA || !bodyB) continue

      const [axA, ayA] = jointDef.anchorA || [0, 0]
      const [axB, ayB] = jointDef.anchorB || [0, 0]

      let joint
      if (jointDef.type === 'prismatic') {
        // Prismatic (slider) joint
        const axis = jointDef.axis || [1, 0]
        const lo = jointDef.lowerLimit ?? -2.4
        const hi = jointDef.upperLimit ?? 2.4
        const jointData = RAPIER.JointData.prismatic(
          { x: axA, y: ayA },
          { x: axB, y: ayB },
          { x: axis[0], y: axis[1] }
        )
        jointData.limitsEnabled = true
        jointData.limits = [lo, hi]
        joint = this.world.createImpulseJoint(jointData, bodyA, bodyB, true)
      } else {
        // Revolute (hinge) joint — default
        const jointData = RAPIER.JointData.revolute(
          { x: axA, y: ayA },
          { x: axB, y: ayB }
        )
        joint = this.world.createImpulseJoint(jointData, bodyA, bodyB, true)
        // Set limits AFTER creation (JointData.limits doesn't propagate reliably)
        const lo = jointDef.lowerLimit ?? -Math.PI
        const hi = jointDef.upperLimit ?? Math.PI
        if (lo > -6.28 || hi < 6.28) {  // only enable if not full-rotation
          joint.setLimits(lo, hi)
        }
      }

      // Disable collisions between jointed bodies
      joint.setContactsEnabled(false)

      // Damping is applied via Rapier's built-in body linearDamping/angularDamping

      this.joints[jointDef.id] = joint
    }
  }

  _buildGround() {
    const def = this.def
    const groundBody = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(0, (def.ground?.y ?? 0) - 0.05)
    )
    const groundCollider = RAPIER.ColliderDesc
      .cuboid(50.0, 0.05)
      .setFriction(def.ground?.friction ?? 0.8)
      .setRestitution(def.ground?.restitution ?? 0.2)
    this.world.createCollider(groundCollider, groundBody)
  }

  _buildTerrain() {
    this.terrain = generateTerrain(
      this.terrainSeed,
      this.terrainLength,
      this._effectiveDifficulty ?? this.difficulty
    )
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

  // ─── Reset ──────────────────────────────────────────────────────────────────

  reset() {
    this._episodeCount++
    const def = this.def

    // Terrain curriculum
    if (this.useTerrain) {
      if (this._episodeCount > 50) {
        this.terrainSeed = Math.floor(Math.random() * 100000)
      }
      this._effectiveDifficulty = Math.min(this.difficulty, this._episodeCount / 100)
    }

    this._buildWorld()
    this.stepCount = 0
    this._footContacts = {}

    // Random perturbation
    for (const bodyDef of def.bodies) {
      if (bodyDef.fixed) continue
      const rb = this.bodies[bodyDef.id]
      rb.setTranslation({
        x: (bodyDef.spawnX ?? 0) + (Math.random() - 0.5) * 0.1,
        y: (bodyDef.spawnY ?? 1.0) + (Math.random() - 0.5) * 0.05,
      }, true)
      rb.setRotation((bodyDef.spawnAngle ?? 0) + (Math.random() - 0.5) * 0.3, true)
      rb.setLinvel({ x: (Math.random() - 0.5) * 0.5, y: (Math.random() - 0.5) * 0.3 }, true)
      rb.setAngvel((Math.random() - 0.5) * 0.5, true)
    }

    // Initialize PD controller cache
    if (this.controlMode === 'pd') {
      this._initPDCache()
    }

    const torso = this.bodies[this._forwardBody]
    this._prevTorsoX = torso ? torso.translation().x : 0
    this._maxTorsoX = this._prevTorsoX

    return this._getObs()
  }

  _initPDCache() {
    const def = this.def
    const actuated = def.joints.filter(j => (j.maxTorque ?? 0) > 0)
    this._pdCache = {
      joints: actuated,
      maxTorques: actuated.map(j => j.maxTorque ?? 300),
      kps: actuated.map(j => j.kp ?? DEFAULT_KP),
      kds: actuated.map(j => j.kd ?? DEFAULT_KD),
      bodyAs: actuated.map(j => this.bodies[j.bodyA]),
      bodyBs: actuated.map(j => this.bodies[j.bodyB]),
    }
    this._targetAngles = new Float32Array(actuated.length)
    for (let i = 0; i < actuated.length; i++) {
      const bA = this._pdCache.bodyAs[i]
      const bB = this._pdCache.bodyBs[i]
      if (bA && bB) this._targetAngles[i] = bB.rotation() - bA.rotation()
    }
  }

  // ─── Step ───────────────────────────────────────────────────────────────────

  step(actions) {
    const def = this.def
    const acts = typeof actions === 'number' ? [actions] : actions
    const clampedActions = acts.map(a => Math.max(-1, Math.min(1, a)))

    if (this.controlMode === 'pd') {
      this._stepPD(clampedActions)
    } else {
      this._stepVelocity(clampedActions)
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
    const torso = this.bodies[this._forwardBody]
    const torsoPos = torso.translation()
    const torsoRot = torso.rotation()

    // Forward velocity
    const forwardVel = (torsoPos.x - this._prevTorsoX) / this.agentDt
    this._prevTorsoX = torsoPos.x
    this._maxTorsoX = Math.max(this._maxTorsoX, torsoPos.x)

    // Health check & termination
    const { healthy, done, timedOut } = this._checkTermination(torsoPos, torsoRot)

    // Reward
    const reward = this._computeReward(forwardVel, clampedActions, healthy, done, timedOut, torsoPos, torsoRot)

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
      }
    }
  }

  _stepVelocity(actions) {
    const def = this.def
    def.joints.forEach((jointDef, i) => {
      const joint = this.joints[jointDef.id]
      if ((jointDef.maxTorque ?? 0) <= 0) return
      const targetVel = actions[i] * (jointDef.maxVelocity ?? 8.0)
      joint.configureMotorVelocity(targetVel, jointDef.maxTorque)
    })

    for (let i = 0; i < this.substeps; i++) {
      this.world.step()
    }
  }

  _stepPD(actions) {
    const pd = this._pdCache
    if (!pd) return

    // Map actions to target angles
    for (let i = 0; i < pd.joints.length; i++) {
      const jDef = pd.joints[i]
      const lo = jDef.lowerLimit
      const hi = jDef.upperLimit
      const rest = jDef.restAngle ?? 0
      const a = actions[i]
      const target = a >= 0
        ? rest + a * (hi - rest)
        : rest + a * (rest - lo)
      this._targetAngles[i] = Math.max(lo, Math.min(hi, target))
    }

    // PD control each substep
    for (let sub = 0; sub < this.substeps; sub++) {
      for (let i = 0; i < pd.joints.length; i++) {
        const bA = pd.bodyAs[i]
        const bB = pd.bodyBs[i]
        const currentAngle = bB.rotation() - bA.rotation()
        const currentAngVel = bB.angvel() - bA.angvel()

        let torque = pd.kps[i] * (this._targetAngles[i] - currentAngle) - pd.kds[i] * currentAngVel
        torque = Math.max(-pd.maxTorques[i], Math.min(pd.maxTorques[i], torque))

        bB.addTorque(torque, true)
        bA.addTorque(-torque, true)
      }
      this.world.step()
    }
  }

  _checkTermination(torsoPos, torsoRot) {
    const def = this.def
    const r = def.defaultReward ?? {}
    const torsoDef = def.bodies.find(b => b.id === this._forwardBody) || def.bodies[0]
    const timedOut = this.stepCount >= this.maxSteps

    // Standard health check (height + angle)
    const minY = torsoDef?.minY ?? 0.3
    const maxAngle = torsoDef?.maxAngle ?? 0.6

    let height = torsoPos.y
    if (this.useTerrain) {
      height = torsoPos.y - this._getGroundHeight(torsoPos.x)
    }

    let healthy = height >= minY && Math.abs(torsoRot) <= maxAngle

    // CartPole-specific: check cart position and pole angle limits
    if (r.cartPositionLimit !== undefined) {
      const cartX = torsoPos.x
      if (Math.abs(cartX) > r.cartPositionLimit) healthy = false
    }
    if (r.poleAngleLimit !== undefined) {
      // Find the pole body's angle relative to the cart
      const poleBody = def.bodies.find(b => b.id !== this._forwardBody && !b.fixed)
      if (poleBody) {
        const poleRb = this.bodies[poleBody.id]
        const cartRb = this.bodies[this._forwardBody]
        if (poleRb && cartRb) {
          const poleAngle = poleRb.rotation() - cartRb.rotation()
          if (Math.abs(poleAngle) > r.poleAngleLimit) healthy = false
        }
      }
    }

    const done = !healthy || timedOut
    return { healthy, done, timedOut }
  }

  _computeReward(forwardVel, actions, healthy, done, timedOut, torsoPos, torsoRot) {
    const r = this.def.defaultReward ?? {}
    let reward = 0

    if (healthy || timedOut) {
      // Forward velocity reward
      reward += (r.forwardVelWeight ?? 1.0) * forwardVel
      // Alive bonus
      reward += (r.aliveBonusWeight ?? 1.0)
      // Control cost
      reward -= (r.ctrlCostWeight ?? 0.001) * actions.reduce((s, a) => s + a * a, 0)
      // Tip height reward (acrobot swing-up)
      if (r.tipHeightWeight) {
        const torso = this.bodies[this._forwardBody]
        reward += (torso.translation().y - (r.anchorY ?? 0)) * r.tipHeightWeight
      }
      // CartPole angle/position penalties
      if (r.anglePenaltyWeight !== undefined) {
        const poleBody = this.def.bodies.find(b => b.id !== this._forwardBody && !b.fixed)
        if (poleBody) {
          const poleRb = this.bodies[poleBody.id]
          const cartRb = this.bodies[this._forwardBody]
          if (poleRb && cartRb) {
            const poleAngle = poleRb.rotation() - cartRb.rotation()
            const thetaMax = r.poleAngleLimit ?? (24 * Math.PI / 180)
            reward -= r.anglePenaltyWeight * (poleAngle * poleAngle) / (thetaMax * thetaMax)
          }
        }
      }
      if (r.positionPenaltyWeight !== undefined) {
        const posLimit = r.cartPositionLimit ?? 2.4
        reward -= r.positionPenaltyWeight * (torsoPos.x * torsoPos.x) / (posLimit * posLimit)
      }
    }

    // Termination penalty
    if (done && !timedOut) {
      reward -= (r.terminationPenalty ?? 0)
    }

    return reward
  }

  // ─── Observation ────────────────────────────────────────────────────────────

  _getObs() {
    const def = this.def
    const torso = this.bodies[this._forwardBody]
    if (!torso) return new Array(this._obsSize).fill(0)

    const torsoPos = torso.translation()
    const torsoVel = torso.linvel()
    const torsoAngle = torso.rotation()
    const torsoAngVel = torso.angvel()

    // Height: terrain-relative or absolute
    let height = torsoPos.y
    if (this.useTerrain) {
      height = torsoPos.y - this._getGroundHeight(torsoPos.x)
    }

    const obs = [
      height,
      torsoAngle,
      torsoVel.x,
      torsoVel.y,
      torsoAngVel,
    ]

    // Joint angles and velocities
    for (const jointDef of def.joints) {
      const bodyA = this.bodies[jointDef.bodyA]
      const bodyB = this.bodies[jointDef.bodyB]
      if (!bodyA || !bodyB) {
        obs.push(0, 0)
        continue
      }

      if (jointDef.type === 'prismatic') {
        // For prismatic joints: position along axis and velocity along axis
        // Compute from body positions since Rapier doesn't expose joint.translation()
        const axis = jointDef.axis || [1, 0]
        const posA = bodyA.translation()
        const posB = bodyB.translation()
        const relX = posB.x - posA.x
        const relY = posB.y - posA.y
        const axialPos = relX * axis[0] + relY * axis[1]
        // Velocity along axis
        const velA = bodyA.linvel()
        const velB = bodyB.linvel()
        const relVelX = velB.x - velA.x
        const relVelY = velB.y - velA.y
        const axialVel = relVelX * axis[0] + relVelY * axis[1]
        obs.push(axialPos)
        obs.push(axialVel)
      } else {
        // Revolute: relative angle and angular velocity
        obs.push(bodyB.rotation() - bodyA.rotation())
        obs.push(bodyB.angvel() - bodyA.angvel())
      }
    }

    // Foot contacts
    for (const bodyDef of def.bodies) {
      if (bodyDef.isFootBody) {
        obs.push(this._footContacts[bodyDef.id] ? 1.0 : 0.0)
      }
    }

    // Terrain heightfield perception
    if (this.useTerrain) {
      const heightfield = sampleHeightfield(
        this.terrain?.segments || [],
        torsoPos.x,
        torsoPos.y,
        TERRAIN_OBSERVATION_SAMPLES,
        8.0,  // look ahead
        1.0   // look behind
      )
      for (let i = 0; i < TERRAIN_OBSERVATION_SAMPLES; i++) {
        obs.push(heightfield[i])
      }
    }

    return obs
  }

  _getGroundHeight(x) {
    if (!this.terrain) return 0
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

  // ─── Render Snapshot ────────────────────────────────────────────────────────

  getRenderSnapshot() {
    const snapshot = {
      _charDef: this.def,
    }
    for (const [id, rb] of Object.entries(this.bodies)) {
      snapshot[id] = {
        x: rb.translation().x,
        y: rb.translation().y,
        angle: rb.rotation(),
      }
    }
    // Foot contacts with positions
    // Find the contact point by sampling the collider's lowest point
    const footContacts = {}
    const groundY = this.useTerrain ? null : (this.def.ground?.y ?? 0)
    for (const bodyDef of this.def.bodies) {
      if (!bodyDef.isFootBody) continue
      const inContact = this._footContacts[bodyDef.id] || false
      const rb = this.bodies[bodyDef.id]
      if (!rb) { footContacts[bodyDef.id] = false; continue }
      if (inContact) {
        const pos = rb.translation()
        const rot = rb.rotation()
        const cos = Math.cos(rot)
        const sin = Math.sin(rot)

        // Sample corners/extremes of the shape in local space,
        // transform to world, find the lowest point
        const localPoints = []
        if (bodyDef.shape === 'capsule') {
          const halfLen = (bodyDef.length || 0.3) / 2
          const r = bodyDef.radius || 0.04
          // Top and bottom cap centers, plus side extremes
          localPoints.push([0, -halfLen], [0, halfLen], [-r, -halfLen], [r, -halfLen], [-r, halfLen], [r, halfLen])
        } else if (bodyDef.shape === 'box') {
          const hw = (bodyDef.w || 0.1) / 2
          const hh = (bodyDef.h || 0.1) / 2
          localPoints.push([-hw, -hh], [hw, -hh], [-hw, hh], [hw, hh])
        } else if (bodyDef.shape === 'ball') {
          const r = bodyDef.radius || 0.05
          localPoints.push([0, -r], [0, r], [-r, 0], [r, 0])
        }

        let lowestY = Infinity
        let lowestX = pos.x
        for (const [lx, ly] of localPoints) {
          const wx = pos.x + lx * cos - ly * sin
          const wy = pos.y + lx * sin + ly * cos
          if (wy < lowestY) {
            lowestY = wy
            lowestX = wx
          }
        }

        // Place contact at ground level directly below the lowest point
        const contactY = groundY !== null ? groundY : lowestY
        footContacts[bodyDef.id] = { x: lowestX, y: contactY }
      } else {
        footContacts[bodyDef.id] = false
      }
    }
    snapshot._footContacts = footContacts
    if (this.useTerrain && this.terrain) {
      snapshot._terrain = this.terrain
    }
    snapshot._maxDistance = this._maxTorsoX

    // Joint debug info (always available for physics lab)
    const debugJoints = {}
    for (const jDef of this.def.joints) {
      const bodyA = this.bodies[jDef.bodyA]
      const bodyB = this.bodies[jDef.bodyB]
      if (!bodyA || !bodyB) continue

      if (jDef.type === 'prismatic') {
        // Compute prismatic position from body positions
        const axis = jDef.axis || [1, 0]
        const posA = bodyA.translation()
        const posB = bodyB.translation()
        const axialPos = (posB.x - posA.x) * axis[0] + (posB.y - posA.y) * axis[1]
        debugJoints[jDef.id] = {
          type: 'prismatic',
          translation: axialPos,
          lower: jDef.lowerLimit,
          upper: jDef.upperLimit,
          maxForce: jDef.maxTorque ?? 10,
          axis: jDef.axis || [1, 0],
          bodyA: jDef.bodyA,
        }
      } else {
        const angle = bodyB.rotation() - bodyA.rotation()
        const angVel = bodyB.angvel() - bodyA.angvel()
        debugJoints[jDef.id] = {
          type: 'revolute',
          angle,
          angVel,
          lower: jDef.lowerLimit,
          upper: jDef.upperLimit,
          maxTorque: jDef.maxTorque ?? 0,
          kp: jDef.kp,
          kd: jDef.kd,
          anchorA: jDef.anchorA,
          bodyA: jDef.bodyA,
        }
      }
    }
    snapshot._joints = debugJoints
    snapshot._bodyLabels = this.def.bodies.map(b => ({
      id: b.id,
      mass: b.mass,
      isFootBody: b.isFootBody || false,
      shape: b.shape,
    }))

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
