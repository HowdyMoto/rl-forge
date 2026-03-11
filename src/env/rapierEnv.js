/**
 * RapierEnv
 *
 * A physics environment backed by Rapier2D (WASM).
 * Presents the same reset()/step() interface as CartPoleEnv so the
 * PPO training loop is environment-agnostic.
 *
 * IMPORTANT: Rapier's WASM module must be initialized before constructing
 * this class. The caller is responsible for:
 *
 *   import init from '@dimforge/rapier2d'
 *   await init()
 *   const env = new RapierEnv(characterDef)
 *
 * Rapier coordinate system: +x right, +y up (standard math, not screen).
 *
 * The simulation runs at a fixed timestep; we take multiple substeps per
 * agent step for stability (especially important with joints under torque).
 */

import RAPIER from '@dimforge/rapier2d'

const SIM_DT = 1 / 60          // physics timestep: 60 Hz
const SUBSTEPS = 2              // substeps per agent step
const AGENT_DT = SIM_DT * SUBSTEPS

// Gravity constant matching MuJoCo default
const GRAVITY = -9.81

export class RapierEnv {
  constructor(characterDef) {
    this.def = characterDef
    this.world = null
    this.bodies = {}       // id → RigidBody
    this.joints = {}       // id → ImpulseJoint
    this.groundHandle = null
    this.footSensorHandle = null
    this.stepCount = 0
    this.maxSteps = 1000
    this._prevTorsoX = 0
    this._footContact = false
  }

  // ─── Build the world from the character definition ────────────────────────

  _buildWorld() {
    if (this.world) this.world.free()

    this.world = new RAPIER.World({ x: 0.0, y: GRAVITY })
    this.bodies = {}
    this.joints = {}

    const def = this.def

    // Ground: static collider
    const groundBody = this.world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(0, def.ground.y - 0.05)
    )
    const groundCollider = RAPIER.ColliderDesc
      .cuboid(50.0, 0.05)
      .setFriction(def.ground.friction)
      .setRestitution(def.ground.restitution)
    this.world.createCollider(groundCollider, groundBody)
    this.groundHandle = groundBody.handle

    // Character bodies
    for (const bodyDef of def.bodies) {
      const rbDesc = RAPIER.RigidBodyDesc
        .dynamic()
        .setTranslation(bodyDef.spawnX, bodyDef.spawnY)
        .setRotation(bodyDef.spawnAngle)

      const rb = this.world.createRigidBody(rbDesc)

      let colliderDesc
      if (bodyDef.shape === 'box') {
        colliderDesc = RAPIER.ColliderDesc.cuboid(bodyDef.w / 2, bodyDef.h / 2)
      } else if (bodyDef.shape === 'capsule') {
        // Rapier capsule: halfHeight is half the cylinder part, radius is end cap
        const halfHeight = Math.max(0.001, (bodyDef.length - 2 * bodyDef.radius) / 2)
        colliderDesc = RAPIER.ColliderDesc.capsule(halfHeight, bodyDef.radius)
      } else {
        throw new Error(`Unknown shape: ${bodyDef.shape}`)
      }

      colliderDesc
        .setFriction(bodyDef.friction)
        .setRestitution(bodyDef.restitution)
        .setDensity(bodyDef.mass / this._bodyVolume(bodyDef))

      // Foot sensor: intersection event tells us ground contact
      if (bodyDef.isFootBody) {
        // Small sensor at bottom of shin
        const sensorDesc = RAPIER.ColliderDesc
          .cuboid(bodyDef.radius, 0.02)
          .setTranslation(0, -(bodyDef.length / 2))
          .setSensor(true)
          .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS)
        const sensorCollider = this.world.createCollider(sensorDesc, rb)
        this.footSensorHandle = sensorCollider.handle  // collider handle, not body handle
      }

      this.world.createCollider(colliderDesc, rb)
      this.bodies[bodyDef.id] = rb
    }

    // Joints (revolute = hinge in 2D)
    for (const jointDef of def.joints) {
      const bodyA = this.bodies[jointDef.bodyA]
      const bodyB = this.bodies[jointDef.bodyB]

      const [axA, ayA] = jointDef.anchorA
      const [axB, ayB] = jointDef.anchorB

      const jointData = RAPIER.JointData.revolute(
        { x: axA, y: ayA },
        { x: axB, y: ayB }
      )

      // Set angle limits
      jointData.limitsEnabled = true
      jointData.limits = [jointDef.lowerLimit, jointDef.upperLimit]

      const joint = this.world.createImpulseJoint(jointData, bodyA, bodyB, true)
      this.joints[jointDef.id] = joint
    }
  }

  _bodyVolume(bodyDef) {
    if (bodyDef.shape === 'box') return bodyDef.w * bodyDef.h * 0.1  // assume depth 0.1
    if (bodyDef.shape === 'capsule') {
      const r = bodyDef.radius
      return Math.PI * r * r * bodyDef.length  // cylinder approximation
    }
    return 1.0
  }

  // ─── Reset ────────────────────────────────────────────────────────────────

  reset() {
    const def = this.def

    // Rebuild world fresh each episode to avoid joint drift accumulation
    this._buildWorld()
    this.stepCount = 0
    this._footContact = false

    // Small random perturbation to break symmetry
    for (const bodyDef of def.bodies) {
      const rb = this.bodies[bodyDef.id]
      rb.setTranslation({
        x: bodyDef.spawnX + (Math.random() - 0.5) * 0.01,
        y: bodyDef.spawnY + (Math.random() - 0.5) * 0.01,
      }, true)
      rb.setRotation(bodyDef.spawnAngle + (Math.random() - 0.5) * 0.02, true)
      rb.setLinvel({ x: (Math.random() - 0.5) * 0.01, y: 0 }, true)
      rb.setAngvel((Math.random() - 0.5) * 0.01, true)
    }

    this._prevTorsoX = this.bodies[def.forwardBody].translation().x
    return this._getObs()
  }

  // ─── Step ─────────────────────────────────────────────────────────────────

  step(actions) {
    const def = this.def

    // Clamp actions to [-1, 1]
    const clampedActions = actions.map(a => Math.max(-1, Math.min(1, a)))

    // Apply joint torques
    def.joints.forEach((jointDef, i) => {
      const joint = this.joints[jointDef.id]
      const torque = clampedActions[i] * jointDef.maxTorque
      // Apply equal and opposite torques to the two connected bodies
      const bodyA = this.bodies[jointDef.bodyA]
      const bodyB = this.bodies[jointDef.bodyB]
      bodyA.applyTorqueImpulse(-torque * AGENT_DT, true)
      bodyB.applyTorqueImpulse( torque * AGENT_DT, true)
    })

    // Substep physics
    for (let i = 0; i < SUBSTEPS; i++) {
      this.world.step()
    }
    this.stepCount++

    // Update foot contact via narrow phase intersection queries
    this._footContact = false
    this.world.narrowPhase.intersectionPairsWith(this.footSensorHandle, () => {
      this._footContact = true
    })

    const obs = this._getObs()
    const torso = this.bodies[def.forwardBody]
    const torsoPos = torso.translation()
    const torsoRot = torso.rotation()

    // Compute forward velocity from position delta
    const forwardVel = (torsoPos.x - this._prevTorsoX) / AGENT_DT
    this._prevTorsoX = torsoPos.x

    // Health check
    const torsoDef = def.bodies.find(b => b.id === def.forwardBody)
    const healthy = torsoPos.y >= torsoDef.minY && Math.abs(torsoRot) <= torsoDef.maxAngle

    const done = !healthy || this.stepCount >= this.maxSteps
    const timedOut = this.stepCount >= this.maxSteps

    // Reward (matches MuJoCo Hopper default)
    const r = def.defaultReward
    let reward = 0
    if (healthy || timedOut) {
      reward += forwardVel * r.forwardVelWeight
      reward += r.aliveBonusWeight
      reward -= r.ctrlCostWeight * clampedActions.reduce((s, a) => s + a * a, 0)
    }
    if (done && !timedOut) {
      reward -= r.terminationPenalty
    }

    return { obs, reward, done, info: { forwardVel, healthy, stepCount: this.stepCount } }
  }

  // ─── Observation ──────────────────────────────────────────────────────────

  _getObs() {
    const def = this.def
    const torso = this.bodies['torso']
    const torsoPos = torso.translation()
    const torsoVel = torso.linvel()
    const torsoAngle = torso.rotation()
    const torsoAngVel = torso.angvel()

    const obs = [
      torsoPos.y,      // height — policy needs to know if falling
      torsoAngle,      // tilt
      torsoVel.x,      // forward velocity — key locomotion signal
      torsoVel.y,      // vertical velocity
      torsoAngVel,     // angular velocity
    ]

    // Joint angles and velocities (computed from connected bodies)
    for (const jointDef of def.joints) {
      const bodyA = this.bodies[jointDef.bodyA]
      const bodyB = this.bodies[jointDef.bodyB]
      // Revolute joint angle = relative rotation between bodies
      const jointAngle = bodyB.rotation() - bodyA.rotation()
      // Relative angular velocity
      const jointAngVel = bodyB.angvel() - bodyA.angvel()
      obs.push(jointAngle)
      obs.push(jointAngVel)
    }

    // Ground contact
    obs.push(this._footContact ? 1.0 : 0.0)

    return obs  // length = 5 + 2*numJoints + 1 = 10 for hopper
  }

  // ─── Snapshot for rendering ───────────────────────────────────────────────

  /**
   * Returns a plain-JS snapshot of all body transforms.
   * Safe to postMessage across worker boundary (no Rapier objects).
   */
  getRenderSnapshot() {
    const snapshot = {}
    for (const [id, rb] of Object.entries(this.bodies)) {
      snapshot[id] = {
        x: rb.translation().x,
        y: rb.translation().y,
        angle: rb.rotation(),
      }
    }
    snapshot._footContact = this._footContact
    return snapshot
  }

  get observationSize() { return this.def.obsSize }
  get actionSize() { return this.def.actionSize }

  dispose() {
    if (this.world) {
      this.world.free()
      this.world = null
    }
  }
}
