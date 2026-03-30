/**
 * BodyDef: Internal body definition format for RL-Forge.
 *
 * This is the canonical in-memory representation of a 2D articulated body.
 * All format converters (MJCF, URDF, legacy JSON) target this format.
 * Environments consume BodyDefs to build Rapier physics worlds.
 *
 * Derived fields (obsSize, actionSize) are computed automatically from
 * body topology via computeDerivedFields(), never stored manually.
 */

// Number of terrain heightfield samples added to observation in terrain mode
export const TERRAIN_OBSERVATION_SAMPLES = 40

// Supported joint types
export const JOINT_TYPES = ['revolute', 'prismatic']

/**
 * Validate a BodyDef object. Returns { valid: boolean, errors: string[] }.
 */
export function validateBodyDef(def) {
  const errors = []

  if (!def || typeof def !== 'object') {
    return { valid: false, errors: ['BodyDef must be an object'] }
  }

  // Name
  if (!def.name || typeof def.name !== 'string') {
    errors.push('name: must be a non-empty string')
  }

  // Bodies
  if (!Array.isArray(def.bodies) || def.bodies.length === 0) {
    errors.push('bodies: must be a non-empty array')
  } else {
    const bodyIds = new Set()
    for (let i = 0; i < def.bodies.length; i++) {
      const b = def.bodies[i]
      const prefix = `bodies[${i}]`

      if (!b.id || typeof b.id !== 'string') {
        errors.push(`${prefix}.id: must be a non-empty string`)
      } else if (bodyIds.has(b.id)) {
        errors.push(`${prefix}.id: duplicate body id '${b.id}'`)
      } else {
        bodyIds.add(b.id)
      }

      if (!['box', 'capsule', 'ball'].includes(b.shape)) {
        errors.push(`${prefix}.shape: must be 'box', 'capsule', or 'ball'`)
      }

      if (b.shape === 'box') {
        if (typeof b.w !== 'number' || b.w <= 0) errors.push(`${prefix}.w: must be positive number`)
        if (typeof b.h !== 'number' || b.h <= 0) errors.push(`${prefix}.h: must be positive number`)
      } else if (b.shape === 'capsule') {
        if (typeof b.radius !== 'number' || b.radius <= 0) errors.push(`${prefix}.radius: must be positive number`)
        if (typeof b.length !== 'number' || b.length <= 0) errors.push(`${prefix}.length: must be positive number`)
      } else if (b.shape === 'ball') {
        if (typeof b.radius !== 'number' || b.radius <= 0) errors.push(`${prefix}.radius: must be positive number`)
      }

      if (typeof b.mass !== 'number' || b.mass <= 0) {
        errors.push(`${prefix}.mass: must be positive number`)
      }
      if (typeof b.friction !== 'number' || b.friction < 0) {
        errors.push(`${prefix}.friction: must be non-negative number`)
      }
      if (typeof b.restitution !== 'number' || b.restitution < 0) {
        errors.push(`${prefix}.restitution: must be non-negative number`)
      }
      if (typeof b.spawnX !== 'number') errors.push(`${prefix}.spawnX: must be a number`)
      if (typeof b.spawnY !== 'number') errors.push(`${prefix}.spawnY: must be a number`)
    }

    // Joints reference validation
    if (Array.isArray(def.joints)) {
      for (let i = 0; i < def.joints.length; i++) {
        const j = def.joints[i]
        const prefix = `joints[${i}]`

        if (!j.id || typeof j.id !== 'string') {
          errors.push(`${prefix}.id: must be a non-empty string`)
        }

        if (!bodyIds.has(j.bodyA)) {
          errors.push(`${prefix}.bodyA: references unknown body '${j.bodyA}'`)
        }
        if (!bodyIds.has(j.bodyB)) {
          errors.push(`${prefix}.bodyB: references unknown body '${j.bodyB}'`)
        }

        if (!Array.isArray(j.anchorA) || j.anchorA.length !== 2) {
          errors.push(`${prefix}.anchorA: must be [x, y] array`)
        }
        if (!Array.isArray(j.anchorB) || j.anchorB.length !== 2) {
          errors.push(`${prefix}.anchorB: must be [x, y] array`)
        }

        if (typeof j.lowerLimit !== 'number') errors.push(`${prefix}.lowerLimit: must be a number`)
        if (typeof j.upperLimit !== 'number') errors.push(`${prefix}.upperLimit: must be a number`)
        if (typeof j.lowerLimit === 'number' && typeof j.upperLimit === 'number' && j.lowerLimit > j.upperLimit) {
          errors.push(`${prefix}: lowerLimit (${j.lowerLimit}) > upperLimit (${j.upperLimit})`)
        }

        if (typeof j.maxTorque !== 'number' || j.maxTorque < 0) {
          errors.push(`${prefix}.maxTorque: must be non-negative number`)
        }
      }
    }
  }

  // Joints array
  if (!Array.isArray(def.joints)) {
    errors.push('joints: must be an array')
  }

  return { valid: errors.length === 0, errors }
}

/**
 * Compute derived fields from body topology.
 *
 * For RapierEnv (velocity control):
 *   obsSize = 5 (torso state) + 2 * numJoints + numFootBodies
 *   actionSize = numActuatedJoints (joints with maxTorque > 0)
 *
 * For TerrainEnv (PD control):
 *   obsSize = 5 + 2 * numJoints + numFootBodies + TERRAIN_OBSERVATION_SAMPLES
 *   actionSize = numActuatedJoints
 *
 * @param {object} def - BodyDef object
 * @param {object} options
 * @param {boolean} options.terrain - if true, include terrain heightfield in obs
 * @returns {object} { obsSize, actionSize, numFootBodies, numJoints, forwardBody }
 */
export function computeDerivedFields(def, options = {}) {
  const numJoints = def.joints?.length ?? 0
  const numFootBodies = def.bodies?.filter(b => b.isFootBody).length ?? 0
  const terrain = options.terrain ?? false

  const extraObsSize = def.extraObsSize ?? 0
  // sinCosAngles: torso angle becomes [sin,cos] (+1), each revolute joint becomes [sin,cos,ω] (+1 each)
  const useSinCos = def.sinCosAngles ?? false
  const numRevolute = def.joints?.filter(j => j.type !== 'prismatic').length ?? 0
  const sinCosExtra = useSinCos ? (1 + numRevolute) : 0
  const baseObsSize = 5 + 2 * numJoints + numFootBodies + extraObsSize + sinCosExtra
  // Allow character defs to override obsSize (e.g., when using customObsFn)
  const obsSize = def.obsSize ?? (terrain ? baseObsSize + TERRAIN_OBSERVATION_SAMPLES : baseObsSize)

  // Only count actuated joints (maxTorque > 0) as actions.
  // Passive joints (e.g., CartPole hinge) still contribute observations but no action.
  const numActuated = def.joints?.filter(j => (j.maxTorque ?? 0) > 0).length ?? 0
  const actionSize = numActuated

  // forwardBody: first body named 'torso', or the first body
  const forwardBody = def.forwardBody
    || def.bodies?.find(b => b.id === 'torso')?.id
    || def.bodies?.[0]?.id
    || 'torso'

  return { obsSize, actionSize, numFootBodies, numJoints, forwardBody }
}

/**
 * Ensure a BodyDef has all optional fields filled with defaults.
 * Does not modify the input; returns a new object.
 */
export function normalizeBodyDef(def) {
  return {
    name: def.name || 'unnamed',
    gravityScale: def.gravityScale ?? 1.0,
    ground: {
      y: def.ground?.y ?? 0.0,
      friction: def.ground?.friction ?? 0.8,
      restitution: def.ground?.restitution ?? 0.1,
    },
    forwardBody: def.forwardBody || def.bodies?.find(b => b.id === 'torso')?.id || def.bodies?.[0]?.id,
    bodies: (def.bodies || []).map(b => ({
      id: b.id,
      shape: b.shape || 'box',
      w: b.w,
      h: b.h,
      radius: b.radius,
      length: b.length,
      mass: b.mass ?? 1.0,
      friction: b.friction ?? 0.3,
      restitution: b.restitution ?? 0.0,
      spawnX: b.spawnX ?? 0,
      spawnY: b.spawnY ?? 1.0,
      spawnAngle: b.spawnAngle ?? 0,
      fixed: b.fixed ?? false,
      isFootBody: b.isFootBody ?? false,
      minY: b.minY,
      maxAngle: b.maxAngle,
      terminateOnContact: b.terminateOnContact,
    })),
    joints: (def.joints || []).map(j => ({
      id: j.id,
      type: j.type || 'revolute',
      bodyA: j.bodyA,
      bodyB: j.bodyB,
      anchorA: j.anchorA || [0, 0],
      anchorB: j.anchorB || [0, 0],
      // For revolute: angle limits (radians). For prismatic: position limits (meters).
      lowerLimit: j.lowerLimit ?? (j.type === 'prismatic' ? -2.4 : -Math.PI),
      upperLimit: j.upperLimit ?? (j.type === 'prismatic' ? 2.4 : Math.PI),
      // For revolute: max torque. For prismatic: max force.
      maxTorque: j.maxTorque ?? 40.0,
      maxVelocity: j.maxVelocity ?? 8.0,
      // Prismatic joint axis direction (default horizontal)
      axis: j.axis || (j.type === 'prismatic' ? [1, 0] : undefined),
      controlMode: j.controlMode || (j.kp ? 'pd' : 'velocity'),
      kp: j.kp,
      kd: j.kd,
      stiffness: j.stiffness ?? 0,
      damping: j.damping ?? 5.0,
      restAngle: j.restAngle ?? 0,
    })),
    defaultReward: def.defaultReward || {
      forwardVelWeight: 1.0,
      aliveBonusWeight: 1.0,
      ctrlCostWeight: 0.001,
      terminationPenalty: 0.0,
    },
  }
}
