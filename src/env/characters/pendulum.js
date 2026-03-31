/**
 * Pendulum (swing-up) character definition.
 *
 * Classic Gym Pendulum-v1: a single link hangs from a fixed pivot under
 * gravity. The agent applies torque to swing the pendulum up and balance
 * it inverted (angle = 0 = pointing up).
 *
 * Reward matches Gym: -(θ² + 0.1·ω² + 0.001·τ²)
 *   where θ is the angle from vertical (normalized to [-π, π]),
 *   ω is angular velocity, τ is applied torque.
 *   Best possible reward per step: 0 (perfectly balanced, no torque).
 *   Worst: -(π² + 0.1·8² + 0.001·2²) ≈ -16.3
 *
 * Observation (3D, custom):
 *   [cos(θ), sin(θ), angular_velocity]
 *
 * Action (1D): [torque] on the revolute joint
 *
 * No termination — episode runs for maxSteps (200).
 */

export const PENDULUM = {
  name: 'pendulum',
  gravityScale: 1.0,

  ground: {
    y: -10.0,       // far below, no interaction
    friction: 0,
    restitution: 0,
  },

  bodies: [
    {
      id: 'anchor',
      shape: 'box',
      w: 0.08,
      h: 0.08,
      mass: 1,
      fixed: true,
      noCollider: true,
      friction: 0,
      restitution: 0,
      spawnX: 0,
      spawnY: 2.0,
      spawnAngle: 0,
    },
    {
      // The pendulum arm — hangs below pivot, goal is to swing up
      id: 'torso',
      shape: 'capsule',
      radius: 0.04,
      length: 0.5,
      mass: 1.0,
      friction: 0,
      restitution: 0,
      spawnX: 0,
      spawnY: 1.75,          // center of link, hanging below anchor
      spawnAngle: Math.PI,   // starts hanging down
      minY: -100,            // no height termination
      maxAngle: 100,         // no angle termination
    },
  ],

  joints: [
    {
      id: 'pivot',
      type: 'revolute',
      bodyA: 'anchor',
      bodyB: 'torso',
      anchorA: [0, 0],
      anchorB: [0, 0.25],    // top of link pinned to anchor
      lowerLimit: -100 * Math.PI,
      upperLimit: 100 * Math.PI,
      maxTorque: 2.0,        // matches Gym max torque
      maxVelocity: 8.0,      // matches Gym max angular velocity
      damping: 2.0,          // motor strength factor for velocity control
    },
  ],

  forwardBody: 'torso',

  // Custom 3D observation: [cos(θ), sin(θ), ω]
  obsSize: 3,

  customObsFn(env) {
    const anchor = env.bodies['anchor']
    const arm = env.bodies['torso']
    if (!anchor || !arm) return [0, 0, 0]
    // θ = angle of arm relative to anchor (0 = aligned with anchor, i.e. pointing up)
    const angle = arm.rotation() - anchor.rotation()
    const angVel = arm.angvel() - anchor.angvel()
    return [Math.cos(angle), Math.sin(angle), angVel]
  },

  resetNoise: {
    position: 0.0,
    angle: Math.PI,    // full random starting angle
    velocity: 0.0,
    angvel: 0.5,       // small random initial spin
  },

  defaultReward: {
    forwardVelWeight: 0,
    aliveBonusWeight: 0,
    ctrlCostWeight: 0,
    terminationPenalty: 0,
  },

  // Classic Pendulum-v1 reward: -(θ² + 0.1·ω² + 0.001·τ²)
  customRewardFn(env, actions, healthy, done, timedOut) {
    const anchor = env.bodies['anchor']
    const arm = env.bodies['torso']
    if (!anchor || !arm) return 0

    const rawAngle = arm.rotation() - anchor.rotation()
    // Normalize angle to [-π, π]
    const theta = ((rawAngle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI
    const omega = arm.angvel() - anchor.angvel()
    const torque = actions[0] * 2.0  // action ∈ [-1,1] → torque ∈ [-2,2]

    return -(theta * theta + 0.1 * omega * omega + 0.001 * torque * torque)
  },

  customRewardBreakdownFn(env) {
    const anchor = env.bodies['anchor']
    const arm = env.bodies['torso']
    if (!anchor || !arm) return { components: [], total: 0 }

    const rawAngle = arm.rotation() - anchor.rotation()
    const theta = ((rawAngle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI
    const omega = arm.angvel() - anchor.angvel()

    const angleCost = -(theta * theta)
    const velCost = -(0.1 * omega * omega)
    const total = angleCost + velCost

    return {
      components: [
        { label: 'Angle cost', value: angleCost, color: angleCost > -1 ? '#4ade80' : '#f87171' },
        { label: 'Velocity cost', value: velCost, color: velCost > -0.5 ? '#4ade80' : '#fbbf24' },
        { label: `θ: ${(theta * 180 / Math.PI).toFixed(0)}°`, value: 0, color: '#94a3b8' },
      ],
      total,
      healthy: true,
      done: false,
    }
  },
}
