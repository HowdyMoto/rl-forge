/**
 * Spinner (Constant Speed) — single body on a revolute joint.
 *
 * The agent must learn to rotate the body at exactly 1 revolution
 * every 3 seconds (2π/3 ≈ 2.094 rad/s). Rewarded for matching
 * that target angular velocity; penalized for deviation.
 *
 * Observation (9D, sin/cos encoded):
 *   [torso.y, sin(θ), cos(θ), torso.vx, torso.vy, torso.angVel,
 *    sin(joint_angle), cos(joint_angle), joint_angVel]
 *
 * Action (1D): [torque] on the revolute joint
 */

const TARGET_ANGVEL = (2 * Math.PI) / 3  // 1 rev per 3 sec ≈ 2.094 rad/s

export const SPINNER_CONSTANT = {
  name: 'spinner-constant',
  gravityScale: 0.0,  // no gravity — pure rotation task

  ground: {
    y: -2,
    friction: 0,
    restitution: 0,
  },

  bodies: [
    {
      id: 'anchor',
      shape: 'ball',
      radius: 0.05,
      mass: 0.001,
      fixed: true,
      noCollider: true,
      friction: 0,
      restitution: 0,
      spawnX: 0,
      spawnY: 1.0,
      spawnAngle: 0,
    },
    {
      id: 'torso',
      shape: 'box',
      w: 0.8,
      h: 0.15,
      mass: 1.0,
      friction: 0,
      restitution: 0,
      spawnX: 0,
      spawnY: 1.0,
      spawnAngle: 0,
      minY: -10,       // no height termination
      maxAngle: 100,   // no angle termination (it spins!)
    },
  ],

  joints: [
    {
      id: 'spin',
      type: 'revolute',
      bodyA: 'anchor',
      bodyB: 'torso',
      anchorA: [0, 0],
      anchorB: [0, 0],
      lowerLimit: -100 * Math.PI,  // effectively unlimited
      upperLimit: 100 * Math.PI,
      maxTorque: 5.0,
      maxVelocity: 20.0,
      damping: 0.1,  // slight damping so it doesn't spin forever
    },
  ],

  forwardBody: 'torso',
  sinCosAngles: true,   // smooth encoding for continuous rotation

  resetNoise: {
    position: 0.0,
    angle: Math.PI,      // start at random angle
    velocity: 0.0,
    angvel: 0.5,         // small random initial spin
  },

  defaultReward: {
    forwardVelWeight: 0,
    aliveBonusWeight: 0,
    ctrlCostWeight: 0.001,
    terminationPenalty: 0,
  },

  // Custom reward: penalize deviation from target angular velocity
  customRewardFn(env, actions, healthy, done, timedOut) {
    const body = env.bodies['torso']
    const angVel = body.angvel()

    // Reward peaks at 1.0 when perfectly matching target, falls off as Gaussian
    const error = angVel - TARGET_ANGVEL
    const speedReward = Math.exp(-0.5 * error * error)

    // Small control cost
    const ctrlCost = 0.001 * actions.reduce((s, a) => s + a * a, 0)

    return speedReward - ctrlCost
  },

  customRewardBreakdownFn(env) {
    const body = env.bodies['torso']
    if (!body) return { components: [], total: 0 }
    const angVel = body.angvel()
    const error = angVel - TARGET_ANGVEL
    const speedReward = Math.exp(-0.5 * error * error)

    return {
      components: [
        { label: 'Speed Match', value: speedReward, color: speedReward > 0.5 ? '#4ade80' : '#fbbf24' },
        { label: `ω: ${angVel.toFixed(2)} / ${TARGET_ANGVEL.toFixed(2)}`, value: 0, color: '#94a3b8' },
      ],
      total: speedReward,
      healthy: true,
      done: false,
    }
  },
}
