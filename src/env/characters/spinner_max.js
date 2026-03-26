/**
 * Spinner (Max Speed) — single body on a revolute joint.
 *
 * The agent is rewarded for spinning the body as fast as possible
 * in either direction. Pure angular velocity maximization task.
 *
 * Observation (9D, sin/cos encoded):
 *   [torso.y, sin(θ), cos(θ), torso.vx, torso.vy, torso.angVel,
 *    sin(joint_angle), cos(joint_angle), joint_angVel]
 *
 * Action (1D): [torque] on the revolute joint
 */

export const SPINNER_MAX = {
  name: 'spinner-max',
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
      minY: -10,
      maxAngle: 100,
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
      lowerLimit: -100 * Math.PI,
      upperLimit: 100 * Math.PI,
      maxTorque: 5.0,
      maxVelocity: 50.0,   // high cap — let it rip
      damping: 0.2,         // damping so the agent must keep pushing
    },
  ],

  forwardBody: 'torso',
  sinCosAngles: true,

  resetNoise: {
    position: 0.0,
    angle: Math.PI,
    velocity: 0.0,
    angvel: 0.5,
  },

  defaultReward: {
    forwardVelWeight: 0,
    aliveBonusWeight: 0,
    ctrlCostWeight: 0,
    terminationPenalty: 0,
  },

  // Custom reward: proportional to absolute angular velocity
  customRewardFn(env, actions, healthy, done, timedOut) {
    const body = env.bodies['torso']
    const angVel = Math.abs(body.angvel())

    // Scale so reward is ~1.0 at a moderate speed, grows linearly
    const speedReward = angVel / 10.0

    // Small control cost to prevent bang-bang
    const ctrlCost = 0.0005 * actions.reduce((s, a) => s + a * a, 0)

    return speedReward - ctrlCost
  },

  customRewardBreakdownFn(env) {
    const body = env.bodies['torso']
    if (!body) return { components: [], total: 0 }
    const angVel = Math.abs(body.angvel())
    const speedReward = angVel / 10.0

    return {
      components: [
        { label: 'Speed', value: speedReward, color: '#4ade80' },
        { label: `|ω|: ${angVel.toFixed(2)} rad/s`, value: 0, color: '#94a3b8' },
      ],
      total: speedReward,
      healthy: true,
      done: false,
    }
  },
}
