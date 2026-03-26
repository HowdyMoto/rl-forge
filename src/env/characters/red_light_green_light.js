/**
 * Red Light Green Light — single body on a revolute joint.
 *
 * A signal alternates between green (spin!) and red (stop!).
 * During green light: rewarded for angular velocity.
 * During red light: penalized for any angular velocity.
 *
 * The agent must learn to spin up during green phases and brake
 * hard when the light turns red.
 *
 * Observation (10D = 9 base sin/cos + 1 extra):
 *   [torso.y, sin(θ), cos(θ), torso.vx, torso.vy, torso.angVel,
 *    sin(joint_angle), cos(joint_angle), joint_angVel,
 *    light_state]   ← 1.0 = green, 0.0 = red
 *
 * Action (1D): [torque] on the revolute joint
 */

// Light timing (in policy steps at 30 Hz)
const GREEN_DURATION_MIN = 45   // 1.5 sec
const GREEN_DURATION_MAX = 90   // 3.0 sec
const RED_DURATION_MIN = 30     // 1.0 sec
const RED_DURATION_MAX = 60     // 2.0 sec

function randomDuration(min, max) {
  return min + Math.floor(Math.random() * (max - min + 1))
}

export const RED_LIGHT_GREEN_LIGHT = {
  name: 'red-light-green-light',
  gravityScale: 0.0,  // no gravity

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
      maxVelocity: 30.0,
      damping: 0.3,  // moderate damping — agent must actively brake
    },
  ],

  forwardBody: 'torso',
  sinCosAngles: true,

  // 1 extra observation: light state (green=1, red=0)
  extraObsSize: 1,

  resetNoise: {
    position: 0.0,
    angle: Math.PI,
    velocity: 0.0,
    angvel: 0.0,   // start still
  },

  defaultReward: {
    forwardVelWeight: 0,
    aliveBonusWeight: 0,
    ctrlCostWeight: 0,
    terminationPenalty: 0,
  },

  // ─── Custom hooks ──────────────────────────────────────────────────────────

  onReset(env) {
    // Start with green light
    env._lightGreen = true
    env._lightTimer = randomDuration(GREEN_DURATION_MIN, GREEN_DURATION_MAX)
  },

  onStep(env) {
    // Count down the light timer and toggle
    env._lightTimer--
    if (env._lightTimer <= 0) {
      env._lightGreen = !env._lightGreen
      env._lightTimer = env._lightGreen
        ? randomDuration(GREEN_DURATION_MIN, GREEN_DURATION_MAX)
        : randomDuration(RED_DURATION_MIN, RED_DURATION_MAX)
    }
  },

  extraObs(env) {
    return [env._lightGreen ? 1.0 : 0.0]
  },

  customRewardFn(env, actions, healthy, done, timedOut) {
    const body = env.bodies['torso']
    const angVel = Math.abs(body.angvel())

    if (env._lightGreen) {
      // Green light: reward for spinning
      return angVel / 10.0
    } else {
      // Red light: penalize for any movement
      // Steep penalty so agent learns to stop fast
      return -angVel * 0.5
    }
  },

  customRewardBreakdownFn(env) {
    const body = env.bodies['torso']
    if (!body) return { components: [], total: 0 }
    const angVel = Math.abs(body.angvel())
    const isGreen = env._lightGreen

    let reward, label, color
    if (isGreen) {
      reward = angVel / 10.0
      label = 'Green: Spin!'
      color = '#4ade80'
    } else {
      reward = -angVel * 0.5
      label = 'Red: Stop!'
      color = '#f87171'
    }

    return {
      components: [
        { label, value: reward, color },
        { label: `|ω|: ${angVel.toFixed(2)}`, value: 0, color: '#94a3b8' },
        { label: isGreen ? '🟢 GREEN' : '🔴 RED', value: 0, color: isGreen ? '#4ade80' : '#f87171' },
      ],
      total: reward,
      healthy: true,
      done: false,
    }
  },

  // Provide light state for renderer
  getRenderExtra(env) {
    return {
      lightGreen: env._lightGreen ?? true,
      lightTimer: env._lightTimer ?? 0,
    }
  },
}
