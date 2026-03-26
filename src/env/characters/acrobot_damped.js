/**
 * Acrobot-Damped character definition.
 *
 * Identical to the standard acrobot but with joint damping (2.0) on both
 * shoulder and elbow joints, making the swing-up task harder because the
 * pendulum loses energy to friction.
 */

export const ACROBOT_DAMPED = {
  name: 'acrobot-damped',
  gravityScale: 1.0,

  ground: {
    y: -10.0,
    friction: 0.5,
    restitution: 0.1,
  },

  bodies: [
    {
      id: 'torso',
      shape: 'box',
      w: 0.08,
      h: 0.08,
      mass: 1,
      fixed: true,
      friction: 0,
      restitution: 0,
      spawnX: 0,
      spawnY: 2.0,
      spawnAngle: 0,
      minY: -100,
      maxAngle: 100,
    },
    {
      id: 'link1',
      shape: 'capsule',
      radius: 0.04,
      length: 0.5,
      mass: 1.0,
      friction: 0.3,
      restitution: 0,
      spawnX: 0,
      spawnY: 1.75,
      spawnAngle: Math.PI,
    },
    {
      id: 'link2',
      shape: 'capsule',
      radius: 0.04,
      length: 0.5,
      mass: 1.0,
      friction: 0.3,
      restitution: 0,
      spawnX: 0,
      spawnY: 1.25,
      spawnAngle: Math.PI,
      isFootBody: true,
      minY: -100,
      maxAngle: 100,
    },
  ],

  joints: [
    {
      id: 'shoulder',
      bodyA: 'torso',
      bodyB: 'link1',
      anchorA: [0, 0],
      anchorB: [0, 0.25],
      lowerLimit: -Math.PI * 2,
      upperLimit: Math.PI * 2,
      maxTorque: 0,
      maxVelocity: 15.0,
      stiffness: 0,
      damping: 2.0,
    },
    {
      id: 'elbow',
      bodyA: 'link1',
      bodyB: 'link2',
      anchorA: [0, -0.25],
      anchorB: [0, 0.25],
      lowerLimit: -Math.PI * 2,
      upperLimit: Math.PI * 2,
      maxTorque: 20.0,
      maxVelocity: 15.0,
      stiffness: 0,
      damping: 2.0,
    },
  ],

  forwardBody: 'link2',

  obsSize: 10,
  actionSize: 2,

  defaultReward: {
    forwardVelWeight: 0,
    aliveBonusWeight: 0,
    ctrlCostWeight: 0.001,
    terminationPenalty: 0,
    tipHeightWeight: 1.0,
    anchorY: 2.0,
    healthyYMin: -100,
    healthyAngleMax: 100,
  },
}
