/**
 * Acrobot (swing-up double pendulum) character definition.
 *
 * A double pendulum hangs from a fixed point. The top joint (shoulder) is
 * passive (unactuated); only the elbow joint between link1 and link2 has
 * torque. The task is to swing the tip UP above the anchor height.
 *
 * Coordinate system: +x = right, +y = up
 * All sizes in meters, masses in kg, angles in radians.
 *
 * Architecture note: RapierEnv only creates dynamic bodies, so the fixed
 * pivot is approximated by an extremely heavy anchor body ("torso") that
 * is effectively immovable under gravity.
 *
 * The body named "torso" is the anchor — RapierEnv._getObs() hardcodes
 * this.bodies['torso'] for the observation vector.  forwardBody is set
 * to "link2" so that reward uses the tip of the pendulum.
 *
 * Observations (10D):
 *   [torso.y, torso.angle, torso.vx, torso.vy, torso.angVel,
 *    shoulder.angle, shoulder.angVel,
 *    elbow.angle, elbow.angVel,
 *    foot_contact]
 *
 * Actions (2D):
 *   [shoulder_torque, elbow_torque]
 *   shoulder maxTorque is 0, so action[0] has no effect.
 */

export const ACROBOT = {
  name: 'acrobot',
  gravityScale: 1.0,

  // Ground placed far below so it never interacts with the pendulum
  ground: {
    y: -10.0,
    friction: 0.5,
    restitution: 0.1,
  },

  bodies: [
    {
      // Fixed anchor acting as the pivot point
      // Named "torso" so RapierEnv._getObs() can find it
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
      spawnY: 1.75,       // center of link1, hanging below anchor
      spawnAngle: Math.PI, // flipped so local bottom is pinned at top
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
      spawnY: 1.25,       // center of link2, hanging below link1
      spawnAngle: Math.PI, // flipped
      isFootBody: true,    // needed so footSensorHandle is not null
      // No early termination — let it swing freely
      minY: -100,
      maxAngle: 100,
    },
  ],

  joints: [
    {
      id: 'shoulder',
      bodyA: 'torso',       // anchor
      bodyB: 'link1',
      anchorA: [0, 0],      // center of anchor
      anchorB: [0, 0.25],   // top of link1 (link1 hangs below anchor)
      lowerLimit: -Math.PI * 2,
      upperLimit: Math.PI * 2,
      maxTorque: 0,         // PASSIVE -- unactuated joint
      maxVelocity: 15.0,
      stiffness: 0,
      damping: 0,
    },
    {
      id: 'elbow',
      bodyA: 'link1',
      bodyB: 'link2',
      anchorA: [0, -0.25],  // bottom of link1
      anchorB: [0, 0.25],   // top of link2 (link2 hangs below link1)
      lowerLimit: -Math.PI * 2,
      upperLimit: Math.PI * 2,
      maxTorque: 10.0,      // ACTUATED -- only this joint has torque
      maxVelocity: 15.0,
      stiffness: 0,
      damping: 0,
    },
  ],

  // Reward / health checks use link2 (the tip body)
  forwardBody: 'link2',

  obsSize: 10,
  actionSize: 2,

  // Reward: swing the tip as high as possible
  defaultReward: {
    forwardVelWeight: 0,        // no forward locomotion goal
    aliveBonusWeight: 0,        // no alive bonus (never terminates early)
    ctrlCostWeight: 0.001,      // small penalty for torque use
    terminationPenalty: 0,      // no termination penalty
    tipHeightWeight: 1.0,       // reward = (link2.y - anchorY) per step
    anchorY: 2.0,               // y position of the pivot
    healthyYMin: -100,
    healthyAngleMax: 100,
  },
}
