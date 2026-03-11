/**
 * Acrobot (inverted double pendulum) character definition.
 *
 * A double pendulum hung from a fixed point. The top joint (shoulder) is
 * passive (unactuated); only the elbow joint between link1 and link2 has
 * torque. The task is to balance the pendulum in the inverted position.
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
 * to "link2" so that health / termination checks use the tip of the
 * pendulum (it must stay above minY to survive).
 *
 * Observations (10D):
 *   [torso.y, torso.angle, torso.vx, torso.vy, torso.angVel,   -- anchor (near-constant)
 *    shoulder.angle, shoulder.angVel,
 *    elbow.angle, elbow.angVel,
 *    foot_contact]                                               -- always 0 (ground far below)
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
      // Heavy anchor acting as a fixed pivot point
      // Named "torso" so RapierEnv._getObs() can find it
      id: 'torso',
      shape: 'box',
      w: 0.08,
      h: 0.08,
      mass: 1000,
      friction: 0,
      restitution: 0,
      spawnX: 0,
      spawnY: 2.0,
      spawnAngle: 0,
      // Not used for termination (forwardBody is link2)
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
      spawnY: 2.25,      // center of link1, above anchor
      spawnAngle: 0,      // vertical (pointing up when inverted)
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
      spawnY: 2.75,      // center of link2, above link1
      spawnAngle: 0,
      isFootBody: true,   // needed so footSensorHandle is not null
      // Termination thresholds (checked via forwardBody = 'link2')
      minY: 1.5,          // if link2 center drops below 1.5m, episode ends
      maxAngle: 6.3,      // effectively no angle limit (> 2*pi)
    },
  ],

  joints: [
    {
      id: 'shoulder',
      bodyA: 'torso',       // anchor
      bodyB: 'link1',
      anchorA: [0, 0],      // center of anchor
      anchorB: [0, -0.25],  // bottom of link1
      lowerLimit: -Math.PI,
      upperLimit: Math.PI,
      maxTorque: 0,         // PASSIVE -- unactuated joint
      maxVelocity: 15.0,
      stiffness: 0,
      damping: 0,
    },
    {
      id: 'elbow',
      bodyA: 'link1',
      bodyB: 'link2',
      anchorA: [0, 0.25],   // top of link1
      anchorB: [0, -0.25],  // bottom of link2
      lowerLimit: -Math.PI,
      upperLimit: Math.PI,
      maxTorque: 5.0,       // ACTUATED -- only this joint has torque
      maxVelocity: 15.0,
      stiffness: 0,
      damping: 0,
    },
  ],

  // Health / forward-velocity checks use link2 (the tip body)
  forwardBody: 'link2',

  // obs = [torso.y, torso.angle, torso.vx, torso.vy, torso.angVel,
  //        shoulder.angle, shoulder.angVel, elbow.angle, elbow.angVel,
  //        foot_contact]
  obsSize: 10,
  // actions = [shoulder_torque (zeroed), elbow_torque] in [-1,1]
  actionSize: 2,

  // Reward: survive as long as possible in the inverted position
  defaultReward: {
    forwardVelWeight: 0,        // no forward locomotion goal
    aliveBonusWeight: 1.0,      // +1 per step for staying balanced
    ctrlCostWeight: 0.001,      // small penalty for torque use
    terminationPenalty: 50.0,   // penalty when link2 drops below minY
    healthyYMin: 1.5,
    healthyAngleMax: 6.3,
  },
}
