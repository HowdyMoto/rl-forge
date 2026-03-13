/**
 * Walker2D character definition.
 *
 * A bipedal walker: torso → left_thigh → left_shin
 *                         → right_thigh → right_shin
 * Modeled after MuJoCo's classic Walker2d-v4 but scaled for 2D Rapier.
 *
 * Coordinate system: +x = right (forward), +y = up
 * All sizes in meters, masses in kg, angles in radians.
 *
 * Both shins have foot contact sensors, giving the policy ground-contact
 * information for each leg — essential for learning alternating gait.
 */

export const WALKER2D = {
  name: 'walker2d',
  gravityScale: 1.0,

  // Static ground collider config (not a body, just the floor)
  ground: {
    y: 0.0,
    friction: 0.8,
    restitution: 0.1,
  },

  bodies: [
    {
      id: 'torso',
      shape: 'box',
      w: 0.30,
      h: 0.40,
      mass: 3.53,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 1.30,
      spawnAngle: 0.0,
      // Termination: if torso falls below this y or tilts past this angle, episode ends
      minY: 0.5,
      maxAngle: 0.8,   // ~46 degrees — wider tolerance for early exploration
    },
    {
      id: 'left_thigh',
      shape: 'capsule',
      radius: 0.05,
      length: 0.35,   // total length; capsule extends ±length/2 from center
      mass: 0.898,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.05,
      spawnY: 0.875,
      spawnAngle: 0.0,
    },
    {
      id: 'left_shin',
      shape: 'capsule',
      radius: 0.04,
      length: 0.30,
      mass: 0.357,
      friction: 0.8,
      restitution: 0.0,
      spawnX: 0.05,
      spawnY: 0.50,
      spawnAngle: 0.0,
      isFootBody: true,   // ground contact sensor for left foot
    },
    {
      id: 'right_thigh',
      shape: 'capsule',
      radius: 0.05,
      length: 0.35,
      mass: 0.898,
      friction: 0.3,
      restitution: 0.0,
      spawnX: -0.05,
      spawnY: 0.875,
      spawnAngle: 0.0,
    },
    {
      id: 'right_shin',
      shape: 'capsule',
      radius: 0.04,
      length: 0.30,
      mass: 0.357,
      friction: 0.8,
      restitution: 0.0,
      spawnX: -0.05,
      spawnY: 0.50,
      spawnAngle: 0.0,
      isFootBody: true,   // ground contact sensor goes on this body
    },
  ],

  joints: [
    {
      id: 'left_hip',
      bodyA: 'torso',
      bodyB: 'left_thigh',
      anchorA: [0.05, -0.20],   // bottom-left of torso
      anchorB: [0.0,   0.175],  // top of thigh
      lowerLimit: -0.698,  // -40°
      upperLimit:  0.698,  //  40°
      maxTorque: 250.0,
      maxVelocity: 10.0,
      stiffness: 0,
      damping: 5.0,
    },
    {
      id: 'left_knee',
      bodyA: 'left_thigh',
      bodyB: 'left_shin',
      anchorA: [0.0, -0.175],   // bottom of thigh
      anchorB: [0.0,  0.150],   // top of shin
      lowerLimit: -1.396,  // -80° (knee bends backward)
      upperLimit:  0.0,    //   0° (can't hyperextend)
      maxTorque: 250.0,
      maxVelocity: 10.0,
      stiffness: 0,
      damping: 5.0,
    },
    {
      id: 'right_hip',
      bodyA: 'torso',
      bodyB: 'right_thigh',
      anchorA: [-0.05, -0.20],  // bottom-right of torso
      anchorB: [0.0,    0.175], // top of thigh
      lowerLimit: -0.698,  // -40°
      upperLimit:  0.698,  //  40°
      maxTorque: 100.0,
      maxVelocity: 10.0,
      stiffness: 0,
      damping: 5.0,
    },
    {
      id: 'right_knee',
      bodyA: 'right_thigh',
      bodyB: 'right_shin',
      anchorA: [0.0, -0.175],   // bottom of thigh
      anchorB: [0.0,  0.150],   // top of shin
      lowerLimit: -1.396,  // -80°
      upperLimit:  0.0,    //   0°
      maxTorque: 100.0,
      maxVelocity: 10.0,
      stiffness: 0,
      damping: 5.0,
    },
  ],

  // Which body's vx is the "forward velocity" signal for reward
  forwardBody: 'torso',

  // PPO network sizing derived from this character
  // obs = [torso.y, torso.angle, torso.vx, torso.vy, torso.angVel,        (5)
  //        left_hip.angle, left_hip.angVel,                                (2)
  //        left_knee.angle, left_knee.angVel,                              (2)
  //        right_hip.angle, right_hip.angVel,                              (2)
  //        right_knee.angle, right_knee.angVel,                            (2)
  //        left_shin_ground_contact, right_shin_ground_contact]            (2)
  //                                                                  total: 15
  obsSize: 15,
  // actions = [left_hip_torque, left_knee_torque, right_hip_torque, right_knee_torque] in [-1,1]
  actionSize: 4,

  // Default reward function config
  defaultReward: {
    forwardVelWeight: 1.0,
    aliveBonusWeight: 1.0,
    ctrlCostWeight: 0.001,
    terminationPenalty: 50.0,
    healthyYMin: 0.5,
    healthyAngleMax: 0.8,
  },
}
