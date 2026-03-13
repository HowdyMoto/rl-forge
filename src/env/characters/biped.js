/**
 * Default biped character for the terrain platformer.
 *
 * A simple bipedal walker optimized for terrain traversal with PD controllers.
 * Users can modify this via the creature builder or create entirely new characters.
 *
 * PD gains (kp, kd) are set per-joint to produce smooth, natural-looking motion.
 * The agent outputs target joint angles rather than raw torques.
 */

export const BIPED = {
  name: 'biped',
  gravityScale: 1.0,

  ground: {
    y: 0.0,
    friction: 0.8,
    restitution: 0.1,
  },

  bodies: [
    {
      id: 'torso',
      shape: 'box',
      w: 0.25,
      h: 0.35,
      mass: 4.0,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 1.25,
      spawnAngle: 0.0,
      minY: 0.4,       // early termination: terrain-relative height
      maxAngle: 0.7,    // ~40 degrees tilt
      terminateOnContact: true,  // episode ends if torso touches ground
    },
    {
      id: 'left_thigh',
      shape: 'capsule',
      radius: 0.055,
      length: 0.38,
      mass: 1.2,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.06,
      spawnY: 0.83,
      spawnAngle: 0.0,
    },
    {
      id: 'left_shin',
      shape: 'capsule',
      radius: 0.045,
      length: 0.35,
      mass: 0.5,
      friction: 0.9,
      restitution: 0.0,
      spawnX: 0.06,
      spawnY: 0.45,
      spawnAngle: 0.0,
      isFootBody: true,
    },
    {
      id: 'right_thigh',
      shape: 'capsule',
      radius: 0.055,
      length: 0.38,
      mass: 1.2,
      friction: 0.3,
      restitution: 0.0,
      spawnX: -0.06,
      spawnY: 0.83,
      spawnAngle: 0.0,
    },
    {
      id: 'right_shin',
      shape: 'capsule',
      radius: 0.045,
      length: 0.35,
      mass: 0.5,
      friction: 0.9,
      restitution: 0.0,
      spawnX: -0.06,
      spawnY: 0.45,
      spawnAngle: 0.0,
      isFootBody: true,
    },
  ],

  joints: [
    {
      id: 'left_hip',
      bodyA: 'torso',
      bodyB: 'left_thigh',
      anchorA: [0.06, -0.175],
      anchorB: [0.0, 0.19],
      lowerLimit: -0.9,   // ~-52 degrees
      upperLimit: 0.9,    // ~52 degrees
      maxTorque: 300.0,
      maxVelocity: 12.0,
      kp: 300,
      kd: 30,
      stiffness: 0,
      damping: 5.0,
    },
    {
      id: 'left_knee',
      bodyA: 'left_thigh',
      bodyB: 'left_shin',
      anchorA: [0.0, -0.19],
      anchorB: [0.0, 0.175],
      lowerLimit: -1.5,   // ~-86 degrees
      upperLimit: 0.05,   // slight hyperextension allowed
      maxTorque: 250.0,
      maxVelocity: 12.0,
      kp: 250,
      kd: 25,
      stiffness: 0,
      damping: 5.0,
    },
    {
      id: 'right_hip',
      bodyA: 'torso',
      bodyB: 'right_thigh',
      anchorA: [-0.06, -0.175],
      anchorB: [0.0, 0.19],
      lowerLimit: -0.9,
      upperLimit: 0.9,
      maxTorque: 300.0,
      maxVelocity: 12.0,
      kp: 300,
      kd: 30,
      stiffness: 0,
      damping: 5.0,
    },
    {
      id: 'right_knee',
      bodyA: 'right_thigh',
      bodyB: 'right_shin',
      anchorA: [0.0, -0.19],
      anchorB: [0.0, 0.175],
      lowerLimit: -1.5,
      upperLimit: 0.05,
      maxTorque: 250.0,
      maxVelocity: 12.0,
      kp: 250,
      kd: 25,
      stiffness: 0,
      damping: 5.0,
    },
  ],

  forwardBody: 'torso',

  // Computed dynamically from character definition:
  // obsSize = 5 + 4*2 + 2 + 10 = 25  (base + joints + feet + terrain)
  // actionSize = 4  (4 actuated joints)
  obsSize: 25,
  actionSize: 4,

  defaultReward: {
    forwardVelWeight: 1.5,
    aliveBonusWeight: 0.5,
    ctrlCostWeight: 0.0005,
    terminationPenalty: 50.0,
  },
}
