/**
 * Default biped character for the terrain platformer.
 *
 * Scaled to realistic human proportions (~1.75m standing height) so that
 * Earth gravity (9.81 m/s²) looks and feels correct.
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
      w: 0.40,
      h: 0.55,
      mass: 8.0,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 1.50,
      spawnAngle: 0.0,
      minY: 0.5,       // early termination: terrain-relative height
      maxAngle: 1.0,    // ~57 degrees tilt — more forgiving for learning
      terminateOnContact: true,  // episode ends if torso touches ground
    },
    {
      id: 'left_thigh',
      shape: 'capsule',
      radius: 0.085,
      length: 0.60,
      mass: 2.5,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.09,
      spawnY: 0.92,
      spawnAngle: 0.0,
    },
    {
      id: 'left_shin',
      shape: 'capsule',
      radius: 0.07,
      length: 0.55,
      mass: 1.0,
      friction: 0.9,
      restitution: 0.0,
      spawnX: 0.09,
      spawnY: 0.345,
      spawnAngle: 0.0,
      isFootBody: true,
    },
    {
      id: 'right_thigh',
      shape: 'capsule',
      radius: 0.085,
      length: 0.60,
      mass: 2.5,
      friction: 0.3,
      restitution: 0.0,
      spawnX: -0.09,
      spawnY: 0.92,
      spawnAngle: 0.0,
    },
    {
      id: 'right_shin',
      shape: 'capsule',
      radius: 0.07,
      length: 0.55,
      mass: 1.0,
      friction: 0.9,
      restitution: 0.0,
      spawnX: -0.09,
      spawnY: 0.345,
      spawnAngle: 0.0,
      isFootBody: true,
    },
  ],

  joints: [
    {
      id: 'left_hip',
      bodyA: 'torso',
      bodyB: 'left_thigh',
      anchorA: [0.09, -0.275],
      anchorB: [0.0, 0.30],
      lowerLimit: -1.5,   // ~-86 degrees (leg back, like a stride)
      upperLimit: 1.2,    // ~69 degrees (leg forward, like a kick)
      maxTorque: 500.0,
      maxVelocity: 12.0,
      kp: 500,
      kd: 50,
      stiffness: 0,
      damping: 8.0,
    },
    {
      id: 'left_knee',
      bodyA: 'left_thigh',
      bodyB: 'left_shin',
      anchorA: [0.0, -0.30],
      anchorB: [0.0, 0.275],
      lowerLimit: -2.4,   // ~-137 degrees (full bend, like a deep squat)
      upperLimit: 0.05,   // slight hyperextension allowed
      maxTorque: 400.0,
      maxVelocity: 12.0,
      kp: 400,
      kd: 40,
      stiffness: 0,
      damping: 8.0,
    },
    {
      id: 'right_hip',
      bodyA: 'torso',
      bodyB: 'right_thigh',
      anchorA: [-0.09, -0.275],
      anchorB: [0.0, 0.30],
      lowerLimit: -0.9,
      upperLimit: 0.9,
      maxTorque: 500.0,
      maxVelocity: 12.0,
      kp: 500,
      kd: 50,
      stiffness: 0,
      damping: 8.0,
    },
    {
      id: 'right_knee',
      bodyA: 'right_thigh',
      bodyB: 'right_shin',
      anchorA: [0.0, -0.30],
      anchorB: [0.0, 0.275],
      lowerLimit: -1.5,
      upperLimit: 0.05,
      maxTorque: 400.0,
      maxVelocity: 12.0,
      kp: 400,
      kd: 40,
      stiffness: 0,
      damping: 8.0,
    },
  ],

  forwardBody: 'torso',

  // Computed dynamically from character definition:
  // obsSize = 5 + 4*2 + 2 + 40 = 55  (base + joints + feet + terrain)
  // actionSize = 4  (4 actuated joints — policy outputs target angles)
  obsSize: 55,
  actionSize: 4,

  defaultReward: {
    // Linear velocity + alive bonus (clear gradient at all speeds)
    forwardVelWeight: 1.0,    // MuJoCo standard: reward = velocity (m/s)
    aliveBonusWeight: 1.0,    // +1 per step alive
    ctrlCostWeight: 0.001,    // penalize large action magnitudes
    terminationPenalty: 0.0,  // no termination penalty (alive bonus incentivizes survival)
  },
}
