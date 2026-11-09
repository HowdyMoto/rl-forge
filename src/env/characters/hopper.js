/**
 * Hopper character definition.
 *
 * A monopod (single-leg) hopper: torso → thigh → shin → foot.
 * Now calibrated to match the terrain biped's physics parameters
 * (PD control, 240Hz, lower gravity, strong torques) which successfully
 * learn locomotion in this engine.
 */

export const HOPPER = {
  name: 'hopper',
  gravityScale: 1.0,
  gravity: -5.5,       // match terrain mode — gentler than -9.81

  ground: {
    y: 0.0,
    friction: 0.9,     // match terrain biped foot friction
    restitution: 0.1,
  },

  bodies: [
    {
      id: 'torso',
      shape: 'box',
      w: 0.25,
      h: 0.45,
      mass: 3.53,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 1.45,
      spawnAngle: 0.0,
      minY: 0.4,
      maxAngle: 1.0,    // ~57° — match terrain biped tolerance
    },
    {
      id: 'thigh',
      shape: 'capsule',
      radius: 0.05,
      length: 0.35,
      mass: 0.898,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 0.97,
      spawnAngle: 0.0,
    },
    {
      id: 'shin',
      shape: 'capsule',
      radius: 0.04,
      length: 0.30,
      mass: 0.357,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 0.57,
      spawnAngle: 0.0,
    },
    {
      id: 'foot',
      shape: 'box',
      w: 0.16,
      h: 0.06,
      mass: 0.2,
      friction: 0.9,     // high friction foot
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 0.23,
      spawnAngle: 0.0,
      isFootBody: true,
    },
  ],

  joints: [
    {
      id: 'hip',
      bodyA: 'torso',
      bodyB: 'thigh',
      anchorA: [0.0, -0.225],
      anchorB: [0.0,  0.175],
      lowerLimit: -1.5,   // -86° — match terrain biped range
      upperLimit:  1.2,   //  69°
      restAngle: 0,
      maxTorque: 300.0,
      kp: 300,
      kd: 30,
      damping: 5.0,
    },
    {
      id: 'knee',
      bodyA: 'thigh',
      bodyB: 'shin',
      anchorA: [0.0, -0.175],
      anchorB: [0.0,  0.150],
      lowerLimit: -2.4,   // -137° — deep squat allowed
      upperLimit:  0.05,
      restAngle: -0.3,
      maxTorque: 300.0,
      kp: 300,
      kd: 30,
      damping: 5.0,
    },
    {
      id: 'ankle',
      bodyA: 'shin',
      bodyB: 'foot',
      anchorA: [0.0, -0.150],
      anchorB: [0.0,  0.03],
      lowerLimit: -0.785,
      upperLimit:  0.785,
      restAngle: 0,
      maxTorque: 150.0,
      kp: 150,
      kd: 15,
      damping: 3.0,
    },
  ],

  resetNoise: {
    position: 0.05,
    angle: 0.1,
    velocity: 0.5,
    angvel: 0.3,
  },

  forwardBody: 'torso',

  obsSize: 12,
  actionSize: 3,

  defaultReward: {
    forwardVelWeight: 1.5,
    aliveBonusWeight: 1.0,
    ctrlCostWeight: 0.001,
    terminationPenalty: 0.0,
    healthyYMin: 0.4,
    healthyAngleMax: 1.0,
  },
}
