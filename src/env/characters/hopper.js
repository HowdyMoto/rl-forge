/**
 * Hopper character definition.
 *
 * A monopod (single-leg) hopper: torso → thigh → shin → foot.
 * Modeled after MuJoCo's classic Hopper-v4 but scaled for 2D Rapier.
 *
 * Coordinate system: +x = right (forward), +y = up
 * All sizes in meters, masses in kg, angles in radians.
 *
 * The character JSON format is the shared representation between:
 *   - The authoring canvas (M2 future)
 *   - RapierEnv (builds the simulation from this)
 *   - MJCF exporter (future)
 */

export const HOPPER = {
  name: 'hopper',
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
      w: 0.25,
      h: 0.45,
      mass: 3.53,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 1.45,
      spawnAngle: 0.0,
      minY: 0.5,
      maxAngle: 0.8,   // ~46 degrees — relaxed so the agent can learn recovery
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
      friction: 0.8,
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
      anchorA: [0.0, -0.225],   // bottom of torso
      anchorB: [0.0,  0.175],   // top of thigh
      lowerLimit: -0.698,  // -40°
      upperLimit:  0.698,  //  40°
      restAngle: 0,
      maxTorque: 80.0,
      kp: 80,
      kd: 8,
    },
    {
      id: 'knee',
      bodyA: 'thigh',
      bodyB: 'shin',
      anchorA: [0.0, -0.175],   // bottom of thigh
      anchorB: [0.0,  0.150],   // top of shin
      lowerLimit: -1.396,  // -80°
      upperLimit:  0.0,    //   0°
      restAngle: -0.3,
      maxTorque: 80.0,
      kp: 80,
      kd: 8,
    },
    {
      id: 'ankle',
      bodyA: 'shin',
      bodyB: 'foot',
      anchorA: [0.0, -0.150],   // bottom of shin
      anchorB: [0.0,  0.03],    // top of foot
      lowerLimit: -0.785,  // -45°
      upperLimit:  0.785,  //  45°
      restAngle: 0,
      maxTorque: 40.0,
      kp: 40,
      kd: 4,
    },
  ],

  resetNoise: {
    position: 0.05,
    angle: 0.1,
    velocity: 0.5,
    angvel: 0.3,
  },

  // Which body's vx is the "forward velocity" signal for reward
  forwardBody: 'torso',

  // obs = [torso.y, torso.angle, torso.vx, torso.vy, torso.angVel,
  //        hip.angle, hip.angVel, knee.angle, knee.angVel,
  //        ankle.angle, ankle.angVel, foot_ground_contact]
  obsSize: 12,
  // actions = [hip_target, knee_target, ankle_target] in [-1,1]
  actionSize: 3,

  // Default reward function config
  defaultReward: {
    forwardVelWeight: 3.0,
    aliveBonusWeight: 0.0,
    ctrlCostWeight: 0.001,
    terminationPenalty: 0.0,
    healthyYMin: 0.5,
    healthyAngleMax: 0.8,
  },
}
