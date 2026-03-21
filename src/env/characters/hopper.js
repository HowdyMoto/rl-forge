/**
 * Hopper character definition.
 *
 * A monopod (single-leg) hopper: torso → thigh → shin.
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
      // Spawn position (will be reset each episode)
      spawnX: 0.0,
      spawnY: 1.35,
      spawnAngle: 0.0,
      // Termination: if torso falls below this y or tilts past this angle, episode ends
      minY: 0.7,
      maxAngle: 0.4,   // ~23 degrees — torso must stay roughly upright
    },
    {
      id: 'thigh',
      shape: 'capsule',
      radius: 0.05,
      length: 0.35,   // total length; capsule extends ±length/2 from center
      mass: 0.898,
      friction: 0.3,
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 0.87,
      spawnAngle: 0.0,
    },
    {
      id: 'shin',
      shape: 'capsule',
      radius: 0.04,
      length: 0.30,
      mass: 0.357,
      friction: 0.8,   // shin/foot needs high friction
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 0.47,
      spawnAngle: 0.0,
      isFootBody: true,   // ground contact sensor goes on this body
    },
  ],

  joints: [
    {
      id: 'hip',
      bodyA: 'torso',
      bodyB: 'thigh',
      // Anchor positions in each body's local frame (relative to body center)
      anchorA: [0.0, -0.225],   // bottom of torso
      anchorB: [0.0,  0.175],   // top of thigh
      lowerLimit: -0.698,  // -40°
      upperLimit:  0.698,  //  40°
      maxTorque: 40.0,
      maxVelocity: 8.0,
      stiffness: 0,      // purely actuated (no spring)
      damping: 5.0,
    },
    {
      id: 'knee',
      bodyA: 'thigh',
      bodyB: 'shin',
      anchorA: [0.0, -0.175],   // bottom of thigh
      anchorB: [0.0,  0.150],   // top of shin
      lowerLimit: -1.396,  // -80° (knee bends backward)
      upperLimit:  0.0,    //   0° (can't hyperextend)
      maxTorque: 40.0,
      maxVelocity: 8.0,
      stiffness: 0,
      damping: 5.0,
    },
  ],

  // Which body's vx is the "forward velocity" signal for reward
  forwardBody: 'torso',

  // PPO network sizing derived from this character
  // obs = [torso.y, torso.angle, torso.vx, torso.vy, torso.angVel,
  //        hip.angle, hip.angVel, knee.angle, knee.angVel,
  //        shin_ground_contact]
  obsSize: 10,
  // actions = [hip_torque, knee_torque] in [-1,1]
  actionSize: 2,

  // Default reward function config (used by visual builder in M2)
  defaultReward: {
    forwardVelWeight: 1.0,
    aliveBonusWeight: 1.0,
    ctrlCostWeight: 0.001,
    terminationPenalty: 0.0,
    healthyYMin: 0.7,
    healthyAngleMax: 0.4,
  },
}
