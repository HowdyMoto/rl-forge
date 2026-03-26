/**
 * CartPole character definition for unified Rapier environment.
 *
 * Classic inverted pendulum: a cart slides on a horizontal rail,
 * a pole is attached to the cart via a revolute joint.
 *
 * Physics parameters match Barto, Sutton & Anderson (1983):
 *   - Cart mass: 1.0 kg
 *   - Pole mass: 0.1 kg
 *   - Pole half-length: 0.5 m
 *   - Max force: 10 N
 *   - Position threshold: ±2.4 m
 *   - Angle threshold: ±24°
 *
 * Control mode: 'force' — action is raw force applied to the cart
 * along the prismatic joint axis.
 *
 * Observation (9D unified):
 *   [cart.y, cart.angle, cart.vx, cart.vy, cart.angVel,    (5 - torso/cart state)
 *    rail_position, rail_velocity,                         (2 - prismatic joint)
 *    pole_angle, pole_angVel]                              (2 - revolute joint)
 *
 * Action (1D — only actuated joints count):
 *   [rail_force]
 *   Pole hinge is passive (maxTorque=0), excluded from action space.
 */

export const CARTPOLE = {
  name: 'cartpole',
  gravityScale: 1.0,

  ground: {
    y: -0.05,      // just below cart rail height
    friction: 0.0, // frictionless ground (cart slides freely)
    restitution: 0.0,
  },

  bodies: [
    {
      // Fixed anchor point for the prismatic rail (no collider — pure pivot)
      id: 'anchor',
      shape: 'box',
      w: 0.001,
      h: 0.001,
      mass: 0.001,
      fixed: true,
      noCollider: true,
      friction: 0,
      restitution: 0,
      spawnX: 0,
      spawnY: 0.15,
      spawnAngle: 0,
    },
    {
      // Cart: slides left/right on the rail
      id: 'torso',   // named 'torso' so observation builder finds it
      shape: 'box',
      w: 0.5,
      h: 0.15,
      mass: 1.0,
      friction: 0.0,
      restitution: 0.0,
      spawnX: 0.0,
      spawnY: 0.15,
      spawnAngle: 0.0,
      // Termination: if cart goes past ±2.4m or tilts (shouldn't tilt, but safety)
      minY: -10,       // no height termination
      maxAngle: 100,   // no angle termination on cart itself
    },
    {
      // Pole: inverted pendulum attached to top of cart
      id: 'pole',
      shape: 'capsule',
      radius: 0.025,
      length: 1.0,     // total length = 2 * half-length
      mass: 0.1,
      friction: 0.0,
      restitution: 0.0,
      noCollider: true, // no physical collision — constrained by hinge, terminates on angle
      spawnX: 0.0,
      spawnY: 0.725,   // cart top (0.225) + pole half-length (0.5)
      spawnAngle: 0.0,
    },
  ],

  joints: [
    {
      // Rail: prismatic joint constraining cart to horizontal movement
      id: 'rail',
      type: 'prismatic',
      bodyA: 'anchor',
      bodyB: 'torso',
      anchorA: [0, 0],
      anchorB: [0, 0],
      axis: [1, 0],          // horizontal axis
      lowerLimit: -2.4,      // position limits (meters)
      upperLimit: 2.4,
      maxTorque: 10.0,       // max force (N) — for prismatic, this is force not torque
      maxVelocity: 20.0,
      damping: 0.0,          // frictionless rail
    },
    {
      // Hinge: revolute joint between cart and pole
      id: 'hinge',
      type: 'revolute',
      bodyA: 'torso',
      bodyB: 'pole',
      anchorA: [0, 0.075],   // top center of cart
      anchorB: [0, -0.5],    // bottom of pole
      lowerLimit: -Math.PI,  // full rotation range (no angle limits)
      upperLimit: Math.PI,
      maxTorque: 0,           // PASSIVE — pole swings freely
      maxVelocity: 20.0,
      damping: 0.0,           // no friction in the hinge
    },
  ],

  forwardBody: 'torso',

  // Gentle reset noise matching classic CartPole (±0.05 uniform)
  resetNoise: {
    position: 0.05,   // ± meters
    angle:    0.1,     // ± radians (~5.7°)
    velocity: 0.1,     // ± m/s
    angvel:   0.1,     // ± rad/s
  },

  // These are now auto-computed by computeDerivedFields():
  // obsSize: 9 (5 base + 2 prismatic + 2 revolute)
  // actionSize: 1 (only rail — pole hinge is passive)

  defaultReward: {
    // CartPole reward: +1 for surviving, penalties for drifting
    forwardVelWeight: 0,       // no forward velocity reward
    aliveBonusWeight: 1.0,     // +1 per step alive
    ctrlCostWeight: 0.0,       // no control cost
    terminationPenalty: 1.0,   // -1 for falling
    // Custom termination conditions (checked by env)
    cartPositionLimit: 1.8,    // terminate if |cart_x| > 1.8 (well inside joint limit of 2.4)
    poleAngleLimit: 24 * (Math.PI / 180),  // terminate if |pole_angle| > 24°
    // Reward shaping (optional)
    anglePenaltyWeight: 0.5,   // penalty for pole angle
    positionPenaltyWeight: 0.5, // penalty for cart position (strong centering signal)
  },
}
