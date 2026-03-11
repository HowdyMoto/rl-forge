/**
 * CartPole Environment
 * Classic inverted pendulum on a cart.
 * State: [x, x_dot, theta, theta_dot]
 *   x        - cart position (m)
 *   x_dot    - cart velocity (m/s)
 *   theta    - pole angle from vertical (rad), positive = right lean
 *   theta_dot- pole angular velocity (rad/s)
 *
 * Action: continuous force on cart [-1, 1], scaled to [-maxForce, maxForce]
 *
 * Physics: closed-form equations of motion for cart-pole system.
 * No external physics engine needed — this is the canonical formulation
 * from Barto, Sutton & Anderson (1983).
 */

export const CARTPOLE_PARAMS = {
  gravity: 9.8,
  massCart: 1.0,
  massPole: 0.1,
  poleHalfLen: 0.5,   // half the pole length
  maxForce: 10.0,
  dt: 0.02,           // 50Hz simulation
  // Episode termination thresholds
  xThreshold: 2.4,
  thetaThreshold: 24 * (Math.PI / 180), // 24 degrees in radians
  maxSteps: 500,
}

export class CartPoleEnv {
  constructor(params = CARTPOLE_PARAMS) {
    this.p = params
    this.state = null
    this.stepCount = 0
  }

  /**
   * Reset to a random state near equilibrium.
   * Returns initial observation.
   */
  reset(rng = Math.random) {
    // Small random perturbation around upright
    this.state = [
      (rng() - 0.5) * 0.1,  // x
      (rng() - 0.5) * 0.1,  // x_dot
      (rng() - 0.5) * 0.1,  // theta
      (rng() - 0.5) * 0.1,  // theta_dot
    ]
    this.stepCount = 0
    return [...this.state]
  }

  /**
   * Step the physics forward one timestep.
   * action: scalar in [-1, 1]
   * Returns { obs, reward, done, info }
   */
  step(action) {
    const { gravity, massCart, massPole, poleHalfLen, maxForce, dt,
            xThreshold, thetaThreshold, maxSteps } = this.p

    const force = Math.max(-1, Math.min(1, action)) * maxForce
    const [x, x_dot, theta, theta_dot] = this.state

    const cosTheta = Math.cos(theta)
    const sinTheta = Math.sin(theta)
    const totalMass = massCart + massPole
    const poleMassLen = massPole * poleHalfLen

    // Equations of motion (standard Lagrangian derivation)
    const temp = (force + poleMassLen * theta_dot * theta_dot * sinTheta) / totalMass
    const thetaAcc = (gravity * sinTheta - cosTheta * temp) /
      (poleHalfLen * (4.0 / 3.0 - massPole * cosTheta * cosTheta / totalMass))
    const xAcc = temp - poleMassLen * thetaAcc * cosTheta / totalMass

    // Euler integration
    const newX = x + dt * x_dot
    const newXDot = x_dot + dt * xAcc
    const newTheta = theta + dt * theta_dot
    const newThetaDot = theta_dot + dt * thetaAcc

    this.state = [newX, newXDot, newTheta, newThetaDot]
    this.stepCount++

    const done =
      Math.abs(newX) > xThreshold ||
      Math.abs(newTheta) > thetaThreshold ||
      this.stepCount >= maxSteps

    // Reward shaping:
    // Base: +1 for surviving
    // Bonus: reward for staying near center and upright
    const anglePenalty = (newTheta * newTheta) / (thetaThreshold * thetaThreshold)
    const posPenalty = (newX * newX) / (xThreshold * xThreshold)
    const reward = 1.0 - 0.5 * anglePenalty - 0.1 * posPenalty

    return {
      obs: [...this.state],
      reward: done && this.stepCount < maxSteps ? -1.0 : reward,
      done,
      info: { stepCount: this.stepCount }
    }
  }

  get observationSize() { return 4 }
  get actionSize() { return 1 }
}
