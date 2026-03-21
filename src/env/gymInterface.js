/**
 * Gym-compatible interface definitions for RL-Forge environments.
 *
 * Follows Gymnasium v1.0 conventions:
 *   reset() -> { obs, info }
 *   step(action) -> { obs, reward, terminated, truncated, info }
 *
 * Observation and action spaces are described as typed objects
 * with shape, dtype, and bounds.
 */

/**
 * Continuous observation/action space (Box space in Gym terms).
 */
export class Box {
  /**
   * @param {number[]} low - lower bounds per dimension
   * @param {number[]} high - upper bounds per dimension
   * @param {string} [dtype='float32']
   */
  constructor(low, high, dtype = 'float32') {
    if (low.length !== high.length) {
      throw new Error(`Box: low (${low.length}) and high (${high.length}) must have same length`)
    }
    this.low = low
    this.high = high
    this.shape = [low.length]
    this.dtype = dtype
  }

  /** Check if a value is within bounds */
  contains(x) {
    if (x.length !== this.shape[0]) return false
    for (let i = 0; i < x.length; i++) {
      if (x[i] < this.low[i] || x[i] > this.high[i]) return false
    }
    return true
  }

  /** Sample a random value uniformly within bounds */
  sample() {
    const result = new Float32Array(this.shape[0])
    for (let i = 0; i < result.length; i++) {
      result[i] = this.low[i] + Math.random() * (this.high[i] - this.low[i])
    }
    return result
  }
}

/**
 * Environment specification: metadata about a registered environment.
 */
export class EnvSpec {
  /**
   * @param {object} params
   * @param {string} params.id - unique environment identifier (e.g., 'Hopper-v1')
   * @param {number} [params.maxEpisodeSteps=1000] - max steps before truncation
   * @param {number} [params.rewardThreshold] - reward threshold to consider "solved"
   */
  constructor({ id, maxEpisodeSteps = 1000, rewardThreshold = undefined }) {
    this.id = id
    this.maxEpisodeSteps = maxEpisodeSteps
    this.rewardThreshold = rewardThreshold
  }
}

/**
 * Base environment class. All RL-Forge environments extend this.
 *
 * Subclasses must implement:
 *   _reset() -> obs (array)
 *   _step(action) -> { obs, reward, terminated, info }
 *   _getObsSpace() -> Box
 *   _getActSpace() -> Box
 */
export class BaseEnv {
  constructor(spec = null) {
    this.spec = spec
    this._stepCount = 0
    this._observationSpace = null
    this._actionSpace = null
  }

  get observationSpace() {
    if (!this._observationSpace) this._observationSpace = this._getObsSpace()
    return this._observationSpace
  }

  get actionSpace() {
    if (!this._actionSpace) this._actionSpace = this._getActSpace()
    return this._actionSpace
  }

  /**
   * Reset the environment.
   * @returns {{ obs: Float32Array|number[], info: object }}
   */
  reset() {
    this._stepCount = 0
    const obs = this._reset()
    return { obs, info: {} }
  }

  /**
   * Take one step.
   * @param {Float32Array|number[]} action
   * @returns {{ obs, reward: number, terminated: boolean, truncated: boolean, info: object }}
   */
  step(action) {
    this._stepCount++
    const result = this._step(action)

    // Handle truncation (max episode steps)
    const truncated = !result.terminated
      && this.spec?.maxEpisodeSteps
      && this._stepCount >= this.spec.maxEpisodeSteps

    return {
      obs: result.obs,
      reward: result.reward,
      terminated: result.terminated,
      truncated: truncated || false,
      info: { ...result.info, stepCount: this._stepCount },
    }
  }

  /** Subclass interface */
  _reset() { throw new Error('_reset() not implemented') }
  _step(_action) { throw new Error('_step() not implemented') }
  _getObsSpace() { throw new Error('_getObsSpace() not implemented') }
  _getActSpace() { throw new Error('_getActSpace() not implemented') }

  /** Clean up resources */
  close() {}

  /** Get a render snapshot (for visualization) */
  render() { return null }
}
