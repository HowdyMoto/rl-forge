/**
 * VecEnv — Vectorized environment wrapper.
 *
 * Runs N independent copies of an environment in lockstep.
 * Each env steps with its own action; envs that hit `done` are
 * automatically reset. The main loop sees a persistent [N, obsSize]
 * observation matrix that never has "gaps".
 *
 * This exists to amortize GPU dispatch overhead: one batched forward
 * pass for N observations instead of N individual calls.
 */

export class VecEnv {
  /**
   * @param {Function} envFactory - () => env instance (must have reset/step/getRenderSnapshot)
   * @param {number} numEnvs - number of parallel environments
   */
  constructor(envFactory, numEnvs) {
    this.numEnvs = numEnvs
    this.envs = Array.from({ length: numEnvs }, () => envFactory())
    this.obsSize = this.envs[0].observationSize
    this.actionSize = this.envs[0].actionSize

    // Current observations for each env
    this.currentObs = new Array(numEnvs)
  }

  /** Reset all envs, return [N][obsSize] observation array */
  resetAll() {
    for (let i = 0; i < this.numEnvs; i++) {
      this.currentObs[i] = this.envs[i].reset()
    }
    return this.currentObs
  }

  /**
   * Step all envs with batched actions.
   * @param {Array<Array<number>>} actions - [N][actionSize] actions
   * @returns {{ obs, rewards, dones }} all arrays of length N.
   *   obs[i] is the next observation (auto-reset if done).
   */
  stepAll(actions) {
    const rewards = new Float32Array(this.numEnvs)
    const dones = new Uint8Array(this.numEnvs)

    for (let i = 0; i < this.numEnvs; i++) {
      const result = this.envs[i].step(actions[i])
      rewards[i] = result.reward
      dones[i] = result.done ? 1 : 0

      if (result.done) {
        // Auto-reset and use the fresh obs as currentObs
        this.currentObs[i] = this.envs[i].reset()
      } else {
        this.currentObs[i] = result.obs
      }
    }

    return { obs: this.currentObs, rewards, dones }
  }

  /** Render snapshot from env[0] (the "display" env) */
  getRenderSnapshot() {
    return this.envs[0].getRenderSnapshot()
  }

  get observationSize() { return this.obsSize }

  dispose() {
    for (const env of this.envs) {
      env.dispose?.()
    }
  }
}
