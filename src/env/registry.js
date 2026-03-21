/**
 * Environment Registry
 *
 * Central registry for all RL-Forge environments.
 * Provides makeEnv(envId, options) factory function.
 *
 * NOTE: This module imports Rapier-based environments which require
 * WASM to be initialized. It should be imported in the worker context,
 * not the main UI thread.
 */

import { EnvSpec } from './gymInterface.js'
import { CartPoleEnv } from './cartpole.js'
import { RapierEnv } from './rapierEnv.js'
import { HOPPER } from './characters/hopper.js'
import { WALKER2D } from './characters/walker2d.js'
import { ACROBOT } from './characters/acrobot.js'

// Registry storage: envId -> { spec, factory }
const _registry = new Map()

/**
 * Register an environment.
 * @param {string} id - unique environment identifier
 * @param {object} config
 * @param {function} config.factory - (spec, options) => BaseEnv instance
 * @param {number} [config.maxEpisodeSteps=1000]
 * @param {number} [config.rewardThreshold]
 */
export function registerEnv(id, config) {
  const spec = new EnvSpec({
    id,
    maxEpisodeSteps: config.maxEpisodeSteps || 1000,
    rewardThreshold: config.rewardThreshold,
  })
  _registry.set(id, { spec, factory: config.factory })
}

/**
 * Create an environment instance.
 * @param {string} envId - registered environment ID
 * @param {object} [options] - passed to the environment factory
 * @returns {import('./gymInterface.js').BaseEnv}
 */
export function makeEnv(envId, options = {}) {
  const entry = _registry.get(envId)
  if (!entry) {
    throw new Error(`Environment '${envId}' not registered. Available: ${listEnvs().join(', ')}`)
  }
  return entry.factory(entry.spec, options)
}

/**
 * List all registered environment IDs.
 * @returns {string[]}
 */
export function listEnvs() {
  return Array.from(_registry.keys())
}

/**
 * Get the spec for a registered environment.
 * @param {string} envId
 * @returns {EnvSpec|null}
 */
export function getEnvSpec(envId) {
  return _registry.get(envId)?.spec || null
}

// ─── Register built-in environments ────────────────────────────────────────

registerEnv('CartPole-v1', {
  factory: (spec) => new CartPoleEnv(spec),
  maxEpisodeSteps: 500,
  rewardThreshold: 450,
})

registerEnv('Hopper-v1', {
  factory: (spec) => new RapierEnv(HOPPER, spec),
  maxEpisodeSteps: 1000,
  rewardThreshold: 1500,
})

registerEnv('Walker2d-v1', {
  factory: (spec) => new RapierEnv(WALKER2D, spec),
  maxEpisodeSteps: 1000,
  rewardThreshold: 1500,
})

registerEnv('Acrobot-v1', {
  factory: (spec) => new RapierEnv(ACROBOT, spec),
  maxEpisodeSteps: 500,
  rewardThreshold: 500,
})
