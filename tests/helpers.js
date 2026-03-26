/**
 * Shared test helpers for physics tests.
 *
 * Provides convenience factories that import Rapier + UnifiedRapierEnv
 * and create ready-to-use environments for specific characters.
 */

import RAPIER from '@dimforge/rapier2d'
import { UnifiedRapierEnv } from '../src/env/unifiedEnv.js'
import { HOPPER } from '../src/env/characters/hopper.js'
import { CARTPOLE } from '../src/env/characters/cartpole.js'
import { ACROBOT } from '../src/env/characters/acrobot.js'

export { RAPIER, UnifiedRapierEnv, HOPPER, CARTPOLE, ACROBOT }

// ─── Default env options ────────────────────────────────────────────────────

const DEFAULT_OPTS = {
  controlMode: 'velocity',
  terrain: false,
  gravity: -9.81,
  physicsHz: 240,
  policyHz: 30,
  maxSteps: 1000,
}

// ─── Env factories ──────────────────────────────────────────────────────────

/** Create an env from any charDef + opts. Calls reset() and returns it. */
export function createEnv(charDef, opts = {}) {
  const env = new UnifiedRapierEnv(charDef, { ...DEFAULT_OPTS, ...opts })
  env.reset()
  return env
}

/** Hopper env (velocity mode, standard opts). */
export function createHopperEnv(opts = {}) {
  return createEnv(HOPPER, opts)
}

/** CartPole env (velocity mode, standard opts). */
export function createCartPoleEnv(opts = {}) {
  return createEnv(CARTPOLE, opts)
}

/** Acrobot env (velocity mode, standard opts). */
export function createAcrobotEnv(opts = {}) {
  return createEnv(ACROBOT, opts)
}

/**
 * Minimal single-body env for isolated physics tests.
 * One dynamic body + one joint (optional) — keeps tests focused.
 */
export function createMinimalEnv(overrides = {}) {
  const def = {
    name: 'minimal',
    gravityScale: 1.0,
    ground: { y: 0, friction: 0.8, restitution: 0.1 },
    bodies: [
      {
        id: 'torso',
        shape: 'box',
        w: 0.2,
        h: 0.2,
        mass: 1.0,
        friction: 0.3,
        restitution: 0,
        spawnX: 0,
        spawnY: 2.0,
        spawnAngle: 0,
      },
    ],
    joints: [],
    forwardBody: 'torso',
    obsSize: 5,
    actionSize: 0,
    defaultReward: {
      forwardVelWeight: 0,
      aliveBonusWeight: 1.0,
      ctrlCostWeight: 0,
      terminationPenalty: 0,
      healthyYMin: -100,
      healthyAngleMax: 100,
    },
    ...overrides,
  }
  return createEnv(def, { maxSteps: 9999 })
}

/**
 * Minimal two-body env with one revolute joint for motor tests.
 */
export function createTwoBodyEnv(jointOverrides = {}) {
  const def = {
    name: 'two-body',
    gravityScale: 1.0,
    ground: { y: -10, friction: 0.5, restitution: 0.1 },
    bodies: [
      {
        id: 'torso',
        shape: 'box',
        w: 0.1,
        h: 0.1,
        mass: 1.0,
        fixed: true,
        friction: 0,
        restitution: 0,
        spawnX: 0,
        spawnY: 2.0,
        spawnAngle: 0,
        minY: -100,
        maxAngle: 100,
      },
      {
        id: 'arm',
        shape: 'capsule',
        radius: 0.03,
        length: 0.4,
        mass: 1.0,
        friction: 0.3,
        restitution: 0,
        spawnX: 0,
        spawnY: 1.75,
        spawnAngle: 0,
      },
    ],
    joints: [
      {
        id: 'shoulder',
        bodyA: 'torso',
        bodyB: 'arm',
        anchorA: [0, 0],
        anchorB: [0, 0.2],
        lowerLimit: -Math.PI,
        upperLimit: Math.PI,
        maxTorque: 20.0,
        maxVelocity: 8.0,
        damping: 0,
        stiffness: 0,
        ...jointOverrides,
      },
    ],
    forwardBody: 'torso',
    defaultReward: {
      forwardVelWeight: 0,
      aliveBonusWeight: 0,
      ctrlCostWeight: 0,
      terminationPenalty: 0,
      healthyYMin: -100,
      healthyAngleMax: 100,
    },
  }
  return createEnv(def)
}

/**
 * Step an env N times with given actions, return final result.
 */
export function stepN(env, actions, n) {
  let result
  for (let i = 0; i < n; i++) {
    result = env.step(actions)
  }
  return result
}
