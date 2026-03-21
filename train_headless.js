/**
 * Headless training script for terrain environment.
 * Runs PPO training entirely in Node.js — no browser needed.
 *
 * Usage: npx vite-node train_headless.js [--updates 500] [--envs 8]
 *
 * Prints stats every 10 updates. Use this to iterate on env/reward tuning.
 */

import * as tf from '@tensorflow/tfjs'
import RAPIER from '@dimforge/rapier2d'
import { TerrainRapierEnv } from './src/env/terrainEnv.js'
import { BIPED } from './src/env/characters/biped.js'
import { VecEnv } from './src/env/vecEnv.js'
import { PPOAgent } from './src/rl/ppo.js'

// ── Parse CLI args ──
const args = process.argv.slice(2)
const getArg = (name, fallback) => {
  const i = args.indexOf(name)
  return i >= 0 && args[i + 1] ? Number(args[i + 1]) : fallback
}
const MAX_UPDATES = getArg('--updates', 300)
const NUM_ENVS = getArg('--envs', 8)
const PRINT_EVERY = getArg('--print', 10)

// ── Strip foot sensors (Rapier WASM callback crashes in Node) ──
const bipedNode = JSON.parse(JSON.stringify(BIPED))
const numFeetRemoved = bipedNode.bodies.filter(b => b.isFootBody).length
bipedNode.bodies.forEach(b => delete b.isFootBody)
bipedNode.obsSize = BIPED.obsSize - numFeetRemoved

// ── PPO config (matches browser terrain config) ──
const ppoConfig = {
  hiddenSizes: [128, 128],
  stepsPerUpdate: 2048,
  numEpochs: 10,
  learningRate: 3e-4,
  entropyCoef: 0.05,
}

// ── Setup ──
console.log('Initializing...')
console.log(`  TF backend: ${tf.getBackend() || 'cpu'}`)
console.log(`  Envs: ${NUM_ENVS}, Updates: ${MAX_UPDATES}`)
console.log(`  Biped obs=${bipedNode.obsSize}, act=${bipedNode.actionSize}`)
console.log(`  PD: kp=${bipedNode.joints[0].kp}, kd=${bipedNode.joints[0].kd}, maxTorque=${bipedNode.joints[0].maxTorque}`)
console.log(`  Reward: forwardVelWeight=${bipedNode.defaultReward.forwardVelWeight}, aliveBonusWeight=${bipedNode.defaultReward.aliveBonusWeight}, terminationPenalty=${bipedNode.defaultReward.terminationPenalty}`)
console.log(`  PPO: lr=${ppoConfig.learningRate}, entropy=${ppoConfig.entropyCoef}, epochs=${ppoConfig.numEpochs}`)
console.log()

const env = new VecEnv(() => new TerrainRapierEnv(bipedNode, { difficulty: 0.3 }), NUM_ENVS)
const agent = new PPOAgent(env.observationSize, env.actionSize, ppoConfig)

// ── Training loop ──
let totalSteps = 0
let totalEpisodes = 0
let updateCount = 0
const allEpisodeRewards = []
const epRewards = new Float32Array(NUM_ENVS)
const epSteps = new Uint32Array(NUM_ENVS)

let obsArray = env.resetAll()
const startTime = Date.now()

console.log('Training...')
console.log('─'.repeat(100))

while (updateCount < MAX_UPDATES) {
  // Forward pass
  const { actions, values, logProbs } = await agent.actBatch(obsArray)

  // Step all envs
  const { obs: nextObsArray, rewards, dones } = env.stepAll(actions)

  // Store transitions
  agent.storeTransitions(obsArray, actions, rewards, dones, values, logProbs)
  totalSteps += NUM_ENVS

  // Track episodes
  for (let i = 0; i < NUM_ENVS; i++) {
    epRewards[i] += rewards[i]
    epSteps[i]++
    if (dones[i]) {
      allEpisodeRewards.push(epRewards[i])
      epRewards[i] = 0
      epSteps[i] = 0
      totalEpisodes++
    }
  }

  obsArray = nextObsArray

  // PPO update when buffer full
  if (agent.bufferFull) {
    const t0 = Date.now()
    const metrics = await agent.update(obsArray[0])
    const dt = Date.now() - t0
    updateCount++

    if (updateCount % PRINT_EVERY === 0 || updateCount === 1) {
      const recent = allEpisodeRewards.slice(-50)
      const meanR = recent.length > 0 ? recent.reduce((a, b) => a + b, 0) / recent.length : 0
      const maxR = recent.length > 0 ? Math.max(...recent) : 0
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(0)
      const sps = Math.round(totalSteps / ((Date.now() - startTime) / 1000))

      // Episode length estimate: reward = steps * (alive + vel) - termPenalty
      // So steps ≈ (reward + 50) / ~1.5
      const avgLen = recent.length > 0 ? recent.map(r => Math.max(1, Math.round((r + 50) / 1.5))).reduce((a, b) => a + b, 0) / recent.length : 0

      console.log(
        `update ${String(updateCount).padStart(4)} | ` +
        `reward: ${meanR.toFixed(1).padStart(7)} (max ${maxR.toFixed(1).padStart(7)}) | ` +
        `~len: ${avgLen.toFixed(0).padStart(4)} | ` +
        `entropy: ${metrics.entropy.toFixed(3)} | ` +
        `pLoss: ${metrics.policyLoss.toFixed(4)} | ` +
        `vLoss: ${metrics.valueLoss.toFixed(1).padStart(6)} | ` +
        `eps: ${String(totalEpisodes).padStart(6)} | ` +
        `steps: ${(totalSteps/1000).toFixed(0).padStart(5)}k | ` +
        `${sps} sps | ` +
        `${elapsed}s`
      )
    }
  }
}

console.log('─'.repeat(100))
console.log()

// ── Final statistics ──
const last100 = allEpisodeRewards.slice(-100)
const mean = arr => arr.reduce((a, b) => a + b, 0) / arr.length
const std = arr => { const m = mean(arr); return Math.sqrt(arr.reduce((s, x) => s + (x - m) ** 2, 0) / arr.length) }

console.log('=== FINAL RESULTS ===')
console.log(`Total updates: ${updateCount}`)
console.log(`Total steps: ${totalSteps}`)
console.log(`Total episodes: ${totalEpisodes}`)
console.log(`Total time: ${((Date.now() - startTime) / 1000).toFixed(1)}s`)
console.log()
console.log(`Last 100 episodes:`)
console.log(`  Mean reward: ${mean(last100).toFixed(2)}`)
console.log(`  Std reward:  ${std(last100).toFixed(2)}`)
console.log(`  Max reward:  ${Math.max(...last100).toFixed(2)}`)
console.log(`  Min reward:  ${Math.min(...last100).toFixed(2)}`)

// Reward breakdown
const posRewards = last100.filter(r => r > 0).length
console.log(`  Episodes with positive reward: ${posRewards}/${last100.length} (${(100*posRewards/last100.length).toFixed(0)}%)`)

env.dispose()
console.log('\nDone.')
