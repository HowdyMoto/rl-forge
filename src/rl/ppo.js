/**
 * PPO (Proximal Policy Optimization) Agent
 * Implemented in TensorFlow.js.
 *
 * Architecture:
 *   - Actor: MLP producing mean + log_std for a Gaussian action distribution
 *   - Critic: MLP producing scalar value estimate (shares trunk or separate)
 *
 * The actor outputs a continuous action. For CartPole we squash via tanh
 * to keep actions in [-1, 1].
 *
 * Key PPO hyperparameters:
 *   clipEpsilon   - surrogate objective clip range (typically 0.1–0.2)
 *   entropyCoef   - encourages exploration by penalizing low entropy
 *   valueLossCoef - weight of value function loss vs policy loss
 *   gamma         - discount factor
 *   lambda        - GAE smoothing parameter
 */

import * as tf from '@tensorflow/tfjs'

export const DEFAULT_PPO_CONFIG = {
  // Network
  hiddenSizes: [64, 64],
  // Rollout
  stepsPerUpdate: 512,   // steps to collect before each PPO update
  numEpochs: 4,          // passes over the rollout buffer per update
  minibatchSize: 128,
  // PPO
  clipEpsilon: 0.2,
  entropyCoef: 0.01,
  valueLossCoef: 0.5,
  maxGradNorm: 0.5,
  // Returns
  gamma: 0.99,
  lambda: 0.95,          // GAE lambda
  // Optimizer
  learningRate: 3e-4,
}

// ─── Network builders ────────────────────────────────────────────────────────

function buildMLP(inputSize, outputSize, hiddenSizes, outputActivation = null) {
  const model = tf.sequential()
  model.add(tf.layers.dense({
    inputShape: [inputSize],
    units: hiddenSizes[0],
    activation: 'tanh',
    kernelInitializer: 'glorotUniform'
  }))
  for (let i = 1; i < hiddenSizes.length; i++) {
    model.add(tf.layers.dense({
      units: hiddenSizes[i],
      activation: 'tanh',
      kernelInitializer: 'glorotUniform'
    }))
  }
  model.add(tf.layers.dense({
    units: outputSize,
    activation: outputActivation,
    kernelInitializer: 'glorotUniform'
  }))
  return model
}

// ─── Gaussian log probability ─────────────────────────────────────────────────

function gaussianLogProb(action, mean, logStd) {
  return tf.tidy(() => {
    const std = tf.exp(logStd)
    const variance = tf.square(std)
    const logProb = tf.mul(-0.5, tf.add(
      tf.div(tf.square(tf.sub(action, mean)), variance),
      tf.add(tf.mul(2, logStd), Math.log(2 * Math.PI))
    ))
    return tf.sum(logProb, -1) // sum over action dims
  })
}

// ─── GAE (Generalized Advantage Estimation) ───────────────────────────────────

function computeGAE(rewards, values, dones, nextValue, gamma, lambda) {
  const n = rewards.length
  const advantages = new Float32Array(n)
  let lastGAE = 0

  for (let t = n - 1; t >= 0; t--) {
    const nextVal = t === n - 1 ? nextValue : values[t + 1]
    const mask = dones[t] ? 0 : 1
    const delta = rewards[t] + gamma * nextVal * mask - values[t]
    lastGAE = delta + gamma * lambda * mask * lastGAE
    advantages[t] = lastGAE
  }

  const returns = new Float32Array(n)
  for (let t = 0; t < n; t++) {
    returns[t] = advantages[t] + values[t]
  }

  return { advantages, returns }
}

// ─── PPO Agent ────────────────────────────────────────────────────────────────

export class PPOAgent {
  constructor(obsSize, actionSize, config = DEFAULT_PPO_CONFIG) {
    this.obsSize = obsSize
    this.actionSize = actionSize
    this.cfg = { ...DEFAULT_PPO_CONFIG, ...config }

    // Actor outputs mean of Gaussian; log_std is a separate learnable parameter
    this.actor = buildMLP(obsSize, actionSize, this.cfg.hiddenSizes)
    // log_std as a standalone variable (not input-dependent, simpler to start)
    this.logStd = tf.variable(
      tf.fill([actionSize], Math.log(0.5)),
      true, 'logStd'
    )

    // Critic outputs scalar value
    this.critic = buildMLP(obsSize, 1, this.cfg.hiddenSizes)

    this.optimizer = tf.train.adam(this.cfg.learningRate, 0.9, 0.999, 1e-5)

    // Rollout buffer
    this.buffer = {
      obs: [], actions: [], rewards: [], dones: [], values: [], logProbs: []
    }
  }

  /**
   * Get action and value for a single observation (inference).
   * Returns { action, value, logProb } all as plain JS numbers/arrays.
   */
  act(obs) {
    return tf.tidy(() => {
      const obsTensor = tf.tensor2d([obs])
      const mean = this.actor.predict(obsTensor)
      const std = tf.exp(this.logStd)

      // Sample from Gaussian
      const noise = tf.randomNormal(mean.shape)
      const rawAction = tf.add(mean, tf.mul(std, noise))
      const action = tf.tanh(rawAction) // squash to [-1, 1]

      // Log prob (account for tanh squashing)
      const logProbRaw = gaussianLogProb(rawAction, mean, this.logStd)
      // Tanh correction: log(1 - tanh^2(x)) = log(1 - action^2)
      const tanhCorrection = tf.sum(
        tf.log(tf.add(tf.scalar(1e-6), tf.sub(tf.scalar(1), tf.square(action)))),
        -1
      )
      const logProb = tf.sub(logProbRaw, tanhCorrection)

      const value = this.critic.predict(obsTensor)

      return {
        action: action.dataSync()[0],
        value: value.dataSync()[0],
        logProb: logProb.dataSync()[0],
        mean: mean.dataSync()[0],
      }
    })
  }

  /**
   * Multi-dimensional action inference for continuous control (hopper, walker).
   * Returns { actions: Float32Array, value, logProb }
   * Each action element is tanh-squashed to [-1, 1].
   */
  actMulti(obs) {
    return tf.tidy(() => {
      const obsTensor = tf.tensor2d([obs])
      const mean = this.actor.predict(obsTensor)
      const std = tf.exp(this.logStd)
      const noise = tf.randomNormal(mean.shape)
      const rawAction = tf.add(mean, tf.mul(std, noise))
      const action = tf.tanh(rawAction)

      const logProbRaw = gaussianLogProb(rawAction, mean, this.logStd)
      const tanhCorrection = tf.sum(
        tf.log(tf.add(tf.scalar(1e-6), tf.sub(tf.scalar(1), tf.square(action)))),
        -1
      )
      const logProb = tf.sub(logProbRaw, tanhCorrection)
      const value = this.critic.predict(obsTensor)

      return {
        actions: new Float32Array(action.dataSync()),
        value: value.dataSync()[0],
        logProb: logProb.dataSync()[0],
      }
    })
  }

  /** Deterministic action for evaluation/rendering (no noise) — scalar */
  actDeterministic(obs) {
    return tf.tidy(() => {
      const obsTensor = tf.tensor2d([obs])
      const mean = this.actor.predict(obsTensor)
      const action = tf.tanh(mean)
      return action.dataSync()[0]
    })
  }

  /** Deterministic multi-dim action for continuous control (hopper, walker) */
  actDeterministicMulti(obs) {
    return tf.tidy(() => {
      const obsTensor = tf.tensor2d([obs])
      const mean = this.actor.predict(obsTensor)
      const action = tf.tanh(mean)
      return new Float32Array(action.dataSync())
    })
  }

  /** Load actor weights from an exported JSON object */
  importWeights(weightsObj) {
    this.actor.trainableWeights.forEach(w => {
      const data = weightsObj[w.name]
      if (data) {
        const oldVal = w.val
        w.val = tf.variable(tf.tensor(data, oldVal.shape), true, w.name)
        oldVal.dispose()
      }
    })
    if (weightsObj['logStd']) {
      const oldLogStd = this.logStd
      this.logStd = tf.variable(
        tf.tensor(weightsObj['logStd'], oldLogStd.shape),
        true, 'logStd'
      )
      oldLogStd.dispose()
    }
  }

  /** Store a transition in the rollout buffer */
  storeTransition(obs, action, reward, done, value, logProb) {
    this.buffer.obs.push(obs)
    this.buffer.actions.push(action)
    this.buffer.rewards.push(reward)
    this.buffer.dones.push(done)
    this.buffer.values.push(value)
    this.buffer.logProbs.push(logProb)
  }

  /** Run PPO update on collected buffer. Returns training metrics. */
  async update(lastObs) {
    const cfg = this.cfg
    const n = this.buffer.obs.length

    // Bootstrap value for last state
    const lastValue = tf.tidy(() => {
      const obsTensor = tf.tensor2d([lastObs])
      return this.critic.predict(obsTensor).dataSync()[0]
    })

    // Compute GAE advantages and returns
    const { advantages, returns } = computeGAE(
      this.buffer.rewards,
      this.buffer.values,
      this.buffer.dones,
      lastValue,
      cfg.gamma,
      cfg.lambda
    )

    // Normalize advantages
    const advMean = advantages.reduce((a, b) => a + b, 0) / n
    const advStd = Math.sqrt(
      advantages.reduce((a, b) => a + (b - advMean) ** 2, 0) / n
    ) + 1e-8
    const advNorm = advantages.map(a => (a - advMean) / advStd)

    // Flatten buffer to tensors
    const obsTensor = tf.tensor2d(this.buffer.obs)
    const actionTensor = tf.tensor2d(this.buffer.actions.map(a => Array.isArray(a) ? a : [a]))
    const returnTensor = tf.tensor1d(Array.from(returns))
    const oldLogProbTensor = tf.tensor1d(this.buffer.logProbs)
    const advTensor = tf.tensor1d(Array.from(advNorm))

    let totalPolicyLoss = 0
    let totalValueLoss = 0
    let totalEntropyLoss = 0
    let updateCount = 0

    // Multiple epochs over the buffer
    for (let epoch = 0; epoch < cfg.numEpochs; epoch++) {
      // Shuffle indices
      const indices = Array.from({ length: n }, (_, i) => i)
        .sort(() => Math.random() - 0.5)

      for (let start = 0; start < n; start += cfg.minibatchSize) {
        const mbIndices = indices.slice(start, start + cfg.minibatchSize)
        if (mbIndices.length < 4) continue

        const mbIdxTensor = tf.tensor1d(mbIndices, 'int32')
        const mbObs = tf.gather(obsTensor, mbIdxTensor)
        const mbActions = tf.gather(actionTensor, mbIdxTensor)
        const mbReturns = tf.gather(returnTensor, mbIdxTensor)
        const mbOldLogProbs = tf.gather(oldLogProbTensor, mbIdxTensor)
        const mbAdvantages = tf.gather(advTensor, mbIdxTensor)

        const { grads, value: lossValue } = this.optimizer.computeGradients(() => {
          return tf.tidy(() => {
            // Actor forward pass
            const mean = this.actor.apply(mbObs)
            const std = tf.exp(this.logStd)

            // Recompute log probs for current policy
            // Actions are stored post-tanh, so invert to get raw actions
            const rawAction = tf.atanh(tf.clipByValue(mbActions, -0.999, 0.999))
            const newLogProbsRaw = gaussianLogProb(rawAction, mean, this.logStd)
            // Apply tanh correction to match how oldLogProbs were computed in act()
            const tanhCorrection = tf.sum(
              tf.log(tf.add(tf.scalar(1e-6), tf.sub(tf.scalar(1), tf.square(mbActions)))),
              -1
            )
            const newLogProbs = tf.sub(newLogProbsRaw, tanhCorrection)

            // Entropy of Gaussian
            const entropy = tf.mean(
              tf.add(this.logStd, tf.scalar(0.5 * Math.log(2 * Math.PI * Math.E)))
            )

            // PPO clipped surrogate objective
            const ratio = tf.exp(tf.sub(newLogProbs, mbOldLogProbs))
            const surr1 = tf.mul(ratio, mbAdvantages)
            const surr2 = tf.mul(
              tf.clipByValue(ratio, 1 - cfg.clipEpsilon, 1 + cfg.clipEpsilon),
              mbAdvantages
            )
            const policyLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)))

            // Value loss
            const values = tf.squeeze(this.critic.apply(mbObs))
            const valueLoss = tf.mean(tf.square(tf.sub(values, mbReturns)))

            // Total loss
            const totalLoss = tf.add(
              tf.add(policyLoss, tf.mul(tf.scalar(cfg.valueLossCoef), valueLoss)),
              tf.neg(tf.mul(tf.scalar(cfg.entropyCoef), entropy))
            )

            totalPolicyLoss += policyLoss.dataSync()[0]
            totalValueLoss += valueLoss.dataSync()[0]
            totalEntropyLoss += entropy.dataSync()[0]
            updateCount++

            return totalLoss
          })
        })

        // Gradient clipping
        const allVars = [...this.actor.trainableWeights.map(w => w.val),
                         this.logStd,
                         ...this.critic.trainableWeights.map(w => w.val)]
        const clippedGrads = {}
        const gradNorm = tf.tidy(() => {
          const allGradVals = allVars.map(v => grads[v.name]).filter(Boolean)
          if (allGradVals.length === 0) return 0
          const norm = tf.sqrt(tf.addN(allGradVals.map(g => tf.sum(tf.square(g)))))
          return norm.dataSync()[0]
        })

        const scale = gradNorm > cfg.maxGradNorm ? cfg.maxGradNorm / gradNorm : 1.0
        for (const [name, grad] of Object.entries(grads)) {
          clippedGrads[name] = tf.mul(grad, scale)
        }

        this.optimizer.applyGradients(clippedGrads)

        // Cleanup
        Object.values(grads).forEach(g => g.dispose())
        Object.values(clippedGrads).forEach(g => g.dispose())
        mbIdxTensor.dispose()
        mbObs.dispose()
        mbActions.dispose()
        mbReturns.dispose()
        mbOldLogProbs.dispose()
        mbAdvantages.dispose()

        lossValue.dispose()
      }
    }

    // Cleanup buffer tensors
    obsTensor.dispose()
    actionTensor.dispose()
    returnTensor.dispose()
    oldLogProbTensor.dispose()
    advTensor.dispose()

    // Clear buffer
    this.buffer = { obs: [], actions: [], rewards: [], dones: [], values: [], logProbs: [] }

    return {
      policyLoss: totalPolicyLoss / Math.max(updateCount, 1),
      valueLoss: totalValueLoss / Math.max(updateCount, 1),
      entropy: totalEntropyLoss / Math.max(updateCount, 1),
    }
  }

  get bufferFull() {
    return this.buffer.obs.length >= this.cfg.stepsPerUpdate
  }

  /** Export actor weights as JSON blob URL */
  async exportModel() {
    const blob = await new Promise(resolve => {
      const weights = {}
      this.actor.trainableWeights.forEach(w => {
        weights[w.name] = Array.from(w.val.dataSync())
      })
      weights['logStd'] = Array.from(this.logStd.dataSync())
      resolve(new Blob([JSON.stringify(weights)], { type: 'application/json' }))
    })
    return URL.createObjectURL(blob)
  }
}
