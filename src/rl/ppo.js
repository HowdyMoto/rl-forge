/**
 * PPO (Proximal Policy Optimization) Agent
 * Implemented in TensorFlow.js.
 *
 * Architecture:
 *   - Actor: MLP producing mean + log_std for a Gaussian action distribution
 *   - Critic: MLP producing scalar value estimate (separate network)
 *
 * Matches CleanRL / Stable Baselines 3 reference implementations:
 *   - Orthogonal weight initialization (sqrt(2) hidden, 0.01 policy, 1.0 value)
 *   - Observation normalization (running mean/std)
 *   - Reward normalization (running std of discounted returns)
 *   - Linear learning rate annealing
 *   - Entropy coef = 0.0 for continuous control
 *   - log_std initialized to 0.0
 */

import * as tf from '@tensorflow/tfjs'

export const DEFAULT_PPO_CONFIG = {
  // Network
  hiddenSizes: [64, 64],
  // Rollout
  stepsPerUpdate: 2048,
  numEpochs: 10,
  minibatchSize: 64,
  // PPO
  clipEpsilon: 0.2,
  entropyCoef: 0.0,     // CleanRL/SB3 use exactly 0.0 for continuous control
  valueLossCoef: 0.5,
  maxGradNorm: 0.5,
  // Returns
  gamma: 0.99,
  lambda: 0.95,
  // Optimizer
  learningRate: 3e-4,
  // Training budget (for LR annealing)
  maxUpdates: 3000,
}

// ─── Orthogonal initialization ──────────────────────────────────────────────

/**
 * Orthogonal weight initialization (matches PyTorch nn.init.orthogonal_).
 * Returns a 2D tensor of shape [rows, cols].
 */
function orthogonalInit(shape, scale = 1.0) {
  const [rows, cols] = shape
  // Generate random values and orthogonalize via Gram-Schmidt-like approach
  // For neural network init, scaled random normal is a good approximation
  // when exact orthogonality isn't critical. But we do proper QR:
  const size = Math.max(rows, cols)
  const flat = new Float32Array(size * size)
  for (let i = 0; i < flat.length; i++) {
    // Box-Muller transform for normal distribution
    const u1 = Math.random() || 1e-10
    const u2 = Math.random()
    flat[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }

  // Simple QR via modified Gram-Schmidt (CPU, but only runs once at init)
  const q = new Float32Array(size * size)
  const a = flat // working copy

  for (let j = 0; j < size; j++) {
    // Copy column j
    for (let i = 0; i < size; i++) q[i * size + j] = a[i * size + j]

    // Subtract projections of previous columns
    for (let k = 0; k < j; k++) {
      let dot = 0
      for (let i = 0; i < size; i++) dot += q[i * size + k] * a[i * size + j]
      for (let i = 0; i < size; i++) q[i * size + j] -= dot * q[i * size + k]
    }

    // Normalize
    let norm = 0
    for (let i = 0; i < size; i++) norm += q[i * size + j] * q[i * size + j]
    norm = Math.sqrt(norm) || 1
    for (let i = 0; i < size; i++) q[i * size + j] /= norm
  }

  // Extract [rows, cols] submatrix and scale
  const result = new Float32Array(rows * cols)
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      result[i * cols + j] = q[i * size + j] * scale
    }
  }

  return tf.tensor2d(result, [rows, cols])
}

/**
 * Build MLP with orthogonal initialization.
 * @param {number} inputSize
 * @param {number} outputSize
 * @param {number[]} hiddenSizes
 * @param {number} outputScale - scale for output layer init (0.01 for policy, 1.0 for value)
 */
function buildMLP(inputSize, outputSize, hiddenSizes, outputScale = 1.0) {
  const model = tf.sequential()

  let prevSize = inputSize
  for (let i = 0; i < hiddenSizes.length; i++) {
    model.add(tf.layers.dense({
      inputShape: i === 0 ? [inputSize] : undefined,
      units: hiddenSizes[i],
      activation: 'tanh',
      kernelInitializer: 'zeros',
      biasInitializer: 'zeros',
    }))
    prevSize = hiddenSizes[i]
  }

  model.add(tf.layers.dense({
    units: outputSize,
    activation: null,
    kernelInitializer: 'zeros',
    biasInitializer: 'zeros',
  }))

  // Build the model so weights are allocated, then overwrite with orthogonal init
  model.build([null, inputSize])

  prevSize = inputSize
  for (let i = 0; i < hiddenSizes.length; i++) {
    const kernel = orthogonalInit([prevSize, hiddenSizes[i]], Math.sqrt(2))
    const bias = tf.zeros([hiddenSizes[i]])
    model.layers[i].setWeights([kernel, bias])
    kernel.dispose()
    bias.dispose()
    prevSize = hiddenSizes[i]
  }

  // Output layer
  const outKernel = orthogonalInit([prevSize, outputSize], outputScale)
  const outBias = tf.zeros([outputSize])
  model.layers[hiddenSizes.length].setWeights([outKernel, outBias])
  outKernel.dispose()
  outBias.dispose()

  return model
}

// ─── Running statistics for normalization ───────────────────────────────────

/**
 * Welford's online algorithm for running mean/variance.
 * Matches gym.wrappers.NormalizeObservation behavior.
 */
export class RunningMeanStd {
  constructor(size) {
    this.size = size
    this.mean = new Float32Array(size)    // running mean
    this.var = new Float32Array(size).fill(1.0) // running variance
    this.count = 1e-4  // small epsilon to avoid division by zero initially
  }

  update(batch) {
    // batch is an array of Float32Arrays or number arrays
    const batchSize = batch.length
    if (batchSize === 0) return

    const batchMean = new Float32Array(this.size)
    const batchVar = new Float32Array(this.size)

    // Compute batch mean
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < this.size; j++) {
        batchMean[j] += batch[i][j]
      }
    }
    for (let j = 0; j < this.size; j++) batchMean[j] /= batchSize

    // Compute batch variance
    for (let i = 0; i < batchSize; i++) {
      for (let j = 0; j < this.size; j++) {
        const d = batch[i][j] - batchMean[j]
        batchVar[j] += d * d
      }
    }
    for (let j = 0; j < this.size; j++) batchVar[j] /= batchSize

    // Combine with running stats (parallel algorithm)
    const newCount = this.count + batchSize
    for (let j = 0; j < this.size; j++) {
      const delta = batchMean[j] - this.mean[j]
      const m_a = this.var[j] * this.count
      const m_b = batchVar[j] * batchSize
      const M2 = m_a + m_b + delta * delta * this.count * batchSize / newCount
      this.mean[j] = this.mean[j] + delta * batchSize / newCount
      this.var[j] = M2 / newCount
    }
    this.count = newCount
  }

  normalize(obs, clipRange = 10.0) {
    const result = new Float32Array(this.size)
    for (let j = 0; j < this.size; j++) {
      const std = Math.sqrt(this.var[j] + 1e-8)
      result[j] = Math.max(-clipRange, Math.min(clipRange, (obs[j] - this.mean[j]) / std))
    }
    return result
  }

  /** Normalize a batch of observations in-place, returns new arrays */
  normalizeBatch(obsArray) {
    return obsArray.map(obs => this.normalize(obs))
  }

  /** Serialize for export (so imported weights can denormalize) */
  toJSON() {
    return {
      mean: Array.from(this.mean),
      var: Array.from(this.var),
      count: this.count,
    }
  }

  /** Restore from serialized data */
  static fromJSON(data) {
    const rms = new RunningMeanStd(data.mean.length)
    rms.mean = new Float32Array(data.mean)
    rms.var = new Float32Array(data.var)
    rms.count = data.count
    return rms
  }
}

/**
 * Running reward normalizer.
 * Tracks the variance of discounted returns and divides rewards by the std.
 * Matches gym.wrappers.NormalizeReward.
 */
export class RewardNormalizer {
  constructor(numEnvs, gamma = 0.99) {
    this.gamma = gamma
    this.returnRunning = new Float32Array(numEnvs)  // per-env discounted return
    this.mean = 0
    this.var = 1
    this.count = 1e-4
  }

  normalize(rewards, dones) {
    const normalized = new Float32Array(rewards.length)
    for (let i = 0; i < rewards.length; i++) {
      this.returnRunning[i] = this.returnRunning[i] * this.gamma + rewards[i]
      if (dones[i]) this.returnRunning[i] = 0
    }

    // Update running variance of returns
    const batch = this.returnRunning
    let batchMean = 0, batchVar = 0
    for (let i = 0; i < batch.length; i++) batchMean += batch[i]
    batchMean /= batch.length
    for (let i = 0; i < batch.length; i++) batchVar += (batch[i] - batchMean) ** 2
    batchVar /= batch.length

    const newCount = this.count + batch.length
    const delta = batchMean - this.mean
    this.var = (this.var * this.count + batchVar * batch.length + delta * delta * this.count * batch.length / newCount) / newCount
    this.mean = this.mean + delta * batch.length / newCount
    this.count = newCount

    // Normalize rewards by std of returns
    const std = Math.sqrt(this.var + 1e-8)
    for (let i = 0; i < rewards.length; i++) {
      normalized[i] = Math.max(-10, Math.min(10, rewards[i] / std))
    }
    return normalized
  }
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

    // Actor: orthogonal init, output scale 0.01 (keeps initial actions near zero)
    this.actor = buildMLP(obsSize, actionSize, this.cfg.hiddenSizes, 0.01)
    // log_std initialized to 0.0 (std = 1.0) — standard for continuous control
    this.logStd = tf.variable(
      tf.fill([actionSize], 0.0),
      true, 'logStd'
    )

    // Critic: orthogonal init, output scale 1.0
    this.critic = buildMLP(obsSize, 1, this.cfg.hiddenSizes, 1.0)

    this.optimizer = tf.train.adam(this.cfg.learningRate, 0.9, 0.999, 1e-5)

    // Observation normalization (running mean/std)
    this.obsRms = new RunningMeanStd(obsSize)

    // Track update count for LR annealing
    this.updateCount = 0

    // Rollout buffer
    this.buffer = {
      obs: [], actions: [], rewards: [], dones: [], values: [], logProbs: []
    }
  }

  /** Normalize an observation using running statistics */
  normalizeObs(obs) {
    return this.obsRms.normalize(obs)
  }

  /** Normalize a batch of observations */
  normalizeObsBatch(obsArray) {
    return this.obsRms.normalizeBatch(obsArray)
  }

  /** Update observation running stats with a batch */
  updateObsStats(obsArray) {
    this.obsRms.update(obsArray)
  }

  /**
   * Get action and value for a single observation (inference).
   * Observation should already be normalized.
   */
  async act(obs) {
    const tensors = tf.tidy(() => {
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
        action: action.clone(),
        value: value.clone(),
        logProb: logProb.clone(),
        mean: mean.clone(),
      }
    })

    const [actionData, valueData, logProbData, meanData] = await Promise.all([
      tensors.action.data(),
      tensors.value.data(),
      tensors.logProb.data(),
      tensors.mean.data(),
    ])

    tensors.action.dispose()
    tensors.value.dispose()
    tensors.logProb.dispose()
    tensors.mean.dispose()

    return {
      action: actionData[0],
      value: valueData[0],
      logProb: logProbData[0],
      mean: meanData[0],
    }
  }

  /**
   * Multi-dimensional action inference.
   * Observation should already be normalized.
   */
  async actMulti(obs) {
    const tensors = tf.tidy(() => {
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
        action: action.clone(),
        value: value.clone(),
        logProb: logProb.clone(),
      }
    })

    const [actionData, valueData, logProbData] = await Promise.all([
      tensors.action.data(),
      tensors.value.data(),
      tensors.logProb.data(),
    ])

    tensors.action.dispose()
    tensors.value.dispose()
    tensors.logProb.dispose()

    return {
      actions: new Float32Array(actionData),
      value: valueData[0],
      logProb: logProbData[0],
    }
  }

  /**
   * Batched inference for vectorized environments.
   * Observations should already be normalized.
   */
  async actBatch(obsArray) {
    const N = obsArray.length
    const tensors = tf.tidy(() => {
      const obsTensor = tf.tensor2d(obsArray)
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
      const value = tf.squeeze(this.critic.predict(obsTensor), -1)

      return {
        action: action.clone(),
        value: value.clone(),
        logProb: logProb.clone(),
      }
    })

    const [actionData, valueData, logProbData] = await Promise.all([
      tensors.action.data(),
      tensors.value.data(),
      tensors.logProb.data(),
    ])

    tensors.action.dispose()
    tensors.value.dispose()
    tensors.logProb.dispose()

    const actions = new Array(N)
    for (let i = 0; i < N; i++) {
      actions[i] = new Float32Array(actionData.buffer, actionData.byteOffset + i * this.actionSize * 4, this.actionSize)
    }

    return {
      actions,
      values: new Float32Array(valueData),
      logProbs: new Float32Array(logProbData),
    }
  }

  /**
   * Store N transitions at once (one per vectorized env).
   */
  storeTransitions(obsArray, actions, rewards, dones, values, logProbs) {
    for (let i = 0; i < obsArray.length; i++) {
      this.buffer.obs.push(obsArray[i])
      this.buffer.actions.push(Array.from(actions[i]))
      this.buffer.rewards.push(rewards[i])
      this.buffer.dones.push(dones[i] === 1)
      this.buffer.values.push(values[i])
      this.buffer.logProbs.push(logProbs[i])
    }
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

  /** Deterministic multi-dim action for continuous control */
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
    this.actor.trainableWeights.forEach((w, i) => {
      const key = `actor_${i}`
      const data = weightsObj[key] ?? weightsObj[w.name]
      if (data) {
        const newTensor = tf.tensor(data, w.val.shape)
        w.val.assign(newTensor)
        newTensor.dispose()
      }
    })
    if (weightsObj['logStd']) {
      const newTensor = tf.tensor(weightsObj['logStd'], this.logStd.shape)
      this.logStd.assign(newTensor)
      newTensor.dispose()
    }
    // Restore observation normalization stats if present
    if (weightsObj['obsRms']) {
      this.obsRms = RunningMeanStd.fromJSON(weightsObj['obsRms'])
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

    // Linear LR annealing
    this.updateCount++
    const frac = 1 - (this.updateCount - 1) / cfg.maxUpdates
    const currentLR = cfg.learningRate * Math.max(frac, 0)
    this.optimizer.learningRate = currentLR

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

    // Flatten buffer to tensors
    const obsTensor = tf.tensor2d(this.buffer.obs)
    const actionTensor = tf.tensor2d(this.buffer.actions.map(a => Array.isArray(a) ? a : [a]))
    const returnTensor = tf.tensor1d(Array.from(returns))
    const oldLogProbTensor = tf.tensor1d(this.buffer.logProbs)
    const advArray = Array.from(advantages)

    // Accumulate loss tensors across minibatches
    const lossAccum = { policy: [], value: [], entropy: [] }
    let updateCount = 0

    const maxGradNormTensor = tf.scalar(cfg.maxGradNorm)

    for (let epoch = 0; epoch < cfg.numEpochs; epoch++) {
      const indices = Array.from({ length: n }, (_, i) => i)
        .sort(() => Math.random() - 0.5)

      for (let start = 0; start < n; start += cfg.minibatchSize) {
        const mbIndices = indices.slice(start, start + cfg.minibatchSize)
        if (mbIndices.length < 4) continue

        // Normalize advantages at minibatch level (matches CleanRL)
        let mbAdvNorm
        {
          const mbAdvRaw = mbIndices.map(i => advArray[i])
          const mean = mbAdvRaw.reduce((a, b) => a + b, 0) / mbAdvRaw.length
          const std = Math.sqrt(mbAdvRaw.reduce((a, b) => a + (b - mean) ** 2, 0) / mbAdvRaw.length) + 1e-8
          mbAdvNorm = mbAdvRaw.map(a => (a - mean) / std)
        }

        const mbIdxTensor = tf.tensor1d(mbIndices, 'int32')
        const mbObs = tf.gather(obsTensor, mbIdxTensor)
        const mbActions = tf.gather(actionTensor, mbIdxTensor)
        const mbReturns = tf.gather(returnTensor, mbIdxTensor)
        const mbOldLogProbs = tf.gather(oldLogProbTensor, mbIdxTensor)
        const mbAdvantages = tf.tensor1d(mbAdvNorm)

        let mbPolicyLoss, mbValueLoss, mbEntropy

        const { grads, value: lossValue } = this.optimizer.computeGradients(() => {
          return tf.tidy(() => {
            const mean = this.actor.apply(mbObs)

            const rawAction = tf.atanh(tf.clipByValue(mbActions, -0.999, 0.999))
            const newLogProbsRaw = gaussianLogProb(rawAction, mean, this.logStd)
            const tanhCorrection = tf.sum(
              tf.log(tf.add(tf.scalar(1e-6), tf.sub(tf.scalar(1), tf.square(mbActions)))),
              -1
            )
            const newLogProbs = tf.sub(newLogProbsRaw, tanhCorrection)

            const entropy = tf.mean(
              tf.add(this.logStd, tf.scalar(0.5 * Math.log(2 * Math.PI * Math.E)))
            )

            const ratio = tf.exp(tf.sub(newLogProbs, mbOldLogProbs))
            const surr1 = tf.mul(ratio, mbAdvantages)
            const surr2 = tf.mul(
              tf.clipByValue(ratio, 1 - cfg.clipEpsilon, 1 + cfg.clipEpsilon),
              mbAdvantages
            )
            const policyLoss = tf.neg(tf.mean(tf.minimum(surr1, surr2)))

            const values = tf.squeeze(this.critic.apply(mbObs))
            const valueLoss = tf.mean(tf.square(tf.sub(values, mbReturns)))

            const totalLoss = tf.add(
              tf.add(policyLoss, tf.mul(tf.scalar(cfg.valueLossCoef), valueLoss)),
              tf.neg(tf.mul(tf.scalar(cfg.entropyCoef), entropy))
            )

            mbPolicyLoss = tf.keep(policyLoss.clone())
            mbValueLoss = tf.keep(valueLoss.clone())
            mbEntropy = tf.keep(entropy.clone())

            return totalLoss
          })
        })

        lossAccum.policy.push(mbPolicyLoss)
        lossAccum.value.push(mbValueLoss)
        lossAccum.entropy.push(mbEntropy)
        updateCount++

        const clippedGrads = tf.tidy(() => {
          const allVars = [...this.actor.trainableWeights.map(w => w.val),
                           this.logStd,
                           ...this.critic.trainableWeights.map(w => w.val)]
          const allGradVals = allVars.map(v => grads[v.name]).filter(Boolean)

          if (allGradVals.length === 0) return {}

          const gradNorm = tf.sqrt(tf.addN(allGradVals.map(g => tf.sum(tf.square(g)))))
          const scale = tf.minimum(tf.div(maxGradNormTensor, tf.add(gradNorm, tf.scalar(1e-6))), tf.scalar(1.0))

          const result = {}
          for (const [name, grad] of Object.entries(grads)) {
            result[name] = tf.mul(grad, scale)
          }
          return result
        })

        this.optimizer.applyGradients(clippedGrads)

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

    maxGradNormTensor.dispose()

    obsTensor.dispose()
    actionTensor.dispose()
    returnTensor.dispose()
    oldLogProbTensor.dispose()

    // Clear buffer
    this.buffer = { obs: [], actions: [], rewards: [], dones: [], values: [], logProbs: [] }

    const allLossData = await Promise.all([
      ...lossAccum.policy.map(t => t.data()),
      ...lossAccum.value.map(t => t.data()),
      ...lossAccum.entropy.map(t => t.data()),
    ])

    ;[...lossAccum.policy, ...lossAccum.value, ...lossAccum.entropy].forEach(t => t.dispose())

    const k = updateCount
    let totalPolicyLoss = 0, totalValueLoss = 0, totalEntropyLoss = 0
    for (let i = 0; i < k; i++) {
      totalPolicyLoss += allLossData[i][0]
      totalValueLoss += allLossData[k + i][0]
      totalEntropyLoss += allLossData[2 * k + i][0]
    }

    return {
      policyLoss: totalPolicyLoss / Math.max(k, 1),
      valueLoss: totalValueLoss / Math.max(k, 1),
      entropy: totalEntropyLoss / Math.max(k, 1),
      learningRate: currentLR,
    }
  }

  get bufferFull() {
    return this.buffer.obs.length >= this.cfg.stepsPerUpdate
  }

  /** Export actor weights + obs normalization stats as JSON blob URL */
  async exportModel() {
    const blob = await new Promise(resolve => {
      const weights = {}
      this.actor.trainableWeights.forEach((w, i) => {
        weights[`actor_${i}`] = Array.from(w.val.dataSync())
      })
      weights['logStd'] = Array.from(this.logStd.dataSync())
      // Include obs normalization stats so imported weights work correctly
      weights['obsRms'] = this.obsRms.toJSON()
      resolve(new Blob([JSON.stringify(weights)], { type: 'application/json' }))
    })
    return URL.createObjectURL(blob)
  }
}
