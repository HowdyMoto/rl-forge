/**
 * Procedural Terrain Generation
 *
 * Generates 2D platformer terrain as a series of segments.
 * Each segment has a start/end x position and a height.
 * Terrain types: flat, ramp_up, ramp_down, gap, stairs_up, stairs_down, bump.
 *
 * The terrain is generated procedurally with a seed for reproducibility.
 * Difficulty scales with how far the agent has progressed.
 */

// Simple seeded PRNG (mulberry32)
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = seed + 0x6D2B79F5 | 0
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed)
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t
    return ((t ^ t >>> 14) >>> 0) / 4294967296
  }
}

/**
 * Terrain segment types and their generators.
 * Each returns an array of { x1, y1, x2, y2 } line segments for Rapier colliders.
 */

const SEGMENT_GENERATORS = {
  flat(startX, startY, rng, difficulty) {
    const length = 1.5 + rng() * 2.0
    return {
      segments: [{ x1: startX, y1: startY, x2: startX + length, y2: startY }],
      endX: startX + length,
      endY: startY,
    }
  },

  ramp_up(startX, startY, rng, difficulty) {
    const length = 1.0 + rng() * 1.5
    const rise = 0.2 + rng() * 0.4 * difficulty
    return {
      segments: [{ x1: startX, y1: startY, x2: startX + length, y2: startY + rise }],
      endX: startX + length,
      endY: startY + rise,
    }
  },

  ramp_down(startX, startY, rng, difficulty) {
    const length = 1.0 + rng() * 1.5
    const drop = 0.2 + rng() * 0.4 * difficulty
    return {
      segments: [{ x1: startX, y1: startY, x2: startX + length, y2: startY - drop }],
      endX: startX + length,
      endY: startY - drop,
    }
  },

  gap(startX, startY, rng, difficulty) {
    const gapWidth = 0.3 + rng() * 0.5 * difficulty
    // Add small ledge before and after
    const ledgeBefore = 0.3
    const ledgeAfter = 0.3
    return {
      segments: [
        { x1: startX, y1: startY, x2: startX + ledgeBefore, y2: startY },
        // gap here — no ground
        { x1: startX + ledgeBefore + gapWidth, y1: startY, x2: startX + ledgeBefore + gapWidth + ledgeAfter, y2: startY },
      ],
      endX: startX + ledgeBefore + gapWidth + ledgeAfter,
      endY: startY,
    }
  },

  stairs_up(startX, startY, rng, difficulty) {
    const numSteps = 2 + Math.floor(rng() * 3)
    const stepWidth = 0.4 + rng() * 0.2
    const stepHeight = 0.1 + rng() * 0.1 * difficulty
    const segments = []
    let x = startX
    let y = startY
    for (let i = 0; i < numSteps; i++) {
      // horizontal tread
      segments.push({ x1: x, y1: y, x2: x + stepWidth, y2: y })
      // vertical riser (approximated as steep ramp)
      segments.push({ x1: x + stepWidth, y1: y, x2: x + stepWidth, y2: y + stepHeight })
      x += stepWidth
      y += stepHeight
    }
    // Final flat landing
    segments.push({ x1: x, y1: y, x2: x + 0.5, y2: y })
    return { segments, endX: x + 0.5, endY: y }
  },

  stairs_down(startX, startY, rng, difficulty) {
    const numSteps = 2 + Math.floor(rng() * 3)
    const stepWidth = 0.4 + rng() * 0.2
    const stepHeight = 0.1 + rng() * 0.1 * difficulty
    const segments = []
    let x = startX
    let y = startY
    for (let i = 0; i < numSteps; i++) {
      segments.push({ x1: x, y1: y, x2: x + stepWidth, y2: y })
      segments.push({ x1: x + stepWidth, y1: y, x2: x + stepWidth, y2: y - stepHeight })
      x += stepWidth
      y -= stepHeight
    }
    segments.push({ x1: x, y1: y, x2: x + 0.5, y2: y })
    return { segments, endX: x + 0.5, endY: y }
  },

  bump(startX, startY, rng, difficulty) {
    const width = 0.6 + rng() * 0.8
    const height = 0.1 + rng() * 0.2 * difficulty
    const halfW = width / 2
    return {
      segments: [
        { x1: startX, y1: startY, x2: startX + halfW, y2: startY + height },
        { x1: startX + halfW, y1: startY + height, x2: startX + width, y2: startY },
      ],
      endX: startX + width,
      endY: startY,
    }
  },
}

// Terrain type weights by difficulty level
const EASY_TYPES = ['flat', 'flat', 'flat', 'ramp_up', 'ramp_down', 'bump']
const MEDIUM_TYPES = ['flat', 'flat', 'ramp_up', 'ramp_down', 'bump', 'stairs_up', 'stairs_down', 'gap']
const HARD_TYPES = ['flat', 'ramp_up', 'ramp_down', 'bump', 'stairs_up', 'stairs_down', 'gap', 'gap']

/**
 * Generate a terrain course.
 * @param {number} seed - Random seed
 * @param {number} totalLength - Approximate total length in meters
 * @param {number} difficulty - 0.0 to 1.0
 * @returns {{ segments: Array<{x1,y1,x2,y2}>, totalLength: number }}
 */
export function generateTerrain(seed = 42, totalLength = 50, difficulty = 0.3) {
  const rng = mulberry32(seed)
  const allSegments = []
  let x = -3.0  // Start a bit behind origin so agent spawns on flat ground
  let y = 0.0

  // Always start with a flat section for spawning
  allSegments.push({ x1: x, y1: y, x2: x + 3.0, y2: y })
  x += 3.0

  // Select terrain types based on difficulty
  const types = difficulty < 0.3 ? EASY_TYPES :
                difficulty < 0.7 ? MEDIUM_TYPES : HARD_TYPES

  while (x < totalLength) {
    // Gradually increase local difficulty as agent progresses
    const localDifficulty = Math.min(1.0, difficulty + (x / totalLength) * 0.3)

    const typeIdx = Math.floor(rng() * types.length)
    const type = types[typeIdx]
    const gen = SEGMENT_GENERATORS[type]
    const { segments, endX, endY } = gen(x, y, rng, localDifficulty)

    allSegments.push(...segments)
    x = endX
    y = endY

    // Clamp y to reasonable range
    y = Math.max(-1.0, Math.min(3.0, y))
  }

  // End with a flat finish zone
  allSegments.push({ x1: x, y1: y, x2: x + 3.0, y2: y })

  return { segments: allSegments, totalLength: x + 3.0 }
}

/**
 * Sample terrain height at a given x position.
 * Finds the segment that contains x and interpolates height.
 * @param {Array} segments - Terrain segments
 * @param {number} x - World x position
 * @returns {number} Ground height at x
 */
export function sampleTerrainHeight(segments, x) {
  // Find the segment containing x
  for (const seg of segments) {
    if (x >= seg.x1 && x <= seg.x2) {
      if (Math.abs(seg.x2 - seg.x1) < 0.001) return seg.y1 // Vertical segment
      const t = (x - seg.x1) / (seg.x2 - seg.x1)
      return seg.y1 + t * (seg.y2 - seg.y1)
    }
  }
  // If x is before all segments, return first segment height
  if (segments.length > 0 && x < segments[0].x1) return segments[0].y1
  // If x is after all segments, return last segment height
  if (segments.length > 0) return segments[segments.length - 1].y2
  return 0
}

/**
 * Sample a heightfield ahead of the agent for terrain perception.
 * Returns an array of height samples relative to the agent's position.
 * @param {Array} segments - Terrain segments
 * @param {number} agentX - Agent's current x position
 * @param {number} agentY - Agent's current y position
 * @param {number} numSamples - Number of height samples
 * @param {number} lookAhead - How far ahead to sample (meters)
 * @param {number} lookBehind - How far behind to sample (meters)
 * @returns {Float32Array} Relative terrain heights
 */
export function sampleHeightfield(segments, agentX, agentY, numSamples = 10, lookAhead = 3.0, lookBehind = 0.5) {
  const heights = new Float32Array(numSamples)
  const totalRange = lookBehind + lookAhead
  for (let i = 0; i < numSamples; i++) {
    const t = i / (numSamples - 1)
    const sampleX = agentX - lookBehind + t * totalRange
    const groundY = sampleTerrainHeight(segments, sampleX)
    // Store height relative to agent — policy learns terrain shape relative to self
    heights[i] = groundY - agentY
  }
  return heights
}
