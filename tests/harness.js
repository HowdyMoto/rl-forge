/**
 * Minimal in-browser test harness.
 *
 * Usage (inside a test HTML page):
 *   import { describe, it, assert, report } from './harness.js'
 *
 *   describe('My Suite', () => {
 *     it('does something', () => { assert.equal(1, 1) })
 *   })
 *
 *   await report()   // writes "X passed, Y failed" to #log
 *
 * The Puppeteer runner (test_physics.mjs) looks for that summary line.
 */

let passed = 0
let failed = 0
const suites = []    // [{ name, tests: [{ name, fn }] }]
let currentSuite = null

// ─── Public API ─────────────────────────────────────────────────────────────

export function describe(name, fn) {
  const suite = { name, tests: [] }
  suites.push(suite)
  const prev = currentSuite
  currentSuite = suite
  fn()
  currentSuite = prev
}

export function it(name, fn) {
  if (!currentSuite) throw new Error('it() must be called inside describe()')
  currentSuite.tests.push({ name, fn })
}

export const assert = {
  equal(actual, expected, msg) {
    if (actual !== expected) {
      throw new Error(msg || `Expected ${expected}, got ${actual}`)
    }
  },

  notEqual(actual, expected, msg) {
    if (actual === expected) {
      throw new Error(msg || `Expected value to differ from ${expected}`)
    }
  },

  near(actual, expected, tolerance, msg) {
    const diff = Math.abs(actual - expected)
    if (diff > tolerance) {
      throw new Error(
        msg || `Expected ~${expected} ±${tolerance}, got ${actual} (off by ${diff.toFixed(6)})`
      )
    }
  },

  arrayNear(actual, expected, tolerance, msg) {
    if (actual.length !== expected.length) {
      throw new Error(
        msg || `Array length mismatch: got ${actual.length}, expected ${expected.length}`
      )
    }
    for (let i = 0; i < expected.length; i++) {
      const diff = Math.abs(actual[i] - expected[i])
      if (diff > tolerance) {
        throw new Error(
          msg || `Element [${i}]: expected ~${expected[i]} ±${tolerance}, got ${actual[i]}`
        )
      }
    }
  },

  true(val, msg) {
    if (val !== true) throw new Error(msg || `Expected true, got ${val}`)
  },

  false(val, msg) {
    if (val !== false) throw new Error(msg || `Expected false, got ${val}`)
  },

  greaterThan(a, b, msg) {
    if (!(a > b)) throw new Error(msg || `Expected ${a} > ${b}`)
  },

  lessThan(a, b, msg) {
    if (!(a < b)) throw new Error(msg || `Expected ${a} < ${b}`)
  },

  greaterThanOrEqual(a, b, msg) {
    if (!(a >= b)) throw new Error(msg || `Expected ${a} >= ${b}`)
  },

  lessThanOrEqual(a, b, msg) {
    if (!(a <= b)) throw new Error(msg || `Expected ${a} <= ${b}`)
  },

  throws(fn, msg) {
    let threw = false
    try { fn() } catch { threw = true }
    if (!threw) throw new Error(msg || 'Expected function to throw')
  },

  async asyncThrows(fn, msg) {
    let threw = false
    try { await fn() } catch { threw = true }
    if (!threw) throw new Error(msg || 'Expected async function to throw')
  },
}

// ─── Runner ─────────────────────────────────────────────────────────────────

function log(msg) {
  const el = document.getElementById('log')
  if (el) el.textContent += msg + '\n'
  console.log(msg)
}

export async function report() {
  passed = 0
  failed = 0

  for (const suite of suites) {
    log(`\n── ${suite.name} ──`)
    for (const test of suite.tests) {
      try {
        await test.fn()
        passed++
        log(`  ✓ ${test.name}`)
      } catch (e) {
        failed++
        log(`  ✗ ${test.name}`)
        log(`    ${e.message}`)
      }
    }
  }

  log(`\n=== RESULTS ===`)
  log(`${passed} passed, ${failed} failed`)
  return { passed, failed }
}
