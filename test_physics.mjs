/**
 * Automated physics test runner.
 *
 * Discovers all tests/test_*.html files, launches headless Chrome via
 * Puppeteer, runs each test page against the Vite dev server, and
 * aggregates results.
 *
 * Usage:
 *   1. Start Vite dev server:  npm run dev
 *   2. Run tests:              npm test
 */
import puppeteer from 'puppeteer'
import { readdirSync } from 'fs'
import { resolve } from 'path'

const VITE_BASE = 'http://localhost:5173'
const TIMEOUT = 30_000
const TESTS_DIR = resolve(import.meta.dirname, 'tests')

// ─── Discover test pages ────────────────────────────────────────────────────

function discoverTestPages() {
  const files = readdirSync(TESTS_DIR)
    .filter(f => f.startsWith('test_') && f.endsWith('.html'))
    .sort()
  return files.map(f => ({
    name: f.replace('.html', ''),
    url: `${VITE_BASE}/tests/${f}`,
  }))
}

// ─── Run a single test page ─────────────────────────────────────────────────

async function runPage(browser, { name, url }) {
  const page = await browser.newPage()
  const errors = []

  page.on('pageerror', err => errors.push(err.message))

  try {
    await page.goto(url, { waitUntil: 'domcontentloaded' })
  } catch (e) {
    await page.close()
    return { name, passed: 0, failed: 1, error: `Failed to load: ${e.message}` }
  }

  // Wait for test completion
  const startTime = Date.now()
  while (Date.now() - startTime < TIMEOUT) {
    const content = await page.$eval('#log', el => el.textContent).catch(() => '')
    if (content.includes('=== RESULTS') || content.includes('=== DONE')) break
    await new Promise(r => setTimeout(r, 300))
  }

  const output = await page.$eval('#log', el => el.textContent).catch(() => '')
  await page.close()

  // Parse "X passed, Y failed"
  const match = output.match(/(\d+) passed, (\d+) failed/)
  if (match) {
    return { name, passed: +match[1], failed: +match[2], output }
  }

  if (errors.length) {
    return { name, passed: 0, failed: 1, error: `Page errors: ${errors.join('; ')}`, output }
  }

  return { name, passed: 0, failed: 1, error: 'Timeout — no results found', output }
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  const pages = discoverTestPages()
  if (pages.length === 0) {
    console.log('No test pages found in tests/')
    process.exit(1)
  }

  console.log(`Found ${pages.length} test page(s):\n`)
  for (const p of pages) console.log(`  ${p.name}`)
  console.log('')

  // Verify Vite is running
  try {
    const resp = await fetch(VITE_BASE)
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  } catch {
    console.error('ERROR: Vite dev server not running on port 5173.')
    console.error('Start it first:  npm run dev')
    process.exit(1)
  }

  console.log('Launching headless browser...\n')
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--enable-unsafe-webgpu'],
  })

  let totalPassed = 0
  let totalFailed = 0
  const results = []

  for (const page of pages) {
    process.stdout.write(`  ${page.name} ... `)
    const result = await runPage(browser, page)
    results.push(result)

    totalPassed += result.passed
    totalFailed += result.failed

    if (result.error) {
      console.log(`ERROR (${result.error})`)
    } else if (result.failed > 0) {
      console.log(`${result.passed} passed, ${result.failed} FAILED`)
    } else {
      console.log(`${result.passed} passed`)
    }
  }

  await browser.close()

  // ─── Summary ────────────────────────────────────────────────────────────

  console.log('\n' + '═'.repeat(50))
  console.log(`TOTAL: ${totalPassed} passed, ${totalFailed} failed`)
  console.log('═'.repeat(50))

  // Print failure details
  const failedPages = results.filter(r => r.failed > 0)
  if (failedPages.length > 0) {
    console.log('\nFailed test output:\n')
    for (const r of failedPages) {
      console.log(`── ${r.name} ──`)
      if (r.output) {
        // Print only failing lines
        const lines = r.output.split('\n')
        for (const line of lines) {
          if (line.includes('✗') || line.trim().startsWith('Expected') || line.trim().startsWith('Element')) {
            console.log(`  ${line}`)
          }
        }
      }
      if (r.error) console.log(`  ${r.error}`)
      console.log('')
    }
  }

  process.exit(totalFailed === 0 ? 0 : 1)
}

main().catch(e => {
  console.error(e)
  process.exit(1)
})
