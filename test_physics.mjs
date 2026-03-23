/**
 * Automated physics test runner.
 * Launches headless Chrome, loads test_damping.html via Vite dev server,
 * captures the output, and reports pass/fail.
 */
import puppeteer from 'puppeteer'

const VITE_URL = 'http://localhost:5173/test_damping.html'
const TIMEOUT = 30000

async function run() {
  console.log('Launching headless browser...')
  const browser = await puppeteer.launch({ headless: true, args: ['--no-sandbox'] })
  const page = await browser.newPage()

  const logs = []
  page.on('console', msg => {
    const text = msg.text()
    logs.push(text)
  })

  page.on('pageerror', err => {
    logs.push(`PAGE ERROR: ${err.message}`)
  })

  console.log(`Loading ${VITE_URL}...`)
  await page.goto(VITE_URL, { waitUntil: 'domcontentloaded' })

  // Wait for the test to complete (look for "=== DONE ===" or "=== RESULTS")
  const startTime = Date.now()
  while (Date.now() - startTime < TIMEOUT) {
    const content = await page.$eval('#log', el => el.textContent).catch(() => '')
    if (content.includes('=== RESULTS') || content.includes('=== DONE')) {
      break
    }
    await new Promise(r => setTimeout(r, 500))
  }

  const finalOutput = await page.$eval('#log', el => el.textContent).catch(() => '')
  await browser.close()

  console.log('\n' + finalOutput)

  // Parse results
  const match = finalOutput.match(/(\d+) passed, (\d+) failed/)
  if (match) {
    const [, p, f] = match
    console.log(`\n${f === '0' ? 'ALL TESTS PASSED' : `${f} TESTS FAILED`}`)
    process.exit(f === '0' ? 0 : 1)
  } else if (finalOutput.includes('ERROR')) {
    console.log('\nTEST ERROR — see output above')
    process.exit(1)
  } else {
    console.log('\nTEST TIMEOUT — no results found')
    process.exit(1)
  }
}

run().catch(e => {
  console.error(e)
  process.exit(1)
})
