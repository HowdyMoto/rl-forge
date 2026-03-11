import { useEffect, useRef } from 'react'
import { CARTPOLE_PARAMS } from '../env/cartpole.js'

const COLORS = {
  bg: '#0a0a0f',
  ground: '#1a1a2e',
  groundLine: '#2a2a4e',
  track: '#16213e',
  cart: '#e2b96f',
  cartHighlight: '#f0d090',
  pole: '#e05a5a',
  poleHighlight: '#ff7070',
  wheel: '#444466',
  axle: '#888899',
  shadow: 'rgba(0,0,0,0.4)',
  grid: 'rgba(255,255,255,0.03)',
}

export default function CartPoleRenderer({ state, episodeReward, episodeSteps, isRunning }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const W = canvas.width
    const H = canvas.height

    const scale = W / (CARTPOLE_PARAMS.xThreshold * 2.5)
    const groundY = H * 0.7
    const cartW = 60
    const cartH = 28
    const poleLen = CARTPOLE_PARAMS.poleHalfLen * 2 * scale
    const wheelR = 9

    ctx.clearRect(0, 0, W, H)

    // Background
    ctx.fillStyle = COLORS.bg
    ctx.fillRect(0, 0, W, H)

    // Subtle grid
    ctx.strokeStyle = COLORS.grid
    ctx.lineWidth = 1
    for (let x = 0; x < W; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
    }
    for (let y = 0; y < H; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
    }

    // Ground
    const groundGrad = ctx.createLinearGradient(0, groundY, 0, H)
    groundGrad.addColorStop(0, '#1a1a2e')
    groundGrad.addColorStop(1, '#0d0d1a')
    ctx.fillStyle = groundGrad
    ctx.fillRect(0, groundY, W, H - groundY)

    // Ground line
    ctx.strokeStyle = COLORS.groundLine
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(0, groundY)
    ctx.lineTo(W, groundY)
    ctx.stroke()

    // Track bounds indicator
    const leftBound = W / 2 + (-CARTPOLE_PARAMS.xThreshold) * scale
    const rightBound = W / 2 + CARTPOLE_PARAMS.xThreshold * scale
    ctx.setLineDash([4, 6])
    ctx.strokeStyle = 'rgba(255,80,80,0.25)'
    ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(leftBound, groundY - 80); ctx.lineTo(leftBound, groundY); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(rightBound, groundY - 80); ctx.lineTo(rightBound, groundY); ctx.stroke()
    ctx.setLineDash([])

    if (!state) {
      // Idle state - just show ground
      ctx.fillStyle = 'rgba(255,255,255,0.1)'
      ctx.font = '500 14px "DM Mono", monospace'
      ctx.textAlign = 'center'
      ctx.fillText('Press TRAIN to begin', W / 2, groundY / 2)
      return
    }

    const [x, , theta] = state
    const cartX = W / 2 + x * scale
    const cartY = groundY - cartH / 2 - wheelR

    // Cart shadow
    ctx.fillStyle = COLORS.shadow
    ctx.beginPath()
    ctx.ellipse(cartX, groundY + 4, cartW / 2, 6, 0, 0, Math.PI * 2)
    ctx.fill()

    // Pole
    const poleBaseX = cartX
    const poleBaseY = cartY - cartH / 2
    const poleTipX = poleBaseX + Math.sin(theta) * poleLen
    const poleTipY = poleBaseY - Math.cos(theta) * poleLen

    // Pole glow
    const normalizedAngle = Math.abs(theta) / CARTPOLE_PARAMS.thetaThreshold
    const danger = Math.min(normalizedAngle, 1)
    const poleColor = `rgb(${Math.round(180 + 75 * danger)},${Math.round(90 - 60 * danger)},${Math.round(90 - 60 * danger)})`

    ctx.strokeStyle = poleColor
    ctx.lineWidth = 8
    ctx.lineCap = 'round'
    ctx.shadowColor = poleColor
    ctx.shadowBlur = 12 * danger
    ctx.beginPath()
    ctx.moveTo(poleBaseX, poleBaseY)
    ctx.lineTo(poleTipX, poleTipY)
    ctx.stroke()

    // Pole highlight
    ctx.strokeStyle = 'rgba(255,255,255,0.3)'
    ctx.lineWidth = 2
    ctx.shadowBlur = 0
    ctx.beginPath()
    ctx.moveTo(poleBaseX, poleBaseY)
    ctx.lineTo(poleTipX, poleTipY)
    ctx.stroke()

    // Pole tip ball
    ctx.fillStyle = COLORS.poleHighlight
    ctx.shadowColor = COLORS.poleHighlight
    ctx.shadowBlur = 8
    ctx.beginPath()
    ctx.arc(poleTipX, poleTipY, 5, 0, Math.PI * 2)
    ctx.fill()
    ctx.shadowBlur = 0

    // Cart body
    const cartGrad = ctx.createLinearGradient(cartX - cartW/2, cartY - cartH/2, cartX - cartW/2, cartY + cartH/2)
    cartGrad.addColorStop(0, COLORS.cartHighlight)
    cartGrad.addColorStop(1, COLORS.cart)
    ctx.fillStyle = cartGrad
    ctx.shadowColor = 'rgba(226,185,111,0.3)'
    ctx.shadowBlur = 10
    const r = 5
    ctx.beginPath()
    ctx.roundRect(cartX - cartW / 2, cartY - cartH / 2, cartW, cartH, r)
    ctx.fill()
    ctx.shadowBlur = 0

    // Cart outline
    ctx.strokeStyle = 'rgba(255,255,255,0.15)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.roundRect(cartX - cartW / 2, cartY - cartH / 2, cartW, cartH, r)
    ctx.stroke()

    // Wheels
    for (const wx of [cartX - cartW / 3, cartX + cartW / 3]) {
      ctx.fillStyle = COLORS.wheel
      ctx.beginPath()
      ctx.arc(wx, groundY - wheelR, wheelR, 0, Math.PI * 2)
      ctx.fill()
      ctx.fillStyle = COLORS.axle
      ctx.beginPath()
      ctx.arc(wx, groundY - wheelR, 3, 0, Math.PI * 2)
      ctx.fill()
    }

    // HUD
    ctx.font = '600 11px "DM Mono", monospace'
    ctx.textAlign = 'left'
    ctx.fillStyle = 'rgba(255,255,255,0.4)'
    ctx.fillText(`θ ${(theta * 180 / Math.PI).toFixed(1)}°`, 12, H - 28)
    ctx.fillText(`x ${x.toFixed(2)}m`, 12, H - 12)
    if (episodeSteps !== undefined) {
      ctx.textAlign = 'right'
      ctx.fillText(`t ${episodeSteps}`, W - 12, H - 12)
      ctx.fillText(`r ${(episodeReward || 0).toFixed(1)}`, W - 12, H - 28)
    }

  }, [state, episodeReward, episodeSteps])

  return (
    <canvas
      ref={canvasRef}
      width={500}
      height={260}
      style={{ width: '100%', height: '100%', display: 'block' }}
    />
  )
}
