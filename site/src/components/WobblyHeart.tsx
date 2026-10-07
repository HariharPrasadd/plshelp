import { useEffect, useRef } from 'react'
import rough from 'roughjs'
import { cachedShape, INK, pickInk, WOBBLE_STEP_MS } from '../lib/doodleLifecycle'
import styles from './WobblyHeart.module.css'

// how many wobbles (of WOBBLE_STEP_MS each) a color holds before cycling to
// the next - 8 wobbles ≈ 1360ms per color
const COLOR_CYCLE_WOBBLES = 8

const SIZE = 16

// one continuous closed curve traced from the classic parametric heart
// equation - a single boundary line, not several overlapping shapes, so at
// this tiny size it stays clean. roughness/bowing are kept very low (just
// enough for the wobble to read) rather than the normal doodle amount,
// which would be too chaotic at 16px
function heartPoints(scale: number): [number, number][] {
  const points: [number, number][] = []
  const steps = 48
  for (let i = 0; i <= steps; i++) {
    const t = (i / steps) * Math.PI * 2
    const x = 16 * Math.pow(Math.sin(t), 3)
    const y = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t))
    points.push([x * scale, y * scale])
  }
  return points
}

export function WobblyHeart() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = SIZE * dpr
    canvas.height = SIZE * dpr
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)

    const rc = rough.canvas(canvas)
    const owner = {}
    const points = heartPoints(0.34)

    let rafId: number
    let colorCycle = -1
    let color = INK
    const start = performance.now()

    const tick = () => {
      const wobbleStep = Math.floor((performance.now() - start) / WOBBLE_STEP_MS)

      const cycle = Math.floor(wobbleStep / COLOR_CYCLE_WOBBLES)
      if (cycle !== colorCycle) {
        colorCycle = cycle
        color = pickInk()
      }

      ctx.clearRect(0, 0, SIZE, SIZE)
      ctx.save()
      ctx.translate(SIZE / 2, SIZE / 2)

      cachedShape(rc, owner, 'heart', wobbleStep, () =>
        rc.generator.curve(points, { stroke: color, strokeWidth: 1, roughness: 0.3, bowing: 0.2, seed: wobbleStep }),
      )

      ctx.restore()
      rafId = requestAnimationFrame(tick)
    }
    rafId = requestAnimationFrame(tick)

    return () => cancelAnimationFrame(rafId)
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className={styles.heart}
      style={{ width: SIZE, height: SIZE }}
      aria-hidden="true"
    />
  )
}
