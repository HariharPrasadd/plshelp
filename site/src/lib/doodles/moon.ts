import type { RoughCanvas } from 'roughjs/bin/canvas'
import { BASE_OPTS, cachedShape, computeLifecycle, pick, pickInk, rand } from '../doodleLifecycle'

type Phase = 'crescent' | 'half' | 'full'

export interface MoonState {
  x: number
  y: number
  baseRotation: number
  idleSpeed: number
  spawnTime: number
  phase: Phase
  r: number
  offset: number
  stars: { dx: number; dy: number; d: number; seed: number }[]
  ink: string
  seed: number
}

export function createMoon(x: number, y: number): MoonState {
  const phase = pick<Phase>(['crescent', 'half', 'full'])
  const r = rand(12, 24)
  const starCount = Math.floor(rand(0, 5))
  const offsetRatio = phase === 'crescent' ? rand(0.75, 1.05) : phase === 'half' ? rand(0.4, 0.55) : 0

  return {
    x,
    y,
    baseRotation: rand(0, Math.PI * 2),
    idleSpeed: rand(0.00002, 0.00008) * (Math.random() < 0.5 ? -1 : 1),
    spawnTime: performance.now(),
    phase,
    r,
    offset: r * offsetRatio,
    stars: Array.from({ length: starCount }, () => ({
      dx: rand(-r * 2.6, r * 2.6),
      dy: rand(-r * 2.6, r * 2.6),
      d: rand(1.5, 3.2),
      seed: Math.floor(rand(1, 100000)),
    })),
    ink: pickInk(),
    seed: Math.floor(rand(1, 100000)),
  }
}

export function drawMoon(
  ctx: CanvasRenderingContext2D,
  rc: RoughCanvas,
  f: MoonState,
  now: number,
): boolean {
  const { scale, revealT, wobbleStep, alive } = computeLifecycle(now - f.spawnTime)
  if (!alive) return false

  const bodyParts = f.phase === 'full' ? 1 : 2
  const parts = Math.ceil(revealT * (bodyParts + f.stars.length))
  const opts = { ...BASE_OPTS, stroke: f.ink }

  ctx.save()
  ctx.translate(f.x, f.y)
  ctx.rotate(f.baseRotation)
  ctx.scale(scale, scale)

  // the main disc
  if (parts > 0) {
    cachedShape(rc, f, 'body', wobbleStep, () => rc.generator.circle(0, 0, f.r * 2, { ...opts, seed: f.seed + wobbleStep }))
  }

  // crescent/half: a smaller displaced circle overlapping the main disc,
  // their outlines alone read as a phase in line art. full moons skip this.
  if (f.phase !== 'full' && parts > 1) {
    cachedShape(rc, f, 'shadow', wobbleStep, () =>
      rc.generator.circle(f.offset, -f.offset * 0.3, f.r * 1.7, {
        ...opts,
        seed: f.seed + 7919 + wobbleStep,
      }),
    )
  }

  for (let i = 0; i < f.stars.length && parts > i + bodyParts; i++) {
    const star = f.stars[i]
    cachedShape(rc, f, `star${i}`, wobbleStep, () =>
      rc.generator.circle(star.dx, star.dy, star.d, {
        ...opts,
        strokeWidth: 0.9,
        seed: star.seed + wobbleStep,
      }),
    )
  }

  ctx.restore()
  return true
}
