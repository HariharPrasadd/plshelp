import type { RoughCanvas } from 'roughjs/bin/canvas'
import { BASE_OPTS, cachedShape, computeLifecycle, pick, pickInk, rand } from '../doodleLifecycle'

type RayStyle = 'rays' | 'dotted' | 'alternating'

export interface SunState {
  x: number
  y: number
  baseRotation: number
  idleSpeed: number
  spawnTime: number
  r: number
  rayCount: number
  rayStyle: RayStyle
  ink: string
  seed: number
}

export function createSun(x: number, y: number): SunState {
  return {
    x,
    y,
    baseRotation: rand(0, Math.PI * 2),
    idleSpeed: rand(0.00004, 0.00014) * (Math.random() < 0.5 ? -1 : 1),
    spawnTime: performance.now(),
    r: rand(7, 15),
    rayCount: Math.floor(rand(7, 14)),
    rayStyle: pick<RayStyle>(['rays', 'dotted', 'alternating']),
    ink: pickInk(),
    seed: Math.floor(rand(1, 100000)),
  }
}

export function drawSun(
  ctx: CanvasRenderingContext2D,
  rc: RoughCanvas,
  f: SunState,
  now: number,
): boolean {
  const { scale, revealT, wobbleStep, alive } = computeLifecycle(now - f.spawnTime)
  if (!alive) return false

  const raysVisible = Math.ceil(revealT * f.rayCount)
  const opts = { ...BASE_OPTS, stroke: f.ink }

  ctx.save()
  ctx.translate(f.x, f.y)
  ctx.rotate(f.baseRotation)
  ctx.scale(scale, scale)

  const rayLen = f.r * 1.3
  for (let i = 0; i < raysVisible; i++) {
    const angle = (i / f.rayCount) * Math.PI * 2
    const short = f.rayStyle === 'alternating' && i % 2 === 1
    const len = short ? rayLen * 0.55 : rayLen
    ctx.save()
    ctx.rotate(angle)

    if (f.rayStyle === 'dotted') {
      cachedShape(rc, f, `ray${i}`, wobbleStep, () =>
        rc.generator.circle(0, f.r + len * 0.4, f.r * 0.3, {
          ...opts,
          strokeWidth: 1,
          seed: f.seed + i * 7919 + wobbleStep,
        }),
      )
    } else {
      cachedShape(rc, f, `ray${i}`, wobbleStep, () =>
        rc.generator.ellipse(0, f.r + len / 2, f.r * 0.22, len, {
          ...opts,
          strokeWidth: 1,
          seed: f.seed + i * 7919 + wobbleStep,
        }),
      )
    }
    ctx.restore()
  }

  // the sun itself, drawn last so it sits on top of the ray bases
  cachedShape(rc, f, 'core', wobbleStep, () =>
    rc.generator.circle(0, 0, f.r * 2, { ...opts, strokeWidth: 1.4, roughness: 1.1, seed: f.seed + wobbleStep }),
  )

  ctx.restore()
  return true
}
