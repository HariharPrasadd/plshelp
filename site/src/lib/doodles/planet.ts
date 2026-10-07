import type { RoughCanvas } from 'roughjs/bin/canvas'
import { BASE_OPTS, cachedShape, computeLifecycle, pickInk, rand } from '../doodleLifecycle'

export interface PlanetState {
  x: number
  y: number
  baseRotation: number
  idleSpeed: number
  spawnTime: number
  r: number
  hasRing: boolean
  ringTilt: number
  banded: boolean
  moons: { angle: number; dist: number }[]
  ink: string
  seed: number
}

export function createPlanet(x: number, y: number): PlanetState {
  const moonCount = Math.floor(rand(0, 3))
  return {
    x,
    y,
    baseRotation: rand(-0.5, 0.5),
    idleSpeed: rand(0.00002, 0.00007) * (Math.random() < 0.5 ? -1 : 1),
    spawnTime: performance.now(),
    r: rand(8, 17),
    hasRing: Math.random() < 0.6,
    ringTilt: rand(0.5, 1.4),
    banded: Math.random() < 0.45,
    moons: Array.from({ length: moonCount }, (_, i) => ({
      angle: (i / Math.max(1, moonCount)) * Math.PI * 2 + 0.6,
      dist: rand(2.6, 3.4),
    })),
    ink: pickInk(),
    seed: Math.floor(rand(1, 100000)),
  }
}

export function drawPlanet(
  ctx: CanvasRenderingContext2D,
  rc: RoughCanvas,
  f: PlanetState,
  now: number,
): boolean {
  const { scale, revealT, wobbleStep, alive } = computeLifecycle(now - f.spawnTime)
  if (!alive) return false

  const bandCount = f.banded ? 3 : 0
  const parts = Math.ceil(revealT * (2 + bandCount + f.moons.length))
  const opts = { ...BASE_OPTS, stroke: f.ink }

  ctx.save()
  ctx.translate(f.x, f.y)
  ctx.rotate(f.baseRotation)
  ctx.scale(scale, scale)

  // ring, behind the body - tilt (height ratio) varies per planet
  if (parts > 0 && f.hasRing) {
    cachedShape(rc, f, 'ring', wobbleStep, () =>
      rc.generator.ellipse(0, 0, f.r * 3.1, f.r * f.ringTilt, { ...opts, strokeWidth: 1, seed: f.seed + wobbleStep }),
    )
  }

  // body
  if (parts > 1) {
    cachedShape(rc, f, 'body', wobbleStep, () =>
      rc.generator.circle(0, 0, f.r * 2, { ...opts, roughness: 1.1, seed: f.seed + 7919 + wobbleStep }),
    )
  }

  // gas-giant bands: a few arcs across the body
  for (let i = 0; i < bandCount && parts > i + 2; i++) {
    const t = (i + 1) / (bandCount + 1)
    const y = (t - 0.5) * f.r * 1.7
    const w = Math.sqrt(Math.max(0.05, 1 - (t - 0.5) * (t - 0.5) * 4)) * f.r * 1.9
    cachedShape(rc, f, `band${i}`, wobbleStep, () =>
      rc.generator.line(-w / 2, y, w / 2, y, {
        ...opts,
        strokeWidth: 0.9,
        seed: f.seed + i * 7919 + 2 + wobbleStep,
      }),
    )
  }

  // little moons orbiting nearby
  for (let i = 0; i < f.moons.length && parts > i + 2 + bandCount; i++) {
    const moon = f.moons[i]
    const dist = f.r * moon.dist
    cachedShape(rc, f, `moon${i}`, wobbleStep, () =>
      rc.generator.circle(Math.cos(moon.angle) * dist, Math.sin(moon.angle) * dist, f.r * 0.4, {
        ...opts,
        strokeWidth: 0.9,
        seed: f.seed + i * 7919 + 1 + wobbleStep,
      }),
    )
  }

  ctx.restore()
  return true
}
