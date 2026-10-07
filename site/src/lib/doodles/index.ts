import type { RoughCanvas } from 'roughjs/bin/canvas'
import { createFlower, drawFlowerFrame, type FlowerState } from '../roughFlower'
import { createBird, drawBird, type BirdState } from './bird'
import { createMoon, drawMoon, type MoonState } from './moon'
import { createSun, drawSun, type SunState } from './sun'
import { createPlanet, drawPlanet, type PlanetState } from './planet'

export type Doodle =
  | { kind: 'flower'; state: FlowerState }
  | { kind: 'bird'; state: BirdState }
  | { kind: 'moon'; state: MoonState }
  | { kind: 'sun'; state: SunState }
  | { kind: 'planet'; state: PlanetState }

// weight controls how often each species gets picked - higher spawns more often
const CREATORS: Array<{ weight: number; create: (x: number, y: number) => Doodle }> = [
  { weight: 2.5, create: (x, y) => ({ kind: 'flower', state: createFlower(x, y) }) },
  { weight: 1, create: (x, y) => ({ kind: 'bird', state: createBird(x, y) }) },
  { weight: 0.7, create: (x, y) => ({ kind: 'moon', state: createMoon(x, y) }) },
  { weight: 1, create: (x, y) => ({ kind: 'sun', state: createSun(x, y) }) },
  { weight: 0.7, create: (x, y) => ({ kind: 'planet', state: createPlanet(x, y) }) },
]

const TOTAL_WEIGHT = CREATORS.reduce((sum, c) => sum + c.weight, 0)

export function createDoodle(x: number, y: number): Doodle {
  let roll = Math.random() * TOTAL_WEIGHT
  for (const entry of CREATORS) {
    roll -= entry.weight
    if (roll <= 0) return entry.create(x, y)
  }
  return CREATORS[0].create(x, y)
}

// returns false once the doodle's lifecycle is finished, so the caller can drop it
export function drawDoodleFrame(
  rc: RoughCanvas,
  canvas: HTMLCanvasElement,
  doodle: Doodle,
  now: number,
): boolean {
  const ctx = canvas.getContext('2d')
  if (!ctx) return false

  switch (doodle.kind) {
    case 'flower':
      return drawFlowerFrame(rc, canvas, doodle.state, now)
    case 'bird':
      return drawBird(ctx, rc, doodle.state, now)
    case 'moon':
      return drawMoon(ctx, rc, doodle.state, now)
    case 'sun':
      return drawSun(ctx, rc, doodle.state, now)
    case 'planet':
      return drawPlanet(ctx, rc, doodle.state, now)
  }
}
