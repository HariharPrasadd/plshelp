import type { RoughCanvas } from 'roughjs/bin/canvas'
import type { Drawable } from 'roughjs/bin/core'
import { cachedShape, pickInk } from './doodleLifecycle'

const INTRO_MS = 550
const HOLD_MS = 2200
const OUTRO_MS = 500
const TOTAL_MS = INTRO_MS + HOLD_MS + OUTRO_MS

// how often the linework re-sketches itself in place - the "wobble"
const WOBBLE_STEP_MS = 170

// every shape here is built from ellipses/arcs/circles - only proportions,
// shear, and how many overlap change between them, so nothing ever has a
// hard corner or a point
type PetalShape = 'round' | 'thin' | 'wide' | 'curved' | 'teardrop' | 'notched' | 'cupped' | 'ruffled' | 'bell'
const PETAL_SHAPES: PetalShape[] = [
  'round',
  'thin',
  'wide',
  'curved',
  'teardrop',
  'notched',
  'cupped',
  'ruffled',
  'bell',
]

function rand(min: number, max: number) {
  return min + Math.random() * (max - min)
}

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

function easeOutCubic(t: number) {
  return 1 - Math.pow(1 - t, 3)
}

function easeInCubic(t: number) {
  return t * t * t
}

interface Petal {
  shape: PetalShape
  angle: number
  lenScale: number
  widthScale: number
  skew: number
  seed: number
}

interface Leaf {
  angle: number
  lenScale: number
  widthScale: number
  skew: number
  seed: number
}

export interface FlowerState {
  x: number
  y: number
  petalLen: number
  petalWidthRatio: number
  budRadius: number
  budCluster: boolean
  budPacked: boolean
  baseRotation: number
  idleSpeed: number
  spawnTime: number
  budSeed: number
  petals: Petal[]
  leaves: Leaf[]
  ink: string
}

export function createFlower(x: number, y: number): FlowerState {
  const petalCount = Math.floor(rand(4, 13))
  // most flowers are one species of petal, occasionally a mixed/wild one
  const uniformShape = Math.random() < 0.75 ? pick(PETAL_SHAPES) : null

  const petals: Petal[] = Array.from({ length: petalCount }, (_, i) => ({
    shape: uniformShape ?? pick(PETAL_SHAPES),
    angle: (i / petalCount) * Math.PI * 2,
    lenScale: rand(0.82, 1.18),
    widthScale: rand(0.8, 1.2),
    skew: rand(-0.22, 0.22),
    seed: Math.floor(rand(1, 100000)),
  }))

  const leafCount = Math.random() < 0.6 ? Math.floor(rand(2, 5)) : 0
  const leaves: Leaf[] = Array.from({ length: leafCount }, () => ({
    angle: rand(0, Math.PI * 2),
    lenScale: rand(0.7, 1.3),
    widthScale: rand(0.7, 1.3),
    skew: rand(-0.3, 0.3),
    seed: Math.floor(rand(1, 100000)),
  }))

  return {
    x,
    y,
    petalLen: rand(16, 38),
    petalWidthRatio: rand(0.42, 0.8),
    budRadius: rand(3, 8),
    budCluster: Math.random() < 0.3,
    // separate, proper "seeds packed into a disc" mode - independent of the
    // loose cluster above, which stays exactly as it was
    budPacked: Math.random() < 0.2,
    baseRotation: rand(0, Math.PI * 2),
    // radians/ms - very slow, lazy drift, full turn takes a minute or two
    idleSpeed: rand(0.00003, 0.00012) * (Math.random() < 0.5 ? -1 : 1),
    spawnTime: performance.now(),
    budSeed: Math.floor(rand(1, 100000)),
    petals,
    leaves,
    ink: pickInk(),
  }
}

function drawPetal(
  ctx: CanvasRenderingContext2D,
  rc: RoughCanvas,
  owner: object,
  keyPrefix: string,
  petal: Petal,
  budRadius: number,
  baseLen: number,
  baseWidth: number,
  wobbleStep: number,
  opts: object,
) {
  const len = baseLen * petal.lenScale
  const width = baseWidth * petal.widthScale
  const o = { ...opts, seed: petal.seed + wobbleStep }
  const draw = (key: string, generate: () => Drawable) => cachedShape(rc, owner, key, wobbleStep, generate)

  switch (petal.shape) {
    case 'round':
      draw(keyPrefix, () => rc.generator.ellipse(0, budRadius + len / 2, width, len, o))
      return
    case 'thin':
      draw(keyPrefix, () => rc.generator.ellipse(0, budRadius + len / 2, width * 0.5, len * 1.25, o))
      return
    case 'wide':
      draw(keyPrefix, () => rc.generator.ellipse(0, budRadius + len * 0.42, width * 1.3, len * 0.78, o))
      return
    case 'curved': {
      // same soft ellipse, just leaned over with a shear so it reads as a
      // gentle curve rather than a straight petal - no hard edges anywhere
      ctx.save()
      ctx.transform(1, 0, petal.skew, 1, 0, 0)
      draw(keyPrefix, () => rc.generator.ellipse(0, budRadius + len / 2, width, len, o))
      ctx.restore()
      return
    }
    case 'teardrop': {
      // two overlapping circles, big at the base tapering to small at the
      // tip - a soft taper with no straight edges at all
      const baseD = width
      const tipD = width * 0.5
      draw(`${keyPrefix}a`, () => rc.generator.circle(0, budRadius + baseD / 2, baseD, o))
      draw(`${keyPrefix}b`, () => rc.generator.circle(0, budRadius + len - tipD / 2, tipD, o))
      return
    }
    case 'notched': {
      // a short rounded base with two small lobes at the tip, like a
      // carnation or heart-shaped petal
      const lobeD = width * 0.55
      draw(`${keyPrefix}a`, () => rc.generator.ellipse(0, budRadius + len * 0.4, width, len * 0.75, o))
      draw(`${keyPrefix}b`, () => rc.generator.circle(-width * 0.26, budRadius + len - lobeD * 0.45, lobeD, o))
      draw(`${keyPrefix}c`, () => rc.generator.circle(width * 0.26, budRadius + len - lobeD * 0.45, lobeD, o))
      return
    }
    case 'cupped': {
      // two slightly offset, slightly rotated ellipses - reads as a folded,
      // layered petal rather than one flat shape
      draw(`${keyPrefix}a`, () => rc.generator.ellipse(0, budRadius + len / 2, width, len, o))
      ctx.save()
      ctx.rotate(petal.skew * 0.5)
      draw(`${keyPrefix}b`, () =>
        rc.generator.ellipse(width * 0.08, budRadius + len * 0.46, width * 0.8, len * 0.85, {
          ...o,
          seed: petal.seed + 1 + wobbleStep,
        }),
      )
      ctx.restore()
      return
    }
    case 'ruffled':
      // same silhouette as round, but a much shakier hand gives the outline
      // a crinkled, wavy-edged feel instead of a clean oval
      draw(keyPrefix, () => rc.generator.ellipse(0, budRadius + len / 2, width, len, { ...o, roughness: 2.6, bowing: 1.8 }))
      return
    case 'bell': {
      // a rounded scoop/cup shape via a closed arc, facing outward
      draw(keyPrefix, () =>
        rc.generator.arc(0, budRadius + len * 0.52, width, len * 0.95, Math.PI * 0.12, Math.PI * 0.88, true, o),
      )
      return
    }
  }
}

function drawLeaf(
  ctx: CanvasRenderingContext2D,
  rc: RoughCanvas,
  owner: object,
  key: string,
  leaf: Leaf,
  budRadius: number,
  wobbleStep: number,
  opts: object,
) {
  const len = (budRadius + 10) * leaf.lenScale * 2.2
  const width = len * 0.42 * leaf.widthScale
  const o = { ...opts, strokeWidth: 1, roughness: 0.9, seed: leaf.seed + wobbleStep }

  ctx.save()
  ctx.rotate(leaf.angle)
  ctx.transform(1, 0, leaf.skew, 1, 0, 0)
  cachedShape(rc, owner, key, wobbleStep, () => rc.generator.ellipse(0, budRadius * 0.5 + len / 2, width, len, o))
  ctx.restore()
}

function drawBud(
  rc: RoughCanvas,
  owner: object,
  budRadius: number,
  cluster: boolean,
  seed: number,
  wobbleStep: number,
  opts: object,
) {
  if (!cluster) {
    cachedShape(rc, owner, 'bud', wobbleStep, () => rc.generator.circle(0, 0, budRadius * 2, { ...opts, seed: seed + wobbleStep }))
    return
  }
  // a little cluster of tiny circles instead of one disc, like a
  // sunflower or daisy center
  const n = 6
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2
    const r = budRadius * 0.55
    cachedShape(rc, owner, `bud${i}`, wobbleStep, () =>
      rc.generator.circle(Math.cos(a) * r, Math.sin(a) * r, budRadius * 0.7, {
        ...opts,
        seed: seed + i * 7919 + wobbleStep,
      }),
    )
  }
  cachedShape(rc, owner, 'budCenter', wobbleStep, () => rc.generator.circle(0, 0, budRadius * 0.9, { ...opts, seed: seed + wobbleStep }))
}

// the "proper" packed version: small circles arranged on a sunflower-seed
// spiral (golden angle), sized so neighbors overlap enough to actually read
// as one dense disc instead of floating apart like the loose cluster above
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))

function drawBudPacked(
  rc: RoughCanvas,
  owner: object,
  budRadius: number,
  seed: number,
  wobbleStep: number,
  opts: object,
) {
  const n = 14
  const seedDiameter = budRadius * 0.62
  for (let i = 0; i < n; i++) {
    const a = i * GOLDEN_ANGLE
    const r = budRadius * Math.sqrt(i / n) * 0.75
    cachedShape(rc, owner, `seed${i}`, wobbleStep, () =>
      rc.generator.circle(Math.cos(a) * r, Math.sin(a) * r, seedDiameter, {
        ...opts,
        seed: seed + i * 7919 + wobbleStep,
      }),
    )
  }
}

// returns false once the flower's lifecycle is finished, so the caller can drop it
export function drawFlowerFrame(
  rc: RoughCanvas,
  canvas: HTMLCanvasElement,
  f: FlowerState,
  now: number,
): boolean {
  const ctx = canvas.getContext('2d')
  if (!ctx) return false

  const elapsed = now - f.spawnTime
  if (elapsed > TOTAL_MS) return false

  let scale: number
  let petalsVisible: number

  if (elapsed < INTRO_MS) {
    // growing into existence, strokes appearing one at a time
    const t = easeOutCubic(elapsed / INTRO_MS)
    scale = t
    petalsVisible = Math.ceil(t * f.petals.length)
  } else if (elapsed < INTRO_MS + HOLD_MS) {
    scale = 1
    petalsVisible = f.petals.length
  } else {
    // shrinking back out
    const t = easeInCubic((elapsed - INTRO_MS - HOLD_MS) / OUTRO_MS)
    scale = 1 - t
    petalsVisible = f.petals.length
  }

  // the shape doesn't translate, but a very slow lazy spin plus a wobbling
  // seed (stepped every WOBBLE_STEP_MS) keeps it feeling alive while still
  const idleRotation = f.idleSpeed * elapsed
  const wobbleStep = Math.floor(elapsed / WOBBLE_STEP_MS)

  // kept gentle: low roughness/bowing reads as a light hand, not a scribble.
  // no `fill` key at all - rough.js treats any string (even 'none') as a real
  // fill color to hachure with, which was the source of the messy infill
  const roughOpts = {
    stroke: f.ink,
    strokeWidth: 1.2,
    roughness: 0.9,
    bowing: 0.6,
  }

  ctx.save()
  ctx.translate(f.x, f.y)
  ctx.rotate(f.baseRotation + idleRotation)
  ctx.scale(scale, scale)

  // leaves sit behind the bloom
  const leafOpts = { ...roughOpts, strokeWidth: 1 }
  f.leaves.forEach((leaf, i) => {
    drawLeaf(ctx, rc, f, `leaf${i}`, leaf, f.budRadius, wobbleStep, leafOpts)
  })

  const budOpts = { ...roughOpts, strokeWidth: 1.5, roughness: 1.1 }
  if (f.budPacked) {
    drawBudPacked(rc, f, f.budRadius, f.budSeed, wobbleStep, budOpts)
  } else {
    drawBud(rc, f, f.budRadius, f.budCluster, f.budSeed, wobbleStep, budOpts)
  }

  const sector = ((Math.PI * 2) / f.petals.length) * 0.7
  const tipRadius = f.budRadius + f.petalLen
  const maxWidthByAngle = 2 * tipRadius * Math.sin(sector / 2)
  const width = Math.min(f.petalLen * f.petalWidthRatio, maxWidthByAngle)

  for (let i = 0; i < petalsVisible; i++) {
    const petal = f.petals[i]
    ctx.save()
    ctx.rotate(petal.angle)
    drawPetal(ctx, rc, f, `petal${i}`, petal, f.budRadius, f.petalLen, width, wobbleStep, roughOpts)
    ctx.restore()
  }

  ctx.restore()
  return true
}
