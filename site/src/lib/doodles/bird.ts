import type { RoughCanvas } from 'roughjs/bin/canvas'
import type { Drawable } from 'roughjs/bin/core'
import { BASE_OPTS, cachedShape, computeLifecycle, pick, pickInk, rand } from '../doodleLifecycle'

type Pose = 'perched' | 'flying' | 'resting' | 'soaring'
type BodyShape = 'plump' | 'slender' | 'longNecked'
type BeakShape = 'short' | 'pointed' | 'hooked' | 'stubby' | 'duckbill'
type WingShape = 'rounded' | 'pointed' | 'feathered' | 'folded'
type WingMode = 'single' | 'spread'
type TailShape = 'fan' | 'forked' | 'stubby' | 'streamer'

export interface BirdState {
  x: number
  y: number
  baseRotation: number
  facing: 1 | -1
  idleSpeed: number
  spawnTime: number
  pose: Pose
  bodyShape: BodyShape
  beakShape: BeakShape
  wingShape: WingShape
  wingMode: WingMode
  tailShape: TailShape
  bodyLen: number
  beakLen: number
  hasLegs: boolean
  hasCrest: boolean
  wingAngle: number
  ink: string
  seed: number
}

export function createBird(x: number, y: number): BirdState {
  const pose = pick<Pose>(['perched', 'flying', 'resting', 'soaring'])
  return {
    x,
    y,
    // perched/resting birds stay close to upright, flying/soaring ones can tilt
    // a lot more since they're banking through the air
    baseRotation: pose === 'flying' || pose === 'soaring' ? rand(-0.9, 0.9) : rand(-0.25, 0.25),
    facing: Math.random() < 0.5 ? 1 : -1,
    idleSpeed: rand(0.00003, 0.00012) * (Math.random() < 0.5 ? -1 : 1),
    spawnTime: performance.now(),
    pose,
    bodyShape: pick<BodyShape>(['plump', 'slender', 'longNecked']),
    beakShape: pick<BeakShape>(['short', 'pointed', 'hooked', 'stubby', 'duckbill']),
    wingShape: pick<WingShape>(['rounded', 'pointed', 'feathered', 'folded']),
    wingMode: pose === 'flying' || pose === 'soaring' ? pick<WingMode>(['single', 'spread']) : 'single',
    tailShape: pick<TailShape>(['fan', 'forked', 'stubby', 'streamer']),
    bodyLen: rand(18, 36),
    beakLen: rand(0.7, 1.3),
    hasLegs: pose === 'perched' && Math.random() < 0.7,
    hasCrest: Math.random() < 0.3,
    wingAngle: pose === 'flying' ? rand(-0.9, -0.5) : pose === 'soaring' ? rand(-0.25, -0.05) : rand(-0.3, -0.1),
    ink: pickInk(),
    seed: Math.floor(rand(1, 100000)),
  }
}

function drawBeak(
  rc: RoughCanvas,
  owner: object,
  shape: BeakShape,
  headCx: number,
  headR: number,
  len: number,
  wobbleStep: number,
  opts: object,
) {
  const base = headCx - headR * 0.9
  const draw = (key: string, generate: () => Drawable) => cachedShape(rc, owner, key, wobbleStep, generate)

  switch (shape) {
    case 'pointed': {
      const tip = headCx - headR * (1.5 + len * 1.2)
      draw('beak', () =>
        rc.generator.polygon(
          [
            [base, -headR * 0.28],
            [tip, 0],
            [base, headR * 0.28],
          ],
          opts,
        ),
      )
      return
    }
    case 'hooked': {
      const tip = headCx - headR * (1.3 + len)
      draw('beak', () =>
        rc.generator.polygon(
          [
            [base, -headR * 0.3],
            [tip, -headR * 0.05],
            [tip + headR * 0.25, headR * 0.3],
            [base, headR * 0.22],
          ],
          opts,
        ),
      )
      return
    }
    case 'stubby': {
      const tip = headCx - headR * (1.1 + len * 0.4)
      draw('beak', () =>
        rc.generator.polygon(
          [
            [base, -headR * 0.4],
            [tip, 0],
            [base, headR * 0.4],
          ],
          opts,
        ),
      )
      return
    }
    case 'duckbill':
      draw('beak', () =>
        rc.generator.ellipse(headCx - headR * (0.9 + len * 0.5), 0, headR * 1.1 * len, headR * 0.34, opts),
      )
      return
    default:
      draw('beak', () =>
        rc.generator.ellipse(headCx - headR * (0.9 + len * 0.4), 0, headR * 0.7 * len, headR * 0.4, opts),
      )
  }
}

function drawWing(
  ctx: CanvasRenderingContext2D,
  rc: RoughCanvas,
  owner: object,
  keyPrefix: string,
  shape: WingShape,
  bodyWidth: number,
  bodyLen: number,
  seedBase: number,
  wobbleStep: number,
  opts: { stroke: string; strokeWidth: number; roughness: number; bowing: number },
) {
  const s = (n: number) => seedBase + n * 7919 + wobbleStep
  switch (shape) {
    case 'pointed':
      cachedShape(rc, owner, keyPrefix, wobbleStep, () =>
        rc.generator.polygon(
          [
            [0, -bodyWidth * 0.15],
            [bodyLen * 0.7, -bodyWidth * 0.05],
            [0, bodyWidth * 0.25],
          ],
          { ...opts, seed: s(0) },
        ),
      )
      return
    case 'feathered':
      for (let i = 0; i < 3; i++) {
        ctx.save()
        ctx.rotate(i * 0.12 - 0.12)
        cachedShape(rc, owner, `${keyPrefix}f${i}`, wobbleStep, () =>
          rc.generator.polygon(
            [
              [0, -bodyWidth * 0.08],
              [bodyLen * (0.45 + i * 0.1), 0],
              [0, bodyWidth * 0.1],
            ],
            { ...opts, seed: s(i + 1) },
          ),
        )
        ctx.restore()
      }
      return
    case 'folded':
      cachedShape(rc, owner, keyPrefix, wobbleStep, () =>
        rc.generator.ellipse(0, -bodyWidth * 0.1, bodyWidth * 0.3, bodyLen * 0.3, { ...opts, seed: s(0) }),
      )
      return
    default:
      cachedShape(rc, owner, keyPrefix, wobbleStep, () =>
        rc.generator.ellipse(0, -bodyWidth * 0.1, bodyWidth * 0.55, bodyLen * 0.55, { ...opts, seed: s(0) }),
      )
  }
}

function drawTail(
  rc: RoughCanvas,
  owner: object,
  shape: TailShape,
  bodyWidth: number,
  bodyLen: number,
  wobbleStep: number,
  opts: object,
) {
  switch (shape) {
    case 'forked':
      cachedShape(rc, owner, 'tail', wobbleStep, () =>
        rc.generator.polygon(
          [
            [0, -bodyWidth * 0.3],
            [bodyLen * 0.75, -bodyWidth * 0.05],
            [bodyLen * 0.45, 0],
            [bodyLen * 0.75, bodyWidth * 0.05],
            [0, bodyWidth * 0.3],
          ],
          opts,
        ),
      )
      return
    case 'stubby':
      cachedShape(rc, owner, 'tail', wobbleStep, () =>
        rc.generator.ellipse(bodyLen * 0.2, 0, bodyWidth * 0.3, bodyLen * 0.3, opts),
      )
      return
    case 'streamer':
      cachedShape(rc, owner, 'tail', wobbleStep, () =>
        rc.generator.polygon(
          [
            [0, -bodyWidth * 0.1],
            [bodyLen * 1.1, 0],
            [0, bodyWidth * 0.1],
          ],
          opts,
        ),
      )
      return
    default:
      cachedShape(rc, owner, 'tail', wobbleStep, () =>
        rc.generator.ellipse(bodyLen * 0.3, 0, bodyWidth * 0.35, bodyLen * 0.6, opts),
      )
  }
}

export function drawBird(
  ctx: CanvasRenderingContext2D,
  rc: RoughCanvas,
  f: BirdState,
  now: number,
): boolean {
  const { scale, revealT, wobbleStep, alive } = computeLifecycle(now - f.spawnTime)
  if (!alive) return false

  const totalParts = 5 + (f.hasLegs ? 1 : 0) + (f.hasCrest ? 1 : 0)
  const parts = Math.ceil(revealT * totalParts)
  const s = (n: number) => f.seed + n * 7919 + wobbleStep

  const bodyLen = f.bodyLen
  const bodyWidth =
    bodyLen *
    (f.bodyShape === 'plump' ? 0.72 : f.bodyShape === 'slender' ? 0.46 : 0.5) *
    (f.pose === 'resting' ? 1.1 : 1)
  const headR = bodyWidth * (f.bodyShape === 'slender' ? 0.48 : 0.4)
  const neckLen = f.bodyShape === 'longNecked' ? bodyLen * 0.5 : 0

  const opts = { ...BASE_OPTS, stroke: f.ink }

  ctx.save()
  ctx.translate(f.x, f.y)
  ctx.rotate(f.baseRotation)
  ctx.scale(scale * f.facing, scale)

  let step = 0

  // tail, drawn behind the body
  if (parts > step++) {
    ctx.save()
    ctx.translate(bodyLen * 0.5, 0)
    ctx.rotate(0.25)
    drawTail(rc, f, f.tailShape, bodyWidth, bodyLen, wobbleStep, { ...opts, seed: s(0) })
    ctx.restore()
  }

  // wing(s)
  if (parts > step++) {
    if (f.wingMode === 'spread') {
      for (const side of [-1, 1]) {
        ctx.save()
        ctx.rotate(side * (0.5 + Math.abs(f.wingAngle)))
        drawWing(
          ctx,
          rc,
          f,
          side === -1 ? 'wingL' : 'wingR',
          f.wingShape,
          bodyWidth,
          bodyLen,
          s(side === -1 ? 10 : 30),
          wobbleStep,
          opts,
        )
        ctx.restore()
      }
    } else {
      ctx.save()
      ctx.rotate(f.wingAngle)
      drawWing(ctx, rc, f, 'wing', f.wingShape, bodyWidth, bodyLen, s(10), wobbleStep, opts)
      ctx.restore()
    }
  }

  // body
  if (parts > step++) {
    ctx.save()
    ctx.rotate(Math.PI / 2 + (f.pose === 'flying' ? -0.25 : 0))
    cachedShape(rc, f, 'body', wobbleStep, () => rc.generator.ellipse(0, 0, bodyWidth, bodyLen, { ...opts, seed: s(1) }))
    ctx.restore()
  }

  const headCx = -bodyLen * 0.55 - neckLen

  // neck, for long-necked birds
  if (neckLen > 0 && parts > step) {
    cachedShape(rc, f, 'neck', wobbleStep, () =>
      rc.generator.line(-bodyLen * 0.55, 0, headCx, 0, { ...opts, strokeWidth: 1.5, seed: s(2) }),
    )
  }
  if (parts > step++) {
    // head
    cachedShape(rc, f, 'head', wobbleStep, () =>
      rc.generator.circle(headCx, -headR * 0.3, headR * 2, { ...opts, seed: s(3) }),
    )
  }

  // beak
  if (parts > step++) {
    drawBeak(rc, f, f.beakShape, headCx, headR, f.beakLen, wobbleStep, { ...opts, strokeWidth: 1, seed: s(4) })
  }

  // crest, a couple of small pointed tufts on the head
  if (f.hasCrest && parts > step++) {
    for (const dx of [-1, 1]) {
      cachedShape(rc, f, `crest${dx}`, wobbleStep, () =>
        rc.generator.polygon(
          [
            [headCx + dx * headR * 0.2, -headR * 1.1],
            [headCx + dx * headR * 0.5, -headR * 1.9],
            [headCx + dx * headR * 0.6, -headR * 1.0],
          ],
          { ...opts, strokeWidth: 0.9, seed: s(5 + dx) },
        ),
      )
    }
  }

  // legs - only when perched
  if (f.hasLegs && parts > step) {
    const legOpts = { ...opts, strokeWidth: 0.9, seed: s(6) }
    cachedShape(rc, f, 'legL', wobbleStep, () =>
      rc.generator.line(-bodyLen * 0.1, bodyWidth * 0.5, -bodyLen * 0.1, bodyWidth * 0.5 + headR, legOpts),
    )
    cachedShape(rc, f, 'legR', wobbleStep, () =>
      rc.generator.line(bodyLen * 0.05, bodyWidth * 0.5, bodyLen * 0.05, bodyWidth * 0.5 + headR, legOpts),
    )
  }

  ctx.restore()
  return true
}
