import type { RoughCanvas } from 'roughjs/bin/canvas'
import type { Drawable } from 'roughjs/bin/core'

export const INK = '#3a3836'

// a curated set of ink tones - every doodle's stroke color is a blend
// BETWEEN two of these (never a fully random hue), so variety stays
// anchored to a palette that was actually chosen rather than wandering.
// mixes muted darks with a few genuinely bright tones; since blends are
// between any two entries, pairing a bright one with a dark one also
// produces a whole range of mid-tones for free
const INK_PALETTE = [
  '#3a3836', // charcoal
  '#5b4636', // warm ink brown
  '#3d4a57', // slate blue-grey
  '#5a3f52', // muted plum
  '#3f4f3c', // forest ink
  '#6b4332', // terracotta
  '#d9534f', // coral red
  '#e08a2e', // marigold
  '#3a8fd1', // sky blue
  '#2fa876', // emerald
  '#d1558a', // rose pink
  '#8c6fd9', // violet
]

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

function rgbToHex(r: number, g: number, b: number): string {
  const c = (v: number) => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0')
  return `#${c(r)}${c(g)}${c(b)}`
}

export function pickInk(): string {
  const a = hexToRgb(pick(INK_PALETTE))
  const b = hexToRgb(pick(INK_PALETTE))
  const t = Math.random()
  return rgbToHex(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)
}

export const INTRO_MS = 550
export const HOLD_MS = 2200
export const OUTRO_MS = 500
export const TOTAL_MS = INTRO_MS + HOLD_MS + OUTRO_MS

// how often the linework re-sketches itself in place - the "wobble"
export const WOBBLE_STEP_MS = 170

export function rand(min: number, max: number) {
  return min + Math.random() * (max - min)
}

export function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)]
}

function easeOutCubic(t: number) {
  return 1 - Math.pow(1 - t, 3)
}

function easeInCubic(t: number) {
  return t * t * t
}

export interface Lifecycle {
  scale: number
  revealT: number
  wobbleStep: number
  alive: boolean
}

// shared grow-in / hold / shrink-out + wobble-seed stepping, used by every
// species so they all breathe at the same pace and feel like one family
export function computeLifecycle(elapsed: number): Lifecycle {
  if (elapsed > TOTAL_MS) {
    return { scale: 0, revealT: 1, wobbleStep: 0, alive: false }
  }

  let scale: number
  let revealT: number

  if (elapsed < INTRO_MS) {
    const t = easeOutCubic(elapsed / INTRO_MS)
    scale = t
    revealT = t
  } else if (elapsed < INTRO_MS + HOLD_MS) {
    scale = 1
    revealT = 1
  } else {
    const t = easeInCubic((elapsed - INTRO_MS - HOLD_MS) / OUTRO_MS)
    scale = 1 - t
    revealT = 1
  }

  return { scale, revealT, wobbleStep: Math.floor(elapsed / WOBBLE_STEP_MS), alive: true }
}

// no `stroke` here - callers set it per-instance via pickInk()
export const BASE_OPTS = {
  strokeWidth: 1.2,
  roughness: 0.9,
  bowing: 0.6,
}

// the wobble only actually changes a shape's sketchy path once every
// WOBBLE_STEP_MS, but without this, every species was re-running rough.js's
// (fairly expensive) jitter-path generation on EVERY animation frame -
// ~10x more often than the result ever visibly changes. This caches the
// generated Drawable per (doodle instance, shape key) and only regenerates
// it when the wobble step actually advances; every other frame just replays
// the cached path via rc.draw(), which is cheap. The cache is keyed off the
// doodle's own state object, so it's naturally garbage collected once a
// doodle's lifecycle ends and nothing references it anymore.
const shapeCache = new WeakMap<object, Map<string, { step: number; drawable: Drawable }>>()

export function cachedShape(
  rc: RoughCanvas,
  owner: object,
  key: string,
  step: number,
  generate: () => Drawable,
) {
  let cache = shapeCache.get(owner)
  if (!cache) {
    cache = new Map()
    shapeCache.set(owner, cache)
  }
  const hit = cache.get(key)
  if (hit && hit.step === step) {
    rc.draw(hit.drawable)
    return
  }
  const drawable = generate()
  cache.set(key, { step, drawable })
  rc.draw(drawable)
}
