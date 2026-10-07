import { forwardRef, useEffect, useImperativeHandle, useRef } from 'react'
import rough from 'roughjs'
import type { RoughCanvas as RoughCanvasInstance } from 'roughjs/bin/canvas'
import { createDoodle, drawDoodleFrame, type Doodle } from '../lib/doodles'
import styles from './MarginDoodles.module.css'

export interface MarginDoodlesHandle {
  spawn: (docX: number, docY: number) => void
}

// a single persistent, viewport-sized doodle canvas. it never resizes on
// navigation or page height (only on an actual window resize), and never
// grows to document size - doodles are stored in document coordinates and
// the whole canvas is translated by -scroll each frame, like a camera, so a
// doodle stays anchored to the content it was drawn on as you scroll past
// it. this also means the canvas buffer is always small (viewport area, not
// document area), which is what actually matters for frame cost - a canvas
// sized to a long page's full scrollHeight was multiple times more pixels
// to clear and redraw every frame than the viewport ever shows at once.
export const MarginDoodles = forwardRef<MarginDoodlesHandle>(function MarginDoodles(_props, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rcRef = useRef<RoughCanvasInstance | null>(null)
  const doodlesRef = useRef<Doodle[]>([])
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    rcRef.current = rough.canvas(canvas)

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [])

  useEffect(
    () => () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    },
    [],
  )

  const startLoop = () => {
    if (rafRef.current !== null) return
    const tick = () => {
      const rc = rcRef.current
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      if (rc && canvas && ctx) {
        // clear in plain viewport space, then shift into document space for
        // the actual drawing so doodles track their original page position
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.save()
        ctx.translate(-window.scrollX, -window.scrollY)
        const now = performance.now()
        doodlesRef.current = doodlesRef.current.filter((d) => drawDoodleFrame(rc, canvas, d, now))
        ctx.restore()
      }
      if (doodlesRef.current.length > 0) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        rafRef.current = null
      }
    }
    rafRef.current = requestAnimationFrame(tick)
  }

  useImperativeHandle(ref, () => ({
    spawn(docX: number, docY: number) {
      doodlesRef.current.push(createDoodle(docX, docY))
      startLoop()
    },
  }))

  // scrolling alone doesn't need a new frame from the click/intro/outro
  // loop, but a doodle's position needs to visibly track the page while
  // scrolling even if nothing else is animating - so nudge the loop on
  // scroll too, it naturally stops again once nothing is left to draw
  useEffect(() => {
    const onScroll = () => {
      if (doodlesRef.current.length > 0) startLoop()
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  return <canvas ref={canvasRef} className={styles.canvas} />
})
