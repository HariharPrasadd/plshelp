import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useRef } from 'react'
import type { RoughCanvas as RoughCanvasInstance } from 'roughjs/bin/canvas'
import { RoughCanvas } from '../components/RoughCanvas'
import { createDoodle, drawDoodleFrame, type Doodle } from '../lib/doodles'

export const Route = createFileRoute('/playground')({
  component: Playground,
})

function Playground() {
  const doodlesRef = useRef<Doodle[]>([])
  const rcRef = useRef<RoughCanvasInstance | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)

  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
  }, [])

  const startLoop = () => {
    if (rafRef.current !== null) return
    const tick = () => {
      const rc = rcRef.current
      const canvas = canvasRef.current
      if (rc && canvas) {
        const ctx = canvas.getContext('2d')
        ctx?.clearRect(0, 0, canvas.width, canvas.height)
        const now = performance.now()
        doodlesRef.current = doodlesRef.current.filter((d) => drawDoodleFrame(rc, canvas, d, now))
      }
      if (doodlesRef.current.length > 0) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        rafRef.current = null
      }
    }
    rafRef.current = requestAnimationFrame(tick)
  }

  return (
    <div style={{ position: 'fixed', inset: 0 }}>
      <RoughCanvas
        onReady={(rc, canvas) => {
          rcRef.current = rc
          canvasRef.current = canvas
        }}
        onClick={(x, y) => {
          doodlesRef.current.push(createDoodle(x, y))
          startLoop()
        }}
      />
    </div>
  )
}
