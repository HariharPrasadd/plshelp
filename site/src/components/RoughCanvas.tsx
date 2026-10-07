import { useEffect, useRef } from 'react'
import rough from 'roughjs'
import type { RoughCanvas as RoughCanvasInstance } from 'roughjs/bin/canvas'

export function RoughCanvas({
  onReady,
  onClick,
}: {
  onReady?: (rc: RoughCanvasInstance, canvas: HTMLCanvasElement) => void
  onClick?: (x: number, y: number, rc: RoughCanvasInstance, canvas: HTMLCanvasElement) => void
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rcRef = useRef<RoughCanvasInstance | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const rc = rough.canvas(canvas)
    rcRef.current = rc
    onReady?.(rc, canvas)

    return () => window.removeEventListener('resize', resize)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    const rc = rcRef.current
    if (!canvas || !rc) return
    onClick?.(e.clientX, e.clientY, rc, canvas)
  }

  return (
    <canvas
      ref={canvasRef}
      onClick={handleClick}
      style={{ display: 'block', width: '100vw', height: '100vh' }}
    />
  )
}
