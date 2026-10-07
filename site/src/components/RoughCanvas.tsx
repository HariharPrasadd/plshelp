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
      const dpr = window.devicePixelRatio || 1
      canvas.width = window.innerWidth * dpr
      canvas.height = window.innerHeight * dpr
      canvas.getContext('2d')?.setTransform(dpr, 0, 0, dpr, 0, 0)
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
