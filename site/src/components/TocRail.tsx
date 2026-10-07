import { useEffect, useRef, useState } from 'react'
import styles from './TocRail.module.css'
import type { TocEntry } from '../content/docsToc'

// how far down the viewport a heading can be and still count as "current" -
// a direct, fixed comparison against each section's live position
const ACTIVE_LINE_PX = 150

// same hover-swell tuning as the desktop Toc, so the two feel like one family
const INFLUENCE_MAX = 46 // px, swell radius at high cursor speed
const INFLUENCE_MIN = 9 // px, swell radius at rest or crawling slowly
const MAX_SHIFT = 14 // px, how far a row pushes out at peak influence
const VELOCITY_SCALE = 0.55 // px/ms of cursor speed that maps to the full radius
const VELOCITY_CAP = 1.6 // px/ms, speed is clamped here before anything else sees it
const VELOCITY_SMOOTH = 0.25 // how much each new sample blends into the smoothed speed
const VELOCITY_FRICTION = 0.88 // per-frame decay, so speed relaxes even without new events
const SIGMA_SMOOTH = 0.12 // per-frame lerp of the radius toward its target (lower = more trailing)

function smoothstep(t: number) {
  const x = Math.min(1, Math.max(0, t))
  return x * x * (3 - 2 * x)
}

export function TocRail({ entries }: { entries: TocEntry[] }) {
  const [activeId, setActiveId] = useState(entries[0]?.id)
  const [open, setOpen] = useState(false)
  const [mouseY, setMouseY] = useState<number | null>(null)
  const [sigma, setSigma] = useState(INFLUENCE_MIN)
  const scrollRafRef = useRef<number | null>(null)
  const railRef = useRef<HTMLElement>(null)
  const rowRefs = useRef<Array<HTMLAnchorElement | null>>([])
  const sigmaRef = useRef(INFLUENCE_MIN)
  const velocityRef = useRef(0)
  const prevSampleRef = useRef<{ y: number; t: number } | null>(null)
  const hoverRafRef = useRef<number | null>(null)

  // which section is active, tracked directly off live scroll position
  useEffect(() => {
    const ids = entries.map((e) => e.id)

    const update = () => {
      let current = ids[0]
      for (const id of ids) {
        const el = document.getElementById(id)
        if (el && el.getBoundingClientRect().top <= ACTIVE_LINE_PX) {
          current = id
        }
      }
      setActiveId(current)
    }

    const onScroll = () => {
      if (scrollRafRef.current !== null) return
      scrollRafRef.current = requestAnimationFrame(() => {
        scrollRafRef.current = null
        update()
      })
    }

    update()
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    return () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      if (scrollRafRef.current !== null) cancelAnimationFrame(scrollRafRef.current)
    }
  }, [entries])

  useEffect(() => () => {
    if (hoverRafRef.current !== null) cancelAnimationFrame(hoverRafRef.current)
  }, [])

  const startHoverLoop = () => {
    if (hoverRafRef.current !== null) return
    const tick = () => {
      velocityRef.current *= VELOCITY_FRICTION
      const target =
        INFLUENCE_MIN + (INFLUENCE_MAX - INFLUENCE_MIN) * smoothstep(velocityRef.current / VELOCITY_SCALE)
      sigmaRef.current += (target - sigmaRef.current) * SIGMA_SMOOTH
      setSigma(sigmaRef.current)

      if (velocityRef.current > 0.001 || Math.abs(target - sigmaRef.current) > 0.1) {
        hoverRafRef.current = requestAnimationFrame(tick)
      } else {
        hoverRafRef.current = null
      }
    }
    hoverRafRef.current = requestAnimationFrame(tick)
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLElement>) => {
    const railRect = railRef.current?.getBoundingClientRect()
    if (!railRect) return
    const y = e.clientY - railRect.top
    setMouseY(y)

    const now = performance.now()
    const prev = prevSampleRef.current
    if (prev) {
      const dt = now - prev.t
      if (dt > 0) {
        const instVelocity = Math.min(VELOCITY_CAP, Math.abs(y - prev.y) / dt)
        velocityRef.current += (instVelocity - velocityRef.current) * VELOCITY_SMOOTH
      }
    }
    prevSampleRef.current = { y, t: now }
    startHoverLoop()
  }

  const handleMouseLeave = () => {
    setMouseY(null)
    prevSampleRef.current = null
    velocityRef.current = 0
    sigmaRef.current = INFLUENCE_MIN
    setSigma(INFLUENCE_MIN)
    if (hoverRafRef.current !== null) {
      cancelAnimationFrame(hoverRafRef.current)
      hoverRafRef.current = null
    }
  }

  return (
    <nav
      ref={railRef}
      className={`${styles.rail} ${open ? styles.railOpen : ''}`}
      aria-label="Table of contents"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {/* one shared blur panel behind the whole label column, instead of a
          separate blur per row - avoids visible seams between labels */}
      <div className={styles.labelsBackdrop} aria-hidden="true" />

      {/* the only part of the rail that's actually tappable to open/close
          the label list - a narrow strip right at the ticks, not the whole
          box, so normal scrolling never misfires it */}
      <button
        className={styles.hitStrip}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-label="Toggle section list"
      />

      {entries.map((entry, i) => {
        const active = entry.id === activeId
        let style: React.CSSProperties | undefined

        if (mouseY !== null && railRef.current) {
          const el = rowRefs.current[i]
          if (el) {
            const railRect = railRef.current.getBoundingClientRect()
            const elRect = el.getBoundingClientRect()
            const center = elRect.top + elRect.height / 2 - railRect.top
            const dist = mouseY - center
            const influence = Math.exp(-(dist * dist) / (2 * sigma * sigma))
            style = { transform: `translateX(${-MAX_SHIFT * influence}px)` }
          }
        }

        return (
          <a
            key={entry.id}
            ref={(el) => {
              rowRefs.current[i] = el
            }}
            href={`#${entry.id}`}
            className={`${styles.row} ${active ? styles.rowActive : ''}`}
            style={style}
            onClick={(e) => {
              e.stopPropagation()
              e.preventDefault()
              document.getElementById(entry.id)?.scrollIntoView({ behavior: 'smooth' })
              setOpen(false)
            }}
          >
            <span className={styles.labelText}>{entry.title}</span>
            <span className={`${styles.tick} ${active ? styles.tickActive : ''}`} />
          </a>
        )
      })}
    </nav>
  )
}
