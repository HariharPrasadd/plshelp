import { useEffect, useRef, useState } from 'react'
import styles from './Toc.module.css'
import type { TocEntry } from '../content/docsToc'

const INFLUENCE_MAX = 46 // px, swell radius at high cursor speed
const INFLUENCE_MIN = 9 // px, swell radius at rest or crawling slowly
const MAX_SHIFT = 14 // px, how far an item pushes right at peak influence
const VELOCITY_SCALE = 0.55 // px/ms of cursor speed that maps to the full radius
const VELOCITY_CAP = 1.6 // px/ms, speed is clamped here before anything else sees it
const VELOCITY_SMOOTH = 0.25 // how much each new sample blends into the smoothed speed
const VELOCITY_FRICTION = 0.88 // per-frame decay, so speed relaxes even without new events
const SIGMA_SMOOTH = 0.12 // per-frame lerp of the radius toward its target (lower = more trailing)

function smoothstep(t: number) {
  const x = Math.min(1, Math.max(0, t))
  return x * x * (3 - 2 * x)
}

export function Toc({ entries }: { entries: TocEntry[] }) {
  const [activeId, setActiveId] = useState(entries[0]?.id)
  const [mouseY, setMouseY] = useState<number | null>(null)
  const [sigma, setSigma] = useState(INFLUENCE_MIN)
  const observerRef = useRef<IntersectionObserver | null>(null)
  const navRef = useRef<HTMLElement>(null)
  const linkRefs = useRef<Array<HTMLAnchorElement | null>>([])
  const sigmaRef = useRef(INFLUENCE_MIN)
  const velocityRef = useRef(0)
  const prevSampleRef = useRef<{ y: number; t: number } | null>(null)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    const sections = entries
      .map((e) => document.getElementById(e.id))
      .filter((el): el is HTMLElement => el !== null)

    observerRef.current = new IntersectionObserver(
      (observerEntries) => {
        const visible = observerEntries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)
        if (visible.length > 0) {
          setActiveId(visible[0].target.id)
        }
      },
      { rootMargin: '-15% 0px -70% 0px', threshold: [0.1, 0.25, 0.5, 0.75] },
    )

    sections.forEach((el) => observerRef.current?.observe(el))
    return () => observerRef.current?.disconnect()
  }, [entries])

  useEffect(() => () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
  }, [])

  const startLoop = () => {
    if (rafRef.current !== null) return
    const tick = () => {
      // speed relaxes toward 0 every frame on its own, so a fast flick that
      // suddenly stops still settles instead of leaving the radius stuck wide
      velocityRef.current *= VELOCITY_FRICTION

      const target =
        INFLUENCE_MIN + (INFLUENCE_MAX - INFLUENCE_MIN) * smoothstep(velocityRef.current / VELOCITY_SCALE)
      sigmaRef.current += (target - sigmaRef.current) * SIGMA_SMOOTH
      setSigma(sigmaRef.current)

      if (velocityRef.current > 0.001 || Math.abs(target - sigmaRef.current) > 0.1) {
        rafRef.current = requestAnimationFrame(tick)
      } else {
        rafRef.current = null
      }
    }
    rafRef.current = requestAnimationFrame(tick)
  }

  const handleClick = (e: React.MouseEvent, id: string) => {
    e.preventDefault()
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    setActiveId(id)
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLElement>) => {
    const navRect = navRef.current?.getBoundingClientRect()
    if (!navRect) return
    const y = e.clientY - navRect.top
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
    startLoop()
  }

  const handleMouseLeave = () => {
    setMouseY(null)
    prevSampleRef.current = null
    velocityRef.current = 0
    sigmaRef.current = INFLUENCE_MIN
    setSigma(INFLUENCE_MIN)
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }

  return (
    <nav
      ref={navRef}
      className={styles.toc}
      aria-label="Table of contents"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {entries.map((entry, i) => {
        let style: React.CSSProperties | undefined

        if (mouseY !== null && navRef.current) {
          const el = linkRefs.current[i]
          if (el) {
            const navRect = navRef.current.getBoundingClientRect()
            const elRect = el.getBoundingClientRect()
            const center = elRect.top + elRect.height / 2 - navRect.top
            const dist = mouseY - center
            // gaussian falloff: the radius (sigma) tracks cursor speed, so a slow
            // deliberate scan stays tight to one item while a fast sweep swells wide
            const influence = Math.exp(-(dist * dist) / (2 * sigma * sigma))
            style = { transform: `translateX(${MAX_SHIFT * influence}px)` }
          }
        }

        return (
          <a
            key={entry.id}
            ref={(el) => {
              linkRefs.current[i] = el
            }}
            href={`#${entry.id}`}
            className={`${styles.link} ${entry.id === activeId ? styles.linkActive : ''}`}
            onClick={(e) => handleClick(e, entry.id)}
            style={style}
          >
            <span className={styles.num}>{String(i + 1).padStart(2, '0')}</span>
            {entry.title}
          </a>
        )
      })}
    </nav>
  )
}
