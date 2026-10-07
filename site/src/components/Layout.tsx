import { useRef, type MouseEvent, type ReactNode } from 'react'
import { Link } from '@tanstack/react-router'
import styles from './Layout.module.css'
import { MarginDoodles, type MarginDoodlesHandle } from './MarginDoodles'

const INTERACTIVE_TAGS = new Set(['A', 'BUTTON', 'INPUT', 'TEXTAREA', 'SELECT', 'LABEL'])

// `e.target === e.currentTarget` only matched the single outermost .page div,
// so any plain layout wrapper in between (a page's own <main> container, a
// header div, its own top padding before any real content starts, grid gaps
// between cards, etc.) silently ate the click instead of letting it reach
// .page - even though visually it's just as empty as the true page margin.
// treat anything as "background" that isn't interactive and has no text of
// its own (only descendants can have text) - that covers every pure layout
// div/main without risking a real content element or button/link.
function isBackgroundClick(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false
  if (INTERACTIVE_TAGS.has(target.tagName)) return false
  for (const node of target.childNodes) {
    if (node.nodeType === Node.TEXT_NODE && (node.textContent ?? '').trim().length > 0) {
      return false
    }
  }
  return true
}

export function Layout({ children }: { children: ReactNode }) {
  const doodlesRef = useRef<MarginDoodlesHandle>(null)

  const handleClick = (e: MouseEvent<HTMLDivElement>) => {
    // the canvas itself is pointer-events:none, so this click-bubbling path
    // on .page is the only way doodles ever get spawned
    if (!isBackgroundClick(e.target)) return
    doodlesRef.current?.spawn(e.pageX, e.pageY)
  }

  return (
    <div className={styles.page} onClick={handleClick}>
      <MarginDoodles ref={doodlesRef} />
      <div className={styles.fogTop} />
      <div className={styles.fogBottom} />
      <nav className={styles.nav}>
        <Link to="/" activeProps={{ className: styles.navActive }}>
          plshelp
        </Link>
        <Link to="/docs" activeProps={{ className: styles.navActive }}>
          docs
        </Link>
        <Link to="/registry" activeProps={{ className: styles.navActive }}>
          registry
        </Link>
        <a href="https://github.com/HariharPrasadd/plshelp" target="_blank" rel="noreferrer">
          github
        </a>
      </nav>
      {children}
    </div>
  )
}
