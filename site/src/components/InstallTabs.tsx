import { useState } from 'react'
import styles from './InstallTabs.module.css'

const TABS = [
  { id: 'curl', label: 'curl', cmd: 'curl -fsSL https://plshelp.run/install.sh | sh' },
  { id: 'irm', label: 'irm', cmd: 'irm https://plshelp.run/install.ps1 | iex' },
  { id: 'npm', label: 'npm', cmd: 'npm i -g @generalinteraction/plshelp' },
  { id: 'bun', label: 'bun', cmd: 'bun add -g @generalinteraction/plshelp' },
] as const

export function InstallTabs() {
  const [active, setActive] = useState<(typeof TABS)[number]['id']>('curl')
  const [copied, setCopied] = useState(false)
  const current = TABS.find((t) => t.id === active)!

  const copy = () => {
    navigator.clipboard.writeText(current.cmd).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    })
  }

  return (
    <div className={styles.wrap}>
      <div className={styles.tabs}>
        {TABS.map((t) => (
          <button
            key={t.id}
            className={`${styles.tab} ${t.id === active ? styles.tabActive : ''}`}
            onClick={() => setActive(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className={styles.cmdRow}>
        <span className={styles.cmdText}>{current.cmd}</span>
        <button
          className={`${styles.copyBtn} ${copied ? styles.copied : ''}`}
          onClick={copy}
          aria-label="Copy"
        >
          <svg
            className={styles.copy}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <rect x="9" y="9" width="13" height="13" rx="2" />
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
          </svg>
          <svg
            className={styles.check}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <polyline points="20 6 9 17 4 12" />
          </svg>
        </button>
      </div>
    </div>
  )
}
