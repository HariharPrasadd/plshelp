import { useState } from 'react'
import { createFileRoute } from '@tanstack/react-router'
import styles from '../components/Home.module.css'
import { InstallTabs } from '../components/InstallTabs'
import { WobblyHeart } from '../components/WobblyHeart'

export const Route = createFileRoute('/')({
  component: Home,
})

const TRY_CMD = 'plshelp add nextjs https://nextjs.org/docs\nplshelp query nextjs "how does app router work"'

function Home() {
  const [copied, setCopied] = useState(false)

  const copy = () => {
    navigator.clipboard.writeText(TRY_CMD).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    })
  }

  return (
    <main className={styles.col}>
      <h1>plshelp</h1>
      <p className={styles.meta}>
        local-first documentation search for agents and humans <WobblyHeart />
      </p>

      <InstallTabs />

      <p className={styles.block}>
        plshelp is a local first documentation search tool for agents and humans. it's open
        source, entirely on-device, and can crawl, index and embed the entirety of nextjs
        documentation from a single url in &lt; 5 mins on a MacBook Air.
      </p>

      <p className={styles.block}>
        since it's entirely local, you can spec the embedding model to your device capabilities.
        it can run on anything from a potato to a gpu cluster to a VPS. you can also create a
        searchable index of your local text files, single page articles you've read, and generally
        any text modality that lives on the internet or on your laptop.
      </p>

      <p className={styles.block}>
        along with the cli tool, we're also releasing the{' '}
        <a href="/registry">plshelp registry</a>, which is a repository of more than 500
        pre-scraped documentation libraries in .md and .txt formats. you can use it to fine-tune
        models, make your own local documentation store, or to immanentize the eschaton.
      </p>

      <p className={styles.block}>once you download plshelp, give this a shot to see it in action:</p>
      <div className={styles.cmdWrap}>
        <pre className={styles.cmd}>{TRY_CMD}</pre>
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
    </main>
  )
}
