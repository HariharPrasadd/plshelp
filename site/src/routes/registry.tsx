import { createFileRoute } from '@tanstack/react-router'
import { useEffect, useMemo, useState } from 'react'
import styles from '../components/Registry.module.css'

export const Route = createFileRoute('/registry')({
  component: Registry,
})

const INDEX_URL = 'https://registry.plshelp.run/index.json'

interface RegistryEntry {
  name: string
  slug: string
  source_url: string
  pages: number
  content_size_chars: number
  artifacts: {
    markdown_path: string
    text_path: string
    markdown_bytes: number
    text_bytes: number
    markdown_sha256: string
    text_sha256: string
  }
  last_crawled_at: string
  last_successful_crawled_at: string
  crawl_duration_ms: number
  status: string
  error_message?: string
}

interface RegistryIndex {
  version: number
  generated_at: string
  entries: RegistryEntry[]
}

function fuzzyScore(query: string, target: string): number {
  const q = query.toLowerCase()
  const t = target.toLowerCase()
  if (!q) return 1

  const idx = t.indexOf(q)
  if (idx !== -1) return 1000 - idx

  let qi = 0
  let score = 0
  let lastMatch = -1
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) {
      score += lastMatch !== -1 ? (ti - lastMatch === 1 ? 10 : 1) : 5
      lastMatch = ti
      qi++
    }
  }
  return qi === q.length ? score : 0
}

function highlight(text: string, query: string) {
  if (!query.trim()) return text
  const idx = text.toLowerCase().indexOf(query.toLowerCase())
  if (idx === -1) return text
  return (
    <>
      {text.slice(0, idx)}
      <mark>{text.slice(idx, idx + query.length)}</mark>
      {text.slice(idx + query.length)}
    </>
  )
}

function formatBytes(n: number): string {
  if (!n) return ''
  if (n < 1024) return n + ' B'
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB'
  return (n / (1024 * 1024)).toFixed(1) + ' MB'
}

function Entry({
  entry,
  query,
}: {
  entry: RegistryEntry
  query: string
}) {
  const [copied, setCopied] = useState(false)
  const cmd = `plshelp add ${entry.slug} ${entry.source_url}`

  const copy = () => {
    navigator.clipboard.writeText(cmd).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1300)
    })
  }

  return (
    <div className={styles.entry}>
      <button className={styles.copyArea} onClick={copy} title={cmd}>
        <div className={styles.entryTop}>
          <span className={styles.name}>{highlight(entry.name, query)}</span>
        </div>
        <span className={`${styles.sub} ${copied ? styles.copied : ''}`}>
          {copied ? 'copied' : highlight(entry.slug, query)}
          {!copied && entry.artifacts.markdown_bytes
            ? ` · ${formatBytes(entry.artifacts.markdown_bytes)}`
            : ''}
        </span>
      </button>
      <div className={styles.links}>
        <a href={entry.artifacts.markdown_path} download="docs.md">
          md
        </a>
        <a href={entry.artifacts.text_path} download="docs.txt">
          txt
        </a>
        <a href={entry.source_url} target="_blank" rel="noreferrer">
          source
        </a>
      </div>
    </div>
  )
}

function Registry() {
  const [index, setIndex] = useState<RegistryIndex | null>(null)
  const [error, setError] = useState(false)
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')

  useEffect(() => {
    fetch(INDEX_URL)
      .then((res) => res.json())
      .then((data: RegistryIndex) => setIndex(data))
      .catch(() => setError(true))
  }, [])

  useEffect(() => {
    const t = setTimeout(() => setDebouncedQuery(query), 80)
    return () => clearTimeout(t)
  }, [query])

  const entries = useMemo(() => {
    if (!index) return []
    return index.entries
      .filter((e) => e.status === 'success')
      .sort((a, b) => b.artifacts.markdown_bytes - a.artifacts.markdown_bytes)
  }, [index])

  const results = useMemo(() => {
    if (!debouncedQuery.trim()) return entries.map((entry) => ({ entry, score: 1 }))
    return entries
      .map((entry) => {
        const nameScore = fuzzyScore(debouncedQuery, entry.name) * 1.2
        const slugScore = fuzzyScore(debouncedQuery, entry.slug)
        return { entry, score: Math.max(nameScore, slugScore) }
      })
      .filter((r) => r.score > 0)
      .sort((a, b) => b.score - a.score)
  }, [entries, debouncedQuery])

  return (
    <main className={styles.wrap}>
      <div className={styles.header}>
        <h1>Registry</h1>
        <p className={styles.meta}>
          {index
            ? `${entries.length} libraries, updated ${new Date(index.generated_at).toLocaleDateString(
                'en-US',
                { month: 'short', day: 'numeric', year: 'numeric' },
              )}. Click one to copy its install command.`
            : 'Pre-crawled documentation libraries, ready to use without crawling.'}
        </p>

        <input
          className={styles.search}
          type="text"
          placeholder="Search libraries"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          spellCheck={false}
        />
        {debouncedQuery.trim() && entries.length > 0 && (
          <p className={styles.count}>
            {results.length} / {entries.length}
          </p>
        )}
      </div>

      <div className={styles.list}>
        {error && <p className={styles.state}>Could not load the registry. Try again shortly.</p>}

        {!error && !index && <p className={styles.state}>Loading registry...</p>}

        {index && results.length === 0 && (
          <p className={styles.state}>No libraries match "{debouncedQuery}".</p>
        )}

        {results.map(({ entry }) => (
          <Entry entry={entry} query={debouncedQuery} key={entry.slug} />
        ))}
      </div>
    </main>
  )
}
