import { createFileRoute } from '@tanstack/react-router'
import styles from '../components/Prose.module.css'
import { Toc } from '../components/Toc'
import { TocRail } from '../components/TocRail'
import { docsToc } from '../content/docsToc'
import { DocsContent } from '../content/DocsContent'

export const Route = createFileRoute('/docs')({
  component: Docs,
})

function Docs() {
  return (
    <>
      <Toc entries={docsToc} />
      <TocRail entries={docsToc} />
      <main className={styles.col}>
        <h1>Documentation</h1>
        <DocsContent />
      </main>
    </>
  )
}
