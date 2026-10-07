import { createFileRoute } from '@tanstack/react-router'
import styles from '../components/Home.module.css'
import { InstallTabs } from '../components/InstallTabs'
import { WobblyHeart } from '../components/WobblyHeart'

export const Route = createFileRoute('/')({
  component: Home,
})

function Home() {
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
      <pre className={styles.cmd}>{'plshelp add nextjs https://nextjs.org/docs\nplshelp query nextjs "how does app router work"'}</pre>
    </main>
  )
}
