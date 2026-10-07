import { Fragment, type ReactNode } from 'react'
import styles from '../components/Prose.module.css'
import { CodeBlock } from '../components/CodeBlock'

function Callout({ label, children }: { label: string; children: ReactNode }) {
  return (
    <p className={styles.block}>
      <strong className={styles.calloutLabel}>{label}:</strong> {children}
    </p>
  )
}

function Opt({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div className={styles.optRow}>
      <p className={styles.optTerm}>{term}</p>
      <p className={styles.optDesc}>{children}</p>
    </div>
  )
}

const modelGroups: {
  group: string
  models: { name: string; tokens: string; chars: string; notes: string }[]
}[] = [
  {
    group: 'Fast / quantized',
    models: [
      {
        name: 'AllMiniLML6V2Q',
        tokens: '256',
        chars: '400-800',
        notes: 'Default. Best choice for large corpora on average hardware.',
      },
      {
        name: 'BGESmallENV15Q',
        tokens: '512',
        chars: '700-1400',
        notes: 'Quantized, slightly better quality than MiniLM.',
      },
      {
        name: 'SnowflakeArcticEmbedSQ',
        tokens: '512',
        chars: '700-1400',
        notes: 'Fast quantized option with competitive retrieval quality.',
      },
    ],
  },
  {
    group: 'Balanced',
    models: [
      {
        name: 'AllMiniLML6V2',
        tokens: '256',
        chars: '400-800',
        notes: 'Unquantized MiniLM. Slightly better quality than the Q variant, somewhat slower.',
      },
      {
        name: 'BGESmallENV15',
        tokens: '512',
        chars: '700-1400',
        notes: 'Good quality/speed balance for English docs. Reasonable on most hardware.',
      },
      {
        name: 'SnowflakeArcticEmbedM',
        tokens: '512',
        chars: '700-1400',
        notes: 'Solid mid-tier model with good English retrieval.',
      },
      {
        name: 'MultilingualE5Small',
        tokens: '512',
        chars: '700-1400',
        notes: 'Smallest multilingual option.',
      },
    ],
  },
  {
    group: 'Higher quality, slower',
    models: [
      {
        name: 'BGEBaseENV15',
        tokens: '512',
        chars: '700-1400',
        notes: 'Better retrieval than Small. Needs more RAM and CPU time.',
      },
      {
        name: 'GTEBaseENV15',
        tokens: '512',
        chars: '700-1400',
        notes: 'Comparable to BGE Base. Good general-purpose English model.',
      },
      {
        name: 'MxbaiEmbedLargeV1',
        tokens: '512',
        chars: '700-1400',
        notes: 'High quality English model. Best reserved for small, high-value corpora.',
      },
      {
        name: 'BGELargeENV15',
        tokens: '512',
        chars: '700-1400',
        notes: 'Large BGE. Strong retrieval, expensive to run.',
      },
      {
        name: 'ModernBertEmbedLarge',
        tokens: '8192',
        chars: '1400-2000',
        notes: 'Modern architecture, large context. Slow on CPU.',
      },
    ],
  },
  {
    group: 'Long context',
    models: [
      {
        name: 'BGEM3',
        tokens: '8192',
        chars: '1400-2000',
        notes: 'Long-context, multilingual. Slow on CPU.',
      },
      {
        name: 'NomicEmbedTextV15',
        tokens: '8192',
        chars: '1400-2000',
        notes: 'Long-context English model. Good for papers and longform content.',
      },
      {
        name: 'SnowflakeArcticEmbedMLong',
        tokens: '2048',
        chars: '1000-1800',
        notes: 'Mid-range long-context option. Faster than BGEM3.',
      },
    ],
  },
  {
    group: 'Domain-specific',
    models: [
      {
        name: 'JinaEmbeddingsV2BaseCode',
        tokens: '8192',
        chars: '600-1200',
        notes:
          'Trained on code. Best for code-heavy or source-annotated corpora. Smaller child sizes recommended due to denser token counts in code.',
      },
      {
        name: 'MultilingualE5Base',
        tokens: '512',
        chars: '700-1400',
        notes: 'Cross-language retrieval. Use when your corpus spans multiple languages.',
      },
      {
        name: 'MultilingualE5Large',
        tokens: '512',
        chars: '700-1400',
        notes: 'Best multilingual quality. Heavy.',
      },
      {
        name: 'BGESmallZHV15',
        tokens: '512',
        chars: '700-1400',
        notes: 'Chinese-language corpora.',
      },
      {
        name: 'BGELargeZHV15',
        tokens: '512',
        chars: '700-1400',
        notes: 'Higher quality Chinese retrieval.',
      },
    ],
  },
]

export function DocsContent() {
  return (
    <>
      <section id="what-is-it" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>01</span>What is plshelp?</h2>
        <p className={styles.block}>
          plshelp is a local documentation search tool. Point it at any docs website, or your own
          notes and Markdown files, and it indexes everything locally so you can search it
          instantly from your terminal.
        </p>
        <p className={styles.block}>
          You can use it yourself, the same way you'd use a search box on a docs site, except it
          works offline and across everything you've indexed at once. And because it speaks JSON,
          coding agents like Claude Code and Codex can call it too, automatically pulling in the
          right docs context without touching the web.
        </p>
        <p className={styles.block}>
          Everything stays on your machine. No cloud sync, no third-party vector database, no data
          leaving your system. Your docs corpus is yours.
        </p>
        <Callout label="The core workflow">
          Index a library once with <code>plshelp add</code>. Search it yourself with{' '}
          <code>plshelp query</code>, or run <code>plshelp init</code> to wire it up to your
          coding agent automatically.
        </Callout>
      </section>

      <section id="install" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>02</span>Installation</h2>
        <p className={styles.block}>
          You can install plshelp with curl, irm, npm, or bun. The shell installers download a
          prebuilt binary and put it on your PATH, while the package manager routes through{' '}
          <code>@generalinteraction/plshelp</code>.
        </p>

        <h3 className={styles.h3}>macOS and Linux</h3>
        <CodeBlock code="curl -fsSL https://plshelp.run/install.sh | sh" />

        <h3 className={styles.h3}>Windows</h3>
        <CodeBlock label="powershell" code="irm https://plshelp.run/install.ps1 | iex" />

        <h3 className={styles.h3}>npm or bun</h3>
        <CodeBlock
          label="package manager"
          code={'npm install -g @generalinteraction/plshelp\nbun add -g @generalinteraction/plshelp'}
        />

        <p className={styles.block}>
          Confirm it worked by running <code>plshelp help</code> or <code>plshelp --help</code>,
          both print the same usage output. If the shell says it can't find the command, you may
          need to add the install directory to your PATH and open a new terminal.
        </p>
      </section>

      <section id="quickstart" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>03</span>Your first library in 2 minutes</h2>
        <p className={styles.block}>
          Pick any docs site. This example uses Next.js, but the same commands work for anything.
        </p>
        <CodeBlock
          code={
            '# Step 1: index the docs (this takes a minute)\nplshelp add nextjs https://nextjs.org/docs\n\n# Step 2: ask it something\nplshelp nextjs "how does the app router work"'
          }
        />
        <p className={styles.block}>
          That's it. <code>add</code> crawls the site, chunks the content, and builds a search
          index. After that, <code>query</code> is instant, it searches locally, no network
          needed.
        </p>
        <Callout label="Tip">
          Always quote your questions. It's not strictly required, but it prevents the shell from
          doing unexpected things with question marks and special characters.
        </Callout>
      </section>

      <section id="indexing-sites" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>04</span>Indexing websites and web pages</h2>
        <p className={styles.block}>
          Use <code>plshelp add &lt;name&gt; &lt;url&gt;</code> for any documentation website.
          Give the library a short name you'll remember, you'll use it in every query.
        </p>
        <CodeBlock
          code={
            'plshelp add rust https://doc.rust-lang.org/book/\nplshelp add react https://react.dev/reference/react\nplshelp add tailwind https://tailwindcss.com/docs'
          }
        />

        <h3 className={styles.h3}>Indexing a single page</h3>
        <p className={styles.block}>
          By default, <code>add</code> follows links and crawls an entire site. Use{' '}
          <code>--single</code> when you want to capture exactly one page: an article, a paper, a
          changelog, any URL you want to make locally searchable.
        </p>
        <CodeBlock
          code={
            '# Save a single article\nplshelp add article https://example.com/some-post --single\n\n# Save a research paper page\nplshelp add paper https://some-site.com/research --single\n\n# Save a changelog\nplshelp add changelog https://lib.rs/releases --single'
          }
        />
        <p className={styles.block}>
          This is what turns plshelp into a general-purpose internet reader. You can build a local
          library of anything you find useful: articles you want to reference later, pages you
          want your agent to know about, research you want to search across. Index each page as
          its own library, then <code>merge</code> them into one so you can query across
          everything at once.
        </p>
        <CodeBlock
          code={
            '# Index each page as its own library\nplshelp add reading-post-1 https://blog.example.com/post-1 --single\nplshelp add reading-post-2 https://blog.example.com/post-2 --single\nplshelp add reading-hn-12345 "https://news.ycombinator.com/item?id=12345" --single\n\n# Merge them into one library\nplshelp merge reading reading-post-1 reading-post-2 reading-hn-12345\n\n# Query across everything you\'ve saved\nplshelp ask "what did I read about distributed systems" --libraries reading\n\n# Wire it up to your agent\nplshelp init'
          }
        />

        <h3 className={styles.h3}>Respecting robots.txt</h3>
        <p className={styles.block}>
          By default, plshelp crawls without checking <code>robots.txt</code>. Pass{' '}
          <code>--respect-robots</code> to opt in to the target site's crawl policy, useful when
          you want more conservative behavior or are crawling sites where it matters.
        </p>
        <CodeBlock code="plshelp add mylib https://example.com/docs --respect-robots" />
        <p className={styles.block}>
          Both flags work with <code>crawl</code> as well, and can be combined:{' '}
          <code>plshelp add &lt;library&gt; &lt;url&gt; --single --respect-robots</code>.
        </p>
        <p className={styles.block}>
          Behind the scenes, <code>add</code> runs three stages: it crawls the site, splits the
          content into chunks, and generates embeddings for semantic search. You can run these
          stages separately if you want more control, see the{' '}
          <a href="#pipeline">pipeline section</a> below.
        </p>
        <p className={styles.block}>
          Once a library is indexed, check on it with <code>plshelp show &lt;name&gt;</code> to
          see page counts, chunk counts, and whether embeddings finished.
        </p>
      </section>

      <section id="indexing-files" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>05</span>Indexing local files</h2>
        <p className={styles.block}>
          plshelp isn't just for docs sites, you can index anything you have locally. Markdown
          notes, a team wiki, architecture decision records, exported API specs, internal
          playbooks. If it's text, you can search it the same way you'd search any other library.
        </p>
        <p className={styles.block}>
          Use <code>plshelp index --file</code> instead of <code>add</code>. It skips the crawl
          and goes straight to chunking and indexing.
        </p>
        <CodeBlock
          code={
            'plshelp index mynotes --file ./notes/architecture.md\nplshelp mynotes "how is the auth service structured"'
          }
        />
        <p className={styles.block}>
          Works with <code>.md</code> and <code>.txt</code> files. The library name is yours to
          choose, pick something short that you'll remember when querying. This is also a good
          way to give a coding agent access to private context it wouldn't otherwise have, like
          internal docs that aren't on the web.
        </p>
      </section>

      <section id="querying" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>06</span>Querying</h2>
        <p className={styles.block}>
          The basic query takes a library name and a question. For normal libraries, the easiest
          syntax is the shorthand form: <code>plshelp &lt;library&gt; "&lt;question&gt;"</code>. By
          default it returns the single most relevant result.
        </p>
        <CodeBlock code='plshelp rust "how do lifetimes work"' />

        <h3 className={styles.h3}>Getting more results</h3>
        <p className={styles.block}>
          Use <code>--top-k</code> to get more than one result, and <code>--context</code> to
          include neighboring sections alongside each result.
        </p>
        <CodeBlock code='plshelp rust "how do lifetimes work" --top-k 3 --context 1' />

        <h3 className={styles.h3}>Searching across multiple libraries</h3>
        <p className={styles.block}>
          Use <code>ask</code> when your question might span multiple docs sets. You can name
          specific libraries or leave it out to search everything.
        </p>
        <CodeBlock
          code={
            '# Search specific libraries\nplshelp ask "how do I handle errors" --libraries rust,tokio\n\n# Search everything you\'ve indexed\nplshelp ask "how do I handle errors"'
          }
        />

        <h3 className={styles.h3}>Single-library queries</h3>
        <p className={styles.block}>The shorthand form works the same way as <code>query</code>:</p>
        <CodeBlock code='plshelp rust "how do lifetimes work"' />

        <h3 className={styles.h3}>Query options</h3>
        <Opt term="--top-k N">
          Return N results instead of just the best one. Default is 1.
        </Opt>
        <Opt term="--context N">
          Include N neighboring sections alongside each result, for more surrounding context.
          Default is 0.
        </Opt>
        <Opt term="--mode">
          Choose <code>hybrid</code> (default), <code>vector</code> (semantic only), or{' '}
          <code>keyword</code> (exact text match).
        </Opt>
        <Opt term="--json">
          Output as JSON. Use this when querying from a script or coding agent.
        </Opt>
      </section>

      <section id="agent-setup" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>07</span>Usage with coding agents</h2>
        <p className={styles.block}>
          Claude Code, Codex, and other terminal-based agents can use plshelp automatically, but
          first they need to know it exists. That's what <code>plshelp init</code> is for.
        </p>
        <p className={styles.block}>
          Run it once at your project root. It writes instruction files (<code>CLAUDE.md</code>{' '}
          and <code>AGENTS.md</code>) that tell the agent how to call plshelp and when to use it.
          Claude Code reads <code>CLAUDE.md</code> on startup; Codex reads <code>AGENTS.md</code>.
        </p>
        <CodeBlock
          code={
            '# Write both CLAUDE.md and AGENTS.md\nplshelp init\n\n# Write only one of them\nplshelp init --claude\nplshelp init --agents\n\n# Preview what would be written without touching disk\nplshelp init --print'
          }
        />
        <p className={styles.block}>
          If you already have a <code>CLAUDE.md</code>, don't worry, <code>init</code> won't
          overwrite it. It appends a clearly marked plshelp section so your existing content stays
          intact.
        </p>
        <Callout label="Recommended workflow">
          Index your libraries first, then run <code>plshelp init</code>. Commit the generated
          files to the repo so every teammate, and the agent, gets the same setup automatically.
        </Callout>
        <p className={styles.block}>
          The <code>--json</code> flag is specifically for agent use, it makes plshelp output a
          structured JSON response that's easy to parse programmatically. When you're querying for
          yourself, you don't need it; the default output is formatted for human reading. The
          generated init files instruct the agent to always use <code>--json</code> automatically.
        </p>
      </section>

      <section id="managing" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>08</span>Managing your libraries</h2>
        <p className={styles.block}>
          A few commands cover the day-to-day of keeping your local corpus in good shape.
        </p>

        <h3 className={styles.h3}>See what you have</h3>
        <CodeBlock
          code={
            '# List all indexed libraries\nplshelp list\n\n# Inspect a specific library (page count, chunk count, embedding status)\nplshelp show rust'
          }
        />

        <h3 className={styles.h3}>Combine libraries into one</h3>
        <p className={styles.block}>
          If you want one query to cover multiple docs sets, say, numpy's user guide and its API
          reference, merge them into a single named library.
        </p>
        <CodeBlock
          code={
            'plshelp merge numpy numpy-userguide numpy-reference\nplshelp numpy "what is broadcasting"'
          }
        />

        <h3 className={styles.h3}>Bulk operations</h3>
        <p className={styles.block}>
          Most pipeline commands accept <code>--all</code> to operate across every indexed library
          at once.
        </p>
        <Opt term="embed --all --force">
          Clears and regenerates embeddings for every library. Use after changing the embedding
          model in config.
        </Opt>
        <Opt term="index --all --force">
          Rechunks and re-embeds everything from scratch. Use after changing chunk size config.
        </Opt>
        <Opt term="chunk --all --force">
          Rebuilds the chunk and BM25 index for every library. Chunking always rebuilds its own
          stage regardless, <code>--force</code> here clears derived state before rechunking but
          does not touch embeddings.
        </Opt>
        <Opt term="refresh --all">Re-runs the default refresh logic across all libraries.</Opt>
        <Opt term="export --all [path]">
          Exports every library to <code>&lt;path&gt;/&lt;slug&gt;/docs.md</code> and{' '}
          <code>docs.txt</code>. If you omit the path, plshelp writes each library to its default
          compiled output directory.
        </Opt>
        <Opt term="remove --all">
          Deletes every indexed library from the database. Requires typing{' '}
          <code>REMOVE ALL</code> exactly to confirm, this is destructive and cannot be undone. It
          does not touch the binary or runtime data; see <code>uninstall</code> for that.
        </Opt>

        <h3 className={styles.h3}>Delete a library</h3>
        <p className={styles.block}>
          Use <code>remove</code> to delete a library from the index, useful when docs are stale
          or the source URL changed and you want to re-index from scratch.
        </p>
        <CodeBlock code="plshelp remove nextjs" />
        <p className={styles.block}>
          This only removes the library from the database. Any exported files on disk (like{' '}
          <code>docs.md</code>) are left alone, delete those yourself if you want them gone.
        </p>

        <h3 className={styles.h3}>Uninstalling plshelp</h3>
        <p className={styles.block}>
          Use <code>plshelp uninstall</code> to remove the binary, the local runtime data, or
          both. This is different from <code>remove --all</code>, which only wipes indexed
          libraries from the database, <code>uninstall</code> removes plshelp itself from your
          system. The command is interactive: it shows you the exact resolved paths it will
          delete, tells you what will remain, and requires you to type the target path back
          exactly before anything is removed.
        </p>
        <CodeBlock
          code={
            '# Remove everything (binary + all local data)\nplshelp uninstall --all\n\n# Remove only the installed binary\nplshelp uninstall --binary\n\n# Remove only the local runtime data (database, embeddings, cache, config)\nplshelp uninstall --data'
          }
        />
      </section>

      <section id="debugging" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>09</span>When results seem wrong</h2>
        <p className={styles.block}>
          If a query returns something that doesn't look right, here's how to investigate.
        </p>

        <h3 className={styles.h3}>Check if the library is fully indexed</h3>
        <p className={styles.block}>
          Run <code>plshelp show &lt;name&gt;</code>. If embeddings aren't done yet, vector search
          won't be working, try <code>--mode keyword</code> as a workaround, or wait for
          embeddings to finish.
        </p>

        <h3 className={styles.h3}>See how results are being ranked</h3>
        <p className={styles.block}>
          <code>trace</code> runs the same search as <code>query</code> but shows you the scoring
          details: which chunks matched, what their scores were, and why they ranked where they
          did.
        </p>
        <CodeBlock code='plshelp trace rust "how do lifetimes work" --mode hybrid --top-k 5' />

        <h3 className={styles.h3}>Inspect a specific chunk</h3>
        <p className={styles.block}>
          If a trace shows a chunk ID you want to look at in full, use <code>open</code> to pull
          it up with its surrounding parent context.
        </p>
        <CodeBlock code="plshelp open 127" />
      </section>

      <section id="pipeline" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>10</span>The indexing pipeline</h2>
        <p className={styles.block}>
          <code>add</code> is convenient, but it's actually three stages running in sequence. You
          can run them individually if you want more control.
        </p>
        <Opt term="crawl">
          Download pages from a URL. Stops before chunking, useful if you want to inspect the raw
          content first.
        </Opt>
        <Opt term="chunk">
          Split pages into chunks and build the keyword (BM25) index. After this step, keyword
          search works immediately.
        </Opt>
        <Opt term="embed">
          Generate embeddings for all chunks. Required for vector and hybrid search.
        </Opt>
        <Opt term="index">
          Shortcut for chunk + embed. Use this with <code>--file</code> for local files.
        </Opt>
        <Opt term="add">Shortcut for crawl + index. The default for docs websites.</Opt>
        <p className={styles.block}>
          A practical reason to use stages separately: chunking is fast, embedding is slow. If you
          want to start searching immediately with keyword mode while embeddings run, do{' '}
          <code>crawl</code> then <code>chunk</code> first, and kick off <code>embed</code>{' '}
          separately.
        </p>
      </section>

      <section id="exports" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>11</span>Exporting</h2>
        <p className={styles.block}>
          <code>export</code> writes a library out as plain files, a <code>docs.md</code> and a{' '}
          <code>docs.txt</code>. Useful if you want a snapshot you can share, paste into a prompt,
          or pipe into another tool.
        </p>
        <CodeBlock
          code={
            '# Export to default location (artifacts/rust/)\nplshelp export rust\n\n# Export to a specific path\nplshelp export rust ./exports/rust-docs'
          }
        />
        <p className={styles.block}>
          If you want exports generated automatically during indexing rather than as a separate
          step, pass <code>--include-artifacts</code> to <code>add</code> or <code>crawl</code>.
          Without a path it writes to the default <code>artifacts/&lt;library&gt;/</code>{' '}
          location; with a path it writes there instead:
        </p>
        <CodeBlock
          code={
            '# Write to artifacts/rust/ automatically after crawling\nplshelp add rust https://doc.rust-lang.org/book/ --include-artifacts\n\n# Write to a custom path\nplshelp add rust https://doc.rust-lang.org/book/ --include-artifacts=./my-exports/rust'
          }
        />
        <p className={styles.block}>
          Use <code>--include-artifacts</code> when you want the export to happen in one step as
          part of indexing. Use <code>plshelp export</code> when the library is already indexed
          and you just want to produce or refresh the files on demand.
        </p>
      </section>

      <section id="config" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>12</span>Configuration</h2>
        <p className={styles.block}>
          plshelp writes a <code>config.toml</code> on first run. Most people never need to touch
          it, the defaults are sensible. But if you want to tune retrieval behavior or chunk
          sizes, this is where to do it.
        </p>
        <p className={styles.block}>See where your config lives and what's in it:</p>
        <CodeBlock code="plshelp config" />
        <p className={styles.block}>The full default config looks like this:</p>
        <CodeBlock
          label="config.toml"
          code={
            '[embedding]\nmodel = "AllMiniLML6V2Q"\nbatch_size = 128\n\n[chunking]\nparent_min_chars = 1400\nparent_max_chars = 3000\nchild_min_chars = 400\nchild_max_chars = 800\nchild_split_window_chars = 50\n\n[retrieval]\ndefault_mode = "hybrid"\ndefault_top_k = 1\ndefault_context_window = 0\nhybrid_vector_weight = 0.9\nhybrid_bm25_weight = 0.1\n\n[sqlite]\njournal_mode = "WAL"\nbusy_timeout_ms = 5000'
          }
        />
        <p className={styles.block}>
          The retrieval section is the one you're most likely to tweak. Bump{' '}
          <code>default_top_k</code> if you want more results by default, or adjust the hybrid
          weights if you find keyword matching is over or under-contributing.
        </p>
      </section>

      <section id="model-config" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>13</span>Embedding model configuration</h2>
        <p className={styles.block}>
          You probably don't need to touch this. The default model, <code>AllMiniLML6V2Q</code>,
          is a good fit for most machines and most corpora. This section is for users who have a
          specific reason to change it.
        </p>

        <h3 className={styles.h3}>Why this is a serious change</h3>
        <p className={styles.block}>
          The embedding model is used twice: when indexing, to generate chunk embeddings, and when
          querying, to embed your question before retrieval. Both sides must use the same model.
          If you change the model after indexing a library, every existing embedding in that
          library is incompatible with your queries, and results will be meaningless.
        </p>
        <p className={styles.block}>
          Changing the model also means downloading new weights, which can be several hundred
          megabytes to over a gigabyte depending on the model. If you experiment with several
          models, the cache grows accordingly. This is not a setting to rotate casually.
        </p>
        <p className={styles.block}>
          The right time to choose a model is before you start indexing. Pick one, index
          everything with it, and leave it alone.
        </p>

        <h3 className={styles.h3}>How chunking relates to model choice</h3>
        <p className={styles.block}>
          plshelp uses a parent/child chunking model. When a document is indexed, it's split into
          large parent chunks and smaller child chunks nested inside them. The child chunks are
          what get embedded and BM25-indexed, small enough for the model to handle well and for
          BM25 to match precisely. When retrieval finds a child chunk, plshelp returns its parent,
          so the text you actually read has enough surrounding context to be coherent.
        </p>
        <p className={styles.block}>
          This means the model's token input limit should drive your child chunk sizing. The
          default child range of <code>400-800 chars</code> is calibrated for the default model's
          256-token input window, using a rough conversion of 3-4 chars per token for prose and
          2-3 chars per token for code-heavy content. If you switch to a model with a larger input
          window, you can raise <code>child_max_chars</code> accordingly and get better semantic
          coherence without truncation.
        </p>

        <h3 className={styles.h3}>Batch size</h3>
        <p className={styles.block}>
          The <code>batch_size</code> setting controls how many chunks are embedded in a single
          pass. It defaults to <code>128</code> and should always be set to a multiple of 64.
        </p>
        <p className={styles.block}>
          A larger batch size means more chunks are processed in parallel, which improves
          throughput, but only up to the limit your hardware can support. If the batch doesn't fit
          in available RAM, the OS will start swapping and performance will collapse. The right
          value depends on which model you're using and how much memory your machine has.
        </p>
        <Opt term="64">
          Safe floor for constrained machines (4-8 GB RAM). Use with larger models or when memory
          is tight.
        </Opt>
        <Opt term="128">Default. Works well with small/quantized models on 8-16 GB RAM.</Opt>
        <Opt term="192-256">
          Reasonable on 16-32 GB RAM with small or medium models. Diminishing returns beyond this
          point on CPU.
        </Opt>
        <Opt term="256+">
          Only beneficial with large models on well-provisioned machines. If you're on CPU only,
          more RAM won't help much past 256, the bottleneck shifts to compute.
        </Opt>
        <p className={styles.block}>
          If you see memory pressure or swapping during <code>embed</code>, lower the batch size.
          If indexing feels slow and you have headroom, try stepping it up by 64 at a time.
        </p>

        <h3 className={styles.h3}>What to consider if you change models</h3>
        <p className={styles.block}>
          Speed, quality, and context window are the main axes. How these land in practice depends
          heavily on your hardware. On a slow CPU or a machine with limited RAM, a quantized small
          model may be the only option that runs at usable speed. Quantized models (the ones
          ending in <code>Q</code>) are faster and lighter with a modest quality tradeoff. Larger
          models produce better embeddings but are slower and heavier to store. Long-context
          models like <code>BGEM3</code> and <code>NomicEmbedTextV15</code> support up to 8192
          tokens per chunk, enabling much less aggressive child chunking, but they're
          significantly slower on CPU and may not be practical on lower-end hardware.
        </p>
        <p className={styles.block}>
          The full list of supported models is in the{' '}
          <a
            href="https://docs.rs/fastembed/latest/fastembed/enum.EmbeddingModel.html"
            target="_blank"
            rel="noopener noreferrer"
          >
            fastembed documentation
          </a>
          . A summary of the most useful ones is in the table below.
        </p>

        <details className={styles.details}>
          <summary className={styles.summary}>
            Supported models <span>click to expand</span>
          </summary>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Input tokens</th>
                  <th>child_max_chars</th>
                  <th>Notes</th>
                </tr>
              </thead>
              <tbody>
                {modelGroups.map((g) => (
                  <Fragment key={g.group}>
                    <tr className={styles.tableGroup}>
                      <td colSpan={4}>{g.group}</td>
                    </tr>
                    {g.models.map((m) => (
                      <tr key={m.name}>
                        <td>
                          <code>{m.name}</code>
                        </td>
                        <td>{m.tokens}</td>
                        <td>{m.chars}</td>
                        <td>{m.notes}</td>
                      </tr>
                    ))}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </details>

        <h3 className={styles.h3}>How to change models</h3>
        <p className={styles.block}>
          Edit the <code>model</code> field in your <code>config.toml</code>:
        </p>
        <CodeBlock label="config.toml" code={'[embedding]\nmodel = "BGESmallENV15"'} />
        <p className={styles.block}>
          After changing the model, you must regenerate embeddings for any library you want to
          keep using with vector or hybrid search. Use <code>--force</code> to clear and
          regenerate, without it, <code>embed</code> only fills missing embeddings and won't
          replace existing ones.
        </p>
        <CodeBlock
          code={
            '# Re-embed a single library\nplshelp embed <library> --force\n\n# Re-embed every indexed library at once\nplshelp embed --all --force'
          }
        />
        <p className={styles.block}>
          <code>embed --force</code> preserves your existing crawl and chunk data, it only
          regenerates the embeddings. If you also want to rebuild chunking from scratch (for
          example, after changing <code>child_max_chars</code> to match a new model's input
          window):
        </p>
        <CodeBlock
          code={
            '# Rechunk and re-embed a single library\nplshelp index <library> --force\n\n# Rechunk and re-embed everything\nplshelp index --all --force'
          }
        />
        <p className={styles.block}>
          If you only want to rebuild chunks without re-embedding (unusual, but possible), use{' '}
          <code>chunk --force</code>. This rebuilds the chunks, rebuilds BM25, and deletes the old
          embeddings along with the old child chunks. If you want vector or hybrid search again
          afterward, run <code>embed</code> to generate embeddings for the new chunks.
        </p>
        <p className={styles.block}>
          Keyword search continues to work without re-embedding, since BM25 is built from the
          chunk text itself, not the embedding model.
        </p>
      </section>

      <section id="how-search-works" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>14</span>How search works</h2>
        <p className={styles.block}>
          plshelp has three search modes. <code>hybrid</code> is the default and almost always the
          right choice.
        </p>
        <Opt term="hybrid">
          Blends keyword matching with semantic similarity. Best for technical docs where you need
          both exact API names and conceptual questions to work well.
        </Opt>
        <Opt term="vector">
          Pure semantic search based on meaning. Good for natural language questions, but can miss
          exact identifiers.
        </Opt>
        <Opt term="keyword">
          Exact text matching using BM25. Fast, and works as soon as a library is chunked, no
          embeddings needed. Best for looking up specific flags, error strings, or function names.
        </Opt>
        <p className={styles.block}>
          Internally, plshelp splits content into two levels: small <em>child chunks</em> used for
          matching, and larger <em>parent chunks</em> returned as results. This means search stays
          precise while the content you get back is actually readable.
        </p>
      </section>

      <section id="storage" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>15</span>Where data lives</h2>
        <p className={styles.block}>
          Everything, the database, embeddings, model cache, and config, lives in your system's
          app data directory. The binary itself is installed separately in a bin directory.
        </p>
        <Opt term="macOS">
          <code>~/Library/Application Support/plshelp</code>
        </Opt>
        <Opt term="Linux">
          <code>~/.local/share/plshelp</code>
        </Opt>
        <Opt term="Windows">
          <code>%APPDATA%\plshelp</code>
        </Opt>
        <p className={styles.block}>
          The database is a single SQLite file. It's yours, back it up, move it, inspect it
          directly if you want.
        </p>
      </section>

      <section id="reference" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>16</span>Full command reference</h2>
        <CodeBlock
          label="all commands"
          code={[
            'plshelp help',
            'plshelp init [--agents] [--claude] [--print] [--json]',
            'plshelp add <library> <url> [--single] [--respect-robots] [--force] [--include-artifacts[=/path]] [--json]',
            'plshelp crawl <library> <url> [--single] [--respect-robots] [--force] [--include-artifacts[=/path]] [--json]',
            'plshelp index <library> [--file /path/to/file] [--force] [--json]',
            'plshelp index --all [--force] [--json]',
            'plshelp chunk <library> [--file /path/to/file] [--force] [--json]',
            'plshelp chunk --all [--force] [--json]',
            'plshelp embed <library> [--force] [--json]',
            'plshelp embed --all [--force] [--json]',
            'plshelp refresh [library ...] [--json]',
            'plshelp refresh --all [--json]',
            'plshelp merge <new> <lib1> <lib2> [...] [--replace] [--json]',
            'plshelp export <library> [path] [--json]',
            'plshelp export --all [path] [--json]',
            'plshelp query <library> "<question>" [--mode hybrid|vector|keyword] [--top-k N] [--context N] [--json]',
            'plshelp <library> "<question>"',
            'plshelp ask "<question>" [--libraries a,b,c] [--mode ...] [--top-k N] [--context N] [--json]',
            'plshelp trace <library> "<question>" [--mode ...] [--top-k N] [--context N] [--json]',
            'plshelp alias <library> <alias> [--json]',
            'plshelp list [--json]',
            'plshelp show <library> [--json]',
            'plshelp open <chunk_id> [--json]',
            'plshelp config [--json]',
            'plshelp remove <library> [--json]',
            'plshelp remove --all [--json]',
            'plshelp uninstall --all | --binary | --data',
          ].join('\n')}
        />
      </section>

      <section id="registry-curl" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>17</span>Using the registry without the CLI</h2>
        <p className={styles.block}>
          The plshelp registry is a collection of pre-crawled documentation libraries hosted at{' '}
          <code>registry.plshelp.run</code>. You don't need the CLI to use it, every library is
          available as a plain <code>.md</code> or <code>.txt</code> file you can download
          directly with curl.
        </p>

        <h3 className={styles.h3}>See what's available</h3>
        <p className={styles.block}>
          The registry publishes an <code>index.json</code> with every available library. Pipe it
          through <code>jq</code> to get a clean list:
        </p>
        <CodeBlock
          code={`curl -s https://registry.plshelp.run/index.json | jq -r '.entries[] | select(.status=="success") | .name'`}
        />
        <p className={styles.block}>
          If you want the slug alongside the name, useful for building download URLs:
        </p>
        <CodeBlock
          code={`curl -s https://registry.plshelp.run/index.json | jq -r '.entries[] | select(.status=="success") | "\\(.name) (\\(.slug))"'`}
        />

        <h3 className={styles.h3}>Download a single library</h3>
        <p className={styles.block}>
          Every library is at <code>registry.plshelp.run/docs/&lt;slug&gt;/docs.md</code>. Use{' '}
          <code>-O</code> to save it with the default filename, or <code>-o</code> to name it
          yourself:
        </p>
        <CodeBlock
          code={
            '# Save as docs.md\ncurl -O https://registry.plshelp.run/docs/accelerate/docs.md\n\n# Save with a custom name\ncurl -o accelerate-docs.md https://registry.plshelp.run/docs/accelerate/docs.md\n\n# Plain text version\ncurl -O https://registry.plshelp.run/docs/accelerate/docs.txt'
          }
        />

        <h3 className={styles.h3}>Download everything at once</h3>
        <p className={styles.block}>
          Combine the index with <code>xargs</code> to bulk-download all available libraries in
          one shot:
        </p>
        <CodeBlock
          code={`curl -s https://registry.plshelp.run/index.json \\\n  | jq -r '.entries[] | select(.status=="success") | .slug' \\\n  | xargs -I{} curl -O https://registry.plshelp.run/docs/{}/docs.md`}
        />
        <Callout label="Tip">
          You can also browse and download libraries from the{' '}
          <a href="/registry">registry page</a> directly in your browser, no terminal needed.
        </Callout>
      </section>

      <section id="faq" className={styles.section}>
        <h2 className={styles.h2}><span className={styles.num}>18</span>FAQ</h2>

        <h3 className={styles.h3}>Can I search before embeddings finish?</h3>
        <p className={styles.block}>
          Yes. Keyword search (BM25) is built during chunking, so <code>--mode keyword</code>{' '}
          works immediately. Vector and hybrid search require embeddings to be done first.
        </p>

        <h3 className={styles.h3}>Does plshelp need an internet connection to search?</h3>
        <p className={styles.block}>
          No. Once a library is indexed, everything runs locally. The only time you need a
          connection is when first crawling a docs site with <code>add</code>.
        </p>

        <h3 className={styles.h3}>What file types can I index?</h3>
        <p className={styles.block}>
          Markdown (<code>.md</code>) and plain text (<code>.txt</code>) for local files. For
          websites, plshelp crawls and cleans the HTML automatically.
        </p>

        <h3 className={styles.h3}>How do I update a library when the docs change?</h3>
        <p className={styles.block}>
          Remove the old version with <code>plshelp remove &lt;name&gt;</code>, then re-run{' '}
          <code>plshelp add</code>. Or use <code>plshelp add &lt;name&gt; &lt;url&gt; --force</code>{' '}
          to recrawl, rechunk, and re-embed from scratch without removing the library first.
        </p>

        <h3 className={styles.h3}>Is there an MCP server?</h3>
        <p className={styles.block}>
          Not yet. The CLI is the integration surface for now. If you need MCP, the practical path
          is wrapping the <code>--json</code> commands in a thin adapter.
        </p>

        <h3 className={styles.h3}>Where does plshelp store everything?</h3>
        <p className={styles.block}>
          All runtime data (database, embeddings, model cache, config) lives in your system's app
          data directory. See the <a href="#storage">storage section</a> above for
          platform-specific paths.
        </p>
      </section>
    </>
  )
}
