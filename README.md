# plshelp

`plshelp` is a local documentation crawler, indexer, and retrieval CLI written in Rust. You point it at a documentation site, it crawls the pages, extracts the content, converts that content into markdown, chunks it into retrieval-friendly sections, embeds those chunks with `fastembed`, stores everything in SQLite, and then lets you search the result with vector search, BM25 keyword search, or a hybrid of both.

The short version is simple: this project turns docs into a local knowledge base you can query from the terminal.

It is designed around a few practical assumptions. The data should live on your machine. The indexed content should be inspectable. Re-indexing should be explicit. The search results should point back to real source pages. And if you want plain text artifacts for debugging or downstream use, the tool should be able to export those too.

## What the project does

There are really five stages in the pipeline.

First, `crawl` downloads a documentation site and extracts the main content from each page.

Second, `chunk` breaks each page into larger parent sections and then smaller child sections that are easier to rank.

Third, `embed` computes vector embeddings for every child chunk that does not already have one.

Fourth, `query`, `trace`, and `ask` score the indexed content and return the best matching parent sections.

Finally, `export` and `--include-artifacts` let you write the compiled documentation back out as plain text and markdown files.

If you want one command that does the normal thing, use `add`. It crawls first and then indexes.

## What lives in this repository

The root crate is the actual CLI. The `src/` folder contains the application logic:

`src/main.rs` is the thin process entrypoint.

`src/lib.rs` defines shared types, default constants, and the command dispatcher.

`src/runtime.rs` creates the config and data directories, writes the default config on first run, resolves file paths, and applies runtime settings such as ONNX thread counts and SQLite settings.

`src/db.rs` creates the SQLite schema and runs lightweight migrations.

`src/crawl.rs` handles crawling, HTML extraction, markdown cleanup, and page storage.

`src/chunk.rs` turns markdown pages into parent and child chunks.

`src/embed.rs` handles embedding storage, query embedding, vector scoring, and parent-context loading.

`src/search.rs` builds and queries the BM25 FTS index, then combines keyword and vector scores.

`src/libraries.rs` handles aliases, merged library groups, stats, metadata views, deletes, and chunk inspection.

`src/artifacts.rs` writes `docs.txt` and `docs.md`.

`src/ui.rs` contains the terminal spinner and the echo guard.

There is also a small `site/` tree. Right now the release automation and the install scripts are more meaningful than the site builder itself. The release workflow packages binaries for macOS on Apple Silicon and Intel, Linux x86_64, and Windows x86_64. The shell and PowerShell installers download GitHub release assets and verify checksums before installing the binary.

## How the pipeline works in practice

When you run:

```bash
plshelp add rust-std https://doc.rust-lang.org/std/
```

the command flow is:

```text
add
  -> crawl_library
  -> store pages + compiled library text
  -> chunk_library
  -> rebuild BM25 index
  -> embed_library
  -> update library rollups
```

That means a library is only really queryable once it has chunks and a BM25 index, and vector mode is only fully available once all chunks are embedded.

The search layer returns parent sections, not raw child chunks. Child chunks are used for ranking because they are more precise. Once the best child match is found, the CLI shows the parent section so the result reads like a real piece of documentation instead of a tiny fragment.

## Requirements

You need a working Rust toolchain if you are building from source.

The project uses SQLite through `rusqlite` with the `bundled` feature, so you do not need a system SQLite installation.

The crawler and embedding stack may need network access the first time you use them. Crawling obviously needs to fetch a site. Embedding models are cached locally under the configured models directory, but they still need to be downloaded once.

In practical terms, the safest assumption is this:

Rust is required to build.

Internet access is required the first time you crawl a site or fetch an embedding model.

Disk space matters if you plan to index large documentation sets.

## Building from source

From the repository root:

```bash
cargo build
```

For a release build:

```bash
cargo build --release
```

Then run it directly:

```bash
cargo run -- --help
```

or use the built binary:

```bash
./target/release/plshelp --help
```

The package name in `Cargo.toml` is currently `plshelp`, and the binary produced by the repo is also `plshelp`.

## Installing from GitHub releases

The repository includes cross-platform install scripts in [`site/public/install.sh`](/Users/hariharprasad/MyDocuments/Code/Rust/deep-spider/site/public/install.sh) and [`site/public/install.ps1`](/Users/hariharprasad/MyDocuments/Code/Rust/deep-spider/site/public/install.ps1).

On macOS or Linux, the intended usage is:

```bash
curl -fsSL https://raw.githubusercontent.com/HariharPrasadd/plshelp/main/site/public/install.sh | sh
```

On Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/HariharPrasadd/plshelp/main/site/public/install.ps1 | iex
```

The scripts support a few environment variables:

`PLSHELP_GITHUB_REPO` lets you point at a fork.

`PLSHELP_VERSION` lets you install a specific release tag instead of `latest`.

`PLSHELP_INSTALL_DIR` lets you choose where the binary is installed.

The shell script currently supports macOS arm64, macOS x86_64, and Linux x86_64. The PowerShell script supports Windows x86_64. Arm64 Linux and arm64 Windows are explicitly not wired up yet in the installer logic.

## First run and local storage

On first launch, `plshelp` creates its config and data directories and writes a default `config.toml` if one does not already exist.

On macOS, both the default config directory and the default data directory are:

```text
~/Library/Application Support/plshelp
```

On Linux, the defaults follow XDG conventions:

```text
~/.config/plshelp
~/.local/share/plshelp
```

On Windows, the default base is `%APPDATA%\plshelp`.

The important files and folders are:

`config.toml` for runtime configuration.

`plshelp.db` for the SQLite database.

`artifacts/` for exported `docs.txt` and `docs.md`.

`models/` for cached embedding models.

If you want to see exactly where your machine will put these, inspect the logic in [`src/runtime.rs`](/Users/hariharprasad/MyDocuments/Code/Rust/deep-spider/src/runtime.rs).

## The default config file

The application writes a default config file the first time it runs. A representative example looks like this:

```toml
[paths]
data_dir = "/Users/you/Library/Application Support/plshelp"
db_path = "/Users/you/Library/Application Support/plshelp/plshelp.db"
artifacts_dir = "/Users/you/Library/Application Support/plshelp/artifacts"
models_dir = "/Users/you/Library/Application Support/plshelp/models"

[embedding]
model = "AllMiniLML6V2Q"
batch_size = 128

[chunking]
parent_min_chars = 1400
parent_max_chars = 3000
child_min_chars = 400
child_max_chars = 800
child_split_window_chars = 50

[retrieval]
default_mode = "hybrid"
default_top_k = 1
default_context_window = 0
hybrid_vector_weight = 0.90
hybrid_bm25_weight = 0.10

[sqlite]
journal_mode = "WAL"
busy_timeout_ms = 5000

[onnx]
intra_threads = 8
inter_threads = 1
```

You can change the storage paths if you want the database and model cache somewhere else. Relative paths are resolved relative to the config directory. `~` expansion is supported.

The supported embedding model names are normalized fairly loosely, so values such as `AllMiniLML6V2Q`, `all-mini-lm-l6-v2-q`, and `allminilml6v2q` all map to the same internal model when the parser strips punctuation and lowercases the string.

The currently recognized models are:

`AllMiniLML6V2Q`

`AllMiniLML6V2`

`AllMiniLML12V2`

`AllMiniLML12V2Q`

`BGESmallENV15`

`BGESmallENV15Q`

`MxbaiEmbedLargeV1Q`

## Core commands

The current CLI surface, verified from the binary, is:

```text
plshelp <command>
  add <library_name> <source_url> [--include-artifacts[=/path]] [--json]
  crawl <library_name> <source_url> [--include-artifacts[=/path]] [--json]
  index <library_name> [--file /path/to/file] [--json]
  chunk <library_name> [--file /path/to/file] [--json]
  embed <library_name> [--json]
  refresh [library_name ...] [--json]
  merge <new_library_name> <library1> <library2> [library3 ...] [--replace] [--include-artifacts[=/path]] [--json]
  export <library_name> [path] [--json]
  query <library_name> "<question>" [--mode hybrid|vector|keyword] [--top-k N] [--context N] [--json]
  <library_name> "<question>"
  ask "<question>" [--libraries a,b,c] [--mode ...] [--top-k N] [--context N] [--json]
  alias <library_name> <alias>
  list [--json]
  show <library_name> [--json]
  remove <library_name> [--json]
  open <chunk_id> [--json]
  trace <library_name> "<question>" [--mode ...] [--top-k N] [--context N] [--json]
```

The rest of this section explains what each command actually does.

## `add`: the normal end-to-end workflow

`add` is the best entrypoint if you are starting from a live documentation site.

```bash
plshelp add rust-std https://doc.rust-lang.org/std/
```

This command checks whether the library exists. If it does not, or if it exists but has no stored pages, it crawls the source URL first. Then it runs indexing, which means chunking followed by embedding if needed.

If you want plain text artifacts written at the same time, use:

```bash
plshelp add rust-std https://doc.rust-lang.org/std/ --include-artifacts
```

If you want those artifacts in a specific directory, use:

```bash
plshelp add rust-std https://doc.rust-lang.org/std/ --include-artifacts ./out/rust-std
```

or:

```bash
plshelp add rust-std https://doc.rust-lang.org/std/ --include-artifacts=./out/rust-std
```

By default, artifact output goes to the configured artifacts root under a subdirectory named after the library.

## `crawl`: fetch pages but stop before indexing

Use `crawl` when you want to fetch and store the cleaned page content without immediately chunking and embedding it.

```bash
plshelp crawl mdn-js https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide
```

This command stores:

the per-page cleaned markdown in the `pages` table,

the concatenated full text in `library_texts`,

the library metadata in `libraries`.

It does not build chunks or embeddings. That is why `query` will not work meaningfully until you run `chunk` or `index`.

## `chunk`: build parent and child chunks

Use `chunk` when you already have crawled content and want to rebuild the retrieval chunks.

```bash
plshelp chunk rust-std
```

This command:

loads page content,

splits pages into parent sections using markdown headings,

applies min and max parent size constraints,

splits each parent into smaller child chunks,

stores both parents and children in SQLite,

rebuilds the BM25 FTS index,

updates library rollups.

It does not generate embeddings by itself unless you call `index` instead.

You can also chunk a local file directly:

```bash
plshelp chunk local-notes --file ./notes.md
```

When you use `--file`, the command can create a new logical library if one does not exist yet. The file is treated as a single page and stored with a `file://` source URL.

## `embed`: fill in missing embeddings

Use `embed` when chunks already exist but vector embeddings are missing.

```bash
plshelp embed rust-std
```

The implementation only loads chunks whose `embedding` blob is empty, batches them according to `embedding.batch_size`, stores the resulting bytes back into the `chunks` table, and then refreshes library rollups.

This is a good command to rerun after changing chunking parameters and rebuilding chunks.

## `index`: chunk and embed in one pass

`index` is the explicit indexing command.

```bash
plshelp index rust-std
```

It behaves like this:

If there are no chunks yet, it runs `chunk`.

If there are chunks but some are not embedded yet, it runs `embed`.

If everything is already present, it avoids unnecessary work.

You can also index a local markdown or text file directly:

```bash
plshelp index local-python --file ./python_notes.md
```

This is the easiest way to turn a single document into a searchable local library.

## `query`: search one library or one merged group

The main search command is:

```bash
plshelp query rust-std "how do I format a string"
```

The shorter alias is:

```bash
plshelp rust-std "how do I format a string"
```

You can control the retrieval mode:

```bash
plshelp query rust-std "iterator flatten examples" --mode hybrid
plshelp query rust-std "iterator flatten examples" --mode vector
plshelp query rust-std "iterator flatten examples" --mode keyword
```

You can request more than one result:

```bash
plshelp query rust-std "trait objects" --top-k 3
```

You can also include neighboring parent sections from the same page:

```bash
plshelp query rust-std "lifetimes" --top-k 2 --context 1
```

`hybrid` combines normalized vector and BM25 scores using the configured retrieval weights.

`vector` uses cosine similarity over chunk embeddings.

`keyword` uses SQLite FTS5 BM25 ranking.

If a library is only partially embedded and you ask for `hybrid`, the code falls back to keyword mode for that library instead of pretending the vector side is complete.

## `trace`: query, but show scoring details

`trace` is the debugging form of `query`.

```bash
plshelp trace rust-std "where is Vec documented" --top-k 3 --context 1
```

It prints the same parent content, but it also shows:

the final score,

the vector component,

the BM25 component,

the child chunk location,

the parent location,

the library name.

This is the command to use when you are tuning chunk sizes, retrieval weights, or deciding whether a result came from semantic similarity or keyword overlap.

## `ask`: search across many libraries

`ask` searches across all libraries unless you filter it.

```bash
plshelp ask "how do I sort a vector in rust"
```

To constrain the search:

```bash
plshelp ask "how do I sort a vector in rust" --libraries rust-std,mdn-js
```

The retrieval options are the same as `query`:

```bash
plshelp ask "state management patterns" --libraries react-docs,next-docs --mode hybrid --top-k 5 --context 1
```

Internally, `ask` embeds the query once, then scores each eligible library and merges the ranked hits into one combined result list.

## `merge`: create a synthetic group from multiple libraries

Merged groups let you treat several libraries as one search target without duplicating their chunks.

```bash
plshelp merge frontend react-docs next-docs mdn-js
```

After that, you can query the group as if it were a library:

```bash
plshelp query frontend "fetch data on the client"
```

If you need to replace the membership:

```bash
plshelp merge frontend react-docs next-docs mdn-js --replace
```

If you also want a compiled text artifact for the merged group:

```bash
plshelp merge frontend react-docs next-docs mdn-js --include-artifacts
```

The group metadata lives in the `library_groups` table. The underlying chunks stay attached to their original libraries.

## `alias`: add a short name for a library

```bash
plshelp alias rust-std rust
```

After that:

```bash
plshelp rust "how do I collect into a vec"
```

Aliases cannot collide with an existing library name.

## `list`: inspect what is indexed

```bash
plshelp list
```

This shows each library and merged group with source URL, page count, chunk count, BM25 chunk count, character count, status, and last refreshed date.

This is the quickest way to confirm whether a crawl or index job finished in a way the CLI considers healthy.

## `show`: inspect one library or group in detail

```bash
plshelp show rust-std
```

For a real library, this includes:

source URL,

page count,

parent count,

chunk count,

embedded chunk count,

BM25 indexed chunk count,

average, minimum, and maximum chunks per page,

empty page count,

content size,

embedding model,

embedding dimension,

latest job status,

last successful crawl time,

last successful index time,

latest error message if any,

aliases.

For a merged group, it shows aggregate counts and the member libraries.

One implementation detail worth knowing: the displayed embedding dimension is currently hard-coded as `1024` in the output path, even though the actual model dimension depends on the embedding model you choose. Treat that field as a placeholder until the code computes it dynamically.

## `open`: inspect a specific chunk and its parent

When `query` or `trace` shows a chunk id, you can inspect it directly:

```bash
plshelp open 42
```

This prints the child chunk, its parent id, the source URL, and then the full parent block that contains that child.

It is useful when you want to understand why a result ranked well or when you want to inspect chunk boundaries without digging in the database.

## `export`: write compiled artifacts on demand

If you want a library written back out as plain files:

```bash
plshelp export rust-std
```

That writes `docs.txt` and `docs.md` into the default artifacts directory for that library.

To choose the output directory yourself:

```bash
plshelp export rust-std ./exports/rust-std
```

For merged groups, `export` concatenates the compiled text from all member libraries in group order.

## `refresh`: recompute metadata without crawling

```bash
plshelp refresh
```

This recomputes and backfills rollups for every library.

To refresh only a few:

```bash
plshelp refresh rust-std mdn-js
```

This command is useful after schema changes, migrations, or older database states where some rollup fields may be missing or stale. It does not fetch remote content.

## `remove`: delete a library or delete a merged group

```bash
plshelp remove rust-std
```

For a library, this deletes aliases, chunks, FTS rows, parents, pages, compiled library text, job history, library metadata, and the artifact directory for that library.

For a merged group, it deletes the group membership rows from `library_groups`.

This is intentionally a real delete, so use it like one.

## JSON output mode

Most commands support `--json`.

For example:

```bash
plshelp show rust-std --json
plshelp list --json
plshelp query rust-std "hashmap entry api" --top-k 3 --json
```

Successful commands return a structured JSON payload. Errors from the process entrypoint are also formatted as JSON if `--json` is present in the argument list.

If you want to script around `plshelp`, prefer `--json` everywhere.

## How crawling works

The crawler uses the `spider` crate with a fairly permissive configuration. The current defaults allow up to 5,000 pages and depth up to 25, stay on the same host path family, and avoid subdomains and alternate top-level domains.

The code normalizes the seed URL to ensure it ends with a trailing slash and then constructs a whitelist regex that keeps the crawl constrained to the same scheme, authority, and path prefix.

After download, the HTML pipeline does not keep the full raw page. Instead, it tries to find the most content-rich node from selectors such as:

`article`

`main`

`[role="main"]`

`.content`

`.docs-content`

`.markdown-body`

`#content`

`body`

It then removes a lot of common navigation and chrome patterns like scripts, styles, headers, footers, sidebars, nav blocks, buttons, forms, and some classes or ids that smell like menus, breadcrumbs, pagination, cookie banners, or promotional furniture.

The cleaned HTML is converted to markdown with `html2md`, then a second cleanup pass removes noisy leftover lines such as `was this helpful?`, `copy page`, and other common doc-site debris.

The final markdown is what gets stored.

## How chunking works

Chunking is a two-level system.

At the parent level, a page is split by markdown headings. Setext headings are normalized into ATX-style headings first, and front matter is stripped if it exists. Then the code enforces upper bounds by splitting oversized chunks first by paragraphs, then by newlines, and finally by raw character boundaries if it has to.

After that, it enforces a lower bound by merging very small parent chunks forward. If the final parent chunk is still too short, it gets appended to the previous one.

At the child level, each parent block is split into smaller windows intended for retrieval. The code aims for roughly balanced child sizes, searching for a whitespace boundary near an ideal split point and falling back to a hard split when necessary.

The important defaults are:

parent chunks between roughly 1400 and 3000 characters,

child chunks between roughly 400 and 800 characters,

a split search window of 50 characters around the ideal child boundary.

Those values come from the defaults in [`src/lib.rs`](/Users/hariharprasad/MyDocuments/Code/Rust/deep-spider/src/lib.rs) and the config resolution in [`src/runtime.rs`](/Users/hariharprasad/MyDocuments/Code/Rust/deep-spider/src/runtime.rs).

## How search works

There are two ranking signals in the codebase.

The first is vector similarity. `embed_query` uses the same embedding model as the document index, converts the question into a vector, and then compares that vector against each chunk embedding using cosine similarity.

The second is BM25 over an FTS5 virtual table named `chunks_fts`. The query string is tokenized into unique lowercase alphanumeric terms and then joined with `OR` so the FTS layer can score content that overlaps with any of the terms.

Both raw score sets are normalized independently to the range `[0, 1]`.

In hybrid mode, the final score is:

```text
final = vector_weight * normalized_vector + bm25_weight * normalized_bm25
```

The default weights are `0.90` for vector and `0.10` for BM25, and the runtime normalizes them if you provide different numbers that do not sum to exactly `1.0`.

The system ranks child chunks but only returns one hit per parent block, which avoids showing five tiny fragments from the exact same section.

## The SQLite schema

The database is created in [`src/db.rs`](/Users/hariharprasad/MyDocuments/Code/Rust/deep-spider/src/db.rs). The main tables are:

`libraries` stores library metadata and rollups.

`library_aliases` stores alternate names.

`pages` stores cleaned markdown per crawled page.

`parents` stores larger page sections.

`chunks` stores child retrieval chunks and their embeddings.

`chunks_fts` is the FTS5 virtual table used for BM25 search.

`library_groups` stores merged groups.

`jobs` stores command history with status and messages.

`library_texts` stores the concatenated full text for a library.

The `embedding` column in `chunks` is a `BLOB`. The code writes embeddings as little-endian `f32` bytes and reconstructs them when needed.

SQLite is configured with `journal_mode = WAL` and a busy timeout of 5000 ms by default.

## Example workflows

If you want the simplest possible start, this is enough:

```bash
plshelp add rust-std https://doc.rust-lang.org/std/
plshelp rust-std "how do I join strings"
```

If you want a more controlled step-by-step flow:

```bash
plshelp crawl rust-std https://doc.rust-lang.org/std/
plshelp chunk rust-std
plshelp embed rust-std
plshelp trace rust-std "result and option combinators" --top-k 3 --context 1
```

If you want to build a local library from your own notes:

```bash
plshelp index team-notes --file ./docs/internal_notes.md
plshelp query team-notes "deploy rollback steps"
```

If you want one umbrella search target across several libraries:

```bash
plshelp add rust-std https://doc.rust-lang.org/std/
plshelp add mdn-js https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/
plshelp merge general-dev rust-std mdn-js
plshelp ask "how does iteration work" --libraries general-dev --top-k 5
```

If you want files you can inspect outside the database:

```bash
plshelp export rust-std ./exports/rust-std
```

After that you will have:

```text
./exports/rust-std/docs.txt
./exports/rust-std/docs.md
```

## Operational notes

The first embedding run may take noticeably longer than later runs because the model cache has to be populated.

Large documentation sites can create a lot of page and chunk rows, so SQLite write performance matters. The code already uses transactions for the heavy write phases and uses WAL mode by default.

Search quality is sensitive to chunk sizes. If results feel too broad, lower the parent and child max sizes a bit. If results feel too fragmented, raise them.

If you are indexing multiple related libraries and want cross-library exploration, merged groups are usually a better fit than concatenating everything into one physical library.

## Known limitations and rough edges

This project is already useful, but it is still early enough that some behavior is more practical than polished.

The tiny builder crate under `site/registry/builder/` is currently just a placeholder and does not participate in the CLI behavior.

The README examples use live documentation sites, but crawl quality depends heavily on how cleanly a site maps into the current HTML extraction heuristics.

The displayed `embedding_dim` in `show` is hard-coded to `1024`, which is not a reliable reflection of every possible embedding model.

The CLI hides terminal echo through a `TerminalEchoGuard`, even though most commands are not currently reading secrets. That is harmless, but it is worth knowing if you are inspecting terminal behavior.

The file [`doclist.txt`](/Users/hariharprasad/MyDocuments/Code/Rust/deep-spider/doclist.txt) appears to be a loose seed list and currently contains malformed entries like `tps://...` and `sttps://...`. It is not referenced by the runtime path, so treat it as scratch input rather than a trusted built-in catalog.

## Troubleshooting

If `query` says a library is not indexed yet, run `plshelp chunk <library>` or `plshelp add <library> <url>`.

If `vector` mode complains about partial embeddings, run `plshelp embed <library>`.

If crawling produces empty or noisy output, inspect the exported `docs.md` and `docs.txt` to see whether the content extraction heuristics are missing the real document body.

If you changed config values and the results still look stale, rerun `chunk`, then `embed`, then `refresh`.

If you want to inspect the actual stored rows, open the SQLite database at the configured `db_path` and check `pages`, `parents`, `chunks`, and `jobs`.

If a library seems present but not queryable, `plshelp show <library>` and `plshelp list` will usually tell you whether the problem is missing chunks, missing BM25 rows, missing embeddings, or a failed job.

## Development notes

The repository includes a release workflow in [`.github/workflows/release.yml`](/Users/hariharprasad/MyDocuments/Code/Rust/deep-spider/.github/workflows/release.yml). Tagging a version that matches `v*` builds release binaries for the supported target matrix, packages them, generates checksums, and publishes a GitHub release.

For day-to-day development, the most useful commands are still the plain Rust ones:

```bash
cargo build
cargo run -- --help
cargo test
```

There are no Rust test files in the repository right now, so `cargo test` is mostly a compile check unless new tests are added.

## One final mental model

If you are trying to understand the code quickly, think of `plshelp` as three systems sharing one local SQLite database.

The crawl system turns remote docs into clean markdown pages.

The indexing system turns pages into parent and child retrieval units, plus embeddings and FTS rows.

The retrieval system ranks those units and returns readable parent sections with enough metadata to inspect what happened.

That is the whole project.
