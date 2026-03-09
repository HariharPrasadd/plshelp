# plshelp

`plshelp` is a local-first Rust CLI for crawling documentation, indexing it into local search context, and querying it for RAG-style workflows.

## What It Does

`plshelp` turns documentation into a local searchable context engine:

- crawl a documentation subtree
- clean HTML and convert it to Markdown/text
- store pages, parent chunks, child chunks, and embeddings in SQLite
- build a BM25 full-text index for lexical retrieval
- answer local semantic and hybrid queries
- export compiled `.md` and `.txt` artifacts

It is designed for:

- developer documentation search
- local RAG pipelines
- code assistants that need deterministic local context
- private, reproducible indexing without hosted vector infrastructure

## Core Concepts

### Libraries

A library is a named corpus in the local database.

Examples:

- `rust`
- `numpyuser`
- `nextjs`

Most commands operate on a library name.

### Pages

Each crawled page is stored in the `pages` table with:

- source URL
- page order
- cleaned page content

### Parents and Children

The index is two-level:

- `parents`: larger coherent chunks returned to the user
- `chunks`: smaller child chunks used for retrieval and embedding

Retrieval matches child chunks, then returns the associated parent chunk for coherence.

### Retrieval Modes

`plshelp` supports:

- `hybrid`: embeddings + BM25
- `vector`: embeddings only
- `keyword`: BM25 only

If a library has been chunked but not embedded, hybrid falls back to BM25.

## Install

### Local development

```bash
cargo build
```

### Run directly

```bash
cargo run -- --help
```

### Install as a CLI

```bash
cargo install --path .
```

That should install the `plshelp` binary.

## Runtime Layout

On first run, `plshelp` creates a config file and runtime directories using OS defaults.

Default runtime paths:

### macOS

- config: `~/Library/Application Support/plshelp/config.toml`
- data dir: `~/Library/Application Support/plshelp`
- db: `~/Library/Application Support/plshelp/plshelp.db`
- artifacts: `~/Library/Application Support/plshelp/artifacts`
- models: `~/Library/Application Support/plshelp/models`

### Linux

- config: `$XDG_CONFIG_HOME/plshelp/config.toml` or `~/.config/plshelp/config.toml`
- data: `$XDG_DATA_HOME/plshelp` or `~/.local/share/plshelp`

### Windows

- config: `%APPDATA%\\plshelp\\config.toml`
- data: `%APPDATA%\\plshelp`

## Configuration

`plshelp` reads `config.toml` at process startup on every command invocation.

Behavior:

- built-in defaults are always present
- valid values in `config.toml` override defaults
- missing values fall back to defaults
- malformed values fall back to defaults
- if the entire TOML file is malformed, `plshelp` falls back to defaults

### Sample `config.toml`

```toml
[paths]
data_dir = "~/Library/Application Support/plshelp"
db_path = "~/Library/Application Support/plshelp/plshelp.db"
artifacts_dir = "~/Library/Application Support/plshelp/artifacts"
models_dir = "~/Library/Application Support/plshelp/models"

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
hybrid_vector_weight = 0.9
hybrid_bm25_weight = 0.1

[sqlite]
journal_mode = "WAL"
busy_timeout_ms = 5000

[onnx]
intra_threads = 8
inter_threads = 1
```

### Supported Embedding Models in Config

The config parser currently accepts these model names:

- `AllMiniLML6V2Q`
- `AllMiniLML6V2`
- `AllMiniLML12V2`
- `AllMiniLML12V2Q`
- `BGESmallENV15`
- `BGESmallENV15Q`
- `MxbaiEmbedLargeV1Q`

The matching is forgiving:

- case-insensitive
- punctuation-insensitive

So values like `all-mini-lm-l6-v2-q` and `AllMiniLML6V2Q` both resolve.

## Command Reference

## `add`

Crawl and index a library in one command.

```bash
plshelp add <library_name> <source_url> [--include-artifacts[=/path]] [--json]
```

What it does:

1. crawls the source URL
2. writes pages into the DB
3. chunks the content
4. builds BM25
5. embeds child chunks

Example:

```bash
plshelp add numpyuser https://numpy.org/doc/2.4/user
```

With artifacts:

```bash
plshelp add numpyuser https://numpy.org/doc/2.4/user --include-artifacts
```

## `crawl`

Crawl a docs subtree and persist pages, but do not chunk or embed.

```bash
plshelp crawl <library_name> <source_url> [--include-artifacts[=/path]] [--json]
```

Use this when you want:

- crawl only
- crawl first, index later
- exported cleaned docs without embeddings yet

Example:

```bash
plshelp crawl nextjs https://nextjs.org/docs
```

## `chunk`

Generate parent and child chunks and build the BM25 index.

```bash
plshelp chunk <library_name> [--file /path/to/file] [--json]
```

Behavior:

- if the library already has crawled pages, chunk those pages
- if `--file` is passed, chunk a local file into the named library

Examples:

```bash
plshelp chunk numpyuser
plshelp chunk mynotes --file /Users/hariharprasad/notes/api.md
```

## `embed`

Generate embeddings for child chunks.

```bash
plshelp embed <library_name> [--json]
```

Use this after `chunk` if you want semantic or hybrid retrieval.

Example:

```bash
plshelp embed numpyuser
```

## `index`

Convenience wrapper over `chunk` + `embed`.

```bash
plshelp index <library_name> [--file /path/to/file] [--json]
```

Examples:

```bash
plshelp index numpyuser
plshelp index mynotes --file /Users/hariharprasad/notes/system-design.txt
```

## `query`

Query one library or merged group.

```bash
plshelp query <library_name> "<question>" [--mode hybrid|vector|keyword] [--top-k N] [--context N] [--json]
```

Notes:

- unquoted multi-word queries are supported
- everything after the library name is treated as the query until the first `--flag`
- default mode, `top-k`, and `context` come from config if not supplied

Examples:

```bash
plshelp query numpyuser "what is broadcasting"
plshelp query numpyuser what is broadcasting
plshelp query numpyuser what is broadcasting --mode keyword
plshelp query numpyuser what is broadcasting --top-k 3 --context 1
```

## `trace`

Same as `query`, but includes scoring and chunk-location metadata.

```bash
plshelp trace <library_name> "<question>" [--mode hybrid|vector|keyword] [--top-k N] [--context N] [--json]
```

Use this when debugging:

- ranking quality
- hybrid weighting
- child-to-parent retrieval behavior

Example:

```bash
plshelp trace numpyuser "what is broadcasting" --mode hybrid --top-k 5
```

## `ask`

Query across multiple libraries or across all libraries.

```bash
plshelp ask "<question>" [--libraries a,b,c] [--mode ...] [--top-k N] [--context N] [--json]
```

Examples:

```bash
plshelp ask "how does routing work" --libraries nextjs,react
plshelp ask "what is broadcasting" --mode keyword
```

If `--libraries` is omitted, `ask` searches all libraries.

## Alias query form

You can query a library directly without the `query` subcommand:

```bash
plshelp <library_name> "<question>"
```

Example:

```bash
plshelp numpyuser "what is broadcasting"
```

## `merge`

Create a merged group from multiple libraries.

```bash
plshelp merge <new_library_name> <library1> <library2> [library3 ...] [--replace] [--include-artifacts[=/path]] [--json]
```

Use this when you want multiple corpora to behave like one searchable library.

Example:

```bash
plshelp merge numpy numpyuser numpyref
```

With replacement:

```bash
plshelp merge numpy numpyuser numpyref --replace
```

## `export`

Export compiled library text as `docs.md` and `docs.txt`.

```bash
plshelp export <library_name> [path] [--json]
```

Behavior:

- if no path is provided, export goes to `artifacts/<library>/`
- filenames are always:
  - `docs.md`
  - `docs.txt`

Examples:

```bash
plshelp export numpyuser
plshelp export numpyuser /tmp/numpy-export
```

## `refresh`

Recompute derived library stats and rebuild library text from stored content.

```bash
plshelp refresh [library_name ...] [--json]
```

Use this when:

- stats are stale
- you changed indexing logic and want rollups refreshed
- you want to rebuild compiled library text from stored pages/parents

Examples:

```bash
plshelp refresh
plshelp refresh numpyuser
```

## `alias`

Add an alias for an existing library.

```bash
plshelp alias <library_name> <alias> [--json]
```

Example:

```bash
plshelp alias numpyuser np
```

Then:

```bash
plshelp np "what is broadcasting"
```

## `list`

List libraries and merged groups.

```bash
plshelp list [--json]
```

Shows:

- source URL
- page count
- chunk count
- BM25-indexed chunk count
- content size
- status
- refresh time

## `show`

Show detailed metadata for one library or merged group.

```bash
plshelp show <library_name> [--json]
```

Includes:

- page count
- parent count
- chunk count
- embedded chunk count
- BM25-indexed chunk count
- min/max/avg chunks per page
- indexed model
- job status
- crawl/index timestamps
- aliases

## `open`

Open a child chunk by chunk ID and show its parent.

```bash
plshelp open <chunk_id> [--json]
```

This is useful for:

- debugging retrieval
- inspecting chunk boundaries
- inspecting parent/child relationships

## `remove`

Remove a library or merged group.

```bash
plshelp remove <library_name> [--json]
```

Behavior:

- for a real library, removes DB rows and local compiled artifacts
- for a merged group, removes the group membership entry

## Output Modes

### Default terminal output

By default, commands print human-readable output and progress spinners.

### JSON output

Most commands support `--json`.

This is intended for:

- automation
- agent integration
- scripts
- external tooling

Examples:

```bash
plshelp list --json
plshelp query numpyuser "what is broadcasting" --json
plshelp trace numpyuser "what is broadcasting" --json
plshelp add numpyuser https://numpy.org/doc/2.4/user --json
```

JSON behavior:

- success responses are structured JSON
- errors are also structured JSON when `--json` is present

## Typical Workflows

## Workflow: index a docs site

```bash
plshelp add nextjs https://nextjs.org/docs
plshelp query nextjs "how do i use app router"
```

## Workflow: crawl first, embed later

```bash
plshelp crawl numpyuser https://numpy.org/doc/2.4/user
plshelp chunk numpyuser
plshelp query numpyuser "what is broadcasting" --mode keyword
plshelp embed numpyuser
plshelp query numpyuser "what is broadcasting" --mode hybrid
```

## Workflow: local Markdown file

```bash
plshelp index mynotes --file /Users/hariharprasad/notes/rust.md
plshelp query mynotes "what did i write about ownership"
```

## Workflow: local text file

```bash
plshelp index design-notes --file /Users/hariharprasad/notes/system-design.txt
plshelp query design-notes "how do we handle retries"
```

## Workflow: multiple docs sets under one merged library

```bash
plshelp add numpyuser https://numpy.org/doc/2.4/user
plshelp add numpyref https://numpy.org/doc/2.4/reference
plshelp merge numpy numpyuser numpyref
plshelp query numpy "what is broadcasting"
```

## Workflow: export cleaned docs to files

```bash
plshelp crawl rustdocs https://doc.rust-lang.org/std --include-artifacts
plshelp export rustdocs
```

This writes:

- `artifacts/rustdocs/docs.md`
- `artifacts/rustdocs/docs.txt`

## Indexing and Retrieval Behavior

### Crawl behavior

The crawler:

- normalizes the seed URL
- stays within the configured URL subtree
- extracts main content from docs pages
- strips obvious chrome like nav/footer/sidebar/script/style blocks
- converts HTML to Markdown

### Chunking behavior

Parent chunks:

- heading-aware
- bounded by parent min/max character settings

Child chunks:

- derived within each parent only
- sized using child min/max character settings
- split near a target size on whitespace

### Retrieval behavior

BM25:

- built during `chunk`
- works immediately after chunking

Embeddings:

- added during `embed`
- used by `vector` mode
- blended with BM25 in `hybrid` mode

### Returned results

Retrieval ranks child chunks, but `plshelp` returns the parent content and source URL.

## Performance Notes

- Smaller child chunks are safer for smaller embedding models like MiniLM.
- More child chunks improve recall but increase indexing cost.
- BM25 is cheap and available right after `chunk`.
- Embeddings are the expensive step.
- SQLite is configured with WAL mode and a busy timeout for better mixed read/write behavior.

## Troubleshooting

## Query returns no results

Try:

- `--mode keyword`
- `trace`
- checking that the library was chunked or embedded
- `show <library>`

Examples:

```bash
plshelp show numpyuser
plshelp trace numpyuser "what is broadcasting" --mode keyword
```

## Query behaves strangely for exact text

Check:

- whether the library has been embedded with the current model
- whether `keyword` mode returns better exact-match results
- whether your chunk sizes are too large for the configured model

## Shell treats `?` or `*` strangely

Quote the question:

```bash
plshelp query numpyuser "how do i use generics?"
```

Unquoted `?` and `*` can be consumed by the shell before `plshelp` runs.

## Config changes do not seem to apply

`plshelp` reads config per process invocation.

So:

- edit `config.toml`
- run a new command

That is enough. No daemon restart is needed.

## Model files are downloading into the wrong place

They should go under:

- `models_dir` from config

By default:

- `~/Library/Application Support/plshelp/models` on macOS

## Current Defaults

At the time of writing, the built-in defaults are:

- embedding model: `AllMiniLML6V2Q`
- embed batch size: `128`
- parent chunk range: `1400-3000` chars
- child chunk range: `400-800` chars
- split window: `50` chars
- retrieval default mode: `hybrid`
- default top-k: `1`
- default context window: `0`
- hybrid weights: `0.9 vector / 0.1 bm25`
- SQLite journal mode: `WAL`
- SQLite busy timeout: `5000ms`

## Development Notes

This repository currently contains:

- a library crate
- a thin binary entrypoint

The binary is `plshelp`.

If you are contributing or debugging behavior, the main subsystems are split into:

- `runtime`
- `db`
- `crawl`
- `chunk`
- `embed`
- `search`
- `commands`
- `libraries`
- `artifacts`
- `cli`
