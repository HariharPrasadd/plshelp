<!-- plshelp:start -->
## plshelp

Use `plshelp` as the local documentation retrieval tool for this repository.

### Setup (if no libraries are indexed yet)

- `plshelp add <name> <docs-url>` to index a library
- `plshelp show <name> --json` to confirm it's ready before querying

Preferred command pattern:

- `plshelp query <library> "<question>" --json`
- `plshelp trace <library> "<question>" --json` when debugging ranking or retrieval quality
- `plshelp ask "<question>" --libraries a,b,c --json` when the answer may span multiple libraries
- `plshelp show <library> --json` to inspect indexing state and chunk / embedding counts
- `plshelp list --json` to discover available libraries
- `plshelp open <chunk_id> --json` to inspect a specific retrieved child chunk and its parent
- `plshelp config --json` to inspect active runtime configuration

Operational guidance:

- Prefer `--json` for any agent-driven call.
- Prefer `query` before `trace`; use `trace` only when retrieval seems wrong or you need scores.
- `query` ranks child chunks but returns parent content. Treat the returned `content` field as the user-facing context block.
- `source_url` is the canonical citation for a returned result.
- `keyword` mode is BM25 / FTS-based lexical retrieval.
- `vector` mode requires embeddings.
- `hybrid` combines both and is usually the default choice.
- If a library is not ready or retrieval seems stale, check `show <library> --json` before assuming the query is wrong.

Do not assume remote search is needed if `plshelp` can answer the question locally.
<!-- plshelp:end -->
