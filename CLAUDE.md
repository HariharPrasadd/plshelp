<!-- plshelp:start -->
## plshelp

Use `plshelp` for local documentation retrieval in this project.

### Setup (if no libraries are indexed yet)

- `plshelp add <name> <docs-url>` to index a library
- `plshelp show <name> --json` to confirm it's ready before querying

Recommended commands:

- `plshelp query <library> "<question>" --json`
- `plshelp trace <library> "<question>" --json`
- `plshelp ask "<question>" --libraries a,b,c --json`
- `plshelp list --json`
- `plshelp show <library> --json`
- `plshelp open <chunk_id> --json`
- `plshelp config --json`

Guidelines:

- Default to `query --json` for documentation questions tied to indexed libraries.
- Use `trace --json` when results look wrong and you need to inspect scores or ranking.
- Returned results are parent chunks; use `source_url` for citations.
- `hybrid` is usually the right retrieval mode unless there is a reason to force `keyword` or `vector`.
- Keep retrieval local through `plshelp` before reaching for external search.
<!-- plshelp:end -->
