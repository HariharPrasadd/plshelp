# plshelp

`plshelp` is a local-first Rust CLI for crawling documentation, indexing it into local search context, and querying it for RAG-style workflows.

## Sample `config.toml`

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
