pub(crate) use chrono::{DateTime, Utc};
pub(crate) use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
pub(crate) use rayon::prelude::*;
pub(crate) use regex::Regex;
pub(crate) use rusqlite::{params, Connection, OptionalExtension};
pub(crate) use scraper::{Html, Selector};
pub(crate) use serde::{Deserialize, Serialize};
pub(crate) use serde_json::{json, Value};
pub(crate) use spider::compact_str::CompactString;
pub(crate) use spider::configuration::Configuration;
pub(crate) use spider::website::Website;
pub(crate) use std::cmp::Ordering as CmpOrdering;
pub(crate) use std::collections::{HashMap, HashSet};
pub(crate) use std::env;
pub(crate) use std::error::Error;
pub(crate) use std::fs;
pub(crate) use std::io::{stdin, stdout, Write};
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::atomic::{AtomicBool, Ordering};
pub(crate) use std::sync::{Arc, LazyLock, Mutex, OnceLock};
pub(crate) use std::thread;
pub(crate) use std::time::{Duration, SystemTime, UNIX_EPOCH};
pub(crate) use url::Url;

pub(crate) const DEFAULT_TOP_K: usize = 1;
pub(crate) const DEFAULT_CONTEXT_WINDOW: usize = 0;
pub(crate) const DEFAULT_EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::AllMiniLML6V2Q;
pub(crate) const MIN_CHILD_LENGTH: usize = 400;
pub(crate) const MAX_CHILD_LENGTH: usize = 800;
pub(crate) const CHILD_SPLIT_WINDOW: usize = 50;
pub(crate) const DEFAULT_EMBED_BATCH_SIZE: usize = 128;
pub(crate) const SQLITE_BUSY_TIMEOUT_MS: u64 = 5_000;
pub(crate) const APP_NAME: &str = "plshelp";
pub(crate) const CONFIG_FILE_NAME: &str = "config.toml";
pub(crate) const DEFAULT_PARENT_MIN_CHARS: usize = 1400;
pub(crate) const DEFAULT_PARENT_MAX_CHARS: usize = 3000;
pub(crate) const PLSHELP_AGENT_BLOCK_START: &str = "<!-- plshelp:start -->";
pub(crate) const PLSHELP_AGENT_BLOCK_END: &str = "<!-- plshelp:end -->";
pub(crate) const AGENTS_TEMPLATE: &str = r#"<!-- plshelp:start -->
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

### Setup (if no libraries are indexed yet)

- `plshelp add <name> <docs-url>` to index a library
- `plshelp show <name> --json` to confirm it's ready before querying

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
"#;
pub(crate) const CLAUDE_TEMPLATE: &str = r#"<!-- plshelp:start -->
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

### Setup (if no libraries are indexed yet)

- `plshelp add <name> <docs-url>` to index a library
- `plshelp show <name> --json` to confirm it's ready before querying

Guidelines:

- Default to `query --json` for documentation questions tied to indexed libraries.
- Use `trace --json` when results look wrong and you need to inspect scores or ranking.
- Returned results are parent chunks; use `source_url` for citations.
- `hybrid` is usually the right retrieval mode unless there is a reason to force `keyword` or `vector`.
- Keep retrieval local through `plshelp` before reaching for external search.
<!-- plshelp:end -->
"#;

pub(crate) static CONTENT_SELECTORS: LazyLock<Vec<Selector>> = LazyLock::new(|| {
    [
        "article",
        "main",
        r#"[role=\"main\"]"#,
        ".content",
        ".docs-content",
        ".doc-content",
        ".markdown-body",
        ".theme-doc-markdown",
        "#content",
        "#main-content",
        "body",
    ]
    .iter()
    .filter_map(|s| Selector::parse(s).ok())
    .collect()
});

pub(crate) static HTML_CLEANUP_REGEXES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?is)<!--.*?-->",
        r"(?is)<script\b[^>]*>.*?</script>",
        r"(?is)<style\b[^>]*>.*?</style>",
        r"(?is)<noscript\b[^>]*>.*?</noscript>",
        r"(?is)<template\b[^>]*>.*?</template>",
        r"(?is)<nav\b[^>]*>.*?</nav>",
        r"(?is)<header\b[^>]*>.*?</header>",
        r"(?is)<footer\b[^>]*>.*?</footer>",
        r"(?is)<aside\b[^>]*>.*?</aside>",
        r#"(?is)<(div|section)[^>]*(id|class)\s*=\s*[\"'][^\"']*(nav|menu|sidebar|footer|header|toc|breadcrumb|pagination|cookie|consent|search|feedback|promo|banner|advert|ads|social|share)[^\"']*[\"'][^>]*>.*?</\1>"#,
        r"(?is)<button\b[^>]*>.*?</button>",
        r"(?is)<form\b[^>]*>.*?</form>",
    ]
    .iter()
    .filter_map(|p| Regex::new(p).ok())
    .collect()
});

pub(crate) static MARKDOWN_LINE_REGEXES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
    [
        r"(?im)^\s*was this helpful\?\s*$",
        r"(?im)^\s*copy page\s*$",
        r"(?im)^\s*menu\s*$",
        r"(?im)^\s*send\s*$",
        r"(?im)^\s*latest version\s*$",
        r"(?im)^\s*supported\.\s*$",
    ]
    .iter()
    .filter_map(|p| Regex::new(p).ok())
    .collect()
});

pub(crate) static MULTI_NEWLINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n{3,}").expect("valid regex"));
pub(crate) static MD_ATX_HEADING_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[ ]{0,3}#{1,6}[ \t]+.*$").expect("valid regex")
});

pub(crate) static HTML_REGEX_HINTS: &[&str] = &[
    "<!--",
    "<script",
    "<style",
    "<noscript",
    "<template",
    "<nav",
    "<header",
    "<footer",
    "<aside",
    "sidebar",
    "breadcrumb",
    "cookie",
    "<button",
    "<form",
];

pub(crate) static MARKDOWN_HINTS: &[&str] = &[
    "was this helpful?",
    "copy page",
    "menu",
    "send",
    "latest version",
    "supported.",
];

pub(crate) static RUNTIME_PATHS: OnceLock<RuntimePaths> = OnceLock::new();
pub(crate) static RUNTIME_SETTINGS: OnceLock<RuntimeSettings> = OnceLock::new();

#[derive(Debug, Clone)]
pub(crate) struct RuntimePaths {
    pub(crate) config_dir: PathBuf,
    pub(crate) config_file: PathBuf,
    pub(crate) data_dir: PathBuf,
    pub(crate) db_path: PathBuf,
    pub(crate) artifacts_dir: PathBuf,
    pub(crate) models_dir: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct RuntimeSettings {
    pub(crate) embedding_model: EmbeddingModel,
    pub(crate) embed_batch_size: usize,
    pub(crate) parent_min_chars: usize,
    pub(crate) parent_max_chars: usize,
    pub(crate) child_min_chars: usize,
    pub(crate) child_max_chars: usize,
    pub(crate) child_split_window_chars: usize,
    pub(crate) default_mode: SearchMode,
    pub(crate) default_top_k: usize,
    pub(crate) default_context_window: usize,
    pub(crate) hybrid_vector_weight: f32,
    pub(crate) hybrid_bm25_weight: f32,
    pub(crate) sqlite_journal_mode: String,
    pub(crate) sqlite_busy_timeout_ms: u64,
    pub(crate) onnx_intra_threads: usize,
    pub(crate) onnx_inter_threads: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct AppConfigFile {
    #[serde(default)]
    pub(crate) paths: PathsConfig,
    #[serde(default)]
    pub(crate) embedding: EmbeddingConfig,
    #[serde(default)]
    pub(crate) chunking: ChunkingConfig,
    #[serde(default)]
    pub(crate) retrieval: RetrievalConfig,
    #[serde(default)]
    pub(crate) sqlite: SqliteConfig,
    #[serde(default)]
    pub(crate) onnx: OnnxConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct PathsConfig {
    pub(crate) data_dir: Option<PathBuf>,
    pub(crate) db_path: Option<PathBuf>,
    pub(crate) artifacts_dir: Option<PathBuf>,
    pub(crate) models_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct EmbeddingConfig {
    pub(crate) model: Option<String>,
    pub(crate) batch_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct ChunkingConfig {
    pub(crate) parent_min_chars: Option<usize>,
    pub(crate) parent_max_chars: Option<usize>,
    pub(crate) child_min_chars: Option<usize>,
    pub(crate) child_max_chars: Option<usize>,
    pub(crate) child_split_window_chars: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct RetrievalConfig {
    pub(crate) default_mode: Option<String>,
    pub(crate) default_top_k: Option<usize>,
    pub(crate) default_context_window: Option<usize>,
    pub(crate) hybrid_vector_weight: Option<f32>,
    pub(crate) hybrid_bm25_weight: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct SqliteConfig {
    pub(crate) journal_mode: Option<String>,
    pub(crate) busy_timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct OnnxConfig {
    pub(crate) intra_threads: Option<usize>,
    pub(crate) inter_threads: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SearchMode {
    Hybrid,
    Vector,
    Keyword,
}

impl SearchMode {
    pub(crate) fn from_str(input: &str) -> Self {
        match input.to_ascii_lowercase().as_str() {
            "vector" => Self::Vector,
            "keyword" => Self::Keyword,
            _ => Self::Hybrid,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Vector => "vector",
            Self::Keyword => "keyword",
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ParentRecord {
    pub(crate) id: i64,
    pub(crate) library_name: String,
    pub(crate) source_url: String,
    pub(crate) source_page_order: i64,
    pub(crate) parent_index_in_page: i64,
    pub(crate) global_parent_index: i64,
    pub(crate) content: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ChunkRecord {
    pub(crate) id: i64,
    pub(crate) parent_id: i64,
    pub(crate) library_name: String,
    pub(crate) source_page_order: i64,
    pub(crate) parent_index_in_page: i64,
    pub(crate) child_index_in_parent: i64,
    pub(crate) global_chunk_index: i64,
    pub(crate) embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct ScoredChunk {
    pub(crate) chunk: ChunkRecord,
    pub(crate) vector_score: f32,
    pub(crate) bm25_score: f32,
    pub(crate) final_score: f32,
}

pub mod ui;
pub mod runtime;
pub mod db;
pub mod cli;
pub mod artifacts;
pub mod libraries;
pub mod commands;
pub mod crawl;
pub mod chunk;
pub mod embed;
pub mod search;

pub(crate) use artifacts::*;
pub(crate) use chunk::*;
pub(crate) use cli::*;
pub(crate) use commands::*;
pub(crate) use crawl::*;
pub(crate) use db::*;
pub(crate) use embed::*;
pub(crate) use libraries::*;
pub(crate) use runtime::*;
pub(crate) use search::*;
pub(crate) use ui::*;

fn parse_init_flags(flags: &[String]) -> Result<(bool, bool, bool), Box<dyn Error>> {
    let mut write_agents = false;
    let mut write_claude = false;
    let mut print_only = false;

    for flag in flags {
        match flag.as_str() {
            "--agents" => write_agents = true,
            "--claude" => write_claude = true,
            "--print" => print_only = true,
            _ => return Err("Usage: plshelp init [--agents] [--claude] [--print] [--json]".into()),
        }
    }

    if !write_agents && !write_claude {
        write_agents = true;
        write_claude = true;
    }

    Ok((write_agents, write_claude, print_only))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UninstallScope {
    All,
    Data,
    Binary,
}

fn parse_uninstall_flags(flags: &[String]) -> Result<UninstallScope, Box<dyn Error>> {
    let mut scope: Option<UninstallScope> = None;
    for flag in flags {
        let next = match flag.as_str() {
            "--all" => UninstallScope::All,
            "--data" => UninstallScope::Data,
            "--binary" => UninstallScope::Binary,
            _ => return Err("Usage: plshelp uninstall --all | --data | --binary".into()),
        };
        if scope.replace(next).is_some() {
            return Err("Usage: plshelp uninstall --all | --data | --binary".into());
        }
    }
    scope.ok_or_else(|| "Usage: plshelp uninstall --all | --data | --binary".into())
}

fn ensure_safe_delete_path(path: &Path, label: &str) -> Result<(), Box<dyn Error>> {
    if path.as_os_str().is_empty() {
        return Err(format!("Refusing to delete empty {} path.", label).into());
    }
    if !path.is_absolute() {
        return Err(format!("Refusing to delete non-absolute {} path: {}", label, path.display()).into());
    }
    if path == Path::new("/") {
        return Err(format!("Refusing to delete root {} path.", label).into());
    }
    if let Some(home) = home_dir() {
        if path == home {
            return Err(format!("Refusing to delete home directory as {} path.", label).into());
        }
    }
    if let Ok(cwd) = env::current_dir() {
        if path == cwd {
            return Err(format!("Refusing to delete current working directory as {} path.", label).into());
        }
    }
    let component_count = path.components().count();
    if component_count < 3 {
        return Err(format!(
            "Refusing to delete suspiciously broad {} path: {}",
            label,
            path.display()
        )
        .into());
    }
    Ok(())
}

fn prompt_exact_path_confirmation(expected: &Path) -> Result<(), Box<dyn Error>> {
    println!();
    println!("WARNING: THIS IS A DESTRUCTIVE OPERATION.");
    println!("WARNING: ONCE DELETED, THIS DATA CANNOT BE RECOVERED BY PLSHELP.");
    println!();
    println!("Type this exact path to confirm deletion:");
    println!("  {}", expected.display());
    print!("> ");
    stdout().flush()?;
    let mut input = String::new();
    stdin().read_line(&mut input)?;
    let typed = input.trim();
    let expected_str = expected.display().to_string();
    if typed != expected_str {
        return Err("Confirmation path did not match. Aborting.".into());
    }
    Ok(())
}

fn print_uninstall_plan(
    scope: UninstallScope,
    binary_path: &Path,
    runtime: &RuntimePaths,
    config_path: &Path,
) {
    match scope {
        UninstallScope::Binary => {
            println!("Deleting binary:");
            println!("  {}", binary_path.display());
            println!();
            println!("Data will remain at:");
            println!("  {}", runtime.data_dir.display());
            println!("  {}", config_path.display());
            println!("  {}", runtime.artifacts_dir.display());
            println!("  {}", runtime.models_dir.display());
            println!("  {}", runtime.db_path.display());
            println!();
            println!("NOTE: This only removes the executable.");
            println!("If you want to delete the local database, artifacts, model cache, and config too, use:");
            println!("  plshelp uninstall --all");
        }
        UninstallScope::Data => {
            println!("Deleting data:");
            println!("  {}", runtime.data_dir.display());
            if runtime.config_dir != runtime.data_dir {
                println!("  {}", runtime.config_dir.display());
            }
            println!();
            println!("Binary will remain at:");
            println!("  {}", binary_path.display());
            println!();
            println!("NOTE: This only removes local runtime data.");
            println!("If you also want to remove the installed executable, use:");
            println!("  plshelp uninstall --all");
        }
        UninstallScope::All => {
            println!("Deleting binary:");
            println!("  {}", binary_path.display());
            println!();
            println!("Deleting data:");
            println!("  {}", runtime.data_dir.display());
            if runtime.config_dir != runtime.data_dir {
                println!("  {}", runtime.config_dir.display());
            }
            println!();
            println!("Nothing else under plshelp runtime paths will remain.");
        }
    }
}

fn uninstall_scope(scope: UninstallScope) -> Result<(), Box<dyn Error>> {
    let runtime = runtime_paths().clone();
    let config_path = config_file_path();
    let binary_path = env::current_exe()?;

    match scope {
        UninstallScope::Binary => {
            ensure_safe_delete_path(&binary_path, "binary")?;
        }
        UninstallScope::Data => {
            ensure_safe_delete_path(&runtime.data_dir, "data")?;
            if runtime.config_dir != runtime.data_dir {
                ensure_safe_delete_path(&runtime.config_dir, "config")?;
            }
        }
        UninstallScope::All => {
            ensure_safe_delete_path(&binary_path, "binary")?;
            ensure_safe_delete_path(&runtime.data_dir, "data")?;
            if runtime.config_dir != runtime.data_dir {
                ensure_safe_delete_path(&runtime.config_dir, "config")?;
            }
        }
    }

    print_uninstall_plan(scope, &binary_path, &runtime, &config_path);
    println!();

    let confirm_path = match scope {
        UninstallScope::Binary => binary_path.as_path(),
        UninstallScope::Data | UninstallScope::All => runtime.data_dir.as_path(),
    };
    prompt_exact_path_confirmation(confirm_path)?;

    match scope {
        UninstallScope::Binary => {
            fs::remove_file(&binary_path)?;
            println!("Deleted {}", binary_path.display());
        }
        UninstallScope::Data => {
            if runtime.config_dir != runtime.data_dir && runtime.config_dir.exists() {
                fs::remove_dir_all(&runtime.config_dir)?;
                println!("Deleted {}", runtime.config_dir.display());
            }
            if runtime.data_dir.exists() {
                fs::remove_dir_all(&runtime.data_dir)?;
                println!("Deleted {}", runtime.data_dir.display());
            }
        }
        UninstallScope::All => {
            if runtime.config_dir != runtime.data_dir && runtime.config_dir.exists() {
                fs::remove_dir_all(&runtime.config_dir)?;
                println!("Deleted {}", runtime.config_dir.display());
            }
            if runtime.data_dir.exists() {
                fs::remove_dir_all(&runtime.data_dir)?;
                println!("Deleted {}", runtime.data_dir.display());
            }
            fs::remove_file(&binary_path)?;
        }
    }

    Ok(())
}

fn upsert_marked_block(existing: &str, block: &str) -> String {
    if let (Some(start), Some(end)) = (
        existing.find(PLSHELP_AGENT_BLOCK_START),
        existing.find(PLSHELP_AGENT_BLOCK_END),
    ) {
        let end = end + PLSHELP_AGENT_BLOCK_END.len();
        let mut updated = String::new();
        updated.push_str(&existing[..start]);
        if !updated.is_empty() && !updated.ends_with('\n') {
            updated.push('\n');
        }
        updated.push_str(block);
        if end < existing.len() {
            if !updated.ends_with('\n') {
                updated.push('\n');
            }
            updated.push_str(existing[end..].trim_start_matches('\n'));
        }
        return updated;
    }

    if existing.trim().is_empty() {
        return format!("{block}\n");
    }

    let mut updated = existing.trim_end().to_string();
    updated.push_str("\n\n");
    updated.push_str(block);
    updated.push('\n');
    updated
}

fn write_agent_file(path: &Path, block: &str) -> Result<(), Box<dyn Error>> {
    let existing = fs::read_to_string(path).unwrap_or_default();
    let updated = upsert_marked_block(&existing, block);
    fs::write(path, updated)?;
    Ok(())
}

pub async fn run(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let _echo_guard = TerminalEchoGuard::new();
    initialize_runtime_paths()?;
    configure_onnx_runtime_env();

    if args.is_empty() {
        print_help();
        return Ok(());
    }

    let command = args[0].as_str();

    match command {
        "uninstall" => {
            let scope = parse_uninstall_flags(&args[1..])?;
            uninstall_scope(scope)?;
        }
        "add" => {
            let conn = init_db(&db_path())?;
            if args.len() < 3 {
                return Err(
                    "Usage: plshelp add <library_name> <source_url> [--single] [--respect-robots] [--include-artifacts[=/path]]".into(),
                );
            }
            let (output_json, flags) = extract_json_flag(&args[3..]);
            let (single_page, flags) = extract_single_flag(&flags);
            let (respect_robots, flags) = extract_respect_robots_flag(&flags);
            let include_artifacts = parse_include_artifacts_flag(&flags, &args[1]);
            add_library(
                &conn,
                &args[1],
                &args[2],
                single_page,
                respect_robots,
                include_artifacts,
            )
            .await?;
            print_command_result(
                "add",
                output_json,
                json!({
                    "library_name": args[1],
                    "source_url": args[2],
                    "single_page": single_page,
                    "respect_robots": respect_robots,
                    "artifacts_path": flags.iter().find_map(|f| f.strip_prefix("--include-artifacts=").map(|s| s.to_string())),
                }),
            )?;
        }
        "crawl" => {
            let conn = init_db(&db_path())?;
            if args.len() < 3 {
                return Err(
                    "Usage: plshelp crawl <library_name> <source_url> [--single] [--respect-robots] [--include-artifacts[=/path]]".into(),
                );
            }
            let (output_json, flags) = extract_json_flag(&args[3..]);
            let (single_page, flags) = extract_single_flag(&flags);
            let (respect_robots, flags) = extract_respect_robots_flag(&flags);
            let include_artifacts = parse_include_artifacts_flag(&flags, &args[1]);
            crawl_library(
                &conn,
                &args[1],
                &args[2],
                single_page,
                respect_robots,
                "crawl",
                include_artifacts,
            )
            .await?;
            print_command_result(
                "crawl",
                output_json,
                json!({
                    "library_name": args[1],
                    "source_url": args[2],
                    "single_page": single_page,
                    "respect_robots": respect_robots,
                }),
            )?;
        }
        "init" => {
            let (output_json, flags) = extract_json_flag(&args[1..]);
            let (write_agents, write_claude, print_only) = parse_init_flags(&flags)?;
            let cwd = env::current_dir()?;
            let agents_path = cwd.join("AGENTS.md");
            let claude_path = cwd.join("CLAUDE.md");

            if !print_only {
                if write_agents {
                    write_agent_file(&agents_path, AGENTS_TEMPLATE)?;
                }
                if write_claude {
                    write_agent_file(&claude_path, CLAUDE_TEMPLATE)?;
                }
            }

            let mut written = Vec::new();
            let mut templates = serde_json::Map::new();

            if write_agents {
                if !print_only {
                    written.push(agents_path.clone());
                }
                templates.insert("AGENTS.md".to_string(), Value::String(AGENTS_TEMPLATE.to_string()));
            }
            if write_claude {
                if !print_only {
                    written.push(claude_path.clone());
                }
                templates.insert("CLAUDE.md".to_string(), Value::String(CLAUDE_TEMPLATE.to_string()));
            }

            if output_json {
                print_json(&json!({
                    "command": "init",
                    "status": "success",
                    "result": {
                        "print_only": print_only,
                        "files": written,
                        "templates": templates,
                    }
                }))?;
            } else if print_only {
                if write_agents {
                    println!("--- AGENTS.md ---\n");
                    print!("{}", AGENTS_TEMPLATE);
                    if write_claude {
                        println!();
                    }
                }
                if write_claude {
                    println!("--- CLAUDE.md ---\n");
                    print!("{}", CLAUDE_TEMPLATE);
                }
            } else {
                for path in &written {
                    println!("{}", path.display());
                }
            }
        }
        "index" => {
            let conn = init_db(&db_path())?;
            if args.len() < 2 {
                return Err("Usage: plshelp index <library_name> [--file /path/to/file]".into());
            }
            let (output_json, flags) = extract_json_flag(&args[2..]);
            let file = parse_index_file_flag(&flags);
            index_library(&conn, &args[1], file.as_deref())?;
            print_command_result(
                "index",
                output_json,
                json!({
                    "input_name": args[1],
                    "file": file,
                }),
            )?;
        }
        "chunk" => {
            let conn = init_db(&db_path())?;
            if args.len() < 2 {
                return Err("Usage: plshelp chunk <library_name> [--file /path/to/file]".into());
            }
            let (output_json, flags) = extract_json_flag(&args[2..]);
            let file = parse_index_file_flag(&flags);
            chunk_targets(&conn, &args[1], file.as_deref(), "chunk")?;
            print_command_result(
                "chunk",
                output_json,
                json!({
                    "input_name": args[1],
                    "file": file,
                }),
            )?;
        }
        "embed" => {
            let conn = init_db(&db_path())?;
            if args.len() < 2 {
                return Err("Usage: plshelp embed <library_name>".into());
            }
            let (output_json, flags) = extract_json_flag(&args[2..]);
            if !flags.is_empty() {
                return Err("Usage: plshelp embed <library_name> [--json]".into());
            }
            embed_library(&conn, &args[1], "embed")?;
            print_command_result(
                "embed",
                output_json,
                json!({
                    "input_name": args[1],
                }),
            )?;
        }
        "refresh" => {
            let conn = init_db(&db_path())?;
            let (output_json, flags) = extract_json_flag(&args[1..]);
            refresh_stats(&conn, &flags)?;
            print_command_result(
                "refresh",
                output_json,
                json!({
                    "targets": flags,
                }),
            )?;
        }
        "merge" => {
            let conn = init_db(&db_path())?;
            if args.len() < 4 {
                return Err("Usage: plshelp merge <new_library_name> <library1> <library2> [library3 ...] [--replace] [--include-artifacts[=/path]]".into());
            }
            let (output_json, flags) = extract_json_flag(&args[2..]);
            let (members, replace, include_artifacts) = parse_merge_args(&flags, &args[1])?;
            merge_libraries(
                &conn,
                &args[1],
                &members,
                replace,
                include_artifacts.as_deref(),
            )?;
            print_command_result(
                "merge",
                output_json,
                json!({
                    "group_name": args[1],
                    "members": members,
                    "replace": replace,
                    "artifacts_path": include_artifacts,
                }),
            )?;
        }
        "export" => {
            let conn = init_db(&db_path())?;
            if args.len() < 2 {
                return Err("Usage: plshelp export <library_name> [path]".into());
            }
            let (output_json, flags) = extract_json_flag(&args[2..]);
            let output_dir = if !flags.is_empty() {
                Some(PathBuf::from(flags[0].clone()))
            } else {
                None
            };
            export_library(&conn, &args[1], output_dir.as_deref())?;
            print_command_result(
                "export",
                output_json,
                json!({
                    "input_name": args[1],
                    "output_dir": output_dir,
                }),
            )?;
        }
        "query" => {
            let conn = init_db(&db_path())?;
            if args.len() < 3 {
                return Err(
                    "Usage: plshelp query <library_name> \"<question>\" [--mode hybrid|vector|keyword] [--top-k N] [--context N]".into(),
                );
            }
            let (question, flags) = split_query_and_flags(&args[2..]);
            if question.is_empty() {
                return Err("Usage: plshelp query <library_name> <question> [--mode ...]".into());
            }
            let (output_json, flags) = extract_json_flag(&flags);
            let (mode, top_k, context) = parse_query_flags(&flags)?;
            query_library(
                &conn,
                &args[1],
                &question,
                mode,
                top_k,
                context,
                false,
                output_json,
            )?;
        }
        "trace" => {
            let conn = init_db(&db_path())?;
            if args.len() < 3 {
                return Err(
                    "Usage: plshelp trace <library_name> \"<question>\" [--mode hybrid|vector|keyword] [--top-k N] [--context N]".into(),
                );
            }
            let (question, flags) = split_query_and_flags(&args[2..]);
            if question.is_empty() {
                return Err("Usage: plshelp trace <library_name> <question> [--mode ...]".into());
            }
            let (output_json, flags) = extract_json_flag(&flags);
            let (mode, top_k, context) = parse_query_flags(&flags)?;
            query_library(
                &conn,
                &args[1],
                &question,
                mode,
                top_k,
                context,
                true,
                output_json,
            )?;
        }
        "ask" => {
            let conn = init_db(&db_path())?;
            if args.len() < 2 {
                return Err(
                    "Usage: plshelp ask \"<question>\" [--libraries a,b,c] [--mode ...] [--top-k N] [--context N]".into(),
                );
            }
            let (question, flags) = split_query_and_flags(&args[1..]);
            if question.is_empty() {
                return Err("Usage: plshelp ask <question> [--libraries ...] [--mode ...]".into());
            }
            let (output_json, flags) = extract_json_flag(&flags);
            ask_libraries(&conn, &question, &flags, output_json)?;
        }
        "alias" => {
            let conn = init_db(&db_path())?;
            if args.len() < 3 {
                return Err("Usage: plshelp alias <library_name> <alias>".into());
            }
            let (output_json, flags) = extract_json_flag(&args[3..]);
            if !flags.is_empty() {
                return Err("Usage: plshelp alias <library_name> <alias> [--json]".into());
            }
            add_alias(&conn, &args[1], &args[2])?;
            print_command_result(
                "alias",
                output_json,
                json!({
                    "input_name": args[1],
                    "alias": args[2],
                }),
            )?;
        }
        "list" => {
            let conn = init_db(&db_path())?;
            let (output_json, _flags) = extract_json_flag(&args[1..]);
            list_libraries(&conn, output_json)?;
        }
        "config" => {
            let (output_json, flags) = extract_json_flag(&args[1..]);
            if !flags.is_empty() {
                return Err("Usage: plshelp config [--json]".into());
            }
            let path = config_file_path();
            let raw = fs::read_to_string(&path)?;
            if output_json {
                print_json(&json!({
                    "command": "config",
                    "status": "success",
                    "result": {
                        "path": path,
                        "content": raw,
                    }
                }))?;
            } else {
                println!("{}", path.display());
                println!();
                print!("{}", raw);
            }
        }
        "show" => {
            let conn = init_db(&db_path())?;
            if args.len() < 2 {
                return Err("Usage: plshelp show <library_name>".into());
            }
            let (output_json, flags) = extract_json_flag(&args[2..]);
            if !flags.is_empty() {
                return Err("Usage: plshelp show <library_name> [--json]".into());
            }
            show_library(&conn, &args[1], output_json)?;
        }
        "remove" => {
            let conn = init_db(&db_path())?;
            if args.len() < 2 {
                return Err("Usage: plshelp remove <library_name>".into());
            }
            let (output_json, flags) = extract_json_flag(&args[2..]);
            if !flags.is_empty() {
                return Err("Usage: plshelp remove <library_name> [--json]".into());
            }
            remove_library(&conn, &args[1])?;
            print_command_result(
                "remove",
                output_json,
                json!({
                    "input_name": args[1],
                }),
            )?;
        }
        "open" => {
            let conn = init_db(&db_path())?;
            if args.len() < 2 {
                return Err("Usage: plshelp open <chunk_id>".into());
            }
            let chunk_id: i64 = args[1].parse()?;
            let (output_json, flags) = extract_json_flag(&args[2..]);
            if !flags.is_empty() {
                return Err("Usage: plshelp open <chunk_id> [--json]".into());
            }
            open_chunk(&conn, chunk_id, output_json)?;
        }
        "help" | "--help" | "-h" => print_help(),
        _ => {
            let conn = init_db(&db_path())?;
            if args.len() < 2 {
                return Err("Usage: plshelp <library_name> \"<question>\"".into());
            }
            let (question, flags) = split_query_and_flags(&args[1..]);
            if question.is_empty() {
                return Err("Usage: plshelp <library_name> <question> [--mode ...]".into());
            }
            let (output_json, flags) = extract_json_flag(&flags);
            let (mode, top_k, context) = parse_query_flags(&flags)?;
            query_library(
                &conn,
                &args[0],
                &question,
                mode,
                top_k,
                context,
                false,
                output_json,
            )?;
        }
    }

    Ok(())
}
