use chrono::{DateTime, Utc};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use rayon::prelude::*;
use regex::Regex;
use rusqlite::{Connection, OptionalExtension, params};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use spider::compact_str::CompactString;
use spider::configuration::Configuration;
use spider::tokio;
use spider::website::Website;
use std::cmp::Ordering as CmpOrdering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error;
use std::fs;
use std::io::{Write, stdin, stdout};
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use termios::{ECHO, TCSANOW, Termios, tcsetattr};
use url::Url;

const DEFAULT_TOP_K: usize = 1;
const DEFAULT_CONTEXT_WINDOW: usize = 0;
const DEFAULT_EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::AllMiniLML6V2Q;
const MIN_CHILD_LENGTH: usize = 700;
const MAX_CHILD_LENGTH: usize = 1400;
const CHILD_SPLIT_WINDOW: usize = 50;
const DEFAULT_EMBED_BATCH_SIZE: usize = 128;
const SQLITE_BUSY_TIMEOUT_MS: u64 = 5_000;
const APP_NAME: &str = "plshelp";
const CONFIG_FILE_NAME: &str = "config.toml";

static CONTENT_SELECTORS: LazyLock<Vec<Selector>> = LazyLock::new(|| {
    [
        "article",
        "main",
        r#"[role="main"]"#,
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

static HTML_CLEANUP_REGEXES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
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
        r#"(?is)<(div|section)[^>]*(id|class)\s*=\s*["'][^"']*(nav|menu|sidebar|footer|header|toc|breadcrumb|pagination|cookie|consent|search|feedback|promo|banner|advert|ads|social|share)[^"']*["'][^>]*>.*?</\1>"#,
        r"(?is)<button\b[^>]*>.*?</button>",
        r"(?is)<form\b[^>]*>.*?</form>",
    ]
    .iter()
    .filter_map(|p| Regex::new(p).ok())
    .collect()
});

static MARKDOWN_LINE_REGEXES: LazyLock<Vec<Regex>> = LazyLock::new(|| {
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

static MULTI_NEWLINE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\n{3,}").expect("valid regex"));
static MD_ATX_HEADING_RE: LazyLock<Regex> = LazyLock::new(|| {
    // CommonMark-style ATX headings:
    // up to 3 leading spaces, 1-6 hashes, then at least one space/tab.
    Regex::new(r"^[ ]{0,3}#{1,6}[ \t]+.*$").expect("valid regex")
});

static HTML_REGEX_HINTS: &[&str] = &[
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

static MARKDOWN_HINTS: &[&str] = &[
    "was this helpful?",
    "copy page",
    "menu",
    "send",
    "latest version",
    "supported.",
];

static RUNTIME_PATHS: OnceLock<RuntimePaths> = OnceLock::new();

#[derive(Debug, Clone)]
struct RuntimePaths {
    config_dir: PathBuf,
    config_file: PathBuf,
    data_dir: PathBuf,
    db_path: PathBuf,
    artifacts_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AppConfigFile {
    #[serde(default)]
    paths: PathsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PathsConfig {
    data_dir: Option<PathBuf>,
    db_path: Option<PathBuf>,
    artifacts_dir: Option<PathBuf>,
}

struct TerminalEchoGuard {
    fd: i32,
    original: Termios,
}

impl TerminalEchoGuard {
    fn new() -> Option<Self> {
        let fd = stdin().as_raw_fd();
        let mut current = Termios::from_fd(fd).ok()?;
        let original = current.clone();
        current.c_lflag &= !ECHO;
        tcsetattr(fd, TCSANOW, &current).ok()?;
        Some(Self { fd, original })
    }
}

impl Drop for TerminalEchoGuard {
    fn drop(&mut self) {
        let _ = tcsetattr(self.fd, TCSANOW, &self.original);
    }
}

struct ProgressSpinner {
    done: Arc<AtomicBool>,
    stage: Arc<Mutex<String>>,
    handle: Option<thread::JoinHandle<()>>,
}

impl ProgressSpinner {
    fn new(initial_stage: impl Into<String>) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let stage = Arc::new(Mutex::new(initial_stage.into()));
        let done_for_spinner = Arc::clone(&done);
        let stage_for_spinner = Arc::clone(&stage);
        let handle = thread::spawn(move || {
            let frames = ["|", "/", "-", "\\"];
            let mut idx = 0usize;
            let mut last_len = 0usize;
            while !done_for_spinner.load(Ordering::Relaxed) {
                let current_stage = stage_for_spinner
                    .lock()
                    .map(|s| s.clone())
                    .unwrap_or_else(|_| String::from("Working"));
                let line = format!("{}... {}", current_stage, frames[idx % frames.len()]);
                let padding = " ".repeat(last_len.saturating_sub(line.len()));
                print!("\r{}{}", line, padding);
                let _ = stdout().flush();
                last_len = line.len();
                idx += 1;
                thread::sleep(Duration::from_millis(150));
            }
        });
        Self {
            done,
            stage,
            handle: Some(handle),
        }
    }

    fn set_stage(&self, message: impl Into<String>) {
        if let Ok(mut stage) = self.stage.lock() {
            *stage = message.into();
        }
    }

    fn finish(mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        print!("\r{}\r", " ".repeat(80));
        let _ = stdout().flush();
    }
}

impl Drop for ProgressSpinner {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        print!("\r{}\r", " ".repeat(80));
        let _ = stdout().flush();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SearchMode {
    Hybrid,
    Vector,
    Keyword,
}

impl SearchMode {
    fn from_str(input: &str) -> Self {
        match input.to_ascii_lowercase().as_str() {
            "vector" => Self::Vector,
            "keyword" => Self::Keyword,
            _ => Self::Hybrid,
        }
    }
}

impl SearchMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Hybrid => "hybrid",
            Self::Vector => "vector",
            Self::Keyword => "keyword",
        }
    }
}

#[derive(Debug, Clone)]
struct ParentRecord {
    id: i64,
    library_name: String,
    source_url: String,
    source_page_order: i64,
    parent_index_in_page: i64,
    global_parent_index: i64,
    content: String,
}

#[derive(Debug, Clone)]
struct ChunkRecord {
    id: i64,
    parent_id: i64,
    library_name: String,
    source_page_order: i64,
    parent_index_in_page: i64,
    child_index_in_parent: i64,
    global_chunk_index: i64,
    embedding: Vec<f32>,
}

#[derive(Debug, Clone)]
struct ScoredChunk {
    chunk: ChunkRecord,
    vector_score: f32,
    bm25_score: f32,
    final_score: f32,
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let output_json = args.iter().any(|arg| arg == "--json");
    if let Err(err) = run(args).await {
        if output_json {
            let payload = json!({
                "status": "error",
                "error": format!("{err}"),
            });
            eprintln!(
                "{}",
                serde_json::to_string_pretty(&payload)
                    .unwrap_or_else(|_| format!(r#"{{"status":"error","error":"{}"}}"#, err))
            );
        } else {
            eprintln!("Error: {err}");
        }
        std::process::exit(1);
    }
}

async fn run(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    let _echo_guard = TerminalEchoGuard::new();
    configure_onnx_runtime_env();
    initialize_runtime_paths()?;

    if args.is_empty() {
        print_help();
        return Ok(());
    }

    let conn = init_db(&db_path())?;
    let command = args[0].as_str();

    match command {
        "add" => {
            if args.len() < 3 {
                return Err(
                    "Usage: plshelp add <library_name> <source_url> [--include-artifacts[=/path]]"
                        .into(),
                );
            }
            let (output_json, flags) = extract_json_flag(&args[3..]);
            let include_artifacts = parse_include_artifacts_flag(&flags, &args[1]);
            add_library(&conn, &args[1], &args[2], include_artifacts).await?;
            print_command_result(
                "add",
                output_json,
                json!({
                    "library_name": args[1],
                    "source_url": args[2],
                    "artifacts_path": flags.iter().find_map(|f| f.strip_prefix("--include-artifacts=").map(|s| s.to_string())),
                }),
            )?;
        }
        "crawl" => {
            if args.len() < 3 {
                return Err(
                    "Usage: plshelp crawl <library_name> <source_url> [--include-artifacts[=/path]]"
                        .into(),
                );
            }
            let (output_json, flags) = extract_json_flag(&args[3..]);
            let include_artifacts = parse_include_artifacts_flag(&flags, &args[1]);
            crawl_library(&conn, &args[1], &args[2], "crawl", include_artifacts).await?;
            print_command_result(
                "crawl",
                output_json,
                json!({
                    "library_name": args[1],
                    "source_url": args[2],
                }),
            )?;
        }
        "index" => {
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
            if args.len() < 3 {
                return Err(
                    "Usage: plshelp query <library_name> \"<question>\" [--mode hybrid|vector|keyword] [--top-k N] [--context N]"
                        .into(),
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
            if args.len() < 3 {
                return Err(
                    "Usage: plshelp trace <library_name> \"<question>\" [--mode hybrid|vector|keyword] [--top-k N] [--context N]"
                        .into(),
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
            if args.len() < 2 {
                return Err(
                    "Usage: plshelp ask \"<question>\" [--libraries a,b,c] [--mode ...] [--top-k N] [--context N]"
                        .into(),
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
            let (output_json, _flags) = extract_json_flag(&args[1..]);
            list_libraries(&conn, output_json)?;
        }
        "show" => {
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

fn print_help() {
    println!("plshelp <command>");
    println!("  add <library_name> <source_url> [--include-artifacts[=/path]] [--json]");
    println!("  crawl <library_name> <source_url> [--include-artifacts[=/path]] [--json]");
    println!("  index <library_name> [--file /path/to/file] [--json]");
    println!("  chunk <library_name> [--file /path/to/file] [--json]");
    println!("  embed <library_name> [--json]");
    println!("  refresh [library_name ...] [--json]   # recompute/backfill stats; no crawl");
    println!(
        "  merge <new_library_name> <library1> <library2> [library3 ...] [--replace] [--include-artifacts[=/path]] [--json]"
    );
    println!("  export <library_name> [path] [--json]");
    println!(
        "  query <library_name> \"<question>\" [--mode hybrid|vector|keyword] [--top-k N] [--context N] [--json]"
    );
    println!("  <library_name> \"<question>\"   # query alias");
    println!(
        "  ask \"<question>\" [--libraries a,b,c] [--mode ...] [--top-k N] [--context N] [--json]"
    );
    println!("  note: quote questions in the shell, especially if they contain ? or *");
    println!("  alias <library_name> <alias>");
    println!("  list [--json]");
    println!("  show <library_name> [--json]");
    println!("  remove <library_name> [--json]");
    println!("  open <chunk_id> [--json]");
    println!(
        "  trace <library_name> \"<question>\" [--mode ...] [--top-k N] [--context N] [--json]"
    );
}

fn configure_onnx_runtime_env() {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // These must be set before ONNX Runtime initializes.
    unsafe {
        env::set_var("OMP_NUM_THREADS", num_cpus.to_string());
        env::set_var("ORT_NUM_INTRA_THREADS", num_cpus.to_string());
        env::set_var("ORT_NUM_INTER_THREADS", "1");
    }
}

fn runtime_paths() -> &'static RuntimePaths {
    RUNTIME_PATHS
        .get()
        .expect("runtime paths must be initialized before use")
}

fn db_path() -> PathBuf {
    runtime_paths().db_path.clone()
}

fn artifacts_root() -> PathBuf {
    runtime_paths().artifacts_dir.clone()
}

fn compiled_dir(library_name: &str) -> PathBuf {
    artifacts_root().join(library_name)
}

fn initialize_runtime_paths() -> Result<&'static RuntimePaths, Box<dyn Error>> {
    if let Some(paths) = RUNTIME_PATHS.get() {
        return Ok(paths);
    }

    let default_config_dir = default_config_dir()?;
    let default_data_dir = default_data_dir()?;
    let default_paths = RuntimePaths {
        config_file: default_config_dir.join(CONFIG_FILE_NAME),
        config_dir: default_config_dir,
        data_dir: default_data_dir.clone(),
        db_path: default_data_dir.join("plshelp.db"),
        artifacts_dir: default_data_dir.join("artifacts"),
    };

    fs::create_dir_all(&default_paths.config_dir)?;
    fs::create_dir_all(&default_paths.data_dir)?;
    fs::create_dir_all(&default_paths.artifacts_dir)?;

    if !default_paths.config_file.exists() {
        write_default_config(&default_paths)?;
    }

    let config = load_config_file(&default_paths.config_file)?;
    let data_dir = resolve_config_path(
        config.paths.data_dir.as_ref(),
        &default_paths.data_dir,
        &default_paths.config_dir,
    );
    let db_path = resolve_config_path(
        config.paths.db_path.as_ref(),
        &data_dir.join("plshelp.db"),
        &default_paths.config_dir,
    );
    let artifacts_dir = resolve_config_path(
        config.paths.artifacts_dir.as_ref(),
        &data_dir.join("artifacts"),
        &default_paths.config_dir,
    );

    fs::create_dir_all(&data_dir)?;
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir_all(&artifacts_dir)?;

    let runtime = RuntimePaths {
        config_dir: default_paths.config_dir,
        config_file: default_paths.config_file,
        data_dir,
        db_path,
        artifacts_dir,
    };
    let _ = RUNTIME_PATHS.set(runtime);
    Ok(runtime_paths())
}

fn write_default_config(defaults: &RuntimePaths) -> Result<(), Box<dyn Error>> {
    let config = AppConfigFile {
        paths: PathsConfig {
            data_dir: Some(defaults.data_dir.clone()),
            db_path: Some(defaults.db_path.clone()),
            artifacts_dir: Some(defaults.artifacts_dir.clone()),
        },
    };
    let serialized = toml::to_string_pretty(&config)?;
    fs::write(&defaults.config_file, serialized)?;
    Ok(())
}

fn load_config_file(path: &Path) -> Result<AppConfigFile, Box<dyn Error>> {
    let raw = fs::read_to_string(path)?;
    let config = toml::from_str::<AppConfigFile>(&raw)?;
    Ok(config)
}

fn resolve_config_path(value: Option<&PathBuf>, fallback: &Path, base_dir: &Path) -> PathBuf {
    match value {
        Some(path) => {
            let expanded = expand_home(path);
            if expanded.is_absolute() {
                expanded
            } else {
                base_dir.join(expanded)
            }
        }
        None => fallback.to_path_buf(),
    }
}

fn expand_home(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        return home_dir().unwrap_or_else(|| path.to_path_buf());
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    path.to_path_buf()
}

fn home_dir() -> Option<PathBuf> {
    if let Some(home) = env::var_os("HOME") {
        return Some(PathBuf::from(home));
    }
    if let Some(profile) = env::var_os("USERPROFILE") {
        return Some(PathBuf::from(profile));
    }
    let drive = env::var_os("HOMEDRIVE");
    let path = env::var_os("HOMEPATH");
    match (drive, path) {
        (Some(drive), Some(path)) => {
            let mut buf = PathBuf::from(drive);
            buf.push(path);
            Some(buf)
        }
        _ => None,
    }
}

fn default_config_dir() -> Result<PathBuf, Box<dyn Error>> {
    if cfg!(target_os = "macos") {
        let home = home_dir().ok_or("Unable to resolve home directory for config path.")?;
        return Ok(home
            .join("Library")
            .join("Application Support")
            .join(APP_NAME));
    }
    if cfg!(target_os = "windows") {
        let appdata = env::var_os("APPDATA")
            .map(PathBuf::from)
            .ok_or("APPDATA is not set.")?;
        return Ok(appdata.join(APP_NAME));
    }
    if let Some(xdg) = env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(xdg).join(APP_NAME));
    }
    let home = home_dir().ok_or("Unable to resolve home directory for config path.")?;
    Ok(home.join(".config").join(APP_NAME))
}

fn default_data_dir() -> Result<PathBuf, Box<dyn Error>> {
    if cfg!(target_os = "macos") {
        let home = home_dir().ok_or("Unable to resolve home directory for data path.")?;
        return Ok(home
            .join("Library")
            .join("Application Support")
            .join(APP_NAME));
    }
    if cfg!(target_os = "windows") {
        let appdata = env::var_os("APPDATA")
            .map(PathBuf::from)
            .ok_or("APPDATA is not set.")?;
        return Ok(appdata.join(APP_NAME));
    }
    if let Some(xdg) = env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(xdg).join(APP_NAME));
    }
    let home = home_dir().ok_or("Unable to resolve home directory for data path.")?;
    Ok(home.join(".local").join("share").join(APP_NAME))
}

fn now_epoch() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    secs.to_string()
}

fn human_time(epoch: &str) -> String {
    if let Ok(secs) = epoch.parse::<i64>() {
        if let Some(dt) = DateTime::<Utc>::from_timestamp(secs, 0) {
            return dt.format("%B %-d, %Y").to_string();
        }
    }
    epoch.to_string()
}

fn configure_sqlite_connection(conn: &Connection) -> Result<(), Box<dyn Error>> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.busy_timeout(Duration::from_millis(SQLITE_BUSY_TIMEOUT_MS))?;
    Ok(())
}

fn init_db(db_path: &Path) -> Result<Connection, Box<dyn Error>> {
    let conn = Connection::open(db_path)?;
    configure_sqlite_connection(&conn)?;
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS libraries (
            library_name TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_refreshed_at TEXT NOT NULL,
            content_size_chars INTEGER NOT NULL DEFAULT 0,
            page_count INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            embedded_chunk_count INTEGER NOT NULL DEFAULT 0,
            empty_page_count INTEGER NOT NULL DEFAULT 0,
            min_chunks_per_page INTEGER NOT NULL DEFAULT 0,
            max_chunks_per_page INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS library_aliases (
            alias TEXT PRIMARY KEY,
            library_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS parents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            library_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_page_order INTEGER NOT NULL,
            parent_index_in_page INTEGER NOT NULL,
            global_parent_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER NOT NULL,
            library_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_page_order INTEGER NOT NULL,
            parent_index_in_page INTEGER NOT NULL,
            child_index_in_parent INTEGER NOT NULL,
            global_chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            library_name UNINDEXED,
            content
        );
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            library_name TEXT NOT NULL,
            page_order INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            content TEXT NOT NULL,
            content_size_chars INTEGER NOT NULL,
            crawled_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS library_groups (
            group_name TEXT NOT NULL,
            member_library_name TEXT NOT NULL,
            member_order INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (group_name, member_library_name)
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            library_name TEXT NOT NULL,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            message TEXT
        );
        CREATE TABLE IF NOT EXISTS library_texts (
            library_name TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_parents_library_name ON parents(library_name);
        CREATE INDEX IF NOT EXISTS idx_parents_library_page ON parents(library_name, source_url, parent_index_in_page);
        CREATE INDEX IF NOT EXISTS idx_chunks_library_name ON chunks(library_name);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_library_page ON chunks(library_name, source_url, parent_index_in_page, child_index_in_parent);
        CREATE INDEX IF NOT EXISTS idx_pages_library_name ON pages(library_name);
        CREATE INDEX IF NOT EXISTS idx_pages_library_order ON pages(library_name, page_order);
        CREATE INDEX IF NOT EXISTS idx_library_groups_name ON library_groups(group_name);
        ",
    )?;
    run_db_migrations(&conn)?;
    Ok(conn)
}

fn run_db_migrations(conn: &Connection) -> Result<(), Box<dyn Error>> {
    let mut has_page_size_bytes = false;
    let mut has_page_size_chars = false;
    let mut library_columns = HashSet::new();

    let mut stmt = conn.prepare("PRAGMA table_info(pages)")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        let col = row?;
        if col == "content_size_bytes" {
            has_page_size_bytes = true;
        }
        if col == "content_size_chars" {
            has_page_size_chars = true;
        }
    }
    if has_page_size_bytes && !has_page_size_chars {
        conn.execute_batch(
            "ALTER TABLE pages RENAME COLUMN content_size_bytes TO content_size_chars;",
        )?;
    }

    let mut stmt = conn.prepare("PRAGMA table_info(libraries)")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        library_columns.insert(row?);
    }
    if !library_columns.contains("content_size_chars") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN content_size_chars INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("page_count") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN page_count INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("chunk_count") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("embedded_chunk_count") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN embedded_chunk_count INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("empty_page_count") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN empty_page_count INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("min_chunks_per_page") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN min_chunks_per_page INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("max_chunks_per_page") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN max_chunks_per_page INTEGER NOT NULL DEFAULT 0;",
        )?;
    }

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS parents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            library_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_page_order INTEGER NOT NULL,
            parent_index_in_page INTEGER NOT NULL,
            global_parent_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER NOT NULL,
            library_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_page_order INTEGER NOT NULL,
            parent_index_in_page INTEGER NOT NULL,
            child_index_in_parent INTEGER NOT NULL,
            global_chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            library_name UNINDEXED,
            content
        );
        CREATE INDEX IF NOT EXISTS idx_parents_library_name ON parents(library_name);
        CREATE INDEX IF NOT EXISTS idx_parents_library_page ON parents(library_name, source_url, parent_index_in_page);
        CREATE INDEX IF NOT EXISTS idx_chunks_library_name ON chunks(library_name);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_library_page ON chunks(library_name, source_url, parent_index_in_page, child_index_in_parent);
        ",
    )?;

    Ok(())
}

fn start_job(conn: &Connection, library_name: &str, job_type: &str) -> Result<i64, Box<dyn Error>> {
    conn.execute(
        "INSERT INTO jobs (library_name, job_type, status, started_at) VALUES (?1, ?2, 'running', ?3)",
        params![library_name, job_type, now_epoch()],
    )?;
    Ok(conn.last_insert_rowid())
}

fn finish_job(
    conn: &Connection,
    job_id: i64,
    status: &str,
    message: &str,
) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "UPDATE jobs SET status = ?1, ended_at = ?2, message = ?3 WHERE id = ?4",
        params![status, now_epoch(), message, job_id],
    )?;
    Ok(())
}

fn parse_query_flags(flags: &[String]) -> Result<(SearchMode, usize, usize), Box<dyn Error>> {
    let mut mode = SearchMode::Hybrid;
    let mut top_k = DEFAULT_TOP_K;
    let mut context = DEFAULT_CONTEXT_WINDOW;
    let mut i = 0usize;
    while i < flags.len() {
        match flags[i].as_str() {
            "--mode" if i + 1 < flags.len() => {
                mode = SearchMode::from_str(&flags[i + 1]);
                i += 2;
            }
            "--top-k" if i + 1 < flags.len() => {
                top_k = flags[i + 1].parse()?;
                i += 2;
            }
            "--context" if i + 1 < flags.len() => {
                context = flags[i + 1].parse()?;
                i += 2;
            }
            _ => i += 1,
        }
    }
    Ok((mode, top_k, context))
}

fn extract_json_flag(flags: &[String]) -> (bool, Vec<String>) {
    let mut output_json = false;
    let mut out = Vec::with_capacity(flags.len());
    for flag in flags {
        if flag == "--json" {
            output_json = true;
        } else {
            out.push(flag.clone());
        }
    }
    (output_json, out)
}

fn split_query_and_flags(args: &[String]) -> (String, Vec<String>) {
    let first_flag = args
        .iter()
        .position(|arg| arg.starts_with("--"))
        .unwrap_or(args.len());
    let query = args[..first_flag].join(" ").trim().to_string();
    let flags = args[first_flag..].to_vec();
    (query, flags)
}

fn print_json(value: &Value) -> Result<(), Box<dyn Error>> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn print_command_result(
    command: &str,
    output_json: bool,
    payload: Value,
) -> Result<(), Box<dyn Error>> {
    if output_json {
        print_json(&json!({
            "command": command,
            "status": "success",
            "result": payload,
        }))
    } else {
        println!("Done.");
        Ok(())
    }
}

fn context_to_json(context: &[ParentRecord], active_parent_id: i64) -> Vec<Value> {
    context
        .iter()
        .filter(|parent| parent.id != active_parent_id)
        .map(|parent| {
            json!({
                "parent_id": parent.id,
                "library_name": parent.library_name,
                "source_url": parent.source_url,
                "source_page_order": parent.source_page_order,
                "parent_index_in_page": parent.parent_index_in_page,
                "global_parent_index": parent.global_parent_index,
                "content": parent.content,
            })
        })
        .collect()
}

fn query_hit_to_json(
    rank: usize,
    hit: &ScoredChunk,
    parent: &ParentRecord,
    context: &[ParentRecord],
) -> Value {
    json!({
        "rank": rank,
        "chunk_id": hit.chunk.id,
        "parent_id": hit.chunk.parent_id,
        "library_name": hit.chunk.library_name,
        "source_url": parent.source_url,
        "content": parent.content,
        "scores": {
            "final": hit.final_score,
            "vector": hit.vector_score,
            "bm25": hit.bm25_score,
        },
        "child_location": {
            "source_page_order": hit.chunk.source_page_order,
            "parent_index_in_page": hit.chunk.parent_index_in_page,
            "child_index_in_parent": hit.chunk.child_index_in_parent,
            "global_chunk_index": hit.chunk.global_chunk_index,
        },
        "parent_location": {
            "source_page_order": parent.source_page_order,
            "parent_index_in_page": parent.parent_index_in_page,
            "global_parent_index": parent.global_parent_index,
        },
        "context": context_to_json(context, parent.id),
    })
}

fn ask_flags(
    flags: &[String],
) -> Result<(SearchMode, usize, usize, Option<Vec<String>>), Box<dyn Error>> {
    let mut mode = SearchMode::Hybrid;
    let mut top_k = DEFAULT_TOP_K;
    let mut context = DEFAULT_CONTEXT_WINDOW;
    let mut libraries: Option<Vec<String>> = None;
    let mut i = 0usize;
    while i < flags.len() {
        match flags[i].as_str() {
            "--mode" if i + 1 < flags.len() => {
                mode = SearchMode::from_str(&flags[i + 1]);
                i += 2;
            }
            "--top-k" if i + 1 < flags.len() => {
                top_k = flags[i + 1].parse()?;
                i += 2;
            }
            "--context" if i + 1 < flags.len() => {
                context = flags[i + 1].parse()?;
                i += 2;
            }
            "--libraries" if i + 1 < flags.len() => {
                libraries = Some(
                    flags[i + 1]
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                );
                i += 2;
            }
            _ => i += 1,
        }
    }
    Ok((mode, top_k, context, libraries))
}

fn parse_index_file_flag(flags: &[String]) -> Option<String> {
    let mut i = 0usize;
    while i < flags.len() {
        if flags[i] == "--file" && i + 1 < flags.len() {
            return Some(flags[i + 1].clone());
        }
        i += 1;
    }
    None
}

fn parse_include_artifacts_flag(flags: &[String], library_name: &str) -> Option<PathBuf> {
    let mut artifacts: Option<PathBuf> = None;
    let mut i = 0usize;
    while i < flags.len() {
        if flags[i] == "--include-artifacts" {
            if i + 1 < flags.len() && !flags[i + 1].starts_with("--") {
                artifacts = Some(PathBuf::from(flags[i + 1].clone()));
                i += 2;
            } else {
                artifacts = Some(compiled_dir(library_name));
                i += 1;
            }
            continue;
        }
        if let Some(raw_path) = flags[i].strip_prefix("--include-artifacts=") {
            if raw_path.is_empty() {
                artifacts = Some(compiled_dir(library_name));
            } else {
                artifacts = Some(PathBuf::from(raw_path));
            }
            i += 1;
            continue;
        }
        i += 1;
    }
    artifacts
}

fn parse_merge_args(
    args: &[String],
    group_name: &str,
) -> Result<(Vec<String>, bool, Option<PathBuf>), Box<dyn Error>> {
    let mut members = Vec::new();
    let mut replace = false;
    let mut include_artifacts: Option<PathBuf> = None;
    let mut i = 0usize;
    while i < args.len() {
        let token = &args[i];
        if token == "--replace" {
            replace = true;
            i += 1;
            continue;
        }
        if token == "--include-artifacts" {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                include_artifacts = Some(PathBuf::from(args[i + 1].clone()));
                i += 2;
            } else {
                include_artifacts = Some(compiled_dir(group_name));
                i += 1;
            }
            continue;
        }
        if let Some(raw_path) = token.strip_prefix("--include-artifacts=") {
            include_artifacts = if raw_path.is_empty() {
                Some(compiled_dir(group_name))
            } else {
                Some(PathBuf::from(raw_path))
            };
            i += 1;
            continue;
        }
        members.push(token.clone());
        i += 1;
    }
    if members.len() < 2 {
        return Err("merge requires at least two source libraries.".into());
    }
    Ok((members, replace, include_artifacts))
}

fn write_artifacts(output_dir: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    fs::write(output_dir.join("docs.txt"), content)?;
    fs::write(output_dir.join("docs.md"), content)?;
    Ok(())
}

fn export_library(
    conn: &Connection,
    input_name: &str,
    output_dir: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let spinner = ProgressSpinner::new("Preparing export");
    let members = resolve_target_libraries(conn, input_name)?;
    let output_dir = output_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| compiled_dir(input_name));
    let mut compiled_parts = Vec::new();
    for member in &members {
        spinner.set_stage(format!("Collecting {}", member));
        compiled_parts.push(compiled_text_for_library(conn, member)?);
    }
    let mut compiled = compiled_parts.join("\n\n");
    if !compiled.is_empty() {
        compiled.push_str("\n\n");
    }

    spinner.set_stage("Writing export artifacts");
    write_artifacts(&output_dir, &compiled)?;
    spinner.finish();
    Ok(())
}

fn resolve_library_name(conn: &Connection, input: &str) -> Result<String, Box<dyn Error>> {
    if let Ok(name) = conn.query_row(
        "SELECT library_name FROM libraries WHERE library_name = ?1",
        params![input],
        |row| row.get::<_, String>(0),
    ) {
        return Ok(name);
    }
    let name = conn.query_row(
        "SELECT library_name FROM library_aliases WHERE alias = ?1",
        params![input],
        |row| row.get::<_, String>(0),
    )?;
    Ok(name)
}

fn group_members(conn: &Connection, group_name: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut stmt = conn.prepare(
        "SELECT member_library_name
         FROM library_groups
         WHERE group_name = ?1
         ORDER BY member_order ASC",
    )?;
    let rows = stmt.query_map(params![group_name], |row| row.get::<_, String>(0))?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn resolve_target_libraries(conn: &Connection, input: &str) -> Result<Vec<String>, Box<dyn Error>> {
    if let Ok(name) = resolve_library_name(conn, input) {
        return Ok(vec![name]);
    }
    let members = group_members(conn, input)?;
    if members.is_empty() {
        return Err(format!("Unknown library or merged group '{}'.", input).into());
    }
    Ok(members)
}

fn compiled_text_for_library(
    conn: &Connection,
    input_name: &str,
) -> Result<String, Box<dyn Error>> {
    let library_name = resolve_library_name(conn, input_name)?;
    let mut stmt = conn.prepare(
        "SELECT content
         FROM pages
         WHERE library_name = ?1
         ORDER BY page_order ASC",
    )?;
    let rows = stmt.query_map(params![library_name], |row| row.get::<_, String>(0))?;
    let mut page_contents = Vec::new();
    for row in rows {
        page_contents.push(row?);
    }
    if !page_contents.is_empty() {
        let mut text = page_contents.join("\n\n");
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        return Ok(text);
    }
    let fallback: Option<String> = conn
        .query_row(
            "SELECT content FROM library_texts WHERE library_name = ?1",
            params![library_name],
            |row| row.get(0),
        )
        .optional()?;
    fallback.ok_or_else(|| {
        format!(
            "No crawled content found for '{}'. Run crawl/add first.",
            input_name
        )
        .into()
    })
}

fn backfill_pages_from_parents(
    conn: &Connection,
    library_name: &str,
    now: &str,
) -> Result<(), Box<dyn Error>> {
    let page_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM pages WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    if page_count > 0 {
        return Ok(());
    }

    let parent_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM parents WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    if parent_count == 0 {
        return Ok(());
    }

    let mut page_stmt = conn.prepare(
        "SELECT DISTINCT source_page_order, source_url
         FROM parents
         WHERE library_name = ?1
         ORDER BY source_page_order ASC, source_url ASC",
    )?;
    let page_rows = page_stmt.query_map(params![library_name], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut page_pairs = Vec::new();
    for row in page_rows {
        page_pairs.push(row?);
    }

    let tx = conn.unchecked_transaction()?;
    for (page_order, source_url) in page_pairs {
        let mut chunk_stmt = tx.prepare(
            "SELECT content
             FROM parents
             WHERE library_name = ?1 AND source_page_order = ?2 AND source_url = ?3
             ORDER BY parent_index_in_page ASC",
        )?;
        let chunk_rows = chunk_stmt
            .query_map(params![library_name, page_order, source_url], |r| {
                r.get::<_, String>(0)
            })?;
        let mut parts = Vec::new();
        for c in chunk_rows {
            parts.push(c?);
        }
        let content = parts.join("\n\n");
        tx.execute(
            "INSERT INTO pages (
                library_name, page_order, source_url, content, content_size_chars, crawled_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                library_name,
                page_order,
                source_url,
                content,
                content.chars().count() as i64,
                now
            ],
        )?;
    }
    tx.commit()?;
    Ok(())
}

fn backfill_library_text(
    conn: &Connection,
    library_name: &str,
    now: &str,
) -> Result<(), Box<dyn Error>> {
    let exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM library_texts WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    if exists > 0 {
        return Ok(());
    }

    let mut stmt = conn.prepare(
        "SELECT content
         FROM pages
         WHERE library_name = ?1
         ORDER BY page_order ASC",
    )?;
    let rows = stmt.query_map(params![library_name], |row| row.get::<_, String>(0))?;
    let mut page_contents = Vec::new();
    for row in rows {
        page_contents.push(row?);
    }
    if page_contents.is_empty() {
        return Ok(());
    }

    let mut compiled = page_contents.join("\n\n");
    if !compiled.is_empty() {
        compiled.push_str("\n\n");
    }
    conn.execute(
        "INSERT OR REPLACE INTO library_texts (library_name, content, updated_at) VALUES (?1, ?2, ?3)",
        params![library_name, compiled, now],
    )?;
    Ok(())
}

fn compute_library_chars(conn: &Connection, library_name: &str) -> Result<i64, Box<dyn Error>> {
    let from_pages: i64 = conn.query_row(
        "SELECT COALESCE(SUM(content_size_chars), 0) FROM pages WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    if from_pages > 0 {
        return Ok(from_pages);
    }

    let from_text: Option<i64> = conn
        .query_row(
            "SELECT LENGTH(content) FROM library_texts WHERE library_name = ?1",
            params![library_name],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(v) = from_text {
        return Ok(v);
    }

    let from_parents: i64 = conn.query_row(
        "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM parents WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    Ok(from_parents)
}

fn compute_page_chunk_rollups(
    conn: &Connection,
    library_name: &str,
) -> Result<(i64, i64, i64, i64, i64, i64), Box<dyn Error>> {
    let page_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM pages WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    let chunk_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM chunks WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    let embedded_chunk_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM chunks WHERE library_name = ?1 AND LENGTH(embedding) > 0",
        params![library_name],
        |row| row.get(0),
    )?;

    if page_count == 0 {
        return Ok((0, chunk_count, embedded_chunk_count, 0, 0, 0));
    }

    let mut stmt = conn.prepare(
        "SELECT p.page_order, COUNT(c.id) AS chunk_count
         FROM pages p
         LEFT JOIN chunks c
           ON c.library_name = p.library_name
          AND c.source_page_order = p.page_order
         WHERE p.library_name = ?1
         GROUP BY p.page_order
         ORDER BY p.page_order ASC",
    )?;
    let rows = stmt.query_map(params![library_name], |row| row.get::<_, i64>(1))?;
    let mut min_chunks = i64::MAX;
    let mut max_chunks = i64::MIN;
    let mut empty_pages = 0i64;
    let mut seen = 0i64;
    for row in rows {
        let count = row?;
        if count == 0 {
            empty_pages += 1;
        }
        if count < min_chunks {
            min_chunks = count;
        }
        if count > max_chunks {
            max_chunks = count;
        }
        seen += 1;
    }
    if seen == 0 {
        min_chunks = 0;
        max_chunks = 0;
    }
    Ok((
        page_count,
        chunk_count,
        embedded_chunk_count,
        empty_pages,
        min_chunks,
        max_chunks,
    ))
}

fn update_library_rollups(conn: &Connection, library_name: &str) -> Result<(), Box<dyn Error>> {
    let content_size_chars = compute_library_chars(conn, library_name)?;
    let (
        page_count,
        chunk_count,
        embedded_chunk_count,
        empty_page_count,
        min_chunks_per_page,
        max_chunks_per_page,
    ) = compute_page_chunk_rollups(conn, library_name)?;
    conn.execute(
        "UPDATE libraries
         SET content_size_chars = ?1,
             page_count = ?2,
             chunk_count = ?3,
             embedded_chunk_count = ?4,
             empty_page_count = ?5,
             min_chunks_per_page = ?6,
             max_chunks_per_page = ?7,
             updated_at = ?8
         WHERE library_name = ?9",
        params![
            content_size_chars,
            page_count,
            chunk_count,
            embedded_chunk_count,
            empty_page_count,
            min_chunks_per_page,
            max_chunks_per_page,
            now_epoch(),
            library_name
        ],
    )?;
    Ok(())
}

fn merge_libraries(
    conn: &Connection,
    group_name: &str,
    member_inputs: &[String],
    replace: bool,
    include_artifacts: Option<&Path>,
) -> Result<(), Box<dyn Error>> {
    let spinner = ProgressSpinner::new("Preparing merge");
    if resolve_library_name(conn, group_name).is_ok() {
        return Err(format!("'{}' already exists as a library/alias.", group_name).into());
    }

    let mut resolved_members = Vec::new();
    let mut seen = HashSet::new();
    for input in member_inputs {
        spinner.set_stage(format!("Resolving {}", input));
        for name in resolve_target_libraries(conn, input)? {
            if seen.insert(name.clone()) {
                resolved_members.push(name);
            }
        }
    }
    if resolved_members.len() < 2 {
        return Err("Merged group must contain at least two distinct libraries.".into());
    }

    let exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM library_groups WHERE group_name = ?1",
        params![group_name],
        |row| row.get(0),
    )?;
    if exists > 0 && !replace {
        return Err(format!(
            "Merged group '{}' already exists. Use --replace to overwrite membership.",
            group_name
        )
        .into());
    }

    let tx = conn.unchecked_transaction()?;
    if exists > 0 {
        tx.execute(
            "DELETE FROM library_groups WHERE group_name = ?1",
            params![group_name],
        )?;
    }
    let now = now_epoch();
    for (idx, member) in resolved_members.iter().enumerate() {
        tx.execute(
            "INSERT INTO library_groups (group_name, member_library_name, member_order, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![group_name, member, idx as i64, now],
        )?;
    }
    tx.commit()?;

    if let Some(path) = include_artifacts {
        let mut compiled_parts = Vec::new();
        for member in &resolved_members {
            spinner.set_stage(format!("Compiling {}", member));
            compiled_parts.push(compiled_text_for_library(conn, member)?);
        }
        let mut compiled = compiled_parts.join("\n\n");
        if !compiled.is_empty() {
            compiled.push_str("\n\n");
        }
        spinner.set_stage("Writing merged artifacts");
        write_artifacts(path, &compiled)?;
    }

    spinner.finish();
    Ok(())
}

async fn add_library(
    conn: &Connection,
    library_name: &str,
    source_url: &str,
    include_artifacts: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let exists: i64 = conn.query_row(
        "SELECT COUNT(*) FROM libraries WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    if exists == 0 {
        crawl_library(
            conn,
            library_name,
            source_url,
            "add-crawl",
            include_artifacts.clone(),
        )
        .await?;
    } else {
        let page_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM pages WHERE library_name = ?1",
            params![library_name],
            |row| row.get(0),
        )?;
        if page_count == 0 {
            crawl_library(
                conn,
                library_name,
                source_url,
                "add-crawl",
                include_artifacts.clone(),
            )
            .await?;
        }
    }
    index_library(conn, library_name, None)?;
    Ok(())
}

fn refresh_stats(conn: &Connection, input_names: &[String]) -> Result<(), Box<dyn Error>> {
    let spinner = ProgressSpinner::new("Preparing refresh");
    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    if input_names.is_empty() {
        let mut stmt =
            conn.prepare("SELECT library_name FROM libraries ORDER BY library_name ASC")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        for row in rows {
            let name = row?;
            if seen.insert(name.clone()) {
                targets.push(name);
            }
        }
    } else {
        for input in input_names {
            for name in resolve_target_libraries(conn, input)? {
                if seen.insert(name.clone()) {
                    targets.push(name);
                }
            }
        }
    }

    if targets.is_empty() {
        return Err("No libraries found to refresh.".into());
    }

    let now = now_epoch();
    for library_name in &targets {
        spinner.set_stage(format!("Refreshing {}", library_name));
        backfill_pages_from_parents(conn, library_name, &now)?;
        spinner.set_stage(format!("Rebuilding text for {}", library_name));
        backfill_library_text(conn, library_name, &now)?;
        spinner.set_stage(format!("Updating stats for {}", library_name));
        update_library_rollups(conn, library_name)?;
        conn.execute(
            "UPDATE libraries SET last_refreshed_at = ?1 WHERE library_name = ?2",
            params![now, library_name],
        )?;
    }

    let job_id = start_job(conn, "_system", "refresh-stats")?;
    finish_job(
        conn,
        job_id,
        "success",
        &format!("Refreshed {} libraries.", targets.len()),
    )?;
    spinner.finish();
    Ok(())
}

// ---- Crawl pipeline: restored to your original behavior ----
fn normalize_seed_url(seed_url: &str) -> Result<String, String> {
    let mut parsed =
        Url::parse(seed_url).map_err(|e| format!("Invalid seed URL '{}': {}", seed_url, e))?;
    let path = parsed.path();
    if !path.ends_with('/') {
        let normalized_path = if path.is_empty() {
            "/".to_string()
        } else {
            format!("{}/", path)
        };
        parsed.set_path(&normalized_path);
    }
    Ok(parsed.to_string())
}

fn whitelist_for_url(seed_url: &str) -> Result<Vec<CompactString>, String> {
    let parsed =
        Url::parse(seed_url).map_err(|e| format!("Invalid seed URL '{}': {}", seed_url, e))?;
    let host = parsed
        .host_str()
        .ok_or_else(|| format!("Seed URL '{}' has no host", seed_url))?;
    let scheme_pattern = regex::escape(parsed.scheme());
    let authority = match parsed.port() {
        Some(port) => format!("{}:{}", host, port),
        None => host.to_string(),
    };
    let authority_pattern = regex::escape(&authority);
    let trimmed_path = parsed.path().trim_end_matches('/');
    let regex_pattern = if trimmed_path.is_empty() {
        format!(r"^{}://{}(/|$)", scheme_pattern, authority_pattern)
    } else {
        let path_pattern = regex::escape(trimmed_path);
        format!(
            r"^{}://{}{}(/|$)",
            scheme_pattern, authority_pattern, path_pattern
        )
    };
    Ok(vec![CompactString::new(regex_pattern)])
}

fn extract_content_html(html: &str) -> String {
    let document = Html::parse_document(html);
    let mut best_html: Option<String> = None;
    let mut best_text_len = 0usize;

    for selector in CONTENT_SELECTORS.iter() {
        for node in document.select(selector) {
            // Avoid allocating a full string only to estimate section size.
            let text_len: usize = node.text().map(|s| s.trim().len()).sum();
            if text_len > best_text_len {
                let selected_html = node.html();
                if !selected_html.trim().is_empty() {
                    best_text_len = text_len;
                    best_html = Some(selected_html);
                }
            }
        }
    }

    let mut cleaned = best_html.unwrap_or_else(|| html.to_string());
    let lower = cleaned.to_ascii_lowercase();
    if HTML_REGEX_HINTS.iter().any(|h| lower.contains(h)) {
        for re in HTML_CLEANUP_REGEXES.iter() {
            cleaned = re.replace_all(&cleaned, "").into_owned();
        }
    }
    cleaned
}

fn cleanup_markdown(markdown: &str) -> String {
    let mut cleaned = markdown.to_string();
    let lower = cleaned.to_ascii_lowercase();
    if MARKDOWN_HINTS.iter().any(|h| lower.contains(h)) {
        for re in MARKDOWN_LINE_REGEXES.iter() {
            cleaned = re.replace_all(&cleaned, "").into_owned();
        }
    }
    if cleaned.contains("\n\n\n") {
        cleaned = MULTI_NEWLINE_RE.replace_all(&cleaned, "\n\n").into_owned();
    }

    cleaned.trim().to_string()
}

async fn crawl_library(
    conn: &Connection,
    library_name: &str,
    source_url: &str,
    job_type: &str,
    include_artifacts: Option<PathBuf>,
) -> Result<(), Box<dyn Error>> {
    let job_id = start_job(conn, library_name, job_type)?;

    let run_result = async {
        let spinner = ProgressSpinner::new("Preparing crawl");
        let normalized_seed_url =
            normalize_seed_url(source_url).map_err(|e| format!("URL error: {e}"))?;
        let whitelist =
            whitelist_for_url(&normalized_seed_url).map_err(|e| format!("Whitelist error: {e}"))?;

        let mut config = Configuration::new();
        config
            .with_limit(5_000)
            .with_depth(25)
            .with_subdomains(false)
            .with_tld(false)
            .with_user_agent(Some("DocumentationScraper/1.0"))
            .with_whitelist_url(Some(whitelist));

        let mut website = Website::new(&normalized_seed_url)
            .with_config(config)
            .build()
            .map_err(|e| format!("Failed to build website: {e}"))?;
        spinner.set_stage("Downloading files");
        website.scrape().await;

        spinner.set_stage("Converting files");
        let pages = match website.get_pages() {
            Some(p) => p,
            None => {
                spinner.finish();
                return Err("No pages collected".into());
            }
        };

        spinner.set_stage("Writing files");
        let page_inputs: Vec<(String, String)> = pages
            .iter()
            .map(|p| (p.get_url().to_string(), p.get_html()))
            .collect();
        // Indexed parallel iterators preserve order on collect.
        let converted: Vec<(String, String)> = page_inputs
            .into_par_iter()
            .map(|(url, html)| {
                let extracted_html = extract_content_html(&html);
                let markdown = cleanup_markdown(&html2md::parse_html(&extracted_html));
                (url, markdown)
            })
            .collect();
        let mut compiled_parts = Vec::with_capacity(converted.len());
        let mut total_chars = 0i64;
        for (_, markdown) in &converted {
            total_chars += markdown.chars().count() as i64;
            compiled_parts.push(markdown.clone());
        }
        let mut compiled = compiled_parts.join("\n\n");
        if !compiled.is_empty() {
            compiled.push_str("\n\n");
        }

        let now = now_epoch();
        conn.execute(
            "INSERT OR REPLACE INTO library_texts (library_name, content, updated_at) VALUES (?1, ?2, ?3)",
            params![library_name, compiled, now],
        )?;

        if let Some(path) = include_artifacts.clone() {
            write_artifacts(&path, &compiled)?;
        }

        let tx = conn.unchecked_transaction()?;
        tx.execute("DELETE FROM pages WHERE library_name = ?1", params![library_name])?;
        for (page_order, (url, markdown)) in converted.iter().enumerate() {
            tx.execute(
                "INSERT INTO pages (
                    library_name, page_order, source_url, content, content_size_chars, crawled_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    library_name,
                    page_order as i64,
                    url,
                    markdown,
                    markdown.chars().count() as i64,
                    now
                ],
            )?;
        }
        tx.commit()?;

        spinner.set_stage("Finalizing");

        conn.execute(
            "INSERT OR REPLACE INTO libraries (
               library_name, source_url, created_at, updated_at, last_refreshed_at,
               content_size_chars, page_count, chunk_count, embedded_chunk_count,
               empty_page_count, min_chunks_per_page, max_chunks_per_page
             )
             VALUES (
               ?1, ?2,
               COALESCE((SELECT created_at FROM libraries WHERE library_name = ?1), ?3),
               ?3, ?3, ?4, ?5,
               COALESCE((SELECT chunk_count FROM libraries WHERE library_name = ?1), 0),
               COALESCE((SELECT embedded_chunk_count FROM libraries WHERE library_name = ?1), 0),
               COALESCE((SELECT empty_page_count FROM libraries WHERE library_name = ?1), 0),
               COALESCE((SELECT min_chunks_per_page FROM libraries WHERE library_name = ?1), 0),
               COALESCE((SELECT max_chunks_per_page FROM libraries WHERE library_name = ?1), 0)
             )",
            params![library_name, source_url, now, total_chars, converted.len() as i64],
        )?;

        spinner.finish();

        Ok::<String, Box<dyn Error>>(format!("Crawled {} pages.", pages.len()))
    }
    .await;

    match run_result {
        Ok(msg) => {
            finish_job(conn, job_id, "success", &msg)?;
            Ok(())
        }
        Err(err) => {
            let msg = format!("{err}");
            let _ = finish_job(conn, job_id, "failed", &msg);
            Err(err)
        }
    }
}

fn strip_front_matter(input: &str) -> &str {
    if !input.starts_with("---\n") {
        return input;
    }
    if let Some(end) = input[4..].find("\n---\n") {
        let idx = 4 + end + 5;
        return &input[idx..];
    }
    input
}

fn preprocess_for_chunking(input: &str) -> String {
    let normalized = input.replace("\r\n", "\n");
    let stripped = strip_front_matter(&normalized);
    let setext_h1 = Regex::new(r"(?m)^([^\n][^\n]*)\n=+\s*$").expect("valid regex");
    let setext_h2 = Regex::new(r"(?m)^([^\n][^\n]*)\n-+\s*$").expect("valid regex");
    let out = setext_h1.replace_all(stripped, "# $1");
    setext_h2.replace_all(&out, "## $1").into_owned()
}

const CHUNK_MIN_CHARS: usize = 1400;
const CHUNK_MAX_CHARS: usize = 3000;

fn split_by_paragraph_upper_bound(chunks: Vec<String>, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.chars().count() <= max_chars {
            out.push(chunk);
            continue;
        }

        // Split paragraphs only outside fenced code blocks.
        let mut paragraphs: Vec<String> = Vec::new();
        let mut current_para = String::new();
        let mut in_fence = false;
        for line in chunk.split_inclusive('\n') {
            let line_no_nl = line.trim_end_matches('\n').trim_end_matches('\r');
            let starts_fence = line_no_nl.trim_start().starts_with("```");
            current_para.push_str(line);
            if starts_fence {
                in_fence = !in_fence;
            }
            if !in_fence && line_no_nl.trim().is_empty() {
                if !current_para.trim().is_empty() {
                    paragraphs.push(std::mem::take(&mut current_para));
                } else {
                    current_para.clear();
                }
            }
        }
        if !current_para.trim().is_empty() {
            paragraphs.push(current_para);
        }

        let mut current = String::new();
        for para in paragraphs {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }
            if current.is_empty() {
                current.push_str(para);
                continue;
            }
            let candidate = format!("{current}\n\n{para}");
            if candidate.chars().count() <= max_chars {
                current = candidate;
            } else {
                out.push(current);
                current = para.to_string();
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

fn split_by_newline_upper_bound(chunks: Vec<String>, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.chars().count() <= max_chars {
            out.push(chunk);
            continue;
        }

        let mut current = String::new();
        let mut in_fence = false;
        for line in chunk.split_inclusive('\n') {
            let line_no_nl = line.trim_end_matches('\n').trim_end_matches('\r');
            let starts_fence = line_no_nl.trim_start().starts_with("```");

            if !in_fence {
                let candidate_len = current.chars().count() + line.chars().count();
                if !current.is_empty() && candidate_len > max_chars {
                    out.push(std::mem::take(&mut current));
                }
            }

            current.push_str(line);
            if starts_fence {
                in_fence = !in_fence;
            }
        }

        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

fn split_by_char_upper_bound(chunks: Vec<String>, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.chars().count() <= max_chars {
            out.push(chunk);
            continue;
        }

        let chars: Vec<char> = chunk.chars().collect();
        let mut start = 0usize;
        while start < chars.len() {
            let end = (start + max_chars).min(chars.len());
            let piece: String = chars[start..end].iter().collect();
            let trimmed = piece.trim().to_string();
            if !trimmed.is_empty() {
                out.push(trimmed);
            }
            start = end;
        }
    }
    out
}

fn split_markdown_by_headings(content: &str) -> Vec<String> {
    let processed = preprocess_for_chunking(content);
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_fence = false;
    for line in processed.split_inclusive('\n') {
        let line_no_nl = line.trim_end_matches('\n').trim_end_matches('\r');
        let starts_fence = line_no_nl.trim_start().starts_with("```");
        let is_heading = !in_fence && MD_ATX_HEADING_RE.is_match(line_no_nl);
        if is_heading && !current.trim().is_empty() {
            out.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        if starts_fence {
            in_fence = !in_fence;
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

fn chunk_markdown_page(content: &str) -> Vec<String> {
    let out = split_markdown_by_headings(content);

    // Final trim + dedupe pass.
    let mut cleaned = Vec::new();
    for chunk in out {
        let t = chunk.trim().to_string();
        if !t.is_empty() && cleaned.last() != Some(&t) {
            cleaned.push(t);
        }
    }
    cleaned = split_by_paragraph_upper_bound(cleaned, CHUNK_MAX_CHARS);
    cleaned = split_by_newline_upper_bound(cleaned, CHUNK_MAX_CHARS);
    cleaned = split_by_char_upper_bound(cleaned, CHUNK_MAX_CHARS);

    // Lower-bound only top-up: merge tiny chunks forward until min size.
    let mut topped = Vec::new();
    let mut pending: Option<String> = None;
    for chunk in cleaned {
        match pending.take() {
            None => {
                pending = Some(chunk);
            }
            Some(prev) => {
                if prev.chars().count() >= CHUNK_MIN_CHARS {
                    topped.push(prev);
                    pending = Some(chunk);
                } else {
                    pending = Some(format!("{prev}\n\n{chunk}"));
                }
            }
        }
    }
    if let Some(last) = pending {
        topped.push(last);
    }

    // Tail fix: if the final chunk is below min size, append it to previous.
    if topped.len() >= 2 {
        let last_len = topped.last().map(|s| s.chars().count()).unwrap_or(0);
        if last_len < CHUNK_MIN_CHARS {
            if let Some(last_chunk) = topped.pop() {
                if let Some(prev) = topped.last_mut() {
                    prev.push_str("\n\n");
                    prev.push_str(&last_chunk);
                }
            }
        }
    }

    topped
}

fn chunk_parent_into_children(parent_content: &str) -> Vec<String> {
    let trimmed = parent_content.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let total_len = chars.len();
    if total_len <= MAX_CHILD_LENGTH {
        return vec![trimmed.to_string()];
    }

    let target_children = total_len.div_ceil(MAX_CHILD_LENGTH).max(1);
    let mut children = Vec::with_capacity(target_children);
    let mut start = 0usize;

    for part_idx in 0..target_children {
        let remaining_parts = target_children - part_idx;
        let remaining_len = total_len - start;
        if remaining_parts == 1 {
            let chunk: String = chars[start..].iter().collect();
            let trimmed_chunk = chunk.trim().to_string();
            if !trimmed_chunk.is_empty() {
                children.push(trimmed_chunk);
            }
            break;
        }

        let ideal_end = start + remaining_len.div_ceil(remaining_parts);
        let search_start = ideal_end.saturating_sub(CHILD_SPLIT_WINDOW).max(start + MIN_CHILD_LENGTH);
        let search_end = (ideal_end + CHILD_SPLIT_WINDOW)
            .min(total_len)
            .min(start + MAX_CHILD_LENGTH);

        let mut best_split = None;
        let mut best_distance = usize::MAX;
        for idx in search_start..search_end {
            if chars[idx].is_whitespace() {
                let distance = idx.abs_diff(ideal_end);
                if distance < best_distance {
                    best_distance = distance;
                    best_split = Some(idx);
                }
            }
        }

        let end = best_split.unwrap_or_else(|| {
            ideal_end
                .max(start + MIN_CHILD_LENGTH)
                .min(start + MAX_CHILD_LENGTH)
                .min(total_len)
        });
        let chunk: String = chars[start..end].iter().collect();
        let trimmed_chunk = chunk.trim().to_string();
        if !trimmed_chunk.is_empty() {
            children.push(trimmed_chunk);
        }
        start = end;
        while start < total_len && chars[start].is_whitespace() {
            start += 1;
        }
    }

    children
}

fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut a_norm = 0.0f32;
    let mut b_norm = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        a_norm += x * x;
        b_norm += y * y;
    }
    if a_norm == 0.0 || b_norm == 0.0 {
        return 0.0;
    }
    dot / (a_norm.sqrt() * b_norm.sqrt())
}

fn resolve_or_create_library_for_index(
    conn: &Connection,
    input_name: &str,
    custom_file: Option<&str>,
) -> Result<(String, String), Box<dyn Error>> {
    let resolved_name = resolve_library_name(conn, input_name);
    let custom_file_source_url = if let Some(file_path) = custom_file {
        let canonical_path = PathBuf::from(file_path).canonicalize()?;
        Some(format!("file://{}", canonical_path.display()))
    } else {
        None
    };
    let (library_name, source_url) = match resolved_name {
        Ok(name) => {
            let source_url: String = conn.query_row(
                "SELECT source_url FROM libraries WHERE library_name = ?1",
                params![name],
                |row| row.get(0),
            )?;
            (name, source_url)
        }
        Err(_) => {
            let file_path = custom_file.ok_or_else(|| {
                format!(
                    "Library '{}' not found. Use add/crawl first, or pass --file.",
                    input_name
                )
            })?;
            let source_url = custom_file_source_url
                .clone()
                .unwrap_or_else(|| format!("file://{}", file_path));
            let now = now_epoch();
            conn.execute(
                "INSERT INTO libraries (library_name, source_url, created_at, updated_at, last_refreshed_at)
                 VALUES (?1, ?2, ?3, ?3, ?3)",
                params![input_name, source_url, now],
            )?;
            (input_name.to_string(), source_url)
        }
    };
    Ok((library_name, source_url))
}

fn load_page_inputs(
    conn: &Connection,
    library_name: &str,
    source_url: &str,
    custom_file: Option<&str>,
) -> Result<Vec<(i64, String, String)>, Box<dyn Error>> {
    let custom_file_source_url = if let Some(file_path) = custom_file {
        let canonical_path = PathBuf::from(file_path).canonicalize()?;
        Some(format!("file://{}", canonical_path.display()))
    } else {
        None
    };
    let page_inputs: Vec<(i64, String, String)> = if let Some(file_path) = custom_file {
        let source_text = fs::read_to_string(file_path)?;
        let page_url = custom_file_source_url
            .clone()
            .unwrap_or_else(|| source_url.to_string());
        vec![(0i64, page_url, source_text)]
    } else {
        let mut stmt = conn.prepare(
            "SELECT page_order, source_url, content
                 FROM pages
                 WHERE library_name = ?1
                 ORDER BY page_order ASC",
        )?;
        let rows = stmt.query_map(params![library_name], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        if !out.is_empty() {
            out
        } else {
            let from_db: Option<String> = conn
                .query_row(
                    "SELECT content FROM library_texts WHERE library_name = ?1",
                    params![library_name],
                    |row| row.get(0),
                )
                .optional()?;
            if let Some(content) = from_db {
                vec![(0i64, source_url.to_string(), content)]
            } else {
                let out_dir = compiled_dir(&library_name);
                let txt = out_dir.join("docs.txt");
                let md = out_dir.join("docs.md");
                if txt.exists() {
                    vec![(0i64, source_url.to_string(), fs::read_to_string(txt)?)]
                } else if md.exists() {
                    vec![(0i64, source_url.to_string(), fs::read_to_string(md)?)]
                } else {
                    return Err(format!(
                        "No crawled text found for '{}'. Run add first.",
                        library_name
                    )
                    .into());
                }
            }
        }
    };

    Ok(page_inputs)
}

fn chunk_library(
    conn: &Connection,
    input_name: &str,
    custom_file: Option<&str>,
    job_type: &str,
) -> Result<(), Box<dyn Error>> {
    let (library_name, source_url) =
        resolve_or_create_library_for_index(conn, input_name, custom_file)?;
    let job_id = start_job(conn, &library_name, job_type)?;
    let spinner = ProgressSpinner::new(format!("Preparing chunks for {}", library_name));
    let result = (|| -> Result<String, Box<dyn Error>> {
        spinner.set_stage(format!("Loading pages for {}", library_name));
        let page_inputs = load_page_inputs(conn, &library_name, &source_url, custom_file)?;
        if page_inputs.is_empty() {
            return Err("No pages available for chunking.".into());
        }

        if custom_file.is_some() {
            spinner.set_stage(format!("Writing pages for {}", library_name));
            let tx = conn.unchecked_transaction()?;
            tx.execute(
                "DELETE FROM pages WHERE library_name = ?1",
                params![library_name],
            )?;
            let now = now_epoch();
            for (page_order, page_url, content) in &page_inputs {
                tx.execute(
                    "INSERT INTO pages (
                        library_name, page_order, source_url, content, content_size_chars, crawled_at
                    ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![
                        library_name,
                        page_order,
                        page_url,
                        content,
                        content.chars().count() as i64,
                        now
                    ],
                )?;
            }
            tx.commit()?;
        }

        let mut parent_rows: Vec<(i64, String, i64, String)> = Vec::new();
        let mut chunk_rows: Vec<(i64, i64, String, i64, i64, String)> = Vec::new();
        let mut per_page_chunk_counts: Vec<i64> = Vec::with_capacity(page_inputs.len());
        let mut global_parent_index = 0i64;
        for (page_order, page_url, page_content) in &page_inputs {
            spinner.set_stage(format!("Chunking page {}", page_order + 1));
            let page_parents = chunk_markdown_page(page_content);
            let mut child_count_for_page = 0i64;
            for (parent_index_in_page, parent) in page_parents.into_iter().enumerate() {
                let parent_index_in_page = parent_index_in_page as i64;
                let children = chunk_parent_into_children(&parent);
                child_count_for_page += children.len() as i64;
                parent_rows.push((
                    *page_order,
                    page_url.clone(),
                    parent_index_in_page,
                    parent.clone(),
                ));
                for (child_index_in_parent, child) in children.into_iter().enumerate() {
                    chunk_rows.push((
                        global_parent_index,
                        *page_order,
                        page_url.clone(),
                        parent_index_in_page,
                        child_index_in_parent as i64,
                        child,
                    ));
                }
                global_parent_index += 1;
            }
            per_page_chunk_counts.push(child_count_for_page);
        }

        if parent_rows.is_empty() {
            return Err("No parent chunks generated from input.".into());
        }
        if chunk_rows.is_empty() {
            return Err("No child chunks generated from input.".into());
        }

        spinner.set_stage(format!("Saving chunks for {}", library_name));
        let tx = conn.unchecked_transaction()?;
        tx.execute(
            "DELETE FROM parents WHERE library_name = ?1",
            params![library_name],
        )?;
        tx.execute(
            "DELETE FROM chunks WHERE library_name = ?1",
            params![library_name],
        )?;

        let now = now_epoch();
        let mut parent_ids = Vec::with_capacity(parent_rows.len());
        for (i, (source_page_order, parent_source_url, parent_index_in_page, parent)) in
            parent_rows.iter().enumerate()
        {
            let token_count = parent.chars().count() as i64;
            tx.execute(
                "INSERT INTO parents (
                    library_name, source_url, source_page_order, parent_index_in_page,
                    global_parent_index, content, token_count, created_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    library_name,
                    parent_source_url,
                    source_page_order,
                    parent_index_in_page,
                    i as i64,
                    parent,
                    token_count,
                    now
                ],
            )?;
            parent_ids.push(tx.last_insert_rowid());
        }

        for (i, (parent_row_index, source_page_order, chunk_source_url, parent_index_in_page, child_index_in_parent, chunk)) in
            chunk_rows.iter().enumerate()
        {
            let token_count = chunk.chars().count() as i64;
            let parent_id = parent_ids[*parent_row_index as usize];
            tx.execute(
                "INSERT INTO chunks (
                    parent_id, library_name, source_url, source_page_order, parent_index_in_page,
                    child_index_in_parent, global_chunk_index, content, embedding, token_count, created_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                params![
                    parent_id,
                    library_name,
                    chunk_source_url,
                    source_page_order,
                    parent_index_in_page,
                    child_index_in_parent,
                    i as i64,
                    chunk,
                    Vec::<u8>::new(),
                    token_count,
                    now
                ],
            )?;
        }
        tx.commit()?;

        let page_count = page_inputs.len() as i64;
        let total_chars: i64 = page_inputs
            .iter()
            .map(|(_, _, content)| content.chars().count() as i64)
            .sum();
        let chunk_count = chunk_rows.len() as i64;
        let empty_page_count = per_page_chunk_counts.iter().filter(|c| **c == 0).count() as i64;
        let min_chunks_per_page = per_page_chunk_counts.iter().copied().min().unwrap_or(0);
        let max_chunks_per_page = per_page_chunk_counts.iter().copied().max().unwrap_or(0);
        conn.execute(
            "UPDATE libraries
             SET content_size_chars = ?1,
                 page_count = ?2,
                 chunk_count = ?3,
                 embedded_chunk_count = 0,
                 empty_page_count = ?4,
                 min_chunks_per_page = ?5,
                 max_chunks_per_page = ?6,
                 updated_at = ?7
             WHERE library_name = ?8",
            params![
                total_chars,
                page_count,
                chunk_count,
                empty_page_count,
                min_chunks_per_page,
                max_chunks_per_page,
                now_epoch(),
                library_name
            ],
        )?;
        spinner.set_stage(format!("Building BM25 index for {}", library_name));
        rebuild_bm25_index_for_library(conn, &library_name)?;
        spinner.finish();
        Ok(format!(
            "Chunked {} parents into {} child chunks.",
            parent_rows.len(),
            chunk_rows.len()
        ))
    })();

    match result {
        Ok(msg) => {
            finish_job(conn, job_id, "success", &msg)?;
            Ok(())
        }
        Err(err) => {
            let msg = format!("{err}");
            let _ = finish_job(conn, job_id, "failed", &msg);
            Err(err)
        }
    }
}

fn chunk_targets(
    conn: &Connection,
    input_name: &str,
    custom_file: Option<&str>,
    job_type: &str,
) -> Result<(), Box<dyn Error>> {
    let targets = match resolve_target_libraries(conn, input_name) {
        Ok(t) => t,
        Err(_) if custom_file.is_some() => vec![input_name.to_string()],
        Err(_) => return Err(format!("Unknown library or merged group '{}'.", input_name).into()),
    };
    if targets.len() > 1 && custom_file.is_some() {
        return Err("Cannot use --file when target is a merged group.".into());
    }
    for target in targets {
        chunk_library(conn, &target, custom_file, job_type)?;
    }
    Ok(())
}

fn embed_library(
    conn: &Connection,
    input_name: &str,
    job_type: &str,
) -> Result<(), Box<dyn Error>> {
    let library_name = resolve_library_name(conn, input_name)?;
    let (chunk_count, _) = embedding_readiness(conn, &library_name)?;
    if chunk_count == 0 {
        return Err(format!(
            "No chunks found for '{}'. Run add or chunk first.",
            library_name
        )
        .into());
    }

    let job_id = start_job(conn, &library_name, job_type)?;
    let spinner = ProgressSpinner::new(format!("Preparing embeddings for {}", library_name));
    let result = (|| -> Result<String, Box<dyn Error>> {
        spinner.set_stage(format!("Loading chunks for {}", library_name));
        let mut stmt = conn.prepare(
            "SELECT id, content FROM chunks WHERE library_name = ?1 AND LENGTH(embedding) = 0 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(params![library_name], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut pending = Vec::new();
        for row in rows {
            pending.push(row?);
        }
        if pending.is_empty() {
            return Ok("All chunks already embedded.".to_string());
        }

        let mut model = TextEmbedding::try_new(
            InitOptions::new(DEFAULT_EMBEDDING_MODEL).with_show_download_progress(true),
        )?;
        let batch_size = DEFAULT_EMBED_BATCH_SIZE;
        let tx = conn.unchecked_transaction()?;
        let total_batches = pending.len().div_ceil(batch_size);
        for (batch_idx, batch) in pending.chunks(batch_size).enumerate() {
            spinner.set_stage(format!(
                "Embedding batch {} of {} for {}",
                batch_idx + 1,
                total_batches,
                library_name
            ));
            let texts: Vec<String> = batch
                .iter()
                .map(|(_, content)| content.clone())
                .collect();
            let embeds = model.embed(&texts, Some(batch_size))?;
            if embeds.len() != batch.len() {
                return Err("Embedding count mismatch.".into());
            }
            for (idx, (chunk_id, _)) in batch.iter().enumerate() {
                tx.execute(
                    "UPDATE chunks SET embedding = ?1 WHERE id = ?2",
                    params![embedding_to_bytes(&embeds[idx]), chunk_id],
                )?;
            }
        }
        tx.commit()?;
        spinner.set_stage(format!("Updating stats for {}", library_name));
        update_library_rollups(conn, &library_name)?;
        spinner.finish();
        Ok(format!("Embedded {} chunks.", pending.len()))
    })();

    match result {
        Ok(msg) => {
            finish_job(conn, job_id, "success", &msg)?;
            Ok(())
        }
        Err(err) => {
            let msg = format!("{err}");
            let _ = finish_job(conn, job_id, "failed", &msg);
            Err(err)
        }
    }
}

fn index_library(
    conn: &Connection,
    input_name: &str,
    custom_file: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let targets = match resolve_target_libraries(conn, input_name) {
        Ok(t) => t,
        Err(_) if custom_file.is_some() => vec![input_name.to_string()],
        Err(_) => return Err(format!("Unknown library or merged group '{}'.", input_name).into()),
    };
    if targets.len() > 1 && custom_file.is_some() {
        return Err("Cannot use --file when target is a merged group.".into());
    }

    for target_name in targets {
        let (total, _embedded) = embedding_readiness(conn, &target_name).unwrap_or((0, 0));
        if custom_file.is_some() || total == 0 {
            chunk_library(conn, &target_name, custom_file, "index-chunk")?;
        }
        let (total_after_chunk, embedded_after_chunk) =
            embedding_readiness(conn, &target_name).unwrap_or((0, 0));
        if total_after_chunk > 0 && embedded_after_chunk < total_after_chunk {
            embed_library(conn, &target_name, "index-embed")?;
        }
    }
    Ok(())
}

fn load_chunks_for_library(
    conn: &Connection,
    library_name: &str,
) -> Result<Vec<ChunkRecord>, Box<dyn Error>> {
    let mut stmt = conn.prepare(
        "SELECT id, parent_id, library_name, source_page_order, parent_index_in_page,
                child_index_in_parent, global_chunk_index, embedding
         FROM chunks
         WHERE library_name = ?1",
    )?;
    let rows = stmt.query_map(params![library_name], |row| {
        let bytes: Vec<u8> = row.get(7)?;
        Ok(ChunkRecord {
            id: row.get(0)?,
            parent_id: row.get(1)?,
            library_name: row.get(2)?,
            source_page_order: row.get(3)?,
            parent_index_in_page: row.get(4)?,
            child_index_in_parent: row.get(5)?,
            global_chunk_index: row.get(6)?,
            embedding: bytes_to_embedding(&bytes),
        })
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn load_parent_by_id(conn: &Connection, parent_id: i64) -> Result<ParentRecord, Box<dyn Error>> {
    let parent = conn.query_row(
        "SELECT id, library_name, source_url, source_page_order, parent_index_in_page, global_parent_index, content
         FROM parents
         WHERE id = ?1",
        params![parent_id],
        |row| {
            Ok(ParentRecord {
                id: row.get(0)?,
                library_name: row.get(1)?,
                source_url: row.get(2)?,
                source_page_order: row.get(3)?,
                parent_index_in_page: row.get(4)?,
                global_parent_index: row.get(5)?,
                content: row.get(6)?,
            })
        },
    )?;
    Ok(parent)
}

fn score_chunks(
    chunks: &[ChunkRecord],
    mode: SearchMode,
    query_embedding: Option<&[f32]>,
    bm25_scores: &HashMap<i64, f32>,
    use_vector_scores: bool,
) -> Vec<ScoredChunk> {
    let vector_scores_raw = if use_vector_scores {
        chunks
            .iter()
            .map(|chunk| {
                let score = match (mode, query_embedding) {
                    (SearchMode::Keyword, _) => 0.0,
                    (_, Some(embed)) if !chunk.embedding.is_empty() => {
                        cosine_similarity(embed, &chunk.embedding)
                    }
                    _ => 0.0,
                };
                (chunk.id, score)
            })
            .collect::<HashMap<_, _>>()
    } else {
        HashMap::new()
    };
    let normalized_vector_scores = normalized_scores(&vector_scores_raw);
    let normalized_bm25_scores = normalized_scores(bm25_scores);
    let mut scored = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        let vector_score = normalized_vector_scores
            .get(&chunk.id)
            .copied()
            .unwrap_or(0.0);
        let bm25_score = normalized_bm25_scores.get(&chunk.id).copied().unwrap_or(0.0);
        let final_score = match mode {
            SearchMode::Vector => vector_score,
            SearchMode::Keyword => bm25_score,
            SearchMode::Hybrid => {
                if use_vector_scores {
                    0.90 * vector_score + 0.10 * bm25_score
                } else {
                    bm25_score
                }
            }
        };
        scored.push(ScoredChunk {
            chunk: chunk.clone(),
            vector_score,
            bm25_score,
            final_score,
        });
    }
    scored.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(CmpOrdering::Equal)
    });
    scored
}

fn parent_neighbors(
    conn: &Connection,
    library_name: &str,
    source_url: &str,
    parent_index_in_page: i64,
    context: usize,
) -> Result<Vec<ParentRecord>, Box<dyn Error>> {
    if context == 0 {
        return Ok(Vec::new());
    }
    let low = parent_index_in_page - context as i64;
    let high = parent_index_in_page + context as i64;
    let mut stmt = conn.prepare(
        "SELECT id, library_name, source_url, source_page_order, parent_index_in_page,
                global_parent_index, content
         FROM parents
         WHERE library_name = ?1 AND source_url = ?2 AND parent_index_in_page BETWEEN ?3 AND ?4
         ORDER BY parent_index_in_page ASC",
    )?;
    let rows = stmt.query_map(params![library_name, source_url, low, high], |row| {
        Ok(ParentRecord {
            id: row.get(0)?,
            library_name: row.get(1)?,
            source_url: row.get(2)?,
            source_page_order: row.get(3)?,
            parent_index_in_page: row.get(4)?,
            global_parent_index: row.get(5)?,
            content: row.get(6)?,
        })
    })?;
    let mut out = Vec::new();
    for row in rows {
        out.push(row?);
    }
    Ok(out)
}

fn embed_query(mode: SearchMode, question: &str) -> Result<Option<Vec<f32>>, Box<dyn Error>> {
    if let SearchMode::Keyword = mode {
        return Ok(None);
    }
    let mut model = TextEmbedding::try_new(
        InitOptions::new(DEFAULT_EMBEDDING_MODEL).with_show_download_progress(false),
    )?;
    let embedding = model.embed([question], None)?;
    Ok(embedding.first().cloned())
}

fn embedding_readiness(
    conn: &Connection,
    library_name: &str,
) -> Result<(i64, i64), Box<dyn Error>> {
    let (total, embedded): (i64, i64) = conn.query_row(
        "SELECT COALESCE(chunk_count, 0), COALESCE(embedded_chunk_count, 0)
         FROM libraries
         WHERE library_name = ?1",
        params![library_name],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    Ok((total, embedded))
}

fn bm25_readiness(conn: &Connection, library_name: &str) -> Result<i64, Box<dyn Error>> {
    let count = conn.query_row(
        "SELECT COUNT(*) FROM chunks_fts WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    Ok(count)
}

fn rebuild_bm25_index_for_library(
    conn: &Connection,
    library_name: &str,
) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "DELETE FROM chunks_fts WHERE library_name = ?1",
        params![library_name],
    )?;
    let mut stmt = conn.prepare(
        "SELECT id, content
         FROM chunks
         WHERE library_name = ?1
         ORDER BY id ASC",
    )?;
    let rows = stmt.query_map(params![library_name], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (chunk_id, content) = row?;
        conn.execute(
            "INSERT INTO chunks_fts(rowid, library_name, content) VALUES (?1, ?2, ?3)",
            params![chunk_id, library_name, content],
        )?;
    }
    Ok(())
}

fn tokenize_fts_query(question: &str) -> String {
    let mut seen = HashSet::new();
    let terms = question
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map(str::trim)
        .filter(|term| !term.is_empty())
        .map(|term| term.to_ascii_lowercase())
        .filter(|term| seen.insert(term.clone()))
        .map(|term| format!("\"{term}\""))
        .collect::<Vec<_>>();
    terms.join(" OR ")
}

fn bm25_scores_for_library(
    conn: &Connection,
    library_name: &str,
    question: &str,
    limit: usize,
) -> Result<HashMap<i64, f32>, Box<dyn Error>> {
    let fts_query = tokenize_fts_query(question);
    if fts_query.is_empty() || limit == 0 {
        return Ok(HashMap::new());
    }

    let mut stmt = conn.prepare(
        "SELECT rowid, -bm25(chunks_fts) AS score
         FROM chunks_fts
         WHERE chunks_fts MATCH ?1 AND library_name = ?2
         ORDER BY bm25(chunks_fts)
         LIMIT ?3",
    )?;
    let rows = stmt.query_map(params![fts_query, library_name, limit as i64], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, f32>(1)?))
    })?;
    let mut scores = HashMap::new();
    for row in rows {
        let (chunk_id, score) = row?;
        scores.insert(chunk_id, score);
    }
    Ok(scores)
}

fn normalized_scores(scores: &HashMap<i64, f32>) -> HashMap<i64, f32> {
    if scores.is_empty() {
        return HashMap::new();
    }
    let min = scores.values().copied().fold(f32::INFINITY, f32::min);
    let max = scores
        .values()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() < f32::EPSILON {
        return scores
            .keys()
            .copied()
            .map(|id| (id, 1.0))
            .collect::<HashMap<_, _>>();
    }
    scores
        .iter()
        .map(|(id, score)| (*id, (*score - min) / (max - min)))
        .collect()
}

fn query_library(
    conn: &Connection,
    input_name: &str,
    question: &str,
    mode: SearchMode,
    top_k: usize,
    context: usize,
    trace: bool,
    output_json: bool,
) -> Result<(), Box<dyn Error>> {
    let spinner = ProgressSpinner::new(format!("Preparing query for {}", input_name));
    let target_libraries = resolve_target_libraries(conn, input_name)?;
    let mut all_chunks = Vec::new();
    let mut all_bm25_scores = HashMap::new();
    let mut use_vector_scores = mode != SearchMode::Keyword;
    for library_name in &target_libraries {
        spinner.set_stage(format!("Checking readiness for {}", library_name));
        let (total, embedded) = embedding_readiness(conn, library_name)?;
        let bm25_count = bm25_readiness(conn, library_name)?;
        if total == 0 || bm25_count == 0 {
            spinner.finish();
            println!(
                "Library '{}' is not indexed yet. Run `plshelp chunk {}` (or `plshelp add {}`) first.",
                library_name, library_name, library_name
            );
            return Ok(());
        }
        if matches!(mode, SearchMode::Vector) && embedded < total {
            spinner.finish();
            println!(
                "Library '{}' has partial embeddings ({}/{}). Run `plshelp embed {}`.",
                library_name, embedded, total, library_name
            );
            return Ok(());
        }
        spinner.set_stage(format!("Loading chunks for {}", library_name));
        all_chunks.extend(load_chunks_for_library(conn, library_name)?);
        spinner.set_stage(format!("Loading BM25 scores for {}", library_name));
        all_bm25_scores.extend(bm25_scores_for_library(
            conn,
            library_name,
            question,
            top_k.saturating_mul(10).max(50),
        )?);
        if embedded < total {
            use_vector_scores = false;
        }
    }
    if all_chunks.is_empty() {
        spinner.finish();
        println!("No chunks indexed for '{}'.", input_name);
        return Ok(());
    }
    let effective_mode = if matches!(mode, SearchMode::Hybrid) && !use_vector_scores {
        SearchMode::Keyword
    } else {
        mode
    };
    let query_embedding = if use_vector_scores {
        spinner.set_stage("Embedding query");
        embed_query(effective_mode, question)?
    } else {
        None
    };
    spinner.set_stage("Ranking results");
    let scored = score_chunks(
        &all_chunks,
        effective_mode,
        query_embedding.as_deref(),
        &all_bm25_scores,
        use_vector_scores,
    );
    let mut top_hits = Vec::new();
    let mut seen_parents = HashSet::new();
    for hit in scored {
        if hit.final_score <= 0.0 {
            continue;
        }
        if seen_parents.insert(hit.chunk.parent_id) {
            top_hits.push(hit);
        }
        if top_hits.len() >= top_k {
            break;
        }
    }

    spinner.finish();

    if top_hits.is_empty() {
        if output_json {
            print_json(&json!({
                "command": if trace { "trace" } else { "query" },
                "input_name": input_name,
                "question": question,
                "mode": mode.as_str(),
                "effective_mode": effective_mode.as_str(),
                "top_k": top_k,
                "context_window": context,
                "libraries": target_libraries,
                "results": [],
            }))?;
        } else {
            println!("No results for '{}'.", question);
        }
        return Ok(());
    }

    let mut json_results = Vec::new();
    for (rank, hit) in top_hits.iter().enumerate() {
        let parent = load_parent_by_id(conn, hit.chunk.parent_id)?;
        let around = if context > 0 {
            parent_neighbors(
                conn,
                &parent.library_name,
                &parent.source_url,
                parent.parent_index_in_page,
                context,
            )?
        } else {
            Vec::new()
        };
        if output_json {
            json_results.push(query_hit_to_json(rank + 1, hit, &parent, &around));
            continue;
        }
        println!("{}. [{}]", rank + 1, hit.chunk.id);
        println!("source: {}", parent.source_url);
        if target_libraries.len() > 1 {
            println!("   library: {}", hit.chunk.library_name);
        }
        if trace {
            println!(
                "   scores: final={:.4} vector={:.4} bm25={:.4}",
                hit.final_score, hit.vector_score, hit.bm25_score
            );
            println!(
                "   child location: page_order={} parent_in_page={} child_in_parent={} global_child_index={}",
                hit.chunk.source_page_order,
                hit.chunk.parent_index_in_page,
                hit.chunk.child_index_in_parent,
                hit.chunk.global_chunk_index
            );
            println!(
                "   parent location: parent_id={} page_order={} global_parent_index={}",
                parent.id, parent.source_page_order, parent.global_parent_index
            );
            println!("   library: {}", hit.chunk.library_name);
        }
        println!("{}", parent.content);
        if context > 0 {
            if !around.is_empty() {
                println!("--- context ---");
                for c in around {
                    if c.id != parent.id {
                        println!("[{}] {}", c.id, c.content);
                    }
                }
            }
        }
        println!();
    }
    if output_json {
        print_json(&json!({
            "command": if trace { "trace" } else { "query" },
            "input_name": input_name,
            "question": question,
            "mode": mode.as_str(),
            "effective_mode": effective_mode.as_str(),
            "top_k": top_k,
            "context_window": context,
            "libraries": target_libraries,
            "results": json_results,
        }))?;
    }
    Ok(())
}

fn ask_libraries(
    conn: &Connection,
    question: &str,
    flags: &[String],
    output_json: bool,
) -> Result<(), Box<dyn Error>> {
    let spinner = ProgressSpinner::new("Preparing multi-library query");
    let (mode, top_k, context, filter) = ask_flags(flags)?;
    let libraries = if let Some(libs) = filter {
        let mut out = Vec::new();
        let mut seen = HashSet::new();
        for lib in libs {
            for expanded in resolve_target_libraries(conn, &lib)? {
                if seen.insert(expanded.clone()) {
                    out.push(expanded);
                }
            }
        }
        out
    } else {
        let mut stmt =
            conn.prepare("SELECT library_name FROM libraries ORDER BY library_name ASC")?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        out
    };
    if libraries.is_empty() {
        spinner.finish();
        println!("No libraries indexed.");
        return Ok(());
    }

    let query_embedding = if mode != SearchMode::Keyword {
        spinner.set_stage("Embedding query");
        embed_query(mode, question)?
    } else {
        None
    };
    let mut combined = Vec::new();
    for lib in libraries {
        spinner.set_stage(format!("Scoring {}", lib));
        let (total, embedded) = embedding_readiness(conn, &lib)?;
        let bm25_count = bm25_readiness(conn, &lib)?;
        if total == 0 || bm25_count == 0 {
            continue;
        }
        let chunks = load_chunks_for_library(conn, &lib)?;
        if chunks.is_empty() {
            continue;
        }
        let bm25_scores =
            bm25_scores_for_library(conn, &lib, question, top_k.saturating_mul(10).max(50))?;
        let library_use_vector = mode != SearchMode::Keyword && embedded == total;
        let effective_mode = if matches!(mode, SearchMode::Hybrid) && !library_use_vector {
            SearchMode::Keyword
        } else {
            mode
        };
        combined.extend(score_chunks(
            &chunks,
            effective_mode,
            query_embedding.as_deref(),
            &bm25_scores,
            library_use_vector,
        ));
    }
    combined.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(CmpOrdering::Equal)
    });
    let mut top_hits = Vec::new();
    let mut seen_parents = HashSet::new();
    for hit in combined {
        if hit.final_score <= 0.0 {
            continue;
        }
        if seen_parents.insert(hit.chunk.parent_id) {
            top_hits.push(hit);
        }
        if top_hits.len() >= top_k {
            break;
        }
    }

    spinner.finish();

    if top_hits.is_empty() {
        if output_json {
            print_json(&json!({
                "command": "ask",
                "question": question,
                "mode": mode.as_str(),
                "top_k": top_k,
                "context_window": context,
                "libraries": [],
                "results": [],
            }))?;
        } else {
            println!("No results for '{}'.", question);
        }
        return Ok(());
    }

    let mut json_results = Vec::new();
    for (rank, hit) in top_hits.iter().enumerate() {
        let parent = load_parent_by_id(conn, hit.chunk.parent_id)?;
        let around = if context > 0 {
            parent_neighbors(
                conn,
                &parent.library_name,
                &parent.source_url,
                parent.parent_index_in_page,
                context,
            )?
        } else {
            Vec::new()
        };
        if output_json {
            json_results.push(query_hit_to_json(rank + 1, hit, &parent, &around));
            continue;
        }
        println!("{}. [{}] ({})", rank + 1, hit.chunk.id, hit.chunk.library_name);
        println!("source: {}", parent.source_url);
        println!("{}", parent.content);
        if context > 0 {
            if !around.is_empty() {
                println!("--- context ---");
                for c in around {
                    if c.id != parent.id {
                        println!("[{}] {}", c.id, c.content);
                    }
                }
            }
        }
        println!();
    }
    if output_json {
        print_json(&json!({
            "command": "ask",
            "question": question,
            "mode": mode.as_str(),
            "top_k": top_k,
            "context_window": context,
            "results": json_results,
        }))?;
    }
    Ok(())
}

fn add_alias(conn: &Connection, input_name: &str, alias: &str) -> Result<(), Box<dyn Error>> {
    let library_name = resolve_library_name(conn, input_name)?;
    let collision: i64 = conn.query_row(
        "SELECT COUNT(*) FROM libraries WHERE library_name = ?1",
        params![alias],
        |row| row.get(0),
    )?;
    if collision > 0 {
        return Err(format!("Alias '{}' conflicts with an existing library name.", alias).into());
    }
    conn.execute(
        "INSERT OR REPLACE INTO library_aliases (alias, library_name, created_at) VALUES (?1, ?2, ?3)",
        params![alias, library_name, now_epoch()],
    )?;
    Ok(())
}

fn library_status(conn: &Connection, library_name: &str) -> Result<String, Box<dyn Error>> {
    let status: Option<String> = conn
        .query_row(
            "SELECT status FROM jobs WHERE library_name = ?1 ORDER BY id DESC LIMIT 1",
            params![library_name],
            |row| row.get(0),
        )
        .optional()?;
    Ok(status.unwrap_or_else(|| "unknown".to_string()))
}

fn latest_success_time_by_kind(
    conn: &Connection,
    library_name: &str,
    kind: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let pattern = format!("%{kind}%");
    let value: Option<String> = conn
        .query_row(
            "SELECT ended_at
             FROM jobs
             WHERE library_name = ?1 AND status = 'success' AND job_type LIKE ?2
             ORDER BY id DESC
             LIMIT 1",
            params![library_name, pattern],
            |row| row.get(0),
        )
        .optional()?;
    Ok(value)
}

fn latest_failed_message(
    conn: &Connection,
    library_name: &str,
) -> Result<Option<String>, Box<dyn Error>> {
    let msg: Option<String> = conn
        .query_row(
            "SELECT message FROM jobs WHERE library_name = ?1 AND status = 'failed' ORDER BY id DESC LIMIT 1",
            params![library_name],
            |row| row.get(0),
        )
        .optional()?;
    Ok(msg)
}

fn parent_count_for_library(conn: &Connection, library_name: &str) -> Result<i64, Box<dyn Error>> {
    let count = conn.query_row(
        "SELECT COUNT(*) FROM parents WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    Ok(count)
}

fn bm25_count_for_library(conn: &Connection, library_name: &str) -> Result<i64, Box<dyn Error>> {
    bm25_readiness(conn, library_name)
}

fn library_rollups(
    conn: &Connection,
    library_name: &str,
) -> Result<(i64, i64, i64, i64, i64, i64, i64), Box<dyn Error>> {
    let values = conn.query_row(
        "SELECT
            COALESCE(content_size_chars, 0),
            COALESCE(page_count, 0),
            COALESCE(chunk_count, 0),
            COALESCE(embedded_chunk_count, 0),
            COALESCE(empty_page_count, 0),
            COALESCE(min_chunks_per_page, 0),
            COALESCE(max_chunks_per_page, 0)
         FROM libraries
         WHERE library_name = ?1",
        params![library_name],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
                row.get::<_, i64>(6)?,
            ))
        },
    )?;
    Ok(values)
}

fn aggregate_rollups_for_libraries(
    conn: &Connection,
    libraries: &[String],
) -> Result<(i64, i64, i64, i64, i64, i64), Box<dyn Error>> {
    let mut total_chars = 0i64;
    let mut total_pages = 0i64;
    let mut total_chunks = 0i64;
    let mut total_empty_pages = 0i64;
    let mut min_chunks = i64::MAX;
    let mut max_chunks = i64::MIN;
    let mut saw_min_max = false;
    for lib in libraries {
        let (chars, pages, chunks, _embedded, empty_pages, min_per_page, max_per_page) =
            library_rollups(conn, lib)?;
        total_chars += chars;
        total_pages += pages;
        total_chunks += chunks;
        total_empty_pages += empty_pages;
        if pages > 0 {
            if !saw_min_max || min_per_page < min_chunks {
                min_chunks = min_per_page;
            }
            if !saw_min_max || max_per_page > max_chunks {
                max_chunks = max_per_page;
            }
            saw_min_max = true;
        }
    }
    if !saw_min_max {
        min_chunks = 0;
        max_chunks = 0;
    }
    Ok((
        total_chars,
        total_pages,
        total_chunks,
        total_empty_pages,
        min_chunks,
        max_chunks,
    ))
}

fn list_libraries(conn: &Connection, output_json: bool) -> Result<(), Box<dyn Error>> {
    let spinner = ProgressSpinner::new("Loading libraries");
    let mut stmt = conn.prepare(
        "SELECT library_name, source_url, last_refreshed_at FROM libraries ORDER BY library_name ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let mut libraries = Vec::new();
    for row in rows {
        libraries.push(row?);
    }

    let mut lines = Vec::new();
    let mut json_libraries = Vec::new();
    for (library_name, source_url, refreshed) in libraries {
        spinner.set_stage(format!("Reading {}", library_name));
        let (chars, pages, chunks, _embedded, _empty, _min, _max) =
            library_rollups(conn, &library_name)?;
        let bm25_chunks = bm25_count_for_library(conn, &library_name)?;
        let status = library_status(conn, &library_name)?;
        json_libraries.push(json!({
            "kind": "library",
            "library_name": library_name,
            "source_url": source_url,
            "page_count": pages,
            "chunk_count": chunks,
            "bm25_indexed_chunk_count": bm25_chunks,
            "content_size_chars": chars,
            "status": status,
            "last_refreshed_at": refreshed,
        }));
        lines.push(format!("{library_name}"));
        lines.push(format!("  source: {source_url}"));
        lines.push(format!("  pages: {pages}"));
        lines.push(format!("  chunks: {chunks}"));
        lines.push(format!("  bm25 chunks: {bm25_chunks}"));
        lines.push(format!("  chars: {chars}"));
        lines.push(format!("  status: {status}"));
        lines.push(format!("  last refreshed: {}", human_time(&refreshed)));
    }

    let mut group_stmt =
        conn.prepare("SELECT DISTINCT group_name FROM library_groups ORDER BY group_name ASC")?;
    let group_rows = group_stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut group_names = Vec::new();
    for row in group_rows {
        group_names.push(row?);
    }
    let mut json_groups = Vec::new();
    for group_name in group_names {
        spinner.set_stage(format!("Reading {}", group_name));
        let members = group_members(conn, &group_name)?;
        let (content_size_chars, pages, chunks, _empty, _min, _max) =
            aggregate_rollups_for_libraries(conn, &members)?;
        json_groups.push(json!({
            "kind": "group",
            "library_name": group_name,
            "source_url": "merged group",
            "page_count": pages,
            "chunk_count": chunks,
            "content_size_chars": content_size_chars,
            "status": "merged",
            "last_refreshed_at": Value::Null,
            "members": members,
        }));
        lines.push(format!("{group_name}"));
        lines.push("  source: merged group".to_string());
        lines.push(format!("  pages: {pages}"));
        lines.push(format!("  chunks: {chunks}"));
        lines.push(format!("  chars: {content_size_chars}"));
        lines.push("  status: merged".to_string());
        lines.push("  last refreshed: n/a".to_string());
    }
    spinner.finish();
    if output_json {
        let mut entries = json_libraries;
        entries.extend(json_groups);
        return print_json(&json!({
            "command": "list",
            "libraries": entries,
        }));
    }
    for line in lines {
        println!("{line}");
    }
    Ok(())
}

fn show_library(conn: &Connection, input_name: &str, output_json: bool) -> Result<(), Box<dyn Error>> {
    let spinner = ProgressSpinner::new(format!("Loading {}", input_name));
    if let Ok(library_name) = resolve_library_name(conn, input_name) {
        spinner.set_stage(format!("Reading {}", library_name));
        let (source_url, refreshed): (String, String) = conn.query_row(
            "SELECT source_url, last_refreshed_at FROM libraries WHERE library_name = ?1",
            params![library_name],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let (
            content_size_chars,
            pages,
            chunks,
            embedded_chunks,
            empty_pages,
            min_chunks,
            max_chunks,
        ) = library_rollups(conn, &library_name)?;
        let parent_count = parent_count_for_library(conn, &library_name)?;
        let bm25_chunks = bm25_count_for_library(conn, &library_name)?;
        let avg_chunks = if pages > 0 {
            chunks as f64 / pages as f64
        } else {
            0.0
        };
        let latest_status = library_status(conn, &library_name)?;
        let last_crawled_at = latest_success_time_by_kind(conn, &library_name, "crawl")?;
        let last_indexed_at = latest_success_time_by_kind(conn, &library_name, "index")?;
        let latest_error = latest_failed_message(conn, &library_name)?;

        let mut alias_stmt = conn.prepare(
            "SELECT alias FROM library_aliases WHERE library_name = ?1 ORDER BY alias ASC",
        )?;
        let alias_rows =
            alias_stmt.query_map(params![library_name], |row| row.get::<_, String>(0))?;
        let mut aliases = Vec::new();
        for row in alias_rows {
            aliases.push(row?);
        }
        let mut lines = Vec::new();
        lines.push(format!("library_name: {library_name}"));
        lines.push(format!("source_url: {source_url}"));
        lines.push(format!("page_count: {pages}"));
        lines.push(format!("parent_count: {parent_count}"));
        lines.push(format!("chunk_count: {chunks}"));
        lines.push(format!("embedded_chunk_count: {embedded_chunks}"));
        lines.push(format!("bm25_indexed_chunk_count: {bm25_chunks}"));
        lines.push(format!("avg_chunks_per_page: {:.2}", avg_chunks));
        lines.push(format!("min_chunks_per_page: {min_chunks}"));
        lines.push(format!("max_chunks_per_page: {max_chunks}"));
        lines.push(format!("pages_with_no_chunks: {empty_pages}"));
        lines.push(format!("content_size_chars: {content_size_chars}"));
        lines.push(format!("indexed_model: {:?}", DEFAULT_EMBEDDING_MODEL));
        lines.push("embedding_dim: 1024".to_string());
        lines.push(format!("latest_job_status: {latest_status}"));
        lines.push(format!(
            "last_crawled_at: {}",
            last_crawled_at
                .as_deref()
                .map(human_time)
                .unwrap_or_else(|| "n/a".to_string())
        ));
        lines.push(format!(
            "last_indexed_at: {}",
            last_indexed_at
                .as_deref()
                .map(human_time)
                .unwrap_or_else(|| "n/a".to_string())
        ));
        if let Some(ref err) = latest_error {
            lines.push(format!("latest_error: {err}"));
        }
        lines.push(format!("last_refreshed_at: {}", human_time(&refreshed)));
        if !aliases.is_empty() {
            lines.push(format!("aliases: {}", aliases.join(", ")));
        }
        spinner.finish();
        if output_json {
            return print_json(&json!({
                "command": "show",
                "kind": "library",
                "library_name": library_name,
                "source_url": source_url,
                "page_count": pages,
                "parent_count": parent_count,
                "chunk_count": chunks,
                "embedded_chunk_count": embedded_chunks,
                "bm25_indexed_chunk_count": bm25_chunks,
                "avg_chunks_per_page": avg_chunks,
                "min_chunks_per_page": min_chunks,
                "max_chunks_per_page": max_chunks,
                "pages_with_no_chunks": empty_pages,
                "content_size_chars": content_size_chars,
                "indexed_model": format!("{:?}", DEFAULT_EMBEDDING_MODEL),
                "embedding_dim": 1024,
                "latest_job_status": latest_status,
                "last_crawled_at": last_crawled_at,
                "last_indexed_at": last_indexed_at,
                "latest_error": latest_error,
                "last_refreshed_at": refreshed,
                "aliases": aliases,
            }));
        }
        for line in lines {
            println!("{line}");
        }
        return Ok(());
    }

    spinner.set_stage(format!("Reading {}", input_name));
    let members = group_members(conn, input_name)?;
    if members.is_empty() {
        return Err(format!("Unknown library or merged group '{}'.", input_name).into());
    }
    let (content_size_chars, pages, chunks, empty_pages, min_chunks, max_chunks) =
        aggregate_rollups_for_libraries(conn, &members)?;
    let mut parent_count = 0i64;
    let mut embedded_chunks = 0i64;
    let mut bm25_chunks = 0i64;
    for member in &members {
        parent_count += parent_count_for_library(conn, member)?;
        bm25_chunks += bm25_count_for_library(conn, member)?;
        let (_, _, _, embedded, _, _, _) = library_rollups(conn, member)?;
        embedded_chunks += embedded;
    }
    let avg_chunks = if pages > 0 {
        chunks as f64 / pages as f64
    } else {
        0.0
    };
    let lines = vec![
        format!("library_name: {input_name}"),
        "source_url: merged group".to_string(),
        format!("page_count: {pages}"),
        format!("parent_count: {parent_count}"),
        format!("chunk_count: {chunks}"),
        format!("embedded_chunk_count: {embedded_chunks}"),
        format!("bm25_indexed_chunk_count: {bm25_chunks}"),
        format!("avg_chunks_per_page: {:.2}", avg_chunks),
        format!("min_chunks_per_page: {min_chunks}"),
        format!("max_chunks_per_page: {max_chunks}"),
        format!("pages_with_no_chunks: {empty_pages}"),
        format!("content_size_chars: {content_size_chars}"),
        format!("indexed_model: {:?}", DEFAULT_EMBEDDING_MODEL),
        "embedding_dim: 1024".to_string(),
        "latest_job_status: merged".to_string(),
        "last_crawled_at: n/a".to_string(),
        "last_indexed_at: n/a".to_string(),
        "last_refreshed_at: n/a".to_string(),
        format!("members: {}", members.join(", ")),
    ];
    spinner.finish();
    if output_json {
        return print_json(&json!({
            "command": "show",
            "kind": "group",
            "library_name": input_name,
            "source_url": "merged group",
            "page_count": pages,
            "parent_count": parent_count,
            "chunk_count": chunks,
            "embedded_chunk_count": embedded_chunks,
            "bm25_indexed_chunk_count": bm25_chunks,
            "avg_chunks_per_page": avg_chunks,
            "min_chunks_per_page": min_chunks,
            "max_chunks_per_page": max_chunks,
            "pages_with_no_chunks": empty_pages,
            "content_size_chars": content_size_chars,
            "indexed_model": format!("{:?}", DEFAULT_EMBEDDING_MODEL),
            "embedding_dim": 1024,
            "latest_job_status": "merged",
            "last_crawled_at": Value::Null,
            "last_indexed_at": Value::Null,
            "last_refreshed_at": Value::Null,
            "members": members,
        }));
    }
    for line in lines {
        println!("{line}");
    }
    Ok(())
}

fn remove_library(conn: &Connection, input_name: &str) -> Result<(), Box<dyn Error>> {
    let library_name = match resolve_library_name(conn, input_name) {
        Ok(name) => name,
        Err(_) => {
            let deleted = conn.execute(
                "DELETE FROM library_groups WHERE group_name = ?1",
                params![input_name],
            )?;
            if deleted > 0 {
                return Ok(());
            }
            return Err(format!("Unknown library or merged group '{}'.", input_name).into());
        }
    };
    conn.execute(
        "DELETE FROM library_aliases WHERE library_name = ?1",
        params![library_name],
    )?;
    conn.execute(
        "DELETE FROM chunks WHERE library_name = ?1",
        params![library_name],
    )?;
    conn.execute(
        "DELETE FROM chunks_fts WHERE library_name = ?1",
        params![library_name],
    )?;
    conn.execute(
        "DELETE FROM parents WHERE library_name = ?1",
        params![library_name],
    )?;
    conn.execute(
        "DELETE FROM pages WHERE library_name = ?1",
        params![library_name],
    )?;
    conn.execute(
        "DELETE FROM library_texts WHERE library_name = ?1",
        params![library_name],
    )?;
    conn.execute(
        "DELETE FROM jobs WHERE library_name = ?1",
        params![library_name],
    )?;
    conn.execute(
        "DELETE FROM libraries WHERE library_name = ?1",
        params![library_name],
    )?;
    let dir = compiled_dir(&library_name);
    if dir.exists() {
        fs::remove_dir_all(dir)?;
    }
    Ok(())
}

fn open_chunk(conn: &Connection, chunk_id: i64, output_json: bool) -> Result<(), Box<dyn Error>> {
    let (parent_id, library_name, source_url, content, child_index_in_parent): (
        i64,
        String,
        String,
        String,
        i64,
    ) = conn.query_row(
        "SELECT parent_id, library_name, source_url, content, child_index_in_parent FROM chunks WHERE id = ?1",
        params![chunk_id],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
    )?;
    let parent = load_parent_by_id(conn, parent_id)?;
    if output_json {
        return print_json(&json!({
            "command": "open",
            "chunk": {
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "library_name": library_name,
                "source_url": source_url,
                "child_index_in_parent": child_index_in_parent,
                "content": content,
            },
            "parent": {
                "parent_id": parent.id,
                "library_name": parent.library_name,
                "source_url": parent.source_url,
                "source_page_order": parent.source_page_order,
                "parent_index_in_page": parent.parent_index_in_page,
                "global_parent_index": parent.global_parent_index,
                "content": parent.content,
            }
        }));
    }
    println!("chunk_id: {chunk_id}");
    println!("parent_id: {parent_id}");
    println!("library_name: {library_name}");
    println!("source_url: {source_url}");
    println!("child_index_in_parent: {child_index_in_parent}");
    println!();
    println!("--- child ---");
    println!("{content}");
    println!();
    println!("--- parent ---");
    println!("{}", parent.content);
    Ok(())
}
