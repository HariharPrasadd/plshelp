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
pub(crate) use std::os::fd::AsRawFd;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::atomic::{AtomicBool, Ordering};
pub(crate) use std::sync::{Arc, LazyLock, Mutex, OnceLock};
pub(crate) use std::thread;
pub(crate) use std::time::{Duration, SystemTime, UNIX_EPOCH};
pub(crate) use termios::{tcsetattr, Termios, ECHO, TCSANOW};
pub(crate) use url::Url;

pub(crate) const DEFAULT_TOP_K: usize = 1;
pub(crate) const DEFAULT_CONTEXT_WINDOW: usize = 0;
pub(crate) const DEFAULT_EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::AllMiniLML6V2Q;
pub(crate) const MIN_CHILD_LENGTH: usize = 700;
pub(crate) const MAX_CHILD_LENGTH: usize = 1400;
pub(crate) const CHILD_SPLIT_WINDOW: usize = 50;
pub(crate) const DEFAULT_EMBED_BATCH_SIZE: usize = 128;
pub(crate) const SQLITE_BUSY_TIMEOUT_MS: u64 = 5_000;
pub(crate) const APP_NAME: &str = "plshelp";
pub(crate) const CONFIG_FILE_NAME: &str = "config.toml";

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

#[derive(Debug, Clone)]
pub(crate) struct RuntimePaths {
    pub(crate) config_dir: PathBuf,
    pub(crate) config_file: PathBuf,
    pub(crate) data_dir: PathBuf,
    pub(crate) db_path: PathBuf,
    pub(crate) artifacts_dir: PathBuf,
    pub(crate) models_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct AppConfigFile {
    #[serde(default)]
    pub(crate) paths: PathsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct PathsConfig {
    pub(crate) data_dir: Option<PathBuf>,
    pub(crate) db_path: Option<PathBuf>,
    pub(crate) artifacts_dir: Option<PathBuf>,
    pub(crate) models_dir: Option<PathBuf>,
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

pub async fn run(args: Vec<String>) -> Result<(), Box<dyn Error>> {
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
                    "Usage: plshelp add <library_name> <source_url> [--include-artifacts[=/path]]".into(),
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
                    "Usage: plshelp crawl <library_name> <source_url> [--include-artifacts[=/path]]".into(),
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
