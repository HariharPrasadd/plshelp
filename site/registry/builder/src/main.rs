use chrono::{DateTime, Utc};
use spider::compact_str::CompactString;
use rayon::prelude::*;
use regex::Regex;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use spider::configuration::Configuration;
use spider::website::Website;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use std::time::Instant;
use url::Url;

static CONTENT_SELECTORS: LazyLock<Vec<Selector>> = LazyLock::new(|| {
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
        r#"(?is)<(div|section)[^>]*(id|class)\s*=\s*[\"'][^\"']*(nav|menu|sidebar|footer|header|toc|breadcrumb|pagination|cookie|consent|search|feedback|promo|banner|advert|ads|social|share)[^\"']*[\"'][^>]*>.*?</\1>"#,
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

#[derive(Debug, Deserialize)]
struct SourcesFile {
    version: u32,
    entries: Vec<SourceEntry>,
}

#[derive(Debug, Deserialize)]
struct SourceEntry {
    name: String,
    slug: String,
    source_url: String,
    enabled: bool,
}

#[derive(Debug, Serialize)]
struct IndexFile {
    version: u32,
    generated_at: DateTime<Utc>,
    entries: Vec<IndexEntry>,
}

#[derive(Debug, Serialize)]
struct IndexEntry {
    name: String,
    slug: String,
    source_url: String,
    pages: usize,
    content_size_chars: usize,
    artifacts: ArtifactMetadata,
    last_crawled_at: DateTime<Utc>,
    last_successful_crawled_at: Option<DateTime<Utc>>,
    crawl_duration_ms: u128,
    status: &'static str,
    error_message: Option<String>,
}

#[derive(Debug, Serialize)]
struct ArtifactMetadata {
    markdown_path: String,
    text_path: String,
    markdown_bytes: u64,
    text_bytes: u64,
    markdown_sha256: Option<String>,
    text_sha256: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let slug_filter = parse_slug_filter()?;
    let builder_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let registry_root = builder_root
        .parent()
        .ok_or("builder directory has no registry parent")?
        .to_path_buf();
    let site_root = registry_root
        .parent()
        .ok_or("registry directory has no site parent")?
        .to_path_buf();

    let sources_path = registry_root.join("sources.json");
    let public_registry_root = site_root.join("public").join("registry");
    let public_docs_root = public_registry_root.join("docs");
    fs::create_dir_all(&public_docs_root)?;
    fs::create_dir_all(&public_registry_root)?;

    let sources: SourcesFile = serde_json::from_str(&fs::read_to_string(&sources_path)?)?;
    let mut index = IndexFile {
        version: sources.version,
        generated_at: Utc::now(),
        entries: Vec::new(),
    };

    for source in sources
        .entries
        .into_iter()
        .filter(|entry| entry.enabled)
        .filter(|entry| slug_filter.as_ref().is_none_or(|slug| slug == &entry.slug))
    {
        println!("Crawling {} ({})", source.slug, source.source_url);
        let result = crawl_registry_entry(&source, &public_docs_root).await;
        println!("Done crawling {}", source.slug);
        index.entries.push(result);
        index.generated_at = Utc::now();
        write_index_file(&public_registry_root, &index)?;
    }

    print_run_summary(&index);
    Ok(())
}

fn write_index_file(output_root: &Path, index: &IndexFile) -> Result<(), Box<dyn Error>> {
    let tmp_path = output_root.join("index.json.tmp");
    let final_path = output_root.join("index.json");
    fs::write(&tmp_path, serde_json::to_string_pretty(index)?)?;
    fs::rename(tmp_path, final_path)?;
    Ok(())
}

fn print_run_summary(index: &IndexFile) {
    let total = index.entries.len();
    let succeeded = index
        .entries
        .iter()
        .filter(|entry| entry.status == "success")
        .count();
    let failed_entries: Vec<&IndexEntry> = index
        .entries
        .iter()
        .filter(|entry| entry.status == "failed")
        .collect();

    println!();
    println!("Registry crawl complete.");
    println!("Processed: {}", total);
    println!("Succeeded: {}", succeeded);
    println!("Failed: {}", failed_entries.len());

    if !failed_entries.is_empty() {
        println!();
        println!("Failed entries:");
        for entry in failed_entries {
            match &entry.error_message {
                Some(message) => println!("- {}: {}", entry.slug, message),
                None => println!("- {}", entry.slug),
            }
        }
    }
}

fn parse_slug_filter() -> Result<Option<String>, Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut slug = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--slug" => {
                let value = args
                    .next()
                    .ok_or("--slug requires a value")?;
                slug = Some(value);
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --manifest-path site/registry/builder/Cargo.toml -- [--slug <slug>]");
                std::process::exit(0);
            }
            other => {
                return Err(format!("Unknown argument: {}", other).into());
            }
        }
    }
    Ok(slug)
}

async fn crawl_registry_entry(
    source: &SourceEntry,
    docs_root: &Path,
) -> IndexEntry {
    let started_at = Utc::now();
    let timer = Instant::now();
    let markdown_path = format!("/registry/docs/{}/docs.md", source.slug);
    let text_path = format!("/registry/docs/{}/docs.txt", source.slug);

    match crawl_source_to_compiled_text(&source.source_url).await {
        Ok(result) => {
            let output_dir = docs_root.join(&source.slug);
            if let Err(err) = write_artifacts(&output_dir, &result.compiled) {
                return IndexEntry {
                    name: source.name.clone(),
                    slug: source.slug.clone(),
                    source_url: source.source_url.clone(),
                    pages: 0,
                    content_size_chars: 0,
                    artifacts: ArtifactMetadata {
                        markdown_path,
                        text_path,
                        markdown_bytes: 0,
                        text_bytes: 0,
                        markdown_sha256: None,
                        text_sha256: None,
                    },
                    last_crawled_at: started_at,
                    last_successful_crawled_at: None,
                    crawl_duration_ms: timer.elapsed().as_millis(),
                    status: "failed",
                    error_message: Some(err.to_string()),
                };
            }

            let markdown_file = output_dir.join("docs.md");
            let text_file = output_dir.join("docs.txt");
            let markdown_bytes = file_len(&markdown_file).unwrap_or(0);
            let text_bytes = file_len(&text_file).unwrap_or(0);
            let markdown_sha256 = sha256_file(&markdown_file).ok();
            let text_sha256 = sha256_file(&text_file).ok();

            IndexEntry {
                name: source.name.clone(),
                slug: source.slug.clone(),
                source_url: source.source_url.clone(),
                pages: result.pages,
                content_size_chars: result.content_size_chars,
                artifacts: ArtifactMetadata {
                    markdown_path,
                    text_path,
                    markdown_bytes,
                    text_bytes,
                    markdown_sha256,
                    text_sha256,
                },
                last_crawled_at: started_at,
                last_successful_crawled_at: Some(Utc::now()),
                crawl_duration_ms: timer.elapsed().as_millis(),
                status: "success",
                error_message: None,
            }
        }
        Err(err) => IndexEntry {
            name: source.name.clone(),
            slug: source.slug.clone(),
            source_url: source.source_url.clone(),
            pages: 0,
            content_size_chars: 0,
            artifacts: ArtifactMetadata {
                markdown_path,
                text_path,
                markdown_bytes: 0,
                text_bytes: 0,
                markdown_sha256: None,
                text_sha256: None,
            },
            last_crawled_at: started_at,
            last_successful_crawled_at: None,
            crawl_duration_ms: timer.elapsed().as_millis(),
            status: "failed",
            error_message: Some(err.to_string()),
        },
    }
}

struct CrawlResult {
    compiled: String,
    pages: usize,
    content_size_chars: usize,
}

async fn crawl_source_to_compiled_text(source_url: &str) -> Result<CrawlResult, Box<dyn Error>> {
    let normalized_seed_url =
        normalize_seed_url(source_url).map_err(|e| format!("URL error: {e}"))?;
    let whitelist = whitelist_for_url(&normalized_seed_url)
        .map_err(|e| format!("Whitelist error: {e}"))?;

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
    website.scrape().await;

    let pages = match website.get_pages() {
        Some(p) => p,
        None => return Err("No pages collected".into()),
    };

    let page_inputs: Vec<(String, String)> = pages
        .iter()
        .map(|p| (p.get_url().to_string(), p.get_html()))
        .collect();

    let converted: Vec<(String, String)> = page_inputs
        .into_par_iter()
        .map(|(url, html)| {
            let extracted_html = extract_content_html(&html);
            let markdown = cleanup_markdown(&html2md::parse_html(&extracted_html));
            (url, markdown)
        })
        .collect();

    let mut compiled_parts = Vec::with_capacity(converted.len());
    let mut total_chars = 0usize;
    for (_, markdown) in &converted {
        total_chars += markdown.chars().count();
        compiled_parts.push(markdown.clone());
    }

    let mut compiled = compiled_parts.join("\n\n");
    if !compiled.is_empty() {
        compiled.push_str("\n\n");
    }

    Ok(CrawlResult {
        compiled,
        pages: converted.len(),
        content_size_chars: total_chars,
    })
}

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

fn write_artifacts(output_dir: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    fs::write(output_dir.join("docs.txt"), content)?;
    fs::write(output_dir.join("docs.md"), content)?;
    Ok(())
}

fn file_len(path: &Path) -> Result<u64, Box<dyn Error>> {
    Ok(fs::metadata(path)?.len())
}

fn sha256_file(path: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{:x}", digest))
}
