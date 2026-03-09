use html2md;
use rayon::prelude::*;
use regex::Regex;
use scraper::{Html, Selector};
use spider::compact_str::CompactString;
use spider::configuration::Configuration;
use spider::website::Website;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;
use url::Url;

const OUTPUT_ROOT: &str = "/Users/hariharprasad/MyDocuments/Code/Rust/deep-spider/artifacts";

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

pub async fn crawl_domain_to_artifacts(
    seed_url: &str,
    relative_output_dir: &str,
) -> Result<(), Box<dyn Error>> {
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

    fn whitelist_for_domain(seed_url: &str) -> Result<Vec<CompactString>, String> {
        let parsed =
            Url::parse(seed_url).map_err(|e| format!("Invalid seed URL '{}': {}", seed_url, e))?;
        let host = parsed
            .host_str()
            .ok_or_else(|| format!("Seed URL '{}' has no host", seed_url))?;
        let scheme_pattern = regex::escape(parsed.scheme());
        let root_host = host
            .split('.')
            .collect::<Vec<_>>();
        let root_host = if root_host.len() >= 2 {
            format!("{}.{}", root_host[root_host.len() - 2], root_host[root_host.len() - 1])
        } else {
            host.to_string()
        };
        let authority_pattern = format!(r"(?:[A-Za-z0-9-]+\.)*{}", regex::escape(&root_host));
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
        fs::write(output_dir.join("docs.md"), content)?;
        fs::write(output_dir.join("docs.txt"), content)?;
        Ok(())
    }

    let normalized_seed_url = normalize_seed_url(seed_url)?;
    let whitelist = whitelist_for_domain(&normalized_seed_url)?;

    let mut config = Configuration::new();
    config
        .with_limit(5_000)
        .with_depth(25)
        .with_subdomains(true)
        .with_tld(false)
        .with_user_agent(Some("DocumentationScraper/1.0"))
        .with_whitelist_url(Some(whitelist));

    let mut website = Website::new(&normalized_seed_url)
        .with_config(config)
        .build()
        .map_err(|e| format!("Failed to build website: {e}"))?;

    website.scrape().await;
    let pages = website
        .get_pages()
        .ok_or_else(|| "No pages collected".to_string())?;

    let converted: Vec<String> = pages
        .iter()
        .map(|page| (page.get_url().to_string(), page.get_html()))
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|(_url, html)| {
            let extracted_html = extract_content_html(&html);
            cleanup_markdown(&html2md::parse_html(&extracted_html))
        })
        .collect();

    let mut compiled = converted.join("\n\n");
    if !compiled.is_empty() {
        compiled.push_str("\n\n");
    }

    let output_dir = PathBuf::from(OUTPUT_ROOT).join(relative_output_dir);
    write_artifacts(&output_dir, &compiled)?;
    Ok(())
}
