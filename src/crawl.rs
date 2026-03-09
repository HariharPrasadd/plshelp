use crate::*;

pub(crate) fn normalize_seed_url(seed_url: &str) -> Result<String, String> {
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

pub(crate) fn whitelist_for_url(seed_url: &str) -> Result<Vec<CompactString>, String> {
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

pub(crate) fn extract_content_html(html: &str) -> String {
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

pub(crate) fn cleanup_markdown(markdown: &str) -> String {
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

pub(crate) async fn crawl_library(
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

// ============================================================================
