use crate::*;

pub(crate) fn resolve_library_name(conn: &Connection, input: &str) -> Result<String, Box<dyn Error>> {
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

pub(crate) fn group_members(conn: &Connection, group_name: &str) -> Result<Vec<String>, Box<dyn Error>> {
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

pub(crate) fn resolve_target_libraries(conn: &Connection, input: &str) -> Result<Vec<String>, Box<dyn Error>> {
    if let Ok(name) = resolve_library_name(conn, input) {
        return Ok(vec![name]);
    }
    let members = group_members(conn, input)?;
    if members.is_empty() {
        return Err(format!("Unknown library or merged group '{}'.", input).into());
    }
    Ok(members)
}

pub(crate) fn compiled_text_for_library(
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

pub(crate) fn backfill_pages_from_parents(
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

pub(crate) fn backfill_library_text(
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

pub(crate) fn compute_library_chars(conn: &Connection, library_name: &str) -> Result<i64, Box<dyn Error>> {
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

pub(crate) fn compute_page_chunk_rollups(
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

pub(crate) fn update_library_rollups(conn: &Connection, library_name: &str) -> Result<(), Box<dyn Error>> {
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

// ============================================================================
pub(crate) fn add_alias(conn: &Connection, input_name: &str, alias: &str) -> Result<(), Box<dyn Error>> {
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

pub(crate) fn library_status(conn: &Connection, library_name: &str) -> Result<String, Box<dyn Error>> {
    let status: Option<String> = conn
        .query_row(
            "SELECT status FROM jobs WHERE library_name = ?1 ORDER BY id DESC LIMIT 1",
            params![library_name],
            |row| row.get(0),
        )
        .optional()?;
    Ok(status.unwrap_or_else(|| "unknown".to_string()))
}

pub(crate) fn latest_success_time_by_kind(
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

pub(crate) fn latest_failed_message(
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

pub(crate) fn parent_count_for_library(conn: &Connection, library_name: &str) -> Result<i64, Box<dyn Error>> {
    let count = conn.query_row(
        "SELECT COUNT(*) FROM parents WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    Ok(count)
}

pub(crate) fn bm25_count_for_library(conn: &Connection, library_name: &str) -> Result<i64, Box<dyn Error>> {
    bm25_readiness(conn, library_name)
}

pub(crate) fn library_rollups(
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

pub(crate) fn aggregate_rollups_for_libraries(
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

pub(crate) fn list_libraries(conn: &Connection, output_json: bool) -> Result<(), Box<dyn Error>> {
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

pub(crate) fn show_library(conn: &Connection, input_name: &str, output_json: bool) -> Result<(), Box<dyn Error>> {
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

pub(crate) fn remove_library(conn: &Connection, input_name: &str) -> Result<(), Box<dyn Error>> {
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

pub(crate) fn open_chunk(conn: &Connection, chunk_id: i64, output_json: bool) -> Result<(), Box<dyn Error>> {
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
