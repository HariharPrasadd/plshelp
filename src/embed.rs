use crate::*;

pub(crate) fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub(crate) fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(crate) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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

pub(crate) fn resolve_or_create_library_for_index(
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

pub(crate) fn load_page_inputs(
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

pub(crate) fn chunk_library(
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

pub(crate) fn chunk_targets(
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

pub(crate) fn embed_library(
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
            InitOptions::new(DEFAULT_EMBEDDING_MODEL)
                .with_cache_dir(models_dir())
                .with_show_download_progress(true),
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

pub(crate) fn index_library(
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

pub(crate) fn load_chunks_for_library(
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

pub(crate) fn load_parent_by_id(conn: &Connection, parent_id: i64) -> Result<ParentRecord, Box<dyn Error>> {
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

pub(crate) fn score_chunks(
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

pub(crate) fn parent_neighbors(
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

pub(crate) fn embed_query(mode: SearchMode, question: &str) -> Result<Option<Vec<f32>>, Box<dyn Error>> {
    if let SearchMode::Keyword = mode {
        return Ok(None);
    }
    let mut model = TextEmbedding::try_new(
        InitOptions::new(DEFAULT_EMBEDDING_MODEL)
            .with_cache_dir(models_dir())
            .with_show_download_progress(false),
    )?;
    let embedding = model.embed([question], None)?;
    Ok(embedding.first().cloned())
}

pub(crate) fn embedding_readiness(
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
