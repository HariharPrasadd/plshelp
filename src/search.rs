use crate::*;

pub(crate) fn bm25_readiness(conn: &Connection, library_name: &str) -> Result<i64, Box<dyn Error>> {
    let count = conn.query_row(
        "SELECT COUNT(*) FROM chunks_fts WHERE library_name = ?1",
        params![library_name],
        |row| row.get(0),
    )?;
    Ok(count)
}

pub(crate) fn rebuild_bm25_index_for_library(
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

pub(crate) fn tokenize_fts_query(question: &str) -> String {
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

pub(crate) fn bm25_scores_for_library(
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

pub(crate) fn normalized_scores(scores: &HashMap<i64, f32>) -> HashMap<i64, f32> {
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

pub(crate) fn query_library(
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

pub(crate) fn ask_libraries(
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

// ============================================================================
