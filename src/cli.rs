use crate::*;

pub(crate) fn print_help() {
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

// ============================================================================
pub(crate) fn parse_query_flags(flags: &[String]) -> Result<(SearchMode, usize, usize), Box<dyn Error>> {
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

pub(crate) fn extract_json_flag(flags: &[String]) -> (bool, Vec<String>) {
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

pub(crate) fn split_query_and_flags(args: &[String]) -> (String, Vec<String>) {
    let first_flag = args
        .iter()
        .position(|arg| arg.starts_with("--"))
        .unwrap_or(args.len());
    let query = args[..first_flag].join(" ").trim().to_string();
    let flags = args[first_flag..].to_vec();
    (query, flags)
}

pub(crate) fn print_json(value: &Value) -> Result<(), Box<dyn Error>> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

pub(crate) fn print_command_result(
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

pub(crate) fn context_to_json(context: &[ParentRecord], active_parent_id: i64) -> Vec<Value> {
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

pub(crate) fn query_hit_to_json(
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

pub(crate) fn ask_flags(
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

pub(crate) fn parse_index_file_flag(flags: &[String]) -> Option<String> {
    let mut i = 0usize;
    while i < flags.len() {
        if flags[i] == "--file" && i + 1 < flags.len() {
            return Some(flags[i + 1].clone());
        }
        i += 1;
    }
    None
}

pub(crate) fn parse_include_artifacts_flag(flags: &[String], library_name: &str) -> Option<PathBuf> {
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

pub(crate) fn parse_merge_args(
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
