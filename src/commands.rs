use crate::*;

pub(crate) fn merge_libraries(
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

pub(crate) async fn add_library(
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

pub(crate) fn refresh_stats(conn: &Connection, input_names: &[String]) -> Result<(), Box<dyn Error>> {
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
