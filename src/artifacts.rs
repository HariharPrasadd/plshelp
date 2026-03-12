use crate::*;

pub(crate) fn write_artifacts(output_dir: &Path, content: &str) -> Result<(), Box<dyn Error>> {
    fs::create_dir_all(output_dir)?;
    fs::write(output_dir.join("docs.txt"), content)?;
    fs::write(output_dir.join("docs.md"), content)?;
    Ok(())
}

pub(crate) fn export_library(
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

pub(crate) fn export_all_libraries(
    conn: &Connection,
    output_root: Option<&Path>,
) -> Result<usize, Box<dyn Error>> {
    let library_names = all_library_names(conn)?;
    for library_name in &library_names {
        let output_dir = output_root
            .map(|root| root.join(library_name))
            .unwrap_or_else(|| compiled_dir(library_name));
        export_library(conn, library_name, Some(&output_dir))?;
    }
    Ok(library_names.len())
}

// ============================================================================
