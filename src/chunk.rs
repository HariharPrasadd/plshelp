use crate::*;

pub(crate) fn strip_front_matter(input: &str) -> &str {
    if !input.starts_with("---\n") {
        return input;
    }
    if let Some(end) = input[4..].find("\n---\n") {
        let idx = 4 + end + 5;
        return &input[idx..];
    }
    input
}

pub(crate) fn preprocess_for_chunking(input: &str) -> String {
    let normalized = input.replace("\r\n", "\n");
    let stripped = strip_front_matter(&normalized);
    let setext_h1 = Regex::new(r"(?m)^([^\n][^\n]*)\n=+\s*$").expect("valid regex");
    let setext_h2 = Regex::new(r"(?m)^([^\n][^\n]*)\n-+\s*$").expect("valid regex");
    let out = setext_h1.replace_all(stripped, "# $1");
    setext_h2.replace_all(&out, "## $1").into_owned()
}

pub(crate) fn split_by_paragraph_upper_bound(chunks: Vec<String>, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.chars().count() <= max_chars {
            out.push(chunk);
            continue;
        }

        // Split paragraphs only outside fenced code blocks.
        let mut paragraphs: Vec<String> = Vec::new();
        let mut current_para = String::new();
        let mut in_fence = false;
        for line in chunk.split_inclusive('\n') {
            let line_no_nl = line.trim_end_matches('\n').trim_end_matches('\r');
            let starts_fence = line_no_nl.trim_start().starts_with("```");
            current_para.push_str(line);
            if starts_fence {
                in_fence = !in_fence;
            }
            if !in_fence && line_no_nl.trim().is_empty() {
                if !current_para.trim().is_empty() {
                    paragraphs.push(std::mem::take(&mut current_para));
                } else {
                    current_para.clear();
                }
            }
        }
        if !current_para.trim().is_empty() {
            paragraphs.push(current_para);
        }

        let mut current = String::new();
        for para in paragraphs {
            let para = para.trim();
            if para.is_empty() {
                continue;
            }
            if current.is_empty() {
                current.push_str(para);
                continue;
            }
            let candidate = format!("{current}\n\n{para}");
            if candidate.chars().count() <= max_chars {
                current = candidate;
            } else {
                out.push(current);
                current = para.to_string();
            }
        }
        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

pub(crate) fn split_by_newline_upper_bound(chunks: Vec<String>, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.chars().count() <= max_chars {
            out.push(chunk);
            continue;
        }

        let mut current = String::new();
        let mut in_fence = false;
        for line in chunk.split_inclusive('\n') {
            let line_no_nl = line.trim_end_matches('\n').trim_end_matches('\r');
            let starts_fence = line_no_nl.trim_start().starts_with("```");

            if !in_fence {
                let candidate_len = current.chars().count() + line.chars().count();
                if !current.is_empty() && candidate_len > max_chars {
                    out.push(std::mem::take(&mut current));
                }
            }

            current.push_str(line);
            if starts_fence {
                in_fence = !in_fence;
            }
        }

        if !current.is_empty() {
            out.push(current);
        }
    }
    out
}

pub(crate) fn split_by_char_upper_bound(chunks: Vec<String>, max_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for chunk in chunks {
        if chunk.chars().count() <= max_chars {
            out.push(chunk);
            continue;
        }

        let chars: Vec<char> = chunk.chars().collect();
        let mut start = 0usize;
        while start < chars.len() {
            let end = (start + max_chars).min(chars.len());
            let piece: String = chars[start..end].iter().collect();
            let trimmed = piece.trim().to_string();
            if !trimmed.is_empty() {
                out.push(trimmed);
            }
            start = end;
        }
    }
    out
}

pub(crate) fn split_markdown_by_headings(content: &str) -> Vec<String> {
    let processed = preprocess_for_chunking(content);
    let mut out = Vec::new();
    let mut current = String::new();
    let mut in_fence = false;
    for line in processed.split_inclusive('\n') {
        let line_no_nl = line.trim_end_matches('\n').trim_end_matches('\r');
        let starts_fence = line_no_nl.trim_start().starts_with("```");
        let is_heading = !in_fence && MD_ATX_HEADING_RE.is_match(line_no_nl);
        if is_heading && !current.trim().is_empty() {
            out.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        if starts_fence {
            in_fence = !in_fence;
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }
    out
}

pub(crate) fn chunk_markdown_page(content: &str) -> Vec<String> {
    let out = split_markdown_by_headings(content);

    // Final trim + dedupe pass.
    let mut cleaned = Vec::new();
    for chunk in out {
        let t = chunk.trim().to_string();
        if !t.is_empty() && cleaned.last() != Some(&t) {
            cleaned.push(t);
        }
    }
    cleaned = split_by_paragraph_upper_bound(cleaned, parent_max_chars());
    cleaned = split_by_newline_upper_bound(cleaned, parent_max_chars());
    cleaned = split_by_char_upper_bound(cleaned, parent_max_chars());

    // Lower-bound only top-up: merge tiny chunks forward until min size.
    let mut topped = Vec::new();
    let mut pending: Option<String> = None;
    for chunk in cleaned {
        match pending.take() {
            None => {
                pending = Some(chunk);
            }
            Some(prev) => {
                if prev.chars().count() >= parent_min_chars() {
                    topped.push(prev);
                    pending = Some(chunk);
                } else {
                    pending = Some(format!("{prev}\n\n{chunk}"));
                }
            }
        }
    }
    if let Some(last) = pending {
        topped.push(last);
    }

    // Tail fix: if the final chunk is below min size, append it to previous.
    if topped.len() >= 2 {
        let last_len = topped.last().map(|s| s.chars().count()).unwrap_or(0);
        if last_len < parent_min_chars() {
            if let Some(last_chunk) = topped.pop() {
                if let Some(prev) = topped.last_mut() {
                    prev.push_str("\n\n");
                    prev.push_str(&last_chunk);
                }
            }
        }
    }

    topped
}

pub(crate) fn chunk_parent_into_children(parent_content: &str) -> Vec<String> {
    let trimmed = parent_content.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let total_len = chars.len();
    if total_len <= child_max_chars() {
        return vec![trimmed.to_string()];
    }

    let target_children = total_len.div_ceil(child_max_chars()).max(1);
    let mut children = Vec::with_capacity(target_children);
    let mut start = 0usize;

    for part_idx in 0..target_children {
        let remaining_parts = target_children - part_idx;
        let remaining_len = total_len - start;
        if remaining_parts == 1 {
            let chunk: String = chars[start..].iter().collect();
            let trimmed_chunk = chunk.trim().to_string();
            if !trimmed_chunk.is_empty() {
                children.push(trimmed_chunk);
            }
            break;
        }

        let ideal_end = start + remaining_len.div_ceil(remaining_parts);
        let search_start = ideal_end
            .saturating_sub(child_split_window_chars())
            .max(start + child_min_chars());
        let search_end = (ideal_end + child_split_window_chars())
            .min(total_len)
            .min(start + child_max_chars());

        let mut best_split = None;
        let mut best_distance = usize::MAX;
        for idx in search_start..search_end {
            if chars[idx].is_whitespace() {
                let distance = idx.abs_diff(ideal_end);
                if distance < best_distance {
                    best_distance = distance;
                    best_split = Some(idx);
                }
            }
        }

        let end = best_split.unwrap_or_else(|| {
            ideal_end
                .max(start + child_min_chars())
                .min(start + child_max_chars())
                .min(total_len)
        });
        let chunk: String = chars[start..end].iter().collect();
        let trimmed_chunk = chunk.trim().to_string();
        if !trimmed_chunk.is_empty() {
            children.push(trimmed_chunk);
        }
        start = end;
        while start < total_len && chars[start].is_whitespace() {
            start += 1;
        }
    }

    children
}

// ============================================================================
