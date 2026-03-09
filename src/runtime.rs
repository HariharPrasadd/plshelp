use crate::*;

pub(crate) fn runtime_settings() -> &'static RuntimeSettings {
    RUNTIME_SETTINGS
        .get()
        .expect("runtime settings must be initialized before use")
}

pub(crate) fn embedding_model() -> EmbeddingModel {
    runtime_settings().embedding_model.clone()
}

pub(crate) fn embed_batch_size() -> usize {
    runtime_settings().embed_batch_size
}

pub(crate) fn parent_min_chars() -> usize {
    runtime_settings().parent_min_chars
}

pub(crate) fn parent_max_chars() -> usize {
    runtime_settings().parent_max_chars
}

pub(crate) fn child_min_chars() -> usize {
    runtime_settings().child_min_chars
}

pub(crate) fn child_max_chars() -> usize {
    runtime_settings().child_max_chars
}

pub(crate) fn child_split_window_chars() -> usize {
    runtime_settings().child_split_window_chars
}

pub(crate) fn default_search_mode() -> SearchMode {
    runtime_settings().default_mode
}

pub(crate) fn default_top_k() -> usize {
    runtime_settings().default_top_k
}

pub(crate) fn default_context_window() -> usize {
    runtime_settings().default_context_window
}

pub(crate) fn hybrid_vector_weight() -> f32 {
    runtime_settings().hybrid_vector_weight
}

pub(crate) fn hybrid_bm25_weight() -> f32 {
    runtime_settings().hybrid_bm25_weight
}

pub(crate) fn sqlite_journal_mode() -> String {
    runtime_settings().sqlite_journal_mode.clone()
}

pub(crate) fn sqlite_busy_timeout_ms() -> u64 {
    runtime_settings().sqlite_busy_timeout_ms
}

pub(crate) fn onnx_intra_threads() -> usize {
    runtime_settings().onnx_intra_threads
}

pub(crate) fn onnx_inter_threads() -> usize {
    runtime_settings().onnx_inter_threads
}

pub(crate) fn configure_onnx_runtime_env() {
    // These must be set before ONNX Runtime initializes.
    unsafe {
        env::set_var("OMP_NUM_THREADS", onnx_intra_threads().to_string());
        env::set_var("ORT_NUM_INTRA_THREADS", onnx_intra_threads().to_string());
        env::set_var("ORT_NUM_INTER_THREADS", onnx_inter_threads().to_string());
    }
}

pub(crate) fn runtime_paths() -> &'static RuntimePaths {
    RUNTIME_PATHS
        .get()
        .expect("runtime paths must be initialized before use")
}

pub(crate) fn db_path() -> PathBuf {
    runtime_paths().db_path.clone()
}

pub(crate) fn artifacts_root() -> PathBuf {
    runtime_paths().artifacts_dir.clone()
}

pub(crate) fn models_dir() -> PathBuf {
    runtime_paths().models_dir.clone()
}

pub(crate) fn compiled_dir(library_name: &str) -> PathBuf {
    artifacts_root().join(library_name)
}

pub(crate) fn initialize_runtime_paths() -> Result<&'static RuntimePaths, Box<dyn Error>> {
    if let Some(paths) = RUNTIME_PATHS.get() {
        return Ok(paths);
    }

    let default_config_dir = default_config_dir()?;
    let default_data_dir = default_data_dir()?;
    let default_paths = RuntimePaths {
        config_file: default_config_dir.join(CONFIG_FILE_NAME),
        config_dir: default_config_dir,
        data_dir: default_data_dir.clone(),
        db_path: default_data_dir.join("plshelp.db"),
        artifacts_dir: default_data_dir.join("artifacts"),
        models_dir: default_data_dir.join("models"),
    };

    fs::create_dir_all(&default_paths.config_dir)?;
    fs::create_dir_all(&default_paths.data_dir)?;
    fs::create_dir_all(&default_paths.artifacts_dir)?;
    fs::create_dir_all(&default_paths.models_dir)?;

    if !default_paths.config_file.exists() {
        write_default_config(&default_paths)?;
    }

    let config = load_config_file(&default_paths.config_file);
    let data_dir = resolve_config_path(
        config.paths.data_dir.as_ref(),
        &default_paths.data_dir,
        &default_paths.config_dir,
    );
    let db_path = resolve_config_path(
        config.paths.db_path.as_ref(),
        &data_dir.join("plshelp.db"),
        &default_paths.config_dir,
    );
    let artifacts_dir = resolve_config_path(
        config.paths.artifacts_dir.as_ref(),
        &data_dir.join("artifacts"),
        &default_paths.config_dir,
    );
    let models_dir = resolve_config_path(
        config.paths.models_dir.as_ref(),
        &data_dir.join("models"),
        &default_paths.config_dir,
    );

    fs::create_dir_all(&data_dir)?;
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir_all(&artifacts_dir)?;
    fs::create_dir_all(&models_dir)?;

    let runtime = RuntimePaths {
        config_dir: default_paths.config_dir,
        config_file: default_paths.config_file,
        data_dir,
        db_path,
        artifacts_dir,
        models_dir,
    };
    let settings = resolve_runtime_settings(&config);
    let _ = RUNTIME_PATHS.set(runtime);
    let _ = RUNTIME_SETTINGS.set(settings);
    Ok(runtime_paths())
}

pub(crate) fn write_default_config(defaults: &RuntimePaths) -> Result<(), Box<dyn Error>> {
    let config = AppConfigFile {
        paths: PathsConfig {
            data_dir: Some(defaults.data_dir.clone()),
            db_path: Some(defaults.db_path.clone()),
            artifacts_dir: Some(defaults.artifacts_dir.clone()),
            models_dir: Some(defaults.models_dir.clone()),
        },
        embedding: EmbeddingConfig {
            model: Some(format!("{:?}", DEFAULT_EMBEDDING_MODEL)),
            batch_size: Some(DEFAULT_EMBED_BATCH_SIZE),
        },
        chunking: ChunkingConfig {
            parent_min_chars: Some(DEFAULT_PARENT_MIN_CHARS),
            parent_max_chars: Some(DEFAULT_PARENT_MAX_CHARS),
            child_min_chars: Some(MIN_CHILD_LENGTH),
            child_max_chars: Some(MAX_CHILD_LENGTH),
            child_split_window_chars: Some(CHILD_SPLIT_WINDOW),
        },
        retrieval: RetrievalConfig {
            default_mode: Some(SearchMode::Hybrid.as_str().to_string()),
            default_top_k: Some(DEFAULT_TOP_K),
            default_context_window: Some(DEFAULT_CONTEXT_WINDOW),
            hybrid_vector_weight: Some(0.90),
            hybrid_bm25_weight: Some(0.10),
        },
        sqlite: SqliteConfig {
            journal_mode: Some("WAL".to_string()),
            busy_timeout_ms: Some(SQLITE_BUSY_TIMEOUT_MS),
        },
        onnx: OnnxConfig {
            intra_threads: Some(
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4),
            ),
            inter_threads: Some(1),
        },
    };
    let serialized = toml::to_string_pretty(&config)?;
    fs::write(&defaults.config_file, serialized)?;
    Ok(())
}

pub(crate) fn load_config_file(path: &Path) -> AppConfigFile {
    let Ok(raw) = fs::read_to_string(path) else {
        return AppConfigFile::default();
    };
    toml::from_str::<AppConfigFile>(&raw).unwrap_or_default()
}

pub(crate) fn resolve_config_path(value: Option<&PathBuf>, fallback: &Path, base_dir: &Path) -> PathBuf {
    match value {
        Some(path) => {
            let expanded = expand_home(path);
            if expanded.is_absolute() {
                expanded
            } else {
                base_dir.join(expanded)
            }
        }
        None => fallback.to_path_buf(),
    }
}

pub(crate) fn expand_home(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        return home_dir().unwrap_or_else(|| path.to_path_buf());
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    path.to_path_buf()
}

pub(crate) fn home_dir() -> Option<PathBuf> {
    if let Some(home) = env::var_os("HOME") {
        return Some(PathBuf::from(home));
    }
    if let Some(profile) = env::var_os("USERPROFILE") {
        return Some(PathBuf::from(profile));
    }
    let drive = env::var_os("HOMEDRIVE");
    let path = env::var_os("HOMEPATH");
    match (drive, path) {
        (Some(drive), Some(path)) => {
            let mut buf = PathBuf::from(drive);
            buf.push(path);
            Some(buf)
        }
        _ => None,
    }
}

pub(crate) fn default_config_dir() -> Result<PathBuf, Box<dyn Error>> {
    if cfg!(target_os = "macos") {
        let home = home_dir().ok_or("Unable to resolve home directory for config path.")?;
        return Ok(home
            .join("Library")
            .join("Application Support")
            .join(APP_NAME));
    }
    if cfg!(target_os = "windows") {
        let appdata = env::var_os("APPDATA")
            .map(PathBuf::from)
            .ok_or("APPDATA is not set.")?;
        return Ok(appdata.join(APP_NAME));
    }
    if let Some(xdg) = env::var_os("XDG_CONFIG_HOME") {
        return Ok(PathBuf::from(xdg).join(APP_NAME));
    }
    let home = home_dir().ok_or("Unable to resolve home directory for config path.")?;
    Ok(home.join(".config").join(APP_NAME))
}

pub(crate) fn default_data_dir() -> Result<PathBuf, Box<dyn Error>> {
    if cfg!(target_os = "macos") {
        let home = home_dir().ok_or("Unable to resolve home directory for data path.")?;
        return Ok(home
            .join("Library")
            .join("Application Support")
            .join(APP_NAME));
    }
    if cfg!(target_os = "windows") {
        let appdata = env::var_os("APPDATA")
            .map(PathBuf::from)
            .ok_or("APPDATA is not set.")?;
        return Ok(appdata.join(APP_NAME));
    }
    if let Some(xdg) = env::var_os("XDG_DATA_HOME") {
        return Ok(PathBuf::from(xdg).join(APP_NAME));
    }
    let home = home_dir().ok_or("Unable to resolve home directory for data path.")?;
    Ok(home.join(".local").join("share").join(APP_NAME))
}

pub(crate) fn now_epoch() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    secs.to_string()
}

pub(crate) fn human_time(epoch: &str) -> String {
    if let Ok(secs) = epoch.parse::<i64>() {
        if let Some(dt) = DateTime::<Utc>::from_timestamp(secs, 0) {
            return dt.format("%B %-d, %Y").to_string();
        }
    }
    epoch.to_string()
}

pub(crate) fn resolve_runtime_settings(config: &AppConfigFile) -> RuntimeSettings {
    let available_parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let embedding_model = config
        .embedding
        .model
        .as_deref()
        .and_then(parse_embedding_model)
        .unwrap_or(DEFAULT_EMBEDDING_MODEL);
    let embed_batch_size = config
        .embedding
        .batch_size
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_EMBED_BATCH_SIZE);

    let parent_min = config
        .chunking
        .parent_min_chars
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_PARENT_MIN_CHARS);
    let parent_max = config
        .chunking
        .parent_max_chars
        .filter(|v| *v >= parent_min)
        .unwrap_or(DEFAULT_PARENT_MAX_CHARS.max(parent_min));
    let child_min = config
        .chunking
        .child_min_chars
        .filter(|v| *v > 0)
        .unwrap_or(MIN_CHILD_LENGTH);
    let child_max = config
        .chunking
        .child_max_chars
        .filter(|v| *v >= child_min)
        .unwrap_or(MAX_CHILD_LENGTH.max(child_min));
    let child_split_window = config
        .chunking
        .child_split_window_chars
        .filter(|v| *v > 0)
        .unwrap_or(CHILD_SPLIT_WINDOW);

    let default_mode = config
        .retrieval
        .default_mode
        .as_deref()
        .map(SearchMode::from_str)
        .unwrap_or(SearchMode::Hybrid);
    let default_top_k = config
        .retrieval
        .default_top_k
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_TOP_K);
    let default_context_window = config
        .retrieval
        .default_context_window
        .unwrap_or(DEFAULT_CONTEXT_WINDOW);
    let mut hybrid_vector_weight = config
        .retrieval
        .hybrid_vector_weight
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(0.90);
    let mut hybrid_bm25_weight = config
        .retrieval
        .hybrid_bm25_weight
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(0.10);
    let weight_sum = hybrid_vector_weight + hybrid_bm25_weight;
    if weight_sum > 0.0 {
        hybrid_vector_weight /= weight_sum;
        hybrid_bm25_weight /= weight_sum;
    } else {
        hybrid_vector_weight = 0.90;
        hybrid_bm25_weight = 0.10;
    }

    let sqlite_journal_mode = match config.sqlite.journal_mode.as_deref() {
        Some(mode) if mode.eq_ignore_ascii_case("wal") => "WAL".to_string(),
        _ => "WAL".to_string(),
    };
    let sqlite_busy_timeout_ms = config
        .sqlite
        .busy_timeout_ms
        .filter(|v| *v > 0)
        .unwrap_or(SQLITE_BUSY_TIMEOUT_MS);
    let onnx_intra_threads = config
        .onnx
        .intra_threads
        .filter(|v| *v > 0)
        .unwrap_or(available_parallelism);
    let onnx_inter_threads = config
        .onnx
        .inter_threads
        .filter(|v| *v > 0)
        .unwrap_or(1);

    RuntimeSettings {
        embedding_model,
        embed_batch_size,
        parent_min_chars: parent_min,
        parent_max_chars: parent_max,
        child_min_chars: child_min,
        child_max_chars: child_max,
        child_split_window_chars: child_split_window,
        default_mode,
        default_top_k,
        default_context_window,
        hybrid_vector_weight,
        hybrid_bm25_weight,
        sqlite_journal_mode,
        sqlite_busy_timeout_ms,
        onnx_intra_threads,
        onnx_inter_threads,
    }
}

pub(crate) fn parse_embedding_model(input: &str) -> Option<EmbeddingModel> {
    let normalized = input
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_ascii_lowercase();
    match normalized.as_str() {
        "allminilml6v2q" => Some(EmbeddingModel::AllMiniLML6V2Q),
        "allminilml6v2" => Some(EmbeddingModel::AllMiniLML6V2),
        "allminilml12v2" => Some(EmbeddingModel::AllMiniLML12V2),
        "allminilml12v2q" => Some(EmbeddingModel::AllMiniLML12V2Q),
        "bgesmallenv15" => Some(EmbeddingModel::BGESmallENV15),
        "bgesmallenv15q" => Some(EmbeddingModel::BGESmallENV15Q),
        "mxbaiembedlargev1q" => Some(EmbeddingModel::MxbaiEmbedLargeV1Q),
        _ => None,
    }
}

// ============================================================================
