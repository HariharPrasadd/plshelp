use crate::*;

pub(crate) fn configure_onnx_runtime_env() {
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);

    // These must be set before ONNX Runtime initializes.
    unsafe {
        env::set_var("OMP_NUM_THREADS", num_cpus.to_string());
        env::set_var("ORT_NUM_INTRA_THREADS", num_cpus.to_string());
        env::set_var("ORT_NUM_INTER_THREADS", "1");
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

    let config = load_config_file(&default_paths.config_file)?;
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
    let _ = RUNTIME_PATHS.set(runtime);
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
    };
    let serialized = toml::to_string_pretty(&config)?;
    fs::write(&defaults.config_file, serialized)?;
    Ok(())
}

pub(crate) fn load_config_file(path: &Path) -> Result<AppConfigFile, Box<dyn Error>> {
    let raw = fs::read_to_string(path)?;
    let config = toml::from_str::<AppConfigFile>(&raw)?;
    Ok(config)
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

// ============================================================================
