use crate::*;

pub(crate) fn configure_sqlite_connection(conn: &Connection) -> Result<(), Box<dyn Error>> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.busy_timeout(Duration::from_millis(SQLITE_BUSY_TIMEOUT_MS))?;
    Ok(())
}

pub(crate) fn init_db(db_path: &Path) -> Result<Connection, Box<dyn Error>> {
    let conn = Connection::open(db_path)?;
    configure_sqlite_connection(&conn)?;
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS libraries (
            library_name TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_refreshed_at TEXT NOT NULL,
            content_size_chars INTEGER NOT NULL DEFAULT 0,
            page_count INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            embedded_chunk_count INTEGER NOT NULL DEFAULT 0,
            empty_page_count INTEGER NOT NULL DEFAULT 0,
            min_chunks_per_page INTEGER NOT NULL DEFAULT 0,
            max_chunks_per_page INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS library_aliases (
            alias TEXT PRIMARY KEY,
            library_name TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS parents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            library_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_page_order INTEGER NOT NULL,
            parent_index_in_page INTEGER NOT NULL,
            global_parent_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER NOT NULL,
            library_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_page_order INTEGER NOT NULL,
            parent_index_in_page INTEGER NOT NULL,
            child_index_in_parent INTEGER NOT NULL,
            global_chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            library_name UNINDEXED,
            content
        );
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            library_name TEXT NOT NULL,
            page_order INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            content TEXT NOT NULL,
            content_size_chars INTEGER NOT NULL,
            crawled_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS library_groups (
            group_name TEXT NOT NULL,
            member_library_name TEXT NOT NULL,
            member_order INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (group_name, member_library_name)
        );
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            library_name TEXT NOT NULL,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            message TEXT
        );
        CREATE TABLE IF NOT EXISTS library_texts (
            library_name TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_parents_library_name ON parents(library_name);
        CREATE INDEX IF NOT EXISTS idx_parents_library_page ON parents(library_name, source_url, parent_index_in_page);
        CREATE INDEX IF NOT EXISTS idx_chunks_library_name ON chunks(library_name);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_library_page ON chunks(library_name, source_url, parent_index_in_page, child_index_in_parent);
        CREATE INDEX IF NOT EXISTS idx_pages_library_name ON pages(library_name);
        CREATE INDEX IF NOT EXISTS idx_pages_library_order ON pages(library_name, page_order);
        CREATE INDEX IF NOT EXISTS idx_library_groups_name ON library_groups(group_name);
        ",
    )?;
    run_db_migrations(&conn)?;
    Ok(conn)
}

pub(crate) fn run_db_migrations(conn: &Connection) -> Result<(), Box<dyn Error>> {
    let mut has_page_size_bytes = false;
    let mut has_page_size_chars = false;
    let mut library_columns = HashSet::new();

    let mut stmt = conn.prepare("PRAGMA table_info(pages)")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        let col = row?;
        if col == "content_size_bytes" {
            has_page_size_bytes = true;
        }
        if col == "content_size_chars" {
            has_page_size_chars = true;
        }
    }
    if has_page_size_bytes && !has_page_size_chars {
        conn.execute_batch(
            "ALTER TABLE pages RENAME COLUMN content_size_bytes TO content_size_chars;",
        )?;
    }

    let mut stmt = conn.prepare("PRAGMA table_info(libraries)")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for row in rows {
        library_columns.insert(row?);
    }
    if !library_columns.contains("content_size_chars") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN content_size_chars INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("page_count") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN page_count INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("chunk_count") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("embedded_chunk_count") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN embedded_chunk_count INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("empty_page_count") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN empty_page_count INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("min_chunks_per_page") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN min_chunks_per_page INTEGER NOT NULL DEFAULT 0;",
        )?;
    }
    if !library_columns.contains("max_chunks_per_page") {
        conn.execute_batch(
            "ALTER TABLE libraries ADD COLUMN max_chunks_per_page INTEGER NOT NULL DEFAULT 0;",
        )?;
    }

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS parents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            library_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_page_order INTEGER NOT NULL,
            parent_index_in_page INTEGER NOT NULL,
            global_parent_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER NOT NULL,
            library_name TEXT NOT NULL,
            source_url TEXT NOT NULL,
            source_page_order INTEGER NOT NULL,
            parent_index_in_page INTEGER NOT NULL,
            child_index_in_parent INTEGER NOT NULL,
            global_chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            token_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            library_name UNINDEXED,
            content
        );
        CREATE INDEX IF NOT EXISTS idx_parents_library_name ON parents(library_name);
        CREATE INDEX IF NOT EXISTS idx_parents_library_page ON parents(library_name, source_url, parent_index_in_page);
        CREATE INDEX IF NOT EXISTS idx_chunks_library_name ON chunks(library_name);
        CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id);
        CREATE INDEX IF NOT EXISTS idx_chunks_library_page ON chunks(library_name, source_url, parent_index_in_page, child_index_in_parent);
        ",
    )?;

    Ok(())
}

pub(crate) fn start_job(conn: &Connection, library_name: &str, job_type: &str) -> Result<i64, Box<dyn Error>> {
    conn.execute(
        "INSERT INTO jobs (library_name, job_type, status, started_at) VALUES (?1, ?2, 'running', ?3)",
        params![library_name, job_type, now_epoch()],
    )?;
    Ok(conn.last_insert_rowid())
}

pub(crate) fn finish_job(
    conn: &Connection,
    job_id: i64,
    status: &str,
    message: &str,
) -> Result<(), Box<dyn Error>> {
    conn.execute(
        "UPDATE jobs SET status = ?1, ended_at = ?2, message = ?3 WHERE id = ?4",
        params![status, now_epoch(), message, job_id],
    )?;
    Ok(())
}

