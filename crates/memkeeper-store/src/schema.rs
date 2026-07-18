//! Memkeeper schema definitions, migrations, validation, and lazy sidecars.

use std::path::Path;

use memkeeper_core::DEFAULT_SPACE;
use rusqlite::{Connection, OptionalExtension, Transaction};

#[cfg(feature = "semantic")]
use crate::MAX_SEMANTIC_EMBEDDING_DIMS;
use crate::{count, user_version, Error, Result};

/// Current `SQLite` schema version expected by the store/migration layer.
pub const SCHEMA_VERSION: i32 = 6;

/// `SQLite` schema draft used for v0.1 store initialization.
pub(crate) const SCHEMA_SQL: &str = include_str!("../../../schema-v0.1.sql");

#[cfg(feature = "semantic")]
const SCHEMA_UPGRADE_SQL: &str = "
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (2, 'schema-v0.2', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (3, 'schema-v0.3-semantic-1536', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (4, 'schema-v0.4-semantic-1024', CURRENT_TIMESTAMP, NULL);
DROP TABLE IF EXISTS memory_vec_1536;
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (5, 'schema-v0.5-token-embeddings', CURRENT_TIMESTAMP, NULL);
CREATE TABLE IF NOT EXISTS memory_token_embeddings (
  memory_id TEXT PRIMARY KEY,
  embedding_model TEXT NOT NULL,
  dims INTEGER NOT NULL,
  n_tokens INTEGER NOT NULL,
  vector_blob BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
UPDATE config_kv SET value = '5', updated_at = CURRENT_TIMESTAMP
WHERE key = 'schema_version' AND CAST(value AS INTEGER) < 5;
PRAGMA user_version = 5;
";

#[cfg(feature = "semantic")]
pub(crate) const SEMANTIC_TABLE_SQL: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec_1024 USING vec0(
  memory_id TEXT NOT NULL,
  embedding FLOAT[1024]
);
";

#[cfg(not(feature = "semantic"))]
const SCHEMA_UPGRADE_SQL: &str = "
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (2, 'schema-v0.2', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (3, 'schema-v0.3', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (4, 'schema-v0.4', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (5, 'schema-v0.5-token-embeddings', CURRENT_TIMESTAMP, NULL);
CREATE TABLE IF NOT EXISTS memory_token_embeddings (
  memory_id TEXT PRIMARY KEY,
  embedding_model TEXT NOT NULL,
  dims INTEGER NOT NULL,
  n_tokens INTEGER NOT NULL,
  vector_blob BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
UPDATE config_kv SET value = '5', updated_at = CURRENT_TIMESTAMP
WHERE key = 'schema_version' AND CAST(value AS INTEGER) < 5;
PRAGMA user_version = 5;
";

const SCHEMA_V6_MIGRATION_SQL: &str = "
CREATE TABLE IF NOT EXISTS memory_representations (
  version_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('contextual-card-v1')),
  text TEXT NOT NULL,
  text_sha256 TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY (version_id) REFERENCES memory_versions(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE memory_fts_v6 USING fts5(
  memory_id UNINDEXED, version_id UNINDEXED, space_name UNINDEXED,
  silo_name UNINDEXED, status UNINDEXED, kind UNINDEXED,
  content, retrieval_text, tags, source_text, metadata_text,
  tokenize = 'unicode61 remove_diacritics 2'
);
INSERT INTO memory_fts_v6
SELECT m.id, v.id, m.space_name, m.silo_name, m.status, m.kind,
       v.content, v.summary,
       COALESCE((
         SELECT group_concat(tag, ' ')
         FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)
       ), ''),
       (SELECT old.source_text FROM memory_fts old
        WHERE old.memory_id = m.id AND old.version_id = v.id
        ORDER BY old.rowid LIMIT 1),
       COALESCE((SELECT old.metadata_text FROM memory_fts old
                 WHERE old.memory_id = m.id AND old.version_id = v.id
                 ORDER BY old.rowid LIMIT 1), '')
FROM memories m
JOIN memory_versions v ON v.id = m.active_version_id
ORDER BY COALESCE((SELECT old.rowid FROM memory_fts old
                   WHERE old.memory_id = m.id AND old.version_id = v.id
                   ORDER BY old.rowid LIMIT 1), 9223372036854775807), m.id;
DROP TABLE memory_fts;
CREATE VIRTUAL TABLE memory_fts USING fts5(
  memory_id UNINDEXED, version_id UNINDEXED, space_name UNINDEXED,
  silo_name UNINDEXED, status UNINDEXED, kind UNINDEXED,
  content, retrieval_text, tags, source_text, metadata_text,
  tokenize = 'unicode61 remove_diacritics 2'
);
INSERT INTO memory_fts SELECT * FROM memory_fts_v6;
DROP TABLE memory_fts_v6;

CREATE VIRTUAL TABLE memory_fts_public_v6 USING fts5(
  memory_id UNINDEXED, version_id UNINDEXED, space_name UNINDEXED,
  silo_name UNINDEXED, status UNINDEXED, kind UNINDEXED,
  content, retrieval_text, tags, metadata_text,
  tokenize = 'unicode61 remove_diacritics 2'
);
INSERT INTO memory_fts_public_v6
SELECT m.id, v.id, m.space_name, m.silo_name, m.status, m.kind,
       v.content, v.summary,
       COALESCE((
         SELECT group_concat(tag, ' ')
         FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)
       ), ''),
       COALESCE((SELECT old.metadata_text FROM memory_fts_public old
                 WHERE old.memory_id = m.id AND old.version_id = v.id
                 ORDER BY old.rowid LIMIT 1), '')
FROM memories m
JOIN memory_versions v ON v.id = m.active_version_id
ORDER BY COALESCE((SELECT old.rowid FROM memory_fts_public old
                   WHERE old.memory_id = m.id AND old.version_id = v.id
                   ORDER BY old.rowid LIMIT 1), 9223372036854775807), m.id;
DROP TABLE memory_fts_public;
CREATE VIRTUAL TABLE memory_fts_public USING fts5(
  memory_id UNINDEXED, version_id UNINDEXED, space_name UNINDEXED,
  silo_name UNINDEXED, status UNINDEXED, kind UNINDEXED,
  content, retrieval_text, tags, metadata_text,
  tokenize = 'unicode61 remove_diacritics 2'
);
INSERT INTO memory_fts_public SELECT * FROM memory_fts_public_v6;
DROP TABLE memory_fts_public_v6;
";

const SCHEMA_V6_FINALIZE_SQL: &str = "
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (6, 'schema-v0.6-memory-representations', CURRENT_TIMESTAMP, NULL);
UPDATE config_kv SET value = '6', updated_at = CURRENT_TIMESTAMP
WHERE key = 'schema_version' AND value != '6';
PRAGMA user_version = 6;
";

/// Required table names that should exist in the v0.1 schema.
pub const REQUIRED_TABLES: &[&str] = &[
    "schema_migrations",
    "config_kv",
    "spaces",
    "silos",
    "source_episodes",
    "memories",
    "memory_versions",
    "memory_representations",
    "memory_events",
    "memory_tags",
    "memory_links",
    "conflicts",
    "entities",
    "entity_aliases",
    "relationships",
    "processing_jobs",
    "embeddings",
    "dream_runs",
    "probe_runs",
];

/// Required FTS virtual table names.
pub const REQUIRED_FTS_TABLES: &[&str] = &["memory_fts", "memory_fts_public", "source_episode_fts"];

const CANDIDATES_DDL: &str = "
CREATE TABLE IF NOT EXISTS memory_candidates (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'pending',
  space TEXT,
  silo TEXT,
  scope TEXT,
  project TEXT,
  kind TEXT,
  content TEXT NOT NULL,
  summary TEXT,
  rationale TEXT,
  tags_json TEXT,
  entity_key TEXT,
  claim_key TEXT,
  confidence REAL NOT NULL DEFAULT 1.0,
  source_type TEXT NOT NULL DEFAULT 'assistant-inference',
  source_json TEXT,
  sensitivity TEXT NOT NULL DEFAULT 'normal',
  supersedes_json TEXT,
  created_at TEXT NOT NULL,
  decided_at TEXT,
  decided_reason TEXT,
  resulting_memory_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_status ON memory_candidates(status);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_created ON memory_candidates(created_at);
";

/// `recall_events` is engine-owned telemetry. It is created lazily and excluded
/// from export/import because recall history is local operational data.
const RECALL_EVENTS_DDL: &str = "
CREATE TABLE IF NOT EXISTS recall_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  source TEXT,
  query TEXT,
  rank INTEGER,
  score REAL,
  session_id TEXT,
  batch_id TEXT,
  latency_ms REAL,
  latency_source TEXT
);
CREATE INDEX IF NOT EXISTS idx_recall_events_memory ON recall_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_recall_events_ts ON recall_events(ts);
";

/// `source_episode_recall_events` stores local document retrieval telemetry.
const SOURCE_EPISODE_RECALL_DDL: &str = "
CREATE TABLE IF NOT EXISTS source_episode_recall_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_episode_id TEXT NOT NULL,
  space_name TEXT NOT NULL,
  ts TEXT NOT NULL,
  query TEXT,
  query_sha256 TEXT,
  rank INTEGER,
  score REAL,
  match_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_se_recall_src ON source_episode_recall_events(source_episode_id);
CREATE INDEX IF NOT EXISTS idx_se_recall_ts ON source_episode_recall_events(ts);
";

/// Return true if the embedded schema text contains the objects expected by the scaffold.
#[must_use]
pub fn schema_mentions_required_objects() -> bool {
    let base_schema_mentions_required_objects =
        REQUIRED_TABLES.iter().all(|name| SCHEMA_SQL.contains(name))
            && REQUIRED_FTS_TABLES
                .iter()
                .all(|name| SCHEMA_SQL.contains(name))
            && SCHEMA_SQL.contains(DEFAULT_SPACE)
            && SCHEMA_SQL.contains("PRAGMA user_version = 1");
    #[cfg(feature = "semantic")]
    {
        base_schema_mentions_required_objects
            && SEMANTIC_TABLE_SQL.contains("memory_vec_1024")
            && SCHEMA_V6_FINALIZE_SQL.contains(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))
    }
    #[cfg(not(feature = "semantic"))]
    {
        base_schema_mentions_required_objects
            && SCHEMA_V6_FINALIZE_SQL.contains(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))
    }
}

pub(crate) fn apply_schema(connection: &Connection) -> Result<()> {
    let previous_schema_version = user_version(connection)?;
    connection.execute_batch("BEGIN IMMEDIATE;")?;
    let schema_sql = transactional_schema_sql();
    let result = connection
        .execute_batch(&schema_sql)
        .and_then(|()| apply_feature_schema(connection, previous_schema_version))
        .and_then(|()| connection.execute_batch("COMMIT;"));

    if let Err(error) = result {
        let _ = connection.execute_batch("ROLLBACK;");
        return Err(Error::Database(error));
    }

    Ok(())
}

pub(crate) fn normalize_imported_schema_metadata(
    transaction: &Transaction<'_>,
    source_schema_version: i32,
) -> Result<()> {
    if source_schema_version == 5 {
        transaction.execute(
            "INSERT OR IGNORE INTO schema_migrations
             (version, name, applied_at, checksum_sha256)
             VALUES (6, 'schema-v0.6-memory-representations', CURRENT_TIMESTAMP, NULL)",
            [],
        )?;
        transaction.execute(
            "INSERT INTO config_kv (key, value, updated_at)
             VALUES ('schema_version', '6', CURRENT_TIMESTAMP)
             ON CONFLICT(key) DO UPDATE SET
               value = excluded.value,
               updated_at = excluded.updated_at",
            [],
        )?;
    }
    Ok(())
}

fn apply_feature_schema(
    connection: &Connection,
    previous_schema_version: i32,
) -> rusqlite::Result<()> {
    connection.execute_batch(SCHEMA_UPGRADE_SQL)?;
    if previous_schema_version < SCHEMA_VERSION {
        connection.execute_batch(SCHEMA_V6_MIGRATION_SQL)?;
    }
    connection.execute_batch(SCHEMA_V6_FINALIZE_SQL)
}

fn transactional_schema_sql() -> String {
    let mut sql = String::with_capacity(SCHEMA_SQL.len());
    for line in SCHEMA_SQL.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("PRAGMA foreign_keys")
            || trimmed.starts_with("PRAGMA journal_mode")
            || trimmed.starts_with("PRAGMA busy_timeout")
        {
            continue;
        }
        sql.push_str(line);
        sql.push('\n');
    }
    sql
}

pub(crate) fn validate_initialized(path: &Path, connection: &Connection) -> Result<()> {
    let actual = user_version(connection)?;
    if actual == 0 {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }
    if actual != SCHEMA_VERSION {
        return Err(Error::SchemaMismatch {
            expected: SCHEMA_VERSION,
            actual,
        });
    }

    for table in REQUIRED_TABLES {
        if !table_exists(connection, table)? {
            return Err(Error::NotInitialized {
                path: path.to_path_buf(),
            });
        }
    }
    for table in REQUIRED_FTS_TABLES {
        if !table_exists(connection, table)? {
            return Err(Error::NotInitialized {
                path: path.to_path_buf(),
            });
        }
    }
    let applied = count(
        connection,
        &format!("SELECT COUNT(*) FROM schema_migrations WHERE version = {SCHEMA_VERSION}"),
    )?;
    if applied == 0 {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    let schema_config = required_config_value(path, connection, "schema_version")?;
    let expected_schema = SCHEMA_VERSION.to_string();
    if schema_config != expected_schema {
        let actual = schema_config.parse().unwrap_or_default();
        return Err(Error::SchemaMismatch {
            expected: SCHEMA_VERSION,
            actual,
        });
    }

    let protocol = required_config_value(path, connection, "protocol_version")?;
    if protocol != "memkeeper.v0.1" {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    let default_space = required_config_value(path, connection, "default_space")?;
    if default_space != DEFAULT_SPACE {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    let seeded_space = count(
        connection,
        "SELECT COUNT(*) FROM spaces WHERE name = 'workspace-memory'",
    )?;
    if seeded_space == 0 {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    Ok(())
}

pub(crate) fn older_schema_is_memkeeper(connection: &Connection, actual: i32) -> Result<bool> {
    if actual <= 0 || actual >= SCHEMA_VERSION {
        return Ok(false);
    }
    for table in REQUIRED_TABLES {
        if actual < 6 && *table == "memory_representations" {
            continue;
        }
        if !table_exists(connection, table)? {
            return Ok(false);
        }
    }
    for table in REQUIRED_FTS_TABLES {
        if !table_exists(connection, table)? {
            return Ok(false);
        }
    }
    let migration_count = count(
        connection,
        &format!("SELECT COUNT(*) FROM schema_migrations WHERE version = {actual}"),
    )?;
    if migration_count == 0 {
        return Ok(false);
    }
    let schema_config = connection
        .query_row(
            "SELECT value FROM config_kv WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let protocol = connection
        .query_row(
            "SELECT value FROM config_kv WHERE key = 'protocol_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let default_space = connection
        .query_row(
            "SELECT value FROM config_kv WHERE key = 'default_space'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(schema_config.as_deref() == Some(&actual.to_string())
        && protocol.as_deref() == Some("memkeeper.v0.1")
        && default_space.as_deref() == Some(DEFAULT_SPACE))
}

pub(crate) fn table_exists(connection: &Connection, name: &str) -> Result<bool> {
    let mut statement = connection.prepare_cached(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?1)",
    )?;
    let exists = statement.query_row([name], |row| row.get::<_, i64>(0))?;
    Ok(exists == 1)
}

pub(crate) fn required_config_value(
    path: &Path,
    connection: &Connection,
    key: &str,
) -> Result<String> {
    let value = connection
        .query_row("SELECT value FROM config_kv WHERE key = ?1", [key], |row| {
            row.get(0)
        })
        .optional()?;
    value.ok_or_else(|| Error::NotInitialized {
        path: path.to_path_buf(),
    })
}

pub(crate) fn ensure_memory_candidates(connection: &Connection) -> Result<()> {
    connection.execute_batch(CANDIDATES_DDL)?;
    Ok(())
}

pub(crate) fn ensure_recall_events(transaction: &Transaction<'_>) -> Result<()> {
    transaction.execute_batch(RECALL_EVENTS_DDL)?;
    ensure_recall_events_optional_columns(transaction)
}

pub(crate) fn ensure_source_episode_recall_events(transaction: &Transaction<'_>) -> Result<()> {
    transaction.execute_batch(SOURCE_EPISODE_RECALL_DDL)?;
    Ok(())
}

fn ensure_recall_events_optional_columns(transaction: &Transaction<'_>) -> Result<()> {
    ensure_recall_events_column(transaction, "session_id", "TEXT")?;
    ensure_recall_events_column(transaction, "batch_id", "TEXT")?;
    ensure_recall_events_column(transaction, "latency_ms", "REAL")?;
    ensure_recall_events_column(transaction, "latency_source", "TEXT")
}

fn ensure_recall_events_column(
    transaction: &Transaction<'_>,
    name: &str,
    sql_type: &str,
) -> Result<()> {
    let sql = format!("ALTER TABLE recall_events ADD COLUMN {name} {sql_type}");
    match transaction.execute(&sql, []) {
        Ok(_) => Ok(()),
        Err(rusqlite::Error::SqliteFailure(_, Some(msg)))
            if msg.contains("duplicate column name") =>
        {
            Ok(())
        }
        Err(error) => Err(Error::Database(error)),
    }
}

#[cfg(feature = "semantic")]
pub(crate) fn semantic_table_for_dims(dims: usize) -> Result<String> {
    if !(1..=MAX_SEMANTIC_EMBEDDING_DIMS).contains(&dims) {
        return Err(Error::InvalidRequest {
            message: format!(
                "embedding dimension {dims} is not supported (expected 1..={MAX_SEMANTIC_EMBEDDING_DIMS})"
            ),
        });
    }
    Ok(format!("memory_vec_{dims}"))
}

#[cfg(feature = "semantic")]
pub(crate) fn ensure_memory_vector_table(
    connection: &Connection,
    dimensions: usize,
) -> Result<String> {
    let table = semantic_table_for_dims(dimensions)?;
    if !table_exists(connection, &table)? {
        connection.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(\n  memory_id TEXT NOT NULL,\n  embedding FLOAT[{dimensions}]\n);"
        ))?;
    }
    Ok(table)
}

#[cfg(feature = "semantic")]
pub(crate) fn source_episode_vector_table(dimensions: usize) -> String {
    format!("source_episode_vec_{dimensions}")
}

#[cfg(feature = "semantic")]
pub(crate) fn ensure_source_episode_vector_table(
    connection: &Connection,
    dimensions: usize,
) -> Result<String> {
    let table = source_episode_vector_table(dimensions);
    if !table_exists(connection, &table)? {
        connection.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(\n  source_episode_id TEXT NOT NULL,\n  embedding FLOAT[{dimensions}]\n);"
        ))?;
    }
    Ok(table)
}

#[cfg(feature = "semantic")]
pub(crate) fn drop_all_vector_tables(transaction: &Transaction<'_>) -> Result<()> {
    let tables: Vec<String> = {
        let mut statement = transaction.prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'memory_vec_%'",
        )?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    for table in tables {
        // Table name comes from sqlite_master and is therefore trusted.
        transaction.execute(&format!("DROP TABLE IF EXISTS {table}"), [])?;
    }
    Ok(())
}
