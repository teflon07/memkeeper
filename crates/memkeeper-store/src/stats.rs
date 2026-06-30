//! `stats` helpers extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::fs;
use std::path::Path;

use memkeeper_core::status;
use rusqlite::{Connection, OptionalExtension};

use crate::{
    configure_connection, count, inspect_on_copy, journal_mode, open_initialized_read_fast,
    reject_sqlite_sidecar_symlinks, required_config_value, sqlite_version, table_exists,
    user_version, validate_initialized, validate_store_path, Error, HealthStats, IndexStats,
    Result, SpaceStats, Stats,
};

/// Return deterministic statistics for an initialized memkeeper store.
///
/// # Errors
///
/// Returns an error when the store file is missing, is not initialized, has an
/// unsupported schema version, or cannot be queried.
pub fn store_stats(path: impl AsRef<Path>, include_indexes: bool) -> Result<Stats> {
    let path = path.as_ref();
    let connection = open_initialized_read_fast(path)?;
    store_stats_on_connection(path, &connection, include_indexes, false)
}
/// Return store statistics including the opt-in memory-governance `health`
/// rollup (lifecycle/provenance hygiene). Separate entry point so existing
/// `store_stats` callers keep the cheaper default path.
///
/// # Errors
///
/// Returns an error when the store file is missing, is not initialized, has an
/// unsupported schema version, or cannot be queried.
pub fn store_stats_with_health(path: impl AsRef<Path>, include_indexes: bool) -> Result<Stats> {
    let path = path.as_ref();
    let connection = open_initialized_read_fast(path)?;
    store_stats_on_connection(path, &connection, include_indexes, true)
}
/// Return deterministic statistics for diagnostics without creating `SQLite` sidecars near the store.
///
/// # Errors
///
/// Returns an error when the store file is missing, is not initialized, has an
/// unsupported schema version, or cannot be inspected from a private copy.
pub fn inspect_store_stats(path: impl AsRef<Path>, include_indexes: bool) -> Result<Stats> {
    let path = path.as_ref();
    validate_store_path(path)?;
    if !path.exists() {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }
    reject_sqlite_sidecar_symlinks(path)?;
    inspect_on_copy(path, |connection| {
        configure_connection(connection)?;
        validate_initialized(path, connection)?;
        store_stats_on_connection(path, connection, include_indexes, false)
    })
}
/// Returns the `started_at` timestamp of the most recent successful dream (synthesis) run,
/// or `None` when no succeeded run exists.
///
/// # Errors
///
/// Returns an error when the store file is missing, is not initialized, has an
/// unsupported schema version, or cannot be queried.
pub fn last_synthesis_run(path: impl AsRef<Path>) -> Result<Option<String>> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    let val: Option<String> = connection
        .query_row(
            "SELECT started_at FROM dream_runs WHERE status = 'succeeded' \
             ORDER BY started_at DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    Ok(val)
}
pub(crate) fn store_stats_on_connection(
    path: &Path,
    connection: &Connection,
    include_indexes: bool,
    include_health: bool,
) -> Result<Stats> {
    let metadata = fs::metadata(path)?;
    let indexes = if include_indexes {
        Some(IndexStats {
            fts_memory_rows: count(connection, "SELECT COUNT(*) FROM memory_fts")?,
            fts_source_episode_rows: count(connection, "SELECT COUNT(*) FROM source_episode_fts")?,
            document_duplicate_clusters: count(
                connection,
                "SELECT COUNT(*) FROM (
                    SELECT 1 FROM source_episodes
                    WHERE content_sha256 IS NOT NULL
                    GROUP BY space_name, content_sha256
                    HAVING COUNT(*) > 1
                 )",
            )?,
            pending_jobs: count(
                connection,
                "SELECT COUNT(*) FROM processing_jobs WHERE status IN ('queued','running')",
            )?,
        })
    } else {
        None
    };
    let health = if include_health {
        Some(health_stats(connection)?)
    } else {
        None
    };

    Ok(Stats {
        path: path.to_path_buf(),
        database_bytes: metadata.len(),
        schema_version: user_version(connection)?,
        protocol_version: required_config_value(path, connection, "protocol_version")?,
        sqlite_version: sqlite_version(connection)?,
        journal_mode: journal_mode(connection)?,
        space_count: count(connection, "SELECT COUNT(*) FROM spaces")?,
        silo_count: count(connection, "SELECT COUNT(*) FROM silos")?,
        memory_count: count(connection, "SELECT COUNT(*) FROM memories")?,
        active_count: count(
            connection,
            "SELECT COUNT(*) FROM memories WHERE status = 'active'",
        )?,
        source_episode_count: count(connection, "SELECT COUNT(*) FROM source_episodes")?,
        spaces: space_stats(connection)?,
        indexes,
        health,
    })
}
/// Compute the opt-in memory-governance health rollup. All queries are indexed
/// aggregates over the local store.
pub(crate) fn health_stats(connection: &Connection) -> Result<HealthStats> {
    let (mut active, mut superseded, mut conflicted, mut tombstoned, mut expired) =
        (0i64, 0i64, 0i64, 0i64, 0i64);
    let mut statement =
        connection.prepare("SELECT status, COUNT(*) FROM memories GROUP BY status")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    for row in rows {
        let (status_value, total) = row?;
        match status_value.as_str() {
            status::ACTIVE => active = total,
            status::SUPERSEDED => superseded = total,
            status::CONFLICTED => conflicted = total,
            status::TOMBSTONED => tombstoned = total,
            status::EXPIRED => expired = total,
            _ => {}
        }
    }

    let last_embedding_at: Option<String> = connection.query_row(
        "SELECT MAX(updated_at) FROM embeddings WHERE status = 'ready'",
        [],
        |row| row.get::<_, Option<String>>(0),
    )?;

    // The candidate review queue is a lazily-created table; count by status when
    // it exists, otherwise report zeros (the queue has simply never been used).
    let (candidates_pending, candidates_approved, candidates_rejected) =
        if table_exists(connection, "memory_candidates")? {
            (
                count(
                    connection,
                    "SELECT COUNT(*) FROM memory_candidates WHERE status = 'pending'",
                )?,
                count(
                    connection,
                    "SELECT COUNT(*) FROM memory_candidates WHERE status = 'approved'",
                )?,
                count(
                    connection,
                    "SELECT COUNT(*) FROM memory_candidates WHERE status = 'rejected'",
                )?,
            )
        } else {
            (0, 0, 0)
        };

    Ok(HealthStats {
        active,
        superseded,
        conflicted,
        tombstoned,
        expired,
        active_without_keys: count(
            connection,
            "SELECT COUNT(*) FROM memories \
             WHERE status = 'active' AND (entity_key IS NULL OR claim_key IS NULL)",
        )?,
        duplicate_key_groups: count(
            connection,
            "SELECT COUNT(*) FROM ( \
               SELECT 1 FROM memories \
                WHERE status = 'active' AND entity_key IS NOT NULL AND claim_key IS NOT NULL \
                GROUP BY space_name, entity_key, claim_key HAVING COUNT(*) > 1 \
             )",
        )?,
        short_term_active: count(
            connection,
            "SELECT COUNT(*) FROM memories WHERE status = 'active' AND silo_name = 'short-term'",
        )?,
        active_past_valid_to: count(
            connection,
            "SELECT COUNT(*) FROM memories \
             WHERE status = 'active' AND valid_to IS NOT NULL \
               AND julianday(valid_to) < julianday('now')",
        )?,
        active_without_embedding: count(
            connection,
            "SELECT COUNT(*) FROM memories m \
             WHERE m.status = 'active' \
               AND NOT EXISTS (SELECT 1 FROM embeddings e \
                                WHERE e.memory_id = m.id AND e.status = 'ready')",
        )?,
        last_embedding_at,
        candidates_pending,
        candidates_approved,
        candidates_rejected,
    })
}
pub(crate) fn space_names(connection: &Connection) -> Result<Vec<String>> {
    let mut statement = connection.prepare("SELECT name FROM spaces ORDER BY name")?;
    let rows = statement.query_map([], |row| row.get(0))?;
    let mut names = Vec::new();
    for row in rows {
        names.push(row?);
    }
    Ok(names)
}
pub(crate) fn space_stats(connection: &Connection) -> Result<Vec<SpaceStats>> {
    let mut statement = connection.prepare(
        "SELECT s.name,
                COUNT(m.id) AS memory_count,
                COALESCE(SUM(CASE WHEN m.status = 'active' THEN 1 ELSE 0 END), 0) AS active_count
           FROM spaces s
           LEFT JOIN memories m ON m.space_name = s.name
          GROUP BY s.name
          ORDER BY s.name",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(SpaceStats {
            name: row.get(0)?,
            memory_count: row.get(1)?,
            active_count: row.get(2)?,
        })
    })?;
    let mut stats = Vec::new();
    for row in rows {
        stats.push(row?);
    }
    Ok(stats)
}
