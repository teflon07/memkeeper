//! `recall` helpers extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::path::Path;

use rusqlite::{params, Transaction};

use crate::{
    limit_i64, now_timestamp, open_initialized_write, Error, RecallLogReport, RecallLogRequest,
    Result, ShadowRerankBatch, MAX_METADATA_VALUE_CHARS, MAX_RECALL_EVENTS, MAX_SEARCH_QUERY_CHARS,
};

/// `recall_events` is engine-owned telemetry. It is created lazily (not part
/// of schema validation) so pre-existing stores need no migration, and it is
/// intentionally excluded from export/import: recall history is local
/// operational data, not memory content.
pub(crate) const RECALL_EVENTS_DDL: &str = "
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

/// `reranker_shadow_events` records production-vs-shadow reranker comparisons.
/// Like `recall_events` it is engine-owned telemetry: created lazily and
/// excluded from export/import (local operational data, not memory content).
pub(crate) const RERANKER_SHADOW_DDL: &str = "
CREATE TABLE IF NOT EXISTS reranker_shadow_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch INTEGER NOT NULL,
  ts TEXT NOT NULL,
  query TEXT,
  prod_model_id TEXT,
  shadow_model_id TEXT,
  memory_id TEXT NOT NULL,
  prod_rank INTEGER,
  prod_score REAL,
  shadow_rank INTEGER,
  shadow_score REAL
);
CREATE INDEX IF NOT EXISTS idx_reranker_shadow_batch ON reranker_shadow_events(batch);
CREATE INDEX IF NOT EXISTS idx_reranker_shadow_ts ON reranker_shadow_events(ts);
";

/// Record one pack's production-vs-shadow reranker comparison. Every row shares a
/// monotonic `batch` id so offline analysis can group a single pack's candidates.
/// Returns the number of rows written (0 when the batch is empty).
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible or `SQLite` rejects the
/// transaction. Callers on the retrieval hot path should treat failures as
/// non-fatal (shadow telemetry must never break production retrieval).
pub fn record_reranker_shadow(path: impl AsRef<Path>, batch: &ShadowRerankBatch) -> Result<usize> {
    if batch.rows.is_empty() {
        return Ok(0);
    }
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    transaction.execute_batch(RERANKER_SHADOW_DDL)?;
    let now = now_timestamp(&transaction)?;
    // Next batch id under the write lock: unique and monotonic without an external
    // dependency, so all rows from this call group cleanly for offline analysis.
    let batch_id: i64 = transaction.query_row(
        "SELECT COALESCE(MAX(batch), 0) + 1 FROM reranker_shadow_events",
        [],
        |row| row.get(0),
    )?;
    let mut recorded = 0_usize;
    {
        let mut insert = transaction.prepare_cached(
            "INSERT INTO reranker_shadow_events
               (batch, ts, query, prod_model_id, shadow_model_id, memory_id,
                prod_rank, prod_score, shadow_rank, shadow_score)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        )?;
        for row in &batch.rows {
            insert.execute(params![
                batch_id,
                &now,
                batch.query.as_deref(),
                batch.prod_model_id.as_deref(),
                batch.shadow_model_id.as_deref(),
                &row.memory_id,
                limit_i64(row.prod_rank)?,
                f64::from(row.prod_score),
                limit_i64(row.shadow_rank)?,
                f64::from(row.shadow_score),
            ])?;
            recorded += 1;
        }
    }
    transaction.commit()?;
    Ok(recorded)
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

/// Add optional telemetry columns to pre-existing `recall_events` tables.
/// Idempotent: duplicate-column errors mean a migration already ran.
pub(crate) fn ensure_recall_events_optional_columns(transaction: &Transaction<'_>) -> Result<()> {
    ensure_recall_events_column(transaction, "session_id", "TEXT")?;
    ensure_recall_events_column(transaction, "batch_id", "TEXT")?;
    ensure_recall_events_column(transaction, "latency_ms", "REAL")?;
    ensure_recall_events_column(transaction, "latency_source", "TEXT")
}

/// Record recall telemetry events and optionally touch `accessed_at` for
/// retrieved memories. Events may reference memories that no longer exist.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is
/// invalid (no events, too many events, unsupported kind, empty memory id,
/// oversized query/source), or `SQLite` rejects the transaction.
pub fn record_recall(
    path: impl AsRef<Path>,
    request: &RecallLogRequest,
) -> Result<RecallLogReport> {
    validate_recall_log_request(request)?;
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    transaction.execute_batch(RECALL_EVENTS_DDL)?;
    ensure_recall_events_optional_columns(&transaction)?;
    let now = now_timestamp(&transaction)?;
    let mut recorded = 0_usize;
    let mut touched = 0_usize;
    {
        let mut insert = transaction.prepare_cached(
            "INSERT INTO recall_events
               (memory_id, ts, kind, source, query, rank, score, session_id,
                batch_id, latency_ms, latency_source)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        )?;
        let mut touch =
            transaction.prepare_cached("UPDATE memories SET accessed_at = ?1 WHERE id = ?2")?;
        for event in &request.events {
            let rank = event.rank.map(limit_i64).transpose()?;
            insert.execute(params![
                &event.memory_id,
                &now,
                &event.kind,
                request.source.as_deref(),
                event.query.as_deref(),
                rank,
                event.score,
                request.session_id.as_deref(),
                request.batch_id.as_deref(),
                request.latency_ms,
                request.latency_source.as_deref(),
            ])?;
            recorded += 1;
            if request.touch_accessed && event.kind == "retrieved" {
                touched += touch.execute(params![&now, &event.memory_id])?;
            }
        }
    }
    transaction.commit()?;
    Ok(RecallLogReport { recorded, touched })
}

pub(crate) fn validate_recall_log_request(request: &RecallLogRequest) -> Result<()> {
    if request.events.is_empty() {
        return Err(Error::InvalidRequest {
            message: "recall-log requires at least one event".to_string(),
        });
    }
    if request.events.len() > MAX_RECALL_EVENTS {
        return Err(Error::InvalidRequest {
            message: format!("recall-log accepts at most {MAX_RECALL_EVENTS} events"),
        });
    }
    if request
        .source
        .as_deref()
        .is_some_and(|source| source.chars().count() > MAX_METADATA_VALUE_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: "recall-log source is too long".to_string(),
        });
    }
    if request
        .session_id
        .as_deref()
        .is_some_and(|s| s.chars().count() > MAX_METADATA_VALUE_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: "recall-log session_id exceeds maximum length".to_string(),
        });
    }
    if request
        .batch_id
        .as_deref()
        .is_some_and(|s| s.chars().count() > MAX_METADATA_VALUE_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: "recall-log batch_id exceeds maximum length".to_string(),
        });
    }
    if let Some(latency_ms) = request.latency_ms {
        if !latency_ms.is_finite() || latency_ms < 0.0 {
            return Err(Error::InvalidRequest {
                message: "recall-log latency_ms must be finite and non-negative".to_string(),
            });
        }
    }
    if request
        .latency_source
        .as_deref()
        .is_some_and(|s| s.chars().count() > MAX_METADATA_VALUE_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: "recall-log latency_source exceeds maximum length".to_string(),
        });
    }
    for event in &request.events {
        if event.memory_id.trim().is_empty() {
            return Err(Error::InvalidRequest {
                message: "recall event memory_id must not be empty".to_string(),
            });
        }
        if !matches!(event.kind.as_str(), "surfaced" | "retrieved") {
            return Err(Error::InvalidRequest {
                message: "recall event kind must be 'surfaced' or 'retrieved'".to_string(),
            });
        }
        if event
            .query
            .as_deref()
            .is_some_and(|query| query.chars().count() > MAX_SEARCH_QUERY_CHARS)
        {
            return Err(Error::InvalidRequest {
                message: "recall event query is too long".to_string(),
            });
        }
        if event.score.is_some_and(|score| !score.is_finite()) {
            return Err(Error::InvalidRequest {
                message: "recall event score must be finite".to_string(),
            });
        }
    }
    Ok(())
}
