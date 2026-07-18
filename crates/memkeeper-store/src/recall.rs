//! `recall` helpers extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::path::Path;

use rusqlite::params;

use crate::{
    ensure_recall_events, limit_i64, now_timestamp, open_initialized_write, Error, RecallLogReport,
    RecallLogRequest, Result, MAX_METADATA_VALUE_CHARS, MAX_RECALL_EVENTS, MAX_SEARCH_QUERY_CHARS,
};

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
    ensure_recall_events(&transaction)?;
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
