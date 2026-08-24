//! Candidate memory review queue extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::path::Path;

use rusqlite::{params, OptionalExtension, Transaction};

use memkeeper_core::{scope, DEFAULT_SPACE};

use crate::{
    collect_rows, default_silo, ensure_memory_candidates, ensure_silo_exists, ensure_space_exists,
    json_string_for_store, limit_i64, next_id, normalized_tags, now_timestamp, open_initialized_write,
    remember_memory_tx, string_array_json, validate_memory_link_ids, validate_optional_metadata_value,
    validate_remember_request, reject_all_spaces_sentinel, JsonValidator, CANDIDATE_COLUMNS, CANDIDATE_SENSITIVITIES,
    CANDIDATE_SOURCE_TYPES, CANDIDATE_STATUSES, CANDIDATE_STATUS_APPROVED, CANDIDATE_STATUS_PENDING,
    CANDIDATE_STATUS_QUARANTINED, CANDIDATE_STATUS_REJECTED, CANDIDATE_SOURCE_CAPTURE,
    DEFAULT_CANDIDATE_SENSITIVITY, DEFAULT_CANDIDATE_SOURCE_TYPE,
    CandidateApproveReport, CandidateApproveRequest, CandidateListReport, CandidateListRequest,
    CandidateQuarantineReport, CandidateQuarantineRequest, CandidateRecord, CandidateRejectReport,
    CandidateRejectRequest, CandidateSubmitReport, CandidateSubmitRequest, Error, RememberRequest,
    Result, MAX_CONTENT_CHARS, MAX_METADATA_VALUE_CHARS, MAX_SOURCE_REF_JSON_CHARS, MAX_TAGS,
    DEFAULT_CANDIDATE_LIST_LIMIT, MAX_SUMMARY_CHARS, REMEMBER_MODE_AUTO,
};

pub(crate) fn capture_require_adjudication() -> bool {
    std::env::var("MEMKEEPER_CAPTURE_REQUIRE_ADJUDICATION")
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Promotion posture for a capture candidate at the `approve` boundary.
#[derive(Debug, PartialEq, Eq)]
pub enum AdjudicationGuard {
    /// Verdict present (or not a capture candidate) — promote normally.
    Ok,
    /// Unadjudicated but no hard requirement — promote and warn loudly.
    Degraded,
    /// Unadjudicated and adjudication is required — refuse promotion (fail closed).
    Refuse,
}

/// Pure decision mirroring the serve `guard`: may a capture candidate be promoted
/// given whether it carries an adjudication verdict and whether the deployment
/// requires adjudication? Kept pure (no I/O) so it is unit-testable.
#[must_use]
pub fn adjudication_guard(has_verdict: bool, require: bool) -> AdjudicationGuard {
    if has_verdict {
        AdjudicationGuard::Ok
    } else if require {
        AdjudicationGuard::Refuse
    } else {
        AdjudicationGuard::Degraded
    }
}

/// Whether a candidate came from the capture write-path (subject to the gate).
/// Non-capture candidates (assistant-inference, docs, import, …) are unaffected.
fn candidate_is_capture(candidate: &CandidateRecord) -> bool {
    candidate.source_type == CANDIDATE_SOURCE_CAPTURE
}

/// Whether a capture candidate carries a recorded adjudication verdict. The
/// adjudication orchestrator stamps an `"adjudication"` object into `source_json`
/// when it approves a capture candidate; its presence is the promotion token.
fn candidate_has_adjudication_verdict(candidate: &CandidateRecord) -> bool {
    candidate
        .source_json
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .is_some_and(|value| value.get("adjudication").is_some())
}
fn candidate_string_array_to_column(values: &[String]) -> Option<String> {
    if values.is_empty() {
        None
    } else {
        Some(serde_json::to_string(values).unwrap_or_else(|_| "[]".to_string()))
    }
}

fn candidate_string_array_from_column(raw: Option<String>) -> Vec<String> {
    raw.and_then(|text| serde_json::from_str::<Vec<String>>(&text).ok())
        .unwrap_or_default()
}

fn memory_candidate_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CandidateRecord> {
    Ok(CandidateRecord {
        id: row.get(0)?,
        status: row.get(1)?,
        space: row.get(2)?,
        silo: row.get(3)?,
        scope: row.get(4)?,
        project: row.get(5)?,
        kind: row.get(6)?,
        content: row.get(7)?,
        summary: row.get(8)?,
        rationale: row.get(9)?,
        tags: candidate_string_array_from_column(row.get(10)?),
        entity_key: row.get(11)?,
        claim_key: row.get(12)?,
        confidence: row.get(13)?,
        source_type: row.get(14)?,
        source_json: row.get(15)?,
        sensitivity: row.get(16)?,
        supersedes: candidate_string_array_from_column(row.get(17)?),
        created_at: row.get(18)?,
        decided_at: row.get(19)?,
        decided_reason: row.get(20)?,
        resulting_memory_id: row.get(21)?,
    })
}

fn load_candidate(transaction: &Transaction<'_>, id: &str) -> Result<Option<CandidateRecord>> {
    let sql = format!("SELECT {CANDIDATE_COLUMNS} FROM memory_candidates WHERE id = ?1");
    let mut statement = transaction.prepare(&sql)?;
    let mut rows = statement.query(params![id])?;
    match rows.next()? {
        Some(row) => Ok(Some(memory_candidate_from_row(row)?)),
        None => Ok(None),
    }
}

fn validate_candidate_submit_request(request: &CandidateSubmitRequest) -> Result<()> {
    if request.content.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "content must not be empty".to_string(),
        });
    }
    if request.content.chars().count() > MAX_CONTENT_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("content must be at most {MAX_CONTENT_CHARS} characters"),
        });
    }
    if request
        .summary
        .as_deref()
        .is_some_and(|s| s.chars().count() > MAX_SUMMARY_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: format!("summary must be at most {MAX_SUMMARY_CHARS} characters"),
        });
    }
    if request
        .rationale
        .as_deref()
        .is_some_and(|s| s.chars().count() > MAX_SUMMARY_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: format!("rationale must be at most {MAX_SUMMARY_CHARS} characters"),
        });
    }
    validate_optional_metadata_value("space", request.space.as_deref())?;
    reject_all_spaces_sentinel(request.space.as_deref())?;
    validate_optional_metadata_value("silo", request.silo.as_deref())?;
    validate_optional_metadata_value("scope", request.scope.as_deref())?;
    validate_optional_metadata_value("project", request.project.as_deref())?;
    validate_optional_metadata_value("kind", request.kind.as_deref())?;
    validate_optional_metadata_value("entity_key", request.entity_key.as_deref())?;
    validate_optional_metadata_value("claim_key", request.claim_key.as_deref())?;
    if !(0.0..=1.0).contains(&request.confidence) {
        return Err(Error::InvalidRequest {
            message: "confidence must be between 0.0 and 1.0".to_string(),
        });
    }
    if let Some(source_type) = request.source_type.as_deref() {
        if !CANDIDATE_SOURCE_TYPES.contains(&source_type) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported source_type: {source_type} (expected one of {})",
                    CANDIDATE_SOURCE_TYPES.join(", ")
                ),
            });
        }
    }
    if let Some(sensitivity) = request.sensitivity.as_deref() {
        if !CANDIDATE_SENSITIVITIES.contains(&sensitivity) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported sensitivity: {sensitivity} (expected one of {})",
                    CANDIDATE_SENSITIVITIES.join(", ")
                ),
            });
        }
    }
    if let Some(source_json) = &request.source_json {
        if source_json.chars().count() > MAX_SOURCE_REF_JSON_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "source JSON must be at most {MAX_SOURCE_REF_JSON_CHARS} characters"
                ),
            });
        }
        if !JsonValidator::is_object(source_json) {
            return Err(Error::InvalidRequest {
                message: "source JSON must be a valid JSON object".to_string(),
            });
        }
    }
    validate_memory_link_ids("supersedes", &request.supersedes)?;
    let _ = normalized_tags(&request.tags)?;
    Ok(())
}

/// Submit a candidate memory for later review.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is
/// invalid, or `SQLite` rejects the transaction.
pub fn submit_candidate(
    path: impl AsRef<Path>,
    request: &CandidateSubmitRequest,
) -> Result<CandidateSubmitReport> {
    validate_candidate_submit_request(request)?;
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    ensure_memory_candidates(&transaction)?;
    let now = now_timestamp(&transaction)?;
    let id = next_id("cand");
    let tags = normalized_tags(&request.tags)?;
    let source_type = request
        .source_type
        .clone()
        .unwrap_or_else(|| DEFAULT_CANDIDATE_SOURCE_TYPE.to_string());
    let sensitivity = request
        .sensitivity
        .clone()
        .unwrap_or_else(|| DEFAULT_CANDIDATE_SENSITIVITY.to_string());
    transaction.execute(
        "INSERT INTO memory_candidates (id, status, space, silo, scope, project, kind, content, \
         summary, rationale, tags_json, entity_key, claim_key, confidence, source_type, \
         source_json, sensitivity, supersedes_json, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
        params![
            id,
            CANDIDATE_STATUS_PENDING,
            request.space,
            request.silo,
            request.scope,
            request.project,
            request.kind,
            request.content,
            request.summary,
            request.rationale,
            candidate_string_array_to_column(&tags),
            request.entity_key,
            request.claim_key,
            request.confidence,
            source_type,
            request.source_json,
            sensitivity,
            candidate_string_array_to_column(&request.supersedes),
            now,
        ],
    )?;
    let candidate = load_candidate(&transaction, &id)?.ok_or_else(|| Error::InvalidRequest {
        message: "candidate insert did not persist".to_string(),
    })?;
    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }
    Ok(CandidateSubmitReport {
        candidate,
        dry_run: request.dry_run,
    })
}

/// List candidates for review, newest first.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the status filter is
/// unknown, or `SQLite` rejects the transaction.
pub fn list_candidates(
    path: impl AsRef<Path>,
    request: &CandidateListRequest,
) -> Result<CandidateListReport> {
    if let Some(status) = request.status.as_deref() {
        if !CANDIDATE_STATUSES.contains(&status) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported status filter: {status} (expected one of {})",
                    CANDIDATE_STATUSES.join(", ")
                ),
            });
        }
    }
    let limit = request.limit.clamp(1, DEFAULT_CANDIDATE_LIST_LIMIT * 2);
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    ensure_memory_candidates(&transaction)?;

    // Build a WHERE clause from the optional status/space filters.
    let mut clauses: Vec<&str> = Vec::new();
    if request.status.is_some() {
        clauses.push("status = :status");
    }
    if request.space.is_some() {
        clauses.push("space = :space");
    }
    let where_sql = if clauses.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", clauses.join(" AND "))
    };

    let count_sql = format!("SELECT COUNT(*) FROM memory_candidates{where_sql}");
    let list_sql = format!(
        "SELECT {CANDIDATE_COLUMNS} FROM memory_candidates{where_sql} \
         ORDER BY created_at DESC, id DESC LIMIT :limit OFFSET :offset"
    );

    let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
    let offset_i64 = i64::try_from(request.offset).unwrap_or(0);

    let total: i64 = {
        let mut statement = transaction.prepare(&count_sql)?;
        let mut bindings: Vec<(&str, &dyn rusqlite::ToSql)> = Vec::new();
        if let Some(status) = &request.status {
            bindings.push((":status", status));
        }
        if let Some(space) = &request.space {
            bindings.push((":space", space));
        }
        statement.query_row(bindings.as_slice(), |row| row.get(0))?
    };

    let candidates = {
        let mut statement = transaction.prepare(&list_sql)?;
        let mut bindings: Vec<(&str, &dyn rusqlite::ToSql)> = Vec::new();
        if let Some(status) = &request.status {
            bindings.push((":status", status));
        }
        if let Some(space) = &request.space {
            bindings.push((":space", space));
        }
        bindings.push((":limit", &limit_i64));
        bindings.push((":offset", &offset_i64));
        let mut rows = statement.query(bindings.as_slice())?;
        let mut collected = Vec::new();
        while let Some(row) = rows.next()? {
            collected.push(memory_candidate_from_row(row)?);
        }
        collected
    };

    transaction.commit()?;
    Ok(CandidateListReport {
        candidates,
        total: usize::try_from(total).unwrap_or(0),
    })
}

/// Build the canonical source/provenance JSON for an approved candidate,
/// folding the candidate's `source_type` into any supplied source object.
/// Mirrors the metadata-merge pattern used by `verify_memory`.
fn candidate_source_ref_json(source_json: Option<&str>, source_type: &str) -> String {
    let mut map: serde_json::Map<String, serde_json::Value> = match source_json {
        Some(raw) => serde_json::from_str(raw).unwrap_or_default(),
        None => serde_json::Map::new(),
    };
    map.insert(
        "source_type".to_string(),
        serde_json::Value::String(source_type.to_string()),
    );
    serde_json::to_string(&map)
        .unwrap_or_else(|_| format!("{{\"source_type\":{}}}", json_string_for_store(source_type)))
}

fn remember_request_from_candidate(
    candidate: &CandidateRecord,
    embedding: Option<Vec<f32>>,
    embedding_model_id: Option<String>,
) -> RememberRequest {
    let (derived_entity_key, derived_claim_key) =
        memkeeper_core::derive::keys(&candidate.content, candidate.summary.as_deref());
    let metadata_json = candidate
        .rationale
        .as_deref()
        .map(|rationale| format!("{{\"rationale\":{}}}", json_string_for_store(rationale)));
    RememberRequest {
        space: candidate.space.clone(),
        silo: candidate.silo.clone(),
        scope: candidate.scope.clone(),
        project_key: candidate.project.clone(),
        kind: candidate.kind.clone(),
        content: candidate.content.clone(),
        summary: candidate.summary.clone(),
        retrieval_representation: None,
        tags: candidate.tags.clone(),
        entity_key: candidate.entity_key.clone().or(derived_entity_key),
        claim_key: candidate.claim_key.clone().or(derived_claim_key),
        graph: None,
        confidence: candidate.confidence,
        observed_at: None,
        valid_from: None,
        valid_to: None,
        expires_at: None,
        source_ref_json: Some(candidate_source_ref_json(
            candidate.source_json.as_deref(),
            &candidate.source_type,
        )),
        metadata_json,
        source_episode_id: None,
        pinned: false,
        supersedes: candidate.supersedes.clone(),
        contradicts: Vec::new(),
        embedding,
        embedding_model_id,
        token_embedding: None,
        token_embedding_model_id: None,
        // The outer candidate transaction controls rollback; the inner write
        // must commit within it (dry-run is handled by rolling back the whole tx).
        dry_run: false,
        mode: REMEMBER_MODE_AUTO.to_string(),
    }
}

/// Approve a candidate: promote it into a real memory via the remember write
/// path, then mark the candidate approved with the resulting memory id.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the candidate is
/// missing or not pending, the promoted memory fails validation, or `SQLite`
/// rejects the transaction.
pub fn approve_candidate(
    path: impl AsRef<Path>,
    request: &CandidateApproveRequest,
) -> Result<CandidateApproveReport> {
    if request.id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "candidate id must not be empty".to_string(),
        });
    }
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    ensure_memory_candidates(&transaction)?;
    let candidate =
        load_candidate(&transaction, &request.id)?.ok_or_else(|| Error::InvalidRequest {
            message: format!("candidate not found: {}", request.id),
        })?;
    if candidate.status != CANDIDATE_STATUS_PENDING {
        return Err(Error::InvalidRequest {
            message: format!("candidate {} is already {}", candidate.id, candidate.status),
        });
    }
    // Fail-closed adjudication gate: a capture-sourced candidate may not become an
    // active memory without an adjudication verdict when the deployment requires it.
    if candidate_is_capture(&candidate) {
        match adjudication_guard(
            candidate_has_adjudication_verdict(&candidate),
            capture_require_adjudication(),
        ) {
            AdjudicationGuard::Refuse => {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "candidate {} is capture-sourced and unadjudicated; \
                         MEMKEEPER_CAPTURE_REQUIRE_ADJUDICATION refuses promotion \
                         (adjudicate or quarantine it first)",
                        candidate.id
                    ),
                });
            }
            AdjudicationGuard::Degraded => {
                eprintln!(
                    "[memkeeper] NOTE: promoting capture candidate {} without an adjudication \
                     verdict (MEMKEEPER_CAPTURE_REQUIRE_ADJUDICATION not set).",
                    candidate.id
                );
            }
            AdjudicationGuard::Ok => {}
        }
    }
    let remember = remember_request_from_candidate(
        &candidate,
        request.embedding.clone(),
        request.embedding_model_id.clone(),
    );
    validate_remember_request(&remember)?;
    let remember_report = remember_memory_tx(&transaction, &remember)?;
    let memory = remember_report.memory;
    let now = now_timestamp(&transaction)?;
    transaction.execute(
        "UPDATE memory_candidates SET status = ?1, decided_at = ?2, resulting_memory_id = ?3 \
         WHERE id = ?4",
        params![CANDIDATE_STATUS_APPROVED, now, memory.id, candidate.id],
    )?;
    let updated =
        load_candidate(&transaction, &candidate.id)?.ok_or_else(|| Error::InvalidRequest {
            message: "candidate update did not persist".to_string(),
        })?;
    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }
    Ok(CandidateApproveReport {
        candidate: updated,
        memory,
        dry_run: request.dry_run,
    })
}

/// Reject a candidate, recording an optional reason.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the candidate is
/// missing or not pending, or `SQLite` rejects the transaction.
pub fn reject_candidate(
    path: impl AsRef<Path>,
    request: &CandidateRejectRequest,
) -> Result<CandidateRejectReport> {
    if request.id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "candidate id must not be empty".to_string(),
        });
    }
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    ensure_memory_candidates(&transaction)?;
    let candidate =
        load_candidate(&transaction, &request.id)?.ok_or_else(|| Error::InvalidRequest {
            message: format!("candidate not found: {}", request.id),
        })?;
    if candidate.status != CANDIDATE_STATUS_PENDING {
        return Err(Error::InvalidRequest {
            message: format!("candidate {} is already {}", candidate.id, candidate.status),
        });
    }
    let now = now_timestamp(&transaction)?;
    transaction.execute(
        "UPDATE memory_candidates SET status = ?1, decided_at = ?2, decided_reason = ?3 \
         WHERE id = ?4",
        params![CANDIDATE_STATUS_REJECTED, now, request.reason, candidate.id],
    )?;
    let updated =
        load_candidate(&transaction, &candidate.id)?.ok_or_else(|| Error::InvalidRequest {
            message: "candidate update did not persist".to_string(),
        })?;
    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }
    Ok(CandidateRejectReport {
        candidate: updated,
        dry_run: request.dry_run,
    })
}

/// Quarantine a candidate an adjudicator flagged, recording an optional reason.
/// Terminal like `reject` but a distinct status so quarantined captures can be
/// reviewed separately from human/assistant rejections.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the candidate is
/// missing or not pending, or `SQLite` rejects the transaction.
pub fn quarantine_candidate(
    path: impl AsRef<Path>,
    request: &CandidateQuarantineRequest,
) -> Result<CandidateQuarantineReport> {
    if request.id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "candidate id must not be empty".to_string(),
        });
    }
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    ensure_memory_candidates(&transaction)?;
    let candidate =
        load_candidate(&transaction, &request.id)?.ok_or_else(|| Error::InvalidRequest {
            message: format!("candidate not found: {}", request.id),
        })?;
    if candidate.status != CANDIDATE_STATUS_PENDING {
        return Err(Error::InvalidRequest {
            message: format!("candidate {} is already {}", candidate.id, candidate.status),
        });
    }
    let now = now_timestamp(&transaction)?;
    transaction.execute(
        "UPDATE memory_candidates SET status = ?1, decided_at = ?2, decided_reason = ?3 \
         WHERE id = ?4",
        params![
            CANDIDATE_STATUS_QUARANTINED,
            now,
            request.reason,
            candidate.id
        ],
    )?;
    let updated =
        load_candidate(&transaction, &candidate.id)?.ok_or_else(|| Error::InvalidRequest {
            message: "candidate update did not persist".to_string(),
        })?;
    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }
    Ok(CandidateQuarantineReport {
        candidate: updated,
        dry_run: request.dry_run,
    })
}
