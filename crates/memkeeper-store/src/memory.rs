//! Core memory CRUD extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::path::Path;

use rusqlite::{params, Connection, OptionalExtension, Row, Transaction};

use memkeeper_core::{
    infer_kind_from_prefix, kind, scope, status, DEFAULT_DURABLE_SILO, DEFAULT_SPACE,
};

use crate::{
    apply_graph_capture, bounded_char_slice, collapse_whitespace, collect_rows, default_silo,
    enforce_active_colbert_model, ensure_memory_candidates, ensure_silo_exists,
    ensure_space_exists, insert_memory_embedding, insert_representation, is_prefixable_search_term,
    json_string_for_store, limit_i64, load_representation, next_id, normalize_utc_timestamp,
    normalized_tags, now_timestamp, open_initialized_read_fast, open_initialized_write,
    push_unique, reject_all_spaces_sentinel, retrieval_companion, search_term_stems, search_terms,
    sha256_hex, sha256_text, string_array_json, upsert_memory_token_embedding,
    upsert_relationship_tx, validate_forget_request, validate_graph_capture,
    validate_history_request, validate_optional_metadata_value, validate_optional_timestamp,
    validate_remember_request, validate_retrieval_representation, with_read_snapshot, Error,
    ForgetReport, ForgetRequest, GetOptions, HistoryOptions, HistoryReport, MemoryEventRecord,
    MemoryLinkRecord, MemoryRecord, MemoryVersionRecord, RememberCandidate,
    RememberConflictCandidate, RememberReport, RememberRequest, RepresentationWriteStatus, Result,
    RetrievalRepresentationInput, VerifyReport, VerifyRequest, MAX_CONTENT_CHARS, MAX_GET_LINKS,
    MAX_HISTORY_LIMIT, MAX_REMEMBER_CANDIDATES, MAX_REMEMBER_CONFLICT_CANDIDATES,
    MAX_REMEMBER_LEXICAL_SCAN, MAX_REMEMBER_LEXICAL_TERMS, MAX_SEARCH_TERMS, MAX_SNIPPET_CHARS,
    MAX_SOURCE_REF_JSON_CHARS, MAX_TIMESTAMP_CHARS, REMEMBER_LEXICAL_THRESHOLD, REMEMBER_MODE_AUTO,
    REMEMBER_SUPERSEDE_MODES,
};

pub fn remember_memory(
    path: impl AsRef<Path>,
    request: &RememberRequest,
) -> Result<RememberReport> {
    validate_remember_request(request)?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = remember_memory_tx(&transaction, request)?;

    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }

    Ok(report)
}

/// Fetch one memory by id from an initialized store.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the id is empty, the
/// memory is not found, or `SQLite` rejects the query.
pub fn get_memory(path: impl AsRef<Path>, id: &str, options: GetOptions) -> Result<MemoryRecord> {
    if id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "memory id must not be empty".to_string(),
        });
    }
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        load_memory(connection, id, options)
    })
}

/// Tombstone one memory by id and preserve an audit event.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the memory does not exist, the memory is already tombstoned, or `SQLite`
/// rejects the transaction.
pub fn forget_memory(path: impl AsRef<Path>, request: &ForgetRequest) -> Result<ForgetReport> {
    validate_forget_request(request)?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = forget_memory_tx(&transaction, request)?;

    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }

    Ok(report)
}
pub fn verify_memory(path: impl AsRef<Path>, request: &VerifyRequest) -> Result<VerifyReport> {
    if request.memory_id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "memory id must not be empty".to_string(),
        });
    }
    validate_optional_timestamp("now", request.now.as_deref())?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = verify_memory_tx(&transaction, request)?;
    transaction.commit()?;
    Ok(report)
}

fn verify_memory_tx(
    transaction: &Transaction<'_>,
    request: &VerifyRequest,
) -> Result<VerifyReport> {
    let now = match request.now.as_deref() {
        Some(ts) => normalize_utc_timestamp(ts),
        None => now_timestamp(transaction)?,
    };

    let existing: Option<String> = transaction
        .query_row(
            "SELECT metadata_json FROM memories WHERE id = ?1",
            [&request.memory_id],
            |row| row.get(0),
        )
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "memory",
            id: request.memory_id.clone(),
        })?;

    // Parse existing metadata_json into a map. A present-but-unparseable (or
    // non-object) value is store corruption; erroring preserves the evidence
    // instead of silently replacing it with just the verify keys.
    let mut map: serde_json::Map<String, serde_json::Value> = match existing.as_deref() {
        None => serde_json::Map::new(),
        Some(raw) => serde_json::from_str::<serde_json::Value>(raw)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .ok_or_else(|| Error::Conflict {
                message: format!(
                    "memory {} has corrupt metadata_json; refusing to overwrite it",
                    request.memory_id
                ),
            })?,
    };

    map.insert(
        "verified_at".to_string(),
        serde_json::Value::String(now.clone()),
    );
    if let Some(ref src) = request.verified_against {
        map.insert(
            "verified_against".to_string(),
            serde_json::Value::String(src.clone()),
        );
    }

    let merged = serde_json::to_string(&map).map_err(|e| Error::InvalidRequest {
        message: format!("failed to serialize metadata_json: {e}"),
    })?;

    // Existence was established by the SELECT above, inside this transaction.
    transaction.execute(
        "UPDATE memories SET metadata_json = ?1, updated_at = ?2 WHERE id = ?3",
        params![&merged, &now, &request.memory_id],
    )?;

    Ok(VerifyReport {
        memory_id: request.memory_id.clone(),
        verified_at: now,
    })
}

/// Fetch bounded audit history for one memory by id.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the memory does not exist, or `SQLite` rejects the query.
pub fn memory_history(
    path: impl AsRef<Path>,
    id: &str,
    options: HistoryOptions,
) -> Result<HistoryReport> {
    validate_history_request(id, options)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        load_history(connection, id, options)
    })
}

/// Search memories deterministically with `SQLite` FTS5/BM25 and metadata filters.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
#[derive(Debug, Clone)]
struct RememberCandidateRow {
    memory_id: String,
    space: String,
    silo: String,
    kind: String,
    status: String,
    entity_key: Option<String>,
    claim_key: Option<String>,
    content: String,
    summary: Option<String>,
    content_sha256: String,
}

#[derive(Debug, Clone)]
struct RememberCandidateAccumulator {
    row: RememberCandidateRow,
    relationship: String,
    score: f64,
    matched_on: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct RememberCandidateDetection<'a> {
    space: &'a str,
    silo: &'a str,
    kind: &'a str,
    content_sha256: &'a str,
    entity_key: Option<&'a str>,
    claim_key: Option<&'a str>,
    request_terms: BTreeSet<String>,
    excluded_ids: &'a BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct SameClaimCandidate {
    memory_id: String,
    kind: String,
    observed_at: String,
    content: String,
    pinned: bool,
}

pub(crate) fn forget_memory_tx(
    transaction: &Transaction<'_>,
    request: &ForgetRequest,
) -> Result<ForgetReport> {
    let now = now_timestamp(transaction)?;
    let old_status = transaction
        .query_row(
            "SELECT status FROM memories WHERE id = ?1",
            [&request.id],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "memory",
            id: request.id.clone(),
        })?;

    if old_status == status::TOMBSTONED {
        return Err(Error::Conflict {
            message: format!("memory is already tombstoned: {}", request.id),
        });
    }

    transaction.execute(
        "UPDATE memories SET status = ?1, updated_at = ?2, deleted_at = ?2 WHERE id = ?3",
        params![status::TOMBSTONED, &now, &request.id],
    )?;
    transaction.execute(
        "UPDATE memory_fts SET status = ?1 WHERE memory_id = ?2",
        params![status::TOMBSTONED, &request.id],
    )?;
    transaction.execute(
        "UPDATE memory_fts_public SET status = ?1 WHERE memory_id = ?2",
        params![status::TOMBSTONED, &request.id],
    )?;

    let is_correction = request.mode == "correct";
    let event_type = if is_correction { "correct" } else { "forget" };
    let data_json = if is_correction {
        correction_event_data_json(
            request.dry_run,
            &request.mode,
            request.corrected_by.as_deref(),
            &load_tags(transaction, &request.id)?,
        )
    } else {
        forget_event_data_json(request.dry_run, &request.mode)
    };

    let event_id = next_id("evt");
    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, old_status, new_status, actor, reason, data_json, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, 'memkeeper', ?6, ?7, ?8)",
        params![
            &event_id,
            &request.id,
            event_type,
            &old_status,
            status::TOMBSTONED,
            request.reason.as_deref(),
            data_json,
            &now,
        ],
    )?;

    // For a correction with a named replacement, record the directed
    // `contradicts` edge (replacement -> wrong memory). This is the explicit,
    // intent-captured correction signal the synthesis loop can key off without
    // having to guess correction-vs-evolution from supersession history.
    if let Some(corrected_by) = &request.corrected_by {
        link_memory(transaction, corrected_by, &request.id, "contradicts", &now)?;
    }

    Ok(ForgetReport {
        memory_id: request.id.clone(),
        old_status,
        new_status: status::TOMBSTONED.to_string(),
        event_id,
        dry_run: request.dry_run,
    })
}

fn load_history(
    connection: &Connection,
    id: &str,
    options: HistoryOptions,
) -> Result<HistoryReport> {
    let current_status = connection
        .query_row("SELECT status FROM memories WHERE id = ?1", [id], |row| {
            row.get::<_, String>(0)
        })
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "memory",
            id: id.to_string(),
        })?;
    let total_events = count_for_id(connection, "memory_events", "memory_id", id)?;
    let total_versions = count_for_id(connection, "memory_versions", "memory_id", id)?;
    let mut versions = load_versions_limited(connection, id, options.limit)?;
    if !options.include_source {
        for version in &mut versions {
            version.source_ref_json = None;
        }
    }
    let events = load_events_limited(connection, id, options.limit)?;
    let truncated = total_events > events.len() || total_versions > versions.len();

    Ok(HistoryReport {
        memory_id: id.to_string(),
        current_status,
        events,
        versions,
        truncated,
    })
}
pub(crate) fn remember_memory_tx(
    transaction: &Transaction<'_>,
    request: &RememberRequest,
) -> Result<RememberReport> {
    let now = now_timestamp(transaction)?;
    let space = request.space.as_deref().unwrap_or(DEFAULT_SPACE);
    let silo = match request.silo.as_deref() {
        Some(value) => value.to_string(),
        None => {
            default_silo(transaction, space)?.unwrap_or_else(|| DEFAULT_DURABLE_SILO.to_string())
        }
    };
    ensure_silo_exists(transaction, space, &silo)?;

    let source_episode_id = match request.source_episode_id.as_deref() {
        Some(id) => {
            ensure_source_episode_exists(transaction, space, id)?;
            Some(id.to_string())
        }
        None => None,
    };

    let scope = request
        .scope
        .clone()
        .unwrap_or_else(|| scope::WORKSPACE.to_string());
    let memory_kind = request
        .kind
        .clone()
        .or_else(|| infer_kind_from_prefix(&request.content).map(str::to_string))
        .unwrap_or_else(|| kind::FACT.to_string());
    // Caller-supplied timestamps normalize to fixed millisecond precision so
    // lexical ordering (auto-supersede, freshness, ORDER BY) stays temporal.
    let observed_at = request
        .observed_at
        .as_deref()
        .map_or_else(|| now.clone(), normalize_utc_timestamp);
    let valid_from = request.valid_from.as_deref().map(normalize_utc_timestamp);
    let valid_to = request.valid_to.as_deref().map(normalize_utc_timestamp);
    let expires_at = request.expires_at.as_deref().map(normalize_utc_timestamp);
    let memory_id = next_id("mem");
    let version_id = next_id("ver");
    let event_id = next_id("evt");
    let content_sha256 = sha256_hex(request.content.as_bytes());
    let pinned = i64::from(request.pinned);
    let tags = normalized_tags(&request.tags)?;
    let tags_text = tags.join(" ");
    let metadata_text = memory_fts_metadata_text(
        request.project_key.as_deref(),
        request.entity_key.as_deref(),
        request.claim_key.as_deref(),
        &request.content,
        request.summary.as_deref(),
        &tags_text,
    );
    let excluded_ids = request
        .supersedes
        .iter()
        .chain(request.contradicts.iter())
        .cloned()
        .collect::<BTreeSet<_>>();
    let candidate_detection = RememberCandidateDetection {
        space,
        silo: &silo,
        kind: &memory_kind,
        content_sha256: &content_sha256,
        entity_key: request.entity_key.as_deref(),
        claim_key: request.claim_key.as_deref(),
        request_terms: lexical_terms(&request.content),
        excluded_ids: &excluded_ids,
    };
    let (candidates, candidates_truncated) =
        detect_remember_candidates(transaction, &candidate_detection)?;
    let same_claim_candidates = same_claim_candidates(transaction, space, request, &excluded_ids)?;
    let non_pinned_same_claim: Vec<String> = same_claim_candidates
        .iter()
        .filter(|candidate| !candidate.pinned)
        .map(|candidate| candidate.memory_id.clone())
        .collect();
    // Supersession mode governs how this write resolves against active memories
    // sharing its entity/claim key. `auto` is the historical policy; the others
    // were added so callers can declare intent explicitly.
    //   auto      -> older same-key memories of eligible kinds (current default)
    //   append    -> coexist; supersede nothing
    //   supersede -> force-retire all non-pinned same-key actives, any kind
    //   suggest   -> mutate nothing; return the would-be set for review
    //   conflict  -> mutate nothing; open a conflict row per same-key active
    let (auto_superseded, supersede_suggestions, open_conflicts) = match request.mode.as_str() {
        "append" => (Vec::new(), Vec::new(), false),
        "supersede" => (non_pinned_same_claim.clone(), Vec::new(), false),
        "suggest" => (Vec::new(), non_pinned_same_claim.clone(), false),
        "conflict" => (Vec::new(), Vec::new(), true),
        // `auto` (and the validated default) keep the historical behavior.
        _ => (
            auto_supersede_candidates(&memory_kind, &observed_at, &same_claim_candidates),
            Vec::new(),
            false,
        ),
    };
    let conflict_candidates = if memory_kind == kind::CONTINUITY || open_conflicts {
        same_claim_candidates
            .iter()
            .map(same_claim_conflict_candidate)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    transaction.execute(
        "INSERT INTO memories (
            id, space_name, silo_name, scope, project_key, kind, entity_key, claim_key,
            status, active_version_id, confidence, pinned, source_episode_id, valid_from,
            valid_to, observed_at, created_at, updated_at, expires_at, metadata_json
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)",
        params![
            &memory_id,
            space,
            &silo,
            &scope,
            request.project_key.as_deref(),
            &memory_kind,
            request.entity_key.as_deref(),
            request.claim_key.as_deref(),
            status::ACTIVE,
            &version_id,
            request.confidence,
            pinned,
            source_episode_id.as_deref(),
            valid_from.as_deref(),
            valid_to.as_deref(),
            &observed_at,
            &now,
            &now,
            expires_at.as_deref(),
            request.metadata_json.as_deref(),
        ],
    )?;

    transaction.execute(
        "INSERT INTO memory_versions (
            id, memory_id, version_num, content, summary, content_sha256,
            source_episode_id, source_ref_json, created_at, created_by, event_id
         ) VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            &version_id,
            &memory_id,
            &request.content,
            request.summary.as_deref(),
            &content_sha256,
            source_episode_id.as_deref(),
            request.source_ref_json.as_deref(),
            &now,
            "memkeeper",
            &event_id,
        ],
    )?;

    let stored_representation = request
        .retrieval_representation
        .as_ref()
        .map(|input| insert_representation(transaction, &version_id, input, &now))
        .transpose()?;

    let retrieval_text = retrieval_companion(
        request.summary.as_deref(),
        request.retrieval_representation.as_ref(),
    );
    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, new_status, actor, data_json, created_at)
         VALUES (?1, ?2, 'remember', ?3, 'memkeeper', ?4, ?5)",
        params![
            &event_id,
            &memory_id,
            status::ACTIVE,
            event_data_json(
                request.dry_run,
                &request.supersedes,
                &request.contradicts,
                &auto_superseded,
                &conflict_candidates,
            ),
            &now,
        ],
    )?;

    for tag in &tags {
        transaction.execute(
            "INSERT INTO memory_tags (memory_id, tag, created_at) VALUES (?1, ?2, ?3)",
            params![&memory_id, tag, &now],
        )?;
    }

    for superseded_id in &request.supersedes {
        supersede_memory(transaction, space, &memory_id, superseded_id, &now)?;
    }
    for superseded_id in &auto_superseded {
        supersede_memory(transaction, space, &memory_id, superseded_id, &now)?;
    }
    for contradicted_id in &request.contradicts {
        ensure_memory_in_space(transaction, contradicted_id, space)?;
        link_memory(
            transaction,
            &memory_id,
            contradicted_id,
            "contradicts",
            &now,
        )?;
        link_memory(
            transaction,
            contradicted_id,
            &memory_id,
            "contradicts",
            &now,
        )?;
    }
    if open_conflicts {
        for candidate in &same_claim_candidates {
            open_conflict(transaction, space, &memory_id, &candidate.memory_id, &now)?;
        }
    }

    transaction.execute(
        "INSERT INTO memory_fts (
            memory_id, version_id, space_name, silo_name, status, kind, content, retrieval_text,
            tags, source_text, metadata_text
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            &memory_id,
            &version_id,
            space,
            &silo,
            status::ACTIVE,
            &memory_kind,
            &request.content,
            retrieval_text,
            &tags_text,
            request.source_ref_json.as_deref(),
            &metadata_text,
        ],
    )?;
    transaction.execute(
        "INSERT INTO memory_fts_public (
            memory_id, version_id, space_name, silo_name, status, kind, content, retrieval_text,
            tags, metadata_text
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            &memory_id,
            &version_id,
            space,
            &silo,
            status::ACTIVE,
            &memory_kind,
            &request.content,
            retrieval_text,
            &tags_text,
            &metadata_text,
        ],
    )?;

    insert_memory_embedding(
        transaction,
        &memory_id,
        &version_id,
        &now,
        request.embedding.as_deref(),
        request.embedding_model_id.as_deref(),
    )?;

    if let (Some(token_vecs), Some(token_model)) = (
        request.token_embedding.as_ref(),
        request.token_embedding_model_id.as_deref(),
    ) {
        enforce_active_colbert_model(transaction, token_model, &now)?;
        upsert_memory_token_embedding(transaction, &memory_id, token_model, token_vecs)?;
    }

    if let Some(entity_key) = request.entity_key.as_deref() {
        upsert_memory_entity_projection(
            transaction,
            space,
            entity_key,
            source_episode_id.as_deref(),
            &now,
        )?;
    }
    let graph_capture = request
        .graph
        .as_ref()
        .map(|graph| apply_graph_capture(transaction, space, &memory_id, request, graph))
        .transpose()?;

    let mut memory = load_memory(
        transaction,
        &memory_id,
        GetOptions {
            include_history: false,
            include_links: true,
            include_source: false,
        },
    )?;
    if !request.dry_run {
        memory.versions = None;
        memory.events = None;
    }

    Ok(RememberReport {
        memory,
        event_id,
        graph_capture,
        processing_status: if request.dry_run {
            "dry_run"
        } else {
            "indexed"
        }
        .to_string(),
        representation: stored_representation.as_ref().map(|stored| {
            let semantic_indexed = request.token_embedding.is_some();
            RepresentationWriteStatus {
                kind: stored.kind.clone(),
                text_sha256: stored.text_sha256.clone(),
                fts_indexed: true,
                semantic_indexed,
                status: if semantic_indexed {
                    "indexed"
                } else {
                    "lexical_only"
                }
                .to_string(),
            }
        }),
        candidates,
        candidates_truncated,
        auto_superseded,
        conflict_candidates,
        supersede_suggestions,
        dry_run: request.dry_run,
    })
}
pub(crate) fn upsert_memory_entity_projection(
    transaction: &Transaction<'_>,
    space: &str,
    entity_key: &str,
    source_episode_id: Option<&str>,
    now: &str,
) -> Result<()> {
    let entity_id = next_id("ent");
    let canonical_name = canonical_name_from_entity_key(entity_key);
    transaction.execute(
        "INSERT INTO entities (
            id, space_name, entity_key, entity_type, canonical_name, status, confidence,
            source_episode_id, metadata_json, created_at, updated_at
         ) VALUES (?1, ?2, ?3, 'MemorySubject', ?4, 'active', 1.0, ?5, NULL, ?6, ?6)
         ON CONFLICT(space_name, entity_key) DO UPDATE SET
            canonical_name = excluded.canonical_name,
            source_episode_id = COALESCE(entities.source_episode_id, excluded.source_episode_id),
            updated_at = excluded.updated_at",
        params![
            &entity_id,
            space,
            entity_key,
            &canonical_name,
            source_episode_id,
            now,
        ],
    )?;
    Ok(())
}

fn canonical_name_from_entity_key(entity_key: &str) -> String {
    let readable = entity_key
        .rsplit([':', '/', '#'])
        .next()
        .unwrap_or(entity_key)
        .chars()
        .map(|character| match character {
            '_' | '-' | '.' => ' ',
            other => other,
        })
        .collect::<String>();
    let collapsed = collapse_whitespace(&readable);
    if collapsed.is_empty() {
        entity_key.to_string()
    } else {
        collapsed
    }
}

pub(crate) fn normalized_alias(value: &str) -> String {
    collapse_whitespace(&value.to_ascii_lowercase())
}

fn detect_remember_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
) -> Result<(Vec<RememberCandidate>, bool)> {
    let mut candidates = BTreeMap::<String, RememberCandidateAccumulator>::new();
    collect_exact_content_candidates(connection, request, &mut candidates)?;
    collect_claim_key_candidates(connection, request, &mut candidates)?;
    collect_entity_key_candidates(connection, request, &mut candidates)?;
    collect_lexical_candidates(connection, request, &mut candidates)?;

    let mut output = candidates
        .into_values()
        .map(RememberCandidateAccumulator::into_candidate)
        .collect::<Vec<_>>();
    output.sort_by(compare_remember_candidates);
    let truncated = output.len() > MAX_REMEMBER_CANDIDATES;
    if truncated {
        output.truncate(MAX_REMEMBER_CANDIDATES);
    }
    Ok((output, truncated))
}

fn same_claim_candidates(
    connection: &Connection,
    space: &str,
    request: &RememberRequest,
    excluded_ids: &BTreeSet<String>,
) -> Result<Vec<SameClaimCandidate>> {
    if !request.supersedes.is_empty() || !request.contradicts.is_empty() {
        return Ok(Vec::new());
    }
    let (Some(entity_key), Some(claim_key)) =
        (request.entity_key.as_deref(), request.claim_key.as_deref())
    else {
        return Ok(Vec::new());
    };

    let mut statement = connection.prepare_cached(
        "SELECT m.id, m.kind, m.observed_at, v.content, m.pinned
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.space_name = ?1 AND m.entity_key = ?2 AND m.claim_key = ?3 AND m.status = 'active'
         ORDER BY m.observed_at DESC, m.updated_at DESC, m.id ASC
         LIMIT ?4",
    )?;
    let rows = statement.query_map(
        params![
            space,
            entity_key,
            claim_key,
            limit_i64(MAX_REMEMBER_CONFLICT_CANDIDATES.saturating_add(1))?,
        ],
        |row| {
            Ok(SameClaimCandidate {
                memory_id: row.get(0)?,
                kind: row.get(1)?,
                observed_at: row.get(2)?,
                content: row.get(3)?,
                pinned: row.get::<_, i64>(4)? == 1,
            })
        },
    )?;
    Ok(collect_rows(rows)?
        .into_iter()
        .filter(|candidate| !excluded_ids.contains(&candidate.memory_id))
        .take(MAX_REMEMBER_CONFLICT_CANDIDATES)
        .collect())
}

fn same_claim_conflict_candidate(candidate: &SameClaimCandidate) -> RememberConflictCandidate {
    RememberConflictCandidate {
        memory_id: candidate.memory_id.clone(),
        kind: candidate.kind.clone(),
        observed_at: candidate.observed_at.clone(),
        snippet: bounded_char_slice(&candidate.content, 0, MAX_SNIPPET_CHARS.min(240)),
    }
}

fn auto_supersede_candidates(
    incoming_kind: &str,
    observed_at: &str,
    candidates: &[SameClaimCandidate],
) -> Vec<String> {
    if incoming_kind == kind::CONTINUITY || !is_auto_supersede_kind(incoming_kind) {
        return Vec::new();
    }
    candidates
        .iter()
        .filter(|candidate| !candidate.pinned && candidate.observed_at.as_str() < observed_at)
        .map(|candidate| candidate.memory_id.clone())
        .collect()
}

fn is_auto_supersede_kind(value: &str) -> bool {
    matches!(
        value,
        kind::FACT | kind::PREFERENCE | kind::DECISION | kind::LESSON
    )
}

fn collect_exact_content_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
) -> Result<()> {
    let mut statement = connection.prepare_cached(
        "SELECT m.id, m.space_name, m.silo_name, m.kind, m.status, m.entity_key, m.claim_key,
                v.content, v.summary, v.content_sha256
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.space_name = ?1 AND m.silo_name = ?2 AND m.status = 'active' AND v.content_sha256 = ?3
         ORDER BY m.id ASC
         LIMIT ?4",
    )?;
    let rows = statement.query_map(
        params![
            request.space,
            request.silo,
            request.content_sha256,
            limit_i64(MAX_REMEMBER_CANDIDATES.saturating_add(1))?,
        ],
        remember_candidate_row_from_row,
    )?;
    for row in collect_rows(rows)? {
        merge_remember_candidate(candidates, request, row, "duplicate", 1.0, "content_sha256");
    }
    Ok(())
}

fn collect_claim_key_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
) -> Result<()> {
    let Some(claim_key) = request.claim_key else {
        return Ok(());
    };
    let mut statement = connection.prepare_cached(
        "SELECT m.id, m.space_name, m.silo_name, m.kind, m.status, m.entity_key, m.claim_key,
                v.content, v.summary, v.content_sha256
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.space_name = ?1 AND m.silo_name = ?2 AND m.status = 'active' AND m.claim_key = ?3
         ORDER BY m.observed_at DESC, m.updated_at DESC, m.id ASC
         LIMIT ?4",
    )?;
    let rows = statement.query_map(
        params![
            request.space,
            request.silo,
            claim_key,
            limit_i64(MAX_REMEMBER_CANDIDATES.saturating_add(1))?,
        ],
        remember_candidate_row_from_row,
    )?;
    for row in collect_rows(rows)? {
        merge_remember_candidate(
            candidates,
            request,
            row,
            "update_candidate",
            0.95,
            "claim_key",
        );
    }
    Ok(())
}

fn collect_entity_key_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
) -> Result<()> {
    let Some(entity_key) = request.entity_key else {
        return Ok(());
    };
    let mut statement = connection.prepare_cached(
        "SELECT m.id, m.space_name, m.silo_name, m.kind, m.status, m.entity_key, m.claim_key,
                v.content, v.summary, v.content_sha256
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.space_name = ?1 AND m.silo_name = ?2 AND m.status = 'active' AND m.entity_key = ?3 AND m.kind = ?4
         ORDER BY m.observed_at DESC, m.updated_at DESC, m.id ASC
         LIMIT ?5",
    )?;
    let rows = statement.query_map(
        params![
            request.space,
            request.silo,
            entity_key,
            request.kind,
            limit_i64(MAX_REMEMBER_CANDIDATES.saturating_add(1))?,
        ],
        remember_candidate_row_from_row,
    )?;
    for row in collect_rows(rows)? {
        merge_remember_candidate(
            candidates,
            request,
            row,
            "update_candidate",
            0.75,
            "entity_key_kind",
        );
    }
    Ok(())
}

fn collect_lexical_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
) -> Result<()> {
    if request.request_terms.len() < 3 {
        return Ok(());
    }
    let fts_query = remember_candidate_fts_query(&request.request_terms);
    if fts_query.is_empty() {
        return Ok(());
    }
    let sql = format!(
        "SELECT m.id, m.space_name, m.silo_name, m.kind, m.status, m.entity_key, m.claim_key,
                v.content, v.summary, v.content_sha256
         FROM memory_fts_public
         JOIN memories m ON m.id = memory_fts_public.memory_id
         JOIN memory_versions v ON v.id = memory_fts_public.version_id
         WHERE memory_fts_public MATCH ?1 AND m.space_name = ?2 AND m.silo_name = ?3 AND m.status = 'active'
         ORDER BY bm25(memory_fts_public), m.observed_at DESC, m.id ASC
         LIMIT {MAX_REMEMBER_LEXICAL_SCAN}"
    );
    let mut statement = connection.prepare_cached(&sql)?;
    let rows = statement.query_map(
        params![&fts_query, request.space, request.silo],
        remember_candidate_row_from_row,
    )?;
    for row in collect_rows(rows)? {
        let row_terms = lexical_terms(&row.content);
        let similarity = jaccard_similarity(&request.request_terms, &row_terms);
        if similarity >= REMEMBER_LEXICAL_THRESHOLD {
            merge_remember_candidate(
                candidates,
                request,
                row,
                "related_candidate",
                similarity.min(0.94),
                "lexical_similarity",
            );
        }
    }
    Ok(())
}

fn remember_candidate_fts_query(terms: &BTreeSet<String>) -> String {
    terms
        .iter()
        .filter(|term| term.chars().count() >= 4)
        .take(MAX_REMEMBER_LEXICAL_TERMS)
        .map(|term| format!("{{content retrieval_text tags metadata_text}} : {term}"))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn remember_candidate_row_from_row(row: &Row<'_>) -> rusqlite::Result<RememberCandidateRow> {
    Ok(RememberCandidateRow {
        memory_id: row.get(0)?,
        space: row.get(1)?,
        silo: row.get(2)?,
        kind: row.get(3)?,
        status: row.get(4)?,
        entity_key: row.get(5)?,
        claim_key: row.get(6)?,
        content: row.get(7)?,
        summary: row.get(8)?,
        content_sha256: row.get(9)?,
    })
}

fn merge_remember_candidate(
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
    request: &RememberCandidateDetection<'_>,
    row: RememberCandidateRow,
    relationship: &str,
    score: f64,
    matched_on: &str,
) {
    if request.excluded_ids.contains(&row.memory_id) {
        return;
    }
    let entry =
        candidates
            .entry(row.memory_id.clone())
            .or_insert_with(|| RememberCandidateAccumulator {
                row,
                relationship: relationship.to_string(),
                score,
                matched_on: BTreeSet::new(),
            });
    let new_priority = relationship_priority(relationship);
    let old_priority = relationship_priority(&entry.relationship);
    if new_priority < old_priority || (new_priority == old_priority && score > entry.score) {
        entry.relationship = relationship.to_string();
        entry.score = score;
    }
    entry.matched_on.insert(matched_on.to_string());
}

impl RememberCandidateAccumulator {
    fn into_candidate(self) -> RememberCandidate {
        let snippet = bounded_char_slice(
            self.row.summary.as_deref().unwrap_or(&self.row.content),
            0,
            MAX_SNIPPET_CHARS.min(240),
        );
        RememberCandidate {
            memory_id: self.row.memory_id,
            relationship: self.relationship,
            score: self.score,
            matched_on: self.matched_on.into_iter().collect(),
            space: self.row.space,
            silo: self.row.silo,
            kind: self.row.kind,
            status: self.row.status,
            summary: self.row.summary,
            snippet,
            content_sha256: self.row.content_sha256,
            entity_key: self.row.entity_key,
            claim_key: self.row.claim_key,
        }
    }
}

fn compare_remember_candidates(
    left: &RememberCandidate,
    right: &RememberCandidate,
) -> std::cmp::Ordering {
    relationship_priority(&left.relationship)
        .cmp(&relationship_priority(&right.relationship))
        .then_with(|| right.score.total_cmp(&left.score))
        .then_with(|| left.memory_id.cmp(&right.memory_id))
}

fn relationship_priority(relationship: &str) -> u8 {
    match relationship {
        "duplicate" => 0,
        "update_candidate" => 1,
        _ => 2,
    }
}

fn lexical_terms(value: &str) -> BTreeSet<String> {
    search_terms(value)
        .into_iter()
        .filter(|term| term.chars().count() >= 3)
        .take(MAX_SEARCH_TERMS.saturating_mul(2))
        .collect()
}

fn jaccard_similarity(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count();
    let union = left.union(right).count();
    if union == 0 {
        0.0
    } else {
        f64::from(u32::try_from(intersection).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(union).unwrap_or(u32::MAX))
    }
}

pub(crate) fn ensure_source_episode_exists(
    connection: &Connection,
    space: &str,
    id: &str,
) -> Result<()> {
    let exists: i64 = connection.query_row(
        "SELECT COUNT(*) FROM source_episodes WHERE id = ?1 AND space_name = ?2",
        params![id, space],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Err(Error::NotFound {
            entity: "source_episode",
            id: id.to_string(),
        });
    }
    Ok(())
}

/// Open a contradiction conflict between a new memory and an existing one,
/// for `conflict`-mode writes. Does not change either memory's status; the
/// conflict awaits human resolution.
fn open_conflict(
    transaction: &Transaction<'_>,
    space: &str,
    new_memory_id: &str,
    other_memory_id: &str,
    now: &str,
) -> Result<()> {
    transaction.execute(
        "INSERT INTO conflicts (id, space_name, status, memory_a_id, memory_b_id, \
         conflict_type, created_at, updated_at) \
         VALUES (?1, ?2, 'open', ?3, ?4, 'contradiction', ?5, ?5)",
        params![next_id("cfl"), space, new_memory_id, other_memory_id, now],
    )?;
    Ok(())
}

fn supersede_memory(
    transaction: &Transaction<'_>,
    space: &str,
    new_memory_id: &str,
    superseded_id: &str,
    now: &str,
) -> Result<()> {
    let target = transaction
        .query_row(
            "SELECT status, pinned FROM memories WHERE id = ?1 AND space_name = ?2",
            params![superseded_id, space],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()?;
    let Some((old_status, pinned)) = target else {
        return Err(Error::NotFound {
            entity: "memory",
            id: superseded_id.to_string(),
        });
    };
    if old_status != status::ACTIVE {
        return Err(Error::Conflict {
            message: format!("cannot supersede non-active memory: {superseded_id}"),
        });
    }
    if pinned == 1 {
        return Err(Error::Conflict {
            message: format!("cannot supersede pinned memory: {superseded_id}"),
        });
    }

    transaction.execute(
        "UPDATE memories SET status = ?1, updated_at = ?2 WHERE id = ?3",
        params![status::SUPERSEDED, now, superseded_id],
    )?;
    transaction.execute(
        "UPDATE memory_fts SET status = ?1 WHERE memory_id = ?2",
        params![status::SUPERSEDED, superseded_id],
    )?;
    transaction.execute(
        "UPDATE memory_fts_public SET status = ?1 WHERE memory_id = ?2",
        params![status::SUPERSEDED, superseded_id],
    )?;

    let event_id = next_id("evt");
    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, old_status, new_status, actor, reason, created_at)
         VALUES (?1, ?2, 'supersede', ?3, ?4, 'memkeeper', ?5, ?6)",
        params![
            event_id,
            superseded_id,
            old_status,
            status::SUPERSEDED,
            format!("superseded by {new_memory_id}"),
            now,
        ],
    )?;
    link_memory(transaction, new_memory_id, superseded_id, "supersedes", now)?;
    link_memory(
        transaction,
        superseded_id,
        new_memory_id,
        "superseded_by",
        now,
    )?;
    Ok(())
}

fn link_memory(
    transaction: &Transaction<'_>,
    src_memory_id: &str,
    dst_memory_id: &str,
    link_type: &str,
    now: &str,
) -> Result<()> {
    ensure_memory_exists(transaction, src_memory_id)?;
    ensure_memory_exists(transaction, dst_memory_id)?;
    transaction.execute(
        "INSERT OR IGNORE INTO memory_links (
            src_memory_id, dst_memory_id, link_type, status, confidence, created_at
         ) VALUES (?1, ?2, ?3, 'active', 1.0, ?4)",
        params![src_memory_id, dst_memory_id, link_type, now],
    )?;
    Ok(())
}

fn ensure_memory_exists(connection: &Connection, id: &str) -> Result<()> {
    let exists: i64 =
        connection.query_row("SELECT COUNT(*) FROM memories WHERE id = ?1", [id], |row| {
            row.get(0)
        })?;
    if exists == 0 {
        return Err(Error::NotFound {
            entity: "memory",
            id: id.to_string(),
        });
    }
    Ok(())
}

pub(crate) fn ensure_memory_in_space(connection: &Connection, id: &str, space: &str) -> Result<()> {
    let exists: i64 = connection.query_row(
        "SELECT COUNT(*) FROM memories WHERE id = ?1 AND space_name = ?2",
        params![id, space],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Err(Error::NotFound {
            entity: "memory",
            id: id.to_string(),
        });
    }
    Ok(())
}

pub(crate) fn load_memory(
    connection: &Connection,
    id: &str,
    options: GetOptions,
) -> Result<MemoryRecord> {
    let mut memory = connection
        .query_row(
            "SELECT
                m.id, m.active_version_id, m.space_name, m.silo_name, m.scope, m.project_key,
                m.kind, m.entity_key, m.claim_key, m.status, m.confidence, m.pinned,
                m.source_episode_id, m.observed_at, m.created_at, m.updated_at, m.valid_from,
                m.valid_to, m.expires_at, m.deleted_at, v.content, v.summary,
                v.content_sha256, v.source_ref_json, m.metadata_json
             FROM memories m
             JOIN memory_versions v ON v.id = m.active_version_id
             WHERE m.id = ?1",
            [id],
            |row| {
                Ok(MemoryRecord {
                    id: row.get(0)?,
                    version_id: row.get(1)?,
                    space: row.get(2)?,
                    silo: row.get(3)?,
                    scope: row.get(4)?,
                    project_key: row.get(5)?,
                    kind: row.get(6)?,
                    entity_key: row.get(7)?,
                    claim_key: row.get(8)?,
                    status: row.get(9)?,
                    confidence: row.get(10)?,
                    pinned: row.get::<_, i64>(11)? == 1,
                    source_episode_id: if options.include_source {
                        row.get(12)?
                    } else {
                        None
                    },
                    observed_at: row.get(13)?,
                    created_at: row.get(14)?,
                    updated_at: row.get(15)?,
                    valid_from: row.get(16)?,
                    valid_to: row.get(17)?,
                    expires_at: row.get(18)?,
                    deleted_at: row.get(19)?,
                    content: row.get(20)?,
                    summary: row.get(21)?,
                    retrieval_representation: None,
                    content_sha256: row.get(22)?,
                    source_ref_json: if options.include_source {
                        row.get(23)?
                    } else {
                        None
                    },
                    metadata_json: row.get(24)?,
                    tags: Vec::new(),
                    versions: None,
                    events: None,
                    links: None,
                })
            },
        )
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "memory",
            id: id.to_string(),
        })?;

    memory.tags = load_tags(connection, id)?;
    memory.retrieval_representation = load_representation(connection, &memory.version_id)?;
    if options.include_history {
        let mut versions = load_versions_limited(connection, id, MAX_HISTORY_LIMIT)?;
        if !options.include_source {
            for version in &mut versions {
                version.source_ref_json = None;
            }
        }
        memory.versions = Some(versions);
        memory.events = Some(load_events_limited(connection, id, MAX_HISTORY_LIMIT)?);
    }
    if options.include_links {
        memory.links = Some(load_links(connection, id)?);
    }
    Ok(memory)
}

fn load_tags(connection: &Connection, id: &str) -> Result<Vec<String>> {
    let mut statement =
        connection.prepare("SELECT tag FROM memory_tags WHERE memory_id = ?1 ORDER BY tag")?;
    let rows = statement.query_map([id], |row| row.get(0))?;
    collect_rows(rows)
}

fn load_versions_limited(
    connection: &Connection,
    id: &str,
    limit: usize,
) -> Result<Vec<MemoryVersionRecord>> {
    let mut statement = connection.prepare(
        "SELECT id, version_num, content, summary, content_sha256, created_at, source_ref_json
         FROM memory_versions WHERE memory_id = ?1 ORDER BY version_num ASC LIMIT ?2",
    )?;
    let rows = statement.query_map(params![id, limit_i64(limit)?], |row| {
        Ok(MemoryVersionRecord {
            id: row.get(0)?,
            version_num: row.get(1)?,
            content: row.get(2)?,
            summary: row.get(3)?,
            retrieval_representation: None,
            content_sha256: row.get(4)?,
            created_at: row.get(5)?,
            source_ref_json: row.get(6)?,
        })
    })?;
    let mut versions = collect_rows(rows)?;
    for version in &mut versions {
        version.retrieval_representation = load_representation(connection, &version.id)?;
    }
    Ok(versions)
}

fn load_events_limited(
    connection: &Connection,
    id: &str,
    limit: usize,
) -> Result<Vec<MemoryEventRecord>> {
    let mut statement = connection.prepare(
        "SELECT id, event_type, old_status, new_status, reason, created_at
         FROM memory_events WHERE memory_id = ?1 ORDER BY created_at ASC, id ASC LIMIT ?2",
    )?;
    let rows = statement.query_map(params![id, limit_i64(limit)?], |row| {
        Ok(MemoryEventRecord {
            id: row.get(0)?,
            event_type: row.get(1)?,
            old_status: row.get(2)?,
            new_status: row.get(3)?,
            reason: row.get(4)?,
            created_at: row.get(5)?,
        })
    })?;
    collect_rows(rows)
}

fn count_for_id(connection: &Connection, table: &str, column: &str, id: &str) -> Result<usize> {
    let sql = format!("SELECT COUNT(*) FROM {table} WHERE {column} = ?1");
    let count: i64 = connection.query_row(&sql, [id], |row| row.get(0))?;
    usize::try_from(count).map_err(|_| Error::InvalidRequest {
        message: "history count overflowed usize".to_string(),
    })
}

fn load_links(connection: &Connection, id: &str) -> Result<Vec<MemoryLinkRecord>> {
    let mut statement = connection.prepare(
        "SELECT src_memory_id, dst_memory_id, link_type, status, confidence
         FROM memory_links
         WHERE src_memory_id = ?1 OR dst_memory_id = ?1
         ORDER BY link_type ASC, src_memory_id ASC, dst_memory_id ASC
         LIMIT ?2",
    )?;
    let rows = statement.query_map(params![id, limit_i64(MAX_GET_LINKS)?], |row| {
        Ok(MemoryLinkRecord {
            src_memory_id: row.get(0)?,
            dst_memory_id: row.get(1)?,
            link_type: row.get(2)?,
            status: row.get(3)?,
            confidence: row.get(4)?,
        })
    })?;
    collect_rows(rows)
}

pub(crate) fn memory_fts_metadata_text(
    project_key: Option<&str>,
    entity_key: Option<&str>,
    claim_key: Option<&str>,
    content: &str,
    summary: Option<&str>,
    tags_text: &str,
) -> String {
    let base_metadata = [project_key, entity_key, claim_key]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
    let normalized_tokens = normalized_search_token_text(&[
        Some(content),
        summary,
        Some(tags_text),
        Some(base_metadata.as_str()),
    ]);
    [base_metadata.as_str(), normalized_tokens.as_str()]
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalized_search_token_text(values: &[Option<&str>]) -> String {
    let mut tokens = BTreeSet::new();
    for value in values.iter().flatten() {
        for term in search_terms(value) {
            for token in normalized_search_tokens_for_term(&term) {
                tokens.insert(token);
            }
        }
    }
    tokens.into_iter().collect::<Vec<_>>().join(" ")
}

fn normalized_search_tokens_for_term(term: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for stem in search_term_stems(term) {
        if stem != term && is_prefixable_search_term(&stem) {
            push_unique(&mut tokens, stem);
        }
    }
    tokens
}

fn event_data_json(
    dry_run: bool,
    supersedes: &[String],
    contradicts: &[String],
    auto_superseded: &[String],
    conflict_candidates: &[RememberConflictCandidate],
) -> String {
    format!(
        "{{\"dry_run\":{dry_run},\"supersedes\":{},\"contradicts\":{},\"auto_superseded\":{},\"conflict_candidates\":{}}}",
        string_array_json(supersedes),
        string_array_json(contradicts),
        string_array_json(auto_superseded),
        remember_conflict_candidates_json(conflict_candidates)
    )
}

fn remember_conflict_candidates_json(candidates: &[RememberConflictCandidate]) -> String {
    let mut output = String::from("[");
    for (index, candidate) in candidates.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"memory_id\":{},\"kind\":{},\"observed_at\":{},\"snippet\":{}}}",
            json_string_for_store(&candidate.memory_id),
            json_string_for_store(&candidate.kind),
            json_string_for_store(&candidate.observed_at),
            json_string_for_store(&candidate.snippet)
        );
    }
    output.push(']');
    output
}

fn forget_event_data_json(dry_run: bool, mode: &str) -> String {
    format!(
        "{{\"dry_run\":{dry_run},\"mode\":{}}}",
        json_string_for_store(mode)
    )
}

/// Build the `data_json` payload stamped on a `correct` event. Captures the
/// corrected memory's provenance (was it synthesis-derived? which session?) plus
/// the optional replacement id, so the nightly synthesis loop can measure
/// correction density and target the sessions whose cards proved wrong without
/// re-deriving any of it from supersession history. Serialized with `serde_json`
/// so the persisted blob is always a valid object.
pub(crate) fn correction_event_data_json(
    dry_run: bool,
    mode: &str,
    corrected_by: Option<&str>,
    tags: &[String],
) -> String {
    let synthesis_derived = tags.iter().any(|tag| tag == "synthesis-derived");
    let session = tags
        .iter()
        .find_map(|tag| tag.strip_prefix("session:"))
        .map(str::to_string);
    serde_json::json!({
        "dry_run": dry_run,
        "mode": mode,
        "corrected_by": corrected_by,
        "synthesis_derived": synthesis_derived,
        "session": session,
    })
    .to_string()
}
