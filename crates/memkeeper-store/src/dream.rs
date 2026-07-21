//! `dream` consolidation extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.
//!
//! Every function here takes `&Transaction<'_>`: the whole subsystem runs
//! inside the single transaction opened by `dream_store`, so a failed task
//! rolls the entire run back.

use std::collections::BTreeSet;

use rusqlite::{params, params_from_iter, types::Value, Connection, Transaction};

use memkeeper_core::{status, DEFAULT_DURABLE_SILO};

use crate::{
    collect_rows, count, ensure_recall_events, ensure_silo_exists, ensure_space_exists,
    json_string_for_store, limit_i64, next_id, normalize_space_name, now_timestamp, rebuild_fts,
    reject_all_spaces_sentinel, string_array_json, upsert_memory_entity_projection,
    upsert_relationship_tx, validate_optional_metadata_value, DreamDedupeReport,
    DreamDuplicateProposal, DreamEntityProjection, DreamExpireReport, DreamGraphReport,
    DreamLinkReport, DreamPromoteReport, DreamReindexReport, DreamRelationshipProposal,
    DreamReport, DreamRequest, Error, PreparedRelationshipUpsertRequest, Result,
    MAX_DREAM_DUPLICATE_IDS, MAX_DREAM_MAX_MEMORIES, SHORT_TERM_SILO,
};

const DREAM_TASK_PROMOTE: &str = "promote";
const DREAM_TASK_EXPIRE: &str = "expire";
const DREAM_TASK_REINDEX: &str = "reindex";
const DREAM_TASK_DEDUPE: &str = "dedupe";
const DREAM_TASK_GRAPH: &str = "graph";
const DREAM_TASK_LINK: &str = "link";
const DREAM_TASK_ALL: &str = "all";

/// A tag this common is not discriminative: linking every pair that shares it
/// would flood the graph, so the `link` task skips tags carried by more than this
/// many active memories.
const DREAM_LINK_MAX_TAG_MEMORIES: usize = 25;
/// Two independently shared topic tags are the minimum evidence for an automatic
/// evidence-backed link. One-tag pairs remain available to explicit/manual linking,
/// but are too noisy for unattended graph growth.
const DREAM_LINK_MIN_SHARED_TAGS: usize = 2;
const DREAM_TASK_ORDER: &[&str] = &[
    DREAM_TASK_PROMOTE,
    DREAM_TASK_EXPIRE,
    DREAM_TASK_REINDEX,
    DREAM_TASK_DEDUPE,
    // `link` writes the cross-entity memory_links that `graph` then projects into
    // weighted relationships, so it must run before `graph` in a combined run.
    DREAM_TASK_LINK,
    DREAM_TASK_GRAPH,
];

#[derive(Debug, Clone, PartialEq)]
struct PreparedDreamRequest {
    space: Option<String>,
    silos: Vec<String>,
    tasks: Vec<String>,
    max_memories: usize,
    dry_run: bool,
    include_pinned: bool,
    promote_threshold: usize,
    promote_score_floor: f64,
    promote_rank_cap: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DreamExpireCandidate {
    id: String,
    pinned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DreamDuplicateGroup {
    space: String,
    content_sha256: String,
    total_count: usize,
    pinned_count: usize,
}

pub(crate) fn validate_dream_request(request: &DreamRequest) -> Result<()> {
    validate_optional_metadata_value("space", request.space.as_deref())?;
    reject_all_spaces_sentinel(request.space.as_deref())?;
    for silo in &request.silos {
        validate_optional_metadata_value("silo", Some(silo))?;
    }
    if !request.silos.is_empty() && request.space.is_none() {
        return Err(Error::InvalidRequest {
            message: "dream silo filters require a space in v0.1".to_string(),
        });
    }
    if request.max_memories == 0 || request.max_memories > MAX_DREAM_MAX_MEMORIES {
        return Err(Error::InvalidRequest {
            message: format!("dream max_memories must be between 1 and {MAX_DREAM_MAX_MEMORIES}"),
        });
    }
    let _ = normalize_dream_tasks(&request.tasks)?;
    if request.promote_threshold == 0 {
        return Err(Error::InvalidRequest {
            message: "dream promote_threshold must be at least 1".to_string(),
        });
    }
    if request.promote_rank_cap == 0 {
        return Err(Error::InvalidRequest {
            message: "dream promote_rank_cap must be at least 1".to_string(),
        });
    }
    if !request.promote_score_floor.is_finite()
        || request.promote_score_floor < 0.0
        || request.promote_score_floor > 2.0
    {
        return Err(Error::InvalidRequest {
            message: "dream promote_score_floor must be between 0.0 and 2.0".to_string(),
        });
    }
    Ok(())
}

fn normalize_dream_tasks(tasks: &[String]) -> Result<Vec<String>> {
    if tasks.is_empty() {
        return Ok(DREAM_TASK_ORDER
            .iter()
            .map(|task| (*task).to_string())
            .collect());
    }
    let mut requested = BTreeSet::new();
    let mut all = false;
    for task in tasks {
        let trimmed = task.trim();
        if trimmed.is_empty() {
            return Err(Error::InvalidRequest {
                message: "dream tasks must be non-empty".to_string(),
            });
        }
        match trimmed {
            DREAM_TASK_ALL => all = true,
            DREAM_TASK_PROMOTE | DREAM_TASK_EXPIRE | DREAM_TASK_REINDEX | DREAM_TASK_DEDUPE
            | DREAM_TASK_LINK | DREAM_TASK_GRAPH => {
                requested.insert(trimmed.to_string());
            }
            _ => {
                return Err(Error::InvalidRequest {
                    message: format!("unsupported dream task: {trimmed}"),
                });
            }
        }
    }
    if all {
        return Ok(DREAM_TASK_ORDER
            .iter()
            .map(|task| (*task).to_string())
            .collect());
    }
    Ok(DREAM_TASK_ORDER
        .iter()
        .filter(|task| requested.contains(**task))
        .map(|task| (*task).to_string())
        .collect())
}

fn prepare_dream_request(
    transaction: &Transaction<'_>,
    request: &DreamRequest,
) -> Result<PreparedDreamRequest> {
    let space = normalize_space_name(request.space.as_deref())?;
    if let Some(space) = &space {
        ensure_space_exists(transaction, space)?;
        for silo in &request.silos {
            ensure_silo_exists(transaction, space, silo)?;
        }
    }
    Ok(PreparedDreamRequest {
        space,
        silos: request.silos.clone(),
        tasks: normalize_dream_tasks(&request.tasks)?,
        max_memories: request.max_memories,
        dry_run: request.dry_run,
        include_pinned: request.include_pinned,
        promote_threshold: request.promote_threshold,
        promote_score_floor: request.promote_score_floor,
        promote_rank_cap: request.promote_rank_cap,
    })
}

pub(crate) fn dream_store_tx(
    transaction: &Transaction<'_>,
    request: &DreamRequest,
) -> Result<DreamReport> {
    let prepared = prepare_dream_request(transaction, request)?;
    let started_at = now_timestamp(transaction)?;
    let run_id = next_id("dream");

    if !prepared.dry_run {
        transaction.execute(
            "INSERT INTO dream_runs (id, space_name, status, started_at, budget_json)
             VALUES (?1, ?2, 'running', ?3, ?4)",
            params![
                &run_id,
                prepared.space.as_deref(),
                &started_at,
                dream_budget_json(&prepared),
            ],
        )?;
    }

    let promote = if dream_has_task(&prepared, DREAM_TASK_PROMOTE) {
        dream_promote_memories(transaction, &prepared, &run_id, &started_at)?
    } else {
        empty_dream_promote_report(false)
    };
    let expire = if dream_has_task(&prepared, DREAM_TASK_EXPIRE) {
        dream_expire_memories(transaction, &prepared, &run_id, &started_at)?
    } else {
        empty_dream_expire_report(false)
    };
    let reindex = if dream_has_task(&prepared, DREAM_TASK_REINDEX) {
        dream_reindex(transaction, prepared.dry_run)?
    } else {
        DreamReindexReport {
            attempted: false,
            memory_rows: 0,
            source_episode_rows: 0,
        }
    };
    let dedupe = if dream_has_task(&prepared, DREAM_TASK_DEDUPE) {
        dream_exact_duplicate_proposals(transaction, &prepared)?
    } else {
        DreamDedupeReport {
            attempted: false,
            proposals: Vec::new(),
            truncated: false,
        }
    };
    // `link` runs before `graph` so the cross-entity links it writes are visible
    // to the graph projection in the same run.
    let link = if dream_has_task(&prepared, DREAM_TASK_LINK) {
        dream_tag_link(transaction, &prepared)?
    } else {
        DreamLinkReport {
            attempted: false,
            candidates: 0,
            links_written: 0,
            truncated: false,
        }
    };
    let graph = if dream_has_task(&prepared, DREAM_TASK_GRAPH) {
        dream_graph_diagnostics(transaction, &prepared)?
    } else {
        DreamGraphReport {
            attempted: false,
            orphan_entities: 0,
            dangling_relationships: 0,
            inactive_evidence_relationships: 0,
            missing_entity_projections: Vec::new(),
            relationship_proposals: Vec::new(),
            truncated: false,
            orphan_entity_ids: Vec::new(),
            dangling_relationship_ids: Vec::new(),
            inactive_evidence_relationship_ids: Vec::new(),
        }
    };

    let finished_at = now_timestamp(transaction)?;
    let report = DreamReport {
        run_id: run_id.clone(),
        space: prepared.space.clone(),
        silos: prepared.silos.clone(),
        tasks: prepared.tasks.clone(),
        status: "succeeded".to_string(),
        started_at,
        finished_at: finished_at.clone(),
        dry_run: prepared.dry_run,
        journaled: !prepared.dry_run,
        max_memories: prepared.max_memories,
        promote,
        expire,
        reindex,
        dedupe,
        graph,
        link,
    };

    if !prepared.dry_run {
        transaction.execute(
            "UPDATE dream_runs
             SET status = 'succeeded', finished_at = ?1, summary_json = ?2
             WHERE id = ?3",
            params![&finished_at, dream_summary_json(&report), &run_id],
        )?;
    }

    Ok(report)
}

fn dream_has_task(request: &PreparedDreamRequest, task: &str) -> bool {
    request.tasks.iter().any(|candidate| candidate == task)
}

fn empty_dream_expire_report(attempted: bool) -> DreamExpireReport {
    DreamExpireReport {
        attempted,
        scanned: 0,
        expired: 0,
        skipped_pinned: 0,
        truncated: false,
        memory_ids: Vec::new(),
        skipped_pinned_ids: Vec::new(),
    }
}

fn empty_dream_promote_report(attempted: bool) -> DreamPromoteReport {
    DreamPromoteReport {
        attempted,
        scanned: 0,
        promoted: 0,
        truncated: false,
        memory_ids: Vec::new(),
    }
}

fn dream_promote_memories(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
    run_id: &str,
    now: &str,
) -> Result<DreamPromoteReport> {
    // recall_events is created lazily; ensure it exists so the aggregate is valid
    // even on stores that have never logged a recall.
    ensure_recall_events(transaction)?;

    // Conditions and bound values are kept in lock-step: every `?` placeholder
    // appended to `conditions` has its value pushed to `values` in the same order.
    let mut conditions = vec![
        "m.status = 'active'".to_string(),
        "m.silo_name = ?".to_string(),
        "(SELECT COUNT(DISTINCT r.session_id) FROM recall_events r \
          WHERE r.memory_id = m.id AND r.kind = 'retrieved' \
            AND r.session_id IS NOT NULL \
            AND r.score >= ? AND r.rank <= ?) >= ?"
            .to_string(),
    ];
    let mut values = vec![
        Value::Text(SHORT_TERM_SILO.to_string()),
        Value::Real(request.promote_score_floor),
        Value::Integer(limit_i64(request.promote_rank_cap)?),
        Value::Integer(limit_i64(request.promote_threshold)?),
    ];
    append_dream_memory_filters(&mut conditions, &mut values, request, "m");
    let sql = format!(
        "SELECT m.id
         FROM memories m
         WHERE {}
         ORDER BY m.id ASC
         LIMIT ?",
        conditions.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(
        request.max_memories.saturating_add(1),
    )?));

    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    let mut candidates = collect_rows(rows)?;
    let truncated = candidates.len() > request.max_memories;
    if truncated {
        candidates.truncate(request.max_memories);
    }

    let mut report = empty_dream_promote_report(true);
    report.truncated = truncated;
    report.scanned = candidates.len();

    for id in candidates {
        report.promoted = report.promoted.saturating_add(1);
        report.memory_ids.push(id.clone());
        if !request.dry_run {
            promote_memory_for_dream(transaction, &id, run_id, now)?;
        }
    }

    Ok(report)
}

fn promote_memory_for_dream(
    transaction: &Transaction<'_>,
    memory_id: &str,
    run_id: &str,
    now: &str,
) -> Result<()> {
    transaction.execute(
        "UPDATE memories
         SET silo_name = ?1, expires_at = NULL, updated_at = ?2
         WHERE id = ?3 AND status = ?4 AND silo_name = ?5",
        params![
            DEFAULT_DURABLE_SILO,
            now,
            memory_id,
            status::ACTIVE,
            SHORT_TERM_SILO
        ],
    )?;
    transaction.execute(
        "UPDATE memory_fts SET silo_name = ?1 WHERE memory_id = ?2",
        params![DEFAULT_DURABLE_SILO, memory_id],
    )?;
    transaction.execute(
        "UPDATE memory_fts_public SET silo_name = ?1 WHERE memory_id = ?2",
        params![DEFAULT_DURABLE_SILO, memory_id],
    )?;
    let event_id = next_id("evt");
    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, old_status, new_status, actor, reason, data_json, created_at)
         VALUES (?1, ?2, 'dream', ?3, ?4, 'memkeeper', ?5, ?6, ?7)",
        params![
            &event_id,
            memory_id,
            status::ACTIVE,
            status::ACTIVE,
            "promoted short-term to durable",
            dream_promote_event_json(run_id),
            now,
        ],
    )?;
    Ok(())
}

fn dream_expire_memories(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
    run_id: &str,
    now: &str,
) -> Result<DreamExpireReport> {
    let mut conditions = vec![
        "m.status = 'active'".to_string(),
        "m.expires_at IS NOT NULL".to_string(),
        "julianday(m.expires_at) <= julianday(?)".to_string(),
    ];
    let mut values = vec![Value::Text(now.to_string())];
    append_dream_memory_filters(&mut conditions, &mut values, request, "m");
    let sql = format!(
        "SELECT m.id, m.pinned
         FROM memories m
         WHERE {}
         ORDER BY m.expires_at ASC, m.id ASC
         LIMIT ?",
        conditions.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(
        request.max_memories.saturating_add(1),
    )?));
    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        Ok(DreamExpireCandidate {
            id: row.get(0)?,
            pinned: row.get::<_, i64>(1)? != 0,
        })
    })?;
    let mut candidates = collect_rows(rows)?;
    let truncated = candidates.len() > request.max_memories;
    if truncated {
        candidates.truncate(request.max_memories);
    }

    let mut report = empty_dream_expire_report(true);
    report.truncated = truncated;
    report.scanned = candidates.len();

    for candidate in candidates {
        if candidate.pinned && !request.include_pinned {
            report.skipped_pinned = report.skipped_pinned.saturating_add(1);
            report.skipped_pinned_ids.push(candidate.id);
            continue;
        }
        report.expired = report.expired.saturating_add(1);
        report.memory_ids.push(candidate.id.clone());
        if !request.dry_run {
            expire_memory_for_dream(transaction, &candidate.id, run_id, now)?;
        }
    }

    Ok(report)
}

fn expire_memory_for_dream(
    transaction: &Transaction<'_>,
    memory_id: &str,
    run_id: &str,
    now: &str,
) -> Result<()> {
    transaction.execute(
        "UPDATE memories SET status = ?1, updated_at = ?2 WHERE id = ?3 AND status = ?4",
        params![status::EXPIRED, now, memory_id, status::ACTIVE],
    )?;
    transaction.execute(
        "UPDATE memory_fts SET status = ?1 WHERE memory_id = ?2",
        params![status::EXPIRED, memory_id],
    )?;
    transaction.execute(
        "UPDATE memory_fts_public SET status = ?1 WHERE memory_id = ?2",
        params![status::EXPIRED, memory_id],
    )?;
    let event_id = next_id("evt");
    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, old_status, new_status, actor, reason, data_json, created_at)
         VALUES (?1, ?2, 'dream', ?3, ?4, 'memkeeper', ?5, ?6, ?7)",
        params![
            &event_id,
            memory_id,
            status::ACTIVE,
            status::EXPIRED,
            "expires_at reached",
            dream_expire_event_json(run_id),
            now,
        ],
    )?;
    Ok(())
}

fn dream_reindex(transaction: &Transaction<'_>, dry_run: bool) -> Result<DreamReindexReport> {
    let (memory_rows, source_episode_rows) = if dry_run {
        (
            count(
                transaction,
                "SELECT COUNT(*) FROM memories m JOIN memory_versions v ON v.id = m.active_version_id",
            )?,
            count(transaction, "SELECT COUNT(*) FROM source_episodes")?,
        )
    } else {
        rebuild_fts(transaction)?
    };
    Ok(DreamReindexReport {
        attempted: true,
        memory_rows,
        source_episode_rows,
    })
}

fn dream_exact_duplicate_proposals(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<DreamDedupeReport> {
    let mut conditions = vec!["m.status = 'active'".to_string()];
    let mut values = Vec::new();
    append_dream_memory_filters(&mut conditions, &mut values, request, "m");
    let sql = format!(
        "SELECT m.space_name, v.content_sha256, COUNT(*), COALESCE(SUM(m.pinned), 0)
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {}
         GROUP BY m.space_name, v.content_sha256
         HAVING COUNT(*) > 1
         ORDER BY m.space_name ASC, v.content_sha256 ASC
         LIMIT ?",
        conditions.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(
        request.max_memories.saturating_add(1),
    )?));
    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        Ok(DreamDuplicateGroup {
            space: row.get(0)?,
            content_sha256: row.get(1)?,
            total_count: usize::try_from(row.get::<_, i64>(2)?).unwrap_or(usize::MAX),
            pinned_count: usize::try_from(row.get::<_, i64>(3)?).unwrap_or(usize::MAX),
        })
    })?;
    let mut groups = collect_rows(rows)?;
    let truncated = groups.len() > request.max_memories;
    if truncated {
        groups.truncate(request.max_memories);
    }

    let proposals = groups
        .iter()
        .map(|group| dream_duplicate_proposal(transaction, request, group))
        .collect::<Result<Vec<_>>>()?;
    Ok(DreamDedupeReport {
        attempted: true,
        proposals,
        truncated,
    })
}

/// `link` task: write cross-entity `memory_links` (`related_to`) between active
/// memories that share at least `DREAM_LINK_MIN_SHARED_TAGS` discriminative topic
/// tags (each carried by at most `DREAM_LINK_MAX_TAG_MEMORIES` active memories), so
/// the `graph` task can project them into weighted relationships and evidence
/// join can bridge curated memories. Session/proposal provenance and generic
/// kind/status bookkeeping tags are excluded because they describe how a memory
/// was captured, not what it means. Existing links are filtered before the
/// per-run cap, making bounded runs resumable instead of repeatedly selecting an
/// already-written first page.
struct DreamTagLinkCandidate {
    src: String,
    dst: String,
    shared_tags: Vec<String>,
}

fn dream_tag_link_candidates(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<(Vec<DreamTagLinkCandidate>, bool)> {
    let mut scope_conditions = vec![
        "m.status = 'active'".to_string(),
        "m.entity_key IS NOT NULL AND TRIM(m.entity_key) <> ''".to_string(),
        "mt.tag NOT LIKE 'session:%'".to_string(),
        "mt.tag NOT LIKE 'proposal-kind-%'".to_string(),
        "LOWER(mt.tag) NOT IN (\
            'action', 'decision', 'summary', 'reference', 'fact', 'lesson',\
            'continuity', 'preference', 'task', 'status', 'done', 'resolved',\
            'review', 'commit', 'synthesis', 'synthesis-derived', 'built',\
            'implemented'\
        )"
        .to_string(),
    ];
    let mut scope_values = Vec::new();
    append_dream_memory_filters(&mut scope_conditions, &mut scope_values, request, "m");
    // Materialize the request-scoped eligible tag rows ONCE, derive
    // discriminative tags per space from that same scope, then self-join the
    // reduced set. A correlated tag-frequency subquery here is catastrophically
    // slow on a real store; CTEs keep it ~1s.
    let select = format!(
        "WITH scoped AS (
             SELECT mt.memory_id AS mid, mt.tag AS tag, m.entity_key AS ek,
                    m.space_name AS sp
               FROM memory_tags mt
               JOIN memories m ON m.id = mt.memory_id
              WHERE {scope_predicate}
         ),
         disc AS (
             SELECT sp, tag
               FROM scoped
              GROUP BY sp, tag
             HAVING COUNT(*) <= {max_tag}
         ),
         tagged AS (
             SELECT scoped.mid, scoped.tag, scoped.ek, scoped.sp
               FROM scoped
               JOIN disc ON disc.sp = scoped.sp AND disc.tag = scoped.tag
         )
         SELECT t1.mid, t2.mid, COUNT(*), json_group_array(t1.tag)
           FROM tagged t1
           JOIN tagged t2 ON t1.tag = t2.tag AND t1.mid < t2.mid
          WHERE t1.ek <> t2.ek
            AND t1.sp = t2.sp
            AND NOT EXISTS (
                SELECT 1
                  FROM memory_links existing
                 WHERE existing.link_type = 'related_to'
                   AND ((existing.src_memory_id = t1.mid AND existing.dst_memory_id = t2.mid)
                     OR (existing.src_memory_id = t2.mid AND existing.dst_memory_id = t1.mid))
            )
          GROUP BY t1.mid, t2.mid
         HAVING COUNT(*) >= {min_shared}
          ORDER BY COUNT(*) DESC, t1.mid ASC, t2.mid ASC
          LIMIT {limit}",
        max_tag = DREAM_LINK_MAX_TAG_MEMORIES,
        min_shared = DREAM_LINK_MIN_SHARED_TAGS,
        limit = request.max_memories.saturating_add(1),
        scope_predicate = scope_conditions.join(" AND "),
    );
    let mut statement = transaction.prepare(&select)?;
    let rows = statement.query_map(params_from_iter(scope_values.iter()), |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            usize::try_from(row.get::<_, i64>(2)?).unwrap_or(usize::MAX),
            row.get::<_, String>(3)?,
        ))
    })?;
    let raw_pairs = collect_rows(rows)?;
    let mut pairs =
        raw_pairs
            .into_iter()
            .map(|(src, dst, shared_tag_count, shared_tags_json)| {
                let mut shared_tags = serde_json::from_str::<Vec<String>>(&shared_tags_json)
                    .map_err(|error| Error::InvalidRequest {
                        message: format!("dream link produced invalid shared-tag JSON: {error}"),
                    })?;
                shared_tags.sort();
                shared_tags.dedup();
                if shared_tags.len() != shared_tag_count {
                    return Err(Error::InvalidRequest {
                        message: "dream link shared-tag evidence count mismatch".to_string(),
                    });
                }
                Ok(DreamTagLinkCandidate {
                    src,
                    dst,
                    shared_tags,
                })
            })
            .collect::<Result<Vec<_>>>()?;
    let truncated = pairs.len() > request.max_memories;
    if truncated {
        pairs.truncate(request.max_memories);
    }
    Ok((pairs, truncated))
}

fn dream_tag_link(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<DreamLinkReport> {
    let (pairs, truncated) = dream_tag_link_candidates(transaction, request)?;
    let candidates = pairs.len();
    if request.dry_run {
        return Ok(DreamLinkReport {
            attempted: true,
            candidates,
            links_written: 0,
            truncated,
        });
    }
    let now = now_timestamp(transaction)?;
    let mut links_written = 0usize;
    for candidate in pairs {
        let DreamTagLinkCandidate {
            src,
            dst,
            shared_tags,
        } = candidate;
        let confidence = relationship_confidence_from_evidence(shared_tags.len());
        let metadata_json = serde_json::json!({
            "source": "dream_tag_link",
            "shared_tag_count": shared_tags.len(),
            "shared_tags": shared_tags,
        })
        .to_string();
        links_written += transaction.execute(
            "INSERT OR IGNORE INTO memory_links \
             (src_memory_id, dst_memory_id, link_type, status, confidence, metadata_json, created_at) \
             VALUES (?1, ?2, 'related_to', 'active', ?3, ?4, ?5)",
            params![src, dst, confidence, metadata_json, now],
        )?;
    }
    Ok(DreamLinkReport {
        attempted: true,
        candidates,
        links_written,
        truncated,
    })
}

fn dream_graph_diagnostics(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<DreamGraphReport> {
    let mut missing_entity_projections =
        dream_graph_missing_entity_projections(transaction, request)?;
    let missing_entity_projections_truncated =
        missing_entity_projections.len() > request.max_memories;
    missing_entity_projections.truncate(request.max_memories);

    // Stage exactly the bounded repair set before deriving relationship
    // proposals. The outer dream transaction rolls back every dry-run, so this
    // yields apply-parity without persisting dry-run mutations or admitting
    // relationships whose missing endpoints fall outside the bound.
    let now = now_timestamp(transaction)?;
    for projection in &missing_entity_projections {
        upsert_memory_entity_projection(
            transaction,
            &projection.space,
            &projection.entity_key,
            None,
            &now,
        )?;
    }

    let mut orphan_entity_ids = dream_graph_orphan_entity_ids(transaction, request)?;
    let mut dangling_relationship_ids =
        dream_graph_dangling_relationship_ids(transaction, request)?;
    let mut inactive_evidence_relationship_ids =
        dream_graph_inactive_evidence_relationship_ids(transaction, request)?;
    let mut relationship_proposals = dream_graph_relationship_proposals(transaction, request)?;

    // The queries fetch max_memories + 1 rows to detect truncation. Truncate
    // before acting so the tombstoned rows and the reported ids are the same
    // set (the sentinel row must not be mutated without being reported).
    let truncated = missing_entity_projections_truncated
        || orphan_entity_ids.len() > request.max_memories
        || dangling_relationship_ids.len() > request.max_memories
        || inactive_evidence_relationship_ids.len() > request.max_memories
        || relationship_proposals.len() > request.max_memories;
    orphan_entity_ids.truncate(request.max_memories);
    dangling_relationship_ids.truncate(request.max_memories);
    inactive_evidence_relationship_ids.truncate(request.max_memories);
    relationship_proposals.truncate(request.max_memories);

    // Reconcile drift: tombstone projection rows that lost their backing --
    // orphan entities (no active memory, no relationships), relationships with a
    // missing or non-active (merged/tombstoned) endpoint entity, and relationships
    // whose evidence memory is gone or inactive. relationship_proposals are
    // link-derived entity edges (deterministic projections of explicit
    // memory_links) that are materialized into the graph on a non-dry-run via
    // apply_graph_relationship_proposals. A dry run reports without mutating
    // (the MCP graph tool path).
    if !request.dry_run {
        let now = now_timestamp(transaction)?;
        apply_graph_relationship_proposals(transaction, &relationship_proposals)?;
        tombstone_graph_rows(
            transaction,
            GraphTable::Relationships,
            &dangling_relationship_ids,
            &now,
        )?;
        tombstone_graph_rows(
            transaction,
            GraphTable::Relationships,
            &inactive_evidence_relationship_ids,
            &now,
        )?;
        tombstone_graph_rows(transaction, GraphTable::Entities, &orphan_entity_ids, &now)?;
    }

    Ok(DreamGraphReport {
        attempted: true,
        orphan_entities: orphan_entity_ids.len(),
        dangling_relationships: dangling_relationship_ids.len(),
        inactive_evidence_relationships: inactive_evidence_relationship_ids.len(),
        missing_entity_projections,
        relationship_proposals,
        truncated,
        orphan_entity_ids,
        dangling_relationship_ids,
        inactive_evidence_relationship_ids,
    })
}

fn dream_graph_missing_entity_projections(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<Vec<DreamEntityProjection>> {
    let mut conditions = vec![
        "m.status = 'active'".to_string(),
        "m.entity_key IS NOT NULL".to_string(),
        "TRIM(m.entity_key) <> ''".to_string(),
        "NOT EXISTS ( \
           SELECT 1 FROM entities e \
            WHERE e.space_name = m.space_name \
              AND e.entity_key = m.entity_key \
         )"
        .to_string(),
    ];
    let mut values = Vec::new();
    append_dream_memory_filters(&mut conditions, &mut values, request, "m");
    let sql = format!(
        "SELECT DISTINCT m.space_name, m.entity_key
           FROM memories m
          WHERE {}
          ORDER BY m.space_name ASC, m.entity_key ASC
          LIMIT {}",
        conditions.join(" AND "),
        request.max_memories.saturating_add(1)
    );
    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        Ok(DreamEntityProjection {
            space: row.get(0)?,
            entity_key: row.get(1)?,
        })
    })?;
    collect_rows(rows)
}

/// Differentiated edge confidence from evidence multiplicity: a saturating map
/// `count / (count + 1)` in `(0, 1)`. One supporting link → 0.5, two → 0.67,
/// three → 0.75, … so a densely co-occurring association out-weights an incidental
/// single-link edge. This is what lets `edge_weight` modulate graph-expansion
/// activation (resolves the uniform-`1.0` debt); the ordering, not the absolute
/// value, is what selects neighbors.
#[must_use]
pub fn relationship_confidence_from_evidence(evidence_count: usize) -> f64 {
    let count = u32::try_from(evidence_count.max(1)).map_or(f64::from(u32::MAX), f64::from);
    count / (count + 1.0)
}

/// Materialize dream-proposed relationships (deterministic projections of explicit
/// `memory_links` onto the entity graph) as active edges. Idempotent via
/// `upsert_relationship_tx`. Confidence is derived from evidence multiplicity (see
/// [`relationship_confidence_from_evidence`]) so well-supported edges carry more
/// activation weight than incidental ones. Runs only on a non-dry-run dream
/// `graph` task.
fn apply_graph_relationship_proposals(
    transaction: &Transaction<'_>,
    proposals: &[DreamRelationshipProposal],
) -> Result<()> {
    for proposal in proposals {
        upsert_relationship_tx(
            transaction,
            &PreparedRelationshipUpsertRequest {
                space: proposal.space.clone(),
                subject_entity_id: None,
                subject_entity_key: Some(proposal.subject_entity_key.clone()),
                relation_type: proposal.relation_type.clone(),
                object_entity_id: None,
                object_entity_key: Some(proposal.object_entity_key.clone()),
                memory_id: Some(proposal.src_memory_id.clone()),
                source_episode_id: None,
                status: status::ACTIVE.to_string(),
                confidence: relationship_confidence_from_evidence(proposal.evidence_count),
                observed_at: None,
                valid_from: None,
                valid_to: None,
                metadata_json: None,
                include_source: false,
            },
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum GraphTable {
    Relationships,
    Entities,
}

/// Tombstone graph projection rows by id (idempotent: only flips `active` rows).
/// `table` is a fixed enum, never user input, so the SQL is a static literal.
fn tombstone_graph_rows(
    transaction: &Transaction<'_>,
    table: GraphTable,
    ids: &[String],
    now: &str,
) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let sql = match table {
        GraphTable::Relationships => {
            "UPDATE relationships SET status = ?1, updated_at = ?2 WHERE id = ?3 AND status = 'active'"
        }
        GraphTable::Entities => {
            "UPDATE entities SET status = ?1, updated_at = ?2 WHERE id = ?3 AND status = 'active'"
        }
    };
    let mut statement = transaction.prepare(sql)?;
    for id in ids {
        statement.execute(params![status::TOMBSTONED, now, id])?;
    }
    Ok(())
}

fn dream_graph_orphan_entity_ids(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<Vec<String>> {
    let (space_predicate, space_values) =
        optional_space_predicate("e.space_name", request.space.as_deref());
    let sql = format!(
        "SELECT e.id
         FROM entities e
         WHERE {space_predicate}
           AND e.status = 'active'
           AND NOT EXISTS (
             SELECT 1 FROM memories m
             WHERE m.space_name = e.space_name AND m.entity_key = e.entity_key AND m.status = 'active'
           )
           AND NOT EXISTS (
             SELECT 1 FROM relationships r
             WHERE r.space_name = e.space_name
              AND (r.subject_entity_id = e.id OR r.object_entity_id = e.id)
              AND r.status = 'active'
           )
         ORDER BY e.space_name ASC, e.entity_key ASC, e.id ASC
         LIMIT {}",
        request.max_memories.saturating_add(1)
    );
    collect_string_query(transaction, &sql, &space_values)
}

fn dream_graph_dangling_relationship_ids(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<Vec<String>> {
    let (space_predicate, space_values) =
        optional_space_predicate("r.space_name", request.space.as_deref());
    let sql = format!(
        "SELECT r.id
         FROM relationships r
         LEFT JOIN entities subject
           ON subject.id = r.subject_entity_id AND subject.space_name = r.space_name
         LEFT JOIN entities object
           ON object.id = r.object_entity_id AND object.space_name = r.space_name
         WHERE {space_predicate}
          AND (
            subject.id IS NULL OR object.id IS NULL
            OR subject.status != 'active' OR object.status != 'active'
          )
          AND r.status = 'active'
         ORDER BY r.space_name ASC, r.id ASC
         LIMIT {}",
        request.max_memories.saturating_add(1)
    );
    collect_string_query(transaction, &sql, &space_values)
}

fn dream_graph_inactive_evidence_relationship_ids(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<Vec<String>> {
    let (space_predicate, space_values) =
        optional_space_predicate("r.space_name", request.space.as_deref());
    let sql = format!(
        "SELECT r.id
         FROM relationships r
         LEFT JOIN memories m
           ON m.id = r.memory_id AND m.space_name = r.space_name
         WHERE {space_predicate}
           AND r.memory_id IS NOT NULL
          AND (m.id IS NULL OR m.status != 'active')
          AND r.status = 'active'
         ORDER BY r.space_name ASC, r.id ASC
         LIMIT {}",
        request.max_memories.saturating_add(1)
    );
    collect_string_query(transaction, &sql, &space_values)
}

/// Optional space filter as a (predicate, bound values) pair; the predicate
/// uses `?1` so callers must bind these values first.
fn optional_space_predicate(column: &str, space: Option<&str>) -> (String, Vec<Value>) {
    space.map_or_else(
        || ("1=1".to_string(), Vec::new()),
        |space| {
            (
                format!("{column} = ?1"),
                vec![Value::Text(space.to_string())],
            )
        },
    )
}

fn collect_string_query(
    connection: &Connection,
    sql: &str,
    values: &[Value],
) -> Result<Vec<String>> {
    let mut statement = connection.prepare(sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    collect_rows(rows)
}

fn dream_graph_relationship_proposals(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<Vec<DreamRelationshipProposal>> {
    let (space_predicate, space_values) =
        optional_space_predicate("src.space_name", request.space.as_deref());
    let sql = format!(
        // Aggregate links by (subject entity, link type, object entity) so an edge
        // backed by many co-occurring memory_links proposes once, carrying the
        // evidence multiplicity that differentiates its confidence (see
        // `apply_graph_relationship_proposals`). MIN() picks a deterministic
        // representative evidence memory for the edge.
        "SELECT src.space_name, MIN(l.src_memory_id), MIN(l.dst_memory_id), l.link_type,
                src.entity_key, l.link_type, dst.entity_key, COUNT(*)
         FROM memory_links l
         JOIN memories src ON src.id = l.src_memory_id
         JOIN memories dst ON dst.id = l.dst_memory_id
         JOIN entities subject ON subject.space_name = src.space_name
              AND subject.entity_key = src.entity_key
              AND subject.status = 'active'
         JOIN entities object ON object.space_name = dst.space_name
              AND object.entity_key = dst.entity_key
              AND object.status = 'active'
         LEFT JOIN relationships existing ON existing.space_name = src.space_name
              AND existing.subject_entity_id = subject.id
              AND existing.relation_type = l.link_type
              AND existing.object_entity_id = object.id
              AND existing.status = 'active'
         WHERE l.status = 'active'
           AND src.status = 'active'
           AND dst.status = 'active'
           AND src.space_name = dst.space_name
           AND src.entity_key IS NOT NULL
           AND TRIM(src.entity_key) <> ''
           AND dst.entity_key IS NOT NULL
           AND TRIM(dst.entity_key) <> ''
           AND src.entity_key <> dst.entity_key
           AND existing.id IS NULL
           AND {space_predicate}
         GROUP BY src.space_name, src.entity_key, l.link_type, dst.entity_key
         ORDER BY src.space_name ASC, l.link_type ASC, src.entity_key ASC, dst.entity_key ASC
         LIMIT {}",
        request.max_memories.saturating_add(1)
    );
    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(space_values.iter()), |row| {
        Ok(DreamRelationshipProposal {
            space: row.get(0)?,
            src_memory_id: row.get(1)?,
            dst_memory_id: row.get(2)?,
            link_type: row.get(3)?,
            subject_entity_key: row.get(4)?,
            relation_type: row.get(5)?,
            object_entity_key: row.get(6)?,
            evidence_count: usize::try_from(row.get::<_, i64>(7)?).unwrap_or(1).max(1),
        })
    })?;
    collect_rows(rows)
}

fn dream_duplicate_proposal(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
    group: &DreamDuplicateGroup,
) -> Result<DreamDuplicateProposal> {
    let mut conditions = vec![
        "m.status = 'active'".to_string(),
        "m.space_name = ?".to_string(),
        "v.content_sha256 = ?".to_string(),
    ];
    let mut values = vec![
        Value::Text(group.space.clone()),
        Value::Text(group.content_sha256.clone()),
    ];
    append_dream_silo_filters(&mut conditions, &mut values, request, "m");
    let sql = format!(
        "SELECT m.id
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {}
         ORDER BY m.pinned DESC, m.id ASC
         LIMIT ?",
        conditions.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(
        MAX_DREAM_DUPLICATE_IDS.saturating_add(2),
    )?));
    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    let ids = collect_rows(rows)?;
    let canonical_memory_id = ids.first().cloned().unwrap_or_default();
    let duplicate_memory_ids = ids
        .iter()
        .skip(1)
        .take(MAX_DREAM_DUPLICATE_IDS)
        .cloned()
        .collect::<Vec<_>>();
    let duplicate_ids_truncated = group.total_count.saturating_sub(1) > duplicate_memory_ids.len();
    Ok(DreamDuplicateProposal {
        space: group.space.clone(),
        content_sha256: group.content_sha256.clone(),
        canonical_memory_id,
        duplicate_memory_ids,
        total_count: group.total_count,
        pinned_count: group.pinned_count,
        duplicate_ids_truncated,
    })
}

fn append_dream_memory_filters(
    conditions: &mut Vec<String>,
    values: &mut Vec<Value>,
    request: &PreparedDreamRequest,
    alias: &str,
) {
    if let Some(space) = &request.space {
        conditions.push(format!("{alias}.space_name = ?"));
        values.push(Value::Text(space.clone()));
    }
    append_dream_silo_filters(conditions, values, request, alias);
}

fn append_dream_silo_filters(
    conditions: &mut Vec<String>,
    values: &mut Vec<Value>,
    request: &PreparedDreamRequest,
    alias: &str,
) {
    if request.silos.is_empty() {
        return;
    }
    let placeholders = std::iter::repeat_n("?", request.silos.len())
        .collect::<Vec<_>>()
        .join(", ");
    conditions.push(format!("{alias}.silo_name IN ({placeholders})"));
    for silo in &request.silos {
        values.push(Value::Text(silo.clone()));
    }
}

fn dream_budget_json(request: &PreparedDreamRequest) -> String {
    format!(
        "{{\"tasks\":{},\"space\":{},\"silos\":{},\"max_memories\":{},\"dry_run\":{},\"include_pinned\":{}}}",
        string_array_json(&request.tasks),
        optional_json_string_for_store(request.space.as_deref()),
        string_array_json(&request.silos),
        request.max_memories,
        request.dry_run,
        request.include_pinned
    )
}

fn dream_summary_json(report: &DreamReport) -> String {
    format!(
        "{{\"status\":{},\"promoted\":{},\"expired\":{},\"skipped_pinned\":{},\"reindex_memory_rows\":{},\"reindex_source_episode_rows\":{},\"duplicate_proposals\":{},\"graph_orphan_entities\":{},\"graph_dangling_relationships\":{},\"graph_inactive_evidence_relationships\":{},\"graph_missing_entity_projections\":{},\"graph_relationship_proposals\":{},\"tag_links_written\":{},\"dry_run\":{}}}",
        json_string_for_store(&report.status),
        report.promote.promoted,
        report.expire.expired,
        report.expire.skipped_pinned,
        report.reindex.memory_rows,
        report.reindex.source_episode_rows,
        report.dedupe.proposals.len(),
        report.graph.orphan_entities,
        report.graph.dangling_relationships,
        report.graph.inactive_evidence_relationships,
        report.graph.missing_entity_projections.len(),
        report.graph.relationship_proposals.len(),
        report.link.links_written,
        report.dry_run
    )
}

fn dream_expire_event_json(run_id: &str) -> String {
    format!(
        "{{\"run_id\":{},\"task\":\"expire\"}}",
        json_string_for_store(run_id)
    )
}

fn dream_promote_event_json(run_id: &str) -> String {
    format!(
        "{{\"run_id\":{},\"task\":\"promote\"}}",
        json_string_for_store(run_id)
    )
}

fn optional_json_string_for_store(value: Option<&str>) -> String {
    value.map_or_else(|| "null".to_string(), json_string_for_store)
}
