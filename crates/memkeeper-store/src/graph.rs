//! `graph` entity/relationship projection extracted from `lib.rs` (pure code
//! movement). Re-exported from the crate root so the public API is unchanged.
//!
//! Scope is the graph CRUD, query, and capture layer. Evidence-graph pool
//! expansion deliberately stays in `lib.rs`: it consumes `PackPoolItem` and is
//! driven from inside the rerank pool builder, so moving it here would make
//! this module depend on the pack hot path.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use rusqlite::{
    params, params_from_iter, types::Value, Connection, OptionalExtension, Row, Transaction,
};

use memkeeper_core::{status, DEFAULT_SPACE};

use crate::{
    collect_rows, ensure_memory_in_space, ensure_source_episode_exists, ensure_space_exists,
    exact_entities_for_span, format_pack_markdown, is_supported_entity_status,
    is_supported_relationship_status, limit_i64, next_id, normalize_filter_values,
    normalize_space_name, normalized_alias, now_timestamp, open_initialized_read_fast,
    open_initialized_write, split_tags, validate_optional_metadata_value,
    validate_optional_timestamp, with_read_snapshot, EntityMergeReport, EntityMergeRequest,
    EntityRecord, EntitySearchReport, EntitySearchRequest, EntitySearchResult, EntityUpsertReport,
    EntityUpsertRequest, Error, GraphCapture, GraphCaptureStatus, GraphContextReport,
    GraphContextRequest, GraphEntityRecord, GraphNeighborsReport, GraphNeighborsRequest,
    GraphRelationshipRecord, JsonValidator, PackReport, PackRequest, RelationshipRecord,
    RelationshipUpsertReport, RelationshipUpsertRequest, RememberRequest, Result, ScoreBreakdown,
    SearchFilters, SearchResult, MAX_CAPTURE_ENTITIES, MAX_CAPTURE_RELATIONSHIPS,
    MAX_ENTITY_ALIASES, MAX_ENTITY_SEARCH_LIMIT, MAX_GRAPH_NEIGHBOR_DEPTH,
    MAX_GRAPH_NEIGHBOR_EDGES, MAX_METADATA_VALUE_CHARS, MAX_PACK_CHARS, MAX_PACK_MEMORIES,
    MAX_SEARCH_OFFSET, MAX_SEARCH_QUERY_CHARS, MAX_SOURCE_REF_JSON_CHARS, MAX_TAGS,
};

/// Create or update one projected graph entity deterministically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the source episode is not in the same space, or `SQLite` rejects the transaction.
pub fn upsert_entity(
    path: impl AsRef<Path>,
    request: &EntityUpsertRequest,
) -> Result<EntityUpsertReport> {
    let prepared = prepare_entity_upsert_request(request)?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = upsert_entity_tx(&transaction, &prepared)?;
    transaction.commit()?;
    Ok(report)
}

/// Create or update one projected graph relationship deterministically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, endpoints or evidence
/// are missing/outside the requested space, the request is invalid, or `SQLite`
/// rejects the transaction.
pub fn upsert_relationship(
    path: impl AsRef<Path>,
    request: &RelationshipUpsertRequest,
) -> Result<RelationshipUpsertReport> {
    let prepared = prepare_relationship_upsert_request(request)?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = upsert_relationship_tx(&transaction, &prepared)?;
    transaction.commit()?;
    Ok(report)
}

/// Merge one projected graph entity into another deterministically.
///
/// Relinks the source entity's active relationships onto the target (collapsing
/// duplicates and self-loops), carries the source's key/name/aliases over as
/// aliases of the target, and tombstones the source entity. With `dry_run` the
/// merge is computed and reported but rolled back.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, either endpoint is
/// unspecified or not found, `from` and `into` resolve to the same entity, or
/// `SQLite` rejects the transaction.
pub fn merge_entity(
    path: impl AsRef<Path>,
    request: &EntityMergeRequest,
) -> Result<EntityMergeReport> {
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = merge_entity_tx(&transaction, request)?;
    if request.dry_run {
        // Drop without committing to roll back the computed (but unwanted) mutation.
        drop(transaction);
    } else {
        transaction.commit()?;
    }
    Ok(report)
}

/// Search projected graph entities deterministically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// or `SQLite` rejects the query.
pub fn search_entities(
    path: impl AsRef<Path>,
    request: &EntitySearchRequest,
) -> Result<EntitySearchReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        search_entities_on_connection(connection, request)
    })
}

/// Traverse projected graph neighbors deterministically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the seed entity does not exist, or `SQLite` rejects the query.
pub fn graph_neighbors(
    path: impl AsRef<Path>,
    request: &GraphNeighborsRequest,
) -> Result<GraphNeighborsReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        graph_neighbors_on_connection(connection, request)
    })
}

/// Build a compact memory context pack around graph neighbors.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the seed entity does not exist, or `SQLite` rejects the query.
pub fn graph_context(
    path: impl AsRef<Path>,
    request: &GraphContextRequest,
) -> Result<GraphContextReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        graph_context_on_connection(connection, request)
    })
}

/// One node in the whole-graph projection: an active entity participating in
/// at least one active relationship, weighted by its total edge count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFullNode {
    /// Stable entity key (node id).
    pub entity_key: String,
    /// Human-friendly name.
    pub canonical_name: String,
    /// Entity type, when set.
    pub entity_type: Option<String>,
    /// Sum of incident active-edge weights (node size hint).
    pub degree: i64,
}

/// One edge in the whole-graph projection: active relationships between two
/// active entities, collapsed and weighted by pair count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFullLink {
    /// Subject entity key.
    pub source: String,
    /// Object entity key.
    pub target: String,
    /// Number of active relationships connecting the pair.
    pub weight: i64,
}

/// The entire connected entity graph for whole-graph visualization. Isolated
/// entities (no active edges) are omitted — they would render as loose dots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFullReport {
    /// Connected entities.
    pub nodes: Vec<GraphFullNode>,
    /// Weighted edges between them.
    pub links: Vec<GraphFullLink>,
}

/// Project the entire connected entity graph (all active relationships between
/// active entities, plus the entities that participate in at least one).
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible or `SQLite` rejects a query.
pub fn graph_full(path: impl AsRef<Path>) -> Result<GraphFullReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, graph_full_on_connection)
}

fn graph_full_on_connection(connection: &Connection) -> Result<GraphFullReport> {
    let mut edge_stmt = connection.prepare(
        "SELECT s.entity_key, o.entity_key, COUNT(*) \
         FROM relationships r \
         JOIN entities s ON s.id = r.subject_entity_id AND s.status = 'active' \
         JOIN entities o ON o.id = r.object_entity_id AND o.status = 'active' \
         WHERE r.status = 'active' \
         GROUP BY s.entity_key, o.entity_key",
    )?;
    let rows = edge_stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;
    let mut links = Vec::new();
    let mut degree: BTreeMap<String, i64> = BTreeMap::new();
    for row in rows {
        let (source, target, weight) = row?;
        *degree.entry(source.clone()).or_insert(0) += weight;
        *degree.entry(target.clone()).or_insert(0) += weight;
        links.push(GraphFullLink {
            source,
            target,
            weight,
        });
    }
    let mut node_stmt = connection.prepare(
        "SELECT entity_key, canonical_name, entity_type FROM entities WHERE status = 'active'",
    )?;
    let node_rows = node_stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
        ))
    })?;
    let mut nodes = Vec::new();
    for row in node_rows {
        let (entity_key, canonical_name, entity_type) = row?;
        if let Some(&deg) = degree.get(&entity_key) {
            nodes.push(GraphFullNode {
                entity_key,
                canonical_name,
                entity_type,
                degree: deg,
            });
        }
    }
    Ok(GraphFullReport { nodes, links })
}
#[derive(Debug, Clone)]
struct PreparedEntityUpsertRequest {
    space: String,
    entity_key: String,
    entity_type: String,
    canonical_name: String,
    aliases: Vec<PreparedEntityAlias>,
    status: String,
    confidence: f64,
    source_episode_id: Option<String>,
    metadata_json: Option<String>,
    include_source: bool,
}

#[derive(Debug, Clone)]
struct PreparedEntityAlias {
    alias: String,
    normalized_alias: String,
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedRelationshipUpsertRequest {
    pub(crate) space: String,
    pub(crate) subject_entity_id: Option<String>,
    pub(crate) subject_entity_key: Option<String>,
    pub(crate) relation_type: String,
    pub(crate) object_entity_id: Option<String>,
    pub(crate) object_entity_key: Option<String>,
    pub(crate) memory_id: Option<String>,
    pub(crate) source_episode_id: Option<String>,
    pub(crate) status: String,
    pub(crate) confidence: f64,
    pub(crate) observed_at: Option<String>,
    pub(crate) valid_from: Option<String>,
    pub(crate) valid_to: Option<String>,
    pub(crate) metadata_json: Option<String>,
    pub(crate) include_source: bool,
}

#[derive(Debug, Clone)]
struct PreparedEntitySearchRequest {
    space: String,
    query: Option<String>,
    normalized_query: Option<String>,
    entity_key: Option<String>,
    entity_types: Vec<String>,
    statuses: Vec<String>,
    limit: usize,
    offset: usize,
    include_source: bool,
}

#[derive(Debug, Clone)]
struct PreparedGraphNeighborsRequest {
    space: String,
    entity_id: Option<String>,
    entity_key: Option<String>,
    depth: usize,
    relation_types: Vec<String>,
    statuses: Vec<String>,
    max_edges: usize,
    include_tombstoned: bool,
    include_source: bool,
}

#[derive(Debug, Clone)]
struct GraphRelationshipRow {
    subject_depth: usize,
    object_depth: usize,
    relationship: RelationshipRecord,
}
fn upsert_entity_tx(
    transaction: &Transaction<'_>,
    request: &PreparedEntityUpsertRequest,
) -> Result<EntityUpsertReport> {
    let now = now_timestamp(transaction)?;
    ensure_space_exists(transaction, &request.space)?;
    if let Some(source_episode_id) = request.source_episode_id.as_deref() {
        ensure_source_episode_exists(transaction, &request.space, source_episode_id)?;
    }

    let existing_id = transaction
        .query_row(
            "SELECT id FROM entities WHERE space_name = ?1 AND entity_key = ?2",
            params![&request.space, &request.entity_key],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let created = existing_id.is_none();
    let entity_id = existing_id.unwrap_or_else(|| next_id("ent"));

    transaction.execute(
        "INSERT INTO entities (
            id, space_name, entity_key, entity_type, canonical_name, status, confidence,
            source_episode_id, metadata_json, created_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)
         ON CONFLICT(space_name, entity_key) DO UPDATE SET
            entity_type = excluded.entity_type,
            canonical_name = excluded.canonical_name,
            status = excluded.status,
            confidence = excluded.confidence,
            source_episode_id = COALESCE(excluded.source_episode_id, entities.source_episode_id),
            metadata_json = COALESCE(excluded.metadata_json, entities.metadata_json),
            updated_at = excluded.updated_at",
        params![
            &entity_id,
            &request.space,
            &request.entity_key,
            &request.entity_type,
            &request.canonical_name,
            &request.status,
            request.confidence,
            request.source_episode_id.as_deref(),
            request.metadata_json.as_deref(),
            &now,
        ],
    )?;

    let canonical_normalized_alias = normalized_alias(&request.canonical_name);
    let mut aliases = request.aliases.clone();
    aliases.retain(|alias| alias.normalized_alias != canonical_normalized_alias);
    aliases.push(PreparedEntityAlias {
        alias: request.canonical_name.clone(),
        normalized_alias: canonical_normalized_alias,
    });
    upsert_entity_aliases(
        transaction,
        &entity_id,
        &aliases,
        request.source_episode_id.as_deref(),
        &now,
    )?;
    let entity = load_entity(transaction, &entity_id, request.include_source)?;

    Ok(EntityUpsertReport {
        strategy: "deterministic_entity_upsert_v0".to_string(),
        created,
        entity,
    })
}

fn upsert_entity_aliases(
    transaction: &Transaction<'_>,
    entity_id: &str,
    aliases: &[PreparedEntityAlias],
    source_episode_id: Option<&str>,
    now: &str,
) -> Result<()> {
    for alias in aliases {
        transaction.execute(
            "INSERT INTO entity_aliases (entity_id, alias, normalized_alias, source_episode_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(entity_id, normalized_alias) DO UPDATE SET
                alias = excluded.alias,
                source_episode_id = COALESCE(excluded.source_episode_id, entity_aliases.source_episode_id)",
            params![
                entity_id,
                &alias.alias,
                &alias.normalized_alias,
                source_episode_id,
                now,
            ],
        )?;
    }
    Ok(())
}

pub(crate) fn upsert_relationship_tx(
    transaction: &Transaction<'_>,
    request: &PreparedRelationshipUpsertRequest,
) -> Result<RelationshipUpsertReport> {
    let now = now_timestamp(transaction)?;
    ensure_space_exists(transaction, &request.space)?;
    let subject_entity_id = resolve_relationship_endpoint(
        transaction,
        &request.space,
        request.subject_entity_id.as_deref(),
        request.subject_entity_key.as_deref(),
        "subject_entity",
    )?;
    let object_entity_id = resolve_relationship_endpoint(
        transaction,
        &request.space,
        request.object_entity_id.as_deref(),
        request.object_entity_key.as_deref(),
        "object_entity",
    )?;
    if let Some(memory_id) = request.memory_id.as_deref() {
        ensure_memory_in_space(transaction, memory_id, &request.space)?;
    }
    if let Some(source_episode_id) = request.source_episode_id.as_deref() {
        ensure_source_episode_exists(transaction, &request.space, source_episode_id)?;
    }

    let existing_id = find_existing_relationship(
        transaction,
        &request.space,
        &subject_entity_id,
        &request.relation_type,
        &object_entity_id,
        request.memory_id.as_deref(),
        request.source_episode_id.as_deref(),
    )?;
    let created = existing_id.is_none();
    let relationship_id = existing_id.unwrap_or_else(|| next_id("rel"));

    transaction.execute(
        "INSERT INTO relationships (
            id, space_name, subject_entity_id, relation_type, object_entity_id,
            memory_id, source_episode_id, status, confidence, observed_at, valid_from, valid_to,
            metadata_json, created_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?14)
         ON CONFLICT(id) DO UPDATE SET
            relation_type = excluded.relation_type,
            memory_id = COALESCE(excluded.memory_id, relationships.memory_id),
            source_episode_id = COALESCE(excluded.source_episode_id, relationships.source_episode_id),
            status = excluded.status,
            confidence = excluded.confidence,
            observed_at = COALESCE(excluded.observed_at, relationships.observed_at),
            valid_from = COALESCE(excluded.valid_from, relationships.valid_from),
            valid_to = COALESCE(excluded.valid_to, relationships.valid_to),
            metadata_json = COALESCE(excluded.metadata_json, relationships.metadata_json),
            updated_at = excluded.updated_at",
        params![
            &relationship_id,
            &request.space,
            &subject_entity_id,
            &request.relation_type,
            &object_entity_id,
            request.memory_id.as_deref(),
            request.source_episode_id.as_deref(),
            &request.status,
            request.confidence,
            request.observed_at.as_deref(),
            request.valid_from.as_deref(),
            request.valid_to.as_deref(),
            request.metadata_json.as_deref(),
            &now,
        ],
    )?;
    let relationship = load_relationship(transaction, &relationship_id, request.include_source)?;
    Ok(RelationshipUpsertReport {
        strategy: "deterministic_relationship_upsert_v0".to_string(),
        created,
        relationship,
    })
}

fn merge_entity_tx(
    transaction: &Transaction<'_>,
    request: &EntityMergeRequest,
) -> Result<EntityMergeReport> {
    let from_entity_id =
        normalize_optional_graph_value("from_entity_id", request.from_entity_id.as_deref())?;
    let from_entity_key =
        normalize_optional_graph_value("from_entity_key", request.from_entity_key.as_deref())?;
    validate_exactly_one_endpoint(
        "from_entity_id",
        from_entity_id.as_ref(),
        "from_entity_key",
        from_entity_key.as_ref(),
    )?;
    let into_entity_id =
        normalize_optional_graph_value("into_entity_id", request.into_entity_id.as_deref())?;
    let into_entity_key =
        normalize_optional_graph_value("into_entity_key", request.into_entity_key.as_deref())?;
    validate_exactly_one_endpoint(
        "into_entity_id",
        into_entity_id.as_ref(),
        "into_entity_key",
        into_entity_key.as_ref(),
    )?;
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    ensure_space_exists(transaction, &space)?;
    let from_id = resolve_relationship_endpoint(
        transaction,
        &space,
        from_entity_id.as_deref(),
        from_entity_key.as_deref(),
        "from_entity",
    )?;
    let into_id = resolve_relationship_endpoint(
        transaction,
        &space,
        into_entity_id.as_deref(),
        into_entity_key.as_deref(),
        "into_entity",
    )?;
    if from_id == into_id {
        return Err(Error::InvalidRequest {
            message: "from and into resolve to the same entity".to_string(),
        });
    }
    let now = now_timestamp(transaction)?;
    let from_entity = load_entity(transaction, &from_id, false)?;
    let into_before = load_entity(transaction, &into_id, false)?;
    if from_entity.status != "active" || into_before.status != "active" {
        return Err(Error::Conflict {
            message: "both from and into entities must be active to merge".to_string(),
        });
    }

    let (repointed, tombstoned_duplicate, tombstoned_self_loop) =
        relink_active_relationships(transaction, &space, &from_id, &into_id, &now)?;

    // Carry the merged-away identifiers over as aliases of the target.
    let alias_count_before = entity_alias_count(transaction, &into_id)?;
    let mut alias_source = vec![
        from_entity.entity_key.clone(),
        from_entity.canonical_name.clone(),
    ];
    alias_source.extend(from_entity.aliases.iter().cloned());
    let prepared_aliases = merge_alias_candidates(&alias_source);
    upsert_entity_aliases(transaction, &into_id, &prepared_aliases, None, &now)?;
    let aliases_added =
        entity_alias_count(transaction, &into_id)?.saturating_sub(alias_count_before);

    // Tombstone the merged-away source entity (audit-preserving).
    transaction.execute(
        "UPDATE entities SET status = 'tombstoned', updated_at = ?1 WHERE id = ?2",
        params![&now, &from_id],
    )?;

    let into = load_entity(transaction, &into_id, request.include_source)?;
    Ok(EntityMergeReport {
        strategy: "deterministic_entity_merge_v0".to_string(),
        dry_run: request.dry_run,
        from_entity_key: from_entity.entity_key,
        into_entity_key: into_before.entity_key,
        relationships_repointed: repointed,
        relationships_tombstoned_duplicate: tombstoned_duplicate,
        relationships_tombstoned_self_loop: tombstoned_self_loop,
        aliases_added,
        from_tombstoned: true,
        into,
    })
}
/// Relink the source entity's active relationships onto the target, collapsing
/// duplicates and self-loops. Returns `(repointed, tombstoned_duplicate,
/// tombstoned_self_loop)`.
fn relink_active_relationships(
    transaction: &Transaction<'_>,
    space: &str,
    from_id: &str,
    into_id: &str,
    now: &str,
) -> Result<(usize, usize, usize)> {
    let edges: Vec<(String, String, String, String)> = {
        let mut statement = transaction.prepare(
            "SELECT id, subject_entity_id, relation_type, object_entity_id
             FROM relationships
             WHERE space_name = ?1 AND status = 'active'
               AND (subject_entity_id = ?2 OR object_entity_id = ?2)
             ORDER BY id ASC",
        )?;
        let rows = statement.query_map(params![space, from_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        collect_rows(rows)?
    };

    let mut repointed = 0usize;
    let mut tombstoned_duplicate = 0usize;
    let mut tombstoned_self_loop = 0usize;
    for (rel_id, subject, relation, object) in edges {
        let new_subject = if subject == from_id {
            into_id
        } else {
            subject.as_str()
        };
        let new_object = if object == from_id {
            into_id
        } else {
            object.as_str()
        };
        if new_subject == new_object {
            tombstone_relationship(transaction, &rel_id, now)?;
            tombstoned_self_loop += 1;
            continue;
        }
        if active_relationship_exists(
            transaction,
            space,
            new_subject,
            &relation,
            new_object,
            &rel_id,
        )? {
            tombstone_relationship(transaction, &rel_id, now)?;
            tombstoned_duplicate += 1;
            continue;
        }
        transaction.execute(
            "UPDATE relationships SET subject_entity_id = ?1, object_entity_id = ?2, updated_at = ?3 WHERE id = ?4",
            params![new_subject, new_object, now, &rel_id],
        )?;
        repointed += 1;
    }
    Ok((repointed, tombstoned_duplicate, tombstoned_self_loop))
}

/// Build deduplicated, non-empty alias candidates for a merge target.
fn merge_alias_candidates(values: &[String]) -> Vec<PreparedEntityAlias> {
    let mut seen = BTreeSet::new();
    let mut prepared = Vec::new();
    for value in values {
        let alias = value.trim();
        if alias.is_empty() {
            continue;
        }
        let normalized = normalized_alias(alias);
        if normalized.is_empty() || !seen.insert(normalized.clone()) {
            continue;
        }
        prepared.push(PreparedEntityAlias {
            alias: alias.to_string(),
            normalized_alias: normalized,
        });
    }
    prepared
}

fn tombstone_relationship(
    transaction: &Transaction<'_>,
    relationship_id: &str,
    now: &str,
) -> Result<()> {
    transaction.execute(
        "UPDATE relationships SET status = 'tombstoned', updated_at = ?1 WHERE id = ?2",
        params![now, relationship_id],
    )?;
    Ok(())
}

fn active_relationship_exists(
    transaction: &Transaction<'_>,
    space: &str,
    subject_entity_id: &str,
    relation_type: &str,
    object_entity_id: &str,
    exclude_id: &str,
) -> Result<bool> {
    let found = transaction
        .query_row(
            "SELECT 1 FROM relationships
             WHERE space_name = ?1 AND status = 'active'
               AND subject_entity_id = ?2 AND relation_type = ?3 AND object_entity_id = ?4
               AND id <> ?5
             LIMIT 1",
            params![
                space,
                subject_entity_id,
                relation_type,
                object_entity_id,
                exclude_id
            ],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    Ok(found.is_some())
}

fn entity_alias_count(transaction: &Transaction<'_>, entity_id: &str) -> Result<usize> {
    let count: i64 = transaction.query_row(
        "SELECT COUNT(*) FROM entity_aliases WHERE entity_id = ?1",
        params![entity_id],
        |row| row.get(0),
    )?;
    Ok(usize::try_from(count).unwrap_or(0))
}

fn search_entities_on_connection(
    connection: &Connection,
    request: &EntitySearchRequest,
) -> Result<EntitySearchReport> {
    let prepared = prepare_entity_search_request(request)?;
    ensure_space_exists(connection, &prepared.space)?;
    let mut results = entity_search_results(connection, &prepared)?;
    let truncated = results.len() > prepared.limit;
    if truncated {
        results.truncate(prepared.limit);
    }
    let total_estimate = if results.is_empty() && !truncated {
        0
    } else {
        prepared
            .offset
            .saturating_add(results.len())
            .saturating_add(usize::from(truncated))
    };
    Ok(EntitySearchReport {
        strategy: "deterministic_entity_search_v0".to_string(),
        space: prepared.space,
        total_estimate,
        truncated,
        results,
    })
}

fn graph_neighbors_on_connection(
    connection: &Connection,
    request: &GraphNeighborsRequest,
) -> Result<GraphNeighborsReport> {
    let prepared = prepare_graph_neighbors_request(request)?;
    ensure_space_exists(connection, &prepared.space)?;
    let seed = resolve_graph_seed(connection, &prepared)?;
    let mut rows = graph_relationship_rows(connection, &seed.id, &prepared)?;
    let truncated = rows.len() > prepared.max_edges;
    if truncated {
        rows.truncate(prepared.max_edges);
    }

    let mut entity_depths = BTreeMap::from([(seed.id.clone(), 0_usize)]);
    for row in &rows {
        entity_depths
            .entry(row.relationship.subject_entity_id.clone())
            .and_modify(|depth| *depth = (*depth).min(row.subject_depth))
            .or_insert(row.subject_depth);
        entity_depths
            .entry(row.relationship.object_entity_id.clone())
            .and_modify(|depth| *depth = (*depth).min(row.object_depth))
            .or_insert(row.object_depth);
    }
    let mut entities = entity_depths
        .into_iter()
        .map(|(entity_id, depth)| {
            load_entity(connection, &entity_id, prepared.include_source)
                .map(|entity| GraphEntityRecord { depth, entity })
        })
        .collect::<Result<Vec<_>>>()?;
    entities.sort_by(|left, right| {
        left.depth
            .cmp(&right.depth)
            .then_with(|| left.entity.entity_key.cmp(&right.entity.entity_key))
            .then_with(|| left.entity.id.cmp(&right.entity.id))
    });

    Ok(GraphNeighborsReport {
        strategy: "deterministic_graph_neighbors_v0".to_string(),
        seed,
        depth: prepared.depth,
        max_edges: prepared.max_edges,
        truncated,
        entities,
        relationships: rows
            .into_iter()
            .map(|row| GraphRelationshipRecord {
                subject_depth: row.subject_depth,
                object_depth: row.object_depth,
                relationship: row.relationship,
            })
            .collect(),
    })
}

fn graph_context_on_connection(
    connection: &Connection,
    request: &GraphContextRequest,
) -> Result<GraphContextReport> {
    validate_graph_context_request(request)?;
    let graph_request = GraphNeighborsRequest {
        space: request.space.clone(),
        entity_id: request.entity_id.clone(),
        entity_key: request.entity_key.clone(),
        depth: request.depth,
        relation_types: request.relation_types.clone(),
        statuses: request.statuses.clone(),
        max_edges: request.max_edges,
        include_tombstoned: request.include_tombstoned,
        include_source: request.include_source,
    };
    let graph = graph_neighbors_on_connection(connection, &graph_request)?;
    let evidence_memory_ids = graph_context_evidence_memory_ids(&graph);
    let entity_memory_ids = graph_context_entity_memory_ids(connection, &graph, request)?;
    let memory_ids = graph_context_memory_ids(&evidence_memory_ids, &entity_memory_ids);
    let results = graph_context_memory_results(connection, &memory_ids, request)?;
    let title = format!("graph context: {}", graph.seed.entity_key);
    let pack_request = PackRequest {
        title,
        queries: vec![graph.seed.entity_key.clone()],
        filters: SearchFilters::default(),
        max_memories: request.max_memories,
        max_chars: request.max_chars,
        format: "markdown".to_string(),
        min_score: 0.0,
        rerank_candidates: 0,
        query_embeddings: None,
        query_token_embeddings: None,
        token_model_id: None,
    };
    let selected_len = results.len().min(request.max_memories);
    let graph_last_synth: Option<String> = connection
        .query_row(
            "SELECT started_at FROM dream_runs WHERE status = 'succeeded' \
             ORDER BY started_at DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let (content, pack_memory_ids, pack_scores, text_truncated) = format_pack_markdown(
        &pack_request,
        &results[..selected_len],
        graph_last_synth.as_deref(),
    );
    Ok(GraphContextReport {
        strategy: "deterministic_graph_context_v0".to_string(),
        graph,
        pack: PackReport {
            title: pack_request.title,
            format: pack_request.format,
            content,
            memory_ids: pack_memory_ids,
            scores: pack_scores,
            truncated: text_truncated || results.len() > request.max_memories,
            top_score: None, // graph-context packs are not scored-retrieval packs
        },
        evidence_memory_ids,
        entity_memory_ids,
    })
}
fn validate_graph_context_request(request: &GraphContextRequest) -> Result<()> {
    let graph_request = GraphNeighborsRequest {
        space: request.space.clone(),
        entity_id: request.entity_id.clone(),
        entity_key: request.entity_key.clone(),
        depth: request.depth,
        relation_types: request.relation_types.clone(),
        statuses: request.statuses.clone(),
        max_edges: request.max_edges,
        include_tombstoned: request.include_tombstoned,
        include_source: request.include_source,
    };
    let _ = prepare_graph_neighbors_request(&graph_request)?;
    if request.max_memories == 0 || request.max_memories > MAX_PACK_MEMORIES {
        return Err(Error::InvalidRequest {
            message: format!("max_memories must be between 1 and {MAX_PACK_MEMORIES}"),
        });
    }
    if request.max_chars == 0 || request.max_chars > MAX_PACK_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("max_chars must be between 1 and {MAX_PACK_CHARS}"),
        });
    }
    Ok(())
}
fn prepare_entity_upsert_request(
    request: &EntityUpsertRequest,
) -> Result<PreparedEntityUpsertRequest> {
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let entity_key = normalize_required_graph_value("entity_key", &request.entity_key)?;
    let entity_type =
        normalize_optional_graph_value("entity_type", request.entity_type.as_deref())?
            .unwrap_or_else(|| "Entity".to_string());
    let canonical_name = normalize_required_graph_value("canonical_name", &request.canonical_name)?;
    let status = normalize_optional_graph_value("status", request.status.as_deref())?
        .unwrap_or_else(|| "active".to_string());
    if !is_supported_entity_status(&status) {
        return Err(Error::InvalidRequest {
            message: format!("unsupported entity status: {status}"),
        });
    }
    if !(0.0..=1.0).contains(&request.confidence) {
        return Err(Error::InvalidRequest {
            message: "confidence must be between 0.0 and 1.0".to_string(),
        });
    }
    let source_episode_id =
        normalize_optional_graph_value("source_episode_id", request.source_episode_id.as_deref())?;
    let metadata_json = match request.metadata_json.as_deref() {
        Some(value) => {
            if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value)
            {
                return Err(Error::InvalidRequest {
                    message: "metadata_json must be a valid JSON object".to_string(),
                });
            }
            Some(value.to_string())
        }
        None => None,
    };
    let aliases = normalize_entity_aliases(&request.aliases)?;
    Ok(PreparedEntityUpsertRequest {
        space,
        entity_key,
        entity_type,
        canonical_name,
        aliases,
        status,
        confidence: request.confidence,
        source_episode_id,
        metadata_json,
        include_source: request.include_source,
    })
}

fn normalize_required_graph_value(name: &str, value: &str) -> Result<String> {
    normalize_optional_graph_value(name, Some(value))?.ok_or_else(|| Error::InvalidRequest {
        message: format!(
            "{name} must be non-empty and at most {MAX_METADATA_VALUE_CHARS} characters"
        ),
    })
}

fn normalize_entity_aliases(values: &[String]) -> Result<Vec<PreparedEntityAlias>> {
    if values.len() > MAX_ENTITY_ALIASES {
        return Err(Error::InvalidRequest {
            message: format!("aliases must contain at most {MAX_ENTITY_ALIASES} values"),
        });
    }
    let mut seen = BTreeSet::new();
    let mut aliases = Vec::with_capacity(values.len());
    for value in values {
        let alias = normalize_required_graph_value("alias", value)?;
        let normalized_alias = normalized_alias(&alias);
        if normalized_alias.is_empty() {
            return Err(Error::InvalidRequest {
                message: "alias must contain searchable text".to_string(),
            });
        }
        if !seen.insert(normalized_alias.clone()) {
            return Err(Error::InvalidRequest {
                message: format!("aliases contain duplicate normalized value: {normalized_alias}"),
            });
        }
        aliases.push(PreparedEntityAlias {
            alias,
            normalized_alias,
        });
    }
    aliases.sort_by(|left, right| {
        left.normalized_alias
            .cmp(&right.normalized_alias)
            .then_with(|| left.alias.cmp(&right.alias))
    });
    Ok(aliases)
}
fn prepare_relationship_upsert_request(
    request: &RelationshipUpsertRequest,
) -> Result<PreparedRelationshipUpsertRequest> {
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let subject_entity_id =
        normalize_optional_graph_value("subject_entity_id", request.subject_entity_id.as_deref())?;
    let subject_entity_key = normalize_optional_graph_value(
        "subject_entity_key",
        request.subject_entity_key.as_deref(),
    )?;
    validate_exactly_one_endpoint(
        "subject_entity_id",
        subject_entity_id.as_ref(),
        "subject_entity_key",
        subject_entity_key.as_ref(),
    )?;
    let object_entity_id =
        normalize_optional_graph_value("object_entity_id", request.object_entity_id.as_deref())?;
    let object_entity_key =
        normalize_optional_graph_value("object_entity_key", request.object_entity_key.as_deref())?;
    validate_exactly_one_endpoint(
        "object_entity_id",
        object_entity_id.as_ref(),
        "object_entity_key",
        object_entity_key.as_ref(),
    )?;
    let relation_type = normalize_required_graph_value("relation_type", &request.relation_type)?;
    let memory_id = normalize_optional_graph_value("memory_id", request.memory_id.as_deref())?;
    let source_episode_id =
        normalize_optional_graph_value("source_episode_id", request.source_episode_id.as_deref())?;
    let status = normalize_optional_graph_value("status", request.status.as_deref())?
        .unwrap_or_else(|| status::ACTIVE.to_string());
    if !is_supported_relationship_status(&status) {
        return Err(Error::InvalidRequest {
            message: format!("unsupported relationship status: {status}"),
        });
    }
    if !(0.0..=1.0).contains(&request.confidence) {
        return Err(Error::InvalidRequest {
            message: "confidence must be between 0.0 and 1.0".to_string(),
        });
    }
    validate_optional_timestamp("observed_at", request.observed_at.as_deref())?;
    validate_optional_timestamp("valid_from", request.valid_from.as_deref())?;
    validate_optional_timestamp("valid_to", request.valid_to.as_deref())?;
    let observed_at = request.observed_at.clone();
    let valid_from = request.valid_from.clone();
    let valid_to = request.valid_to.clone();
    let metadata_json = normalize_optional_json_object(request.metadata_json.as_deref())?;
    Ok(PreparedRelationshipUpsertRequest {
        space,
        subject_entity_id,
        subject_entity_key,
        relation_type,
        object_entity_id,
        object_entity_key,
        memory_id,
        source_episode_id,
        status,
        confidence: request.confidence,
        observed_at,
        valid_from,
        valid_to,
        metadata_json,
        include_source: request.include_source,
    })
}

fn validate_exactly_one_endpoint(
    left_name: &str,
    left: Option<&String>,
    right_name: &str,
    right: Option<&String>,
) -> Result<()> {
    if left.is_some() == right.is_some() {
        return Err(Error::InvalidRequest {
            message: format!("exactly one of {left_name} or {right_name} is required"),
        });
    }
    Ok(())
}

fn normalize_optional_json_object(value: Option<&str>) -> Result<Option<String>> {
    match value {
        Some(value) => {
            if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value)
            {
                return Err(Error::InvalidRequest {
                    message: "metadata_json must be a valid JSON object".to_string(),
                });
            }
            Ok(Some(value.to_string()))
        }
        None => Ok(None),
    }
}

fn prepare_entity_search_request(
    request: &EntitySearchRequest,
) -> Result<PreparedEntitySearchRequest> {
    if request.limit == 0 || request.limit > MAX_ENTITY_SEARCH_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("limit must be between 1 and {MAX_ENTITY_SEARCH_LIMIT}"),
        });
    }
    if request.offset > MAX_SEARCH_OFFSET {
        return Err(Error::InvalidRequest {
            message: format!("offset must be at most {MAX_SEARCH_OFFSET}"),
        });
    }
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let query = normalize_optional_query("query", request.query.as_deref())?;
    let entity_key = normalize_optional_graph_value("entity_key", request.entity_key.as_deref())?;
    let entity_types = normalize_filter_values(&request.entity_types)?;
    let mut statuses = normalize_filter_values(&request.statuses)?;
    if statuses.is_empty() {
        statuses.push("active".to_string());
    }
    for status_value in &statuses {
        if !is_supported_entity_status(status_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported entity status: {status_value}"),
            });
        }
    }
    Ok(PreparedEntitySearchRequest {
        space,
        normalized_query: query.as_deref().map(normalized_alias),
        query,
        entity_key,
        entity_types,
        statuses,
        limit: request.limit,
        offset: request.offset,
        include_source: request.include_source,
    })
}

fn prepare_graph_neighbors_request(
    request: &GraphNeighborsRequest,
) -> Result<PreparedGraphNeighborsRequest> {
    if request.depth == 0 || request.depth > MAX_GRAPH_NEIGHBOR_DEPTH {
        return Err(Error::InvalidRequest {
            message: format!("depth must be between 1 and {MAX_GRAPH_NEIGHBOR_DEPTH}"),
        });
    }
    if request.max_edges == 0 || request.max_edges > MAX_GRAPH_NEIGHBOR_EDGES {
        return Err(Error::InvalidRequest {
            message: format!("max_edges must be between 1 and {MAX_GRAPH_NEIGHBOR_EDGES}"),
        });
    }
    if request.entity_id.is_some() == request.entity_key.is_some() {
        return Err(Error::InvalidRequest {
            message: "exactly one of entity_id or entity_key is required".to_string(),
        });
    }
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let entity_id = normalize_optional_graph_value("entity_id", request.entity_id.as_deref())?;
    let entity_key = normalize_optional_graph_value("entity_key", request.entity_key.as_deref())?;
    let relation_types = normalize_filter_values(&request.relation_types)?;
    let mut statuses = normalize_filter_values(&request.statuses)?;
    if statuses.is_empty() {
        statuses.push("active".to_string());
    }
    for status_value in &statuses {
        if !is_supported_relationship_status(status_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported relationship status: {status_value}"),
            });
        }
    }
    if !request.include_tombstoned && statuses.iter().any(|value| value == "tombstoned") {
        return Err(Error::InvalidRequest {
            message: "include_tombstoned=true is required for tombstoned relationships".to_string(),
        });
    }
    Ok(PreparedGraphNeighborsRequest {
        space,
        entity_id,
        entity_key,
        depth: request.depth,
        relation_types,
        statuses,
        max_edges: request.max_edges,
        include_tombstoned: request.include_tombstoned,
        include_source: request.include_source,
    })
}
fn normalize_optional_query(name: &str, value: Option<&str>) -> Result<Option<String>> {
    value
        .map(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed.chars().count() > MAX_SEARCH_QUERY_CHARS {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "{name} must be non-empty and at most {MAX_SEARCH_QUERY_CHARS} characters"
                    ),
                });
            }
            Ok(trimmed.to_string())
        })
        .transpose()
}

fn normalize_optional_graph_value(name: &str, value: Option<&str>) -> Result<Option<String>> {
    value
        .map(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "{name} must be non-empty and at most {MAX_METADATA_VALUE_CHARS} characters"
                    ),
                });
            }
            Ok(trimmed.to_string())
        })
        .transpose()
}
fn entity_search_results(
    connection: &Connection,
    request: &PreparedEntitySearchRequest,
) -> Result<Vec<EntitySearchResult>> {
    let mut predicates = vec!["e.space_name = ?".to_string()];
    let mut values = vec![Value::Text(request.space.clone())];
    if let Some(entity_key) = &request.entity_key {
        predicates.push("e.entity_key = ?".to_string());
        values.push(Value::Text(entity_key.clone()));
    }
    append_sql_in_filter(
        &mut predicates,
        &mut values,
        "e.entity_type",
        &request.entity_types,
    );
    append_sql_in_filter(&mut predicates, &mut values, "e.status", &request.statuses);
    if let Some(query) = &request.query {
        predicates.push(
            "(instr(lower(e.entity_key), lower(?)) > 0 \
              OR instr(lower(e.canonical_name), lower(?)) > 0 \
              OR EXISTS (SELECT 1 FROM entity_aliases ea \
                         WHERE ea.entity_id = e.id AND instr(ea.normalized_alias, ?) > 0))"
                .to_string(),
        );
        values.push(Value::Text(query.clone()));
        values.push(Value::Text(query.clone()));
        values.push(Value::Text(
            request.normalized_query.clone().unwrap_or_default(),
        ));
    }
    let sql = format!(
        "SELECT e.id, e.space_name, e.entity_key, e.entity_type, e.canonical_name,
                e.status, e.confidence, e.source_episode_id, e.created_at, e.updated_at
         FROM entities e
         WHERE {}
         ORDER BY e.status ASC, e.entity_key ASC, e.id ASC
         LIMIT ? OFFSET ?",
        predicates.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(request.limit.saturating_add(1))?));
    values.push(Value::Integer(limit_i64(request.offset)?));
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        entity_from_row(row, request.include_source)
    })?;
    let mut entities = collect_rows(rows)?;
    for entity in &mut entities {
        entity.aliases = load_entity_aliases(connection, &entity.id)?;
    }
    entities
        .into_iter()
        .enumerate()
        .map(|(index, entity)| {
            matched_entity_aliases(connection, &entity.id, request.normalized_query.as_deref()).map(
                |matched_aliases| EntitySearchResult {
                    rank: request.offset + index + 1,
                    entity,
                    matched_aliases,
                },
            )
        })
        .collect()
}

fn append_sql_in_filter(
    predicates: &mut Vec<String>,
    values: &mut Vec<Value>,
    column: &str,
    filter_values: &[String],
) {
    if filter_values.is_empty() {
        return;
    }
    let placeholders = std::iter::repeat_n("?", filter_values.len())
        .collect::<Vec<_>>()
        .join(", ");
    predicates.push(format!("{column} IN ({placeholders})"));
    values.extend(filter_values.iter().cloned().map(Value::Text));
}

fn resolve_graph_seed(
    connection: &Connection,
    request: &PreparedGraphNeighborsRequest,
) -> Result<EntityRecord> {
    let mut predicates = vec!["space_name = ?".to_string()];
    let mut values = vec![Value::Text(request.space.clone())];
    if let Some(entity_id) = &request.entity_id {
        predicates.push("id = ?".to_string());
        values.push(Value::Text(entity_id.clone()));
    }
    if let Some(entity_key) = &request.entity_key {
        predicates.push("entity_key = ?".to_string());
        values.push(Value::Text(entity_key.clone()));
    }
    if !request.include_tombstoned {
        predicates.push("status != 'tombstoned'".to_string());
    }
    let sql = format!(
        "SELECT id FROM entities WHERE {} ORDER BY id ASC LIMIT 1",
        predicates.join(" AND ")
    );
    let entity_id = connection
        .query_row(&sql, params_from_iter(values.iter()), |row| {
            row.get::<_, String>(0)
        })
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "entity",
            id: request
                .entity_id
                .as_deref()
                .or(request.entity_key.as_deref())
                .unwrap_or_default()
                .to_string(),
        })?;
    load_entity(connection, &entity_id, request.include_source)
}

fn graph_relationship_rows(
    connection: &Connection,
    seed_id: &str,
    request: &PreparedGraphNeighborsRequest,
) -> Result<Vec<GraphRelationshipRow>> {
    // All string values below bind as `?` parameters; the IN-lists expand to a
    // matching run of placeholders. `space`, `statuses`, and the relation-type
    // list each appear twice (recursive term + final select), so the bind order
    // repeats them in that order. `depth`/`limit` are validated integers, not
    // user strings, so they stay interpolated.
    let status_placeholders = std::iter::repeat_n("?", request.statuses.len())
        .collect::<Vec<_>>()
        .join(",");
    let relation_type_predicate = if request.relation_types.is_empty() {
        String::new()
    } else {
        let placeholders = std::iter::repeat_n("?", request.relation_types.len())
            .collect::<Vec<_>>()
            .join(",");
        format!(" AND r.relation_type IN ({placeholders})")
    };
    let entity_status_predicate = if request.include_tombstoned {
        String::new()
    } else {
        " AND next_entity.status != 'tombstoned'".to_string()
    };
    let edge_entity_status_predicate = if request.include_tombstoned {
        String::new()
    } else {
        " AND subject_entity.status != 'tombstoned' AND object_entity.status != 'tombstoned'"
            .to_string()
    };
    let sql = format!(
        "WITH RECURSIVE walk(entity_id, depth) AS (
             SELECT ?, 0
             UNION
             SELECT CASE
                      WHEN r.subject_entity_id = walk.entity_id THEN r.object_entity_id
                      ELSE r.subject_entity_id
                    END,
                    walk.depth + 1
               FROM walk
               JOIN relationships r
                 ON r.space_name = ?
                AND (r.subject_entity_id = walk.entity_id OR r.object_entity_id = walk.entity_id)
               JOIN entities next_entity
                 ON next_entity.id = CASE
                      WHEN r.subject_entity_id = walk.entity_id THEN r.object_entity_id
                      ELSE r.subject_entity_id
                    END
               LEFT JOIN memories evidence_memory ON evidence_memory.id = r.memory_id
              WHERE walk.depth < {depth}
                AND r.status IN ({status_placeholders})
                {relation_type_predicate}
                {entity_status_predicate}
                AND (r.memory_id IS NULL OR evidence_memory.status = 'active')
           ),
           nodes AS (
             SELECT entity_id, MIN(depth) AS depth FROM walk GROUP BY entity_id
           )
         SELECT r.id, r.space_name, r.subject_entity_id, r.relation_type, r.object_entity_id,
                r.memory_id, r.source_episode_id, r.status, r.confidence, r.observed_at,
                r.valid_from, r.valid_to, r.created_at, r.updated_at, s.depth, o.depth
           FROM relationships r
           JOIN nodes s ON s.entity_id = r.subject_entity_id
           JOIN nodes o ON o.entity_id = r.object_entity_id
           JOIN entities subject_entity ON subject_entity.id = r.subject_entity_id
           JOIN entities object_entity ON object_entity.id = r.object_entity_id
           LEFT JOIN memories evidence_memory ON evidence_memory.id = r.memory_id
          WHERE r.space_name = ?
            AND r.status IN ({status_placeholders})
            {relation_type_predicate}
            {edge_entity_status_predicate}
            AND (r.memory_id IS NULL OR evidence_memory.status = 'active')
          ORDER BY min(s.depth, o.depth) ASC, max(s.depth, o.depth) ASC, r.relation_type ASC, r.id ASC
          LIMIT {limit}",
        depth = request.depth,
        limit = request.max_edges.saturating_add(1),
    );
    let mut binds: Vec<&str> = Vec::new();
    binds.push(seed_id);
    binds.push(request.space.as_str());
    binds.extend(request.statuses.iter().map(String::as_str));
    binds.extend(request.relation_types.iter().map(String::as_str));
    binds.push(request.space.as_str());
    binds.extend(request.statuses.iter().map(String::as_str));
    binds.extend(request.relation_types.iter().map(String::as_str));
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(binds.iter()), |row| {
        graph_relationship_row_from_row(row, request)
    })?;
    collect_rows(rows)
}

fn graph_context_evidence_memory_ids(graph: &GraphNeighborsReport) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for relationship in &graph.relationships {
        if let Some(memory_id) = &relationship.relationship.memory_id {
            if seen.insert(memory_id.clone()) {
                ids.push(memory_id.clone());
            }
        }
    }
    ids
}

fn graph_context_entity_memory_ids(
    connection: &Connection,
    graph: &GraphNeighborsReport,
    request: &GraphContextRequest,
) -> Result<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for entity in &graph.entities {
        if seen.contains(&entity.entity.entity_key) {
            continue;
        }
        seen.insert(entity.entity.entity_key.clone());
        let mut statement = connection.prepare(
            "SELECT id
             FROM memories
             WHERE space_name = ?1
               AND status = 'active'
               AND entity_key = ?2
             ORDER BY pinned DESC, observed_at DESC, updated_at DESC, id ASC
             LIMIT ?3",
        )?;
        let limit = limit_i64(request.max_memories.saturating_add(1))?;
        let rows = statement.query_map(
            params![&entity.entity.space, &entity.entity.entity_key, limit],
            |row| row.get::<_, String>(0),
        )?;
        for memory_id in collect_rows(rows)? {
            if ids.len()
                >= request
                    .max_memories
                    .saturating_add(graph.relationships.len())
            {
                break;
            }
            if ids.iter().all(|existing| existing != &memory_id) {
                ids.push(memory_id);
            }
        }
    }
    Ok(ids)
}

fn graph_context_memory_ids(evidence_ids: &[String], entity_ids: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for memory_id in evidence_ids.iter().chain(entity_ids.iter()) {
        if seen.insert(memory_id.clone()) {
            ids.push(memory_id.clone());
        }
    }
    ids
}

fn graph_context_memory_results(
    connection: &Connection,
    memory_ids: &[String],
    request: &GraphContextRequest,
) -> Result<Vec<SearchResult>> {
    memory_ids
        .iter()
        .filter_map(|memory_id| {
            match graph_context_memory_result(connection, memory_id, request.include_source) {
                Ok(Some(result)) => Some(Ok(result)),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .enumerate()
        .map(|(index, result)| {
            let mut result = result?;
            result.rank = index + 1;
            Ok(result)
        })
        .collect()
}

fn graph_context_memory_result(
    connection: &Connection,
    memory_id: &str,
    include_source: bool,
) -> Result<Option<SearchResult>> {
    let source_sql = if include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let sql = format!(
        "SELECT
            m.id, m.active_version_id, m.space_name, m.silo_name, m.scope, m.project_key,
            m.kind, m.status, v.summary, substr(v.content, 1, 1000),
            COALESCE((SELECT group_concat(tag, char(31)) FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)), ''),
            m.entity_key, m.claim_key, m.observed_at, {source_sql}
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.id = ?1 AND m.status = 'active'"
    );
    connection
        .query_row(&sql, [memory_id], |row| {
            let tags_joined = row.get::<_, Option<String>>(10)?.unwrap_or_default();
            Ok(SearchResult {
                rank: 0,
                memory_id: row.get(0)?,
                version_id: row.get(1)?,
                score: 0.0,
                scores: ScoreBreakdown {
                    fts: 0.0,
                    metadata: 0.0,
                    recency: 0.0,
                    scope: 0.0,
                    status: 0.0,
                    pin: 0.0,
                    source_tier: 0.0,
                },
                space: row.get(2)?,
                silo: row.get(3)?,
                scope: row.get(4)?,
                project_key: row.get(5)?,
                kind: row.get(6)?,
                status: row.get(7)?,
                summary: row.get(8)?,
                snippet: row.get(9)?,
                content: None,
                tags: split_tags(&tags_joined),
                entity_key: row.get(11)?,
                claim_key: row.get(12)?,
                observed_at: row.get(13)?,
                source_ref_json: row.get(14)?,
                metadata_json: None,
            })
        })
        .optional()
        .map_err(Into::into)
}

fn graph_relationship_row_from_row(
    row: &Row<'_>,
    request: &PreparedGraphNeighborsRequest,
) -> rusqlite::Result<GraphRelationshipRow> {
    let source_episode_id = if request.include_source {
        row.get(6)?
    } else {
        None
    };
    Ok(GraphRelationshipRow {
        relationship: RelationshipRecord {
            id: row.get(0)?,
            space: row.get(1)?,
            subject_entity_id: row.get(2)?,
            relation_type: row.get(3)?,
            object_entity_id: row.get(4)?,
            memory_id: row.get(5)?,
            source_episode_id,
            status: row.get(7)?,
            confidence: row.get(8)?,
            observed_at: row.get(9)?,
            valid_from: row.get(10)?,
            valid_to: row.get(11)?,
            created_at: row.get(12)?,
            updated_at: row.get(13)?,
        },
        subject_depth: usize::try_from(row.get::<_, i64>(14)?).unwrap_or(usize::MAX),
        object_depth: usize::try_from(row.get::<_, i64>(15)?).unwrap_or(usize::MAX),
    })
}

fn resolve_relationship_endpoint(
    connection: &Connection,
    space: &str,
    entity_id: Option<&str>,
    entity_key: Option<&str>,
    label: &'static str,
) -> Result<String> {
    let mut predicates = vec!["space_name = ?".to_string()];
    let mut values = vec![Value::Text(space.to_string())];
    if let Some(entity_id) = entity_id {
        predicates.push("id = ?".to_string());
        values.push(Value::Text(entity_id.to_string()));
    }
    if let Some(entity_key) = entity_key {
        predicates.push("entity_key = ?".to_string());
        values.push(Value::Text(entity_key.to_string()));
    }
    let sql = format!(
        "SELECT id FROM entities WHERE {} ORDER BY id ASC LIMIT 1",
        predicates.join(" AND ")
    );
    connection
        .query_row(&sql, params_from_iter(values.iter()), |row| {
            row.get::<_, String>(0)
        })
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: label,
            id: entity_id.or(entity_key).unwrap_or_default().to_string(),
        })
}

fn find_existing_relationship(
    connection: &Connection,
    space: &str,
    subject_entity_id: &str,
    relation_type: &str,
    object_entity_id: &str,
    memory_id: Option<&str>,
    source_episode_id: Option<&str>,
) -> Result<Option<String>> {
    connection
        .query_row(
            "SELECT id FROM relationships
             WHERE space_name = ?1
               AND subject_entity_id = ?2
               AND relation_type = ?3
               AND object_entity_id = ?4
               AND ((memory_id IS NULL AND ?5 IS NULL) OR memory_id = ?5)
               AND ((source_episode_id IS NULL AND ?6 IS NULL) OR source_episode_id = ?6)
             ORDER BY id ASC
             LIMIT 1",
            params![
                space,
                subject_entity_id,
                relation_type,
                object_entity_id,
                memory_id,
                source_episode_id,
            ],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(Into::into)
}

fn load_relationship(
    connection: &Connection,
    relationship_id: &str,
    include_source: bool,
) -> Result<RelationshipRecord> {
    connection
        .query_row(
            "SELECT id, space_name, subject_entity_id, relation_type, object_entity_id,
                    memory_id, source_episode_id, status, confidence, observed_at,
                    valid_from, valid_to, created_at, updated_at
             FROM relationships
             WHERE id = ?1",
            [relationship_id],
            |row| relationship_from_row(row, include_source),
        )
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "relationship",
            id: relationship_id.to_string(),
        })
}

fn relationship_from_row(
    row: &Row<'_>,
    include_source: bool,
) -> rusqlite::Result<RelationshipRecord> {
    let source_episode_id = if include_source { row.get(6)? } else { None };
    Ok(RelationshipRecord {
        id: row.get(0)?,
        space: row.get(1)?,
        subject_entity_id: row.get(2)?,
        relation_type: row.get(3)?,
        object_entity_id: row.get(4)?,
        memory_id: row.get(5)?,
        source_episode_id,
        status: row.get(7)?,
        confidence: row.get(8)?,
        observed_at: row.get(9)?,
        valid_from: row.get(10)?,
        valid_to: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
    })
}

fn load_entity(
    connection: &Connection,
    entity_id: &str,
    include_source: bool,
) -> Result<EntityRecord> {
    // Cached for the same reason as `load_entity_aliases`: callers loop over an
    // entity set, and the SQL text is fixed. Idiom consistency, not a measured
    // win -- the per-entity cost is well under a millisecond.
    let mut statement = connection.prepare_cached(
        "SELECT id, space_name, entity_key, entity_type, canonical_name, status, confidence,
                source_episode_id, created_at, updated_at
         FROM entities
         WHERE id = ?1",
    )?;
    let mut entity = statement
        .query_row([entity_id], |row| entity_from_row(row, include_source))
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "entity",
            id: entity_id.to_string(),
        })?;
    entity.aliases = load_entity_aliases(connection, &entity.id)?;
    Ok(entity)
}

fn entity_from_row(row: &Row<'_>, include_source: bool) -> rusqlite::Result<EntityRecord> {
    let source_episode_id = if include_source { row.get(7)? } else { None };
    Ok(EntityRecord {
        id: row.get(0)?,
        space: row.get(1)?,
        entity_key: row.get(2)?,
        entity_type: row.get(3)?,
        canonical_name: row.get(4)?,
        status: row.get(5)?,
        confidence: row.get(6)?,
        aliases: Vec::new(),
        source_episode_id,
        created_at: row.get(8)?,
        updated_at: row.get(9)?,
    })
}

fn load_entity_aliases(connection: &Connection, entity_id: &str) -> Result<Vec<String>> {
    // Cached because callers loop over an entity set; the SQL text is fixed, so
    // this is idiom consistency with the rest of the file rather than a
    // measured win (the per-entity saving is well under a millisecond).
    let mut statement = connection.prepare_cached(
        "SELECT alias
         FROM entity_aliases
         WHERE entity_id = ?1
         ORDER BY normalized_alias ASC, alias ASC",
    )?;
    let rows = statement.query_map([entity_id], |row| row.get::<_, String>(0))?;
    collect_rows(rows)
}

fn matched_entity_aliases(
    connection: &Connection,
    entity_id: &str,
    normalized_query: Option<&str>,
) -> Result<Vec<String>> {
    let Some(normalized_query) = normalized_query.filter(|value| !value.is_empty()) else {
        return Ok(Vec::new());
    };
    let mut statement = connection.prepare(
        "SELECT alias
         FROM entity_aliases
         WHERE entity_id = ?1 AND instr(normalized_alias, ?2) > 0
         ORDER BY normalized_alias ASC, alias ASC
         LIMIT ?3",
    )?;
    let rows = statement.query_map(
        params![entity_id, normalized_query, limit_i64(MAX_TAGS)?],
        |row| row.get::<_, String>(0),
    )?;
    collect_rows(rows)
}
pub(crate) fn validate_graph_capture(graph: Option<&GraphCapture>) -> Result<()> {
    let Some(graph) = graph else {
        return Ok(());
    };
    if graph.entities.is_empty() || graph.entities.len() > MAX_CAPTURE_ENTITIES {
        return Err(Error::InvalidRequest {
            message: format!("graph.entities must contain 1..={MAX_CAPTURE_ENTITIES} values"),
        });
    }
    if graph.relationships.len() > MAX_CAPTURE_RELATIONSHIPS {
        return Err(Error::InvalidRequest {
            message: format!(
                "graph.relationships must contain at most {MAX_CAPTURE_RELATIONSHIPS} values"
            ),
        });
    }
    validate_optional_metadata_value("graph.extractor", Some(&graph.extractor))?;
    validate_optional_metadata_value(
        "graph.extractor_version",
        graph.extractor_version.as_deref(),
    )?;
    let mut keys = BTreeSet::new();
    for entity in &graph.entities {
        let request = EntityUpsertRequest {
            space: None,
            entity_key: entity.entity_key.clone(),
            entity_type: Some(entity.entity_type.clone()),
            canonical_name: entity.canonical_name.clone(),
            aliases: entity.aliases.clone(),
            status: Some(status::ACTIVE.to_string()),
            confidence: 1.0,
            source_episode_id: None,
            metadata_json: None,
            include_source: false,
        };
        let prepared = prepare_entity_upsert_request(&request)?;
        if !keys.insert(prepared.entity_key) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "graph.entities contains duplicate entity_key: {}",
                    entity.entity_key
                ),
            });
        }
    }
    let mut triples = BTreeSet::new();
    for relationship in &graph.relationships {
        if relationship.relation_type == "related_to" {
            return Err(Error::InvalidRequest {
                message: "graph relationships must use a typed predicate, not related_to"
                    .to_string(),
            });
        }
        if relationship.subject_entity_key == relationship.object_entity_key {
            return Err(Error::InvalidRequest {
                message: "graph relationship endpoints must be different entities".to_string(),
            });
        }
        if !keys.contains(relationship.subject_entity_key.trim())
            || !keys.contains(relationship.object_entity_key.trim())
        {
            return Err(Error::InvalidRequest {
                message: "graph relationship endpoints must be declared in graph.entities"
                    .to_string(),
            });
        }
        if !relationship.confidence.is_finite() || !(0.0..=1.0).contains(&relationship.confidence) {
            return Err(Error::InvalidRequest {
                message: "graph relationship confidence must be between 0.0 and 1.0".to_string(),
            });
        }
        let triple = (
            relationship.subject_entity_key.trim(),
            relationship.relation_type.trim(),
            relationship.object_entity_key.trim(),
        );
        if !triples.insert(triple) {
            return Err(Error::InvalidRequest {
                message: "graph.relationships contains a duplicate typed edge".to_string(),
            });
        }
    }
    Ok(())
}
pub(crate) fn apply_graph_capture(
    transaction: &Transaction<'_>,
    space: &str,
    memory_id: &str,
    remember: &RememberRequest,
    graph: &GraphCapture,
) -> Result<GraphCaptureStatus> {
    let entity_metadata = serde_json::json!({
        "origin": "automatic_capture",
        "extractor": graph.extractor,
        "extractor_version": graph.extractor_version,
    })
    .to_string();
    let resolved_keys = resolve_graph_capture_keys(transaction, space, graph)?;
    upsert_graph_capture_entities(
        transaction,
        space,
        remember,
        graph,
        &resolved_keys,
        &entity_metadata,
    )?;
    upsert_graph_capture_relationships(
        transaction,
        space,
        memory_id,
        remember,
        graph,
        &resolved_keys,
    )?;
    Ok(GraphCaptureStatus {
        routing_contract: "evidence_join_v2",
        entities: graph.entities.len(),
        relationships: graph.relationships.len(),
    })
}

fn resolve_graph_capture_keys(
    transaction: &Transaction<'_>,
    space: &str,
    graph: &GraphCapture,
) -> Result<BTreeMap<String, String>> {
    let mut resolved_keys = BTreeMap::new();
    let mut seen_resolved = BTreeSet::new();
    for entity in &graph.entities {
        let exact_key = transaction
            .query_row(
                "SELECT entity_key
                   FROM entities
                  WHERE space_name = ?1 AND entity_key = ?2 AND status = 'active'",
                params![space, entity.entity_key],
                |row| row.get(0),
            )
            .optional()?;
        let resolved_key = if let Some(exact_key) = exact_key {
            exact_key
        } else {
            let mut matches = BTreeSet::new();
            for surface in std::iter::once(entity.canonical_name.as_str())
                .chain(entity.aliases.iter().map(String::as_str))
            {
                for (_, entity_key, _) in exact_entities_for_span(
                    transaction,
                    &[space.to_string()],
                    &normalized_alias(surface),
                )? {
                    matches.insert(entity_key);
                }
            }
            if matches.len() > 1 {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "graph entity {} matches multiple canonical entities",
                        entity.entity_key
                    ),
                });
            }
            matches
                .into_iter()
                .next()
                .unwrap_or_else(|| entity.entity_key.clone())
        };
        if !seen_resolved.insert(resolved_key.clone()) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "multiple graph entities resolve to canonical entity {resolved_key}"
                ),
            });
        }
        resolved_keys.insert(entity.entity_key.clone(), resolved_key);
    }
    Ok(resolved_keys)
}

fn upsert_graph_capture_entities(
    transaction: &Transaction<'_>,
    space: &str,
    remember: &RememberRequest,
    graph: &GraphCapture,
    resolved_keys: &BTreeMap<String, String>,
    entity_metadata: &str,
) -> Result<()> {
    for entity in &graph.entities {
        let resolved_key = &resolved_keys[&entity.entity_key];
        let existing: Option<(Option<String>, String)> = transaction
            .query_row(
                "SELECT entity_type, canonical_name
                   FROM entities
                  WHERE space_name = ?1 AND entity_key = ?2 AND status = 'active'",
                params![space, resolved_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        let (entity_type, canonical_name, aliases) =
            if let Some((existing_type, existing_name)) = existing {
                let mut aliases = entity.aliases.clone();
                aliases.push(entity.canonical_name.clone());
                (
                    existing_type.or_else(|| Some(entity.entity_type.clone())),
                    existing_name,
                    aliases,
                )
            } else {
                (
                    Some(entity.entity_type.clone()),
                    entity.canonical_name.clone(),
                    entity.aliases.clone(),
                )
            };
        let prepared = prepare_entity_upsert_request(&EntityUpsertRequest {
            space: Some(space.to_string()),
            entity_key: resolved_key.clone(),
            entity_type,
            canonical_name,
            aliases,
            status: Some(status::ACTIVE.to_string()),
            confidence: remember.confidence,
            source_episode_id: remember.source_episode_id.clone(),
            metadata_json: Some(entity_metadata.to_string()),
            include_source: false,
        })?;
        upsert_entity_tx(transaction, &prepared)?;
    }
    Ok(())
}

fn upsert_graph_capture_relationships(
    transaction: &Transaction<'_>,
    space: &str,
    memory_id: &str,
    remember: &RememberRequest,
    graph: &GraphCapture,
    resolved_keys: &BTreeMap<String, String>,
) -> Result<()> {
    for relationship in &graph.relationships {
        let subject_entity_key = resolved_keys
            .get(&relationship.subject_entity_key)
            .expect("validated graph subject");
        let object_entity_key = resolved_keys
            .get(&relationship.object_entity_key)
            .expect("validated graph object");
        if subject_entity_key == object_entity_key {
            return Err(Error::InvalidRequest {
                message: "graph relationship endpoints resolve to the same canonical entity"
                    .to_string(),
            });
        }
        let metadata_json = serde_json::json!({
            "routing": true,
            "origin": "automatic_capture",
            "routing_contract": "evidence_join_v2",
            "routing_contract_version": 2,
            "extractor": graph.extractor,
            "extractor_version": graph.extractor_version,
        })
        .to_string();
        let prepared = prepare_relationship_upsert_request(&RelationshipUpsertRequest {
            space: Some(space.to_string()),
            subject_entity_id: None,
            subject_entity_key: Some(subject_entity_key.clone()),
            relation_type: relationship.relation_type.clone(),
            object_entity_id: None,
            object_entity_key: Some(object_entity_key.clone()),
            memory_id: Some(memory_id.to_string()),
            source_episode_id: remember.source_episode_id.clone(),
            status: Some(status::ACTIVE.to_string()),
            confidence: relationship.confidence,
            observed_at: remember.observed_at.clone(),
            valid_from: remember.valid_from.clone(),
            valid_to: remember.valid_to.clone(),
            metadata_json: Some(metadata_json),
            include_source: false,
        })?;
        upsert_relationship_tx(transaction, &prepared)?;
    }
    Ok(())
}
