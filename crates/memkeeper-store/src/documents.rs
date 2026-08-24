//! Document ingest and retrieval extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::path::Path;

use rusqlite::{
    params, params_from_iter, types::Value, Connection, OptionalExtension, Row, Transaction,
};

use memkeeper_core::{status, DEFAULT_DURABLE_SILO, DEFAULT_SPACE};

use crate::{
    collect_rows, ensure_source_episode_recall_events, ensure_space_exists, limit_i64, next_id,
    now_timestamp, open_initialized_read_fast, open_initialized_write, reject_all_spaces_sentinel,
    search_terms, seed_standard_silos, sha256_hex, sha256_text, space_exists, table_exists,
    validate_optional_metadata_value, validate_optional_timestamp, with_read_snapshot,
    DocumentSearchReport, DocumentSearchRequest, DocumentSearchResult, Error, IngestReport,
    IngestRequest, JsonValidator, Result, DEFAULT_DOCUMENT_SEARCH_LIMIT,
    DEFAULT_INGEST_SOURCE_TYPE, DOCUMENTS_SPACE, MAX_CONTENT_CHARS, MAX_SEARCH_LIMIT,
    MAX_SOURCE_REF_JSON_CHARS,
};

#[cfg(feature = "semantic")]
use crate::{
    embedding_json, embedding_to_blob, enforce_active_embedding_model,
    ensure_source_episode_vector_table, source_episode_vector_table,
};

/// Internal chunk hit shared by the lexical and semantic arms before fusion.
struct DocumentChunkHit {
    source_episode_id: String,
    space: String,
    source_type: String,
    source_path: Option<String>,
    source_uri: Option<String>,
    chunk_index: i64,
    chunk_count: i64,
    snippet: String,
    content: Option<String>,
}

pub fn ingest_source(path: impl AsRef<Path>, request: &IngestRequest) -> Result<IngestReport> {
    validate_ingest_request(request)?;
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let source_type = request
        .source_type
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_INGEST_SOURCE_TYPE)
        .to_string();

    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let now = now_timestamp(&transaction)?;

    let created_space = ensure_ingest_space(&transaction, &space, &now)?;

    let chunk_count = request.chunks.len();
    let chunk_count_i64 = i64::try_from(chunk_count).unwrap_or(i64::MAX);
    let mut created = Vec::new();
    let mut skipped = 0usize;
    for (index, content) in request.chunks.iter().enumerate() {
        let sha = sha256_text(content);
        let chunk_index = i64::try_from(index).unwrap_or(i64::MAX);
        if maybe_repair_ingested_chunk(
            &transaction,
            &space,
            &sha,
            request,
            &source_type,
            chunk_index,
            chunk_count_i64,
            content,
            &now,
        )? {
            skipped += 1;
            continue;
        }
        let id = next_id("src");
        transaction.execute(
            "INSERT INTO source_episodes (
                id, space_name, source_type, source_uri, source_path, source_description,
                content, content_sha256, chunk_index, chunk_count, metadata_json,
                ingest_status, ingested_at, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'indexed', ?12, ?12, ?12)",
            params![
                &id,
                &space,
                &source_type,
                request.source_uri,
                request.source_path,
                request.source_description,
                content,
                &sha,
                chunk_index,
                chunk_count_i64,
                request.metadata_json,
                &now,
            ],
        )?;
        transaction.execute(
            "INSERT INTO source_episode_fts (
                source_episode_id, space_name, source_type, source_path, source_description,
                content, metadata_text
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &id,
                &space,
                &source_type,
                request.source_path,
                request.source_description,
                content,
                request.metadata_json,
            ],
        )?;
        maybe_write_chunk_embedding(&transaction, &id, request, index, &now)?;
        created.push(id);
    }

    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }

    Ok(IngestReport {
        space,
        source_path: request.source_path.clone(),
        chunk_count,
        created,
        skipped,
        created_space: created_space && !request.dry_run,
        dry_run: request.dry_run,
    })
}

/// If this chunk already exists at the same `(space, content, source_path)`,
/// repair its mutable provenance/metadata in place and return `true` (a re-sync,
/// counted as skipped). Returns `false` when no such row exists, so the caller
/// inserts a fresh chunk.
///
/// Identity deliberately includes `source_path`: identical content under a
/// *different* path is an independent chunk (kept, to be surfaced as a duplicate
/// later), not a dedup collision. On a hit, content (and thus the embedding) is
/// unchanged, so the vector is left intact.
#[allow(clippy::too_many_arguments)]
fn maybe_repair_ingested_chunk(
    transaction: &Transaction<'_>,
    space: &str,
    sha: &str,
    request: &IngestRequest,
    source_type: &str,
    chunk_index: i64,
    chunk_count: i64,
    content: &str,
    now: &str,
) -> Result<bool> {
    let existing_id: Option<String> = transaction
        .query_row(
            "SELECT id FROM source_episodes
             WHERE space_name = ?1 AND content_sha256 = ?2 AND source_path IS ?3
             LIMIT 1",
            params![space, sha, request.source_path],
            |row| row.get(0),
        )
        .optional()?;
    let Some(existing_id) = existing_id else {
        return Ok(false);
    };
    transaction.execute(
        "UPDATE source_episodes
         SET source_type = ?2, source_uri = ?3, source_description = ?4,
             chunk_index = ?5, chunk_count = ?6, metadata_json = ?7, updated_at = ?8
         WHERE id = ?1",
        params![
            existing_id,
            source_type,
            request.source_uri,
            request.source_description,
            chunk_index,
            chunk_count,
            request.metadata_json,
            now,
        ],
    )?;
    transaction.execute(
        "UPDATE source_episode_fts
         SET source_type = ?2, source_path = ?3, source_description = ?4,
             content = ?5, metadata_text = ?6
         WHERE source_episode_id = ?1",
        params![
            existing_id,
            source_type,
            request.source_path,
            request.source_description,
            content,
            request.metadata_json,
        ],
    )?;
    Ok(true)
}

/// Create the ingest target space (with standard silos) if it does not yet
/// exist. Returns `true` when this call created it. Idempotent.
fn ensure_ingest_space(transaction: &Transaction<'_>, space: &str, now: &str) -> Result<bool> {
    if space_exists(transaction, space)? {
        return Ok(false);
    }
    transaction.execute(
        "INSERT INTO spaces (
            name, display_name, description, default_silo, created_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
        params![
            space,
            "Documents",
            "Ingested document chunks (isolated from curated memory).",
            DEFAULT_DURABLE_SILO,
            now,
        ],
    )?;
    seed_standard_silos(transaction, space, DEFAULT_DURABLE_SILO, now)?;
    Ok(true)
}

fn validate_ingest_request(request: &IngestRequest) -> Result<()> {
    if request.chunks.is_empty() {
        return Err(Error::InvalidRequest {
            message: "ingest request must include at least one chunk".to_string(),
        });
    }
    if let Some(space) = request.space.as_deref() {
        if space.trim().is_empty() {
            return Err(Error::InvalidRequest {
                message: "ingest space must not be blank when provided".to_string(),
            });
        }
    }
    for (index, content) in request.chunks.iter().enumerate() {
        if content.trim().is_empty() {
            return Err(Error::InvalidRequest {
                message: format!("ingest chunk {index} must not be empty"),
            });
        }
        if content.chars().count() > MAX_CONTENT_CHARS {
            return Err(Error::InvalidRequest {
                message: format!("ingest chunk {index} exceeds the maximum content size"),
            });
        }
    }
    if let Some(metadata) = request.metadata_json.as_deref() {
        // Match the import invariant (`validate_import_source_metadata_json`): a
        // malformed payload here would otherwise create a store the engine's own
        // import validation later rejects.
        if metadata.chars().count() > MAX_SOURCE_REF_JSON_CHARS
            || !JsonValidator::is_object(metadata)
        {
            return Err(Error::InvalidRequest {
                message: "ingest metadata_json must be a valid JSON object".to_string(),
            });
        }
    }
    if let Some(embeddings) = request.embeddings.as_ref() {
        if embeddings.len() != request.chunks.len() {
            return Err(Error::InvalidRequest {
                message: "ingest embeddings length must match chunks length".to_string(),
            });
        }
        if embeddings.iter().any(Vec::is_empty) {
            return Err(Error::InvalidRequest {
                message: "ingest embedding vectors must not be empty".to_string(),
            });
        }
        if request
            .embedding_model_id
            .as_deref()
            .is_none_or(|model| model.trim().is_empty())
        {
            return Err(Error::InvalidRequest {
                message: "ingest embedding_model_id is required when embeddings are supplied"
                    .to_string(),
            });
        }
    }
    Ok(())
}

/// Write the chunk embedding for a just-created `source_episodes` row, when the
/// request carries one for this chunk index. No-op on non-semantic builds.
#[cfg(feature = "semantic")]
fn maybe_write_chunk_embedding(
    transaction: &Transaction<'_>,
    source_episode_id: &str,
    request: &IngestRequest,
    index: usize,
    now: &str,
) -> Result<()> {
    let Some(embeddings) = request.embeddings.as_ref() else {
        return Ok(());
    };
    let Some(embedding) = embeddings.get(index) else {
        return Ok(());
    };
    let model = request.embedding_model_id.as_deref().unwrap_or("unknown");
    write_source_episode_embedding(
        transaction,
        source_episode_id,
        model,
        embedding.len(),
        embedding,
        now,
    )
}

#[cfg(not(feature = "semantic"))]
#[allow(clippy::unnecessary_wraps)]
fn maybe_write_chunk_embedding(
    _transaction: &Transaction<'_>,
    _source_episode_id: &str,
    _request: &IngestRequest,
    _index: usize,
    _now: &str,
) -> Result<()> {
    Ok(())
}

/// Write one chunk embedding: the canonical `embeddings` sidecar row (keyed by
/// `source_episode_id`, `memory_id`/`version_id` NULL) plus the rebuildable ANN
/// projection in `source_episode_vec_{dims}`. Enforces the store's single active
/// embedding model so chunk vectors share the model used for query embeddings.
#[cfg(feature = "semantic")]
fn write_source_episode_embedding(
    transaction: &Transaction<'_>,
    source_episode_id: &str,
    model: &str,
    dims: usize,
    embedding: &[f32],
    now: &str,
) -> Result<()> {
    enforce_active_embedding_model(transaction, model, dims, now)?;
    let table = ensure_source_episode_vector_table(transaction, dims)?;
    transaction.execute(
        "INSERT INTO embeddings (id, source_episode_id, embedding_model, dimensions, vector_blob, status, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, 'ready', ?6, ?6)",
        params![
            next_id("emb"),
            source_episode_id,
            model,
            i64::try_from(dims).unwrap_or(i64::MAX),
            embedding_to_blob(embedding),
            now,
        ],
    )?;
    let embedding_json = embedding_json(embedding)?;
    transaction.execute(
        &format!("INSERT INTO {table} (source_episode_id, embedding) VALUES (?1, ?2)"),
        params![source_episode_id, embedding_json],
    )?;
    Ok(())
}

/// Hybrid (BM25 + vector) search over ingested document chunks in one space.
///
/// Runs a lexical arm over `source_episode_fts` and, when a query embedding is
/// supplied and a `source_episode_vec_{dims}` index exists, a semantic arm; the
/// two ranked lists are fused with Reciprocal Rank Fusion. Results carry a
/// citation back to `source_path` + `chunk_index`. Searches only the requested
/// space (default [`DOCUMENTS_SPACE`]); curated memories are never returned.
///
/// # Errors
/// Returns an error when the store is missing/incompatible or `SQLite` fails.
pub fn search_documents(
    path: impl AsRef<Path>,
    request: &DocumentSearchRequest,
) -> Result<DocumentSearchReport> {
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let limit = if request.limit == 0 {
        DEFAULT_DOCUMENT_SEARCH_LIMIT
    } else {
        request.limit.min(MAX_SEARCH_LIMIT)
    };
    let connection = open_initialized_read_fast(path.as_ref())?;
    let report = with_read_snapshot(&connection, move |connection| {
        search_documents_on_connection(connection, request, space, limit)
    })?;
    // Best-effort retrieval instrumentation: records which chunks earned traffic,
    // the signal that later drives usage-based promotion. Never fails the search.
    if !request.skip_recall_log && !report.results.is_empty() {
        if let Err(error) = record_document_retrievals(path.as_ref(), &request.query, &report) {
            eprintln!("[memkeeper] document retrieval logging failed: {error}");
        }
    }
    Ok(report)
}

fn search_documents_on_connection(
    connection: &Connection,
    request: &DocumentSearchRequest,
    space: String,
    limit: usize,
) -> Result<DocumentSearchReport> {
    let candidate_limit = limit
        .saturating_mul(4)
        .max(limit.saturating_add(32))
        .min(MAX_SEARCH_LIMIT.saturating_mul(4));
    let terms = search_terms(&request.query);
    let fts_query = document_fts_query(&terms);
    let lexical = document_lexical_candidates(
        connection,
        &space,
        &fts_query,
        candidate_limit,
        request.include_content,
        request.snippet_chars,
    )?;
    let semantic_attempted = request.embedding.is_some();
    let semantic = match request.embedding.as_deref() {
        Some(embedding) => document_semantic_candidates(
            connection,
            &space,
            embedding,
            candidate_limit,
            request.include_content,
            request.snippet_chars,
        )?,
        None => Vec::new(),
    };
    let results = document_rrf_merge(semantic, lexical, limit);
    let strategy = if semantic_attempted {
        "hybrid_rrf_v0"
    } else {
        "lexical_only_v0"
    };
    Ok(DocumentSearchReport {
        strategy: strategy.to_string(),
        semantic_attempted,
        space,
        results,
    })
}

/// Build a safe FTS5 MATCH string: quoted, OR-joined terms. `search_terms`
/// already lowercases and strips to alphanumerics, so quoting cannot inject.
fn document_fts_query(terms: &[String]) -> String {
    terms
        .iter()
        .map(|term| format!("\"{term}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn document_snippet_sql(snippet_chars: usize) -> String {
    if snippet_chars == 0 {
        "''".to_string()
    } else {
        format!("substr(se.content, 1, {snippet_chars})")
    }
}

fn document_hit_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DocumentChunkHit> {
    Ok(DocumentChunkHit {
        source_episode_id: row.get(0)?,
        space: row.get(1)?,
        source_type: row.get(2)?,
        source_path: row.get(3)?,
        source_uri: row.get(4)?,
        chunk_index: row.get(5)?,
        chunk_count: row.get(6)?,
        snippet: row.get::<_, Option<String>>(7)?.unwrap_or_default(),
        content: row.get(8)?,
    })
}

fn document_lexical_candidates(
    connection: &Connection,
    space: &str,
    fts_query: &str,
    candidate_limit: usize,
    include_content: bool,
    snippet_chars: usize,
) -> Result<Vec<DocumentChunkHit>> {
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }
    let snippet_sql = document_snippet_sql(snippet_chars);
    let content_sql = if include_content {
        "se.content"
    } else {
        "NULL"
    };
    let sql = format!(
        "SELECT se.id, se.space_name, se.source_type, se.source_path, se.source_uri,
                se.chunk_index, se.chunk_count, {snippet_sql}, {content_sql}
         FROM source_episode_fts
         JOIN source_episodes se ON se.id = source_episode_fts.source_episode_id
         WHERE source_episode_fts MATCH ?1 AND source_episode_fts.space_name = ?2
         ORDER BY bm25(source_episode_fts) ASC
         LIMIT {candidate_limit}"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params![fts_query, space], document_hit_from_row)?;
    let mut hits = Vec::new();
    for row in rows {
        hits.push(row?);
    }
    Ok(hits)
}

#[cfg(feature = "semantic")]
fn document_semantic_candidates(
    connection: &Connection,
    space: &str,
    embedding: &[f32],
    candidate_limit: usize,
    include_content: bool,
    snippet_chars: usize,
) -> Result<Vec<DocumentChunkHit>> {
    let dims = embedding.len();
    let table = source_episode_vector_table(dims);
    if !table_exists(connection, &table)? {
        return Ok(Vec::new());
    }
    let snippet_sql = document_snippet_sql(snippet_chars);
    let content_sql = if include_content {
        "se.content"
    } else {
        "NULL"
    };
    let sql = format!(
        "SELECT se.id, se.space_name, se.source_type, se.source_path, se.source_uri,
                se.chunk_index, se.chunk_count, {snippet_sql}, {content_sql}
         FROM {table} t
         JOIN source_episodes se ON se.id = t.source_episode_id
         WHERE t.embedding MATCH ?1 AND k = {candidate_limit} AND se.space_name = ?2
         ORDER BY t.distance ASC"
    );
    let embedding_json = embedding_json(embedding)?;
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params![embedding_json, space], document_hit_from_row)?;
    let mut hits = Vec::new();
    for row in rows {
        hits.push(row?);
    }
    Ok(hits)
}

#[cfg(not(feature = "semantic"))]
#[allow(clippy::unnecessary_wraps)]
fn document_semantic_candidates(
    _connection: &Connection,
    _space: &str,
    _embedding: &[f32],
    _candidate_limit: usize,
    _include_content: bool,
    _snippet_chars: usize,
) -> Result<Vec<DocumentChunkHit>> {
    Ok(Vec::new())
}

struct DocumentMergeEntry {
    score: f64,
    hit: DocumentChunkHit,
    semantic: bool,
    lexical: bool,
}

/// Fuse the semantic and lexical ranked lists with Reciprocal Rank Fusion
/// (`score = sum 1/(K + rank)`, K=60). Deterministic: ties break on chunk id.
#[allow(clippy::cast_precision_loss)]
fn document_rrf_merge(
    semantic: Vec<DocumentChunkHit>,
    lexical: Vec<DocumentChunkHit>,
    limit: usize,
) -> Vec<DocumentSearchResult> {
    const RRF_K: f64 = 60.0;
    let mut entries: std::collections::BTreeMap<String, DocumentMergeEntry> =
        std::collections::BTreeMap::new();
    for (rank, hit) in semantic.into_iter().enumerate() {
        let contribution = 1.0 / (RRF_K + rank as f64 + 1.0);
        match entries.get_mut(&hit.source_episode_id) {
            Some(entry) => {
                entry.score += contribution;
                entry.semantic = true;
            }
            None => {
                entries.insert(
                    hit.source_episode_id.clone(),
                    DocumentMergeEntry {
                        score: contribution,
                        hit,
                        semantic: true,
                        lexical: false,
                    },
                );
            }
        }
    }
    for (rank, hit) in lexical.into_iter().enumerate() {
        let contribution = 1.0 / (RRF_K + rank as f64 + 1.0);
        match entries.get_mut(&hit.source_episode_id) {
            Some(entry) => {
                entry.score += contribution;
                entry.lexical = true;
            }
            None => {
                entries.insert(
                    hit.source_episode_id.clone(),
                    DocumentMergeEntry {
                        score: contribution,
                        hit,
                        semantic: false,
                        lexical: true,
                    },
                );
            }
        }
    }
    let mut merged: Vec<DocumentMergeEntry> = entries.into_values().collect();
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.hit.source_episode_id.cmp(&b.hit.source_episode_id))
    });
    merged.truncate(limit);
    merged
        .into_iter()
        .enumerate()
        .map(|(index, entry)| {
            let match_type = match (entry.semantic, entry.lexical) {
                (true, true) => "hybrid",
                (true, false) => "semantic",
                _ => "lexical",
            };
            DocumentSearchResult {
                rank: index + 1,
                source_episode_id: entry.hit.source_episode_id,
                space: entry.hit.space,
                source_type: entry.hit.source_type,
                source_path: entry.hit.source_path,
                source_uri: entry.hit.source_uri,
                chunk_index: entry.hit.chunk_index,
                chunk_count: entry.hit.chunk_count,
                snippet: entry.hit.snippet,
                content: entry.hit.content,
                score: entry.score,
                match_type: match_type.to_string(),
            }
        })
        .collect()
}

/// Record one retrieval event per returned chunk. `query_sha256` lets the
/// promotion step count distinct queries (query diversity) without storing the
/// raw query repeatedly. Opens its own short write transaction.
fn record_document_retrievals(
    path: &Path,
    query: &str,
    report: &DocumentSearchReport,
) -> Result<usize> {
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    ensure_source_episode_recall_events(&transaction)?;
    let now = now_timestamp(&transaction)?;
    let query_opt = (!query.is_empty()).then_some(query);
    let query_sha = query_opt.map(sha256_text);
    let mut recorded = 0_usize;
    {
        let mut insert = transaction.prepare_cached(
            "INSERT INTO source_episode_recall_events
                (source_episode_id, space_name, ts, query, query_sha256, rank, score, match_type)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )?;
        for result in &report.results {
            insert.execute(params![
                &result.source_episode_id,
                &report.space,
                &now,
                query_opt,
                query_sha.as_deref(),
                i64::try_from(result.rank).unwrap_or(i64::MAX),
                result.score,
                &result.match_type,
            ])?;
            recorded += 1;
        }
    }
    transaction.commit()?;
    Ok(recorded)
}

/// Default chunk limit for [`get_document`].
pub const DEFAULT_DOCUMENT_GET_LIMIT: usize = 500;
/// Default candidate limit for [`promotion_candidates`].
pub const DEFAULT_PROMOTION_CANDIDATE_LIMIT: usize = 20;

/// Request to fetch a document's chunks by path or by chunk id.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocumentGetRequest {
    /// Fetch all chunks of the document at this `source_path`.
    pub source_path: Option<String>,
    /// Fetch a single chunk by its `source_episodes` id.
    pub source_episode_id: Option<String>,
    /// Space to look in (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// Include full chunk content (else metadata only).
    pub include_content: bool,
    /// Max chunks to return.
    pub limit: usize,
}

/// One stored document chunk with full provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentChunk {
    /// `source_episodes` id.
    pub source_episode_id: String,
    /// Space.
    pub space: String,
    /// Provenance type.
    pub source_type: String,
    /// Source document path.
    pub source_path: Option<String>,
    /// Source document URI.
    pub source_uri: Option<String>,
    /// Chunk position within the source.
    pub chunk_index: i64,
    /// Total chunks in the source.
    pub chunk_count: i64,
    /// Content hash.
    pub content_sha256: Option<String>,
    /// Ingest status (`indexed`/`extracted`/...).
    pub ingest_status: String,
    /// Chunk content (when requested).
    pub content: Option<String>,
}

/// Result of a document fetch.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentGetReport {
    /// Space searched.
    pub space: String,
    /// Chunks, ordered by `chunk_index`.
    pub chunks: Vec<DocumentChunk>,
}

/// Request for usage-driven promotion candidates.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PromotionCandidatesRequest {
    /// Space to rank within (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// Minimum total retrieval hits to qualify.
    pub min_hits: usize,
    /// Minimum distinct queries that retrieved the chunk (query diversity).
    pub min_distinct_queries: usize,
    /// Max candidates to return.
    pub limit: usize,
    /// Include full chunk content.
    pub include_content: bool,
    /// Include chunks already promoted (`ingest_status = 'extracted'`).
    pub include_extracted: bool,
}

/// One promotion candidate: a chunk that earned retrieval traffic.
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionCandidate {
    /// `source_episodes` id.
    pub source_episode_id: String,
    /// Space.
    pub space: String,
    /// Source document path.
    pub source_path: Option<String>,
    /// Source document URI.
    pub source_uri: Option<String>,
    /// Chunk position within the source.
    pub chunk_index: i64,
    /// Total chunks in the source.
    pub chunk_count: i64,
    /// Total retrieval hits.
    pub hits: i64,
    /// Distinct queries that retrieved the chunk.
    pub distinct_queries: i64,
    /// Most recent retrieval timestamp.
    pub last_hit: String,
    /// Chunk content (when requested).
    pub content: Option<String>,
}

/// Result of ranking promotion candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionCandidatesReport {
    /// Space ranked.
    pub space: String,
    /// Candidates, highest signal first.
    pub candidates: Vec<PromotionCandidate>,
}

/// Default cluster cap for [`document_duplicates`].
pub const DEFAULT_DUPLICATE_CLUSTER_LIMIT: usize = 50;

/// Request to surface exact-content duplicate document chunks: independent rows
/// that ingest kept because they share content but came from different sources.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocumentDuplicatesRequest {
    /// Space to scan (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// Max duplicate clusters to return (0 = [`DEFAULT_DUPLICATE_CLUSTER_LIMIT`]).
    pub limit: usize,
    /// Snippet length in characters for the shared-content preview (0 = none).
    pub snippet_chars: usize,
}

/// One member of a duplicate cluster: a stored chunk that shares its content
/// with the other members.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateChunkMember {
    /// `source_episodes` id.
    pub source_episode_id: String,
    /// Source document path (citation).
    pub source_path: Option<String>,
    /// Source document URI (citation).
    pub source_uri: Option<String>,
    /// Chunk position within its source.
    pub chunk_index: i64,
    /// Total chunks in its source.
    pub chunk_count: i64,
    /// Ingest status (`indexed`/`extracted`/...).
    pub ingest_status: String,
    /// When the chunk was ingested.
    pub ingested_at: String,
}

/// A set of document chunks sharing identical content (`content_sha256`) across
/// different sources — the independent duplicates ingest intentionally keeps.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateChunkCluster {
    /// Shared content hash.
    pub content_sha256: String,
    /// Number of chunks sharing this content.
    pub member_count: i64,
    /// Preview of the shared content (empty when not requested).
    pub snippet: String,
    /// The duplicate chunks, ordered by source path then chunk index.
    pub members: Vec<DuplicateChunkMember>,
}

/// Result of scanning for duplicate document chunks.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentDuplicatesReport {
    /// Space scanned.
    pub space: String,
    /// Duplicate clusters, most-duplicated first.
    pub clusters: Vec<DuplicateChunkCluster>,
}

/// Fetch a document's chunks by `source_path` (all chunks, ordered) or by a
/// single `source_episode_id`. One of the two must be set.
///
/// # Errors
/// Returns [`Error::InvalidRequest`] when neither selector is set, or a storage
/// error.
pub fn get_document(
    path: impl AsRef<Path>,
    request: &DocumentGetRequest,
) -> Result<DocumentGetReport> {
    if request.source_path.is_none() && request.source_episode_id.is_none() {
        return Err(Error::InvalidRequest {
            message: "document-get requires source_path or source_episode_id".to_string(),
        });
    }
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let limit = if request.limit == 0 {
        DEFAULT_DOCUMENT_GET_LIMIT
    } else {
        request.limit.min(5_000)
    };
    let content_sql = if request.include_content {
        "content"
    } else {
        "NULL"
    };
    let (selector, selector_value) = if let Some(id) = request.source_episode_id.as_deref() {
        ("id = ?2", id.to_string())
    } else {
        (
            "source_path = ?2",
            request.source_path.clone().unwrap_or_default(),
        )
    };
    let sql = format!(
        "SELECT id, space_name, source_type, source_path, source_uri, chunk_index, chunk_count,
                content_sha256, ingest_status, {content_sql}
         FROM source_episodes
         WHERE space_name = ?1 AND {selector}
         ORDER BY chunk_index ASC, id ASC
         LIMIT {limit}"
    );
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(params![space, selector_value], document_chunk_from_row)?;
        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row?);
        }
        Ok(DocumentGetReport {
            space: space.clone(),
            chunks,
        })
    })
}

fn document_chunk_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DocumentChunk> {
    Ok(DocumentChunk {
        source_episode_id: row.get(0)?,
        space: row.get(1)?,
        source_type: row.get(2)?,
        source_path: row.get(3)?,
        source_uri: row.get(4)?,
        chunk_index: row.get(5)?,
        chunk_count: row.get(6)?,
        content_sha256: row.get(7)?,
        ingest_status: row.get(8)?,
        content: row.get(9)?,
    })
}

/// Rank document chunks that have earned retrieval traffic, as promotion
/// candidates (usage-driven promotion signal). Aggregates
/// `source_episode_recall_events` by chunk: total hits, distinct queries (via
/// `query_sha256`), and recency. Returns empty if nothing has been searched yet.
///
/// # Errors
/// Returns a storage error on `SQLite` failure.
pub fn promotion_candidates(
    path: impl AsRef<Path>,
    request: &PromotionCandidatesRequest,
) -> Result<PromotionCandidatesReport> {
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let limit = if request.limit == 0 {
        DEFAULT_PROMOTION_CANDIDATE_LIMIT
    } else {
        request.limit.min(200)
    };
    let min_hits = i64::try_from(request.min_hits.max(1)).unwrap_or(1);
    let min_distinct = i64::try_from(request.min_distinct_queries).unwrap_or(0);
    let content_sql = if request.include_content {
        "se.content"
    } else {
        "NULL"
    };
    let extracted_filter = if request.include_extracted {
        ""
    } else {
        "AND se.ingest_status != 'extracted'"
    };
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        if !table_exists(connection, "source_episode_recall_events")? {
            return Ok(PromotionCandidatesReport {
                space: space.clone(),
                candidates: Vec::new(),
            });
        }
        let sql = format!(
            "SELECT r.source_episode_id, se.space_name, se.source_path, se.source_uri,
                    se.chunk_index, se.chunk_count,
                    COUNT(*) AS hits,
                    COUNT(DISTINCT r.query_sha256) AS distinct_queries,
                    MAX(r.ts) AS last_hit,
                    {content_sql}
             FROM source_episode_recall_events r
             JOIN source_episodes se ON se.id = r.source_episode_id
             WHERE r.space_name = ?1 {extracted_filter}
             GROUP BY r.source_episode_id
             HAVING COUNT(*) >= ?2 AND COUNT(DISTINCT r.query_sha256) >= ?3
             ORDER BY distinct_queries DESC, hits DESC, last_hit DESC
             LIMIT {limit}"
        );
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(
            params![space, min_hits, min_distinct],
            promotion_candidate_from_row,
        )?;
        let mut candidates = Vec::new();
        for row in rows {
            candidates.push(row?);
        }
        Ok(PromotionCandidatesReport {
            space: space.clone(),
            candidates,
        })
    })
}

/// Surface exact-content duplicate document chunks: clusters of chunks in a
/// space that share the same `content_sha256` across two or more rows (e.g.
/// identical content ingested under different `source_path`s). Read-only; never
/// records retrieval telemetry. Clusters are returned most-duplicated first.
///
/// # Errors
/// Returns a storage error when the store cannot be read.
pub fn document_duplicates(
    path: impl AsRef<Path>,
    request: &DocumentDuplicatesRequest,
) -> Result<DocumentDuplicatesReport> {
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let limit = if request.limit == 0 {
        DEFAULT_DUPLICATE_CLUSTER_LIMIT
    } else {
        request.limit.min(500)
    };
    let snippet_chars = i64::try_from(request.snippet_chars).unwrap_or(0).max(0);
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        if !table_exists(connection, "source_episodes")? {
            return Ok(DocumentDuplicatesReport {
                space: space.clone(),
                clusters: Vec::new(),
            });
        }
        // 1. Qualifying content hashes (2+ rows in the space), most-duplicated
        //    first, capped to `limit` clusters.
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut hash_statement = connection.prepare(
            "SELECT content_sha256, COUNT(*) AS n
             FROM source_episodes
             WHERE space_name = ?1 AND content_sha256 IS NOT NULL
             GROUP BY content_sha256
             HAVING n > 1
             ORDER BY n DESC, content_sha256 ASC
             LIMIT ?2",
        )?;
        let hashes: Vec<(String, i64)> = hash_statement
            .query_map(params![space, limit_i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<rusqlite::Result<_>>()?;

        // 2. Members per cluster. All members share content, so any member's
        //    snippet represents the cluster.
        let mut member_statement = connection.prepare(
            "SELECT id, source_path, source_uri, chunk_index, chunk_count,
                    ingest_status, ingested_at, substr(content, 1, ?3)
             FROM source_episodes
             WHERE space_name = ?1 AND content_sha256 = ?2
             ORDER BY source_path, chunk_index, id",
        )?;
        let mut clusters = Vec::with_capacity(hashes.len());
        for (content_sha256, member_count) in hashes {
            let mut snippet = String::new();
            let mut members = Vec::new();
            let rows = member_statement.query_map(
                params![space, content_sha256, snippet_chars],
                |row| {
                    Ok((
                        DuplicateChunkMember {
                            source_episode_id: row.get(0)?,
                            source_path: row.get(1)?,
                            source_uri: row.get(2)?,
                            chunk_index: row.get(3)?,
                            chunk_count: row.get(4)?,
                            ingest_status: row.get(5)?,
                            ingested_at: row.get(6)?,
                        },
                        row.get::<_, Option<String>>(7)?,
                    ))
                },
            )?;
            for row in rows {
                let (member, member_snippet) = row?;
                if snippet.is_empty() {
                    if let Some(text) = member_snippet {
                        snippet = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    }
                }
                members.push(member);
            }
            clusters.push(DuplicateChunkCluster {
                content_sha256,
                member_count,
                snippet,
                members,
            });
        }
        Ok(DocumentDuplicatesReport {
            space: space.clone(),
            clusters,
        })
    })
}

/// Request to mark document chunks as extracted (i.e. promoted to memory), so
/// they stop surfacing as promotion candidates.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MarkExtractedRequest {
    /// Space the chunks live in (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// `source_episodes` ids to mark `extracted`.
    pub source_episode_ids: Vec<String>,
}

/// Result of marking chunks extracted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkExtractedReport {
    /// Space operated on.
    pub space: String,
    /// Number of chunks whose status changed.
    pub updated: usize,
}

/// Mark the given chunks `ingest_status = 'extracted'`. Used after a chunk is
/// promoted into a memory, so usage-driven promotion does not re-propose it.
///
/// # Errors
/// Returns [`Error::InvalidRequest`] when no ids are given, or a storage error.
pub fn mark_source_episodes_extracted(
    path: impl AsRef<Path>,
    request: &MarkExtractedRequest,
) -> Result<MarkExtractedReport> {
    if request.source_episode_ids.is_empty() {
        return Err(Error::InvalidRequest {
            message: "mark-extracted requires at least one source_episode_id".to_string(),
        });
    }
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let now = now_timestamp(&transaction)?;
    let mut updated = 0usize;
    {
        let mut statement = transaction.prepare(
            "UPDATE source_episodes SET ingest_status = 'extracted', updated_at = ?1
             WHERE space_name = ?2 AND id = ?3",
        )?;
        for id in &request.source_episode_ids {
            updated += statement.execute(params![now, space, id])?;
        }
    }
    transaction.commit()?;
    Ok(MarkExtractedReport { space, updated })
}

/// Request to prune (delete) specific document chunks by id. User-driven cleanup
/// of duplicates surfaced by [`document_duplicates`]: the caller chooses exactly
/// which chunks to remove, so deletion is always explicit (never automatic).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocumentPruneRequest {
    /// Space the chunks live in (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// `source_episodes` ids to delete.
    pub source_episode_ids: Vec<String>,
    /// Validate and report without deleting.
    pub dry_run: bool,
}

/// Result of pruning document chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentPruneReport {
    /// Space operated on.
    pub space: String,
    /// Number of ids requested for deletion.
    pub requested: usize,
    /// Ids actually deleted (those that existed in the space).
    pub deleted: Vec<String>,
    /// True when validated but rolled back.
    pub dry_run: bool,
}

/// Delete the given document chunks and all their derived rows (FTS, canonical
/// embedding, and rebuildable ANN projections). Only ids present in `space` are
/// removed; unknown ids are ignored (surfaced via `requested` vs `deleted`).
/// Intended for user-driven duplicate cleanup after reviewing
/// [`document_duplicates`].
///
/// # Errors
/// Returns [`Error::InvalidRequest`] when no ids are given, or a storage error.
pub fn prune_documents(
    path: impl AsRef<Path>,
    request: &DocumentPruneRequest,
) -> Result<DocumentPruneReport> {
    if request.source_episode_ids.is_empty() {
        return Err(Error::InvalidRequest {
            message: "document-prune requires at least one source_episode_id".to_string(),
        });
    }
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let vec_tables = source_episode_vec_tables(&transaction)?;
    let mut deleted = Vec::new();
    for id in &request.source_episode_ids {
        // Scope the delete to the space: an id outside it is left untouched and
        // its derived rows are not removed.
        let removed = transaction.execute(
            "DELETE FROM source_episodes WHERE space_name = ?1 AND id = ?2",
            params![space, id],
        )?;
        if removed == 0 {
            continue;
        }
        transaction.execute(
            "DELETE FROM source_episode_fts WHERE source_episode_id = ?1",
            params![id],
        )?;
        transaction.execute(
            "DELETE FROM embeddings WHERE source_episode_id = ?1",
            params![id],
        )?;
        for table in &vec_tables {
            transaction.execute(
                &format!("DELETE FROM {table} WHERE source_episode_id = ?1"),
                params![id],
            )?;
        }
        deleted.push(id.clone());
    }
    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }
    Ok(DocumentPruneReport {
        space,
        requested: request.source_episode_ids.len(),
        deleted,
        dry_run: request.dry_run,
    })
}

/// Names of the rebuildable `source_episode_vec_{dims}` ANN tables present in the
/// store (zero on non-semantic stores). Restricted to the `vec0` virtual tables:
/// the `sql LIKE '%USING vec0%'` filter excludes vec0's internal shadow tables
/// (`_info`/`_chunks`/`_rowids`/...), which carry no `source_episode_id` column
/// and are managed automatically when the virtual table row is deleted.
fn source_episode_vec_tables(transaction: &Transaction<'_>) -> Result<Vec<String>> {
    let mut statement = transaction.prepare(
        "SELECT name FROM sqlite_master
         WHERE type = 'table' AND name LIKE 'source_episode_vec_%'
           AND sql LIKE '%USING vec0%'",
    )?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

fn promotion_candidate_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<PromotionCandidate> {
    Ok(PromotionCandidate {
        source_episode_id: row.get(0)?,
        space: row.get(1)?,
        source_path: row.get(2)?,
        source_uri: row.get(3)?,
        chunk_index: row.get(4)?,
        chunk_count: row.get(5)?,
        hits: row.get(6)?,
        distinct_queries: row.get(7)?,
        last_hit: row.get(8)?,
        content: row.get(9)?,
    })
}
