//! Vector index and embedding helpers extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::collections::BTreeSet;
use std::path::Path;

use rusqlite::{params, params_from_iter, Connection, OptionalExtension, Transaction};

use crate::pack::{AdmissionObservation, AdmissionSource, PackPoolItem};
use crate::{
    collect_rows, limit_i64, next_id, now_timestamp, open_initialized_read_fast, open_initialized_write,
    representation_document, table_exists, Error, Result,
};

#[cfg(feature = "semantic")]
use crate::{
    drop_all_vector_tables, ensure_memory_vector_table, embedding_json, filters_where_clause,
    semantic_table_for_dims, PackRequest, SearchFilters, SqlArgs,
};

pub(crate) fn maxsim_score(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f64 {
    query
        .iter()
        .map(|query_token| {
            doc.iter()
                .map(|doc_token| {
                    query_token
                        .iter()
                        .zip(doc_token)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                })
                .fold(f32::MIN, f32::max)
        })
        .map(f64::from)
        .sum()
}

/// Exhaustive late-interaction candidate generation: MaxSim-score every
/// eligible active memory's token embeddings against the query tokens, then
/// return the top `limit`.
///
/// # Errors
///
/// Returns an error on `SQLite` failure or malformed blobs.
pub(crate) fn maxsim_candidates(
    connection: &Connection,
    query_tokens: &[Vec<f32>],
    model_id: &str,
    eligible_ids: &BTreeSet<String>,
    limit: usize,
    query_index: usize,
) -> Result<Vec<PackPoolItem>> {
    let docs = load_token_embeddings_cached(connection, model_id)?;
    let mut scored: Vec<PackPoolItem> = docs
        .iter()
        .filter(|(memory_id, matrix)| eligible_ids.contains(memory_id) && !matrix.is_empty())
        .map(|(memory_id, matrix)| PackPoolItem {
            memory_id: memory_id.clone(),
            score: maxsim_score(query_tokens, matrix),
            admissions: vec![AdmissionObservation {
                source: AdmissionSource::Maxsim,
                query_index,
                source_rank: 0,
                seed_memory_id: None,
                activation: None,
                graph_route: None,
            }],
        })
        .collect();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(limit);
    for (index, candidate) in scored.iter_mut().enumerate() {
        candidate.admissions[0].source_rank = index + 1;
    }
    Ok(scored)
}

/// Bounded, scope-correct shortlist for late-interaction selection: the top
/// `n` eligible memories by single-vector distance to `embedding`, unioned
/// with eligible memories that have no row in the vector table (those cannot
/// be ranked here, so they stay eligible for `MaxSim` instead of being
/// silently dropped).
///
/// The KNN scan is restricted to eligible rowids via `rowid IN` so the top-N
/// cannot be crowded out by out-of-scope vectors. The result is intersected
/// with `eligible_ids` as a guard against drift between the SQL filter
/// derivation and the caller's eligible set.
///
/// # Errors
///
/// Returns an error on `SQLite` failure.
#[cfg(feature = "semantic")]
pub(crate) fn maxsim_shortlist_ids(
    connection: &Connection,
    table: &str,
    embedding: &[f32],
    filters: &SearchFilters,
    eligible_ids: &BTreeSet<String>,
    n: usize,
) -> Result<BTreeSet<String>> {
    let mut args = SqlArgs::with_reserved(1);
    let where_clause = filters_where_clause(filters, &mut args);
    // `k` is interpolated like the sibling semantic_search_sql: sqlite-vec
    // expects an integer constraint, and the cached-statement LRU stays
    // bounded because the cap takes only a handful of distinct values.
    let sql = format!(
        "SELECT t.memory_id FROM {table} t
         WHERE t.embedding MATCH ?1 AND k = {k}
         AND t.rowid IN (
            SELECT v.rowid FROM {table} v
            JOIN memories m ON m.id = v.memory_id
            WHERE {where_clause}
         )",
        k = i64::try_from(n).unwrap_or(i64::MAX)
    );
    let embedding_json = embedding_json(embedding)?;
    let mut statement = connection.prepare_cached(&sql)?;
    let params = std::iter::once(embedding_json).chain(args.values);
    let rows = statement.query_map(params_from_iter(params), |row| row.get::<_, String>(0))?;
    let mut shortlist: BTreeSet<String> = collect_rows(rows)?
        .into_iter()
        .filter(|memory_id| eligible_ids.contains(memory_id))
        .collect();

    let mut vectorless_args = SqlArgs::with_reserved(0);
    let vectorless_where = filters_where_clause(filters, &mut vectorless_args);
    let vectorless_sql = format!(
        "SELECT m.id FROM memories m WHERE {vectorless_where}
         AND m.id NOT IN (SELECT memory_id FROM {table})"
    );
    let mut vectorless_statement = connection.prepare_cached(&vectorless_sql)?;
    let vectorless_rows = vectorless_statement
        .query_map(params_from_iter(vectorless_args.values), |row| {
            row.get::<_, String>(0)
        })?;
    for memory_id in collect_rows(vectorless_rows)? {
        if eligible_ids.contains(&memory_id) {
            shortlist.insert(memory_id);
        }
    }
    Ok(shortlist)
}

/// Pack-path wrapper for `maxsim_shortlist_ids`: returns `Some(shortlist)`
/// when the cap is active and this query has a dense embedding to rank by
/// (`query_embeddings` aligned with the token queries), `None` to keep the
/// exhaustive eligible set.
///
/// # Errors
///
/// Returns an error on `SQLite` failure.
#[cfg(feature = "semantic")]
pub(crate) fn pack_maxsim_shortlist(
    connection: &Connection,
    pool_request: &PackRequest,
    filters: &SearchFilters,
    eligible_ids: &BTreeSet<String>,
    query_count: usize,
    query_index: usize,
    pool_width: usize,
) -> Result<Option<BTreeSet<String>>> {
    let cap = pool_request.maxsim_shortlist;
    if cap == 0 || eligible_ids.len() <= cap {
        return Ok(None);
    }
    let Some(dense) = pool_request
        .query_embeddings
        .as_ref()
        .filter(|embeddings| embeddings.len() == query_count)
        .map(|embeddings| &embeddings[query_index])
    else {
        return Ok(None);
    };
    let table = semantic_table_for_dims(dense.len())?;
    if !table_exists(connection, &table)? {
        return Ok(None);
    }
    maxsim_shortlist_ids(
        connection,
        &table,
        dense,
        filters,
        eligible_ids,
        cap.max(pool_width),
    )
    .map(Some)
}


/// Rows of (memory id, token matrix) loaded for late-interaction scoring.
type TokenMatrixRows = Vec<(String, Vec<Vec<f32>>)>;

/// Process-global token-matrix cache for the warm daemon: keyed by
/// (model id, row count, max `created_at`); any token write changes the key.
/// Returns a cheap `Arc` clone on hit.
pub(crate) fn load_token_embeddings_cached(
    connection: &Connection,
    model_id: &str,
) -> Result<std::sync::Arc<TokenMatrixRows>> {
    type CacheEntry = (String, i64, String, std::sync::Arc<TokenMatrixRows>);
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Option<CacheEntry>>> =
        std::sync::OnceLock::new();
    let (count, max_created): (i64, String) = connection.query_row(
        "SELECT count(*), COALESCE(max(created_at), '') FROM memory_token_embeddings \
         WHERE embedding_model = ?1",
        [model_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = cache.lock().map_err(|_| Error::Conflict {
        message: "token cache lock poisoned".to_string(),
    })?;
    if let Some((cached_model, cached_count, cached_created, rows)) = guard.as_ref() {
        if cached_model == model_id && *cached_count == count && cached_created == &max_created {
            return Ok(std::sync::Arc::clone(rows));
        }
    }
    let rows = std::sync::Arc::new(load_token_embeddings(connection, model_id)?);
    *guard = Some((
        model_id.to_string(),
        count,
        max_created,
        std::sync::Arc::clone(&rows),
    ));
    Ok(rows)
}

/// Interleave semantic ANN and lexical BM25 candidates, preserving each source's
/// internal order while deduplicating by memory id. ANN candidates stay first at
/// each rank so the embedding path remains the primary recall tier; BM25 acts as
/// an exact-keyword safety net before cross-encoder reranking.
pub(crate) fn token_vecs_to_blob(vecs: &[Vec<f32>]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vecs.iter().map(Vec::len).sum::<usize>() * 4);
    for vector in vecs {
        for value in vector {
            blob.extend_from_slice(&value.to_le_bytes());
        }
    }
    blob
}

pub(crate) fn blob_to_token_vecs(blob: &[u8], dims: usize, n_tokens: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(n_tokens);
    for token in 0..n_tokens {
        let base = token * dims * 4;
        out.push(
            (0..dims)
                .map(|d| {
                    let offset = base + d * 4;
                    f32::from_le_bytes([
                        blob[offset],
                        blob[offset + 1],
                        blob[offset + 2],
                        blob[offset + 3],
                    ])
                })
                .collect(),
        );
    }
    out
}

/// Insert or replace the late-interaction token-embedding row for a memory.
///
/// # Errors
///
/// Returns an error on `SQLite` failure or empty/ragged input.
pub(crate) fn upsert_memory_token_embedding(
    connection: &Connection,
    memory_id: &str,
    model_id: &str,
    vecs: &[Vec<f32>],
) -> Result<()> {
    let n_tokens = vecs.len();
    if n_tokens == 0 {
        return Err(Error::InvalidRequest {
            message: "empty token embedding".to_string(),
        });
    }
    let dims = vecs[0].len();
    if !vecs.iter().all(|vector| vector.len() == dims) {
        return Err(Error::InvalidRequest {
            message: "ragged token embedding".to_string(),
        });
    }
    connection.execute(
        "INSERT OR REPLACE INTO memory_token_embeddings \
         (memory_id, embedding_model, dims, n_tokens, vector_blob, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)",
        rusqlite::params![
            memory_id,
            model_id,
            i64::try_from(dims).map_err(|_| Error::InvalidRequest {
                message: "token dims overflow".to_string(),
            })?,
            i64::try_from(n_tokens).map_err(|_| Error::InvalidRequest {
                message: "token count overflow".to_string(),
            })?,
            token_vecs_to_blob(vecs)
        ],
    )?;
    Ok(())
}

/// Load all token embeddings for ACTIVE memories under the given model.
///
/// # Errors
///
/// Returns an error on `SQLite` failure or malformed blobs.
pub(crate) fn load_token_embeddings(
    connection: &Connection,
    model_id: &str,
) -> Result<Vec<(String, Vec<Vec<f32>>)>> {
    let mut statement = connection.prepare_cached(
        "SELECT t.memory_id, t.dims, t.n_tokens, t.vector_blob \
         FROM memory_token_embeddings t JOIN memories m ON m.id = t.memory_id \
         WHERE t.embedding_model = ?1 AND m.status = 'active'",
    )?;
    let rows = statement.query_map([model_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, Vec<u8>>(3)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (id, dims, n_tokens, blob) = row?;
        let (Ok(dims), Ok(n_tokens)) = (usize::try_from(dims), usize::try_from(n_tokens)) else {
            return Err(Error::InvalidRequest {
                message: format!("negative token dims for {id}"),
            });
        };
        if blob.len() != dims * n_tokens * 4 {
            return Err(Error::InvalidRequest {
                message: format!("token blob size mismatch for {id}"),
            });
        }
        out.push((id, blob_to_token_vecs(&blob, dims, n_tokens)));
    }
    Ok(out)
}

#[cfg(feature = "semantic")]
pub(crate) fn insert_memory_embedding(
    transaction: &Transaction<'_>,
    memory_id: &str,
    version_id: &str,
    now: &str,
    embedding: Option<&[f32]>,
    model_id: Option<&str>,
) -> Result<()> {
    let Some(embedding) = embedding else {
        return Ok(());
    };
    let dims = embedding.len();
    let model = model_id.unwrap_or("unknown");
    enforce_active_embedding_model(transaction, model, dims, now)?;
    write_embedding_row(
        transaction,
        memory_id,
        version_id,
        model,
        dims,
        embedding,
        now,
    )
}

#[cfg(feature = "semantic")]
pub(crate) fn write_embedding_row(
    transaction: &Transaction<'_>,
    memory_id: &str,
    version_id: &str,
    model: &str,
    dims: usize,
    embedding: &[f32],
    now: &str,
) -> Result<()> {
    let table = ensure_memory_vector_table(transaction, dims)?;
    // Canonical vector sidecar (source of truth for export/import).
    transaction.execute(
        "INSERT INTO embeddings (id, memory_id, version_id, embedding_model, dimensions, vector_blob, status, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'ready', ?7, ?7)",
        params![
            next_id("emb"),
            memory_id,
            version_id,
            model,
            i64::try_from(dims).unwrap_or(i64::MAX),
            embedding_to_blob(embedding),
            now,
        ],
    )?;
    // Rebuildable ANN projection.
    let embedding_json = embedding_json(embedding)?;
    transaction.execute(
        &format!("INSERT INTO {table} (memory_id, embedding) VALUES (?1, ?2)"),
        params![memory_id, embedding_json],
    )?;
    Ok(())
}

#[cfg(feature = "semantic")]
pub(crate) fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

#[cfg(feature = "semantic")]
pub(crate) fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

pub(crate) fn read_config_value(connection: &Connection, key: &str) -> Result<Option<String>> {
    let mut statement = connection.prepare("SELECT value FROM config_kv WHERE key = ?1")?;
    let mut rows = statement.query(params![key])?;
    match rows.next()? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

pub(crate) fn set_config_value(connection: &Connection, key: &str, value: &str, now: &str) -> Result<()> {
    connection.execute(
        "INSERT INTO config_kv (key, value, updated_at) VALUES (?1, ?2, ?3)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        params![key, value, now],
    )?;
    Ok(())
}

#[cfg(feature = "semantic")]
pub(crate) fn enforce_active_embedding_model(
    transaction: &Transaction<'_>,
    model: &str,
    dims: usize,
    now: &str,
) -> Result<()> {
    let active_model = read_config_value(transaction, "active_embedding_model")?;
    let active_dims = read_config_value(transaction, "active_embedding_dims")?;
    if let (Some(active_model), Some(active_dims)) = (active_model, active_dims) {
        if active_model != model || active_dims != dims.to_string() {
            return Err(Error::InvalidRequest {
                message: format!(
                    "store embeddings use model '{active_model}' ({active_dims} dims); cannot mix with '{model}' ({dims} dims) — run `memkeeper reindex` to switch"
                ),
            });
        }
    } else {
        set_config_value(transaction, "active_embedding_model", model, now)?;
        set_config_value(transaction, "active_embedding_dims", &dims.to_string(), now)?;
    }
    Ok(())
}

/// Reject mixing token-embedding models within one store (mirrors
/// `enforce_active_embedding_model`; key `active_colbert_model`).
pub(crate) fn enforce_active_colbert_model(
    transaction: &Transaction<'_>,
    model: &str,
    now: &str,
) -> Result<()> {
    let active_model = read_config_value(transaction, "active_colbert_model")?;
    if let Some(active_model) = active_model {
        if active_model != model {
            return Err(Error::InvalidRequest {
                message: format!(
                    "store token embeddings use model '{active_model}'; cannot mix with '{model}' — run `memkeeper reindex --tokens --force` to switch"
                ),
            });
        }
    } else {
        set_config_value(transaction, "active_colbert_model", model, now)?;
    }
    Ok(())
}

#[cfg(feature = "semantic")]
pub(crate) fn rebuild_vector_index(connection: &Connection) -> Result<usize> {
    let collected: Vec<(String, Vec<u8>, i64)> = {
        let mut statement = connection.prepare(
            "SELECT memory_id, vector_blob, dimensions FROM embeddings \
             WHERE vector_blob IS NOT NULL AND dimensions IS NOT NULL AND status = 'ready'",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let mut count = 0usize;
    for (memory_id, blob, dims_i64) in collected {
        let Ok(dims) = usize::try_from(dims_i64) else {
            continue;
        };
        let embedding = blob_to_embedding(&blob);
        if dims == 0 || embedding.len() != dims {
            continue;
        }
        let table = ensure_memory_vector_table(connection, dims)?;
        connection.execute(
            &format!("DELETE FROM {table} WHERE memory_id = ?1"),
            params![memory_id],
        )?;
        let embedding_json = embedding_json(&embedding)?;
        connection.execute(
            &format!("INSERT INTO {table} (memory_id, embedding) VALUES (?1, ?2)"),
            params![memory_id, embedding_json],
        )?;
        count += 1;
    }
    Ok(count)
}

/// Rebuild the semantic ANN index from the canonical `embeddings` table.
///
/// Re-projects stored vectors into the `memory_vec_<dims>` index without
/// re-running the embedding model. Useful after a logical import or to recover
/// the vector index. Returns the number of vectors reindexed.
///
/// # Errors
///
/// Returns an error if the store is missing/uninitialized or `SQLite` rejects a
/// statement.
#[cfg(feature = "semantic")]
pub fn reindex_vectors(path: impl AsRef<Path>) -> Result<usize> {
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let count = rebuild_vector_index(&transaction)?;
    transaction.commit()?;
    Ok(count)
}

/// One active memory's content, to be re-embedded under a new model.
#[cfg(feature = "semantic")]
pub struct ReembedTarget {
    /// Memory id.
    pub memory_id: String,
    /// Active version id.
    pub version_id: String,
    /// Active version content to embed.
    pub content: String,
}

/// Collect every active memory's content for a model-switching re-embed.
///
/// # Errors
///
/// Returns an error if the store is missing/uninitialized or `SQLite` rejects a
/// statement.
#[cfg(feature = "semantic")]
pub fn collect_reembed_targets(path: impl AsRef<Path>) -> Result<Vec<ReembedTarget>> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    let mut statement = connection.prepare(
        "SELECT m.id, m.active_version_id, v.content
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.status = 'active'
         ORDER BY m.id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(ReembedTarget {
            memory_id: row.get(0)?,
            version_id: row.get(1)?,
            content: row.get(2)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

/// Collect active memories needing late-interaction token embeddings.
///
/// Text uses the persisted representation when present, otherwise summary,
/// followed by canonical content.
/// With `force`, every active memory is returned (model switch / re-embed);
/// otherwise only memories without a token row.
///
/// # Errors
///
/// Returns an error if the store is missing/uninitialized or `SQLite` rejects a
/// statement.
pub fn collect_token_backfill_targets(
    path: impl AsRef<Path>,
    force: bool,
) -> Result<Vec<(String, String)>> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    let sql = if force {
        "SELECT m.id, COALESCE(r.text, v.summary, ''), v.content FROM memories m \
         JOIN memory_versions v ON v.id = m.active_version_id \
         LEFT JOIN memory_representations r ON r.version_id = v.id \
         WHERE m.status = 'active' ORDER BY m.id"
    } else {
        "SELECT m.id, COALESCE(r.text, v.summary, ''), v.content FROM memories m \
         JOIN memory_versions v ON v.id = m.active_version_id \
         LEFT JOIN memory_representations r ON r.version_id = v.id \
         LEFT JOIN memory_token_embeddings t ON t.memory_id = m.id \
         WHERE m.status = 'active' AND t.memory_id IS NULL ORDER BY m.id"
    };
    let mut statement = connection.prepare(sql)?;
    let rows = statement.query_map([], |row| {
        let id: String = row.get(0)?;
        let companion: String = row.get(1)?;
        let content: String = row.get(2)?;
        let text = representation_document(
            &content,
            (!companion.is_empty()).then_some(companion.as_str()),
        );
        Ok((id, text))
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

/// Write a batch of late-interaction token embeddings (backfill / model switch).
///
/// With `reset`, clears existing token rows and the active-model record first.
/// Returns the number of rows written.
///
/// # Errors
///
/// Returns an error if the store is missing or `SQLite` rejects a statement.
pub fn apply_token_embeddings(
    path: impl AsRef<Path>,
    model_id: &str,
    rows: &[(String, Vec<Vec<f32>>)],
    reset: bool,
) -> Result<usize> {
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    if reset {
        transaction.execute("DELETE FROM memory_token_embeddings", [])?;
        transaction.execute(
            "DELETE FROM config_kv WHERE key = 'active_colbert_model'",
            [],
        )?;
    }
    let now = now_timestamp(&transaction)?;
    enforce_active_colbert_model(&transaction, model_id, &now)?;
    let mut written = 0usize;
    for (memory_id, vecs) in rows {
        if vecs.is_empty() {
            continue;
        }
        upsert_memory_token_embedding(&transaction, memory_id, model_id, vecs)?;
        written += 1;
    }
    transaction.commit()?;
    Ok(written)
}

/// Replace every stored embedding with vectors produced by a new model.
///
/// Clears the canonical embeddings table and all ANN index tables, resets the
/// active embedding model record, then writes the supplied vectors. Returns the
/// number of vectors written.
///
/// # Errors
///
/// Returns an error if a vector length does not match `dims`, the store is
/// missing, or `SQLite` rejects a statement.
#[cfg(feature = "semantic")]
pub fn apply_reembed(
    path: impl AsRef<Path>,
    model_id: &str,
    dims: usize,
    vectors: &[(String, String, Vec<f32>)],
) -> Result<usize> {
    semantic_table_for_dims(dims)?;
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let now = now_timestamp(&transaction)?;
    transaction.execute("DELETE FROM embeddings", [])?;
    drop_all_vector_tables(&transaction)?;
    set_config_value(&transaction, "active_embedding_model", model_id, &now)?;
    set_config_value(
        &transaction,
        "active_embedding_dims",
        &dims.to_string(),
        &now,
    )?;
    let mut count = 0usize;
    for (memory_id, version_id, embedding) in vectors {
        if embedding.len() != dims {
            return Err(Error::InvalidRequest {
                message: format!(
                    "re-embed produced dimension {} but {dims} was expected",
                    embedding.len()
                ),
            });
        }
        write_embedding_row(
            &transaction,
            memory_id,
            version_id,
            model_id,
            dims,
            embedding,
            &now,
        )?;
        count += 1;
    }
    transaction.commit()?;
    Ok(count)
}
