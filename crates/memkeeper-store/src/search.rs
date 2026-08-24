//! Memory search and listing extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use rusqlite::{params, params_from_iter, types::Value, Connection, OptionalExtension, Row};

use memkeeper_core::{status, ALL_SPACES, DEFAULT_DURABLE_SILO, DEFAULT_SPACE};

use crate::{
    bounded_char_slice, collect_rows, is_supported_kind, is_supported_scope, is_supported_status,
    limit_i64, make_snippet, normalize_search_filters, normalized_tags, open_initialized_read_fast,
    validate_batch_search_request, validate_optional_embedding, validate_search_filters,
    with_read_snapshot, BatchSearchItemReport, BatchSearchReport, BatchSearchRequest, Error,
    MemoryListItem, MemoryListReport, MemoryListRequest, Result, ScoreBreakdown, SearchFilters,
    SearchReport, SearchRequest, SearchResult, DURABLE_RECENCY_HALF_LIFE_DAYS,
    MAX_BATCH_QUERY_LIMIT, MAX_MEMORY_LIST_LIMIT, MAX_RECENCY_SCORE, MAX_SEARCH_LIMIT,
    MAX_SEARCH_OFFSET, MAX_SEARCH_QUERY_CHARS, MAX_SEARCH_TERMS, MAX_SNIPPET_CHARS, MAX_TAGS,
    SHORT_TERM_SILO, VOLATILE_RECENCY_HALF_LIFE_DAYS,
};

#[cfg(feature = "semantic")]
use crate::{maxsim_candidates, maxsim_shortlist_ids, semantic_table_for_dims, table_exists};

pub fn search_memories(path: impl AsRef<Path>, request: &SearchRequest) -> Result<SearchReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        search_memories_on_connection(connection, request)
    })
}

/// List recent memories deterministically for review/admin workflows.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// or `SQLite` rejects the query.
pub fn list_memories(
    path: impl AsRef<Path>,
    request: &MemoryListRequest,
) -> Result<MemoryListReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        list_memories_on_connection(connection, request)
    })
}

/// Run multiple deterministic searches against one read-only store snapshot.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// or `SQLite` rejects a query.
pub fn batch_search_memories(
    path: impl AsRef<Path>,
    request: &BatchSearchRequest,
) -> Result<BatchSearchReport> {
    validate_batch_search_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        batch_search_memories_on_connection(connection, request)
    })
}

pub(crate) fn batch_search_memories_on_connection(
    connection: &Connection,
    request: &BatchSearchRequest,
) -> Result<BatchSearchReport> {
    let mut results = Vec::with_capacity(request.queries.len());
    // One reusable request; only the per-query fields change between loops.
    let mut search_request = SearchRequest {
        query: String::new(),
        filters: request.common_filters.clone(),
        limit: request.limit,
        offset: request.offset,
        snippet_chars: request.snippet_chars,
        include_content: request.include_content,
        include_source: request.include_source,
        semantic_fallback: request.semantic_fallback.clone(),
        lexical_fallback: "conservative".to_string(),
        embedding: None,
        query_token_embedding: None,
        token_model_id: None,
        maxsim_shortlist: 0,
    };
    for query in &request.queries {
        search_request.query.clone_from(&query.query);
        search_request.limit = query.limit.unwrap_or(request.limit);
        results.push(BatchSearchItemReport {
            name: query.name.clone(),
            query: query.query.clone(),
            report: search_memories_on_connection(connection, &search_request)?,
        });
    }
    Ok(BatchSearchReport { results })
}

pub(crate) fn list_memories_on_connection(
    connection: &Connection,
    request: &MemoryListRequest,
) -> Result<MemoryListReport> {
    let prepared = prepare_memory_list_request(request)?;
    let mut candidates = list_memory_candidates(connection, &prepared)?;
    let truncated = candidates.len() > prepared.limit;
    if truncated {
        candidates.truncate(prepared.limit);
    }
    let total_estimate = if candidates.is_empty() && !truncated {
        0
    } else {
        prepared
            .offset
            .saturating_add(candidates.len())
            .saturating_add(usize::from(truncated))
    };
    let results = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| candidate.into_item(prepared.offset + index + 1, &prepared))
        .collect();
    Ok(MemoryListReport {
        strategy: "deterministic_list_v0".to_string(),
        total_estimate,
        truncated,
        results,
    })
}

pub(crate) fn search_memories_on_connection(
    connection: &Connection,
    request: &SearchRequest,
) -> Result<SearchReport> {
    let prepared = prepare_search_request(request)?;
    search_prepared(connection, &prepared)
}

pub(crate) fn search_prepared(
    connection: &Connection,
    prepared: &PreparedSearchRequest,
) -> Result<SearchReport> {
    // Semantic-primary: when a query embedding and its ANN index are available,
    // semantic relevance is the primary ranker. BM25/FTS is graceful degradation
    // (no embedding, a missing index, or an empty semantic result set).
    #[cfg(feature = "semantic")]
    if prepared.semantic_fallback != "disabled" {
        if let Some(embedding) = prepared.embedding.as_ref() {
            let table = semantic_table_for_dims(embedding.len())?;
            if table_exists(connection, &table)? {
                let report = semantic_ranked_report(
                    connection,
                    prepared,
                    embedding,
                    "semantic_primary_v0",
                    "semantic_primary",
                )?;
                if !report.results.is_empty() {
                    return Ok(report);
                }
            }
        }
    }

    let mut candidates = search_candidates(connection, prepared, &prepared.fts_query)?;
    if prepared.lexical_fallback != "disabled" && candidates.len() < prepared.limit {
        if let Some(prefix_fts_query) = prepared.prefix_fts_query.as_deref() {
            fill_lexical_candidates(connection, prepared, &mut candidates, prefix_fts_query, 1)?;
        }
    }
    if prepared.lexical_fallback != "disabled" && candidates.is_empty() {
        if let Some(fallback_fts_query) = prepared.fallback_fts_query.as_deref() {
            fill_lexical_candidates(connection, prepared, &mut candidates, fallback_fts_query, 2)?;
        }
    }

    // Best (most negative) bm25 in the matched set anchors the relative FTS
    // normalization in `fts_score`. Computed across all candidates before the
    // per-candidate scoring pass so the top match normalizes to 1.0.
    let best_bm25 = candidates
        .iter()
        .map(|candidate| candidate.bm25)
        .fold(f64::INFINITY, f64::min);
    let now_jd = now_julian_day(connection)?;
    for candidate in &mut candidates {
        candidate.score = score_candidate(candidate, prepared, best_bm25, now_jd);
    }
    candidates.sort_by(compare_candidates);

    let truncated = candidates.len() > prepared.limit;
    if truncated {
        candidates.truncate(prepared.limit);
    }
    let total_estimate = prepared
        .offset
        .saturating_add(candidates.len())
        .saturating_add(usize::from(truncated));
    let results = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| candidate.into_result(prepared.offset + index + 1, prepared))
        .collect::<Vec<_>>();

    let report = SearchReport {
        strategy: "deterministic_fts_v0".to_string(),
        semantic_attempted: false,
        semantic_reason: if prepared.semantic_fallback == "disabled" {
            "disabled_v0_1"
        } else {
            "fts_results"
        }
        .to_string(),
        total_estimate,
        truncated,
        results,
    };

    if !report.results.is_empty() || prepared.semantic_fallback == "disabled" {
        return Ok(report);
    }

    semantic_fallback_search(connection, prepared, report)
}

#[cfg(not(feature = "semantic"))]
#[allow(clippy::unnecessary_wraps)]
fn semantic_fallback_search(
    _connection: &Connection,
    prepared: &PreparedSearchRequest,
    mut empty_fts_report: SearchReport,
) -> Result<SearchReport> {
    let _embedding_was_supplied = prepared.embedding.is_some();
    empty_fts_report.semantic_reason = "semantic_feature_disabled".to_string();
    Ok(empty_fts_report)
}

fn fill_lexical_candidates(
    connection: &Connection,
    prepared: &PreparedSearchRequest,
    candidates: &mut Vec<SearchCandidate>,
    fts_query: &str,
    lexical_tier: u8,
) -> Result<()> {
    let mut seen = candidates
        .iter()
        .map(|candidate| candidate.memory_id.clone())
        .collect::<BTreeSet<_>>();
    for mut candidate in search_candidates(connection, prepared, fts_query)? {
        if seen.insert(candidate.memory_id.clone()) {
            candidate.lexical_tier = lexical_tier;
            candidates.push(candidate);
        }
    }
    Ok(())
}

#[cfg(feature = "semantic")]
fn semantic_fallback_search(
    connection: &Connection,
    prepared: &PreparedSearchRequest,
    mut empty_fts_report: SearchReport,
) -> Result<SearchReport> {
    let Some(embedding) = prepared.embedding.as_ref() else {
        empty_fts_report.semantic_reason = "missing_embedding".to_string();
        return Ok(empty_fts_report);
    };
    let table = semantic_table_for_dims(embedding.len())?;
    if !table_exists(connection, &table)? {
        empty_fts_report.semantic_reason = "semantic_index_missing".to_string();
        return Ok(empty_fts_report);
    }
    semantic_ranked_report(
        connection,
        prepared,
        embedding,
        "semantic_fallback",
        "fts_empty",
    )
}

/// Rank active memories by semantic (ANN) relevance to the query embedding.
/// Shared by the semantic-primary path and the empty-FTS degradation path.
#[cfg(feature = "semantic")]
fn semantic_ranked_report(
    connection: &Connection,
    prepared: &PreparedSearchRequest,
    embedding: &[f32],
    strategy: &str,
    reason: &str,
) -> Result<SearchReport> {
    let mut candidates = semantic_candidates(connection, prepared, embedding)?;
    let now_jd = now_julian_day(connection)?;
    for candidate in &mut candidates {
        candidate.score = score_semantic_candidate(candidate, prepared, now_jd);
    }
    candidates.sort_by(compare_candidates);
    let truncated = candidates.len() > prepared.limit;
    if truncated {
        candidates.truncate(prepared.limit);
    }
    let total_estimate = prepared
        .offset
        .saturating_add(candidates.len())
        .saturating_add(usize::from(truncated));
    let results = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| candidate.into_result(prepared.offset + index + 1, prepared))
        .collect::<Vec<_>>();
    Ok(SearchReport {
        strategy: strategy.to_string(),
        semantic_attempted: true,
        semantic_reason: reason.to_string(),
        total_estimate,
        truncated,
        results,
    })
}

struct PreparedMemoryListRequest {
    filters: SearchFilters,
    limit: usize,
    offset: usize,
    snippet_chars: usize,
    include_content: bool,
    include_source: bool,
    order: String,
}

#[derive(Debug, Clone)]
struct MemoryListCandidate {
    memory_id: String,
    version_id: String,
    space: String,
    silo: String,
    scope: String,
    project_key: Option<String>,
    kind: String,
    status: String,
    entity_key: Option<String>,
    claim_key: Option<String>,
    confidence: f64,
    pinned: bool,
    observed_at: String,
    created_at: String,
    updated_at: String,
    snippet_text: String,
    content: Option<String>,
    summary: Option<String>,
    tags: Vec<String>,
    source_ref_json: Option<String>,
}

impl MemoryListCandidate {
    fn into_item(self, rank: usize, request: &PreparedMemoryListRequest) -> MemoryListItem {
        MemoryListItem {
            rank,
            memory_id: self.memory_id,
            version_id: self.version_id,
            space: self.space,
            silo: self.silo,
            scope: self.scope,
            project_key: self.project_key,
            kind: self.kind,
            status: self.status,
            summary: self.summary,
            snippet: bounded_char_slice(&self.snippet_text, 0, request.snippet_chars),
            content: self.content.filter(|_| request.include_content),
            tags: self.tags,
            entity_key: self.entity_key,
            claim_key: self.claim_key,
            confidence: self.confidence,
            pinned: self.pinned,
            observed_at: self.observed_at,
            created_at: self.created_at,
            updated_at: self.updated_at,
            source_ref_json: self.source_ref_json.filter(|_| request.include_source),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PreparedSearchRequest {
    fts_query: String,
    fallback_fts_query: Option<String>,
    prefix_fts_query: Option<String>,
    terms: Vec<String>,
    /// Normalized 1..=3-word contiguous shingles of the raw query, used to match
    /// reserved `alias::<normalized>` tags for the alias-exact-match boost. Built
    /// from the raw query (not `terms`, which is sorted/deduped and loses order).
    query_alias_shingles: std::collections::HashSet<String>,
    filters: SearchFilters,
    pub(crate) limit: usize,
    /// Inflated SQL LIMIT for the initial candidate fetch. Always >= limit.
    /// Gives Rust re-scoring room to surface candidates SQL would otherwise
    /// undervalue (e.g. recent volatile memories boosted 4x by the silo-aware
    /// recency curve but ranked equal to durable by SQL's uniform recency score).
    candidate_pool_limit: usize,
    offset: usize,
    snippet_chars: usize,
    include_content: bool,
    include_source: bool,
    semantic_fallback: String,
    lexical_fallback: String,
    embedding: Option<Vec<f32>>,
    // Read only by the semantic-gated late-interaction candidate path.
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    query_token_embedding: Option<Vec<Vec<f32>>>,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    token_model_id: Option<String>,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    maxsim_shortlist: usize,
}

pub(crate) struct SearchCandidate {
    memory_id: String,
    version_id: String,
    space: String,
    silo: String,
    scope: String,
    project_key: Option<String>,
    kind: String,
    status: String,
    summary: Option<String>,
    content: Option<String>,
    snippet_text: String,
    tags: Vec<String>,
    entity_key: Option<String>,
    claim_key: Option<String>,
    observed_at: String,
    recency_jd: Option<f64>,
    source_ref_json: Option<String>,
    /// Coarse source `type` (e.g. `mcp`, `manual`, `synthesis`) extracted for
    /// ranking only. Always loaded, unlike `source_ref_json`, which is gated
    /// behind `include_source` for output privacy.
    source_type: Option<String>,
    metadata_json: Option<String>,
    confidence: f64,
    pinned: bool,
    bm25: f64,
    lexical_tier: u8,
    pub(crate) score: f64,
    scores: ScoreBreakdown,
}

impl SearchCandidate {
    /// Consumes the candidate; result assembly moves the owned strings
    /// instead of cloning them (the candidate vec is always discarded after
    /// this conversion).
    pub(crate) fn into_result(self, rank: usize, request: &PreparedSearchRequest) -> SearchResult {
        let snippet = self.snippet(request);
        SearchResult {
            rank,
            memory_id: self.memory_id,
            version_id: self.version_id,
            score: self.score,
            scores: self.scores,
            space: self.space,
            silo: self.silo,
            scope: self.scope,
            project_key: self.project_key,
            kind: self.kind,
            status: self.status,
            summary: self.summary,
            snippet,
            content: self.content.filter(|_| request.include_content),
            tags: self.tags,
            entity_key: self.entity_key,
            claim_key: self.claim_key,
            observed_at: self.observed_at,
            source_ref_json: request
                .include_source
                .then_some(self.source_ref_json)
                .flatten(),
            metadata_json: self.metadata_json,
        }
    }

    fn snippet(&self, request: &PreparedSearchRequest) -> String {
        if request.snippet_chars == 0 {
            return String::new();
        }
        if let Some(content) = &self.content {
            make_snippet(content, &request.terms, request.snippet_chars)
        } else {
            bounded_char_slice(&self.snippet_text, 0, request.snippet_chars)
        }
    }
}
pub(crate) fn prepare_memory_list_request(
    request: &MemoryListRequest,
) -> Result<PreparedMemoryListRequest> {
    if request.limit == 0 || request.limit > MAX_MEMORY_LIST_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("limit must be between 1 and {MAX_MEMORY_LIST_LIMIT}"),
        });
    }
    if request.offset > MAX_SEARCH_OFFSET {
        return Err(Error::InvalidRequest {
            message: format!("offset must be at most {MAX_SEARCH_OFFSET}"),
        });
    }
    if request.snippet_chars > MAX_SNIPPET_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("snippet_chars must be at most {MAX_SNIPPET_CHARS}"),
        });
    }
    if !matches!(
        request.order.as_str(),
        "updated_desc" | "observed_desc" | "created_desc"
    ) {
        return Err(Error::InvalidRequest {
            message: "order must be updated_desc, observed_desc, or created_desc".to_string(),
        });
    }
    let mut filters = request.filters.clone();
    filters.spaces = resolve_space_filter(&filters.spaces);
    if filters.statuses.is_empty() {
        filters.statuses.push(status::ACTIVE.to_string());
    }
    filters = normalize_search_filters(filters)?;
    validate_search_filters(&filters)?;
    Ok(PreparedMemoryListRequest {
        filters,
        limit: request.limit,
        offset: request.offset,
        snippet_chars: request.snippet_chars,
        include_content: request.include_content,
        include_source: request.include_source,
        order: request.order.clone(),
    })
}

/// Reserved tag prefix marking an alias/canonical surface form for a memory.
/// A memory tagged `alias::k8s` is boosted when a query contains the token `k8s`.
pub(crate) const ALIAS_TAG_PREFIX: &str = "alias::";
/// Additive boost applied once when any query shingle matches a candidate's
/// `alias::` tag. Sized in the `fts_score` band (max 1.0) so an exact alias hit
/// lifts a topically-weak-but-correct match above semantic noise near the
/// abstention floor, without overriding a strong topical match outright.
pub(crate) const ALIAS_MATCH_BOOST: f64 = 0.5;
/// Longest multi-word alias we shingle for (e.g. "point of view", "ci pipeline").
const MAX_ALIAS_SHINGLE_WORDS: usize = 3;

/// Build the set of normalized 1..=`MAX_ALIAS_SHINGLE_WORDS`-word contiguous
/// shingles from the raw query, preserving word order. Normalization matches
/// `normalized_alias` (lowercase + single-space) so shingles compare directly
/// against the suffix of an `alias::<normalized>` tag.
pub(crate) fn query_alias_words(query: &str) -> Vec<String> {
    query
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .collect()
}

pub(crate) fn query_alias_shingles(query: &str) -> std::collections::HashSet<String> {
    let words = query_alias_words(query);
    let mut shingles = std::collections::HashSet::new();
    for start in 0..words.len() {
        for span in 1..=MAX_ALIAS_SHINGLE_WORDS {
            if start + span <= words.len() {
                shingles.insert(words[start..start + span].join(" "));
            }
        }
    }
    shingles
}

pub(crate) fn prepare_search_request(request: &SearchRequest) -> Result<PreparedSearchRequest> {
    if request.limit == 0 || request.limit > MAX_SEARCH_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("limit must be between 1 and {MAX_SEARCH_LIMIT}"),
        });
    }
    if request.offset > MAX_SEARCH_OFFSET {
        return Err(Error::InvalidRequest {
            message: format!("offset must be at most {MAX_SEARCH_OFFSET}"),
        });
    }
    if request.snippet_chars > MAX_SNIPPET_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("snippet_chars must be at most {MAX_SNIPPET_CHARS}"),
        });
    }
    if !matches!(request.semantic_fallback.as_str(), "disabled" | "fallback") {
        return Err(Error::InvalidRequest {
            message: "semantic_fallback must be disabled or fallback".to_string(),
        });
    }
    if !matches!(
        request.lexical_fallback.as_str(),
        "disabled" | "conservative"
    ) {
        return Err(Error::InvalidRequest {
            message: "lexical_fallback must be disabled or conservative".to_string(),
        });
    }
    validate_optional_embedding("embedding", request.embedding.as_deref())?;

    if request.query.chars().count() > MAX_SEARCH_QUERY_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("query must be at most {MAX_SEARCH_QUERY_CHARS} characters"),
        });
    }
    let terms = search_terms(&request.query);
    if terms.len() > MAX_SEARCH_TERMS {
        return Err(Error::InvalidRequest {
            message: format!("query must contain at most {MAX_SEARCH_TERMS} searchable terms"),
        });
    }
    if terms.is_empty() {
        return Err(Error::InvalidRequest {
            message: "query must contain at least one searchable term".to_string(),
        });
    }

    let filters = prepare_recall_filters(&request.filters)?;

    let fts_terms = search_variant_terms(&terms);
    let fts_query = search_fts_query(&fts_terms, request.include_source, " AND ");
    let fallback_fts_query = if fts_terms.len() > 1 {
        Some(search_fts_query(&fts_terms, request.include_source, " OR "))
            .filter(|fallback| fallback != &fts_query)
    } else {
        None
    };
    let prefix_terms = search_prefix_terms(&terms);
    let prefix_fts_query = if prefix_terms.is_empty() {
        None
    } else {
        Some(search_fts_query(
            &prefix_terms,
            request.include_source,
            " AND ",
        ))
        .filter(|prefix| prefix != &fts_query)
    };

    // Inflate the SQL candidate pool so Rust re-scoring can surface memories
    // that SQL undervalues (e.g. recent volatile memories get a 4x recency
    // boost in Rust but look equal-recency to durable in SQL). Use a generous
    // floor — max(limit * 4, limit + 32) — capped at MAX_SEARCH_LIMIT * 4 to
    // stay bounded. Final result is always truncated to `limit` after Rust sort.
    let candidate_pool_limit = request
        .limit
        .saturating_mul(4)
        .max(request.limit.saturating_add(32))
        .min(MAX_SEARCH_LIMIT.saturating_mul(4));

    Ok(PreparedSearchRequest {
        fts_query,
        fallback_fts_query,
        prefix_fts_query,
        terms,
        query_alias_shingles: query_alias_shingles(&request.query),
        filters,
        limit: request.limit,
        candidate_pool_limit,
        offset: request.offset,
        snippet_chars: request.snippet_chars,
        include_content: request.include_content,
        include_source: request.include_source,
        semantic_fallback: request.semantic_fallback.clone(),
        lexical_fallback: request.lexical_fallback.clone(),
        embedding: request.embedding.clone(),
        query_token_embedding: request.query_token_embedding.clone(),
        token_model_id: request.token_model_id.clone(),
        maxsim_shortlist: request.maxsim_shortlist,
    })
}

fn search_fts_query(terms: &[String], include_source: bool, joiner: &str) -> String {
    if include_source {
        terms.join(joiner)
    } else {
        terms
            .iter()
            .map(|term| format!("{{content retrieval_text tags metadata_text}} : {term}"))
            .collect::<Vec<_>>()
            .join(joiner)
    }
}

fn search_variant_terms(terms: &[String]) -> Vec<String> {
    terms
        .iter()
        .map(|term| search_term_expression(&search_term_variants(term)))
        .collect()
}

fn search_prefix_terms(terms: &[String]) -> Vec<String> {
    let mut has_prefix_term = false;
    let prefix_terms = terms
        .iter()
        .map(|term| {
            let mut variants = search_term_variants(term);
            for variant in variants.clone() {
                if is_prefixable_search_term(&variant) {
                    has_prefix_term = true;
                    push_unique(&mut variants, format!("{variant}*"));
                }
            }
            search_term_expression(&variants)
        })
        .collect();
    if has_prefix_term {
        prefix_terms
    } else {
        Vec::new()
    }
}

fn search_term_expression(variants: &[String]) -> String {
    if variants.len() == 1 {
        variants[0].clone()
    } else {
        format!("({})", variants.join(" OR "))
    }
}

fn search_term_variants(term: &str) -> Vec<String> {
    let mut variants = vec![term.to_string()];
    if term.chars().count() < 5 || !is_prefixable_search_term(term) {
        return variants;
    }

    for stem in search_term_stems(term) {
        if stem.chars().count() >= 4 && stem != term {
            push_unique(&mut variants, stem);
        }
    }
    variants
}

pub(crate) fn search_term_stems(term: &str) -> Vec<String> {
    let mut stems = Vec::new();
    if let Some(stem) = term.strip_suffix("ies") {
        if !stem.is_empty() {
            stems.push(format!("{stem}y"));
        }
    }
    if let Some(stem) = term.strip_suffix("ied") {
        if !stem.is_empty() {
            stems.push(format!("{stem}y"));
        }
    }
    if let Some(stem) = term.strip_suffix("ing") {
        if stem.chars().count() >= 4 {
            stems.push(trim_doubled_suffix(stem).to_string());
        }
    }
    if let Some(stem) = term.strip_suffix("ed") {
        if stem.chars().count() >= 4 {
            stems.push(trim_doubled_suffix(stem).to_string());
        }
    }
    if let Some(stem) = term.strip_suffix("es") {
        if stem.chars().count() >= 4 {
            stems.push(stem.to_string());
        }
    }
    if term.ends_with('s') && !term.ends_with("ss") {
        let stem = term.trim_end_matches('s');
        if stem.chars().count() >= 4 {
            stems.push(stem.to_string());
        }
    }
    if term.ends_with("ate") || term.ends_with("ize") || term.ends_with("ise") {
        if let Some(stem) = term.strip_suffix('e') {
            if stem.chars().count() >= 4 {
                stems.push(stem.to_string());
            }
        }
    }
    stems
}

fn trim_doubled_suffix(value: &str) -> &str {
    let mut chars = value.char_indices().rev();
    let Some((last_index, last)) = chars.next() else {
        return value;
    };
    let Some((previous_index, previous)) = chars.next() else {
        return value;
    };
    if last == previous {
        &value[..last_index.max(previous_index)]
    } else {
        value
    }
}

pub(crate) fn is_prefixable_search_term(term: &str) -> bool {
    term.chars().count() >= 4
        && term
            .chars()
            .all(|character| character.is_alphanumeric() || character == '_')
}

pub(crate) fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}

/// Resolve a raw read-side space filter into predicate-ready values.
///
/// Three scopes, chosen so that empty stays back-compatible (existing scoped
/// callers rely on empty == default) while `["*"]` gains an explicit "every
/// space" meaning:
/// - contains the [`ALL_SPACES`] sentinel -> **all spaces**: an empty vector,
///   which [`push_in_predicate`] renders as *no* `space_name` predicate.
/// - empty -> **default space**: `[DEFAULT_SPACE]`.
/// - named -> **those spaces**, unchanged (SQL `IN (...)`).
///
/// The sentinel is matched trimmed so a stray `" * "` from a host still resolves
/// to all-spaces rather than being treated as a (non-existent) literal space.
pub(crate) fn resolve_space_filter(spaces: &[String]) -> Vec<String> {
    if spaces.iter().any(|value| value.trim() == ALL_SPACES) {
        Vec::new()
    } else if spaces.is_empty() {
        vec![DEFAULT_SPACE.to_string()]
    } else {
        spaces.to_vec()
    }
}

/// Reject the reserved all-spaces sentinel on any write path. A real memory
/// (or candidate) must never live in space `*`, or it would collide with the
/// read-side union scope and become impossible to address explicitly. Entity,
/// relationship, and space-create paths are already guarded at
/// `normalize_required_space_component`; this covers the write paths that only
/// run `validate_optional_metadata_value` (remember, `candidate_submit`, ingest).
pub(crate) fn prepare_recall_filters(filters: &SearchFilters) -> Result<SearchFilters> {
    let mut filters = filters.clone();
    filters.spaces = resolve_space_filter(&filters.spaces);
    if filters.statuses.is_empty() {
        filters.statuses.push(status::ACTIVE.to_string());
    }
    // Recall must not surface logically stale facts. Past-`valid_to` and
    // reached-`expires_at` memories are excluded before any retrieval route
    // selects candidates, even before the dream expire task deletes them.
    filters.hide_expired = true;
    filters = normalize_search_filters(filters)?;
    validate_search_filters(&filters)?;
    Ok(filters)
}

pub(crate) fn search_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    for character in query.chars() {
        if character.is_alphanumeric() || character == '_' {
            current.push(character.to_ascii_lowercase());
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        terms.push(current);
    }
    terms.sort();
    terms.dedup();
    let filtered_terms = terms
        .iter()
        .filter(|term| !is_search_stopword(term))
        .cloned()
        .collect::<Vec<_>>();
    if filtered_terms.is_empty() {
        terms
    } else {
        filtered_terms
    }
}

fn is_search_stopword(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "been"
            | "but"
            | "by"
            | "can"
            | "could"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "had"
            | "has"
            | "have"
            | "he"
            | "her"
            | "hers"
            | "him"
            | "his"
            | "how"
            | "i"
            | "in"
            | "is"
            | "it"
            | "its"
            | "of"
            | "on"
            | "or"
            | "our"
            | "s"
            | "she"
            | "should"
            | "t"
            | "that"
            | "the"
            | "their"
            | "them"
            | "then"
            | "there"
            | "they"
            | "this"
            | "to"
            | "was"
            | "were"
            | "what"
            | "which"
            | "who"
            | "why"
            | "will"
            | "with"
            | "would"
            | "you"
            | "your"
    )
}

fn list_memory_candidates(
    connection: &Connection,
    request: &PreparedMemoryListRequest,
) -> Result<Vec<MemoryListCandidate>> {
    let (sql, args) = memory_list_sql(request);
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(args.iter()), |row| {
        let tags_joined = row.get::<_, Option<String>>(17)?.unwrap_or_default();
        Ok(MemoryListCandidate {
            memory_id: row.get(0)?,
            version_id: row.get(1)?,
            space: row.get(2)?,
            silo: row.get(3)?,
            scope: row.get(4)?,
            project_key: row.get(5)?,
            kind: row.get(6)?,
            status: row.get(7)?,
            entity_key: row.get(8)?,
            claim_key: row.get(9)?,
            confidence: row.get(10)?,
            pinned: row.get::<_, i64>(11)? == 1,
            observed_at: row.get(12)?,
            created_at: row.get(13)?,
            updated_at: row.get(14)?,
            snippet_text: row.get(15)?,
            content: row.get(16)?,
            tags: split_tags(&tags_joined),
            summary: row.get(18)?,
            source_ref_json: row.get(19)?,
        })
    })?;
    collect_rows(rows)
}

fn memory_list_sql(request: &PreparedMemoryListRequest) -> (String, Vec<String>) {
    let mut args = SqlArgs::with_reserved(0);
    let where_clause = filters_where_clause(&request.filters, &mut args);
    let order = memory_list_order_sql(&request.order);
    let row_limit = request.limit.saturating_add(1);
    let row_offset = request.offset;
    let snippet_sql = if request.snippet_chars == 0 {
        "''".to_string()
    } else {
        format!(
            "substr(COALESCE(v.summary, v.content), 1, {})",
            request.snippet_chars.saturating_add(1)
        )
    };
    let content_sql = if request.include_content {
        "v.content"
    } else {
        "NULL"
    };
    let source_sql = if request.include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let sql = format!(
        "SELECT
            m.id,
            m.active_version_id,
            m.space_name,
            m.silo_name,
            m.scope,
            m.project_key,
            m.kind,
            m.status,
            m.entity_key,
            m.claim_key,
            m.confidence,
            m.pinned,
            m.observed_at,
            m.created_at,
            m.updated_at,
            {snippet_sql},
            {content_sql},
            COALESCE((SELECT group_concat(tag, char(31)) FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)), ''),
            v.summary,
            {source_sql}
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {where_clause}
         ORDER BY {order}, m.id ASC
         LIMIT {row_limit} OFFSET {row_offset}"
    );
    (sql, args.values)
}

fn memory_list_order_sql(order: &str) -> &'static str {
    match order {
        "observed_desc" => "m.observed_at DESC, m.updated_at DESC",
        "created_desc" => "m.created_at DESC, m.updated_at DESC",
        _ => "m.updated_at DESC, m.observed_at DESC",
    }
}

/// Maps one row of a candidate SELECT (see `candidate_select_columns`) onto a
/// `SearchCandidate`. The column order here and in `candidate_select_columns`
/// must stay in lockstep; sharing one mapper keeps the FTS and semantic paths
/// from drifting apart.
fn candidate_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SearchCandidate> {
    let tags_joined = row.get::<_, Option<String>>(13)?.unwrap_or_default();
    Ok(SearchCandidate {
        memory_id: row.get(0)?,
        version_id: row.get(1)?,
        space: row.get(2)?,
        silo: row.get(3)?,
        scope: row.get(4)?,
        project_key: row.get(5)?,
        kind: row.get(6)?,
        status: row.get(7)?,
        entity_key: row.get(8)?,
        claim_key: row.get(9)?,
        observed_at: row.get(10)?,
        recency_jd: row.get(11)?,
        pinned: row.get::<_, i64>(12)? == 1,
        tags: split_tags(&tags_joined),
        confidence: row.get(14)?,
        content: row.get(15)?,
        snippet_text: row.get(16)?,
        summary: row.get(17)?,
        source_ref_json: row.get(18)?,
        bm25: row.get(19)?,
        metadata_json: row.get(20)?,
        source_type: row.get(21)?,
        score: 0.0,
        lexical_tier: 0,
        scores: ScoreBreakdown {
            fts: 0.0,
            metadata: 0.0,
            recency: 0.0,
            scope: 0.0,
            status: 0.0,
            pin: 0.0,
            source_tier: 0.0,
        },
    })
}

/// Shared SELECT column list consumed by `candidate_from_row`. `rank_sql` is
/// the path-specific relevance column (FTS bm25 or vector distance).
fn candidate_select_columns(
    recency_jd: &str,
    content_sql: &str,
    snippet_sql: &str,
    source_sql: &str,
    rank_sql: &str,
) -> String {
    // `{recency_jd}` and `{rank_sql}` are aliased so a wrapping subquery can
    // reference each by name and reuse the single evaluation rather than
    // recomputing the expensive julianday()/bm25() call in ORDER BY. Positional
    // column reads in `candidate_from_row` are unaffected by the aliases.
    format!(
        "m.id,
            m.active_version_id,
            m.space_name,
            m.silo_name,
            m.scope,
            m.project_key,
            m.kind,
            m.status,
            m.entity_key,
            m.claim_key,
            m.observed_at,
            {recency_jd} AS rec_jd,
            m.pinned,
            COALESCE((SELECT group_concat(tag, char(31)) FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)), '') AS tags_joined,
            m.confidence,
            {content_sql},
            {snippet_sql},
            v.summary,
            {source_sql},
            {rank_sql} AS relevance,
            m.metadata_json,
            json_extract(v.source_ref_json, '$.type') AS source_type"
    )
}

fn search_candidates(
    connection: &Connection,
    request: &PreparedSearchRequest,
    fts_query: &str,
) -> Result<Vec<SearchCandidate>> {
    let (sql, args) = search_sql(request);
    let mut statement = connection.prepare_cached(&sql)?;
    let params = std::iter::once(fts_query.to_string()).chain(args);
    let rows = statement.query_map(params_from_iter(params), candidate_from_row)?;
    collect_rows(rows)
}

#[cfg(feature = "semantic")]
pub(crate) fn semantic_candidates(
    connection: &Connection,
    request: &PreparedSearchRequest,
    embedding: &[f32],
) -> Result<Vec<SearchCandidate>> {
    if let (Some(query_tokens), Some(token_model)) = (
        request.query_token_embedding.as_ref(),
        request.token_model_id.as_deref(),
    ) {
        return semantic_candidates_late_interaction(
            connection,
            request,
            query_tokens,
            token_model,
            embedding,
        );
    }
    let table = semantic_table_for_dims(embedding.len())?;
    let (sql, args) = semantic_search_sql(request, &table);
    let embedding_json = embedding_json(embedding)?;
    // Cached to match the FTS sibling `search_candidates`: both are per-query
    // retrieval hot paths. rusqlite's cache is a bounded LRU, so the
    // limit/offset interpolation in the SQL text cannot grow it without bound.
    let mut statement = connection.prepare_cached(&sql)?;
    let params = std::iter::once(embedding_json).chain(args);
    let rows = statement.query_map(params_from_iter(params), candidate_from_row)?;
    collect_rows(rows)
}

/// Late-interaction semantic candidates: exhaustive `MaxSim` selects WHICH
/// memories enter the pipeline; each selected candidate carries its exact
/// single-vector L2 distance (the same statistic the vec0 ANN table reports
/// for unit vectors), so downstream fused scoring keeps its scale.
#[cfg(feature = "semantic")]
pub(crate) fn semantic_candidates_late_interaction(
    connection: &Connection,
    request: &PreparedSearchRequest,
    query_tokens: &[Vec<f32>],
    token_model: &str,
    embedding: &[f32],
) -> Result<Vec<SearchCandidate>> {
    // Selection is capped at limit+offset (NOT candidate_pool_limit): MaxSim
    // decides WHICH memories proceed; the fused single-vector score must only
    // order within that set. With the inflated pool, fused sort-then-truncate
    // lets the single-vector ranking veto MaxSim's selection and the recall
    // gain disappears (observed in the first acceptance run).
    let selection = request.limit.saturating_add(request.offset);
    let mut eligible_ids = memory_ids_matching_filters(connection, &request.filters)?;
    // Bounded shortlist: cap the MaxSim scan (linear in eligible memories, the
    // dominant retrieval cost as the store grows) to the top-N eligible by
    // single-vector distance. 0 keeps the exhaustive scan.
    if request.maxsim_shortlist > 0 && eligible_ids.len() > request.maxsim_shortlist {
        let table = semantic_table_for_dims(embedding.len())?;
        if table_exists(connection, &table)? {
            eligible_ids = maxsim_shortlist_ids(
                connection,
                &table,
                embedding,
                &request.filters,
                &eligible_ids,
                request.maxsim_shortlist.max(selection),
            )?;
        }
    }
    let pool = maxsim_candidates(
        connection,
        query_tokens,
        token_model,
        &eligible_ids,
        selection,
        0,
    )?;
    if pool.is_empty() {
        return Ok(vec![]);
    }
    // Exact L2 distance from the stored single vector (2.0 = max for unit
    // vectors, used when a candidate has no ready single vector).
    let mut vector_statement = connection.prepare_cached(
        "SELECT vector_blob FROM embeddings WHERE memory_id = ?1 AND status = 'ready' \
         AND vector_blob IS NOT NULL ORDER BY updated_at DESC LIMIT 1",
    )?;
    let mut args = SqlArgs::with_reserved(0);
    let mut values = Vec::with_capacity(pool.len());
    for item in &pool {
        let blob: Option<Vec<u8>> = vector_statement
            .query_row([&item.memory_id], |row| row.get(0))
            .optional()?;
        let distance = blob
            .filter(|blob| blob.len() == embedding.len() * 4)
            .map_or(2.0_f64, |blob| {
                let mut sum = 0.0_f64;
                for (chunk, query_value) in blob.chunks_exact(4).zip(embedding) {
                    let doc_value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    sum += f64::from(query_value - doc_value).powi(2);
                }
                sum.sqrt()
            });
        let placeholder = args.push(&item.memory_id);
        values.push(format!("({placeholder}, {distance:.17})"));
    }
    let values_sql = values.join(",");
    let where_clause = filters_where_clause(&request.filters, &mut args);
    let recency_jd = recency_jd_sql();
    let content_sql = if request.include_content {
        "v.content"
    } else {
        "NULL"
    };
    let snippet_sql = if request.snippet_chars == 0 {
        "''".to_string()
    } else {
        format!("substr(v.content, 1, {})", request.snippet_chars)
    };
    let source_sql = if request.include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let columns = candidate_select_columns(
        &recency_jd,
        content_sql,
        &snippet_sql,
        source_sql,
        "li.distance",
    );
    let sql = format!(
        "WITH li(memory_id, distance) AS (VALUES {values_sql})
         SELECT
            {columns}
         FROM li
         JOIN memories m ON m.id = li.memory_id
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {where_clause}
         ORDER BY li.distance ASC, m.observed_at DESC, m.id ASC
         LIMIT {} OFFSET {}",
        request.candidate_pool_limit, request.offset
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(args.values), candidate_from_row)?;
    collect_rows(rows)
}

fn search_fts_table(request: &PreparedSearchRequest) -> &'static str {
    if request.include_source {
        "memory_fts"
    } else {
        "memory_fts_public"
    }
}

fn search_sql(request: &PreparedSearchRequest) -> (String, Vec<String>) {
    let fts_table = search_fts_table(request);
    let mut args = SqlArgs::with_reserved(1);
    let where_clause = search_where_clause(request, &mut args);
    let score_sql = search_score_sql(request, &mut args);
    let recency_jd = recency_jd_sql();
    // Use the inflated pool so Rust re-scoring has room to surface SQL-undervalued
    // candidates (e.g. recent volatile memories). Final results are truncated to
    // request.limit after Rust scoring + sort.
    let candidate_limit = request.candidate_pool_limit.saturating_add(request.offset);
    let candidate_offset = request.offset;
    let content_sql = if request.include_content {
        "v.content"
    } else {
        "NULL"
    };
    let snippet_sql = if request.snippet_chars == 0 {
        "''".to_string()
    } else {
        format!("snippet({fts_table}, 6, '', '', ' … ', 32)")
    };
    let source_sql = if request.include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let bm25_sql = format!("bm25({fts_table})");
    let columns = candidate_select_columns(
        &recency_jd,
        content_sql,
        &snippet_sql,
        source_sql,
        &bm25_sql,
    );
    let sql = format!(
        "SELECT
            {columns}
         FROM {fts_table}
         JOIN memories m ON m.id = {fts_table}.memory_id
         JOIN memory_versions v ON v.id = {fts_table}.version_id
         WHERE {where_clause}
         ORDER BY {bm25_sql} ASC, {score_sql} DESC, m.observed_at DESC, m.id ASC
         LIMIT {candidate_limit} OFFSET {candidate_offset}"
    );
    (sql, args.values)
}

#[cfg(feature = "semantic")]
fn semantic_search_sql(request: &PreparedSearchRequest, table: &str) -> (String, Vec<String>) {
    let mut args = SqlArgs::with_reserved(1);
    let where_clause = filters_where_clause(&request.filters, &mut args);
    // Same filters, second placeholder set: binds the KNN prefilter subquery.
    let prefilter_clause = filters_where_clause(&request.filters, &mut args);
    let recency_jd = recency_jd_sql();
    // Use the inflated pool (same logic as FTS search_sql) so Rust re-scoring
    // can surface SQL-undervalued candidates after silo-aware recency boosting.
    let candidate_limit = request.candidate_pool_limit.saturating_add(request.offset);
    let content_sql = if request.include_content {
        "v.content"
    } else {
        "NULL"
    };
    // The consumer only ever takes a prefix of `snippet_text` (no term
    // centering on the semantic path), so fetch just that prefix instead of
    // full content for every pool row.
    let snippet_sql = if request.snippet_chars == 0 {
        "''".to_string()
    } else {
        format!("substr(v.content, 1, {})", request.snippet_chars)
    };
    let source_sql = if request.include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let distance_sql = format!("{table}.distance");
    let columns = candidate_select_columns(
        &recency_jd,
        content_sql,
        &snippet_sql,
        source_sql,
        &distance_sql,
    );
    let sql = format!(
        "SELECT
            {columns}
         FROM {table}
         JOIN memories m ON m.id = {table}.memory_id
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {table}.embedding MATCH ?1 AND k = {candidate_limit}
         AND {table}.rowid IN (
            SELECT prefilter.rowid FROM {table} prefilter
            JOIN memories m ON m.id = prefilter.memory_id
            WHERE {prefilter_clause}
         )
         AND {where_clause}
         ORDER BY {table}.distance ASC, m.observed_at DESC, m.id ASC
         LIMIT {} OFFSET {}",
        request.candidate_pool_limit, request.offset
    );
    (sql, args.values)
}

fn search_score_sql(request: &PreparedSearchRequest, args: &mut SqlArgs) -> String {
    // NOTE: this expression is used ONLY as the bm25 tiebreaker in the candidate
    // pool ORDER BY (`bm25 ASC, score_sql DESC, observed_at DESC, id ASC`); it is
    // never selected or stored. The relevance/`fts` term was dropped here on
    // purpose: `fts` is a pure function of `bm25`, so it can only differ between
    // two rows when their `bm25` differs — but in that case the primary
    // `bm25 ASC` key has already decided their order. Whenever the tiebreaker
    // actually matters (equal bm25), `fts` is identical across the tied rows and
    // contributes an equal constant, so removing it leaves the ordering
    // byte-identical for all inputs while avoiding two extra bm25() evaluations
    // per matched row (bm25 is FTS5's most expensive scalar).
    let confidence = "(m.confidence * 0.05)";
    let recency = recency_score_sql(&recency_jd_sql());
    let kind = boost_in_sql("m.kind", &request.filters.kinds, 0.05, args);
    let entity = boost_in_sql("m.entity_key", &request.filters.entity_keys, 0.10, args);
    let claim = boost_in_sql("m.claim_key", &request.filters.claim_keys, 0.10, args);
    let scope = boost_in_sql("m.scope", &request.filters.scopes, 0.03, args);
    let tag = if request.filters.tags.is_empty() {
        "0.0".to_string()
    } else {
        format!(
            "(CASE WHEN EXISTS (SELECT 1 FROM memory_tags mt WHERE mt.memory_id = m.id AND mt.tag IN ({})) THEN 0.05 ELSE 0.0 END)",
            args.placeholder_list(&normalized_tags(&request.filters.tags).expect("validated tags"))
        )
    };
    let status = "(CASE WHEN m.status = 'active' THEN 0.02 ELSE 0.0 END)";
    let pin = "(CASE WHEN m.pinned = 1 THEN 0.05 ELSE 0.0 END)";
    format!(
        "({confidence} + {kind} + {entity} + {claim} + {scope} + {tag} + {recency} + {status} + {pin})"
    )
}

fn recency_jd_sql() -> String {
    // Bit-identical to the previous 4-branch CASE (most-recent of updated/observed,
    // null-tolerant) but evaluates julianday() ~2x per row instead of ~4x: the
    // multi-arg max() returns the larger non-null value, returning NULL only when
    // BOTH are null, and coalesce supplies the single-null fallbacks. The produced
    // value (hence the Rust recency score, overall sort order, and the SELECTed
    // recency_jd column) is unchanged for every input.
    "coalesce(max(julianday(m.updated_at), julianday(m.observed_at)), \
       julianday(m.updated_at), julianday(m.observed_at))"
        .to_string()
}

fn recency_score_sql(recency_jd_sql: &str) -> String {
    // Linear approximation of the Rust half-life curve (zero at two
    // half-lives), silo-aware. Used only for SQL-side candidate-pool
    // ordering; the Rust re-scoring pass is authoritative.
    let durable_window = DURABLE_RECENCY_HALF_LIFE_DAYS * 2.0;
    let volatile_window = VOLATILE_RECENCY_HALF_LIFE_DAYS * 2.0;
    // The explicit `WHEN {recency_jd} IS NULL THEN 0.0` guard re-evaluated the
    // recency_jd expression an extra time per row. Fold it into an outer
    // coalesce instead: when recency_jd is NULL the inner arithmetic yields NULL
    // (julianday(NULL) -> NULL propagates through max/subtraction), which
    // coalesce maps to 0.0 -- bit-identical to the guard, but recency_jd is now
    // evaluated once (in the taken silo branch) rather than twice.
    format!(
        "coalesce(CASE \
          WHEN m.silo_name = '{DEFAULT_DURABLE_SILO}' \
          THEN {MAX_RECENCY_SCORE} * max(0.0, 1.0 - (max(0.0, julianday('now') - {recency_jd_sql}) / {durable_window})) \
          ELSE {VOLATILE_MAX_RECENCY_SCORE} * max(0.0, 1.0 - (max(0.0, julianday('now') - {recency_jd_sql}) / {volatile_window})) END, 0.0)"
    )
}

fn boost_in_sql(column: &str, values: &[String], boost: f64, args: &mut SqlArgs) -> String {
    if values.is_empty() {
        "0.0".to_string()
    } else {
        format!(
            "(CASE WHEN {column} IN ({}) THEN {boost} ELSE 0.0 END)",
            args.placeholder_list(values)
        )
    }
}

/// Accumulates bound SQL parameter values, handing out `?N` placeholders.
/// `reserved` counts placeholders the caller binds ahead of these values
/// (e.g. `?1` for the FTS query string or the query embedding).
pub(crate) struct SqlArgs {
    reserved: usize,
    pub(crate) values: Vec<String>,
}

impl SqlArgs {
    pub(crate) fn with_reserved(reserved: usize) -> Self {
        Self {
            reserved,
            values: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, value: &str) -> String {
        self.values.push(value.to_string());
        format!("?{}", self.reserved + self.values.len())
    }

    pub(crate) fn placeholder_list(&mut self, values: &[String]) -> String {
        values
            .iter()
            .map(|value| self.push(value))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn search_where_clause(request: &PreparedSearchRequest, args: &mut SqlArgs) -> String {
    let mut predicates = vec![format!("{} MATCH ?1", search_fts_table(request))];
    predicates.extend(filter_predicates(&request.filters, args));
    predicates.join(" AND ")
}

/// Filter predicates shared by the FTS search, semantic search, and
/// memory-list queries. All values are bound via `args`.
fn filter_predicates(filters: &SearchFilters, args: &mut SqlArgs) -> Vec<String> {
    let mut predicates = Vec::new();
    push_in_predicate(&mut predicates, "m.space_name", &filters.spaces, args);
    push_in_predicate(&mut predicates, "m.silo_name", &filters.silos, args);
    push_in_predicate(&mut predicates, "m.scope", &filters.scopes, args);
    push_in_predicate(&mut predicates, "m.project_key", &filters.projects, args);
    push_in_predicate(&mut predicates, "m.kind", &filters.kinds, args);
    push_in_predicate(&mut predicates, "m.status", &filters.statuses, args);
    push_in_predicate(&mut predicates, "m.entity_key", &filters.entity_keys, args);
    push_in_predicate(&mut predicates, "m.claim_key", &filters.claim_keys, args);
    if !filters.tags.is_empty() {
        predicates.push(format!(
            "EXISTS (SELECT 1 FROM memory_tags mt WHERE mt.memory_id = m.id AND mt.tag IN ({}))",
            args.placeholder_list(&normalized_tags(&filters.tags).expect("validated tags"))
        ));
    }
    if filters.hide_expired {
        // Exclude logically stale facts from recall: a memory whose `valid_to`
        // has passed, or whose `expires_at` has been reached, is no longer
        // current even if the dream expire task has not yet removed it.
        // Mirrors the `active_past_valid_to` stats diagnostic's clock
        // (`julianday('now')`, consistent within a single SQLite statement).
        predicates
            .push("(m.valid_to IS NULL OR julianday(m.valid_to) >= julianday('now'))".to_string());
        predicates.push(
            "(m.expires_at IS NULL OR julianday(m.expires_at) > julianday('now'))".to_string(),
        );
    }
    predicates
}

pub(crate) fn filters_where_clause(filters: &SearchFilters, args: &mut SqlArgs) -> String {
    let predicates = filter_predicates(filters, args);
    if predicates.is_empty() {
        "1=1".to_string()
    } else {
        predicates.join(" AND ")
    }
}

pub(crate) fn memory_ids_matching_filters(
    connection: &Connection,
    filters: &SearchFilters,
) -> Result<BTreeSet<String>> {
    let mut args = SqlArgs::with_reserved(0);
    let where_clause = filters_where_clause(filters, &mut args);
    let sql = format!("SELECT m.id FROM memories m WHERE {where_clause}");
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(args.values.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    Ok(collect_rows(rows)?.into_iter().collect())
}

fn push_in_predicate(
    predicates: &mut Vec<String>,
    column: &str,
    values: &[String],
    args: &mut SqlArgs,
) {
    if !values.is_empty() {
        predicates.push(format!("{column} IN ({})", args.placeholder_list(values)));
    }
}

#[cfg(feature = "semantic")]
pub(crate) fn embedding_json(embedding: &[f32]) -> Result<String> {
    serde_json::to_string(embedding).map_err(|error| Error::InvalidRequest {
        message: format!("failed to encode embedding: {error}"),
    })
}

pub(crate) fn split_tags(tags: &str) -> Vec<String> {
    if tags.is_empty() {
        Vec::new()
    } else {
        tags.split('\u{1f}').map(str::to_string).collect()
    }
}

fn score_candidate(
    candidate: &mut SearchCandidate,
    request: &PreparedSearchRequest,
    best_bm25: f64,
    now_jd: f64,
) -> f64 {
    let fts = fts_score(candidate.bm25, best_bm25);
    let metadata = metadata_score(candidate, request);
    let recency = recency_score_for_silo(candidate.recency_jd, &candidate.silo, now_jd);
    let scope_score = if request.filters.scopes.contains(&candidate.scope) {
        0.03
    } else {
        0.0
    };
    let status_score = if candidate.status == status::ACTIVE {
        0.02
    } else {
        0.0
    };
    let pin = if candidate.pinned { 0.05 } else { 0.0 };
    let source_tier = source_tier_score(candidate.source_type.as_deref(), &candidate.tags);
    candidate.scores = ScoreBreakdown {
        fts,
        metadata,
        recency,
        scope: scope_score,
        status: status_score,
        pin,
        source_tier,
    };
    fts + metadata + recency + scope_score + status_score + pin + source_tier
}

#[cfg(feature = "semantic")]
pub(crate) fn score_semantic_candidate(
    candidate: &mut SearchCandidate,
    request: &PreparedSearchRequest,
    now_jd: f64,
) -> f64 {
    let semantic = (1.0 / (1.0 + candidate.bm25.max(0.0))).min(10.0);
    let metadata = metadata_score(candidate, request);
    let recency = recency_score_for_silo(candidate.recency_jd, &candidate.silo, now_jd);
    let scope_score = if request.filters.scopes.contains(&candidate.scope) {
        0.03
    } else {
        0.0
    };
    let status_score = if candidate.status == status::ACTIVE {
        0.02
    } else {
        0.0
    };
    let pin = if candidate.pinned { 0.05 } else { 0.0 };
    let source_tier = source_tier_score(candidate.source_type.as_deref(), &candidate.tags);
    candidate.scores = ScoreBreakdown {
        fts: semantic,
        metadata,
        recency,
        scope: scope_score,
        status: status_score,
        pin,
        source_tier,
    };
    semantic + metadata + recency + scope_score + status_score + pin + source_tier
}

pub(crate) fn fts_score(bm25: f64, best_bm25: f64) -> f64 {
    // SQLite FTS5 `bm25()` returns <= 0 for matches; more negative = better.
    // Normalize relative to the best (most negative) match in this result set so
    // the score is corpus-independent, monotonic, and bounded in (0, 1] with the
    // best match at 1.0.
    //
    // The previous version computed `(-bm25 * 1_000_000.0).min(10.0)`, which
    // saturated every real-corpus match to 10.0 -- destroying both relevance
    // ranking and the `min_score` floor. It only appeared to work in tiny test
    // stores where raw bm25 magnitudes were near zero (so the product stayed
    // below the 10.0 clamp).
    if best_bm25 < 0.0 && bm25 < 0.0 {
        (bm25 / best_bm25).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn metadata_score(candidate: &SearchCandidate, request: &PreparedSearchRequest) -> f64 {
    let mut score = (candidate.confidence.clamp(0.0, 1.0)) * 0.05;
    if request.filters.kinds.contains(&candidate.kind) {
        score += 0.05;
    }
    if candidate
        .entity_key
        .as_ref()
        .is_some_and(|key| request.filters.entity_keys.contains(key))
    {
        score += 0.10;
    }
    if candidate
        .claim_key
        .as_ref()
        .is_some_and(|key| request.filters.claim_keys.contains(key))
    {
        score += 0.10;
    }
    if candidate
        .tags
        .iter()
        .any(|tag| request.filters.tags.contains(tag))
    {
        score += 0.05;
    }
    // Alias-exact-match boost: a query token matching a memory's reserved
    // `alias::<normalized>` tag is a strong, precise relevance signal that BM25
    // dilutes across a long query. Boosting it lets a topically-weak-but-correct
    // alias hit (e.g. "TypeScript" -> the TS preference) clear the abstention
    // floor while semantic neighbors stay below it. Fires at most once.
    if !request.query_alias_shingles.is_empty()
        && candidate.tags.iter().any(|tag| {
            tag.strip_prefix(ALIAS_TAG_PREFIX)
                .is_some_and(|alias| request.query_alias_shingles.contains(alias))
        })
    {
        score += ALIAS_MATCH_BOOST;
    }
    score
}

/// Source-trust tier boost. Explicit in-session (`mcp`) and manually authored
/// (`manual`) memories rank above auto-harvested synthesis memories, with
/// legacy/unknown provenance sitting in between. Additive nudge in the same
/// magnitude band as the pin/recency boosts: a tiebreaker, not a dominator.
///
/// The nightly synthesis harvester always co-tags its writes `synthesis-derived`,
/// so that tag is treated as the authoritative auto-harvest signal even if the
/// source envelope drifts; otherwise the tier is read from the source `type`
/// (extracted into `source_type`, which is always loaded for ranking).
pub(crate) fn source_tier_score(source_type: Option<&str>, tags: &[String]) -> f64 {
    if tags.iter().any(|tag| tag == "synthesis-derived") {
        return 0.0;
    }
    match source_type {
        Some("synthesis") => 0.0,
        Some("manual" | "mcp") => 0.04,
        // Legacy (no provenance) or unrecognized source: a small edge over
        // auto-harvest, below explicit/manual writes.
        _ => 0.02,
    }
}

/// Maximum deterministic recency boost for volatile (non-durable) silos.
/// Steeper curve ensures a recent volatile claim decisively outranks an old one.
pub(crate) const VOLATILE_MAX_RECENCY_SCORE: f64 = 0.20;

/// Current time as a Julian day, read from `SQLite` so the Rust re-scoring
/// pass and the SQL-side `recency_jd_sql` ordering share one clock. Called
/// once per query (never per candidate), so the round trip is not hot.
pub(crate) fn now_julian_day(connection: &Connection) -> Result<f64> {
    Ok(connection.query_row("SELECT julianday('now')", [], |row| row.get(0))?)
}

/// Age-based exponential recency boost: `max * 0.5^(age_days / half_life)`.
/// Future timestamps (clock skew) clamp to zero age, i.e. the full boost.
pub(crate) fn recency_score_for_silo(recency_jd: Option<f64>, silo: &str, now_jd: f64) -> f64 {
    let Some(recency_jd) = recency_jd.filter(|value| value.is_finite()) else {
        return 0.0;
    };
    let age_days = (now_jd - recency_jd).max(0.0);
    let (max_score, half_life_days) = if silo == DEFAULT_DURABLE_SILO {
        (MAX_RECENCY_SCORE, DURABLE_RECENCY_HALF_LIFE_DAYS)
    } else {
        (VOLATILE_MAX_RECENCY_SCORE, VOLATILE_RECENCY_HALF_LIFE_DAYS)
    };
    max_score * 0.5_f64.powf(age_days / half_life_days)
}

pub(crate) fn compare_candidates(
    left: &SearchCandidate,
    right: &SearchCandidate,
) -> std::cmp::Ordering {
    left.lexical_tier
        .cmp(&right.lexical_tier)
        .then_with(|| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| right.observed_at.cmp(&left.observed_at))
        .then_with(|| left.memory_id.cmp(&right.memory_id))
}

pub(crate) fn freshness_marker(
    silo: &str,
    metadata_json: Option<&str>,
    last_synth: Option<&str>,
) -> String {
    if silo == DEFAULT_DURABLE_SILO {
        return String::new();
    }
    let md = metadata_json.and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok());
    let ptr = md
        .as_ref()
        .and_then(|v| v.get("verified_against"))
        .and_then(|v| v.as_str());
    let Some(ptr) = ptr else {
        return String::new();
    };
    let verified_at = md
        .as_ref()
        .and_then(|v| v.get("verified_at"))
        .and_then(|v| v.as_str());
    let fresh = matches!((verified_at, last_synth), (Some(v), Some(ls)) if v >= ls);
    if fresh {
        format!(" [confirmed {} vs {}]", verified_at.unwrap_or("?"), ptr)
    } else {
        format!(
            " [VERIFY vs {}; last {}]",
            ptr,
            verified_at.unwrap_or("never")
        )
    }
}
