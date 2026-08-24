//! Memory pack assembly extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

use rusqlite::{params, params_from_iter, types::Value, Connection, OptionalExtension, Row};

use memkeeper_core::{status, DEFAULT_DURABLE_SILO, DEFAULT_SPACE};

use crate::{
    bounded_char_slice, collapse_whitespace, collect_rows, filters_where_clause, freshness_marker,
    is_supported_entity_status, is_supported_relationship_status, limit_i64, load_memory,
    load_token_embeddings_cached, maxsim_candidates, memory_ids_matching_filters,
    normalize_search_filters, open_initialized_read_fast, prepare_recall_filters,
    prepare_search_request, query_alias_words, resolve_space_filter, search_memories_on_connection,
    search_prepared, split_tags, validate_pack_request, validate_search_filters,
    with_read_snapshot, Error, EvidenceJoinOptions, PackReport, PackRequest, Result,
    ScoreBreakdown, SearchFilters, SearchReport, SearchRequest, SearchResult, SqlArgs,
    MAX_BATCH_QUERY_LIMIT, MAX_GRAPH_NEIGHBOR_EDGES, MAX_PACK_CHARS, MAX_PACK_MEMORIES,
    MAX_SEARCH_TERMS,
};

#[cfg(feature = "semantic")]
use crate::{
    compare_candidates, embedding_json, now_julian_day, pack_maxsim_shortlist,
    score_semantic_candidate, semantic_candidates, semantic_table_for_dims, table_exists,
};

pub fn build_pack(path: impl AsRef<Path>, request: &PackRequest) -> Result<PackReport> {
    validate_pack_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_pack_on_connection(connection, request)
    })
}

/// Retrieval route that observed a candidate before cross-encoder reranking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AdmissionSource {
    /// Semantic approximate-nearest-neighbor retrieval.
    Ann,
    /// Exhaustive late-interaction `MaxSim` retrieval replacing the ANN leg.
    Maxsim,
    /// Lexical BM25 retrieval.
    Bm25,
    /// Relationship-graph expansion from a seed candidate.
    Graph,
}

/// Source that supplied the graph traversal seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GraphSeedSource {
    /// A direct ANN, `MaxSim`, or BM25 memory candidate.
    Memory,
    /// A deterministic exact query-entity match.
    Entity,
}

/// Evidence class for one graph-derived memory admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GraphEvidenceClass {
    /// Exact endpoint support stored on a routing relationship.
    EndpointSupport,
    /// Another active memory attached to the reached entity.
    EntityFallback,
}

/// Diagnostic-only evidence-join path observation.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphRouteObservation {
    /// Seed arm.
    pub seed_source: GraphSeedSource,
    /// Direct memory seed, when the memory arm supplied the foothold.
    pub seed_memory_id: Option<String>,
    /// Canonical entity id where traversal began.
    pub seed_entity_id: String,
    /// Query index that supplied an exact entity span.
    pub matched_query_index: Option<usize>,
    /// Normalized exact entity span.
    pub matched_query_span: Option<String>,
    /// One-based traversal depth.
    pub hop_depth: usize,
    /// Relationship ids in path order.
    pub relationship_ids: Vec<String>,
    /// Stored predicate names in path order.
    pub predicate_names: Vec<String>,
    /// Traversal directions in path order.
    pub traversal_directions: Vec<String>,
    /// Exact endpoint support or entity fallback.
    pub evidence_class: GraphEvidenceClass,
    /// Route outcome for this admission.
    pub route_outcome: String,
}

/// One route-specific observation of a candidate before pool truncation.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmissionObservation {
    /// Retrieval route.
    pub source: AdmissionSource,
    /// Zero-based query or deterministic variant index.
    pub query_index: usize,
    /// One-based rank within the route-local result list.
    pub source_rank: usize,
    /// Expansion seed for thread or graph observations.
    pub seed_memory_id: Option<String>,
    /// Graph activation when the route is [`AdmissionSource::Graph`].
    pub activation: Option<f64>,
    /// Evidence-join path details. `None` for direct observations.
    pub graph_route: Option<GraphRouteObservation>,
}

/// One scored candidate from the pack retrieval pool before reranking.
#[derive(Debug, Clone, PartialEq)]
pub struct PackPoolItem {
    /// Candidate memory id.
    pub memory_id: String,
    /// Retrieval score: cosine similarity on the ANN/embed path, BM25 otherwise.
    pub score: f64,
    /// Every retrieval route that observed this memory before reranking.
    pub admissions: Vec<AdmissionObservation>,
}

impl PackPoolItem {
    pub(crate) fn direct_candidate(
        memory_id: String,
        score: f64,
        source: AdmissionSource,
        query_index: usize,
        source_rank: usize,
    ) -> Self {
        Self {
            memory_id,
            score,
            admissions: vec![AdmissionObservation {
                source,
                query_index,
                source_rank,
                seed_memory_id: None,
                activation: None,
                graph_route: None,
            }],
        }
    }

    fn merge_admissions_from(&mut self, other: &Self) {
        for observation in &other.admissions {
            if !self.admissions.contains(observation) {
                self.admissions.push(observation.clone());
            }
        }
    }
}

/// One pack candidate after cross-encoder reranking: its memory id, full
/// content, and the reranker's relevance score. Consumed by
/// [`assemble_reranked_pack`].
#[derive(Debug, Clone, PartialEq)]
pub struct RerankCandidate {
    /// Candidate memory id.
    pub memory_id: String,
    /// Canonical memory content used for reranking and rendered after source
    /// time in the final pack.
    pub content: String,
    /// Source observation time rendered with the injected content.
    pub observed_at: String,
    /// Cross-encoder relevance score for `(query, content)`.
    pub rerank_score: f32,
    /// True when both direct retrieval and the graph reached this memory.
    pub consensus: bool,
    /// Graph-expansion activation when this candidate was pulled through a
    /// relationship hop (`None` for ordinary ANN/BM25 candidates). It only
    /// breaks remaining ordering ties.
    pub activation: Option<f64>,
}

fn compare_rerank_candidates(a: &RerankCandidate, b: &RerankCandidate) -> std::cmp::Ordering {
    b.rerank_score
        .total_cmp(&a.rerank_score)
        .then_with(|| b.consensus.cmp(&a.consensus))
        .then_with(|| {
            b.activation
                .partial_cmp(&a.activation)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| a.memory_id.cmp(&b.memory_id))
}

/// Build an empty pack (no injection) that preserves the request's title/format.
#[must_use]
pub fn empty_pack(request: &PackRequest) -> PackReport {
    PackReport {
        title: request.title.clone(),
        format: request.format.clone(),
        content: String::new(),
        memory_ids: Vec::new(),
        scores: Vec::new(),
        truncated: false,
        top_score: None,
    }
}

/// Render a single oversized pack entry truncated to fit the whole char budget.
///
/// Used only by the budget fallback in [`assemble_reranked_pack`] when the
/// top-ranked candidate is itself larger than `max_chars`. Returns the rendered
/// `- <prefix>…\n` entry whose byte length is `<= max_chars`, cut on a UTF-8 char
/// boundary, or `None` when the budget cannot hold the bullet, marker, and at
/// least one character of content.
fn truncate_pack_entry(content: &str, max_chars: usize) -> Option<String> {
    const PREFIX: &str = "- ";
    const SUFFIX: &str = "…\n"; // ellipsis marker + newline
    let overhead = PREFIX.len() + SUFFIX.len();
    if max_chars <= overhead {
        return None;
    }
    let budget = max_chars - overhead;
    let mut end = budget.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut entry = String::with_capacity(max_chars);
    entry.push_str(PREFIX);
    entry.push_str(&content[..end]);
    entry.push_str(SUFFIX);
    Some(entry)
}

fn rerank_pack_line(candidate: &RerankCandidate) -> String {
    format!(
        "- [Observed at: {}] {}\n",
        candidate.observed_at, candidate.content
    )
}

/// Assemble the final pack from a reranked candidate pool, applying one
/// pack-level top-score gate, rerank ordering, and the char/count budget.
///
/// Pure retrieval policy: no store or model access, so it is fully unit-testable.
#[must_use]
pub fn assemble_reranked_pack(request: &PackRequest, candidates: &[RerankCandidate]) -> PackReport {
    let total = candidates.len();
    let rr_top = candidates
        .iter()
        .map(|candidate| candidate.rerank_score)
        .fold(f32::MIN, f32::max);

    let mut ordered: Vec<&RerankCandidate> = candidates.iter().collect();
    ordered.sort_by(|a, b| compare_rerank_candidates(a, b));

    if candidates.is_empty() || f64::from(rr_top) < request.min_score {
        return empty_pack(request);
    }

    let mut content = String::new();
    let mut memory_ids = Vec::new();
    let mut scores = Vec::new();
    for candidate in &ordered {
        if memory_ids.len() >= request.max_memories {
            break;
        }
        let entry = rerank_pack_line(candidate);
        if content.len() + entry.len() > request.max_chars {
            break;
        }
        content.push_str(&entry);
        memory_ids.push(candidate.memory_id.clone());
        scores.push(f64::from(candidate.rerank_score));
    }

    // Budget fallback: the gate passed, but if the highest-ranked eligible
    // candidate is by itself larger than the whole char budget, the loop above
    // injects nothing and a confidently-relevant memory is dropped purely for
    // being long. Inject the top eligible candidate truncated to the budget.
    // Fires only for the char-budget case: the early top-score gate already
    // excluded the ineligible pack.
    let mut text_truncated = false;
    if memory_ids.is_empty() {
        if let Some(top) = ordered.first() {
            let line = rerank_pack_line(top);
            if let Some(entry) = truncate_pack_entry(
                line.strip_prefix("- ")
                    .and_then(|value| value.strip_suffix('\n'))
                    .unwrap_or(&line),
                request.max_chars,
            ) {
                content.push_str(&entry);
                memory_ids.push(top.memory_id.clone());
                // Keep `scores` aligned 1:1 with `memory_ids` (PackReport invariant);
                // the budget-fallback path injects the top candidate, so its score too.
                scores.push(f64::from(top.rerank_score));
                text_truncated = true;
            }
        }
    }

    debug_assert_eq!(
        scores.len(),
        memory_ids.len(),
        "PackReport scores must stay aligned 1:1 with memory_ids"
    );
    PackReport {
        title: request.title.clone(),
        format: request.format.clone(),
        content,
        memory_ids: memory_ids.clone(),
        scores,
        truncated: memory_ids.len() < total || text_truncated,
        top_score: if candidates.is_empty() {
            None
        } else {
            Some(f64::from(rr_top))
        },
    }
}

/// Build the raw, deduplicated, scored candidate pool for a pack request
/// *without* applying the `min_score` precision floor.
///
/// The CLI rerank path uses this to gate injection on the pool's top retrieval
/// score (cosine on the embed path) independently of the cross-encoder rerank
/// score, then reranks survivors for ordering. `max_memories` bounds the pool
/// size; callers set it to the desired candidate-pool width.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is
/// invalid, or retrieval fails.
pub fn build_pack_pool(path: impl AsRef<Path>, request: &PackRequest) -> Result<Vec<PackPoolItem>> {
    validate_pack_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_pack_pool_basic_on_connection(connection, request)
    })
}

fn build_pack_pool_basic_on_connection(
    connection: &Connection,
    request: &PackRequest,
) -> Result<Vec<PackPoolItem>> {
    let per_query_limit = request.max_memories.min(MAX_BATCH_QUERY_LIMIT);
    let mut query_reports: Vec<SearchReport> = Vec::with_capacity(request.queries.len());

    for (i, query) in request.queries.iter().enumerate() {
        #[cfg(not(feature = "semantic"))]
        let _ = i;
        #[cfg(feature = "semantic")]
        let embedding_opt = request
            .query_embeddings
            .as_ref()
            .and_then(|embs| embs.get(i));

        #[cfg(feature = "semantic")]
        if let Some(embedding) = embedding_opt {
            query_reports.push(ann_search_for_pack(
                connection,
                query,
                embedding,
                per_query_limit,
                &request.filters,
            )?);
            continue;
        }

        let search_request = SearchRequest {
            query: query.clone(),
            filters: request.filters.clone(),
            limit: per_query_limit,
            offset: 0,
            snippet_chars: 240,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
            maxsim_shortlist: 0,
        };
        query_reports.push(search_memories_on_connection(connection, &search_request)?);
    }

    // Dedupe by memory id, preserving the interleaved retrieval order, with no
    // precision floor: gating is the caller's responsibility on this path.
    let mut pool = Vec::new();
    let max_result_len = query_reports
        .iter()
        .map(|r| r.results.len())
        .max()
        .unwrap_or(0);
    for result_index in 0..max_result_len {
        for (query_index, report) in query_reports.iter().enumerate() {
            if let Some(result) = report.results.get(result_index) {
                let source = if request
                    .query_embeddings
                    .as_ref()
                    .and_then(|embeddings| embeddings.get(query_index))
                    .is_some()
                {
                    AdmissionSource::Ann
                } else {
                    AdmissionSource::Bm25
                };
                let candidate = PackPoolItem::direct_candidate(
                    result.memory_id.clone(),
                    result.score,
                    source,
                    query_index,
                    result_index + 1,
                );
                if let Some(existing) = pool
                    .iter_mut()
                    .find(|existing: &&mut PackPoolItem| existing.memory_id == result.memory_id)
                {
                    existing.merge_admissions_from(&candidate);
                } else if pool.len() < request.max_memories {
                    pool.push(PackPoolItem::direct_candidate(
                        result.memory_id.clone(),
                        result.score,
                        source,
                        query_index,
                        result_index + 1,
                    ));
                }
            }
        }
    }
    Ok(pool)
}

/// Internal cap on relationship edges examined per graph-expansion seed. We want
/// the seed's full one-hop neighborhood; the candidate budget is enforced
/// downstream by `max_graph_neighbors`, not here.
const GRAPH_EXPANSION_MAX_EDGES: usize = MAX_GRAPH_NEIGHBOR_EDGES;

/// Result of the evidence-backed graph join over the semantic candidate pool.
struct GraphExpandedPool {
    pool: Vec<PackPoolItem>,
    activations: BTreeMap<String, f64>,
    allocation_ranks: BTreeMap<String, usize>,
    route_admissions: BTreeMap<String, Vec<AdmissionObservation>>,
    outcome: Option<String>,
}

impl GraphExpandedPool {
    fn unchanged(pool: Vec<PackPoolItem>, outcome: Option<&str>) -> Self {
        Self {
            pool,
            activations: BTreeMap::new(),
            allocation_ranks: BTreeMap::new(),
            route_admissions: BTreeMap::new(),
            outcome: outcome.map(str::to_string),
        }
    }
}

pub(crate) fn evidence_join_filters(request: &PackRequest) -> Result<SearchFilters> {
    let mut filters = request.filters.clone();
    filters.spaces = resolve_space_filter(&filters.spaces);
    if filters.statuses.is_empty() {
        filters.statuses.push(status::ACTIVE.to_string());
    }
    let filters = normalize_search_filters(filters)?;
    validate_search_filters(&filters)?;
    Ok(filters)
}

fn graph_expand_pool(
    connection: &Connection,
    request: &PackRequest,
    pool: &[PackPoolItem],
    expansion: EvidenceJoinOptions,
    trace_graph_allocation: bool,
) -> Result<GraphExpandedPool> {
    evidence_graph_expand_pool(connection, request, pool, expansion, trace_graph_allocation)
}

#[derive(Debug, Clone)]
pub(crate) struct EvidenceGraphSeed {
    pub(crate) source: GraphSeedSource,
    pub(crate) entity_id: String,
    pub(crate) space_name: String,
    pub(crate) score: f64,
    pub(crate) memory_id: Option<String>,
    pub(crate) matched_query_index: Option<usize>,
    pub(crate) matched_query_span: Option<String>,
}

#[derive(Debug, Clone)]
struct EvidenceTraversalState {
    entity_id: String,
    depth: usize,
    activation: f64,
    relationship_ids: Vec<String>,
    predicate_names: Vec<String>,
    traversal_directions: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct EvidenceCandidateRoute {
    pub(crate) activation: f64,
    pub(crate) route: GraphRouteObservation,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct EvidenceRouteSeed {
    pub(crate) source: GraphSeedSource,
    pub(crate) memory_id: Option<String>,
    pub(crate) entity_id: String,
}

pub(crate) type EvidenceCandidateRoutes =
    BTreeMap<String, BTreeMap<EvidenceRouteSeed, EvidenceCandidateRoute>>;

#[derive(Debug)]
struct EvidenceRelationship {
    id: String,
    subject_entity_id: String,
    subject_entity_key: String,
    relation_type: String,
    object_entity_id: String,
    object_entity_key: String,
    memory_id: Option<String>,
    confidence: f64,
    metadata_json: Option<String>,
}

pub(crate) const MAX_EVIDENCE_ENTITY_SPANS: usize = 32;
const MAX_EVIDENCE_ENTITY_SPAN_WORDS: usize = 5;
fn evidence_graph_expand_pool(
    connection: &Connection,
    request: &PackRequest,
    pool: &[PackPoolItem],
    expansion: EvidenceJoinOptions,
    trace_graph_allocation: bool,
) -> Result<GraphExpandedPool> {
    if expansion.max_graph_seeds == 0 || expansion.max_graph_neighbors == 0 {
        return Ok(evidence_graph_route_miss(pool));
    }
    let filters = evidence_join_filters(request)?;
    let memory_seeds = evidence_memory_seeds(connection, pool)?;
    let entity_seeds = evidence_entity_seeds(connection, request, &filters)?;
    let seeds = allocate_evidence_seeds(&memory_seeds, &entity_seeds, expansion.max_graph_seeds);
    if seeds.is_empty() {
        return Ok(evidence_graph_route_miss(pool));
    }

    let eligible_memory_ids = memory_ids_matching_filters(connection, &filters)?;
    let mut candidates = EvidenceCandidateRoutes::new();
    for seed in &seeds {
        traverse_evidence_seed(
            connection,
            &filters,
            &eligible_memory_ids,
            seed,
            expansion,
            &mut candidates,
        )?;
    }
    if candidates.is_empty() {
        return Ok(evidence_graph_route_miss(pool));
    }
    let route_admissions = evidence_route_admissions(&candidates);

    let pool_ids: BTreeSet<&str> = pool.iter().map(|item| item.memory_id.as_str()).collect();
    let new_candidate_routes: EvidenceCandidateRoutes = candidates
        .iter()
        .filter(|(memory_id, _)| !pool_ids.contains(memory_id.as_str()))
        .map(|(memory_id, routes)| (memory_id.clone(), routes.clone()))
        .collect();
    let allocation_order = evidence_candidate_order(&new_candidate_routes);
    let selected_new: BTreeSet<&str> = allocation_order
        .iter()
        .take(expansion.max_graph_neighbors)
        .map(String::as_str)
        .collect();
    let graph_allocation_ranks = allocation_order
        .iter()
        .enumerate()
        .filter(|(_, memory_id)| {
            trace_graph_allocation || selected_new.contains(memory_id.as_str())
        })
        .map(|(index, memory_id)| (memory_id.clone(), index + 1))
        .collect();
    let mut activations = BTreeMap::new();
    for (memory_id, routes) in &candidates {
        let activation = routes
            .values()
            .map(|candidate| candidate.activation)
            .fold(f64::MIN, f64::max);
        activations.insert(memory_id.clone(), activation);
    }

    let mut expanded = pool.to_vec();
    for item in &mut expanded {
        if let Some(routes) = candidates.get(&item.memory_id) {
            append_evidence_admissions(item, routes);
        }
    }
    for memory_id in allocation_order
        .iter()
        .filter(|memory_id| selected_new.contains(memory_id.as_str()))
    {
        let routes = &new_candidate_routes[memory_id];
        let activation = routes
            .values()
            .map(|candidate| candidate.activation)
            .fold(f64::MIN, f64::max);
        let mut item = PackPoolItem {
            memory_id: memory_id.clone(),
            score: activation,
            admissions: Vec::new(),
        };
        append_evidence_admissions(&mut item, routes);
        expanded.push(item);
    }

    Ok(GraphExpandedPool {
        pool: expanded,
        activations,
        allocation_ranks: graph_allocation_ranks,
        route_admissions,
        outcome: Some("active".to_string()),
    })
}

fn evidence_graph_route_miss(pool: &[PackPoolItem]) -> GraphExpandedPool {
    GraphExpandedPool::unchanged(pool.to_vec(), Some("no_eligible_seed_route"))
}

fn evidence_route_admissions(
    candidates: &EvidenceCandidateRoutes,
) -> BTreeMap<String, Vec<AdmissionObservation>> {
    candidates
        .iter()
        .map(|(memory_id, routes)| {
            let admissions = routes
                .values()
                .map(|candidate| AdmissionObservation {
                    source: AdmissionSource::Graph,
                    query_index: candidate.route.matched_query_index.unwrap_or(0),
                    source_rank: 0,
                    seed_memory_id: candidate.route.seed_memory_id.clone(),
                    activation: Some(candidate.activation),
                    graph_route: Some(candidate.route.clone()),
                })
                .collect();
            (memory_id.clone(), admissions)
        })
        .collect()
}

fn append_evidence_admissions(
    item: &mut PackPoolItem,
    routes: &BTreeMap<EvidenceRouteSeed, EvidenceCandidateRoute>,
) {
    for candidate in routes.values() {
        let query_index = candidate.route.matched_query_index.unwrap_or(0);
        item.admissions.push(AdmissionObservation {
            source: AdmissionSource::Graph,
            query_index,
            source_rank: 0,
            seed_memory_id: candidate.route.seed_memory_id.clone(),
            activation: Some(candidate.activation),
            graph_route: Some(candidate.route.clone()),
        });
    }
}

fn evidence_memory_seeds(
    connection: &Connection,
    pool: &[PackPoolItem],
) -> Result<Vec<EvidenceGraphSeed>> {
    let mut seeds = Vec::new();
    let mut seen_entities = BTreeSet::new();
    let mut support_statement = connection.prepare_cached(
        "SELECT DISTINCT e.id, e.entity_key, e.space_name
           FROM relationships r
           JOIN entities e ON e.id IN (r.subject_entity_id, r.object_entity_id)
          WHERE r.status = 'active'
            AND e.status = 'active'
            AND json_extract(r.metadata_json, '$.routing') = 1
            AND r.memory_id = ?1
          ORDER BY e.id",
    )?;
    let mut attached_statement = connection.prepare_cached(
        "SELECT e.id, e.entity_key, e.space_name
           FROM memories m
           JOIN entities e
             ON e.space_name = m.space_name
            AND e.entity_key = m.entity_key
          WHERE m.id = ?1
            AND e.status = 'active'",
    )?;
    for item in pool {
        let supported_entities = support_statement
            .query_map([&item.memory_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let attached_entity = attached_statement
            .query_row([&item.memory_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .optional()?;
        for (entity_id, _entity_key, space_name) in
            supported_entities.into_iter().chain(attached_entity)
        {
            if seen_entities.insert(entity_id.clone()) {
                seeds.push(EvidenceGraphSeed {
                    source: GraphSeedSource::Memory,
                    entity_id,
                    space_name,
                    score: item.score,
                    memory_id: Some(item.memory_id.clone()),
                    matched_query_index: None,
                    matched_query_span: None,
                });
            }
        }
    }
    Ok(seeds)
}

#[derive(Debug)]
pub(crate) struct EvidenceQuerySpan {
    pub(crate) query_index: usize,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) normalized: String,
}

pub(crate) fn evidence_query_spans(queries: &[String]) -> Vec<EvidenceQuerySpan> {
    let mut spans = Vec::new();
    for (query_index, query) in queries.iter().enumerate() {
        let words = query_alias_words(query);
        let max_span = MAX_EVIDENCE_ENTITY_SPAN_WORDS.min(words.len());
        for width in (1..=max_span).rev() {
            for start in 0..=words.len().saturating_sub(width) {
                if spans.len() == MAX_EVIDENCE_ENTITY_SPANS {
                    return spans;
                }
                let normalized = words[start..start + width].join(" ");
                if width == 1 && evidence_entity_stopword(&normalized) {
                    continue;
                }
                spans.push(EvidenceQuerySpan {
                    query_index,
                    start,
                    end: start + width,
                    normalized,
                });
            }
        }
    }
    spans
}

fn evidence_entity_stopword(value: &str) -> bool {
    matches!(
        value,
        "a" | "an"
            | "and"
            | "are"
            | "at"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "how"
            | "in"
            | "is"
            | "of"
            | "on"
            | "the"
            | "to"
            | "was"
            | "were"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "why"
            | "with"
            | "work"
    )
}

pub(crate) fn evidence_entity_seeds(
    connection: &Connection,
    request: &PackRequest,
    filters: &SearchFilters,
) -> Result<Vec<EvidenceGraphSeed>> {
    let mut seeds = Vec::new();
    let mut seen_entities = BTreeSet::new();
    let mut matched_ranges: BTreeMap<usize, Vec<(usize, usize)>> = BTreeMap::new();
    for span in evidence_query_spans(&request.queries) {
        if matched_ranges.get(&span.query_index).is_some_and(|ranges| {
            ranges
                .iter()
                .any(|(start, end)| span.start < *end && span.end > *start)
        }) {
            continue;
        }
        let matches = exact_entities_for_span(connection, &filters.spaces, &span.normalized)?;
        if matches.len() != 1 {
            continue;
        }
        let (entity_id, _entity_key, space_name) = matches.into_iter().next().expect("one match");
        matched_ranges
            .entry(span.query_index)
            .or_default()
            .push((span.start, span.end));
        if !seen_entities.insert(entity_id.clone()) {
            continue;
        }
        seeds.push(EvidenceGraphSeed {
            source: GraphSeedSource::Entity,
            entity_id,
            space_name,
            score: 1.0,
            memory_id: None,
            matched_query_index: Some(span.query_index),
            matched_query_span: Some(span.normalized),
        });
    }
    Ok(seeds)
}

pub(crate) fn exact_entities_for_span(
    connection: &Connection,
    spaces: &[String],
    normalized_span: &str,
) -> Result<Vec<(String, String, String)>> {
    let mut args = SqlArgs::with_reserved(0);
    let span = args.push(normalized_span);
    let space_predicate = if spaces.is_empty() {
        String::new()
    } else {
        format!(" AND e.space_name IN ({})", args.placeholder_list(spaces))
    };
    let sql = format!(
        "SELECT DISTINCT e.id, e.entity_key, e.space_name
           FROM entities e
           LEFT JOIN entity_aliases ea ON ea.entity_id = e.id
          WHERE e.status = 'active'
            AND (
                 e.entity_key = {span}
              OR lower(trim(e.canonical_name)) = {span}
              OR ea.normalized_alias = {span}
            )
            {space_predicate}
          ORDER BY e.space_name, e.entity_key, e.id
          LIMIT 3"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(args.values.iter()), |row| {
        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
    })?;
    collect_rows(rows)
}

pub(crate) fn allocate_evidence_seeds(
    memory_seeds: &[EvidenceGraphSeed],
    entity_seeds: &[EvidenceGraphSeed],
    limit: usize,
) -> Vec<EvidenceGraphSeed> {
    if limit == 0 {
        return Vec::new();
    }
    if memory_seeds.is_empty() {
        return entity_seeds.iter().take(limit).cloned().collect();
    }
    if entity_seeds.is_empty() || limit == 1 {
        return memory_seeds.iter().take(limit).cloned().collect();
    }
    let mut selected = vec![memory_seeds[0].clone(), entity_seeds[0].clone()];
    let mut memory_index = 1;
    let mut entity_index = 1;
    while selected.len() < limit {
        let mut advanced = false;
        if memory_index < memory_seeds.len() {
            selected.push(memory_seeds[memory_index].clone());
            memory_index += 1;
            advanced = true;
            if selected.len() == limit {
                break;
            }
        }
        if entity_index < entity_seeds.len() {
            selected.push(entity_seeds[entity_index].clone());
            entity_index += 1;
            advanced = true;
        }
        if !advanced {
            break;
        }
    }
    selected
}

fn traverse_evidence_seed(
    connection: &Connection,
    filters: &SearchFilters,
    eligible_memory_ids: &BTreeSet<String>,
    seed: &EvidenceGraphSeed,
    expansion: EvidenceJoinOptions,
    candidates: &mut EvidenceCandidateRoutes,
) -> Result<()> {
    let mut frontier = VecDeque::from([EvidenceTraversalState {
        entity_id: seed.entity_id.clone(),
        depth: 0,
        activation: seed.score,
        relationship_ids: Vec::new(),
        predicate_names: Vec::new(),
        traversal_directions: Vec::new(),
    }]);
    let mut reached_depth = BTreeMap::from([(seed.entity_id.clone(), 0_usize)]);
    let mut examined_edges = 0_usize;
    while let Some(state) = frontier.pop_front() {
        if state.depth >= 2 || examined_edges >= GRAPH_EXPANSION_MAX_EDGES {
            continue;
        }
        let relationships = evidence_incident_relationships(
            connection,
            &seed.space_name,
            &state.entity_id,
            GRAPH_EXPANSION_MAX_EDGES - examined_edges,
        )?;
        for relationship in relationships {
            examined_edges += 1;
            if state.relationship_ids.contains(&relationship.id) {
                continue;
            }
            let Some(route) = qualified_evidence_relationship(
                connection,
                &seed.space_name,
                &state.entity_id,
                relationship,
            )?
            else {
                continue;
            };
            let depth = state.depth + 1;
            let activation = state.activation * expansion.graph_decay * route.confidence;
            let mut relationship_ids = state.relationship_ids.clone();
            relationship_ids.push(route.relationship_id);
            let mut predicate_names = state.predicate_names.clone();
            predicate_names.push(route.predicate_name);
            let mut traversal_directions = state.traversal_directions.clone();
            traversal_directions.push(route.direction);

            if eligible_memory_ids.contains(&route.target_memory_id) {
                record_evidence_candidate(
                    candidates,
                    &route.target_memory_id,
                    activation,
                    evidence_route_observation(
                        seed,
                        depth,
                        &relationship_ids,
                        &predicate_names,
                        &traversal_directions,
                        GraphEvidenceClass::EndpointSupport,
                    ),
                );
            }
            for fallback_memory_id in entity_memory_ids_with_filters(
                connection,
                filters,
                &route.neighbor_entity_key,
                expansion.max_graph_neighbors,
            )? {
                if fallback_memory_id == route.target_memory_id {
                    continue;
                }
                record_evidence_candidate(
                    candidates,
                    &fallback_memory_id,
                    activation,
                    evidence_route_observation(
                        seed,
                        depth,
                        &relationship_ids,
                        &predicate_names,
                        &traversal_directions,
                        GraphEvidenceClass::EntityFallback,
                    ),
                );
            }

            if depth < 2
                && reached_depth
                    .get(&route.neighbor_entity_id)
                    .is_none_or(|existing_depth| depth < *existing_depth)
            {
                reached_depth.insert(route.neighbor_entity_id.clone(), depth);
                frontier.push_back(EvidenceTraversalState {
                    entity_id: route.neighbor_entity_id,
                    depth,
                    activation,
                    relationship_ids,
                    predicate_names,
                    traversal_directions,
                });
            }
        }
    }
    Ok(())
}

struct QualifiedEvidenceRelationship {
    relationship_id: String,
    predicate_name: String,
    neighbor_entity_id: String,
    neighbor_entity_key: String,
    target_memory_id: String,
    confidence: f64,
    direction: String,
}

fn qualified_evidence_relationship(
    connection: &Connection,
    space: &str,
    current_entity_id: &str,
    relationship: EvidenceRelationship,
) -> Result<Option<QualifiedEvidenceRelationship>> {
    let Some(metadata_json) = relationship.metadata_json.as_deref() else {
        return Ok(None);
    };
    let metadata: serde_json::Value =
        serde_json::from_str(metadata_json).map_err(|error| Error::InvalidRequest {
            message: format!(
                "routing relationship {} has invalid metadata_json: {error}",
                relationship.id
            ),
        })?;
    if metadata.get("routing").and_then(serde_json::Value::as_bool) != Some(true) {
        return Ok(None);
    }
    if metadata
        .get("routing_contract")
        .and_then(serde_json::Value::as_str)
        != Some("evidence_join_v2")
        || metadata
            .get("routing_contract_version")
            .and_then(serde_json::Value::as_u64)
            != Some(2)
    {
        return Err(Error::InvalidRequest {
            message: format!(
                "routing relationship {} must use evidence_join_v2 contract version 2",
                relationship.id
            ),
        });
    }
    if metadata.get("object_memory_id").is_some() {
        return Err(Error::InvalidRequest {
            message: format!(
                "routing relationship {} uses removed object_memory_id; memory_id is the sole evidence id",
                relationship.id
            ),
        });
    }
    if relationship.relation_type == "related_to" {
        return Ok(None);
    }
    let evidence_memory_id =
        relationship
            .memory_id
            .as_deref()
            .ok_or_else(|| Error::InvalidRequest {
                message: format!(
                    "routing relationship {} is missing evidence memory_id",
                    relationship.id
                ),
            })?;
    if !routing_support_memory_is_valid(connection, evidence_memory_id, space)? {
        return Err(Error::InvalidRequest {
            message: format!(
                "routing relationship {} has invalid evidence memory {}",
                relationship.id, evidence_memory_id
            ),
        });
    }

    let (neighbor_entity_id, neighbor_entity_key, target_memory_id, direction) =
        if relationship.subject_entity_id == current_entity_id {
            (
                relationship.object_entity_id,
                relationship.object_entity_key,
                evidence_memory_id.to_string(),
                "forward".to_string(),
            )
        } else if relationship.object_entity_id == current_entity_id {
            (
                relationship.subject_entity_id,
                relationship.subject_entity_key,
                evidence_memory_id.to_string(),
                "reverse".to_string(),
            )
        } else {
            return Ok(None);
        };
    Ok(Some(QualifiedEvidenceRelationship {
        relationship_id: relationship.id,
        predicate_name: relationship.relation_type,
        neighbor_entity_id,
        neighbor_entity_key,
        target_memory_id,
        confidence: relationship.confidence,
        direction,
    }))
}

fn routing_support_memory_is_valid(
    connection: &Connection,
    memory_id: &str,
    space: &str,
) -> Result<bool> {
    connection
        .query_row(
            "SELECT EXISTS(
                 SELECT 1
                   FROM memories m
                  WHERE m.id = ?1
                    AND m.space_name = ?2
                    AND m.status = 'active'
                    AND m.active_version_id IS NOT NULL
                    AND (m.valid_to IS NULL OR julianday(m.valid_to) >= julianday('now'))
                    AND (m.expires_at IS NULL OR julianday(m.expires_at) > julianday('now'))
             )",
            params![memory_id, space],
            |row| row.get(0),
        )
        .map_err(Into::into)
}

fn evidence_incident_relationships(
    connection: &Connection,
    space: &str,
    entity_id: &str,
    limit: usize,
) -> Result<Vec<EvidenceRelationship>> {
    let sql = format!(
        "SELECT r.id,
                r.subject_entity_id,
                subject.entity_key,
                r.relation_type,
                r.object_entity_id,
                object.entity_key,
                r.memory_id,
                r.confidence,
                r.metadata_json
           FROM relationships r
           JOIN entities subject ON subject.id = r.subject_entity_id
           JOIN entities object ON object.id = r.object_entity_id
          WHERE r.space_name = ?1
            AND (r.subject_entity_id = ?2 OR r.object_entity_id = ?2)
            AND r.status = 'active'
            AND subject.status = 'active'
            AND object.status = 'active'
            AND (r.valid_from IS NULL OR julianday(r.valid_from) <= julianday('now'))
            AND (r.valid_to IS NULL OR julianday(r.valid_to) >= julianday('now'))
          ORDER BY r.relation_type, r.subject_entity_id, r.object_entity_id, r.id
          LIMIT {limit}"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params![space, entity_id], |row| {
        Ok(EvidenceRelationship {
            id: row.get(0)?,
            subject_entity_id: row.get(1)?,
            subject_entity_key: row.get(2)?,
            relation_type: row.get(3)?,
            object_entity_id: row.get(4)?,
            object_entity_key: row.get(5)?,
            memory_id: row.get(6)?,
            confidence: row.get(7)?,
            metadata_json: row.get(8)?,
        })
    })?;
    collect_rows(rows)
}

fn evidence_route_observation(
    seed: &EvidenceGraphSeed,
    hop_depth: usize,
    relationship_ids: &[String],
    predicate_names: &[String],
    traversal_directions: &[String],
    evidence_class: GraphEvidenceClass,
) -> GraphRouteObservation {
    GraphRouteObservation {
        seed_source: seed.source,
        seed_memory_id: seed.memory_id.clone(),
        seed_entity_id: seed.entity_id.clone(),
        matched_query_index: seed.matched_query_index,
        matched_query_span: seed.matched_query_span.clone(),
        hop_depth,
        relationship_ids: relationship_ids.to_vec(),
        predicate_names: predicate_names.to_vec(),
        traversal_directions: traversal_directions.to_vec(),
        evidence_class,
        route_outcome: "active".to_string(),
    }
}

pub(crate) fn record_evidence_candidate(
    candidates: &mut EvidenceCandidateRoutes,
    memory_id: &str,
    activation: f64,
    route: GraphRouteObservation,
) {
    let seed = EvidenceRouteSeed {
        source: route.seed_source,
        memory_id: route.seed_memory_id.clone(),
        entity_id: route.seed_entity_id.clone(),
    };
    let candidate = EvidenceCandidateRoute { activation, route };
    candidates
        .entry(memory_id.to_string())
        .or_default()
        .entry(seed)
        .and_modify(|existing| {
            if evidence_route_is_better(&candidate, existing) {
                *existing = candidate.clone();
            }
        })
        .or_insert(candidate);
}

fn evidence_route_is_better(
    candidate: &EvidenceCandidateRoute,
    existing: &EvidenceCandidateRoute,
) -> bool {
    let activation_order = candidate.activation.total_cmp(&existing.activation);
    candidate.route.evidence_class < existing.route.evidence_class
        || (candidate.route.evidence_class == existing.route.evidence_class
            && (candidate.route.hop_depth < existing.route.hop_depth
                || (candidate.route.hop_depth == existing.route.hop_depth
                    && (activation_order.is_gt()
                        || (activation_order.is_eq()
                            && candidate.route.relationship_ids
                                < existing.route.relationship_ids)))))
}

pub(crate) fn evidence_candidate_order(candidates: &EvidenceCandidateRoutes) -> Vec<String> {
    let mut ranked: Vec<(&String, &EvidenceCandidateRoute)> = candidates
        .iter()
        .filter_map(|(memory_id, routes)| {
            routes
                .values()
                .reduce(|best, candidate| {
                    if evidence_route_is_better(candidate, best) {
                        candidate
                    } else {
                        best
                    }
                })
                .map(|route| (memory_id, route))
        })
        .collect();
    ranked.sort_by(|left, right| {
        left.1
            .route
            .evidence_class
            .cmp(&right.1.route.evidence_class)
            .then_with(|| left.1.route.hop_depth.cmp(&right.1.route.hop_depth))
            .then_with(|| right.1.activation.total_cmp(&left.1.activation))
            .then_with(|| left.0.cmp(right.0))
    });
    ranked
        .into_iter()
        .map(|(memory_id, _)| memory_id.clone())
        .collect()
}

/// Active memory ids attached to `entity_key`, passing the normalized pack filters
/// (space/status/kind/project/silo/scope), in the deterministic order graph
/// selection uses. Mirrors `thread_neighbor_pool_items`' filter preservation,
/// anchored on a neighbor entity instead of the seed's entity/claim anchors.
fn entity_memory_ids_with_filters(
    connection: &Connection,
    filters: &SearchFilters,
    entity_key: &str,
    limit: usize,
) -> Result<Vec<String>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let mut args = SqlArgs::with_reserved(0);
    let where_clause = filters_where_clause(filters, &mut args);
    let entity_placeholder = args.push(entity_key);
    let sql = format!(
        "SELECT m.id
           FROM memories m
          WHERE {where_clause}
            AND m.entity_key = {entity_placeholder}
          ORDER BY m.pinned DESC, m.observed_at DESC, m.updated_at DESC, m.id ASC
          LIMIT {limit}"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(args.values.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    collect_rows(rows)
}

/// One reranking candidate from the hybrid retrieval pool: memory id plus the
/// full active-version content the cross-encoder scores against.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankPoolCandidate {
    /// Candidate memory id.
    pub memory_id: String,
    /// Full active-version content.
    pub content: String,
    /// Source observation time. This is rendered into the final pack but is
    /// not supplied to the cross-encoder, so temporal context does not alter
    /// candidate ranking.
    pub observed_at: String,
    /// True when both a direct route and graph route observed this memory.
    pub consensus: bool,
    /// Graph-expansion activation when this candidate was pulled through a
    /// relationship hop (`None` for ordinary ANN/BM25 candidates). It only
    /// breaks exact cross-encoder score ties.
    pub activation: Option<f64>,
    /// Every retrieval or expansion route that observed this candidate.
    pub admissions: Vec<AdmissionObservation>,
}

/// Hybrid pre-rerank retrieval pool: ANN-primary with a BM25 safety net,
/// deduplicated, with contents fetched from the same read snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankPool {
    /// Top retrieval score of the ANN pool (cosine on the embed path);
    /// `f64::MIN` when the ANN pool is empty.
    pub cos_top: f64,
    /// Deduplicated candidates in interleaved retrieval order.
    pub candidates: Vec<RerankPoolCandidate>,
    /// Evidence-join treatment outcome. `None` when that strategy was not active.
    pub graph_outcome: Option<String>,
    /// Every candidate observed by ANN/BM25 before the final hybrid cutoff,
    /// plus admitted structural-expansion candidates.
    pub observed: Vec<RerankPoolObservedCandidate>,
    /// Requested final hybrid width before structural additions.
    pub pool_width: usize,
}

/// ID-only diagnostic record for one candidate observed during pool assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankPoolObservedCandidate {
    /// Candidate memory id.
    pub memory_id: String,
    /// One-based position in the unbounded ANN/BM25 interleave.
    pub merged_rank: usize,
    /// Whether this candidate reached the exact reranker input pool.
    pub admitted: bool,
    /// Stage that excluded the candidate, or `None` when admitted.
    pub dropped_at: Option<String>,
    /// One-based position in the exact diversified graph-allocation order.
    /// `None` means the memory already belonged to the pre-expansion pool and
    /// therefore did not consume a graph allocation slot.
    pub graph_allocation_rank: Option<usize>,
    /// Every route that observed the candidate.
    pub admissions: Vec<AdmissionObservation>,
}

/// Build the hybrid pre-rerank candidate pool for a pack request: a scored ANN
/// pool (no precision floor), a bounded BM25 pool as a recall safety net when
/// query embeddings are present, deduplicated ANN-first, with candidate
/// contents fetched on the same connection/read snapshot. The returned cosine
/// top score is computed from the ANN pool only, preserving the query-level
/// embedding gate semantics.
///
/// Candidates whose active version has no content are dropped (they cannot be
/// reranked meaningfully).
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is
/// invalid, or retrieval fails.
pub fn build_hybrid_rerank_pool(
    path: impl AsRef<Path>,
    request: &PackRequest,
    pool_width: usize,
) -> Result<RerankPool> {
    validate_pack_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_hybrid_rerank_pool_on_connection(connection, request, pool_width)
    })
}

/// Build the hybrid pre-rerank pool with diagnostic evidence-join allocation
/// controls. Normal pack construction uses [`build_hybrid_rerank_pool`] and its
/// fixed policy.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is
/// invalid, or retrieval fails.
pub fn build_hybrid_rerank_pool_with_evidence_options(
    path: impl AsRef<Path>,
    request: &PackRequest,
    pool_width: usize,
    expansion: EvidenceJoinOptions,
) -> Result<RerankPool> {
    validate_pack_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_hybrid_rerank_pool_with_evidence_options_on_connection(
            connection, request, pool_width, expansion, false,
        )
    })
}

/// Build the hybrid pre-rerank pool with full admission observations for a
/// diagnostic trace. Unlike the production builder, this includes candidates
/// excluded by the graph-neighbor cap in `RerankPool::observed`.
///
/// Admitted candidate membership and ordering are identical to
/// [`build_hybrid_rerank_pool_with_evidence_options`].
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is
/// invalid, or retrieval fails.
pub fn build_hybrid_rerank_pool_trace_with_evidence_options(
    path: impl AsRef<Path>,
    request: &PackRequest,
    pool_width: usize,
    expansion: EvidenceJoinOptions,
) -> Result<RerankPool> {
    validate_pack_request(request)?;
    if expansion.max_graph_seeds > MAX_PACK_MEMORIES
        || expansion.max_graph_neighbors > MAX_PACK_MEMORIES
    {
        return Err(Error::InvalidRequest {
            message: format!(
                "pool-trace max_graph_seeds and max_graph_neighbors must not exceed {MAX_PACK_MEMORIES}"
            ),
        });
    }
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_hybrid_rerank_pool_with_evidence_options_on_connection(
            connection, request, pool_width, expansion, true,
        )
    })
}

fn build_hybrid_rerank_pool_on_connection(
    connection: &Connection,
    request: &PackRequest,
    pool_width: usize,
) -> Result<RerankPool> {
    build_hybrid_rerank_pool_with_evidence_options_on_connection(
        connection,
        request,
        pool_width,
        EvidenceJoinOptions::default(),
        false,
    )
}

fn expand_and_observe_rerank_graph(
    connection: &Connection,
    request: &PackRequest,
    pool: &[PackPoolItem],
    expansion: EvidenceJoinOptions,
    trace_graph_allocation: bool,
    observed: &mut Vec<RerankPoolObservedCandidate>,
) -> Result<GraphExpandedPool> {
    let mut expanded =
        graph_expand_pool(connection, request, pool, expansion, trace_graph_allocation)?;
    apply_graph_admission_observations(
        &mut expanded.pool,
        &expanded.activations,
        &expanded.allocation_ranks,
        &expanded.route_admissions,
        observed,
    );
    Ok(expanded)
}

fn materialize_expanded_rerank_candidates(
    connection: &Connection,
    expanded: GraphExpandedPool,
) -> Result<(Vec<RerankPoolCandidate>, Option<String>)> {
    let GraphExpandedPool {
        pool,
        activations,
        allocation_ranks: _,
        outcome,
        ..
    } = expanded;
    let mut statement = connection.prepare_cached(
        "SELECT v.content, m.observed_at FROM memories m \
         JOIN memory_versions v ON v.id = m.active_version_id \
         WHERE m.id = ?1",
    )?;
    let mut candidates = Vec::with_capacity(pool.len());
    for item in pool {
        let materialized: Option<(String, String)> = statement
            .query_row([&item.memory_id], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()?;
        let Some((content, observed_at)) = materialized else {
            continue;
        };
        if content.is_empty() {
            continue;
        }
        let activation = activations.get(&item.memory_id).copied();
        let consensus = activation.is_some()
            && item
                .admissions
                .iter()
                .any(|admission| admission.source != AdmissionSource::Graph);
        candidates.push(RerankPoolCandidate {
            memory_id: item.memory_id,
            content,
            observed_at,
            consensus,
            activation,
            admissions: item.admissions,
        });
    }
    Ok((candidates, outcome))
}

fn build_hybrid_rerank_pool_with_evidence_options_on_connection(
    connection: &Connection,
    request: &PackRequest,
    pool_width: usize,
    expansion: EvidenceJoinOptions,
    include_graph_cap_observations: bool,
) -> Result<RerankPool> {
    let mut pool_request = request.clone();
    pool_request.max_memories = pool_width;
    pool_request.max_chars = MAX_PACK_CHARS;
    pool_request.rerank_candidates = 0;
    pool_request.min_score = 0.0;
    let dense_active = pool_request
        .query_embeddings
        .as_ref()
        .is_some_and(|embeddings| !embeddings.is_empty());
    let late_interaction = pool_request.query_token_embeddings.is_some();
    let initial_pool = build_pack_pool_basic_on_connection(connection, &pool_request)?;
    let (ann_pool, bm25_pool) = if dense_active {
        let mut lexical_request = pool_request.clone();
        lexical_request.query_embeddings = None;
        (
            initial_pool,
            build_pack_pool_basic_on_connection(connection, &lexical_request)?,
        )
    } else if late_interaction {
        // MaxSim replaces ANN for selection. Without a dense query, the initial
        // lexical pool is already the BM25 safety net.
        (Vec::new(), initial_pool)
    } else {
        (initial_pool, Vec::new())
    };
    // Keep the raw ANN top score in the diagnostic trace. Final pack admission
    // uses one top-score gate on the cross-encoder output.
    let cos_top = ann_pool
        .iter()
        .map(|item| item.score)
        .reduce(f64::max)
        .unwrap_or(f64::NAN);
    // Late-interaction: MaxSim over the filtered recall scope replaces the ANN
    // leg for candidate SELECTION only; the BM25 safety net stays merged in.
    let primary_pool = match (
        pool_request.query_token_embeddings.as_ref(),
        pool_request.token_model_id.as_deref(),
    ) {
        (Some(token_queries), Some(token_model)) if !token_queries.is_empty() => {
            let filters = prepare_recall_filters(&pool_request.filters)?;
            let eligible_ids = memory_ids_matching_filters(connection, &filters)?;
            let mut per_query = Vec::with_capacity(token_queries.len());
            for (query_index, tokens) in token_queries.iter().enumerate() {
                // Bounded shortlist (see semantic_candidates_late_interaction):
                // per query, when that query's dense embedding is available to
                // rank by. Queries without a dense embedding stay exhaustive,
                // as do non-semantic builds (no vec0 table to shortlist from).
                #[cfg(feature = "semantic")]
                let shortlisted = pack_maxsim_shortlist(
                    connection,
                    &pool_request,
                    &filters,
                    &eligible_ids,
                    token_queries.len(),
                    query_index,
                    pool_width,
                )?;
                #[cfg(not(feature = "semantic"))]
                let shortlisted: Option<BTreeSet<String>> = None;
                let query_eligible = shortlisted.as_ref().unwrap_or(&eligible_ids);
                per_query.push(maxsim_candidates(
                    connection,
                    tokens,
                    token_model,
                    query_eligible,
                    pool_width,
                    query_index,
                )?);
            }
            interleave_pools(&per_query, pool_width)
        }
        _ => ann_pool,
    };
    let (merged, mut observed) =
        merge_rerank_pools_with_trace(&primary_pool, &bm25_pool, pool_width);

    // Join graph evidence after direct semantic and lexical observations have
    // been merged on canonical memory identity.
    let expanded = expand_and_observe_rerank_graph(
        connection,
        &pool_request,
        &merged,
        expansion,
        include_graph_cap_observations,
        &mut observed,
    )?;
    let (candidates, graph_outcome) = materialize_expanded_rerank_candidates(connection, expanded)?;
    Ok(RerankPool {
        cos_top,
        candidates,
        graph_outcome,
        observed,
        pool_width,
    })
}

fn replace_graph_admissions(
    item: &mut PackPoolItem,
    graph_ranks: &BTreeMap<&str, usize>,
    activation: f64,
) {
    let mut route_observations = item
        .admissions
        .iter()
        .filter(|source| source.source == AdmissionSource::Graph && source.graph_route.is_some())
        .cloned()
        .collect::<Vec<_>>();
    item.admissions
        .retain(|source| source.source != AdmissionSource::Graph);
    let canonical_rank = graph_ranks
        .get(item.memory_id.as_str())
        .copied()
        .unwrap_or(1);
    if route_observations.is_empty() {
        item.admissions.push(AdmissionObservation {
            source: AdmissionSource::Graph,
            query_index: 0,
            source_rank: canonical_rank,
            seed_memory_id: None,
            activation: Some(activation),
            graph_route: None,
        });
    } else {
        for observation in &mut route_observations {
            observation.source_rank = canonical_rank;
        }
        item.admissions.extend(route_observations);
    }
}

pub(crate) fn apply_graph_admission_observations(
    merged: &mut [PackPoolItem],
    graph_activations: &BTreeMap<String, f64>,
    graph_allocation_ranks: &BTreeMap<String, usize>,
    graph_route_admissions: &BTreeMap<String, Vec<AdmissionObservation>>,
    observed: &mut Vec<RerankPoolObservedCandidate>,
) {
    let mut activation_order: Vec<(&String, &f64)> = graph_activations.iter().collect();
    activation_order.sort_by(|left, right| {
        right
            .1
            .partial_cmp(left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(right.0))
    });
    let graph_ranks: BTreeMap<&str, usize> = activation_order
        .iter()
        .enumerate()
        .map(|(index, (memory_id, _))| (memory_id.as_str(), index + 1))
        .collect();
    let graph_observation = |memory_id: &str, activation: f64| AdmissionObservation {
        source: AdmissionSource::Graph,
        query_index: 0,
        source_rank: graph_ranks.get(memory_id).copied().unwrap_or(1),
        seed_memory_id: None,
        activation: Some(activation),
        graph_route: None,
    };
    for item in merged.iter_mut() {
        if let Some(activation) = graph_activations.get(&item.memory_id) {
            replace_graph_admissions(item, &graph_ranks, *activation);
        }
    }
    let admitted_ids: BTreeSet<&str> = merged.iter().map(|item| item.memory_id.as_str()).collect();
    for record in observed.iter_mut() {
        record.admitted = admitted_ids.contains(record.memory_id.as_str());
        record.dropped_at = (!record.admitted).then(|| "hybrid_cap".to_string());
        record.graph_allocation_rank = graph_allocation_ranks.get(&record.memory_id).copied();
        if let Some(item) = merged
            .iter()
            .find(|item| item.memory_id == record.memory_id)
        {
            record.admissions = item.admissions.clone();
        } else if let Some(activation) = graph_activations.get(&record.memory_id) {
            record
                .admissions
                .retain(|source| source.source != AdmissionSource::Graph);
            if let Some(route_admissions) = graph_route_admissions.get(&record.memory_id) {
                let mut route_admissions = route_admissions.clone();
                let rank = graph_ranks
                    .get(record.memory_id.as_str())
                    .copied()
                    .unwrap_or(1);
                for admission in &mut route_admissions {
                    admission.source_rank = rank;
                }
                record.admissions.extend(route_admissions);
            } else {
                record
                    .admissions
                    .push(graph_observation(&record.memory_id, *activation));
            }
        }
    }
    for item in merged.iter() {
        if !observed
            .iter()
            .any(|record| record.memory_id == item.memory_id)
        {
            observed.push(RerankPoolObservedCandidate {
                memory_id: item.memory_id.clone(),
                merged_rank: observed.len() + 1,
                admitted: true,
                dropped_at: None,
                graph_allocation_rank: graph_allocation_ranks.get(&item.memory_id).copied(),
                admissions: item.admissions.clone(),
            });
        }
    }
    let mut graph_allocation_order: Vec<(&String, &usize)> =
        graph_allocation_ranks.iter().collect();
    graph_allocation_order.sort_by_key(|(_, rank)| **rank);
    for (memory_id, allocation_rank) in graph_allocation_order {
        if observed.iter().any(|record| record.memory_id == *memory_id) {
            continue;
        }
        let activation = graph_activations
            .get(memory_id)
            .copied()
            .unwrap_or_default();
        let admissions = graph_route_admissions
            .get(memory_id)
            .cloned()
            .unwrap_or_else(|| vec![graph_observation(memory_id, activation)]);
        observed.push(RerankPoolObservedCandidate {
            memory_id: memory_id.clone(),
            merged_rank: observed.len() + 1,
            admitted: false,
            dropped_at: Some("graph_cap".to_string()),
            graph_allocation_rank: Some(*allocation_rank),
            admissions,
        });
    }
}

/// `MaxSim`: sum over query tokens of the max dot-product against doc tokens.
/// Vectors are L2-normalized by the encoder, so dots are cosines.
pub(crate) fn merge_rerank_pools(
    ann_pool: &[PackPoolItem],
    bm25_pool: &[PackPoolItem],
    max_candidates: usize,
) -> Vec<PackPoolItem> {
    let mut merged = Vec::new();
    let max_len = ann_pool.len().max(bm25_pool.len());
    for index in 0..max_len {
        for pool in [ann_pool, bm25_pool] {
            let Some(item) = pool.get(index) else {
                continue;
            };
            if let Some(existing) = merged
                .iter_mut()
                .find(|existing: &&mut PackPoolItem| existing.memory_id == item.memory_id)
            {
                existing.merge_admissions_from(item);
            } else if merged.len() < max_candidates {
                merged.push(item.clone());
            }
        }
        if merged.len() >= max_candidates {
            break;
        }
    }
    merged
}

pub(crate) fn merge_rerank_pools_with_trace(
    ann_pool: &[PackPoolItem],
    bm25_pool: &[PackPoolItem],
    max_candidates: usize,
) -> (Vec<PackPoolItem>, Vec<RerankPoolObservedCandidate>) {
    let all = merge_rerank_pools(
        ann_pool,
        bm25_pool,
        ann_pool.len().saturating_add(bm25_pool.len()),
    );
    let admitted = all.iter().take(max_candidates).cloned().collect::<Vec<_>>();
    let observed = all
        .into_iter()
        .enumerate()
        .map(|(index, item)| {
            let admitted = index < max_candidates;
            RerankPoolObservedCandidate {
                memory_id: item.memory_id,
                merged_rank: index + 1,
                admitted,
                dropped_at: (!admitted).then(|| "hybrid_cap".to_string()),
                graph_allocation_rank: None,
                admissions: item.admissions,
            }
        })
        .collect();
    (admitted, observed)
}

/// Write a deterministic logical JSONL export of canonical store tables.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is invalid,
/// the output path already exists, or filesystem/SQLite export work fails.
fn build_pack_on_connection(connection: &Connection, request: &PackRequest) -> Result<PackReport> {
    let per_query_limit = request.max_memories.min(MAX_BATCH_QUERY_LIMIT);
    let mut query_reports: Vec<SearchReport> = Vec::with_capacity(request.queries.len());

    for (i, query) in request.queries.iter().enumerate() {
        #[cfg(not(feature = "semantic"))]
        let _ = i;
        #[cfg(feature = "semantic")]
        let embedding_opt = request
            .query_embeddings
            .as_ref()
            .and_then(|embs| embs.get(i));

        #[cfg(feature = "semantic")]
        if let Some(embedding) = embedding_opt {
            let report = ann_search_for_pack(
                connection,
                query,
                embedding,
                per_query_limit,
                &request.filters,
            )?;
            query_reports.push(report);
            continue;
        }

        // BM25 FTS path (default or when no embedding available for this query)
        let search_request = SearchRequest {
            query: query.clone(),
            filters: request.filters.clone(),
            limit: per_query_limit,
            offset: 0,
            snippet_chars: 240,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
            maxsim_shortlist: 0,
        };
        query_reports.push(search_memories_on_connection(connection, &search_request)?);
    }

    let mut seen = BTreeSet::new();
    let mut unique_results = Vec::new();
    let mut truncated = query_reports.iter().any(|report| report.truncated);

    // Round-robin across the per-query result lists (rank 1 of each query,
    // then rank 2, ...), draining the reports so results move instead of clone.
    let mut result_queues = query_reports
        .into_iter()
        .map(|report| report.results.into_iter())
        .collect::<Vec<_>>();
    'items: loop {
        let mut any_remaining = false;
        for queue in &mut result_queues {
            let Some(result) = queue.next() else {
                continue;
            };
            any_remaining = true;
            if seen.insert(result.memory_id.clone()) {
                if unique_results.len() >= request.max_memories {
                    truncated = true;
                    break 'items;
                }
                unique_results.push(result);
            }
        }
        if !any_remaining {
            break;
        }
    }

    let top_score = unique_results
        .iter()
        .map(|result| result.score)
        .reduce(f64::max);
    if top_score.is_some_and(|score| score < request.min_score) {
        return Ok(empty_pack(request));
    }

    let last_synth: Option<String> = connection
        .query_row(
            "SELECT started_at FROM dream_runs WHERE status = 'succeeded' \
             ORDER BY started_at DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let (content, memory_ids, scores, text_truncated) =
        format_pack_markdown(request, &unique_results, last_synth.as_deref());
    Ok(PackReport {
        title: request.title.clone(),
        format: request.format.clone(),
        content,
        memory_ids,
        scores,
        truncated: truncated || text_truncated,
        top_score,
    })
}

#[cfg(feature = "semantic")]
fn ann_search_for_pack(
    connection: &Connection,
    query: &str,
    embedding: &[f32],
    limit: usize,
    filters: &SearchFilters,
) -> Result<SearchReport> {
    let table = semantic_table_for_dims(embedding.len())?;
    if !table_exists(connection, &table)? {
        // Vec index missing; fall back to BM25.
        let search_request = SearchRequest {
            query: query.to_string(),
            filters: filters.clone(),
            limit,
            offset: 0,
            snippet_chars: 240,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
            maxsim_shortlist: 0,
        };
        return search_memories_on_connection(connection, &search_request);
    }

    // Build a PreparedSearchRequest so we can use semantic_candidates.
    let search_request = SearchRequest {
        query: query.to_string(),
        filters: filters.clone(),
        limit,
        offset: 0,
        snippet_chars: 240,
        include_content: false,
        include_source: false,
        semantic_fallback: "fallback".to_string(),
        lexical_fallback: "conservative".to_string(),
        embedding: Some(embedding.to_vec()),
        query_token_embedding: None,
        token_model_id: None,
        maxsim_shortlist: 0,
    };
    let prepared = prepare_search_request(&search_request)?;

    let mut candidates = semantic_candidates(connection, &prepared, embedding)?;
    let now_jd = now_julian_day(connection)?;
    for candidate in &mut candidates {
        candidate.score = score_semantic_candidate(candidate, &prepared, now_jd);
    }
    candidates.sort_by(compare_candidates);
    let truncated = candidates.len() > prepared.limit;
    if truncated {
        candidates.truncate(prepared.limit);
    }
    let total_estimate = candidates.len().saturating_add(usize::from(truncated));
    let results = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| candidate.into_result(index + 1, &prepared))
        .collect::<Vec<_>>();
    Ok(SearchReport {
        strategy: "ann_pack_v0".to_string(),
        semantic_attempted: true,
        semantic_reason: "query_embedding_provided".to_string(),
        total_estimate,
        truncated,
        results,
    })
}
pub(crate) fn format_pack_markdown(
    request: &PackRequest,
    results: &[SearchResult],
    last_synth: Option<&str>,
) -> (String, Vec<String>, Vec<f64>, bool) {
    let mut content = format!("## Retrieved Memory: {}\n", request.title.trim());
    let mut memory_ids = Vec::new();
    let mut scores = Vec::new();
    let mut truncated = false;

    if content.chars().count() > request.max_chars {
        return (
            bounded_char_slice(&content, 0, request.max_chars),
            memory_ids,
            scores,
            true,
        );
    }

    if results.is_empty() {
        let line = "- No matching active memories.\n";
        if !append_with_char_budget(&mut content, line, request.max_chars) {
            truncated = true;
        }
        return (content, memory_ids, scores, truncated);
    }

    let build_line = |result: &SearchResult| -> String {
        let text = result
            .summary
            .as_deref()
            .filter(|summary| !summary.trim().is_empty())
            .unwrap_or(&result.snippet);
        let text = collapse_whitespace(text);
        let tags = if result.tags.is_empty() {
            "-".to_string()
        } else {
            result.tags.join(",")
        };
        let marker = freshness_marker(&result.silo, result.metadata_json.as_deref(), last_synth);
        format!(
            "- [Observed at: {}] [{}:{}] {} (space={}, silo={}, scope={}, tags={}){}\n",
            result.observed_at,
            result.kind,
            result.memory_id,
            text,
            result.space,
            result.silo,
            result.scope,
            tags,
            marker
        )
    };

    for result in results {
        let line = build_line(result);
        if append_with_char_budget(&mut content, &line, request.max_chars) {
            memory_ids.push(result.memory_id.clone());
            scores.push(result.score);
        } else {
            truncated = true;
            break;
        }
    }

    // Budget fallback: the loop injected nothing because the top eligible
    // result's line is by itself larger than the remaining char budget. Rather
    // than drop a confidently-matched memory purely for being long, inject the
    // top result truncated to fit. Mirrors `assemble_reranked_pack`'s budget
    // fallback so the non-rerank pack path (no reranker / rerank failure /
    // FTS-only build) degrades the same way instead of injecting an empty pack.
    // `build_pack_on_connection` already filtered `results` by `min_score`, so
    // every entry here is eligible.
    if memory_ids.is_empty() {
        if let Some(result) = results.first() {
            let line = build_line(result);
            let remaining = request.max_chars.saturating_sub(content.chars().count());
            if let Some(entry) = truncate_pack_line(&line, remaining) {
                content.push_str(&entry);
                memory_ids.push(result.memory_id.clone());
                scores.push(result.score);
                truncated = true;
            }
        }
    }

    if memory_ids.len() < results.len() {
        truncated = true;
    }
    (content, memory_ids, scores, truncated)
}

/// Truncate a single rendered pack line (with its trailing newline) to fit
/// `budget` characters, cutting on a `char` boundary and appending an ellipsis
/// marker + newline. Returns `None` when `budget` cannot hold the marker plus at
/// least one character of the line. Char-count based to match the rest of the
/// `format_pack_markdown` budget path (`append_with_char_budget`).
fn truncate_pack_line(line: &str, budget: usize) -> Option<String> {
    const SUFFIX: &str = "…\n"; // ellipsis marker + newline
    let suffix_len = SUFFIX.chars().count();
    if budget <= suffix_len {
        return None;
    }
    let body = line.strip_suffix('\n').unwrap_or(line);
    let keep = budget - suffix_len;
    let truncated: String = body.chars().take(keep).collect();
    if truncated.is_empty() {
        return None;
    }
    let mut entry = String::with_capacity(truncated.len() + SUFFIX.len());
    entry.push_str(&truncated);
    entry.push_str(SUFFIX);
    Some(entry)
}

fn append_with_char_budget(output: &mut String, text: &str, max_chars: usize) -> bool {
    let current = output.chars().count();
    let additional = text.chars().count();
    if current.saturating_add(additional) <= max_chars {
        output.push_str(text);
        true
    } else {
        false
    }
}

/// Interleave per-query candidate pools rank-by-rank, deduplicating by memory
/// id (multi-query analog of the dedupe in pack-pool construction).
pub(crate) fn interleave_pools(pools: &[Vec<PackPoolItem>], cap: usize) -> Vec<PackPoolItem> {
    let mut merged = Vec::new();
    let max_len = pools.iter().map(Vec::len).max().unwrap_or(0);
    for index in 0..max_len {
        for pool in pools {
            if let Some(item) = pool.get(index) {
                if let Some(existing) = merged
                    .iter_mut()
                    .find(|existing: &&mut PackPoolItem| existing.memory_id == item.memory_id)
                {
                    existing.merge_admissions_from(item);
                } else if merged.len() < cap {
                    merged.push(item.clone());
                }
            }
        }
    }
    merged
}
