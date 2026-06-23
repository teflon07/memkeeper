#![forbid(unsafe_code)]

//! Minimal CLI for `memkeeper`.
//!
//! v0.1 starts with deterministic store initialization and diagnostics before
//! adding write/search commands.

use std::{
    collections::BTreeSet,
    env,
    fmt::Write as _,
    fs,
    io::{self, BufRead, Write as IoWrite},
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use memkeeper_protocol::{Command, ErrorCode, PROTOCOL_VERSION};
use memkeeper_store::{
    approve_candidate, backup_store, batch_search_memories,
    build_hybrid_rerank_pool_with_expansion, build_pack, create_space, document_duplicates,
    dream_store, expanded_pack_queries, export_store, forget_memory, get_document, get_memory,
    graph_context, graph_full, graph_neighbors, import_store, ingest_source, init_store,
    inspect_store_stats, last_synthesis_run, list_candidates, list_memories, list_silos,
    list_spaces, mark_source_episodes_extracted, memory_history, merge_entity,
    promotion_candidates, prune_documents, record_recall, reject_candidate, remember_memory,
    schema_mentions_required_objects, search_documents, search_entities, search_memories,
    store_stats, store_stats_with_health, submit_candidate, upsert_entity, upsert_relationship,
    verify_memory, BackupReport, BackupRequest, BatchSearchItemReport, BatchSearchQuery,
    BatchSearchReport, BatchSearchRequest, CandidateApproveReport, CandidateApproveRequest,
    CandidateListReport, CandidateListRequest, CandidateRecord, CandidateRejectReport,
    CandidateRejectRequest, CandidateSubmitReport, CandidateSubmitRequest, DocumentChunk,
    DocumentDuplicatesReport, DocumentDuplicatesRequest, DocumentGetReport, DocumentGetRequest,
    DocumentPruneReport, DocumentPruneRequest, DocumentSearchReport, DocumentSearchRequest,
    DocumentSearchResult, DreamDedupeReport, DreamDuplicateProposal, DreamExpireReport,
    DreamGraphReport, DreamPromoteReport, DreamReindexReport, DreamRelationshipProposal,
    DreamReport, DreamRequest, DuplicateChunkCluster, DuplicateChunkMember, EntityMergeReport,
    EntityMergeRequest, EntityRecord, EntitySearchReport, EntitySearchRequest, EntitySearchResult,
    EntityUpsertReport, EntityUpsertRequest, Error as StoreError, ExportReport, ExportRequest,
    ExportTableReport, ForgetReport, ForgetRequest, GetOptions, GraphContextReport,
    GraphContextRequest, GraphEntityRecord, GraphFullReport, GraphNeighborsReport,
    GraphNeighborsRequest, GraphRelationshipRecord, HealthStats, HistoryOptions, HistoryReport,
    ImportReport, ImportRequest, IndexStats, IngestReport, IngestRequest, InitReport,
    MarkExtractedReport, MarkExtractedRequest, MemoryEventRecord, MemoryLinkRecord, MemoryListItem,
    MemoryListReport, MemoryListRequest, MemoryRecord, MemoryVersionRecord, PackExpansionOptions,
    PackReport, PackRequest, PromotionCandidate, PromotionCandidatesReport,
    PromotionCandidatesRequest, RecallEvent, RecallLogReport, RecallLogRequest, RelationshipRecord,
    RelationshipUpsertReport, RelationshipUpsertRequest, RememberCandidate,
    RememberConflictCandidate, RememberReport, RememberRequest, SearchFilters, SearchReport,
    SearchRequest, SearchResult, SiloListReport, SiloListRequest, SiloRecord, SpaceCreateReport,
    SpaceCreateRequest, SpaceListReport, SpaceRecord, Stats, VerifyReport, VerifyRequest,
    DEFAULT_DREAM_MAX_MEMORIES, DEFAULT_PROMOTE_RANK_CAP, DEFAULT_PROMOTE_SCORE_FLOOR,
    DEFAULT_PROMOTE_THRESHOLD, PROJECT_STORE_RELATIVE_PATH, SCHEMA_VERSION, USER_STORE_PATH_HINT,
};
#[cfg(feature = "embed")]
use std::sync::Mutex;

const MAX_JSON_BYTES: usize = 1_048_576;
const MAX_JSON_DEPTH: usize = 64;
const MAX_JSON_OBJECT_FIELDS: usize = 256;
const MAX_JSON_ARRAY_ITEMS: usize = 4_096;
const MAX_SPACE_TEXT_OUTPUT_CHARS: usize = 512;
const MAX_SPACE_CONFIG_OUTPUT_CHARS: usize = 2_048;

mod dashboard;
mod hook;
mod json;
mod mcp;
mod output;
mod requests;
mod schema;
mod serve;

#[allow(clippy::wildcard_imports)]
pub(crate) use hook::*;
#[allow(clippy::wildcard_imports)]
pub(crate) use json::*;
#[allow(clippy::wildcard_imports)]
pub(crate) use mcp::*;
#[allow(clippy::wildcard_imports)]
pub(crate) use output::*;
#[allow(clippy::wildcard_imports)]
pub(crate) use requests::*;
#[allow(clippy::wildcard_imports)]
pub(crate) use serve::*;

#[allow(clippy::too_many_lines)]
fn main() {
    let mut args = env::args().skip(1);
    let Some(command) = args.next() else {
        print_help();
        return;
    };

    let exit_code = match command.as_str() {
        "--help" | "-h" | "help" => {
            print_help();
            0
        }
        "--version" | "-V" | "version" => {
            println!("memkeeper {PROTOCOL_VERSION} schema {SCHEMA_VERSION}");
            0
        }
        "schema-status" => {
            print_schema_status();
            0
        }
        "schema" => {
            let args = args.collect::<Vec<_>>();
            schema::run_schema(&args)
        }
        "serve" => {
            let args = args.collect::<Vec<_>>();
            run_serve(&args)
        }
        "mcp" => {
            let args = args.collect::<Vec<_>>();
            run_mcp(&args)
        }
        "doctor" => {
            let args = args.collect::<Vec<_>>();
            run_doctor(&args)
        }
        "init" => {
            let args = args.collect::<Vec<_>>();
            run_init(&args)
        }
        "stats" => {
            let args = args.collect::<Vec<_>>();
            run_stats(&args)
        }
        "space-list" => {
            let args = args.collect::<Vec<_>>();
            run_space_list(&args)
        }
        "space-create" => {
            let args = args.collect::<Vec<_>>();
            run_space_create(&args)
        }
        "silo-list" => {
            let args = args.collect::<Vec<_>>();
            run_silo_list(&args)
        }
        "remember" => {
            let args = args.collect::<Vec<_>>();
            run_remember(&args)
        }
        "ingest" => {
            let args = args.collect::<Vec<_>>();
            run_ingest(&args)
        }
        "document-search" => {
            let args = args.collect::<Vec<_>>();
            run_document_search(&args)
        }
        "document-get" => {
            let args = args.collect::<Vec<_>>();
            run_document_get(&args)
        }
        "promotion-candidates" => {
            let args = args.collect::<Vec<_>>();
            run_promotion_candidates(&args)
        }
        "document-duplicates" => {
            let args = args.collect::<Vec<_>>();
            run_document_duplicates(&args)
        }
        "document-prune" => {
            let args = args.collect::<Vec<_>>();
            run_document_prune(&args)
        }
        "mark-extracted" => {
            let args = args.collect::<Vec<_>>();
            run_mark_extracted(&args)
        }
        "search" => {
            let args = args.collect::<Vec<_>>();
            run_search(&args)
        }
        "entity-upsert" => {
            let args = args.collect::<Vec<_>>();
            run_entity_upsert(&args)
        }
        "relationship-upsert" => {
            let args = args.collect::<Vec<_>>();
            run_relationship_upsert(&args)
        }
        "entity-merge" => {
            let args = args.collect::<Vec<_>>();
            run_entity_merge(&args)
        }
        "entity-search" => {
            let args = args.collect::<Vec<_>>();
            run_entity_search(&args)
        }
        "graph-neighbors" => {
            let args = args.collect::<Vec<_>>();
            run_graph_neighbors(&args)
        }
        "graph-context" => {
            let args = args.collect::<Vec<_>>();
            run_graph_context(&args)
        }
        "graph-full" => {
            let args = args.collect::<Vec<_>>();
            run_graph_full(&args)
        }
        "memory-list" => {
            let args = args.collect::<Vec<_>>();
            run_memory_list(&args)
        }
        "batch-search" => {
            let args = args.collect::<Vec<_>>();
            run_batch_search(&args)
        }
        "pack" => {
            let args = args.collect::<Vec<_>>();
            run_pack(&args)
        }
        "rerank" => {
            let args = args.collect::<Vec<_>>();
            run_rerank(&args)
        }
        "get" => {
            let args = args.collect::<Vec<_>>();
            run_get(&args)
        }
        "forget" => {
            let args = args.collect::<Vec<_>>();
            run_forget(&args)
        }
        "verify" => {
            let args = args.collect::<Vec<_>>();
            run_verify(&args)
        }
        "recall-log" => {
            let args = args.collect::<Vec<_>>();
            run_recall_log(&args)
        }
        "candidate-submit" => {
            let args = args.collect::<Vec<_>>();
            run_candidate_submit(&args)
        }
        "candidate-list" => {
            let args = args.collect::<Vec<_>>();
            run_candidate_list(&args)
        }
        "candidate-approve" => {
            let args = args.collect::<Vec<_>>();
            run_candidate_approve(&args)
        }
        "candidate-reject" => {
            let args = args.collect::<Vec<_>>();
            run_candidate_reject(&args)
        }
        "history" => {
            let args = args.collect::<Vec<_>>();
            run_history(&args)
        }
        "export" => {
            let args = args.collect::<Vec<_>>();
            run_export(&args)
        }
        "import" => {
            let args = args.collect::<Vec<_>>();
            run_import(&args)
        }
        "dream" => {
            let args = args.collect::<Vec<_>>();
            run_dream(&args)
        }
        "backup" => {
            let args = args.collect::<Vec<_>>();
            run_backup(&args)
        }
        "reindex" => {
            let args = args.collect::<Vec<_>>();
            run_reindex(&args)
        }
        "hook" => {
            let args = args.collect::<Vec<_>>();
            run_hook(&args)
        }
        "pull-models" => {
            let args = args.collect::<Vec<_>>();
            run_pull_models(&args)
        }
        unsupported => {
            eprintln!("Unsupported command '{unsupported}'.");
            eprintln!("Run `memkeeper --help` for supported commands.");
            2
        }
    };

    process::exit(exit_code);
}

fn run_init(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Init;
    let result = parse_store_args(args)
        .and_then(|options| {
            init_store(&options.store)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                report.schema_version,
                &init_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_doctor(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Doctor;
    let result = parse_doctor_args(args).map(|options| {
        let (result_json, schema_version) = doctor_result_json(&options);
        doctor_success_envelope(&options.store, schema_version, &result_json, started)
    });
    print_result(command, started, result)
}

fn run_stats(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Stats;
    let result = parse_stats_args(args)
        .and_then(|options| {
            let stats = if options.include_health {
                store_stats_with_health(&options.store, options.include_indexes)
            } else {
                store_stats(&options.store, options.include_indexes)
            };
            stats
                .map(|stats| (options.store, stats))
                .map_err(Into::into)
        })
        .map(|(path, stats)| {
            success_envelope(
                command,
                &path,
                stats.schema_version,
                &stats_result_json(&stats),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_space_list(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::SpaceList;
    let result = parse_space_list_args(args)
        .and_then(|options| {
            list_spaces(&options.store)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &space_list_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_space_create(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::SpaceCreate;
    let result = parse_space_create_args(args)
        .and_then(|options| {
            create_space(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &space_create_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_silo_list(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::SiloList;
    let result = parse_silo_list_args(args)
        .and_then(|options| {
            list_silos(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &silo_list_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

#[cfg(feature = "embed")]
struct SemanticModels {
    embed: Option<Mutex<Box<dyn memkeeper_embed::Embedder>>>,
    rerank: Option<Mutex<Box<dyn memkeeper_embed::Reranker>>>,
    colbert: Option<Mutex<Box<dyn memkeeper_embed::TokenEmbedder>>>,
}

/// Late-interaction retrieval gate: `MEMKEEPER_LATE_INTERACTION=1` plus
/// `MEMKEEPER_COLBERT_MODEL_DIR` (checked at model load).
#[cfg(feature = "embed")]
fn late_interaction_enabled() -> bool {
    std::env::var("MEMKEEPER_LATE_INTERACTION").is_ok_and(|value| value == "1")
}

#[cfg(feature = "embed")]
impl SemanticModels {
    fn load(embed: bool, rerank: bool) -> Self {
        let colbert = (embed && late_interaction_enabled())
            .then(memkeeper_embed::colbert_from_env)
            .flatten();
        if let Some(model) = colbert.as_ref() {
            eprintln!("[memkeeper] colbert model loaded: {}", model.model_id());
        }
        Self {
            embed: embed
                .then(memkeeper_embed::embedder_from_env)
                .flatten()
                .map(Mutex::new),
            rerank: rerank
                .then(memkeeper_embed::reranker_from_env)
                .flatten()
                .map(Mutex::new),
            colbert: colbert.map(Mutex::new),
        }
    }

    fn for_remember_or_search() -> Self {
        Self::load(true, false)
    }

    fn for_search(rerank: bool) -> Self {
        Self::load(true, rerank)
    }

    fn for_pack() -> Self {
        Self::load(true, true)
    }

    fn for_serve() -> Self {
        Self::for_pack()
    }

    /// Whether the embedder actually loaded. False means the model files were
    /// missing/unreadable, so search will silently fall back to BM25 — the serve
    /// runtime guard keys on this to fail loud instead.
    fn embed_active(&self) -> bool {
        self.embed.is_some()
    }
}

#[cfg(not(feature = "embed"))]
struct SemanticModels;

#[cfg(not(feature = "embed"))]
impl SemanticModels {
    const fn for_remember_or_search() -> Self {
        Self
    }

    const fn for_search(_rerank: bool) -> Self {
        Self
    }

    const fn for_pack() -> Self {
        Self
    }

    const fn for_serve() -> Self {
        Self::for_pack()
    }
}

fn run_remember(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_remember_or_search();
    run_remember_with_models(args, &semantic_models)
}

fn run_remember_with_models(args: &[String], semantic_models: &SemanticModels) -> i32 {
    let started = Instant::now();
    let command = Command::Remember;
    let result = parse_remember_args(args).and_then(|mut options| {
        maybe_embed_remember_request(&mut options.request, semantic_models);
        maybe_colbert_embed_remember_request(&mut options.request, semantic_models);
        remember_memory(&options.store, &options.request)
            .map(|report| (options.store, report))
            .map_err(Into::into)
    });
    let result = result.map(|(path, report)| {
        success_envelope(
            command,
            &path,
            SCHEMA_VERSION,
            &remember_result_json(&report),
            started,
        )
    });
    print_result(command, started, result)
}

fn run_search(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Search;
    // Parse before loading models so the reranker only loads when requested.
    let result = parse_search_args(args).and_then(|options| {
        let semantic_models = SemanticModels::for_search(options.rerank);
        execute_search(
            &options.store,
            options.request,
            options.rerank,
            options.rerank_candidates,
            &semantic_models,
        )
        .map(|result_json| (options.store, result_json))
    });
    let result = result.map(|(path, result_json)| {
        success_envelope(command, &path, SCHEMA_VERSION, &result_json, started)
    });
    print_result(command, started, result)
}

/// Pure fail-closed decision for the one-shot search path, mirroring serve's
/// `runtime_semantic_guard`: refuse when the operator requires semantic
/// retrieval but this request cannot be embedded — no caller-supplied vector
/// (`request_has_embedding`) and no active embedder (`embed_active`, always
/// false on a non-semantic build). Kept pure so it is unit-testable without
/// touching the `MEMKEEPER_REQUIRE_SEMANTIC` process env.
fn must_refuse_search_without_semantic(
    request_has_embedding: bool,
    embed_active: bool,
    require_semantic: bool,
) -> bool {
    require_semantic && !request_has_embedding && !embed_active
}

/// Run one search, optionally reranking the widened result pool with the
/// warm cross-encoder, and return the result JSON. Shared by the CLI path
/// and the serve dispatcher. Rerank is best-effort: any reranker failure
/// falls back to the plain order truncated to the requested limit.
fn execute_search(
    store: &Path,
    mut request: SearchRequest,
    rerank: bool,
    rerank_candidates: usize,
    semantic_models: &SemanticModels,
) -> Result<String, CliError> {
    if let Err(message) = maybe_embed_search_request(&mut request, semantic_models) {
        // Fail closed when the operator requires semantic retrieval: a runtime
        // embedding failure must not be served as a silent FTS-only success.
        if serve::require_semantic_env() {
            return Err(CliError::SemanticUnavailable(format!(
                "semantic retrieval required but {message}"
            )));
        }
    }
    // Also fail closed for the *absent*-embedder case (no embedder configured, or
    // a non-semantic build), not just runtime embed failures. Without this, a
    // one-shot `search` under MEMKEEPER_REQUIRE_SEMANTIC silently returned a BM25
    // result with semantic.reason="missing_embedding" — serve refused but the
    // direct CLI/dispatch path did not. Mirrors serve's runtime guard.
    #[cfg(feature = "embed")]
    let embed_active = semantic_models.embed_active();
    #[cfg(not(feature = "embed"))]
    let embed_active = false;
    if must_refuse_search_without_semantic(
        request.embedding.is_some(),
        embed_active,
        serve::require_semantic_env(),
    ) {
        return Err(CliError::SemanticUnavailable(
            "semantic retrieval required but no embedder is available".to_string(),
        ));
    }
    maybe_colbert_embed_search_request(&mut request, semantic_models);
    let rerank = rerank && has_reranker(semantic_models) && request.limit > 0;
    if !rerank {
        let report = search_memories(store, &request)?;
        let last_synth = last_synthesis_run(store).unwrap_or(None);
        return Ok(search_result_json(&report, last_synth.as_deref(), false));
    }

    let user_limit = request.limit;
    let user_include_content = request.include_content;
    // Widen the pool so the reranker has candidates to reorder, and fetch
    // contents for cross-encoder scoring regardless of the caller's
    // include_content choice.
    request.limit = rerank_candidates
        .max(user_limit)
        .min(memkeeper_store::MAX_SEARCH_LIMIT);
    request.include_content = true;
    let late_interaction = request.query_token_embedding.is_some();
    let mut report = search_memories(store, &request)?;
    let reranked = rerank_search_report(
        &mut report,
        &request.query,
        semantic_models,
        late_interaction,
    );
    finalize_search_window(
        &mut report,
        user_limit,
        user_include_content,
        request.offset,
    );
    let last_synth = last_synthesis_run(store).unwrap_or(None);
    Ok(search_result_json(&report, last_synth.as_deref(), reranked))
}

/// Truncate a (possibly reranked) report back to the caller's limit, renumber
/// ranks, and strip contents the caller did not ask for.
fn finalize_search_window(
    report: &mut SearchReport,
    limit: usize,
    include_content: bool,
    offset: usize,
) {
    if report.results.len() > limit {
        report.results.truncate(limit);
        report.truncated = true;
    }
    for (index, result) in report.results.iter_mut().enumerate() {
        result.rank = offset + index + 1;
        if !include_content {
            result.content = None;
        }
    }
}

/// Default cap (chars) on cross-encoder rerank inputs. Scoring cost scales
/// with sequence length and memory claims lead with their key sentence, so a
/// prefix preserves ranking quality at a fraction of the cost (measured: 12
/// full-length docs ~1.0s, 256-char prefixes ~0.34s warm). Tunable via
/// `MEMKEEPER_RERANK_DOC_CHARS`; 0 disables truncation. Scoring input only --
/// injected/returned content is never truncated by this.
#[cfg(feature = "embed")]
const DEFAULT_RERANK_DOC_CHARS: usize = 512;

#[cfg(feature = "embed")]
fn rerank_doc_chars() -> usize {
    std::env::var("MEMKEEPER_RERANK_DOC_CHARS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_RERANK_DOC_CHARS)
}

/// Char-boundary-safe prefix of `content` for rerank scoring. `limit == 0`
/// disables truncation.
#[cfg(feature = "embed")]
fn rerank_doc(content: &str, limit: usize) -> &str {
    if limit == 0 {
        return content;
    }
    match content.char_indices().nth(limit) {
        Some((byte_index, _)) => &content[..byte_index],
        None => content,
    }
}

/// Combine content scores with per-candidate summary scores by max.
/// `summary_idx[k]` is the index of the candidate whose summary produced
/// `summary_scores[k]`. Scoring summary and content separately (instead of
/// concatenating) keeps prefix-duplicate summaries from corrupting the
/// cross-encoder ordering while still letting a summary-carried answer win.
#[cfg(any(test, feature = "embed"))]
fn max_combine_rerank_scores(
    content_scores: &[f32],
    summary_scores: &[f32],
    summary_idx: &[usize],
) -> Vec<f32> {
    let mut combined = content_scores.to_vec();
    for (offset, &index) in summary_idx.iter().enumerate() {
        if let (Some(slot), Some(&score)) = (combined.get_mut(index), summary_scores.get(offset)) {
            if score > *slot {
                *slot = score;
            }
        }
    }
    combined
}

#[cfg(feature = "embed")]
fn has_reranker(semantic_models: &SemanticModels) -> bool {
    semantic_models.rerank.is_some()
}

#[cfg(not(feature = "embed"))]
const fn has_reranker(_semantic_models: &SemanticModels) -> bool {
    false
}

/// Reorder a search report's results by cross-encoder relevance. Returns
/// whether reranking was applied; on any failure the report keeps the plain
/// retrieval order.
#[cfg(feature = "embed")]
fn rerank_search_report(
    report: &mut SearchReport,
    query: &str,
    semantic_models: &SemanticModels,
    late_interaction: bool,
) -> bool {
    let Some(reranker) = semantic_models.rerank.as_ref() else {
        return false;
    };
    if report.results.is_empty() || query.is_empty() {
        return false;
    }
    let doc_limit = rerank_doc_chars();
    // Late-interaction mode scores summary + content (the eval-validated A1
    // input); the legacy path stays content-only so flag-off behavior is
    // byte-identical.
    let owned_docs: Vec<String> = report
        .results
        .iter()
        .map(|result| {
            let content = result
                .content
                .as_deref()
                .filter(|content| !content.is_empty())
                .unwrap_or(result.snippet.as_str());
            match result.summary.as_deref().filter(|s| !s.is_empty()) {
                Some(summary) if late_interaction => format!("{summary}\n\n{content}"),
                _ => content.to_string(),
            }
        })
        .collect();
    let docs: Vec<&str> = owned_docs
        .iter()
        .map(|doc| rerank_doc(doc, doc_limit))
        .collect();
    let scores = match reranker.lock() {
        Ok(mut reranker) => match reranker.rerank(query, &docs) {
            Ok(scores) => scores,
            Err(error) => {
                eprintln!("[memkeeper] search rerank failed: {error}");
                return false;
            }
        },
        Err(error) => {
            eprintln!("[memkeeper] rerank model lock failed: {error}");
            return false;
        }
    };
    if scores.len() != report.results.len() {
        eprintln!(
            "[memkeeper] reranker returned {} scores for {} results -- keeping plain order",
            scores.len(),
            report.results.len()
        );
        return false;
    }
    let mut order: Vec<usize> = (0..report.results.len()).collect();
    order.sort_by(|a, b| {
        scores[*b]
            .partial_cmp(&scores[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut results = std::mem::take(&mut report.results);
    let mut by_index: Vec<Option<_>> = results.drain(..).map(Some).collect();
    report.results = order
        .into_iter()
        .filter_map(|index| by_index[index].take())
        .collect();
    true
}

#[cfg(not(feature = "embed"))]
fn rerank_search_report(
    _report: &mut SearchReport,
    _query: &str,
    _semantic_models: &SemanticModels,
    _late_interaction: bool,
) -> bool {
    false
}

fn run_entity_upsert(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::EntityUpsert;
    let result = parse_entity_upsert_args(args)
        .and_then(|options| {
            upsert_entity(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &entity_upsert_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_relationship_upsert(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::RelationshipUpsert;
    let result = parse_relationship_upsert_args(args)
        .and_then(|options| {
            upsert_relationship(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &relationship_upsert_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_entity_merge(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::EntityMerge;
    let result = parse_entity_merge_args(args)
        .and_then(|options| {
            merge_entity(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &entity_merge_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_entity_search(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::EntitySearch;
    let result = parse_entity_search_args(args)
        .and_then(|options| {
            search_entities(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &entity_search_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_graph_neighbors(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::GraphNeighbors;
    let result = parse_graph_neighbors_args(args)
        .and_then(|options| {
            graph_neighbors(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &graph_neighbors_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_graph_context(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::GraphContext;
    let result = parse_graph_context_args(args)
        .and_then(|options| {
            graph_context(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &graph_context_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_graph_full(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::GraphFull;
    let result = parse_store_args(args)
        .and_then(|options| {
            graph_full(&options.store)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &graph_full_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_memory_list(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::MemoryList;
    let result = parse_memory_list_args(args)
        .and_then(|options| {
            list_memories(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &memory_list_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_batch_search(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::BatchSearch;
    let result = parse_batch_search_args(args)
        .and_then(|options| {
            batch_search_memories(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            let last_synth = last_synthesis_run(&path).unwrap_or(None);
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &batch_search_result_json(&report, last_synth.as_deref()),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_pack(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_pack();
    run_pack_with_models(args, &semantic_models)
}

fn run_pack_with_models(args: &[String], semantic_models: &SemanticModels) -> i32 {
    let started = Instant::now();
    let command = Command::Pack;
    let result = parse_pack_args(args)
        .and_then(|mut options| {
            apply_pack_query_expansion(&mut options.request, &mut options.expansion);
            maybe_embed_pack_request(&mut options.request, semantic_models);
            maybe_colbert_embed_pack_request(&mut options.request, semantic_models);
            let mut report =
                build_pack(&options.store, &options.request).map_err(CliError::from)?;
            maybe_rerank_pack_report(
                &options.store,
                &options.request,
                options.cosine_gate,
                options.expansion,
                &mut report,
                semantic_models,
            );

            Ok((options.store, report))
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &pack_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn apply_pack_query_expansion(request: &mut PackRequest, expansion: &mut PackExpansionOptions) {
    if !expansion.query_expansion {
        return;
    }
    request.queries = expanded_pack_queries(&request.queries, *expansion);
    request.query_embeddings = None;
    request.query_token_embeddings = None;
    request.token_model_id = None;
    expansion.query_expansion = false;
}

#[cfg(feature = "embed")]
fn maybe_embed_remember_request(request: &mut RememberRequest, semantic_models: &SemanticModels) {
    if request.embedding.is_some() {
        return;
    }
    let Some(model) = semantic_models.embed.as_ref() else {
        return;
    };
    match model.lock() {
        Ok(mut model) => match model.embed_one(&request.content) {
            Ok(vec) => {
                request.embedding = Some(vec);
                request.embedding_model_id = Some(model.model_id().to_string());
            }
            Err(e) => eprintln!("[memkeeper] embedding failed: {e}"),
        },
        Err(e) => eprintln!("[memkeeper] embed model lock failed: {e}"),
    }
}

#[cfg(not(feature = "embed"))]
fn maybe_embed_remember_request(_request: &mut RememberRequest, _semantic_models: &SemanticModels) {
}

#[cfg(feature = "embed")]
fn maybe_embed_ingest_request(request: &mut IngestRequest, semantic_models: &SemanticModels) {
    if request.embeddings.is_some() || request.chunks.is_empty() {
        return;
    }
    let Some(model) = semantic_models.embed.as_ref() else {
        return;
    };
    let chunks: Vec<&str> = request.chunks.iter().map(String::as_str).collect();
    match model.lock() {
        Ok(mut model) => match model.embed(&chunks) {
            Ok(vectors) => {
                request.embeddings = Some(vectors);
                request.embedding_model_id = Some(model.model_id().to_string());
            }
            Err(e) => eprintln!("[memkeeper] ingest embedding failed: {e}"),
        },
        Err(e) => eprintln!("[memkeeper] embed model lock failed: {e}"),
    }
}

#[cfg(not(feature = "embed"))]
fn maybe_embed_ingest_request(_request: &mut IngestRequest, _semantic_models: &SemanticModels) {}

#[cfg(feature = "embed")]
fn maybe_embed_document_search_request(
    request: &mut DocumentSearchRequest,
    semantic_models: &SemanticModels,
) {
    if request.embedding.is_some() || request.query.is_empty() {
        return;
    }
    let Some(model) = semantic_models.embed.as_ref() else {
        return;
    };
    match model.lock() {
        Ok(mut model) => match model.embed_one(&request.query) {
            Ok(vec) => request.embedding = Some(vec),
            Err(e) => eprintln!("[memkeeper] document-search embedding failed: {e}"),
        },
        Err(e) => eprintln!("[memkeeper] embed model lock failed: {e}"),
    }
}

#[cfg(not(feature = "embed"))]
fn maybe_embed_document_search_request(
    _request: &mut DocumentSearchRequest,
    _semantic_models: &SemanticModels,
) {
}

#[cfg(feature = "embed")]
fn maybe_colbert_embed_remember_request(
    request: &mut RememberRequest,
    semantic_models: &SemanticModels,
) {
    if request.token_embedding.is_some() {
        return;
    }
    let Some(model) = semantic_models.colbert.as_ref() else {
        return;
    };
    // Doc text convention (eval-validated): summary + "\n\n" + content.
    let text = match request.summary.as_deref() {
        Some(summary) if !summary.is_empty() => format!("{summary}\n\n{}", request.content),
        _ => request.content.clone(),
    };
    match model.lock() {
        Ok(mut model) => match model.encode_docs(&[&text]) {
            Ok(mut vecs) => {
                request.token_embedding = vecs.pop().filter(|tokens| !tokens.is_empty());
                if request.token_embedding.is_some() {
                    request.token_embedding_model_id = Some(model.model_id().to_string());
                }
            }
            Err(error) => eprintln!("[memkeeper] colbert embedding failed: {error}"),
        },
        Err(error) => eprintln!("[memkeeper] colbert model lock failed: {error}"),
    }
}

#[cfg(not(feature = "embed"))]
fn maybe_colbert_embed_remember_request(
    _request: &mut RememberRequest,
    _semantic_models: &SemanticModels,
) {
}

#[cfg(feature = "embed")]
fn maybe_colbert_embed_search_request(
    request: &mut SearchRequest,
    semantic_models: &SemanticModels,
) {
    if request.query_token_embedding.is_some() {
        return;
    }
    let Some(model) = semantic_models.colbert.as_ref() else {
        return;
    };
    match model.lock() {
        Ok(mut model) => match model.encode_query(&request.query) {
            Ok(tokens) if !tokens.is_empty() => {
                request.query_token_embedding = Some(tokens);
                request.token_model_id = Some(model.model_id().to_string());
            }
            Ok(_) => {}
            Err(error) => eprintln!("[memkeeper] colbert query encode failed: {error}"),
        },
        Err(error) => eprintln!("[memkeeper] colbert model lock failed: {error}"),
    }
}

#[cfg(not(feature = "embed"))]
fn maybe_colbert_embed_search_request(
    _request: &mut SearchRequest,
    _semantic_models: &SemanticModels,
) {
}

#[cfg(feature = "embed")]
fn maybe_colbert_embed_pack_request(request: &mut PackRequest, semantic_models: &SemanticModels) {
    if request.query_token_embeddings.is_some() {
        return;
    }
    let Some(model) = semantic_models.colbert.as_ref() else {
        return;
    };
    let Ok(mut model) = model.lock() else {
        eprintln!("[memkeeper] colbert model lock failed");
        return;
    };
    let mut all = Vec::with_capacity(request.queries.len());
    for query in &request.queries {
        match model.encode_query(query) {
            Ok(tokens) if !tokens.is_empty() => all.push(tokens),
            Ok(_) => {}
            Err(error) => {
                eprintln!("[memkeeper] colbert query encode failed: {error}");
                return;
            }
        }
    }
    if !all.is_empty() {
        request.query_token_embeddings = Some(all);
        request.token_model_id = Some(model.model_id().to_string());
    }
}

#[cfg(not(feature = "embed"))]
fn maybe_colbert_embed_pack_request(_request: &mut PackRequest, _semantic_models: &SemanticModels) {
}

#[cfg(feature = "embed")]
fn maybe_embed_search_request(
    request: &mut SearchRequest,
    semantic_models: &SemanticModels,
) -> Result<(), String> {
    if request.embedding.is_some() {
        return Ok(());
    }
    let Some(model) = semantic_models.embed.as_ref() else {
        // No embedder configured for this run; not a runtime failure.
        return Ok(());
    };
    // The embedder loaded, so semantic was expected. A failure here is a
    // runtime degradation (provider rate-limit/outage), distinct from "no
    // embedder" — return Err so callers can fail closed under
    // MEMKEEPER_REQUIRE_SEMANTIC instead of silently serving FTS.
    match model.lock() {
        Ok(mut model) => match model.embed_one(&request.query) {
            Ok(vec) => {
                request.embedding = Some(vec);
                Ok(())
            }
            Err(e) => {
                eprintln!("[memkeeper] search embedding failed: {e}");
                Err(format!("query embedding failed: {e}"))
            }
        },
        Err(e) => {
            eprintln!("[memkeeper] embed model lock failed: {e}");
            Err(format!("embed model lock failed: {e}"))
        }
    }
}

#[cfg(not(feature = "embed"))]
fn maybe_embed_search_request(
    _request: &mut SearchRequest,
    _semantic_models: &SemanticModels,
) -> Result<(), String> {
    Ok(())
}

#[cfg(feature = "embed")]
fn maybe_embed_pack_request(request: &mut PackRequest, semantic_models: &SemanticModels) {
    if request.query_embeddings.is_some() {
        return;
    }
    let Some(model) = semantic_models.embed.as_ref() else {
        return;
    };
    let query_strs: Vec<&str> = request.queries.iter().map(String::as_str).collect();
    match model.lock() {
        Ok(mut model) => match model.embed(&query_strs) {
            Ok(vecs) => request.query_embeddings = Some(vecs),
            Err(e) => eprintln!("[memkeeper] pack embedding failed: {e}"),
        },
        Err(e) => eprintln!("[memkeeper] embed model lock failed: {e}"),
    }
}

#[cfg(not(feature = "embed"))]
fn maybe_embed_pack_request(_request: &mut PackRequest, _semantic_models: &SemanticModels) {}

#[cfg(feature = "embed")]
fn maybe_rerank_pack_report(
    store: &Path,
    request: &PackRequest,
    cosine_gate: f64,
    expansion: PackExpansionOptions,
    report: &mut PackReport,
    semantic_models: &SemanticModels,
) {
    // `build_pack` populated `report.scores` with RETRIEVAL-scale values. The
    // promote score-floor (and the auto-retrieve hook that logs these) are
    // calibrated to the cross-encoder RERANK scale, so retrieval-scale scores
    // must never leak out as if they were rerank scores. Clear them up front:
    // every early return below (no reranker / pool or rerank failure) then
    // yields empty scores -> the hook logs NULL -> the floor excludes them
    // (safe degradation: no promotion without the reranker). The success path
    // replaces the whole report (with rerank-scale scores) via
    // `assemble_reranked_pack`.
    report.scores.clear();
    let Some(reranker) = semantic_models.rerank.as_ref() else {
        // No cross-encoder available: when a gate is configured, still suppress
        // off-topic injection using the embedding's top retrieval score alone.
        if cosine_gate > 0.0 {
            let pool_width = request.rerank_candidates.max(request.max_memories);
            let mut pool_request = request.clone();
            pool_request.max_memories = pool_width;
            pool_request.min_score = 0.0;
            pool_request.rerank_candidates = 0;
            match memkeeper_store::build_pack_pool_with_expansion(store, &pool_request, expansion) {
                Ok(pool) => {
                    let cos_top = pool.iter().map(|item| item.score).fold(f64::MIN, f64::max);
                    if memkeeper_store::pack_blocked_by_cosine_gate(cos_top, cosine_gate) {
                        *report = memkeeper_store::empty_pack(request);
                    }
                }
                Err(error) => eprintln!("[memkeeper] cosine gate pool build failed: {error}"),
            }
        }
        return;
    };
    let query = request.queries.first().map_or("", String::as_str);
    if query.is_empty() {
        return;
    }

    // Pool building (ANN + BM25 safety net + content fetch) runs in the store
    // on one read snapshot; the CLI only invokes the models.
    let pool_width = request.rerank_candidates.max(request.max_memories);
    let pool = match build_hybrid_rerank_pool_with_expansion(store, request, pool_width, expansion)
    {
        Ok(pool) => pool,
        Err(error) => {
            eprintln!("[memkeeper] rerank pool build failed: {error}");
            return;
        }
    };
    if pool.candidates.is_empty() {
        if cosine_gate > 0.0 {
            *report = memkeeper_store::empty_pack(request);
        }
        return;
    }
    let doc_limit = rerank_doc_chars();
    // Content is always scored; late-interaction candidates with a summary get
    // a second pass in the same rerank batch and keep the better score.
    let mut doc_refs: Vec<&str> = pool
        .candidates
        .iter()
        .map(|candidate| rerank_doc(&candidate.content, doc_limit))
        .collect();
    let summary_idx: Vec<usize> = pool
        .candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| candidate.summary.as_ref().map(|_| index))
        .collect();
    for &index in &summary_idx {
        if let Some(summary) = pool.candidates[index].summary.as_deref() {
            doc_refs.push(rerank_doc(summary, doc_limit));
        }
    }
    let scores = match reranker.lock() {
        Ok(mut reranker) => match reranker.rerank(query, &doc_refs) {
            Ok(scores) => scores,
            Err(e) => {
                eprintln!("[memkeeper] rerank failed: {e}");
                return;
            }
        },
        Err(e) => {
            eprintln!("[memkeeper] rerank model lock failed: {e}");
            return;
        }
    };
    if scores.len() != doc_refs.len() {
        eprintln!(
            "[memkeeper] reranker returned {} scores for {} rerank texts -- skipping rerank",
            scores.len(),
            doc_refs.len()
        );
        return;
    }
    let (content_scores, summary_scores) = scores.split_at(pool.candidates.len());
    let scores = max_combine_rerank_scores(content_scores, summary_scores, &summary_idx);

    // Hand the scored candidates to the store's pure pack-assembly policy: gate
    // on cos_top / cross-encoder confidence, order by rerank score, then apply
    // the precision floor and char/count budget.
    let candidates: Vec<memkeeper_store::RerankCandidate> = pool
        .candidates
        .into_iter()
        .zip(scores)
        .map(
            |(candidate, rerank_score)| memkeeper_store::RerankCandidate {
                memory_id: candidate.memory_id,
                content: candidate.content,
                rerank_score,
            },
        )
        .collect();
    *report =
        memkeeper_store::assemble_reranked_pack(request, cosine_gate, pool.cos_top, &candidates);
}

#[cfg(not(feature = "embed"))]
fn maybe_rerank_pack_report(
    _store: &Path,
    _request: &PackRequest,
    _cosine_gate: f64,
    _expansion: PackExpansionOptions,
    report: &mut PackReport,
    _semantic_models: &SemanticModels,
) {
    // No reranker in this build: `build_pack` left RETRIEVAL-scale scores in
    // `report.scores`, but the promote floor and the auto-retrieve hook expect
    // the cross-encoder RERANK scale. Clear them so the contract holds (pack
    // scores present <=> rerank-scale); the hook then logs NULL and the floor
    // excludes them (no promotion without the reranker).
    report.scores.clear();
}

/// Standalone cross-encoder scoring of arbitrary documents against a query.
///
/// Unlike `pack`, this touches no store -- documents are supplied inline -- so
/// it can score RSS items, search candidates, or any text the caller has in
/// hand using the reranker the serve loop already holds warm.
#[cfg_attr(not(feature = "semantic"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
}

fn rerank_request_from_json(input: &str) -> Result<RerankRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("rerank request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["query", "documents"])?;
    let query = required_string_field(object, "query")?;
    let documents = required_string_array_field(object, "documents")?;
    Ok(RerankRequest { query, documents })
}

fn parse_rerank_args(args: &[String]) -> Result<RerankRequest, CliError> {
    let mut parser = ArgParser::new(args);
    let mut request_json = None;
    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--json" | "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            value if value.starts_with("--request-json=") => {
                request_json = Some(value.trim_start_matches("--request-json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported rerank flag: {unknown}"
                )));
            }
        }
    }
    let request_json = request_json.ok_or_else(|| {
        CliError::InvalidRequest("missing rerank request JSON after --json".to_string())
    })?;
    rerank_request_from_json(&request_json)
}

fn run_rerank(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_pack();
    let started = Instant::now();
    let command = Command::Rerank;
    let result = parse_rerank_args(args)
        .and_then(|request| run_rerank_payload(&request, &semantic_models))
        .map(|result_json| {
            success_envelope(
                command,
                Path::new(""),
                SCHEMA_VERSION,
                &result_json,
                started,
            )
        });
    print_result(command, started, result)
}

/// Identify the active reranker by the basename of its model directory
/// (e.g. `mxbai-rerank-base`). `RerankerModel` exposes no id accessor, and the
/// daemon always sets `MEMKEEPER_RERANK_MODEL_DIR`, so the dir name is the stable
/// label callers persist alongside scores.
#[cfg(feature = "semantic")]
fn rerank_model_id() -> String {
    std::env::var("MEMKEEPER_RERANK_MODEL_DIR")
        .ok()
        .and_then(|dir| {
            std::path::Path::new(&dir)
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "rerank".to_string())
}

#[cfg(feature = "semantic")]
fn run_rerank_payload(
    request: &RerankRequest,
    semantic_models: &SemanticModels,
) -> Result<String, CliError> {
    let Some(reranker) = semantic_models.rerank.as_ref() else {
        return Err(CliError::InvalidRequest(
            "reranker model unavailable (set MEMKEEPER_RERANK_MODEL_DIR)".to_string(),
        ));
    };
    if request.documents.is_empty() {
        return Ok("{\"model_id\":null,\"scores\":[]}".to_string());
    }
    let docs: Vec<&str> = request.documents.iter().map(String::as_str).collect();
    let model_id = rerank_model_id();
    let mut reranker = reranker
        .lock()
        .map_err(|e| CliError::InvalidRequest(format!("rerank model lock failed: {e}")))?;
    let scores = reranker
        .rerank(&request.query, &docs)
        .map_err(|e| CliError::InvalidRequest(format!("rerank failed: {e}")))?;
    let scores_json = scores
        .iter()
        .map(|score| finite_number_json(f64::from(*score)))
        .collect::<Vec<_>>()
        .join(",");
    Ok(format!(
        "{{\"model_id\":{},\"scores\":[{scores_json}]}}",
        json_string(&model_id)
    ))
}

#[cfg(not(feature = "semantic"))]
fn run_rerank_payload(
    _request: &RerankRequest,
    _semantic_models: &SemanticModels,
) -> Result<String, CliError> {
    Err(CliError::InvalidRequest(
        "rerank requires the semantic feature; this build has it disabled".to_string(),
    ))
}

fn run_get(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Get;
    let result = parse_get_args(args)
        .and_then(|options| {
            get_memory(&options.store, &options.id, options.options)
                .map(|memory| (options.store, memory, options.options))
                .map_err(Into::into)
        })
        .map(|(path, memory, options)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &format!("{{\"memory\":{}}}", memory_json(&memory, options)),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_forget(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Forget;
    let result = parse_forget_args(args)
        .and_then(|options| {
            forget_memory(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &forget_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_verify(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Verify;
    let result = parse_verify_args(args)
        .and_then(|options| {
            verify_memory(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &verify_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_recall_log(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::RecallLog;
    let result = parse_recall_log_args(args)
        .and_then(|options| {
            record_recall(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &recall_log_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct IngestArgs {
    store: PathBuf,
    request: IngestRequest,
}

fn parse_ingest_args(args: &[String]) -> Result<IngestArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "ingest")?;
    Ok(IngestArgs {
        store,
        request: ingest_request_from_json(&request_json)?,
    })
}

struct DocumentGetArgs {
    store: PathBuf,
    request: DocumentGetRequest,
}

fn parse_document_get_args(args: &[String]) -> Result<DocumentGetArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "document-get")?;
    Ok(DocumentGetArgs {
        store,
        request: document_get_request_from_json(&request_json)?,
    })
}

fn run_document_get(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::DocumentGet;
    let result = parse_document_get_args(args)
        .and_then(|options| {
            get_document(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &document_get_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct PromotionCandidatesArgs {
    store: PathBuf,
    request: PromotionCandidatesRequest,
}

fn parse_promotion_candidates_args(args: &[String]) -> Result<PromotionCandidatesArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "promotion-candidates")?;
    Ok(PromotionCandidatesArgs {
        store,
        request: promotion_candidates_request_from_json(&request_json)?,
    })
}

fn run_promotion_candidates(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::PromotionCandidates;
    let result = parse_promotion_candidates_args(args)
        .and_then(|options| {
            promotion_candidates(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &promotion_candidates_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct DocumentDuplicatesArgs {
    store: PathBuf,
    request: DocumentDuplicatesRequest,
}

fn parse_document_duplicates_args(args: &[String]) -> Result<DocumentDuplicatesArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "document-duplicates")?;
    Ok(DocumentDuplicatesArgs {
        store,
        request: document_duplicates_request_from_json(&request_json)?,
    })
}

fn run_document_duplicates(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::DocumentDuplicates;
    let result = parse_document_duplicates_args(args)
        .and_then(|options| {
            document_duplicates(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &document_duplicates_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct DocumentPruneArgs {
    store: PathBuf,
    request: DocumentPruneRequest,
}

fn parse_document_prune_args(args: &[String]) -> Result<DocumentPruneArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "document-prune")?;
    Ok(DocumentPruneArgs {
        store,
        request: document_prune_request_from_json(&request_json)?,
    })
}

fn run_document_prune(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::DocumentPrune;
    let result = parse_document_prune_args(args)
        .and_then(|options| {
            prune_documents(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &document_prune_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct MarkExtractedArgs {
    store: PathBuf,
    request: MarkExtractedRequest,
}

fn parse_mark_extracted_args(args: &[String]) -> Result<MarkExtractedArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "mark-extracted")?;
    Ok(MarkExtractedArgs {
        store,
        request: mark_extracted_request_from_json(&request_json)?,
    })
}

fn run_mark_extracted(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::MarkExtracted;
    let result = parse_mark_extracted_args(args)
        .and_then(|options| {
            mark_source_episodes_extracted(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &mark_extracted_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct DocumentSearchArgs {
    store: PathBuf,
    request: DocumentSearchRequest,
}

fn parse_document_search_args(args: &[String]) -> Result<DocumentSearchArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "document-search")?;
    Ok(DocumentSearchArgs {
        store,
        request: document_search_request_from_json(&request_json)?,
    })
}

fn run_document_search(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_remember_or_search();
    run_document_search_with_models(args, &semantic_models)
}

fn run_document_search_with_models(args: &[String], semantic_models: &SemanticModels) -> i32 {
    let started = Instant::now();
    let command = Command::DocumentSearch;
    let result = parse_document_search_args(args)
        .and_then(|mut options| {
            maybe_embed_document_search_request(&mut options.request, semantic_models);
            search_documents(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &document_search_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_ingest(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_remember_or_search();
    run_ingest_with_models(args, &semantic_models)
}

fn run_ingest_with_models(args: &[String], semantic_models: &SemanticModels) -> i32 {
    let started = Instant::now();
    let command = Command::Ingest;
    let result = parse_ingest_args(args)
        .and_then(|mut options| {
            maybe_embed_ingest_request(&mut options.request, semantic_models);
            ingest_source(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &ingest_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct CandidateSubmitArgs {
    store: PathBuf,
    request: CandidateSubmitRequest,
}

fn parse_candidate_submit_args(args: &[String]) -> Result<CandidateSubmitArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "candidate-submit")?;
    Ok(CandidateSubmitArgs {
        store,
        request: candidate_submit_request_from_json(&request_json)?,
    })
}

fn run_candidate_submit(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::CandidateSubmit;
    let result = parse_candidate_submit_args(args)
        .and_then(|options| {
            submit_candidate(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &candidate_submit_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct CandidateListArgs {
    store: PathBuf,
    request: CandidateListRequest,
}

fn parse_candidate_list_args(args: &[String]) -> Result<CandidateListArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "candidate-list")?;
    Ok(CandidateListArgs {
        store,
        request: candidate_list_request_from_json(&request_json)?,
    })
}

fn run_candidate_list(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::CandidateList;
    let result = parse_candidate_list_args(args)
        .and_then(|options| {
            list_candidates(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &candidate_list_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct CandidateApproveArgs {
    store: PathBuf,
    request: CandidateApproveRequest,
}

fn parse_candidate_approve_args(args: &[String]) -> Result<CandidateApproveArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "candidate-approve")?;
    Ok(CandidateApproveArgs {
        store,
        request: candidate_approve_request_from_json(&request_json)?,
    })
}

fn run_candidate_approve(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::CandidateApprove;
    let result = parse_candidate_approve_args(args)
        .and_then(|options| {
            approve_candidate(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &candidate_approve_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct CandidateRejectArgs {
    store: PathBuf,
    request: CandidateRejectRequest,
}

fn parse_candidate_reject_args(args: &[String]) -> Result<CandidateRejectArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "candidate-reject")?;
    Ok(CandidateRejectArgs {
        store,
        request: candidate_reject_request_from_json(&request_json)?,
    })
}

fn run_candidate_reject(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::CandidateReject;
    let result = parse_candidate_reject_args(args)
        .and_then(|options| {
            reject_candidate(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &candidate_reject_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

struct RecallLogArgs {
    store: PathBuf,
    request: RecallLogRequest,
}

fn parse_recall_log_args(args: &[String]) -> Result<RecallLogArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "recall-log")?;
    Ok(RecallLogArgs {
        store,
        request: recall_log_request_from_json(&request_json)?,
    })
}

fn run_history(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::History;
    let result = parse_history_args(args)
        .and_then(|options| {
            memory_history(&options.store, &options.id, options.options)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &history_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_export(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Export;
    let result = parse_export_args(args)
        .and_then(|options| {
            export_store(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                report.schema_version,
                &export_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_import(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Import;
    let result = parse_import_args(args)
        .and_then(|options| {
            import_store(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                report.schema_version,
                &import_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_dream(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Dream;
    let result = parse_dream_args(args)
        .and_then(|options| {
            dream_store(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &dream_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_backup(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Backup;
    let result = parse_backup_args(args)
        .and_then(|options| {
            backup_store(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                report.schema_version,
                &backup_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

fn run_reindex(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Reindex;
    let result = parse_reindex_args(args).and_then(|parsed| {
        let (schema, count) = if parsed.tokens {
            token_backfill_store(&parsed.store, parsed.force)?
        } else if parsed.embed {
            reembed_store(&parsed.store)?
        } else {
            reindex_count(&parsed.store)?
        };
        Ok(success_envelope(
            command,
            &parsed.store,
            schema,
            &format!("{{\"reindexed\":{count}}}"),
            started,
        ))
    });
    print_result(command, started, result)
}

struct ReindexArgs {
    store: PathBuf,
    embed: bool,
    tokens: bool,
    force: bool,
}

fn parse_reindex_args(args: &[String]) -> Result<ReindexArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut embed = false;
    let mut tokens = false;
    let mut force = false;
    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            "--embed" => embed = true,
            "--tokens" => tokens = true,
            "--force" => force = true,
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported reindex flag: {unknown}"
                )));
            }
        }
    }
    let store = store.unwrap_or_else(resolve_store_default);
    Ok(ReindexArgs {
        store,
        embed,
        tokens,
        force,
    })
}

#[cfg(feature = "embed")]
fn reindex_count(store: &Path) -> Result<(i32, usize), CliError> {
    let count = memkeeper_store::reindex_vectors(store).map_err(CliError::from)?;
    Ok((SCHEMA_VERSION, count))
}

#[cfg(not(feature = "embed"))]
#[allow(clippy::unnecessary_wraps)]
fn reindex_count(_store: &Path) -> Result<(i32, usize), CliError> {
    Err(CliError::InvalidRequest(
        "reindex requires a build with the 'semantic' or 'api' feature".to_string(),
    ))
}

#[cfg(feature = "embed")]
fn reembed_store(store: &Path) -> Result<(i32, usize), CliError> {
    let mut embedder = memkeeper_embed::embedder_from_env().ok_or_else(|| {
        CliError::InvalidRequest(
            "no embedding provider configured; set MEMKEEPER_EMBED_PROVIDER and model env"
                .to_string(),
        )
    })?;
    let targets = memkeeper_store::collect_reembed_targets(store).map_err(CliError::from)?;
    let model_id = embedder.model_id().to_string();
    let dims = embedder.dims();
    let mut vectors: Vec<(String, String, Vec<f32>)> = Vec::with_capacity(targets.len());
    for chunk in targets.chunks(32) {
        let texts: Vec<&str> = chunk.iter().map(|target| target.content.as_str()).collect();
        let embeddings = embedder
            .embed(&texts)
            .map_err(|error| CliError::InvalidRequest(format!("re-embed failed: {error}")))?;
        for (target, embedding) in chunk.iter().zip(embeddings) {
            vectors.push((
                target.memory_id.clone(),
                target.version_id.clone(),
                embedding,
            ));
        }
    }
    let count =
        memkeeper_store::apply_reembed(store, &model_id, dims, &vectors).map_err(CliError::from)?;
    Ok((SCHEMA_VERSION, count))
}

#[cfg(not(feature = "embed"))]
#[allow(clippy::unnecessary_wraps)]
fn reembed_store(_store: &Path) -> Result<(i32, usize), CliError> {
    Err(CliError::InvalidRequest(
        "reindex --embed requires a build with the 'semantic' or 'api' feature".to_string(),
    ))
}

#[cfg(feature = "embed")]
fn token_backfill_store(store: &Path, force: bool) -> Result<(i32, usize), CliError> {
    let mut model = memkeeper_embed::colbert_from_env().ok_or_else(|| {
        CliError::InvalidRequest(
            "no colbert model configured; set MEMKEEPER_COLBERT_MODEL_DIR (and build with the 'semantic' feature)"
                .to_string(),
        )
    })?;
    let targets =
        memkeeper_store::collect_token_backfill_targets(store, force).map_err(CliError::from)?;
    let model_id = model.model_id().to_string();
    let total = targets.len();
    let mut rows: Vec<(String, Vec<Vec<f32>>)> = Vec::with_capacity(total);
    for chunk in targets.chunks(32) {
        let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
        let encoded = model
            .encode_docs(&texts)
            .map_err(|error| CliError::InvalidRequest(format!("token encode failed: {error}")))?;
        for ((memory_id, _), vecs) in chunk.iter().zip(encoded) {
            rows.push((memory_id.clone(), vecs));
        }
        eprintln!("[memkeeper] token backfill encoded {}/{total}", rows.len());
    }
    let count = memkeeper_store::apply_token_embeddings(store, &model_id, &rows, force)
        .map_err(CliError::from)?;
    Ok((SCHEMA_VERSION, count))
}

#[cfg(not(feature = "embed"))]
#[allow(clippy::unnecessary_wraps)]
fn token_backfill_store(_store: &Path, _force: bool) -> Result<(i32, usize), CliError> {
    Err(CliError::InvalidRequest(
        "reindex --tokens requires a build with the 'semantic' feature".to_string(),
    ))
}

fn print_result(command: Command, started: Instant, result: Result<String, CliError>) -> i32 {
    match result {
        Ok(envelope) => {
            println!("{envelope}");
            0
        }
        Err(error) => {
            println!("{}", failure_envelope(command, &error, started));
            error.exit_code()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StoreArgs {
    store: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DoctorArgs {
    store: PathBuf,
    store_source: String,
    include_indexes: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StatsArgs {
    store: PathBuf,
    include_indexes: bool,
    include_health: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpaceListArgs {
    store: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SpaceCreateArgs {
    store: PathBuf,
    request: SpaceCreateRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SiloListArgs {
    store: PathBuf,
    request: SiloListRequest,
}

#[derive(Debug, Clone, PartialEq)]
struct RememberArgs {
    store: PathBuf,
    request: RememberRequest,
}

#[derive(Debug, Clone, PartialEq)]
struct SearchArgs {
    store: PathBuf,
    request: SearchRequest,
    /// Native cross-encoder rerank of the result pool (opt-in).
    rerank: bool,
    /// Candidate pool width when reranking.
    rerank_candidates: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct EntityUpsertArgs {
    store: PathBuf,
    request: EntityUpsertRequest,
}

#[derive(Debug, Clone, PartialEq)]
struct RelationshipUpsertArgs {
    store: PathBuf,
    request: RelationshipUpsertRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EntityMergeArgs {
    store: PathBuf,
    request: EntityMergeRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EntitySearchArgs {
    store: PathBuf,
    request: EntitySearchRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphNeighborsArgs {
    store: PathBuf,
    request: GraphNeighborsRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphContextArgs {
    store: PathBuf,
    request: GraphContextRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MemoryListArgs {
    store: PathBuf,
    request: MemoryListRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BatchSearchArgs {
    store: PathBuf,
    request: BatchSearchRequest,
}

#[derive(Debug, Clone, PartialEq)]
struct PackArgs {
    store: PathBuf,
    request: PackRequest,
    /// Query-level cosine OR-gate (0.0 = legacy per-item rerank floor).
    cosine_gate: f64,
    /// Optional deterministic expansion for pack retrieval.
    expansion: PackExpansionOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GetArgs {
    store: PathBuf,
    id: String,
    options: GetOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ForgetArgs {
    store: PathBuf,
    request: ForgetRequest,
}

#[derive(Debug, Clone)]
struct VerifyArgs {
    store: PathBuf,
    request: VerifyRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct HistoryArgs {
    store: PathBuf,
    id: String,
    options: HistoryOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExportArgs {
    store: PathBuf,
    request: ExportRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ImportArgs {
    store: PathBuf,
    request: ImportRequest,
}

#[derive(Debug, Clone, PartialEq)]
struct DreamArgs {
    store: PathBuf,
    request: DreamRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BackupArgs {
    store: PathBuf,
    request: BackupRequest,
}

#[derive(Debug)]
enum CliError {
    InvalidRequest(String),
    /// A required semantic dependency failed at request time (e.g. the embedder
    /// errored mid-request) while `MEMKEEPER_REQUIRE_SEMANTIC` is set, so the
    /// request is failed closed instead of silently degrading to FTS.
    SemanticUnavailable(String),
    Store(StoreError),
}

impl CliError {
    fn code(&self) -> ErrorCode {
        match self {
            Self::SemanticUnavailable(_) => ErrorCode::SemanticUnavailable,
            Self::InvalidRequest(_)
            | Self::Store(StoreError::InvalidRequest { .. } | StoreError::InvalidPath { .. }) => {
                ErrorCode::InvalidRequest
            }
            Self::Store(StoreError::NotInitialized { .. }) => ErrorCode::StoreNotInitialized,
            Self::Store(StoreError::NotFound { .. }) => ErrorCode::NotFound,
            Self::Store(
                StoreError::UnsafeExistingDatabase { .. }
                | StoreError::WalUnavailable { .. }
                | StoreError::Conflict { .. },
            ) => ErrorCode::Conflict,
            Self::Store(StoreError::SchemaMismatch { .. }) => ErrorCode::SchemaMismatch,
            Self::Store(StoreError::Io(_)) => ErrorCode::IoError,
            Self::Store(error) if error.is_locked() => ErrorCode::Locked,
            Self::Store(StoreError::Database(_)) => ErrorCode::InternalError,
        }
    }

    fn exit_code(&self) -> i32 {
        if self.code() == ErrorCode::InvalidRequest {
            2
        } else {
            1
        }
    }

    fn retryable(&self) -> bool {
        match self {
            // Embedder/reranker runtime failures are usually transient
            // (rate limit, provider blip), so callers may retry.
            Self::SemanticUnavailable(_) => true,
            Self::Store(error) => error.is_retryable(),
            Self::InvalidRequest(_) => false,
        }
    }

    fn hint(&self) -> &'static str {
        match self.code() {
            ErrorCode::InvalidRequest => "Run `memkeeper --help` for supported flags.",
            ErrorCode::StoreNotInitialized => "Run `memkeeper init --store <path> --json` first.",
            ErrorCode::SchemaMismatch => "Use a compatible memkeeper binary or migrate the store.",
            ErrorCode::Conflict => {
                "Choose an empty/new path or an existing initialized memkeeper store."
            }
            ErrorCode::Locked => "Retry after the other writer releases the store lock.",
            ErrorCode::IoError => "Check the store path and filesystem permissions.",
            ErrorCode::InternalError => {
                "Re-run with the same store after checking the schema and SQLite build."
            }
            ErrorCode::NotFound => "Inspect the request and current store state.",
            ErrorCode::SemanticUnavailable => {
                "Semantic retrieval is required (MEMKEEPER_REQUIRE_SEMANTIC) but the embedder \
                 failed at request time. Check the embedding provider/key/rate limits, or unset \
                 MEMKEEPER_REQUIRE_SEMANTIC to allow degraded FTS fallback."
            }
        }
    }

    fn details_json(&self) -> String {
        match self {
            Self::SemanticUnavailable(_)
            | Self::InvalidRequest(_)
            | Self::Store(
                StoreError::InvalidRequest { .. }
                | StoreError::Conflict { .. }
                | StoreError::Io(_)
                | StoreError::Database(_),
            ) => "{}".to_string(),
            Self::Store(
                StoreError::NotInitialized { path }
                | StoreError::InvalidPath { path, .. }
                | StoreError::UnsafeExistingDatabase { path },
            ) => format!("{{\"path\":{}}}", json_path(path)),
            Self::Store(StoreError::NotFound { entity, id }) => format!(
                "{{\"entity\":{},\"id\":{}}}",
                json_string(entity),
                json_string(id)
            ),
            Self::Store(StoreError::SchemaMismatch { expected, actual }) => {
                format!("{{\"expected\":{expected},\"actual\":{actual}}}")
            }
            Self::Store(StoreError::WalUnavailable { path, journal_mode }) => format!(
                "{{\"path\":{},\"journal_mode\":{}}}",
                json_path(path),
                json_string(journal_mode)
            ),
        }
    }
}

impl From<StoreError> for CliError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

impl std::fmt::Display for CliError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(message) | Self::SemanticUnavailable(message) => {
                formatter.write_str(message)
            }
            Self::Store(error) => write!(formatter, "{error}"),
        }
    }
}

/// Resolve the store path when no explicit `--store` flag was supplied.
///
/// Precedence is `--store` flag (handled by callers) > `MEMKEEPER_STORE` env >
/// the default `~/.memkeeper/store.sqlite`. When `$HOME` is unset (minimal CI
/// containers, some service managers) the default falls back to the
/// project-relative `.memkeeper/store.sqlite` so the CLI still resolves to a
/// deterministic location instead of an empty path.
fn resolve_store_default() -> PathBuf {
    if let Ok(env_store) = std::env::var("MEMKEEPER_STORE") {
        let trimmed = env_store.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed.strip_prefix('@').unwrap_or(trimmed));
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        if !home.is_empty() {
            return PathBuf::from(home).join(".memkeeper").join("store.sqlite");
        }
    }
    PathBuf::from(PROJECT_STORE_RELATIVE_PATH)
}

fn parse_store_args(args: &[String]) -> Result<StoreArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--json" => {}
            "--store" => store = Some(parser.required_value("--store")?),
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported init flag: {unknown}"
                )));
            }
        }
    }

    Ok(StoreArgs {
        store: store.unwrap_or_else(resolve_store_default),
    })
}

fn parse_doctor_args(args: &[String]) -> Result<DoctorArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut include_indexes = false;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--json" => {}
            "--store" => store = Some(parser.required_value("--store")?),
            "--include-indexes" => include_indexes = true,
            "--no-indexes" => include_indexes = false,
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--include-indexes=") => {
                include_indexes = parse_bool(value.trim_start_matches("--include-indexes="))?;
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported doctor flag: {unknown}"
                )));
            }
        }
    }

    let (store, store_source) = match store {
        Some(store) => (store, "flag".to_string()),
        None => diagnostic_store_candidate(),
    };

    Ok(DoctorArgs {
        store,
        store_source,
        include_indexes,
    })
}

fn parse_stats_args(args: &[String]) -> Result<StatsArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut include_indexes = true;
    let mut include_health = false;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--json" => {}
            "--store" => store = Some(parser.required_value("--store")?),
            "--include-indexes" => include_indexes = true,
            "--no-indexes" => include_indexes = false,
            "--health" => include_health = true,
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--include-indexes=") => {
                include_indexes = parse_bool(value.trim_start_matches("--include-indexes="))?;
            }
            value if value.starts_with("--health=") => {
                include_health = parse_bool(value.trim_start_matches("--health="))?;
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported stats flag: {unknown}"
                )));
            }
        }
    }

    Ok(StatsArgs {
        store: store.unwrap_or_else(resolve_store_default),
        include_indexes,
        include_health,
    })
}

fn parse_space_list_args(args: &[String]) -> Result<SpaceListArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--json" => {}
            "--store" => store = Some(parser.required_value("--store")?),
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported space-list flag: {unknown}"
                )));
            }
        }
    }

    Ok(SpaceListArgs {
        store: store.unwrap_or_else(resolve_store_default),
    })
}

fn parse_space_create_args(args: &[String]) -> Result<SpaceCreateArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut request_json = None;
    let mut name = None;
    let mut display_name = None;
    let mut description = None;
    let mut default_silo = None;
    let mut ontology = None;
    let mut if_not_exists = false;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--json" | "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            "--name" => name = Some(parser.required_string("--name")?),
            "--display-name" => display_name = Some(parser.required_string("--display-name")?),
            "--description" => description = Some(parser.required_string("--description")?),
            "--default-silo" => default_silo = Some(parser.required_string("--default-silo")?),
            "--ontology" => ontology = Some(parser.required_string("--ontology")?),
            "--if-not-exists" => if_not_exists = true,
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            value if value.starts_with("--request-json=") => {
                request_json = Some(value.trim_start_matches("--request-json=").to_string());
            }
            value if value.starts_with("--name=") => {
                name = Some(value.trim_start_matches("--name=").to_string());
            }
            value if value.starts_with("--display-name=") => {
                display_name = Some(value.trim_start_matches("--display-name=").to_string());
            }
            value if value.starts_with("--description=") => {
                description = Some(value.trim_start_matches("--description=").to_string());
            }
            value if value.starts_with("--default-silo=") => {
                default_silo = Some(value.trim_start_matches("--default-silo=").to_string());
            }
            value if value.starts_with("--ontology=") => {
                ontology = Some(value.trim_start_matches("--ontology=").to_string());
            }
            value if value.starts_with("--if-not-exists=") => {
                if_not_exists = parse_bool(value.trim_start_matches("--if-not-exists="))?;
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported space-create flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        space_create_request_from_json(&request_json)?
    } else {
        SpaceCreateRequest {
            name: name.ok_or_else(|| {
                CliError::InvalidRequest("missing required --name <space>".to_string())
            })?,
            display_name,
            description,
            default_silo,
            ontology,
            config_json: None,
            if_not_exists,
        }
    };

    Ok(SpaceCreateArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

fn parse_silo_list_args(args: &[String]) -> Result<SiloListArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut request_json = None;
    let mut space = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--json" => {
                if parser
                    .peek()
                    .is_some_and(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(parser.required_string("--json")?);
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            "--store" => store = Some(parser.required_value("--store")?),
            "--space" => space = Some(parser.required_string("--space")?),
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            value if value.starts_with("--request-json=") => {
                request_json = Some(value.trim_start_matches("--request-json=").to_string());
            }
            value if value.starts_with("--space=") => {
                space = Some(value.trim_start_matches("--space=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported silo-list flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        silo_list_request_from_json(&request_json)?
    } else {
        SiloListRequest { space }
    };

    Ok(SiloListArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

fn parse_remember_args(args: &[String]) -> Result<RememberArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "remember")?;
    Ok(RememberArgs {
        store,
        request: remember_request_from_json(&request_json)?,
    })
}

fn parse_search_args(args: &[String]) -> Result<SearchArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "search")?;
    let (rerank, rerank_candidates) = search_rerank_options_from_json(&request_json)?;
    Ok(SearchArgs {
        store,
        request: search_request_from_json(&request_json)?,
        rerank,
        rerank_candidates,
    })
}

fn parse_entity_upsert_args(args: &[String]) -> Result<EntityUpsertArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "entity-upsert")?;
    Ok(EntityUpsertArgs {
        store,
        request: entity_upsert_request_from_json(&request_json)?,
    })
}

fn parse_relationship_upsert_args(args: &[String]) -> Result<RelationshipUpsertArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "relationship-upsert")?;
    Ok(RelationshipUpsertArgs {
        store,
        request: relationship_upsert_request_from_json(&request_json)?,
    })
}

fn parse_entity_merge_args(args: &[String]) -> Result<EntityMergeArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "entity-merge")?;
    Ok(EntityMergeArgs {
        store,
        request: entity_merge_request_from_json(&request_json)?,
    })
}

fn parse_entity_search_args(args: &[String]) -> Result<EntitySearchArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "entity-search")?;
    Ok(EntitySearchArgs {
        store,
        request: entity_search_request_from_json(&request_json)?,
    })
}

fn parse_graph_neighbors_args(args: &[String]) -> Result<GraphNeighborsArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "graph-neighbors")?;
    Ok(GraphNeighborsArgs {
        store,
        request: graph_neighbors_request_from_json(&request_json)?,
    })
}

fn parse_graph_context_args(args: &[String]) -> Result<GraphContextArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "graph-context")?;
    Ok(GraphContextArgs {
        store,
        request: graph_context_request_from_json(&request_json)?,
    })
}

fn parse_json_command_args(
    args: &[String],
    command_name: &str,
) -> Result<(PathBuf, String), CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--json" | "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            value if value.starts_with("--request-json=") => {
                request_json = Some(value.trim_start_matches("--request-json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported {command_name} flag: {unknown}"
                )));
            }
        }
    }

    let request_json = request_json.ok_or_else(|| {
        CliError::InvalidRequest(format!("missing {command_name} request JSON after --json"))
    })?;
    let store = store.unwrap_or_else(resolve_store_default);
    Ok((store, request_json))
}

fn parse_memory_list_args(args: &[String]) -> Result<MemoryListArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut request_json = None;
    let mut limit = 20;
    let mut offset = 0;
    let mut snippet_chars = 240;
    let mut include_content = false;
    let mut include_source = false;
    let mut order = "updated_desc".to_string();

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            "--limit" => limit = parse_usize_arg("--limit", &parser.required_string("--limit")?)?,
            "--offset" => {
                offset = parse_usize_arg("--offset", &parser.required_string("--offset")?)?;
            }
            "--snippet-chars" => {
                snippet_chars = parse_usize_arg(
                    "--snippet-chars",
                    &parser.required_string("--snippet-chars")?,
                )?;
            }
            "--include-content" => include_content = true,
            "--no-content" => include_content = false,
            "--include-source" => include_source = true,
            "--no-source" => include_source = false,
            "--order" => order = parser.required_string("--order")?,
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            value if value.starts_with("--request-json=") => {
                request_json = Some(value.trim_start_matches("--request-json=").to_string());
            }
            value if value.starts_with("--limit=") => {
                limit = parse_usize_arg("--limit", value.trim_start_matches("--limit="))?;
            }
            value if value.starts_with("--offset=") => {
                offset = parse_usize_arg("--offset", value.trim_start_matches("--offset="))?;
            }
            value if value.starts_with("--snippet-chars=") => {
                snippet_chars = parse_usize_arg(
                    "--snippet-chars",
                    value.trim_start_matches("--snippet-chars="),
                )?;
            }
            value if value.starts_with("--include-content=") => {
                include_content = parse_bool(value.trim_start_matches("--include-content="))?;
            }
            value if value.starts_with("--include-source=") => {
                include_source = parse_bool(value.trim_start_matches("--include-source="))?;
            }
            value if value.starts_with("--order=") => {
                order = value.trim_start_matches("--order=").to_string();
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported memory-list flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        memory_list_request_from_json(&request_json)?
    } else {
        MemoryListRequest {
            filters: SearchFilters::default(),
            limit,
            offset,
            snippet_chars,
            include_content,
            include_source,
            order,
        }
    };

    Ok(MemoryListArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

fn parse_batch_search_args(args: &[String]) -> Result<BatchSearchArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "batch-search")?;
    Ok(BatchSearchArgs {
        store,
        request: batch_search_request_from_json(&request_json)?,
    })
}

fn parse_pack_args(args: &[String]) -> Result<PackArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "pack")?;
    Ok(PackArgs {
        store,
        request: pack_request_from_json(&request_json)?,
        cosine_gate: pack_cosine_gate_from_json(&request_json)?,
        expansion: pack_expansion_options_from_json(&request_json)?,
    })
}

fn parse_get_args(args: &[String]) -> Result<GetArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut id = None;
    let mut request_json = None;
    let mut include_history = false;
    let mut include_links = true;
    let mut include_source = false;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--id" => id = Some(parser.required_string("--id")?),
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            "--include-history" => include_history = true,
            "--no-history" => include_history = false,
            "--include-links" => include_links = true,
            "--no-links" => include_links = false,
            "--include-source" => include_source = true,
            "--no-source" => include_source = false,
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--id=") => {
                id = Some(value.trim_start_matches("--id=").to_string());
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported get flag: {unknown}"
                )));
            }
        }
    }

    if let Some(request_json) = request_json {
        let request = get_request_from_json(&request_json)?;
        id = Some(request.id);
        include_history = request.options.include_history;
        include_links = request.options.include_links;
        include_source = request.options.include_source;
    }

    Ok(GetArgs {
        store: store.unwrap_or_else(resolve_store_default),
        id: id.ok_or_else(|| CliError::InvalidRequest("missing required --id <id>".to_string()))?,
        options: GetOptions {
            include_history,
            include_links,
            include_source,
        },
    })
}

fn parse_forget_args(args: &[String]) -> Result<ForgetArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut id = None;
    let mut reason = None;
    let mut mode = "tombstone".to_string();
    let mut corrected_by = None;
    let mut dry_run = false;
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--id" => id = Some(parser.required_string("--id")?),
            "--reason" => reason = Some(parser.required_string("--reason")?),
            "--mode" => mode = parser.required_string("--mode")?,
            "--corrected-by" => corrected_by = Some(parser.required_string("--corrected-by")?),
            "--dry-run" => dry_run = true,
            "--commit" | "--apply" => dry_run = false,
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--id=") => {
                id = Some(value.trim_start_matches("--id=").to_string());
            }
            value if value.starts_with("--reason=") => {
                reason = Some(value.trim_start_matches("--reason=").to_string());
            }
            value if value.starts_with("--mode=") => {
                mode = value.trim_start_matches("--mode=").to_string();
            }
            value if value.starts_with("--corrected-by=") => {
                corrected_by = Some(value.trim_start_matches("--corrected-by=").to_string());
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported forget flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        forget_request_from_json(&request_json)?
    } else {
        ForgetRequest {
            id: id.ok_or_else(|| {
                CliError::InvalidRequest("missing required --id <id>".to_string())
            })?,
            reason,
            mode,
            corrected_by,
            dry_run,
        }
    };

    Ok(ForgetArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

fn parse_verify_args(args: &[String]) -> Result<VerifyArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut memory_id = None;
    let mut verified_against = None;
    let mut now = None;
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--id" => memory_id = Some(parser.required_string("--id")?),
            "--verified-against" => {
                verified_against = Some(parser.required_string("--verified-against")?);
            }
            "--now" => now = Some(parser.required_string("--now")?),
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--id=") => {
                memory_id = Some(value.trim_start_matches("--id=").to_string());
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported verify flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        verify_request_from_json(&request_json)?
    } else {
        VerifyRequest {
            memory_id: memory_id.ok_or_else(|| {
                CliError::InvalidRequest("missing required --id <id>".to_string())
            })?,
            verified_against,
            now,
        }
    };

    Ok(VerifyArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

fn parse_history_args(args: &[String]) -> Result<HistoryArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut id = None;
    let mut limit = 50;
    let mut include_source = false;
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--id" => id = Some(parser.required_string("--id")?),
            "--limit" => limit = parse_usize_arg("--limit", &parser.required_string("--limit")?)?,
            "--include-source" => include_source = true,
            "--no-source" => include_source = false,
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--id=") => {
                id = Some(value.trim_start_matches("--id=").to_string());
            }
            value if value.starts_with("--limit=") => {
                limit = parse_usize_arg("--limit", value.trim_start_matches("--limit="))?;
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported history flag: {unknown}"
                )));
            }
        }
    }

    if let Some(request_json) = request_json {
        let request = history_request_from_json(&request_json)?;
        id = Some(request.id);
        limit = request.options.limit;
        include_source = request.options.include_source;
    }

    Ok(HistoryArgs {
        store: store.unwrap_or_else(resolve_store_default),
        id: id.ok_or_else(|| CliError::InvalidRequest("missing required --id <id>".to_string()))?,
        options: HistoryOptions {
            limit,
            include_source,
        },
    })
}

fn parse_export_args(args: &[String]) -> Result<ExportArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut output = None;
    let mut format = "jsonl".to_string();
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--output" | "--out" => output = Some(parser.required_value(arg.as_str())?),
            "--format" => format = parser.required_string("--format")?,
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--output=") => {
                output = Some(PathBuf::from(value.trim_start_matches("--output=")));
            }
            value if value.starts_with("--out=") => {
                output = Some(PathBuf::from(value.trim_start_matches("--out=")));
            }
            value if value.starts_with("--format=") => {
                format = value.trim_start_matches("--format=").to_string();
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported export flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        export_request_from_json(&request_json)?
    } else {
        ExportRequest {
            output_path: output.ok_or_else(|| {
                CliError::InvalidRequest("missing required --output <path>".to_string())
            })?,
            format,
        }
    };

    Ok(ExportArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

fn parse_import_args(args: &[String]) -> Result<ImportArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut input = None;
    let mut format = "jsonl".to_string();
    let mut dry_run = false;
    let mut conflict_policy = "fail_if_exists".to_string();
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--input" | "--in" => input = Some(parser.required_value(arg.as_str())?),
            "--format" => format = parser.required_string("--format")?,
            "--dry-run" => dry_run = true,
            "--commit" | "--apply" => dry_run = false,
            "--conflict-policy" => conflict_policy = parser.required_string("--conflict-policy")?,
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--input=") => {
                input = Some(PathBuf::from(value.trim_start_matches("--input=")));
            }
            value if value.starts_with("--in=") => {
                input = Some(PathBuf::from(value.trim_start_matches("--in=")));
            }
            value if value.starts_with("--format=") => {
                format = value.trim_start_matches("--format=").to_string();
            }
            value if value.starts_with("--dry-run=") => {
                dry_run = parse_bool(value.trim_start_matches("--dry-run="))?;
            }
            value if value.starts_with("--conflict-policy=") => {
                conflict_policy = value.trim_start_matches("--conflict-policy=").to_string();
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported import flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        import_request_from_json(&request_json)?
    } else {
        ImportRequest {
            input_path: input.ok_or_else(|| {
                CliError::InvalidRequest("missing required --input <path>".to_string())
            })?,
            format,
            dry_run,
            conflict_policy,
        }
    };

    Ok(ImportArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

#[allow(clippy::too_many_lines)]
fn parse_dream_args(args: &[String]) -> Result<DreamArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut space = None;
    let mut silos = Vec::new();
    let mut tasks = Vec::new();
    // Track flags as Option so explicitly-passed values can override a --json payload.
    // (None means "not supplied on the command line".)
    let mut max_memories = None;
    let mut dry_run = None;
    let mut include_pinned = None;
    let mut promote_threshold = None;
    let mut promote_score_floor = None;
    let mut promote_rank_cap = None;
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--space" => space = Some(parser.required_string("--space")?),
            "--silo" => append_csv_values(&mut silos, &parser.required_string("--silo")?),
            "--task" => append_csv_values(&mut tasks, &parser.required_string("--task")?),
            "--tasks" => append_csv_values(&mut tasks, &parser.required_string("--tasks")?),
            "--max-memories" => {
                max_memories = Some(parse_usize_arg(
                    "--max-memories",
                    &parser.required_string("--max-memories")?,
                )?);
            }
            "--promote-threshold" => {
                promote_threshold = Some(parse_usize_arg(
                    "--promote-threshold",
                    &parser.required_string("--promote-threshold")?,
                )?);
            }
            "--promote-score-floor" => {
                promote_score_floor = Some(parse_f64_arg(
                    "--promote-score-floor",
                    &parser.required_string("--promote-score-floor")?,
                )?);
            }
            "--promote-rank-cap" => {
                promote_rank_cap = Some(parse_usize_arg(
                    "--promote-rank-cap",
                    &parser.required_string("--promote-rank-cap")?,
                )?);
            }
            "--dry-run" => dry_run = Some(true),
            "--commit" | "--apply" => dry_run = Some(false),
            "--include-pinned" => include_pinned = Some(true),
            "--exclude-pinned" => include_pinned = Some(false),
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--space=") => {
                space = Some(value.trim_start_matches("--space=").to_string());
            }
            value if value.starts_with("--silo=") => {
                append_csv_values(&mut silos, value.trim_start_matches("--silo="));
            }
            value if value.starts_with("--task=") => {
                append_csv_values(&mut tasks, value.trim_start_matches("--task="));
            }
            value if value.starts_with("--tasks=") => {
                append_csv_values(&mut tasks, value.trim_start_matches("--tasks="));
            }
            value if value.starts_with("--max-memories=") => {
                max_memories = Some(parse_usize_arg(
                    "--max-memories",
                    value.trim_start_matches("--max-memories="),
                )?);
            }
            value if value.starts_with("--promote-threshold=") => {
                promote_threshold = Some(parse_usize_arg(
                    "--promote-threshold",
                    value.trim_start_matches("--promote-threshold="),
                )?);
            }
            value if value.starts_with("--promote-score-floor=") => {
                promote_score_floor = Some(parse_f64_arg(
                    "--promote-score-floor",
                    value.trim_start_matches("--promote-score-floor="),
                )?);
            }
            value if value.starts_with("--promote-rank-cap=") => {
                promote_rank_cap = Some(parse_usize_arg(
                    "--promote-rank-cap",
                    value.trim_start_matches("--promote-rank-cap="),
                )?);
            }
            value if value.starts_with("--dry-run=") => {
                dry_run = Some(parse_bool(value.trim_start_matches("--dry-run="))?);
            }
            value if value.starts_with("--include-pinned=") => {
                include_pinned = Some(parse_bool(value.trim_start_matches("--include-pinned="))?);
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported dream flag: {unknown}"
                )));
            }
        }
    }

    // Start from the --json payload when supplied, otherwise from defaults.
    let mut request = if let Some(request_json) = request_json {
        dream_request_from_json(&request_json)?
    } else {
        DreamRequest {
            space: None,
            silos: Vec::new(),
            tasks: Vec::new(),
            max_memories: DEFAULT_DREAM_MAX_MEMORIES,
            dry_run: false,
            include_pinned: false,
            promote_threshold: DEFAULT_PROMOTE_THRESHOLD,
            promote_score_floor: DEFAULT_PROMOTE_SCORE_FLOOR,
            promote_rank_cap: DEFAULT_PROMOTE_RANK_CAP,
        }
    };

    // Explicit CLI flags are the user's direct intent and override the payload.
    // This keeps dry-run trust intact: `--dry-run` always wins, even alongside `--json`.
    if space.is_some() {
        request.space = space;
    }
    if !silos.is_empty() {
        request.silos = silos;
    }
    if !tasks.is_empty() {
        request.tasks = tasks;
    }
    if let Some(value) = max_memories {
        request.max_memories = value;
    }
    if let Some(value) = dry_run {
        request.dry_run = value;
    }
    if let Some(value) = include_pinned {
        request.include_pinned = value;
    }
    if let Some(value) = promote_threshold {
        request.promote_threshold = value;
    }
    if let Some(value) = promote_score_floor {
        request.promote_score_floor = value;
    }
    if let Some(value) = promote_rank_cap {
        request.promote_rank_cap = value;
    }

    Ok(DreamArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

fn parse_backup_args(args: &[String]) -> Result<BackupArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut output = None;
    let mut format = "sqlite".to_string();
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--output" | "--out" => output = Some(parser.required_value(arg.as_str())?),
            "--format" => format = parser.required_string("--format")?,
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--output=") => {
                output = Some(PathBuf::from(value.trim_start_matches("--output=")));
            }
            value if value.starts_with("--out=") => {
                output = Some(PathBuf::from(value.trim_start_matches("--out=")));
            }
            value if value.starts_with("--format=") => {
                format = value.trim_start_matches("--format=").to_string();
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported backup flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        backup_request_from_json(&request_json)?
    } else {
        BackupRequest {
            output_path: output.ok_or_else(|| {
                CliError::InvalidRequest("missing required --output <path>".to_string())
            })?,
            format,
        }
    };

    Ok(BackupArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

struct ArgParser<'a> {
    args: &'a [String],
    index: usize,
}

impl<'a> ArgParser<'a> {
    const fn new(args: &'a [String]) -> Self {
        Self { args, index: 0 }
    }

    fn next(&mut self) -> Option<String> {
        let value = self.args.get(self.index)?.clone();
        self.index += 1;
        Some(value)
    }

    fn peek(&self) -> Option<&str> {
        self.args.get(self.index).map(String::as_str)
    }

    fn required_value(&mut self, flag: &str) -> Result<PathBuf, CliError> {
        let value = self
            .next()
            .ok_or_else(|| CliError::InvalidRequest(format!("missing value for {flag}")))?;
        if value.starts_with('-') {
            return Err(CliError::InvalidRequest(format!(
                "missing value for {flag}"
            )));
        }
        Ok(PathBuf::from(value))
    }

    fn required_string(&mut self, flag: &str) -> Result<String, CliError> {
        self.next()
            .ok_or_else(|| CliError::InvalidRequest(format!("missing value for {flag}")))
    }
}

fn parse_bool(value: &str) -> Result<bool, CliError> {
    match value {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(CliError::InvalidRequest(format!(
            "invalid boolean value: {value}"
        ))),
    }
}

fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, CliError> {
    value.parse::<usize>().map_err(|_| {
        CliError::InvalidRequest(format!("invalid non-negative integer for {flag}: {value}"))
    })
}

fn parse_f64_arg(flag: &str, value: &str) -> Result<f64, CliError> {
    value
        .parse::<f64>()
        .map_err(|_| CliError::InvalidRequest(format!("invalid number for {flag}: {value}")))
}

fn append_csv_values(target: &mut Vec<String>, value: &str) {
    target.extend(value.split(',').map(str::to_string));
}

// ---------------------------------------------------------------------------
// hook subcommands — thin clients for Claude Code hook integration
// ---------------------------------------------------------------------------

/// Download the ONNX models the semantic path needs (embedder + reranker) via
/// `curl`, mirroring `scripts/fetch-models.sh` so a `cargo install`-ed binary can
/// fetch models without the repo checked out. Writes `model.onnx` +
/// `tokenizer.json` into `<dir>/<model>` (subdir names match the adapter's
/// documented defaults, so the bridge finds them with no extra config) and
/// prints the env vars to point the daemon at them.
fn run_pull_models(args: &[String]) -> i32 {
    let mut quantized = false;
    let mut dir: Option<PathBuf> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--quantized" => quantized = true,
            "--dir" => {
                let Some(value) = iter.next() else {
                    eprintln!("pull-models: --dir needs a path");
                    return 2;
                };
                dir = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                println!(
                    "Usage: memkeeper pull-models [--quantized] [--dir DIR]\n\n  \
                     --quantized  fetch smaller INT8 models (~0.6GB) instead of fp32 (~2.1GB);\n               \
                     recall drifts from the fp32 baseline, so prefer fp32 for parity.\n  \
                     --dir DIR    install root (default: $MEMKEEPER_MODELS_DIR or ~/.memkeeper/models)"
                );
                return 0;
            }
            other => {
                eprintln!("pull-models: unknown argument: {other}");
                return 2;
            }
        }
    }

    // curl is the one external dependency, same as scripts/fetch-models.sh.
    let curl_ok = process::Command::new("curl")
        .arg("--version")
        .stdout(process::Stdio::null())
        .stderr(process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !curl_ok {
        eprintln!("pull-models: curl is required but was not found on PATH");
        return 1;
    }

    let dir = dir.unwrap_or_else(|| {
        env::var_os("MEMKEEPER_MODELS_DIR").map_or_else(
            || {
                let home = env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
                home.join(".memkeeper").join("models")
            },
            PathBuf::from,
        )
    });

    let onnx = if quantized {
        "model_quantized.onnx"
    } else {
        "model.onnx"
    };
    let hf = "https://huggingface.co";
    // (HF repo, local subdir). The HF repos carry the `-v1` suffix; the local
    // subdirs intentionally omit it to match MEMKEEPER_EMBED_MODEL_DIR/
    // MEMKEEPER_RERANK_MODEL_DIR's documented defaults.
    let models = [
        ("mixedbread-ai/mxbai-embed-large-v1", "mxbai-embed-large"),
        ("mixedbread-ai/mxbai-rerank-base-v1", "mxbai-rerank-base"),
    ];

    for (repo, subdir) in models {
        let dest = dir.join(subdir);
        if let Err(error) = fs::create_dir_all(&dest) {
            eprintln!("pull-models: cannot create {}: {error}", dest.display());
            return 1;
        }
        println!("==> {repo}  ->  {}  ({onnx})", dest.display());
        let downloads = [
            (
                format!("{hf}/{repo}/resolve/main/onnx/{onnx}"),
                dest.join("model.onnx"),
            ),
            (
                format!("{hf}/{repo}/resolve/main/tokenizer.json"),
                dest.join("tokenizer.json"),
            ),
        ];
        for (url, out) in downloads {
            if !curl_download(&url, &out) {
                eprintln!("pull-models: FAILED downloading {url}");
                return 1;
            }
        }
    }

    let embed = dir.join("mxbai-embed-large");
    let rerank = dir.join("mxbai-rerank-base");
    println!(
        "\nDone. Models installed under: {}\n\
         Point the daemon at them (add to your shell profile or memkeeper launch env):\n\n  \
         export MEMKEEPER_EMBED_MODEL_DIR=\"{}\"\n  \
         export MEMKEEPER_RERANK_MODEL_DIR=\"{}\"\n\n\
         Then `memkeeper serve` runs with semantics on. To require it (fail closed if\n\
         the models go missing), also set MEMKEEPER_REQUIRE_SEMANTIC=1.",
        dir.display(),
        embed.display(),
        rerank.display(),
    );
    0
}

/// `curl -fL --retry 3 --proto =https --tlsv1.2 -o <out> <url>` — fail loud on
/// any HTTP error, follow CDN redirects, https-only. Returns true on success.
fn curl_download(url: &str, out: &Path) -> bool {
    process::Command::new("curl")
        .args([
            "-fL",
            "--retry",
            "3",
            "--proto",
            "=https",
            "--tlsv1.2",
            "-o",
        ])
        .arg(out)
        .arg(url)
        .status()
        .is_ok_and(|status| status.success())
}

fn print_help() {
    println!(
        "memkeeper {PROTOCOL_VERSION} schema {SCHEMA_VERSION}\n\n\
         Commands:\n\
           init --store <path> --json             Initialize a local store.\n\
           doctor [--store <path>] --json         Diagnose binary/config/store readiness.\n\
           stats --store <path> --json            Show deterministic store stats.\n\
           space-list --store <path> --json       List configured spaces.\n\
           space-create --store <path> --json '{{}}' Create a space and default silos.\n\
           silo-list --store <path> [--space <name>] --json List silos.\n\
           remember --store <path> --json '{{}}'   Store one explicit memory.\n\
           search --store <path> --json '{{}}'     Search memories with FTS5/BM25.\n\
           entity-upsert --store <path> --json '{{}}' Create/update a graph entity.\n\
           relationship-upsert --store <path> --json '{{}}' Create/update a graph relationship.\n\
           entity-merge --store <path> --json '{{}}' Merge a graph entity into another (relink + tombstone).\n\
           entity-search --store <path> --json '{{}}' Search graph entities.\n\
           graph-neighbors --store <path> --json '{{}}' Traverse graph neighbors.\n\
           graph-context --store <path> --json '{{}}' Build graph-centered memory context.\n\
           memory-list --store <path> --json '{{}}' List recent memories for review.\n\
           batch-search --store <path> --json '{{}}' Run multiple searches.\n\
           ingest --store <path> --json '{{}}'     Ingest a document as isolated, embedded chunks (RAG store).\n\
           document-search --store <path> --json '{{}}' Hybrid search over ingested document chunks.\n\
           document-get --store <path> --json '{{}}' Fetch a document's chunks by path or chunk id.\n\
           document-duplicates --store <path> --json '{{}}' List exact-content duplicate chunks (clusters).\n\
           document-prune --store <path> --json '{{}}' Delete chosen document chunks by id (--dry-run via JSON).\n\
           promotion-candidates --store <path> --json '{{}}' Rank document chunks that earned retrieval traffic.\n\
           mark-extracted --store <path> --json '{{}}' Mark document chunks extracted (promoted).\n\
           candidate-submit --store <path> --json '{{}}' Submit a candidate memory for review.\n\
           candidate-list --store <path> --json '{{}}' List candidate memories (filter by status).\n\
           candidate-approve --store <path> --json '{{}}' Approve a candidate, promoting it to a memory.\n\
           candidate-reject --store <path> --json '{{}}' Reject a candidate memory.\n\
           pack --store <path> --json '{{\"title\":<str>,\"queries\":[<str>,...]}}' Build a compact memory pack. Optional: max_memories, max_chars, min_score, filters, format.\n\
           get --store <path> --id <id> --json    Fetch one memory.\n\
           forget --store <path> --id <id> --json Tombstone one memory.\n\
           history --store <path> --id <id> --json Show memory versions/events.\n\
           export --store <path> --output <path> --json Write logical JSONL export.\n\
           import --store <path> --input <path> --json  Import logical JSONL into a new store.\n\
           dream --store <path> [--task promote|expire|reindex|dedupe|graph] [--promote-threshold <N>] [--promote-score-floor <F>] [--promote-rank-cap <N>] [--dry-run|--apply] --json '{{}}' Run bounded maintenance tasks.\n\
           (--dry-run previews; --apply/--commit mutates. forget/import/dream accept both; mutating commands always report dry_run + changed ids.)\n\
           backup --store <path> --output <path> --json Create physical SQLite backup.\n\
           hook retrieve [--store <path>] [--sock <path>]  Claude Code UserPromptSubmit hook client.\n\
           serve --stdio | --socket <path>       Serve newline-delimited JSON requests (stdio or Unix socket).\n\
           mcp [--store <path>]                  Speak MCP (JSON-RPC 2.0) over stdio for any MCP client.\n\
           serve --http [addr]                   Serve the read-only local dashboard (default 127.0.0.1:7777).\n\
           schema [command] [--json]              Show accepted JSON payload fields for a command (or list all).\n\
           schema-status                          Check embedded schema metadata.\n\
           pull-models [--quantized] [--dir <path>] Download the ONNX embed+rerank models (needs curl).\n\
           --help, help                           Show this help.\n\
           --version, version                     Show protocol/schema version.\n\n\
         Per-command request/response shapes: run `memkeeper schema <command>`.\n\
         CLI commands are store-path explicit except read-only doctor diagnostics; hints are user={USER_STORE_PATH_HINT} project={PROJECT_STORE_RELATIVE_PATH}"
    );
}

fn print_schema_status() {
    let schema_mentions_required_objects = schema_mentions_required_objects();
    println!(
        "{{\"protocol_version\":\"{PROTOCOL_VERSION}\",\"schema_version\":{SCHEMA_VERSION},\"user_store_path_hint\":\"{USER_STORE_PATH_HINT}\",\"project_store_relative_path\":\"{PROJECT_STORE_RELATIVE_PATH}\",\"schema_mentions_required_objects\":{schema_mentions_required_objects},\"scaffold_only\":false}}"
    );
}

#[cfg(test)]
mod tests;
