//! `pack` command handlers.

#![allow(clippy::wildcard_imports)]

use std::{
    env,
    fmt::Write as _,
    fs,
    io::{self, BufRead, Write as IoWrite},
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use memkeeper_protocol::Command;
use memkeeper_store::*;
use std::result::Result;
#[cfg(feature = "embed")]
use memkeeper_store::build_hybrid_rerank_pool_trace_with_evidence_options;

use crate::{CliError, hook::*, json::*, output::*, requests::*, serve::*};
use super::{ArgParser, 
    append_csv_values, maybe_colbert_embed_pack_request, maybe_colbert_embed_remember_request,
    maybe_colbert_embed_remember_request_with_requirement, maybe_colbert_embed_search_request,
    maybe_embed_document_search_request, maybe_embed_ingest_request, maybe_embed_pack_request,
    maybe_embed_remember_request,
    maybe_embed_search_request, parse_bool, parse_f64_arg, parse_json_command_args,
    parse_usize_arg, print_result, SemanticModels,
};


pub(crate) fn run_pack(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_pack();
    run_pack_with_models(args, &semantic_models)
}

pub(crate) fn run_pack_with_models(args: &[String], semantic_models: &SemanticModels) -> i32 {
    let started = Instant::now();
    let command = Command::Pack;
    let result = parse_pack_args(args)
        .and_then(|options| {
            execute_pack_request(
                &options.store,
                options.request,
                semantic_models,
                crate::serve::require_rerank_env(),
            )
            .map(|report| (options.store, report))
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

/// Pure fail-closed decision for commands that request cross-encoder reranking.
pub(crate) fn must_refuse_without_reranker(requested: bool, active: bool, required: bool) -> bool {
    requested && required && !active
}

/// Apply `MEMKEEPER_REQUIRE_RERANK` to one-shot/shared request paths. Long-lived
/// servers apply the same posture at startup in `serve.rs`.
pub(crate) fn require_primary_reranker(
    requested: bool,
    semantic_models: &SemanticModels,
) -> Result<(), CliError> {
    if must_refuse_without_reranker(
        requested,
        semantic_models.rerank_active(),
        crate::serve::require_rerank_env(),
    ) {
        eprintln!(
            "[memkeeper] ERROR: MEMKEEPER_REQUIRE_RERANK=1 but the primary reranker is not active; \
             refusing to serve the request (set MEMKEEPER_RERANK_MODEL_DIR to a valid model dir)"
        );
        return Err(CliError::InvalidRequest(
            "primary reranker required but unavailable".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn run_pool_trace(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_remember_or_search();
    run_pool_trace_with_models(args, &semantic_models)
}

pub(crate) fn run_pool_trace_with_models(args: &[String], semantic_models: &SemanticModels) -> i32 {
    let started = Instant::now();
    let command = Command::PoolTrace;
    let result = parse_pool_trace_args(args)
        .and_then(|options| {
            execute_pool_trace(
                &options.store,
                options.request,
                options.expansion,
                semantic_models,
            )
            .map(|result_json| (options.store, result_json))
        })
        .map(|(path, result_json)| {
            success_envelope(command, &path, SCHEMA_VERSION, &result_json, started)
        });
    print_result(command, started, result)
}

#[cfg(feature = "embed")]
pub(crate) fn execute_pool_trace(
    store: &Path,
    mut request: PackRequest,
    evidence_join: EvidenceJoinOptions,
    semantic_models: &SemanticModels,
) -> Result<String, CliError> {
    if semantic_models.embed.is_none() {
        return Err(CliError::SemanticUnavailable(
            "pool-trace requires an active semantic embedder".to_string(),
        ));
    }
    maybe_prepare_pack_semantics(&mut request, semantic_models);
    if request.query_embeddings.is_none() && request.query_token_embeddings.is_none() {
        return Err(CliError::SemanticUnavailable(
            "pool-trace could not encode the requested queries".to_string(),
        ));
    }
    let pool_width = request.rerank_candidates.max(request.max_memories);
    let pool = build_hybrid_rerank_pool_trace_with_evidence_options(
        store,
        &request,
        pool_width,
        evidence_join,
    )?;
    Ok(pool_trace_result_json(&pool))
}

#[cfg(not(feature = "embed"))]
pub(crate) fn execute_pool_trace(
    _store: &Path,
    _request: PackRequest,
    _expansion: EvidenceJoinOptions,
    _semantic_models: &SemanticModels,
) -> Result<String, CliError> {
    Err(CliError::SemanticUnavailable(
        "pool-trace requires a semantic build with an active embedder".to_string(),
    ))
}
fn maybe_prepare_pack_semantics(request: &mut PackRequest, semantic_models: &SemanticModels) {
    maybe_colbert_embed_pack_request(request, semantic_models);
    if request.query_token_embeddings.is_none() {
        maybe_embed_pack_request(request, semantic_models);
    } else {
        request.query_embeddings = None;
    }
}

/// Zip the rerank pool candidates with their cross-encoder scores into
/// `RerankCandidate`s for the store's pack-assembly policy.
fn rerank_candidates_from_pool(
    pool_candidates: Vec<memkeeper_store::RerankPoolCandidate>,
    scores: Vec<f32>,
) -> Vec<memkeeper_store::RerankCandidate> {
    pool_candidates
        .into_iter()
        .zip(scores)
        .map(
            |(candidate, rerank_score)| memkeeper_store::RerankCandidate {
                memory_id: candidate.memory_id,
                content: candidate.content,
                observed_at: candidate.observed_at,
                rerank_score,
                consensus: candidate.consensus,
                activation: candidate.activation,
            },
        )
        .collect()
}

#[cfg(feature = "embed")]
fn rerank_pack_scores(
    request: &PackRequest,
    pool: &memkeeper_store::RerankPool,
    semantic_models: &SemanticModels,
) -> Result<Vec<f32>, CliError> {
    let Some(reranker) = semantic_models.rerank.as_ref() else {
        return Err(CliError::SemanticUnavailable(
            "primary reranker is not active".to_string(),
        ));
    };
    let query = request.queries.first().map_or("", String::as_str);
    if query.is_empty() {
        return Ok(Vec::new());
    }

    let doc_limit = rerank_doc_chars();
    // Contextual representations improve candidate admission through FTS and
    // late interaction. The cross-encoder scores the canonical memory content
    // once so a companion summary cannot override the evidence-bearing text.
    let doc_refs = rerank_pool_documents(&pool.candidates, doc_limit);
    let scores = reranker
        .lock()
        .map_err(|error| {
            CliError::SemanticUnavailable(format!("rerank model lock failed: {error}"))
        })?
        .rerank(query, &doc_refs)
        .map_err(|error| CliError::SemanticUnavailable(format!("rerank failed: {error}")))?;
    if scores.len() != doc_refs.len() {
        return Err(CliError::SemanticUnavailable(format!(
            "reranker returned {} scores for {} rerank texts",
            scores.len(),
            doc_refs.len()
        )));
    }
    Ok(scores)
}

#[cfg(not(feature = "embed"))]
fn rerank_pack_scores(
    _request: &PackRequest,
    _pool: &memkeeper_store::RerankPool,
    _semantic_models: &SemanticModels,
) -> Result<Vec<f32>, CliError> {
    Err(CliError::SemanticUnavailable(
        "primary reranker is unavailable in this build".to_string(),
    ))
}

pub(crate) fn retrieval_only_pack_report(
    request: &PackRequest,
    pool_candidates: Vec<memkeeper_store::RerankPoolCandidate>,
) -> PackReport {
    let count = pool_candidates.len();
    let order_scores = (0..count)
        .map(|index| u16::try_from(count - index).map_or(f32::from(u16::MAX), f32::from))
        .collect();
    let candidates = rerank_candidates_from_pool(pool_candidates, order_scores);
    let mut ungated = request.clone();
    ungated.min_score = 0.0;
    let heading = format!("## Retrieved Memory: {}\n", request.title.trim());
    ungated.max_chars = ungated.max_chars.saturating_sub(heading.len());
    let mut report = memkeeper_store::assemble_reranked_pack(&ungated, &candidates);
    if !report.memory_ids.is_empty() {
        report.content.insert_str(0, &heading);
    }
    // Synthetic scores preserve the unified pool's retrieval order only. Do
    // not expose them as model confidence or feed them into recall promotion.
    report.scores.clear();
    report.top_score = None;
    report
}

pub(crate) fn execute_pack_request(
    store: &Path,
    mut request: PackRequest,
    semantic_models: &SemanticModels,
    require_rerank: bool,
) -> Result<PackReport, CliError> {
    require_primary_reranker(require_rerank, semantic_models)?;
    maybe_prepare_pack_semantics(&mut request, semantic_models);
    let pool_width = request.rerank_candidates.max(request.max_memories);
    let pool = build_hybrid_rerank_pool(store, &request, pool_width).map_err(CliError::from)?;
    if pool.candidates.is_empty() {
        return Ok(memkeeper_store::empty_pack(&request));
    }
    match rerank_pack_scores(&request, &pool, semantic_models) {
        Ok(scores) => {
            // Hand the scored candidates to the store's pure pack-assembly
            // policy: top-score gate, rerank order, then char/count budget.
            let candidates = rerank_candidates_from_pool(pool.candidates, scores);
            Ok(memkeeper_store::assemble_reranked_pack(
                &request,
                &candidates,
            ))
        }
        Err(error) if require_rerank => Err(error),
        Err(error) => {
            eprintln!(
                "[memkeeper] NOTE: rerank unavailable at request time; serving one retrieval-only \
                 pack: {error}"
            );
            Ok(retrieval_only_pack_report(&request, pool.candidates))
        }
    }
}

/// Standalone cross-encoder scoring of arbitrary documents against a query.
///
/// Unlike `pack`, this touches no store -- documents are supplied inline -- so
/// it can score RSS items, search candidates, or any text the caller has in
/// hand using the reranker the serve loop already holds warm.
#[cfg_attr(not(feature = "semantic"), allow(dead_code))]
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RerankRequest {
    pub(crate) query: String,
    pub(crate) documents: Vec<String>,
}

pub(crate) fn rerank_request_from_json(input: &str) -> Result<RerankRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("rerank request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["query", "documents"])?;
    let query = required_string_field(object, "query")?;
    let documents = required_string_array_field(object, "documents")?;
    Ok(RerankRequest { query, documents })
}

pub(crate) fn parse_rerank_args(args: &[String]) -> Result<RerankRequest, CliError> {
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

pub(crate) fn run_rerank(args: &[String]) -> i32 {
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

#[cfg(feature = "semantic")]
pub(crate) fn run_rerank_payload(
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
    let mut reranker = reranker
        .lock()
        .map_err(|e| CliError::InvalidRequest(format!("rerank model lock failed: {e}")))?;
    let model_id = reranker.model_id().to_string();
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
pub(crate) fn run_rerank_payload(
    _request: &RerankRequest,
    _semantic_models: &SemanticModels,
) -> Result<String, CliError> {
    Err(CliError::InvalidRequest(
        "rerank requires the semantic feature; this build has it disabled".to_string(),
    ))
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PackArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: PackRequest,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PoolTraceArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: PackRequest,
    pub(crate) expansion: EvidenceJoinOptions,
}

pub(crate) fn parse_pack_args(args: &[String]) -> Result<PackArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "pack")?;
    Ok(PackArgs {
        store,
        request: pack_request_from_json(&request_json)?,
    })
}

pub(crate) fn parse_pool_trace_args(args: &[String]) -> Result<PoolTraceArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "pool-trace")?;
    Ok(PoolTraceArgs {
        store,
        request: pool_trace_pack_request_from_json(&request_json)?,
        expansion: evidence_join_options_from_json(&request_json)?,
    })
}
