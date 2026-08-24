//! `search` command handlers.

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
    append_csv_values, maybe_colbert_embed_remember_request,
    maybe_colbert_embed_remember_request_with_requirement, maybe_colbert_embed_search_request,
    maybe_embed_document_search_request, maybe_embed_ingest_request, maybe_embed_remember_request,
    maybe_embed_search_request, parse_bool, parse_f64_arg, parse_json_command_args,
    parse_usize_arg, print_result, require_primary_reranker, SemanticModels,
};

pub(crate) struct SearchArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: SearchRequest,
    /// Native cross-encoder rerank of the result pool (opt-in).
    pub(crate) rerank: bool,
    /// Candidate pool width when reranking.
    pub(crate) rerank_candidates: usize,
}


pub(crate) struct BatchSearchArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: BatchSearchRequest,
}

pub(crate) fn run_search(args: &[String]) -> i32 {
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
/// the serve `guard`: refuse when the operator requires semantic
/// retrieval but this request cannot be embedded — no caller-supplied vector
/// (`request_has_embedding`) and no active embedder (`embed_active`, always
/// false on a non-semantic build). Kept pure so it is unit-testable without
/// touching the `MEMKEEPER_REQUIRE_SEMANTIC` process env.
pub(crate) fn must_refuse_search_without_semantic(
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
pub(crate) fn execute_search(
    store: &Path,
    mut request: SearchRequest,
    rerank: bool,
    rerank_candidates: usize,
    semantic_models: &SemanticModels,
) -> Result<String, CliError> {
    require_primary_reranker(rerank, semantic_models)?;
    if let Err(message) = maybe_embed_search_request(&mut request, semantic_models) {
        // Fail closed when the operator requires semantic retrieval: a runtime
        // embedding failure must not be served as a silent FTS-only success.
        if crate::serve::require_semantic_env() {
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
        crate::serve::require_semantic_env(),
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
#[cfg(any(test, feature = "embed"))]
fn rerank_doc(content: &str, limit: usize) -> &str {
    if limit == 0 {
        return content;
    }
    match content.char_indices().nth(limit) {
        Some((byte_index, _)) => &content[..byte_index],
        None => content,
    }
}

#[cfg(any(test, feature = "embed"))]
pub(crate) fn rerank_pool_documents(
    candidates: &[memkeeper_store::RerankPoolCandidate],
    limit: usize,
) -> Vec<&str> {
    candidates
        .iter()
        .map(|candidate| rerank_doc(&candidate.content, limit))
        .collect()
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
pub(crate) fn run_batch_search(args: &[String]) -> i32 {
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
pub(crate) fn parse_search_args(args: &[String]) -> Result<SearchArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "search")?;
    let (rerank, rerank_candidates) = search_rerank_options_from_json(&request_json)?;
    Ok(SearchArgs {
        store,
        request: search_request_from_json(&request_json)?,
        rerank,
        rerank_candidates,
    })
}
pub(crate) fn parse_batch_search_args(args: &[String]) -> Result<BatchSearchArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "batch-search")?;
    Ok(BatchSearchArgs {
        store,
        request: batch_search_request_from_json(&request_json)?,
    })
}
