//! CLI command handlers split from `main.rs`.

#![allow(clippy::wildcard_imports)]

use std::{
    path::{Path, PathBuf},
    time::Instant,
};

use memkeeper_protocol::Command;
use memkeeper_store::*;
use std::result::Result;

use crate::{hook::*, json::*, output::*, requests::*, serve::*, CliError};

mod archive;
mod candidates;
mod documents;
mod graph;
mod init;
mod memory;
mod models;
mod pack;
mod search;

pub(crate) use archive::*;
pub(crate) use candidates::*;
pub(crate) use documents::*;
pub(crate) use graph::*;
pub(crate) use init::*;
pub(crate) use memory::*;
pub(crate) use models::*;
pub(crate) use pack::*;
pub(crate) use search::*;

#[cfg(feature = "embed")]
use std::sync::Mutex;

#[cfg(feature = "embed")]
pub(crate) struct SemanticModels {
    pub(crate) embed: Option<Mutex<Box<dyn memkeeper_embed::Embedder>>>,
    pub(crate) rerank: Option<Mutex<Box<dyn memkeeper_embed::Reranker>>>,
    pub(crate) colbert: Option<Mutex<Box<dyn memkeeper_embed::TokenEmbedder>>>,
}

/// Late-interaction retrieval gate: `MEMKEEPER_LATE_INTERACTION=1` plus
/// `MEMKEEPER_COLBERT_MODEL_DIR` (checked at model load).
#[cfg(feature = "embed")]
pub(crate) fn late_interaction_enabled() -> bool {
    std::env::var("MEMKEEPER_LATE_INTERACTION").is_ok_and(|value| value == "1")
}

#[cfg(feature = "embed")]
impl SemanticModels {
    pub(crate) fn load(embed: bool, rerank: bool) -> Self {
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

    pub(crate) fn for_remember_or_search() -> Self {
        Self::load(true, false)
    }

    pub(crate) fn for_search(rerank: bool) -> Self {
        Self::load(true, rerank)
    }

    pub(crate) fn for_pack() -> Self {
        Self::load(true, true)
    }

    pub(crate) fn for_serve() -> Self {
        Self::for_pack()
    }

    #[cfg(test)]
    pub(crate) const fn for_test() -> Self {
        Self {
            embed: None,
            rerank: None,
            colbert: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_colbert_for_test(model: Box<dyn memkeeper_embed::TokenEmbedder>) -> Self {
        Self {
            embed: None,
            rerank: None,
            colbert: Some(Mutex::new(model)),
        }
    }

    /// Whether the embedder actually loaded. False means the model files were
    /// missing/unreadable, so search will silently fall back to BM25 — the serve
    /// runtime guard keys on this to fail loud instead.
    pub(crate) fn embed_active(&self) -> bool {
        self.embed.is_some()
    }

    /// Whether the reranker actually loaded. When a rerank-requesting load
    /// (`for_pack`/`for_serve`) leaves this false, the cross-encoder is off and
    /// results serve in plain retrieval order — the serve guard keys on this to
    /// warn loud instead of degrading silently.
    pub(crate) fn rerank_active(&self) -> bool {
        self.rerank.is_some()
    }

    /// Whether the `ColBERT` (late-interaction) model actually loaded. Only
    /// meaningful when `MEMKEEPER_LATE_INTERACTION=1`; false there means the
    /// explicitly-requested late-interaction path is silently off.
    pub(crate) fn colbert_active(&self) -> bool {
        self.colbert.is_some()
    }
}

#[cfg(not(feature = "embed"))]
pub(crate) struct SemanticModels;

#[cfg(not(feature = "embed"))]
impl SemanticModels {
    pub(crate) const fn for_remember_or_search() -> Self {
        Self
    }

    pub(crate) const fn for_search(_rerank: bool) -> Self {
        Self
    }

    pub(crate) const fn for_pack() -> Self {
        Self
    }

    pub(crate) const fn for_serve() -> Self {
        Self::for_pack()
    }

    #[cfg(test)]
    pub(crate) const fn for_test() -> Self {
        Self
    }

    #[allow(clippy::unused_self)]
    pub(crate) const fn rerank_active(&self) -> bool {
        false
    }
}

#[cfg(feature = "embed")]
pub(crate) fn maybe_embed_remember_request(
    request: &mut RememberRequest,
    semantic_models: &SemanticModels,
) {
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
pub(crate) fn maybe_embed_remember_request(
    _request: &mut RememberRequest,
    _semantic_models: &SemanticModels,
) {
}

#[cfg(feature = "embed")]
pub(crate) fn maybe_embed_ingest_request(
    request: &mut IngestRequest,
    semantic_models: &SemanticModels,
) {
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
pub(crate) fn maybe_embed_ingest_request(
    _request: &mut IngestRequest,
    _semantic_models: &SemanticModels,
) {
}

#[cfg(feature = "embed")]
pub(crate) fn maybe_embed_document_search_request(
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
pub(crate) fn maybe_embed_document_search_request(
    _request: &mut DocumentSearchRequest,
    _semantic_models: &SemanticModels,
) {
}

#[cfg(any(test, feature = "embed"))]
pub(crate) fn remember_token_document(request: &RememberRequest) -> String {
    let companion = memkeeper_store::retrieval_companion(
        request.summary.as_deref(),
        request.retrieval_representation.as_ref(),
    );
    memkeeper_store::representation_document(&request.content, companion)
}

#[cfg(feature = "embed")]
pub(crate) fn maybe_colbert_embed_remember_request(
    request: &mut RememberRequest,
    semantic_models: &SemanticModels,
) -> Result<(), CliError> {
    maybe_colbert_embed_remember_request_with_requirement(
        request,
        semantic_models,
        crate::serve::require_semantic_env(),
    )
}

#[cfg(feature = "embed")]
pub(crate) fn maybe_colbert_embed_remember_request_with_requirement(
    request: &mut RememberRequest,
    semantic_models: &SemanticModels,
    require_semantic: bool,
) -> Result<(), CliError> {
    if request.token_embedding.is_some() {
        return Ok(());
    }
    let Some(model) = semantic_models.colbert.as_ref() else {
        if request.retrieval_representation.is_some() && require_semantic {
            return Err(CliError::SemanticUnavailable(
                "retrieval representation token encoder is unavailable".to_string(),
            ));
        }
        return Ok(());
    };
    let text = remember_token_document(request);
    match model.lock() {
        Ok(mut model) => match model.encode_docs(&[&text]) {
            Ok(mut vecs) => {
                request.token_embedding = vecs.pop().filter(|tokens| !tokens.is_empty());
                if request.token_embedding.is_some() {
                    request.token_embedding_model_id = Some(model.model_id().to_string());
                    return Ok(());
                }
                let message = "retrieval representation token encoding returned no tokens";
                if request.retrieval_representation.is_some() && require_semantic {
                    return Err(CliError::SemanticUnavailable(message.to_string()));
                }
                eprintln!("[memkeeper] {message}");
            }
            Err(error) => {
                let message = format!("retrieval representation token encoding failed: {error}");
                if request.retrieval_representation.is_some() && require_semantic {
                    return Err(CliError::SemanticUnavailable(message));
                }
                eprintln!("[memkeeper] colbert embedding failed: {error}");
            }
        },
        Err(error) => {
            let message = format!("retrieval representation token encoder lock failed: {error}");
            if request.retrieval_representation.is_some() && require_semantic {
                return Err(CliError::SemanticUnavailable(message));
            }
            eprintln!("[memkeeper] colbert model lock failed: {error}");
        }
    }
    Ok(())
}

#[cfg(not(feature = "embed"))]
pub(crate) fn maybe_colbert_embed_remember_request(
    request: &mut RememberRequest,
    semantic_models: &SemanticModels,
) -> Result<(), CliError> {
    maybe_colbert_embed_remember_request_with_requirement(
        request,
        semantic_models,
        crate::serve::require_semantic_env(),
    )
}

#[cfg(not(feature = "embed"))]
pub(crate) fn maybe_colbert_embed_remember_request_with_requirement(
    request: &mut RememberRequest,
    _semantic_models: &SemanticModels,
    require_semantic: bool,
) -> Result<(), CliError> {
    if request.retrieval_representation.is_some() && require_semantic {
        return Err(CliError::SemanticUnavailable(
            "retrieval representation token encoder is unavailable in this build".to_string(),
        ));
    }
    Ok(())
}

#[cfg(feature = "embed")]
pub(crate) fn maybe_colbert_embed_search_request(
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
pub(crate) fn maybe_colbert_embed_search_request(
    _request: &mut SearchRequest,
    _semantic_models: &SemanticModels,
) {
}

#[cfg(feature = "embed")]
pub(crate) fn maybe_colbert_embed_pack_request(
    request: &mut PackRequest,
    semantic_models: &SemanticModels,
) {
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
pub(crate) fn maybe_colbert_embed_pack_request(
    _request: &mut PackRequest,
    _semantic_models: &SemanticModels,
) {
}

#[cfg(feature = "embed")]
pub(crate) fn maybe_embed_search_request(
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

// Mirrors the `embed` variant's fallible signature for a shared call site; the
// lexical stub can only ever return `Ok`, so silence `unnecessary_wraps` as the
// sibling non-embed stubs (reindex_count, reembed_store, ...) already do.
#[cfg(not(feature = "embed"))]
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn maybe_embed_search_request(
    _request: &mut SearchRequest,
    _semantic_models: &SemanticModels,
) -> Result<(), String> {
    Ok(())
}

#[cfg(feature = "embed")]
pub(crate) fn maybe_embed_pack_request(
    request: &mut PackRequest,
    semantic_models: &SemanticModels,
) {
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
pub(crate) fn maybe_embed_pack_request(
    _request: &mut PackRequest,
    _semantic_models: &SemanticModels,
) {
}

pub(crate) struct ArgParser<'a> {
    args: &'a [String],
    index: usize,
}

impl<'a> ArgParser<'a> {
    pub(crate) const fn new(args: &'a [String]) -> Self {
        Self { args, index: 0 }
    }

    pub(crate) fn next(&mut self) -> Option<String> {
        let value = self.args.get(self.index)?.clone();
        self.index += 1;
        Some(value)
    }

    pub(crate) fn peek(&self) -> Option<&str> {
        self.args.get(self.index).map(String::as_str)
    }

    pub(crate) fn required_value(&mut self, flag: &str) -> Result<PathBuf, CliError> {
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

    pub(crate) fn required_string(&mut self, flag: &str) -> Result<String, CliError> {
        self.next()
            .ok_or_else(|| CliError::InvalidRequest(format!("missing value for {flag}")))
    }
}

pub(crate) fn print_result(
    command: Command,
    started: Instant,
    result: Result<String, CliError>,
) -> i32 {
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
pub(crate) fn parse_json_command_args(
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
pub(crate) fn parse_bool(value: &str) -> Result<bool, CliError> {
    match value {
        "true" | "1" | "yes" => Ok(true),
        "false" | "0" | "no" => Ok(false),
        _ => Err(CliError::InvalidRequest(format!(
            "invalid boolean value: {value}"
        ))),
    }
}

pub(crate) fn parse_usize_arg(flag: &str, value: &str) -> Result<usize, CliError> {
    value.parse::<usize>().map_err(|_| {
        CliError::InvalidRequest(format!("invalid non-negative integer for {flag}: {value}"))
    })
}

pub(crate) fn parse_f64_arg(flag: &str, value: &str) -> Result<f64, CliError> {
    value
        .parse::<f64>()
        .map_err(|_| CliError::InvalidRequest(format!("invalid number for {flag}: {value}")))
}

pub(crate) fn append_csv_values(target: &mut Vec<String>, value: &str) {
    target.extend(value.split(',').map(str::to_string));
}
