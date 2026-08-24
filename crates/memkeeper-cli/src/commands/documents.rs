//! `documents` command handlers.

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
    parse_usize_arg, print_result, SemanticModels,
};


pub(crate) struct IngestArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: IngestRequest,
}

pub(crate) fn parse_ingest_args(args: &[String]) -> Result<IngestArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "ingest")?;
    Ok(IngestArgs {
        store,
        request: ingest_request_from_json(&request_json)?,
    })
}

pub(crate) struct DocumentGetArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: DocumentGetRequest,
}

pub(crate) fn parse_document_get_args(args: &[String]) -> Result<DocumentGetArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "document-get")?;
    Ok(DocumentGetArgs {
        store,
        request: document_get_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_document_get(args: &[String]) -> i32 {
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

pub(crate) struct PromotionCandidatesArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: PromotionCandidatesRequest,
}

pub(crate) fn parse_promotion_candidates_args(args: &[String]) -> Result<PromotionCandidatesArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "promotion-candidates")?;
    Ok(PromotionCandidatesArgs {
        store,
        request: promotion_candidates_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_promotion_candidates(args: &[String]) -> i32 {
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

pub(crate) struct DocumentDuplicatesArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: DocumentDuplicatesRequest,
}

pub(crate) fn parse_document_duplicates_args(args: &[String]) -> Result<DocumentDuplicatesArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "document-duplicates")?;
    Ok(DocumentDuplicatesArgs {
        store,
        request: document_duplicates_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_document_duplicates(args: &[String]) -> i32 {
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

pub(crate) struct DocumentPruneArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: DocumentPruneRequest,
}

pub(crate) fn parse_document_prune_args(args: &[String]) -> Result<DocumentPruneArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "document-prune")?;
    Ok(DocumentPruneArgs {
        store,
        request: document_prune_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_document_prune(args: &[String]) -> i32 {
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

pub(crate) struct MarkExtractedArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: MarkExtractedRequest,
}

pub(crate) fn parse_mark_extracted_args(args: &[String]) -> Result<MarkExtractedArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "mark-extracted")?;
    Ok(MarkExtractedArgs {
        store,
        request: mark_extracted_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_mark_extracted(args: &[String]) -> i32 {
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

pub(crate) struct DocumentSearchArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: DocumentSearchRequest,
}

pub(crate) fn parse_document_search_args(args: &[String]) -> Result<DocumentSearchArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "document-search")?;
    Ok(DocumentSearchArgs {
        store,
        request: document_search_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_document_search(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_remember_or_search();
    run_document_search_with_models(args, &semantic_models)
}

pub(crate) fn run_document_search_with_models(args: &[String], semantic_models: &SemanticModels) -> i32 {
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

pub(crate) fn run_ingest(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_remember_or_search();
    run_ingest_with_models(args, &semantic_models)
}

pub(crate) fn run_ingest_with_models(args: &[String], semantic_models: &SemanticModels) -> i32 {
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

pub(crate) struct CandidateSubmitArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: CandidateSubmitRequest,
}
