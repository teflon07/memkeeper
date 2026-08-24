//! `candidates` command handlers.

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
#[cfg(feature = "embed")]
use memkeeper_store::build_hybrid_rerank_pool_trace_with_evidence_options;
use memkeeper_store::*;
use std::result::Result;

use super::{
    append_csv_values, maybe_colbert_embed_remember_request,
    maybe_colbert_embed_remember_request_with_requirement, maybe_colbert_embed_search_request,
    maybe_embed_document_search_request, maybe_embed_ingest_request, maybe_embed_remember_request,
    maybe_embed_search_request, parse_bool, parse_f64_arg, parse_json_command_args,
    parse_usize_arg, print_result, ArgParser, SemanticModels,
};
use crate::{hook::*, json::*, output::*, requests::*, serve::*, CliError};

pub(crate) struct CandidateSubmitArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: CandidateSubmitRequest,
}

pub(crate) fn parse_candidate_submit_args(
    args: &[String],
) -> Result<CandidateSubmitArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "candidate-submit")?;
    Ok(CandidateSubmitArgs {
        store,
        request: candidate_submit_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_candidate_submit(args: &[String]) -> i32 {
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

pub(crate) struct CandidateListArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: CandidateListRequest,
}

pub(crate) fn parse_candidate_list_args(args: &[String]) -> Result<CandidateListArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "candidate-list")?;
    Ok(CandidateListArgs {
        store,
        request: candidate_list_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_candidate_list(args: &[String]) -> i32 {
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

pub(crate) struct CandidateApproveArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: CandidateApproveRequest,
}

pub(crate) fn parse_candidate_approve_args(
    args: &[String],
) -> Result<CandidateApproveArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "candidate-approve")?;
    Ok(CandidateApproveArgs {
        store,
        request: candidate_approve_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_candidate_approve(args: &[String]) -> i32 {
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

pub(crate) struct CandidateRejectArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: CandidateRejectRequest,
}

pub(crate) fn parse_candidate_reject_args(
    args: &[String],
) -> Result<CandidateRejectArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "candidate-reject")?;
    Ok(CandidateRejectArgs {
        store,
        request: candidate_reject_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_candidate_reject(args: &[String]) -> i32 {
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

pub(crate) struct CandidateQuarantineArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: CandidateQuarantineRequest,
}

pub(crate) fn parse_candidate_quarantine_args(
    args: &[String],
) -> Result<CandidateQuarantineArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "candidate-quarantine")?;
    Ok(CandidateQuarantineArgs {
        store,
        request: candidate_quarantine_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_candidate_quarantine(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::CandidateQuarantine;
    let result = parse_candidate_quarantine_args(args)
        .and_then(|options| {
            quarantine_candidate(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &candidate_quarantine_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

pub(crate) struct RecallLogArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: RecallLogRequest,
}
