//! `memory` command handlers.

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
use super::{resolve_store_default, ArgParser, 
    append_csv_values, maybe_colbert_embed_remember_request,
    maybe_colbert_embed_remember_request_with_requirement, maybe_colbert_embed_search_request,
    maybe_embed_document_search_request, maybe_embed_ingest_request, maybe_embed_remember_request,
    maybe_embed_search_request, parse_bool, parse_f64_arg, parse_json_command_args,
    parse_usize_arg, print_result, SemanticModels,
};

#[derive(Debug)]
pub(crate) struct RememberArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: RememberRequest,
}


pub(crate) struct GetArgs {
    pub(crate) store: PathBuf,
    pub(crate) id: String,
    pub(crate) options: GetOptions,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ForgetArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: ForgetRequest,
}

#[derive(Debug, Clone)]
pub(crate) struct VerifyArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: VerifyRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HistoryArgs {
    pub(crate) store: PathBuf,
    pub(crate) id: String,
    pub(crate) options: HistoryOptions,
}


pub(crate) struct RecallLogArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: RecallLogRequest,
}


pub(crate) fn run_remember(args: &[String]) -> i32 {
    let semantic_models = SemanticModels::for_remember_or_search();
    run_remember_with_models(args, &semantic_models)
}

pub(crate) fn run_remember_with_models(args: &[String], semantic_models: &SemanticModels) -> i32 {
    run_remember_with_models_and_requirement(args, semantic_models, crate::serve::require_semantic_env())
}

pub(crate) fn run_remember_with_models_and_requirement(
    args: &[String],
    semantic_models: &SemanticModels,
    require_semantic: bool,
) -> i32 {
    let started = Instant::now();
    let command = Command::Remember;
    let result = parse_remember_args(args).and_then(|mut options| {
        maybe_embed_remember_request(&mut options.request, semantic_models);
        maybe_colbert_embed_remember_request_with_requirement(
            &mut options.request,
            semantic_models,
            require_semantic,
        )?;
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
pub(crate) fn run_get(args: &[String]) -> i32 {
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

pub(crate) fn run_forget(args: &[String]) -> i32 {
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

pub(crate) fn run_verify(args: &[String]) -> i32 {
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

pub(crate) fn run_recall_log(args: &[String]) -> i32 {
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
pub(crate) fn parse_recall_log_args(args: &[String]) -> Result<RecallLogArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "recall-log")?;
    Ok(RecallLogArgs {
        store,
        request: recall_log_request_from_json(&request_json)?,
    })
}

pub(crate) fn run_history(args: &[String]) -> i32 {
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
pub(crate) fn parse_remember_args(args: &[String]) -> Result<RememberArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "remember")?;
    Ok(RememberArgs {
        store,
        request: remember_request_from_json(&request_json)?,
    })
}

pub(crate) fn parse_get_args(args: &[String]) -> Result<GetArgs, CliError> {
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

pub(crate) fn parse_forget_args(args: &[String]) -> Result<ForgetArgs, CliError> {
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

pub(crate) fn parse_verify_args(args: &[String]) -> Result<VerifyArgs, CliError> {
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

pub(crate) fn parse_history_args(args: &[String]) -> Result<HistoryArgs, CliError> {
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

