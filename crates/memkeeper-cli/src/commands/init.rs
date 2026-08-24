//! `init` command handlers.

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

pub(crate) struct StoreArgs {
    pub(crate) store: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DoctorArgs {
    pub(crate) store: PathBuf,
    pub(crate) store_source: String,
    pub(crate) include_indexes: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StatsArgs {
    pub(crate) store: PathBuf,
    pub(crate) include_indexes: bool,
    pub(crate) include_health: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpaceListArgs {
    pub(crate) store: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SpaceCreateArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: SpaceCreateRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SiloListArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: SiloListRequest,
}

pub(crate) fn run_init(args: &[String]) -> i32 {
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

pub(crate) fn run_doctor(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Doctor;
    let result = parse_doctor_args(args).map(|options| {
        let (result_json, schema_version) = doctor_result_json(&options);
        doctor_success_envelope(&options.store, schema_version, &result_json, started)
    });
    print_result(command, started, result)
}

pub(crate) fn run_stats(args: &[String]) -> i32 {
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

#[cfg(feature = "semantic")]
pub(crate) fn run_local_usage() -> i32 {
    println!("{}", memkeeper_embed::local_usage_report());
    0
}

#[cfg(not(feature = "semantic"))]
pub(crate) fn run_local_usage() -> i32 {
    eprintln!("local usage requires the semantic build");
    1
}

pub(crate) fn run_space_list(args: &[String]) -> i32 {
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

pub(crate) fn run_space_create(args: &[String]) -> i32 {
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

pub(crate) fn run_silo_list(args: &[String]) -> i32 {
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
pub(crate) fn resolve_store_default() -> PathBuf {
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
    // Windows rarely sets HOME; use the native profile dir instead.
    #[cfg(windows)]
    if let Some(profile) = std::env::var_os("USERPROFILE") {
        if !profile.is_empty() {
            return PathBuf::from(profile)
                .join(".memkeeper")
                .join("store.sqlite");
        }
    }
    PathBuf::from(PROJECT_STORE_RELATIVE_PATH)
}

pub(crate) fn parse_store_args(args: &[String]) -> Result<StoreArgs, CliError> {
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

pub(crate) fn parse_doctor_args(args: &[String]) -> Result<DoctorArgs, CliError> {
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

pub(crate) fn parse_stats_args(args: &[String]) -> Result<StatsArgs, CliError> {
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

pub(crate) fn parse_space_list_args(args: &[String]) -> Result<SpaceListArgs, CliError> {
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

pub(crate) fn parse_space_create_args(args: &[String]) -> Result<SpaceCreateArgs, CliError> {
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

pub(crate) fn parse_silo_list_args(args: &[String]) -> Result<SiloListArgs, CliError> {
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

