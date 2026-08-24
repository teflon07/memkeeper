//! `graph` command handlers.

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
    parse_store_args, parse_usize_arg, print_result, resolve_store_default, ArgParser,
    SemanticModels,
};
use crate::{hook::*, json::*, output::*, requests::*, serve::*, CliError};

pub(crate) struct EntityUpsertArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: EntityUpsertRequest,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RelationshipUpsertArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: RelationshipUpsertRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EntityMergeArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: EntityMergeRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EntitySearchArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: EntitySearchRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphNeighborsArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: GraphNeighborsRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GraphContextArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: GraphContextRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MemoryListArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: MemoryListRequest,
}

pub(crate) fn run_entity_upsert(args: &[String]) -> i32 {
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

pub(crate) fn run_relationship_upsert(args: &[String]) -> i32 {
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

pub(crate) fn run_entity_merge(args: &[String]) -> i32 {
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

pub(crate) fn run_entity_search(args: &[String]) -> i32 {
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

pub(crate) fn run_graph_neighbors(args: &[String]) -> i32 {
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

pub(crate) fn run_graph_context(args: &[String]) -> i32 {
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

pub(crate) fn run_graph_full(args: &[String]) -> i32 {
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

pub(crate) fn run_memory_list(args: &[String]) -> i32 {
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
pub(crate) fn parse_entity_upsert_args(args: &[String]) -> Result<EntityUpsertArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "entity-upsert")?;
    Ok(EntityUpsertArgs {
        store,
        request: entity_upsert_request_from_json(&request_json)?,
    })
}

pub(crate) fn parse_relationship_upsert_args(
    args: &[String],
) -> Result<RelationshipUpsertArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "relationship-upsert")?;
    Ok(RelationshipUpsertArgs {
        store,
        request: relationship_upsert_request_from_json(&request_json)?,
    })
}

pub(crate) fn parse_entity_merge_args(args: &[String]) -> Result<EntityMergeArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "entity-merge")?;
    Ok(EntityMergeArgs {
        store,
        request: entity_merge_request_from_json(&request_json)?,
    })
}

pub(crate) fn parse_entity_search_args(args: &[String]) -> Result<EntitySearchArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "entity-search")?;
    Ok(EntitySearchArgs {
        store,
        request: entity_search_request_from_json(&request_json)?,
    })
}

pub(crate) fn parse_graph_neighbors_args(args: &[String]) -> Result<GraphNeighborsArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "graph-neighbors")?;
    Ok(GraphNeighborsArgs {
        store,
        request: graph_neighbors_request_from_json(&request_json)?,
    })
}

pub(crate) fn parse_graph_context_args(args: &[String]) -> Result<GraphContextArgs, CliError> {
    let (store, request_json) = parse_json_command_args(args, "graph-context")?;
    Ok(GraphContextArgs {
        store,
        request: graph_context_request_from_json(&request_json)?,
    })
}
pub(crate) fn parse_memory_list_args(args: &[String]) -> Result<MemoryListArgs, CliError> {
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
