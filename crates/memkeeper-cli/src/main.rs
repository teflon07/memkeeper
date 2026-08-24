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
// Only used by the embed-gated rerank path; gate the import so a lexical-only
// (`--no-default-features`) build doesn't warn on an unused symbol.
#[cfg(feature = "embed")]
use memkeeper_store::build_hybrid_rerank_pool_trace_with_evidence_options;
use memkeeper_store::{
    approve_candidate, backup_store, batch_search_memories, build_hybrid_rerank_pool, create_space,
    document_duplicates, dream_store, export_store, forget_memory, get_document, get_memory,
    graph_context, graph_full, graph_neighbors, import_store, ingest_source, init_store,
    inspect_store_stats, last_synthesis_run, list_candidates, list_memories, list_silos,
    list_spaces, mark_source_episodes_extracted, memory_history, merge_entity,
    promotion_candidates, prune_documents, quarantine_candidate, record_recall, reject_candidate,
    remember_memory, schema_mentions_required_objects, search_documents, search_entities,
    search_memories, store_stats, store_stats_with_health, submit_candidate, upsert_entity,
    upsert_relationship, verify_memory, BackupReport, BackupRequest, BatchSearchItemReport,
    BatchSearchQuery, BatchSearchReport, BatchSearchRequest, CandidateApproveReport,
    CandidateApproveRequest, CandidateListReport, CandidateListRequest, CandidateQuarantineReport,
    CandidateQuarantineRequest, CandidateRecord, CandidateRejectReport, CandidateRejectRequest,
    CandidateSubmitReport, CandidateSubmitRequest, CapturedEntity, CapturedRelationship,
    DocumentChunk, DocumentDuplicatesReport, DocumentDuplicatesRequest, DocumentGetReport,
    DocumentGetRequest, DocumentPruneReport, DocumentPruneRequest, DocumentSearchReport,
    DocumentSearchRequest, DocumentSearchResult, DreamDedupeReport, DreamDuplicateProposal,
    DreamEntityProjection, DreamExpireReport, DreamGraphReport, DreamPromoteReport,
    DreamReindexReport, DreamRelationshipProposal, DreamReport, DreamRequest,
    DuplicateChunkCluster, DuplicateChunkMember, EntityMergeReport, EntityMergeRequest,
    EntityRecord, EntitySearchReport, EntitySearchRequest, EntitySearchResult, EntityUpsertReport,
    EntityUpsertRequest, Error as StoreError, EvidenceJoinOptions, ExportReport, ExportRequest,
    ExportTableReport, ForgetReport, ForgetRequest, GetOptions, GraphCapture, GraphContextReport,
    GraphContextRequest, GraphEntityRecord, GraphFullReport, GraphNeighborsReport,
    GraphNeighborsRequest, GraphRelationshipRecord, HealthStats, HistoryOptions, HistoryReport,
    ImportReport, ImportRequest, IndexStats, IngestReport, IngestRequest, InitReport,
    MarkExtractedReport, MarkExtractedRequest, MemoryEventRecord, MemoryLinkRecord, MemoryListItem,
    MemoryListReport, MemoryListRequest, MemoryRecord, MemoryRepresentationRecord,
    MemoryVersionRecord, PackReport, PackRequest, PromotionCandidate, PromotionCandidatesReport,
    PromotionCandidatesRequest, RecallEvent, RecallLogReport, RecallLogRequest, RelationshipRecord,
    RelationshipUpsertReport, RelationshipUpsertRequest, RememberCandidate,
    RememberConflictCandidate, RememberReport, RememberRequest, RepresentationWriteStatus,
    RetrievalRepresentationInput, SearchFilters, SearchReport, SearchRequest, SearchResult,
    SiloListReport, SiloListRequest, SiloRecord, SpaceCreateReport, SpaceCreateRequest,
    SpaceListReport, SpaceRecord, Stats, VerifyReport, VerifyRequest, DEFAULT_DREAM_MAX_MEMORIES,
    DEFAULT_PROMOTE_RANK_CAP, DEFAULT_PROMOTE_SCORE_FLOOR, DEFAULT_PROMOTE_THRESHOLD,
    PROJECT_STORE_RELATIVE_PATH, SCHEMA_VERSION, USER_STORE_PATH_HINT,
};
#[cfg(feature = "embed")]
use std::sync::Mutex;

const MAX_JSON_BYTES: usize = 1_048_576;
const MAX_JSON_DEPTH: usize = 64;
const MAX_JSON_OBJECT_FIELDS: usize = 256;
const MAX_JSON_ARRAY_ITEMS: usize = 4_096;
const MAX_SPACE_TEXT_OUTPUT_CHARS: usize = 512;
const MAX_SPACE_CONFIG_OUTPUT_CHARS: usize = 2_048;

mod commands;
mod dashboard;
mod hook;
mod json;
mod mcp;
mod output;
mod requests;
mod schema;
mod serve;

#[allow(clippy::wildcard_imports)]
pub(crate) use commands::*;
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

#[derive(Debug)]
pub(crate) enum CliError {
    InvalidRequest(String),
    /// A required semantic dependency failed at request time, so the request is
    /// failed closed instead of silently degrading.
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
                "A required semantic component failed at request time. Check the embedding or \
                 reranking model and provider, then retry. Unset MEMKEEPER_REQUIRE_SEMANTIC or \
                 MEMKEEPER_REQUIRE_RERANK only when degraded fallback is acceptable."
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

/// Resolve curl-style `--json @<file>` / `--json -` (stdin) sentinels in the raw
/// arg list to the literal payload, so every command's `--json` handling sees
/// plain JSON. Applies to `--json` / `--request-json` / `--request`, space- or
/// `=`-separated. Anything not starting with `@` (or equal to `-`) is unchanged.
fn resolve_json_args(args: Vec<String>) -> Result<Vec<String>, CliError> {
    const FLAGS: [&str; 3] = ["--json", "--request-json", "--request"];
    let mut out = Vec::with_capacity(args.len());
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        if FLAGS.contains(&arg.as_str()) {
            out.push(arg);
            if let Some(value) = iter.next() {
                out.push(resolve_json_payload(&value)?);
            }
        } else if let Some((flag, value)) = FLAGS.iter().find_map(|flag| {
            arg.strip_prefix(&format!("{flag}="))
                .map(|v| (*flag, v.to_owned()))
        }) {
            out.push(format!("{flag}={}", resolve_json_payload(&value)?));
        } else {
            out.push(arg);
        }
    }
    Ok(out)
}

/// `@<path>` -> file contents; `-` -> stdin; anything else -> unchanged.
fn resolve_json_payload(value: &str) -> Result<String, CliError> {
    if value == "-" {
        let mut buf = String::new();
        let mut stdin = std::io::stdin();
        std::io::Read::read_to_string(&mut stdin, &mut buf).map_err(|error| {
            CliError::InvalidRequest(format!("failed to read --json from stdin: {error}"))
        })?;
        Ok(buf)
    } else if let Some(path) = value.strip_prefix('@') {
        std::fs::read_to_string(path).map_err(|error| {
            CliError::InvalidRequest(format!("failed to read --json file '{path}': {error}"))
        })
    } else {
        Ok(value.to_string())
    }
}

#[allow(clippy::too_many_lines)]
fn main() {
    // Resolve `--json @file` / `--json -` (stdin) up front so every command's
    // `--json` handling sees a literal payload (avoids brittle inline JSON quoting,
    // especially in Windows PowerShell).
    let resolved_args = match resolve_json_args(env::args().skip(1).collect()) {
        Ok(resolved) => resolved,
        Err(error) => {
            eprintln!("[memkeeper] {error}");
            process::exit(error.exit_code());
        }
    };
    let mut args = resolved_args.into_iter();
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
        "local-usage" => run_local_usage(),
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
        "pool-trace" => {
            let args = args.collect::<Vec<_>>();
            run_pool_trace(&args)
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
        "candidate-quarantine" => {
            let args = args.collect::<Vec<_>>();
            run_candidate_quarantine(&args)
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

#[cfg(test)]
mod tests;
