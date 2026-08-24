#![cfg_attr(not(feature = "semantic"), forbid(unsafe_code))]

//! Store boundary for the local `SQLite` canonical memory database.
//!
//! The store crate owns initialization, schema validation, and deterministic
//! store diagnostics. It intentionally exposes structured results instead of a
//! host-specific JSON shape so CLI, Pi, MCP, and future adapters can share the
//! same semantics.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    env,
    fmt::{self, Write as _},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, BufWriter, Write as IoWrite},
    path::{Path, PathBuf},
    process,
    sync::atomic::Ordering,
    time::Duration,
};

use memkeeper_core::{
    infer_kind_from_prefix, kind, scope, status, ALL_SPACES, DEFAULT_DURABLE_SILO, DEFAULT_SPACE,
};
use rusqlite::{
    backup, params, params_from_iter,
    types::{Value, ValueRef},
    Connection, OpenFlags, OptionalExtension, Row, Transaction,
};

mod spaces;
pub(crate) use spaces::{
    cleanup_vestigial_long_term_silo, default_silo, ensure_silo_exists, ensure_space_exists,
    normalize_space_name, seed_standard_silos, space_exists,
};
pub use spaces::{create_space, list_silos, list_spaces};

mod common;
pub(crate) use common::{
    collect_rows, is_supported_kind, is_supported_scope, is_supported_status,
    json_string_for_store, limit_i64, next_id, normalized_tags, now_timestamp, sha256_hex,
    sha256_path, sha256_text, string_array_json, unique_nanos, Sha256, ID_COUNTER,
};

mod archive_spec;
pub(crate) use archive_spec::{ExportTableSpec, EXPORT_TABLES};

mod representation;
pub(crate) use representation::{
    insert_representation, load_representation, validate_retrieval_representation,
};
pub use representation::{representation_document, retrieval_companion};

mod recall;
pub use recall::record_recall;

mod stats;
pub(crate) use stats::space_names;
pub use stats::{inspect_store_stats, last_synthesis_run, store_stats, store_stats_with_health};

mod schema;
#[cfg(test)]
pub(crate) use schema::SCHEMA_SQL;
#[cfg(all(test, feature = "semantic"))]
pub(crate) use schema::SEMANTIC_TABLE_SQL;
pub(crate) use schema::{
    apply_schema, ensure_memory_candidates, ensure_recall_events,
    ensure_source_episode_recall_events, normalize_imported_schema_metadata,
    older_schema_is_memkeeper, required_config_value, table_exists, validate_initialized,
};
#[cfg(feature = "semantic")]
pub(crate) use schema::{
    drop_all_vector_tables, ensure_memory_vector_table, ensure_source_episode_vector_table,
    semantic_table_for_dims, source_episode_vector_table,
};
pub use schema::{
    schema_mentions_required_objects, REQUIRED_FTS_TABLES, REQUIRED_TABLES, SCHEMA_VERSION,
};

mod connection;
pub use connection::init_store;
pub(crate) use connection::{
    cleanup_inspection_copy, configure_connection, enable_wal, inspect_on_copy,
    inspection_copy_path, journal_mode, open_initialized_read_fast, open_initialized_write,
    register_sqlite_vec_extension, reject_sqlite_sidecar_symlinks, sidecar_path, sqlite_version,
    user_version, validate_store_path,
};

mod dream;
pub use dream::relationship_confidence_from_evidence;
pub(crate) use dream::{dream_store_tx, validate_dream_request};

mod graph;
pub(crate) use graph::{
    apply_graph_capture, upsert_relationship_tx, validate_graph_capture,
    PreparedRelationshipUpsertRequest,
};
pub use graph::{
    graph_context, graph_full, graph_neighbors, merge_entity, search_entities, upsert_entity,
    upsert_relationship, GraphFullLink, GraphFullNode, GraphFullReport,
};

mod snapshot;
pub(crate) use snapshot::with_read_snapshot;

mod validate;
pub(crate) use validate::{
    normalize_filter_values, normalize_search_filters, normalize_utc_timestamp,
    reject_all_spaces_sentinel, validate_optional_metadata_value, validate_optional_timestamp,
    validate_export_request, validate_backup_request, validate_output_path,
    validate_forget_request, validate_history_request, validate_remember_request,
    validate_batch_search_request, validate_search_filters, validate_pack_request,
    validate_optional_embedding, validate_memory_link_ids, is_utc_rfc3339_like,
    timestamp_parts_are_valid,
};

mod documents;
pub use documents::{
    document_duplicates, get_document, ingest_source, mark_source_episodes_extracted,
    prune_documents, promotion_candidates, search_documents, DocumentChunk,
    DocumentDuplicatesReport, DocumentDuplicatesRequest, DocumentGetReport, DocumentGetRequest,
    DocumentPruneReport, DocumentPruneRequest, DuplicateChunkCluster, DuplicateChunkMember,
    MarkExtractedReport, MarkExtractedRequest, PromotionCandidate, PromotionCandidatesReport,
    PromotionCandidatesRequest,
};

mod archive;
pub use archive::{backup_store, export_store, import_store};
pub(crate) use archive::rebuild_fts;
pub(crate) use archive::JsonValidator;
pub(crate) use validate::reject_existing_output_sidecars;

mod candidates;
pub use candidates::{
    adjudication_guard, approve_candidate, list_candidates, quarantine_candidate,
    reject_candidate, submit_candidate, AdjudicationGuard,
};
pub(crate) use candidates::capture_require_adjudication;

mod memory;
pub use memory::{
    forget_memory, get_memory, memory_history, remember_memory, verify_memory,
};
pub(crate) use memory::{
    forget_memory_tx, load_memory, remember_memory_tx, upsert_memory_entity_projection,
    ensure_memory_in_space, ensure_source_episode_exists, normalized_alias, memory_fts_metadata_text,
    correction_event_data_json,
};

mod search;
pub use search::{batch_search_memories, list_memories, search_memories};
pub(crate) use search::{
    batch_search_memories_on_connection, list_memories_on_connection, prepare_memory_list_request,
    prepare_search_request, search_memories_on_connection, search_prepared, split_tags,
    resolve_space_filter, ALIAS_TAG_PREFIX, ALIAS_MATCH_BOOST, compare_candidates, freshness_marker,
    now_julian_day, search_terms, SqlArgs, prepare_recall_filters, memory_ids_matching_filters,
    filters_where_clause, search_term_stems, push_unique, is_prefixable_search_term, query_alias_words,
    fts_score, recency_score_for_silo, source_tier_score, VOLATILE_MAX_RECENCY_SCORE,
    query_alias_shingles,
};
#[cfg(feature = "semantic")]
pub(crate) use search::{embedding_json, score_semantic_candidate, semantic_candidates};

mod pack;
pub use pack::{
    assemble_reranked_pack, build_hybrid_rerank_pool, build_hybrid_rerank_pool_trace_with_evidence_options,
    build_hybrid_rerank_pool_with_evidence_options, build_pack, build_pack_pool, empty_pack,
    RerankCandidate,
};
pub(crate) use pack::{
    format_pack_markdown, exact_entities_for_span, apply_graph_admission_observations,
    merge_rerank_pools, merge_rerank_pools_with_trace, interleave_pools, evidence_query_spans,
    evidence_join_filters, evidence_entity_seeds, allocate_evidence_seeds,
    record_evidence_candidate, evidence_candidate_order,
    AdmissionObservation, AdmissionSource, GraphSeedSource, GraphEvidenceClass,
    GraphRouteObservation, EvidenceGraphSeed, EvidenceCandidateRoutes, EvidenceQuerySpan,
    PackPoolItem, RerankPoolObservedCandidate, MAX_EVIDENCE_ENTITY_SPANS,
};

mod vectors;
pub use vectors::{
    apply_token_embeddings, collect_token_backfill_targets,
};
#[cfg(feature = "semantic")]
pub use vectors::{apply_reembed, collect_reembed_targets, reindex_vectors, ReembedTarget};
pub(crate) use vectors::{
    blob_to_token_vecs, enforce_active_colbert_model, load_token_embeddings,
    load_token_embeddings_cached, maxsim_candidates, maxsim_score, read_config_value,
    set_config_value, token_vecs_to_blob, upsert_memory_token_embedding,
};
#[cfg(feature = "semantic")]
pub(crate) use vectors::{
    blob_to_embedding, embedding_to_blob, enforce_active_embedding_model, insert_memory_embedding,
    maxsim_shortlist_ids, pack_maxsim_shortlist, rebuild_vector_index, write_embedding_row,
};

pub(crate) const MAX_CAPTURE_ENTITIES: usize = 32;
pub(crate) const MAX_CAPTURE_RELATIONSHIPS: usize = 64;

/// Suggested user-scoped store path for hosts that intentionally choose a user-global store.
pub const USER_STORE_PATH_HINT: &str = "<user-data-dir>/memkeeper/store.sqlite";

/// Project/workspace-local store path used by the Pi adapter when no env override is set.
pub const PROJECT_STORE_RELATIVE_PATH: &str = ".memkeeper/store.sqlite";

/// Busy timeout used by deterministic local store operations.
pub(crate) const BUSY_TIMEOUT_MS: u64 = 5_000;

/// Maximum accepted explicit memory content size in Unicode scalar values.
pub const MAX_CONTENT_CHARS: usize = 131_072;

/// Maximum accepted memory summary size in Unicode scalar values.
pub const MAX_SUMMARY_CHARS: usize = 8_192;

/// Maximum accepted tag count for one memory.
pub const MAX_TAGS: usize = 64;

/// Maximum accepted tag size in Unicode scalar values.
pub const MAX_TAG_CHARS: usize = 128;

/// Maximum accepted canonical source/provenance JSON size in Unicode scalar values.
pub const MAX_SOURCE_REF_JSON_CHARS: usize = 32_768;

/// Maximum accepted space/silo/project/entity/claim/id string size.
pub const MAX_METADATA_VALUE_CHARS: usize = 256;

/// Maximum accepted timestamp string size.
pub const MAX_TIMESTAMP_CHARS: usize = 64;

/// Maximum explicit memory links accepted in one remember request.
pub const MAX_MEMORY_LINKS: usize = 128;

/// Maximum returned results for one deterministic search.
pub const MAX_SEARCH_LIMIT: usize = 50;

/// Maximum returned rows for one deterministic memory review/list call.
pub const MAX_MEMORY_LIST_LIMIT: usize = 100;

/// Maximum returned rows for one deterministic entity search call.
pub const MAX_ENTITY_SEARCH_LIMIT: usize = 50;

/// Maximum accepted aliases on one deterministic entity upsert.
pub const MAX_ENTITY_ALIASES: usize = 64;

/// Maximum recursive graph traversal depth accepted by `graph-neighbors`.
pub(crate) const MAX_GRAPH_NEIGHBOR_DEPTH: usize = 4;

/// Maximum relationship edges returned by one `graph-neighbors` call.
pub const MAX_GRAPH_NEIGHBOR_EDGES: usize = 200;

/// Maximum accepted search offset.
pub const MAX_SEARCH_OFFSET: usize = 1_000;

/// Maximum deterministic recency boost added to a search score.
pub const MAX_RECENCY_SCORE: f64 = 0.05;

/// Half-life in days of the durable-silo recency boost. Durable facts decay
/// gently: a 6-month-old memory keeps half its (small) freshness boost.
pub(crate) const DURABLE_RECENCY_HALF_LIFE_DAYS: f64 = 180.0;

/// Half-life in days of the volatile-silo recency boost. Volatile claims
/// decay fast so a recent claim decisively outranks a stale one.
pub(crate) const VOLATILE_RECENCY_HALF_LIFE_DAYS: f64 = 30.0;

/// Maximum emitted snippet length in Unicode scalar values.
pub const MAX_SNIPPET_CHARS: usize = 1_000;

/// Maximum accepted search query size in Unicode scalar values.
pub const MAX_SEARCH_QUERY_CHARS: usize = 4_096;

/// Maximum accepted sanitized search terms.
pub const MAX_SEARCH_TERMS: usize = 64;

/// Maximum deterministic batch-search queries in one request.
pub(crate) const MAX_BATCH_QUERIES: usize = 20;

/// Maximum deterministic batch-search results per query.
pub const MAX_BATCH_QUERY_LIMIT: usize = 20;

/// Maximum memories included in one deterministic prompt pack.
pub const MAX_PACK_MEMORIES: usize = 50;

/// Maximum emitted prompt pack size in Unicode scalar values.
pub const MAX_PACK_CHARS: usize = 10_000;

/// Maximum prompt pack title size in Unicode scalar values.
pub const MAX_PACK_TITLE_CHARS: usize = 256;

/// Maximum spaces returned by one `space-list` call.
pub const MAX_SPACE_LIST_LIMIT: usize = 50;

/// Maximum silos returned by one `silo-list` call.
pub const MAX_SILO_LIST_LIMIT: usize = 100;

/// Maximum returned history versions/events for one memory.
pub const MAX_HISTORY_LIMIT: usize = 500;

/// Maximum returned links for one memory get response.
pub const MAX_GET_LINKS: usize = MAX_MEMORY_LINKS * 4;

/// Maximum accepted forget reason size in Unicode scalar values.
pub const MAX_FORGET_REASON_CHARS: usize = 2_048;

/// Maximum accepted logical import line size in bytes.
pub const MAX_IMPORT_LINE_BYTES: usize = 16 * 1024 * 1024;

/// Maximum JSON nesting accepted while parsing logical import files.
pub(crate) const MAX_IMPORT_JSON_DEPTH: usize = 64;

/// Maximum JSON object fields accepted in one logical import record.
pub const MAX_IMPORT_JSON_OBJECT_FIELDS: usize = 512;

/// Maximum JSON array items accepted in one logical import record.
pub(crate) const MAX_IMPORT_JSON_ARRAY_ITEMS: usize = 2_048;

/// Default maximum memories scanned by one explicit maintenance/dream run.
pub const DEFAULT_DREAM_MAX_MEMORIES: usize = 1_000;

/// Default number of `retrieved` recall events that promotes a short-term
/// memory to the durable silo. Overridable per dream run via
/// `DreamRequest::promote_threshold`.
pub const DEFAULT_PROMOTE_THRESHOLD: usize = 3;

/// Default minimum per-memory rerank score for a recall event to count toward
/// promotion. Calibrated to the live cross-encoder distribution (measured
/// 2026-06-11): strong/relevant matches land ~0.6-0.98, noise sits below ~0.3,
/// with a clean gap ~0.3-0.6. A floor of 0.5 sits in that gap -- it counts
/// genuine matches while excluding noise. Overridable per run.
pub const DEFAULT_PROMOTE_SCORE_FLOOR: f64 = 0.5;

/// Default maximum rank (position in the injected set) for a recall event to
/// count toward promotion.
pub const DEFAULT_PROMOTE_RANK_CAP: usize = 3;

/// Silo a memory is promoted *from*. Memories in any non-durable silo decay on
/// the volatile curve; `short-term` is the seeded volatile silo.
pub(crate) const SHORT_TERM_SILO: &str = "short-term";

/// Maximum memories scanned by one explicit maintenance/dream run.
pub const MAX_DREAM_MAX_MEMORIES: usize = 10_000;

/// Maximum duplicate ids emitted per duplicate proposal.
pub(crate) const MAX_DREAM_DUPLICATE_IDS: usize = 20;

/// Maximum duplicate/update candidates returned by one remember preflight.
pub const MAX_REMEMBER_CANDIDATES: usize = 5;

/// Maximum active memories scanned by lexical remember candidate detection.
pub(crate) const MAX_REMEMBER_LEXICAL_SCAN: usize = 50;

/// Maximum query terms used by lexical remember candidate detection.
pub(crate) const MAX_REMEMBER_LEXICAL_TERMS: usize = 12;

/// Default vector dimension for `mxbai-embed-large`.
pub const DEFAULT_SEMANTIC_EMBEDDING_DIMS: usize = 1024;

/// Maximum supported semantic embedding dimension. A model may use any dimension
/// up to this bound; the vector index table is created per-dimension.
pub const MAX_SEMANTIC_EMBEDDING_DIMS: usize = 8192;

/// Maximum same-claim conflict candidates returned by one remember operation.
pub(crate) const MAX_REMEMBER_CONFLICT_CANDIDATES: usize = 5;

/// Minimum token Jaccard overlap for lexical remember candidate detection.
pub(crate) const REMEMBER_LEXICAL_THRESHOLD: f64 = 0.35;

/// Result type for store operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Store-layer errors with enough structure for host adapters to map to stable protocol errors.
#[derive(Debug)]
pub enum Error {
    /// Filesystem failure while preparing or inspecting a store path.
    Io(std::io::Error),
    /// `SQLite` failure while opening, migrating, or querying a store.
    Database(rusqlite::Error),
    /// Input failed store-level validation.
    InvalidRequest {
        /// Safe diagnostic message.
        message: String,
    },
    /// The path does not point to an initialized memkeeper store.
    NotInitialized {
        /// Store path that failed initialization validation.
        path: PathBuf,
    },
    /// Store path is not acceptable for a durable local database.
    InvalidPath {
        /// Rejected store path.
        path: PathBuf,
        /// Safe explanation for the rejection.
        reason: &'static str,
    },
    /// Existing database is not recognized as a memkeeper store, so init refuses to mutate it.
    UnsafeExistingDatabase {
        /// Existing database path that was refused.
        path: PathBuf,
    },
    /// Store schema is not supported by this binary.
    SchemaMismatch {
        /// Schema version expected by this binary.
        expected: i32,
        /// Schema version found in the database.
        actual: i32,
    },
    /// `SQLite` could not enable the required durable WAL journal mode.
    WalUnavailable {
        /// Store path that could not use WAL.
        path: PathBuf,
        /// Journal mode returned by `SQLite`.
        journal_mode: String,
    },
    /// Requested store object does not exist.
    NotFound {
        /// Object kind, such as `memory`, `space`, or `silo`.
        entity: &'static str,
        /// Requested object id/name.
        id: String,
    },
    /// Request conflicts with current store state or policy.
    Conflict {
        /// Safe diagnostic message.
        message: String,
    },
}

impl Error {
    /// Return whether the underlying database reported a lock/busy condition.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        matches!(
            self,
            Self::Database(rusqlite::Error::SqliteFailure(error, _))
                if matches!(
                    error.code,
                    rusqlite::ErrorCode::DatabaseBusy | rusqlite::ErrorCode::DatabaseLocked
                )
        )
    }

    /// Return whether a retry may succeed without changing the request.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        self.is_locked()
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "filesystem error: {error}"),
            Self::Database(error) => write!(formatter, "SQLite error: {error}"),
            Self::InvalidRequest { message } | Self::Conflict { message } => {
                formatter.write_str(message)
            }
            Self::NotInitialized { path } => {
                write!(formatter, "store is not initialized: {}", path.display())
            }
            Self::InvalidPath { path, reason } => {
                write!(formatter, "invalid store path {}: {reason}", path.display())
            }
            Self::UnsafeExistingDatabase { path } => write!(
                formatter,
                "refusing to initialize non-memkeeper existing database: {}",
                path.display()
            ),
            Self::SchemaMismatch { expected, actual } => write!(
                formatter,
                "schema mismatch: expected version {expected}, found version {actual}"
            ),
            Self::WalUnavailable { path, journal_mode } => write!(
                formatter,
                "WAL journal mode unavailable for {}: got {journal_mode}",
                path.display()
            ),
            Self::NotFound { entity, id } => write!(formatter, "{entity} not found: {id}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Database(error) => Some(error),
            Self::InvalidRequest { .. }
            | Self::NotInitialized { .. }
            | Self::InvalidPath { .. }
            | Self::UnsafeExistingDatabase { .. }
            | Self::SchemaMismatch { .. }
            | Self::WalUnavailable { .. }
            | Self::NotFound { .. }
            | Self::Conflict { .. } => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<rusqlite::Error> for Error {
    fn from(error: rusqlite::Error) -> Self {
        Self::Database(error)
    }
}

mod types;
pub use types::*;

/// Store one explicit memory and update deterministic indexes atomically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the referenced space/silo/source/memory does not exist, or `SQLite` rejects the
/// transaction.
pub const MAX_RECALL_EVENTS: usize = 100;

/// One recall telemetry event: a memory either surfaced in search results or
/// explicitly retrieved by id.
#[derive(Debug, Clone, PartialEq)]
pub struct RecallEvent {
    /// Memory id the event refers to (not required to still exist).
    pub memory_id: String,
    /// Event kind: `surfaced` (appeared in results) or `retrieved` (fetched by id).
    pub kind: String,
    /// Query that surfaced the memory, when applicable.
    pub query: Option<String>,
    /// Result rank, when applicable.
    pub rank: Option<usize>,
    /// Result score, when applicable.
    pub score: Option<f64>,
}

/// Engine-owned recall telemetry write: events plus an optional
/// `memories.accessed_at` touch for retrieved memories.
#[derive(Debug, Clone, PartialEq)]
pub struct RecallLogRequest {
    /// Recall source label (e.g. the host adapter name).
    pub source: Option<String>,
    /// Optional session/conversation id, written to every event in this batch.
    pub session_id: Option<String>,
    /// Optional batch id shared by every event in this recall-log write.
    pub batch_id: Option<String>,
    /// Optional caller-observed retrieval latency for this recall batch.
    pub latency_ms: Option<f64>,
    /// Optional label describing what `latency_ms` measured.
    pub latency_source: Option<String>,
    /// Events to record (1..=`MAX_RECALL_EVENTS`).
    pub events: Vec<RecallEvent>,
    /// Update `memories.accessed_at` for `retrieved` events.
    pub touch_accessed: bool,
}

/// Result of one `record_recall` call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecallLogReport {
    /// Events written to `recall_events`.
    pub recorded: usize,
    /// Memories whose `accessed_at` was updated.
    pub touched: usize,
}

/// Accepted candidate provenance/source types, lowest-trust last.
pub(crate) const CANDIDATE_SOURCE_TYPES: &[&str] = &[
    "explicit-user",
    "auto-harvest",
    "import",
    "docs",
    "test",
    "assistant-inference",
    // Capture write-path: subject to the adjudication gate (see `adjudication_guard`).
    "capture",
];

/// Accepted candidate sensitivity labels.
pub(crate) const CANDIDATE_SENSITIVITIES: &[&str] = &["normal", "sensitive"];

/// Accepted candidate lifecycle statuses. `quarantined` is a terminal state for
/// capture candidates an adjudicator flagged (unsupported/distorted); it is
/// distinct from `rejected` so a human can review adjudicator calls.
pub(crate) const CANDIDATE_STATUSES: &[&str] = &["pending", "approved", "rejected", "quarantined"];

/// Accepted `remember` supersession modes (how a write resolves against active
/// memories sharing its `entity_key` + `claim_key`). `auto` is the default and
/// preserves the historical policy.
pub const REMEMBER_SUPERSEDE_MODES: &[&str] =
    &["auto", "append", "supersede", "suggest", "conflict"];

/// Default supersession mode: the historical same entity/claim policy.
pub const REMEMBER_MODE_AUTO: &str = "auto";

pub(crate) const CANDIDATE_STATUS_PENDING: &str = "pending";
pub(crate) const CANDIDATE_STATUS_APPROVED: &str = "approved";
pub(crate) const CANDIDATE_STATUS_REJECTED: &str = "rejected";
pub(crate) const CANDIDATE_STATUS_QUARANTINED: &str = "quarantined";
pub(crate) const DEFAULT_CANDIDATE_SOURCE_TYPE: &str = "assistant-inference";
pub(crate) const DEFAULT_CANDIDATE_SENSITIVITY: &str = "normal";
/// `source_type` marking a candidate that came from the capture write-path; only
/// these are subject to the adjudication gate.
pub(crate) const CANDIDATE_SOURCE_CAPTURE: &str = "capture";

/// Default and maximum rows returned by one `candidate-list` call.
pub(crate) const DEFAULT_CANDIDATE_LIST_LIMIT: usize = 50;

// Candidate memories: a review queue for not-yet-trusted writes.
//
// Candidates are engine-owned operational data. Like `recall_events`, the
// `memory_candidates` table is created lazily (CREATE TABLE IF NOT EXISTS at
// operation time) and is intentionally NOT part of schema validation, so
// pre-existing stores need no migration and the schema version is unchanged.
// Candidates never enter retrieval; approving one promotes it into `memories`
// through the normal remember write path (inheriting supersession/conflict
// handling), and stamps the resulting memory id back onto the candidate row.
// ---------------------------------------------------------------------------

/// SELECT column list for `memory_candidates`, matched 1:1 by `memory_candidate_from_row`.
pub(crate) const CANDIDATE_COLUMNS: &str = "id, status, space, silo, scope, project, kind, content, \
     summary, rationale, tags_json, entity_key, claim_key, confidence, source_type, \
     source_json, sensitivity, supersedes_json, created_at, decided_at, decided_reason, \
     resulting_memory_id";

/// Request to submit a candidate memory for review.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateSubmitRequest {
    /// Target space (applied when the candidate is approved).
    pub space: Option<String>,
    /// Target silo.
    pub silo: Option<String>,
    /// Target scope.
    pub scope: Option<String>,
    /// Project key.
    pub project: Option<String>,
    /// Memory kind.
    pub kind: Option<String>,
    /// Candidate content.
    pub content: String,
    /// Optional summary.
    pub summary: Option<String>,
    /// Why this candidate was proposed (preserved into the memory on approval).
    pub rationale: Option<String>,
    /// Tags to attach on approval.
    pub tags: Vec<String>,
    /// Stable entity key.
    pub entity_key: Option<String>,
    /// Stable claim key.
    pub claim_key: Option<String>,
    /// Confidence 0.0-1.0.
    pub confidence: f64,
    /// Provenance/source type (one of `CANDIDATE_SOURCE_TYPES`).
    pub source_type: Option<String>,
    /// Optional canonical JSON source/provenance object.
    pub source_json: Option<String>,
    /// Sensitivity label (one of `CANDIDATE_SENSITIVITIES`).
    pub sensitivity: Option<String>,
    /// Memory ids this candidate would supersede on approval.
    pub supersedes: Vec<String>,
    /// Validate and return a report without writing.
    pub dry_run: bool,
}

/// A stored candidate memory.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRecord {
    /// Candidate id (`cand_...`).
    pub id: String,
    /// Lifecycle status: pending, approved, or rejected.
    pub status: String,
    /// Target space.
    pub space: Option<String>,
    /// Target silo.
    pub silo: Option<String>,
    /// Target scope.
    pub scope: Option<String>,
    /// Project key.
    pub project: Option<String>,
    /// Memory kind.
    pub kind: Option<String>,
    /// Candidate content.
    pub content: String,
    /// Optional summary.
    pub summary: Option<String>,
    /// Rationale for the proposal.
    pub rationale: Option<String>,
    /// Tags.
    pub tags: Vec<String>,
    /// Entity key.
    pub entity_key: Option<String>,
    /// Claim key.
    pub claim_key: Option<String>,
    /// Confidence 0.0-1.0.
    pub confidence: f64,
    /// Provenance/source type.
    pub source_type: String,
    /// Canonical JSON source/provenance object.
    pub source_json: Option<String>,
    /// Sensitivity label.
    pub sensitivity: String,
    /// Memory ids this candidate would supersede.
    pub supersedes: Vec<String>,
    /// Submission timestamp.
    pub created_at: String,
    /// Decision (approve/reject) timestamp, when decided.
    pub decided_at: Option<String>,
    /// Reason recorded at rejection.
    pub decided_reason: Option<String>,
    /// Memory id created when the candidate was approved.
    pub resulting_memory_id: Option<String>,
}

/// Result of submitting a candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateSubmitReport {
    /// The stored (or dry-run) candidate.
    pub candidate: CandidateRecord,
    /// True when validated but rolled back.
    pub dry_run: bool,
}

/// Default space for ingested document chunks, isolated from the curated tier so
/// supersession/dedup/graph/promotion never touch raw document content.
pub const DOCUMENTS_SPACE: &str = "documents";

/// Default `source_type` recorded on ingested document chunks.
pub(crate) const DEFAULT_INGEST_SOURCE_TYPE: &str = "import";

/// Request to ingest one document source as embedded-ready, isolated chunks.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IngestRequest {
    /// Target space (defaults to [`DOCUMENTS_SPACE`]); kept isolated from curated memory.
    pub space: Option<String>,
    /// Provenance type recorded on each chunk (defaults to `import`).
    pub source_type: Option<String>,
    /// Filesystem path of the source document (for citations and re-sync).
    pub source_path: Option<String>,
    /// URI of the source document, when it is not a local path.
    pub source_uri: Option<String>,
    /// Human-readable description of the source.
    pub source_description: Option<String>,
    /// Optional canonical JSON metadata stored on each chunk.
    pub metadata_json: Option<String>,
    /// Ordered chunk contents for this source; `chunk_index`/`chunk_count` are derived.
    pub chunks: Vec<String>,
    /// Optional per-chunk embedding vectors, parallel to `chunks` (caller-computed,
    /// same model as query embeddings). Populated by the CLI/daemon embed step, not
    /// from the wire payload.
    pub embeddings: Option<Vec<Vec<f32>>>,
    /// Embedding model id for `embeddings`; required when embeddings are supplied.
    pub embedding_model_id: Option<String>,
    /// Validate and report without writing.
    pub dry_run: bool,
}

/// Result of ingesting a document source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestReport {
    /// Space the chunks were written to.
    pub space: String,
    /// Source path echoed back (when supplied).
    pub source_path: Option<String>,
    /// Total chunks supplied in the request.
    pub chunk_count: usize,
    /// `source_episodes` ids created by this call.
    pub created: Vec<String>,
    /// Chunks skipped because identical content already exists in the space (dedup).
    pub skipped: usize,
    /// True when this call created the target space.
    pub created_space: bool,
    /// True when validated but rolled back.
    pub dry_run: bool,
}

/// Default result limit for document-chunk search.
pub const DEFAULT_DOCUMENT_SEARCH_LIMIT: usize = 10;

/// Request for hybrid (BM25 + vector) search over ingested document chunks.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocumentSearchRequest {
    /// Free-text query.
    pub query: String,
    /// Space to search (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// Max results (defaults to [`DEFAULT_DOCUMENT_SEARCH_LIMIT`]).
    pub limit: usize,
    /// Include full chunk content in each result (else snippet only).
    pub include_content: bool,
    /// Snippet length in characters (0 = no snippet).
    pub snippet_chars: usize,
    /// Query embedding (CLI/daemon-computed); enables the semantic arm.
    pub embedding: Option<Vec<f32>>,
    /// Skip retrieval instrumentation for this search (e.g. eval/benchmark runs
    /// that must not pollute the promotion signal). Default `false` = log.
    pub skip_recall_log: bool,
}

/// One matched document chunk, with a citation back to its source.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSearchResult {
    /// 1-based result rank.
    pub rank: usize,
    /// `source_episodes` id of the chunk.
    pub source_episode_id: String,
    /// Space the chunk lives in.
    pub space: String,
    /// Provenance type.
    pub source_type: String,
    /// Source document path (citation).
    pub source_path: Option<String>,
    /// Source document URI (citation).
    pub source_uri: Option<String>,
    /// Chunk position within the source.
    pub chunk_index: i64,
    /// Total chunks in the source.
    pub chunk_count: i64,
    /// Snippet of the chunk content.
    pub snippet: String,
    /// Full chunk content (when requested).
    pub content: Option<String>,
    /// Fused relevance score (Reciprocal Rank Fusion).
    pub score: f64,
    /// Which arms matched: `hybrid`, `semantic`, or `lexical`.
    pub match_type: String,
}

/// Result of a document-chunk search.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentSearchReport {
    /// Retrieval strategy: `hybrid_rrf_v0` or `lexical_only_v0`.
    pub strategy: String,
    /// True when a semantic arm was attempted (query embedding present).
    pub semantic_attempted: bool,
    /// Space searched.
    pub space: String,
    /// Ranked chunk matches.
    pub results: Vec<DocumentSearchResult>,
}

/// Request to list candidates for review.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateListRequest {
    /// Filter by status (pending/approved/rejected); None lists all.
    pub status: Option<String>,
    /// Filter by target space.
    pub space: Option<String>,
    /// Max rows (capped at `DEFAULT_CANDIDATE_LIST_LIMIT` * 2).
    pub limit: usize,
    /// Row offset.
    pub offset: usize,
}

/// Result of listing candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateListReport {
    /// Matching candidates (newest first).
    pub candidates: Vec<CandidateRecord>,
    /// Total candidates matching the filters (ignoring limit/offset).
    pub total: usize,
}

/// Request to approve a candidate, promoting it into a real memory.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateApproveRequest {
    /// Candidate id to approve.
    pub id: String,
    /// Optional precomputed embedding for the promoted memory.
    pub embedding: Option<Vec<f32>>,
    /// Embedding model id for `embedding`.
    pub embedding_model_id: Option<String>,
    /// Validate and return a report without writing.
    pub dry_run: bool,
}

/// Result of approving a candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateApproveReport {
    /// The updated candidate (status=approved, `resulting_memory_id` set).
    pub candidate: CandidateRecord,
    /// The memory created from the candidate.
    pub memory: MemoryRecord,
    /// True when validated but rolled back.
    pub dry_run: bool,
}

/// Request to reject a candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateRejectRequest {
    /// Candidate id to reject.
    pub id: String,
    /// Optional reason recorded on the candidate.
    pub reason: Option<String>,
    /// Validate and return a report without writing.
    pub dry_run: bool,
}

/// Result of rejecting a candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRejectReport {
    /// The updated candidate (status=rejected).
    pub candidate: CandidateRecord,
    /// True when validated but rolled back.
    pub dry_run: bool,
}

/// Request to quarantine a candidate (an adjudicator flagged it). Distinct from
/// reject so the call remains human-reviewable.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateQuarantineRequest {
    /// Candidate id to quarantine.
    pub id: String,
    /// Optional reason (e.g. the adjudicator finding) recorded on the candidate.
    pub reason: Option<String>,
    /// Validate and return a report without writing.
    pub dry_run: bool,
}

/// Result of quarantining a candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateQuarantineReport {
    /// The updated candidate (status=quarantined).
    pub candidate: CandidateRecord,
    /// True when validated but rolled back.
    pub dry_run: bool,
}

pub fn dream_store(path: impl AsRef<Path>, request: &DreamRequest) -> Result<DreamReport> {
    validate_dream_request(request)?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = dream_store_tx(&transaction, request)?;

    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }

    Ok(report)
}

pub(crate) fn collapse_whitespace(value: &str) -> String {
    let mut output = String::new();
    let mut previous_was_space = false;
    for character in value.chars() {
        if character.is_whitespace() {
            if !previous_was_space {
                output.push(' ');
                previous_was_space = true;
            }
        } else {
            output.push(character);
            previous_was_space = false;
        }
    }
    output.trim().to_string()
}

pub(crate) fn make_snippet(content: &str, terms: &[String], max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let lower = content.to_ascii_lowercase();
    let first_match = terms
        .iter()
        .filter_map(|term| lower.find(term))
        .min()
        .unwrap_or(0);
    let prefix_chars = content[..first_match.min(content.len())].chars().count();
    let half = max_chars / 2;
    let start_char = prefix_chars.saturating_sub(half);
    bounded_char_slice(content, start_char, max_chars)
}

pub(crate) fn bounded_char_slice(value: &str, start_char: usize, max_chars: usize) -> String {
    value.chars().skip(start_char).take(max_chars).collect()
}

pub(crate) fn is_supported_entity_status(value: &str) -> bool {
    matches!(value, "active" | "merged" | "tombstoned")
}

pub(crate) fn is_supported_relationship_status(value: &str) -> bool {
    matches!(
        value,
        status::ACTIVE | status::SUPERSEDED | status::CONFLICTED | status::TOMBSTONED
    )
}
#[cfg(not(feature = "semantic"))]
#[allow(clippy::unnecessary_wraps)]
fn insert_memory_embedding(
    _transaction: &Transaction<'_>,
    _memory_id: &str,
    _version_id: &str,
    _now: &str,
    _embedding: Option<&[f32]>,
    _model_id: Option<&str>,
) -> Result<()> {
    Ok(())
}
pub(crate) fn reject_source_sidecar_output(source_store: &Path, output_path: &Path) -> Result<()> {
    let source_store = normalize_path_lexically(&fs::canonicalize(source_store)?);
    let output_path = normalized_output_path(output_path)?;
    let output_key = path_identity_key(&output_path);
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = normalize_path_lexically(&sidecar_path(&source_store, suffix));
        let sidecar_key = path_identity_key(&sidecar);
        let nested_sidecar_key = format!("{sidecar_key}{}", std::path::MAIN_SEPARATOR);
        if output_path == sidecar
            || output_path.starts_with(&sidecar)
            || output_key == sidecar_key
            || output_key.starts_with(&nested_sidecar_key)
        {
            return Err(Error::Conflict {
                message: format!(
                    "output path is reserved for the source store SQLite sidecar: {}",
                    output_path.display()
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn normalized_output_path(path: &Path) -> Result<PathBuf> {
    let absolute = normalize_path_lexically(&if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    });
    if absolute.exists() {
        return fs::canonicalize(&absolute)
            .map(|path| normalize_path_lexically(&path))
            .map_err(Into::into);
    }

    let mut missing_components = Vec::new();
    let mut cursor = absolute.as_path();
    loop {
        if cursor.exists() {
            let mut combined = fs::canonicalize(cursor)?;
            for component in missing_components.iter().rev() {
                combined.push(component);
            }
            return Ok(normalize_path_lexically(&combined));
        }
        let Some(file_name) = cursor.file_name() else {
            return Ok(normalize_path_lexically(&absolute));
        };
        missing_components.push(file_name.to_os_string());
        let Some(parent) = cursor.parent() else {
            return Ok(normalize_path_lexically(&absolute));
        };
        cursor = parent;
    }
}

pub(crate) fn normalize_path_lexically(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            std::path::Component::RootDir => normalized.push(component.as_os_str()),
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                let _ = normalized.pop();
            }
            std::path::Component::Normal(part) => normalized.push(part),
        }
    }
    normalized
}

pub(crate) fn path_identity_key(path: &Path) -> String {
    path.to_string_lossy().to_lowercase()
}

pub(crate) fn create_temp_output_file(destination: &Path) -> Result<(PathBuf, File)> {
    validate_output_path(destination)?;
    create_parent_dirs(destination)?;
    let parent = destination
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let file_name = destination.file_name().ok_or_else(|| Error::InvalidPath {
        path: destination.to_path_buf(),
        reason: "output path must include a file name",
    })?;
    let file_name = file_name.to_string_lossy();
    for _ in 0..16 {
        let temp_path = parent.join(format!(
            ".{file_name}.memkeeper-tmp-{}-{}-{}",
            process::id(),
            unique_nanos(),
            ID_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        match create_new_private_file(&temp_path, true) {
            Ok(file) => return Ok((temp_path, file)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(Error::Io(error)),
        }
    }
    Err(Error::InvalidPath {
        path: destination.to_path_buf(),
        reason: "could not create a temporary output file",
    })
}

pub(crate) fn publish_temp_output(temp_path: &Path, destination: &Path) -> Result<()> {
    match fs::hard_link(temp_path, destination) {
        Ok(()) => match fs::remove_file(temp_path) {
            Ok(()) => Ok(()),
            Err(error) => {
                let _ = fs::remove_file(destination);
                Err(Error::Io(error))
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            cleanup_temp_output(temp_path);
            Err(Error::Conflict {
                message: format!("output path already exists: {}", destination.display()),
            })
        }
        Err(error) => {
            cleanup_temp_output(temp_path);
            Err(Error::Io(error))
        }
    }
}

pub(crate) fn reject_output_sidecar_files(path: &Path) -> Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = sidecar_path(path, suffix);
        if sidecar.exists() {
            return Err(Error::Conflict {
                message: format!(
                    "backup produced SQLite sidecar {}; refusing non-self-contained backup",
                    sidecar.display()
                ),
            });
        }
    }
    Ok(())
}


pub(crate) fn cleanup_temp_output(path: &Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(sidecar_path(path, "-wal"));
    let _ = fs::remove_file(sidecar_path(path, "-shm"));
    let _ = fs::remove_file(sidecar_path(path, "-journal"));
}

pub(crate) fn create_new_private_file(path: &Path, read: bool) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    if read {
        options.read(true);
    }
    apply_private_file_mode(&mut options);
    options.open(path)
}

#[cfg(unix)]
fn apply_private_file_mode(options: &mut OpenOptions) {
    use std::os::unix::fs::OpenOptionsExt;
    options.mode(0o600);
}

#[cfg(not(unix))]
fn apply_private_file_mode(_options: &mut OpenOptions) {}

pub(crate) fn create_parent_dirs(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

pub(crate) fn count(connection: &Connection, sql: &str) -> Result<i64> {
    connection
        .query_row(sql, [], |row| row.get(0))
        .map_err(Into::into)
}

#[cfg(test)]
mod tests;
