#![cfg_attr(not(feature = "semantic"), forbid(unsafe_code))]

//! Store boundary for the local `SQLite` canonical memory database.
//!
//! The store crate owns initialization, schema validation, and deterministic
//! store diagnostics. It intentionally exposes structured results instead of a
//! host-specific JSON shape so CLI, Pi, MCP, and future adapters can share the
//! same semantics.

use std::{
    collections::{BTreeMap, BTreeSet},
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
    infer_kind_from_prefix, kind, scope, status, DEFAULT_DURABLE_SILO, DEFAULT_SPACE,
};
use rusqlite::{
    backup, params, params_from_iter,
    types::{Value, ValueRef},
    Connection, OpenFlags, OptionalExtension, Row, Transaction,
};

#[cfg(feature = "semantic")]
use rusqlite::auto_extension::{register_auto_extension, RawAutoExtension};

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

mod recall;
pub use recall::record_recall;
pub(crate) use recall::{ensure_recall_events_session_column, RECALL_EVENTS_DDL};

mod stats;
pub(crate) use stats::space_names;
pub use stats::{inspect_store_stats, last_synthesis_run, store_stats, store_stats_with_health};

#[cfg(feature = "semantic")]
static SQLITE_VEC_EXTENSION_REGISTERED: std::sync::OnceLock<std::result::Result<(), String>> =
    std::sync::OnceLock::new();

/// Current `SQLite` schema version expected by the store/migration layer.
pub const SCHEMA_VERSION: i32 = 5;

/// `SQLite` schema draft used for v0.1 store initialization.
pub(crate) const SCHEMA_SQL: &str = include_str!("../../../schema-v0.1.sql");

#[cfg(feature = "semantic")]
const SCHEMA_UPGRADE_SQL: &str = "
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (2, 'schema-v0.2', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (3, 'schema-v0.3-semantic-1536', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (4, 'schema-v0.4-semantic-1024', CURRENT_TIMESTAMP, NULL);
DROP TABLE IF EXISTS memory_vec_1536;
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (5, 'schema-v0.5-token-embeddings', CURRENT_TIMESTAMP, NULL);
CREATE TABLE IF NOT EXISTS memory_token_embeddings (
  memory_id TEXT PRIMARY KEY,
  embedding_model TEXT NOT NULL,
  dims INTEGER NOT NULL,
  n_tokens INTEGER NOT NULL,
  vector_blob BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
UPDATE config_kv SET value = '5', updated_at = CURRENT_TIMESTAMP WHERE key = 'schema_version' AND value != '5';
PRAGMA user_version = 5;
";

#[cfg(feature = "semantic")]
const SEMANTIC_TABLE_SQL: &str = "
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec_1024 USING vec0(
  memory_id TEXT NOT NULL,
  embedding FLOAT[1024]
);
";

#[cfg(not(feature = "semantic"))]
const SCHEMA_UPGRADE_SQL: &str = "
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (2, 'schema-v0.2', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (3, 'schema-v0.3', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (4, 'schema-v0.4', CURRENT_TIMESTAMP, NULL);
INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (5, 'schema-v0.5-token-embeddings', CURRENT_TIMESTAMP, NULL);
CREATE TABLE IF NOT EXISTS memory_token_embeddings (
  memory_id TEXT PRIMARY KEY,
  embedding_model TEXT NOT NULL,
  dims INTEGER NOT NULL,
  n_tokens INTEGER NOT NULL,
  vector_blob BLOB NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
UPDATE config_kv SET value = '5', updated_at = CURRENT_TIMESTAMP WHERE key = 'schema_version' AND value != '5';
PRAGMA user_version = 5;
";

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
const DURABLE_RECENCY_HALF_LIFE_DAYS: f64 = 180.0;

/// Half-life in days of the volatile-silo recency boost. Volatile claims
/// decay fast so a recent claim decisively outranks a stale one.
const VOLATILE_RECENCY_HALF_LIFE_DAYS: f64 = 30.0;

/// Julian day for the Unix epoch, used to derive "now" as a Julian day.
const UNIX_EPOCH_JD: f64 = 2_440_587.5;

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
const SHORT_TERM_SILO: &str = "short-term";

/// Maximum memories scanned by one explicit maintenance/dream run.
pub const MAX_DREAM_MAX_MEMORIES: usize = 10_000;

/// Maximum duplicate ids emitted per duplicate proposal.
pub(crate) const MAX_DREAM_DUPLICATE_IDS: usize = 20;

/// Maximum duplicate/update candidates returned by one remember preflight.
pub const MAX_REMEMBER_CANDIDATES: usize = 5;

/// Maximum active memories scanned by lexical remember candidate detection.
const MAX_REMEMBER_LEXICAL_SCAN: usize = 50;

/// Maximum query terms used by lexical remember candidate detection.
const MAX_REMEMBER_LEXICAL_TERMS: usize = 12;

/// Default vector dimension for `mxbai-embed-large`.
pub const DEFAULT_SEMANTIC_EMBEDDING_DIMS: usize = 1024;

/// Maximum supported semantic embedding dimension. A model may use any dimension
/// up to this bound; the vector index table is created per-dimension.
pub const MAX_SEMANTIC_EMBEDDING_DIMS: usize = 8192;

/// Maximum same-claim conflict candidates returned by one remember operation.
const MAX_REMEMBER_CONFLICT_CANDIDATES: usize = 5;

/// Minimum token Jaccard overlap for lexical remember candidate detection.
const REMEMBER_LEXICAL_THRESHOLD: f64 = 0.35;

/// Required table names that should exist in the v0.1 schema.
pub const REQUIRED_TABLES: &[&str] = &[
    "schema_migrations",
    "config_kv",
    "spaces",
    "silos",
    "source_episodes",
    "memories",
    "memory_versions",
    "memory_events",
    "memory_tags",
    "memory_links",
    "conflicts",
    "entities",
    "entity_aliases",
    "relationships",
    "processing_jobs",
    "embeddings",
    "dream_runs",
    "probe_runs",
];

/// Required FTS virtual table names.
pub const REQUIRED_FTS_TABLES: &[&str] = &["memory_fts", "memory_fts_public", "source_episode_fts"];

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExistingStoreInspection {
    Compatible,
    OlderSchema(i32),
    Unrecognized,
    FutureSchema(i32),
}

/// Return true if the embedded schema text contains the objects expected by the scaffold.
#[must_use]
pub fn schema_mentions_required_objects() -> bool {
    let base_schema_mentions_required_objects =
        REQUIRED_TABLES.iter().all(|name| SCHEMA_SQL.contains(name))
            && REQUIRED_FTS_TABLES
                .iter()
                .all(|name| SCHEMA_SQL.contains(name))
            && SCHEMA_SQL.contains(DEFAULT_SPACE)
            && SCHEMA_SQL.contains("PRAGMA user_version = 1");
    #[cfg(feature = "semantic")]
    {
        base_schema_mentions_required_objects
            && SEMANTIC_TABLE_SQL.contains("memory_vec_1024")
            && SCHEMA_UPGRADE_SQL.contains(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))
    }
    #[cfg(not(feature = "semantic"))]
    {
        base_schema_mentions_required_objects
            && SCHEMA_UPGRADE_SQL.contains(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))
    }
}

/// Initialize a local memkeeper store at `path`.
///
/// The operation creates parent directories, applies the v0.1 schema
/// idempotently, seeds the default workspace space/silos, and validates the
/// resulting schema version.
///
/// # Errors
///
/// Returns an error when the path is not durable, the parent directory cannot
/// be created, an existing non-memkeeper database would be mutated, the existing
/// schema is newer than this binary supports, WAL cannot be enabled, or
/// `SQLite` rejects the schema batch.
pub fn init_store(path: impl AsRef<Path>) -> Result<InitReport> {
    let path = path.as_ref();
    validate_store_path(path)?;
    create_parent_dirs(path)?;
    let created = claim_or_preflight_init_path(path)?;

    reject_sqlite_sidecar_symlinks(path)?;
    register_sqlite_vec_extension()?;
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )?;
    configure_connection(&connection)?;
    let enabled_journal_mode = enable_wal(&connection)?;
    if enabled_journal_mode != "wal" {
        return Err(Error::WalUnavailable {
            path: path.to_path_buf(),
            journal_mode: enabled_journal_mode,
        });
    }

    apply_schema(&connection)?;
    validate_initialized(path, &connection)?;
    cleanup_vestigial_long_term_silo(&connection)?;

    Ok(InitReport {
        created,
        initialized: true,
        schema_version: user_version(&connection)?,
        protocol_version: required_config_value(path, &connection, "protocol_version")?,
        sqlite_version: sqlite_version(&connection)?,
        journal_mode: journal_mode(&connection)?,
        spaces: space_names(&connection)?,
        default_space: required_config_value(path, &connection, "default_space")?,
    })
}

/// Store one explicit memory and update deterministic indexes atomically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the referenced space/silo/source/memory does not exist, or `SQLite` rejects the
/// transaction.
pub fn remember_memory(
    path: impl AsRef<Path>,
    request: &RememberRequest,
) -> Result<RememberReport> {
    validate_remember_request(request)?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = remember_memory_tx(&transaction, request)?;

    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }

    Ok(report)
}

/// Fetch one memory by id from an initialized store.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the id is empty, the
/// memory is not found, or `SQLite` rejects the query.
pub fn get_memory(path: impl AsRef<Path>, id: &str, options: GetOptions) -> Result<MemoryRecord> {
    if id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "memory id must not be empty".to_string(),
        });
    }
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        load_memory(connection, id, options)
    })
}

/// Tombstone one memory by id and preserve an audit event.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the memory does not exist, the memory is already tombstoned, or `SQLite`
/// rejects the transaction.
pub fn forget_memory(path: impl AsRef<Path>, request: &ForgetRequest) -> Result<ForgetReport> {
    validate_forget_request(request)?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = forget_memory_tx(&transaction, request)?;

    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }

    Ok(report)
}

/// Maximum recall events accepted by one `record_recall` call.
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
];

/// Accepted candidate sensitivity labels.
pub(crate) const CANDIDATE_SENSITIVITIES: &[&str] = &["normal", "sensitive"];

/// Accepted candidate lifecycle statuses.
pub(crate) const CANDIDATE_STATUSES: &[&str] = &["pending", "approved", "rejected"];

/// Accepted `remember` supersession modes (how a write resolves against active
/// memories sharing its `entity_key` + `claim_key`). `auto` is the default and
/// preserves the historical policy.
pub const REMEMBER_SUPERSEDE_MODES: &[&str] =
    &["auto", "append", "supersede", "suggest", "conflict"];

/// Default supersession mode: the historical same entity/claim policy.
pub const REMEMBER_MODE_AUTO: &str = "auto";

const CANDIDATE_STATUS_PENDING: &str = "pending";
const CANDIDATE_STATUS_APPROVED: &str = "approved";
const CANDIDATE_STATUS_REJECTED: &str = "rejected";
const DEFAULT_CANDIDATE_SOURCE_TYPE: &str = "assistant-inference";
const DEFAULT_CANDIDATE_SENSITIVITY: &str = "normal";

/// Default and maximum rows returned by one `candidate-list` call.
const DEFAULT_CANDIDATE_LIST_LIMIT: usize = 50;

const CANDIDATES_DDL: &str = "
CREATE TABLE IF NOT EXISTS memory_candidates (
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'pending',
  space TEXT,
  silo TEXT,
  scope TEXT,
  project TEXT,
  kind TEXT,
  content TEXT NOT NULL,
  summary TEXT,
  rationale TEXT,
  tags_json TEXT,
  entity_key TEXT,
  claim_key TEXT,
  confidence REAL NOT NULL DEFAULT 1.0,
  source_type TEXT NOT NULL DEFAULT 'assistant-inference',
  source_json TEXT,
  sensitivity TEXT NOT NULL DEFAULT 'normal',
  supersedes_json TEXT,
  created_at TEXT NOT NULL,
  decided_at TEXT,
  decided_reason TEXT,
  resulting_memory_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_status ON memory_candidates(status);
CREATE INDEX IF NOT EXISTS idx_memory_candidates_created ON memory_candidates(created_at);
";

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
const CANDIDATE_COLUMNS: &str = "id, status, space, silo, scope, project, kind, content, \
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
const DEFAULT_INGEST_SOURCE_TYPE: &str = "import";

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

/// Internal chunk hit shared by the lexical and semantic arms before fusion.
struct DocumentChunkHit {
    source_episode_id: String,
    space: String,
    source_type: String,
    source_path: Option<String>,
    source_uri: Option<String>,
    chunk_index: i64,
    chunk_count: i64,
    snippet: String,
    content: Option<String>,
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

fn candidate_string_array_to_column(values: &[String]) -> Option<String> {
    if values.is_empty() {
        None
    } else {
        Some(serde_json::to_string(values).unwrap_or_else(|_| "[]".to_string()))
    }
}

fn candidate_string_array_from_column(raw: Option<String>) -> Vec<String> {
    raw.and_then(|text| serde_json::from_str::<Vec<String>>(&text).ok())
        .unwrap_or_default()
}

fn memory_candidate_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<CandidateRecord> {
    Ok(CandidateRecord {
        id: row.get(0)?,
        status: row.get(1)?,
        space: row.get(2)?,
        silo: row.get(3)?,
        scope: row.get(4)?,
        project: row.get(5)?,
        kind: row.get(6)?,
        content: row.get(7)?,
        summary: row.get(8)?,
        rationale: row.get(9)?,
        tags: candidate_string_array_from_column(row.get(10)?),
        entity_key: row.get(11)?,
        claim_key: row.get(12)?,
        confidence: row.get(13)?,
        source_type: row.get(14)?,
        source_json: row.get(15)?,
        sensitivity: row.get(16)?,
        supersedes: candidate_string_array_from_column(row.get(17)?),
        created_at: row.get(18)?,
        decided_at: row.get(19)?,
        decided_reason: row.get(20)?,
        resulting_memory_id: row.get(21)?,
    })
}

fn load_candidate(transaction: &Transaction<'_>, id: &str) -> Result<Option<CandidateRecord>> {
    let sql = format!("SELECT {CANDIDATE_COLUMNS} FROM memory_candidates WHERE id = ?1");
    let mut statement = transaction.prepare(&sql)?;
    let mut rows = statement.query(params![id])?;
    match rows.next()? {
        Some(row) => Ok(Some(memory_candidate_from_row(row)?)),
        None => Ok(None),
    }
}

fn validate_candidate_submit_request(request: &CandidateSubmitRequest) -> Result<()> {
    if request.content.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "content must not be empty".to_string(),
        });
    }
    if request.content.chars().count() > MAX_CONTENT_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("content must be at most {MAX_CONTENT_CHARS} characters"),
        });
    }
    if request
        .summary
        .as_deref()
        .is_some_and(|s| s.chars().count() > MAX_SUMMARY_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: format!("summary must be at most {MAX_SUMMARY_CHARS} characters"),
        });
    }
    if request
        .rationale
        .as_deref()
        .is_some_and(|s| s.chars().count() > MAX_SUMMARY_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: format!("rationale must be at most {MAX_SUMMARY_CHARS} characters"),
        });
    }
    validate_optional_metadata_value("space", request.space.as_deref())?;
    validate_optional_metadata_value("silo", request.silo.as_deref())?;
    validate_optional_metadata_value("scope", request.scope.as_deref())?;
    validate_optional_metadata_value("project", request.project.as_deref())?;
    validate_optional_metadata_value("kind", request.kind.as_deref())?;
    validate_optional_metadata_value("entity_key", request.entity_key.as_deref())?;
    validate_optional_metadata_value("claim_key", request.claim_key.as_deref())?;
    if !(0.0..=1.0).contains(&request.confidence) {
        return Err(Error::InvalidRequest {
            message: "confidence must be between 0.0 and 1.0".to_string(),
        });
    }
    if let Some(source_type) = request.source_type.as_deref() {
        if !CANDIDATE_SOURCE_TYPES.contains(&source_type) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported source_type: {source_type} (expected one of {})",
                    CANDIDATE_SOURCE_TYPES.join(", ")
                ),
            });
        }
    }
    if let Some(sensitivity) = request.sensitivity.as_deref() {
        if !CANDIDATE_SENSITIVITIES.contains(&sensitivity) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported sensitivity: {sensitivity} (expected one of {})",
                    CANDIDATE_SENSITIVITIES.join(", ")
                ),
            });
        }
    }
    if let Some(source_json) = &request.source_json {
        if source_json.chars().count() > MAX_SOURCE_REF_JSON_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "source JSON must be at most {MAX_SOURCE_REF_JSON_CHARS} characters"
                ),
            });
        }
        if !JsonValidator::is_object(source_json) {
            return Err(Error::InvalidRequest {
                message: "source JSON must be a valid JSON object".to_string(),
            });
        }
    }
    validate_memory_link_ids("supersedes", &request.supersedes)?;
    let _ = normalized_tags(&request.tags)?;
    Ok(())
}

/// Submit a candidate memory for later review.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is
/// invalid, or `SQLite` rejects the transaction.
pub fn submit_candidate(
    path: impl AsRef<Path>,
    request: &CandidateSubmitRequest,
) -> Result<CandidateSubmitReport> {
    validate_candidate_submit_request(request)?;
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    transaction.execute_batch(CANDIDATES_DDL)?;
    let now = now_timestamp(&transaction)?;
    let id = next_id("cand");
    let tags = normalized_tags(&request.tags)?;
    let source_type = request
        .source_type
        .clone()
        .unwrap_or_else(|| DEFAULT_CANDIDATE_SOURCE_TYPE.to_string());
    let sensitivity = request
        .sensitivity
        .clone()
        .unwrap_or_else(|| DEFAULT_CANDIDATE_SENSITIVITY.to_string());
    transaction.execute(
        "INSERT INTO memory_candidates (id, status, space, silo, scope, project, kind, content, \
         summary, rationale, tags_json, entity_key, claim_key, confidence, source_type, \
         source_json, sensitivity, supersedes_json, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19)",
        params![
            id,
            CANDIDATE_STATUS_PENDING,
            request.space,
            request.silo,
            request.scope,
            request.project,
            request.kind,
            request.content,
            request.summary,
            request.rationale,
            candidate_string_array_to_column(&tags),
            request.entity_key,
            request.claim_key,
            request.confidence,
            source_type,
            request.source_json,
            sensitivity,
            candidate_string_array_to_column(&request.supersedes),
            now,
        ],
    )?;
    let candidate = load_candidate(&transaction, &id)?.ok_or_else(|| Error::InvalidRequest {
        message: "candidate insert did not persist".to_string(),
    })?;
    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }
    Ok(CandidateSubmitReport {
        candidate,
        dry_run: request.dry_run,
    })
}

/// List candidates for review, newest first.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the status filter is
/// unknown, or `SQLite` rejects the transaction.
pub fn list_candidates(
    path: impl AsRef<Path>,
    request: &CandidateListRequest,
) -> Result<CandidateListReport> {
    if let Some(status) = request.status.as_deref() {
        if !CANDIDATE_STATUSES.contains(&status) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported status filter: {status} (expected one of {})",
                    CANDIDATE_STATUSES.join(", ")
                ),
            });
        }
    }
    let limit = request.limit.clamp(1, DEFAULT_CANDIDATE_LIST_LIMIT * 2);
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    transaction.execute_batch(CANDIDATES_DDL)?;

    // Build a WHERE clause from the optional status/space filters.
    let mut clauses: Vec<&str> = Vec::new();
    if request.status.is_some() {
        clauses.push("status = :status");
    }
    if request.space.is_some() {
        clauses.push("space = :space");
    }
    let where_sql = if clauses.is_empty() {
        String::new()
    } else {
        format!(" WHERE {}", clauses.join(" AND "))
    };

    let count_sql = format!("SELECT COUNT(*) FROM memory_candidates{where_sql}");
    let list_sql = format!(
        "SELECT {CANDIDATE_COLUMNS} FROM memory_candidates{where_sql} \
         ORDER BY created_at DESC, id DESC LIMIT :limit OFFSET :offset"
    );

    let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
    let offset_i64 = i64::try_from(request.offset).unwrap_or(0);

    let total: i64 = {
        let mut statement = transaction.prepare(&count_sql)?;
        let mut bindings: Vec<(&str, &dyn rusqlite::ToSql)> = Vec::new();
        if let Some(status) = &request.status {
            bindings.push((":status", status));
        }
        if let Some(space) = &request.space {
            bindings.push((":space", space));
        }
        statement.query_row(bindings.as_slice(), |row| row.get(0))?
    };

    let candidates = {
        let mut statement = transaction.prepare(&list_sql)?;
        let mut bindings: Vec<(&str, &dyn rusqlite::ToSql)> = Vec::new();
        if let Some(status) = &request.status {
            bindings.push((":status", status));
        }
        if let Some(space) = &request.space {
            bindings.push((":space", space));
        }
        bindings.push((":limit", &limit_i64));
        bindings.push((":offset", &offset_i64));
        let mut rows = statement.query(bindings.as_slice())?;
        let mut collected = Vec::new();
        while let Some(row) = rows.next()? {
            collected.push(memory_candidate_from_row(row)?);
        }
        collected
    };

    transaction.commit()?;
    Ok(CandidateListReport {
        candidates,
        total: usize::try_from(total).unwrap_or(0),
    })
}

/// Build the canonical source/provenance JSON for an approved candidate,
/// folding the candidate's `source_type` into any supplied source object.
/// Mirrors the metadata-merge pattern used by `verify_memory`.
fn candidate_source_ref_json(source_json: Option<&str>, source_type: &str) -> String {
    let mut map: serde_json::Map<String, serde_json::Value> = match source_json {
        Some(raw) => serde_json::from_str(raw).unwrap_or_default(),
        None => serde_json::Map::new(),
    };
    map.insert(
        "source_type".to_string(),
        serde_json::Value::String(source_type.to_string()),
    );
    serde_json::to_string(&map)
        .unwrap_or_else(|_| format!("{{\"source_type\":{}}}", json_string_for_store(source_type)))
}

fn remember_request_from_candidate(
    candidate: &CandidateRecord,
    embedding: Option<Vec<f32>>,
    embedding_model_id: Option<String>,
) -> RememberRequest {
    let metadata_json = candidate
        .rationale
        .as_deref()
        .map(|rationale| format!("{{\"rationale\":{}}}", json_string_for_store(rationale)));
    RememberRequest {
        space: candidate.space.clone(),
        silo: candidate.silo.clone(),
        scope: candidate.scope.clone(),
        project_key: candidate.project.clone(),
        kind: candidate.kind.clone(),
        content: candidate.content.clone(),
        summary: candidate.summary.clone(),
        tags: candidate.tags.clone(),
        entity_key: candidate.entity_key.clone(),
        claim_key: candidate.claim_key.clone(),
        confidence: candidate.confidence,
        observed_at: None,
        valid_from: None,
        valid_to: None,
        expires_at: None,
        source_ref_json: Some(candidate_source_ref_json(
            candidate.source_json.as_deref(),
            &candidate.source_type,
        )),
        metadata_json,
        source_episode_id: None,
        pinned: false,
        supersedes: candidate.supersedes.clone(),
        contradicts: Vec::new(),
        embedding,
        embedding_model_id,
        token_embedding: None,
        token_embedding_model_id: None,
        // The outer candidate transaction controls rollback; the inner write
        // must commit within it (dry-run is handled by rolling back the whole tx).
        dry_run: false,
        mode: REMEMBER_MODE_AUTO.to_string(),
    }
}

/// Approve a candidate: promote it into a real memory via the remember write
/// path, then mark the candidate approved with the resulting memory id.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the candidate is
/// missing or not pending, the promoted memory fails validation, or `SQLite`
/// rejects the transaction.
pub fn approve_candidate(
    path: impl AsRef<Path>,
    request: &CandidateApproveRequest,
) -> Result<CandidateApproveReport> {
    if request.id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "candidate id must not be empty".to_string(),
        });
    }
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    transaction.execute_batch(CANDIDATES_DDL)?;
    let candidate =
        load_candidate(&transaction, &request.id)?.ok_or_else(|| Error::InvalidRequest {
            message: format!("candidate not found: {}", request.id),
        })?;
    if candidate.status != CANDIDATE_STATUS_PENDING {
        return Err(Error::InvalidRequest {
            message: format!("candidate {} is already {}", candidate.id, candidate.status),
        });
    }
    let remember = remember_request_from_candidate(
        &candidate,
        request.embedding.clone(),
        request.embedding_model_id.clone(),
    );
    validate_remember_request(&remember)?;
    let remember_report = remember_memory_tx(&transaction, &remember)?;
    let memory = remember_report.memory;
    let now = now_timestamp(&transaction)?;
    transaction.execute(
        "UPDATE memory_candidates SET status = ?1, decided_at = ?2, resulting_memory_id = ?3 \
         WHERE id = ?4",
        params![CANDIDATE_STATUS_APPROVED, now, memory.id, candidate.id],
    )?;
    let updated =
        load_candidate(&transaction, &candidate.id)?.ok_or_else(|| Error::InvalidRequest {
            message: "candidate update did not persist".to_string(),
        })?;
    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }
    Ok(CandidateApproveReport {
        candidate: updated,
        memory,
        dry_run: request.dry_run,
    })
}

/// Reject a candidate, recording an optional reason.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the candidate is
/// missing or not pending, or `SQLite` rejects the transaction.
pub fn reject_candidate(
    path: impl AsRef<Path>,
    request: &CandidateRejectRequest,
) -> Result<CandidateRejectReport> {
    if request.id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "candidate id must not be empty".to_string(),
        });
    }
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    transaction.execute_batch(CANDIDATES_DDL)?;
    let candidate =
        load_candidate(&transaction, &request.id)?.ok_or_else(|| Error::InvalidRequest {
            message: format!("candidate not found: {}", request.id),
        })?;
    if candidate.status != CANDIDATE_STATUS_PENDING {
        return Err(Error::InvalidRequest {
            message: format!("candidate {} is already {}", candidate.id, candidate.status),
        });
    }
    let now = now_timestamp(&transaction)?;
    transaction.execute(
        "UPDATE memory_candidates SET status = ?1, decided_at = ?2, decided_reason = ?3 \
         WHERE id = ?4",
        params![CANDIDATE_STATUS_REJECTED, now, request.reason, candidate.id],
    )?;
    let updated =
        load_candidate(&transaction, &candidate.id)?.ok_or_else(|| Error::InvalidRequest {
            message: "candidate update did not persist".to_string(),
        })?;
    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }
    Ok(CandidateRejectReport {
        candidate: updated,
        dry_run: request.dry_run,
    })
}

/// Stamp `verified_at` (and optionally `verified_against`) into one memory's `metadata_json`,
/// preserving any existing keys, and touch `updated_at`.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the memory id is empty,
/// the memory does not exist, or `SQLite` rejects the transaction.
pub fn verify_memory(path: impl AsRef<Path>, request: &VerifyRequest) -> Result<VerifyReport> {
    if request.memory_id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "memory id must not be empty".to_string(),
        });
    }
    validate_optional_timestamp("now", request.now.as_deref())?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = verify_memory_tx(&transaction, request)?;
    transaction.commit()?;
    Ok(report)
}

fn verify_memory_tx(
    transaction: &Transaction<'_>,
    request: &VerifyRequest,
) -> Result<VerifyReport> {
    let now = match request.now.as_deref() {
        Some(ts) => normalize_utc_timestamp(ts),
        None => now_timestamp(transaction)?,
    };

    let existing: Option<String> = transaction
        .query_row(
            "SELECT metadata_json FROM memories WHERE id = ?1",
            [&request.memory_id],
            |row| row.get(0),
        )
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "memory",
            id: request.memory_id.clone(),
        })?;

    // Parse existing metadata_json into a map. A present-but-unparseable (or
    // non-object) value is store corruption; erroring preserves the evidence
    // instead of silently replacing it with just the verify keys.
    let mut map: serde_json::Map<String, serde_json::Value> = match existing.as_deref() {
        None => serde_json::Map::new(),
        Some(raw) => serde_json::from_str::<serde_json::Value>(raw)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .ok_or_else(|| Error::Conflict {
                message: format!(
                    "memory {} has corrupt metadata_json; refusing to overwrite it",
                    request.memory_id
                ),
            })?,
    };

    map.insert(
        "verified_at".to_string(),
        serde_json::Value::String(now.clone()),
    );
    if let Some(ref src) = request.verified_against {
        map.insert(
            "verified_against".to_string(),
            serde_json::Value::String(src.clone()),
        );
    }

    let merged = serde_json::to_string(&map).map_err(|e| Error::InvalidRequest {
        message: format!("failed to serialize metadata_json: {e}"),
    })?;

    // Existence was established by the SELECT above, inside this transaction.
    transaction.execute(
        "UPDATE memories SET metadata_json = ?1, updated_at = ?2 WHERE id = ?3",
        params![&merged, &now, &request.memory_id],
    )?;

    Ok(VerifyReport {
        memory_id: request.memory_id.clone(),
        verified_at: now,
    })
}

/// Fetch bounded audit history for one memory by id.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the memory does not exist, or `SQLite` rejects the query.
pub fn memory_history(
    path: impl AsRef<Path>,
    id: &str,
    options: HistoryOptions,
) -> Result<HistoryReport> {
    validate_history_request(id, options)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        load_history(connection, id, options)
    })
}

/// Search memories deterministically with `SQLite` FTS5/BM25 and metadata filters.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// or `SQLite` rejects the query.
pub fn search_memories(path: impl AsRef<Path>, request: &SearchRequest) -> Result<SearchReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        search_memories_on_connection(connection, request)
    })
}

/// Create or update one projected graph entity deterministically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the source episode is not in the same space, or `SQLite` rejects the transaction.
pub fn upsert_entity(
    path: impl AsRef<Path>,
    request: &EntityUpsertRequest,
) -> Result<EntityUpsertReport> {
    let prepared = prepare_entity_upsert_request(request)?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = upsert_entity_tx(&transaction, &prepared)?;
    transaction.commit()?;
    Ok(report)
}

/// Create or update one projected graph relationship deterministically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, endpoints or evidence
/// are missing/outside the requested space, the request is invalid, or `SQLite`
/// rejects the transaction.
pub fn upsert_relationship(
    path: impl AsRef<Path>,
    request: &RelationshipUpsertRequest,
) -> Result<RelationshipUpsertReport> {
    let prepared = prepare_relationship_upsert_request(request)?;
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = upsert_relationship_tx(&transaction, &prepared)?;
    transaction.commit()?;
    Ok(report)
}

/// Merge one projected graph entity into another deterministically.
///
/// Relinks the source entity's active relationships onto the target (collapsing
/// duplicates and self-loops), carries the source's key/name/aliases over as
/// aliases of the target, and tombstones the source entity. With `dry_run` the
/// merge is computed and reported but rolled back.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, either endpoint is
/// unspecified or not found, `from` and `into` resolve to the same entity, or
/// `SQLite` rejects the transaction.
pub fn merge_entity(
    path: impl AsRef<Path>,
    request: &EntityMergeRequest,
) -> Result<EntityMergeReport> {
    let path = path.as_ref();
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    let report = merge_entity_tx(&transaction, request)?;
    if request.dry_run {
        // Drop without committing to roll back the computed (but unwanted) mutation.
        drop(transaction);
    } else {
        transaction.commit()?;
    }
    Ok(report)
}

/// Search projected graph entities deterministically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// or `SQLite` rejects the query.
pub fn search_entities(
    path: impl AsRef<Path>,
    request: &EntitySearchRequest,
) -> Result<EntitySearchReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        search_entities_on_connection(connection, request)
    })
}

/// Traverse projected graph neighbors deterministically.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the seed entity does not exist, or `SQLite` rejects the query.
pub fn graph_neighbors(
    path: impl AsRef<Path>,
    request: &GraphNeighborsRequest,
) -> Result<GraphNeighborsReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        graph_neighbors_on_connection(connection, request)
    })
}

/// Build a compact memory context pack around graph neighbors.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// the seed entity does not exist, or `SQLite` rejects the query.
pub fn graph_context(
    path: impl AsRef<Path>,
    request: &GraphContextRequest,
) -> Result<GraphContextReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        graph_context_on_connection(connection, request)
    })
}

/// One node in the whole-graph projection: an active entity participating in
/// at least one active relationship, weighted by its total edge count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFullNode {
    /// Stable entity key (node id).
    pub entity_key: String,
    /// Human-friendly name.
    pub canonical_name: String,
    /// Entity type, when set.
    pub entity_type: Option<String>,
    /// Sum of incident active-edge weights (node size hint).
    pub degree: i64,
}

/// One edge in the whole-graph projection: active relationships between two
/// active entities, collapsed and weighted by pair count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFullLink {
    /// Subject entity key.
    pub source: String,
    /// Object entity key.
    pub target: String,
    /// Number of active relationships connecting the pair.
    pub weight: i64,
}

/// The entire connected entity graph for whole-graph visualization. Isolated
/// entities (no active edges) are omitted — they would render as loose dots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphFullReport {
    /// Connected entities.
    pub nodes: Vec<GraphFullNode>,
    /// Weighted edges between them.
    pub links: Vec<GraphFullLink>,
}

/// Project the entire connected entity graph (all active relationships between
/// active entities, plus the entities that participate in at least one).
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible or `SQLite` rejects a query.
pub fn graph_full(path: impl AsRef<Path>) -> Result<GraphFullReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, graph_full_on_connection)
}

fn graph_full_on_connection(connection: &Connection) -> Result<GraphFullReport> {
    let mut edge_stmt = connection.prepare(
        "SELECT s.entity_key, o.entity_key, COUNT(*) \
         FROM relationships r \
         JOIN entities s ON s.id = r.subject_entity_id AND s.status = 'active' \
         JOIN entities o ON o.id = r.object_entity_id AND o.status = 'active' \
         WHERE r.status = 'active' \
         GROUP BY s.entity_key, o.entity_key",
    )?;
    let rows = edge_stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;
    let mut links = Vec::new();
    let mut degree: BTreeMap<String, i64> = BTreeMap::new();
    for row in rows {
        let (source, target, weight) = row?;
        *degree.entry(source.clone()).or_insert(0) += weight;
        *degree.entry(target.clone()).or_insert(0) += weight;
        links.push(GraphFullLink {
            source,
            target,
            weight,
        });
    }
    let mut node_stmt = connection.prepare(
        "SELECT entity_key, canonical_name, entity_type FROM entities WHERE status = 'active'",
    )?;
    let node_rows = node_stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
        ))
    })?;
    let mut nodes = Vec::new();
    for row in node_rows {
        let (entity_key, canonical_name, entity_type) = row?;
        if let Some(&deg) = degree.get(&entity_key) {
            nodes.push(GraphFullNode {
                entity_key,
                canonical_name,
                entity_type,
                degree: deg,
            });
        }
    }
    Ok(GraphFullReport { nodes, links })
}

/// List recent memories deterministically for review/admin workflows.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// or `SQLite` rejects the query.
pub fn list_memories(
    path: impl AsRef<Path>,
    request: &MemoryListRequest,
) -> Result<MemoryListReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        list_memories_on_connection(connection, request)
    })
}

/// Run multiple deterministic searches against one read-only store snapshot.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// or `SQLite` rejects a query.
pub fn batch_search_memories(
    path: impl AsRef<Path>,
    request: &BatchSearchRequest,
) -> Result<BatchSearchReport> {
    validate_batch_search_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        batch_search_memories_on_connection(connection, request)
    })
}

/// Build a compact deterministic memory pack from one or more lexical queries.
///
/// # Errors
///
/// Returns an error if the store is missing/incompatible, the request is invalid,
/// or `SQLite` rejects a query.
pub fn build_pack(path: impl AsRef<Path>, request: &PackRequest) -> Result<PackReport> {
    validate_pack_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_pack_on_connection(connection, request)
    })
}

/// One scored candidate from the pack retrieval pool (pre-rerank).
#[derive(Debug, Clone, PartialEq)]
pub struct PackPoolItem {
    /// Candidate memory id.
    pub memory_id: String,
    /// Retrieval score: cosine similarity on the ANN/embed path, BM25 otherwise.
    pub score: f64,
}

/// One pack candidate after cross-encoder reranking: its memory id, full
/// content, and the reranker's relevance score. Consumed by
/// [`assemble_reranked_pack`].
#[derive(Debug, Clone, PartialEq)]
pub struct RerankCandidate {
    /// Candidate memory id.
    pub memory_id: String,
    /// Full memory content used both for reranking and pack injection.
    pub content: String,
    /// Cross-encoder relevance score for `(query, content)`.
    pub rerank_score: f32,
}

/// Build an empty pack (no injection) that preserves the request's title/format.
#[must_use]
pub fn empty_pack(request: &PackRequest) -> PackReport {
    PackReport {
        title: request.title.clone(),
        format: request.format.clone(),
        content: String::new(),
        memory_ids: Vec::new(),
        scores: Vec::new(),
        truncated: false,
        top_score: None,
    }
}

/// Whether a configured cosine gate (`> 0.0`) blocks injection because the pool's
/// top retrieval score is below it. Used by the embed-without-reranker path,
/// where there is no cross-encoder confidence to fall back on.
#[must_use]
pub fn pack_blocked_by_cosine_gate(cos_top: f64, cosine_gate: f64) -> bool {
    cosine_gate > 0.0 && cos_top < cosine_gate
}

/// Render a single oversized pack entry truncated to fit the whole char budget.
///
/// Used only by the budget fallback in [`assemble_reranked_pack`] when the
/// top-ranked candidate is itself larger than `max_chars`. Returns the rendered
/// `- <prefix>…\n` entry whose byte length is `<= max_chars`, cut on a UTF-8 char
/// boundary, or `None` when the budget cannot hold the bullet, marker, and at
/// least one character of content.
fn truncate_pack_entry(content: &str, max_chars: usize) -> Option<String> {
    const PREFIX: &str = "- ";
    const SUFFIX: &str = "…\n"; // ellipsis marker + newline
    let overhead = PREFIX.len() + SUFFIX.len();
    if max_chars <= overhead {
        return None;
    }
    let budget = max_chars - overhead;
    let mut end = budget.min(content.len());
    while end > 0 && !content.is_char_boundary(end) {
        end -= 1;
    }
    if end == 0 {
        return None;
    }
    let mut entry = String::with_capacity(max_chars);
    entry.push_str(PREFIX);
    entry.push_str(&content[..end]);
    entry.push_str(SUFFIX);
    Some(entry)
}

/// Assemble the final pack from a reranked candidate pool, applying the injection
/// gate, rerank ordering, precision floor, and char/count budget.
///
/// `cos_top` is the pool's top retrieval (cosine) score and `cosine_gate` the
/// configured query-level gate. When `cosine_gate > 0.0` (Option-3 gating),
/// injection is allowed iff the embedding is on-topic (`cos_top >= cosine_gate`)
/// or the cross-encoder is confident (`max rerank score >= min_score`); the
/// cross-encoder then only ORDERS survivors, with no per-item floor. When
/// `cosine_gate == 0.0`, the legacy behavior applies `min_score` as a per-item
/// floor on the rerank score.
///
/// Pure retrieval policy: no store or model access, so it is fully unit-testable.
#[must_use]
pub fn assemble_reranked_pack(
    request: &PackRequest,
    cosine_gate: f64,
    cos_top: f64,
    candidates: &[RerankCandidate],
) -> PackReport {
    let total = candidates.len();
    let rr_top = candidates
        .iter()
        .map(|candidate| candidate.rerank_score)
        .fold(f32::MIN, f32::max);

    let mut ordered: Vec<&RerankCandidate> = candidates.iter().collect();
    ordered.sort_by(|a, b| {
        b.rerank_score
            .partial_cmp(&a.rerank_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let per_item_floor = if cosine_gate > 0.0 {
        let emit = cos_top >= cosine_gate || f64::from(rr_top) >= request.min_score;
        if !emit {
            return empty_pack(request);
        }
        None
    } else {
        Some(request.min_score)
    };

    let mut content = String::new();
    let mut memory_ids = Vec::new();
    let mut scores = Vec::new();
    for candidate in &ordered {
        if memory_ids.len() >= request.max_memories {
            break;
        }
        // `ordered` is sorted by rerank score descending, so the first sub-floor
        // score means every remaining candidate is also below the floor.
        if per_item_floor.is_some_and(|floor| f64::from(candidate.rerank_score) < floor) {
            break;
        }
        let entry = format!("- {}\n", candidate.content);
        if content.len() + entry.len() > request.max_chars {
            break;
        }
        content.push_str(&entry);
        memory_ids.push(candidate.memory_id.clone());
        scores.push(f64::from(candidate.rerank_score));
    }

    // Budget fallback: the gate passed, but if the highest-ranked eligible
    // candidate is by itself larger than the whole char budget, the loop above
    // injects nothing and a confidently-relevant memory is dropped purely for
    // being long. Inject the top eligible candidate truncated to the budget.
    // Fires ONLY for the char-budget case: the early gate return and the floor
    // check above already excluded the gate-blocked and floored cases.
    let mut text_truncated = false;
    if memory_ids.is_empty() {
        if let Some(top) = ordered.first() {
            let eligible = match per_item_floor {
                Some(floor) => f64::from(top.rerank_score) >= floor,
                None => true,
            };
            if eligible {
                if let Some(entry) = truncate_pack_entry(&top.content, request.max_chars) {
                    content.push_str(&entry);
                    memory_ids.push(top.memory_id.clone());
                    // Keep `scores` aligned 1:1 with `memory_ids` (PackReport invariant);
                    // the budget-fallback path injects the top candidate, so its score too.
                    scores.push(f64::from(top.rerank_score));
                    text_truncated = true;
                }
            }
        }
    }

    // Loud guard: a future edit that pushes to one vector but not the other would
    // misattribute rerank scores to memories and silently corrupt the promote
    // signal. Fail in tests/debug rather than degrade in production.
    debug_assert_eq!(
        scores.len(),
        memory_ids.len(),
        "PackReport scores must stay aligned 1:1 with memory_ids"
    );
    PackReport {
        title: request.title.clone(),
        format: request.format.clone(),
        content,
        memory_ids: memory_ids.clone(),
        scores,
        truncated: memory_ids.len() < total || text_truncated,
        top_score: if candidates.is_empty() {
            None
        } else {
            Some(f64::from(rr_top))
        },
    }
}

/// Build the raw, deduplicated, scored candidate pool for a pack request
/// *without* applying the `min_score` precision floor.
///
/// The CLI rerank path uses this to gate injection on the pool's top retrieval
/// score (cosine on the embed path) independently of the cross-encoder rerank
/// score, then reranks survivors for ordering. `max_memories` bounds the pool
/// size; callers set it to the desired candidate-pool width.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is
/// invalid, or retrieval fails.
pub fn build_pack_pool(path: impl AsRef<Path>, request: &PackRequest) -> Result<Vec<PackPoolItem>> {
    validate_pack_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_pack_pool_basic_on_connection(connection, request)
    })
}

/// Build a pack retrieval pool with optional deterministic query/thread
/// expansion. Existing callers should use [`build_pack_pool`] unless they
/// explicitly want experimental expansion behavior.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is
/// invalid, or retrieval fails.
pub fn build_pack_pool_with_expansion(
    path: impl AsRef<Path>,
    request: &PackRequest,
    expansion: PackExpansionOptions,
) -> Result<Vec<PackPoolItem>> {
    validate_pack_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_pack_pool_with_expansion_on_connection(connection, request, expansion)
    })
}

fn build_pack_pool_with_expansion_on_connection(
    connection: &Connection,
    request: &PackRequest,
    expansion: PackExpansionOptions,
) -> Result<Vec<PackPoolItem>> {
    let expanded_request = expand_pack_request_queries(request, expansion);
    let pool = build_pack_pool_basic_on_connection(connection, &expanded_request)?;
    if !expansion.thread_expansion {
        return Ok(pool);
    }
    expand_pack_pool_threads(connection, &expanded_request, &pool, expansion)
}

fn build_pack_pool_basic_on_connection(
    connection: &Connection,
    request: &PackRequest,
) -> Result<Vec<PackPoolItem>> {
    let per_query_limit = request.max_memories.min(MAX_BATCH_QUERY_LIMIT);
    let mut query_reports: Vec<SearchReport> = Vec::with_capacity(request.queries.len());

    for (i, query) in request.queries.iter().enumerate() {
        #[cfg(not(feature = "semantic"))]
        let _ = i;
        #[cfg(feature = "semantic")]
        let embedding_opt = request
            .query_embeddings
            .as_ref()
            .and_then(|embs| embs.get(i));

        #[cfg(feature = "semantic")]
        if let Some(embedding) = embedding_opt {
            query_reports.push(ann_search_for_pack(
                connection,
                query,
                embedding,
                per_query_limit,
                &request.filters,
            )?);
            continue;
        }

        let search_request = SearchRequest {
            query: query.clone(),
            filters: request.filters.clone(),
            limit: per_query_limit,
            offset: 0,
            snippet_chars: 240,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        };
        query_reports.push(search_memories_on_connection(connection, &search_request)?);
    }

    // Dedupe by memory id, preserving the interleaved retrieval order, with no
    // precision floor: gating is the caller's responsibility on this path.
    let mut seen = BTreeSet::new();
    let mut pool = Vec::new();
    let max_result_len = query_reports
        .iter()
        .map(|r| r.results.len())
        .max()
        .unwrap_or(0);
    for result_index in 0..max_result_len {
        for report in &query_reports {
            if let Some(result) = report.results.get(result_index) {
                if seen.insert(result.memory_id.clone()) {
                    pool.push(PackPoolItem {
                        memory_id: result.memory_id.clone(),
                        score: result.score,
                    });
                    if pool.len() >= request.max_memories {
                        return Ok(pool);
                    }
                }
            }
        }
    }
    Ok(pool)
}

/// Return the deterministic query set used by pack query expansion.
#[must_use]
pub fn expanded_pack_queries(queries: &[String], expansion: PackExpansionOptions) -> Vec<String> {
    let max_variants = expansion
        .max_query_variants
        .clamp(1, MAX_BATCH_QUERIES)
        .max(queries.len().min(MAX_BATCH_QUERIES));
    let mut expanded = Vec::with_capacity(max_variants);
    for query in queries {
        push_unique_query(&mut expanded, query);
        if expansion.query_expansion {
            for variant in deterministic_query_variants(query) {
                push_unique_query(&mut expanded, &variant);
                if expanded.len() >= max_variants {
                    return expanded;
                }
            }
        }
        if expanded.len() >= max_variants {
            return expanded;
        }
    }
    expanded
}

fn expand_pack_request_queries(
    request: &PackRequest,
    expansion: PackExpansionOptions,
) -> PackRequest {
    if !expansion.query_expansion {
        return request.clone();
    }
    let mut expanded = request.clone();
    expanded.queries = expanded_pack_queries(&request.queries, expansion);
    expanded.query_embeddings = None;
    expanded.query_token_embeddings = None;
    expanded.token_model_id = None;
    expanded
}

fn push_unique_query(queries: &mut Vec<String>, query: &str) {
    let normalized = collapse_whitespace(query);
    if !normalized.is_empty() && !queries.iter().any(|existing| existing == &normalized) {
        queries.push(normalized);
    }
}

fn deterministic_query_variants(query: &str) -> Vec<String> {
    let lower = query.to_ascii_lowercase();
    let mut variants = Vec::new();
    for marker in [
        " about ",
        " regarding ",
        " around ",
        " for ",
        " on ",
        " with ",
    ] {
        if let Some((_, tail)) = lower.split_once(marker) {
            let tail = tail
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != ':')
                .trim();
            if !tail.is_empty() {
                variants.push(tail.to_string());
                variants.push(format!("decision {tail}"));
                variants.push(format!("action {tail}"));
            }
        }
    }
    let question_prefixes = [
        "what did we decide",
        "what did i decide",
        "what changed",
        "what fixed",
        "what caused",
        "why did",
        "how did",
        "how should",
    ];
    for prefix in question_prefixes {
        if let Some(tail) = lower.strip_prefix(prefix) {
            let tail = tail
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != ':')
                .trim();
            if !tail.is_empty() {
                variants.push(tail.to_string());
                variants.push(format!("decision {tail}"));
                variants.push(format!("lesson {tail}"));
            }
        }
    }
    variants
}

fn expand_pack_pool_threads(
    connection: &Connection,
    request: &PackRequest,
    pool: &[PackPoolItem],
    expansion: PackExpansionOptions,
) -> Result<Vec<PackPoolItem>> {
    let mut seen: BTreeSet<String> = pool.iter().map(|item| item.memory_id.clone()).collect();
    let mut expanded = Vec::with_capacity(request.max_memories);
    let mut seed_count = 0usize;
    for item in pool {
        if expanded.len() >= request.max_memories {
            break;
        }
        expanded.push(item.clone());
        if expanded.len() >= request.max_memories {
            break;
        }
        if seed_count >= expansion.max_thread_seeds {
            continue;
        }
        seed_count += 1;
        let neighbors =
            thread_neighbor_pool_items(connection, request, item, expansion.max_thread_neighbors)?;
        for neighbor in neighbors {
            if seen.insert(neighbor.memory_id.clone()) {
                expanded.push(neighbor);
                if expanded.len() >= request.max_memories {
                    break;
                }
            }
        }
    }
    Ok(expanded)
}

fn thread_neighbor_pool_items(
    connection: &Connection,
    request: &PackRequest,
    seed: &PackPoolItem,
    limit: usize,
) -> Result<Vec<PackPoolItem>> {
    if limit == 0 {
        return Ok(Vec::new());
    }
    let anchors: Option<(Option<String>, Option<String>)> = connection
        .query_row(
            "SELECT entity_key, claim_key FROM memories WHERE id = ?1",
            [&seed.memory_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((entity_key, claim_key)) = anchors else {
        return Ok(Vec::new());
    };
    if entity_key.is_none() && claim_key.is_none() {
        return Ok(Vec::new());
    }

    let mut filters = request.filters.clone();
    if filters.spaces.is_empty() {
        filters.spaces.push(DEFAULT_SPACE.to_string());
    }
    if filters.statuses.is_empty() {
        filters.statuses.push(status::ACTIVE.to_string());
    }
    filters = normalize_search_filters(filters)?;
    validate_search_filters(&filters)?;

    let mut args = SqlArgs::with_reserved(0);
    let where_clause = filters_where_clause(&filters, &mut args);
    let entity_predicate = entity_key
        .as_ref()
        .map(|value| format!("m.entity_key = {}", args.push(value)));
    let claim_predicate = claim_key
        .as_ref()
        .map(|value| format!("m.claim_key = {}", args.push(value)));
    let anchor_predicate = match (entity_predicate, claim_predicate) {
        (Some(entity), Some(claim)) => format!("({entity} OR {claim})"),
        (Some(entity), None) => entity,
        (None, Some(claim)) => claim,
        (None, None) => return Ok(Vec::new()),
    };
    let seed_placeholder = args.push(&seed.memory_id);
    let sql = format!(
        "SELECT m.id
           FROM memories m
          WHERE {where_clause}
            AND {anchor_predicate}
            AND m.id != {seed_placeholder}
          ORDER BY m.pinned DESC, m.observed_at DESC, m.updated_at DESC, m.id ASC
          LIMIT {limit}"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(args.values.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    let ids = collect_rows(rows)?;
    Ok(ids
        .into_iter()
        .enumerate()
        .map(|(index, memory_id)| PackPoolItem {
            memory_id,
            score: seed.score
                - 0.000_001_f64 * u32::try_from(index + 1).map_or(f64::from(u32::MAX), f64::from),
        })
        .collect())
}

/// One reranking candidate from the hybrid retrieval pool: memory id plus the
/// full active-version content the cross-encoder scores against.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankPoolCandidate {
    /// Candidate memory id.
    pub memory_id: String,
    /// Full active-version content.
    pub content: String,
    /// Late-interaction only: the memory summary as a SEPARATE alternate
    /// rerank text. Scored independently and max-combined with the content
    /// score; never concatenated (prefix-duplicate summaries corrupt
    /// cross-encoder ordering — 2026-06-12 adversarial diagnosis).
    pub summary: Option<String>,
}

/// Hybrid pre-rerank retrieval pool: ANN-primary with a BM25 safety net,
/// deduplicated, with contents fetched from the same read snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct RerankPool {
    /// Top retrieval score of the ANN pool (cosine on the embed path);
    /// `f64::MIN` when the ANN pool is empty.
    pub cos_top: f64,
    /// Deduplicated candidates in interleaved retrieval order.
    pub candidates: Vec<RerankPoolCandidate>,
}

/// Build the hybrid pre-rerank candidate pool for a pack request: a scored ANN
/// pool (no precision floor), a bounded BM25 pool as a recall safety net when
/// query embeddings are present, deduplicated ANN-first, with candidate
/// contents fetched on the same connection/read snapshot. The returned cosine
/// top score is computed from the ANN pool only, preserving the query-level
/// embedding gate semantics.
///
/// Candidates whose active version has no content are dropped (they cannot be
/// reranked meaningfully).
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is
/// invalid, or retrieval fails.
pub fn build_hybrid_rerank_pool(
    path: impl AsRef<Path>,
    request: &PackRequest,
    pool_width: usize,
) -> Result<RerankPool> {
    validate_pack_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_hybrid_rerank_pool_on_connection(connection, request, pool_width)
    })
}

/// Build the hybrid pre-rerank pool with optional same-thread expansion.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is
/// invalid, or retrieval fails.
pub fn build_hybrid_rerank_pool_with_expansion(
    path: impl AsRef<Path>,
    request: &PackRequest,
    pool_width: usize,
    expansion: PackExpansionOptions,
) -> Result<RerankPool> {
    validate_pack_request(request)?;
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        build_hybrid_rerank_pool_with_expansion_on_connection(
            connection, request, pool_width, expansion,
        )
    })
}

fn build_hybrid_rerank_pool_on_connection(
    connection: &Connection,
    request: &PackRequest,
    pool_width: usize,
) -> Result<RerankPool> {
    build_hybrid_rerank_pool_with_expansion_on_connection(
        connection,
        request,
        pool_width,
        PackExpansionOptions::default(),
    )
}

fn build_hybrid_rerank_pool_with_expansion_on_connection(
    connection: &Connection,
    request: &PackRequest,
    pool_width: usize,
    expansion: PackExpansionOptions,
) -> Result<RerankPool> {
    let mut pool_request = request.clone();
    pool_request.max_memories = pool_width;
    pool_request.max_chars = MAX_PACK_CHARS;
    pool_request.rerank_candidates = 0;
    pool_request.min_score = 0.0;
    let ann_pool =
        build_pack_pool_with_expansion_on_connection(connection, &pool_request, expansion)?;
    let bm25_pool = if pool_request
        .query_embeddings
        .as_ref()
        .is_some_and(|embeddings| !embeddings.is_empty())
    {
        let mut lexical_request = pool_request.clone();
        lexical_request.query_embeddings = None;
        build_pack_pool_with_expansion_on_connection(connection, &lexical_request, expansion)?
    } else {
        Vec::new()
    };
    // cos_top is ALWAYS the max fused ANN score so the hook's cosine-gate
    // statistic is identical with late-interaction on or off (the 0.6-incident
    // lesson: never change a gate's scale silently).
    let cos_top = ann_pool
        .iter()
        .map(|item| item.score)
        .fold(f64::MIN, f64::max);
    // Late-interaction: MaxSim over the whole store replaces the ANN leg for
    // candidate SELECTION only; the BM25 safety net stays merged in.
    let primary_pool = match (
        pool_request.query_token_embeddings.as_ref(),
        pool_request.token_model_id.as_deref(),
    ) {
        (Some(token_queries), Some(token_model)) if !token_queries.is_empty() => {
            let mut per_query = Vec::with_capacity(token_queries.len());
            for tokens in token_queries {
                per_query.push(maxsim_candidates(
                    connection,
                    tokens,
                    token_model,
                    pool_width,
                )?);
            }
            interleave_pools(&per_query, pool_width)
        }
        _ => ann_pool,
    };
    let merged = merge_rerank_pools(&primary_pool, &bm25_pool, pool_width);

    // Late-interaction mode carries the summary as a separate alternate rerank
    // text (scored independently, max-combined by the caller). The legacy path
    // stays content-only so flag-off behavior is byte-identical.
    let late_interaction = pool_request.query_token_embeddings.is_some();
    let mut statement = connection.prepare_cached(
        "SELECT COALESCE(v.summary, ''), v.content FROM memories m \
         JOIN memory_versions v ON v.id = m.active_version_id \
         WHERE m.id = ?1",
    )?;
    let mut candidates = Vec::with_capacity(merged.len());
    for item in merged {
        let row: Option<(String, String)> = statement
            .query_row([&item.memory_id], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()?;
        let Some((summary, content)) = row else {
            continue;
        };
        let summary = (late_interaction && !summary.is_empty()).then_some(summary);
        if !content.is_empty() || summary.is_some() {
            candidates.push(RerankPoolCandidate {
                memory_id: item.memory_id,
                content,
                summary,
            });
        }
    }
    Ok(RerankPool {
        cos_top,
        candidates,
    })
}

/// `MaxSim`: sum over query tokens of the max dot-product against doc tokens.
/// Vectors are L2-normalized by the encoder, so dots are cosines.
fn maxsim_score(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f64 {
    query
        .iter()
        .map(|query_token| {
            doc.iter()
                .map(|doc_token| {
                    query_token
                        .iter()
                        .zip(doc_token)
                        .map(|(a, b)| a * b)
                        .sum::<f32>()
                })
                .fold(f32::MIN, f32::max)
        })
        .map(f64::from)
        .sum()
}

/// Exhaustive late-interaction candidate generation: MaxSim-score ALL active
/// memories' token embeddings against the query tokens, return the top `limit`.
///
/// # Errors
///
/// Returns an error on `SQLite` failure or malformed blobs.
fn maxsim_candidates(
    connection: &Connection,
    query_tokens: &[Vec<f32>],
    model_id: &str,
    limit: usize,
) -> Result<Vec<PackPoolItem>> {
    let docs = load_token_embeddings_cached(connection, model_id)?;
    let mut scored: Vec<PackPoolItem> = docs
        .iter()
        .filter(|(_, matrix)| !matrix.is_empty())
        .map(|(memory_id, matrix)| PackPoolItem {
            memory_id: memory_id.clone(),
            score: maxsim_score(query_tokens, matrix),
        })
        .collect();
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(limit);
    Ok(scored)
}

/// Interleave per-query candidate pools rank-by-rank, deduplicating by memory
/// id (multi-query analog of the dedupe in pack-pool construction).
fn interleave_pools(pools: &[Vec<PackPoolItem>], cap: usize) -> Vec<PackPoolItem> {
    let mut seen = BTreeSet::new();
    let mut merged = Vec::new();
    let max_len = pools.iter().map(Vec::len).max().unwrap_or(0);
    for index in 0..max_len {
        for pool in pools {
            if let Some(item) = pool.get(index) {
                if seen.insert(item.memory_id.clone()) {
                    merged.push(item.clone());
                    if merged.len() >= cap {
                        return merged;
                    }
                }
            }
        }
    }
    merged
}

/// Rows of (memory id, token matrix) loaded for late-interaction scoring.
type TokenMatrixRows = Vec<(String, Vec<Vec<f32>>)>;

/// Process-global token-matrix cache for the warm daemon: keyed by
/// (model id, row count, max `created_at`); any token write changes the key.
/// Returns a cheap `Arc` clone on hit.
fn load_token_embeddings_cached(
    connection: &Connection,
    model_id: &str,
) -> Result<std::sync::Arc<TokenMatrixRows>> {
    type CacheEntry = (String, i64, String, std::sync::Arc<TokenMatrixRows>);
    static CACHE: std::sync::OnceLock<std::sync::Mutex<Option<CacheEntry>>> =
        std::sync::OnceLock::new();
    let (count, max_created): (i64, String) = connection.query_row(
        "SELECT count(*), COALESCE(max(created_at), '') FROM memory_token_embeddings \
         WHERE embedding_model = ?1",
        [model_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(None));
    let mut guard = cache.lock().map_err(|_| Error::Conflict {
        message: "token cache lock poisoned".to_string(),
    })?;
    if let Some((cached_model, cached_count, cached_created, rows)) = guard.as_ref() {
        if cached_model == model_id && *cached_count == count && cached_created == &max_created {
            return Ok(std::sync::Arc::clone(rows));
        }
    }
    let rows = std::sync::Arc::new(load_token_embeddings(connection, model_id)?);
    *guard = Some((
        model_id.to_string(),
        count,
        max_created,
        std::sync::Arc::clone(&rows),
    ));
    Ok(rows)
}

/// Interleave semantic ANN and lexical BM25 candidates, preserving each source's
/// internal order while deduplicating by memory id. ANN candidates stay first at
/// each rank so the embedding path remains the primary recall tier; BM25 acts as
/// an exact-keyword safety net before cross-encoder reranking.
fn merge_rerank_pools(
    ann_pool: &[PackPoolItem],
    bm25_pool: &[PackPoolItem],
    max_candidates: usize,
) -> Vec<PackPoolItem> {
    let mut seen = BTreeSet::new();
    let mut merged = Vec::new();
    let max_len = ann_pool.len().max(bm25_pool.len());
    for index in 0..max_len {
        for pool in [ann_pool, bm25_pool] {
            let Some(item) = pool.get(index) else {
                continue;
            };
            if seen.insert(item.memory_id.clone()) {
                merged.push(item.clone());
                if merged.len() >= max_candidates {
                    return merged;
                }
            }
        }
    }
    merged
}

/// Write a deterministic logical JSONL export of canonical store tables.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is invalid,
/// the output path already exists, or filesystem/SQLite export work fails.
pub fn export_store(path: impl AsRef<Path>, request: &ExportRequest) -> Result<ExportReport> {
    validate_export_request(request)?;
    let path = path.as_ref();
    let connection = open_initialized_read_fast(path)?;
    reject_source_sidecar_output(path, &request.output_path)?;
    with_read_snapshot(&connection, |connection| {
        export_store_on_connection(connection, request)
    })
}

/// Create a consistent physical `SQLite` backup of an initialized store.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is invalid,
/// the output path already exists, backup work fails, or the backup cannot be
/// validated as a self-contained memkeeper database.
pub fn backup_store(path: impl AsRef<Path>, request: &BackupRequest) -> Result<BackupReport> {
    validate_backup_request(request)?;
    let path = path.as_ref();
    let connection = open_initialized_read_fast(path)?;
    reject_source_sidecar_output(path, &request.output_path)?;
    with_read_snapshot(&connection, |connection| {
        backup_store_on_connection(connection, request)
    })
}

/// Import a deterministic logical JSONL export into a new initialized store.
///
/// # Errors
///
/// Returns an error when the request/archive is invalid, the target store already
/// exists, the schema is unsupported, or import/index rebuild validation fails.
pub fn import_store(path: impl AsRef<Path>, request: &ImportRequest) -> Result<ImportReport> {
    validate_import_request(request)?;
    let path = path.as_ref();
    validate_store_path(path)?;
    if request.dry_run {
        return import_store_dry_run(request);
    }
    import_store_create(path, request)
}

/// Run explicit bounded maintenance/dream tasks in the Rust core.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is invalid,
/// a referenced space/silo does not exist, or `SQLite` rejects the transaction.
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

fn with_read_snapshot<T>(
    connection: &Connection,
    read: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    connection.execute_batch("BEGIN DEFERRED TRANSACTION")?;
    let result = read(connection);
    match result {
        Ok(value) => {
            connection.execute_batch("COMMIT")?;
            Ok(value)
        }
        Err(error) => {
            let _ = connection.execute_batch("ROLLBACK");
            Err(error)
        }
    }
}

fn list_memories_on_connection(
    connection: &Connection,
    request: &MemoryListRequest,
) -> Result<MemoryListReport> {
    let prepared = prepare_memory_list_request(request)?;
    let mut candidates = list_memory_candidates(connection, &prepared)?;
    let truncated = candidates.len() > prepared.limit;
    if truncated {
        candidates.truncate(prepared.limit);
    }
    let total_estimate = if candidates.is_empty() && !truncated {
        0
    } else {
        prepared
            .offset
            .saturating_add(candidates.len())
            .saturating_add(usize::from(truncated))
    };
    let results = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| candidate.into_item(prepared.offset + index + 1, &prepared))
        .collect();
    Ok(MemoryListReport {
        strategy: "deterministic_list_v0".to_string(),
        total_estimate,
        truncated,
        results,
    })
}

fn search_memories_on_connection(
    connection: &Connection,
    request: &SearchRequest,
) -> Result<SearchReport> {
    let prepared = prepare_search_request(request)?;
    search_prepared(connection, &prepared)
}

fn search_prepared(
    connection: &Connection,
    prepared: &PreparedSearchRequest,
) -> Result<SearchReport> {
    // Semantic-primary: when a query embedding and its ANN index are available,
    // semantic relevance is the primary ranker. BM25/FTS is graceful degradation
    // (no embedding, a missing index, or an empty semantic result set).
    #[cfg(feature = "semantic")]
    if prepared.semantic_fallback != "disabled" {
        if let Some(embedding) = prepared.embedding.as_ref() {
            let table = semantic_table_for_dims(embedding.len())?;
            if table_exists(connection, &table)? {
                let report = semantic_ranked_report(
                    connection,
                    prepared,
                    embedding,
                    "semantic_primary_v0",
                    "semantic_primary",
                )?;
                if !report.results.is_empty() {
                    return Ok(report);
                }
            }
        }
    }

    let mut candidates = search_candidates(connection, prepared, &prepared.fts_query)?;
    if prepared.lexical_fallback != "disabled" && candidates.len() < prepared.limit {
        if let Some(prefix_fts_query) = prepared.prefix_fts_query.as_deref() {
            fill_lexical_candidates(connection, prepared, &mut candidates, prefix_fts_query, 1)?;
        }
    }
    if prepared.lexical_fallback != "disabled" && candidates.is_empty() {
        if let Some(fallback_fts_query) = prepared.fallback_fts_query.as_deref() {
            fill_lexical_candidates(connection, prepared, &mut candidates, fallback_fts_query, 2)?;
        }
    }

    // Best (most negative) bm25 in the matched set anchors the relative FTS
    // normalization in `fts_score`. Computed across all candidates before the
    // per-candidate scoring pass so the top match normalizes to 1.0.
    let best_bm25 = candidates
        .iter()
        .map(|candidate| candidate.bm25)
        .fold(f64::INFINITY, f64::min);
    let now_jd = now_julian_day();
    for candidate in &mut candidates {
        candidate.score = score_candidate(candidate, prepared, best_bm25, now_jd);
    }
    candidates.sort_by(compare_candidates);

    let truncated = candidates.len() > prepared.limit;
    if truncated {
        candidates.truncate(prepared.limit);
    }
    let total_estimate = prepared
        .offset
        .saturating_add(candidates.len())
        .saturating_add(usize::from(truncated));
    let results = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| candidate.into_result(prepared.offset + index + 1, prepared))
        .collect::<Vec<_>>();

    let report = SearchReport {
        strategy: "deterministic_fts_v0".to_string(),
        semantic_attempted: false,
        semantic_reason: if prepared.semantic_fallback == "disabled" {
            "disabled_v0_1"
        } else {
            "fts_results"
        }
        .to_string(),
        total_estimate,
        truncated,
        results,
    };

    if !report.results.is_empty() || prepared.semantic_fallback == "disabled" {
        return Ok(report);
    }

    semantic_fallback_search(connection, prepared, report)
}

#[cfg(not(feature = "semantic"))]
#[allow(clippy::unnecessary_wraps)]
fn semantic_fallback_search(
    _connection: &Connection,
    prepared: &PreparedSearchRequest,
    mut empty_fts_report: SearchReport,
) -> Result<SearchReport> {
    let _embedding_was_supplied = prepared.embedding.is_some();
    empty_fts_report.semantic_reason = "semantic_feature_disabled".to_string();
    Ok(empty_fts_report)
}

fn fill_lexical_candidates(
    connection: &Connection,
    prepared: &PreparedSearchRequest,
    candidates: &mut Vec<SearchCandidate>,
    fts_query: &str,
    lexical_tier: u8,
) -> Result<()> {
    let mut seen = candidates
        .iter()
        .map(|candidate| candidate.memory_id.clone())
        .collect::<BTreeSet<_>>();
    for mut candidate in search_candidates(connection, prepared, fts_query)? {
        if seen.insert(candidate.memory_id.clone()) {
            candidate.lexical_tier = lexical_tier;
            candidates.push(candidate);
        }
    }
    Ok(())
}

#[cfg(feature = "semantic")]
fn semantic_fallback_search(
    connection: &Connection,
    prepared: &PreparedSearchRequest,
    mut empty_fts_report: SearchReport,
) -> Result<SearchReport> {
    let Some(embedding) = prepared.embedding.as_ref() else {
        empty_fts_report.semantic_reason = "missing_embedding".to_string();
        return Ok(empty_fts_report);
    };
    let table = semantic_table_for_dims(embedding.len())?;
    if !table_exists(connection, &table)? {
        empty_fts_report.semantic_reason = "semantic_index_missing".to_string();
        return Ok(empty_fts_report);
    }
    semantic_ranked_report(
        connection,
        prepared,
        embedding,
        "semantic_fallback",
        "fts_empty",
    )
}

/// Rank active memories by semantic (ANN) relevance to the query embedding.
/// Shared by the semantic-primary path and the empty-FTS degradation path.
#[cfg(feature = "semantic")]
fn semantic_ranked_report(
    connection: &Connection,
    prepared: &PreparedSearchRequest,
    embedding: &[f32],
    strategy: &str,
    reason: &str,
) -> Result<SearchReport> {
    let mut candidates = semantic_candidates(connection, prepared, embedding)?;
    let now_jd = now_julian_day();
    for candidate in &mut candidates {
        candidate.score = score_semantic_candidate(candidate, prepared, now_jd);
    }
    candidates.sort_by(compare_candidates);
    let truncated = candidates.len() > prepared.limit;
    if truncated {
        candidates.truncate(prepared.limit);
    }
    let total_estimate = prepared
        .offset
        .saturating_add(candidates.len())
        .saturating_add(usize::from(truncated));
    let results = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| candidate.into_result(prepared.offset + index + 1, prepared))
        .collect::<Vec<_>>();
    Ok(SearchReport {
        strategy: strategy.to_string(),
        semantic_attempted: true,
        semantic_reason: reason.to_string(),
        total_estimate,
        truncated,
        results,
    })
}

fn batch_search_memories_on_connection(
    connection: &Connection,
    request: &BatchSearchRequest,
) -> Result<BatchSearchReport> {
    let mut results = Vec::with_capacity(request.queries.len());
    // One reusable request; only the per-query fields change between loops.
    let mut search_request = SearchRequest {
        query: String::new(),
        filters: request.common_filters.clone(),
        limit: request.limit,
        offset: request.offset,
        snippet_chars: request.snippet_chars,
        include_content: request.include_content,
        include_source: request.include_source,
        semantic_fallback: request.semantic_fallback.clone(),
        lexical_fallback: "conservative".to_string(),
        embedding: None,
        query_token_embedding: None,
        token_model_id: None,
    };
    for query in &request.queries {
        search_request.query.clone_from(&query.query);
        search_request.limit = query.limit.unwrap_or(request.limit);
        results.push(BatchSearchItemReport {
            name: query.name.clone(),
            query: query.query.clone(),
            report: search_memories_on_connection(connection, &search_request)?,
        });
    }
    Ok(BatchSearchReport { results })
}

fn build_pack_on_connection(connection: &Connection, request: &PackRequest) -> Result<PackReport> {
    let per_query_limit = request.max_memories.min(MAX_BATCH_QUERY_LIMIT);
    let mut query_reports: Vec<SearchReport> = Vec::with_capacity(request.queries.len());

    for (i, query) in request.queries.iter().enumerate() {
        #[cfg(not(feature = "semantic"))]
        let _ = i;
        #[cfg(feature = "semantic")]
        let embedding_opt = request
            .query_embeddings
            .as_ref()
            .and_then(|embs| embs.get(i));

        #[cfg(feature = "semantic")]
        if let Some(embedding) = embedding_opt {
            let report = ann_search_for_pack(
                connection,
                query,
                embedding,
                per_query_limit,
                &request.filters,
            )?;
            query_reports.push(report);
            continue;
        }

        // BM25 FTS path (default or when no embedding available for this query)
        let search_request = SearchRequest {
            query: query.clone(),
            filters: request.filters.clone(),
            limit: per_query_limit,
            offset: 0,
            snippet_chars: 240,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        };
        query_reports.push(search_memories_on_connection(connection, &search_request)?);
    }

    let mut seen = BTreeSet::new();
    let mut unique_results = Vec::new();
    let mut truncated = query_reports.iter().any(|report| report.truncated);

    // Round-robin across the per-query result lists (rank 1 of each query,
    // then rank 2, ...), draining the reports so results move instead of clone.
    let mut result_queues = query_reports
        .into_iter()
        .map(|report| report.results.into_iter())
        .collect::<Vec<_>>();
    'items: loop {
        let mut any_remaining = false;
        for queue in &mut result_queues {
            let Some(result) = queue.next() else {
                continue;
            };
            any_remaining = true;
            if result.score < request.min_score {
                continue;
            }
            if seen.insert(result.memory_id.clone()) {
                if unique_results.len() >= request.max_memories {
                    truncated = true;
                    break 'items;
                }
                unique_results.push(result);
            }
        }
        if !any_remaining {
            break;
        }
    }

    let last_synth: Option<String> = connection
        .query_row(
            "SELECT started_at FROM dream_runs WHERE status = 'succeeded' \
             ORDER BY started_at DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let (content, memory_ids, scores, text_truncated) =
        format_pack_markdown(request, &unique_results, last_synth.as_deref());
    let top_score = if unique_results.is_empty() {
        None
    } else {
        Some(
            unique_results
                .iter()
                .map(|r| r.score)
                .fold(f64::NEG_INFINITY, f64::max),
        )
    };
    Ok(PackReport {
        title: request.title.clone(),
        format: request.format.clone(),
        content,
        memory_ids,
        scores,
        truncated: truncated || text_truncated,
        top_score,
    })
}

#[cfg(feature = "semantic")]
fn ann_search_for_pack(
    connection: &Connection,
    query: &str,
    embedding: &[f32],
    limit: usize,
    filters: &SearchFilters,
) -> Result<SearchReport> {
    let table = semantic_table_for_dims(embedding.len())?;
    if !table_exists(connection, &table)? {
        // Vec index missing; fall back to BM25.
        let search_request = SearchRequest {
            query: query.to_string(),
            filters: filters.clone(),
            limit,
            offset: 0,
            snippet_chars: 240,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        };
        return search_memories_on_connection(connection, &search_request);
    }

    // Build a PreparedSearchRequest so we can use semantic_candidates.
    let search_request = SearchRequest {
        query: query.to_string(),
        filters: filters.clone(),
        limit,
        offset: 0,
        snippet_chars: 240,
        include_content: false,
        include_source: false,
        semantic_fallback: "fallback".to_string(),
        lexical_fallback: "conservative".to_string(),
        embedding: Some(embedding.to_vec()),
        query_token_embedding: None,
        token_model_id: None,
    };
    let prepared = prepare_search_request(&search_request)?;

    let mut candidates = semantic_candidates(connection, &prepared, embedding)?;
    let now_jd = now_julian_day();
    for candidate in &mut candidates {
        candidate.score = score_semantic_candidate(candidate, &prepared, now_jd);
    }
    candidates.sort_by(compare_candidates);
    let truncated = candidates.len() > prepared.limit;
    if truncated {
        candidates.truncate(prepared.limit);
    }
    let total_estimate = candidates.len().saturating_add(usize::from(truncated));
    let results = candidates
        .into_iter()
        .enumerate()
        .map(|(index, candidate)| candidate.into_result(index + 1, &prepared))
        .collect::<Vec<_>>();
    Ok(SearchReport {
        strategy: "ann_pack_v0".to_string(),
        semantic_attempted: true,
        semantic_reason: "query_embedding_provided".to_string(),
        total_estimate,
        truncated,
        results,
    })
}

fn export_store_on_connection(
    connection: &Connection,
    request: &ExportRequest,
) -> Result<ExportReport> {
    let (temp_path, file) = create_temp_output_file(&request.output_path)?;
    let export_result = write_export_file(connection, request, file);
    match export_result {
        Ok(report) => {
            publish_temp_output(&temp_path, &request.output_path)?;
            Ok(report)
        }
        Err(error) => {
            cleanup_temp_output(&temp_path);
            Err(error)
        }
    }
}

fn write_export_file(
    connection: &Connection,
    request: &ExportRequest,
    file: File,
) -> Result<ExportReport> {
    let schema_version = user_version(connection)?;
    let mut writer = HashingFileWriter::new(file);
    writer.write_bytes(b"{\"type\":\"header\",\"format\":")?;
    write_json_string_to(&mut writer, "memkeeper.export.v0.1")?;
    writer.write_bytes(b",\"protocol_version\":\"memkeeper.v0.1\",\"schema_version\":")?;
    writer.write_bytes(schema_version.to_string().as_bytes())?;
    writer.write_bytes(b",\"tables\":")?;
    write_string_array_to(&mut writer, EXPORT_TABLES.iter().map(|table| table.name))?;
    writer.write_bytes(b",\"rebuildable_omitted\":[\"memory_fts\",\"memory_fts_public\",\"source_episode_fts\"]}\n")?;

    let mut tables = Vec::with_capacity(EXPORT_TABLES.len());
    let mut row_count = 0_u64;
    for table in EXPORT_TABLES {
        let rows = export_table(connection, &mut writer, table)?;
        row_count = row_count.saturating_add(rows);
        tables.push(ExportTableReport {
            name: table.name.to_string(),
            rows,
        });
    }

    writer.write_bytes(b"{\"type\":\"footer\",\"row_count\":")?;
    writer.write_bytes(row_count.to_string().as_bytes())?;
    writer.write_bytes(b",\"table_counts\":{")?;
    for (index, table) in tables.iter().enumerate() {
        if index > 0 {
            writer.write_bytes(b",")?;
        }
        write_json_string_to(&mut writer, &table.name)?;
        writer.write_bytes(b":")?;
        writer.write_bytes(table.rows.to_string().as_bytes())?;
    }
    writer.write_bytes(b"}}\n")?;
    let (bytes, sha256) = writer.finish()?;

    Ok(ExportReport {
        output_path: request.output_path.clone(),
        format: request.format.clone(),
        schema_version,
        tables,
        row_count,
        bytes,
        sha256,
    })
}

fn export_table(
    connection: &Connection,
    writer: &mut HashingFileWriter,
    table: &ExportTableSpec,
) -> Result<u64> {
    let sql = format!(
        "SELECT {} FROM {} ORDER BY {}",
        table.columns.join(", "),
        table.name,
        table.order_by
    );
    let mut statement = connection.prepare(&sql)?;
    let mut rows = statement.query([])?;
    let mut row_count = 0_u64;

    while let Some(row) = rows.next()? {
        writer.write_bytes(b"{\"type\":\"row\",\"table\":")?;
        write_json_string_to(writer, table.name)?;
        writer.write_bytes(b",\"data\":{")?;
        for (index, column) in table.columns.iter().enumerate() {
            if index > 0 {
                writer.write_bytes(b",")?;
            }
            write_json_string_to(writer, column)?;
            writer.write_bytes(b":")?;
            write_sql_value_json(writer, row.get_ref(index)?)?;
        }
        writer.write_bytes(b"}}\n")?;
        row_count = row_count.saturating_add(1);
    }

    Ok(row_count)
}

fn backup_store_on_connection(
    connection: &Connection,
    request: &BackupRequest,
) -> Result<BackupReport> {
    let (temp_path, file) = create_temp_output_file(&request.output_path)?;
    drop(file);

    let backup_result = backup_to_temp(connection, &temp_path);
    match backup_result {
        Ok(page_count) => {
            let finalize = (|| -> Result<BackupReport> {
                reject_output_sidecar_files(&temp_path)?;
                let bytes = fs::metadata(&temp_path)?.len();
                let sha256 = sha256_path(&temp_path)?;
                let schema_version = user_version(connection)?;
                reject_existing_output_sidecars(&request.output_path)?;
                publish_temp_output(&temp_path, &request.output_path)?;
                if let Err(error) = reject_existing_output_sidecars(&request.output_path) {
                    let _ = fs::remove_file(&request.output_path);
                    return Err(error);
                }
                Ok(BackupReport {
                    output_path: request.output_path.clone(),
                    format: request.format.clone(),
                    schema_version,
                    page_count,
                    bytes,
                    sha256,
                })
            })();
            if finalize.is_err() {
                cleanup_temp_output(&temp_path);
            }
            finalize
        }
        Err(error) => {
            cleanup_temp_output(&temp_path);
            Err(error)
        }
    }
}

fn backup_to_temp(connection: &Connection, temp_path: &Path) -> Result<i64> {
    register_sqlite_vec_extension()?;
    let mut destination =
        Connection::open_with_flags(temp_path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    configure_connection(&destination)?;
    {
        let backup = backup::Backup::new(connection, &mut destination)?;
        backup.run_to_completion(128, Duration::from_millis(0), None::<fn(backup::Progress)>)?;
    }
    let _journal_mode: String =
        destination.query_row("PRAGMA journal_mode = DELETE", [], |row| row.get(0))?;
    validate_initialized(temp_path, &destination)?;
    count(&destination, "PRAGMA page_count")
}

fn import_store_dry_run(request: &ImportRequest) -> Result<ImportReport> {
    let temp_path = inspection_copy_path()?;
    let result = (|| -> Result<ImportReport> {
        let file = create_new_private_file(&temp_path, false)?;
        drop(file);
        import_store_into_path(&temp_path, request, true, false)
    })();
    cleanup_inspection_copy(&temp_path);
    result
}

fn import_store_create(path: &Path, request: &ImportRequest) -> Result<ImportReport> {
    reject_existing_output_sidecars(path)?;
    let (temp_path, file) = create_temp_output_file(path)?;
    drop(file);

    let result = (|| -> Result<ImportReport> {
        let report = import_store_into_path(&temp_path, request, false, false)?;
        reject_output_sidecar_files(&temp_path)?;
        reject_existing_output_sidecars(path)?;
        publish_temp_output(&temp_path, path)?;
        if let Err(error) = reject_existing_output_sidecars(path) {
            cleanup_import_target(path);
            return Err(error);
        }
        if let Err(error) = init_store(path) {
            cleanup_import_target(path);
            return Err(error);
        }
        Ok(report)
    })();

    if result.is_err() {
        cleanup_temp_output(&temp_path);
    }
    result
}

fn import_store_into_path(
    path: &Path,
    request: &ImportRequest,
    dry_run: bool,
    wal: bool,
) -> Result<ImportReport> {
    let open_path = fs::canonicalize(path)?;
    register_sqlite_vec_extension()?;
    let mut connection = Connection::open_with_flags(
        &open_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )?;
    configure_connection(&connection)?;
    if wal {
        let journal_mode = enable_wal(&connection)?;
        if journal_mode != "wal" {
            return Err(Error::WalUnavailable {
                path: path.to_path_buf(),
                journal_mode,
            });
        }
    } else {
        let _journal_mode: String =
            connection.query_row("PRAGMA journal_mode = DELETE", [], |row| row.get(0))?;
    }
    apply_schema(&connection)?;
    let report = import_archive_into_connection(&mut connection, request, dry_run)?;
    validate_initialized(path, &connection)?;
    Ok(report)
}

fn import_archive_into_connection(
    connection: &mut Connection,
    request: &ImportRequest,
    dry_run: bool,
) -> Result<ImportReport> {
    let transaction = connection.transaction()?;
    clear_import_tables(&transaction)?;
    let (parse, fts_memory_rows, fts_source_episode_rows) = {
        let parse = parse_import_file(&transaction, request)?;
        validate_import_integrity(&transaction)?;
        let (fts_memory_rows, fts_source_episode_rows) = rebuild_fts(&transaction)?;
        #[cfg(feature = "semantic")]
        rebuild_vector_index(&transaction)?;
        (parse, fts_memory_rows, fts_source_episode_rows)
    };
    transaction.commit()?;

    Ok(ImportReport {
        input_path: request.input_path.clone(),
        format: request.format.clone(),
        schema_version: parse.schema_version,
        dry_run,
        tables: parse.tables,
        row_count: parse.row_count,
        bytes: parse.bytes,
        sha256: parse.sha256,
        fts_memory_rows,
        fts_source_episode_rows,
    })
}

fn clear_import_tables(transaction: &Transaction<'_>) -> Result<()> {
    transaction.execute("DELETE FROM memory_fts", [])?;
    transaction.execute("DELETE FROM memory_fts_public", [])?;
    transaction.execute("DELETE FROM source_episode_fts", [])?;
    for table in EXPORT_TABLES.iter().rev() {
        let sql = format!("DELETE FROM {}", table.name);
        transaction.execute(&sql, [])?;
    }
    Ok(())
}

fn validate_import_integrity(transaction: &Transaction<'_>) -> Result<()> {
    let invalid_active_version: Option<String> = transaction
        .query_row(
            "SELECT m.id
             FROM memories m
             LEFT JOIN memory_versions v
                ON v.id = m.active_version_id AND v.memory_id = m.id
             WHERE m.active_version_id IS NULL OR v.id IS NULL
             ORDER BY m.id
             LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(memory_id) = invalid_active_version {
        return Err(Error::InvalidRequest {
            message: format!("import archive has invalid active_version_id for memory {memory_id}"),
        });
    }
    validate_import_spaces_and_silos(transaction)?;
    validate_import_source_refs(transaction)?;
    validate_import_source_episodes(transaction)?;
    validate_import_memory_invariants(transaction)?;
    validate_import_space_isolation(transaction)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn validate_import_spaces_and_silos(transaction: &Transaction<'_>) -> Result<()> {
    let mut spaces = transaction.prepare(
        "SELECT name, display_name, description, default_silo, ontology, config_json,
                created_at, updated_at
         FROM spaces
         ORDER BY name",
    )?;
    let space_rows = spaces.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, String>(7)?,
        ))
    })?;
    for row in space_rows {
        let (
            name,
            display_name,
            description,
            default_silo,
            ontology,
            config_json,
            created_at,
            updated_at,
        ) = row?;
        validate_required_import_metadata_value("space name", &name)?;
        validate_optional_import_metadata_value("space display_name", display_name.as_deref())?;
        validate_optional_import_text_value(
            "space description",
            description.as_deref(),
            MAX_SUMMARY_CHARS,
        )?;
        validate_required_import_metadata_value("space default_silo", &default_silo)?;
        validate_optional_import_text_value(
            "space ontology",
            ontology.as_deref(),
            MAX_SOURCE_REF_JSON_CHARS,
        )?;
        validate_optional_import_json_object("space config_json", config_json.as_deref())?;
        validate_required_import_timestamp("space created_at", &created_at)?;
        validate_required_import_timestamp("space updated_at", &updated_at)?;
    }

    let mut silos = transaction.prepare(
        "SELECT space_name, name, description, retention_policy, default_scope, config_json,
                created_at, updated_at
         FROM silos
         ORDER BY space_name, name",
    )?;
    let silo_rows = silos.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, String>(7)?,
        ))
    })?;
    for row in silo_rows {
        let (
            space_name,
            name,
            description,
            retention_policy,
            default_scope,
            config_json,
            created_at,
            updated_at,
        ) = row?;
        validate_required_import_metadata_value("silo space", &space_name)?;
        validate_required_import_metadata_value("silo name", &name)?;
        validate_optional_import_text_value(
            "silo description",
            description.as_deref(),
            MAX_SUMMARY_CHARS,
        )?;
        validate_required_import_metadata_value("silo retention_policy", &retention_policy)?;
        validate_required_import_metadata_value("silo default_scope", &default_scope)?;
        if !is_supported_scope(&default_scope) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive has unsupported silo default_scope: {default_scope}"
                ),
            });
        }
        validate_optional_import_json_object("silo config_json", config_json.as_deref())?;
        validate_required_import_timestamp("silo created_at", &created_at)?;
        validate_required_import_timestamp("silo updated_at", &updated_at)?;
    }

    let missing_default_silo: Option<String> = transaction
        .query_row(
            "SELECT s.name
             FROM spaces s
             LEFT JOIN silos si
                ON si.space_name = s.name AND si.name = s.default_silo
             WHERE si.name IS NULL
             ORDER BY s.name
             LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(space) = missing_default_silo {
        return Err(Error::InvalidRequest {
            message: format!("import archive has missing default silo for space {space}"),
        });
    }
    Ok(())
}

fn validate_optional_import_text_value(
    name: &str,
    value: Option<&str>,
    max_chars: usize,
) -> Result<()> {
    if let Some(value) = value {
        if value.trim() != value || value.is_empty() || value.chars().count() > max_chars {
            return Err(Error::InvalidRequest {
                message: format!("import archive {name} is not canonical or is too long"),
            });
        }
    }
    Ok(())
}

fn validate_optional_import_json_object(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value) {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid {name}"),
            });
        }
    }
    Ok(())
}

fn validate_required_import_timestamp(name: &str, value: &str) -> Result<()> {
    if value.chars().count() > MAX_TIMESTAMP_CHARS
        || !(is_utc_rfc3339_like(value) || is_sqlite_current_timestamp_like(value))
    {
        return Err(Error::InvalidRequest {
            message: format!("{name} must be a bounded UTC timestamp"),
        });
    }
    Ok(())
}

fn is_sqlite_current_timestamp_like(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 19
        && matches!(bytes.get(4), Some(b'-'))
        && matches!(bytes.get(7), Some(b'-'))
        && matches!(bytes.get(10), Some(b' '))
        && matches!(bytes.get(13), Some(b':'))
        && matches!(bytes.get(16), Some(b':'))
        && timestamp_parts_are_valid(bytes, 0, 5, 8, 11, 14, 17)
}

fn validate_import_source_refs(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare(
        "SELECT id, source_ref_json
         FROM memory_versions
         WHERE source_ref_json IS NOT NULL
         ORDER BY memory_id, version_num, id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (version_id, source_ref_json) = row?;
        if source_ref_json.chars().count() > MAX_SOURCE_REF_JSON_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive source_ref_json exceeds maximum size for version {version_id}"
                ),
            });
        }
        if !JsonValidator::is_object(&source_ref_json) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive has invalid source_ref_json for version {version_id}"
                ),
            });
        }
    }
    Ok(())
}

/// Ingest one document source as isolated, embedded-ready chunks.
///
/// Chunks are written as `source_episodes` rows in a dedicated space (default
/// [`DOCUMENTS_SPACE`]) so they never receive the curated memory tier's
/// supersession/dedup/graph/promotion treatment. Re-ingesting identical content
/// is idempotent: a chunk whose `content_sha256` already exists in the space is
/// skipped rather than duplicated.
///
/// # Errors
/// Returns [`Error::InvalidRequest`] when the request has no chunks or a chunk is
/// empty/too large, or an I/O/storage error when the store cannot be written.
pub fn ingest_source(path: impl AsRef<Path>, request: &IngestRequest) -> Result<IngestReport> {
    validate_ingest_request(request)?;
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let source_type = request
        .source_type
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_INGEST_SOURCE_TYPE)
        .to_string();

    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let now = now_timestamp(&transaction)?;

    let created_space = ensure_ingest_space(&transaction, &space, &now)?;

    let chunk_count = request.chunks.len();
    let chunk_count_i64 = i64::try_from(chunk_count).unwrap_or(i64::MAX);
    let mut created = Vec::new();
    let mut skipped = 0usize;
    for (index, content) in request.chunks.iter().enumerate() {
        let sha = sha256_text(content);
        let chunk_index = i64::try_from(index).unwrap_or(i64::MAX);
        if maybe_repair_ingested_chunk(
            &transaction,
            &space,
            &sha,
            request,
            &source_type,
            chunk_index,
            chunk_count_i64,
            content,
            &now,
        )? {
            skipped += 1;
            continue;
        }
        let id = next_id("src");
        transaction.execute(
            "INSERT INTO source_episodes (
                id, space_name, source_type, source_uri, source_path, source_description,
                content, content_sha256, chunk_index, chunk_count, metadata_json,
                ingest_status, ingested_at, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, 'indexed', ?12, ?12, ?12)",
            params![
                &id,
                &space,
                &source_type,
                request.source_uri,
                request.source_path,
                request.source_description,
                content,
                &sha,
                chunk_index,
                chunk_count_i64,
                request.metadata_json,
                &now,
            ],
        )?;
        transaction.execute(
            "INSERT INTO source_episode_fts (
                source_episode_id, space_name, source_type, source_path, source_description,
                content, metadata_text
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                &id,
                &space,
                &source_type,
                request.source_path,
                request.source_description,
                content,
                request.metadata_json,
            ],
        )?;
        maybe_write_chunk_embedding(&transaction, &id, request, index, &now)?;
        created.push(id);
    }

    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }

    Ok(IngestReport {
        space,
        source_path: request.source_path.clone(),
        chunk_count,
        created,
        skipped,
        created_space: created_space && !request.dry_run,
        dry_run: request.dry_run,
    })
}

/// If this chunk already exists at the same `(space, content, source_path)`,
/// repair its mutable provenance/metadata in place and return `true` (a re-sync,
/// counted as skipped). Returns `false` when no such row exists, so the caller
/// inserts a fresh chunk.
///
/// Identity deliberately includes `source_path`: identical content under a
/// *different* path is an independent chunk (kept, to be surfaced as a duplicate
/// later), not a dedup collision. On a hit, content (and thus the embedding) is
/// unchanged, so the vector is left intact.
#[allow(clippy::too_many_arguments)]
fn maybe_repair_ingested_chunk(
    transaction: &Transaction<'_>,
    space: &str,
    sha: &str,
    request: &IngestRequest,
    source_type: &str,
    chunk_index: i64,
    chunk_count: i64,
    content: &str,
    now: &str,
) -> Result<bool> {
    let existing_id: Option<String> = transaction
        .query_row(
            "SELECT id FROM source_episodes
             WHERE space_name = ?1 AND content_sha256 = ?2 AND source_path IS ?3
             LIMIT 1",
            params![space, sha, request.source_path],
            |row| row.get(0),
        )
        .optional()?;
    let Some(existing_id) = existing_id else {
        return Ok(false);
    };
    transaction.execute(
        "UPDATE source_episodes
         SET source_type = ?2, source_uri = ?3, source_description = ?4,
             chunk_index = ?5, chunk_count = ?6, metadata_json = ?7, updated_at = ?8
         WHERE id = ?1",
        params![
            existing_id,
            source_type,
            request.source_uri,
            request.source_description,
            chunk_index,
            chunk_count,
            request.metadata_json,
            now,
        ],
    )?;
    transaction.execute(
        "UPDATE source_episode_fts
         SET source_type = ?2, source_path = ?3, source_description = ?4,
             content = ?5, metadata_text = ?6
         WHERE source_episode_id = ?1",
        params![
            existing_id,
            source_type,
            request.source_path,
            request.source_description,
            content,
            request.metadata_json,
        ],
    )?;
    Ok(true)
}

/// Create the ingest target space (with standard silos) if it does not yet
/// exist. Returns `true` when this call created it. Idempotent.
fn ensure_ingest_space(transaction: &Transaction<'_>, space: &str, now: &str) -> Result<bool> {
    if space_exists(transaction, space)? {
        return Ok(false);
    }
    transaction.execute(
        "INSERT INTO spaces (
            name, display_name, description, default_silo, created_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?5)",
        params![
            space,
            "Documents",
            "Ingested document chunks (isolated from curated memory).",
            DEFAULT_DURABLE_SILO,
            now,
        ],
    )?;
    seed_standard_silos(transaction, space, DEFAULT_DURABLE_SILO, now)?;
    Ok(true)
}

fn validate_ingest_request(request: &IngestRequest) -> Result<()> {
    if request.chunks.is_empty() {
        return Err(Error::InvalidRequest {
            message: "ingest request must include at least one chunk".to_string(),
        });
    }
    if let Some(space) = request.space.as_deref() {
        if space.trim().is_empty() {
            return Err(Error::InvalidRequest {
                message: "ingest space must not be blank when provided".to_string(),
            });
        }
    }
    for (index, content) in request.chunks.iter().enumerate() {
        if content.trim().is_empty() {
            return Err(Error::InvalidRequest {
                message: format!("ingest chunk {index} must not be empty"),
            });
        }
        if content.chars().count() > MAX_CONTENT_CHARS {
            return Err(Error::InvalidRequest {
                message: format!("ingest chunk {index} exceeds the maximum content size"),
            });
        }
    }
    if let Some(metadata) = request.metadata_json.as_deref() {
        // Match the import invariant (`validate_import_source_metadata_json`): a
        // malformed payload here would otherwise create a store the engine's own
        // import validation later rejects.
        if metadata.chars().count() > MAX_SOURCE_REF_JSON_CHARS
            || !JsonValidator::is_object(metadata)
        {
            return Err(Error::InvalidRequest {
                message: "ingest metadata_json must be a valid JSON object".to_string(),
            });
        }
    }
    if let Some(embeddings) = request.embeddings.as_ref() {
        if embeddings.len() != request.chunks.len() {
            return Err(Error::InvalidRequest {
                message: "ingest embeddings length must match chunks length".to_string(),
            });
        }
        if embeddings.iter().any(Vec::is_empty) {
            return Err(Error::InvalidRequest {
                message: "ingest embedding vectors must not be empty".to_string(),
            });
        }
        if request
            .embedding_model_id
            .as_deref()
            .is_none_or(|model| model.trim().is_empty())
        {
            return Err(Error::InvalidRequest {
                message: "ingest embedding_model_id is required when embeddings are supplied"
                    .to_string(),
            });
        }
    }
    Ok(())
}

/// Write the chunk embedding for a just-created `source_episodes` row, when the
/// request carries one for this chunk index. No-op on non-semantic builds.
#[cfg(feature = "semantic")]
fn maybe_write_chunk_embedding(
    transaction: &Transaction<'_>,
    source_episode_id: &str,
    request: &IngestRequest,
    index: usize,
    now: &str,
) -> Result<()> {
    let Some(embeddings) = request.embeddings.as_ref() else {
        return Ok(());
    };
    let Some(embedding) = embeddings.get(index) else {
        return Ok(());
    };
    let model = request.embedding_model_id.as_deref().unwrap_or("unknown");
    write_source_episode_embedding(
        transaction,
        source_episode_id,
        model,
        embedding.len(),
        embedding,
        now,
    )
}

#[cfg(not(feature = "semantic"))]
#[allow(clippy::unnecessary_wraps)]
fn maybe_write_chunk_embedding(
    _transaction: &Transaction<'_>,
    _source_episode_id: &str,
    _request: &IngestRequest,
    _index: usize,
    _now: &str,
) -> Result<()> {
    Ok(())
}

/// Vector index table name for source-episode chunk embeddings at `dims`.
#[cfg(feature = "semantic")]
fn source_episode_vec_table(dims: usize) -> String {
    format!("source_episode_vec_{dims}")
}

#[cfg(feature = "semantic")]
fn ensure_source_episode_vec_table(
    connection: &Connection,
    table: &str,
    dims: usize,
) -> Result<()> {
    if !table_exists(connection, table)? {
        connection.execute_batch(&format!(
            "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(\n  source_episode_id TEXT NOT NULL,\n  embedding FLOAT[{dims}]\n);"
        ))?;
    }
    Ok(())
}

/// Write one chunk embedding: the canonical `embeddings` sidecar row (keyed by
/// `source_episode_id`, `memory_id`/`version_id` NULL) plus the rebuildable ANN
/// projection in `source_episode_vec_{dims}`. Enforces the store's single active
/// embedding model so chunk vectors share the model used for query embeddings.
#[cfg(feature = "semantic")]
fn write_source_episode_embedding(
    transaction: &Transaction<'_>,
    source_episode_id: &str,
    model: &str,
    dims: usize,
    embedding: &[f32],
    now: &str,
) -> Result<()> {
    enforce_active_embedding_model(transaction, model, dims, now)?;
    let table = source_episode_vec_table(dims);
    ensure_source_episode_vec_table(transaction, &table, dims)?;
    transaction.execute(
        "INSERT INTO embeddings (id, source_episode_id, embedding_model, dimensions, vector_blob, status, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, 'ready', ?6, ?6)",
        params![
            next_id("emb"),
            source_episode_id,
            model,
            i64::try_from(dims).unwrap_or(i64::MAX),
            embedding_to_blob(embedding),
            now,
        ],
    )?;
    let embedding_json = embedding_json(embedding)?;
    transaction.execute(
        &format!("INSERT INTO {table} (source_episode_id, embedding) VALUES (?1, ?2)"),
        params![source_episode_id, embedding_json],
    )?;
    Ok(())
}

/// Hybrid (BM25 + vector) search over ingested document chunks in one space.
///
/// Runs a lexical arm over `source_episode_fts` and, when a query embedding is
/// supplied and a `source_episode_vec_{dims}` index exists, a semantic arm; the
/// two ranked lists are fused with Reciprocal Rank Fusion. Results carry a
/// citation back to `source_path` + `chunk_index`. Searches only the requested
/// space (default [`DOCUMENTS_SPACE`]); curated memories are never returned.
///
/// # Errors
/// Returns an error when the store is missing/incompatible or `SQLite` fails.
pub fn search_documents(
    path: impl AsRef<Path>,
    request: &DocumentSearchRequest,
) -> Result<DocumentSearchReport> {
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let limit = if request.limit == 0 {
        DEFAULT_DOCUMENT_SEARCH_LIMIT
    } else {
        request.limit.min(MAX_SEARCH_LIMIT)
    };
    let connection = open_initialized_read_fast(path.as_ref())?;
    let report = with_read_snapshot(&connection, move |connection| {
        search_documents_on_connection(connection, request, space, limit)
    })?;
    // Best-effort retrieval instrumentation: records which chunks earned traffic,
    // the signal that later drives usage-based promotion. Never fails the search.
    if !request.skip_recall_log && !report.results.is_empty() {
        if let Err(error) = record_document_retrievals(path.as_ref(), &request.query, &report) {
            eprintln!("[memkeeper] document retrieval logging failed: {error}");
        }
    }
    Ok(report)
}

fn search_documents_on_connection(
    connection: &Connection,
    request: &DocumentSearchRequest,
    space: String,
    limit: usize,
) -> Result<DocumentSearchReport> {
    let candidate_limit = limit
        .saturating_mul(4)
        .max(limit.saturating_add(32))
        .min(MAX_SEARCH_LIMIT.saturating_mul(4));
    let terms = search_terms(&request.query);
    let fts_query = document_fts_query(&terms);
    let lexical = document_lexical_candidates(
        connection,
        &space,
        &fts_query,
        candidate_limit,
        request.include_content,
        request.snippet_chars,
    )?;
    let semantic_attempted = request.embedding.is_some();
    let semantic = match request.embedding.as_deref() {
        Some(embedding) => document_semantic_candidates(
            connection,
            &space,
            embedding,
            candidate_limit,
            request.include_content,
            request.snippet_chars,
        )?,
        None => Vec::new(),
    };
    let results = document_rrf_merge(semantic, lexical, limit);
    let strategy = if semantic_attempted {
        "hybrid_rrf_v0"
    } else {
        "lexical_only_v0"
    };
    Ok(DocumentSearchReport {
        strategy: strategy.to_string(),
        semantic_attempted,
        space,
        results,
    })
}

/// Build a safe FTS5 MATCH string: quoted, OR-joined terms. `search_terms`
/// already lowercases and strips to alphanumerics, so quoting cannot inject.
fn document_fts_query(terms: &[String]) -> String {
    terms
        .iter()
        .map(|term| format!("\"{term}\""))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn document_snippet_sql(snippet_chars: usize) -> String {
    if snippet_chars == 0 {
        "''".to_string()
    } else {
        format!("substr(se.content, 1, {snippet_chars})")
    }
}

fn document_hit_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DocumentChunkHit> {
    Ok(DocumentChunkHit {
        source_episode_id: row.get(0)?,
        space: row.get(1)?,
        source_type: row.get(2)?,
        source_path: row.get(3)?,
        source_uri: row.get(4)?,
        chunk_index: row.get(5)?,
        chunk_count: row.get(6)?,
        snippet: row.get::<_, Option<String>>(7)?.unwrap_or_default(),
        content: row.get(8)?,
    })
}

fn document_lexical_candidates(
    connection: &Connection,
    space: &str,
    fts_query: &str,
    candidate_limit: usize,
    include_content: bool,
    snippet_chars: usize,
) -> Result<Vec<DocumentChunkHit>> {
    if fts_query.is_empty() {
        return Ok(Vec::new());
    }
    let snippet_sql = document_snippet_sql(snippet_chars);
    let content_sql = if include_content {
        "se.content"
    } else {
        "NULL"
    };
    let sql = format!(
        "SELECT se.id, se.space_name, se.source_type, se.source_path, se.source_uri,
                se.chunk_index, se.chunk_count, {snippet_sql}, {content_sql}
         FROM source_episode_fts
         JOIN source_episodes se ON se.id = source_episode_fts.source_episode_id
         WHERE source_episode_fts MATCH ?1 AND source_episode_fts.space_name = ?2
         ORDER BY bm25(source_episode_fts) ASC
         LIMIT {candidate_limit}"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params![fts_query, space], document_hit_from_row)?;
    let mut hits = Vec::new();
    for row in rows {
        hits.push(row?);
    }
    Ok(hits)
}

#[cfg(feature = "semantic")]
fn document_semantic_candidates(
    connection: &Connection,
    space: &str,
    embedding: &[f32],
    candidate_limit: usize,
    include_content: bool,
    snippet_chars: usize,
) -> Result<Vec<DocumentChunkHit>> {
    let dims = embedding.len();
    let table = source_episode_vec_table(dims);
    if !table_exists(connection, &table)? {
        return Ok(Vec::new());
    }
    let snippet_sql = document_snippet_sql(snippet_chars);
    let content_sql = if include_content {
        "se.content"
    } else {
        "NULL"
    };
    let sql = format!(
        "SELECT se.id, se.space_name, se.source_type, se.source_path, se.source_uri,
                se.chunk_index, se.chunk_count, {snippet_sql}, {content_sql}
         FROM {table} t
         JOIN source_episodes se ON se.id = t.source_episode_id
         WHERE t.embedding MATCH ?1 AND k = {candidate_limit} AND se.space_name = ?2
         ORDER BY t.distance ASC"
    );
    let embedding_json = embedding_json(embedding)?;
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params![embedding_json, space], document_hit_from_row)?;
    let mut hits = Vec::new();
    for row in rows {
        hits.push(row?);
    }
    Ok(hits)
}

#[cfg(not(feature = "semantic"))]
#[allow(clippy::unnecessary_wraps)]
fn document_semantic_candidates(
    _connection: &Connection,
    _space: &str,
    _embedding: &[f32],
    _candidate_limit: usize,
    _include_content: bool,
    _snippet_chars: usize,
) -> Result<Vec<DocumentChunkHit>> {
    Ok(Vec::new())
}

struct DocumentMergeEntry {
    score: f64,
    hit: DocumentChunkHit,
    semantic: bool,
    lexical: bool,
}

/// Fuse the semantic and lexical ranked lists with Reciprocal Rank Fusion
/// (`score = sum 1/(K + rank)`, K=60). Deterministic: ties break on chunk id.
#[allow(clippy::cast_precision_loss)]
fn document_rrf_merge(
    semantic: Vec<DocumentChunkHit>,
    lexical: Vec<DocumentChunkHit>,
    limit: usize,
) -> Vec<DocumentSearchResult> {
    const RRF_K: f64 = 60.0;
    let mut entries: std::collections::BTreeMap<String, DocumentMergeEntry> =
        std::collections::BTreeMap::new();
    for (rank, hit) in semantic.into_iter().enumerate() {
        let contribution = 1.0 / (RRF_K + rank as f64 + 1.0);
        match entries.get_mut(&hit.source_episode_id) {
            Some(entry) => {
                entry.score += contribution;
                entry.semantic = true;
            }
            None => {
                entries.insert(
                    hit.source_episode_id.clone(),
                    DocumentMergeEntry {
                        score: contribution,
                        hit,
                        semantic: true,
                        lexical: false,
                    },
                );
            }
        }
    }
    for (rank, hit) in lexical.into_iter().enumerate() {
        let contribution = 1.0 / (RRF_K + rank as f64 + 1.0);
        match entries.get_mut(&hit.source_episode_id) {
            Some(entry) => {
                entry.score += contribution;
                entry.lexical = true;
            }
            None => {
                entries.insert(
                    hit.source_episode_id.clone(),
                    DocumentMergeEntry {
                        score: contribution,
                        hit,
                        semantic: false,
                        lexical: true,
                    },
                );
            }
        }
    }
    let mut merged: Vec<DocumentMergeEntry> = entries.into_values().collect();
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.hit.source_episode_id.cmp(&b.hit.source_episode_id))
    });
    merged.truncate(limit);
    merged
        .into_iter()
        .enumerate()
        .map(|(index, entry)| {
            let match_type = match (entry.semantic, entry.lexical) {
                (true, true) => "hybrid",
                (true, false) => "semantic",
                _ => "lexical",
            };
            DocumentSearchResult {
                rank: index + 1,
                source_episode_id: entry.hit.source_episode_id,
                space: entry.hit.space,
                source_type: entry.hit.source_type,
                source_path: entry.hit.source_path,
                source_uri: entry.hit.source_uri,
                chunk_index: entry.hit.chunk_index,
                chunk_count: entry.hit.chunk_count,
                snippet: entry.hit.snippet,
                content: entry.hit.content,
                score: entry.score,
                match_type: match_type.to_string(),
            }
        })
        .collect()
}

/// `source_episode_recall_events` is engine-owned telemetry: which document
/// chunks earned retrieval traffic. Created lazily (not part of schema
/// validation) and excluded from export/import, like `recall_events`. This is
/// the signal that later drives usage-based promotion of chunks into memories.
const SOURCE_EPISODE_RECALL_DDL: &str = "
CREATE TABLE IF NOT EXISTS source_episode_recall_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_episode_id TEXT NOT NULL,
  space_name TEXT NOT NULL,
  ts TEXT NOT NULL,
  query TEXT,
  query_sha256 TEXT,
  rank INTEGER,
  score REAL,
  match_type TEXT
);
CREATE INDEX IF NOT EXISTS idx_se_recall_src ON source_episode_recall_events(source_episode_id);
CREATE INDEX IF NOT EXISTS idx_se_recall_ts ON source_episode_recall_events(ts);
";

/// Record one retrieval event per returned chunk. `query_sha256` lets the
/// promotion step count distinct queries (query diversity) without storing the
/// raw query repeatedly. Opens its own short write transaction.
fn record_document_retrievals(
    path: &Path,
    query: &str,
    report: &DocumentSearchReport,
) -> Result<usize> {
    let mut connection = open_initialized_write(path)?;
    let transaction = connection.transaction()?;
    transaction.execute_batch(SOURCE_EPISODE_RECALL_DDL)?;
    let now = now_timestamp(&transaction)?;
    let query_opt = (!query.is_empty()).then_some(query);
    let query_sha = query_opt.map(sha256_text);
    let mut recorded = 0_usize;
    {
        let mut insert = transaction.prepare_cached(
            "INSERT INTO source_episode_recall_events
                (source_episode_id, space_name, ts, query, query_sha256, rank, score, match_type)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        )?;
        for result in &report.results {
            insert.execute(params![
                &result.source_episode_id,
                &report.space,
                &now,
                query_opt,
                query_sha.as_deref(),
                i64::try_from(result.rank).unwrap_or(i64::MAX),
                result.score,
                &result.match_type,
            ])?;
            recorded += 1;
        }
    }
    transaction.commit()?;
    Ok(recorded)
}

/// Default chunk limit for [`get_document`].
pub const DEFAULT_DOCUMENT_GET_LIMIT: usize = 500;
/// Default candidate limit for [`promotion_candidates`].
pub const DEFAULT_PROMOTION_CANDIDATE_LIMIT: usize = 20;

/// Request to fetch a document's chunks by path or by chunk id.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocumentGetRequest {
    /// Fetch all chunks of the document at this `source_path`.
    pub source_path: Option<String>,
    /// Fetch a single chunk by its `source_episodes` id.
    pub source_episode_id: Option<String>,
    /// Space to look in (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// Include full chunk content (else metadata only).
    pub include_content: bool,
    /// Max chunks to return.
    pub limit: usize,
}

/// One stored document chunk with full provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentChunk {
    /// `source_episodes` id.
    pub source_episode_id: String,
    /// Space.
    pub space: String,
    /// Provenance type.
    pub source_type: String,
    /// Source document path.
    pub source_path: Option<String>,
    /// Source document URI.
    pub source_uri: Option<String>,
    /// Chunk position within the source.
    pub chunk_index: i64,
    /// Total chunks in the source.
    pub chunk_count: i64,
    /// Content hash.
    pub content_sha256: Option<String>,
    /// Ingest status (`indexed`/`extracted`/...).
    pub ingest_status: String,
    /// Chunk content (when requested).
    pub content: Option<String>,
}

/// Result of a document fetch.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentGetReport {
    /// Space searched.
    pub space: String,
    /// Chunks, ordered by `chunk_index`.
    pub chunks: Vec<DocumentChunk>,
}

/// Request for usage-driven promotion candidates.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PromotionCandidatesRequest {
    /// Space to rank within (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// Minimum total retrieval hits to qualify.
    pub min_hits: usize,
    /// Minimum distinct queries that retrieved the chunk (query diversity).
    pub min_distinct_queries: usize,
    /// Max candidates to return.
    pub limit: usize,
    /// Include full chunk content.
    pub include_content: bool,
    /// Include chunks already promoted (`ingest_status = 'extracted'`).
    pub include_extracted: bool,
}

/// One promotion candidate: a chunk that earned retrieval traffic.
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionCandidate {
    /// `source_episodes` id.
    pub source_episode_id: String,
    /// Space.
    pub space: String,
    /// Source document path.
    pub source_path: Option<String>,
    /// Source document URI.
    pub source_uri: Option<String>,
    /// Chunk position within the source.
    pub chunk_index: i64,
    /// Total chunks in the source.
    pub chunk_count: i64,
    /// Total retrieval hits.
    pub hits: i64,
    /// Distinct queries that retrieved the chunk.
    pub distinct_queries: i64,
    /// Most recent retrieval timestamp.
    pub last_hit: String,
    /// Chunk content (when requested).
    pub content: Option<String>,
}

/// Result of ranking promotion candidates.
#[derive(Debug, Clone, PartialEq)]
pub struct PromotionCandidatesReport {
    /// Space ranked.
    pub space: String,
    /// Candidates, highest signal first.
    pub candidates: Vec<PromotionCandidate>,
}

/// Default cluster cap for [`document_duplicates`].
pub const DEFAULT_DUPLICATE_CLUSTER_LIMIT: usize = 50;

/// Request to surface exact-content duplicate document chunks: independent rows
/// that ingest kept because they share content but came from different sources.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocumentDuplicatesRequest {
    /// Space to scan (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// Max duplicate clusters to return (0 = [`DEFAULT_DUPLICATE_CLUSTER_LIMIT`]).
    pub limit: usize,
    /// Snippet length in characters for the shared-content preview (0 = none).
    pub snippet_chars: usize,
}

/// One member of a duplicate cluster: a stored chunk that shares its content
/// with the other members.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateChunkMember {
    /// `source_episodes` id.
    pub source_episode_id: String,
    /// Source document path (citation).
    pub source_path: Option<String>,
    /// Source document URI (citation).
    pub source_uri: Option<String>,
    /// Chunk position within its source.
    pub chunk_index: i64,
    /// Total chunks in its source.
    pub chunk_count: i64,
    /// Ingest status (`indexed`/`extracted`/...).
    pub ingest_status: String,
    /// When the chunk was ingested.
    pub ingested_at: String,
}

/// A set of document chunks sharing identical content (`content_sha256`) across
/// different sources — the independent duplicates ingest intentionally keeps.
#[derive(Debug, Clone, PartialEq)]
pub struct DuplicateChunkCluster {
    /// Shared content hash.
    pub content_sha256: String,
    /// Number of chunks sharing this content.
    pub member_count: i64,
    /// Preview of the shared content (empty when not requested).
    pub snippet: String,
    /// The duplicate chunks, ordered by source path then chunk index.
    pub members: Vec<DuplicateChunkMember>,
}

/// Result of scanning for duplicate document chunks.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentDuplicatesReport {
    /// Space scanned.
    pub space: String,
    /// Duplicate clusters, most-duplicated first.
    pub clusters: Vec<DuplicateChunkCluster>,
}

/// Fetch a document's chunks by `source_path` (all chunks, ordered) or by a
/// single `source_episode_id`. One of the two must be set.
///
/// # Errors
/// Returns [`Error::InvalidRequest`] when neither selector is set, or a storage
/// error.
pub fn get_document(
    path: impl AsRef<Path>,
    request: &DocumentGetRequest,
) -> Result<DocumentGetReport> {
    if request.source_path.is_none() && request.source_episode_id.is_none() {
        return Err(Error::InvalidRequest {
            message: "document-get requires source_path or source_episode_id".to_string(),
        });
    }
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let limit = if request.limit == 0 {
        DEFAULT_DOCUMENT_GET_LIMIT
    } else {
        request.limit.min(5_000)
    };
    let content_sql = if request.include_content {
        "content"
    } else {
        "NULL"
    };
    let (selector, selector_value) = if let Some(id) = request.source_episode_id.as_deref() {
        ("id = ?2", id.to_string())
    } else {
        (
            "source_path = ?2",
            request.source_path.clone().unwrap_or_default(),
        )
    };
    let sql = format!(
        "SELECT id, space_name, source_type, source_path, source_uri, chunk_index, chunk_count,
                content_sha256, ingest_status, {content_sql}
         FROM source_episodes
         WHERE space_name = ?1 AND {selector}
         ORDER BY chunk_index ASC, id ASC
         LIMIT {limit}"
    );
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(params![space, selector_value], document_chunk_from_row)?;
        let mut chunks = Vec::new();
        for row in rows {
            chunks.push(row?);
        }
        Ok(DocumentGetReport {
            space: space.clone(),
            chunks,
        })
    })
}

fn document_chunk_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<DocumentChunk> {
    Ok(DocumentChunk {
        source_episode_id: row.get(0)?,
        space: row.get(1)?,
        source_type: row.get(2)?,
        source_path: row.get(3)?,
        source_uri: row.get(4)?,
        chunk_index: row.get(5)?,
        chunk_count: row.get(6)?,
        content_sha256: row.get(7)?,
        ingest_status: row.get(8)?,
        content: row.get(9)?,
    })
}

/// Rank document chunks that have earned retrieval traffic, as promotion
/// candidates (usage-driven promotion signal). Aggregates
/// `source_episode_recall_events` by chunk: total hits, distinct queries (via
/// `query_sha256`), and recency. Returns empty if nothing has been searched yet.
///
/// # Errors
/// Returns a storage error on `SQLite` failure.
pub fn promotion_candidates(
    path: impl AsRef<Path>,
    request: &PromotionCandidatesRequest,
) -> Result<PromotionCandidatesReport> {
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let limit = if request.limit == 0 {
        DEFAULT_PROMOTION_CANDIDATE_LIMIT
    } else {
        request.limit.min(200)
    };
    let min_hits = i64::try_from(request.min_hits.max(1)).unwrap_or(1);
    let min_distinct = i64::try_from(request.min_distinct_queries).unwrap_or(0);
    let content_sql = if request.include_content {
        "se.content"
    } else {
        "NULL"
    };
    let extracted_filter = if request.include_extracted {
        ""
    } else {
        "AND se.ingest_status != 'extracted'"
    };
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        if !table_exists(connection, "source_episode_recall_events")? {
            return Ok(PromotionCandidatesReport {
                space: space.clone(),
                candidates: Vec::new(),
            });
        }
        let sql = format!(
            "SELECT r.source_episode_id, se.space_name, se.source_path, se.source_uri,
                    se.chunk_index, se.chunk_count,
                    COUNT(*) AS hits,
                    COUNT(DISTINCT r.query_sha256) AS distinct_queries,
                    MAX(r.ts) AS last_hit,
                    {content_sql}
             FROM source_episode_recall_events r
             JOIN source_episodes se ON se.id = r.source_episode_id
             WHERE r.space_name = ?1 {extracted_filter}
             GROUP BY r.source_episode_id
             HAVING COUNT(*) >= ?2 AND COUNT(DISTINCT r.query_sha256) >= ?3
             ORDER BY distinct_queries DESC, hits DESC, last_hit DESC
             LIMIT {limit}"
        );
        let mut statement = connection.prepare(&sql)?;
        let rows = statement.query_map(
            params![space, min_hits, min_distinct],
            promotion_candidate_from_row,
        )?;
        let mut candidates = Vec::new();
        for row in rows {
            candidates.push(row?);
        }
        Ok(PromotionCandidatesReport {
            space: space.clone(),
            candidates,
        })
    })
}

/// Surface exact-content duplicate document chunks: clusters of chunks in a
/// space that share the same `content_sha256` across two or more rows (e.g.
/// identical content ingested under different `source_path`s). Read-only; never
/// records retrieval telemetry. Clusters are returned most-duplicated first.
///
/// # Errors
/// Returns a storage error when the store cannot be read.
pub fn document_duplicates(
    path: impl AsRef<Path>,
    request: &DocumentDuplicatesRequest,
) -> Result<DocumentDuplicatesReport> {
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let limit = if request.limit == 0 {
        DEFAULT_DUPLICATE_CLUSTER_LIMIT
    } else {
        request.limit.min(500)
    };
    let snippet_chars = i64::try_from(request.snippet_chars).unwrap_or(0).max(0);
    let connection = open_initialized_read_fast(path.as_ref())?;
    with_read_snapshot(&connection, |connection| {
        if !table_exists(connection, "source_episodes")? {
            return Ok(DocumentDuplicatesReport {
                space: space.clone(),
                clusters: Vec::new(),
            });
        }
        // 1. Qualifying content hashes (2+ rows in the space), most-duplicated
        //    first, capped to `limit` clusters.
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut hash_statement = connection.prepare(
            "SELECT content_sha256, COUNT(*) AS n
             FROM source_episodes
             WHERE space_name = ?1 AND content_sha256 IS NOT NULL
             GROUP BY content_sha256
             HAVING n > 1
             ORDER BY n DESC, content_sha256 ASC
             LIMIT ?2",
        )?;
        let hashes: Vec<(String, i64)> = hash_statement
            .query_map(params![space, limit_i64], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })?
            .collect::<rusqlite::Result<_>>()?;

        // 2. Members per cluster. All members share content, so any member's
        //    snippet represents the cluster.
        let mut member_statement = connection.prepare(
            "SELECT id, source_path, source_uri, chunk_index, chunk_count,
                    ingest_status, ingested_at, substr(content, 1, ?3)
             FROM source_episodes
             WHERE space_name = ?1 AND content_sha256 = ?2
             ORDER BY source_path, chunk_index, id",
        )?;
        let mut clusters = Vec::with_capacity(hashes.len());
        for (content_sha256, member_count) in hashes {
            let mut snippet = String::new();
            let mut members = Vec::new();
            let rows = member_statement.query_map(
                params![space, content_sha256, snippet_chars],
                |row| {
                    Ok((
                        DuplicateChunkMember {
                            source_episode_id: row.get(0)?,
                            source_path: row.get(1)?,
                            source_uri: row.get(2)?,
                            chunk_index: row.get(3)?,
                            chunk_count: row.get(4)?,
                            ingest_status: row.get(5)?,
                            ingested_at: row.get(6)?,
                        },
                        row.get::<_, Option<String>>(7)?,
                    ))
                },
            )?;
            for row in rows {
                let (member, member_snippet) = row?;
                if snippet.is_empty() {
                    if let Some(text) = member_snippet {
                        snippet = text.split_whitespace().collect::<Vec<_>>().join(" ");
                    }
                }
                members.push(member);
            }
            clusters.push(DuplicateChunkCluster {
                content_sha256,
                member_count,
                snippet,
                members,
            });
        }
        Ok(DocumentDuplicatesReport {
            space: space.clone(),
            clusters,
        })
    })
}

/// Request to mark document chunks as extracted (i.e. promoted to memory), so
/// they stop surfacing as promotion candidates.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct MarkExtractedRequest {
    /// Space the chunks live in (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// `source_episodes` ids to mark `extracted`.
    pub source_episode_ids: Vec<String>,
}

/// Result of marking chunks extracted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkExtractedReport {
    /// Space operated on.
    pub space: String,
    /// Number of chunks whose status changed.
    pub updated: usize,
}

/// Mark the given chunks `ingest_status = 'extracted'`. Used after a chunk is
/// promoted into a memory, so usage-driven promotion does not re-propose it.
///
/// # Errors
/// Returns [`Error::InvalidRequest`] when no ids are given, or a storage error.
pub fn mark_source_episodes_extracted(
    path: impl AsRef<Path>,
    request: &MarkExtractedRequest,
) -> Result<MarkExtractedReport> {
    if request.source_episode_ids.is_empty() {
        return Err(Error::InvalidRequest {
            message: "mark-extracted requires at least one source_episode_id".to_string(),
        });
    }
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let now = now_timestamp(&transaction)?;
    let mut updated = 0usize;
    {
        let mut statement = transaction.prepare(
            "UPDATE source_episodes SET ingest_status = 'extracted', updated_at = ?1
             WHERE space_name = ?2 AND id = ?3",
        )?;
        for id in &request.source_episode_ids {
            updated += statement.execute(params![now, space, id])?;
        }
    }
    transaction.commit()?;
    Ok(MarkExtractedReport { space, updated })
}

/// Request to prune (delete) specific document chunks by id. User-driven cleanup
/// of duplicates surfaced by [`document_duplicates`]: the caller chooses exactly
/// which chunks to remove, so deletion is always explicit (never automatic).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DocumentPruneRequest {
    /// Space the chunks live in (defaults to [`DOCUMENTS_SPACE`]).
    pub space: Option<String>,
    /// `source_episodes` ids to delete.
    pub source_episode_ids: Vec<String>,
    /// Validate and report without deleting.
    pub dry_run: bool,
}

/// Result of pruning document chunks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentPruneReport {
    /// Space operated on.
    pub space: String,
    /// Number of ids requested for deletion.
    pub requested: usize,
    /// Ids actually deleted (those that existed in the space).
    pub deleted: Vec<String>,
    /// True when validated but rolled back.
    pub dry_run: bool,
}

/// Delete the given document chunks and all their derived rows (FTS, canonical
/// embedding, and rebuildable ANN projections). Only ids present in `space` are
/// removed; unknown ids are ignored (surfaced via `requested` vs `deleted`).
/// Intended for user-driven duplicate cleanup after reviewing
/// [`document_duplicates`].
///
/// # Errors
/// Returns [`Error::InvalidRequest`] when no ids are given, or a storage error.
pub fn prune_documents(
    path: impl AsRef<Path>,
    request: &DocumentPruneRequest,
) -> Result<DocumentPruneReport> {
    if request.source_episode_ids.is_empty() {
        return Err(Error::InvalidRequest {
            message: "document-prune requires at least one source_episode_id".to_string(),
        });
    }
    let space = request
        .space
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DOCUMENTS_SPACE)
        .to_string();
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let vec_tables = source_episode_vec_tables(&transaction)?;
    let mut deleted = Vec::new();
    for id in &request.source_episode_ids {
        // Scope the delete to the space: an id outside it is left untouched and
        // its derived rows are not removed.
        let removed = transaction.execute(
            "DELETE FROM source_episodes WHERE space_name = ?1 AND id = ?2",
            params![space, id],
        )?;
        if removed == 0 {
            continue;
        }
        transaction.execute(
            "DELETE FROM source_episode_fts WHERE source_episode_id = ?1",
            params![id],
        )?;
        transaction.execute(
            "DELETE FROM embeddings WHERE source_episode_id = ?1",
            params![id],
        )?;
        for table in &vec_tables {
            transaction.execute(
                &format!("DELETE FROM {table} WHERE source_episode_id = ?1"),
                params![id],
            )?;
        }
        deleted.push(id.clone());
    }
    if request.dry_run {
        transaction.rollback()?;
    } else {
        transaction.commit()?;
    }
    Ok(DocumentPruneReport {
        space,
        requested: request.source_episode_ids.len(),
        deleted,
        dry_run: request.dry_run,
    })
}

/// Names of the rebuildable `source_episode_vec_{dims}` ANN tables present in the
/// store (zero on non-semantic stores). Restricted to the `vec0` virtual tables:
/// the `sql LIKE '%USING vec0%'` filter excludes vec0's internal shadow tables
/// (`_info`/`_chunks`/`_rowids`/...), which carry no `source_episode_id` column
/// and are managed automatically when the virtual table row is deleted.
fn source_episode_vec_tables(transaction: &Transaction<'_>) -> Result<Vec<String>> {
    let mut statement = transaction.prepare(
        "SELECT name FROM sqlite_master
         WHERE type = 'table' AND name LIKE 'source_episode_vec_%'
           AND sql LIKE '%USING vec0%'",
    )?;
    let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
    Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
}

fn promotion_candidate_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<PromotionCandidate> {
    Ok(PromotionCandidate {
        source_episode_id: row.get(0)?,
        space: row.get(1)?,
        source_path: row.get(2)?,
        source_uri: row.get(3)?,
        chunk_index: row.get(4)?,
        chunk_count: row.get(5)?,
        hits: row.get(6)?,
        distinct_queries: row.get(7)?,
        last_hit: row.get(8)?,
        content: row.get(9)?,
    })
}

fn validate_import_source_episodes(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare(
        "SELECT id, space_name, source_type, source_path, source_description, content,
            content_sha256, chunk_index, chunk_count, metadata_json
         FROM source_episodes
         ORDER BY id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(ImportSourceEpisodeInvariantRow {
            id: row.get(0)?,
            space_name: row.get(1)?,
            source_type: row.get(2)?,
            source_path: row.get(3)?,
            source_description: row.get(4)?,
            content: row.get(5)?,
            content_sha256: row.get(6)?,
            chunk_index: row.get(7)?,
            chunk_count: row.get(8)?,
            metadata_json: row.get(9)?,
        })
    })?;
    for row in rows {
        validate_import_source_episode_row(&row?)?;
    }
    Ok(())
}

struct ImportSourceEpisodeInvariantRow {
    id: String,
    space_name: String,
    source_type: String,
    source_path: Option<String>,
    source_description: Option<String>,
    content: Option<String>,
    content_sha256: Option<String>,
    chunk_index: i64,
    chunk_count: i64,
    metadata_json: Option<String>,
}

fn validate_import_source_episode_row(row: &ImportSourceEpisodeInvariantRow) -> Result<()> {
    validate_required_import_metadata_value("source episode id", &row.id)?;
    validate_required_import_metadata_value("source episode space", &row.space_name)?;
    validate_required_import_metadata_value("source episode type", &row.source_type)?;
    validate_optional_import_long_text("source episode path", row.source_path.as_deref())?;
    validate_optional_import_long_text(
        "source episode description",
        row.source_description.as_deref(),
    )?;
    if row.chunk_count < 1 || row.chunk_index < 0 || row.chunk_index >= row.chunk_count {
        return Err(Error::InvalidRequest {
            message: format!(
                "import archive has invalid chunk indexes for source {}",
                row.id
            ),
        });
    }
    if row
        .content
        .as_deref()
        .is_some_and(|content| content.chars().count() > MAX_CONTENT_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: format!(
                "import archive source content is too large for source {}",
                row.id
            ),
        });
    }
    if let Some(hash) = row.content_sha256.as_deref() {
        validate_required_import_metadata_value("source episode content_sha256", hash)?;
        if row.content.as_deref().map(sha256_text) != Some(hash.to_string()) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive has content hash mismatch for source {}",
                    row.id
                ),
            });
        }
    }
    validate_import_source_metadata_json(&row.id, row.metadata_json.as_deref())
}

fn validate_optional_import_long_text(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.trim() != value
            || value.is_empty()
            || value.chars().count() > MAX_SOURCE_REF_JSON_CHARS
        {
            return Err(Error::InvalidRequest {
                message: format!("import archive {name} is not canonical or is too long"),
            });
        }
    }
    Ok(())
}

fn validate_import_source_metadata_json(source_id: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value) {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid metadata_json for source {source_id}"),
            });
        }
    }
    Ok(())
}

fn validate_import_memory_invariants(transaction: &Transaction<'_>) -> Result<()> {
    validate_import_memories(transaction)?;
    validate_import_memory_versions(transaction)?;
    validate_import_memory_tags(transaction)?;
    Ok(())
}

fn validate_import_memories(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare(
        "SELECT id, space_name, silo_name, scope, project_key, kind, entity_key,
            claim_key, status, confidence, source_episode_id, valid_from, valid_to,
            observed_at, created_at, updated_at, accessed_at, expires_at, deleted_at,
            metadata_json
         FROM memories
         ORDER BY id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(ImportMemoryInvariantRow {
            id: row.get(0)?,
            space_name: row.get(1)?,
            silo_name: row.get(2)?,
            scope: row.get(3)?,
            project_key: row.get(4)?,
            kind: row.get(5)?,
            entity_key: row.get(6)?,
            claim_key: row.get(7)?,
            status: row.get(8)?,
            confidence: row.get(9)?,
            source_episode_id: row.get(10)?,
            valid_from: row.get(11)?,
            valid_to: row.get(12)?,
            observed_at: row.get(13)?,
            created_at: row.get(14)?,
            updated_at: row.get(15)?,
            accessed_at: row.get(16)?,
            expires_at: row.get(17)?,
            deleted_at: row.get(18)?,
            metadata_json: row.get(19)?,
        })
    })?;
    for row in rows {
        validate_import_memory_row(&row?)?;
    }
    Ok(())
}

struct ImportMemoryInvariantRow {
    id: String,
    space_name: String,
    silo_name: String,
    scope: String,
    project_key: Option<String>,
    kind: String,
    entity_key: Option<String>,
    claim_key: Option<String>,
    status: String,
    confidence: f64,
    source_episode_id: Option<String>,
    valid_from: Option<String>,
    valid_to: Option<String>,
    observed_at: String,
    created_at: String,
    updated_at: String,
    accessed_at: Option<String>,
    expires_at: Option<String>,
    deleted_at: Option<String>,
    metadata_json: Option<String>,
}

fn validate_import_memory_row(row: &ImportMemoryInvariantRow) -> Result<()> {
    validate_required_import_metadata_value("memory id", &row.id)?;
    validate_required_import_metadata_value("memory space", &row.space_name)?;
    validate_required_import_metadata_value("memory silo", &row.silo_name)?;
    validate_required_import_metadata_value("memory scope", &row.scope)?;
    validate_optional_import_metadata_value("memory project", row.project_key.as_deref())?;
    validate_required_import_metadata_value("memory kind", &row.kind)?;
    validate_optional_import_metadata_value("memory entity_key", row.entity_key.as_deref())?;
    validate_optional_import_metadata_value("memory claim_key", row.claim_key.as_deref())?;
    validate_required_import_metadata_value("memory status", &row.status)?;
    validate_optional_import_metadata_value(
        "memory source_episode_id",
        row.source_episode_id.as_deref(),
    )?;
    if !is_supported_scope(&row.scope) {
        return Err(Error::InvalidRequest {
            message: format!("import archive has unsupported memory scope: {}", row.scope),
        });
    }
    if !is_supported_kind(&row.kind) {
        return Err(Error::InvalidRequest {
            message: format!("import archive has unsupported memory kind: {}", row.kind),
        });
    }
    if !is_supported_status(&row.status) || !(0.0..=1.0).contains(&row.confidence) {
        return Err(Error::InvalidRequest {
            message: format!(
                "import archive has invalid status/confidence for memory {}",
                row.id
            ),
        });
    }
    validate_import_memory_timestamps(row)?;
    validate_import_metadata_json(&row.id, row.metadata_json.as_deref())
}

fn validate_required_import_metadata_value(name: &str, value: &str) -> Result<()> {
    validate_optional_import_metadata_value(name, Some(value))
}

fn validate_optional_import_metadata_value(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.trim() != value {
            return Err(Error::InvalidRequest {
                message: format!("import archive {name} must not need trimming"),
            });
        }
        validate_optional_metadata_value(name, Some(value))?;
    }
    Ok(())
}

fn validate_import_memory_timestamps(row: &ImportMemoryInvariantRow) -> Result<()> {
    validate_required_timestamp("memory observed_at", &row.observed_at)?;
    validate_required_timestamp("memory created_at", &row.created_at)?;
    validate_required_timestamp("memory updated_at", &row.updated_at)?;
    validate_optional_timestamp("memory valid_from", row.valid_from.as_deref())?;
    validate_optional_timestamp("memory valid_to", row.valid_to.as_deref())?;
    validate_optional_timestamp("memory accessed_at", row.accessed_at.as_deref())?;
    validate_optional_timestamp("memory expires_at", row.expires_at.as_deref())?;
    validate_optional_timestamp("memory deleted_at", row.deleted_at.as_deref())
}

fn validate_required_timestamp(name: &str, value: &str) -> Result<()> {
    validate_optional_timestamp(name, Some(value))
}

fn validate_import_metadata_json(memory_id: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value) {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid metadata_json for memory {memory_id}"),
            });
        }
    }
    Ok(())
}

fn validate_import_memory_versions(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare(
        "SELECT id, content, summary, content_sha256
         FROM memory_versions
         ORDER BY memory_id, version_num, id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    for row in rows {
        let (id, content, summary, content_sha256) = row?;
        validate_required_import_metadata_value("memory version id", &id)?;
        if content.trim().is_empty() || content.chars().count() > MAX_CONTENT_CHARS {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid content length for version {id}"),
            });
        }
        if summary
            .as_deref()
            .is_some_and(|summary| summary.chars().count() > MAX_SUMMARY_CHARS)
        {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid summary length for version {id}"),
            });
        }
        if content_sha256 != sha256_hex(content.as_bytes()) {
            return Err(Error::InvalidRequest {
                message: format!("import archive has content hash mismatch for version {id}"),
            });
        }
    }
    Ok(())
}

fn validate_import_memory_tags(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement =
        transaction.prepare("SELECT memory_id, tag FROM memory_tags ORDER BY memory_id, tag")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut tags_by_memory: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for row in rows {
        let (memory_id, tag) = row?;
        tags_by_memory.entry(memory_id).or_default().push(tag);
    }
    for (memory_id, tags) in tags_by_memory {
        let normalized = normalized_tags(&tags).map_err(|_| Error::InvalidRequest {
            message: format!("import archive has invalid tags for memory {memory_id}"),
        })?;
        if normalized != tags {
            return Err(Error::InvalidRequest {
                message: format!("import archive has non-canonical tags for memory {memory_id}"),
            });
        }
    }
    Ok(())
}

fn validate_import_space_isolation(transaction: &Transaction<'_>) -> Result<()> {
    reject_import_space_mismatch(
        transaction,
        "memory source_episode_id crosses spaces",
        "SELECT m.id
         FROM memories m
         JOIN source_episodes s ON s.id = m.source_episode_id
         WHERE s.space_name != m.space_name
         ORDER BY m.id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "memory version source_episode_id crosses spaces",
        "SELECT v.id
         FROM memory_versions v
         JOIN memories m ON m.id = v.memory_id
         JOIN source_episodes s ON s.id = v.source_episode_id
         WHERE s.space_name != m.space_name
         ORDER BY v.id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "memory link crosses spaces",
        "SELECT l.src_memory_id
         FROM memory_links l
         JOIN memories src ON src.id = l.src_memory_id
         JOIN memories dst ON dst.id = l.dst_memory_id
         WHERE src.space_name != dst.space_name
         ORDER BY l.src_memory_id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "conflict memory crosses spaces",
        "SELECT c.id
         FROM conflicts c
         JOIN memories a ON a.id = c.memory_a_id
         JOIN memories b ON b.id = c.memory_b_id
         WHERE c.space_name != a.space_name OR c.space_name != b.space_name
         ORDER BY c.id
         LIMIT 1",
    )?;
    validate_import_entity_space_isolation(transaction)
}

fn validate_import_entity_space_isolation(transaction: &Transaction<'_>) -> Result<()> {
    reject_import_space_mismatch(
        transaction,
        "entity source_episode_id crosses spaces",
        "SELECT e.id
         FROM entities e
         JOIN source_episodes s ON s.id = e.source_episode_id
         WHERE s.space_name != e.space_name
         ORDER BY e.id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "entity alias source_episode_id crosses spaces",
        "SELECT a.entity_id
         FROM entity_aliases a
         JOIN entities e ON e.id = a.entity_id
         JOIN source_episodes s ON s.id = a.source_episode_id
         WHERE s.space_name != e.space_name
         ORDER BY a.entity_id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "relationship crosses spaces",
        "SELECT r.id
         FROM relationships r
         JOIN entities subject ON subject.id = r.subject_entity_id
         JOIN entities object ON object.id = r.object_entity_id
         LEFT JOIN memories m ON m.id = r.memory_id
         LEFT JOIN source_episodes s ON s.id = r.source_episode_id
         WHERE r.space_name != subject.space_name
            OR r.space_name != object.space_name
            OR (m.id IS NOT NULL AND r.space_name != m.space_name)
            OR (s.id IS NOT NULL AND r.space_name != s.space_name)
         ORDER BY r.id
         LIMIT 1",
    )
}

fn reject_import_space_mismatch(
    transaction: &Transaction<'_>,
    reason: &'static str,
    sql: &str,
) -> Result<()> {
    let invalid_id: Option<String> = transaction
        .query_row(sql, [], |row| row.get(0))
        .optional()?;
    if let Some(id) = invalid_id {
        return Err(Error::InvalidRequest {
            message: format!("import archive {reason}: {id}"),
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct FtsMemoryProjection {
    memory_id: String,
    version_id: String,
    space_name: String,
    silo_name: String,
    status: String,
    kind: String,
    content: String,
    summary: Option<String>,
    tags: String,
    source_text: Option<String>,
    project_key: Option<String>,
    entity_key: Option<String>,
    claim_key: Option<String>,
}

fn fts_memory_projection_from_row(row: &Row<'_>) -> rusqlite::Result<FtsMemoryProjection> {
    Ok(FtsMemoryProjection {
        memory_id: row.get(0)?,
        version_id: row.get(1)?,
        space_name: row.get(2)?,
        silo_name: row.get(3)?,
        status: row.get(4)?,
        kind: row.get(5)?,
        content: row.get(6)?,
        summary: row.get(7)?,
        tags: row.get(8)?,
        source_text: row.get(9)?,
        project_key: row.get(10)?,
        entity_key: row.get(11)?,
        claim_key: row.get(12)?,
    })
}

fn rebuild_fts(transaction: &Transaction<'_>) -> Result<(i64, i64)> {
    transaction.execute("DELETE FROM memory_fts", [])?;
    transaction.execute("DELETE FROM memory_fts_public", [])?;
    transaction.execute("DELETE FROM source_episode_fts", [])?;

    let memory_rows = {
        let mut statement = transaction.prepare(
            "SELECT
                m.id,
                v.id,
                m.space_name,
                m.silo_name,
                m.status,
                m.kind,
                v.content,
                v.summary,
                COALESCE((
                    SELECT group_concat(tag, ' ')
                    FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)
                ), ''),
                v.source_ref_json,
                m.project_key,
                m.entity_key,
                m.claim_key
             FROM memories m
             JOIN memory_versions v ON v.id = m.active_version_id",
        )?;
        let rows = statement.query_map([], fts_memory_projection_from_row)?;
        collect_rows(rows)?
    };

    for row in memory_rows {
        let metadata_text = memory_fts_metadata_text(
            row.project_key.as_deref(),
            row.entity_key.as_deref(),
            row.claim_key.as_deref(),
            &row.content,
            row.summary.as_deref(),
            &row.tags,
        );
        transaction.execute(
            "INSERT INTO memory_fts (
                memory_id, version_id, space_name, silo_name, status, kind, content, summary,
                tags, source_text, metadata_text
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                &row.memory_id,
                &row.version_id,
                &row.space_name,
                &row.silo_name,
                &row.status,
                &row.kind,
                &row.content,
                row.summary.as_deref(),
                &row.tags,
                row.source_text.as_deref(),
                &metadata_text,
            ],
        )?;
        transaction.execute(
            "INSERT INTO memory_fts_public (
                memory_id, version_id, space_name, silo_name, status, kind, content, summary,
                tags, metadata_text
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                &row.memory_id,
                &row.version_id,
                &row.space_name,
                &row.silo_name,
                &row.status,
                &row.kind,
                &row.content,
                row.summary.as_deref(),
                &row.tags,
                &metadata_text,
            ],
        )?;
    }
    transaction.execute(
        "INSERT INTO source_episode_fts (
            source_episode_id, space_name, source_type, source_path, source_description,
            content, metadata_text
         )
         SELECT id, space_name, source_type, source_path, source_description,
            content, metadata_json
         FROM source_episodes",
        [],
    )?;
    Ok((
        count(transaction, "SELECT COUNT(*) FROM memory_fts")?,
        count(transaction, "SELECT COUNT(*) FROM source_episode_fts")?,
    ))
}

const DREAM_TASK_PROMOTE: &str = "promote";
const DREAM_TASK_EXPIRE: &str = "expire";
const DREAM_TASK_REINDEX: &str = "reindex";
const DREAM_TASK_DEDUPE: &str = "dedupe";
const DREAM_TASK_GRAPH: &str = "graph";
const DREAM_TASK_ALL: &str = "all";
const DREAM_TASK_ORDER: &[&str] = &[
    DREAM_TASK_PROMOTE,
    DREAM_TASK_EXPIRE,
    DREAM_TASK_REINDEX,
    DREAM_TASK_DEDUPE,
    DREAM_TASK_GRAPH,
];

#[derive(Debug, Clone, PartialEq)]
struct PreparedDreamRequest {
    space: Option<String>,
    silos: Vec<String>,
    tasks: Vec<String>,
    max_memories: usize,
    dry_run: bool,
    include_pinned: bool,
    promote_threshold: usize,
    promote_score_floor: f64,
    promote_rank_cap: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DreamExpireCandidate {
    id: String,
    pinned: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DreamDuplicateGroup {
    space: String,
    content_sha256: String,
    total_count: usize,
    pinned_count: usize,
}

fn validate_dream_request(request: &DreamRequest) -> Result<()> {
    validate_optional_metadata_value("space", request.space.as_deref())?;
    for silo in &request.silos {
        validate_optional_metadata_value("silo", Some(silo))?;
    }
    if !request.silos.is_empty() && request.space.is_none() {
        return Err(Error::InvalidRequest {
            message: "dream silo filters require a space in v0.1".to_string(),
        });
    }
    if request.max_memories == 0 || request.max_memories > MAX_DREAM_MAX_MEMORIES {
        return Err(Error::InvalidRequest {
            message: format!("dream max_memories must be between 1 and {MAX_DREAM_MAX_MEMORIES}"),
        });
    }
    let _ = normalize_dream_tasks(&request.tasks)?;
    if request.promote_threshold == 0 {
        return Err(Error::InvalidRequest {
            message: "dream promote_threshold must be at least 1".to_string(),
        });
    }
    if request.promote_rank_cap == 0 {
        return Err(Error::InvalidRequest {
            message: "dream promote_rank_cap must be at least 1".to_string(),
        });
    }
    if !request.promote_score_floor.is_finite()
        || request.promote_score_floor < 0.0
        || request.promote_score_floor > 2.0
    {
        return Err(Error::InvalidRequest {
            message: "dream promote_score_floor must be between 0.0 and 2.0".to_string(),
        });
    }
    Ok(())
}

fn normalize_dream_tasks(tasks: &[String]) -> Result<Vec<String>> {
    if tasks.is_empty() {
        return Ok(DREAM_TASK_ORDER
            .iter()
            .map(|task| (*task).to_string())
            .collect());
    }
    let mut requested = BTreeSet::new();
    let mut all = false;
    for task in tasks {
        let trimmed = task.trim();
        if trimmed.is_empty() {
            return Err(Error::InvalidRequest {
                message: "dream tasks must be non-empty".to_string(),
            });
        }
        match trimmed {
            DREAM_TASK_ALL => all = true,
            DREAM_TASK_PROMOTE | DREAM_TASK_EXPIRE | DREAM_TASK_REINDEX | DREAM_TASK_DEDUPE
            | DREAM_TASK_GRAPH => {
                requested.insert(trimmed.to_string());
            }
            _ => {
                return Err(Error::InvalidRequest {
                    message: format!("unsupported dream task: {trimmed}"),
                });
            }
        }
    }
    if all {
        return Ok(DREAM_TASK_ORDER
            .iter()
            .map(|task| (*task).to_string())
            .collect());
    }
    Ok(DREAM_TASK_ORDER
        .iter()
        .filter(|task| requested.contains(**task))
        .map(|task| (*task).to_string())
        .collect())
}

fn prepare_dream_request(
    transaction: &Transaction<'_>,
    request: &DreamRequest,
) -> Result<PreparedDreamRequest> {
    let space = normalize_space_name(request.space.as_deref())?;
    if let Some(space) = &space {
        ensure_space_exists(transaction, space)?;
        for silo in &request.silos {
            ensure_silo_exists(transaction, space, silo)?;
        }
    }
    Ok(PreparedDreamRequest {
        space,
        silos: request.silos.clone(),
        tasks: normalize_dream_tasks(&request.tasks)?,
        max_memories: request.max_memories,
        dry_run: request.dry_run,
        include_pinned: request.include_pinned,
        promote_threshold: request.promote_threshold,
        promote_score_floor: request.promote_score_floor,
        promote_rank_cap: request.promote_rank_cap,
    })
}

fn dream_store_tx(transaction: &Transaction<'_>, request: &DreamRequest) -> Result<DreamReport> {
    let prepared = prepare_dream_request(transaction, request)?;
    let started_at = now_timestamp(transaction)?;
    let run_id = next_id("dream");

    if !prepared.dry_run {
        transaction.execute(
            "INSERT INTO dream_runs (id, space_name, status, started_at, budget_json)
             VALUES (?1, ?2, 'running', ?3, ?4)",
            params![
                &run_id,
                prepared.space.as_deref(),
                &started_at,
                dream_budget_json(&prepared),
            ],
        )?;
    }

    let promote = if dream_has_task(&prepared, DREAM_TASK_PROMOTE) {
        dream_promote_memories(transaction, &prepared, &run_id, &started_at)?
    } else {
        empty_dream_promote_report(false)
    };
    let expire = if dream_has_task(&prepared, DREAM_TASK_EXPIRE) {
        dream_expire_memories(transaction, &prepared, &run_id, &started_at)?
    } else {
        empty_dream_expire_report(false)
    };
    let reindex = if dream_has_task(&prepared, DREAM_TASK_REINDEX) {
        dream_reindex(transaction, prepared.dry_run)?
    } else {
        DreamReindexReport {
            attempted: false,
            memory_rows: 0,
            source_episode_rows: 0,
        }
    };
    let dedupe = if dream_has_task(&prepared, DREAM_TASK_DEDUPE) {
        dream_exact_duplicate_proposals(transaction, &prepared)?
    } else {
        DreamDedupeReport {
            attempted: false,
            proposals: Vec::new(),
            truncated: false,
        }
    };
    let graph = if dream_has_task(&prepared, DREAM_TASK_GRAPH) {
        dream_graph_diagnostics(transaction, &prepared)?
    } else {
        DreamGraphReport {
            attempted: false,
            orphan_entities: 0,
            dangling_relationships: 0,
            inactive_evidence_relationships: 0,
            relationship_proposals: Vec::new(),
            truncated: false,
            orphan_entity_ids: Vec::new(),
            dangling_relationship_ids: Vec::new(),
            inactive_evidence_relationship_ids: Vec::new(),
        }
    };

    let finished_at = now_timestamp(transaction)?;
    let report = DreamReport {
        run_id: run_id.clone(),
        space: prepared.space.clone(),
        silos: prepared.silos.clone(),
        tasks: prepared.tasks.clone(),
        status: "succeeded".to_string(),
        started_at,
        finished_at: finished_at.clone(),
        dry_run: prepared.dry_run,
        journaled: !prepared.dry_run,
        max_memories: prepared.max_memories,
        promote,
        expire,
        reindex,
        dedupe,
        graph,
    };

    if !prepared.dry_run {
        transaction.execute(
            "UPDATE dream_runs
             SET status = 'succeeded', finished_at = ?1, summary_json = ?2
             WHERE id = ?3",
            params![&finished_at, dream_summary_json(&report), &run_id],
        )?;
    }

    Ok(report)
}

fn dream_has_task(request: &PreparedDreamRequest, task: &str) -> bool {
    request.tasks.iter().any(|candidate| candidate == task)
}

fn empty_dream_expire_report(attempted: bool) -> DreamExpireReport {
    DreamExpireReport {
        attempted,
        scanned: 0,
        expired: 0,
        skipped_pinned: 0,
        truncated: false,
        memory_ids: Vec::new(),
        skipped_pinned_ids: Vec::new(),
    }
}

fn empty_dream_promote_report(attempted: bool) -> DreamPromoteReport {
    DreamPromoteReport {
        attempted,
        scanned: 0,
        promoted: 0,
        truncated: false,
        memory_ids: Vec::new(),
    }
}

fn dream_promote_memories(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
    run_id: &str,
    now: &str,
) -> Result<DreamPromoteReport> {
    // recall_events is created lazily; ensure it exists so the aggregate is valid
    // even on stores that have never logged a recall.
    transaction.execute_batch(RECALL_EVENTS_DDL)?;
    ensure_recall_events_session_column(transaction)?;

    // Conditions and bound values are kept in lock-step: every `?` placeholder
    // appended to `conditions` has its value pushed to `values` in the same order.
    let mut conditions = vec![
        "m.status = 'active'".to_string(),
        "m.silo_name = ?".to_string(),
        "(SELECT COUNT(DISTINCT r.session_id) FROM recall_events r \
          WHERE r.memory_id = m.id AND r.kind = 'retrieved' \
            AND r.session_id IS NOT NULL \
            AND r.score >= ? AND r.rank <= ?) >= ?"
            .to_string(),
    ];
    let mut values = vec![
        Value::Text(SHORT_TERM_SILO.to_string()),
        Value::Real(request.promote_score_floor),
        Value::Integer(limit_i64(request.promote_rank_cap)?),
        Value::Integer(limit_i64(request.promote_threshold)?),
    ];
    append_dream_memory_filters(&mut conditions, &mut values, request, "m");
    let sql = format!(
        "SELECT m.id
         FROM memories m
         WHERE {}
         ORDER BY m.id ASC
         LIMIT ?",
        conditions.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(
        request.max_memories.saturating_add(1),
    )?));

    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    let mut candidates = collect_rows(rows)?;
    let truncated = candidates.len() > request.max_memories;
    if truncated {
        candidates.truncate(request.max_memories);
    }

    let mut report = empty_dream_promote_report(true);
    report.truncated = truncated;
    report.scanned = candidates.len();

    for id in candidates {
        report.promoted = report.promoted.saturating_add(1);
        report.memory_ids.push(id.clone());
        if !request.dry_run {
            promote_memory_for_dream(transaction, &id, run_id, now)?;
        }
    }

    Ok(report)
}

fn promote_memory_for_dream(
    transaction: &Transaction<'_>,
    memory_id: &str,
    run_id: &str,
    now: &str,
) -> Result<()> {
    transaction.execute(
        "UPDATE memories
         SET silo_name = ?1, expires_at = NULL, updated_at = ?2
         WHERE id = ?3 AND status = ?4 AND silo_name = ?5",
        params![
            DEFAULT_DURABLE_SILO,
            now,
            memory_id,
            status::ACTIVE,
            SHORT_TERM_SILO
        ],
    )?;
    transaction.execute(
        "UPDATE memory_fts SET silo_name = ?1 WHERE memory_id = ?2",
        params![DEFAULT_DURABLE_SILO, memory_id],
    )?;
    transaction.execute(
        "UPDATE memory_fts_public SET silo_name = ?1 WHERE memory_id = ?2",
        params![DEFAULT_DURABLE_SILO, memory_id],
    )?;
    let event_id = next_id("evt");
    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, old_status, new_status, actor, reason, data_json, created_at)
         VALUES (?1, ?2, 'dream', ?3, ?4, 'memkeeper', ?5, ?6, ?7)",
        params![
            &event_id,
            memory_id,
            status::ACTIVE,
            status::ACTIVE,
            "promoted short-term to durable",
            dream_promote_event_json(run_id),
            now,
        ],
    )?;
    Ok(())
}

fn dream_expire_memories(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
    run_id: &str,
    now: &str,
) -> Result<DreamExpireReport> {
    let mut conditions = vec![
        "m.status = 'active'".to_string(),
        "m.expires_at IS NOT NULL".to_string(),
        "julianday(m.expires_at) <= julianday(?)".to_string(),
    ];
    let mut values = vec![Value::Text(now.to_string())];
    append_dream_memory_filters(&mut conditions, &mut values, request, "m");
    let sql = format!(
        "SELECT m.id, m.pinned
         FROM memories m
         WHERE {}
         ORDER BY m.expires_at ASC, m.id ASC
         LIMIT ?",
        conditions.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(
        request.max_memories.saturating_add(1),
    )?));
    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        Ok(DreamExpireCandidate {
            id: row.get(0)?,
            pinned: row.get::<_, i64>(1)? != 0,
        })
    })?;
    let mut candidates = collect_rows(rows)?;
    let truncated = candidates.len() > request.max_memories;
    if truncated {
        candidates.truncate(request.max_memories);
    }

    let mut report = empty_dream_expire_report(true);
    report.truncated = truncated;
    report.scanned = candidates.len();

    for candidate in candidates {
        if candidate.pinned && !request.include_pinned {
            report.skipped_pinned = report.skipped_pinned.saturating_add(1);
            report.skipped_pinned_ids.push(candidate.id);
            continue;
        }
        report.expired = report.expired.saturating_add(1);
        report.memory_ids.push(candidate.id.clone());
        if !request.dry_run {
            expire_memory_for_dream(transaction, &candidate.id, run_id, now)?;
        }
    }

    Ok(report)
}

fn expire_memory_for_dream(
    transaction: &Transaction<'_>,
    memory_id: &str,
    run_id: &str,
    now: &str,
) -> Result<()> {
    transaction.execute(
        "UPDATE memories SET status = ?1, updated_at = ?2 WHERE id = ?3 AND status = ?4",
        params![status::EXPIRED, now, memory_id, status::ACTIVE],
    )?;
    transaction.execute(
        "UPDATE memory_fts SET status = ?1 WHERE memory_id = ?2",
        params![status::EXPIRED, memory_id],
    )?;
    transaction.execute(
        "UPDATE memory_fts_public SET status = ?1 WHERE memory_id = ?2",
        params![status::EXPIRED, memory_id],
    )?;
    let event_id = next_id("evt");
    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, old_status, new_status, actor, reason, data_json, created_at)
         VALUES (?1, ?2, 'dream', ?3, ?4, 'memkeeper', ?5, ?6, ?7)",
        params![
            &event_id,
            memory_id,
            status::ACTIVE,
            status::EXPIRED,
            "expires_at reached",
            dream_expire_event_json(run_id),
            now,
        ],
    )?;
    Ok(())
}

fn dream_reindex(transaction: &Transaction<'_>, dry_run: bool) -> Result<DreamReindexReport> {
    let (memory_rows, source_episode_rows) = if dry_run {
        (
            count(
                transaction,
                "SELECT COUNT(*) FROM memories m JOIN memory_versions v ON v.id = m.active_version_id",
            )?,
            count(transaction, "SELECT COUNT(*) FROM source_episodes")?,
        )
    } else {
        rebuild_fts(transaction)?
    };
    Ok(DreamReindexReport {
        attempted: true,
        memory_rows,
        source_episode_rows,
    })
}

fn dream_exact_duplicate_proposals(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<DreamDedupeReport> {
    let mut conditions = vec!["m.status = 'active'".to_string()];
    let mut values = Vec::new();
    append_dream_memory_filters(&mut conditions, &mut values, request, "m");
    let sql = format!(
        "SELECT m.space_name, v.content_sha256, COUNT(*), COALESCE(SUM(m.pinned), 0)
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {}
         GROUP BY m.space_name, v.content_sha256
         HAVING COUNT(*) > 1
         ORDER BY m.space_name ASC, v.content_sha256 ASC
         LIMIT ?",
        conditions.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(
        request.max_memories.saturating_add(1),
    )?));
    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        Ok(DreamDuplicateGroup {
            space: row.get(0)?,
            content_sha256: row.get(1)?,
            total_count: usize::try_from(row.get::<_, i64>(2)?).unwrap_or(usize::MAX),
            pinned_count: usize::try_from(row.get::<_, i64>(3)?).unwrap_or(usize::MAX),
        })
    })?;
    let mut groups = collect_rows(rows)?;
    let truncated = groups.len() > request.max_memories;
    if truncated {
        groups.truncate(request.max_memories);
    }

    let proposals = groups
        .iter()
        .map(|group| dream_duplicate_proposal(transaction, request, group))
        .collect::<Result<Vec<_>>>()?;
    Ok(DreamDedupeReport {
        attempted: true,
        proposals,
        truncated,
    })
}

fn dream_graph_diagnostics(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<DreamGraphReport> {
    let mut orphan_entity_ids = dream_graph_orphan_entity_ids(transaction, request)?;
    let mut dangling_relationship_ids =
        dream_graph_dangling_relationship_ids(transaction, request)?;
    let mut inactive_evidence_relationship_ids =
        dream_graph_inactive_evidence_relationship_ids(transaction, request)?;
    let mut relationship_proposals = dream_graph_relationship_proposals(transaction, request)?;

    // The queries fetch max_memories + 1 rows to detect truncation. Truncate
    // before acting so the tombstoned rows and the reported ids are the same
    // set (the sentinel row must not be mutated without being reported).
    let truncated = orphan_entity_ids.len() > request.max_memories
        || dangling_relationship_ids.len() > request.max_memories
        || inactive_evidence_relationship_ids.len() > request.max_memories
        || relationship_proposals.len() > request.max_memories;
    orphan_entity_ids.truncate(request.max_memories);
    dangling_relationship_ids.truncate(request.max_memories);
    inactive_evidence_relationship_ids.truncate(request.max_memories);
    relationship_proposals.truncate(request.max_memories);

    // Reconcile drift: tombstone projection rows that lost their backing --
    // orphan entities (no active memory, no relationships), relationships with a
    // missing or non-active (merged/tombstoned) endpoint entity, and relationships
    // whose evidence memory is gone or inactive. relationship_proposals are
    // link-derived entity edges (deterministic projections of explicit
    // memory_links) that are materialized into the graph on a non-dry-run via
    // apply_graph_relationship_proposals. A dry run reports without mutating
    // (the MCP graph tool path).
    if !request.dry_run {
        let now = now_timestamp(transaction)?;
        apply_graph_relationship_proposals(transaction, &relationship_proposals)?;
        tombstone_graph_rows(
            transaction,
            GraphTable::Relationships,
            &dangling_relationship_ids,
            &now,
        )?;
        tombstone_graph_rows(
            transaction,
            GraphTable::Relationships,
            &inactive_evidence_relationship_ids,
            &now,
        )?;
        tombstone_graph_rows(transaction, GraphTable::Entities, &orphan_entity_ids, &now)?;
    }

    Ok(DreamGraphReport {
        attempted: true,
        orphan_entities: orphan_entity_ids.len(),
        dangling_relationships: dangling_relationship_ids.len(),
        inactive_evidence_relationships: inactive_evidence_relationship_ids.len(),
        relationship_proposals,
        truncated,
        orphan_entity_ids,
        dangling_relationship_ids,
        inactive_evidence_relationship_ids,
    })
}

/// Materialize dream-proposed relationships (deterministic projections of
/// explicit `memory_links` onto the entity graph) as active edges. Idempotent
/// via `upsert_relationship_tx`. Confidence is 1.0 because each proposal mirrors
/// a concrete recorded link, not a probabilistic inference. Runs only on a
/// non-dry-run dream `graph` task.
fn apply_graph_relationship_proposals(
    transaction: &Transaction<'_>,
    proposals: &[DreamRelationshipProposal],
) -> Result<()> {
    for proposal in proposals {
        upsert_relationship_tx(
            transaction,
            &PreparedRelationshipUpsertRequest {
                space: proposal.space.clone(),
                subject_entity_id: None,
                subject_entity_key: Some(proposal.subject_entity_key.clone()),
                relation_type: proposal.relation_type.clone(),
                object_entity_id: None,
                object_entity_key: Some(proposal.object_entity_key.clone()),
                memory_id: Some(proposal.src_memory_id.clone()),
                source_episode_id: None,
                status: status::ACTIVE.to_string(),
                confidence: 1.0,
                observed_at: None,
                valid_from: None,
                valid_to: None,
                metadata_json: None,
                include_source: false,
            },
        )?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum GraphTable {
    Relationships,
    Entities,
}

/// Tombstone graph projection rows by id (idempotent: only flips `active` rows).
/// `table` is a fixed enum, never user input, so the SQL is a static literal.
fn tombstone_graph_rows(
    transaction: &Transaction<'_>,
    table: GraphTable,
    ids: &[String],
    now: &str,
) -> Result<()> {
    if ids.is_empty() {
        return Ok(());
    }
    let sql = match table {
        GraphTable::Relationships => {
            "UPDATE relationships SET status = ?1, updated_at = ?2 WHERE id = ?3 AND status = 'active'"
        }
        GraphTable::Entities => {
            "UPDATE entities SET status = ?1, updated_at = ?2 WHERE id = ?3 AND status = 'active'"
        }
    };
    let mut statement = transaction.prepare(sql)?;
    for id in ids {
        statement.execute(params![status::TOMBSTONED, now, id])?;
    }
    Ok(())
}

fn dream_graph_orphan_entity_ids(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<Vec<String>> {
    let (space_predicate, space_values) =
        optional_space_predicate("e.space_name", request.space.as_deref());
    let sql = format!(
        "SELECT e.id
         FROM entities e
         WHERE {space_predicate}
           AND e.status = 'active'
           AND NOT EXISTS (
             SELECT 1 FROM memories m
             WHERE m.space_name = e.space_name AND m.entity_key = e.entity_key AND m.status = 'active'
           )
           AND NOT EXISTS (
             SELECT 1 FROM relationships r
             WHERE r.space_name = e.space_name
              AND (r.subject_entity_id = e.id OR r.object_entity_id = e.id)
              AND r.status = 'active'
           )
         ORDER BY e.space_name ASC, e.entity_key ASC, e.id ASC
         LIMIT {}",
        request.max_memories.saturating_add(1)
    );
    collect_string_query(transaction, &sql, &space_values)
}

fn dream_graph_dangling_relationship_ids(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<Vec<String>> {
    let (space_predicate, space_values) =
        optional_space_predicate("r.space_name", request.space.as_deref());
    let sql = format!(
        "SELECT r.id
         FROM relationships r
         LEFT JOIN entities subject
           ON subject.id = r.subject_entity_id AND subject.space_name = r.space_name
         LEFT JOIN entities object
           ON object.id = r.object_entity_id AND object.space_name = r.space_name
         WHERE {space_predicate}
          AND (
            subject.id IS NULL OR object.id IS NULL
            OR subject.status != 'active' OR object.status != 'active'
          )
          AND r.status = 'active'
         ORDER BY r.space_name ASC, r.id ASC
         LIMIT {}",
        request.max_memories.saturating_add(1)
    );
    collect_string_query(transaction, &sql, &space_values)
}

fn dream_graph_inactive_evidence_relationship_ids(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<Vec<String>> {
    let (space_predicate, space_values) =
        optional_space_predicate("r.space_name", request.space.as_deref());
    let sql = format!(
        "SELECT r.id
         FROM relationships r
         LEFT JOIN memories m
           ON m.id = r.memory_id AND m.space_name = r.space_name
         WHERE {space_predicate}
           AND r.memory_id IS NOT NULL
          AND (m.id IS NULL OR m.status != 'active')
          AND r.status = 'active'
         ORDER BY r.space_name ASC, r.id ASC
         LIMIT {}",
        request.max_memories.saturating_add(1)
    );
    collect_string_query(transaction, &sql, &space_values)
}

/// Optional space filter as a (predicate, bound values) pair; the predicate
/// uses `?1` so callers must bind these values first.
fn optional_space_predicate(column: &str, space: Option<&str>) -> (String, Vec<Value>) {
    space.map_or_else(
        || ("1=1".to_string(), Vec::new()),
        |space| {
            (
                format!("{column} = ?1"),
                vec![Value::Text(space.to_string())],
            )
        },
    )
}

fn collect_string_query(
    connection: &Connection,
    sql: &str,
    values: &[Value],
) -> Result<Vec<String>> {
    let mut statement = connection.prepare(sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    collect_rows(rows)
}

fn dream_graph_relationship_proposals(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
) -> Result<Vec<DreamRelationshipProposal>> {
    let (space_predicate, space_values) =
        optional_space_predicate("src.space_name", request.space.as_deref());
    let sql = format!(
        "SELECT src.space_name, l.src_memory_id, l.dst_memory_id, l.link_type,
                src.entity_key, l.link_type, dst.entity_key
         FROM memory_links l
         JOIN memories src ON src.id = l.src_memory_id
         JOIN memories dst ON dst.id = l.dst_memory_id
         JOIN entities subject ON subject.space_name = src.space_name
              AND subject.entity_key = src.entity_key
              AND subject.status = 'active'
         JOIN entities object ON object.space_name = dst.space_name
              AND object.entity_key = dst.entity_key
              AND object.status = 'active'
         LEFT JOIN relationships existing ON existing.space_name = src.space_name
              AND existing.subject_entity_id = subject.id
              AND existing.relation_type = l.link_type
              AND existing.object_entity_id = object.id
              AND existing.status = 'active'
         WHERE l.status = 'active'
           AND src.status = 'active'
           AND dst.status = 'active'
           AND src.space_name = dst.space_name
           AND src.entity_key IS NOT NULL
           AND TRIM(src.entity_key) <> ''
           AND dst.entity_key IS NOT NULL
           AND TRIM(dst.entity_key) <> ''
           AND src.entity_key <> dst.entity_key
           AND existing.id IS NULL
           AND {space_predicate}
         ORDER BY src.space_name ASC, l.link_type ASC, src.entity_key ASC, dst.entity_key ASC,
                  l.src_memory_id ASC, l.dst_memory_id ASC
         LIMIT {}",
        request.max_memories.saturating_add(1)
    );
    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(space_values.iter()), |row| {
        Ok(DreamRelationshipProposal {
            space: row.get(0)?,
            src_memory_id: row.get(1)?,
            dst_memory_id: row.get(2)?,
            link_type: row.get(3)?,
            subject_entity_key: row.get(4)?,
            relation_type: row.get(5)?,
            object_entity_key: row.get(6)?,
        })
    })?;
    collect_rows(rows)
}

fn dream_duplicate_proposal(
    transaction: &Transaction<'_>,
    request: &PreparedDreamRequest,
    group: &DreamDuplicateGroup,
) -> Result<DreamDuplicateProposal> {
    let mut conditions = vec![
        "m.status = 'active'".to_string(),
        "m.space_name = ?".to_string(),
        "v.content_sha256 = ?".to_string(),
    ];
    let mut values = vec![
        Value::Text(group.space.clone()),
        Value::Text(group.content_sha256.clone()),
    ];
    append_dream_silo_filters(&mut conditions, &mut values, request, "m");
    let sql = format!(
        "SELECT m.id
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {}
         ORDER BY m.pinned DESC, m.id ASC
         LIMIT ?",
        conditions.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(
        MAX_DREAM_DUPLICATE_IDS.saturating_add(2),
    )?));
    let mut statement = transaction.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        row.get::<_, String>(0)
    })?;
    let ids = collect_rows(rows)?;
    let canonical_memory_id = ids.first().cloned().unwrap_or_default();
    let duplicate_memory_ids = ids
        .iter()
        .skip(1)
        .take(MAX_DREAM_DUPLICATE_IDS)
        .cloned()
        .collect::<Vec<_>>();
    let duplicate_ids_truncated = group.total_count.saturating_sub(1) > duplicate_memory_ids.len();
    Ok(DreamDuplicateProposal {
        space: group.space.clone(),
        content_sha256: group.content_sha256.clone(),
        canonical_memory_id,
        duplicate_memory_ids,
        total_count: group.total_count,
        pinned_count: group.pinned_count,
        duplicate_ids_truncated,
    })
}

fn append_dream_memory_filters(
    conditions: &mut Vec<String>,
    values: &mut Vec<Value>,
    request: &PreparedDreamRequest,
    alias: &str,
) {
    if let Some(space) = &request.space {
        conditions.push(format!("{alias}.space_name = ?"));
        values.push(Value::Text(space.clone()));
    }
    append_dream_silo_filters(conditions, values, request, alias);
}

fn append_dream_silo_filters(
    conditions: &mut Vec<String>,
    values: &mut Vec<Value>,
    request: &PreparedDreamRequest,
    alias: &str,
) {
    if request.silos.is_empty() {
        return;
    }
    let placeholders = std::iter::repeat_n("?", request.silos.len())
        .collect::<Vec<_>>()
        .join(", ");
    conditions.push(format!("{alias}.silo_name IN ({placeholders})"));
    for silo in &request.silos {
        values.push(Value::Text(silo.clone()));
    }
}

fn dream_budget_json(request: &PreparedDreamRequest) -> String {
    format!(
        "{{\"tasks\":{},\"space\":{},\"silos\":{},\"max_memories\":{},\"dry_run\":{},\"include_pinned\":{}}}",
        string_array_json(&request.tasks),
        optional_json_string_for_store(request.space.as_deref()),
        string_array_json(&request.silos),
        request.max_memories,
        request.dry_run,
        request.include_pinned
    )
}

fn dream_summary_json(report: &DreamReport) -> String {
    format!(
        "{{\"status\":{},\"promoted\":{},\"expired\":{},\"skipped_pinned\":{},\"reindex_memory_rows\":{},\"reindex_source_episode_rows\":{},\"duplicate_proposals\":{},\"graph_orphan_entities\":{},\"graph_dangling_relationships\":{},\"graph_inactive_evidence_relationships\":{},\"graph_relationship_proposals\":{},\"dry_run\":{}}}",
        json_string_for_store(&report.status),
        report.promote.promoted,
        report.expire.expired,
        report.expire.skipped_pinned,
        report.reindex.memory_rows,
        report.reindex.source_episode_rows,
        report.dedupe.proposals.len(),
        report.graph.orphan_entities,
        report.graph.dangling_relationships,
        report.graph.inactive_evidence_relationships,
        report.graph.relationship_proposals.len(),
        report.dry_run
    )
}

fn dream_expire_event_json(run_id: &str) -> String {
    format!(
        "{{\"run_id\":{},\"task\":\"expire\"}}",
        json_string_for_store(run_id)
    )
}

fn dream_promote_event_json(run_id: &str) -> String {
    format!(
        "{{\"run_id\":{},\"task\":\"promote\"}}",
        json_string_for_store(run_id)
    )
}

fn optional_json_string_for_store(value: Option<&str>) -> String {
    value.map_or_else(|| "null".to_string(), json_string_for_store)
}

struct ParsedImportFile {
    schema_version: i32,
    tables: Vec<ExportTableReport>,
    row_count: u64,
    bytes: u64,
    sha256: String,
}

fn parse_import_file(
    transaction: &Transaction<'_>,
    request: &ImportRequest,
) -> Result<ParsedImportFile> {
    let file = open_import_input_file(&request.input_path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut line = Vec::new();
    let mut line_number = 0_usize;
    let mut state = ImportState::new();

    loop {
        let read = read_limited_jsonl_record(&mut reader, &mut line, line_number + 1)?;
        if read == 0 {
            break;
        }
        hasher.update(&line);
        bytes = bytes.saturating_add(read as u64);
        line_number = line_number.saturating_add(1);
        process_import_line(transaction, &mut state, &line, line_number)?;
    }

    if !state.footer_seen {
        return Err(Error::InvalidRequest {
            message: "import file missing footer".to_string(),
        });
    }

    Ok(ParsedImportFile {
        schema_version: state.schema_version,
        tables: state.table_reports(),
        row_count: state.row_count,
        bytes,
        sha256: hasher.finish_hex(),
    })
}

fn read_limited_jsonl_record<R: BufRead>(
    reader: &mut R,
    buffer: &mut Vec<u8>,
    line_number: usize,
) -> Result<usize> {
    buffer.clear();
    let mut total = 0_usize;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(total);
        }
        let end = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if total.saturating_add(end) > MAX_IMPORT_LINE_BYTES {
            return import_invalid(line_number, "import JSONL record exceeds maximum size");
        }
        buffer.extend_from_slice(&available[..end]);
        reader.consume(end);
        total = total.saturating_add(end);
        if buffer.ends_with(b"\n") {
            return Ok(total);
        }
    }
}

fn process_import_line(
    transaction: &Transaction<'_>,
    state: &mut ImportState,
    raw_line: &[u8],
    line_number: usize,
) -> Result<()> {
    let line = trim_jsonl_newline(raw_line);
    if line.iter().all(u8::is_ascii_whitespace) {
        return import_invalid(line_number, "import JSONL records must not be blank");
    }
    let text = std::str::from_utf8(line).map_err(|_| Error::InvalidRequest {
        message: format!("import JSONL line {line_number} is not valid UTF-8"),
    })?;
    let value = parse_import_json_line(text, line_number)?;
    let object = import_object(&value, line_number, "record")?;
    let record_type = import_string_field(object, "type", line_number)?;
    match record_type {
        "header" => state.accept_header(object, line_number),
        "row" => state.accept_row(transaction, object, line_number),
        "footer" => state.accept_footer(object, line_number),
        _ => import_invalid(line_number, "unknown import record type"),
    }
}

fn trim_jsonl_newline(mut line: &[u8]) -> &[u8] {
    if line.ends_with(b"\n") {
        line = &line[..line.len() - 1];
    }
    if line.ends_with(b"\r") {
        line = &line[..line.len() - 1];
    }
    line
}

struct ImportState {
    header_seen: bool,
    footer_seen: bool,
    current_table_index: usize,
    schema_version: i32,
    row_count: u64,
    table_counts: Vec<u64>,
}

impl ImportState {
    fn new() -> Self {
        Self {
            header_seen: false,
            footer_seen: false,
            current_table_index: 0,
            schema_version: SCHEMA_VERSION,
            row_count: 0,
            table_counts: vec![0; EXPORT_TABLES.len()],
        }
    }

    fn accept_header(
        &mut self,
        object: &BTreeMap<String, ImportJsonValue>,
        line_number: usize,
    ) -> Result<()> {
        if self.header_seen {
            return import_invalid(line_number, "duplicate import header");
        }
        if self.footer_seen || self.row_count > 0 {
            return import_invalid(line_number, "import header must be the first record");
        }
        import_reject_unknown_fields(
            object,
            &[
                "type",
                "format",
                "protocol_version",
                "schema_version",
                "tables",
                "rebuildable_omitted",
            ],
            line_number,
        )?;
        let format = import_string_field(object, "format", line_number)?;
        if format != "memkeeper.export.v0.1" {
            return import_invalid(line_number, "import format must be memkeeper.export.v0.1");
        }
        let protocol = import_string_field(object, "protocol_version", line_number)?;
        if protocol != "memkeeper.v0.1" {
            return import_invalid(
                line_number,
                "import protocol_version must be memkeeper.v0.1",
            );
        }
        let schema_version = import_i64_field(object, "schema_version", line_number)?;
        if schema_version != i64::from(SCHEMA_VERSION) {
            return Err(Error::SchemaMismatch {
                expected: SCHEMA_VERSION,
                actual: i32::try_from(schema_version).unwrap_or_default(),
            });
        }
        validate_import_table_list(object, line_number)?;
        validate_rebuildable_omitted(object, line_number)?;
        self.schema_version = SCHEMA_VERSION;
        self.header_seen = true;
        Ok(())
    }

    fn accept_row(
        &mut self,
        transaction: &Transaction<'_>,
        object: &BTreeMap<String, ImportJsonValue>,
        line_number: usize,
    ) -> Result<()> {
        self.require_open_rows(line_number)?;
        import_reject_unknown_fields(object, &["type", "table", "data"], line_number)?;
        let table_name = import_string_field(object, "table", line_number)?;
        let table_index = export_table_index(table_name).ok_or_else(|| Error::InvalidRequest {
            message: format!("import JSONL line {line_number}: unknown export table {table_name}"),
        })?;
        if table_index < self.current_table_index {
            return import_invalid(line_number, "import rows are not in export table order");
        }
        self.current_table_index = table_index;
        let data = import_object_field(object, "data", line_number)?;
        insert_import_row(transaction, &EXPORT_TABLES[table_index], data, line_number)?;
        self.table_counts[table_index] = self.table_counts[table_index].saturating_add(1);
        self.row_count = self.row_count.saturating_add(1);
        Ok(())
    }

    fn accept_footer(
        &mut self,
        object: &BTreeMap<String, ImportJsonValue>,
        line_number: usize,
    ) -> Result<()> {
        self.require_open_rows(line_number)?;
        import_reject_unknown_fields(object, &["type", "row_count", "table_counts"], line_number)?;
        let row_count = import_i64_field(object, "row_count", line_number)?;
        if row_count < 0 || u64::try_from(row_count).ok() != Some(self.row_count) {
            return import_invalid(line_number, "import footer row_count mismatch");
        }
        let table_counts = import_object_field(object, "table_counts", line_number)?;
        self.validate_footer_table_counts(table_counts, line_number)?;
        self.footer_seen = true;
        Ok(())
    }

    fn require_open_rows(&self, line_number: usize) -> Result<()> {
        if !self.header_seen {
            return import_invalid(line_number, "import row appeared before header");
        }
        if self.footer_seen {
            return import_invalid(line_number, "import record appeared after footer");
        }
        Ok(())
    }

    fn validate_footer_table_counts(
        &self,
        object: &BTreeMap<String, ImportJsonValue>,
        line_number: usize,
    ) -> Result<()> {
        if object.len() != EXPORT_TABLES.len() {
            return import_invalid(line_number, "import footer table_counts mismatch");
        }
        for (index, table) in EXPORT_TABLES.iter().enumerate() {
            let count = import_i64_field(object, table.name, line_number)?;
            if count < 0 || u64::try_from(count).ok() != Some(self.table_counts[index]) {
                return import_invalid(line_number, "import footer table count mismatch");
            }
        }
        Ok(())
    }

    fn table_reports(&self) -> Vec<ExportTableReport> {
        EXPORT_TABLES
            .iter()
            .zip(&self.table_counts)
            .map(|(table, rows)| ExportTableReport {
                name: table.name.to_string(),
                rows: *rows,
            })
            .collect()
    }
}

fn validate_import_table_list(
    object: &BTreeMap<String, ImportJsonValue>,
    line_number: usize,
) -> Result<()> {
    let tables = import_array_field(object, "tables", line_number)?;
    if tables.len() != EXPORT_TABLES.len() {
        return import_invalid(line_number, "import header table list mismatch");
    }
    for (value, table) in tables.iter().zip(EXPORT_TABLES) {
        if import_json_string(value) != Some(table.name) {
            return import_invalid(line_number, "import header table list mismatch");
        }
    }
    Ok(())
}

fn validate_rebuildable_omitted(
    object: &BTreeMap<String, ImportJsonValue>,
    line_number: usize,
) -> Result<()> {
    let omitted = import_array_field(object, "rebuildable_omitted", line_number)?;
    let expected = ["memory_fts", "memory_fts_public", "source_episode_fts"];
    if omitted.len() != expected.len() {
        return import_invalid(line_number, "import header rebuildable_omitted mismatch");
    }
    for (value, expected) in omitted.iter().zip(expected) {
        if import_json_string(value) != Some(expected) {
            return import_invalid(line_number, "import header rebuildable_omitted mismatch");
        }
    }
    Ok(())
}

fn insert_import_row(
    transaction: &Transaction<'_>,
    table: &ExportTableSpec,
    data: &BTreeMap<String, ImportJsonValue>,
    line_number: usize,
) -> Result<()> {
    validate_import_row_columns(table, data, line_number)?;
    let values = table
        .columns
        .iter()
        .map(|column| {
            import_sql_value(
                table,
                column,
                data.get(*column).expect("validated column"),
                line_number,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let placeholders = (1..=table.columns.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        table.name,
        table.columns.join(", "),
        placeholders
    );
    transaction.execute(&sql, params_from_iter(values.iter()))?;
    Ok(())
}

fn validate_import_row_columns(
    table: &ExportTableSpec,
    data: &BTreeMap<String, ImportJsonValue>,
    line_number: usize,
) -> Result<()> {
    if data.len() != table.columns.len() {
        return import_invalid(line_number, "import row column set mismatch");
    }
    for column in table.columns {
        if !data.contains_key(*column) {
            return import_invalid(line_number, "import row missing expected column");
        }
    }
    Ok(())
}

fn import_sql_value(
    table: &ExportTableSpec,
    column: &str,
    value: &ImportJsonValue,
    line_number: usize,
) -> Result<Value> {
    match (import_column_type(table.name, column), value) {
        (_, ImportJsonValue::Null) => Ok(Value::Null),
        (ImportColumnType::Integer, ImportJsonValue::Integer(value)) => Ok(Value::Integer(*value)),
        (ImportColumnType::Real, ImportJsonValue::Integer(value)) => {
            let real = value
                .to_string()
                .parse::<f64>()
                .map_err(|_| Error::InvalidRequest {
                    message: format!(
                        "import JSON line {line_number}: integer for real column is out of range"
                    ),
                })?;
            Ok(Value::Real(real))
        }
        (ImportColumnType::Real, ImportJsonValue::Real(value)) => Ok(Value::Real(*value)),
        (ImportColumnType::Text, ImportJsonValue::String(value)) => Ok(Value::Text(value.clone())),
        (ImportColumnType::Blob, ImportJsonValue::Object(object)) => {
            import_blob_value(object, line_number)
        }
        _ => Err(Error::InvalidRequest {
            message: format!(
                "import JSON line {line_number}: column {}.{} has invalid SQLite storage class",
                table.name, column
            ),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImportColumnType {
    Integer,
    Real,
    Text,
    Blob,
}

fn import_column_type(table: &str, column: &str) -> ImportColumnType {
    match (table, column) {
        ("schema_migrations", "version")
        | (
            _,
            "chunk_index" | "chunk_count" | "version_num" | "pinned" | "priority" | "attempts"
            | "max_attempts" | "dimensions",
        ) => ImportColumnType::Integer,
        (_, "confidence") => ImportColumnType::Real,
        ("embeddings", "vector_blob") => ImportColumnType::Blob,
        _ => ImportColumnType::Text,
    }
}

fn import_blob_value(
    object: &BTreeMap<String, ImportJsonValue>,
    line_number: usize,
) -> Result<Value> {
    if object.len() != 1 {
        return import_invalid(
            line_number,
            "import BLOB wrapper must have exactly one field",
        );
    }
    let hex = import_string_field(object, "$memkeeper_blob_hex", line_number)?;
    Ok(Value::Blob(decode_hex_blob(hex, line_number)?))
}

fn decode_hex_blob(value: &str, line_number: usize) -> Result<Vec<u8>> {
    if !value.len().is_multiple_of(2) {
        return import_invalid(line_number, "import BLOB hex length must be even");
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|chunk| {
            let high = hex_nibble(chunk[0], line_number)?;
            let low = hex_nibble(chunk[1], line_number)?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn hex_nibble(byte: u8, line_number: usize) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => import_invalid(line_number, "import BLOB hex contains non-hex character"),
    }
}

fn export_table_index(name: &str) -> Option<usize> {
    EXPORT_TABLES.iter().position(|table| table.name == name)
}

fn validate_import_request(request: &ImportRequest) -> Result<()> {
    if request.format != "jsonl" {
        return Err(Error::InvalidRequest {
            message: "import format must be jsonl in v0.1".to_string(),
        });
    }
    if request.conflict_policy != "fail_if_exists" {
        return Err(Error::InvalidRequest {
            message: "import conflict_policy must be fail_if_exists in v0.1".to_string(),
        });
    }
    Ok(())
}

fn open_import_input_file(path: &Path) -> Result<File> {
    validate_import_input_path_shape(path)?;
    let file = open_import_input_file_inner(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "input path must be a regular file",
        });
    }
    Ok(file)
}

fn validate_import_input_path_shape(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "input path must not be empty",
        });
    }
    let display_path = path.to_string_lossy();
    if path == Path::new(":memory:") || display_path.starts_with("file:") {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "input paths must use plain filesystem paths",
        });
    }
    Ok(())
}

#[cfg(unix)]
fn open_import_input_file_inner(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(unix_import_input_open_flags());
    options.open(path).map_err(Into::into)
}

#[cfg(not(unix))]
fn open_import_input_file_inner(path: &Path) -> Result<File> {
    File::open(path).map_err(Into::into)
}

#[cfg(target_os = "linux")]
const fn unix_import_input_open_flags() -> i32 {
    0o400_000 | 0o4_000
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
const fn unix_import_input_open_flags() -> i32 {
    0x0100 | 0x0004
}

#[cfg(all(
    unix,
    not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    ))
))]
const fn unix_import_input_open_flags() -> i32 {
    0
}

fn cleanup_import_target(path: &Path) {
    let _ = fs::remove_file(path);
    cleanup_temp_output(path);
}

#[derive(Debug, Clone, PartialEq)]
enum ImportJsonValue {
    Null,
    Bool(bool),
    Integer(i64),
    Real(f64),
    String(String),
    Array(Vec<ImportJsonValue>),
    Object(BTreeMap<String, ImportJsonValue>),
}

impl<'de> serde::Deserialize<'de> for ImportJsonValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(ImportJsonValueVisitor)
    }
}

struct ImportJsonValueVisitor;

impl<'de> serde::de::Visitor<'de> for ImportJsonValueVisitor {
    type Value = ImportJsonValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a bounded JSON value")
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::Null)
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::Integer(value))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        i64::try_from(value)
            .map(ImportJsonValue::Integer)
            .map_err(|_| E::custom("integer out of range"))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if value.is_finite() {
            Ok(ImportJsonValue::Real(value))
        } else {
            Err(E::custom("number must be finite"))
        }
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::String(value))
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<ImportJsonValue>()? {
            if values.len() >= MAX_IMPORT_JSON_ARRAY_ITEMS {
                return Err(serde::de::Error::custom("JSON array has too many items"));
            }
            values.push(value);
        }
        Ok(ImportJsonValue::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut object = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if object.len() >= MAX_IMPORT_JSON_OBJECT_FIELDS {
                return Err(serde::de::Error::custom("JSON object has too many fields"));
            }
            let value = map.next_value::<ImportJsonValue>()?;
            if object.insert(key, value).is_some() {
                return Err(serde::de::Error::custom("duplicate field in JSON object"));
            }
        }
        Ok(ImportJsonValue::Object(object))
    }
}

fn parse_import_json_line(text: &str, line_number: usize) -> Result<ImportJsonValue> {
    let value =
        serde_json::from_str::<ImportJsonValue>(text).map_err(|error| Error::InvalidRequest {
            message: format!("import JSON line {line_number}: {error}"),
        })?;
    validate_import_json_depth(&value, 0, line_number)?;
    Ok(value)
}

fn validate_import_json_depth(
    value: &ImportJsonValue,
    depth: usize,
    line_number: usize,
) -> Result<()> {
    if depth > MAX_IMPORT_JSON_DEPTH {
        return import_invalid(line_number, "import JSON record exceeds maximum depth");
    }
    match value {
        ImportJsonValue::Array(values) => {
            for value in values {
                validate_import_json_depth(value, depth + 1, line_number)?;
            }
        }
        ImportJsonValue::Object(object) => {
            for value in object.values() {
                validate_import_json_depth(value, depth + 1, line_number)?;
            }
        }
        ImportJsonValue::Null
        | ImportJsonValue::Bool(_)
        | ImportJsonValue::Integer(_)
        | ImportJsonValue::Real(_)
        | ImportJsonValue::String(_) => {}
    }
    Ok(())
}

fn import_json_object_is_valid(input: &str) -> bool {
    let Ok(value) = serde_json::from_str::<ImportJsonValue>(input) else {
        return false;
    };
    matches!(value, ImportJsonValue::Object(_)) && validate_import_json_depth(&value, 0, 0).is_ok()
}

fn import_object<'a>(
    value: &'a ImportJsonValue,
    line_number: usize,
    label: &str,
) -> Result<&'a BTreeMap<String, ImportJsonValue>> {
    match value {
        ImportJsonValue::Object(object) => Ok(object),
        _ => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: {label} must be an object"),
        }),
    }
}

fn import_object_field<'a>(
    object: &'a BTreeMap<String, ImportJsonValue>,
    key: &str,
    line_number: usize,
) -> Result<&'a BTreeMap<String, ImportJsonValue>> {
    let value = object.get(key).ok_or_else(|| Error::InvalidRequest {
        message: format!("import JSON line {line_number}: missing field {key}"),
    })?;
    import_object(value, line_number, key)
}

fn import_array_field<'a>(
    object: &'a BTreeMap<String, ImportJsonValue>,
    key: &str,
    line_number: usize,
) -> Result<&'a [ImportJsonValue]> {
    match object.get(key) {
        Some(ImportJsonValue::Array(values)) => Ok(values),
        Some(_) => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: field {key} must be an array"),
        }),
        None => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: missing field {key}"),
        }),
    }
}

fn import_string_field<'a>(
    object: &'a BTreeMap<String, ImportJsonValue>,
    key: &str,
    line_number: usize,
) -> Result<&'a str> {
    match object.get(key) {
        Some(ImportJsonValue::String(value)) => Ok(value),
        Some(_) => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: field {key} must be a string"),
        }),
        None => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: missing field {key}"),
        }),
    }
}

fn import_i64_field(
    object: &BTreeMap<String, ImportJsonValue>,
    key: &str,
    line_number: usize,
) -> Result<i64> {
    match object.get(key) {
        Some(ImportJsonValue::Integer(value)) => Ok(*value),
        Some(_) => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: field {key} must be an integer"),
        }),
        None => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: missing field {key}"),
        }),
    }
}

fn import_json_string(value: &ImportJsonValue) -> Option<&str> {
    match value {
        ImportJsonValue::String(value) => Some(value),
        _ => None,
    }
}

fn import_reject_unknown_fields(
    object: &BTreeMap<String, ImportJsonValue>,
    allowed: &[&str],
    line_number: usize,
) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::InvalidRequest {
                message: format!("import JSON line {line_number}: unknown field {key}"),
            });
        }
    }
    Ok(())
}

fn import_invalid<T>(line_number: usize, message: &str) -> Result<T> {
    Err(Error::InvalidRequest {
        message: format!("import JSON line {line_number}: {message}"),
    })
}

fn validate_export_request(request: &ExportRequest) -> Result<()> {
    if request.format != "jsonl" {
        return Err(Error::InvalidRequest {
            message: "export format must be jsonl in v0.1".to_string(),
        });
    }
    validate_output_path(&request.output_path)
}

fn validate_backup_request(request: &BackupRequest) -> Result<()> {
    if request.format != "sqlite" {
        return Err(Error::InvalidRequest {
            message: "backup format must be sqlite in v0.1".to_string(),
        });
    }
    validate_output_path(&request.output_path)?;
    reject_existing_output_sidecars(&request.output_path)
}

struct HashingFileWriter {
    inner: BufWriter<File>,
    hasher: Sha256,
    bytes: u64,
}

impl HashingFileWriter {
    fn new(file: File) -> Self {
        Self {
            inner: BufWriter::new(file),
            hasher: Sha256::new(),
            bytes: 0,
        }
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.inner.write_all(bytes)?;
        self.hasher.update(bytes);
        self.bytes = self.bytes.saturating_add(bytes.len() as u64);
        Ok(())
    }

    fn finish(mut self) -> Result<(u64, String)> {
        self.inner.flush()?;
        self.inner.get_ref().sync_all()?;
        Ok((self.bytes, self.hasher.finish_hex()))
    }
}

fn write_json_string_to(writer: &mut HashingFileWriter, value: &str) -> Result<()> {
    let encoded = serde_json::to_string(value).map_err(|error| Error::InvalidRequest {
        message: format!("could not encode JSON string: {error}"),
    })?;
    writer.write_bytes(encoded.as_bytes())
}

fn write_string_array_to<'a>(
    writer: &mut HashingFileWriter,
    values: impl Iterator<Item = &'a str>,
) -> Result<()> {
    writer.write_bytes(b"[")?;
    for (index, value) in values.enumerate() {
        if index > 0 {
            writer.write_bytes(b",")?;
        }
        write_json_string_to(writer, value)?;
    }
    writer.write_bytes(b"]")
}

fn write_sql_value_json(writer: &mut HashingFileWriter, value: ValueRef<'_>) -> Result<()> {
    match value {
        ValueRef::Null => writer.write_bytes(b"null"),
        ValueRef::Integer(value) => writer.write_bytes(value.to_string().as_bytes()),
        ValueRef::Real(value) => writer.write_bytes(finite_json_number(value).as_bytes()),
        ValueRef::Text(value) => {
            let text = std::str::from_utf8(value).map_err(|_| Error::InvalidRequest {
                message: "database text value is not valid UTF-8".to_string(),
            })?;
            write_json_string_to(writer, text)
        }
        ValueRef::Blob(value) => {
            writer.write_bytes(b"{\"$memkeeper_blob_hex\":\"")?;
            write_hex_bytes(writer, value)?;
            writer.write_bytes(b"\"}")
        }
    }
}

fn finite_json_number(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".to_string()
    }
}

fn write_hex_bytes(writer: &mut HashingFileWriter, bytes: &[u8]) -> Result<()> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        writer.write_bytes(&[HEX[usize::from(byte >> 4)], HEX[usize::from(byte & 0x0f)]])?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct PreparedMemoryListRequest {
    filters: SearchFilters,
    limit: usize,
    offset: usize,
    snippet_chars: usize,
    include_content: bool,
    include_source: bool,
    order: String,
}

#[derive(Debug, Clone)]
struct MemoryListCandidate {
    memory_id: String,
    version_id: String,
    space: String,
    silo: String,
    scope: String,
    project_key: Option<String>,
    kind: String,
    status: String,
    entity_key: Option<String>,
    claim_key: Option<String>,
    confidence: f64,
    pinned: bool,
    observed_at: String,
    created_at: String,
    updated_at: String,
    snippet_text: String,
    content: Option<String>,
    summary: Option<String>,
    tags: Vec<String>,
    source_ref_json: Option<String>,
}

impl MemoryListCandidate {
    fn into_item(self, rank: usize, request: &PreparedMemoryListRequest) -> MemoryListItem {
        MemoryListItem {
            rank,
            memory_id: self.memory_id,
            version_id: self.version_id,
            space: self.space,
            silo: self.silo,
            scope: self.scope,
            project_key: self.project_key,
            kind: self.kind,
            status: self.status,
            summary: self.summary,
            snippet: bounded_char_slice(&self.snippet_text, 0, request.snippet_chars),
            content: self.content.filter(|_| request.include_content),
            tags: self.tags,
            entity_key: self.entity_key,
            claim_key: self.claim_key,
            confidence: self.confidence,
            pinned: self.pinned,
            observed_at: self.observed_at,
            created_at: self.created_at,
            updated_at: self.updated_at,
            source_ref_json: self.source_ref_json.filter(|_| request.include_source),
        }
    }
}

#[derive(Debug, Clone)]
struct PreparedSearchRequest {
    fts_query: String,
    fallback_fts_query: Option<String>,
    prefix_fts_query: Option<String>,
    terms: Vec<String>,
    /// Normalized 1..=3-word contiguous shingles of the raw query, used to match
    /// reserved `alias::<normalized>` tags for the alias-exact-match boost. Built
    /// from the raw query (not `terms`, which is sorted/deduped and loses order).
    query_alias_shingles: std::collections::HashSet<String>,
    filters: SearchFilters,
    limit: usize,
    /// Inflated SQL LIMIT for the initial candidate fetch. Always >= limit.
    /// Gives Rust re-scoring room to surface candidates SQL would otherwise
    /// undervalue (e.g. recent volatile memories boosted 4x by the silo-aware
    /// recency curve but ranked equal to durable by SQL's uniform recency score).
    candidate_pool_limit: usize,
    offset: usize,
    snippet_chars: usize,
    include_content: bool,
    include_source: bool,
    semantic_fallback: String,
    lexical_fallback: String,
    embedding: Option<Vec<f32>>,
    // Read only by the semantic-gated late-interaction candidate path.
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    query_token_embedding: Option<Vec<Vec<f32>>>,
    #[cfg_attr(not(feature = "semantic"), allow(dead_code))]
    token_model_id: Option<String>,
}

#[derive(Debug, Clone)]
struct PreparedEntityUpsertRequest {
    space: String,
    entity_key: String,
    entity_type: String,
    canonical_name: String,
    aliases: Vec<PreparedEntityAlias>,
    status: String,
    confidence: f64,
    source_episode_id: Option<String>,
    metadata_json: Option<String>,
    include_source: bool,
}

#[derive(Debug, Clone)]
struct PreparedEntityAlias {
    alias: String,
    normalized_alias: String,
}

#[derive(Debug, Clone)]
struct PreparedRelationshipUpsertRequest {
    space: String,
    subject_entity_id: Option<String>,
    subject_entity_key: Option<String>,
    relation_type: String,
    object_entity_id: Option<String>,
    object_entity_key: Option<String>,
    memory_id: Option<String>,
    source_episode_id: Option<String>,
    status: String,
    confidence: f64,
    observed_at: Option<String>,
    valid_from: Option<String>,
    valid_to: Option<String>,
    metadata_json: Option<String>,
    include_source: bool,
}

#[derive(Debug, Clone)]
struct PreparedEntitySearchRequest {
    space: String,
    query: Option<String>,
    normalized_query: Option<String>,
    entity_key: Option<String>,
    entity_types: Vec<String>,
    statuses: Vec<String>,
    limit: usize,
    offset: usize,
    include_source: bool,
}

#[derive(Debug, Clone)]
struct PreparedGraphNeighborsRequest {
    space: String,
    entity_id: Option<String>,
    entity_key: Option<String>,
    depth: usize,
    relation_types: Vec<String>,
    statuses: Vec<String>,
    max_edges: usize,
    include_tombstoned: bool,
    include_source: bool,
}

#[derive(Debug, Clone)]
struct GraphRelationshipRow {
    subject_depth: usize,
    object_depth: usize,
    relationship: RelationshipRecord,
}

#[derive(Debug, Clone)]
struct RememberCandidateRow {
    memory_id: String,
    space: String,
    silo: String,
    kind: String,
    status: String,
    entity_key: Option<String>,
    claim_key: Option<String>,
    content: String,
    summary: Option<String>,
    content_sha256: String,
}

#[derive(Debug, Clone)]
struct RememberCandidateAccumulator {
    row: RememberCandidateRow,
    relationship: String,
    score: f64,
    matched_on: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct RememberCandidateDetection<'a> {
    space: &'a str,
    silo: &'a str,
    kind: &'a str,
    content_sha256: &'a str,
    entity_key: Option<&'a str>,
    claim_key: Option<&'a str>,
    request_terms: BTreeSet<String>,
    excluded_ids: &'a BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct SameClaimCandidate {
    memory_id: String,
    kind: String,
    observed_at: String,
    content: String,
    pinned: bool,
}

#[derive(Debug, Clone)]
struct SearchCandidate {
    memory_id: String,
    version_id: String,
    space: String,
    silo: String,
    scope: String,
    project_key: Option<String>,
    kind: String,
    status: String,
    summary: Option<String>,
    content: Option<String>,
    snippet_text: String,
    tags: Vec<String>,
    entity_key: Option<String>,
    claim_key: Option<String>,
    observed_at: String,
    recency_jd: Option<f64>,
    source_ref_json: Option<String>,
    /// Coarse source `type` (e.g. `mcp`, `manual`, `synthesis`) extracted for
    /// ranking only. Always loaded, unlike `source_ref_json`, which is gated
    /// behind `include_source` for output privacy.
    source_type: Option<String>,
    metadata_json: Option<String>,
    confidence: f64,
    pinned: bool,
    bm25: f64,
    lexical_tier: u8,
    score: f64,
    scores: ScoreBreakdown,
}

impl SearchCandidate {
    /// Consumes the candidate; result assembly moves the owned strings
    /// instead of cloning them (the candidate vec is always discarded after
    /// this conversion).
    fn into_result(self, rank: usize, request: &PreparedSearchRequest) -> SearchResult {
        let snippet = self.snippet(request);
        SearchResult {
            rank,
            memory_id: self.memory_id,
            version_id: self.version_id,
            score: self.score,
            scores: self.scores,
            space: self.space,
            silo: self.silo,
            scope: self.scope,
            project_key: self.project_key,
            kind: self.kind,
            status: self.status,
            summary: self.summary,
            snippet,
            content: self.content.filter(|_| request.include_content),
            tags: self.tags,
            entity_key: self.entity_key,
            claim_key: self.claim_key,
            observed_at: self.observed_at,
            source_ref_json: request
                .include_source
                .then_some(self.source_ref_json)
                .flatten(),
            metadata_json: self.metadata_json,
        }
    }

    fn snippet(&self, request: &PreparedSearchRequest) -> String {
        if request.snippet_chars == 0 {
            return String::new();
        }
        if let Some(content) = &self.content {
            make_snippet(content, &request.terms, request.snippet_chars)
        } else {
            bounded_char_slice(&self.snippet_text, 0, request.snippet_chars)
        }
    }
}

fn upsert_entity_tx(
    transaction: &Transaction<'_>,
    request: &PreparedEntityUpsertRequest,
) -> Result<EntityUpsertReport> {
    let now = now_timestamp(transaction)?;
    ensure_space_exists(transaction, &request.space)?;
    if let Some(source_episode_id) = request.source_episode_id.as_deref() {
        ensure_source_episode_exists(transaction, &request.space, source_episode_id)?;
    }

    let existing_id = transaction
        .query_row(
            "SELECT id FROM entities WHERE space_name = ?1 AND entity_key = ?2",
            params![&request.space, &request.entity_key],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let created = existing_id.is_none();
    let entity_id = existing_id.unwrap_or_else(|| next_id("ent"));

    transaction.execute(
        "INSERT INTO entities (
            id, space_name, entity_key, entity_type, canonical_name, status, confidence,
            source_episode_id, metadata_json, created_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?10)
         ON CONFLICT(space_name, entity_key) DO UPDATE SET
            entity_type = excluded.entity_type,
            canonical_name = excluded.canonical_name,
            status = excluded.status,
            confidence = excluded.confidence,
            source_episode_id = COALESCE(excluded.source_episode_id, entities.source_episode_id),
            metadata_json = COALESCE(excluded.metadata_json, entities.metadata_json),
            updated_at = excluded.updated_at",
        params![
            &entity_id,
            &request.space,
            &request.entity_key,
            &request.entity_type,
            &request.canonical_name,
            &request.status,
            request.confidence,
            request.source_episode_id.as_deref(),
            request.metadata_json.as_deref(),
            &now,
        ],
    )?;

    upsert_entity_aliases(
        transaction,
        &entity_id,
        &request.aliases,
        request.source_episode_id.as_deref(),
        &now,
    )?;
    let entity = load_entity(transaction, &entity_id, request.include_source)?;

    Ok(EntityUpsertReport {
        strategy: "deterministic_entity_upsert_v0".to_string(),
        created,
        entity,
    })
}

fn upsert_entity_aliases(
    transaction: &Transaction<'_>,
    entity_id: &str,
    aliases: &[PreparedEntityAlias],
    source_episode_id: Option<&str>,
    now: &str,
) -> Result<()> {
    for alias in aliases {
        transaction.execute(
            "INSERT INTO entity_aliases (entity_id, alias, normalized_alias, source_episode_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(entity_id, normalized_alias) DO UPDATE SET
                alias = excluded.alias,
                source_episode_id = COALESCE(excluded.source_episode_id, entity_aliases.source_episode_id)",
            params![
                entity_id,
                &alias.alias,
                &alias.normalized_alias,
                source_episode_id,
                now,
            ],
        )?;
    }
    Ok(())
}

fn upsert_relationship_tx(
    transaction: &Transaction<'_>,
    request: &PreparedRelationshipUpsertRequest,
) -> Result<RelationshipUpsertReport> {
    let now = now_timestamp(transaction)?;
    ensure_space_exists(transaction, &request.space)?;
    let subject_entity_id = resolve_relationship_endpoint(
        transaction,
        &request.space,
        request.subject_entity_id.as_deref(),
        request.subject_entity_key.as_deref(),
        "subject_entity",
    )?;
    let object_entity_id = resolve_relationship_endpoint(
        transaction,
        &request.space,
        request.object_entity_id.as_deref(),
        request.object_entity_key.as_deref(),
        "object_entity",
    )?;
    if let Some(memory_id) = request.memory_id.as_deref() {
        ensure_memory_in_space(transaction, memory_id, &request.space)?;
    }
    if let Some(source_episode_id) = request.source_episode_id.as_deref() {
        ensure_source_episode_exists(transaction, &request.space, source_episode_id)?;
    }

    let existing_id = find_existing_relationship(
        transaction,
        &request.space,
        &subject_entity_id,
        &request.relation_type,
        &object_entity_id,
        request.memory_id.as_deref(),
        request.source_episode_id.as_deref(),
    )?;
    let created = existing_id.is_none();
    let relationship_id = existing_id.unwrap_or_else(|| next_id("rel"));

    transaction.execute(
        "INSERT INTO relationships (
            id, space_name, subject_entity_id, relation_type, object_entity_id,
            memory_id, source_episode_id, status, confidence, observed_at, valid_from, valid_to,
            metadata_json, created_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?14)
         ON CONFLICT(id) DO UPDATE SET
            relation_type = excluded.relation_type,
            memory_id = COALESCE(excluded.memory_id, relationships.memory_id),
            source_episode_id = COALESCE(excluded.source_episode_id, relationships.source_episode_id),
            status = excluded.status,
            confidence = excluded.confidence,
            observed_at = COALESCE(excluded.observed_at, relationships.observed_at),
            valid_from = COALESCE(excluded.valid_from, relationships.valid_from),
            valid_to = COALESCE(excluded.valid_to, relationships.valid_to),
            metadata_json = COALESCE(excluded.metadata_json, relationships.metadata_json),
            updated_at = excluded.updated_at",
        params![
            &relationship_id,
            &request.space,
            &subject_entity_id,
            &request.relation_type,
            &object_entity_id,
            request.memory_id.as_deref(),
            request.source_episode_id.as_deref(),
            &request.status,
            request.confidence,
            request.observed_at.as_deref(),
            request.valid_from.as_deref(),
            request.valid_to.as_deref(),
            request.metadata_json.as_deref(),
            &now,
        ],
    )?;
    let relationship = load_relationship(transaction, &relationship_id, request.include_source)?;
    Ok(RelationshipUpsertReport {
        strategy: "deterministic_relationship_upsert_v0".to_string(),
        created,
        relationship,
    })
}

fn merge_entity_tx(
    transaction: &Transaction<'_>,
    request: &EntityMergeRequest,
) -> Result<EntityMergeReport> {
    let from_entity_id =
        normalize_optional_graph_value("from_entity_id", request.from_entity_id.as_deref())?;
    let from_entity_key =
        normalize_optional_graph_value("from_entity_key", request.from_entity_key.as_deref())?;
    validate_exactly_one_endpoint(
        "from_entity_id",
        from_entity_id.as_ref(),
        "from_entity_key",
        from_entity_key.as_ref(),
    )?;
    let into_entity_id =
        normalize_optional_graph_value("into_entity_id", request.into_entity_id.as_deref())?;
    let into_entity_key =
        normalize_optional_graph_value("into_entity_key", request.into_entity_key.as_deref())?;
    validate_exactly_one_endpoint(
        "into_entity_id",
        into_entity_id.as_ref(),
        "into_entity_key",
        into_entity_key.as_ref(),
    )?;
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    ensure_space_exists(transaction, &space)?;
    let from_id = resolve_relationship_endpoint(
        transaction,
        &space,
        from_entity_id.as_deref(),
        from_entity_key.as_deref(),
        "from_entity",
    )?;
    let into_id = resolve_relationship_endpoint(
        transaction,
        &space,
        into_entity_id.as_deref(),
        into_entity_key.as_deref(),
        "into_entity",
    )?;
    if from_id == into_id {
        return Err(Error::InvalidRequest {
            message: "from and into resolve to the same entity".to_string(),
        });
    }
    let now = now_timestamp(transaction)?;
    let from_entity = load_entity(transaction, &from_id, false)?;
    let into_before = load_entity(transaction, &into_id, false)?;
    if from_entity.status != "active" || into_before.status != "active" {
        return Err(Error::Conflict {
            message: "both from and into entities must be active to merge".to_string(),
        });
    }

    let (repointed, tombstoned_duplicate, tombstoned_self_loop) =
        relink_active_relationships(transaction, &space, &from_id, &into_id, &now)?;

    // Carry the merged-away identifiers over as aliases of the target.
    let alias_count_before = entity_alias_count(transaction, &into_id)?;
    let mut alias_source = vec![
        from_entity.entity_key.clone(),
        from_entity.canonical_name.clone(),
    ];
    alias_source.extend(from_entity.aliases.iter().cloned());
    let prepared_aliases = merge_alias_candidates(&alias_source);
    upsert_entity_aliases(transaction, &into_id, &prepared_aliases, None, &now)?;
    let aliases_added =
        entity_alias_count(transaction, &into_id)?.saturating_sub(alias_count_before);

    // Tombstone the merged-away source entity (audit-preserving).
    transaction.execute(
        "UPDATE entities SET status = 'tombstoned', updated_at = ?1 WHERE id = ?2",
        params![&now, &from_id],
    )?;

    let into = load_entity(transaction, &into_id, request.include_source)?;
    Ok(EntityMergeReport {
        strategy: "deterministic_entity_merge_v0".to_string(),
        dry_run: request.dry_run,
        from_entity_key: from_entity.entity_key,
        into_entity_key: into_before.entity_key,
        relationships_repointed: repointed,
        relationships_tombstoned_duplicate: tombstoned_duplicate,
        relationships_tombstoned_self_loop: tombstoned_self_loop,
        aliases_added,
        from_tombstoned: true,
        into,
    })
}

/// Relink the source entity's active relationships onto the target, collapsing
/// duplicates and self-loops. Returns `(repointed, tombstoned_duplicate,
/// tombstoned_self_loop)`.
fn relink_active_relationships(
    transaction: &Transaction<'_>,
    space: &str,
    from_id: &str,
    into_id: &str,
    now: &str,
) -> Result<(usize, usize, usize)> {
    let edges: Vec<(String, String, String, String)> = {
        let mut statement = transaction.prepare(
            "SELECT id, subject_entity_id, relation_type, object_entity_id
             FROM relationships
             WHERE space_name = ?1 AND status = 'active'
               AND (subject_entity_id = ?2 OR object_entity_id = ?2)
             ORDER BY id ASC",
        )?;
        let rows = statement.query_map(params![space, from_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        collect_rows(rows)?
    };

    let mut repointed = 0usize;
    let mut tombstoned_duplicate = 0usize;
    let mut tombstoned_self_loop = 0usize;
    for (rel_id, subject, relation, object) in edges {
        let new_subject = if subject == from_id {
            into_id
        } else {
            subject.as_str()
        };
        let new_object = if object == from_id {
            into_id
        } else {
            object.as_str()
        };
        if new_subject == new_object {
            tombstone_relationship(transaction, &rel_id, now)?;
            tombstoned_self_loop += 1;
            continue;
        }
        if active_relationship_exists(
            transaction,
            space,
            new_subject,
            &relation,
            new_object,
            &rel_id,
        )? {
            tombstone_relationship(transaction, &rel_id, now)?;
            tombstoned_duplicate += 1;
            continue;
        }
        transaction.execute(
            "UPDATE relationships SET subject_entity_id = ?1, object_entity_id = ?2, updated_at = ?3 WHERE id = ?4",
            params![new_subject, new_object, now, &rel_id],
        )?;
        repointed += 1;
    }
    Ok((repointed, tombstoned_duplicate, tombstoned_self_loop))
}

/// Build deduplicated, non-empty alias candidates for a merge target.
fn merge_alias_candidates(values: &[String]) -> Vec<PreparedEntityAlias> {
    let mut seen = BTreeSet::new();
    let mut prepared = Vec::new();
    for value in values {
        let alias = value.trim();
        if alias.is_empty() {
            continue;
        }
        let normalized = normalized_alias(alias);
        if normalized.is_empty() || !seen.insert(normalized.clone()) {
            continue;
        }
        prepared.push(PreparedEntityAlias {
            alias: alias.to_string(),
            normalized_alias: normalized,
        });
    }
    prepared
}

fn tombstone_relationship(
    transaction: &Transaction<'_>,
    relationship_id: &str,
    now: &str,
) -> Result<()> {
    transaction.execute(
        "UPDATE relationships SET status = 'tombstoned', updated_at = ?1 WHERE id = ?2",
        params![now, relationship_id],
    )?;
    Ok(())
}

fn active_relationship_exists(
    transaction: &Transaction<'_>,
    space: &str,
    subject_entity_id: &str,
    relation_type: &str,
    object_entity_id: &str,
    exclude_id: &str,
) -> Result<bool> {
    let found = transaction
        .query_row(
            "SELECT 1 FROM relationships
             WHERE space_name = ?1 AND status = 'active'
               AND subject_entity_id = ?2 AND relation_type = ?3 AND object_entity_id = ?4
               AND id <> ?5
             LIMIT 1",
            params![
                space,
                subject_entity_id,
                relation_type,
                object_entity_id,
                exclude_id
            ],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    Ok(found.is_some())
}

fn entity_alias_count(transaction: &Transaction<'_>, entity_id: &str) -> Result<usize> {
    let count: i64 = transaction.query_row(
        "SELECT COUNT(*) FROM entity_aliases WHERE entity_id = ?1",
        params![entity_id],
        |row| row.get(0),
    )?;
    Ok(usize::try_from(count).unwrap_or(0))
}

fn search_entities_on_connection(
    connection: &Connection,
    request: &EntitySearchRequest,
) -> Result<EntitySearchReport> {
    let prepared = prepare_entity_search_request(request)?;
    ensure_space_exists(connection, &prepared.space)?;
    let mut results = entity_search_results(connection, &prepared)?;
    let truncated = results.len() > prepared.limit;
    if truncated {
        results.truncate(prepared.limit);
    }
    let total_estimate = if results.is_empty() && !truncated {
        0
    } else {
        prepared
            .offset
            .saturating_add(results.len())
            .saturating_add(usize::from(truncated))
    };
    Ok(EntitySearchReport {
        strategy: "deterministic_entity_search_v0".to_string(),
        space: prepared.space,
        total_estimate,
        truncated,
        results,
    })
}

fn graph_neighbors_on_connection(
    connection: &Connection,
    request: &GraphNeighborsRequest,
) -> Result<GraphNeighborsReport> {
    let prepared = prepare_graph_neighbors_request(request)?;
    ensure_space_exists(connection, &prepared.space)?;
    let seed = resolve_graph_seed(connection, &prepared)?;
    let mut rows = graph_relationship_rows(connection, &seed.id, &prepared)?;
    let truncated = rows.len() > prepared.max_edges;
    if truncated {
        rows.truncate(prepared.max_edges);
    }

    let mut entity_depths = BTreeMap::from([(seed.id.clone(), 0_usize)]);
    for row in &rows {
        entity_depths
            .entry(row.relationship.subject_entity_id.clone())
            .and_modify(|depth| *depth = (*depth).min(row.subject_depth))
            .or_insert(row.subject_depth);
        entity_depths
            .entry(row.relationship.object_entity_id.clone())
            .and_modify(|depth| *depth = (*depth).min(row.object_depth))
            .or_insert(row.object_depth);
    }
    let mut entities = entity_depths
        .into_iter()
        .map(|(entity_id, depth)| {
            load_entity(connection, &entity_id, prepared.include_source)
                .map(|entity| GraphEntityRecord { depth, entity })
        })
        .collect::<Result<Vec<_>>>()?;
    entities.sort_by(|left, right| {
        left.depth
            .cmp(&right.depth)
            .then_with(|| left.entity.entity_key.cmp(&right.entity.entity_key))
            .then_with(|| left.entity.id.cmp(&right.entity.id))
    });

    Ok(GraphNeighborsReport {
        strategy: "deterministic_graph_neighbors_v0".to_string(),
        seed,
        depth: prepared.depth,
        max_edges: prepared.max_edges,
        truncated,
        entities,
        relationships: rows
            .into_iter()
            .map(|row| GraphRelationshipRecord {
                subject_depth: row.subject_depth,
                object_depth: row.object_depth,
                relationship: row.relationship,
            })
            .collect(),
    })
}

fn graph_context_on_connection(
    connection: &Connection,
    request: &GraphContextRequest,
) -> Result<GraphContextReport> {
    validate_graph_context_request(request)?;
    let graph_request = GraphNeighborsRequest {
        space: request.space.clone(),
        entity_id: request.entity_id.clone(),
        entity_key: request.entity_key.clone(),
        depth: request.depth,
        relation_types: request.relation_types.clone(),
        statuses: request.statuses.clone(),
        max_edges: request.max_edges,
        include_tombstoned: request.include_tombstoned,
        include_source: request.include_source,
    };
    let graph = graph_neighbors_on_connection(connection, &graph_request)?;
    let evidence_memory_ids = graph_context_evidence_memory_ids(&graph);
    let entity_memory_ids = graph_context_entity_memory_ids(connection, &graph, request)?;
    let memory_ids = graph_context_memory_ids(&evidence_memory_ids, &entity_memory_ids);
    let results = graph_context_memory_results(connection, &memory_ids, request)?;
    let title = format!("graph context: {}", graph.seed.entity_key);
    let pack_request = PackRequest {
        title,
        queries: vec![graph.seed.entity_key.clone()],
        filters: SearchFilters::default(),
        max_memories: request.max_memories,
        max_chars: request.max_chars,
        format: "markdown".to_string(),
        min_score: 0.0,
        rerank_candidates: 0,
        query_embeddings: None,
        query_token_embeddings: None,
        token_model_id: None,
    };
    let selected_len = results.len().min(request.max_memories);
    let graph_last_synth: Option<String> = connection
        .query_row(
            "SELECT started_at FROM dream_runs WHERE status = 'succeeded' \
             ORDER BY started_at DESC LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    let (content, pack_memory_ids, pack_scores, text_truncated) = format_pack_markdown(
        &pack_request,
        &results[..selected_len],
        graph_last_synth.as_deref(),
    );
    Ok(GraphContextReport {
        strategy: "deterministic_graph_context_v0".to_string(),
        graph,
        pack: PackReport {
            title: pack_request.title,
            format: pack_request.format,
            content,
            memory_ids: pack_memory_ids,
            scores: pack_scores,
            truncated: text_truncated || results.len() > request.max_memories,
            top_score: None, // graph-context packs are not scored-retrieval packs
        },
        evidence_memory_ids,
        entity_memory_ids,
    })
}

fn validate_batch_search_request(request: &BatchSearchRequest) -> Result<()> {
    if request.queries.is_empty() || request.queries.len() > MAX_BATCH_QUERIES {
        return Err(Error::InvalidRequest {
            message: format!("queries must contain between 1 and {MAX_BATCH_QUERIES} entries"),
        });
    }
    if request.limit == 0 || request.limit > MAX_BATCH_QUERY_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("batch limit must be between 1 and {MAX_BATCH_QUERY_LIMIT}"),
        });
    }
    if request.offset > MAX_SEARCH_OFFSET {
        return Err(Error::InvalidRequest {
            message: format!("offset must be at most {MAX_SEARCH_OFFSET}"),
        });
    }
    if request.snippet_chars > MAX_SNIPPET_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("snippet_chars must be at most {MAX_SNIPPET_CHARS}"),
        });
    }
    if request.semantic_fallback != "disabled" {
        return Err(Error::InvalidRequest {
            message: "semantic_fallback must be disabled in v0.1".to_string(),
        });
    }
    for query in &request.queries {
        if query
            .name
            .as_deref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err(Error::InvalidRequest {
                message: "batch query name must not be empty".to_string(),
            });
        }
        if query
            .name
            .as_deref()
            .is_some_and(|name| name.chars().count() > MAX_TAG_CHARS)
        {
            return Err(Error::InvalidRequest {
                message: format!("batch query name must be at most {MAX_TAG_CHARS} characters"),
            });
        }
        if query
            .limit
            .is_some_and(|limit| limit == 0 || limit > MAX_BATCH_QUERY_LIMIT)
        {
            return Err(Error::InvalidRequest {
                message: format!("batch query limit must be between 1 and {MAX_BATCH_QUERY_LIMIT}"),
            });
        }
    }
    Ok(())
}

fn validate_graph_context_request(request: &GraphContextRequest) -> Result<()> {
    let graph_request = GraphNeighborsRequest {
        space: request.space.clone(),
        entity_id: request.entity_id.clone(),
        entity_key: request.entity_key.clone(),
        depth: request.depth,
        relation_types: request.relation_types.clone(),
        statuses: request.statuses.clone(),
        max_edges: request.max_edges,
        include_tombstoned: request.include_tombstoned,
        include_source: request.include_source,
    };
    let _ = prepare_graph_neighbors_request(&graph_request)?;
    if request.max_memories == 0 || request.max_memories > MAX_PACK_MEMORIES {
        return Err(Error::InvalidRequest {
            message: format!("max_memories must be between 1 and {MAX_PACK_MEMORIES}"),
        });
    }
    if request.max_chars == 0 || request.max_chars > MAX_PACK_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("max_chars must be between 1 and {MAX_PACK_CHARS}"),
        });
    }
    Ok(())
}

fn validate_pack_request(request: &PackRequest) -> Result<()> {
    if request.title.trim().is_empty() || request.title.chars().count() > MAX_PACK_TITLE_CHARS {
        return Err(Error::InvalidRequest {
            message: format!(
                "title must be non-empty and at most {MAX_PACK_TITLE_CHARS} characters"
            ),
        });
    }
    if request.queries.is_empty() || request.queries.len() > MAX_BATCH_QUERIES {
        return Err(Error::InvalidRequest {
            message: format!("queries must contain between 1 and {MAX_BATCH_QUERIES} entries"),
        });
    }
    if request.max_memories == 0 || request.max_memories > MAX_PACK_MEMORIES {
        return Err(Error::InvalidRequest {
            message: format!("max_memories must be between 1 and {MAX_PACK_MEMORIES}"),
        });
    }
    if request.max_chars == 0 || request.max_chars > MAX_PACK_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("max_chars must be between 1 and {MAX_PACK_CHARS}"),
        });
    }
    if !request.min_score.is_finite() || request.min_score < 0.0 {
        return Err(Error::InvalidRequest {
            message: "min_score must be a finite value >= 0.0".to_string(),
        });
    }
    if request.rerank_candidates > MAX_PACK_MEMORIES {
        return Err(Error::InvalidRequest {
            message: format!("rerank_candidates must be between 0 and {MAX_PACK_MEMORIES}"),
        });
    }
    if request.format != "markdown" {
        return Err(Error::InvalidRequest {
            message: "pack format must be markdown in v0.1".to_string(),
        });
    }
    Ok(())
}

fn prepare_entity_upsert_request(
    request: &EntityUpsertRequest,
) -> Result<PreparedEntityUpsertRequest> {
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let entity_key = normalize_required_graph_value("entity_key", &request.entity_key)?;
    let entity_type =
        normalize_optional_graph_value("entity_type", request.entity_type.as_deref())?
            .unwrap_or_else(|| "Entity".to_string());
    let canonical_name = normalize_required_graph_value("canonical_name", &request.canonical_name)?;
    let status = normalize_optional_graph_value("status", request.status.as_deref())?
        .unwrap_or_else(|| "active".to_string());
    if !is_supported_entity_status(&status) {
        return Err(Error::InvalidRequest {
            message: format!("unsupported entity status: {status}"),
        });
    }
    if !(0.0..=1.0).contains(&request.confidence) {
        return Err(Error::InvalidRequest {
            message: "confidence must be between 0.0 and 1.0".to_string(),
        });
    }
    let source_episode_id =
        normalize_optional_graph_value("source_episode_id", request.source_episode_id.as_deref())?;
    let metadata_json = match request.metadata_json.as_deref() {
        Some(value) => {
            if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value)
            {
                return Err(Error::InvalidRequest {
                    message: "metadata_json must be a valid JSON object".to_string(),
                });
            }
            Some(value.to_string())
        }
        None => None,
    };
    let aliases = normalize_entity_aliases(&request.aliases)?;
    Ok(PreparedEntityUpsertRequest {
        space,
        entity_key,
        entity_type,
        canonical_name,
        aliases,
        status,
        confidence: request.confidence,
        source_episode_id,
        metadata_json,
        include_source: request.include_source,
    })
}

fn normalize_required_graph_value(name: &str, value: &str) -> Result<String> {
    normalize_optional_graph_value(name, Some(value))?.ok_or_else(|| Error::InvalidRequest {
        message: format!(
            "{name} must be non-empty and at most {MAX_METADATA_VALUE_CHARS} characters"
        ),
    })
}

fn normalize_entity_aliases(values: &[String]) -> Result<Vec<PreparedEntityAlias>> {
    if values.len() > MAX_ENTITY_ALIASES {
        return Err(Error::InvalidRequest {
            message: format!("aliases must contain at most {MAX_ENTITY_ALIASES} values"),
        });
    }
    let mut seen = BTreeSet::new();
    let mut aliases = Vec::with_capacity(values.len());
    for value in values {
        let alias = normalize_required_graph_value("alias", value)?;
        let normalized_alias = normalized_alias(&alias);
        if normalized_alias.is_empty() {
            return Err(Error::InvalidRequest {
                message: "alias must contain searchable text".to_string(),
            });
        }
        if !seen.insert(normalized_alias.clone()) {
            return Err(Error::InvalidRequest {
                message: format!("aliases contain duplicate normalized value: {normalized_alias}"),
            });
        }
        aliases.push(PreparedEntityAlias {
            alias,
            normalized_alias,
        });
    }
    aliases.sort_by(|left, right| {
        left.normalized_alias
            .cmp(&right.normalized_alias)
            .then_with(|| left.alias.cmp(&right.alias))
    });
    Ok(aliases)
}

fn prepare_relationship_upsert_request(
    request: &RelationshipUpsertRequest,
) -> Result<PreparedRelationshipUpsertRequest> {
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let subject_entity_id =
        normalize_optional_graph_value("subject_entity_id", request.subject_entity_id.as_deref())?;
    let subject_entity_key = normalize_optional_graph_value(
        "subject_entity_key",
        request.subject_entity_key.as_deref(),
    )?;
    validate_exactly_one_endpoint(
        "subject_entity_id",
        subject_entity_id.as_ref(),
        "subject_entity_key",
        subject_entity_key.as_ref(),
    )?;
    let object_entity_id =
        normalize_optional_graph_value("object_entity_id", request.object_entity_id.as_deref())?;
    let object_entity_key =
        normalize_optional_graph_value("object_entity_key", request.object_entity_key.as_deref())?;
    validate_exactly_one_endpoint(
        "object_entity_id",
        object_entity_id.as_ref(),
        "object_entity_key",
        object_entity_key.as_ref(),
    )?;
    let relation_type = normalize_required_graph_value("relation_type", &request.relation_type)?;
    let memory_id = normalize_optional_graph_value("memory_id", request.memory_id.as_deref())?;
    let source_episode_id =
        normalize_optional_graph_value("source_episode_id", request.source_episode_id.as_deref())?;
    let status = normalize_optional_graph_value("status", request.status.as_deref())?
        .unwrap_or_else(|| status::ACTIVE.to_string());
    if !is_supported_relationship_status(&status) {
        return Err(Error::InvalidRequest {
            message: format!("unsupported relationship status: {status}"),
        });
    }
    if !(0.0..=1.0).contains(&request.confidence) {
        return Err(Error::InvalidRequest {
            message: "confidence must be between 0.0 and 1.0".to_string(),
        });
    }
    validate_optional_timestamp("observed_at", request.observed_at.as_deref())?;
    validate_optional_timestamp("valid_from", request.valid_from.as_deref())?;
    validate_optional_timestamp("valid_to", request.valid_to.as_deref())?;
    let observed_at = request.observed_at.clone();
    let valid_from = request.valid_from.clone();
    let valid_to = request.valid_to.clone();
    let metadata_json = normalize_optional_json_object(request.metadata_json.as_deref())?;
    Ok(PreparedRelationshipUpsertRequest {
        space,
        subject_entity_id,
        subject_entity_key,
        relation_type,
        object_entity_id,
        object_entity_key,
        memory_id,
        source_episode_id,
        status,
        confidence: request.confidence,
        observed_at,
        valid_from,
        valid_to,
        metadata_json,
        include_source: request.include_source,
    })
}

fn validate_exactly_one_endpoint(
    left_name: &str,
    left: Option<&String>,
    right_name: &str,
    right: Option<&String>,
) -> Result<()> {
    if left.is_some() == right.is_some() {
        return Err(Error::InvalidRequest {
            message: format!("exactly one of {left_name} or {right_name} is required"),
        });
    }
    Ok(())
}

fn normalize_optional_json_object(value: Option<&str>) -> Result<Option<String>> {
    match value {
        Some(value) => {
            if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value)
            {
                return Err(Error::InvalidRequest {
                    message: "metadata_json must be a valid JSON object".to_string(),
                });
            }
            Ok(Some(value.to_string()))
        }
        None => Ok(None),
    }
}

fn prepare_entity_search_request(
    request: &EntitySearchRequest,
) -> Result<PreparedEntitySearchRequest> {
    if request.limit == 0 || request.limit > MAX_ENTITY_SEARCH_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("limit must be between 1 and {MAX_ENTITY_SEARCH_LIMIT}"),
        });
    }
    if request.offset > MAX_SEARCH_OFFSET {
        return Err(Error::InvalidRequest {
            message: format!("offset must be at most {MAX_SEARCH_OFFSET}"),
        });
    }
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let query = normalize_optional_query("query", request.query.as_deref())?;
    let entity_key = normalize_optional_graph_value("entity_key", request.entity_key.as_deref())?;
    let entity_types = normalize_filter_values(&request.entity_types)?;
    let mut statuses = normalize_filter_values(&request.statuses)?;
    if statuses.is_empty() {
        statuses.push("active".to_string());
    }
    for status_value in &statuses {
        if !is_supported_entity_status(status_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported entity status: {status_value}"),
            });
        }
    }
    Ok(PreparedEntitySearchRequest {
        space,
        normalized_query: query.as_deref().map(normalized_alias),
        query,
        entity_key,
        entity_types,
        statuses,
        limit: request.limit,
        offset: request.offset,
        include_source: request.include_source,
    })
}

fn prepare_graph_neighbors_request(
    request: &GraphNeighborsRequest,
) -> Result<PreparedGraphNeighborsRequest> {
    if request.depth == 0 || request.depth > MAX_GRAPH_NEIGHBOR_DEPTH {
        return Err(Error::InvalidRequest {
            message: format!("depth must be between 1 and {MAX_GRAPH_NEIGHBOR_DEPTH}"),
        });
    }
    if request.max_edges == 0 || request.max_edges > MAX_GRAPH_NEIGHBOR_EDGES {
        return Err(Error::InvalidRequest {
            message: format!("max_edges must be between 1 and {MAX_GRAPH_NEIGHBOR_EDGES}"),
        });
    }
    if request.entity_id.is_some() == request.entity_key.is_some() {
        return Err(Error::InvalidRequest {
            message: "exactly one of entity_id or entity_key is required".to_string(),
        });
    }
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let entity_id = normalize_optional_graph_value("entity_id", request.entity_id.as_deref())?;
    let entity_key = normalize_optional_graph_value("entity_key", request.entity_key.as_deref())?;
    let relation_types = normalize_filter_values(&request.relation_types)?;
    let mut statuses = normalize_filter_values(&request.statuses)?;
    if statuses.is_empty() {
        statuses.push("active".to_string());
    }
    for status_value in &statuses {
        if !is_supported_relationship_status(status_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported relationship status: {status_value}"),
            });
        }
    }
    if !request.include_tombstoned && statuses.iter().any(|value| value == "tombstoned") {
        return Err(Error::InvalidRequest {
            message: "include_tombstoned=true is required for tombstoned relationships".to_string(),
        });
    }
    Ok(PreparedGraphNeighborsRequest {
        space,
        entity_id,
        entity_key,
        depth: request.depth,
        relation_types,
        statuses,
        max_edges: request.max_edges,
        include_tombstoned: request.include_tombstoned,
        include_source: request.include_source,
    })
}

fn normalize_optional_query(name: &str, value: Option<&str>) -> Result<Option<String>> {
    value
        .map(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed.chars().count() > MAX_SEARCH_QUERY_CHARS {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "{name} must be non-empty and at most {MAX_SEARCH_QUERY_CHARS} characters"
                    ),
                });
            }
            Ok(trimmed.to_string())
        })
        .transpose()
}

fn normalize_optional_graph_value(name: &str, value: Option<&str>) -> Result<Option<String>> {
    value
        .map(|value| {
            let trimmed = value.trim();
            if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
                return Err(Error::InvalidRequest {
                    message: format!(
                        "{name} must be non-empty and at most {MAX_METADATA_VALUE_CHARS} characters"
                    ),
                });
            }
            Ok(trimmed.to_string())
        })
        .transpose()
}

fn prepare_memory_list_request(request: &MemoryListRequest) -> Result<PreparedMemoryListRequest> {
    if request.limit == 0 || request.limit > MAX_MEMORY_LIST_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("limit must be between 1 and {MAX_MEMORY_LIST_LIMIT}"),
        });
    }
    if request.offset > MAX_SEARCH_OFFSET {
        return Err(Error::InvalidRequest {
            message: format!("offset must be at most {MAX_SEARCH_OFFSET}"),
        });
    }
    if request.snippet_chars > MAX_SNIPPET_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("snippet_chars must be at most {MAX_SNIPPET_CHARS}"),
        });
    }
    if !matches!(
        request.order.as_str(),
        "updated_desc" | "observed_desc" | "created_desc"
    ) {
        return Err(Error::InvalidRequest {
            message: "order must be updated_desc, observed_desc, or created_desc".to_string(),
        });
    }
    let mut filters = request.filters.clone();
    if filters.spaces.is_empty() {
        filters.spaces.push(DEFAULT_SPACE.to_string());
    }
    if filters.statuses.is_empty() {
        filters.statuses.push(status::ACTIVE.to_string());
    }
    filters = normalize_search_filters(filters)?;
    validate_search_filters(&filters)?;
    Ok(PreparedMemoryListRequest {
        filters,
        limit: request.limit,
        offset: request.offset,
        snippet_chars: request.snippet_chars,
        include_content: request.include_content,
        include_source: request.include_source,
        order: request.order.clone(),
    })
}

/// Reserved tag prefix marking an alias/canonical surface form for a memory.
/// A memory tagged `alias::k8s` is boosted when a query contains the token `k8s`.
pub const ALIAS_TAG_PREFIX: &str = "alias::";
/// Additive boost applied once when any query shingle matches a candidate's
/// `alias::` tag. Sized in the `fts_score` band (max 1.0) so an exact alias hit
/// lifts a topically-weak-but-correct match above semantic noise near the
/// abstention floor, without overriding a strong topical match outright.
const ALIAS_MATCH_BOOST: f64 = 0.5;
/// Longest multi-word alias we shingle for (e.g. "point of view", "ci pipeline").
const MAX_ALIAS_SHINGLE_WORDS: usize = 3;

/// Build the set of normalized 1..=`MAX_ALIAS_SHINGLE_WORDS`-word contiguous
/// shingles from the raw query, preserving word order. Normalization matches
/// `normalized_alias` (lowercase + single-space) so shingles compare directly
/// against the suffix of an `alias::<normalized>` tag.
fn query_alias_shingles(query: &str) -> std::collections::HashSet<String> {
    let words: Vec<String> = query
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|word| !word.is_empty())
        .map(str::to_ascii_lowercase)
        .collect();
    let mut shingles = std::collections::HashSet::new();
    for start in 0..words.len() {
        for span in 1..=MAX_ALIAS_SHINGLE_WORDS {
            if start + span <= words.len() {
                shingles.insert(words[start..start + span].join(" "));
            }
        }
    }
    shingles
}

fn prepare_search_request(request: &SearchRequest) -> Result<PreparedSearchRequest> {
    if request.limit == 0 || request.limit > MAX_SEARCH_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("limit must be between 1 and {MAX_SEARCH_LIMIT}"),
        });
    }
    if request.offset > MAX_SEARCH_OFFSET {
        return Err(Error::InvalidRequest {
            message: format!("offset must be at most {MAX_SEARCH_OFFSET}"),
        });
    }
    if request.snippet_chars > MAX_SNIPPET_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("snippet_chars must be at most {MAX_SNIPPET_CHARS}"),
        });
    }
    if !matches!(request.semantic_fallback.as_str(), "disabled" | "fallback") {
        return Err(Error::InvalidRequest {
            message: "semantic_fallback must be disabled or fallback".to_string(),
        });
    }
    if !matches!(
        request.lexical_fallback.as_str(),
        "disabled" | "conservative"
    ) {
        return Err(Error::InvalidRequest {
            message: "lexical_fallback must be disabled or conservative".to_string(),
        });
    }
    validate_optional_embedding("embedding", request.embedding.as_deref())?;

    if request.query.chars().count() > MAX_SEARCH_QUERY_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("query must be at most {MAX_SEARCH_QUERY_CHARS} characters"),
        });
    }
    let terms = search_terms(&request.query);
    if terms.len() > MAX_SEARCH_TERMS {
        return Err(Error::InvalidRequest {
            message: format!("query must contain at most {MAX_SEARCH_TERMS} searchable terms"),
        });
    }
    if terms.is_empty() {
        return Err(Error::InvalidRequest {
            message: "query must contain at least one searchable term".to_string(),
        });
    }

    let mut filters = request.filters.clone();
    if filters.spaces.is_empty() {
        filters.spaces.push(DEFAULT_SPACE.to_string());
    }
    if filters.statuses.is_empty() {
        filters.statuses.push(status::ACTIVE.to_string());
    }
    // Recall must not surface logically stale facts. Past-`valid_to` and
    // reached-`expires_at` memories are excluded from search even before the
    // dream expire task deletes them. (An API escape hatch can later set this
    // false; the search path forces it on for now.)
    filters.hide_expired = true;
    filters = normalize_search_filters(filters)?;
    validate_search_filters(&filters)?;

    let fts_terms = search_variant_terms(&terms);
    let fts_query = search_fts_query(&fts_terms, request.include_source, " AND ");
    let fallback_fts_query = if fts_terms.len() > 1 {
        Some(search_fts_query(&fts_terms, request.include_source, " OR "))
            .filter(|fallback| fallback != &fts_query)
    } else {
        None
    };
    let prefix_terms = search_prefix_terms(&terms);
    let prefix_fts_query = if prefix_terms.is_empty() {
        None
    } else {
        Some(search_fts_query(
            &prefix_terms,
            request.include_source,
            " AND ",
        ))
        .filter(|prefix| prefix != &fts_query)
    };

    // Inflate the SQL candidate pool so Rust re-scoring can surface memories
    // that SQL undervalues (e.g. recent volatile memories get a 4x recency
    // boost in Rust but look equal-recency to durable in SQL). Use a generous
    // floor — max(limit * 4, limit + 32) — capped at MAX_SEARCH_LIMIT * 4 to
    // stay bounded. Final result is always truncated to `limit` after Rust sort.
    let candidate_pool_limit = request
        .limit
        .saturating_mul(4)
        .max(request.limit.saturating_add(32))
        .min(MAX_SEARCH_LIMIT.saturating_mul(4));

    Ok(PreparedSearchRequest {
        fts_query,
        fallback_fts_query,
        prefix_fts_query,
        terms,
        query_alias_shingles: query_alias_shingles(&request.query),
        filters,
        limit: request.limit,
        candidate_pool_limit,
        offset: request.offset,
        snippet_chars: request.snippet_chars,
        include_content: request.include_content,
        include_source: request.include_source,
        semantic_fallback: request.semantic_fallback.clone(),
        lexical_fallback: request.lexical_fallback.clone(),
        embedding: request.embedding.clone(),
        query_token_embedding: request.query_token_embedding.clone(),
        token_model_id: request.token_model_id.clone(),
    })
}

fn search_fts_query(terms: &[String], include_source: bool, joiner: &str) -> String {
    if include_source {
        terms.join(joiner)
    } else {
        terms
            .iter()
            .map(|term| format!("{{content summary tags metadata_text}} : {term}"))
            .collect::<Vec<_>>()
            .join(joiner)
    }
}

fn search_variant_terms(terms: &[String]) -> Vec<String> {
    terms
        .iter()
        .map(|term| search_term_expression(&search_term_variants(term)))
        .collect()
}

fn search_prefix_terms(terms: &[String]) -> Vec<String> {
    let mut has_prefix_term = false;
    let prefix_terms = terms
        .iter()
        .map(|term| {
            let mut variants = search_term_variants(term);
            for variant in variants.clone() {
                if is_prefixable_search_term(&variant) {
                    has_prefix_term = true;
                    push_unique(&mut variants, format!("{variant}*"));
                }
            }
            search_term_expression(&variants)
        })
        .collect();
    if has_prefix_term {
        prefix_terms
    } else {
        Vec::new()
    }
}

fn search_term_expression(variants: &[String]) -> String {
    if variants.len() == 1 {
        variants[0].clone()
    } else {
        format!("({})", variants.join(" OR "))
    }
}

fn search_term_variants(term: &str) -> Vec<String> {
    let mut variants = vec![term.to_string()];
    if term.chars().count() < 5 || !is_prefixable_search_term(term) {
        return variants;
    }

    for stem in search_term_stems(term) {
        if stem.chars().count() >= 4 && stem != term {
            push_unique(&mut variants, stem);
        }
    }
    variants
}

fn search_term_stems(term: &str) -> Vec<String> {
    let mut stems = Vec::new();
    if let Some(stem) = term.strip_suffix("ies") {
        if !stem.is_empty() {
            stems.push(format!("{stem}y"));
        }
    }
    if let Some(stem) = term.strip_suffix("ied") {
        if !stem.is_empty() {
            stems.push(format!("{stem}y"));
        }
    }
    if let Some(stem) = term.strip_suffix("ing") {
        if stem.chars().count() >= 4 {
            stems.push(trim_doubled_suffix(stem).to_string());
        }
    }
    if let Some(stem) = term.strip_suffix("ed") {
        if stem.chars().count() >= 4 {
            stems.push(trim_doubled_suffix(stem).to_string());
        }
    }
    if let Some(stem) = term.strip_suffix("es") {
        if stem.chars().count() >= 4 {
            stems.push(stem.to_string());
        }
    }
    if term.ends_with('s') && !term.ends_with("ss") {
        let stem = term.trim_end_matches('s');
        if stem.chars().count() >= 4 {
            stems.push(stem.to_string());
        }
    }
    if term.ends_with("ate") || term.ends_with("ize") || term.ends_with("ise") {
        if let Some(stem) = term.strip_suffix('e') {
            if stem.chars().count() >= 4 {
                stems.push(stem.to_string());
            }
        }
    }
    stems
}

fn trim_doubled_suffix(value: &str) -> &str {
    let mut chars = value.char_indices().rev();
    let Some((last_index, last)) = chars.next() else {
        return value;
    };
    let Some((previous_index, previous)) = chars.next() else {
        return value;
    };
    if last == previous {
        &value[..last_index.max(previous_index)]
    } else {
        value
    }
}

fn is_prefixable_search_term(term: &str) -> bool {
    term.chars().count() >= 4
        && term
            .chars()
            .all(|character| character.is_alphanumeric() || character == '_')
}

fn push_unique(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn normalize_search_filters(mut filters: SearchFilters) -> Result<SearchFilters> {
    filters.spaces = normalize_filter_values(&filters.spaces)?;
    filters.silos = normalize_filter_values(&filters.silos)?;
    filters.scopes = normalize_filter_values(&filters.scopes)?;
    filters.projects = normalize_filter_values(&filters.projects)?;
    filters.kinds = normalize_filter_values(&filters.kinds)?;
    filters.statuses = normalize_filter_values(&filters.statuses)?;
    filters.entity_keys = normalize_filter_values(&filters.entity_keys)?;
    filters.claim_keys = normalize_filter_values(&filters.claim_keys)?;
    filters.tags = normalized_tags(&filters.tags)?;
    Ok(filters)
}

fn normalize_filter_values(values: &[String]) -> Result<Vec<String>> {
    if values.len() > MAX_TAGS {
        return Err(Error::InvalidRequest {
            message: "filter has too many values".to_string(),
        });
    }
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(values.len());
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(Error::InvalidRequest {
                message: "filter contains an invalid value".to_string(),
            });
        }
        if !seen.insert(trimmed.to_string()) {
            return Err(Error::InvalidRequest {
                message: format!("filter contains duplicate value: {trimmed}"),
            });
        }
        normalized.push(trimmed.to_string());
    }
    Ok(normalized)
}

fn validate_search_filters(filters: &SearchFilters) -> Result<()> {
    validate_filter_values("spaces", &filters.spaces)?;
    validate_filter_values("silos", &filters.silos)?;
    validate_filter_values("scopes", &filters.scopes)?;
    validate_filter_values("projects", &filters.projects)?;
    validate_filter_values("kinds", &filters.kinds)?;
    validate_filter_values("statuses", &filters.statuses)?;
    validate_filter_values("entity_keys", &filters.entity_keys)?;
    validate_filter_values("claim_keys", &filters.claim_keys)?;
    let _ = normalized_tags(&filters.tags)?;
    for scope_value in &filters.scopes {
        if !is_supported_scope(scope_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported scope: {scope_value}"),
            });
        }
    }
    for kind_value in &filters.kinds {
        if !is_supported_kind(kind_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported kind: {kind_value}"),
            });
        }
    }
    for status_value in &filters.statuses {
        if !is_supported_status(status_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported status: {status_value}"),
            });
        }
    }
    Ok(())
}

fn validate_filter_values(name: &str, values: &[String]) -> Result<()> {
    if values.len() > MAX_TAGS {
        return Err(Error::InvalidRequest {
            message: format!("filter {name} has too many values"),
        });
    }
    let mut seen = BTreeSet::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(Error::InvalidRequest {
                message: format!("filter {name} contains an invalid value"),
            });
        }
        if !seen.insert(trimmed) {
            return Err(Error::InvalidRequest {
                message: format!("filter {name} contains duplicate value: {trimmed}"),
            });
        }
    }
    Ok(())
}

fn search_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut current = String::new();
    for character in query.chars() {
        if character.is_alphanumeric() || character == '_' {
            current.push(character.to_ascii_lowercase());
        } else if !current.is_empty() {
            terms.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        terms.push(current);
    }
    terms.sort();
    terms.dedup();
    let filtered_terms = terms
        .iter()
        .filter(|term| !is_search_stopword(term))
        .cloned()
        .collect::<Vec<_>>();
    if filtered_terms.is_empty() {
        terms
    } else {
        filtered_terms
    }
}

fn is_search_stopword(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "been"
            | "but"
            | "by"
            | "can"
            | "could"
            | "did"
            | "do"
            | "does"
            | "for"
            | "from"
            | "had"
            | "has"
            | "have"
            | "he"
            | "her"
            | "hers"
            | "him"
            | "his"
            | "how"
            | "i"
            | "in"
            | "is"
            | "it"
            | "its"
            | "of"
            | "on"
            | "or"
            | "our"
            | "s"
            | "she"
            | "should"
            | "t"
            | "that"
            | "the"
            | "their"
            | "them"
            | "then"
            | "there"
            | "they"
            | "this"
            | "to"
            | "was"
            | "were"
            | "what"
            | "which"
            | "who"
            | "why"
            | "will"
            | "with"
            | "would"
            | "you"
            | "your"
    )
}

fn entity_search_results(
    connection: &Connection,
    request: &PreparedEntitySearchRequest,
) -> Result<Vec<EntitySearchResult>> {
    let mut predicates = vec!["e.space_name = ?".to_string()];
    let mut values = vec![Value::Text(request.space.clone())];
    if let Some(entity_key) = &request.entity_key {
        predicates.push("e.entity_key = ?".to_string());
        values.push(Value::Text(entity_key.clone()));
    }
    append_sql_in_filter(
        &mut predicates,
        &mut values,
        "e.entity_type",
        &request.entity_types,
    );
    append_sql_in_filter(&mut predicates, &mut values, "e.status", &request.statuses);
    if let Some(query) = &request.query {
        predicates.push(
            "(instr(lower(e.entity_key), lower(?)) > 0 \
              OR instr(lower(e.canonical_name), lower(?)) > 0 \
              OR EXISTS (SELECT 1 FROM entity_aliases ea \
                         WHERE ea.entity_id = e.id AND instr(ea.normalized_alias, ?) > 0))"
                .to_string(),
        );
        values.push(Value::Text(query.clone()));
        values.push(Value::Text(query.clone()));
        values.push(Value::Text(
            request.normalized_query.clone().unwrap_or_default(),
        ));
    }
    let sql = format!(
        "SELECT e.id, e.space_name, e.entity_key, e.entity_type, e.canonical_name,
                e.status, e.confidence, e.source_episode_id, e.created_at, e.updated_at
         FROM entities e
         WHERE {}
         ORDER BY e.status ASC, e.entity_key ASC, e.id ASC
         LIMIT ? OFFSET ?",
        predicates.join(" AND ")
    );
    values.push(Value::Integer(limit_i64(request.limit.saturating_add(1))?));
    values.push(Value::Integer(limit_i64(request.offset)?));
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values.iter()), |row| {
        entity_from_row(row, request.include_source)
    })?;
    let mut entities = collect_rows(rows)?;
    for entity in &mut entities {
        entity.aliases = load_entity_aliases(connection, &entity.id)?;
    }
    entities
        .into_iter()
        .enumerate()
        .map(|(index, entity)| {
            matched_entity_aliases(connection, &entity.id, request.normalized_query.as_deref()).map(
                |matched_aliases| EntitySearchResult {
                    rank: request.offset + index + 1,
                    entity,
                    matched_aliases,
                },
            )
        })
        .collect()
}

fn append_sql_in_filter(
    predicates: &mut Vec<String>,
    values: &mut Vec<Value>,
    column: &str,
    filter_values: &[String],
) {
    if filter_values.is_empty() {
        return;
    }
    let placeholders = std::iter::repeat_n("?", filter_values.len())
        .collect::<Vec<_>>()
        .join(", ");
    predicates.push(format!("{column} IN ({placeholders})"));
    values.extend(filter_values.iter().cloned().map(Value::Text));
}

fn resolve_graph_seed(
    connection: &Connection,
    request: &PreparedGraphNeighborsRequest,
) -> Result<EntityRecord> {
    let mut predicates = vec!["space_name = ?".to_string()];
    let mut values = vec![Value::Text(request.space.clone())];
    if let Some(entity_id) = &request.entity_id {
        predicates.push("id = ?".to_string());
        values.push(Value::Text(entity_id.clone()));
    }
    if let Some(entity_key) = &request.entity_key {
        predicates.push("entity_key = ?".to_string());
        values.push(Value::Text(entity_key.clone()));
    }
    if !request.include_tombstoned {
        predicates.push("status != 'tombstoned'".to_string());
    }
    let sql = format!(
        "SELECT id FROM entities WHERE {} ORDER BY id ASC LIMIT 1",
        predicates.join(" AND ")
    );
    let entity_id = connection
        .query_row(&sql, params_from_iter(values.iter()), |row| {
            row.get::<_, String>(0)
        })
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "entity",
            id: request
                .entity_id
                .as_deref()
                .or(request.entity_key.as_deref())
                .unwrap_or_default()
                .to_string(),
        })?;
    load_entity(connection, &entity_id, request.include_source)
}

fn graph_relationship_rows(
    connection: &Connection,
    seed_id: &str,
    request: &PreparedGraphNeighborsRequest,
) -> Result<Vec<GraphRelationshipRow>> {
    // All string values below bind as `?` parameters; the IN-lists expand to a
    // matching run of placeholders. `space`, `statuses`, and the relation-type
    // list each appear twice (recursive term + final select), so the bind order
    // repeats them in that order. `depth`/`limit` are validated integers, not
    // user strings, so they stay interpolated.
    let status_placeholders = std::iter::repeat_n("?", request.statuses.len())
        .collect::<Vec<_>>()
        .join(",");
    let relation_type_predicate = if request.relation_types.is_empty() {
        String::new()
    } else {
        let placeholders = std::iter::repeat_n("?", request.relation_types.len())
            .collect::<Vec<_>>()
            .join(",");
        format!(" AND r.relation_type IN ({placeholders})")
    };
    let entity_status_predicate = if request.include_tombstoned {
        String::new()
    } else {
        " AND next_entity.status != 'tombstoned'".to_string()
    };
    let edge_entity_status_predicate = if request.include_tombstoned {
        String::new()
    } else {
        " AND subject_entity.status != 'tombstoned' AND object_entity.status != 'tombstoned'"
            .to_string()
    };
    let sql = format!(
        "WITH RECURSIVE walk(entity_id, depth) AS (
             SELECT ?, 0
             UNION
             SELECT CASE
                      WHEN r.subject_entity_id = walk.entity_id THEN r.object_entity_id
                      ELSE r.subject_entity_id
                    END,
                    walk.depth + 1
               FROM walk
               JOIN relationships r
                 ON r.space_name = ?
                AND (r.subject_entity_id = walk.entity_id OR r.object_entity_id = walk.entity_id)
               JOIN entities next_entity
                 ON next_entity.id = CASE
                      WHEN r.subject_entity_id = walk.entity_id THEN r.object_entity_id
                      ELSE r.subject_entity_id
                    END
               LEFT JOIN memories evidence_memory ON evidence_memory.id = r.memory_id
              WHERE walk.depth < {depth}
                AND r.status IN ({status_placeholders})
                {relation_type_predicate}
                {entity_status_predicate}
                AND (r.memory_id IS NULL OR evidence_memory.status = 'active')
           ),
           nodes AS (
             SELECT entity_id, MIN(depth) AS depth FROM walk GROUP BY entity_id
           )
         SELECT r.id, r.space_name, r.subject_entity_id, r.relation_type, r.object_entity_id,
                r.memory_id, r.source_episode_id, r.status, r.confidence, r.observed_at,
                r.valid_from, r.valid_to, r.created_at, r.updated_at, s.depth, o.depth
           FROM relationships r
           JOIN nodes s ON s.entity_id = r.subject_entity_id
           JOIN nodes o ON o.entity_id = r.object_entity_id
           JOIN entities subject_entity ON subject_entity.id = r.subject_entity_id
           JOIN entities object_entity ON object_entity.id = r.object_entity_id
           LEFT JOIN memories evidence_memory ON evidence_memory.id = r.memory_id
          WHERE r.space_name = ?
            AND r.status IN ({status_placeholders})
            {relation_type_predicate}
            {edge_entity_status_predicate}
            AND (r.memory_id IS NULL OR evidence_memory.status = 'active')
          ORDER BY min(s.depth, o.depth) ASC, max(s.depth, o.depth) ASC, r.relation_type ASC, r.id ASC
          LIMIT {limit}",
        depth = request.depth,
        limit = request.max_edges.saturating_add(1),
    );
    let mut binds: Vec<&str> = Vec::new();
    binds.push(seed_id);
    binds.push(request.space.as_str());
    binds.extend(request.statuses.iter().map(String::as_str));
    binds.extend(request.relation_types.iter().map(String::as_str));
    binds.push(request.space.as_str());
    binds.extend(request.statuses.iter().map(String::as_str));
    binds.extend(request.relation_types.iter().map(String::as_str));
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(binds.iter()), |row| {
        graph_relationship_row_from_row(row, request)
    })?;
    collect_rows(rows)
}

fn graph_context_evidence_memory_ids(graph: &GraphNeighborsReport) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for relationship in &graph.relationships {
        if let Some(memory_id) = &relationship.relationship.memory_id {
            if seen.insert(memory_id.clone()) {
                ids.push(memory_id.clone());
            }
        }
    }
    ids
}

fn graph_context_entity_memory_ids(
    connection: &Connection,
    graph: &GraphNeighborsReport,
    request: &GraphContextRequest,
) -> Result<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for entity in &graph.entities {
        if seen.contains(&entity.entity.entity_key) {
            continue;
        }
        seen.insert(entity.entity.entity_key.clone());
        let mut statement = connection.prepare(
            "SELECT id
             FROM memories
             WHERE space_name = ?1
               AND status = 'active'
               AND entity_key = ?2
             ORDER BY pinned DESC, observed_at DESC, updated_at DESC, id ASC
             LIMIT ?3",
        )?;
        let limit = limit_i64(request.max_memories.saturating_add(1))?;
        let rows = statement.query_map(
            params![&entity.entity.space, &entity.entity.entity_key, limit],
            |row| row.get::<_, String>(0),
        )?;
        for memory_id in collect_rows(rows)? {
            if ids.len()
                >= request
                    .max_memories
                    .saturating_add(graph.relationships.len())
            {
                break;
            }
            if ids.iter().all(|existing| existing != &memory_id) {
                ids.push(memory_id);
            }
        }
    }
    Ok(ids)
}

fn graph_context_memory_ids(evidence_ids: &[String], entity_ids: &[String]) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut ids = Vec::new();
    for memory_id in evidence_ids.iter().chain(entity_ids.iter()) {
        if seen.insert(memory_id.clone()) {
            ids.push(memory_id.clone());
        }
    }
    ids
}

fn graph_context_memory_results(
    connection: &Connection,
    memory_ids: &[String],
    request: &GraphContextRequest,
) -> Result<Vec<SearchResult>> {
    memory_ids
        .iter()
        .filter_map(|memory_id| {
            match graph_context_memory_result(connection, memory_id, request.include_source) {
                Ok(Some(result)) => Some(Ok(result)),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            }
        })
        .enumerate()
        .map(|(index, result)| {
            let mut result = result?;
            result.rank = index + 1;
            Ok(result)
        })
        .collect()
}

fn graph_context_memory_result(
    connection: &Connection,
    memory_id: &str,
    include_source: bool,
) -> Result<Option<SearchResult>> {
    let source_sql = if include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let sql = format!(
        "SELECT
            m.id, m.active_version_id, m.space_name, m.silo_name, m.scope, m.project_key,
            m.kind, m.status, v.summary, substr(v.content, 1, 1000),
            COALESCE((SELECT group_concat(tag, char(31)) FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)), ''),
            m.entity_key, m.claim_key, m.observed_at, {source_sql}
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.id = ?1 AND m.status = 'active'"
    );
    connection
        .query_row(&sql, [memory_id], |row| {
            let tags_joined = row.get::<_, Option<String>>(10)?.unwrap_or_default();
            Ok(SearchResult {
                rank: 0,
                memory_id: row.get(0)?,
                version_id: row.get(1)?,
                score: 0.0,
                scores: ScoreBreakdown {
                    fts: 0.0,
                    metadata: 0.0,
                    recency: 0.0,
                    scope: 0.0,
                    status: 0.0,
                    pin: 0.0,
                    source_tier: 0.0,
                },
                space: row.get(2)?,
                silo: row.get(3)?,
                scope: row.get(4)?,
                project_key: row.get(5)?,
                kind: row.get(6)?,
                status: row.get(7)?,
                summary: row.get(8)?,
                snippet: row.get(9)?,
                content: None,
                tags: split_tags(&tags_joined),
                entity_key: row.get(11)?,
                claim_key: row.get(12)?,
                observed_at: row.get(13)?,
                source_ref_json: row.get(14)?,
                metadata_json: None,
            })
        })
        .optional()
        .map_err(Into::into)
}

fn graph_relationship_row_from_row(
    row: &Row<'_>,
    request: &PreparedGraphNeighborsRequest,
) -> rusqlite::Result<GraphRelationshipRow> {
    let source_episode_id = if request.include_source {
        row.get(6)?
    } else {
        None
    };
    Ok(GraphRelationshipRow {
        relationship: RelationshipRecord {
            id: row.get(0)?,
            space: row.get(1)?,
            subject_entity_id: row.get(2)?,
            relation_type: row.get(3)?,
            object_entity_id: row.get(4)?,
            memory_id: row.get(5)?,
            source_episode_id,
            status: row.get(7)?,
            confidence: row.get(8)?,
            observed_at: row.get(9)?,
            valid_from: row.get(10)?,
            valid_to: row.get(11)?,
            created_at: row.get(12)?,
            updated_at: row.get(13)?,
        },
        subject_depth: usize::try_from(row.get::<_, i64>(14)?).unwrap_or(usize::MAX),
        object_depth: usize::try_from(row.get::<_, i64>(15)?).unwrap_or(usize::MAX),
    })
}

fn resolve_relationship_endpoint(
    connection: &Connection,
    space: &str,
    entity_id: Option<&str>,
    entity_key: Option<&str>,
    label: &'static str,
) -> Result<String> {
    let mut predicates = vec!["space_name = ?".to_string()];
    let mut values = vec![Value::Text(space.to_string())];
    if let Some(entity_id) = entity_id {
        predicates.push("id = ?".to_string());
        values.push(Value::Text(entity_id.to_string()));
    }
    if let Some(entity_key) = entity_key {
        predicates.push("entity_key = ?".to_string());
        values.push(Value::Text(entity_key.to_string()));
    }
    let sql = format!(
        "SELECT id FROM entities WHERE {} ORDER BY id ASC LIMIT 1",
        predicates.join(" AND ")
    );
    connection
        .query_row(&sql, params_from_iter(values.iter()), |row| {
            row.get::<_, String>(0)
        })
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: label,
            id: entity_id.or(entity_key).unwrap_or_default().to_string(),
        })
}

fn find_existing_relationship(
    connection: &Connection,
    space: &str,
    subject_entity_id: &str,
    relation_type: &str,
    object_entity_id: &str,
    memory_id: Option<&str>,
    source_episode_id: Option<&str>,
) -> Result<Option<String>> {
    connection
        .query_row(
            "SELECT id FROM relationships
             WHERE space_name = ?1
               AND subject_entity_id = ?2
               AND relation_type = ?3
               AND object_entity_id = ?4
               AND ((memory_id IS NULL AND ?5 IS NULL) OR memory_id = ?5)
               AND ((source_episode_id IS NULL AND ?6 IS NULL) OR source_episode_id = ?6)
             ORDER BY id ASC
             LIMIT 1",
            params![
                space,
                subject_entity_id,
                relation_type,
                object_entity_id,
                memory_id,
                source_episode_id,
            ],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(Into::into)
}

fn load_relationship(
    connection: &Connection,
    relationship_id: &str,
    include_source: bool,
) -> Result<RelationshipRecord> {
    connection
        .query_row(
            "SELECT id, space_name, subject_entity_id, relation_type, object_entity_id,
                    memory_id, source_episode_id, status, confidence, observed_at,
                    valid_from, valid_to, created_at, updated_at
             FROM relationships
             WHERE id = ?1",
            [relationship_id],
            |row| relationship_from_row(row, include_source),
        )
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "relationship",
            id: relationship_id.to_string(),
        })
}

fn relationship_from_row(
    row: &Row<'_>,
    include_source: bool,
) -> rusqlite::Result<RelationshipRecord> {
    let source_episode_id = if include_source { row.get(6)? } else { None };
    Ok(RelationshipRecord {
        id: row.get(0)?,
        space: row.get(1)?,
        subject_entity_id: row.get(2)?,
        relation_type: row.get(3)?,
        object_entity_id: row.get(4)?,
        memory_id: row.get(5)?,
        source_episode_id,
        status: row.get(7)?,
        confidence: row.get(8)?,
        observed_at: row.get(9)?,
        valid_from: row.get(10)?,
        valid_to: row.get(11)?,
        created_at: row.get(12)?,
        updated_at: row.get(13)?,
    })
}

fn load_entity(
    connection: &Connection,
    entity_id: &str,
    include_source: bool,
) -> Result<EntityRecord> {
    let mut entity = connection
        .query_row(
            "SELECT id, space_name, entity_key, entity_type, canonical_name, status, confidence,
                    source_episode_id, created_at, updated_at
             FROM entities
             WHERE id = ?1",
            [entity_id],
            |row| entity_from_row(row, include_source),
        )
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "entity",
            id: entity_id.to_string(),
        })?;
    entity.aliases = load_entity_aliases(connection, &entity.id)?;
    Ok(entity)
}

fn entity_from_row(row: &Row<'_>, include_source: bool) -> rusqlite::Result<EntityRecord> {
    let source_episode_id = if include_source { row.get(7)? } else { None };
    Ok(EntityRecord {
        id: row.get(0)?,
        space: row.get(1)?,
        entity_key: row.get(2)?,
        entity_type: row.get(3)?,
        canonical_name: row.get(4)?,
        status: row.get(5)?,
        confidence: row.get(6)?,
        aliases: Vec::new(),
        source_episode_id,
        created_at: row.get(8)?,
        updated_at: row.get(9)?,
    })
}

fn load_entity_aliases(connection: &Connection, entity_id: &str) -> Result<Vec<String>> {
    let mut statement = connection.prepare(
        "SELECT alias
         FROM entity_aliases
         WHERE entity_id = ?1
         ORDER BY normalized_alias ASC, alias ASC",
    )?;
    let rows = statement.query_map([entity_id], |row| row.get::<_, String>(0))?;
    collect_rows(rows)
}

fn matched_entity_aliases(
    connection: &Connection,
    entity_id: &str,
    normalized_query: Option<&str>,
) -> Result<Vec<String>> {
    let Some(normalized_query) = normalized_query.filter(|value| !value.is_empty()) else {
        return Ok(Vec::new());
    };
    let mut statement = connection.prepare(
        "SELECT alias
         FROM entity_aliases
         WHERE entity_id = ?1 AND instr(normalized_alias, ?2) > 0
         ORDER BY normalized_alias ASC, alias ASC
         LIMIT ?3",
    )?;
    let rows = statement.query_map(
        params![entity_id, normalized_query, limit_i64(MAX_TAGS)?],
        |row| row.get::<_, String>(0),
    )?;
    collect_rows(rows)
}

fn list_memory_candidates(
    connection: &Connection,
    request: &PreparedMemoryListRequest,
) -> Result<Vec<MemoryListCandidate>> {
    let (sql, args) = memory_list_sql(request);
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(args.iter()), |row| {
        let tags_joined = row.get::<_, Option<String>>(17)?.unwrap_or_default();
        Ok(MemoryListCandidate {
            memory_id: row.get(0)?,
            version_id: row.get(1)?,
            space: row.get(2)?,
            silo: row.get(3)?,
            scope: row.get(4)?,
            project_key: row.get(5)?,
            kind: row.get(6)?,
            status: row.get(7)?,
            entity_key: row.get(8)?,
            claim_key: row.get(9)?,
            confidence: row.get(10)?,
            pinned: row.get::<_, i64>(11)? == 1,
            observed_at: row.get(12)?,
            created_at: row.get(13)?,
            updated_at: row.get(14)?,
            snippet_text: row.get(15)?,
            content: row.get(16)?,
            tags: split_tags(&tags_joined),
            summary: row.get(18)?,
            source_ref_json: row.get(19)?,
        })
    })?;
    collect_rows(rows)
}

fn memory_list_sql(request: &PreparedMemoryListRequest) -> (String, Vec<String>) {
    let mut args = SqlArgs::with_reserved(0);
    let where_clause = filters_where_clause(&request.filters, &mut args);
    let order = memory_list_order_sql(&request.order);
    let row_limit = request.limit.saturating_add(1);
    let row_offset = request.offset;
    let snippet_sql = if request.snippet_chars == 0 {
        "''".to_string()
    } else {
        format!(
            "substr(COALESCE(v.summary, v.content), 1, {})",
            request.snippet_chars.saturating_add(1)
        )
    };
    let content_sql = if request.include_content {
        "v.content"
    } else {
        "NULL"
    };
    let source_sql = if request.include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let sql = format!(
        "SELECT
            m.id,
            m.active_version_id,
            m.space_name,
            m.silo_name,
            m.scope,
            m.project_key,
            m.kind,
            m.status,
            m.entity_key,
            m.claim_key,
            m.confidence,
            m.pinned,
            m.observed_at,
            m.created_at,
            m.updated_at,
            {snippet_sql},
            {content_sql},
            COALESCE((SELECT group_concat(tag, char(31)) FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)), ''),
            v.summary,
            {source_sql}
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {where_clause}
         ORDER BY {order}, m.id ASC
         LIMIT {row_limit} OFFSET {row_offset}"
    );
    (sql, args.values)
}

fn memory_list_order_sql(order: &str) -> &'static str {
    match order {
        "observed_desc" => "m.observed_at DESC, m.updated_at DESC",
        "created_desc" => "m.created_at DESC, m.updated_at DESC",
        _ => "m.updated_at DESC, m.observed_at DESC",
    }
}

/// Maps one row of a candidate SELECT (see `candidate_select_columns`) onto a
/// `SearchCandidate`. The column order here and in `candidate_select_columns`
/// must stay in lockstep; sharing one mapper keeps the FTS and semantic paths
/// from drifting apart.
fn candidate_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SearchCandidate> {
    let tags_joined = row.get::<_, Option<String>>(13)?.unwrap_or_default();
    Ok(SearchCandidate {
        memory_id: row.get(0)?,
        version_id: row.get(1)?,
        space: row.get(2)?,
        silo: row.get(3)?,
        scope: row.get(4)?,
        project_key: row.get(5)?,
        kind: row.get(6)?,
        status: row.get(7)?,
        entity_key: row.get(8)?,
        claim_key: row.get(9)?,
        observed_at: row.get(10)?,
        recency_jd: row.get(11)?,
        pinned: row.get::<_, i64>(12)? == 1,
        tags: split_tags(&tags_joined),
        confidence: row.get(14)?,
        content: row.get(15)?,
        snippet_text: row.get(16)?,
        summary: row.get(17)?,
        source_ref_json: row.get(18)?,
        bm25: row.get(19)?,
        metadata_json: row.get(20)?,
        source_type: row.get(21)?,
        score: 0.0,
        lexical_tier: 0,
        scores: ScoreBreakdown {
            fts: 0.0,
            metadata: 0.0,
            recency: 0.0,
            scope: 0.0,
            status: 0.0,
            pin: 0.0,
            source_tier: 0.0,
        },
    })
}

/// Shared SELECT column list consumed by `candidate_from_row`. `rank_sql` is
/// the path-specific relevance column (FTS bm25 or vector distance).
fn candidate_select_columns(
    recency_jd: &str,
    content_sql: &str,
    snippet_sql: &str,
    source_sql: &str,
    rank_sql: &str,
) -> String {
    // `{recency_jd}` and `{rank_sql}` are aliased so a wrapping subquery can
    // reference each by name and reuse the single evaluation rather than
    // recomputing the expensive julianday()/bm25() call in ORDER BY. Positional
    // column reads in `candidate_from_row` are unaffected by the aliases.
    format!(
        "m.id,
            m.active_version_id,
            m.space_name,
            m.silo_name,
            m.scope,
            m.project_key,
            m.kind,
            m.status,
            m.entity_key,
            m.claim_key,
            m.observed_at,
            {recency_jd} AS rec_jd,
            m.pinned,
            COALESCE((SELECT group_concat(tag, char(31)) FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)), '') AS tags_joined,
            m.confidence,
            {content_sql},
            {snippet_sql},
            v.summary,
            {source_sql},
            {rank_sql} AS relevance,
            m.metadata_json,
            json_extract(v.source_ref_json, '$.type') AS source_type"
    )
}

fn search_candidates(
    connection: &Connection,
    request: &PreparedSearchRequest,
    fts_query: &str,
) -> Result<Vec<SearchCandidate>> {
    let (sql, args) = search_sql(request);
    let mut statement = connection.prepare_cached(&sql)?;
    let params = std::iter::once(fts_query.to_string()).chain(args);
    let rows = statement.query_map(params_from_iter(params), candidate_from_row)?;
    collect_rows(rows)
}

#[cfg(feature = "semantic")]
fn semantic_candidates(
    connection: &Connection,
    request: &PreparedSearchRequest,
    embedding: &[f32],
) -> Result<Vec<SearchCandidate>> {
    if let (Some(query_tokens), Some(token_model)) = (
        request.query_token_embedding.as_ref(),
        request.token_model_id.as_deref(),
    ) {
        return semantic_candidates_late_interaction(
            connection,
            request,
            query_tokens,
            token_model,
            embedding,
        );
    }
    let table = semantic_table_for_dims(embedding.len())?;
    let (sql, args) = semantic_search_sql(request, &table);
    let embedding_json = embedding_json(embedding)?;
    let mut statement = connection.prepare(&sql)?;
    let params = std::iter::once(embedding_json).chain(args);
    let rows = statement.query_map(params_from_iter(params), candidate_from_row)?;
    collect_rows(rows)
}

/// Late-interaction semantic candidates: exhaustive `MaxSim` selects WHICH
/// memories enter the pipeline; each selected candidate carries its exact
/// single-vector L2 distance (the same statistic the vec0 ANN table reports
/// for unit vectors), so downstream fused scoring keeps its scale.
#[cfg(feature = "semantic")]
fn semantic_candidates_late_interaction(
    connection: &Connection,
    request: &PreparedSearchRequest,
    query_tokens: &[Vec<f32>],
    token_model: &str,
    embedding: &[f32],
) -> Result<Vec<SearchCandidate>> {
    // Selection is capped at limit+offset (NOT candidate_pool_limit): MaxSim
    // decides WHICH memories proceed; the fused single-vector score must only
    // order within that set. With the inflated pool, fused sort-then-truncate
    // lets the single-vector ranking veto MaxSim's selection and the recall
    // gain disappears (observed in the first acceptance run).
    let selection = request.limit.saturating_add(request.offset);
    let pool = maxsim_candidates(connection, query_tokens, token_model, selection)?;
    if pool.is_empty() {
        return Ok(vec![]);
    }
    // Exact L2 distance from the stored single vector (2.0 = max for unit
    // vectors, used when a candidate has no ready single vector).
    let mut vector_statement = connection.prepare_cached(
        "SELECT vector_blob FROM embeddings WHERE memory_id = ?1 AND status = 'ready' \
         AND vector_blob IS NOT NULL ORDER BY updated_at DESC LIMIT 1",
    )?;
    let mut args = SqlArgs::with_reserved(0);
    let mut values = Vec::with_capacity(pool.len());
    for item in &pool {
        let blob: Option<Vec<u8>> = vector_statement
            .query_row([&item.memory_id], |row| row.get(0))
            .optional()?;
        let distance = blob
            .filter(|blob| blob.len() == embedding.len() * 4)
            .map_or(2.0_f64, |blob| {
                let mut sum = 0.0_f64;
                for (chunk, query_value) in blob.chunks_exact(4).zip(embedding) {
                    let doc_value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    sum += f64::from(query_value - doc_value).powi(2);
                }
                sum.sqrt()
            });
        let placeholder = args.push(&item.memory_id);
        values.push(format!("({placeholder}, {distance:.17})"));
    }
    let values_sql = values.join(",");
    let where_clause = filters_where_clause(&request.filters, &mut args);
    let recency_jd = recency_jd_sql();
    let content_sql = if request.include_content {
        "v.content"
    } else {
        "NULL"
    };
    let snippet_sql = if request.snippet_chars == 0 {
        "''".to_string()
    } else {
        format!("substr(v.content, 1, {})", request.snippet_chars)
    };
    let source_sql = if request.include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let columns = candidate_select_columns(
        &recency_jd,
        content_sql,
        &snippet_sql,
        source_sql,
        "li.distance",
    );
    let sql = format!(
        "WITH li(memory_id, distance) AS (VALUES {values_sql})
         SELECT
            {columns}
         FROM li
         JOIN memories m ON m.id = li.memory_id
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {where_clause}
         ORDER BY li.distance ASC, m.observed_at DESC, m.id ASC
         LIMIT {} OFFSET {}",
        request.candidate_pool_limit, request.offset
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(args.values), candidate_from_row)?;
    collect_rows(rows)
}

fn search_fts_table(request: &PreparedSearchRequest) -> &'static str {
    if request.include_source {
        "memory_fts"
    } else {
        "memory_fts_public"
    }
}

fn search_sql(request: &PreparedSearchRequest) -> (String, Vec<String>) {
    let fts_table = search_fts_table(request);
    let mut args = SqlArgs::with_reserved(1);
    let where_clause = search_where_clause(request, &mut args);
    let score_sql = search_score_sql(request, &mut args);
    let recency_jd = recency_jd_sql();
    // Use the inflated pool so Rust re-scoring has room to surface SQL-undervalued
    // candidates (e.g. recent volatile memories). Final results are truncated to
    // request.limit after Rust scoring + sort.
    let candidate_limit = request.candidate_pool_limit.saturating_add(request.offset);
    let candidate_offset = request.offset;
    let content_sql = if request.include_content {
        "v.content"
    } else {
        "NULL"
    };
    let snippet_sql = if request.snippet_chars == 0 {
        "''".to_string()
    } else {
        format!("snippet({fts_table}, 6, '', '', ' … ', 32)")
    };
    let source_sql = if request.include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let bm25_sql = format!("bm25({fts_table})");
    let columns = candidate_select_columns(
        &recency_jd,
        content_sql,
        &snippet_sql,
        source_sql,
        &bm25_sql,
    );
    let sql = format!(
        "SELECT
            {columns}
         FROM {fts_table}
         JOIN memories m ON m.id = {fts_table}.memory_id
         JOIN memory_versions v ON v.id = {fts_table}.version_id
         WHERE {where_clause}
         ORDER BY {bm25_sql} ASC, {score_sql} DESC, m.observed_at DESC, m.id ASC
         LIMIT {candidate_limit} OFFSET {candidate_offset}"
    );
    (sql, args.values)
}

#[cfg(feature = "semantic")]
fn semantic_search_sql(request: &PreparedSearchRequest, table: &str) -> (String, Vec<String>) {
    let mut args = SqlArgs::with_reserved(1);
    let where_clause = filters_where_clause(&request.filters, &mut args);
    let recency_jd = recency_jd_sql();
    // Use the inflated pool (same logic as FTS search_sql) so Rust re-scoring
    // can surface SQL-undervalued candidates after silo-aware recency boosting.
    let candidate_limit = request.candidate_pool_limit.saturating_add(request.offset);
    let content_sql = if request.include_content {
        "v.content"
    } else {
        "NULL"
    };
    // The consumer only ever takes a prefix of `snippet_text` (no term
    // centering on the semantic path), so fetch just that prefix instead of
    // full content for every pool row.
    let snippet_sql = if request.snippet_chars == 0 {
        "''".to_string()
    } else {
        format!("substr(v.content, 1, {})", request.snippet_chars)
    };
    let source_sql = if request.include_source {
        "v.source_ref_json"
    } else {
        "NULL"
    };
    let distance_sql = format!("{table}.distance");
    let columns = candidate_select_columns(
        &recency_jd,
        content_sql,
        &snippet_sql,
        source_sql,
        &distance_sql,
    );
    let sql = format!(
        "SELECT
            {columns}
         FROM {table}
         JOIN memories m ON m.id = {table}.memory_id
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE {table}.embedding MATCH ?1 AND k = {candidate_limit} AND {where_clause}
         ORDER BY {table}.distance ASC, m.observed_at DESC, m.id ASC
         LIMIT {} OFFSET {}",
        request.candidate_pool_limit, request.offset
    );
    (sql, args.values)
}

fn search_score_sql(request: &PreparedSearchRequest, args: &mut SqlArgs) -> String {
    // NOTE: this expression is used ONLY as the bm25 tiebreaker in the candidate
    // pool ORDER BY (`bm25 ASC, score_sql DESC, observed_at DESC, id ASC`); it is
    // never selected or stored. The relevance/`fts` term was dropped here on
    // purpose: `fts` is a pure function of `bm25`, so it can only differ between
    // two rows when their `bm25` differs — but in that case the primary
    // `bm25 ASC` key has already decided their order. Whenever the tiebreaker
    // actually matters (equal bm25), `fts` is identical across the tied rows and
    // contributes an equal constant, so removing it leaves the ordering
    // byte-identical for all inputs while avoiding two extra bm25() evaluations
    // per matched row (bm25 is FTS5's most expensive scalar).
    let confidence = "(m.confidence * 0.05)";
    let recency = recency_score_sql(&recency_jd_sql());
    let kind = boost_in_sql("m.kind", &request.filters.kinds, 0.05, args);
    let entity = boost_in_sql("m.entity_key", &request.filters.entity_keys, 0.10, args);
    let claim = boost_in_sql("m.claim_key", &request.filters.claim_keys, 0.10, args);
    let scope = boost_in_sql("m.scope", &request.filters.scopes, 0.03, args);
    let tag = if request.filters.tags.is_empty() {
        "0.0".to_string()
    } else {
        format!(
            "(CASE WHEN EXISTS (SELECT 1 FROM memory_tags mt WHERE mt.memory_id = m.id AND mt.tag IN ({})) THEN 0.05 ELSE 0.0 END)",
            args.placeholder_list(&normalized_tags(&request.filters.tags).expect("validated tags"))
        )
    };
    let status = "(CASE WHEN m.status = 'active' THEN 0.02 ELSE 0.0 END)";
    let pin = "(CASE WHEN m.pinned = 1 THEN 0.05 ELSE 0.0 END)";
    format!(
        "({confidence} + {kind} + {entity} + {claim} + {scope} + {tag} + {recency} + {status} + {pin})"
    )
}

fn recency_jd_sql() -> String {
    // Bit-identical to the previous 4-branch CASE (most-recent of updated/observed,
    // null-tolerant) but evaluates julianday() ~2x per row instead of ~4x: the
    // multi-arg max() returns the larger non-null value, returning NULL only when
    // BOTH are null, and coalesce supplies the single-null fallbacks. The produced
    // value (hence the Rust recency score, overall sort order, and the SELECTed
    // recency_jd column) is unchanged for every input.
    "coalesce(max(julianday(m.updated_at), julianday(m.observed_at)), \
       julianday(m.updated_at), julianday(m.observed_at))"
        .to_string()
}

fn recency_score_sql(recency_jd_sql: &str) -> String {
    // Linear approximation of the Rust half-life curve (zero at two
    // half-lives), silo-aware. Used only for SQL-side candidate-pool
    // ordering; the Rust re-scoring pass is authoritative.
    let durable_window = DURABLE_RECENCY_HALF_LIFE_DAYS * 2.0;
    let volatile_window = VOLATILE_RECENCY_HALF_LIFE_DAYS * 2.0;
    // The explicit `WHEN {recency_jd} IS NULL THEN 0.0` guard re-evaluated the
    // recency_jd expression an extra time per row. Fold it into an outer
    // coalesce instead: when recency_jd is NULL the inner arithmetic yields NULL
    // (julianday(NULL) -> NULL propagates through max/subtraction), which
    // coalesce maps to 0.0 -- bit-identical to the guard, but recency_jd is now
    // evaluated once (in the taken silo branch) rather than twice.
    format!(
        "coalesce(CASE \
          WHEN m.silo_name = '{DEFAULT_DURABLE_SILO}' \
          THEN {MAX_RECENCY_SCORE} * max(0.0, 1.0 - (max(0.0, julianday('now') - {recency_jd_sql}) / {durable_window})) \
          ELSE {VOLATILE_MAX_RECENCY_SCORE} * max(0.0, 1.0 - (max(0.0, julianday('now') - {recency_jd_sql}) / {volatile_window})) END, 0.0)"
    )
}

fn boost_in_sql(column: &str, values: &[String], boost: f64, args: &mut SqlArgs) -> String {
    if values.is_empty() {
        "0.0".to_string()
    } else {
        format!(
            "(CASE WHEN {column} IN ({}) THEN {boost} ELSE 0.0 END)",
            args.placeholder_list(values)
        )
    }
}

/// Accumulates bound SQL parameter values, handing out `?N` placeholders.
/// `reserved` counts placeholders the caller binds ahead of these values
/// (e.g. `?1` for the FTS query string or the query embedding).
struct SqlArgs {
    reserved: usize,
    values: Vec<String>,
}

impl SqlArgs {
    fn with_reserved(reserved: usize) -> Self {
        Self {
            reserved,
            values: Vec::new(),
        }
    }

    fn push(&mut self, value: &str) -> String {
        self.values.push(value.to_string());
        format!("?{}", self.reserved + self.values.len())
    }

    fn placeholder_list(&mut self, values: &[String]) -> String {
        values
            .iter()
            .map(|value| self.push(value))
            .collect::<Vec<_>>()
            .join(",")
    }
}

fn search_where_clause(request: &PreparedSearchRequest, args: &mut SqlArgs) -> String {
    let mut predicates = vec![format!("{} MATCH ?1", search_fts_table(request))];
    predicates.extend(filter_predicates(&request.filters, args));
    predicates.join(" AND ")
}

/// Filter predicates shared by the FTS search, semantic search, and
/// memory-list queries. All values are bound via `args`.
fn filter_predicates(filters: &SearchFilters, args: &mut SqlArgs) -> Vec<String> {
    let mut predicates = Vec::new();
    push_in_predicate(&mut predicates, "m.space_name", &filters.spaces, args);
    push_in_predicate(&mut predicates, "m.silo_name", &filters.silos, args);
    push_in_predicate(&mut predicates, "m.scope", &filters.scopes, args);
    push_in_predicate(&mut predicates, "m.project_key", &filters.projects, args);
    push_in_predicate(&mut predicates, "m.kind", &filters.kinds, args);
    push_in_predicate(&mut predicates, "m.status", &filters.statuses, args);
    push_in_predicate(&mut predicates, "m.entity_key", &filters.entity_keys, args);
    push_in_predicate(&mut predicates, "m.claim_key", &filters.claim_keys, args);
    if !filters.tags.is_empty() {
        predicates.push(format!(
            "EXISTS (SELECT 1 FROM memory_tags mt WHERE mt.memory_id = m.id AND mt.tag IN ({}))",
            args.placeholder_list(&normalized_tags(&filters.tags).expect("validated tags"))
        ));
    }
    if filters.hide_expired {
        // Exclude logically stale facts from recall: a memory whose `valid_to`
        // has passed, or whose `expires_at` has been reached, is no longer
        // current even if the dream expire task has not yet removed it.
        // Mirrors the `active_past_valid_to` stats diagnostic's clock
        // (`julianday('now')`, consistent within a single SQLite statement).
        predicates
            .push("(m.valid_to IS NULL OR julianday(m.valid_to) >= julianday('now'))".to_string());
        predicates.push(
            "(m.expires_at IS NULL OR julianday(m.expires_at) > julianday('now'))".to_string(),
        );
    }
    predicates
}

fn filters_where_clause(filters: &SearchFilters, args: &mut SqlArgs) -> String {
    let predicates = filter_predicates(filters, args);
    if predicates.is_empty() {
        "1=1".to_string()
    } else {
        predicates.join(" AND ")
    }
}

fn push_in_predicate(
    predicates: &mut Vec<String>,
    column: &str,
    values: &[String],
    args: &mut SqlArgs,
) {
    if !values.is_empty() {
        predicates.push(format!("{column} IN ({})", args.placeholder_list(values)));
    }
}

#[cfg(feature = "semantic")]
fn semantic_table_for_dims(dims: usize) -> Result<String> {
    if !(1..=MAX_SEMANTIC_EMBEDDING_DIMS).contains(&dims) {
        return Err(Error::InvalidRequest {
            message: format!(
                "embedding dimension {dims} is not supported (expected 1..={MAX_SEMANTIC_EMBEDDING_DIMS})"
            ),
        });
    }
    Ok(format!("memory_vec_{dims}"))
}

#[cfg(feature = "semantic")]
fn embedding_json(embedding: &[f32]) -> Result<String> {
    serde_json::to_string(embedding).map_err(|error| Error::InvalidRequest {
        message: format!("failed to encode embedding: {error}"),
    })
}

fn split_tags(tags: &str) -> Vec<String> {
    if tags.is_empty() {
        Vec::new()
    } else {
        tags.split('\u{1f}').map(str::to_string).collect()
    }
}

fn score_candidate(
    candidate: &mut SearchCandidate,
    request: &PreparedSearchRequest,
    best_bm25: f64,
    now_jd: f64,
) -> f64 {
    let fts = fts_score(candidate.bm25, best_bm25);
    let metadata = metadata_score(candidate, request);
    let recency = recency_score_for_silo(candidate.recency_jd, &candidate.silo, now_jd);
    let scope_score = if request.filters.scopes.contains(&candidate.scope) {
        0.03
    } else {
        0.0
    };
    let status_score = if candidate.status == status::ACTIVE {
        0.02
    } else {
        0.0
    };
    let pin = if candidate.pinned { 0.05 } else { 0.0 };
    let source_tier = source_tier_score(candidate.source_type.as_deref(), &candidate.tags);
    candidate.scores = ScoreBreakdown {
        fts,
        metadata,
        recency,
        scope: scope_score,
        status: status_score,
        pin,
        source_tier,
    };
    fts + metadata + recency + scope_score + status_score + pin + source_tier
}

#[cfg(feature = "semantic")]
fn score_semantic_candidate(
    candidate: &mut SearchCandidate,
    request: &PreparedSearchRequest,
    now_jd: f64,
) -> f64 {
    let semantic = (1.0 / (1.0 + candidate.bm25.max(0.0))).min(10.0);
    let metadata = metadata_score(candidate, request);
    let recency = recency_score_for_silo(candidate.recency_jd, &candidate.silo, now_jd);
    let scope_score = if request.filters.scopes.contains(&candidate.scope) {
        0.03
    } else {
        0.0
    };
    let status_score = if candidate.status == status::ACTIVE {
        0.02
    } else {
        0.0
    };
    let pin = if candidate.pinned { 0.05 } else { 0.0 };
    let source_tier = source_tier_score(candidate.source_type.as_deref(), &candidate.tags);
    candidate.scores = ScoreBreakdown {
        fts: semantic,
        metadata,
        recency,
        scope: scope_score,
        status: status_score,
        pin,
        source_tier,
    };
    semantic + metadata + recency + scope_score + status_score + pin + source_tier
}

fn fts_score(bm25: f64, best_bm25: f64) -> f64 {
    // SQLite FTS5 `bm25()` returns <= 0 for matches; more negative = better.
    // Normalize relative to the best (most negative) match in this result set so
    // the score is corpus-independent, monotonic, and bounded in (0, 1] with the
    // best match at 1.0.
    //
    // The previous version computed `(-bm25 * 1_000_000.0).min(10.0)`, which
    // saturated every real-corpus match to 10.0 -- destroying both relevance
    // ranking and the `min_score` floor. It only appeared to work in tiny test
    // stores where raw bm25 magnitudes were near zero (so the product stayed
    // below the 10.0 clamp).
    if best_bm25 < 0.0 && bm25 < 0.0 {
        (bm25 / best_bm25).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn metadata_score(candidate: &SearchCandidate, request: &PreparedSearchRequest) -> f64 {
    let mut score = (candidate.confidence.clamp(0.0, 1.0)) * 0.05;
    if request.filters.kinds.contains(&candidate.kind) {
        score += 0.05;
    }
    if candidate
        .entity_key
        .as_ref()
        .is_some_and(|key| request.filters.entity_keys.contains(key))
    {
        score += 0.10;
    }
    if candidate
        .claim_key
        .as_ref()
        .is_some_and(|key| request.filters.claim_keys.contains(key))
    {
        score += 0.10;
    }
    if candidate
        .tags
        .iter()
        .any(|tag| request.filters.tags.contains(tag))
    {
        score += 0.05;
    }
    // Alias-exact-match boost: a query token matching a memory's reserved
    // `alias::<normalized>` tag is a strong, precise relevance signal that BM25
    // dilutes across a long query. Boosting it lets a topically-weak-but-correct
    // alias hit (e.g. "TypeScript" -> the TS preference) clear the abstention
    // floor while semantic neighbors stay below it. Fires at most once.
    if !request.query_alias_shingles.is_empty()
        && candidate.tags.iter().any(|tag| {
            tag.strip_prefix(ALIAS_TAG_PREFIX)
                .is_some_and(|alias| request.query_alias_shingles.contains(alias))
        })
    {
        score += ALIAS_MATCH_BOOST;
    }
    score
}

/// Source-trust tier boost. Explicit in-session (`mcp`) and manually authored
/// (`manual`) memories rank above auto-harvested synthesis memories, with
/// legacy/unknown provenance sitting in between. Additive nudge in the same
/// magnitude band as the pin/recency boosts: a tiebreaker, not a dominator.
///
/// The nightly synthesis harvester always co-tags its writes `synthesis-derived`,
/// so that tag is treated as the authoritative auto-harvest signal even if the
/// source envelope drifts; otherwise the tier is read from the source `type`
/// (extracted into `source_type`, which is always loaded for ranking).
fn source_tier_score(source_type: Option<&str>, tags: &[String]) -> f64 {
    if tags.iter().any(|tag| tag == "synthesis-derived") {
        return 0.0;
    }
    match source_type {
        Some("synthesis") => 0.0,
        Some("manual" | "mcp") => 0.04,
        // Legacy (no provenance) or unrecognized source: a small edge over
        // auto-harvest, below explicit/manual writes.
        _ => 0.02,
    }
}

/// Maximum deterministic recency boost for volatile (non-durable) silos.
/// Steeper curve ensures a recent volatile claim decisively outranks an old one.
const VOLATILE_MAX_RECENCY_SCORE: f64 = 0.20;

/// Current time as a Julian day, matching `SQLite` `julianday('now')`.
fn now_julian_day() -> f64 {
    let unix_seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |duration| duration.as_secs_f64());
    UNIX_EPOCH_JD + unix_seconds / 86_400.0
}

/// Age-based exponential recency boost: `max * 0.5^(age_days / half_life)`.
/// Future timestamps (clock skew) clamp to zero age, i.e. the full boost.
fn recency_score_for_silo(recency_jd: Option<f64>, silo: &str, now_jd: f64) -> f64 {
    let Some(recency_jd) = recency_jd.filter(|value| value.is_finite()) else {
        return 0.0;
    };
    let age_days = (now_jd - recency_jd).max(0.0);
    let (max_score, half_life_days) = if silo == DEFAULT_DURABLE_SILO {
        (MAX_RECENCY_SCORE, DURABLE_RECENCY_HALF_LIFE_DAYS)
    } else {
        (VOLATILE_MAX_RECENCY_SCORE, VOLATILE_RECENCY_HALF_LIFE_DAYS)
    };
    max_score * 0.5_f64.powf(age_days / half_life_days)
}

fn compare_candidates(left: &SearchCandidate, right: &SearchCandidate) -> std::cmp::Ordering {
    left.lexical_tier
        .cmp(&right.lexical_tier)
        .then_with(|| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| right.observed_at.cmp(&left.observed_at))
        .then_with(|| left.memory_id.cmp(&right.memory_id))
}

fn freshness_marker(silo: &str, metadata_json: Option<&str>, last_synth: Option<&str>) -> String {
    if silo == DEFAULT_DURABLE_SILO {
        return String::new();
    }
    let md = metadata_json.and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok());
    let ptr = md
        .as_ref()
        .and_then(|v| v.get("verified_against"))
        .and_then(|v| v.as_str());
    let Some(ptr) = ptr else {
        return String::new();
    };
    let verified_at = md
        .as_ref()
        .and_then(|v| v.get("verified_at"))
        .and_then(|v| v.as_str());
    let fresh = matches!((verified_at, last_synth), (Some(v), Some(ls)) if v >= ls);
    if fresh {
        format!(" [confirmed {} vs {}]", verified_at.unwrap_or("?"), ptr)
    } else {
        format!(
            " [VERIFY vs {}; last {}]",
            ptr,
            verified_at.unwrap_or("never")
        )
    }
}

fn format_pack_markdown(
    request: &PackRequest,
    results: &[SearchResult],
    last_synth: Option<&str>,
) -> (String, Vec<String>, Vec<f64>, bool) {
    let mut content = format!("## Retrieved Memory: {}\n", request.title.trim());
    let mut memory_ids = Vec::new();
    let mut scores = Vec::new();
    let mut truncated = false;

    if content.chars().count() > request.max_chars {
        return (
            bounded_char_slice(&content, 0, request.max_chars),
            memory_ids,
            scores,
            true,
        );
    }

    if results.is_empty() {
        let line = "- No matching active memories.\n";
        if !append_with_char_budget(&mut content, line, request.max_chars) {
            truncated = true;
        }
        return (content, memory_ids, scores, truncated);
    }

    let build_line = |result: &SearchResult| -> String {
        let text = result
            .summary
            .as_deref()
            .filter(|summary| !summary.trim().is_empty())
            .unwrap_or(&result.snippet);
        let text = collapse_whitespace(text);
        let tags = if result.tags.is_empty() {
            "-".to_string()
        } else {
            result.tags.join(",")
        };
        let marker = freshness_marker(&result.silo, result.metadata_json.as_deref(), last_synth);
        format!(
            "- [{}:{}] {} (space={}, silo={}, scope={}, tags={}){}\n",
            result.kind,
            result.memory_id,
            text,
            result.space,
            result.silo,
            result.scope,
            tags,
            marker
        )
    };

    for result in results {
        let line = build_line(result);
        if append_with_char_budget(&mut content, &line, request.max_chars) {
            memory_ids.push(result.memory_id.clone());
            scores.push(result.score);
        } else {
            truncated = true;
            break;
        }
    }

    // Budget fallback: the loop injected nothing because the top eligible
    // result's line is by itself larger than the remaining char budget. Rather
    // than drop a confidently-matched memory purely for being long, inject the
    // top result truncated to fit. Mirrors `assemble_reranked_pack`'s budget
    // fallback so the non-rerank pack path (no reranker / rerank failure /
    // FTS-only build) degrades the same way instead of injecting an empty pack.
    // `build_pack_on_connection` already filtered `results` by `min_score`, so
    // every entry here is eligible.
    if memory_ids.is_empty() {
        if let Some(result) = results.first() {
            let line = build_line(result);
            let remaining = request.max_chars.saturating_sub(content.chars().count());
            if let Some(entry) = truncate_pack_line(&line, remaining) {
                content.push_str(&entry);
                memory_ids.push(result.memory_id.clone());
                scores.push(result.score);
                truncated = true;
            }
        }
    }

    if memory_ids.len() < results.len() {
        truncated = true;
    }
    (content, memory_ids, scores, truncated)
}

/// Truncate a single rendered pack line (with its trailing newline) to fit
/// `budget` characters, cutting on a `char` boundary and appending an ellipsis
/// marker + newline. Returns `None` when `budget` cannot hold the marker plus at
/// least one character of the line. Char-count based to match the rest of the
/// `format_pack_markdown` budget path (`append_with_char_budget`).
fn truncate_pack_line(line: &str, budget: usize) -> Option<String> {
    const SUFFIX: &str = "…\n"; // ellipsis marker + newline
    let suffix_len = SUFFIX.chars().count();
    if budget <= suffix_len {
        return None;
    }
    let body = line.strip_suffix('\n').unwrap_or(line);
    let keep = budget - suffix_len;
    let truncated: String = body.chars().take(keep).collect();
    if truncated.is_empty() {
        return None;
    }
    let mut entry = String::with_capacity(truncated.len() + SUFFIX.len());
    entry.push_str(&truncated);
    entry.push_str(SUFFIX);
    Some(entry)
}

fn append_with_char_budget(output: &mut String, text: &str, max_chars: usize) -> bool {
    let current = output.chars().count();
    let additional = text.chars().count();
    if current.saturating_add(additional) <= max_chars {
        output.push_str(text);
        true
    } else {
        false
    }
}

fn collapse_whitespace(value: &str) -> String {
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

fn make_snippet(content: &str, terms: &[String], max_chars: usize) -> String {
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

fn bounded_char_slice(value: &str, start_char: usize, max_chars: usize) -> String {
    value.chars().skip(start_char).take(max_chars).collect()
}

fn is_supported_entity_status(value: &str) -> bool {
    matches!(value, "active" | "merged" | "tombstoned")
}

fn is_supported_relationship_status(value: &str) -> bool {
    matches!(
        value,
        status::ACTIVE | status::SUPERSEDED | status::CONFLICTED | status::TOMBSTONED
    )
}

fn validate_forget_request(request: &ForgetRequest) -> Result<()> {
    if request.id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "memory id must not be empty".to_string(),
        });
    }
    if request.mode != "tombstone" && request.mode != "correct" {
        return Err(Error::InvalidRequest {
            message: "forget mode must be tombstone or correct".to_string(),
        });
    }
    if request.mode != "correct" && request.corrected_by.is_some() {
        return Err(Error::InvalidRequest {
            message: "corrected_by is only valid in correct mode".to_string(),
        });
    }
    if let Some(corrected_by) = &request.corrected_by {
        if corrected_by.trim().is_empty() {
            return Err(Error::InvalidRequest {
                message: "corrected_by must not be empty".to_string(),
            });
        }
        if corrected_by == &request.id {
            return Err(Error::InvalidRequest {
                message: "corrected_by must differ from the corrected memory id".to_string(),
            });
        }
    }
    if let Some(reason) = &request.reason {
        if reason.trim().is_empty() || reason.chars().count() > MAX_FORGET_REASON_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "reason must be non-empty and at most {MAX_FORGET_REASON_CHARS} characters"
                ),
            });
        }
    }
    Ok(())
}

fn validate_history_request(id: &str, options: HistoryOptions) -> Result<()> {
    if id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "memory id must not be empty".to_string(),
        });
    }
    if options.limit == 0 || options.limit > MAX_HISTORY_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("history limit must be between 1 and {MAX_HISTORY_LIMIT}"),
        });
    }
    Ok(())
}

fn forget_memory_tx(
    transaction: &Transaction<'_>,
    request: &ForgetRequest,
) -> Result<ForgetReport> {
    let now = now_timestamp(transaction)?;
    let old_status = transaction
        .query_row(
            "SELECT status FROM memories WHERE id = ?1",
            [&request.id],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "memory",
            id: request.id.clone(),
        })?;

    if old_status == status::TOMBSTONED {
        return Err(Error::Conflict {
            message: format!("memory is already tombstoned: {}", request.id),
        });
    }

    transaction.execute(
        "UPDATE memories SET status = ?1, updated_at = ?2, deleted_at = ?2 WHERE id = ?3",
        params![status::TOMBSTONED, &now, &request.id],
    )?;
    transaction.execute(
        "UPDATE memory_fts SET status = ?1 WHERE memory_id = ?2",
        params![status::TOMBSTONED, &request.id],
    )?;
    transaction.execute(
        "UPDATE memory_fts_public SET status = ?1 WHERE memory_id = ?2",
        params![status::TOMBSTONED, &request.id],
    )?;

    let is_correction = request.mode == "correct";
    let event_type = if is_correction { "correct" } else { "forget" };
    let data_json = if is_correction {
        correction_event_data_json(
            request.dry_run,
            &request.mode,
            request.corrected_by.as_deref(),
            &load_tags(transaction, &request.id)?,
        )
    } else {
        forget_event_data_json(request.dry_run, &request.mode)
    };

    let event_id = next_id("evt");
    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, old_status, new_status, actor, reason, data_json, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, 'memkeeper', ?6, ?7, ?8)",
        params![
            &event_id,
            &request.id,
            event_type,
            &old_status,
            status::TOMBSTONED,
            request.reason.as_deref(),
            data_json,
            &now,
        ],
    )?;

    // For a correction with a named replacement, record the directed
    // `contradicts` edge (replacement -> wrong memory). This is the explicit,
    // intent-captured correction signal the synthesis loop can key off without
    // having to guess correction-vs-evolution from supersession history.
    if let Some(corrected_by) = &request.corrected_by {
        link_memory(transaction, corrected_by, &request.id, "contradicts", &now)?;
    }

    Ok(ForgetReport {
        memory_id: request.id.clone(),
        old_status,
        new_status: status::TOMBSTONED.to_string(),
        event_id,
        dry_run: request.dry_run,
    })
}

fn load_history(
    connection: &Connection,
    id: &str,
    options: HistoryOptions,
) -> Result<HistoryReport> {
    let current_status = connection
        .query_row("SELECT status FROM memories WHERE id = ?1", [id], |row| {
            row.get::<_, String>(0)
        })
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "memory",
            id: id.to_string(),
        })?;
    let total_events = count_for_id(connection, "memory_events", "memory_id", id)?;
    let total_versions = count_for_id(connection, "memory_versions", "memory_id", id)?;
    let mut versions = load_versions_limited(connection, id, options.limit)?;
    if !options.include_source {
        for version in &mut versions {
            version.source_ref_json = None;
        }
    }
    let events = load_events_limited(connection, id, options.limit)?;
    let truncated = total_events > events.len() || total_versions > versions.len();

    Ok(HistoryReport {
        memory_id: id.to_string(),
        current_status,
        events,
        versions,
        truncated,
    })
}

fn validate_remember_request(request: &RememberRequest) -> Result<()> {
    let content = request.content.trim();
    if content.is_empty() {
        return Err(Error::InvalidRequest {
            message: "content must not be empty".to_string(),
        });
    }
    if request.content.chars().count() > MAX_CONTENT_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("content must be at most {MAX_CONTENT_CHARS} characters"),
        });
    }
    if request
        .summary
        .as_deref()
        .is_some_and(|summary| summary.chars().count() > MAX_SUMMARY_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: format!("summary must be at most {MAX_SUMMARY_CHARS} characters"),
        });
    }
    validate_optional_metadata_value("space", request.space.as_deref())?;
    validate_optional_metadata_value("silo", request.silo.as_deref())?;
    validate_optional_metadata_value("scope", request.scope.as_deref())?;
    validate_optional_metadata_value("project", request.project_key.as_deref())?;
    validate_optional_metadata_value("kind", request.kind.as_deref())?;
    validate_optional_metadata_value("entity_key", request.entity_key.as_deref())?;
    validate_optional_metadata_value("claim_key", request.claim_key.as_deref())?;
    validate_optional_metadata_value("source_episode_id", request.source_episode_id.as_deref())?;
    validate_optional_timestamp("observed_at", request.observed_at.as_deref())?;
    validate_optional_timestamp("valid_from", request.valid_from.as_deref())?;
    validate_optional_timestamp("valid_to", request.valid_to.as_deref())?;
    validate_optional_timestamp("expires_at", request.expires_at.as_deref())?;
    validate_memory_link_ids("supersedes", &request.supersedes)?;
    validate_memory_link_ids("contradicts", &request.contradicts)?;
    if !(0.0..=1.0).contains(&request.confidence) {
        return Err(Error::InvalidRequest {
            message: "confidence must be between 0.0 and 1.0".to_string(),
        });
    }
    validate_optional_embedding("embedding", request.embedding.as_deref())?;
    let _ = normalized_tags(&request.tags)?;
    if let Some(source_ref_json) = &request.source_ref_json {
        if source_ref_json.chars().count() > MAX_SOURCE_REF_JSON_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "source JSON must be at most {MAX_SOURCE_REF_JSON_CHARS} characters"
                ),
            });
        }
        if !JsonValidator::is_object(source_ref_json) {
            return Err(Error::InvalidRequest {
                message: "source JSON must be a valid JSON object".to_string(),
            });
        }
        // Provenance/trust keys recognized inside the source object: when
        // present, source_type and sensitivity must use the shared vocabularies
        // (the same ones candidates validate), so explicit-vs-harvested writes
        // and sensitivity stay consistent across the remember and candidate paths.
        validate_source_object_provenance(source_ref_json)?;
    }
    if !REMEMBER_SUPERSEDE_MODES.contains(&request.mode.as_str()) {
        return Err(Error::InvalidRequest {
            message: format!(
                "unsupported mode: {} (expected one of {})",
                request.mode,
                REMEMBER_SUPERSEDE_MODES.join(", ")
            ),
        });
    }
    let scope = request.scope.as_deref().unwrap_or(scope::WORKSPACE);
    if !is_supported_scope(scope) {
        return Err(Error::InvalidRequest {
            message: format!("unsupported scope: {scope}"),
        });
    }
    let inferred_kind = infer_kind_from_prefix(&request.content);
    let kind = request
        .kind
        .as_deref()
        .or(inferred_kind)
        .unwrap_or(kind::FACT);
    if !is_supported_kind(kind) {
        return Err(Error::InvalidRequest {
            message: format!("unsupported kind: {kind}"),
        });
    }
    Ok(())
}

/// Validate the optional `source_type` / `sensitivity` keys inside a source
/// provenance object against the shared candidate vocabularies. Absent keys are
/// fine; only present-but-invalid values error.
fn validate_source_object_provenance(source_ref_json: &str) -> Result<()> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(source_ref_json) else {
        // Shape was already validated as a JSON object above; nothing to do.
        return Ok(());
    };
    let Some(object) = value.as_object() else {
        return Ok(());
    };
    if let Some(source_type) = object.get("source_type").and_then(|v| v.as_str()) {
        if !CANDIDATE_SOURCE_TYPES.contains(&source_type) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported source.source_type: {source_type} (expected one of {})",
                    CANDIDATE_SOURCE_TYPES.join(", ")
                ),
            });
        }
    }
    if let Some(sensitivity) = object.get("sensitivity").and_then(|v| v.as_str()) {
        if !CANDIDATE_SENSITIVITIES.contains(&sensitivity) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported source.sensitivity: {sensitivity} (expected one of {})",
                    CANDIDATE_SENSITIVITIES.join(", ")
                ),
            });
        }
    }
    Ok(())
}

fn validate_optional_metadata_value(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "{name} must be non-empty and at most {MAX_METADATA_VALUE_CHARS} characters"
                ),
            });
        }
    }
    Ok(())
}

fn validate_optional_timestamp(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.chars().count() > MAX_TIMESTAMP_CHARS || !is_utc_rfc3339_like(value) {
            return Err(Error::InvalidRequest {
                message: format!("{name} must be a UTC RFC3339 timestamp ending in Z"),
            });
        }
    }
    Ok(())
}

/// Normalize a validated UTC RFC3339 timestamp to fixed millisecond
/// precision (`YYYY-MM-DDTHH:MM:SS.fffZ`) so lexical string comparison
/// matches temporal order. Shorter fractions pad with zeros; longer
/// fractions truncate (sub-millisecond precision is not preserved).
/// Callers must validate with `is_utc_rfc3339_like` first.
fn normalize_utc_timestamp(value: &str) -> String {
    let body = &value[..19];
    let digits = value[19..value.len() - 1].strip_prefix('.').unwrap_or("");
    let mut millis = String::with_capacity(3);
    for index in 0..3 {
        millis.push(
            digits
                .as_bytes()
                .get(index)
                .copied()
                .map_or('0', char::from),
        );
    }
    format!("{body}.{millis}Z")
}

fn validate_optional_embedding(name: &str, embedding: Option<&[f32]>) -> Result<()> {
    if let Some(embedding) = embedding {
        let dims = embedding.len();
        if !(1..=MAX_SEMANTIC_EMBEDDING_DIMS).contains(&dims) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "{name} dimension {dims} is not supported (expected 1..={MAX_SEMANTIC_EMBEDDING_DIMS})"
                ),
            });
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidRequest {
                message: format!("{name} must contain only finite floats"),
            });
        }
    }
    Ok(())
}

fn is_utc_rfc3339_like(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 20 || !value.ends_with('Z') {
        return false;
    }
    if !matches!(bytes.get(4), Some(b'-'))
        || !matches!(bytes.get(7), Some(b'-'))
        || !matches!(bytes.get(10), Some(b'T'))
        || !matches!(bytes.get(13), Some(b':'))
        || !matches!(bytes.get(16), Some(b':'))
        || !timestamp_parts_are_valid(bytes, 0, 5, 8, 11, 14, 17)
    {
        return false;
    }
    match bytes.len() {
        20 => true,
        len if len > 21 && bytes[19] == b'.' => bytes[20..len - 1].iter().all(u8::is_ascii_digit),
        _ => false,
    }
}

fn timestamp_parts_are_valid(
    bytes: &[u8],
    year_start: usize,
    month_start: usize,
    day_start: usize,
    hour_start: usize,
    minute_start: usize,
    second_start: usize,
) -> bool {
    let Some(year) = parse_ascii_digits(bytes, year_start, 4) else {
        return false;
    };
    let Some(month) = parse_ascii_digits(bytes, month_start, 2) else {
        return false;
    };
    let Some(day) = parse_ascii_digits(bytes, day_start, 2) else {
        return false;
    };
    let Some(hour) = parse_ascii_digits(bytes, hour_start, 2) else {
        return false;
    };
    let Some(minute) = parse_ascii_digits(bytes, minute_start, 2) else {
        return false;
    };
    let Some(second) = parse_ascii_digits(bytes, second_start, 2) else {
        return false;
    };

    (1..=12).contains(&month)
        && day >= 1
        && day <= days_in_month(year, month)
        && hour <= 23
        && minute <= 59
        && second <= 59
}

fn parse_ascii_digits(bytes: &[u8], start: usize, len: usize) -> Option<u32> {
    let end = start.checked_add(len)?;
    let slice = bytes.get(start..end)?;
    let mut value = 0_u32;
    for byte in slice {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value * 10 + u32::from(byte - b'0');
    }
    Some(value)
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: u32) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}

fn validate_memory_link_ids(name: &str, values: &[String]) -> Result<()> {
    if values.len() > MAX_MEMORY_LINKS {
        return Err(Error::InvalidRequest {
            message: format!("{name} must contain at most {MAX_MEMORY_LINKS} memory ids"),
        });
    }
    let mut seen = BTreeSet::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "{name} ids must be non-empty and at most {MAX_METADATA_VALUE_CHARS} characters"
                ),
            });
        }
        if !seen.insert(trimmed) {
            return Err(Error::InvalidRequest {
                message: format!("{name} contains duplicate memory id: {trimmed}"),
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn remember_memory_tx(
    transaction: &Transaction<'_>,
    request: &RememberRequest,
) -> Result<RememberReport> {
    let now = now_timestamp(transaction)?;
    let space = request.space.as_deref().unwrap_or(DEFAULT_SPACE);
    let silo = match request.silo.as_deref() {
        Some(value) => value.to_string(),
        None => {
            default_silo(transaction, space)?.unwrap_or_else(|| DEFAULT_DURABLE_SILO.to_string())
        }
    };
    ensure_silo_exists(transaction, space, &silo)?;

    let source_episode_id = match request.source_episode_id.as_deref() {
        Some(id) => {
            ensure_source_episode_exists(transaction, space, id)?;
            Some(id.to_string())
        }
        None => None,
    };

    let scope = request
        .scope
        .clone()
        .unwrap_or_else(|| scope::WORKSPACE.to_string());
    let memory_kind = request
        .kind
        .clone()
        .or_else(|| infer_kind_from_prefix(&request.content).map(str::to_string))
        .unwrap_or_else(|| kind::FACT.to_string());
    // Caller-supplied timestamps normalize to fixed millisecond precision so
    // lexical ordering (auto-supersede, freshness, ORDER BY) stays temporal.
    let observed_at = request
        .observed_at
        .as_deref()
        .map_or_else(|| now.clone(), normalize_utc_timestamp);
    let valid_from = request.valid_from.as_deref().map(normalize_utc_timestamp);
    let valid_to = request.valid_to.as_deref().map(normalize_utc_timestamp);
    let expires_at = request.expires_at.as_deref().map(normalize_utc_timestamp);
    let memory_id = next_id("mem");
    let version_id = next_id("ver");
    let event_id = next_id("evt");
    let content_sha256 = sha256_hex(request.content.as_bytes());
    let pinned = i64::from(request.pinned);
    let tags = normalized_tags(&request.tags)?;
    let tags_text = tags.join(" ");
    let metadata_text = memory_fts_metadata_text(
        request.project_key.as_deref(),
        request.entity_key.as_deref(),
        request.claim_key.as_deref(),
        &request.content,
        request.summary.as_deref(),
        &tags_text,
    );
    let excluded_ids = request
        .supersedes
        .iter()
        .chain(request.contradicts.iter())
        .cloned()
        .collect::<BTreeSet<_>>();
    let candidate_detection = RememberCandidateDetection {
        space,
        silo: &silo,
        kind: &memory_kind,
        content_sha256: &content_sha256,
        entity_key: request.entity_key.as_deref(),
        claim_key: request.claim_key.as_deref(),
        request_terms: lexical_terms(&request.content),
        excluded_ids: &excluded_ids,
    };
    let (candidates, candidates_truncated) =
        detect_remember_candidates(transaction, &candidate_detection)?;
    let same_claim_candidates = same_claim_candidates(transaction, space, request, &excluded_ids)?;
    let non_pinned_same_claim: Vec<String> = same_claim_candidates
        .iter()
        .filter(|candidate| !candidate.pinned)
        .map(|candidate| candidate.memory_id.clone())
        .collect();
    // Supersession mode governs how this write resolves against active memories
    // sharing its entity/claim key. `auto` is the historical policy; the others
    // were added so callers can declare intent explicitly.
    //   auto      -> older same-key memories of eligible kinds (current default)
    //   append    -> coexist; supersede nothing
    //   supersede -> force-retire all non-pinned same-key actives, any kind
    //   suggest   -> mutate nothing; return the would-be set for review
    //   conflict  -> mutate nothing; open a conflict row per same-key active
    let (auto_superseded, supersede_suggestions, open_conflicts) = match request.mode.as_str() {
        "append" => (Vec::new(), Vec::new(), false),
        "supersede" => (non_pinned_same_claim.clone(), Vec::new(), false),
        "suggest" => (Vec::new(), non_pinned_same_claim.clone(), false),
        "conflict" => (Vec::new(), Vec::new(), true),
        // `auto` (and the validated default) keep the historical behavior.
        _ => (
            auto_supersede_candidates(&memory_kind, &observed_at, &same_claim_candidates),
            Vec::new(),
            false,
        ),
    };
    let conflict_candidates = if memory_kind == kind::CONTINUITY || open_conflicts {
        same_claim_candidates
            .iter()
            .map(same_claim_conflict_candidate)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    transaction.execute(
        "INSERT INTO memories (
            id, space_name, silo_name, scope, project_key, kind, entity_key, claim_key,
            status, active_version_id, confidence, pinned, source_episode_id, valid_from,
            valid_to, observed_at, created_at, updated_at, expires_at, metadata_json
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)",
        params![
            &memory_id,
            space,
            &silo,
            &scope,
            request.project_key.as_deref(),
            &memory_kind,
            request.entity_key.as_deref(),
            request.claim_key.as_deref(),
            status::ACTIVE,
            &version_id,
            request.confidence,
            pinned,
            source_episode_id.as_deref(),
            valid_from.as_deref(),
            valid_to.as_deref(),
            &observed_at,
            &now,
            &now,
            expires_at.as_deref(),
            request.metadata_json.as_deref(),
        ],
    )?;

    transaction.execute(
        "INSERT INTO memory_versions (
            id, memory_id, version_num, content, summary, content_sha256,
            source_episode_id, source_ref_json, created_at, created_by, event_id
         ) VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            &version_id,
            &memory_id,
            &request.content,
            request.summary.as_deref(),
            &content_sha256,
            source_episode_id.as_deref(),
            request.source_ref_json.as_deref(),
            &now,
            "memkeeper",
            &event_id,
        ],
    )?;

    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, new_status, actor, data_json, created_at)
         VALUES (?1, ?2, 'remember', ?3, 'memkeeper', ?4, ?5)",
        params![
            &event_id,
            &memory_id,
            status::ACTIVE,
            event_data_json(
                request.dry_run,
                &request.supersedes,
                &request.contradicts,
                &auto_superseded,
                &conflict_candidates,
            ),
            &now,
        ],
    )?;

    for tag in &tags {
        transaction.execute(
            "INSERT INTO memory_tags (memory_id, tag, created_at) VALUES (?1, ?2, ?3)",
            params![&memory_id, tag, &now],
        )?;
    }

    for superseded_id in &request.supersedes {
        supersede_memory(transaction, space, &memory_id, superseded_id, &now)?;
    }
    for superseded_id in &auto_superseded {
        supersede_memory(transaction, space, &memory_id, superseded_id, &now)?;
    }
    for contradicted_id in &request.contradicts {
        ensure_memory_in_space(transaction, contradicted_id, space)?;
        link_memory(
            transaction,
            &memory_id,
            contradicted_id,
            "contradicts",
            &now,
        )?;
        link_memory(
            transaction,
            contradicted_id,
            &memory_id,
            "contradicts",
            &now,
        )?;
    }
    if open_conflicts {
        for candidate in &same_claim_candidates {
            open_conflict(transaction, space, &memory_id, &candidate.memory_id, &now)?;
        }
    }

    transaction.execute(
        "INSERT INTO memory_fts (
            memory_id, version_id, space_name, silo_name, status, kind, content, summary,
            tags, source_text, metadata_text
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            &memory_id,
            &version_id,
            space,
            &silo,
            status::ACTIVE,
            &memory_kind,
            &request.content,
            request.summary.as_deref(),
            &tags_text,
            request.source_ref_json.as_deref(),
            &metadata_text,
        ],
    )?;
    transaction.execute(
        "INSERT INTO memory_fts_public (
            memory_id, version_id, space_name, silo_name, status, kind, content, summary,
            tags, metadata_text
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            &memory_id,
            &version_id,
            space,
            &silo,
            status::ACTIVE,
            &memory_kind,
            &request.content,
            request.summary.as_deref(),
            &tags_text,
            &metadata_text,
        ],
    )?;

    insert_memory_embedding(
        transaction,
        &memory_id,
        &version_id,
        &now,
        request.embedding.as_deref(),
        request.embedding_model_id.as_deref(),
    )?;

    if let (Some(token_vecs), Some(token_model)) = (
        request.token_embedding.as_ref(),
        request.token_embedding_model_id.as_deref(),
    ) {
        enforce_active_colbert_model(transaction, token_model, &now)?;
        upsert_memory_token_embedding(transaction, &memory_id, token_model, token_vecs)?;
    }

    if let Some(entity_key) = request.entity_key.as_deref() {
        upsert_memory_entity_projection(
            transaction,
            space,
            entity_key,
            source_episode_id.as_deref(),
            &now,
        )?;
    }

    let mut memory = load_memory(
        transaction,
        &memory_id,
        GetOptions {
            include_history: false,
            include_links: true,
            include_source: false,
        },
    )?;
    if !request.dry_run {
        memory.versions = None;
        memory.events = None;
    }

    Ok(RememberReport {
        memory,
        event_id,
        processing_status: if request.dry_run {
            "dry_run"
        } else {
            "indexed"
        }
        .to_string(),
        candidates,
        candidates_truncated,
        auto_superseded,
        conflict_candidates,
        supersede_suggestions,
        dry_run: request.dry_run,
    })
}

/// Pack per-token vectors into one little-endian f32 blob, `[n_tokens * dims]`.
fn token_vecs_to_blob(vecs: &[Vec<f32>]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(vecs.iter().map(Vec::len).sum::<usize>() * 4);
    for vector in vecs {
        for value in vector {
            blob.extend_from_slice(&value.to_le_bytes());
        }
    }
    blob
}

fn blob_to_token_vecs(blob: &[u8], dims: usize, n_tokens: usize) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(n_tokens);
    for token in 0..n_tokens {
        let base = token * dims * 4;
        out.push(
            (0..dims)
                .map(|d| {
                    let offset = base + d * 4;
                    f32::from_le_bytes([
                        blob[offset],
                        blob[offset + 1],
                        blob[offset + 2],
                        blob[offset + 3],
                    ])
                })
                .collect(),
        );
    }
    out
}

/// Insert or replace the late-interaction token-embedding row for a memory.
///
/// # Errors
///
/// Returns an error on `SQLite` failure or empty/ragged input.
pub(crate) fn upsert_memory_token_embedding(
    connection: &Connection,
    memory_id: &str,
    model_id: &str,
    vecs: &[Vec<f32>],
) -> Result<()> {
    let n_tokens = vecs.len();
    if n_tokens == 0 {
        return Err(Error::InvalidRequest {
            message: "empty token embedding".to_string(),
        });
    }
    let dims = vecs[0].len();
    if !vecs.iter().all(|vector| vector.len() == dims) {
        return Err(Error::InvalidRequest {
            message: "ragged token embedding".to_string(),
        });
    }
    connection.execute(
        "INSERT OR REPLACE INTO memory_token_embeddings \
         (memory_id, embedding_model, dims, n_tokens, vector_blob, created_at) \
         VALUES (?1, ?2, ?3, ?4, ?5, CURRENT_TIMESTAMP)",
        rusqlite::params![
            memory_id,
            model_id,
            i64::try_from(dims).map_err(|_| Error::InvalidRequest {
                message: "token dims overflow".to_string(),
            })?,
            i64::try_from(n_tokens).map_err(|_| Error::InvalidRequest {
                message: "token count overflow".to_string(),
            })?,
            token_vecs_to_blob(vecs)
        ],
    )?;
    Ok(())
}

/// Load all token embeddings for ACTIVE memories under the given model.
///
/// # Errors
///
/// Returns an error on `SQLite` failure or malformed blobs.
pub(crate) fn load_token_embeddings(
    connection: &Connection,
    model_id: &str,
) -> Result<Vec<(String, Vec<Vec<f32>>)>> {
    let mut statement = connection.prepare_cached(
        "SELECT t.memory_id, t.dims, t.n_tokens, t.vector_blob \
         FROM memory_token_embeddings t JOIN memories m ON m.id = t.memory_id \
         WHERE t.embedding_model = ?1 AND m.status = 'active'",
    )?;
    let rows = statement.query_map([model_id], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, Vec<u8>>(3)?,
        ))
    })?;
    let mut out = Vec::new();
    for row in rows {
        let (id, dims, n_tokens, blob) = row?;
        let (Ok(dims), Ok(n_tokens)) = (usize::try_from(dims), usize::try_from(n_tokens)) else {
            return Err(Error::InvalidRequest {
                message: format!("negative token dims for {id}"),
            });
        };
        if blob.len() != dims * n_tokens * 4 {
            return Err(Error::InvalidRequest {
                message: format!("token blob size mismatch for {id}"),
            });
        }
        out.push((id, blob_to_token_vecs(&blob, dims, n_tokens)));
    }
    Ok(out)
}

#[cfg(feature = "semantic")]
fn insert_memory_embedding(
    transaction: &Transaction<'_>,
    memory_id: &str,
    version_id: &str,
    now: &str,
    embedding: Option<&[f32]>,
    model_id: Option<&str>,
) -> Result<()> {
    let Some(embedding) = embedding else {
        return Ok(());
    };
    let dims = embedding.len();
    let model = model_id.unwrap_or("unknown");
    enforce_active_embedding_model(transaction, model, dims, now)?;
    write_embedding_row(
        transaction,
        memory_id,
        version_id,
        model,
        dims,
        embedding,
        now,
    )
}

#[cfg(feature = "semantic")]
fn write_embedding_row(
    transaction: &Transaction<'_>,
    memory_id: &str,
    version_id: &str,
    model: &str,
    dims: usize,
    embedding: &[f32],
    now: &str,
) -> Result<()> {
    let table = semantic_table_for_dims(dims)?;
    ensure_semantic_table(transaction, &table, dims)?;
    // Canonical vector sidecar (source of truth for export/import).
    transaction.execute(
        "INSERT INTO embeddings (id, memory_id, version_id, embedding_model, dimensions, vector_blob, status, created_at, updated_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'ready', ?7, ?7)",
        params![
            next_id("emb"),
            memory_id,
            version_id,
            model,
            i64::try_from(dims).unwrap_or(i64::MAX),
            embedding_to_blob(embedding),
            now,
        ],
    )?;
    // Rebuildable ANN projection.
    let embedding_json = embedding_json(embedding)?;
    transaction.execute(
        &format!("INSERT INTO {table} (memory_id, embedding) VALUES (?1, ?2)"),
        params![memory_id, embedding_json],
    )?;
    Ok(())
}

#[cfg(feature = "semantic")]
fn ensure_semantic_table(connection: &Connection, table: &str, dims: usize) -> Result<()> {
    if !table_exists(connection, table)? {
        connection.execute_batch(&semantic_table_ddl(table, dims))?;
    }
    Ok(())
}

#[cfg(feature = "semantic")]
fn semantic_table_ddl(table: &str, dims: usize) -> String {
    format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(\n  memory_id TEXT NOT NULL,\n  embedding FLOAT[{dims}]\n);"
    )
}

#[cfg(feature = "semantic")]
fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

#[cfg(feature = "semantic")]
fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn read_config_value(connection: &Connection, key: &str) -> Result<Option<String>> {
    let mut statement = connection.prepare("SELECT value FROM config_kv WHERE key = ?1")?;
    let mut rows = statement.query(params![key])?;
    match rows.next()? {
        Some(row) => Ok(Some(row.get(0)?)),
        None => Ok(None),
    }
}

fn set_config_value(connection: &Connection, key: &str, value: &str, now: &str) -> Result<()> {
    connection.execute(
        "INSERT INTO config_kv (key, value, updated_at) VALUES (?1, ?2, ?3)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
        params![key, value, now],
    )?;
    Ok(())
}

#[cfg(feature = "semantic")]
fn enforce_active_embedding_model(
    transaction: &Transaction<'_>,
    model: &str,
    dims: usize,
    now: &str,
) -> Result<()> {
    let active_model = read_config_value(transaction, "active_embedding_model")?;
    let active_dims = read_config_value(transaction, "active_embedding_dims")?;
    if let (Some(active_model), Some(active_dims)) = (active_model, active_dims) {
        if active_model != model || active_dims != dims.to_string() {
            return Err(Error::InvalidRequest {
                message: format!(
                    "store embeddings use model '{active_model}' ({active_dims} dims); cannot mix with '{model}' ({dims} dims) — run `memkeeper reindex` to switch"
                ),
            });
        }
    } else {
        set_config_value(transaction, "active_embedding_model", model, now)?;
        set_config_value(transaction, "active_embedding_dims", &dims.to_string(), now)?;
    }
    Ok(())
}

/// Reject mixing token-embedding models within one store (mirrors
/// `enforce_active_embedding_model`; key `active_colbert_model`).
fn enforce_active_colbert_model(
    transaction: &Transaction<'_>,
    model: &str,
    now: &str,
) -> Result<()> {
    let active_model = read_config_value(transaction, "active_colbert_model")?;
    if let Some(active_model) = active_model {
        if active_model != model {
            return Err(Error::InvalidRequest {
                message: format!(
                    "store token embeddings use model '{active_model}'; cannot mix with '{model}' — run `memkeeper reindex --tokens --force` to switch"
                ),
            });
        }
    } else {
        set_config_value(transaction, "active_colbert_model", model, now)?;
    }
    Ok(())
}

#[cfg(feature = "semantic")]
fn rebuild_vector_index(connection: &Connection) -> Result<usize> {
    let collected: Vec<(String, Vec<u8>, i64)> = {
        let mut statement = connection.prepare(
            "SELECT memory_id, vector_blob, dimensions FROM embeddings \
             WHERE vector_blob IS NOT NULL AND dimensions IS NOT NULL AND status = 'ready'",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Vec<u8>>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let mut count = 0usize;
    for (memory_id, blob, dims_i64) in collected {
        let Ok(dims) = usize::try_from(dims_i64) else {
            continue;
        };
        let embedding = blob_to_embedding(&blob);
        if dims == 0 || embedding.len() != dims {
            continue;
        }
        let table = semantic_table_for_dims(dims)?;
        ensure_semantic_table(connection, &table, dims)?;
        connection.execute(
            &format!("DELETE FROM {table} WHERE memory_id = ?1"),
            params![memory_id],
        )?;
        let embedding_json = embedding_json(&embedding)?;
        connection.execute(
            &format!("INSERT INTO {table} (memory_id, embedding) VALUES (?1, ?2)"),
            params![memory_id, embedding_json],
        )?;
        count += 1;
    }
    Ok(count)
}

/// Rebuild the semantic ANN index from the canonical `embeddings` table.
///
/// Re-projects stored vectors into the `memory_vec_<dims>` index without
/// re-running the embedding model. Useful after a logical import or to recover
/// the vector index. Returns the number of vectors reindexed.
///
/// # Errors
///
/// Returns an error if the store is missing/uninitialized or `SQLite` rejects a
/// statement.
#[cfg(feature = "semantic")]
pub fn reindex_vectors(path: impl AsRef<Path>) -> Result<usize> {
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let count = rebuild_vector_index(&transaction)?;
    transaction.commit()?;
    Ok(count)
}

#[cfg(feature = "semantic")]
fn drop_all_vector_tables(transaction: &Transaction<'_>) -> Result<()> {
    let tables: Vec<String> = {
        let mut statement = transaction.prepare(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'memory_vec_%'",
        )?;
        let rows = statement.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    for table in tables {
        // Table name comes from sqlite_master and is therefore trusted.
        transaction.execute(&format!("DROP TABLE IF EXISTS {table}"), [])?;
    }
    Ok(())
}

/// One active memory's content, to be re-embedded under a new model.
#[cfg(feature = "semantic")]
pub struct ReembedTarget {
    /// Memory id.
    pub memory_id: String,
    /// Active version id.
    pub version_id: String,
    /// Active version content to embed.
    pub content: String,
}

/// Collect every active memory's content for a model-switching re-embed.
///
/// # Errors
///
/// Returns an error if the store is missing/uninitialized or `SQLite` rejects a
/// statement.
#[cfg(feature = "semantic")]
pub fn collect_reembed_targets(path: impl AsRef<Path>) -> Result<Vec<ReembedTarget>> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    let mut statement = connection.prepare(
        "SELECT m.id, m.active_version_id, v.content
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.status = 'active'
         ORDER BY m.id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(ReembedTarget {
            memory_id: row.get(0)?,
            version_id: row.get(1)?,
            content: row.get(2)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

/// Collect active memories needing late-interaction token embeddings.
///
/// Text is `summary + "\n\n" + content` (the eval-validated doc composition).
/// With `force`, every active memory is returned (model switch / re-embed);
/// otherwise only memories without a token row.
///
/// # Errors
///
/// Returns an error if the store is missing/uninitialized or `SQLite` rejects a
/// statement.
pub fn collect_token_backfill_targets(
    path: impl AsRef<Path>,
    force: bool,
) -> Result<Vec<(String, String)>> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    let sql = if force {
        "SELECT m.id, COALESCE(v.summary, ''), v.content FROM memories m \
         JOIN memory_versions v ON v.id = m.active_version_id \
         WHERE m.status = 'active' ORDER BY m.id"
    } else {
        "SELECT m.id, COALESCE(v.summary, ''), v.content FROM memories m \
         JOIN memory_versions v ON v.id = m.active_version_id \
         LEFT JOIN memory_token_embeddings t ON t.memory_id = m.id \
         WHERE m.status = 'active' AND t.memory_id IS NULL ORDER BY m.id"
    };
    let mut statement = connection.prepare(sql)?;
    let rows = statement.query_map([], |row| {
        let id: String = row.get(0)?;
        let summary: String = row.get(1)?;
        let content: String = row.get(2)?;
        let text = if summary.is_empty() {
            content
        } else {
            format!("{summary}\n\n{content}")
        };
        Ok((id, text))
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

/// Write a batch of late-interaction token embeddings (backfill / model switch).
///
/// With `reset`, clears existing token rows and the active-model record first.
/// Returns the number of rows written.
///
/// # Errors
///
/// Returns an error if the store is missing or `SQLite` rejects a statement.
pub fn apply_token_embeddings(
    path: impl AsRef<Path>,
    model_id: &str,
    rows: &[(String, Vec<Vec<f32>>)],
    reset: bool,
) -> Result<usize> {
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    if reset {
        transaction.execute("DELETE FROM memory_token_embeddings", [])?;
        transaction.execute(
            "DELETE FROM config_kv WHERE key = 'active_colbert_model'",
            [],
        )?;
    }
    let now = now_timestamp(&transaction)?;
    enforce_active_colbert_model(&transaction, model_id, &now)?;
    let mut written = 0usize;
    for (memory_id, vecs) in rows {
        if vecs.is_empty() {
            continue;
        }
        upsert_memory_token_embedding(&transaction, memory_id, model_id, vecs)?;
        written += 1;
    }
    transaction.commit()?;
    Ok(written)
}

/// Replace every stored embedding with vectors produced by a new model.
///
/// Clears the canonical embeddings table and all ANN index tables, resets the
/// active embedding model record, then writes the supplied vectors. Returns the
/// number of vectors written.
///
/// # Errors
///
/// Returns an error if a vector length does not match `dims`, the store is
/// missing, or `SQLite` rejects a statement.
#[cfg(feature = "semantic")]
pub fn apply_reembed(
    path: impl AsRef<Path>,
    model_id: &str,
    dims: usize,
    vectors: &[(String, String, Vec<f32>)],
) -> Result<usize> {
    semantic_table_for_dims(dims)?;
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;
    let now = now_timestamp(&transaction)?;
    transaction.execute("DELETE FROM embeddings", [])?;
    drop_all_vector_tables(&transaction)?;
    set_config_value(&transaction, "active_embedding_model", model_id, &now)?;
    set_config_value(
        &transaction,
        "active_embedding_dims",
        &dims.to_string(),
        &now,
    )?;
    let mut count = 0usize;
    for (memory_id, version_id, embedding) in vectors {
        if embedding.len() != dims {
            return Err(Error::InvalidRequest {
                message: format!(
                    "re-embed produced dimension {} but {dims} was expected",
                    embedding.len()
                ),
            });
        }
        write_embedding_row(
            &transaction,
            memory_id,
            version_id,
            model_id,
            dims,
            embedding,
            &now,
        )?;
        count += 1;
    }
    transaction.commit()?;
    Ok(count)
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

fn upsert_memory_entity_projection(
    transaction: &Transaction<'_>,
    space: &str,
    entity_key: &str,
    source_episode_id: Option<&str>,
    now: &str,
) -> Result<()> {
    let entity_id = next_id("ent");
    let canonical_name = canonical_name_from_entity_key(entity_key);
    transaction.execute(
        "INSERT INTO entities (
            id, space_name, entity_key, entity_type, canonical_name, status, confidence,
            source_episode_id, metadata_json, created_at, updated_at
         ) VALUES (?1, ?2, ?3, 'MemorySubject', ?4, 'active', 1.0, ?5, NULL, ?6, ?6)
         ON CONFLICT(space_name, entity_key) DO UPDATE SET
            canonical_name = excluded.canonical_name,
            source_episode_id = COALESCE(entities.source_episode_id, excluded.source_episode_id),
            updated_at = excluded.updated_at",
        params![
            &entity_id,
            space,
            entity_key,
            &canonical_name,
            source_episode_id,
            now,
        ],
    )?;
    Ok(())
}

fn canonical_name_from_entity_key(entity_key: &str) -> String {
    let readable = entity_key
        .rsplit([':', '/', '#'])
        .next()
        .unwrap_or(entity_key)
        .chars()
        .map(|character| match character {
            '_' | '-' | '.' => ' ',
            other => other,
        })
        .collect::<String>();
    let collapsed = collapse_whitespace(&readable);
    if collapsed.is_empty() {
        entity_key.to_string()
    } else {
        collapsed
    }
}

fn normalized_alias(value: &str) -> String {
    collapse_whitespace(&value.to_ascii_lowercase())
}

fn detect_remember_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
) -> Result<(Vec<RememberCandidate>, bool)> {
    let mut candidates = BTreeMap::<String, RememberCandidateAccumulator>::new();
    collect_exact_content_candidates(connection, request, &mut candidates)?;
    collect_claim_key_candidates(connection, request, &mut candidates)?;
    collect_entity_key_candidates(connection, request, &mut candidates)?;
    collect_lexical_candidates(connection, request, &mut candidates)?;

    let mut output = candidates
        .into_values()
        .map(RememberCandidateAccumulator::into_candidate)
        .collect::<Vec<_>>();
    output.sort_by(compare_remember_candidates);
    let truncated = output.len() > MAX_REMEMBER_CANDIDATES;
    if truncated {
        output.truncate(MAX_REMEMBER_CANDIDATES);
    }
    Ok((output, truncated))
}

fn same_claim_candidates(
    connection: &Connection,
    space: &str,
    request: &RememberRequest,
    excluded_ids: &BTreeSet<String>,
) -> Result<Vec<SameClaimCandidate>> {
    if !request.supersedes.is_empty() || !request.contradicts.is_empty() {
        return Ok(Vec::new());
    }
    let (Some(entity_key), Some(claim_key)) =
        (request.entity_key.as_deref(), request.claim_key.as_deref())
    else {
        return Ok(Vec::new());
    };

    let mut statement = connection.prepare_cached(
        "SELECT m.id, m.kind, m.observed_at, v.content, m.pinned
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.space_name = ?1 AND m.entity_key = ?2 AND m.claim_key = ?3 AND m.status = 'active'
         ORDER BY m.observed_at DESC, m.updated_at DESC, m.id ASC
         LIMIT ?4",
    )?;
    let rows = statement.query_map(
        params![
            space,
            entity_key,
            claim_key,
            limit_i64(MAX_REMEMBER_CONFLICT_CANDIDATES.saturating_add(1))?,
        ],
        |row| {
            Ok(SameClaimCandidate {
                memory_id: row.get(0)?,
                kind: row.get(1)?,
                observed_at: row.get(2)?,
                content: row.get(3)?,
                pinned: row.get::<_, i64>(4)? == 1,
            })
        },
    )?;
    Ok(collect_rows(rows)?
        .into_iter()
        .filter(|candidate| !excluded_ids.contains(&candidate.memory_id))
        .take(MAX_REMEMBER_CONFLICT_CANDIDATES)
        .collect())
}

fn same_claim_conflict_candidate(candidate: &SameClaimCandidate) -> RememberConflictCandidate {
    RememberConflictCandidate {
        memory_id: candidate.memory_id.clone(),
        kind: candidate.kind.clone(),
        observed_at: candidate.observed_at.clone(),
        snippet: bounded_char_slice(&candidate.content, 0, MAX_SNIPPET_CHARS.min(240)),
    }
}

fn auto_supersede_candidates(
    incoming_kind: &str,
    observed_at: &str,
    candidates: &[SameClaimCandidate],
) -> Vec<String> {
    if incoming_kind == kind::CONTINUITY || !is_auto_supersede_kind(incoming_kind) {
        return Vec::new();
    }
    candidates
        .iter()
        .filter(|candidate| !candidate.pinned && candidate.observed_at.as_str() < observed_at)
        .map(|candidate| candidate.memory_id.clone())
        .collect()
}

fn is_auto_supersede_kind(value: &str) -> bool {
    matches!(
        value,
        kind::FACT | kind::PREFERENCE | kind::DECISION | kind::LESSON
    )
}

fn collect_exact_content_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
) -> Result<()> {
    let mut statement = connection.prepare_cached(
        "SELECT m.id, m.space_name, m.silo_name, m.kind, m.status, m.entity_key, m.claim_key,
                v.content, v.summary, v.content_sha256
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.space_name = ?1 AND m.silo_name = ?2 AND m.status = 'active' AND v.content_sha256 = ?3
         ORDER BY m.id ASC
         LIMIT ?4",
    )?;
    let rows = statement.query_map(
        params![
            request.space,
            request.silo,
            request.content_sha256,
            limit_i64(MAX_REMEMBER_CANDIDATES.saturating_add(1))?,
        ],
        remember_candidate_row_from_row,
    )?;
    for row in collect_rows(rows)? {
        merge_remember_candidate(candidates, request, row, "duplicate", 1.0, "content_sha256");
    }
    Ok(())
}

fn collect_claim_key_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
) -> Result<()> {
    let Some(claim_key) = request.claim_key else {
        return Ok(());
    };
    let mut statement = connection.prepare_cached(
        "SELECT m.id, m.space_name, m.silo_name, m.kind, m.status, m.entity_key, m.claim_key,
                v.content, v.summary, v.content_sha256
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.space_name = ?1 AND m.silo_name = ?2 AND m.status = 'active' AND m.claim_key = ?3
         ORDER BY m.observed_at DESC, m.updated_at DESC, m.id ASC
         LIMIT ?4",
    )?;
    let rows = statement.query_map(
        params![
            request.space,
            request.silo,
            claim_key,
            limit_i64(MAX_REMEMBER_CANDIDATES.saturating_add(1))?,
        ],
        remember_candidate_row_from_row,
    )?;
    for row in collect_rows(rows)? {
        merge_remember_candidate(
            candidates,
            request,
            row,
            "update_candidate",
            0.95,
            "claim_key",
        );
    }
    Ok(())
}

fn collect_entity_key_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
) -> Result<()> {
    let Some(entity_key) = request.entity_key else {
        return Ok(());
    };
    let mut statement = connection.prepare_cached(
        "SELECT m.id, m.space_name, m.silo_name, m.kind, m.status, m.entity_key, m.claim_key,
                v.content, v.summary, v.content_sha256
         FROM memories m
         JOIN memory_versions v ON v.id = m.active_version_id
         WHERE m.space_name = ?1 AND m.silo_name = ?2 AND m.status = 'active' AND m.entity_key = ?3 AND m.kind = ?4
         ORDER BY m.observed_at DESC, m.updated_at DESC, m.id ASC
         LIMIT ?5",
    )?;
    let rows = statement.query_map(
        params![
            request.space,
            request.silo,
            entity_key,
            request.kind,
            limit_i64(MAX_REMEMBER_CANDIDATES.saturating_add(1))?,
        ],
        remember_candidate_row_from_row,
    )?;
    for row in collect_rows(rows)? {
        merge_remember_candidate(
            candidates,
            request,
            row,
            "update_candidate",
            0.75,
            "entity_key_kind",
        );
    }
    Ok(())
}

fn collect_lexical_candidates(
    connection: &Connection,
    request: &RememberCandidateDetection<'_>,
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
) -> Result<()> {
    if request.request_terms.len() < 3 {
        return Ok(());
    }
    let fts_query = remember_candidate_fts_query(&request.request_terms);
    if fts_query.is_empty() {
        return Ok(());
    }
    let sql = format!(
        "SELECT m.id, m.space_name, m.silo_name, m.kind, m.status, m.entity_key, m.claim_key,
                v.content, v.summary, v.content_sha256
         FROM memory_fts_public
         JOIN memories m ON m.id = memory_fts_public.memory_id
         JOIN memory_versions v ON v.id = memory_fts_public.version_id
         WHERE memory_fts_public MATCH ?1 AND m.space_name = ?2 AND m.silo_name = ?3 AND m.status = 'active'
         ORDER BY bm25(memory_fts_public), m.observed_at DESC, m.id ASC
         LIMIT {MAX_REMEMBER_LEXICAL_SCAN}"
    );
    let mut statement = connection.prepare_cached(&sql)?;
    let rows = statement.query_map(
        params![&fts_query, request.space, request.silo],
        remember_candidate_row_from_row,
    )?;
    for row in collect_rows(rows)? {
        let row_terms = lexical_terms(&row.content);
        let similarity = jaccard_similarity(&request.request_terms, &row_terms);
        if similarity >= REMEMBER_LEXICAL_THRESHOLD {
            merge_remember_candidate(
                candidates,
                request,
                row,
                "related_candidate",
                similarity.min(0.94),
                "lexical_similarity",
            );
        }
    }
    Ok(())
}

fn remember_candidate_fts_query(terms: &BTreeSet<String>) -> String {
    terms
        .iter()
        .filter(|term| term.chars().count() >= 4)
        .take(MAX_REMEMBER_LEXICAL_TERMS)
        .map(|term| format!("{{content summary tags metadata_text}} : {term}"))
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn remember_candidate_row_from_row(row: &Row<'_>) -> rusqlite::Result<RememberCandidateRow> {
    Ok(RememberCandidateRow {
        memory_id: row.get(0)?,
        space: row.get(1)?,
        silo: row.get(2)?,
        kind: row.get(3)?,
        status: row.get(4)?,
        entity_key: row.get(5)?,
        claim_key: row.get(6)?,
        content: row.get(7)?,
        summary: row.get(8)?,
        content_sha256: row.get(9)?,
    })
}

fn merge_remember_candidate(
    candidates: &mut BTreeMap<String, RememberCandidateAccumulator>,
    request: &RememberCandidateDetection<'_>,
    row: RememberCandidateRow,
    relationship: &str,
    score: f64,
    matched_on: &str,
) {
    if request.excluded_ids.contains(&row.memory_id) {
        return;
    }
    let entry =
        candidates
            .entry(row.memory_id.clone())
            .or_insert_with(|| RememberCandidateAccumulator {
                row,
                relationship: relationship.to_string(),
                score,
                matched_on: BTreeSet::new(),
            });
    let new_priority = relationship_priority(relationship);
    let old_priority = relationship_priority(&entry.relationship);
    if new_priority < old_priority || (new_priority == old_priority && score > entry.score) {
        entry.relationship = relationship.to_string();
        entry.score = score;
    }
    entry.matched_on.insert(matched_on.to_string());
}

impl RememberCandidateAccumulator {
    fn into_candidate(self) -> RememberCandidate {
        let snippet = bounded_char_slice(
            self.row.summary.as_deref().unwrap_or(&self.row.content),
            0,
            MAX_SNIPPET_CHARS.min(240),
        );
        RememberCandidate {
            memory_id: self.row.memory_id,
            relationship: self.relationship,
            score: self.score,
            matched_on: self.matched_on.into_iter().collect(),
            space: self.row.space,
            silo: self.row.silo,
            kind: self.row.kind,
            status: self.row.status,
            summary: self.row.summary,
            snippet,
            content_sha256: self.row.content_sha256,
            entity_key: self.row.entity_key,
            claim_key: self.row.claim_key,
        }
    }
}

fn compare_remember_candidates(
    left: &RememberCandidate,
    right: &RememberCandidate,
) -> std::cmp::Ordering {
    relationship_priority(&left.relationship)
        .cmp(&relationship_priority(&right.relationship))
        .then_with(|| right.score.total_cmp(&left.score))
        .then_with(|| left.memory_id.cmp(&right.memory_id))
}

fn relationship_priority(relationship: &str) -> u8 {
    match relationship {
        "duplicate" => 0,
        "update_candidate" => 1,
        _ => 2,
    }
}

fn lexical_terms(value: &str) -> BTreeSet<String> {
    search_terms(value)
        .into_iter()
        .filter(|term| term.chars().count() >= 3)
        .take(MAX_SEARCH_TERMS.saturating_mul(2))
        .collect()
}

fn jaccard_similarity(left: &BTreeSet<String>, right: &BTreeSet<String>) -> f64 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let intersection = left.intersection(right).count();
    let union = left.union(right).count();
    if union == 0 {
        0.0
    } else {
        f64::from(u32::try_from(intersection).unwrap_or(u32::MAX))
            / f64::from(u32::try_from(union).unwrap_or(u32::MAX))
    }
}

fn ensure_source_episode_exists(connection: &Connection, space: &str, id: &str) -> Result<()> {
    let exists: i64 = connection.query_row(
        "SELECT COUNT(*) FROM source_episodes WHERE id = ?1 AND space_name = ?2",
        params![id, space],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Err(Error::NotFound {
            entity: "source_episode",
            id: id.to_string(),
        });
    }
    Ok(())
}

/// Open a contradiction conflict between a new memory and an existing one,
/// for `conflict`-mode writes. Does not change either memory's status; the
/// conflict awaits human resolution.
fn open_conflict(
    transaction: &Transaction<'_>,
    space: &str,
    new_memory_id: &str,
    other_memory_id: &str,
    now: &str,
) -> Result<()> {
    transaction.execute(
        "INSERT INTO conflicts (id, space_name, status, memory_a_id, memory_b_id, \
         conflict_type, created_at, updated_at) \
         VALUES (?1, ?2, 'open', ?3, ?4, 'contradiction', ?5, ?5)",
        params![next_id("cfl"), space, new_memory_id, other_memory_id, now],
    )?;
    Ok(())
}

fn supersede_memory(
    transaction: &Transaction<'_>,
    space: &str,
    new_memory_id: &str,
    superseded_id: &str,
    now: &str,
) -> Result<()> {
    let target = transaction
        .query_row(
            "SELECT status, pinned FROM memories WHERE id = ?1 AND space_name = ?2",
            params![superseded_id, space],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()?;
    let Some((old_status, pinned)) = target else {
        return Err(Error::NotFound {
            entity: "memory",
            id: superseded_id.to_string(),
        });
    };
    if old_status != status::ACTIVE {
        return Err(Error::Conflict {
            message: format!("cannot supersede non-active memory: {superseded_id}"),
        });
    }
    if pinned == 1 {
        return Err(Error::Conflict {
            message: format!("cannot supersede pinned memory: {superseded_id}"),
        });
    }

    transaction.execute(
        "UPDATE memories SET status = ?1, updated_at = ?2 WHERE id = ?3",
        params![status::SUPERSEDED, now, superseded_id],
    )?;
    transaction.execute(
        "UPDATE memory_fts SET status = ?1 WHERE memory_id = ?2",
        params![status::SUPERSEDED, superseded_id],
    )?;
    transaction.execute(
        "UPDATE memory_fts_public SET status = ?1 WHERE memory_id = ?2",
        params![status::SUPERSEDED, superseded_id],
    )?;

    let event_id = next_id("evt");
    transaction.execute(
        "INSERT INTO memory_events (id, memory_id, event_type, old_status, new_status, actor, reason, created_at)
         VALUES (?1, ?2, 'supersede', ?3, ?4, 'memkeeper', ?5, ?6)",
        params![
            event_id,
            superseded_id,
            old_status,
            status::SUPERSEDED,
            format!("superseded by {new_memory_id}"),
            now,
        ],
    )?;
    link_memory(transaction, new_memory_id, superseded_id, "supersedes", now)?;
    link_memory(
        transaction,
        superseded_id,
        new_memory_id,
        "superseded_by",
        now,
    )?;
    Ok(())
}

fn link_memory(
    transaction: &Transaction<'_>,
    src_memory_id: &str,
    dst_memory_id: &str,
    link_type: &str,
    now: &str,
) -> Result<()> {
    ensure_memory_exists(transaction, src_memory_id)?;
    ensure_memory_exists(transaction, dst_memory_id)?;
    transaction.execute(
        "INSERT OR IGNORE INTO memory_links (
            src_memory_id, dst_memory_id, link_type, status, confidence, created_at
         ) VALUES (?1, ?2, ?3, 'active', 1.0, ?4)",
        params![src_memory_id, dst_memory_id, link_type, now],
    )?;
    Ok(())
}

fn ensure_memory_exists(connection: &Connection, id: &str) -> Result<()> {
    let exists: i64 =
        connection.query_row("SELECT COUNT(*) FROM memories WHERE id = ?1", [id], |row| {
            row.get(0)
        })?;
    if exists == 0 {
        return Err(Error::NotFound {
            entity: "memory",
            id: id.to_string(),
        });
    }
    Ok(())
}

fn ensure_memory_in_space(connection: &Connection, id: &str, space: &str) -> Result<()> {
    let exists: i64 = connection.query_row(
        "SELECT COUNT(*) FROM memories WHERE id = ?1 AND space_name = ?2",
        params![id, space],
        |row| row.get(0),
    )?;
    if exists == 0 {
        return Err(Error::NotFound {
            entity: "memory",
            id: id.to_string(),
        });
    }
    Ok(())
}

fn load_memory(connection: &Connection, id: &str, options: GetOptions) -> Result<MemoryRecord> {
    let mut memory = connection
        .query_row(
            "SELECT
                m.id, m.active_version_id, m.space_name, m.silo_name, m.scope, m.project_key,
                m.kind, m.entity_key, m.claim_key, m.status, m.confidence, m.pinned,
                m.source_episode_id, m.observed_at, m.created_at, m.updated_at, m.valid_from,
                m.valid_to, m.expires_at, m.deleted_at, v.content, v.summary,
                v.content_sha256, v.source_ref_json, m.metadata_json
             FROM memories m
             JOIN memory_versions v ON v.id = m.active_version_id
             WHERE m.id = ?1",
            [id],
            |row| {
                Ok(MemoryRecord {
                    id: row.get(0)?,
                    version_id: row.get(1)?,
                    space: row.get(2)?,
                    silo: row.get(3)?,
                    scope: row.get(4)?,
                    project_key: row.get(5)?,
                    kind: row.get(6)?,
                    entity_key: row.get(7)?,
                    claim_key: row.get(8)?,
                    status: row.get(9)?,
                    confidence: row.get(10)?,
                    pinned: row.get::<_, i64>(11)? == 1,
                    source_episode_id: if options.include_source {
                        row.get(12)?
                    } else {
                        None
                    },
                    observed_at: row.get(13)?,
                    created_at: row.get(14)?,
                    updated_at: row.get(15)?,
                    valid_from: row.get(16)?,
                    valid_to: row.get(17)?,
                    expires_at: row.get(18)?,
                    deleted_at: row.get(19)?,
                    content: row.get(20)?,
                    summary: row.get(21)?,
                    content_sha256: row.get(22)?,
                    source_ref_json: if options.include_source {
                        row.get(23)?
                    } else {
                        None
                    },
                    metadata_json: row.get(24)?,
                    tags: Vec::new(),
                    versions: None,
                    events: None,
                    links: None,
                })
            },
        )
        .optional()?
        .ok_or_else(|| Error::NotFound {
            entity: "memory",
            id: id.to_string(),
        })?;

    memory.tags = load_tags(connection, id)?;
    if options.include_history {
        let mut versions = load_versions_limited(connection, id, MAX_HISTORY_LIMIT)?;
        if !options.include_source {
            for version in &mut versions {
                version.source_ref_json = None;
            }
        }
        memory.versions = Some(versions);
        memory.events = Some(load_events_limited(connection, id, MAX_HISTORY_LIMIT)?);
    }
    if options.include_links {
        memory.links = Some(load_links(connection, id)?);
    }
    Ok(memory)
}

fn load_tags(connection: &Connection, id: &str) -> Result<Vec<String>> {
    let mut statement =
        connection.prepare("SELECT tag FROM memory_tags WHERE memory_id = ?1 ORDER BY tag")?;
    let rows = statement.query_map([id], |row| row.get(0))?;
    collect_rows(rows)
}

fn load_versions_limited(
    connection: &Connection,
    id: &str,
    limit: usize,
) -> Result<Vec<MemoryVersionRecord>> {
    let mut statement = connection.prepare(
        "SELECT id, version_num, content, summary, content_sha256, created_at, source_ref_json
         FROM memory_versions WHERE memory_id = ?1 ORDER BY version_num ASC LIMIT ?2",
    )?;
    let rows = statement.query_map(params![id, limit_i64(limit)?], |row| {
        Ok(MemoryVersionRecord {
            id: row.get(0)?,
            version_num: row.get(1)?,
            content: row.get(2)?,
            summary: row.get(3)?,
            content_sha256: row.get(4)?,
            created_at: row.get(5)?,
            source_ref_json: row.get(6)?,
        })
    })?;
    collect_rows(rows)
}

fn load_events_limited(
    connection: &Connection,
    id: &str,
    limit: usize,
) -> Result<Vec<MemoryEventRecord>> {
    let mut statement = connection.prepare(
        "SELECT id, event_type, old_status, new_status, reason, created_at
         FROM memory_events WHERE memory_id = ?1 ORDER BY created_at ASC, id ASC LIMIT ?2",
    )?;
    let rows = statement.query_map(params![id, limit_i64(limit)?], |row| {
        Ok(MemoryEventRecord {
            id: row.get(0)?,
            event_type: row.get(1)?,
            old_status: row.get(2)?,
            new_status: row.get(3)?,
            reason: row.get(4)?,
            created_at: row.get(5)?,
        })
    })?;
    collect_rows(rows)
}

fn count_for_id(connection: &Connection, table: &str, column: &str, id: &str) -> Result<usize> {
    let sql = format!("SELECT COUNT(*) FROM {table} WHERE {column} = ?1");
    let count: i64 = connection.query_row(&sql, [id], |row| row.get(0))?;
    usize::try_from(count).map_err(|_| Error::InvalidRequest {
        message: "history count overflowed usize".to_string(),
    })
}

fn load_links(connection: &Connection, id: &str) -> Result<Vec<MemoryLinkRecord>> {
    let mut statement = connection.prepare(
        "SELECT src_memory_id, dst_memory_id, link_type, status, confidence
         FROM memory_links
         WHERE src_memory_id = ?1 OR dst_memory_id = ?1
         ORDER BY link_type ASC, src_memory_id ASC, dst_memory_id ASC
         LIMIT ?2",
    )?;
    let rows = statement.query_map(params![id, limit_i64(MAX_GET_LINKS)?], |row| {
        Ok(MemoryLinkRecord {
            src_memory_id: row.get(0)?,
            dst_memory_id: row.get(1)?,
            link_type: row.get(2)?,
            status: row.get(3)?,
            confidence: row.get(4)?,
        })
    })?;
    collect_rows(rows)
}

fn memory_fts_metadata_text(
    project_key: Option<&str>,
    entity_key: Option<&str>,
    claim_key: Option<&str>,
    content: &str,
    summary: Option<&str>,
    tags_text: &str,
) -> String {
    let base_metadata = [project_key, entity_key, claim_key]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
    let normalized_tokens = normalized_search_token_text(&[
        Some(content),
        summary,
        Some(tags_text),
        Some(base_metadata.as_str()),
    ]);
    [base_metadata.as_str(), normalized_tokens.as_str()]
        .into_iter()
        .filter(|value| !value.trim().is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalized_search_token_text(values: &[Option<&str>]) -> String {
    let mut tokens = BTreeSet::new();
    for value in values.iter().flatten() {
        for term in search_terms(value) {
            for token in normalized_search_tokens_for_term(&term) {
                tokens.insert(token);
            }
        }
    }
    tokens.into_iter().collect::<Vec<_>>().join(" ")
}

fn normalized_search_tokens_for_term(term: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    for stem in search_term_stems(term) {
        if stem != term && is_prefixable_search_term(&stem) {
            push_unique(&mut tokens, stem);
        }
    }
    tokens
}

fn event_data_json(
    dry_run: bool,
    supersedes: &[String],
    contradicts: &[String],
    auto_superseded: &[String],
    conflict_candidates: &[RememberConflictCandidate],
) -> String {
    format!(
        "{{\"dry_run\":{dry_run},\"supersedes\":{},\"contradicts\":{},\"auto_superseded\":{},\"conflict_candidates\":{}}}",
        string_array_json(supersedes),
        string_array_json(contradicts),
        string_array_json(auto_superseded),
        remember_conflict_candidates_json(conflict_candidates)
    )
}

fn remember_conflict_candidates_json(candidates: &[RememberConflictCandidate]) -> String {
    let mut output = String::from("[");
    for (index, candidate) in candidates.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"memory_id\":{},\"kind\":{},\"observed_at\":{},\"snippet\":{}}}",
            json_string_for_store(&candidate.memory_id),
            json_string_for_store(&candidate.kind),
            json_string_for_store(&candidate.observed_at),
            json_string_for_store(&candidate.snippet)
        );
    }
    output.push(']');
    output
}

fn forget_event_data_json(dry_run: bool, mode: &str) -> String {
    format!(
        "{{\"dry_run\":{dry_run},\"mode\":{}}}",
        json_string_for_store(mode)
    )
}

/// Build the `data_json` payload stamped on a `correct` event. Captures the
/// corrected memory's provenance (was it synthesis-derived? which session?) plus
/// the optional replacement id, so the nightly synthesis loop can measure
/// correction density and target the sessions whose cards proved wrong without
/// re-deriving any of it from supersession history. Serialized with `serde_json`
/// so the persisted blob is always a valid object.
fn correction_event_data_json(
    dry_run: bool,
    mode: &str,
    corrected_by: Option<&str>,
    tags: &[String],
) -> String {
    let synthesis_derived = tags.iter().any(|tag| tag == "synthesis-derived");
    let session = tags
        .iter()
        .find_map(|tag| tag.strip_prefix("session:"))
        .map(str::to_string);
    serde_json::json!({
        "dry_run": dry_run,
        "mode": mode,
        "corrected_by": corrected_by,
        "synthesis_derived": synthesis_derived,
        "session": session,
    })
    .to_string()
}

pub(crate) struct JsonValidator;

impl JsonValidator {
    pub(crate) fn is_object(input: &str) -> bool {
        import_json_object_is_valid(input)
    }
}

pub(crate) fn open_initialized_read_fast(path: &Path) -> Result<Connection> {
    validate_store_path(path)?;
    if !path.exists() {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    match inspect_existing_store_immutable(path)? {
        ExistingStoreInspection::Compatible => {}
        ExistingStoreInspection::OlderSchema(actual)
        | ExistingStoreInspection::FutureSchema(actual) => {
            return Err(Error::SchemaMismatch {
                expected: SCHEMA_VERSION,
                actual,
            });
        }
        ExistingStoreInspection::Unrecognized => match inspect_existing_store(path)? {
            ExistingStoreInspection::Compatible => {}
            ExistingStoreInspection::OlderSchema(actual)
            | ExistingStoreInspection::FutureSchema(actual) => {
                return Err(Error::SchemaMismatch {
                    expected: SCHEMA_VERSION,
                    actual,
                });
            }
            ExistingStoreInspection::Unrecognized => {
                return Err(Error::NotInitialized {
                    path: path.to_path_buf(),
                });
            }
        },
    }

    reject_sqlite_sidecar_symlinks(path)?;
    register_sqlite_vec_extension()?;
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    configure_connection(&connection)?;
    validate_initialized(path, &connection)?;
    Ok(connection)
}

pub(crate) fn open_initialized_write(path: &Path) -> Result<Connection> {
    validate_store_path(path)?;
    if !path.exists() {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    let needs_migration = match inspect_existing_store_immutable(path)? {
        ExistingStoreInspection::Compatible => false,
        ExistingStoreInspection::OlderSchema(_) => true,
        ExistingStoreInspection::Unrecognized => match inspect_existing_store(path)? {
            ExistingStoreInspection::Compatible => false,
            ExistingStoreInspection::OlderSchema(_) => true,
            ExistingStoreInspection::Unrecognized => {
                return Err(Error::NotInitialized {
                    path: path.to_path_buf(),
                });
            }
            ExistingStoreInspection::FutureSchema(actual) => {
                return Err(Error::SchemaMismatch {
                    expected: SCHEMA_VERSION,
                    actual,
                });
            }
        },
        ExistingStoreInspection::FutureSchema(actual) => {
            return Err(Error::SchemaMismatch {
                expected: SCHEMA_VERSION,
                actual,
            });
        }
    };

    reject_sqlite_sidecar_symlinks(path)?;
    register_sqlite_vec_extension()?;
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    configure_connection(&connection)?;
    if needs_migration {
        apply_schema(&connection)?;
    }
    validate_initialized(path, &connection)?;
    Ok(connection)
}

fn reject_symlink_path(path: &Path) -> Result<()> {
    if fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "path must not be a symlink",
        });
    }
    Ok(())
}

pub(crate) fn reject_sqlite_sidecar_symlinks(path: &Path) -> Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = sidecar_path(path, suffix);
        if fs::symlink_metadata(&sidecar).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return Err(Error::InvalidPath {
                path: sidecar,
                reason: "SQLite sidecar path must not be a symlink",
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_store_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "path must not be empty",
        });
    }
    let display_path = path.to_string_lossy();
    if path == Path::new(":memory:") || display_path.starts_with("file:") {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason:
                "durable stores must use plain filesystem paths, not SQLite URI or memory paths",
        });
    }
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() {
            return Err(Error::InvalidPath {
                path: path.to_path_buf(),
                reason: "store path must not be a symlink",
            });
        }
        if metadata.is_dir() {
            return Err(Error::InvalidPath {
                path: path.to_path_buf(),
                reason: "path points to a directory",
            });
        }
    } else if path.is_dir() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "path points to a directory",
        });
    }
    reject_sqlite_sidecar_symlinks(path)?;
    Ok(())
}

fn reject_source_sidecar_output(source_store: &Path, output_path: &Path) -> Result<()> {
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

fn normalized_output_path(path: &Path) -> Result<PathBuf> {
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

fn normalize_path_lexically(path: &Path) -> PathBuf {
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

fn path_identity_key(path: &Path) -> String {
    path.to_string_lossy().to_lowercase()
}

fn validate_output_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "output path must not be empty",
        });
    }
    if path
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "output path must not contain parent directory components",
        });
    }
    let display_path = path.to_string_lossy();
    if path == Path::new(":memory:") || display_path.starts_with("file:") {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "output paths must use plain filesystem paths",
        });
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(Error::InvalidPath {
                    path: path.to_path_buf(),
                    reason: "output path must not be a symlink",
                });
            }
            if metadata.is_dir() {
                return Err(Error::InvalidPath {
                    path: path.to_path_buf(),
                    reason: "output path points to a directory",
                });
            }
            return Err(Error::Conflict {
                message: format!("output path already exists: {}", path.display()),
            });
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(Error::Io(error)),
    }
    reject_sqlite_sidecar_symlinks(path)?;
    Ok(())
}

fn create_temp_output_file(destination: &Path) -> Result<(PathBuf, File)> {
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

fn publish_temp_output(temp_path: &Path, destination: &Path) -> Result<()> {
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

fn reject_output_sidecar_files(path: &Path) -> Result<()> {
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

fn reject_existing_output_sidecars(path: &Path) -> Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = sidecar_path(path, suffix);
        match fs::symlink_metadata(&sidecar) {
            Ok(_) => {
                return Err(Error::Conflict {
                    message: format!(
                        "output SQLite sidecar already exists: {}",
                        sidecar.display()
                    ),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(Error::Io(error)),
        }
    }
    Ok(())
}

fn cleanup_temp_output(path: &Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(sidecar_path(path, "-wal"));
    let _ = fs::remove_file(sidecar_path(path, "-shm"));
    let _ = fs::remove_file(sidecar_path(path, "-journal"));
}

fn claim_or_preflight_init_path(path: &Path) -> Result<bool> {
    if !path.exists() {
        match create_new_private_file(path, false) {
            Ok(_) => return Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(Error::Io(error)),
        }
    }
    preflight_init_path(path)
}

fn preflight_init_path(path: &Path) -> Result<bool> {
    validate_store_path(path)?;
    if !path.exists() {
        return Ok(true);
    }

    let metadata = fs::metadata(path)?;
    if metadata.len() == 0 {
        return Ok(false);
    }

    match inspect_existing_store(path)? {
        ExistingStoreInspection::Compatible | ExistingStoreInspection::OlderSchema(_) => Ok(false),
        ExistingStoreInspection::Unrecognized => Err(Error::UnsafeExistingDatabase {
            path: path.to_path_buf(),
        }),
        ExistingStoreInspection::FutureSchema(actual) => Err(Error::SchemaMismatch {
            expected: SCHEMA_VERSION,
            actual,
        }),
    }
}

fn inspect_existing_store(path: &Path) -> Result<ExistingStoreInspection> {
    match inspect_existing_store_copy(path) {
        Ok(inspection) => Ok(inspection),
        Err(Error::Database(_)) => Ok(ExistingStoreInspection::Unrecognized),
        Err(error) => Err(error),
    }
}

fn inspect_existing_store_immutable(path: &Path) -> Result<ExistingStoreInspection> {
    match inspect_existing_store_immutable_inner(path) {
        Ok(inspection) => Ok(inspection),
        Err(Error::Database(_)) => Ok(ExistingStoreInspection::Unrecognized),
        Err(error) => Err(error),
    }
}

fn inspect_existing_store_immutable_inner(path: &Path) -> Result<ExistingStoreInspection> {
    let uri = sqlite_immutable_uri(path)?;
    register_sqlite_vec_extension()?;
    let connection = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    configure_connection(&connection)?;
    let actual = user_version(&connection)?;
    if actual > SCHEMA_VERSION {
        return Ok(ExistingStoreInspection::FutureSchema(actual));
    }
    if actual < SCHEMA_VERSION {
        return if older_schema_is_memkeeper(&connection, actual)? {
            Ok(ExistingStoreInspection::OlderSchema(actual))
        } else {
            Ok(ExistingStoreInspection::Unrecognized)
        };
    }

    match validate_initialized(path, &connection) {
        Ok(()) => Ok(ExistingStoreInspection::Compatible),
        Err(Error::NotInitialized { .. }) => Ok(ExistingStoreInspection::Unrecognized),
        Err(Error::SchemaMismatch { actual, .. }) => {
            Ok(ExistingStoreInspection::FutureSchema(actual))
        }
        Err(error) => Err(error),
    }
}

fn sqlite_immutable_uri(path: &Path) -> Result<String> {
    let absolute = fs::canonicalize(path)?;
    Ok(format!(
        "file:{}?mode=ro&immutable=1",
        percent_encode_path(&absolute.to_string_lossy())
    ))
}

fn percent_encode_path(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'-' | b'.' | b'_' | b'~' => {
                output.push(char::from(*byte));
            }
            byte => {
                let _ = write!(output, "%{byte:02X}");
            }
        }
    }
    output
}

fn inspect_existing_store_copy(path: &Path) -> Result<ExistingStoreInspection> {
    inspect_on_copy(path, |connection| {
        configure_connection(connection)?;
        let actual = user_version(connection)?;
        if actual > SCHEMA_VERSION {
            return Ok(ExistingStoreInspection::FutureSchema(actual));
        }
        if actual < SCHEMA_VERSION {
            return if older_schema_is_memkeeper(connection, actual)? {
                Ok(ExistingStoreInspection::OlderSchema(actual))
            } else {
                Ok(ExistingStoreInspection::Unrecognized)
            };
        }

        match validate_initialized(path, connection) {
            Ok(()) => Ok(ExistingStoreInspection::Compatible),
            Err(Error::NotInitialized { .. }) => Ok(ExistingStoreInspection::Unrecognized),
            Err(Error::SchemaMismatch { actual, .. }) => {
                Ok(ExistingStoreInspection::FutureSchema(actual))
            }
            Err(error) => Err(error),
        }
    })
}

pub(crate) fn inspect_on_copy<T>(
    path: &Path,
    inspect: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    reject_symlink_path(path)?;
    let copy_path = inspection_copy_path()?;
    fs::copy(path, &copy_path)?;
    copy_sidecar_if_present(path, &copy_path, "-wal")?;
    copy_sidecar_if_present(path, &copy_path, "-shm")?;
    copy_sidecar_if_present(path, &copy_path, "-journal")?;

    register_sqlite_vec_extension()?;
    let result = Connection::open_with_flags(&copy_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(Error::Database)
        .and_then(|connection| inspect(&connection));

    cleanup_inspection_copy(&copy_path);
    result
}

fn inspection_copy_path() -> Result<PathBuf> {
    for _ in 0..16 {
        let dir = env::temp_dir().join(format!(
            "memkeeper-inspect-{}-{}-{}",
            process::id(),
            unique_nanos(),
            ID_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        match create_private_dir(&dir) {
            Ok(()) => return Ok(dir.join("store.sqlite")),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(Error::Io(error)),
        }
    }
    Err(Error::InvalidPath {
        path: env::temp_dir(),
        reason: "could not create private inspection directory",
    })
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new().mode(0o700).create(path)
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    fs::create_dir(path)
}

fn create_new_private_file(path: &Path, read: bool) -> std::io::Result<File> {
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

fn copy_sidecar_if_present(source: &Path, copy: &Path, suffix: &str) -> Result<()> {
    let source_sidecar = sidecar_path(source, suffix);
    match fs::symlink_metadata(&source_sidecar) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(Error::InvalidPath {
                    path: source_sidecar,
                    reason: "SQLite sidecar path must not be a symlink",
                });
            }
            fs::copy(source_sidecar, sidecar_path(copy, suffix))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(Error::Io(error)),
    }
    Ok(())
}

fn cleanup_inspection_copy(path: &Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(sidecar_path(path, "-wal"));
    let _ = fs::remove_file(sidecar_path(path, "-shm"));
    let _ = fs::remove_file(sidecar_path(path, "-journal"));
    if let Some(parent) = path.parent() {
        let _ = fs::remove_dir(parent);
    }
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn create_parent_dirs(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

#[cfg(feature = "semantic")]
fn register_sqlite_vec_extension() -> Result<()> {
    // sqlite-vec exposes SQLite's raw extension entry point; rusqlite requires
    // registering that entry point before opening connections that use vec0. The
    // registration is process-global, so do it once to avoid adding duplicate
    // auto-extension hooks on repeated short-lived CLI/store calls.
    match SQLITE_VEC_EXTENSION_REGISTERED.get_or_init(|| {
        let entry: RawAutoExtension =
            unsafe { std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ()) };
        unsafe { register_auto_extension(entry) }.map_err(|error| error.to_string())
    }) {
        Ok(()) => Ok(()),
        Err(message) => Err(Error::InvalidRequest {
            message: format!("failed to register sqlite-vec extension: {message}"),
        }),
    }
}

#[cfg(not(feature = "semantic"))]
#[allow(clippy::unnecessary_wraps)]
fn register_sqlite_vec_extension() -> Result<()> {
    Ok(())
}

pub(crate) fn configure_connection(connection: &Connection) -> Result<()> {
    connection.busy_timeout(Duration::from_millis(BUSY_TIMEOUT_MS))?;
    connection.execute_batch("PRAGMA foreign_keys = ON;")?;
    Ok(())
}

fn enable_wal(connection: &Connection) -> Result<String> {
    connection
        .query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))
        .map_err(Into::into)
}

pub(crate) fn journal_mode(connection: &Connection) -> Result<String> {
    connection
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .map_err(Into::into)
}

pub(crate) fn user_version(connection: &Connection) -> Result<i32> {
    connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(Into::into)
}

pub(crate) fn sqlite_version(connection: &Connection) -> Result<String> {
    connection
        .query_row("SELECT sqlite_version()", [], |row| row.get(0))
        .map_err(Into::into)
}

fn apply_schema(connection: &Connection) -> Result<()> {
    connection.execute_batch("BEGIN IMMEDIATE;")?;
    let schema_sql = transactional_schema_sql();
    let result = connection
        .execute_batch(&schema_sql)
        .and_then(|()| apply_feature_schema(connection))
        .and_then(|()| connection.execute_batch("COMMIT;"));

    if let Err(error) = result {
        let _ = connection.execute_batch("ROLLBACK;");
        return Err(Error::Database(error));
    }

    Ok(())
}

fn apply_feature_schema(connection: &Connection) -> rusqlite::Result<()> {
    connection.execute_batch(SCHEMA_UPGRADE_SQL)
}

fn transactional_schema_sql() -> String {
    let mut sql = String::with_capacity(SCHEMA_SQL.len());
    for line in SCHEMA_SQL.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("PRAGMA foreign_keys")
            || trimmed.starts_with("PRAGMA journal_mode")
            || trimmed.starts_with("PRAGMA busy_timeout")
        {
            continue;
        }
        sql.push_str(line);
        sql.push('\n');
    }
    sql
}

pub(crate) fn validate_initialized(path: &Path, connection: &Connection) -> Result<()> {
    let actual = user_version(connection)?;
    if actual == 0 {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }
    if actual != SCHEMA_VERSION {
        return Err(Error::SchemaMismatch {
            expected: SCHEMA_VERSION,
            actual,
        });
    }

    for table in REQUIRED_TABLES {
        if !table_exists(connection, table)? {
            return Err(Error::NotInitialized {
                path: path.to_path_buf(),
            });
        }
    }
    for table in REQUIRED_FTS_TABLES {
        if !table_exists(connection, table)? {
            return Err(Error::NotInitialized {
                path: path.to_path_buf(),
            });
        }
    }
    let applied = count(
        connection,
        &format!("SELECT COUNT(*) FROM schema_migrations WHERE version = {SCHEMA_VERSION}"),
    )?;
    if applied == 0 {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    let schema_config = required_config_value(path, connection, "schema_version")?;
    let expected_schema = SCHEMA_VERSION.to_string();
    if schema_config != expected_schema {
        let actual = schema_config.parse().unwrap_or_default();
        return Err(Error::SchemaMismatch {
            expected: SCHEMA_VERSION,
            actual,
        });
    }

    let protocol = required_config_value(path, connection, "protocol_version")?;
    if protocol != "memkeeper.v0.1" {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    let default_space = required_config_value(path, connection, "default_space")?;
    if default_space != DEFAULT_SPACE {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    let seeded_space = count(
        connection,
        "SELECT COUNT(*) FROM spaces WHERE name = 'workspace-memory'",
    )?;
    if seeded_space == 0 {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    Ok(())
}

fn older_schema_is_memkeeper(connection: &Connection, actual: i32) -> Result<bool> {
    if actual <= 0 || actual >= SCHEMA_VERSION {
        return Ok(false);
    }
    for table in REQUIRED_TABLES {
        if !table_exists(connection, table)? {
            return Ok(false);
        }
    }
    for table in REQUIRED_FTS_TABLES {
        if !table_exists(connection, table)? {
            return Ok(false);
        }
    }
    let migration_count = count(
        connection,
        &format!("SELECT COUNT(*) FROM schema_migrations WHERE version = {actual}"),
    )?;
    if migration_count == 0 {
        return Ok(false);
    }
    let schema_config = connection
        .query_row(
            "SELECT value FROM config_kv WHERE key = 'schema_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let protocol = connection
        .query_row(
            "SELECT value FROM config_kv WHERE key = 'protocol_version'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let default_space = connection
        .query_row(
            "SELECT value FROM config_kv WHERE key = 'default_space'",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(schema_config.as_deref() == Some(&actual.to_string())
        && protocol.as_deref() == Some("memkeeper.v0.1")
        && default_space.as_deref() == Some(DEFAULT_SPACE))
}

pub(crate) fn table_exists(connection: &Connection, name: &str) -> Result<bool> {
    let mut statement = connection.prepare_cached(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?1)",
    )?;
    let exists = statement.query_row([name], |row| row.get::<_, i64>(0))?;
    Ok(exists == 1)
}

pub(crate) fn required_config_value(
    path: &Path,
    connection: &Connection,
    key: &str,
) -> Result<String> {
    let value = connection
        .query_row("SELECT value FROM config_kv WHERE key = ?1", [key], |row| {
            row.get(0)
        })
        .optional()?;
    value.ok_or_else(|| Error::NotInitialized {
        path: path.to_path_buf(),
    })
}

pub(crate) fn count(connection: &Connection, sql: &str) -> Result<i64> {
    connection
        .query_row(sql, [], |row| row.get(0))
        .map_err(Into::into)
}

#[cfg(test)]
mod tests;
