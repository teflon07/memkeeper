#![forbid(unsafe_code)]

//! Versioned command/protocol constants for `memkeeper`.
//!
//! JSON serialization is intentionally not implemented in the scaffold because
//! the v0.1 protocol should be reviewed before adding dependencies.

/// Current protocol identifier used by host adapters and CLI responses.
pub const PROTOCOL_VERSION: &str = "memkeeper.v0.1";

/// Stable command names in the v0.1 CLI/protocol surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Initialize or inspect a store.
    Init,
    /// Run setup diagnostics without mutating the store.
    Doctor,
    /// List configured spaces.
    SpaceList,
    /// Create a space.
    SpaceCreate,
    /// List silos.
    SiloList,
    /// Explicitly store a memory.
    Remember,
    /// Ingest a document source as embedded, isolated chunks (RAG document store).
    Ingest,
    /// Hybrid search over ingested document chunks (RAG document search).
    DocumentSearch,
    /// Fetch a document's chunks by path or chunk id.
    DocumentGet,
    /// Rank document chunks that earned retrieval traffic (promotion candidates).
    PromotionCandidates,
    /// Mark document chunks as extracted (promoted) so they stop surfacing.
    MarkExtracted,
    /// Surface exact-content duplicate document chunks (independent duplicates).
    DocumentDuplicates,
    /// Delete document chunks by id (user-driven duplicate cleanup).
    DocumentPrune,
    /// Search indexed memories.
    Search,
    /// Create or update one projected graph entity.
    EntityUpsert,
    /// Create or update one projected graph relationship.
    RelationshipUpsert,
    /// Merge one projected graph entity into another (relink edges, tombstone source).
    EntityMerge,
    /// Search projected graph entities.
    EntitySearch,
    /// Traverse projected graph neighbors.
    GraphNeighbors,
    /// Build compact memory context around projected graph neighbors.
    GraphContext,
    /// Project the entire connected entity graph for whole-graph visualization.
    GraphFull,
    /// List recent memories for review/admin workflows.
    MemoryList,
    /// Batch multiple searches.
    BatchSearch,
    /// Produce a compact memory pack.
    Pack,
    /// Diagnose the exact pre-rerank candidate pool without returning memory content.
    PoolTrace,
    /// Fetch one memory by id.
    Get,
    /// Tombstone a memory by id.
    Forget,
    /// Return memory history/events.
    History,
    /// Return store/index statistics.
    Stats,
    /// Export the store/history.
    Export,
    /// Import a previous export.
    Import,
    /// Run explicit bounded maintenance/dream tasks.
    Dream,
    /// Create a consistent backup.
    Backup,
    /// Score documents against a query with the cross-encoder reranker.
    Rerank,
    /// Rebuild the semantic vector index from stored embeddings.
    Reindex,
    /// Stamp `verified_at` (and optionally `verified_against`) into a memory's metadata.
    Verify,
    /// Record recall telemetry events (surfaced/retrieved memories).
    RecallLog,
    /// Submit a candidate memory for review.
    CandidateSubmit,
    /// List candidate memories awaiting (or past) review.
    CandidateList,
    /// Approve a candidate, promoting it into a memory.
    CandidateApprove,
    /// Reject a candidate memory.
    CandidateReject,
    /// Quarantine a candidate memory (an adjudicator flagged it).
    CandidateQuarantine,
}

impl Command {
    /// Return the stable wire/CLI command name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::Doctor => "doctor",
            Self::SpaceList => "space-list",
            Self::SpaceCreate => "space-create",
            Self::SiloList => "silo-list",
            Self::Remember => "remember",
            Self::Ingest => "ingest",
            Self::DocumentSearch => "document-search",
            Self::DocumentGet => "document-get",
            Self::PromotionCandidates => "promotion-candidates",
            Self::MarkExtracted => "mark-extracted",
            Self::DocumentDuplicates => "document-duplicates",
            Self::DocumentPrune => "document-prune",
            Self::Search => "search",
            Self::EntityUpsert => "entity-upsert",
            Self::RelationshipUpsert => "relationship-upsert",
            Self::EntityMerge => "entity-merge",
            Self::EntitySearch => "entity-search",
            Self::GraphNeighbors => "graph-neighbors",
            Self::GraphContext => "graph-context",
            Self::GraphFull => "graph-full",
            Self::MemoryList => "memory-list",
            Self::BatchSearch => "batch-search",
            Self::Pack => "pack",
            Self::PoolTrace => "pool-trace",
            Self::Get => "get",
            Self::Forget => "forget",
            Self::History => "history",
            Self::Stats => "stats",
            Self::Export => "export",
            Self::Import => "import",
            Self::Dream => "dream",
            Self::Backup => "backup",
            Self::Rerank => "rerank",
            Self::Reindex => "reindex",
            Self::Verify => "verify",
            Self::RecallLog => "recall-log",
            Self::CandidateSubmit => "candidate-submit",
            Self::CandidateList => "candidate-list",
            Self::CandidateApprove => "candidate-approve",
            Self::CandidateReject => "candidate-reject",
            Self::CandidateQuarantine => "candidate-quarantine",
        }
    }

    /// Parse a stable wire/CLI command name.
    #[must_use]
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "init" => Some(Self::Init),
            "doctor" => Some(Self::Doctor),
            "space-list" => Some(Self::SpaceList),
            "space-create" => Some(Self::SpaceCreate),
            "silo-list" => Some(Self::SiloList),
            "remember" => Some(Self::Remember),
            "ingest" => Some(Self::Ingest),
            "document-search" => Some(Self::DocumentSearch),
            "document-get" => Some(Self::DocumentGet),
            "promotion-candidates" => Some(Self::PromotionCandidates),
            "mark-extracted" => Some(Self::MarkExtracted),
            "document-duplicates" => Some(Self::DocumentDuplicates),
            "document-prune" => Some(Self::DocumentPrune),
            "search" => Some(Self::Search),
            "entity-upsert" => Some(Self::EntityUpsert),
            "relationship-upsert" => Some(Self::RelationshipUpsert),
            "entity-merge" => Some(Self::EntityMerge),
            "entity-search" => Some(Self::EntitySearch),
            "graph-neighbors" => Some(Self::GraphNeighbors),
            "graph-context" => Some(Self::GraphContext),
            "graph-full" => Some(Self::GraphFull),
            "memory-list" => Some(Self::MemoryList),
            "batch-search" => Some(Self::BatchSearch),
            "pack" => Some(Self::Pack),
            "pool-trace" => Some(Self::PoolTrace),
            "get" => Some(Self::Get),
            "forget" => Some(Self::Forget),
            "history" => Some(Self::History),
            "stats" => Some(Self::Stats),
            "export" => Some(Self::Export),
            "import" => Some(Self::Import),
            "dream" => Some(Self::Dream),
            "backup" => Some(Self::Backup),
            "rerank" => Some(Self::Rerank),
            "reindex" => Some(Self::Reindex),
            "verify" => Some(Self::Verify),
            "recall-log" => Some(Self::RecallLog),
            "candidate-submit" => Some(Self::CandidateSubmit),
            "candidate-list" => Some(Self::CandidateList),
            "candidate-approve" => Some(Self::CandidateApprove),
            "candidate-reject" => Some(Self::CandidateReject),
            "candidate-quarantine" => Some(Self::CandidateQuarantine),
            _ => None,
        }
    }
}

/// Stable top-level error codes from the protocol draft.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    /// Input shape or value failed validation.
    InvalidRequest,
    /// Store path has no initialized database.
    StoreNotInitialized,
    /// Store schema is newer/unsupported or migration required.
    SchemaMismatch,
    /// Requested memory/space/silo does not exist.
    NotFound,
    /// Operation conflicts with current state or policy.
    Conflict,
    /// Store is busy beyond configured timeout.
    Locked,
    /// Filesystem or backup/export failure.
    IoError,
    /// Unexpected bug; safe diagnostic details only.
    InternalError,
    /// A required semantic dependency (embedder/reranker) failed at request
    /// time, so retrieval could not run in the required mode
    /// (`MEMKEEPER_REQUIRE_SEMANTIC`). Typically transient (rate limit, provider
    /// outage) and retryable.
    SemanticUnavailable,
}

impl ErrorCode {
    /// Return the stable wire error code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::StoreNotInitialized => "store_not_initialized",
            Self::SchemaMismatch => "schema_mismatch",
            Self::NotFound => "not_found",
            Self::Conflict => "conflict",
            Self::Locked => "locked",
            Self::IoError => "io_error",
            Self::InternalError => "internal_error",
            Self::SemanticUnavailable => "semantic_unavailable",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Command, ErrorCode, PROTOCOL_VERSION};

    #[test]
    fn protocol_identity_is_v0_1() {
        assert_eq!(PROTOCOL_VERSION, "memkeeper.v0.1");
    }

    #[test]
    fn error_code_semantic_unavailable_wire_string() {
        assert_eq!(
            ErrorCode::SemanticUnavailable.as_str(),
            "semantic_unavailable"
        );
    }

    #[test]
    fn commands_round_trip() {
        let commands = [
            Command::Init,
            Command::Doctor,
            Command::SpaceList,
            Command::SpaceCreate,
            Command::SiloList,
            Command::Remember,
            Command::Ingest,
            Command::DocumentSearch,
            Command::DocumentGet,
            Command::PromotionCandidates,
            Command::MarkExtracted,
            Command::DocumentDuplicates,
            Command::DocumentPrune,
            Command::Search,
            Command::EntityUpsert,
            Command::RelationshipUpsert,
            Command::EntityMerge,
            Command::EntitySearch,
            Command::GraphNeighbors,
            Command::GraphContext,
            Command::GraphFull,
            Command::Rerank,
            Command::MemoryList,
            Command::BatchSearch,
            Command::Pack,
            Command::PoolTrace,
            Command::Get,
            Command::Forget,
            Command::History,
            Command::Stats,
            Command::Export,
            Command::Import,
            Command::Dream,
            Command::Backup,
            Command::Reindex,
            Command::Verify,
            Command::RecallLog,
            Command::CandidateSubmit,
            Command::CandidateList,
            Command::CandidateApprove,
            Command::CandidateReject,
            Command::CandidateQuarantine,
        ];
        for command in commands {
            assert_eq!(Command::parse(command.as_str()), Some(command));
        }
    }

    #[test]
    fn error_codes_are_snake_case() {
        assert_eq!(ErrorCode::InvalidRequest.as_str(), "invalid_request");
        assert_eq!(
            ErrorCode::StoreNotInitialized.as_str(),
            "store_not_initialized"
        );
    }
}
