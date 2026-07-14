//! Public request, report, and record types for the store boundary.
//!
//! Pure data definitions extracted from `lib.rs` to keep the engine logic
//! navigable. Re-exported from the crate root via `pub use types::*`, so the
//! crate's public API is unchanged.

use std::path::PathBuf;

use crate::MAX_BATCH_QUERIES;

/// Result of initializing a store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitReport {
    /// True when the database file did not exist before this init call.
    pub created: bool,
    /// True when the store was initialized or already compatible.
    pub initialized: bool,
    /// Schema version after initialization.
    pub schema_version: i32,
    /// Protocol version recorded in store metadata.
    pub protocol_version: String,
    /// Runtime `SQLite` library version.
    pub sqlite_version: String,
    /// Active journal mode after initialization.
    pub journal_mode: String,
    /// Configured spaces after initialization.
    pub spaces: Vec<String>,
    /// Default space recorded in store metadata.
    pub default_space: String,
}

/// Request to create one top-level memory space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceCreateRequest {
    /// Stable space name.
    pub name: String,
    /// Optional display name.
    pub display_name: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Optional default silo name. Defaults to `durable`.
    pub default_silo: Option<String>,
    /// Optional ontology identifier or serialized config.
    pub ontology: Option<String>,
    /// Optional canonical JSON config object.
    pub config_json: Option<String>,
    /// Return the existing space instead of failing if it already exists.
    pub if_not_exists: bool,
}

/// Result of listing spaces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceListReport {
    /// Spaces in deterministic name order.
    pub spaces: Vec<SpaceRecord>,
    /// True when additional spaces were omitted because of the v0.1 output cap.
    pub truncated: bool,
}

/// Result of creating a space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceCreateReport {
    /// Stored space projection.
    pub space: SpaceRecord,
    /// True when this call inserted the space.
    pub created: bool,
}

/// Stored space projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceRecord {
    /// Stable space name.
    pub name: String,
    /// Optional display name.
    pub display_name: Option<String>,
    /// Optional description.
    pub description: Option<String>,
    /// Default silo for new memories in this space.
    pub default_silo: String,
    /// Optional ontology identifier or serialized config.
    pub ontology: Option<String>,
    /// Optional canonical JSON config object.
    pub config_json: Option<String>,
    /// Creation timestamp.
    pub created_at: String,
    /// Update timestamp.
    pub updated_at: String,
    /// Total memories in this space.
    pub memory_count: i64,
    /// Active memories in this space.
    pub active_count: i64,
    /// Number of silos configured in this space.
    pub silo_count: i64,
}

/// Request to list silos for a space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiloListRequest {
    /// Optional space. Defaults to `workspace-memory`.
    pub space: Option<String>,
}

/// Result of listing silos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiloListReport {
    /// Space queried.
    pub space: String,
    /// Silos in deterministic policy/name order.
    pub silos: Vec<SiloRecord>,
    /// True when additional silos were omitted because of the v0.1 output cap.
    pub truncated: bool,
}

/// Stored silo projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SiloRecord {
    /// Space containing this silo.
    pub space: String,
    /// Silo name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Retention policy.
    pub retention_policy: String,
    /// Default scope for memories in this silo.
    pub default_scope: String,
    /// Optional canonical JSON config object.
    pub config_json: Option<String>,
    /// Creation timestamp.
    pub created_at: String,
    /// Update timestamp.
    pub updated_at: String,
    /// Total memories in this silo.
    pub memory_count: i64,
    /// Active memories in this silo.
    pub active_count: i64,
    /// True when this is the parent space default silo.
    pub is_default: bool,
}

/// Optional immutable retrieval companion supplied by a source-aware adapter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetrievalRepresentationInput {
    /// Versioned representation kind.
    pub kind: String,
    /// Bounded retrieval text.
    pub text: String,
}

/// Stored retrieval companion owned by one immutable memory version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryRepresentationRecord {
    /// Memory version that owns this representation.
    pub version_id: String,
    /// Versioned representation kind.
    pub kind: String,
    /// Full retrieval text, exposed only on explicit audit reads.
    pub text: String,
    /// SHA-256 digest of `text`.
    pub text_sha256: String,
    /// Storage timestamp.
    pub created_at: String,
}

/// Bounded projection status returned only for representation-bearing writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentationWriteStatus {
    /// Versioned representation kind.
    pub kind: String,
    /// SHA-256 digest of the stored retrieval text.
    pub text_sha256: String,
    /// Whether the lexical projection was written.
    pub fts_indexed: bool,
    /// Whether the semantic token projection was written.
    pub semantic_indexed: bool,
    /// `indexed` or `lexical_only`.
    pub status: String,
}

/// Request to store one explicit memory.
#[derive(Debug, Clone, PartialEq)]
pub struct RememberRequest {
    /// Optional space; defaults to `workspace-memory`.
    pub space: Option<String>,
    /// Optional silo; defaults to the space default silo.
    pub silo: Option<String>,
    /// Optional scope; defaults to `workspace`.
    pub scope: Option<String>,
    /// Optional project key/path.
    pub project_key: Option<String>,
    /// Optional memory kind; inferred from deterministic prefix when omitted.
    pub kind: Option<String>,
    /// Canonical memory text.
    pub content: String,
    /// Optional shorter summary.
    pub summary: Option<String>,
    /// Optional source-aware retrieval companion.
    pub retrieval_representation: Option<RetrievalRepresentationInput>,
    /// Tags to attach to the memory.
    pub tags: Vec<String>,
    /// Optional stable entity key.
    pub entity_key: Option<String>,
    /// Optional stable claim key.
    pub claim_key: Option<String>,
    /// Confidence score from 0.0 to 1.0.
    pub confidence: f64,
    /// Optional observation timestamp; defaults to storage time.
    pub observed_at: Option<String>,
    /// Optional valid-from timestamp.
    pub valid_from: Option<String>,
    /// Optional valid-to timestamp.
    pub valid_to: Option<String>,
    /// Optional expiration timestamp.
    pub expires_at: Option<String>,
    /// Optional canonical JSON source/provenance object.
    pub source_ref_json: Option<String>,
    /// Optional metadata JSON object (e.g. verification provenance).
    pub metadata_json: Option<String>,
    /// Optional existing source episode id.
    pub source_episode_id: Option<String>,
    /// Whether the memory is pinned against automatic supersession.
    pub pinned: bool,
    /// Explicit memory ids superseded by this memory.
    pub supersedes: Vec<String>,
    /// Explicit memory ids contradicted by this memory.
    pub contradicts: Vec<String>,
    /// Optional caller-provided embedding for the semantic sidecar.
    pub embedding: Option<Vec<f32>>,
    /// Stable identifier of the embedding model that produced `embedding`.
    pub embedding_model_id: Option<String>,
    /// Optional late-interaction token embedding (one vector per token).
    pub token_embedding: Option<Vec<Vec<f32>>>,
    /// Stable identifier of the model that produced `token_embedding`.
    pub token_embedding_model_id: Option<String>,
    /// Validate and return a report without committing writes.
    pub dry_run: bool,
    /// Supersession mode (one of `REMEMBER_SUPERSEDE_MODES`); governs how this
    /// write resolves against active memories sharing its entity/claim key.
    pub mode: String,
}

/// Result of an explicit remember write.
#[derive(Debug, Clone, PartialEq)]
pub struct RememberReport {
    /// Stored or dry-run memory projection.
    pub memory: MemoryRecord,
    /// Remember event id.
    pub event_id: String,
    /// Processing status for this deterministic MVP path.
    pub processing_status: String,
    /// Bounded representation projection status when one was supplied.
    pub representation: Option<RepresentationWriteStatus>,
    /// Duplicate/update candidates detected before insertion.
    pub candidates: Vec<RememberCandidate>,
    /// True when more candidates existed beyond the output cap.
    pub candidates_truncated: bool,
    /// Memory ids automatically superseded by the same entity/claim write policy.
    pub auto_superseded: Vec<String>,
    /// Same entity/claim active memories surfaced for continuity conflict review.
    pub conflict_candidates: Vec<RememberConflictCandidate>,
    /// Memory ids that supersede/conflict modes WOULD retire, returned by
    /// `suggest` mode without mutating anything. Empty in other modes.
    pub supersede_suggestions: Vec<String>,
    /// True when the request was validated but rolled back.
    pub dry_run: bool,
}

/// Same entity/claim active memory surfaced during a continuity write.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RememberConflictCandidate {
    /// Existing memory id.
    pub memory_id: String,
    /// Existing memory kind.
    pub kind: String,
    /// Existing observation timestamp.
    pub observed_at: String,
    /// Source-free content snippet.
    pub snippet: String,
}

/// Deterministic duplicate/update candidate reported by `remember`.
#[derive(Debug, Clone, PartialEq)]
pub struct RememberCandidate {
    /// Existing memory id.
    pub memory_id: String,
    /// Candidate relationship: `duplicate`, `update_candidate`, or `related_candidate`.
    pub relationship: String,
    /// Deterministic confidence-like score from 0.0 to 1.0.
    pub score: f64,
    /// Signals that matched this candidate.
    pub matched_on: Vec<String>,
    /// Space containing the candidate.
    pub space: String,
    /// Silo containing the candidate.
    pub silo: String,
    /// Candidate kind.
    pub kind: String,
    /// Candidate status.
    pub status: String,
    /// Optional summary from the existing memory.
    pub summary: Option<String>,
    /// Source-free content snippet.
    pub snippet: String,
    /// Content SHA-256 of the existing memory.
    pub content_sha256: String,
    /// Optional entity key.
    pub entity_key: Option<String>,
    /// Optional claim key.
    pub claim_key: Option<String>,
}

/// Options for fetching one memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GetOptions {
    /// Include immutable versions and event history.
    pub include_history: bool,
    /// Include memory links.
    pub include_links: bool,
    /// Include stored source/provenance JSON.
    pub include_source: bool,
}

/// Request to tombstone one memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgetRequest {
    /// Memory id to tombstone.
    pub id: String,
    /// Optional user-visible reason for the audit event.
    pub reason: Option<String>,
    /// Forget mode. `tombstone` (routine retire) or `correct` (retire because the
    /// memory is wrong). `correct` records a distinct `correct` event carrying the
    /// memory's synthesis/session provenance, so corrections are queryable apart
    /// from routine cleanup and never need to be inferred from supersession.
    pub mode: String,
    /// For `correct` mode only: optional id of the memory that replaces the wrong
    /// one. When set, a `contradicts` link is recorded from the replacement to the
    /// corrected memory.
    pub corrected_by: Option<String>,
    /// Validate and return a report without committing writes.
    pub dry_run: bool,
}

/// Result of a forget/tombstone operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgetReport {
    /// Memory id.
    pub memory_id: String,
    /// Status before forget.
    pub old_status: String,
    /// Status after forget.
    pub new_status: String,
    /// Forget event id.
    pub event_id: String,
    /// True when the request was validated but rolled back.
    pub dry_run: bool,
}

/// Request to stamp verification provenance onto one memory.
#[derive(Debug, Clone)]
pub struct VerifyRequest {
    /// Memory id to verify.
    pub memory_id: String,
    /// Optional source reference for the verification (e.g. a file path or URL).
    pub verified_against: Option<String>,
    /// Optional explicit timestamp string (ISO-8601 UTC). Defaults to store time when omitted.
    pub now: Option<String>,
}

/// Result of a verify operation.
#[derive(Debug, Clone)]
pub struct VerifyReport {
    /// Memory id that was verified.
    pub memory_id: String,
    /// Timestamp written as `verified_at` in `metadata_json`.
    pub verified_at: String,
}

/// Options for fetching memory audit history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryOptions {
    /// Maximum versions/events to return.
    pub limit: usize,
    /// Include stored source/provenance JSON in versions.
    pub include_source: bool,
}

/// Audit history report for one memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HistoryReport {
    /// Memory id.
    pub memory_id: String,
    /// Current memory status.
    pub current_status: String,
    /// Event history, chronological.
    pub events: Vec<MemoryEventRecord>,
    /// Version history, chronological.
    pub versions: Vec<MemoryVersionRecord>,
    /// True when more versions or events exist beyond the limit.
    pub truncated: bool,
}

/// Stored memory projection plus optional related details.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryRecord {
    /// Memory id.
    pub id: String,
    /// Active version id.
    pub version_id: String,
    /// Space name.
    pub space: String,
    /// Silo name.
    pub silo: String,
    /// Scope.
    pub scope: String,
    /// Optional project key.
    pub project_key: Option<String>,
    /// Memory kind.
    pub kind: String,
    /// Optional stable entity key.
    pub entity_key: Option<String>,
    /// Optional stable claim key.
    pub claim_key: Option<String>,
    /// Memory status.
    pub status: String,
    /// Confidence score.
    pub confidence: f64,
    /// Whether the memory is pinned.
    pub pinned: bool,
    /// Canonical memory text.
    pub content: String,
    /// Optional summary.
    pub summary: Option<String>,
    /// Optional retrieval representation for the active immutable version.
    pub retrieval_representation: Option<MemoryRepresentationRecord>,
    /// Content SHA-256 hex digest.
    pub content_sha256: String,
    /// Tags.
    pub tags: Vec<String>,
    /// Optional existing source episode id.
    pub source_episode_id: Option<String>,
    /// Optional canonical JSON source/provenance object.
    pub source_ref_json: Option<String>,
    /// Observation timestamp.
    pub observed_at: String,
    /// Creation timestamp.
    pub created_at: String,
    /// Update timestamp.
    pub updated_at: String,
    /// Optional valid-from timestamp.
    pub valid_from: Option<String>,
    /// Optional valid-to timestamp.
    pub valid_to: Option<String>,
    /// Optional expiration timestamp.
    pub expires_at: Option<String>,
    /// Optional deletion timestamp.
    pub deleted_at: Option<String>,
    /// Optional metadata JSON object.
    pub metadata_json: Option<String>,
    /// Optional immutable version history.
    pub versions: Option<Vec<MemoryVersionRecord>>,
    /// Optional event history.
    pub events: Option<Vec<MemoryEventRecord>>,
    /// Optional links involving this memory.
    pub links: Option<Vec<MemoryLinkRecord>>,
}

/// Immutable memory version record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryVersionRecord {
    /// Version id.
    pub id: String,
    /// Version number.
    pub version_num: i64,
    /// Version content.
    pub content: String,
    /// Optional version summary.
    pub summary: Option<String>,
    /// Optional retrieval representation owned by this version.
    pub retrieval_representation: Option<MemoryRepresentationRecord>,
    /// Version content SHA-256 hex digest.
    pub content_sha256: String,
    /// Version creation timestamp.
    pub created_at: String,
    /// Optional source/provenance JSON.
    pub source_ref_json: Option<String>,
}

/// Memory event record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryEventRecord {
    /// Event id.
    pub id: String,
    /// Event type.
    pub event_type: String,
    /// Optional old status.
    pub old_status: Option<String>,
    /// Optional new status.
    pub new_status: Option<String>,
    /// Optional event reason.
    pub reason: Option<String>,
    /// Event timestamp.
    pub created_at: String,
}

/// Memory link record.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryLinkRecord {
    /// Source memory id.
    pub src_memory_id: String,
    /// Destination memory id.
    pub dst_memory_id: String,
    /// Link type.
    pub link_type: String,
    /// Link status.
    pub status: String,
    /// Optional link confidence.
    pub confidence: Option<f64>,
}

/// Deterministic search request.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchRequest {
    /// Full-text query for deterministic FTS5 search.
    pub query: String,
    /// Metadata filters.
    pub filters: SearchFilters,
    /// Maximum results to return.
    pub limit: usize,
    /// Number of ranked results to skip.
    pub offset: usize,
    /// Maximum snippet length in Unicode scalar values.
    pub snippet_chars: usize,
    /// Include full content in each result.
    pub include_content: bool,
    /// Include source/provenance JSON in each result.
    pub include_source: bool,
    /// Semantic fallback flag. Accepts `disabled` or `fallback`.
    pub semantic_fallback: String,
    /// Lexical fallback flag. Accepts `conservative` or `disabled`.
    pub lexical_fallback: String,
    /// Optional caller-provided query embedding for semantic fallback.
    pub embedding: Option<Vec<f32>>,
    /// Optional late-interaction query token embedding. When present (with
    /// `token_model_id`), exhaustive `MaxSim` selects semantic candidates.
    pub query_token_embedding: Option<Vec<Vec<f32>>>,
    /// Model id for `query_token_embedding`.
    pub token_model_id: Option<String>,
}

/// Deterministic memory review/list request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryListRequest {
    /// Metadata filters.
    pub filters: SearchFilters,
    /// Maximum rows to return.
    pub limit: usize,
    /// Number of ordered rows to skip.
    pub offset: usize,
    /// Maximum snippet length in Unicode scalar values.
    pub snippet_chars: usize,
    /// Include full content in each row.
    pub include_content: bool,
    /// Include source/provenance JSON in each row.
    pub include_source: bool,
    /// Deterministic sort order. v0.1 supports `updated_desc`, `observed_desc`, and `created_desc`.
    pub order: String,
}

/// Deterministic search metadata filters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SearchFilters {
    /// Space names. Defaults to `workspace-memory` when omitted.
    pub spaces: Vec<String>,
    /// Silo names.
    pub silos: Vec<String>,
    /// Scopes.
    pub scopes: Vec<String>,
    /// Project keys.
    pub projects: Vec<String>,
    /// Memory kinds.
    pub kinds: Vec<String>,
    /// Memory statuses. Defaults to `active` when omitted.
    pub statuses: Vec<String>,
    /// Tags; any tag matches.
    pub tags: Vec<String>,
    /// Entity keys.
    pub entity_keys: Vec<String>,
    /// Claim keys.
    pub claim_keys: Vec<String>,
    /// When true, exclude memories that are no longer valid: `valid_to` in the
    /// past or `expires_at` reached. Set on the search path so recall never
    /// surfaces logically stale facts; left false for review listings
    /// (memory-list) so stale memories stay visible for cleanup.
    pub hide_expired: bool,
}

/// Deterministic graph entity upsert request.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityUpsertRequest {
    /// Optional space. Defaults to `workspace-memory`.
    pub space: Option<String>,
    /// Space-local stable entity key.
    pub entity_key: String,
    /// Entity type label. Defaults to `Entity`.
    pub entity_type: Option<String>,
    /// Human-readable canonical name.
    pub canonical_name: String,
    /// Optional aliases to insert/update idempotently.
    pub aliases: Vec<String>,
    /// Entity status. Defaults to `active`.
    pub status: Option<String>,
    /// Confidence score from 0.0 to 1.0.
    pub confidence: f64,
    /// Optional existing source episode id in the same space.
    pub source_episode_id: Option<String>,
    /// Optional JSON metadata object.
    pub metadata_json: Option<String>,
    /// Include source episode ids in returned graph records.
    pub include_source: bool,
}

/// Deterministic graph entity upsert report.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityUpsertReport {
    /// Upsert strategy identifier.
    pub strategy: String,
    /// True when a new entity row was created.
    pub created: bool,
    /// Stored entity projection.
    pub entity: EntityRecord,
}

/// Deterministic graph relationship upsert request.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipUpsertRequest {
    /// Optional space. Defaults to `workspace-memory`.
    pub space: Option<String>,
    /// Existing subject entity id. Mutually exclusive with `subject_entity_key`.
    pub subject_entity_id: Option<String>,
    /// Existing subject entity key. Mutually exclusive with `subject_entity_id`.
    pub subject_entity_key: Option<String>,
    /// Relationship type label.
    pub relation_type: String,
    /// Existing object entity id. Mutually exclusive with `object_entity_key`.
    pub object_entity_id: Option<String>,
    /// Existing object entity key. Mutually exclusive with `object_entity_id`.
    pub object_entity_key: Option<String>,
    /// Optional evidence memory id in the same space.
    pub memory_id: Option<String>,
    /// Optional existing source episode id in the same space.
    pub source_episode_id: Option<String>,
    /// Relationship status. Defaults to `active`.
    pub status: Option<String>,
    /// Confidence score from 0.0 to 1.0.
    pub confidence: f64,
    /// Optional observation timestamp.
    pub observed_at: Option<String>,
    /// Optional valid-from timestamp.
    pub valid_from: Option<String>,
    /// Optional valid-to timestamp.
    pub valid_to: Option<String>,
    /// Optional JSON metadata object.
    pub metadata_json: Option<String>,
    /// Include source episode ids in returned graph records.
    pub include_source: bool,
}

/// Deterministic graph relationship upsert report.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipUpsertReport {
    /// Upsert strategy identifier.
    pub strategy: String,
    /// True when a new relationship row was created.
    pub created: bool,
    /// Stored relationship projection.
    pub relationship: RelationshipRecord,
}

/// Explicit, operator-invoked graph entity merge request.
///
/// Merges the `from` entity into the `into` entity within one space: active
/// relationships are relinked to `into` (collapsing duplicates and self-loops),
/// the merged-away identifiers are carried over as aliases of `into`, and the
/// `from` entity is tombstoned. This is a graph-projection operation and does
/// not rewrite `memories` anchors. There is no automatic merge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntityMergeRequest {
    /// Optional space. Defaults to `workspace-memory`.
    pub space: Option<String>,
    /// Source entity id (merged away). Mutually exclusive with `from_entity_key`.
    pub from_entity_id: Option<String>,
    /// Source entity key (merged away). Mutually exclusive with `from_entity_id`.
    pub from_entity_key: Option<String>,
    /// Target entity id (kept). Mutually exclusive with `into_entity_key`.
    pub into_entity_id: Option<String>,
    /// Target entity key (kept). Mutually exclusive with `into_entity_id`.
    pub into_entity_key: Option<String>,
    /// When true, compute the merge and report counts without committing.
    pub dry_run: bool,
    /// Include source episode ids in the returned target entity record.
    pub include_source: bool,
}

/// Deterministic graph entity merge report.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityMergeReport {
    /// Merge strategy identifier.
    pub strategy: String,
    /// True when the merge was computed but not committed.
    pub dry_run: bool,
    /// Key of the merged-away source entity.
    pub from_entity_key: String,
    /// Key of the surviving target entity.
    pub into_entity_key: String,
    /// Active relationships repointed onto the target.
    pub relationships_repointed: usize,
    /// Relationships tombstoned because the target already had an equivalent edge.
    pub relationships_tombstoned_duplicate: usize,
    /// Relationships tombstoned because the merge collapsed them into a self-loop.
    pub relationships_tombstoned_self_loop: usize,
    /// Newly added aliases on the target (merged-away key/name/aliases).
    pub aliases_added: usize,
    /// True when the source entity was tombstoned (always true on a committed merge).
    pub from_tombstoned: bool,
    /// The surviving target entity after the merge.
    pub into: EntityRecord,
}

/// Deterministic graph entity search request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EntitySearchRequest {
    /// Optional space. Defaults to `workspace-memory`.
    pub space: Option<String>,
    /// Optional substring query matched against entity key, canonical name, and aliases.
    pub query: Option<String>,
    /// Optional exact entity key lookup.
    pub entity_key: Option<String>,
    /// Optional entity type filters.
    pub entity_types: Vec<String>,
    /// Entity statuses. Defaults to `active` when omitted.
    pub statuses: Vec<String>,
    /// Maximum rows to return.
    pub limit: usize,
    /// Number of ordered rows to skip.
    pub offset: usize,
    /// Include source episode ids in returned graph records.
    pub include_source: bool,
}

/// Deterministic graph entity search report.
#[derive(Debug, Clone, PartialEq)]
pub struct EntitySearchReport {
    /// Search strategy identifier.
    pub strategy: String,
    /// Space searched.
    pub space: String,
    /// Bounded estimate/lower bound of matching rows around the returned page.
    pub total_estimate: usize,
    /// True when more ordered rows exist beyond returned rows.
    pub truncated: bool,
    /// Ordered entity rows.
    pub results: Vec<EntitySearchResult>,
}

/// One deterministic entity search result.
#[derive(Debug, Clone, PartialEq)]
pub struct EntitySearchResult {
    /// One-based position after offset.
    pub rank: usize,
    /// Projected entity record.
    pub entity: EntityRecord,
    /// Aliases that matched the query, if any.
    pub matched_aliases: Vec<String>,
}

/// Stored graph entity projection.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityRecord {
    /// Entity id.
    pub id: String,
    /// Space containing the entity.
    pub space: String,
    /// Space-local stable entity key.
    pub entity_key: String,
    /// Entity type label.
    pub entity_type: String,
    /// Human-readable canonical name.
    pub canonical_name: String,
    /// Entity status.
    pub status: String,
    /// Confidence score.
    pub confidence: f64,
    /// Aliases attached to the entity, deterministic order.
    pub aliases: Vec<String>,
    /// Optional source episode id, present only when the caller includes source.
    pub source_episode_id: Option<String>,
    /// Creation timestamp.
    pub created_at: String,
    /// Update timestamp.
    pub updated_at: String,
}

/// Deterministic graph neighbor traversal request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphNeighborsRequest {
    /// Optional space. Defaults to `workspace-memory`.
    pub space: Option<String>,
    /// Optional seed entity id. Mutually exclusive with `entity_key`.
    pub entity_id: Option<String>,
    /// Optional seed entity key. Mutually exclusive with `entity_id`.
    pub entity_key: Option<String>,
    /// Maximum traversal depth. Defaults to one in adapters and is capped by the store.
    pub depth: usize,
    /// Optional relationship type filters.
    pub relation_types: Vec<String>,
    /// Relationship statuses. Defaults to `active` when omitted.
    pub statuses: Vec<String>,
    /// Maximum relationship edges to return.
    pub max_edges: usize,
    /// Include tombstoned entities/relationships when explicitly requested.
    pub include_tombstoned: bool,
    /// Include source episode ids in returned graph records.
    pub include_source: bool,
}

/// Deterministic graph neighbor traversal report.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphNeighborsReport {
    /// Traversal strategy identifier.
    pub strategy: String,
    /// Seed entity.
    pub seed: EntityRecord,
    /// Requested traversal depth after validation.
    pub depth: usize,
    /// Maximum relationship edges requested after validation.
    pub max_edges: usize,
    /// True when relationship edges were omitted because of `max_edges`.
    pub truncated: bool,
    /// Entity records reached by returned edges, including the seed.
    pub entities: Vec<GraphEntityRecord>,
    /// Relationship edges returned by traversal.
    pub relationships: Vec<GraphRelationshipRecord>,
}

/// Deterministic graph-context request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphContextRequest {
    /// Optional space. Defaults to `workspace-memory`.
    pub space: Option<String>,
    /// Optional seed entity id. Mutually exclusive with `entity_key`.
    pub entity_id: Option<String>,
    /// Optional seed entity key. Mutually exclusive with `entity_id`.
    pub entity_key: Option<String>,
    /// Maximum traversal depth.
    pub depth: usize,
    /// Optional relationship type filters.
    pub relation_types: Vec<String>,
    /// Relationship statuses. Defaults to `active` when omitted.
    pub statuses: Vec<String>,
    /// Maximum relationship edges to inspect.
    pub max_edges: usize,
    /// Maximum unique memories to include in the context pack.
    pub max_memories: usize,
    /// Maximum emitted pack size in Unicode scalar values.
    pub max_chars: usize,
    /// Include tombstoned entities/relationships when explicitly requested.
    pub include_tombstoned: bool,
    /// Include source episode ids in returned graph records.
    pub include_source: bool,
}

/// Deterministic graph-context report.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphContextReport {
    /// Context strategy identifier.
    pub strategy: String,
    /// Bounded graph traversal used for context selection.
    pub graph: GraphNeighborsReport,
    /// Compact memory pack around graph evidence and reached entities.
    pub pack: PackReport,
    /// Relationship evidence memories considered, in deterministic order.
    pub evidence_memory_ids: Vec<String>,
    /// Active memories attached to reached entity keys, in deterministic order.
    pub entity_memory_ids: Vec<String>,
}

/// Entity reached by graph traversal.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphEntityRecord {
    /// Minimum hop depth from the seed.
    pub depth: usize,
    /// Projected entity record.
    pub entity: EntityRecord,
}

/// Relationship reached by graph traversal.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphRelationshipRecord {
    /// Hop depth of the subject entity from the seed.
    pub subject_depth: usize,
    /// Hop depth of the object entity from the seed.
    pub object_depth: usize,
    /// Projected relationship record.
    pub relationship: RelationshipRecord,
}

/// Stored graph relationship projection.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationshipRecord {
    /// Relationship id.
    pub id: String,
    /// Space containing the relationship.
    pub space: String,
    /// Subject entity id.
    pub subject_entity_id: String,
    /// Relationship type label.
    pub relation_type: String,
    /// Object entity id.
    pub object_entity_id: String,
    /// Optional evidence memory id.
    pub memory_id: Option<String>,
    /// Optional source episode id, present only when the caller includes source.
    pub source_episode_id: Option<String>,
    /// Relationship status.
    pub status: String,
    /// Confidence score.
    pub confidence: f64,
    /// Optional observation timestamp.
    pub observed_at: Option<String>,
    /// Optional valid-from timestamp.
    pub valid_from: Option<String>,
    /// Optional valid-to timestamp.
    pub valid_to: Option<String>,
    /// Creation timestamp.
    pub created_at: String,
    /// Update timestamp.
    pub updated_at: String,
}

/// Deterministic memory review/list report.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryListReport {
    /// Deterministic review/list strategy identifier.
    pub strategy: String,
    /// Bounded estimate/lower bound of matching rows around the returned page.
    pub total_estimate: usize,
    /// True when more ordered rows exist beyond returned rows.
    pub truncated: bool,
    /// Ordered memory rows.
    pub results: Vec<MemoryListItem>,
}

/// One deterministic memory review/list row.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryListItem {
    /// One-based position after offset.
    pub rank: usize,
    /// Memory id.
    pub memory_id: String,
    /// Active version id.
    pub version_id: String,
    /// Space name.
    pub space: String,
    /// Silo name.
    pub silo: String,
    /// Scope.
    pub scope: String,
    /// Optional project key.
    pub project_key: Option<String>,
    /// Memory kind.
    pub kind: String,
    /// Memory status.
    pub status: String,
    /// Optional summary.
    pub summary: Option<String>,
    /// Bounded snippet.
    pub snippet: String,
    /// Optional full content.
    pub content: Option<String>,
    /// Tags.
    pub tags: Vec<String>,
    /// Optional entity key.
    pub entity_key: Option<String>,
    /// Optional claim key.
    pub claim_key: Option<String>,
    /// Confidence score.
    pub confidence: f64,
    /// Whether memory is pinned.
    pub pinned: bool,
    /// Observation timestamp.
    pub observed_at: String,
    /// Creation timestamp.
    pub created_at: String,
    /// Update timestamp.
    pub updated_at: String,
    /// Optional source/provenance JSON.
    pub source_ref_json: Option<String>,
}

/// Deterministic search report.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchReport {
    /// Search strategy identifier.
    pub strategy: String,
    /// Semantic fallback was attempted.
    pub semantic_attempted: bool,
    /// Reason semantic fallback was skipped.
    pub semantic_reason: String,
    /// Bounded estimate/lower bound of matching rows around the returned page.
    pub total_estimate: usize,
    /// True when more ranked results exist beyond returned results.
    pub truncated: bool,
    /// Ranked results.
    pub results: Vec<SearchResult>,
}

/// One deterministic search result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// One-based rank after offset.
    pub rank: usize,
    /// Memory id.
    pub memory_id: String,
    /// Active version id.
    pub version_id: String,
    /// Overall deterministic score.
    pub score: f64,
    /// Score component details.
    pub scores: ScoreBreakdown,
    /// Space name.
    pub space: String,
    /// Silo name.
    pub silo: String,
    /// Scope.
    pub scope: String,
    /// Optional project key.
    pub project_key: Option<String>,
    /// Memory kind.
    pub kind: String,
    /// Memory status.
    pub status: String,
    /// Optional summary.
    pub summary: Option<String>,
    /// Bounded snippet.
    pub snippet: String,
    /// Optional full content.
    pub content: Option<String>,
    /// Tags.
    pub tags: Vec<String>,
    /// Optional entity key.
    pub entity_key: Option<String>,
    /// Optional claim key.
    pub claim_key: Option<String>,
    /// Observation timestamp.
    pub observed_at: String,
    /// Optional source/provenance JSON.
    pub source_ref_json: Option<String>,
    /// Optional metadata JSON (e.g. `verified_against`, `verified_at`).
    pub metadata_json: Option<String>,
}

/// Request to run multiple deterministic searches in one store read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSearchRequest {
    /// Queries to run.
    pub queries: Vec<BatchSearchQuery>,
    /// Common metadata filters applied to every query.
    pub common_filters: SearchFilters,
    /// Default maximum results per query.
    pub limit: usize,
    /// Number of ranked results to skip in each query.
    pub offset: usize,
    /// Maximum snippet length in Unicode scalar values.
    pub snippet_chars: usize,
    /// Include full content in each result.
    pub include_content: bool,
    /// Include source/provenance JSON in each result.
    pub include_source: bool,
    /// Semantic fallback flag. v0.1 accepts only `disabled`.
    pub semantic_fallback: String,
}

/// One query inside a deterministic batch-search request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchSearchQuery {
    /// Optional caller label.
    pub name: Option<String>,
    /// Full-text query.
    pub query: String,
    /// Optional per-query result limit, bounded by `MAX_BATCH_QUERY_LIMIT`.
    pub limit: Option<usize>,
}

/// Deterministic batch-search report.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchSearchReport {
    /// Per-query search reports.
    pub results: Vec<BatchSearchItemReport>,
}

/// One result set inside a batch-search report.
#[derive(Debug, Clone, PartialEq)]
pub struct BatchSearchItemReport {
    /// Optional caller label.
    pub name: Option<String>,
    /// Query text.
    pub query: String,
    /// Search report for this query.
    pub report: SearchReport,
}

/// Request to build a compact deterministic memory pack.
#[derive(Debug, Clone, PartialEq)]
pub struct PackRequest {
    /// Pack title.
    pub title: String,
    /// Query strings used to retrieve candidate memories.
    pub queries: Vec<String>,
    /// Metadata filters applied to every query.
    pub filters: SearchFilters,
    /// Maximum unique memories to include.
    pub max_memories: usize,
    /// Maximum emitted pack size in Unicode scalar values.
    pub max_chars: usize,
    /// Pack format. v0.1 accepts only `markdown`.
    pub format: String,
    /// Minimum final score a memory must reach to be included (precision floor).
    /// `0.0` disables the floor.
    pub min_score: f64,
    /// Candidate pool retrieved and reranked before truncating to `max_memories`.
    /// `0` reranks only the final `max_memories`. Effective only when a reranker
    /// is active; capped at `MAX_PACK_MEMORIES`.
    pub rerank_candidates: usize,
    /// Optional per-query embeddings for ANN retrieval.
    /// When provided for a query at index i, uses sqlite-vec ANN search instead of BM25 for that query.
    /// Length must match `queries.len()` or be empty (falls back to BM25 for all queries).
    pub query_embeddings: Option<Vec<Vec<f32>>>,
    /// Per-query late-interaction token embeddings (one token matrix per query).
    /// When present, exhaustive `MaxSim` replaces the ANN leg for candidate selection.
    pub query_token_embeddings: Option<Vec<Vec<Vec<f32>>>>,
    /// Model id for `query_token_embeddings`.
    pub token_model_id: Option<String>,
}

/// Within-entity memory-selection policy for graph expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphWithinEntitySelection {
    /// Existing pinned/recency order.
    Recency,
    /// First-query late-interaction `MaxSim` order.
    Maxsim,
}

/// Optional retrieval-expansion behavior for pack construction.
///
/// `PartialEq` only (not `Eq`): `graph_decay` is an `f64` knob, so the struct
/// cannot derive `Eq`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PackExpansionOptions {
    /// Expand each user query into deterministic subqueries before retrieval.
    pub query_expansion: bool,
    /// Add same-entity / same-claim neighbors from top pack-pool anchors.
    pub thread_expansion: bool,
    /// v0.3 associative recall: graph-expand the merged rerank candidate pool by
    /// traversing the entity/relationship graph one hop from the top anchored
    /// seeds, so a relationship-reachable memory below the ANN/BM25 threshold can
    /// still enter the candidate set. Applied AFTER `cos_top` is computed from the
    /// raw ANN pool, so graph-pulled items never influence the cosine gate. The
    /// cross-encoder still gates what lands in the pack. Strategy: `hybrid_assoc_v0`.
    pub graph_expansion: bool,
    /// Within-entity memory-selection policy for graph expansion.
    pub graph_within_entity_selection: GraphWithinEntitySelection,
    /// Maximum total query variants after deterministic expansion.
    pub max_query_variants: usize,
    /// Maximum anchor memories used for same-thread expansion.
    pub max_thread_seeds: usize,
    /// Maximum neighbor memories considered per anchor.
    pub max_thread_neighbors: usize,
    /// Maximum top-of-pool seeds anchored for graph expansion.
    pub max_graph_seeds: usize,
    /// Maximum graph-reachable neighbor memories unioned into the candidate pool
    /// (an activation-ordered budget, not a relevance cap).
    pub max_graph_neighbors: usize,
    /// Per-hop activation decay knob (`activation = seed_score * decay^hop *
    /// edge_weight`). Bounds/orders which graph-reachable memories get fetched and
    /// reranked; it is a candidate-budget allocator, never a relevance score.
    pub graph_decay: f64,
    /// v0.3.1 associative recall: reserve up to this many pack slots for the
    /// highest-activation graph-pulled candidates, letting a hop-reached memory the
    /// cross-encoder scored below the cut still land. `0` (default) keeps graph
    /// expansion recall-widening-only — byte-identical packs to graph-off ranking.
    pub graph_rerank_slots: usize,
    /// Minimum activation a graph candidate must reach to claim a reserved slot.
    /// Bounds precision so low-activation graph noise cannot take a slot.
    pub graph_activation_floor: f64,
}

impl Default for PackExpansionOptions {
    fn default() -> Self {
        Self {
            query_expansion: false,
            thread_expansion: false,
            graph_expansion: false,
            graph_within_entity_selection: GraphWithinEntitySelection::Recency,
            max_query_variants: MAX_BATCH_QUERIES,
            max_thread_seeds: 3,
            max_thread_neighbors: 3,
            max_graph_seeds: 3,
            max_graph_neighbors: 5,
            graph_decay: 0.5,
            graph_rerank_slots: 0,
            graph_activation_floor: 0.0,
        }
    }
}

/// Deterministic memory pack report.
#[derive(Debug, Clone, PartialEq)]
pub struct PackReport {
    /// Pack title.
    pub title: String,
    /// Pack format.
    pub format: String,
    /// Bounded pack content.
    pub content: String,
    /// Memory ids included in content, in display order.
    pub memory_ids: Vec<String>,
    /// Per-memory ranking score, aligned 1:1 with `memory_ids` (same order).
    /// Cross-encoder rerank score on the rerank path; retrieval score otherwise.
    pub scores: Vec<f64>,
    /// True when memories or text were omitted because of limits.
    pub truncated: bool,
    /// Highest relevance score among the candidates retrieved for this pack
    /// (cross-encoder rerank score on the rerank path, retrieval score on the
    /// BM25/ANN path). Reflects the retrieved candidate pool, not only the
    /// memories that survived the floor/char-budget. `None` for an empty pack.
    pub top_score: Option<f64>,
}

/// Request to write a deterministic logical JSONL export file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportRequest {
    /// Output JSONL file path. The file must not already exist.
    pub output_path: PathBuf,
    /// Export format. v0.1 accepts only `jsonl`.
    pub format: String,
}

/// Result of writing a deterministic logical JSONL export file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportReport {
    /// Output JSONL file path.
    pub output_path: PathBuf,
    /// Export format.
    pub format: String,
    /// Store schema version exported.
    pub schema_version: i32,
    /// Per-table row counts in export order.
    pub tables: Vec<ExportTableReport>,
    /// Total exported row records, excluding header/footer lines.
    pub row_count: u64,
    /// Output file size in bytes.
    pub bytes: u64,
    /// SHA-256 hex digest of the exported JSONL file.
    pub sha256: String,
}

/// Per-table export count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportTableReport {
    /// Table name.
    pub name: String,
    /// Rows exported from the table.
    pub rows: u64,
}

/// Request to create a consistent physical `SQLite` backup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackupRequest {
    /// Output `SQLite` backup file path. The file must not already exist.
    pub output_path: PathBuf,
    /// Backup format. v0.1 accepts only `sqlite`.
    pub format: String,
}

/// Result of creating a consistent physical `SQLite` backup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BackupReport {
    /// Output `SQLite` backup file path.
    pub output_path: PathBuf,
    /// Backup format.
    pub format: String,
    /// Store schema version backed up.
    pub schema_version: i32,
    /// `SQLite` page count in the backup file.
    pub page_count: i64,
    /// Output file size in bytes.
    pub bytes: u64,
    /// SHA-256 hex digest of the backup `SQLite` file.
    pub sha256: String,
}

/// Request to import a deterministic logical JSONL export into a new store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportRequest {
    /// Input JSONL export file path.
    pub input_path: PathBuf,
    /// Import format. v0.1 accepts only `jsonl`.
    pub format: String,
    /// Validate the archive without creating or modifying the target store.
    pub dry_run: bool,
    /// Conflict policy. v0.1 accepts only `fail_if_exists`.
    pub conflict_policy: String,
}

/// Result of importing or validating a deterministic logical JSONL export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportReport {
    /// Input JSONL export file path.
    pub input_path: PathBuf,
    /// Import format.
    pub format: String,
    /// Store schema version imported.
    pub schema_version: i32,
    /// True when the archive was validated without creating the target store.
    pub dry_run: bool,
    /// Per-table row counts in import/export order.
    pub tables: Vec<ExportTableReport>,
    /// Total imported row records, excluding header/footer lines.
    pub row_count: u64,
    /// Input file size in bytes.
    pub bytes: u64,
    /// SHA-256 hex digest of the input JSONL file.
    pub sha256: String,
    /// Rows rebuilt into `memory_fts`.
    pub fts_memory_rows: i64,
    /// Rows rebuilt into `source_episode_fts`.
    pub fts_source_episode_rows: i64,
}

/// Request for explicit Rust-core maintenance/dream work.
#[derive(Debug, Clone, PartialEq)]
pub struct DreamRequest {
    /// Optional space to limit expiry and dedupe scans. Reindex rebuilds global FTS tables.
    pub space: Option<String>,
    /// Optional silo filters for expiry/dedupe. Requires `space` in v0.1.
    pub silos: Vec<String>,
    /// Maintenance tasks. Empty defaults to `expire`, `reindex`, and `dedupe`.
    pub tasks: Vec<String>,
    /// Maximum memories or duplicate groups scanned by each bounded task.
    pub max_memories: usize,
    /// Validate/report without committing writes.
    pub dry_run: bool,
    /// Allow automatic expiry of pinned memories whose `expires_at` is reached.
    pub include_pinned: bool,
    /// Minimum `retrieved` recall count that promotes a short-term memory to
    /// durable. Must be >= 1.
    pub promote_threshold: usize,
    /// Minimum per-memory rerank score for a `retrieved` event to count.
    pub promote_score_floor: f64,
    /// Maximum rank for a `retrieved` event to count. Must be >= 1.
    pub promote_rank_cap: usize,
}

/// Result of one explicit maintenance/dream run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamReport {
    /// Dream run id. Dry runs return an id but do not journal it.
    pub run_id: String,
    /// Optional requested space.
    pub space: Option<String>,
    /// Optional requested silos.
    pub silos: Vec<String>,
    /// Normalized tasks run in deterministic order.
    pub tasks: Vec<String>,
    /// Run status.
    pub status: String,
    /// Start timestamp.
    pub started_at: String,
    /// Finish timestamp.
    pub finished_at: String,
    /// True when no writes were committed.
    pub dry_run: bool,
    /// True when this run was recorded in `dream_runs`.
    pub journaled: bool,
    /// Maximum scan bound used by the run.
    pub max_memories: usize,
    /// Expiry task report.
    pub expire: DreamExpireReport,
    /// Promotion task report.
    pub promote: DreamPromoteReport,
    /// Reindex task report.
    pub reindex: DreamReindexReport,
    /// Exact duplicate proposal report.
    pub dedupe: DreamDedupeReport,
    /// Graph projection diagnostics report.
    pub graph: DreamGraphReport,
    /// Shared-tag cross-entity linking report.
    pub link: DreamLinkReport,
}

/// Expiry task report for a dream run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamExpireReport {
    /// Whether the task ran.
    pub attempted: bool,
    /// Candidate memories scanned within the bound.
    pub scanned: usize,
    /// Memories expired, or that would expire in dry-run mode.
    pub expired: usize,
    /// Pinned memories skipped by default.
    pub skipped_pinned: usize,
    /// True when more candidates existed beyond `max_memories`.
    pub truncated: bool,
    /// Memory ids expired or that would expire.
    pub memory_ids: Vec<String>,
    /// Pinned memory ids skipped.
    pub skipped_pinned_ids: Vec<String>,
}

/// Promotion task report for a dream run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamPromoteReport {
    /// Whether the task ran.
    pub attempted: bool,
    /// Short-term candidates scanned within the bound.
    pub scanned: usize,
    /// Memories promoted, or that would promote in dry-run mode.
    pub promoted: usize,
    /// True when more candidates existed beyond `max_memories`.
    pub truncated: bool,
    /// Memory ids promoted or that would promote.
    pub memory_ids: Vec<String>,
}

/// FTS reindex task report for a dream run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamReindexReport {
    /// Whether the task ran.
    pub attempted: bool,
    /// Rows rebuilt or that would be rebuilt into memory FTS tables.
    pub memory_rows: i64,
    /// Rows rebuilt or that would be rebuilt into source episode FTS.
    pub source_episode_rows: i64,
}

/// Exact duplicate proposal task report for a dream run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamDedupeReport {
    /// Whether the task ran.
    pub attempted: bool,
    /// Exact duplicate groups proposed for review.
    pub proposals: Vec<DreamDuplicateProposal>,
    /// True when more duplicate groups existed beyond `max_memories`.
    pub truncated: bool,
}

/// Result of the `link` dream task: cross-entity `memory_links` written between
/// memories that share discriminative tags, so the graph projection (`graph` task)
/// can bridge curated memories for associative recall.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamLinkReport {
    /// Whether the task ran.
    pub attempted: bool,
    /// Cross-entity shared-tag memory-pair links eligible to write (after the
    /// discriminative-tag filter and the per-run cap).
    pub candidates: usize,
    /// Links actually written this run (0 on a dry run; already-present links are
    /// ignored idempotently and not counted).
    pub links_written: usize,
    /// True when more eligible pairs existed beyond the per-run cap.
    pub truncated: bool,
}

/// Graph projection diagnostics task report for a dream run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamGraphReport {
    /// Whether the task ran.
    pub attempted: bool,
    /// Entities with no active memory and no relationships.
    pub orphan_entities: usize,
    /// Relationship rows whose endpoint entity is missing, outside the relationship
    /// space, or non-active (`merged`/`tombstoned`).
    pub dangling_relationships: usize,
    /// Relationship rows with missing, inactive, or cross-space evidence memory ids.
    pub inactive_evidence_relationships: usize,
    /// Keyed active memory subjects lacking any matching entity projection row.
    pub missing_entity_projections: Vec<DreamEntityProjection>,
    /// Proposed relationship candidates derived from explicit memory links.
    pub relationship_proposals: Vec<DreamRelationshipProposal>,
    /// True when any diagnostic/proposal list was bounded by `max_memories`.
    pub truncated: bool,
    /// Bounded orphan entity ids.
    pub orphan_entity_ids: Vec<String>,
    /// Bounded dangling relationship ids.
    pub dangling_relationship_ids: Vec<String>,
    /// Bounded inactive evidence relationship ids.
    pub inactive_evidence_relationship_ids: Vec<String>,
}

/// One missing graph entity projection derived from canonical memory identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamEntityProjection {
    /// Space containing the keyed active memory.
    pub space: String,
    /// Canonical memory entity key that should have an active entity row.
    pub entity_key: String,
}

/// One relationship proposal derived from an explicit memory link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamRelationshipProposal {
    /// Space containing the linked memories/entities.
    pub space: String,
    /// Source memory id carrying the subject entity key.
    pub src_memory_id: String,
    /// Destination memory id carrying the object entity key.
    pub dst_memory_id: String,
    /// Link type mapped directly to the relationship type.
    pub link_type: String,
    /// Proposed subject entity key.
    pub subject_entity_key: String,
    /// Proposed relationship type.
    pub relation_type: String,
    /// Proposed object entity key.
    pub object_entity_key: String,
    /// Number of active `memory_links` supporting this edge (evidence multiplicity).
    /// Drives the materialized relationship's confidence so a well-co-occurring
    /// association out-weights an incidental one in graph-expansion activation.
    pub evidence_count: usize,
}

/// One exact duplicate proposal. The dream run does not auto-merge v0.1 duplicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamDuplicateProposal {
    /// Space containing the duplicate group.
    pub space: String,
    /// SHA-256 of identical active memory content.
    pub content_sha256: String,
    /// Deterministic canonical candidate id.
    pub canonical_memory_id: String,
    /// Other duplicate memory ids, bounded for output.
    pub duplicate_memory_ids: Vec<String>,
    /// Total active memories in the duplicate group.
    pub total_count: usize,
    /// Number of pinned memories in the group.
    pub pinned_count: usize,
    /// True when duplicate ids were omitted because of output bounds.
    pub duplicate_ids_truncated: bool,
}

/// Deterministic search score component details.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoreBreakdown {
    /// FTS/BM25-derived score.
    pub fts: f64,
    /// Metadata/key/tag boost score.
    pub metadata: f64,
    /// Recency score.
    pub recency: f64,
    /// Scope boost score.
    pub scope: f64,
    /// Status boost score.
    pub status: f64,
    /// Pin boost score.
    pub pin: f64,
    /// Source-trust tier boost: explicit/manual writes over auto-harvest.
    pub source_tier: f64,
}

/// Top-level deterministic store statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stats {
    /// Store path that was inspected.
    pub path: PathBuf,
    /// Store file size in bytes, excluding `-wal` and `-shm` sidecar files.
    pub database_bytes: u64,
    /// Schema version from `PRAGMA user_version`.
    pub schema_version: i32,
    /// Protocol version recorded in store metadata.
    pub protocol_version: String,
    /// Runtime `SQLite` library version.
    pub sqlite_version: String,
    /// Current journal mode reported by `SQLite`.
    pub journal_mode: String,
    /// Number of configured spaces.
    pub space_count: i64,
    /// Number of configured silos.
    pub silo_count: i64,
    /// Number of memory headers in the canonical store.
    pub memory_count: i64,
    /// Number of active memory headers in the canonical store.
    pub active_count: i64,
    /// Number of source episode rows in the canonical store.
    pub source_episode_count: i64,
    /// Per-space memory counts.
    pub spaces: Vec<SpaceStats>,
    /// Optional index/queue diagnostics.
    pub indexes: Option<IndexStats>,
    /// Optional memory-governance health rollup (opt-in; see `--health`).
    pub health: Option<HealthStats>,
}

/// Memory-governance health rollup: lifecycle and provenance hygiene signals
/// for review and observability. Opt-in because it runs extra aggregate
/// queries; absent from default `stats` output. Backup recency is intentionally
/// not included: backups write a separate file and leave no row in the store,
/// so there is nothing to report from here without lying.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthStats {
    /// Active memories (the live, retrievable set).
    pub active: i64,
    /// Superseded memories (replaced by a newer claim; retained as history).
    pub superseded: i64,
    /// Conflicted memories surfaced for review.
    pub conflicted: i64,
    /// Tombstoned (soft-deleted) memories.
    pub tombstoned: i64,
    /// Expired memories (retention policy reached).
    pub expired: i64,
    /// Active memories missing an `entity_key` and/or `claim_key`, so they do
    /// not participate in entity/claim grouping or auto-supersession.
    pub active_without_keys: i64,
    /// Active keyed memories without any matching entity row, excluding
    /// deliberately merged or tombstoned projections.
    pub active_missing_entity_projection: i64,
    /// Active `(entity_key, claim_key)` groups with more than one member, i.e.
    /// duplicates that should have collapsed via supersession (expect ~0).
    pub duplicate_key_groups: i64,
    /// Active memories still in the short-term silo (promotion backlog).
    pub short_term_active: i64,
    /// Active memories whose `valid_to` is in the past (logically stale).
    pub active_past_valid_to: i64,
    /// Active memories with no `ready` embedding (semantic-retrieval blind spots).
    pub active_without_embedding: i64,
    /// Timestamp of the most recent `ready` embedding, or `None` if none exist.
    pub last_embedding_at: Option<String>,
    /// Candidate memories awaiting review (0 if the candidate queue is unused).
    pub candidates_pending: i64,
    /// Candidate memories that were approved into the store.
    pub candidates_approved: i64,
    /// Candidate memories that were rejected.
    pub candidates_rejected: i64,
}

/// Per-space deterministic memory statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpaceStats {
    /// Space name.
    pub name: String,
    /// Total memory count in the space.
    pub memory_count: i64,
    /// Active memory count in the space.
    pub active_count: i64,
}

/// Deterministic index and queue statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexStats {
    /// Rows in the memory FTS projection table.
    pub fts_memory_rows: i64,
    /// Rows in the source episode FTS projection table.
    pub fts_source_episode_rows: i64,
    /// Sets of document chunks sharing identical content across sources
    /// (exact-content duplicate clusters; see `document-duplicates`). A nudge
    /// that the store holds duplicates the user may want to prune.
    pub document_duplicate_clusters: i64,
    /// Queued or running background jobs.
    pub pending_jobs: i64,
}
