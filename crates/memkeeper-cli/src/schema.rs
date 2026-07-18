//! Per-command JSON payload schema docs surfaced by the `schema` subcommand.
//!
//! Every command that accepts a `--json '{...}'` payload is described here:
//! required fields, optional fields with defaults, and a runnable example.
//! The descriptors are kept honest by tests that parse each example (and a
//! full all-fields payload) through the real request parsers, so documentation
//! drift fails `cargo test` instead of misleading a user at the CLI.

use crate::output::json_string;

/// One field in a command's JSON payload.
pub(crate) struct FieldDoc {
    pub(crate) name: &'static str,
    /// Wire type: `string`, `string[]`, `int`, `float`, `bool`, `object`,
    /// `float[]`, `float[][]`, or `object[]`.
    pub(crate) ty: &'static str,
    pub(crate) required: bool,
    /// Default applied when the field is omitted (optional fields only).
    pub(crate) default: Option<&'static str>,
    pub(crate) note: &'static str,
}

/// The accepted JSON payload for one command.
pub(crate) struct CommandSchema {
    pub(crate) command: &'static str,
    pub(crate) summary: &'static str,
    pub(crate) fields: &'static [FieldDoc],
    /// A minimal, valid example payload (must parse through the real parser).
    pub(crate) example: &'static str,
}

const fn req(name: &'static str, ty: &'static str, note: &'static str) -> FieldDoc {
    FieldDoc {
        name,
        ty,
        required: true,
        default: None,
        note,
    }
}

const fn opt(
    name: &'static str,
    ty: &'static str,
    default: Option<&'static str>,
    note: &'static str,
) -> FieldDoc {
    FieldDoc {
        name,
        ty,
        required: false,
        default,
        note,
    }
}

/// `filters` / `common_filters` object subfields, shared by search/memory-list/pack.
/// All are optional string arrays that AND-narrow the candidate set.
const FILTERS_NOTE: &str =
    "object; subfields (all string[]): spaces, silos, scopes, projects, kinds, statuses, tags, entity_keys, claim_keys";

pub(crate) static COMMAND_SCHEMAS: &[CommandSchema] = &[
    CommandSchema {
        command: "space-create",
        summary: "Create a space and its default silos.",
        fields: &[
            req("name", "string", "Space identifier."),
            opt("display_name", "string", None, "Human-friendly label."),
            opt("description", "string", None, "Space description."),
            opt("default_silo", "string", None, "Default silo for new memories."),
            opt("ontology", "string", None, "Named ontology to attach."),
            opt("config", "object", None, "Space config (alias: config_json)."),
            opt("if_not_exists", "bool", Some("false"), "Succeed if the space already exists."),
        ],
        example: r#"{"name":"workspace-memory","default_silo":"durable","if_not_exists":true}"#,
    },
    CommandSchema {
        command: "silo-list",
        summary: "List silos in a space.",
        fields: &[opt("space", "string", None, "Space to list (defaults to the active space).")],
        example: r#"{"space":"workspace-memory"}"#,
    },
    CommandSchema {
        command: "remember",
        summary: "Store one explicit memory.",
        fields: &[
            req("content", "string", "The memory text (one atomic fact)."),
            opt("space", "string", None, "Target space."),
            opt("silo", "string", None, "Target silo (e.g. durable, volatile)."),
            opt("scope", "string", None, "Scope (e.g. workspace, project)."),
            opt("project", "string", None, "Project key when scope is project."),
            opt("kind", "string", None, "Memory kind (fact, decision, preference, ...)."),
            opt("summary", "string", None, "Optional short summary."),
            opt(
                "retrieval_representation",
                "object",
                None,
                "Source-aware retrieval companion: {kind: contextual-card-v1, text: <=512 chars}. CLI/serve only.",
            ),
            opt("tags", "string[]", Some("[]"), "Free-form tags."),
            opt("entity_key", "string", None, "Grouping key; derived if omitted."),
            opt("claim_key", "string", None, "Claim key for supersession grouping."),
            opt("confidence", "float", Some("1.0"), "Confidence 0.0-1.0."),
            opt("observed_at", "string", None, "ISO-8601 UTC when observed."),
            opt("valid_from", "string", None, "ISO-8601 UTC validity start."),
            opt("valid_to", "string", None, "ISO-8601 UTC validity end."),
            opt("expires_at", "string", None, "ISO-8601 UTC expiry."),
            opt("source", "object", None, "Provenance object; may carry source_episode_id, plus source_type (explicit-user|auto-harvest|import|docs|test|assistant-inference) and sensitivity (normal|sensitive)."),
            opt("metadata_json", "string", None, "Opaque metadata JSON string."),
            opt("pinned", "bool", Some("false"), "Pin against expiry/dedupe."),
            opt("supersedes", "string[]", Some("[]"), "Memory ids this one supersedes."),
            opt("contradicts", "string[]", Some("[]"), "Memory ids this one contradicts."),
            opt("embedding", "float[]", None, "Precomputed embedding vector."),
            opt("embedding_model_id", "string", None, "Model id for the supplied embedding."),
            opt("dry_run", "bool", Some("false"), "Validate without writing."),
            opt("mode", "string", Some("\"auto\""), "Supersession vs same entity/claim memories: auto | append | supersede | suggest | conflict."),
        ],
        example: r#"{"content":"prefer tabs over spaces in code","kind":"preference","tags":["writing"]}"#,
    },
    CommandSchema {
        command: "search",
        summary: "Search memories (semantic-primary, BM25/FTS fallback).",
        fields: &[
            req("query", "string", "Search text."),
            opt("filters", "object", None, FILTERS_NOTE),
            opt("limit", "int", Some("10"), "Max results."),
            opt("offset", "int", Some("0"), "Result offset."),
            opt("snippet_chars", "int", Some("240"), "Snippet length."),
            opt("include_content", "bool", Some("false"), "Return full content."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
            opt("semantic_fallback", "string", Some("\"fallback\""), "fallback | disabled."),
            opt("lexical_fallback", "string", Some("\"conservative\""), "conservative | disabled."),
            opt("embedding", "float[]", None, "Precomputed query embedding."),
            opt("rerank", "bool", Some("false"), "Cross-encoder rerank the pool."),
            opt("rerank_candidates", "int", Some("16"), "Candidate pool size when reranking."),
        ],
        example: r#"{"query":"memkeeper roadmap","limit":5}"#,
    },
    CommandSchema {
        command: "entity-upsert",
        summary: "Create or update a graph entity.",
        fields: &[
            req("entity_key", "string", "Stable entity key."),
            req("canonical_name", "string", "Canonical display name."),
            opt("space", "string", None, "Target space."),
            opt("entity_type", "string", None, "Entity type (person, tool, ...)."),
            opt("aliases", "string[]", Some("[]"), "Alternate names."),
            opt("status", "string", None, "Entity status."),
            opt("confidence", "float", Some("1.0"), "Confidence 0.0-1.0."),
            opt("source_episode_id", "string", None, "Originating episode id."),
            opt("metadata", "object", None, "Entity metadata (alias: metadata_json)."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
        ],
        example: r#"{"entity_key":"tool-memkeeper","canonical_name":"memkeeper","entity_type":"tool"}"#,
    },
    CommandSchema {
        command: "relationship-upsert",
        summary: "Create or update a graph relationship. Identify each endpoint by *_entity_id or *_entity_key.",
        fields: &[
            req("relation_type", "string", "Relation type (e.g. uses, authored)."),
            opt("space", "string", None, "Target space."),
            opt("subject_entity_id", "string", None, "Subject by id (or use subject_entity_key)."),
            opt("subject_entity_key", "string", None, "Subject by key."),
            opt("object_entity_id", "string", None, "Object by id (or use object_entity_key)."),
            opt("object_entity_key", "string", None, "Object by key."),
            opt("memory_id", "string", None, "Memory id evidencing the edge."),
            opt("source_episode_id", "string", None, "Originating episode id."),
            opt("status", "string", None, "Relationship status."),
            opt("confidence", "float", Some("1.0"), "Confidence 0.0-1.0."),
            opt("observed_at", "string", None, "ISO-8601 UTC when observed."),
            opt("valid_from", "string", None, "ISO-8601 UTC validity start."),
            opt("valid_to", "string", None, "ISO-8601 UTC validity end."),
            opt("metadata", "object", None, "Edge metadata (alias: metadata_json)."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
        ],
        example: r#"{"subject_entity_key":"person-alice","relation_type":"uses","object_entity_key":"tool-memkeeper"}"#,
    },
    CommandSchema {
        command: "entity-merge",
        summary: "Merge one graph entity into another (relink edges + tombstone). Identify each side by *_entity_id or *_entity_key.",
        fields: &[
            opt("space", "string", None, "Target space."),
            opt("from_entity_id", "string", None, "Source by id (or from_entity_key)."),
            opt("from_entity_key", "string", None, "Source by key."),
            opt("into_entity_id", "string", None, "Target by id (or into_entity_key)."),
            opt("into_entity_key", "string", None, "Target by key."),
            opt("dry_run", "bool", Some("false"), "Preview the merge without applying."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
        ],
        example: r#"{"from_entity_key":"tool-memkeeper-dup","into_entity_key":"tool-memkeeper","dry_run":true}"#,
    },
    CommandSchema {
        command: "entity-search",
        summary: "Search graph entities.",
        fields: &[
            opt("space", "string", None, "Target space."),
            opt("query", "string", None, "Text query."),
            opt("entity_key", "string", None, "Exact entity key."),
            opt("entity_types", "string[]", Some("[]"), "Filter by entity types."),
            opt("statuses", "string[]", Some("[]"), "Filter by statuses."),
            opt("limit", "int", Some("20"), "Max results."),
            opt("offset", "int", Some("0"), "Result offset."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
        ],
        example: r#"{"query":"memkeeper","limit":10}"#,
    },
    CommandSchema {
        command: "graph-neighbors",
        summary: "Traverse graph neighbors of an entity. Identify the seed by entity_id or entity_key.",
        fields: &[
            opt("space", "string", None, "Target space."),
            opt("entity_id", "string", None, "Seed by id (or entity_key)."),
            opt("entity_key", "string", None, "Seed by key."),
            opt("depth", "int", Some("1"), "Traversal depth."),
            opt("relation_types", "string[]", Some("[]"), "Restrict to relation types."),
            opt("statuses", "string[]", Some("[]"), "Restrict to statuses."),
            opt("max_edges", "int", Some("50"), "Max edges returned."),
            opt("include_tombstoned", "bool", Some("false"), "Include tombstoned rows."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
        ],
        example: r#"{"entity_key":"tool-memkeeper","depth":2}"#,
    },
    CommandSchema {
        command: "graph-context",
        summary: "Build an entity-centered memory context pack. Identify the seed by entity_id or entity_key.",
        fields: &[
            opt("space", "string", None, "Target space."),
            opt("entity_id", "string", None, "Seed by id (or entity_key)."),
            opt("entity_key", "string", None, "Seed by key."),
            opt("depth", "int", Some("1"), "Traversal depth."),
            opt("relation_types", "string[]", Some("[]"), "Restrict to relation types."),
            opt("statuses", "string[]", Some("[]"), "Restrict to statuses."),
            opt("max_edges", "int", Some("50"), "Max edges traversed."),
            opt("max_memories", "int", Some("10"), "Max memories included."),
            opt("max_chars", "int", Some("4000"), "Context character budget."),
            opt("include_tombstoned", "bool", Some("false"), "Include tombstoned rows."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
        ],
        example: r#"{"entity_key":"tool-memkeeper","max_memories":8}"#,
    },
    CommandSchema {
        command: "memory-list",
        summary: "List recent memories for review.",
        fields: &[
            opt("filters", "object", None, FILTERS_NOTE),
            opt("limit", "int", Some("20"), "Max results."),
            opt("offset", "int", Some("0"), "Result offset."),
            opt("snippet_chars", "int", Some("240"), "Snippet length."),
            opt("include_content", "bool", Some("false"), "Return full content."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
            opt("order", "string", Some("\"updated_desc\""), "updated_desc | updated_asc | created_desc | created_asc."),
        ],
        example: r#"{"limit":10,"order":"updated_desc"}"#,
    },
    CommandSchema {
        command: "batch-search",
        summary: "Run multiple searches in one call.",
        fields: &[
            req("queries", "object[]", "Array of {query (required), name?, limit?}."),
            opt("common_filters", "object", None, FILTERS_NOTE),
            opt("limit", "int", Some("10"), "Default max results per query."),
            opt("offset", "int", Some("0"), "Result offset."),
            opt("snippet_chars", "int", Some("240"), "Snippet length."),
            opt("include_content", "bool", Some("false"), "Return full content."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
            opt("semantic_fallback", "string", Some("\"disabled\""), "fallback | disabled."),
        ],
        example: r#"{"queries":[{"name":"roadmap","query":"memkeeper roadmap","limit":3},{"query":"dream flags"}]}"#,
    },
    CommandSchema {
        command: "pack",
        summary: "Build a compact memory pack for injection.",
        fields: &[
            req("title", "string", "Pack title."),
            req("queries", "string[]", "One or more query strings."),
            opt("filters", "object", None, FILTERS_NOTE),
            opt("max_memories", "int", Some("10"), "Max memories included."),
            opt("max_chars", "int", Some("6000"), "Character budget."),
            opt("format", "string", Some("\"markdown\""), "Output format."),
            opt("min_score", "float", Some("0.0"), "Return an empty pack when the top final score is below this threshold."),
            opt("rerank_candidates", "int", Some("0"), "Candidate pool for rerank (0 = final pack width)."),
            opt("query_embeddings", "float[][]", None, "Precomputed per-query embeddings."),
        ],
        example: r#"{"title":"memkeeper context","queries":["roadmap","dream flags"],"max_memories":5}"#,
    },
    CommandSchema {
        command: "pool-trace",
        summary: "Diagnose the exact ID-only pre-rerank candidate pool.",
        fields: &[
            req("title", "string", "Trace title."),
            req("queries", "string[]", "One or more query strings."),
            opt("filters", "object", None, FILTERS_NOTE),
            opt("max_memories", "int", Some("10"), "Final pack width used to resolve the pool width."),
            opt("max_chars", "int", Some("6000"), "Validated for pack compatibility; content is never returned."),
            opt("format", "string", Some("\"markdown\""), "Validated pack format."),
            opt("min_score", "float", Some("0.0"), "Accepted for exact request replay; ignored before reranking."),
            opt("rerank_candidates", "int", Some("0"), "Requested candidate-pool width."),
            opt("max_graph_seeds", "int", Some("4"), "Graph-expansion seed count."),
            opt("max_graph_neighbors", "int", Some("4"), "Evidence-backed graph neighbor budget."),
            opt("graph_decay", "float", Some("0.5"), "Graph activation decay."),
        ],
        example: r#"{"title":"candidate admission","queries":["memkeeper roadmap"],"max_memories":5,"rerank_candidates":16}"#,
    },
    CommandSchema {
        command: "get",
        summary: "Fetch one memory by id.",
        fields: &[
            req("id", "string", "Memory id."),
            opt("include_history", "bool", Some("false"), "Include version history."),
            opt("include_links", "bool", Some("true"), "Include links."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
        ],
        example: r#"{"id":"mem_...","include_history":true}"#,
    },
    CommandSchema {
        command: "forget",
        summary: "Tombstone one memory.",
        fields: &[
            req("id", "string", "Memory id."),
            opt("reason", "string", None, "Audit reason."),
            opt("mode", "string", Some("\"tombstone\""), "tombstone (only durable mode today)."),
            opt("dry_run", "bool", Some("false"), "Preview without tombstoning."),
        ],
        example: r#"{"id":"mem_...","reason":"superseded"}"#,
    },
    CommandSchema {
        command: "verify",
        summary: "Stamp verification provenance onto one memory.",
        fields: &[
            req("memory_id", "string", "Memory id to verify."),
            opt("verified_against", "string", None, "What it was checked against."),
            opt("now", "string", None, "ISO-8601 UTC timestamp (defaults to store time)."),
        ],
        example: r#"{"memory_id":"mem_...","verified_against":"10-K filing"}"#,
    },
    CommandSchema {
        command: "recall-log",
        summary: "Record recall events for retrieved memories.",
        fields: &[
            req("events", "object[]", "Array of {memory_id (req), kind (req), query?, rank?, score?}."),
            opt("source", "string", None, "Event source label."),
            opt("session_id", "string", None, "Session identifier."),
            opt("batch_id", "string", None, "Recall-log batch identifier shared by events."),
            opt("latency_ms", "float", None, "Caller-observed retrieval latency for this batch."),
            opt("latency_source", "string", None, "Label describing what latency_ms measured."),
            opt("touch_accessed", "bool", Some("true"), "Update last-accessed timestamps."),
        ],
        example: r#"{"source":"hook","events":[{"memory_id":"mem_...","kind":"retrieved","rank":1}]}"#,
    },
    CommandSchema {
        command: "history",
        summary: "Show a memory's versions and events.",
        fields: &[
            req("id", "string", "Memory id."),
            opt("limit", "int", Some("50"), "Max versions/events."),
            opt("include_source", "bool", Some("false"), "Return provenance JSON."),
        ],
        example: r#"{"id":"mem_...","limit":20}"#,
    },
    CommandSchema {
        command: "export",
        summary: "Write a logical JSONL export.",
        fields: &[
            req("output", "string", "Output path (alias: out)."),
            opt("format", "string", Some("\"jsonl\""), "Export format."),
        ],
        example: r#"{"output":"export.jsonl"}"#,
    },
    CommandSchema {
        command: "import",
        summary: "Import a logical JSONL export into a store.",
        fields: &[
            req("input", "string", "Input path (alias: in)."),
            opt("format", "string", Some("\"jsonl\""), "Import format."),
            opt("dry_run", "bool", Some("false"), "Validate without writing."),
            opt("conflict_policy", "string", Some("\"fail_if_exists\""), "fail_if_exists | skip | overwrite."),
        ],
        example: r#"{"input":"export.jsonl","dry_run":true}"#,
    },
    CommandSchema {
        command: "dream",
        summary: "Run bounded maintenance tasks. Explicit CLI flags (--task, --dry-run, ...) override these JSON fields.",
        fields: &[
            opt("space", "string", None, "Restrict to one space."),
            opt("silos", "string[]", Some("[]"), "Restrict to silos."),
            opt("tasks", "string[]", Some("[] (all)"), "Subset of: promote, expire, reindex, dedupe, link, graph."),
            opt("max_memories", "int", Some("1000"), "Max memories scanned."),
            opt("dry_run", "bool", Some("false"), "Preview without journaling/mutating."),
            opt("include_pinned", "bool", Some("false"), "Include pinned memories."),
            opt("promote_threshold", "int", Some("3"), "Recall count to promote."),
            opt("promote_score_floor", "float", Some("0.5"), "Min score to promote."),
            opt("promote_rank_cap", "int", Some("3"), "Max rank to promote."),
        ],
        example: r#"{"tasks":["dedupe"],"dry_run":true}"#,
    },
    CommandSchema {
        command: "backup",
        summary: "Create a physical SQLite backup.",
        fields: &[
            req("output", "string", "Output path (alias: out)."),
            opt("format", "string", Some("\"sqlite\""), "Backup format."),
        ],
        example: r#"{"output":"backup.sqlite"}"#,
    },
    CommandSchema {
        command: "candidate-submit",
        summary: "Submit a candidate memory for review (promoted into a memory only on approval).",
        fields: &[
            req("content", "string", "Candidate memory text."),
            opt("space", "string", None, "Target space (applied on approval)."),
            opt("silo", "string", None, "Target silo."),
            opt("scope", "string", None, "Target scope."),
            opt("project", "string", None, "Project key."),
            opt("kind", "string", None, "Memory kind."),
            opt("summary", "string", None, "Optional summary."),
            opt("rationale", "string", None, "Why this candidate was proposed."),
            opt("tags", "string[]", Some("[]"), "Tags applied on approval."),
            opt("entity_key", "string", None, "Stable entity key."),
            opt("claim_key", "string", None, "Claim key."),
            opt("confidence", "float", Some("1.0"), "Confidence 0.0-1.0."),
            opt("source_type", "string", Some("\"assistant-inference\""), "explicit-user | auto-harvest | import | docs | test | assistant-inference."),
            opt("source", "object", None, "Provenance object."),
            opt("sensitivity", "string", Some("\"normal\""), "normal | sensitive."),
            opt("supersedes", "string[]", Some("[]"), "Memory ids superseded on approval."),
            opt("dry_run", "bool", Some("false"), "Validate without writing."),
        ],
        example: r#"{"content":"prefer tabs over spaces in code","kind":"preference","source_type":"explicit-user"}"#,
    },
    CommandSchema {
        command: "candidate-list",
        summary: "List candidate memories, newest first.",
        fields: &[
            opt("status", "string", None, "Filter: pending | approved | rejected (omit for all)."),
            opt("space", "string", None, "Filter by target space."),
            opt("limit", "int", Some("50"), "Max rows (capped at 100)."),
            opt("offset", "int", Some("0"), "Row offset."),
        ],
        example: r#"{"status":"pending","limit":20}"#,
    },
    CommandSchema {
        command: "candidate-approve",
        summary: "Approve a candidate, promoting it into a memory via the remember write path.",
        fields: &[
            req("id", "string", "Candidate id (cand_...)."),
            opt("embedding", "float[]", None, "Precomputed embedding for the promoted memory."),
            opt("embedding_model_id", "string", None, "Model id for the embedding."),
            opt("dry_run", "bool", Some("false"), "Preview without writing."),
        ],
        example: r#"{"id":"cand_..."}"#,
    },
    CommandSchema {
        command: "candidate-reject",
        summary: "Reject a candidate memory.",
        fields: &[
            req("id", "string", "Candidate id (cand_...)."),
            opt("reason", "string", None, "Reason recorded on the candidate."),
            opt("dry_run", "bool", Some("false"), "Preview without writing."),
        ],
        example: r#"{"id":"cand_...","reason":"duplicate"}"#,
    },
    CommandSchema {
        command: "candidate-quarantine",
        summary: "Quarantine a candidate memory (an adjudicator flagged it).",
        fields: &[
            req("id", "string", "Candidate id (cand_...)."),
            opt("reason", "string", None, "Reason recorded on the candidate."),
            opt("dry_run", "bool", Some("false"), "Preview without writing."),
        ],
        example: r#"{"id":"cand_...","reason":"adjudicator flagged"}"#,
    },
    CommandSchema {
        command: "ingest",
        summary: "Ingest a document source as isolated, embedded chunks (RAG store).",
        fields: &[
            req("chunks", "string[]", "Document chunks; one embedded source_episodes row each."),
            opt("space", "string", Some("\"documents\""), "Target space (isolated from curated memory)."),
            opt("source_type", "string", None, "Provenance type (e.g. markdown, pdf)."),
            opt("source_path", "string", None, "Source document path (citation + re-sync identity)."),
            opt("source_uri", "string", None, "Source document URI (citation)."),
            opt("source_description", "string", None, "Human-friendly source label."),
            opt("metadata_json", "string", None, "Opaque metadata JSON object string."),
            opt("dry_run", "bool", Some("false"), "Validate without writing."),
        ],
        example: r#"{"source_path":"notes/setup.md","chunks":["First chunk.","Second chunk."]}"#,
    },
    CommandSchema {
        command: "document-search",
        summary: "Hybrid (BM25 + vector) search over ingested document chunks.",
        fields: &[
            req("query", "string", "Search text."),
            opt("space", "string", Some("\"documents\""), "Space to search."),
            opt("limit", "int", Some("10"), "Max results."),
            opt("include_content", "bool", Some("false"), "Return full chunk content."),
            opt("snippet_chars", "int", Some("240"), "Snippet length (0 = none)."),
            opt("skip_recall_log", "bool", Some("false"), "Skip retrieval telemetry (eval runs)."),
        ],
        example: r#"{"query":"deployment steps","limit":5}"#,
    },
    CommandSchema {
        command: "document-get",
        summary: "Fetch a document's chunks by path, or one chunk by id.",
        fields: &[
            opt("source_path", "string", None, "Fetch all chunks of the document at this path."),
            opt("source_episode_id", "string", None, "Fetch a single chunk by its id."),
            opt("space", "string", Some("\"documents\""), "Space to look in."),
            opt("include_content", "bool", Some("true"), "Return full chunk content."),
            opt("limit", "int", None, "Max chunks to return."),
        ],
        example: r#"{"source_path":"notes/setup.md"}"#,
    },
    CommandSchema {
        command: "document-duplicates",
        summary: "Surface exact-content duplicate chunks (same content across paths).",
        fields: &[
            opt("space", "string", Some("\"documents\""), "Space to scan."),
            opt("limit", "int", Some("50"), "Max duplicate clusters."),
            opt("snippet_chars", "int", Some("160"), "Shared-content preview length (0 = none)."),
        ],
        example: r#"{"snippet_chars":160}"#,
    },
    CommandSchema {
        command: "promotion-candidates",
        summary: "Rank document chunks that earned retrieval traffic (promotion signal).",
        fields: &[
            opt("space", "string", Some("\"documents\""), "Space to rank within."),
            opt("min_hits", "int", Some("1"), "Minimum total retrieval hits to qualify."),
            opt("min_distinct_queries", "int", Some("1"), "Minimum distinct queries (diversity)."),
            opt("limit", "int", Some("20"), "Max candidates."),
            opt("include_content", "bool", Some("true"), "Return full chunk content."),
            opt("include_extracted", "bool", Some("false"), "Include already-promoted chunks."),
        ],
        example: r#"{"min_hits":2,"min_distinct_queries":2}"#,
    },
    CommandSchema {
        command: "mark-extracted",
        summary: "Mark document chunks extracted (promoted) so they stop surfacing.",
        fields: &[
            req("source_episode_ids", "string[]", "Chunk ids to mark extracted."),
            opt("space", "string", Some("\"documents\""), "Space the chunks live in."),
        ],
        example: r#"{"source_episode_ids":["src_..."]}"#,
    },
    CommandSchema {
        command: "document-prune",
        summary: "Delete document chunks by id (user-driven duplicate cleanup).",
        fields: &[
            req("source_episode_ids", "string[]", "Chunk ids to delete (you choose which copies to remove)."),
            opt("space", "string", Some("\"documents\""), "Space the chunks live in."),
            opt("dry_run", "bool", Some("false"), "Report what would be deleted without deleting."),
        ],
        example: r#"{"source_episode_ids":["src_..."],"dry_run":true}"#,
    },
];

/// Look up one command's schema by name.
pub(crate) fn schema_for(command: &str) -> Option<&'static CommandSchema> {
    COMMAND_SCHEMAS.iter().find(|s| s.command == command)
}

/// Entry point for the `schema` subcommand.
///
/// `schema`                 -> list every documented command.
/// `schema <command>`       -> show that command's payload contract.
/// `--json`                 -> emit machine-readable JSON instead of text.
pub(crate) fn run_schema(args: &[String]) -> i32 {
    let mut command = None;
    let mut as_json = false;
    for arg in args {
        match arg.as_str() {
            "--json" => as_json = true,
            value if value.starts_with("--") => {
                eprintln!("Unsupported schema flag '{value}'.");
                return 2;
            }
            value => {
                if command.is_some() {
                    eprintln!("schema accepts at most one command name.");
                    return 2;
                }
                command = Some(value.to_string());
            }
        }
    }

    let Some(name) = command else {
        if as_json {
            println!("{}", render_index_json());
        } else {
            print_index();
        }
        return 0;
    };

    let Some(schema) = schema_for(&name) else {
        eprintln!("No schema for command '{name}'.");
        eprintln!("Run `memkeeper schema` to list documented commands.");
        return 2;
    };

    if as_json {
        println!("{}", render_schema_json(schema));
    } else {
        print_schema(schema);
    }
    0
}

fn print_index() {
    println!("memkeeper command payload schemas (run `memkeeper schema <command>` for one):\n");
    let width = COMMAND_SCHEMAS
        .iter()
        .map(|s| s.command.len())
        .max()
        .unwrap_or(0);
    for schema in COMMAND_SCHEMAS {
        println!(
            "  {:<width$}  {}",
            schema.command,
            schema.summary,
            width = width
        );
    }
}

fn print_schema(schema: &CommandSchema) {
    println!("{} — {}\n", schema.command, schema.summary);
    let required: Vec<&FieldDoc> = schema.fields.iter().filter(|f| f.required).collect();
    let optional: Vec<&FieldDoc> = schema.fields.iter().filter(|f| !f.required).collect();
    let name_w = schema
        .fields
        .iter()
        .map(|f| f.name.len())
        .max()
        .unwrap_or(0);
    let type_w = schema.fields.iter().map(|f| f.ty.len()).max().unwrap_or(0);

    if !required.is_empty() {
        println!("Required:");
        for f in &required {
            println!(
                "  {:<name_w$}  {:<type_w$}  {}",
                f.name,
                f.ty,
                f.note,
                name_w = name_w,
                type_w = type_w
            );
        }
        println!();
    }
    if !optional.is_empty() {
        println!("Optional:");
        for f in &optional {
            let default = f.default.map(|d| format!("[={d}] ")).unwrap_or_default();
            println!(
                "  {:<name_w$}  {:<type_w$}  {}{}",
                f.name,
                f.ty,
                default,
                f.note,
                name_w = name_w,
                type_w = type_w
            );
        }
        println!();
    }
    println!("Example:");
    println!("  {}", schema.example);
}

fn render_field_json(field: &FieldDoc) -> String {
    let default = match field.default {
        Some(d) => json_string(d),
        None => "null".to_string(),
    };
    format!(
        "{{\"name\":{},\"type\":{},\"required\":{},\"default\":{},\"note\":{}}}",
        json_string(field.name),
        json_string(field.ty),
        field.required,
        default,
        json_string(field.note),
    )
}

fn render_schema_json(schema: &CommandSchema) -> String {
    let fields = schema
        .fields
        .iter()
        .map(render_field_json)
        .collect::<Vec<_>>()
        .join(",");
    // `example` is already valid JSON; embed it raw.
    format!(
        "{{\"command\":{},\"summary\":{},\"fields\":[{}],\"example\":{}}}",
        json_string(schema.command),
        json_string(schema.summary),
        fields,
        schema.example,
    )
}

#[cfg(test)]
pub(crate) fn render_schema_json_for_test(schema: &CommandSchema) -> String {
    render_schema_json(schema)
}

fn render_index_json() -> String {
    let items = COMMAND_SCHEMAS
        .iter()
        .map(render_schema_json)
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"commands\":[{items}]}}")
}
