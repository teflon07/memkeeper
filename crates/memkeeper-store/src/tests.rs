//! Unit tests for the memkeeper store. Lives in its own file so `lib.rs`
//! stays navigable; private access to the parent module is preserved
//! because this is still `mod tests` inside the crate root.

use super::{
    approve_candidate, assemble_reranked_pack, backup_store, batch_search_memories, build_pack,
    build_pack_pool_with_expansion, create_space, document_duplicates, dream_store, empty_pack,
    expanded_pack_queries, export_store, forget_memory, fts_score, get_document, get_memory,
    graph_context, graph_neighbors, import_store, ingest_source, init_store, last_synthesis_run,
    list_candidates, list_memories, list_silos, list_spaces, load_token_embeddings,
    load_token_embeddings_cached, mark_source_episodes_extracted, maxsim_score, memory_history,
    merge_entity, normalize_utc_timestamp, now_julian_day, pack_blocked_by_cosine_gate,
    promotion_candidates, prune_documents, rebuild_fts, recency_score_for_silo, record_recall,
    reject_candidate, remember_memory, schema_mentions_required_objects, search_documents,
    search_entities, search_memories, sha256_hex, sidecar_path, source_tier_score, store_stats,
    store_stats_with_health, submit_candidate, upsert_entity, upsert_memory_token_embedding,
    upsert_relationship, verify_memory, BackupRequest, BatchSearchQuery, BatchSearchRequest,
    CandidateApproveRequest, CandidateListRequest, CandidateRejectRequest, CandidateSubmitRequest,
    DocumentDuplicatesRequest, DocumentGetRequest, DocumentPruneRequest, DocumentSearchRequest,
    DreamRequest, EntityMergeRequest, EntitySearchRequest, EntityUpsertRequest, Error,
    ExportRequest, ForgetRequest, GetOptions, GraphContextRequest, GraphNeighborsRequest,
    HistoryOptions, ImportRequest, IngestRequest, MarkExtractedRequest, MemoryListRequest,
    PackExpansionOptions, PackRequest, PromotionCandidatesRequest, RecallEvent, RecallLogRequest,
    RelationshipUpsertRequest, RememberRequest, RerankCandidate, SearchFilters, SearchRequest,
    SearchResult, SiloListRequest, SpaceCreateRequest, VerifyRequest, DEFAULT_PROMOTE_RANK_CAP,
    DEFAULT_PROMOTE_SCORE_FLOOR, DEFAULT_PROMOTE_THRESHOLD, DOCUMENTS_SPACE, MAX_CONTENT_CHARS,
    MAX_HISTORY_LIMIT, MAX_MEMORY_LINKS, MAX_METADATA_VALUE_CHARS, MAX_SEARCH_LIMIT,
    MAX_TIMESTAMP_CHARS, PROJECT_STORE_RELATIVE_PATH, SCHEMA_SQL, SCHEMA_VERSION,
    USER_STORE_PATH_HINT, VOLATILE_MAX_RECENCY_SCORE,
};
use memkeeper_core::DEFAULT_SPACE;
use rusqlite::{params, Connection};
use std::{env, fs, path::Path, path::PathBuf, process, time::SystemTime};

#[test]
fn schema_embeds_required_objects() {
    assert!(schema_mentions_required_objects());
}

#[test]
fn source_tier_score_ranks_explicit_above_legacy_above_synthesis() {
    let no_tags: Vec<String> = Vec::new();
    let synth_tags = vec!["synthesis-derived".to_string()];
    let close = |actual: f64, expected: f64| (actual - expected).abs() < f64::EPSILON;

    // Explicit in-session and manual writes get the top boost.
    assert!(close(source_tier_score(Some("mcp"), &no_tags), 0.04));
    assert!(close(source_tier_score(Some("manual"), &no_tags), 0.04));

    // Legacy (no provenance) and unrecognized types sit in the middle.
    assert!(close(source_tier_score(None, &no_tags), 0.02));
    assert!(close(source_tier_score(Some("import"), &no_tags), 0.02));

    // Auto-harvested synthesis gets no boost, by source type or by tag.
    assert!(close(source_tier_score(Some("synthesis"), &no_tags), 0.0));
    // The synthesis-derived tag wins even if the source type looks explicit.
    assert!(close(source_tier_score(Some("mcp"), &synth_tags), 0.0));
}

#[test]
fn store_stats_health_rollup_reports_governance_signals() {
    let path = temp_store_path("store_stats_health_rollup");
    init_store(&path).expect("init succeeds");

    let remember_request = |content: &str, keys: Option<(&str, &str)>| RememberRequest {
        space: None,
        silo: None,
        scope: None,
        project_key: None,
        kind: None,
        content: content.to_string(),
        summary: None,
        tags: Vec::new(),
        entity_key: keys.map(|(entity, _)| entity.to_string()),
        claim_key: keys.map(|(_, claim)| claim.to_string()),
        confidence: 1.0,
        observed_at: Some("2026-05-25T21:00:00.000Z".to_string()),
        valid_from: None,
        valid_to: None,
        expires_at: None,
        source_ref_json: None,
        metadata_json: None,
        source_episode_id: None,
        pinned: false,
        supersedes: Vec::new(),
        contradicts: Vec::new(),
        embedding: None,
        embedding_model_id: None,
        token_embedding: None,
        token_embedding_model_id: None,
        dry_run: false,
        mode: "auto".to_string(),
    };

    // One memory with keys, one without.
    remember_memory(
        &path,
        &remember_request("fact: keyed memory", Some(("entity:a", "claim:a"))),
    )
    .expect("keyed remember succeeds");
    remember_memory(&path, &remember_request("fact: keyless memory", None))
        .expect("keyless remember succeeds");

    // Default stats omits the rollup; the opt-in variant includes it.
    assert!(store_stats(&path, false).expect("stats").health.is_none());
    let health = store_stats_with_health(&path, false)
        .expect("health stats")
        .health
        .expect("health present");

    assert_eq!(health.active, 2);
    assert_eq!(health.tombstoned, 0);
    assert_eq!(health.active_without_keys, 1);
    assert_eq!(health.duplicate_key_groups, 0);
    // No embeddings written in this FTS-only test store.
    assert_eq!(health.active_without_embedding, 2);
    assert_eq!(health.last_embedding_at, None);
    // Candidate queue unused -> the lazy table is absent and counts read zero.
    assert_eq!(health.candidates_pending, 0);
    assert_eq!(health.candidates_approved, 0);
    assert_eq!(health.candidates_rejected, 0);

    // Once a candidate is submitted, the rollup reflects the pending queue.
    submit_candidate(&path, &candidate_submit_request("fact: a proposed memory"))
        .expect("candidate submit succeeds");
    let health = store_stats_with_health(&path, false)
        .expect("health stats")
        .health
        .expect("health present");
    assert_eq!(health.candidates_pending, 1);
    assert_eq!(health.candidates_approved, 0);

    cleanup_store(&path);
}

#[test]
fn schema_keeps_sqlite_canonical_objects() {
    assert!(SCHEMA_SQL.contains("CREATE TABLE IF NOT EXISTS memories"));
    assert!(SCHEMA_SQL.contains("CREATE TABLE IF NOT EXISTS memory_events"));
    assert!(SCHEMA_SQL.contains("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5"));
    assert!(SCHEMA_SQL.contains("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts_public USING fts5"));
    assert!(SCHEMA_SQL.contains("idx_memories_space_status_updated"));
    assert!(SCHEMA_SQL.contains("idx_memories_space_status_created"));
    assert!(SCHEMA_SQL.contains("INSERT OR IGNORE INTO spaces"));
}

#[test]
fn store_path_policy_exposes_v0_1_host_hints() {
    assert_eq!(
        USER_STORE_PATH_HINT,
        "<user-data-dir>/memkeeper/store.sqlite"
    );
    assert_eq!(PROJECT_STORE_RELATIVE_PATH, ".memkeeper/store.sqlite");
}

#[test]
fn init_creates_schema_and_seed_data() {
    let path = temp_store_path("init_creates_schema_and_seed_data");
    cleanup_store(&path);

    let report = init_store(&path).expect("init succeeds");
    assert!(report.created);
    assert!(report.initialized);
    assert_eq!(report.schema_version, SCHEMA_VERSION);
    assert_eq!(report.default_space, DEFAULT_SPACE);
    assert_eq!(report.spaces, vec![DEFAULT_SPACE.to_string()]);
    assert_eq!(report.protocol_version, "memkeeper.v0.1");
    assert_eq!(report.journal_mode, "wal");

    let stats = store_stats(&path, true).expect("stats succeeds");
    assert_eq!(stats.schema_version, SCHEMA_VERSION);
    assert_eq!(stats.space_count, 1);
    assert_eq!(stats.silo_count, 2);
    assert_eq!(stats.memory_count, 0);
    assert_eq!(stats.active_count, 0);
    assert_eq!(stats.source_episode_count, 0);
    assert_eq!(stats.spaces.len(), 1);
    assert_eq!(stats.spaces[0].name, DEFAULT_SPACE);
    let indexes = stats.indexes.expect("indexes included");
    assert_eq!(indexes.fts_memory_rows, 0);
    assert_eq!(indexes.fts_source_episode_rows, 0);
    assert_eq!(indexes.pending_jobs, 0);

    cleanup_store(&path);
}

#[test]
fn space_create_lists_spaces_and_silos() {
    let path = temp_store_path("space_create_lists_spaces_and_silos");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let report = create_space(
        &path,
        &SpaceCreateRequest {
            name: "project-notes".to_string(),
            display_name: Some("Project Notes".to_string()),
            description: Some("Durable project notes".to_string()),
            default_silo: Some("long-term".to_string()),
            ontology: Some("notes".to_string()),
            config_json: Some("{\"owner\":\"test\"}".to_string()),
            if_not_exists: false,
        },
    )
    .expect("space create succeeds");
    assert!(report.created);
    assert_eq!(report.space.name, "project-notes");
    assert_eq!(report.space.default_silo, "long-term");
    assert_eq!(report.space.silo_count, 3);

    let spaces = list_spaces(&path).expect("space list succeeds");
    assert_eq!(spaces.spaces.len(), 2);
    assert_eq!(spaces.spaces[0].name, "project-notes");
    assert_eq!(spaces.spaces[1].name, DEFAULT_SPACE);

    let silos = list_silos(
        &path,
        &SiloListRequest {
            space: Some("project-notes".to_string()),
        },
    )
    .expect("silo list succeeds");
    assert_eq!(silos.space, "project-notes");
    assert_eq!(silos.silos.len(), 3);
    assert_eq!(silos.silos[0].name, "short-term");
    assert_eq!(silos.silos[1].name, "durable");
    assert_eq!(silos.silos[2].name, "long-term");
    assert!(silos.silos[2].is_default);

    let mut memory = basic_request("decision: project notes use custom space");
    memory.space = Some("project-notes".to_string());
    memory.silo = None;
    let remembered = remember_memory(&path, &memory).expect("remember in custom space");
    assert_eq!(remembered.memory.space, "project-notes");
    assert_eq!(remembered.memory.silo, "long-term");
    assert_eq!(
        list_spaces(&path).expect("spaces").spaces[0].memory_count,
        1
    );

    cleanup_store(&path);
}

#[test]
fn fresh_store_seeds_two_tier_silos_only() {
    let path = temp_store_path("fresh_store_seeds_two_tier_silos_only");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let report = list_silos(&path, &SiloListRequest { space: None }).expect("list silos");
    let names: Vec<&str> = report.silos.iter().map(|s| s.name.as_str()).collect();
    assert!(
        names.contains(&"short-term"),
        "short-term seeded: {names:?}"
    );
    assert!(names.contains(&"durable"), "durable seeded: {names:?}");
    assert!(
        !names.contains(&"long-term"),
        "long-term must NOT be seeded: {names:?}"
    );
}

#[test]
fn init_cleans_up_empty_long_term_but_keeps_populated() {
    let path = temp_store_path("init_cleans_up_empty_long_term_but_keeps_populated");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    {
        let conn = Connection::open(&path).expect("open");
        conn.execute(
            "INSERT OR IGNORE INTO silos (space_name, name, retention_policy, default_scope, created_at, updated_at)
             VALUES ('workspace-memory', 'long-term', 'keep', 'workspace', '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z')",
            [],
        ).expect("insert legacy long-term");
    }

    init_store(&path).expect("re-init succeeds");
    let after = list_silos(&path, &SiloListRequest { space: None }).expect("list");
    assert!(
        !after.silos.iter().any(|s| s.name == "long-term"),
        "empty long-term removed on init"
    );

    {
        let conn = Connection::open(&path).expect("open");
        conn.execute(
            "INSERT OR IGNORE INTO silos (space_name, name, retention_policy, default_scope, created_at, updated_at)
             VALUES ('workspace-memory', 'long-term', 'keep', 'workspace', '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z')",
            [],
        ).expect("re-insert long-term");
    }
    let mut req = basic_request("fact: lives in long-term");
    req.silo = Some("long-term".to_string());
    remember_memory(&path, &req).expect("remember into long-term");
    init_store(&path).expect("re-init with populated long-term");
    let final_silos = list_silos(&path, &SiloListRequest { space: None }).expect("list");
    assert!(
        final_silos.silos.iter().any(|s| s.name == "long-term"),
        "populated long-term preserved"
    );
}

#[test]
fn init_repoints_legacy_long_term_default_silo() {
    let path = temp_store_path("init_repoints_legacy_long_term_default_silo");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // Simulate a legacy store: space defaults to long-term, with an empty long-term silo.
    {
        let conn = Connection::open(&path).expect("open");
        conn.execute(
            "INSERT OR IGNORE INTO silos (space_name, name, retention_policy, default_scope, created_at, updated_at)
             VALUES ('workspace-memory', 'long-term', 'keep', 'workspace', '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z')",
            [],
        ).expect("insert long-term silo");
        conn.execute(
            "UPDATE spaces SET default_silo = 'long-term' WHERE name = 'workspace-memory'",
            [],
        )
        .expect("set legacy default");
    }

    init_store(&path).expect("re-init migrates");

    // Default-silo remember (no explicit silo) must now succeed and land in durable.
    let report = remember_memory(&path, &basic_request("fact: default silo after migration"))
        .expect("default remember succeeds post-migration");
    assert_eq!(report.memory.silo, "durable");
}

#[test]
fn space_create_is_idempotent_only_when_requested() {
    let path = temp_store_path("space_create_is_idempotent_only_when_requested");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let request = SpaceCreateRequest {
        name: "idempotent-space".to_string(),
        display_name: None,
        description: None,
        default_silo: None,
        ontology: None,
        config_json: None,
        if_not_exists: false,
    };
    let first = create_space(&path, &request).expect("first create succeeds");
    assert!(first.created);
    let duplicate = create_space(&path, &request).expect_err("duplicate fails");
    assert!(matches!(duplicate, Error::Conflict { .. }));

    let idempotent = create_space(
        &path,
        &SpaceCreateRequest {
            if_not_exists: true,
            ..request
        },
    )
    .expect("idempotent create succeeds");
    assert!(!idempotent.created);
    assert_eq!(idempotent.space.name, "idempotent-space");

    cleanup_store(&path);
}

#[test]
fn space_and_silo_requests_validate_bounds_and_missing_space() {
    let path = temp_store_path("space_and_silo_requests_validate_bounds_and_missing_space");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let bad_space = create_space(
        &path,
        &SpaceCreateRequest {
            name: " needs-trim ".to_string(),
            display_name: None,
            description: None,
            default_silo: None,
            ontology: None,
            config_json: None,
            if_not_exists: false,
        },
    )
    .expect_err("trimmed name rejected");
    assert!(matches!(bad_space, Error::InvalidRequest { .. }));

    let bad_config = create_space(
        &path,
        &SpaceCreateRequest {
            name: "bad-config".to_string(),
            display_name: None,
            description: None,
            default_silo: None,
            ontology: None,
            config_json: Some("{not-json}".to_string()),
            if_not_exists: false,
        },
    )
    .expect_err("bad config rejected");
    assert!(matches!(bad_config, Error::InvalidRequest { .. }));

    let missing = list_silos(
        &path,
        &SiloListRequest {
            space: Some("missing-space".to_string()),
        },
    )
    .expect_err("missing space rejected");
    assert!(matches!(
        missing,
        Error::NotFound {
            entity: "space",
            ..
        }
    ));

    cleanup_store(&path);
}

#[test]
fn init_is_idempotent() {
    let path = temp_store_path("init_is_idempotent");
    cleanup_store(&path);

    let first = init_store(&path).expect("first init succeeds");
    let second = init_store(&path).expect("second init succeeds");

    assert!(first.created);
    assert!(!second.created);
    assert_eq!(second.schema_version, SCHEMA_VERSION);
    assert_eq!(
        store_stats(&path, false).expect("stats succeeds").indexes,
        None
    );

    cleanup_store(&path);
}

#[test]
fn init_refuses_unrelated_existing_database_without_mutating_it() {
    let path = temp_store_path("init_refuses_unrelated_existing_database_without_mutating_it");
    cleanup_store(&path);
    let connection = Connection::open(&path).expect("create unrelated sqlite database");
    connection
        .execute_batch(
            "PRAGMA journal_mode = WAL; CREATE TABLE unrelated (id INTEGER PRIMARY KEY);",
        )
        .expect("create unrelated table");
    drop(connection);
    cleanup_store_sidecars(&path);

    let error = init_store(&path).expect_err("init should refuse unrelated database");
    assert!(matches!(error, Error::UnsafeExistingDatabase { .. }));
    assert!(!sidecar_path(&path, "-wal").exists());
    assert!(!sidecar_path(&path, "-shm").exists());

    let connection = Connection::open(&path).expect("reopen unrelated sqlite database");
    let user_version: i32 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("query user version");
    let memkeeper_tables: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'spaces'",
            [],
            |row| row.get(0),
        )
        .expect("query sqlite_master");
    assert_eq!(user_version, 0);
    assert_eq!(memkeeper_tables, 0);

    cleanup_store(&path);
}

#[test]
fn init_rejects_future_schema_without_enabling_wal() {
    let path = temp_store_path("init_rejects_future_schema_without_enabling_wal");
    cleanup_store(&path);
    let connection = Connection::open(&path).expect("create sqlite database");
    connection
        .execute_batch("PRAGMA journal_mode = WAL; PRAGMA user_version = 99;")
        .expect("set future schema");
    drop(connection);
    cleanup_store_sidecars(&path);

    let error = init_store(&path).expect_err("init should fail");
    assert!(matches!(
        error,
        Error::SchemaMismatch {
            expected: SCHEMA_VERSION,
            actual: 99
        }
    ));
    assert!(!sidecar_path(&path, "-wal").exists());
    assert!(!sidecar_path(&path, "-shm").exists());

    cleanup_store(&path);
}

#[test]
fn stats_refuses_unrelated_wal_database_without_sidecars() {
    let path = temp_store_path("stats_refuses_unrelated_wal_database_without_sidecars");
    cleanup_store(&path);
    let connection = Connection::open(&path).expect("create unrelated sqlite database");
    connection
        .execute_batch(
            "PRAGMA journal_mode = WAL; CREATE TABLE unrelated (id INTEGER PRIMARY KEY);",
        )
        .expect("create unrelated table");
    drop(connection);
    cleanup_store_sidecars(&path);

    let error = store_stats(&path, true).expect_err("stats should fail");
    assert!(matches!(error, Error::NotInitialized { .. }));
    assert!(!sidecar_path(&path, "-wal").exists());
    assert!(!sidecar_path(&path, "-shm").exists());

    cleanup_store(&path);
}

#[test]
fn prompt_read_ops_refuse_unrelated_wal_database_without_sidecars() {
    let path = temp_store_path("prompt_read_ops_refuse_unrelated_wal_database_without_sidecars");
    cleanup_store(&path);
    let connection = Connection::open(&path).expect("create unrelated sqlite database");
    connection
        .execute_batch(
            "PRAGMA journal_mode = WAL; CREATE TABLE unrelated (id INTEGER PRIMARY KEY);",
        )
        .expect("create unrelated table");
    drop(connection);
    cleanup_store_sidecars(&path);

    let get_error = get_memory(
        &path,
        "mem_x",
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: false,
        },
    )
    .expect_err("get should fail");
    assert!(matches!(get_error, Error::NotInitialized { .. }));
    let search_error = search_memories(
        &path,
        &SearchRequest {
            query: "needle".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 20,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "disabled".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect_err("search should fail");
    assert!(matches!(search_error, Error::NotInitialized { .. }));
    let history_error = memory_history(
        &path,
        "mem_x",
        HistoryOptions {
            limit: 10,
            include_source: false,
        },
    )
    .expect_err("history should fail");
    assert!(matches!(history_error, Error::NotInitialized { .. }));
    assert!(!sidecar_path(&path, "-wal").exists());
    assert!(!sidecar_path(&path, "-shm").exists());

    cleanup_store(&path);
}

#[test]
fn init_rejects_memory_path() {
    let error = init_store(Path::new(":memory:")).expect_err("memory path should fail");
    assert!(matches!(error, Error::InvalidPath { .. }));
}

#[test]
fn init_rejects_sqlite_uri_without_mutating_target() {
    let target = temp_store_path("init_rejects_sqlite_uri_without_mutating_target");
    cleanup_store(&target);
    let connection = Connection::open(&target).expect("create unrelated sqlite database");
    connection
        .execute_batch("CREATE TABLE unrelated (id INTEGER PRIMARY KEY);")
        .expect("create unrelated table");
    drop(connection);

    let uri = PathBuf::from(format!("file:{}", target.display()));
    let error = init_store(&uri).expect_err("sqlite uri should fail");
    assert!(matches!(error, Error::InvalidPath { .. }));

    let connection = Connection::open(&target).expect("reopen unrelated sqlite database");
    let user_version: i32 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("query user version");
    let memkeeper_tables: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'spaces'",
            [],
            |row| row.get(0),
        )
        .expect("query sqlite_master");
    assert_eq!(user_version, 0);
    assert_eq!(memkeeper_tables, 0);
    assert!(!Path::new("file:").exists());

    cleanup_store(&target);
}

#[test]
fn init_rejects_sqlite_memory_uri() {
    let error = init_store(Path::new("file::memory:?cache=shared"))
        .expect_err("sqlite memory uri should fail");
    assert!(matches!(error, Error::InvalidPath { .. }));
}

#[cfg(unix)]
#[test]
fn store_path_rejects_symlink() {
    use std::os::unix::fs::symlink;

    let target = temp_store_path("store_path_rejects_symlink_target");
    let link = temp_store_path("store_path_rejects_symlink_link");
    cleanup_store(&target);
    cleanup_store(&link);
    symlink(&target, &link).expect("create symlink");

    let error = init_store(&link).expect_err("symlink path should fail");
    assert!(matches!(error, Error::InvalidPath { .. }));

    let _ = fs::remove_file(&link);
    cleanup_store(&target);
}

#[cfg(unix)]
#[test]
fn sqlite_sidecar_symlink_is_rejected() {
    use std::os::unix::fs::symlink;

    let path = temp_store_path("sqlite_sidecar_symlink_is_rejected");
    let target = temp_store_path("sqlite_sidecar_symlink_target");
    cleanup_store(&path);
    cleanup_store(&target);
    fs::write(&target, b"sidecar target").expect("write target");
    symlink(&target, sidecar_path(&path, "-wal")).expect("create wal symlink");

    let error = init_store(&path).expect_err("sidecar symlink should fail");
    assert!(matches!(error, Error::InvalidPath { .. }));

    let _ = fs::remove_file(sidecar_path(&path, "-wal"));
    cleanup_store(&path);
    cleanup_store(&target);
}

#[test]
fn remember_and_get_write_atomic_memory_records() {
    let path = temp_store_path("remember_and_get_write_atomic_memory_records");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let report = remember_memory(
        &path,
        &RememberRequest {
            space: None,
            silo: None,
            scope: None,
            project_key: Some("Workspace".to_string()),
            kind: None,
            content: "decision: keep memkeeper deterministic".to_string(),
            summary: Some("Keep memkeeper deterministic.".to_string()),
            tags: vec!["memory".to_string(), "sqlite".to_string()],
            entity_key: Some("project:memkeeper".to_string()),
            claim_key: Some("mvp.boundary".to_string()),
            confidence: 1.0,
            observed_at: Some("2026-05-25T21:00:00.000Z".to_string()),
            valid_from: None,
            valid_to: None,
            expires_at: None,
            source_ref_json: Some("{\"type\":\"manual\",\"adapter\":\"host\"}".to_string()),
            metadata_json: None,
            source_episode_id: None,
            pinned: false,
            supersedes: Vec::new(),
            contradicts: Vec::new(),
            embedding: None,
            embedding_model_id: None,
            token_embedding: None,
            token_embedding_model_id: None,
            dry_run: false,
            mode: "auto".to_string(),
        },
    )
    .expect("remember succeeds");

    assert_eq!(report.processing_status, "indexed");
    assert!(report.candidates.is_empty());
    assert!(!report.candidates_truncated);
    assert_eq!(report.memory.space, DEFAULT_SPACE);
    assert_eq!(report.memory.silo, "durable");
    assert_eq!(report.memory.kind, "decision");
    assert_eq!(
        report.memory.tags,
        vec!["memory".to_string(), "sqlite".to_string()]
    );
    assert_eq!(
        report.memory.content_sha256,
        sha256_hex(report.memory.content.as_bytes())
    );

    let fetched = get_memory(
        &path,
        &report.memory.id,
        GetOptions {
            include_history: true,
            include_links: true,
            include_source: true,
        },
    )
    .expect("get succeeds");
    assert_eq!(fetched.id, report.memory.id);
    assert_eq!(fetched.content, "decision: keep memkeeper deterministic");
    assert_eq!(
        fetched.source_ref_json.as_deref(),
        Some("{\"type\":\"manual\",\"adapter\":\"host\"}")
    );
    assert_eq!(fetched.versions.expect("versions").len(), 1);
    assert_eq!(fetched.events.expect("events").len(), 1);
    assert_eq!(fetched.links.expect("links").len(), 0);

    let no_source = get_memory(
        &path,
        &report.memory.id,
        GetOptions {
            include_history: true,
            include_links: false,
            include_source: false,
        },
    )
    .expect("get no source succeeds");
    assert!(no_source.source_ref_json.is_none());
    assert!(no_source.versions.expect("versions")[0]
        .source_ref_json
        .is_none());

    let stats = store_stats(&path, true).expect("stats succeeds");
    assert_eq!(stats.memory_count, 1);
    assert_eq!(stats.active_count, 1);
    assert_eq!(stats.indexes.expect("indexes").fts_memory_rows, 1);

    cleanup_store(&path);
}

#[test]
fn remember_persists_and_reads_back_metadata_json() {
    let path = temp_store_path("metadata_json_roundtrip");
    cleanup_store(&path);
    init_store(&path).unwrap();
    let mut req = remember_request("metadata json content");
    req.metadata_json = Some(r#"{"verified_against":"~/.zshrc:FOO"}"#.to_string());
    let report = remember_memory(&path, &req).unwrap();
    let got = get_memory(
        &path,
        &report.memory.id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: false,
        },
    )
    .unwrap();
    assert_eq!(
        got.metadata_json.as_deref(),
        Some(r#"{"verified_against":"~/.zshrc:FOO"}"#)
    );
    cleanup_store(&path);
}

#[test]
fn remember_reports_duplicate_update_and_lexical_candidates() {
    let path = temp_store_path("remember_reports_duplicate_update_and_lexical_candidates");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut base =
        basic_request("decision: deterministic local memory uses sqlite fts bm25 for retrieval");
    base.kind = Some("decision".to_string());
    base.entity_key = Some("project:memkeeper".to_string());
    base.claim_key = Some("retrieval.primary".to_string());
    base.summary = Some("SQLite FTS BM25 is the deterministic retrieval path.".to_string());
    let base_report = remember_memory(&path, &base).expect("remember base succeeds");

    let mut duplicate = basic_request(&base.content);
    duplicate.dry_run = true;
    let duplicate_report = remember_memory(&path, &duplicate).expect("duplicate dry-run succeeds");
    assert_eq!(duplicate_report.processing_status, "dry_run");
    assert_eq!(
        duplicate_report.candidates[0].memory_id,
        base_report.memory.id
    );
    assert_eq!(duplicate_report.candidates[0].relationship, "duplicate");
    assert!(duplicate_report.candidates[0]
        .matched_on
        .contains(&"content_sha256".to_string()));

    let mut update = basic_request(
        "decision: memkeeper retrieval should keep deterministic sqlite fts indexes primary",
    );
    update.kind = Some("decision".to_string());
    update.entity_key = Some("project:memkeeper".to_string());
    update.claim_key = Some("retrieval.primary".to_string());
    update.dry_run = true;
    let update_report = remember_memory(&path, &update).expect("update dry-run succeeds");
    assert_eq!(update_report.candidates[0].memory_id, base_report.memory.id);
    assert_eq!(update_report.candidates[0].relationship, "update_candidate");
    assert!(update_report.candidates[0]
        .matched_on
        .contains(&"claim_key".to_string()));

    let mut lexical =
        basic_request("fact: local memory retrieval uses deterministic sqlite fts bm25 indexes");
    lexical.kind = Some("fact".to_string());
    lexical.dry_run = true;
    let lexical_report = remember_memory(&path, &lexical).expect("lexical dry-run succeeds");
    assert!(lexical_report.candidates.iter().any(|candidate| {
        candidate.memory_id == base_report.memory.id
            && candidate.relationship == "related_candidate"
            && candidate
                .matched_on
                .contains(&"lexical_similarity".to_string())
    }));

    cleanup_store(&path);
}

#[test]
fn remember_with_entity_key_projects_entity() {
    let path = temp_store_path("remember_with_entity_key_projects_entity");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = basic_request("decision: graph projection gets explicit entity anchors");
    request.entity_key = Some("project:memkeeper_graph_projection".to_string());
    let first = remember_memory(&path, &request).expect("remember succeeds");
    assert_eq!(
        first.memory.entity_key.as_deref(),
        Some("project:memkeeper_graph_projection")
    );

    let search = search_entities(
        &path,
        &EntitySearchRequest {
            space: None,
            query: None,
            entity_key: Some("project:memkeeper_graph_projection".to_string()),
            entity_types: Vec::new(),
            statuses: Vec::new(),
            limit: 10,
            offset: 0,
            include_source: false,
        },
    )
    .expect("entity search succeeds");
    assert_eq!(search.results.len(), 1);
    let entity = &search.results[0].entity;
    assert_eq!(entity.entity_type, "MemorySubject");
    assert_eq!(entity.canonical_name, "memkeeper graph projection");
    assert_eq!(entity.status, "active");
    assert!(entity.source_episode_id.is_none());

    let second = remember_memory(&path, &request).expect("second remember succeeds");
    assert_ne!(first.memory.id, second.memory.id);
    let search_again = search_entities(
        &path,
        &EntitySearchRequest {
            entity_key: Some("project:memkeeper_graph_projection".to_string()),
            limit: 10,
            ..entity_search_defaults()
        },
    )
    .expect("entity search succeeds");
    assert_eq!(search_again.results.len(), 1);

    cleanup_store(&path);
}

#[test]
fn remember_entity_projection_respects_dry_run() {
    let path = temp_store_path("remember_entity_projection_respects_dry_run");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = basic_request("fact: dry-run entity projection should roll back");
    request.entity_key = Some("project:dry_run_entity".to_string());
    request.dry_run = true;
    let report = remember_memory(&path, &request).expect("dry run remember succeeds");
    assert!(report.dry_run);

    let search = search_entities(
        &path,
        &EntitySearchRequest {
            entity_key: Some("project:dry_run_entity".to_string()),
            limit: 10,
            ..entity_search_defaults()
        },
    )
    .expect("entity search succeeds");
    assert!(search.results.is_empty());

    cleanup_store(&path);
}

#[test]
fn entity_search_matches_aliases_and_filters() {
    let path = temp_store_path("entity_search_matches_aliases_and_filters");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut request = basic_request("fact: alias search anchor");
    request.entity_key = Some("project:alias_anchor".to_string());
    remember_memory(&path, &request).expect("remember succeeds");
    let entity_id = search_entities(
        &path,
        &EntitySearchRequest {
            entity_key: Some("project:alias_anchor".to_string()),
            limit: 10,
            ..entity_search_defaults()
        },
    )
    .expect("entity search succeeds")
    .results[0]
        .entity
        .id
        .clone();

    let connection = Connection::open(&path).expect("open store");
    connection
        .execute(
            "INSERT INTO entity_aliases (entity_id, alias, normalized_alias, created_at)
             VALUES (?1, 'Memkeeper Alias', 'memkeeper alias', CURRENT_TIMESTAMP)",
            params![&entity_id],
        )
        .expect("insert alias");
    drop(connection);

    let by_alias = search_entities(
        &path,
        &EntitySearchRequest {
            query: Some("alias".to_string()),
            entity_types: vec!["MemorySubject".to_string()],
            statuses: vec!["active".to_string()],
            limit: 10,
            ..entity_search_defaults()
        },
    )
    .expect("entity search succeeds");
    assert_eq!(by_alias.results.len(), 1);
    assert_eq!(by_alias.results[0].entity.id, entity_id);
    assert_eq!(by_alias.results[0].matched_aliases, vec!["Memkeeper Alias"]);

    cleanup_store(&path);
}

#[test]
fn entity_upsert_creates_entity() {
    let path = temp_store_path("entity_upsert_creates_entity");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let report = upsert_entity(
        &path,
        &EntityUpsertRequest {
            aliases: vec!["Memkeeper".to_string()],
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect("entity upsert succeeds");

    assert!(report.created);
    assert_eq!(report.strategy, "deterministic_entity_upsert_v0");
    assert_eq!(report.entity.space, DEFAULT_SPACE);
    assert_eq!(report.entity.entity_key, "project:memkeeper");
    assert_eq!(report.entity.entity_type, "Entity");
    assert_eq!(report.entity.canonical_name, "Memkeeper");
    assert_eq!(report.entity.status, "active");
    assert!((report.entity.confidence - 1.0).abs() < f64::EPSILON);
    assert_eq!(report.entity.aliases, vec!["Memkeeper"]);
    assert!(report.entity.source_episode_id.is_none());

    let search = search_entities(
        &path,
        &EntitySearchRequest {
            entity_key: Some("project:memkeeper".to_string()),
            limit: 10,
            ..entity_search_defaults()
        },
    )
    .expect("entity search succeeds");
    assert_eq!(search.results.len(), 1);
    assert_eq!(search.results[0].entity.id, report.entity.id);

    cleanup_store(&path);
}

#[test]
fn entity_upsert_updates_type_name_status_and_confidence() {
    let path = temp_store_path("entity_upsert_updates_type_name_status_and_confidence");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let first = upsert_entity(
        &path,
        &EntityUpsertRequest {
            entity_type: Some("Project".to_string()),
            confidence: 0.8,
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect("first upsert succeeds");
    let second = upsert_entity(
        &path,
        &EntityUpsertRequest {
            entity_type: Some("System".to_string()),
            canonical_name: "Memkeeper graph projection".to_string(),
            status: Some("tombstoned".to_string()),
            confidence: 0.25,
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect("second upsert succeeds");

    assert!(!second.created);
    assert_eq!(second.entity.id, first.entity.id);
    assert_eq!(second.entity.entity_type, "System");
    assert_eq!(second.entity.canonical_name, "Memkeeper graph projection");
    assert_eq!(second.entity.status, "tombstoned");
    assert!((second.entity.confidence - 0.25).abs() < f64::EPSILON);

    let active_search = search_entities(
        &path,
        &EntitySearchRequest {
            entity_key: Some("project:memkeeper".to_string()),
            limit: 10,
            ..entity_search_defaults()
        },
    )
    .expect("active entity search succeeds");
    assert!(active_search.results.is_empty());

    let tombstoned_search = search_entities(
        &path,
        &EntitySearchRequest {
            entity_key: Some("project:memkeeper".to_string()),
            statuses: vec!["tombstoned".to_string()],
            limit: 10,
            ..entity_search_defaults()
        },
    )
    .expect("tombstoned entity search succeeds");
    assert_eq!(tombstoned_search.results.len(), 1);

    cleanup_store(&path);
}

#[test]
fn entity_upsert_aliases_insert_update_idempotently() {
    let path = temp_store_path("entity_upsert_aliases_insert_update_idempotently");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let first = upsert_entity(
        &path,
        &EntityUpsertRequest {
            aliases: vec!["Graph Projection".to_string(), "Memkeeper".to_string()],
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect("first upsert succeeds");
    // Compare as sets: alias storage order is an unspecified implementation
    // detail (and shifts if the fixture names are renamed); the contract under
    // test is which aliases are present and idempotency, not their order.
    let mut first_aliases = first.entity.aliases.clone();
    first_aliases.sort();
    let mut want_first = vec!["Memkeeper".to_string(), "Graph Projection".to_string()];
    want_first.sort();
    assert_eq!(first_aliases, want_first);

    let second = upsert_entity(
        &path,
        &EntityUpsertRequest {
            aliases: vec!["memkeeper".to_string(), "FM".to_string()],
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect("second upsert succeeds");
    assert_eq!(second.entity.id, first.entity.id);
    let mut second_aliases = second.entity.aliases.clone();
    second_aliases.sort();
    let mut want_second = vec![
        "memkeeper".to_string(),
        "FM".to_string(),
        "Graph Projection".to_string(),
    ];
    want_second.sort();
    assert_eq!(second_aliases, want_second);

    let third = upsert_entity(
        &path,
        &EntityUpsertRequest {
            aliases: vec!["memkeeper".to_string(), "FM".to_string()],
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect("third upsert succeeds");
    assert_eq!(third.entity.aliases, second.entity.aliases);

    let connection = Connection::open(&path).expect("open store");
    let alias_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM entity_aliases", [], |row| row.get(0))
        .expect("count aliases");
    assert_eq!(alias_count, 3);

    cleanup_store(&path);
}

#[test]
fn entity_upsert_hides_source_by_default() {
    let path = temp_store_path("entity_upsert_hides_source_by_default");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute(
            "INSERT INTO source_episodes (
                id, space_name, source_type, content, ingested_at, created_at, updated_at
             ) VALUES ('src-entity', 'workspace-memory', 'manual', 'source text', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            [],
        )
        .expect("insert source episode");
    drop(connection);

    let hidden = upsert_entity(
        &path,
        &EntityUpsertRequest {
            source_episode_id: Some("src-entity".to_string()),
            include_source: false,
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect("source-hidden upsert succeeds");
    assert!(hidden.entity.source_episode_id.is_none());

    let included = upsert_entity(
        &path,
        &EntityUpsertRequest {
            source_episode_id: Some("src-entity".to_string()),
            include_source: true,
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect("source-including upsert succeeds");
    assert_eq!(
        included.entity.source_episode_id.as_deref(),
        Some("src-entity")
    );

    cleanup_store(&path);
}

#[test]
fn entity_upsert_rejects_bad_status_confidence_and_cross_space_source() {
    let path =
        temp_store_path("entity_upsert_rejects_bad_status_confidence_and_cross_space_source");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let bad_status = upsert_entity(
        &path,
        &EntityUpsertRequest {
            status: Some("conflicted".to_string()),
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect_err("bad status rejected");
    assert!(matches!(bad_status, Error::InvalidRequest { .. }));

    let bad_confidence = upsert_entity(
        &path,
        &EntityUpsertRequest {
            confidence: 1.1,
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect_err("bad confidence rejected");
    assert!(matches!(bad_confidence, Error::InvalidRequest { .. }));

    let connection = Connection::open(&path).expect("open store");
    connection
        .execute_batch(
            "INSERT INTO spaces (name, display_name, created_at, updated_at)
             VALUES ('other-space', 'Other Space', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
             INSERT INTO silos (space_name, name, description, retention_policy, default_scope, created_at, updated_at)
             VALUES ('other-space', 'durable', 'Durable', 'keep', 'workspace', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
             INSERT INTO source_episodes (id, space_name, source_type, content, ingested_at, created_at, updated_at)
             VALUES ('src-other-entity', 'other-space', 'manual', 'source', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);",
        )
        .expect("insert cross-space source episode");
    drop(connection);

    let cross_space = upsert_entity(
        &path,
        &EntityUpsertRequest {
            source_episode_id: Some("src-other-entity".to_string()),
            ..entity_upsert_request("project:memkeeper", "Memkeeper")
        },
    )
    .expect_err("cross-space source rejected");
    assert!(matches!(
        cross_space,
        Error::NotFound {
            entity: "source_episode",
            ..
        }
    ));

    cleanup_store(&path);
}

#[test]
fn relationship_upsert_creates_and_updates_edge() {
    let path = temp_store_path("relationship_upsert_creates_and_updates_edge");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let subject = upsert_entity(
        &path,
        &entity_upsert_request("project:memkeeper", "Memkeeper"),
    )
    .expect("subject upsert succeeds");
    let object = upsert_entity(&path, &entity_upsert_request("component:sqlite", "SQLite"))
        .expect("object upsert succeeds");

    let first = upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("project:memkeeper".to_string()),
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:sqlite".to_string()),
            confidence: 0.7,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("relationship upsert succeeds");
    assert!(first.created);
    assert_eq!(first.strategy, "deterministic_relationship_upsert_v0");
    assert_eq!(first.relationship.subject_entity_id, subject.entity.id);
    assert_eq!(first.relationship.object_entity_id, object.entity.id);
    assert_eq!(first.relationship.relation_type, "uses");
    assert!((first.relationship.confidence - 0.7).abs() < f64::EPSILON);
    assert!(first.relationship.source_episode_id.is_none());

    let second = upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_id: Some(subject.entity.id.clone()),
            relation_type: "uses".to_string(),
            object_entity_id: Some(object.entity.id.clone()),
            status: Some("superseded".to_string()),
            confidence: 0.4,
            observed_at: Some("2026-05-28T00:00:00Z".to_string()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("relationship update succeeds");
    assert!(!second.created);
    assert_eq!(second.relationship.id, first.relationship.id);
    assert_eq!(second.relationship.status, "superseded");
    assert!((second.relationship.confidence - 0.4).abs() < f64::EPSILON);
    assert_eq!(
        second.relationship.observed_at.as_deref(),
        Some("2026-05-28T00:00:00Z")
    );

    cleanup_store(&path);
}

fn active_edges_to(connection: &Connection, object_key: &str, relation: &str) -> i64 {
    connection
        .query_row(
            "SELECT COUNT(*) FROM relationships r
             JOIN entities o ON o.id = r.object_entity_id
             WHERE r.status = 'active' AND o.entity_key = ?1 AND r.relation_type = ?2",
            params![object_key, relation],
            |row| row.get::<_, i64>(0),
        )
        .expect("count active edges")
}

fn entity_status(connection: &Connection, entity_key: &str) -> String {
    connection
        .query_row(
            "SELECT status FROM entities WHERE entity_key = ?1",
            params![entity_key],
            |row| row.get::<_, String>(0),
        )
        .expect("entity status")
}

fn merge_request_defaults() -> EntityMergeRequest {
    EntityMergeRequest {
        space: None,
        from_entity_id: None,
        from_entity_key: None,
        into_entity_id: None,
        into_entity_key: None,
        dry_run: false,
        include_source: false,
    }
}

#[test]
fn merge_repoints_edges_and_tombstones_source() {
    let path = temp_store_path("merge_repoints_edges_and_tombstones_source");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(&path, &entity_upsert_request("file:canon", "store.sqlite"))
        .expect("canon upsert");
    upsert_entity(
        &path,
        &entity_upsert_request("file:variant", "variant store.sqlite"),
    )
    .expect("variant upsert");
    upsert_entity(&path, &entity_upsert_request("card:a", "Card A")).expect("card upsert");
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("card:a".to_string()),
            relation_type: "card_mentions_file".to_string(),
            object_entity_key: Some("file:variant".to_string()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("edge upsert");

    let report = merge_entity(
        &path,
        &EntityMergeRequest {
            from_entity_key: Some("file:variant".to_string()),
            into_entity_key: Some("file:canon".to_string()),
            ..merge_request_defaults()
        },
    )
    .expect("merge succeeds");
    assert_eq!(report.relationships_repointed, 1);
    assert_eq!(report.relationships_tombstoned_duplicate, 0);
    assert_eq!(report.relationships_tombstoned_self_loop, 0);
    assert!(report.from_tombstoned);
    assert!(report.into.aliases.iter().any(|a| a == "file:variant"));

    let connection = Connection::open(&path).expect("open store");
    assert_eq!(entity_status(&connection, "file:variant"), "tombstoned");
    assert_eq!(entity_status(&connection, "file:canon"), "active");
    assert_eq!(
        active_edges_to(&connection, "file:canon", "card_mentions_file"),
        1
    );
    assert_eq!(
        active_edges_to(&connection, "file:variant", "card_mentions_file"),
        0
    );
    cleanup_store(&path);
}

#[test]
fn merge_collapses_duplicate_and_self_loop_edges() {
    let path = temp_store_path("merge_collapses_duplicate_and_self_loop_edges");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(&path, &entity_upsert_request("file:canon", "canon")).expect("canon");
    upsert_entity(&path, &entity_upsert_request("file:variant", "variant")).expect("variant");
    upsert_entity(&path, &entity_upsert_request("card:a", "Card A")).expect("card");
    for object in ["file:variant", "file:canon"] {
        upsert_relationship(
            &path,
            &RelationshipUpsertRequest {
                subject_entity_key: Some("card:a".to_string()),
                relation_type: "card_mentions_file".to_string(),
                object_entity_key: Some(object.to_string()),
                ..relationship_upsert_request_defaults()
            },
        )
        .expect("edge upsert");
    }
    // variant -> canon becomes a canon->canon self-loop on merge.
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("file:variant".to_string()),
            relation_type: "related_to".to_string(),
            object_entity_key: Some("file:canon".to_string()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("self-loop edge upsert");

    let report = merge_entity(
        &path,
        &EntityMergeRequest {
            from_entity_key: Some("file:variant".to_string()),
            into_entity_key: Some("file:canon".to_string()),
            ..merge_request_defaults()
        },
    )
    .expect("merge succeeds");
    assert_eq!(report.relationships_repointed, 0);
    assert_eq!(report.relationships_tombstoned_duplicate, 1);
    assert_eq!(report.relationships_tombstoned_self_loop, 1);

    let connection = Connection::open(&path).expect("open store");
    assert_eq!(
        active_edges_to(&connection, "file:canon", "card_mentions_file"),
        1
    );
    assert_eq!(active_edges_to(&connection, "file:canon", "related_to"), 0);
    cleanup_store(&path);
}

#[test]
fn merge_dry_run_is_non_mutating_and_validates() {
    let path = temp_store_path("merge_dry_run_is_non_mutating_and_validates");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(&path, &entity_upsert_request("file:canon", "canon")).expect("canon");
    upsert_entity(&path, &entity_upsert_request("file:variant", "variant")).expect("variant");
    upsert_entity(&path, &entity_upsert_request("card:a", "Card A")).expect("card");
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("card:a".to_string()),
            relation_type: "card_mentions_file".to_string(),
            object_entity_key: Some("file:variant".to_string()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("edge upsert");

    let report = merge_entity(
        &path,
        &EntityMergeRequest {
            from_entity_key: Some("file:variant".to_string()),
            into_entity_key: Some("file:canon".to_string()),
            dry_run: true,
            ..merge_request_defaults()
        },
    )
    .expect("dry-run merge succeeds");
    assert!(report.dry_run);
    assert_eq!(report.relationships_repointed, 1);

    // Dry run rolled back: variant still active, edge unchanged.
    let connection = Connection::open(&path).expect("open store");
    assert_eq!(entity_status(&connection, "file:variant"), "active");
    assert_eq!(
        active_edges_to(&connection, "file:variant", "card_mentions_file"),
        1
    );
    assert_eq!(
        active_edges_to(&connection, "file:canon", "card_mentions_file"),
        0
    );

    let same = merge_entity(
        &path,
        &EntityMergeRequest {
            from_entity_key: Some("file:canon".to_string()),
            into_entity_key: Some("file:canon".to_string()),
            ..merge_request_defaults()
        },
    );
    assert!(matches!(same, Err(Error::InvalidRequest { .. })));

    let missing = merge_entity(
        &path,
        &EntityMergeRequest {
            into_entity_key: Some("file:canon".to_string()),
            ..merge_request_defaults()
        },
    );
    assert!(matches!(missing, Err(Error::InvalidRequest { .. })));

    // Both id and key for one endpoint is rejected (exactly-one).
    let both = merge_entity(
        &path,
        &EntityMergeRequest {
            from_entity_id: Some("ent-x".to_string()),
            from_entity_key: Some("file:variant".to_string()),
            into_entity_key: Some("file:canon".to_string()),
            ..merge_request_defaults()
        },
    );
    assert!(matches!(both, Err(Error::InvalidRequest { .. })));

    // Merging into a tombstoned target is rejected.
    upsert_entity(
        &path,
        &EntityUpsertRequest {
            status: Some("tombstoned".to_string()),
            ..entity_upsert_request("file:dead", "dead")
        },
    )
    .expect("dead upsert");
    let into_dead = merge_entity(
        &path,
        &EntityMergeRequest {
            from_entity_key: Some("file:variant".to_string()),
            into_entity_key: Some("file:dead".to_string()),
            ..merge_request_defaults()
        },
    );
    assert!(matches!(into_dead, Err(Error::Conflict { .. })));
    cleanup_store(&path);
}

#[test]
fn relationship_upsert_validates_evidence_and_hides_source() {
    let path = temp_store_path("relationship_upsert_validates_evidence_and_hides_source");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(
        &path,
        &entity_upsert_request("project:memkeeper", "Memkeeper"),
    )
    .expect("subject upsert succeeds");
    upsert_entity(&path, &entity_upsert_request("component:sqlite", "SQLite"))
        .expect("object upsert succeeds");
    let evidence = remember_memory(
        &path,
        &RememberRequest {
            content: "memkeeper uses sqlite for local storage".to_string(),
            entity_key: Some("project:memkeeper".to_string()),
            ..remember_request("relationship evidence")
        },
    )
    .expect("remember succeeds");
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute(
            "INSERT INTO source_episodes (
                id, space_name, source_type, content, ingested_at, created_at, updated_at
             ) VALUES ('src-rel', 'workspace-memory', 'manual', 'source text', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            [],
        )
        .expect("insert source episode");
    drop(connection);

    let hidden = upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("project:memkeeper".to_string()),
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:sqlite".to_string()),
            memory_id: Some(evidence.memory.id.clone()),
            source_episode_id: Some("src-rel".to_string()),
            include_source: false,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("hidden-source upsert succeeds");
    assert_eq!(
        hidden.relationship.memory_id.as_deref(),
        Some(evidence.memory.id.as_str())
    );
    assert!(hidden.relationship.source_episode_id.is_none());

    let included = upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("project:memkeeper".to_string()),
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:sqlite".to_string()),
            memory_id: Some(evidence.memory.id.clone()),
            source_episode_id: Some("src-rel".to_string()),
            include_source: true,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("source-including upsert succeeds");
    assert_eq!(included.relationship.id, hidden.relationship.id);
    assert_eq!(
        included.relationship.source_episode_id.as_deref(),
        Some("src-rel")
    );

    cleanup_store(&path);
}

#[test]
fn relationship_upsert_rejects_bad_endpoint_status_confidence_and_cross_space_evidence() {
    let path = temp_store_path(
        "relationship_upsert_rejects_bad_endpoint_status_confidence_and_cross_space_evidence",
    );
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(
        &path,
        &entity_upsert_request("project:memkeeper", "Memkeeper"),
    )
    .expect("subject upsert succeeds");
    upsert_entity(&path, &entity_upsert_request("component:sqlite", "SQLite"))
        .expect("object upsert succeeds");

    let missing_endpoint = upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:sqlite".to_string()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect_err("missing endpoint rejected");
    assert!(matches!(missing_endpoint, Error::InvalidRequest { .. }));

    let bad_status = upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("project:memkeeper".to_string()),
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:sqlite".to_string()),
            status: Some("merged".to_string()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect_err("bad status rejected");
    assert!(matches!(bad_status, Error::InvalidRequest { .. }));

    let bad_confidence = upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("project:memkeeper".to_string()),
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:sqlite".to_string()),
            confidence: -0.1,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect_err("bad confidence rejected");
    assert!(matches!(bad_confidence, Error::InvalidRequest { .. }));

    let connection = Connection::open(&path).expect("open store");
    connection
        .execute_batch(
            "INSERT INTO spaces (name, display_name, created_at, updated_at)
             VALUES ('other-space', 'Other Space', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
             INSERT INTO silos (space_name, name, description, retention_policy, default_scope, created_at, updated_at)
             VALUES ('other-space', 'durable', 'Durable', 'keep', 'workspace', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
             INSERT INTO memories (
                id, active_version_id, space_name, silo_name, scope, kind, status, confidence,
                pinned, observed_at, created_at, updated_at
             ) VALUES (
                'mem-other-rel', 'ver-other-rel', 'other-space', 'durable', 'workspace',
                'note', 'active', 1.0, 0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
             );
             INSERT INTO memory_versions (id, memory_id, version_num, content, content_sha256, created_at)
             VALUES ('ver-other-rel', 'mem-other-rel', 1, 'other-space evidence', 'abc', CURRENT_TIMESTAMP);",
        )
        .expect("insert cross-space memory");
    drop(connection);
    let cross_space_memory = upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("project:memkeeper".to_string()),
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:sqlite".to_string()),
            memory_id: Some("mem-other-rel".to_string()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect_err("cross-space memory rejected");
    assert!(matches!(
        cross_space_memory,
        Error::NotFound {
            entity: "memory",
            ..
        }
    ));

    cleanup_store(&path);
}

#[test]
fn graph_context_packs_relationship_evidence_and_entity_memories() {
    let path = temp_store_path("graph_context_packs_relationship_evidence_and_entity_memories");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(
        &path,
        &entity_upsert_request("project:memkeeper", "Memkeeper"),
    )
    .expect("subject upsert succeeds");
    upsert_entity(&path, &entity_upsert_request("component:sqlite", "SQLite"))
        .expect("object upsert succeeds");
    let evidence = remember_memory(
        &path,
        &RememberRequest {
            content: "decision: relationship evidence says memkeeper uses sqlite".to_string(),
            ..remember_request("relationship evidence")
        },
    )
    .expect("evidence remember succeeds");
    let entity_memory = remember_memory(
        &path,
        &RememberRequest {
            content: "lesson: sqlite is the local memkeeper graph store".to_string(),
            entity_key: Some("component:sqlite".to_string()),
            ..remember_request("sqlite entity memory")
        },
    )
    .expect("entity remember succeeds");
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("project:memkeeper".to_string()),
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:sqlite".to_string()),
            memory_id: Some(evidence.memory.id.clone()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("relationship upsert succeeds");

    let report = graph_context(
        &path,
        &GraphContextRequest {
            entity_key: Some("project:memkeeper".to_string()),
            depth: 1,
            max_edges: 10,
            max_memories: 5,
            max_chars: 2_000,
            ..graph_context_defaults()
        },
    )
    .expect("graph context succeeds");

    assert_eq!(report.strategy, "deterministic_graph_context_v0");
    assert_eq!(report.graph.relationships.len(), 1);
    assert_eq!(report.evidence_memory_ids, vec![evidence.memory.id.clone()]);
    assert!(report.entity_memory_ids.contains(&entity_memory.memory.id));
    assert!(report.pack.memory_ids.contains(&evidence.memory.id));
    assert!(report.pack.memory_ids.contains(&entity_memory.memory.id));
    assert!(report.pack.content.contains("Retrieved Memory"));
    assert!(report.pack.content.contains("relationship evidence"));
    assert!(report.pack.content.contains("sqlite is the local"));

    cleanup_store(&path);
}

#[test]
fn graph_context_rejects_invalid_pack_bounds() {
    let path = temp_store_path("graph_context_rejects_invalid_pack_bounds");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(
        &path,
        &entity_upsert_request("project:memkeeper", "Memkeeper"),
    )
    .expect("subject upsert succeeds");

    let error = graph_context(
        &path,
        &GraphContextRequest {
            entity_key: Some("project:memkeeper".to_string()),
            depth: 1,
            max_edges: 10,
            max_memories: 0,
            max_chars: 2_000,
            ..graph_context_defaults()
        },
    )
    .expect_err("max_memories rejected");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    cleanup_store(&path);
}

#[test]
fn graph_neighbors_traverses_bounded_active_edges() {
    let path = temp_store_path("graph_neighbors_traverses_bounded_active_edges");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut alpha_request = basic_request("fact: alpha graph node");
    alpha_request.entity_key = Some("entity:alpha".to_string());
    let alpha_memory = remember_memory(&path, &alpha_request).expect("remember alpha");
    let mut beta_request = basic_request("fact: beta graph node");
    beta_request.entity_key = Some("entity:beta".to_string());
    remember_memory(&path, &beta_request).expect("remember beta");
    let mut gamma_request = basic_request("fact: gamma graph node");
    gamma_request.entity_key = Some("entity:gamma".to_string());
    remember_memory(&path, &gamma_request).expect("remember gamma");

    let alpha = entity_id_for_key(&path, "entity:alpha");
    let beta = entity_id_for_key(&path, "entity:beta");
    let gamma = entity_id_for_key(&path, "entity:gamma");
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute(
            "INSERT INTO relationships (
                id, space_name, subject_entity_id, relation_type, object_entity_id,
                memory_id, status, confidence, created_at, updated_at
             ) VALUES ('rel-alpha-beta', 'workspace-memory', ?1, 'related_to', ?2, ?3, 'active', 1.0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            params![&alpha, &beta, &alpha_memory.memory.id],
        )
        .expect("insert alpha beta relationship");
    connection
        .execute(
            "INSERT INTO relationships (
                id, space_name, subject_entity_id, relation_type, object_entity_id,
                status, confidence, created_at, updated_at
             ) VALUES ('rel-beta-gamma', 'workspace-memory', ?1, 'related_to', ?2, 'active', 1.0, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            params![&beta, &gamma],
        )
        .expect("insert beta gamma relationship");
    drop(connection);

    let depth_one = graph_neighbors(
        &path,
        &GraphNeighborsRequest {
            entity_key: Some("entity:alpha".to_string()),
            depth: 1,
            max_edges: 10,
            ..graph_neighbors_defaults()
        },
    )
    .expect("graph neighbors succeeds");
    assert_eq!(depth_one.relationships.len(), 1);
    assert_eq!(depth_one.entities.len(), 2);
    assert!(depth_one
        .entities
        .iter()
        .any(|entity| entity.entity.entity_key == "entity:beta"));
    assert!(!depth_one
        .entities
        .iter()
        .any(|entity| entity.entity.entity_key == "entity:gamma"));

    let depth_two = graph_neighbors(
        &path,
        &GraphNeighborsRequest {
            entity_key: Some("entity:alpha".to_string()),
            depth: 2,
            max_edges: 10,
            ..graph_neighbors_defaults()
        },
    )
    .expect("graph neighbors succeeds");
    assert_eq!(depth_two.relationships.len(), 2);
    assert!(depth_two
        .entities
        .iter()
        .any(|entity| entity.entity.entity_key == "entity:gamma"));

    forget_memory(
        &path,
        &ForgetRequest {
            id: alpha_memory.memory.id,
            reason: Some("hide inactive graph evidence".to_string()),
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect("forget evidence memory");
    let after_forget = graph_neighbors(
        &path,
        &GraphNeighborsRequest {
            entity_key: Some("entity:alpha".to_string()),
            depth: 1,
            max_edges: 10,
            ..graph_neighbors_defaults()
        },
    )
    .expect("graph neighbors succeeds");
    assert!(after_forget.relationships.is_empty());

    cleanup_store(&path);
}

#[test]
fn get_history_is_bounded() {
    let path = temp_store_path("get_history_is_bounded");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let remembered = remember_memory(&path, &basic_request("decision: bounded get history"))
        .expect("remember succeeds");
    let connection = Connection::open(&path).expect("open store");
    let transaction = connection
        .unchecked_transaction()
        .expect("begin insert events");
    for index in 0..(MAX_HISTORY_LIMIT + 5) {
        transaction
            .execute(
                "INSERT INTO memory_events (id, memory_id, event_type, actor, created_at)
                 VALUES (?1, ?2, 'import', 'test', ?3)",
                params![
                    format!("evt-extra-{index}"),
                    &remembered.memory.id,
                    format!("2026-05-25T00:00:{:02}.000Z", index % 60),
                ],
            )
            .expect("insert event");
    }
    transaction.commit().expect("commit events");

    let loaded = get_memory(
        &path,
        &remembered.memory.id,
        GetOptions {
            include_history: true,
            include_links: false,
            include_source: false,
        },
    )
    .expect("get succeeds");
    assert_eq!(loaded.events.expect("events").len(), MAX_HISTORY_LIMIT);

    cleanup_store(&path);
}

#[test]
fn remember_rejects_duplicate_trimmed_tags() {
    let path = temp_store_path("remember_rejects_duplicate_trimmed_tags");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut request = basic_request("duplicate tag test");
    request.tags = vec!["tag".to_string(), " tag ".to_string()];

    let error = remember_memory(&path, &request).expect_err("duplicate tags should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));
    assert_eq!(store_stats(&path, true).expect("stats").memory_count, 0);

    cleanup_store(&path);
}

#[test]
fn remember_rejects_invalid_source_json() {
    let path = temp_store_path("remember_rejects_invalid_source_json");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut request = basic_request("invalid source test");
    request.source_ref_json = Some("{not-json}".to_string());

    let error = remember_memory(&path, &request).expect_err("invalid source should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    request.source_ref_json = Some("{\"x\":\"\\uD800\"}".to_string());
    let surrogate_error =
        remember_memory(&path, &request).expect_err("surrogate source should fail");
    assert!(matches!(surrogate_error, Error::InvalidRequest { .. }));

    cleanup_store(&path);
}

#[test]
fn remember_rejects_unbounded_metadata_and_links() {
    let path = temp_store_path("remember_rejects_unbounded_metadata_and_links");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut too_long_project = basic_request("metadata bounds");
    too_long_project.project_key = Some("x".repeat(MAX_METADATA_VALUE_CHARS + 1));
    let error =
        remember_memory(&path, &too_long_project).expect_err("long project key should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    let mut bad_time = basic_request("timestamp bounds");
    bad_time.observed_at = Some("2026-05-25 00:00:00".to_string());
    let error = remember_memory(&path, &bad_time).expect_err("bad timestamp should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    let mut invalid_calendar_time = basic_request("timestamp calendar bounds");
    invalid_calendar_time.observed_at = Some("2026-99-99T99:99:99.000Z".to_string());
    let error = remember_memory(&path, &invalid_calendar_time)
        .expect_err("shape-valid invalid timestamp should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    let mut too_many_links = basic_request("link bounds");
    too_many_links.supersedes = (0..=MAX_MEMORY_LINKS)
        .map(|index| format!("mem_{index}"))
        .collect();
    let error = remember_memory(&path, &too_many_links).expect_err("too many links should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    cleanup_store(&path);
}

#[test]
fn remember_rejects_cross_space_source_episode() {
    let path = temp_store_path("remember_rejects_cross_space_source_episode");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute_batch(
            "INSERT INTO spaces (name, display_name, description, default_silo, created_at, updated_at)
             VALUES ('other-space', 'Other', 'Other', 'durable', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
             INSERT INTO silos (space_name, name, description, retention_policy, default_scope, created_at, updated_at)
             VALUES ('other-space', 'durable', 'Durable', 'keep', 'workspace', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
             INSERT INTO source_episodes (id, space_name, source_type, content, ingested_at, created_at, updated_at)
             VALUES ('src-other', 'other-space', 'manual', 'source', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);",
        )
        .expect("insert source episode");
    drop(connection);

    let mut request = basic_request("cross space source");
    request.source_episode_id = Some("src-other".to_string());
    request.source_ref_json =
        Some("{\"type\":\"manual\",\"source_episode_id\":\"src-other\"}".to_string());

    let error = remember_memory(&path, &request).expect_err("cross-space source should fail");
    assert!(matches!(
        error,
        Error::NotFound {
            entity: "source_episode",
            ..
        }
    ));

    cleanup_store(&path);
}

#[test]
fn get_without_source_hides_source_episode_id() {
    let path = temp_store_path("get_without_source_hides_source_episode_id");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute_batch(
            "INSERT INTO source_episodes (id, space_name, source_type, content, ingested_at, created_at, updated_at)
             VALUES ('src-workspace', 'workspace-memory', 'manual', 'source', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);",
        )
        .expect("insert source episode");
    drop(connection);

    let mut request = basic_request("source episode gating");
    request.source_episode_id = Some("src-workspace".to_string());
    request.source_ref_json =
        Some("{\"type\":\"manual\",\"source_episode_id\":\"src-workspace\"}".to_string());
    let remembered = remember_memory(&path, &request).expect("remember succeeds");
    assert!(remembered.memory.source_episode_id.is_none());
    assert!(remembered.memory.source_ref_json.is_none());

    let hidden = get_memory(
        &path,
        &remembered.memory.id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: false,
        },
    )
    .expect("get without source succeeds");
    assert!(hidden.source_episode_id.is_none());
    assert!(hidden.source_ref_json.is_none());

    let visible = get_memory(
        &path,
        &remembered.memory.id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: true,
        },
    )
    .expect("get with source succeeds");
    assert_eq!(visible.source_episode_id.as_deref(), Some("src-workspace"));
    assert!(visible.source_ref_json.is_some());

    cleanup_store(&path);
}

#[test]
fn remember_rejects_cross_space_contradiction() {
    let path = temp_store_path("remember_rejects_cross_space_contradiction");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute_batch(
            "INSERT INTO spaces (name, display_name, description, default_silo, created_at, updated_at)
             VALUES ('other-space', 'Other', 'Other', 'durable', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
             INSERT INTO silos (space_name, name, description, retention_policy, default_scope, created_at, updated_at)
             VALUES ('other-space', 'durable', 'Durable', 'keep', 'workspace', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);",
        )
        .expect("insert other space");
    drop(connection);
    let mut other = basic_request("other space memory");
    other.space = Some("other-space".to_string());
    other.silo = Some("durable".to_string());
    let other_report = remember_memory(&path, &other).expect("remember other space");

    let mut request = basic_request("workspace contradiction");
    request.contradicts = vec![other_report.memory.id];
    let error = remember_memory(&path, &request).expect_err("cross-space contradiction fails");
    assert!(matches!(
        error,
        Error::NotFound {
            entity: "memory",
            ..
        }
    ));

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn search_uses_semantic_fallback_when_fts_has_no_results() {
    let path = temp_store_path("search_uses_semantic_fallback_when_fts_has_no_results");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = basic_request("decision: semantic-only memory content");
    request.embedding = Some(vec![0.25; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
    let remembered = remember_memory(&path, &request).expect("remember succeeds");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "unmatched lexical tokens".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "fallback".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: Some(vec![0.25; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]),
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.strategy, "semantic_primary_v0");
    assert!(report.semantic_attempted);
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].memory_id, remembered.memory.id);

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_supports_non_default_embedding_dimension() {
    // 1536 is deliberately not the 1024 default: dimension must not be a
    // hardcoded limit. The ANN index is created per-dimension on demand.
    let path = temp_store_path("semantic_supports_non_default_embedding_dimension");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let dims = 1536;
    let mut request = basic_request("decision: high dimension semantic memory");
    request.embedding = Some(vec![0.1_f32; dims]);
    let remembered = remember_memory(&path, &request).expect("remember succeeds");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "unmatched lexical tokens".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "fallback".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: Some(vec![0.1_f32; dims]),
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.strategy, "semantic_primary_v0");
    assert!(report.semantic_attempted);
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].memory_id, remembered.memory.id);

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_export_import_round_trip_rebuilds_vector_index() {
    let source = temp_store_path("semantic_round_trip_source");
    let target = temp_store_path("semantic_round_trip_target");
    let export_path = temp_store_path("semantic_round_trip_export").with_extension("jsonl");
    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&export_path);
    init_store(&source).expect("init source");

    let mut request = basic_request("decision: round-trip semantic memory");
    request.embedding = Some(vec![0.3_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
    request.embedding_model_id = Some("rt-model".to_string());
    let remembered = remember_memory(&source, &request).expect("remember source");

    export_store(
        &source,
        &ExportRequest {
            output_path: export_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export source");

    import_store(
        &target,
        &ImportRequest {
            input_path: export_path.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect("import into target");

    // Semantic search on the imported store must still find the memory,
    // proving the ANN index was rebuilt from the canonical embeddings table.
    let report = search_memories(
        &target,
        &SearchRequest {
            query: "no lexical overlap zzz".to_string(),
            filters: SearchFilters::default(),
            limit: 5,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "fallback".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: Some(vec![0.3_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]),
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("semantic search on imported store");
    assert_eq!(report.strategy, "semantic_primary_v0");
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].memory_id, remembered.memory.id);

    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&export_path);
}

#[cfg(feature = "semantic")]
#[test]
fn embedding_model_change_is_rejected() {
    let path = temp_store_path("embedding_model_change_is_rejected");
    cleanup_store(&path);
    init_store(&path).expect("init");

    let mut first = basic_request("decision: first model memory");
    first.embedding = Some(vec![0.1_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
    first.embedding_model_id = Some("model-a".to_string());
    remember_memory(&path, &first).expect("first remember");

    let mut second = basic_request("decision: second model memory");
    second.embedding = Some(vec![0.2_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
    second.embedding_model_id = Some("model-b".to_string());
    let error = remember_memory(&path, &second).expect_err("model change must be rejected");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    // The original model is still accepted.
    let mut third = basic_request("decision: same model memory");
    third.embedding = Some(vec![0.3_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
    third.embedding_model_id = Some("model-a".to_string());
    remember_memory(&path, &third).expect("same model remember");

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn reindex_vectors_reprojects_stored_embeddings() {
    let path = temp_store_path("reindex_vectors_reprojects_stored_embeddings");
    cleanup_store(&path);
    init_store(&path).expect("init");

    for index in 0..2 {
        let mut request = basic_request(&format!("decision: reindex memory {index}"));
        request.embedding = Some(vec![0.15_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
        request.embedding_model_id = Some("reindex-model".to_string());
        remember_memory(&path, &request).expect("remember");
    }

    // Re-projecting from the canonical embeddings table covers two stored rows.
    let count = crate::reindex_vectors(&path).expect("reindex");
    assert_eq!(count, 2);

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn reembed_replaces_vectors_and_switches_active_model() {
    let path = temp_store_path("reembed_replaces_vectors_and_switches_active_model");
    cleanup_store(&path);
    init_store(&path).expect("init");

    for index in 0..2 {
        let mut request = basic_request(&format!("decision: reembed memory {index}"));
        request.embedding = Some(vec![0.1_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
        request.embedding_model_id = Some("model-a".to_string());
        remember_memory(&path, &request).expect("remember");
    }

    let targets = crate::collect_reembed_targets(&path).expect("collect targets");
    assert_eq!(targets.len(), 2);

    // Re-embed under a new model AND a new dimension (768) to prove the switch.
    let new_dims = 768;
    let vectors: Vec<(String, String, Vec<f32>)> = targets
        .iter()
        .map(|target| {
            (
                target.memory_id.clone(),
                target.version_id.clone(),
                vec![0.2_f32; new_dims],
            )
        })
        .collect();
    let count = crate::apply_reembed(&path, "model-b", new_dims, &vectors).expect("reembed");
    assert_eq!(count, 2);

    // The active model is now model-b/768; a model-a write is rejected.
    let mut conflicting = basic_request("decision: stale model write");
    conflicting.embedding = Some(vec![0.3_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
    conflicting.embedding_model_id = Some("model-a".to_string());
    assert!(remember_memory(&path, &conflicting).is_err());

    // The new 768-dim index answers semantic queries for both memories.
    let report = search_memories(
        &path,
        &SearchRequest {
            query: "unmatched zzz".to_string(),
            filters: SearchFilters::default(),
            limit: 5,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "fallback".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: Some(vec![0.2_f32; new_dims]),
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search");
    assert_eq!(report.strategy, "semantic_primary_v0");
    assert_eq!(report.results.len(), 2);

    cleanup_store(&path);
}

#[test]
fn search_uses_fts_and_metadata_filters() {
    let path = temp_store_path("search_uses_fts_and_metadata_filters");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut first = basic_request("decision: use sqlite fts for fast memory search");
    first.kind = None;
    first.tags = vec!["search".to_string(), "sqlite".to_string()];
    first.claim_key = Some("search.primary".to_string());
    let first_report = remember_memory(&path, &first).expect("remember first");
    let mut second = basic_request("lesson: unrelated cooking note");
    second.kind = None;
    second.tags = vec!["kitchen".to_string()];
    remember_memory(&path, &second).expect("remember second");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "sqlite memory search".to_string(),
            filters: SearchFilters {
                kinds: vec!["decision".to_string()],
                tags: vec!["search".to_string()],
                ..SearchFilters::default()
            },
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "disabled".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.strategy, "deterministic_fts_v0");
    assert!(!report.semantic_attempted);
    assert_eq!(report.total_estimate, 1);
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].memory_id, first_report.memory.id);
    assert_eq!(report.results[0].kind, "decision");
    assert!(report.results[0].content.is_none());
    assert!(report.results[0].source_ref_json.is_none());
    assert!(report.results[0].snippet.contains("sqlite"));

    cleanup_store(&path);
}

#[test]
fn search_matches_any_query_term_with_bm25_ranking() {
    let path = temp_store_path("search_matches_any_query_term_with_bm25_ranking");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut first = basic_request("decision: Nora scene rules use scene breaks");
    first.kind = None;
    let first_report = remember_memory(&path, &first).expect("remember first");
    let mut second = basic_request("lesson: unrelated boundaries note");
    second.kind = None;
    let second_report = remember_memory(&path, &second).expect("remember second");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "nora boundaries".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    let ids: Vec<&str> = report
        .results
        .iter()
        .map(|result| result.memory_id.as_str())
        .collect();
    assert!(ids.contains(&first_report.memory.id.as_str()));
    assert!(ids.contains(&second_report.memory.id.as_str()));

    cleanup_store(&path);
}

#[test]
fn search_prefers_all_terms_before_any_term_fallback() {
    let path = temp_store_path("search_prefers_all_terms_before_any_term_fallback");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let exact = remember_memory(
        &path,
        &basic_request("decision: memkeeper deterministic topic3 exact match"),
    )
    .expect("remember exact");
    let partial = remember_memory(
        &path,
        &basic_request("decision: memkeeper deterministic topic4 partial match"),
    )
    .expect("remember partial");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "memkeeper deterministic topic3".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "disabled".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    let ids: Vec<&str> = report
        .results
        .iter()
        .map(|result| result.memory_id.as_str())
        .collect();
    assert!(ids.contains(&exact.memory.id.as_str()));
    assert!(!ids.contains(&partial.memory.id.as_str()));

    cleanup_store(&path);
}

#[test]
fn search_falls_back_to_prefix_terms() {
    let path = temp_store_path("search_falls_back_to_prefix_terms");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let expected = remember_memory(
        &path,
        &basic_request("decision: ai workspace memory uses deterministic sqlite search"),
    )
    .expect("remember expected");
    let distractor = remember_memory(
        &path,
        &basic_request("decision: workspace-only memory uses deterministic sqlite search"),
    )
    .expect("remember distractor");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "ai workspac".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    let ids: Vec<&str> = report
        .results
        .iter()
        .map(|result| result.memory_id.as_str())
        .collect();
    assert!(ids.contains(&expected.memory.id.as_str()));
    assert!(!ids.contains(&distractor.memory.id.as_str()));

    cleanup_store(&path);
}

#[test]
fn search_prefix_fallback_fills_after_exact_matches() {
    let path = temp_store_path("search_prefix_fallback_fills_after_exact_matches");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let exact = remember_memory(
        &path,
        &basic_request("decision: adopt deterministic sqlite search for workspace memory"),
    )
    .expect("remember exact");
    let prefix_only = remember_memory(
        &path,
        &basic_request("decision: adoption of deterministic sqlite search helped workspace memory"),
    )
    .expect("remember prefix-only");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "adopt".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    let ids: Vec<&str> = report
        .results
        .iter()
        .map(|result| result.memory_id.as_str())
        .collect();
    assert_eq!(ids.first().copied(), Some(exact.memory.id.as_str()));
    assert!(ids.contains(&prefix_only.memory.id.as_str()));

    cleanup_store(&path);
}

#[test]
fn search_lexical_fallback_disabled_keeps_exact_only_results() {
    let path = temp_store_path("search_lexical_fallback_disabled_keeps_exact_only_results");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let exact = remember_memory(
        &path,
        &basic_request("decision: adopt deterministic sqlite search for workspace memory"),
    )
    .expect("remember exact");
    let prefix_only = remember_memory(
        &path,
        &basic_request("decision: adoption of deterministic sqlite search helped workspace memory"),
    )
    .expect("remember prefix-only");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "adopt".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "disabled".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    let ids: Vec<&str> = report
        .results
        .iter()
        .map(|result| result.memory_id.as_str())
        .collect();
    assert_eq!(ids, vec![exact.memory.id.as_str()]);
    assert!(!ids.contains(&prefix_only.memory.id.as_str()));

    cleanup_store(&path);
}

#[test]
fn search_matches_inflected_term_variants() {
    let path = temp_store_path("search_matches_inflected_term_variants");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let expected = remember_memory(
        &path,
        &basic_request("decision: migrate workspace memory to sqlite"),
    )
    .expect("remember expected");
    let distractor = remember_memory(
        &path,
        &basic_request("decision: migrated project notes to a new folder"),
    )
    .expect("remember distractor");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "migrated workspaces".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "disabled".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    let ids: Vec<&str> = report
        .results
        .iter()
        .map(|result| result.memory_id.as_str())
        .collect();
    assert!(ids.contains(&expected.memory.id.as_str()));
    assert!(!ids.contains(&distractor.memory.id.as_str()));

    let connection = Connection::open(&path).expect("open store");
    let transaction = connection
        .unchecked_transaction()
        .expect("start transaction");
    rebuild_fts(&transaction).expect("rebuild fts");
    transaction.commit().expect("commit rebuild");

    let rebuilt_report = search_memories(
        &path,
        &SearchRequest {
            query: "migrated workspaces".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "disabled".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("rebuilt search succeeds");
    let rebuilt_ids: Vec<&str> = rebuilt_report
        .results
        .iter()
        .map(|result| result.memory_id.as_str())
        .collect();
    assert!(rebuilt_ids.contains(&expected.memory.id.as_str()));
    assert!(!rebuilt_ids.contains(&distractor.memory.id.as_str()));

    cleanup_store(&path);
}

#[test]
fn search_ignores_question_stopwords_and_possessive_artifacts() {
    let path = temp_store_path("search_ignores_question_stopwords_and_possessive_artifacts");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let expected = remember_memory(
        &path,
        &basic_request("decision: Nora boundary rules use scene breaks"),
    )
    .expect("remember expected");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "what did Nora's boundaries use".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    let ids: Vec<&str> = report
        .results
        .iter()
        .map(|result| result.memory_id.as_str())
        .collect();
    assert!(ids.contains(&expected.memory.id.as_str()));

    cleanup_store(&path);
}

#[test]
fn memory_list_reviews_recent_active_memories_without_source_by_default() {
    let path =
        temp_store_path("memory_list_reviews_recent_active_memories_without_source_by_default");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut first = basic_request("decision: first review memory");
    first.tags = vec!["review".to_string()];
    first.source_ref_json = Some("{\"type\":\"manual\",\"path\":\"/private/source\"}".to_string());
    let first_report = remember_memory(&path, &first).expect("remember first");
    let mut second = basic_request("lesson: second review memory is newer");
    second.tags = vec!["review".to_string()];
    let second_report = remember_memory(&path, &second).expect("remember second");

    let hidden = list_memories(
        &path,
        &MemoryListRequest {
            filters: SearchFilters {
                tags: vec!["review".to_string()],
                ..SearchFilters::default()
            },
            limit: 10,
            offset: 0,
            snippet_chars: 20,
            include_content: false,
            include_source: false,
            order: "updated_desc".to_string(),
        },
    )
    .expect("list succeeds");
    assert_eq!(hidden.strategy, "deterministic_list_v0");
    assert_eq!(hidden.results.len(), 2);
    assert_eq!(hidden.results[0].memory_id, second_report.memory.id);
    assert_eq!(hidden.results[1].memory_id, first_report.memory.id);
    assert!(hidden.results.iter().all(|result| result.content.is_none()));
    assert!(hidden
        .results
        .iter()
        .all(|result| result.source_ref_json.is_none()));
    assert!(hidden.results[0].snippet.chars().count() <= 20);

    let visible = list_memories(
        &path,
        &MemoryListRequest {
            filters: SearchFilters {
                tags: vec!["review".to_string()],
                ..SearchFilters::default()
            },
            limit: 1,
            offset: 1,
            snippet_chars: 80,
            include_content: true,
            include_source: true,
            order: "updated_desc".to_string(),
        },
    )
    .expect("source list succeeds");
    assert_eq!(visible.results.len(), 1);
    assert_eq!(visible.results[0].memory_id, first_report.memory.id);
    assert!(visible.results[0].content.is_some());
    assert!(visible.results[0].source_ref_json.is_some());

    let empty_page = list_memories(
        &path,
        &MemoryListRequest {
            filters: SearchFilters {
                tags: vec!["review".to_string()],
                ..SearchFilters::default()
            },
            limit: 10,
            offset: 1000,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            order: "updated_desc".to_string(),
        },
    )
    .expect("empty page list succeeds");
    assert!(empty_page.results.is_empty());
    assert_eq!(empty_page.total_estimate, 0);

    cleanup_store(&path);
}

#[test]
fn search_without_source_uses_source_free_bm25_score() {
    let path = temp_store_path("search_without_source_uses_source_free_bm25_score");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut first = basic_request("needle alpha");
    first.confidence = 1.0;
    remember_memory(&path, &first).expect("remember first");
    let mut second = basic_request("needle needle needle alpha");
    second.confidence = 1.0;
    remember_memory(&path, &second).expect("remember second");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "needle".to_string(),
            filters: SearchFilters::default(),
            limit: 2,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");
    assert_eq!(report.results.len(), 2);
    assert!(
        report
            .results
            .iter()
            .any(|result| (result.scores.fts - 1.0).abs() > f64::EPSILON),
        "expected source-free BM25 score, got {:?}",
        report
            .results
            .iter()
            .map(|result| result.scores.fts)
            .collect::<Vec<_>>()
    );

    cleanup_store(&path);
}

#[test]
fn search_without_source_does_not_match_source_only_terms() {
    let path = temp_store_path("search_without_source_does_not_match_source_only_terms");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut request = basic_request("decision: visible memory content");
    request.source_ref_json =
        Some("{\"type\":\"manual\",\"path\":\"/private/provenance-only-needle\"}".to_string());
    let remembered = remember_memory(&path, &request).expect("remember succeeds");

    let hidden = search_memories(
        &path,
        &SearchRequest {
            query: "provenance-only-needle".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("no-source search succeeds");
    assert!(hidden.results.is_empty());

    let mixed_hidden = search_memories(
        &path,
        &SearchRequest {
            query: "visible provenance-only-needle".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("mixed no-source search succeeds");
    assert_eq!(mixed_hidden.results.len(), 1);
    assert_eq!(mixed_hidden.results[0].memory_id, remembered.memory.id);
    assert!(mixed_hidden.results[0].source_ref_json.is_none());

    let explicit = search_memories(
        &path,
        &SearchRequest {
            query: "provenance-only-needle".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: true,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("source-enabled search succeeds");
    assert_eq!(explicit.results.len(), 1);
    assert_eq!(explicit.results[0].memory_id, remembered.memory.id);
    assert!(explicit.results[0].source_ref_json.is_some());

    cleanup_store(&path);
}

#[test]
fn search_excludes_expired_and_past_valid_to_but_memory_list_keeps_them() {
    let path = temp_store_path("search_excludes_expired_and_past_valid_to");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // All four share the query terms "retention policy" and a "rtest" tag.
    let mut fresh = basic_request("retention policy stays fresh");
    fresh.tags = vec!["rtest".to_string()];
    let fresh = remember_memory(&path, &fresh).expect("fresh");

    let mut future = basic_request("retention policy valid into the future");
    future.tags = vec!["rtest".to_string()];
    future.valid_to = Some("2999-01-01T00:00:00Z".to_string());
    let future = remember_memory(&path, &future).expect("future");

    let mut stale = basic_request("retention policy went stale long ago");
    stale.tags = vec!["rtest".to_string()];
    stale.valid_to = Some("2000-01-01T00:00:00Z".to_string());
    let stale = remember_memory(&path, &stale).expect("stale");

    let mut expired = basic_request("retention policy has already expired");
    expired.tags = vec!["rtest".to_string()];
    expired.expires_at = Some("2000-01-01T00:00:00Z".to_string());
    let expired = remember_memory(&path, &expired).expect("expired");

    // Search hides the past-valid_to and reached-expires_at memories.
    let report = search_memories(
        &path,
        &SearchRequest {
            query: "retention policy".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 120,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    let returned: Vec<&str> = report
        .results
        .iter()
        .map(|r| r.memory_id.as_str())
        .collect();
    assert!(returned.contains(&fresh.memory.id.as_str()), "fresh kept");
    assert!(
        returned.contains(&future.memory.id.as_str()),
        "future valid_to kept"
    );
    assert!(
        !returned.contains(&stale.memory.id.as_str()),
        "past valid_to hidden from search"
    );
    assert!(
        !returned.contains(&expired.memory.id.as_str()),
        "reached expires_at hidden from search"
    );
    assert_eq!(report.results.len(), 2, "only the two current memories");

    // memory-list is for review and must still surface the stale ones.
    let listed = list_memories(
        &path,
        &MemoryListRequest {
            filters: SearchFilters {
                tags: vec!["rtest".to_string()],
                ..SearchFilters::default()
            },
            limit: 10,
            offset: 0,
            snippet_chars: 40,
            include_content: false,
            include_source: false,
            order: "updated_desc".to_string(),
        },
    )
    .expect("list succeeds");
    assert_eq!(
        listed.results.len(),
        4,
        "memory-list keeps stale memories visible for cleanup"
    );

    cleanup_store(&path);
}

#[test]
fn search_defaults_to_active_workspace_and_supports_explicit_superseded_filter() {
    let path = temp_store_path(
        "search_defaults_to_active_workspace_and_supports_explicit_superseded_filter",
    );
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let old = remember_memory(&path, &basic_request("old sqlite decision")).expect("old");
    let mut replacement = basic_request("new sqlite decision");
    replacement.supersedes = vec![old.memory.id.clone()];
    let new = remember_memory(&path, &replacement).expect("new");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "sqlite decision".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 120,
            include_content: true,
            include_source: true,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].memory_id, new.memory.id);
    assert_eq!(report.results[0].status, "active");
    assert_eq!(
        report.results[0].content.as_deref(),
        Some("new sqlite decision")
    );

    let superseded = search_memories(
        &path,
        &SearchRequest {
            query: "sqlite decision".to_string(),
            filters: SearchFilters {
                statuses: vec!["superseded".to_string()],
                ..SearchFilters::default()
            },
            limit: 10,
            offset: 0,
            snippet_chars: 120,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("superseded search succeeds");
    assert_eq!(superseded.results.len(), 1);
    assert_eq!(superseded.results[0].memory_id, old.memory.id);
    assert_eq!(superseded.results[0].status, "superseded");

    cleanup_store(&path);
}

#[test]
fn search_recency_boosts_newer_observed_and_updated_timestamps() {
    let path = temp_store_path("search_recency_boosts_newer_observed_and_updated_timestamps");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut old = basic_request("needle recency tie");
    old.observed_at = Some("2020-01-01T00:00:00.000Z".to_string());
    let old_report = remember_memory(&path, &old).expect("remember old");
    let mut new = basic_request("needle recency tie");
    new.observed_at = Some("2026-01-01T00:00:00.000Z".to_string());
    let new_report = remember_memory(&path, &new).expect("remember new");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "needle recency".to_string(),
            filters: SearchFilters::default(),
            limit: 2,
            offset: 0,
            snippet_chars: 40,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.results.len(), 2);
    assert_eq!(report.results[0].memory_id, new_report.memory.id);
    assert_eq!(report.results[1].memory_id, old_report.memory.id);
    assert!(report.results[0].scores.recency > report.results[1].scores.recency);
    for result in &report.results {
        assert_score_components_add_up(result);
    }

    let connection = Connection::open(&path).expect("open store");
    connection
        .execute(
            "UPDATE memories SET updated_at = '2027-01-01T00:00:00.000Z' WHERE id = ?1",
            [&old_report.memory.id],
        )
        .expect("update old memory timestamp");
    drop(connection);

    let updated_report = search_memories(
        &path,
        &SearchRequest {
            query: "needle recency".to_string(),
            filters: SearchFilters::default(),
            limit: 2,
            offset: 0,
            snippet_chars: 40,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("updated search succeeds");

    assert_eq!(updated_report.results.len(), 2);
    assert_eq!(updated_report.results[0].memory_id, old_report.memory.id);
    assert_eq!(updated_report.results[1].memory_id, new_report.memory.id);
    assert!(updated_report.results[0].scores.recency > updated_report.results[1].scores.recency);
    for result in &updated_report.results {
        assert_score_components_add_up(result);
    }

    cleanup_store(&path);
}

#[test]
fn search_bm25_dominates_recency_when_relevance_is_stronger() {
    let path = temp_store_path("search_bm25_dominates_recency_when_relevance_is_stronger");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut relevant = basic_request(&format!("{} durable older note", "needle ".repeat(80)));
    relevant.summary = Some("needle needle needle".to_string());
    relevant.tags = vec!["needle".to_string()];
    relevant.observed_at = Some("2000-01-01T00:00:00.000Z".to_string());
    let relevant_report = remember_memory(&path, &relevant).expect("remember relevant");
    let mut recent = basic_request("needle");
    recent.observed_at = Some("2100-01-01T00:00:00.000Z".to_string());
    let recent_report = remember_memory(&path, &recent).expect("remember recent");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "needle".to_string(),
            filters: SearchFilters::default(),
            limit: 2,
            offset: 0,
            snippet_chars: 40,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.results.len(), 2);
    let top = &report.results[0];
    let runner_up = &report.results[1];
    assert_eq!(top.memory_id, relevant_report.memory.id);
    assert_eq!(runner_up.memory_id, recent_report.memory.id);
    assert!(top.scores.fts > runner_up.scores.fts);
    assert!(runner_up.scores.recency > top.scores.recency);
    assert!(
        top.scores.fts - runner_up.scores.fts > runner_up.scores.recency - top.scores.recency,
        "expected FTS delta to dominate recency delta: top={top:?} runner_up={runner_up:?}"
    );
    for result in &report.results {
        assert_score_components_add_up(result);
    }

    cleanup_store(&path);
}

#[test]
fn fts_score_normalizes_relative_to_best_match() {
    // Best (most negative) match anchors to 1.0; weaker matches scale down
    // proportionally; non-match (>= 0) bm25 scores 0.0.
    assert!((fts_score(-10.0, -10.0) - 1.0).abs() < 1e-9);
    assert!((fts_score(-5.0, -10.0) - 0.5).abs() < 1e-9);
    assert!((fts_score(-2.0, -10.0) - 0.2).abs() < 1e-9);
    assert!(fts_score(0.0, -10.0).abs() < 1e-9);
    // Regression guard: the old saturating map clamped everything to 10.0.
    assert!(fts_score(-15.8, -17.9) <= 1.0);
    assert!(fts_score(-15.8, -17.9) > 0.6);
    assert!(fts_score(-1.0, -17.9) < 0.6);
    // No result set (empty / no negative best) must not panic or saturate.
    assert!(fts_score(-5.0, f64::INFINITY).abs() < 1e-9);
}

#[test]
fn search_scores_discriminate_and_do_not_saturate() {
    // Regression test for the bm25 saturation bug: with several memories of
    // varying relevance, reported scores must differ (not all tie at a
    // clamped constant) and the strongest match must normalize to fts == 1.0.
    let path = temp_store_path("search_scores_discriminate_and_do_not_saturate");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // Strong match: many occurrences of the query term.
    remember_memory(
        &path,
        &basic_request(&format!("{}strong", "supervisor ".repeat(12))),
    )
    .expect("remember strong");
    // Weaker match: term appears once amid unrelated tokens.
    remember_memory(
        &path,
        &basic_request(
            "supervisor amid many other unrelated trading ledger report cycle tokens here",
        ),
    )
    .expect("remember weak");
    // Noise: does not contain the query term at all.
    remember_memory(
        &path,
        &basic_request("completely unrelated trading note about portfolios"),
    )
    .expect("remember noise");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "supervisor".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 40,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    assert!(report.results.len() >= 2, "expected multiple matches");
    // Top match normalizes to 1.0.
    assert!((report.results[0].scores.fts - 1.0).abs() < 1e-9);
    // FTS scores are not all identical -- ranking signal is preserved.
    let fts: Vec<f64> = report.results.iter().map(|r| r.scores.fts).collect();
    assert!(
        fts.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-6),
        "expected distinct fts scores, got {fts:?}"
    );
    // All fts scores are bounded in (0, 1].
    assert!(
        fts.iter().all(|&s| s > 0.0 && s <= 1.0 + 1e-9),
        "fts out of (0,1]: {fts:?}"
    );
    for result in &report.results {
        assert_score_components_add_up(result);
    }

    cleanup_store(&path);
}

#[test]
fn search_sql_limit_uses_final_score_order() {
    let path = temp_store_path("search_sql_limit_uses_final_score_order");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut low_a = basic_request("needle");
    low_a.confidence = 0.0;
    low_a.observed_at = Some("2026-05-25T00:00:00.000Z".to_string());
    remember_memory(&path, &low_a).expect("remember low a");
    let mut low_b = basic_request("needle");
    low_b.confidence = 0.0;
    low_b.observed_at = Some("2026-05-24T00:00:00.000Z".to_string());
    remember_memory(&path, &low_b).expect("remember low b");
    let mut pinned = basic_request("needle");
    pinned.confidence = 1.0;
    pinned.pinned = true;
    pinned.observed_at = Some("2020-01-01T00:00:00.000Z".to_string());
    let pinned_report = remember_memory(&path, &pinned).expect("remember pinned");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "needle".to_string(),
            filters: SearchFilters::default(),
            limit: 1,
            offset: 0,
            snippet_chars: 20,
            include_content: true,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].memory_id, pinned_report.memory.id);
    assert!(report.results[0].scores.pin > 0.0);

    cleanup_store(&path);
}

#[test]
fn search_normalizes_trimmed_tag_filter() {
    let path = temp_store_path("search_normalizes_trimmed_tag_filter");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut other_a = basic_request("needle");
    other_a.tags = vec!["other".to_string()];
    other_a.observed_at = Some("2026-05-25T00:00:00.000Z".to_string());
    remember_memory(&path, &other_a).expect("remember other a");
    let mut other_b = basic_request("needle");
    other_b.tags = vec!["other".to_string()];
    other_b.observed_at = Some("2026-05-24T00:00:00.000Z".to_string());
    remember_memory(&path, &other_b).expect("remember other b");
    let mut tagged = basic_request("needle");
    tagged.tags = vec!["tag".to_string()];
    tagged.observed_at = Some("2020-01-01T00:00:00.000Z".to_string());
    let tagged_report = remember_memory(&path, &tagged).expect("remember tagged");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "needle".to_string(),
            filters: SearchFilters {
                tags: vec![" tag ".to_string()],
                ..SearchFilters::default()
            },
            limit: 1,
            offset: 0,
            snippet_chars: 20,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].memory_id, tagged_report.memory.id);

    cleanup_store(&path);
}

#[test]
fn batch_search_runs_multiple_queries_with_common_filters() {
    let path = temp_store_path("batch_search_runs_multiple_queries_with_common_filters");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut storage = basic_request("decision: sqlite storage remains canonical");
    storage.kind = None;
    storage.tags = vec!["memory".to_string()];
    let storage_report = remember_memory(&path, &storage).expect("remember storage");
    let mut search = basic_request("lesson: deterministic search uses fts");
    search.kind = None;
    search.tags = vec!["memory".to_string()];
    let search_report = remember_memory(&path, &search).expect("remember search");
    let mut other = basic_request("decision: cooking storage note");
    other.tags = vec!["kitchen".to_string()];
    remember_memory(&path, &other).expect("remember other");

    let report = batch_search_memories(
        &path,
        &BatchSearchRequest {
            queries: vec![
                BatchSearchQuery {
                    name: Some("storage".to_string()),
                    query: "sqlite storage".to_string(),
                    limit: Some(5),
                },
                BatchSearchQuery {
                    name: Some("search".to_string()),
                    query: "deterministic fts".to_string(),
                    limit: None,
                },
            ],
            common_filters: SearchFilters {
                tags: vec!["memory".to_string()],
                ..SearchFilters::default()
            },
            limit: 5,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
        },
    )
    .expect("batch succeeds");

    assert_eq!(report.results.len(), 2);
    assert_eq!(report.results[0].name.as_deref(), Some("storage"));
    assert_eq!(
        report.results[0].report.results[0].memory_id,
        storage_report.memory.id
    );
    assert_eq!(
        report.results[1].report.results[0].memory_id,
        search_report.memory.id
    );
    assert!(report.results[0].report.results[0]
        .source_ref_json
        .is_none());

    cleanup_store(&path);
}

#[test]
fn pack_builds_bounded_deduped_markdown_without_source() {
    let path = temp_store_path("pack_builds_bounded_deduped_markdown_without_source");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut first = basic_request("decision: sqlite storage remains canonical for memkeeper");
    first.kind = None;
    first.summary = Some("SQLite stays canonical.".to_string());
    first.source_ref_json = Some("{\"type\":\"manual\",\"adapter\":\"host\"}".to_string());
    let first_report = remember_memory(&path, &first).expect("remember first");
    let mut second = basic_request("lesson: pack output should stay compact");
    second.kind = None;
    second.summary = Some("Memory packs stay compact.".to_string());
    let second_report = remember_memory(&path, &second).expect("remember second");

    let report = build_pack(
        &path,
        &PackRequest {
            title: "memkeeper implementation".to_string(),
            queries: vec!["sqlite storage".to_string(), "compact pack".to_string()],
            filters: SearchFilters::default(),
            max_memories: 10,
            max_chars: 2_000,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .expect("pack succeeds");

    assert_eq!(report.title, "memkeeper implementation");
    assert_eq!(report.format, "markdown");
    assert!(report.content.contains("## Retrieved Memory"));
    assert!(report.content.contains("SQLite stays canonical."));
    assert!(report.content.contains("Memory packs stay compact."));
    assert_eq!(
        report.memory_ids,
        vec![first_report.memory.id, second_report.memory.id]
    );
    assert!(!report.content.contains("adapter"));
    assert!(!report.truncated);

    let tiny = build_pack(
        &path,
        &PackRequest {
            title: "memkeeper implementation".to_string(),
            queries: vec!["sqlite storage".to_string()],
            filters: SearchFilters::default(),
            max_memories: 10,
            max_chars: 20,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .expect("tiny pack succeeds");
    assert!(tiny.truncated);
    assert!(tiny.content.chars().count() <= 20);

    cleanup_store(&path);
}

#[test]
fn pack_injects_truncated_top_memory_when_it_exceeds_char_budget() {
    // Regression: the non-rerank pack path (used whenever the reranker is absent
    // or fails) must inject the top eligible memory truncated to fit rather than
    // an empty pack when its line is by itself larger than the char budget. This
    // mirrors the `assemble_reranked_pack` budget fallback for the FTS/non-rerank
    // surface so a confidently-matched long memory is never dropped for length.
    let path = temp_store_path("pack_injects_truncated_top_memory_when_it_exceeds_char_budget");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = basic_request("decision: sqlite storage remains canonical for memkeeper");
    request.summary = Some(
        "SQLite remains the canonical store for memkeeper retrieval, and this summary is \
         deliberately written long enough that its rendered pack line cannot fit inside a \
         small char budget on its own, exercising the truncate-to-fit budget fallback."
            .to_string(),
    );
    remember_memory(&path, &request).expect("remember");

    // Budget large enough for the header but far too small for the full line.
    let report = build_pack(
        &path,
        &PackRequest {
            title: "budget fallback".to_string(),
            queries: vec!["sqlite storage".to_string()],
            filters: SearchFilters::default(),
            max_memories: 5,
            max_chars: 90,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .expect("pack succeeds");

    assert_eq!(
        report.memory_ids.len(),
        1,
        "long top memory should be injected truncated, not dropped"
    );
    assert!(report.truncated, "truncated flag must be set");
    assert!(
        report.content.chars().count() <= 90,
        "content must respect the char budget"
    );
    assert!(
        report
            .content
            .contains("## Retrieved Memory: budget fallback"),
        "header is preserved"
    );
    assert!(
        report.content.contains("- [fact:"),
        "the truncated entry keeps its leading kind/id marker"
    );
    assert!(
        report.content.ends_with("…\n"),
        "the truncated entry ends with the ellipsis marker"
    );

    cleanup_store(&path);
}

#[test]
fn pack_reports_per_memory_scores_aligned_with_ids() {
    let path = temp_store_path("pack_reports_per_memory_scores_aligned_with_ids");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    for i in 0..3 {
        remember_memory(
            &path,
            &basic_request(&format!("fact: sqlite retrieval note {i}")),
        )
        .expect("remember");
    }
    let report = build_pack(
        &path,
        &PackRequest {
            title: "t".to_string(),
            queries: vec!["sqlite retrieval".to_string()],
            filters: SearchFilters::default(),
            max_memories: 5,
            max_chars: 6000,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .expect("pack");
    assert_eq!(
        report.scores.len(),
        report.memory_ids.len(),
        "scores aligned 1:1 with memory_ids"
    );

    cleanup_store(&path);
}

#[test]
fn pack_min_score_floor_filters_low_scoring_memories() {
    let path = temp_store_path("pack_min_score_floor_filters_low_scoring_memories");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let request = basic_request("decision: sqlite storage remains canonical for memkeeper");
    remember_memory(&path, &request).expect("remember");

    let make = |min_score: f64| PackRequest {
        title: "floor".to_string(),
        queries: vec!["sqlite storage".to_string()],
        filters: SearchFilters::default(),
        max_memories: 10,
        max_chars: 2_000,
        format: "markdown".to_string(),
        min_score,
        rerank_candidates: 0,
        query_embeddings: None,
        query_token_embeddings: None,
        token_model_id: None,
    };

    let included = build_pack(&path, &make(0.0)).expect("pack without floor");
    assert_eq!(included.memory_ids.len(), 1);

    // A floor above any attainable score excludes every candidate.
    let excluded = build_pack(&path, &make(1_000.0)).expect("pack with high floor");
    assert!(excluded.memory_ids.is_empty());

    // A negative floor is rejected.
    let mut bad = make(0.0);
    bad.min_score = -1.0;
    assert!(build_pack(&path, &bad).is_err());

    cleanup_store(&path);
}

#[test]
fn pack_query_expansion_derives_bounded_action_and_decision_variants() {
    let queries = expanded_pack_queries(
        &["what did we decide about memkeeper retrieval diagnostics".to_string()],
        PackExpansionOptions {
            query_expansion: true,
            thread_expansion: false,
            max_query_variants: 4,
            max_thread_seeds: 3,
            max_thread_neighbors: 3,
        },
    );

    assert_eq!(
        queries,
        vec![
            "what did we decide about memkeeper retrieval diagnostics".to_string(),
            "memkeeper retrieval diagnostics".to_string(),
            "decision memkeeper retrieval diagnostics".to_string(),
            "action memkeeper retrieval diagnostics".to_string(),
        ]
    );
}

#[test]
fn pack_pool_thread_expansion_interleaves_same_entity_neighbors() {
    let path = temp_store_path("pack_pool_thread_expansion_interleaves_same_entity_neighbors");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut seed = basic_request("decision: memkeeper diagnostics should expose retrieval misses");
    seed.entity_key = Some("thread:fm-thread".to_string());
    seed.claim_key = Some("claim:diagnostics".to_string());
    seed.observed_at = Some("2026-06-01T00:00:00.000Z".to_string());
    let seed_report = remember_memory(&path, &seed).expect("remember seed");

    let mut sibling = basic_request("lesson: carry graph anchors forward into packs");
    sibling.entity_key = Some("thread:fm-thread".to_string());
    sibling.claim_key = Some("claim:thread-context".to_string());
    sibling.observed_at = Some("2026-06-02T00:00:00.000Z".to_string());
    let sibling_report = remember_memory(&path, &sibling).expect("remember sibling");

    let mut unrelated = basic_request("fact: unrelated benchmark fixture");
    unrelated.entity_key = Some("thread:other".to_string());
    unrelated.observed_at = Some("2026-06-03T00:00:00.000Z".to_string());
    remember_memory(&path, &unrelated).expect("remember unrelated");

    let request = PackRequest {
        title: "thread expansion".to_string(),
        queries: vec!["retrieval misses".to_string()],
        filters: SearchFilters::default(),
        max_memories: 4,
        max_chars: 2_000,
        format: "markdown".to_string(),
        min_score: 0.0,
        rerank_candidates: 0,
        query_embeddings: None,
        query_token_embeddings: None,
        token_model_id: None,
    };
    let base_pool = build_pack_pool_with_expansion(
        &path,
        &request,
        PackExpansionOptions {
            query_expansion: false,
            thread_expansion: false,
            max_query_variants: 4,
            max_thread_seeds: 3,
            max_thread_neighbors: 3,
        },
    )
    .expect("base pool");
    assert!(base_pool
        .iter()
        .any(|item| item.memory_id == seed_report.memory.id));
    assert!(!base_pool
        .iter()
        .any(|item| item.memory_id == sibling_report.memory.id));

    let expanded_pool = build_pack_pool_with_expansion(
        &path,
        &request,
        PackExpansionOptions {
            query_expansion: false,
            thread_expansion: true,
            max_query_variants: 4,
            max_thread_seeds: 1,
            max_thread_neighbors: 2,
        },
    )
    .expect("expanded pool");
    assert!(expanded_pool
        .iter()
        .any(|item| item.memory_id == sibling_report.memory.id));

    cleanup_store(&path);
}

fn rerank_pack_request(max_memories: usize, max_chars: usize, min_score: f64) -> PackRequest {
    PackRequest {
        title: "rr".to_string(),
        queries: vec!["q".to_string()],
        filters: SearchFilters::default(),
        max_memories,
        max_chars,
        format: "markdown".to_string(),
        min_score,
        rerank_candidates: 0,
        query_embeddings: None,
        query_token_embeddings: None,
        token_model_id: None,
    }
}

fn rc(id: &str, content: &str, score: f32) -> RerankCandidate {
    RerankCandidate {
        memory_id: id.to_string(),
        content: content.to_string(),
        rerank_score: score,
    }
}

#[test]
fn assemble_reranked_pack_budget_fallback_keeps_scores_aligned() {
    // A single candidate larger than the whole char budget: the main loop emits
    // nothing and the budget fallback injects the top candidate truncated. Scores
    // must stay 1:1 with memory_ids on THIS path specifically — the exact branch
    // the token-embedding port reconciled. Regression guard for silent misalignment.
    let request = rerank_pack_request(5, 20, 0.0);
    let long = "x".repeat(200);
    let candidates = vec![rc("only", &long, 0.8)];
    let report = assemble_reranked_pack(&request, 0.0, 0.0, &candidates);
    assert_eq!(
        report.memory_ids.len(),
        1,
        "budget fallback injects the top candidate"
    );
    assert_eq!(
        report.scores.len(),
        report.memory_ids.len(),
        "scores aligned 1:1 with memory_ids through the budget fallback"
    );
    assert_eq!(report.scores, vec![f64::from(0.8_f32)]);
    assert!(
        report.truncated,
        "fallback truncates the oversized candidate"
    );
}

#[test]
fn assemble_reranked_pack_orders_by_rerank_and_caps_memories() {
    let request = rerank_pack_request(2, 10_000, 0.0);
    let candidates = vec![
        rc("a", "low", 0.1),
        rc("b", "high", 0.9),
        rc("c", "mid", 0.5),
    ];
    // cosine_gate == 0.0 => legacy path, min_score floor is 0.0 (keeps all).
    let report = assemble_reranked_pack(&request, 0.0, 0.0, &candidates);
    assert_eq!(report.memory_ids, vec!["b".to_string(), "c".to_string()]);
    assert_eq!(report.content, "- high\n- mid\n");
    assert!(report.truncated, "3 candidates, 2 injected => truncated");
}

#[test]
fn assemble_reranked_pack_gate_blocks_off_topic() {
    let request = rerank_pack_request(5, 10_000, 0.2);
    let candidates = vec![rc("a", "x", 0.05), rc("b", "y", 0.10)];
    // cos_top below the gate AND rr_top (0.10) below min_score (0.2) => empty.
    let report = assemble_reranked_pack(&request, 0.30, 0.10, &candidates);
    assert!(report.memory_ids.is_empty());
    assert!(report.content.is_empty());
    assert!(!report.truncated);
}

#[test]
fn assemble_reranked_pack_gate_emits_on_cosine_without_item_floor() {
    let request = rerank_pack_request(5, 10_000, 0.5);
    let candidates = vec![rc("a", "x", 0.01), rc("b", "y", 0.02)];
    // cos_top (0.80) clears the gate, so injection proceeds and the low
    // cross-encoder scores are NOT floored on the gated path.
    let report = assemble_reranked_pack(&request, 0.30, 0.80, &candidates);
    assert_eq!(report.memory_ids, vec!["b".to_string(), "a".to_string()]);
}

#[test]
fn assemble_reranked_pack_gate_emits_on_reranker_confidence() {
    let request = rerank_pack_request(5, 10_000, 0.4);
    let candidates = vec![rc("a", "x", 0.45), rc("b", "y", 0.10)];
    // cos_top (0.05) below the gate, but rr_top (0.45) >= min_score (0.4) => emit.
    let report = assemble_reranked_pack(&request, 0.30, 0.05, &candidates);
    assert_eq!(report.memory_ids, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn assemble_reranked_pack_legacy_applies_per_item_floor() {
    let request = rerank_pack_request(5, 10_000, 0.4);
    let candidates = vec![rc("a", "x", 0.9), rc("b", "y", 0.5), rc("c", "z", 0.2)];
    // No gate (0.0): min_score is a per-item floor; c (0.2 < 0.4) is dropped.
    let report = assemble_reranked_pack(&request, 0.0, 0.0, &candidates);
    assert_eq!(report.memory_ids, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn assemble_reranked_pack_respects_char_budget() {
    // Budget fits only the first "- high\n" (7 chars); the second entry would
    // exceed it.
    let request = rerank_pack_request(5, 8, 0.0);
    let candidates = vec![rc("a", "high", 0.9), rc("b", "more", 0.5)];
    let report = assemble_reranked_pack(&request, 0.0, 0.0, &candidates);
    assert_eq!(report.memory_ids, vec!["a".to_string()]);
    assert_eq!(report.content, "- high\n");
    assert!(report.truncated);
}

#[test]
fn assemble_reranked_pack_sets_top_score_to_max_rerank() {
    let request = rerank_pack_request(5, 4000, 0.05);
    let candidates = vec![
        RerankCandidate {
            memory_id: "a".into(),
            content: "alpha".into(),
            rerank_score: 0.20,
        },
        RerankCandidate {
            memory_id: "b".into(),
            content: "beta".into(),
            rerank_score: 0.70,
        },
        RerankCandidate {
            memory_id: "c".into(),
            content: "gamma".into(),
            rerank_score: 0.40,
        },
    ];
    let report = assemble_reranked_pack(&request, 0.0, 0.0, &candidates);
    assert_eq!(report.top_score, Some(f64::from(0.70_f32)));
}

#[test]
fn empty_pack_has_no_top_score() {
    let request = rerank_pack_request(5, 4000, 0.05);
    assert_eq!(empty_pack(&request).top_score, None);
}

#[test]
fn empty_pack_and_cosine_gate_helpers() {
    let request = rerank_pack_request(5, 10_000, 0.0);
    let empty = empty_pack(&request);
    assert!(empty.memory_ids.is_empty());
    assert!(empty.content.is_empty());
    assert_eq!(empty.title, "rr");

    assert!(pack_blocked_by_cosine_gate(0.10, 0.30));
    assert!(!pack_blocked_by_cosine_gate(0.40, 0.30));
    // A disabled gate (0.0) never blocks.
    assert!(!pack_blocked_by_cosine_gate(0.0, 0.0));
}

#[test]
fn pack_rejects_excessive_rerank_candidates() {
    let path = temp_store_path("pack_rejects_excessive_rerank_candidates");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    remember_memory(&path, &basic_request("decision: rerank pool memory")).expect("remember");

    let request = PackRequest {
        title: "pool".to_string(),
        queries: vec!["rerank".to_string()],
        filters: SearchFilters::default(),
        max_memories: 5,
        max_chars: 2_000,
        format: "markdown".to_string(),
        min_score: 0.0,
        rerank_candidates: crate::MAX_PACK_MEMORIES + 1,
        query_embeddings: None,
        query_token_embeddings: None,
        token_model_id: None,
    };
    assert!(build_pack(&path, &request).is_err());

    let ok = PackRequest {
        rerank_candidates: crate::MAX_PACK_MEMORIES,
        ..request.clone()
    };
    assert!(build_pack(&path, &ok).is_ok());

    cleanup_store(&path);
}

#[test]
fn pack_round_robins_query_results_to_avoid_starving_later_queries() {
    let path = temp_store_path("pack_round_robins_query_results_to_avoid_starving_later_queries");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    for index in 0..5 {
        let mut noisy = basic_request(&format!(
            "fact: memkeeper noisy recent benchmark memory {index}"
        ));
        noisy.observed_at = Some(format!("2026-05-1{index}T00:00:00.000Z"));
        remember_memory(&path, &noisy).expect("remember noisy");
    }
    let mut policy = basic_request(
        "decision: Memkeeper workspace-memory stores concise non-secret durable memory for local retrieval",
    );
    policy.observed_at = Some("2020-01-01T00:00:00.000Z".to_string());
    let policy_report = remember_memory(&path, &policy).expect("remember policy");

    let report = build_pack(
        &path,
        &PackRequest {
            title: "fair query merge".to_string(),
            queries: vec![
                "memkeeper".to_string(),
                "memkeeper workspace memory local retrieval policy".to_string(),
            ],
            filters: SearchFilters::default(),
            max_memories: 3,
            max_chars: 2_000,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .expect("pack succeeds");

    assert!(report.memory_ids.contains(&policy_report.memory.id));
    assert!(report.content.contains("Memkeeper workspace-memory stores"));
    assert!(report.memory_ids.len() <= 3);

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn pack_with_query_embedding_uses_ann_search() {
    let path = temp_store_path("pack_with_query_embedding_uses_ann_search");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // Store a memory with a synthetic 1024-dim embedding (unit vector along dim 0).
    let fake_embedding: Vec<f32> = (0..crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS)
        .map(|i| if i == 0 { 1.0_f32 } else { 0.0_f32 })
        .collect();

    let mut req = basic_request("fact: mxbai is a local embedding model");
    req.embedding = Some(fake_embedding.clone());
    let remembered = remember_memory(&path, &req).expect("remember succeeds");

    // Pack with same embedding vector should find the memory via ANN.
    let report = build_pack(
        &path,
        &PackRequest {
            title: "ann pack test".to_string(),
            queries: vec!["embedding model".to_string()],
            filters: SearchFilters::default(),
            max_memories: 5,
            max_chars: 2_000,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: Some(vec![fake_embedding]),
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .expect("pack with ANN embedding succeeds");

    assert!(
        !report.memory_ids.is_empty(),
        "pack with ANN embedding should return at least one memory"
    );
    assert!(
        report.memory_ids.contains(&remembered.memory.id),
        "pack should contain the remembered memory id"
    );

    cleanup_store(&path);
}

#[test]
fn dream_graph_diagnostics_reports_orphans_and_bad_relationship_evidence() {
    let path =
        temp_store_path("dream_graph_diagnostics_reports_orphans_and_bad_relationship_evidence");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(&path, &entity_upsert_request("entity:orphan", "Orphan"))
        .expect("orphan upsert succeeds");
    upsert_entity(&path, &entity_upsert_request("entity:subject", "Subject"))
        .expect("subject upsert succeeds");
    upsert_entity(&path, &entity_upsert_request("entity:object", "Object"))
        .expect("object upsert succeeds");
    let evidence = remember_memory(
        &path,
        &RememberRequest {
            content: "fact: graph diagnostics evidence".to_string(),
            ..remember_request("graph diagnostics evidence")
        },
    )
    .expect("remember evidence succeeds");
    forget_memory(
        &path,
        &ForgetRequest {
            id: evidence.memory.id.clone(),
            reason: Some("make inactive evidence".to_string()),
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect("forget evidence succeeds");
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("entity:subject".to_string()),
            relation_type: "related_to".to_string(),
            object_entity_key: Some("entity:object".to_string()),
            memory_id: Some(evidence.memory.id.clone()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("relationship upsert succeeds");

    let connection = Connection::open(&path).expect("open store");
    connection
        .execute_batch("PRAGMA foreign_keys = OFF;")
        .expect("disable foreign keys");
    connection
        .execute(
            "INSERT INTO relationships (
                id, space_name, subject_entity_id, relation_type, object_entity_id,
                status, confidence, created_at, updated_at
             ) VALUES (
                'rel-dangling-diagnostics', 'workspace-memory', 'missing-subject',
                'related_to', 'missing-object', 'active', 1.0,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
             )",
            [],
        )
        .expect("insert dangling relationship");
    drop(connection);

    let report = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["graph".to_string()],
            max_memories: 10,
            dry_run: true,
            ..dream_request_defaults()
        },
    )
    .expect("graph dream succeeds");
    assert!(report.graph.attempted);
    assert_eq!(
        report.graph.orphan_entity_ids,
        vec![entity_id_for_key(&path, "entity:orphan")]
    );
    assert!(report
        .graph
        .inactive_evidence_relationship_ids
        .iter()
        .any(|id| id.starts_with("rel_")));
    assert_eq!(
        report.graph.dangling_relationship_ids,
        vec!["rel-dangling-diagnostics".to_string()]
    );
    assert!(report.graph.relationship_proposals.is_empty());
    assert!(!report.journaled);

    cleanup_store(&path);
}

#[test]
fn dream_graph_sweeps_edges_into_tombstoned_entity() {
    let path = temp_store_path("dream_graph_sweeps_edges_into_tombstoned_entity");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(&path, &entity_upsert_request("entity:subject", "Subject"))
        .expect("subject upsert");
    upsert_entity(&path, &entity_upsert_request("entity:object", "Object")).expect("object upsert");
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("entity:subject".to_string()),
            relation_type: "related_to".to_string(),
            object_entity_key: Some("entity:object".to_string()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("edge upsert");
    // Tombstone the object entity directly (e.g. junk-node cleanup), leaving
    // an active edge pointing into a now-tombstoned endpoint.
    upsert_entity(
        &path,
        &EntityUpsertRequest {
            status: Some("tombstoned".to_string()),
            ..entity_upsert_request("entity:object", "Object")
        },
    )
    .expect("tombstone object");

    let connection = Connection::open(&path).expect("open store");
    assert_eq!(
        active_edges_to(&connection, "entity:object", "related_to"),
        1
    );
    drop(connection);

    let report = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["graph".to_string()],
            max_memories: 10,
            dry_run: false,
            ..dream_request_defaults()
        },
    )
    .expect("graph dream succeeds");
    assert_eq!(report.graph.dangling_relationships, 1);

    let connection = Connection::open(&path).expect("open store");
    assert_eq!(
        active_edges_to(&connection, "entity:object", "related_to"),
        0
    );
    cleanup_store(&path);
}

// Pre-existing long integration test (already exceeded the clippy line cap
// before the entity-merge work); kept intact to preserve its coverage.
#[allow(clippy::too_many_lines)]
#[test]
fn dream_graph_reconciles_drift_when_not_dry_run() {
    let path = temp_store_path("dream_graph_reconciles_drift_when_not_dry_run");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // Drift fixtures: orphan entity, relationship with inactive evidence.
    upsert_entity(&path, &entity_upsert_request("entity:orphan", "Orphan")).expect("orphan upsert");
    upsert_entity(&path, &entity_upsert_request("entity:subject", "Subject"))
        .expect("subject upsert");
    upsert_entity(&path, &entity_upsert_request("entity:object", "Object")).expect("object upsert");
    let evidence = remember_memory(
        &path,
        &RememberRequest {
            content: "fact: graph reconcile evidence".to_string(),
            ..remember_request("graph reconcile evidence")
        },
    )
    .expect("remember evidence");
    forget_memory(
        &path,
        &ForgetRequest {
            id: evidence.memory.id.clone(),
            reason: Some("make inactive evidence".to_string()),
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect("forget evidence");
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("entity:subject".to_string()),
            relation_type: "related_to".to_string(),
            object_entity_key: Some("entity:object".to_string()),
            memory_id: Some(evidence.memory.id.clone()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("inactive-evidence relationship upsert");

    // Control: valid entities + active evidence + clean relationship; must survive.
    upsert_entity(
        &path,
        &entity_upsert_request("entity:keep-subject", "KeepSubject"),
    )
    .expect("keep-subject upsert");
    upsert_entity(
        &path,
        &entity_upsert_request("entity:keep-object", "KeepObject"),
    )
    .expect("keep-object upsert");
    let keep_evidence = remember_memory(
        &path,
        &RememberRequest {
            content: "fact: live evidence".to_string(),
            ..remember_request("live evidence")
        },
    )
    .expect("remember keep evidence");
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("entity:keep-subject".to_string()),
            relation_type: "supports".to_string(),
            object_entity_key: Some("entity:keep-object".to_string()),
            memory_id: Some(keep_evidence.memory.id.clone()),
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("control relationship upsert");

    // Dangling relationship: endpoints reference nonexistent entities.
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute_batch("PRAGMA foreign_keys = OFF;")
        .expect("disable foreign keys");
    connection
        .execute(
            "INSERT INTO relationships (
                id, space_name, subject_entity_id, relation_type, object_entity_id,
                status, confidence, created_at, updated_at
             ) VALUES (
                'rel-dangling-reconcile', 'workspace-memory', 'missing-subject',
                'related_to', 'missing-object', 'active', 1.0,
                CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
             )",
            [],
        )
        .expect("insert dangling relationship");
    drop(connection);

    let report = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["graph".to_string()],
            max_memories: 10,
            dry_run: false,
            ..dream_request_defaults()
        },
    )
    .expect("graph dream succeeds");
    assert!(report.graph.attempted);
    assert!(report.journaled);
    assert!(!report.graph.orphan_entity_ids.is_empty());
    assert!(!report.graph.dangling_relationship_ids.is_empty());
    assert!(!report.graph.inactive_evidence_relationship_ids.is_empty());

    let connection = Connection::open(&path).expect("reopen store");
    let status_of = |table: &str, id: &str| -> String {
        connection
            .query_row(
                &format!("SELECT status FROM {table} WHERE id = ?1"),
                params![id],
                |row| row.get::<_, String>(0),
            )
            .expect("row exists")
    };
    for id in &report.graph.orphan_entity_ids {
        assert_eq!(
            status_of("entities", id),
            "tombstoned",
            "orphan entity tombstoned"
        );
    }
    for id in &report.graph.dangling_relationship_ids {
        assert_eq!(
            status_of("relationships", id),
            "tombstoned",
            "dangling rel tombstoned"
        );
    }
    for id in &report.graph.inactive_evidence_relationship_ids {
        assert_eq!(
            status_of("relationships", id),
            "tombstoned",
            "inactive-evidence rel tombstoned"
        );
    }
    // Control entity and relationship remain active.
    let keep_entity = entity_id_for_key(&path, "entity:keep-subject");
    assert_eq!(status_of("entities", &keep_entity), "active");
    let control_status: String = connection
        .query_row(
            "SELECT status FROM relationships WHERE relation_type = 'supports'",
            [],
            |row| row.get(0),
        )
        .expect("control relationship exists");
    assert_eq!(control_status, "active");
    drop(connection);

    cleanup_store(&path);
}

#[test]
fn dream_graph_proposes_relationships_from_memory_links() {
    let path = temp_store_path("dream_graph_proposes_relationships_from_memory_links");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(&path, &entity_upsert_request("entity:subject", "Subject"))
        .expect("subject upsert succeeds");
    upsert_entity(&path, &entity_upsert_request("entity:object", "Object"))
        .expect("object upsert succeeds");
    let subject_memory = remember_memory(
        &path,
        &RememberRequest {
            entity_key: Some("entity:subject".to_string()),
            ..remember_request("subject memory")
        },
    )
    .expect("subject memory succeeds");
    let object_memory = remember_memory(
        &path,
        &RememberRequest {
            entity_key: Some("entity:object".to_string()),
            ..remember_request("object memory")
        },
    )
    .expect("object memory succeeds");
    let now = "2026-05-28T00:00:00.000Z";
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute(
            "INSERT INTO memory_links (
                src_memory_id, dst_memory_id, link_type, status, confidence, created_at
             ) VALUES (?1, ?2, 'supports', 'active', 1.0, ?3)",
            params![subject_memory.memory.id, object_memory.memory.id, now],
        )
        .expect("insert memory link");
    drop(connection);

    let report = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["graph".to_string()],
            max_memories: 10,
            dry_run: true,
            ..dream_request_defaults()
        },
    )
    .expect("graph dream succeeds");
    assert_eq!(report.graph.relationship_proposals.len(), 1);
    let proposal = &report.graph.relationship_proposals[0];
    assert_eq!(proposal.link_type, "supports");
    assert_eq!(proposal.relation_type, "supports");
    assert_eq!(proposal.subject_entity_key, "entity:subject");
    assert_eq!(proposal.object_entity_key, "entity:object");

    let applied = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["graph".to_string()],
            max_memories: 10,
            dry_run: false,
            ..dream_request_defaults()
        },
    )
    .expect("graph apply succeeds");
    assert_eq!(applied.graph.relationship_proposals.len(), 1);
    let connection = Connection::open(&path).expect("open store after apply");
    assert_eq!(
        active_edges_to(&connection, "entity:object", "supports"),
        1,
        "apply materializes proposed relationship"
    );
    drop(connection);
    let after_existing = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["graph".to_string()],
            max_memories: 10,
            dry_run: true,
            ..dream_request_defaults()
        },
    )
    .expect("graph dream succeeds");
    assert!(after_existing.graph.relationship_proposals.is_empty());

    cleanup_store(&path);
}

#[test]
#[allow(clippy::too_many_lines)]
fn dream_expires_reindexes_and_reports_duplicate_proposals() {
    let path = temp_store_path("dream_expires_reindexes_and_reports_duplicate_proposals");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut expired = basic_request("temporary expired memory");
    expired.expires_at = Some("2000-01-01T00:00:00Z".to_string());
    let expired_report = remember_memory(&path, &expired).expect("remember expired");

    let mut pinned = basic_request("pinned expired memory");
    pinned.expires_at = Some("2000-01-01T00:00:00.1Z".to_string());
    pinned.pinned = true;
    let pinned_report = remember_memory(&path, &pinned).expect("remember pinned");

    let duplicate_a = remember_memory(&path, &basic_request("duplicate exact content"))
        .expect("remember duplicate a");
    let duplicate_b = remember_memory(&path, &basic_request("duplicate exact content"))
        .expect("remember duplicate b");

    let request = DreamRequest {
        space: Some(DEFAULT_SPACE.to_string()),
        silos: vec!["durable".to_string()],
        tasks: vec!["all".to_string()],
        max_memories: 10,
        dry_run: true,
        include_pinned: false,
        promote_threshold: DEFAULT_PROMOTE_THRESHOLD,
        promote_score_floor: DEFAULT_PROMOTE_SCORE_FLOOR,
        promote_rank_cap: DEFAULT_PROMOTE_RANK_CAP,
    };
    let dry_run = dream_store(&path, &request).expect("dry-run dream succeeds");
    assert!(dry_run.dry_run);
    assert!(!dry_run.journaled);
    assert_eq!(dry_run.expire.expired, 1);
    assert_eq!(dry_run.expire.skipped_pinned, 1);
    assert_eq!(
        dry_run.expire.memory_ids,
        vec![expired_report.memory.id.clone()]
    );
    assert_eq!(
        dry_run.expire.skipped_pinned_ids,
        vec![pinned_report.memory.id.clone()]
    );
    assert_eq!(dry_run.reindex.memory_rows, 4);
    assert_eq!(dry_run.dedupe.proposals.len(), 1);
    assert_eq!(dry_run.dedupe.proposals[0].total_count, 2);
    assert_eq!(
        get_memory(
            &path,
            &expired_report.memory.id,
            GetOptions {
                include_history: false,
                include_links: true,
                include_source: false,
            },
        )
        .expect("expired memory still active after dry-run")
        .status,
        "active"
    );

    let mut commit_request = request.clone();
    commit_request.dry_run = false;
    let committed = dream_store(&path, &commit_request).expect("committed dream succeeds");
    assert!(!committed.dry_run);
    assert!(committed.journaled);
    assert_eq!(committed.expire.expired, 1);
    assert_eq!(committed.expire.skipped_pinned, 1);
    assert_eq!(committed.reindex.memory_rows, 4);
    assert_eq!(committed.dedupe.proposals.len(), 1);
    assert_eq!(
        committed.dedupe.proposals[0].duplicate_memory_ids,
        vec![duplicate_b.memory.id.clone()]
    );
    assert_eq!(
        committed.dedupe.proposals[0].canonical_memory_id,
        duplicate_a.memory.id
    );

    let expired_memory = get_memory(
        &path,
        &expired_report.memory.id,
        GetOptions {
            include_history: true,
            include_links: true,
            include_source: false,
        },
    )
    .expect("expired memory fetch");
    assert_eq!(expired_memory.status, "expired");
    assert!(
        expired_memory
            .events
            .as_ref()
            .expect("events present")
            .iter()
            .any(|event| event.event_type == "dream"
                && event.new_status.as_deref() == Some("expired"))
    );
    assert_eq!(
        get_memory(
            &path,
            &pinned_report.memory.id,
            GetOptions {
                include_history: false,
                include_links: true,
                include_source: false,
            },
        )
        .expect("pinned memory fetch")
        .status,
        "active"
    );
    let dream_runs: i64 = Connection::open(&path)
        .expect("open store")
        .query_row("SELECT COUNT(*) FROM dream_runs", [], |row| row.get(0))
        .expect("count dream runs");
    assert_eq!(dream_runs, 1);

    cleanup_store(&path);
}

#[test]
fn export_rejects_source_sidecar_output_aliases() {
    let path = temp_store_path("export_rejects_source_sidecar_output_aliases");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let source_sidecar = sidecar_path(&path, "-journal");

    let direct_error = export_store(
        &path,
        &ExportRequest {
            output_path: source_sidecar.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect_err("source sidecar export output should fail");
    assert!(matches!(direct_error, Error::Conflict { .. }));
    assert!(!source_sidecar.exists());

    let nested_error = export_store(
        &path,
        &ExportRequest {
            output_path: source_sidecar.join("export.jsonl"),
            format: "jsonl".to_string(),
        },
    )
    .expect_err("nested source sidecar export output should fail");
    assert!(matches!(nested_error, Error::Conflict { .. }));
    assert!(!source_sidecar.exists());

    let aliased = path
        .parent()
        .expect("parent")
        .join("missing")
        .join("..")
        .join(source_sidecar.file_name().expect("sidecar file name"));
    let alias_error = export_store(
        &path,
        &ExportRequest {
            output_path: aliased,
            format: "jsonl".to_string(),
        },
    )
    .expect_err("aliased source sidecar export output should fail");
    assert!(matches!(
        alias_error,
        Error::Conflict { .. } | Error::InvalidPath { .. }
    ));
    assert!(!source_sidecar.exists());

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let link_dir = temp_store_dir("export_source_sidecar_symlink_alias");
        let _ = fs::remove_file(&link_dir);
        let _ = fs::remove_dir_all(&link_dir);
        symlink(path.parent().expect("parent"), &link_dir).expect("create parent symlink");
        let symlink_alias = link_dir
            .join("missing")
            .join("..")
            .join(source_sidecar.file_name().expect("sidecar file name"));
        let symlink_error = export_store(
            &path,
            &ExportRequest {
                output_path: symlink_alias,
                format: "jsonl".to_string(),
            },
        )
        .expect_err("symlink aliased source sidecar export output should fail");
        assert!(matches!(
            symlink_error,
            Error::Conflict { .. } | Error::InvalidPath { .. }
        ));
        assert!(!source_sidecar.exists());
        let child_dir = temp_store_dir("export_source_sidecar_child_target");
        let child_link = temp_store_dir("export_source_sidecar_child_link");
        let _ = fs::remove_file(&child_link);
        let _ = fs::remove_dir_all(&child_dir);
        fs::create_dir(&child_dir).expect("create child target");
        symlink(&child_dir, &child_link).expect("create child symlink");
        let child_alias = child_link
            .join("..")
            .join(source_sidecar.file_name().expect("sidecar file name"));
        let child_alias_error = export_store(
            &path,
            &ExportRequest {
                output_path: child_alias,
                format: "jsonl".to_string(),
            },
        )
        .expect_err("child symlink parent-dir source sidecar export output should fail");
        assert!(matches!(
            child_alias_error,
            Error::Conflict { .. } | Error::InvalidPath { .. }
        ));
        assert!(!source_sidecar.exists());
        let _ = fs::remove_file(&child_link);
        let _ = fs::remove_dir_all(&child_dir);
        let _ = fs::remove_file(&link_dir);
    }

    cleanup_store(&path);
}

#[test]
fn export_writes_deterministic_jsonl_with_audit_history() {
    let path = temp_store_path("export_writes_deterministic_jsonl_with_audit_history");
    let export_a = temp_store_path("export_writes_deterministic_jsonl_a").with_extension("jsonl");
    let export_b = temp_store_path("export_writes_deterministic_jsonl_b").with_extension("jsonl");
    cleanup_store(&path);
    cleanup_store(&export_a);
    cleanup_store(&export_b);
    init_store(&path).expect("init succeeds");
    let mut request = basic_request("decision: export keeps audit history");
    request.tags = vec!["export".to_string()];
    request.source_ref_json =
        Some("{\"type\":\"manual\",\"path\":\"/private/export-source\"}".to_string());
    let remembered = remember_memory(&path, &request).expect("remember succeeds");
    forget_memory(
        &path,
        &ForgetRequest {
            id: remembered.memory.id,
            reason: Some("export tombstone".to_string()),
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect("forget succeeds");

    let first = export_store(
        &path,
        &ExportRequest {
            output_path: export_a.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("first export succeeds");
    let second = export_store(
        &path,
        &ExportRequest {
            output_path: export_b.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("second export succeeds");

    assert_private_file_mode(&export_a);
    assert_private_file_mode(&export_b);
    let first_bytes = fs::read(&export_a).expect("read first export");
    let second_bytes = fs::read(&export_b).expect("read second export");
    assert_eq!(first_bytes, second_bytes);
    assert_eq!(first.bytes, first_bytes.len() as u64);
    assert_eq!(first.sha256, sha256_hex(&first_bytes));
    assert_eq!(first.row_count, second.row_count);
    assert!(first.row_count > 0);
    let text = String::from_utf8(first_bytes).expect("export is utf8");
    assert!(text.starts_with("{\"type\":\"header\""));
    assert!(text.contains("\"table\":\"memories\""));
    assert!(text.contains("\"table\":\"memory_versions\""));
    assert!(text.contains("\"table\":\"memory_events\""));
    assert!(text.contains("/private/export-source"));
    assert!(text.contains("export tombstone"));
    assert!(!text.contains("\"table\":\"memory_fts\""));

    let overwrite_error = export_store(
        &path,
        &ExportRequest {
            output_path: export_a.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect_err("existing export output should fail");
    assert!(matches!(overwrite_error, Error::Conflict { .. }));

    cleanup_store(&path);
    cleanup_store(&export_a);
    cleanup_store(&export_b);
}

#[test]
fn import_round_trips_export_and_rebuilds_indexes() {
    let path = temp_store_path("import_round_trips_export_source");
    let import_path = temp_store_path("import_round_trips_export_target");
    let export_a = temp_store_path("import_round_trips_export_a").with_extension("jsonl");
    let export_b = temp_store_path("import_round_trips_export_b").with_extension("jsonl");
    cleanup_store(&path);
    cleanup_store(&import_path);
    cleanup_store(&export_a);
    cleanup_store(&export_b);
    init_store(&path).expect("init succeeds");
    let connection = Connection::open(&path).expect("open source store");
    connection
        .execute_batch(
            "INSERT INTO source_episodes (
                id, space_name, source_type, source_path, content, ingested_at,
                created_at, updated_at
             ) VALUES (
                'src-import', 'workspace-memory', 'manual', 'refs/import.md',
                'source episode import text', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP,
                CURRENT_TIMESTAMP
             );",
        )
        .expect("insert source episode");
    drop(connection);

    let mut request = basic_request("decision: import round trip keeps searchable content");
    request.tags = vec!["import".to_string(), "roundtrip".to_string()];
    request.source_episode_id = Some("src-import".to_string());
    request.source_ref_json =
        Some("{\"type\":\"manual\",\"path\":\"/private/import-source\"}".to_string());
    let remembered = remember_memory(&path, &request).expect("remember succeeds");
    let export = export_store(
        &path,
        &ExportRequest {
            output_path: export_a.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export succeeds");

    let dry_run = import_store(
        &import_path,
        &ImportRequest {
            input_path: export_a.clone(),
            format: "jsonl".to_string(),
            dry_run: true,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect("dry-run import succeeds");
    assert!(dry_run.dry_run);
    assert_eq!(dry_run.row_count, export.row_count);
    assert!(!import_path.exists());

    let imported = import_store(
        &import_path,
        &ImportRequest {
            input_path: export_a.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect("import succeeds");
    assert_eq!(imported.sha256, export.sha256);
    assert_eq!(imported.fts_memory_rows, 1);
    assert_eq!(imported.fts_source_episode_rows, 1);
    assert_private_file_mode(&import_path);

    let stats = store_stats(&import_path, true).expect("import stats");
    assert_eq!(stats.memory_count, 1);
    assert_eq!(stats.source_episode_count, 1);
    assert_eq!(stats.indexes.expect("indexes").fts_memory_rows, 1);
    let loaded = get_memory(
        &import_path,
        &remembered.memory.id,
        GetOptions {
            include_history: true,
            include_links: true,
            include_source: true,
        },
    )
    .expect("get imported memory");
    assert_eq!(loaded.source_episode_id.as_deref(), Some("src-import"));
    assert!(loaded
        .source_ref_json
        .as_deref()
        .is_some_and(|source| { source.contains("/private/import-source") }));

    export_store(
        &import_path,
        &ExportRequest {
            output_path: export_b.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("round-trip export succeeds");
    assert_eq!(
        fs::read(&export_a).expect("read source export"),
        fs::read(&export_b).expect("read round-trip export")
    );

    cleanup_store(&path);
    cleanup_store(&import_path);
    cleanup_store(&export_a);
    cleanup_store(&export_b);
}

#[test]
fn import_rejects_existing_target_and_cleans_failed_create() {
    let source = temp_store_path("import_failure_cleanup_source");
    let target = temp_store_path("import_failure_cleanup_target");
    let malformed_target = temp_store_path("import_failure_cleanup_malformed_target");
    let export_path = temp_store_path("import_failure_cleanup_export").with_extension("jsonl");
    let malformed_path =
        temp_store_path("import_failure_cleanup_bad_export").with_extension("jsonl");
    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&malformed_target);
    cleanup_store(&export_path);
    cleanup_store(&malformed_path);
    init_store(&source).expect("source init succeeds");
    init_store(&target).expect("target init succeeds");
    export_store(
        &source,
        &ExportRequest {
            output_path: export_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export succeeds");

    let existing_error = import_store(
        &target,
        &ImportRequest {
            input_path: export_path.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect_err("existing target should fail");
    assert!(matches!(existing_error, Error::Conflict { .. }));

    let mut lines = fs::read_to_string(&export_path)
        .expect("read export")
        .lines()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let _ = lines.pop();
    fs::write(&malformed_path, format!("{}\n", lines.join("\n"))).expect("write malformed");
    let malformed_error = import_store(
        &malformed_target,
        &ImportRequest {
            input_path: malformed_path.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect_err("malformed import should fail");
    assert!(matches!(malformed_error, Error::InvalidRequest { .. }));
    assert!(!malformed_target.exists());
    assert!(!sidecar_path(&malformed_target, "-wal").exists());
    assert!(!sidecar_path(&malformed_target, "-shm").exists());
    assert!(!sidecar_path(&malformed_target, "-journal").exists());

    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&malformed_target);
    cleanup_store(&export_path);
    cleanup_store(&malformed_path);
}

#[test]
fn import_rejects_invalid_storage_class_and_active_version() {
    let source = temp_store_path("import_rejects_invalid_rows_source");
    let export_path = temp_store_path("import_rejects_invalid_rows_export").with_extension("jsonl");
    let invalid_type_path =
        temp_store_path("import_rejects_invalid_rows_type").with_extension("jsonl");
    let invalid_version_path =
        temp_store_path("import_rejects_invalid_rows_version").with_extension("jsonl");
    let invalid_source_path =
        temp_store_path("import_rejects_invalid_rows_source_ref").with_extension("jsonl");
    let target = temp_store_path("import_rejects_invalid_rows_target");
    cleanup_store(&source);
    cleanup_store(&export_path);
    cleanup_store(&invalid_type_path);
    cleanup_store(&invalid_version_path);
    cleanup_store(&invalid_source_path);
    cleanup_store(&target);
    init_store(&source).expect("init succeeds");
    let mut request = basic_request("decision: import rejects malformed row values");
    request.source_ref_json =
        Some("{\"type\":\"manual\",\"path\":\"/private/malformed-source\"}".to_string());
    let remembered = remember_memory(&source, &request).expect("remember succeeds");
    export_store(
        &source,
        &ExportRequest {
            output_path: export_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export succeeds");

    let export_text = fs::read_to_string(&export_path).expect("read export");
    fs::write(
        &invalid_type_path,
        export_text.replace(
            "\"content\":\"decision: import rejects malformed row values\"",
            "\"content\":{\"$memkeeper_blob_hex\":\"00\"}",
        ),
    )
    .expect("write invalid type export");
    let type_error = import_store(
        &target,
        &ImportRequest {
            input_path: invalid_type_path.clone(),
            format: "jsonl".to_string(),
            dry_run: true,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect_err("invalid storage class should fail");
    assert!(matches!(type_error, Error::InvalidRequest { .. }));

    fs::write(
        &invalid_version_path,
        export_text.replace(
            &format!("\"active_version_id\":\"{}\"", remembered.memory.version_id),
            "\"active_version_id\":\"missing-version\"",
        ),
    )
    .expect("write invalid active version export");
    let version_error = import_store(
        &target,
        &ImportRequest {
            input_path: invalid_version_path.clone(),
            format: "jsonl".to_string(),
            dry_run: true,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect_err("invalid active version should fail");
    assert!(matches!(version_error, Error::InvalidRequest { .. }));

    fs::write(
        &invalid_source_path,
        export_text.replace(
            "\"source_ref_json\":\"{\\\"type\\\":\\\"manual\\\",\\\"path\\\":\\\"/private/malformed-source\\\"}\"",
            "\"source_ref_json\":\"not-json\"",
        ),
    )
    .expect("write invalid source ref export");
    let source_error = import_store(
        &target,
        &ImportRequest {
            input_path: invalid_source_path.clone(),
            format: "jsonl".to_string(),
            dry_run: true,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect_err("invalid source_ref_json should fail");
    assert!(matches!(source_error, Error::InvalidRequest { .. }));
    assert!(!target.exists());

    cleanup_store(&source);
    cleanup_store(&export_path);
    cleanup_store(&invalid_type_path);
    cleanup_store(&invalid_version_path);
    cleanup_store(&invalid_source_path);
    cleanup_store(&target);
}

#[test]
fn import_rejects_non_regular_input_path() {
    let input_dir = temp_store_dir("import_rejects_non_regular_input_path");
    let target = temp_store_path("import_rejects_non_regular_input_target");
    let _ = fs::remove_dir_all(&input_dir);
    cleanup_store(&target);
    fs::create_dir(&input_dir).expect("create input dir");

    let error = import_store(
        &target,
        &ImportRequest {
            input_path: input_dir.clone(),
            format: "jsonl".to_string(),
            dry_run: true,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect_err("directory input should fail");
    assert!(matches!(error, Error::InvalidPath { .. } | Error::Io(_)));
    assert!(!target.exists());

    let _ = fs::remove_dir_all(&input_dir);
    cleanup_store(&target);
}

#[test]
#[allow(clippy::too_many_lines)]
fn import_rejects_invalid_space_and_silo_config_json() {
    let source = temp_store_path("import_rejects_space_config_source");
    let export_path = temp_store_path("import_rejects_space_config_export").with_extension("jsonl");
    let bad_space_path =
        temp_store_path("import_rejects_space_config_bad_space").with_extension("jsonl");
    let bad_silo_path =
        temp_store_path("import_rejects_space_config_bad_silo").with_extension("jsonl");
    let bad_space_target = temp_store_path("import_rejects_space_config_bad_space_target");
    let bad_timestamp_path =
        temp_store_path("import_rejects_space_config_bad_timestamp").with_extension("jsonl");
    let bad_silo_target = temp_store_path("import_rejects_space_config_bad_silo_target");
    let bad_timestamp_target = temp_store_path("import_rejects_space_config_bad_timestamp_target");
    cleanup_store(&source);
    cleanup_store(&export_path);
    cleanup_store(&bad_space_path);
    cleanup_store(&bad_silo_path);
    cleanup_store(&bad_timestamp_path);
    cleanup_store(&bad_space_target);
    cleanup_store(&bad_silo_target);
    cleanup_store(&bad_timestamp_target);

    init_store(&source).expect("source init succeeds");
    create_space(
        &source,
        &SpaceCreateRequest {
            name: "config-space".to_string(),
            display_name: None,
            description: None,
            default_silo: Some("durable".to_string()),
            ontology: None,
            config_json: Some("{\"space\":true}".to_string()),
            if_not_exists: false,
        },
    )
    .expect("create config space");
    let connection = Connection::open(&source).expect("open source");
    connection
        .execute(
            "UPDATE silos SET config_json = '{\"silo\":true}'
             WHERE space_name = 'config-space' AND name = 'durable'",
            [],
        )
        .expect("set silo config");
    drop(connection);
    export_store(
        &source,
        &ExportRequest {
            output_path: export_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export succeeds");
    let export_text = fs::read_to_string(&export_path).expect("read export");

    fs::write(
        &bad_space_path,
        export_text.replace(
            "\"config_json\":\"{\\\"space\\\":true}\"",
            "\"config_json\":\"not-json\"",
        ),
    )
    .expect("write bad space archive");
    let bad_space_error = import_store(
        &bad_space_target,
        &ImportRequest {
            input_path: bad_space_path.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect_err("bad space config rejected");
    assert!(matches!(bad_space_error, Error::InvalidRequest { .. }));

    fs::write(
        &bad_silo_path,
        export_text.replace(
            "\"config_json\":\"{\\\"silo\\\":true}\"",
            "\"config_json\":\"not-json\"",
        ),
    )
    .expect("write bad silo archive");
    let bad_silo_error = import_store(
        &bad_silo_target,
        &ImportRequest {
            input_path: bad_silo_path.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect_err("bad silo config rejected");
    assert!(matches!(bad_silo_error, Error::InvalidRequest { .. }));

    let bad_timestamp_text = export_text
        .lines()
        .map(|line| {
            if line.contains("\"table\":\"spaces\"") && line.contains("config-space") {
                line.replacen(
                    "\"created_at\":\"",
                    &format!("\"created_at\":\"{}", "x".repeat(MAX_TIMESTAMP_CHARS + 1)),
                    1,
                )
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(&bad_timestamp_path, format!("{bad_timestamp_text}\n"))
        .expect("write bad timestamp archive");
    let bad_timestamp_error = import_store(
        &bad_timestamp_target,
        &ImportRequest {
            input_path: bad_timestamp_path.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect_err("bad space timestamp rejected");
    assert!(matches!(bad_timestamp_error, Error::InvalidRequest { .. }));

    cleanup_store(&source);
    cleanup_store(&export_path);
    cleanup_store(&bad_space_path);
    cleanup_store(&bad_silo_path);
    cleanup_store(&bad_timestamp_path);
    cleanup_store(&bad_space_target);
    cleanup_store(&bad_silo_target);
    cleanup_store(&bad_timestamp_target);
}

#[test]
fn import_rejects_memory_bound_and_space_bypass() {
    let source = temp_store_path("import_rejects_invariant_source");
    let export_path = temp_store_path("import_rejects_invariant_export").with_extension("jsonl");
    let oversized_path =
        temp_store_path("import_rejects_invariant_oversized").with_extension("jsonl");
    let cross_space_path =
        temp_store_path("import_rejects_invariant_cross_space").with_extension("jsonl");
    let target = temp_store_path("import_rejects_invariant_target");
    cleanup_store(&source);
    cleanup_store(&export_path);
    cleanup_store(&oversized_path);
    cleanup_store(&cross_space_path);
    cleanup_store(&target);
    init_store(&source).expect("init succeeds");
    let remembered = remember_memory(
        &source,
        &basic_request("decision: import rejects invariant bypass"),
    )
    .expect("remember succeeds");
    export_store(
        &source,
        &ExportRequest {
            output_path: export_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export succeeds");

    let export_text = fs::read_to_string(&export_path).expect("read export");
    fs::write(
        &oversized_path,
        export_text.replace(
            "\"content\":\"decision: import rejects invariant bypass\"",
            &format!("\"content\":\"{}\"", "x".repeat(MAX_CONTENT_CHARS + 1)),
        ),
    )
    .expect("write oversized export");
    assert!(matches!(
        import_store(&target, &import_request_for(&oversized_path, true))
            .expect_err("oversized import should fail"),
        Error::InvalidRequest { .. }
    ));

    let connection = Connection::open(&source).expect("open source");
    connection
        .execute_batch(
            "INSERT INTO spaces (name, default_silo, created_at, updated_at)
             VALUES ('other-space', 'durable', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
             INSERT INTO silos (space_name, name, retention_policy, default_scope, created_at, updated_at)
             VALUES ('other-space', 'durable', 'keep', 'workspace', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);
             INSERT INTO source_episodes (id, space_name, source_type, content, ingested_at, created_at, updated_at)
             VALUES ('src-cross-import', 'other-space', 'manual', 'source', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);",
        )
        .expect("insert cross-space source");
    connection
        .execute(
            "UPDATE memories SET source_episode_id = 'src-cross-import' WHERE id = ?1",
            [&remembered.memory.id],
        )
        .expect("mutate cross-space source");
    drop(connection);
    export_store(
        &source,
        &ExportRequest {
            output_path: cross_space_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("cross-space export succeeds");
    assert!(matches!(
        import_store(&target, &import_request_for(&cross_space_path, true))
            .expect_err("cross-space import should fail"),
        Error::InvalidRequest { .. }
    ));
    assert!(!target.exists());

    cleanup_store(&source);
    cleanup_store(&export_path);
    cleanup_store(&oversized_path);
    cleanup_store(&cross_space_path);
    cleanup_store(&target);
}

#[test]
fn import_rejects_invalid_source_episode_projection() {
    let source = temp_store_path("import_rejects_source_episode_source");
    let export_path =
        temp_store_path("import_rejects_source_episode_export").with_extension("jsonl");
    let target = temp_store_path("import_rejects_source_episode_target");
    cleanup_store(&source);
    cleanup_store(&export_path);
    cleanup_store(&target);
    init_store(&source).expect("init succeeds");
    let connection = Connection::open(&source).expect("open source");
    let source_hash = sha256_hex(b"source text");
    connection
        .execute(
            "INSERT INTO source_episodes (
                id, space_name, source_type, content, content_sha256,
                metadata_json, ingested_at, created_at, updated_at
             ) VALUES (
                'src-bad-metadata', 'workspace-memory', 'manual', 'source text',
                ?1, '{bad-json}', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP
             )",
            [&source_hash],
        )
        .expect("insert bad source episode");
    drop(connection);
    export_store(
        &source,
        &ExportRequest {
            output_path: export_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export succeeds");

    let error = import_store(&target, &import_request_for(&export_path, true))
        .expect_err("bad source episode should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));
    assert!(!target.exists());

    cleanup_store(&source);
    cleanup_store(&export_path);
    cleanup_store(&target);
}

#[test]
fn backup_creates_restorable_sqlite_snapshot() {
    let path = temp_store_path("backup_creates_restorable_sqlite_snapshot");
    let backup_path = temp_store_path("backup_creates_restorable_sqlite_snapshot_out");
    cleanup_store(&path);
    cleanup_store(&backup_path);
    init_store(&path).expect("init succeeds");
    remember_memory(
        &path,
        &basic_request("decision: backup preserves sqlite store"),
    )
    .expect("remember succeeds");
    let source_sidecar_backup = sidecar_path(&path, "-journal");
    let source_sidecar_error = backup_store(
        &path,
        &BackupRequest {
            output_path: source_sidecar_backup.clone(),
            format: "sqlite".to_string(),
        },
    )
    .expect_err("source sidecar backup output should fail");
    assert!(matches!(source_sidecar_error, Error::Conflict { .. }));
    assert!(!source_sidecar_backup.exists());

    let nested_sidecar_error = backup_store(
        &path,
        &BackupRequest {
            output_path: source_sidecar_backup.join("backup.sqlite"),
            format: "sqlite".to_string(),
        },
    )
    .expect_err("nested source sidecar backup output should fail");
    assert!(matches!(nested_sidecar_error, Error::Conflict { .. }));
    assert!(!source_sidecar_backup.exists());

    let case_variant_sidecar = path.parent().expect("parent").join(
        source_sidecar_backup
            .file_name()
            .expect("sidecar file name")
            .to_string_lossy()
            .to_uppercase(),
    );
    let case_variant_error = backup_store(
        &path,
        &BackupRequest {
            output_path: case_variant_sidecar,
            format: "sqlite".to_string(),
        },
    )
    .expect_err("case-variant source sidecar backup output should fail");
    assert!(matches!(case_variant_error, Error::Conflict { .. }));
    assert!(!source_sidecar_backup.exists());

    fs::write(sidecar_path(&backup_path, "-journal"), b"stale journal")
        .expect("write stale sidecar");
    let sidecar_error = backup_store(
        &path,
        &BackupRequest {
            output_path: backup_path.clone(),
            format: "sqlite".to_string(),
        },
    )
    .expect_err("stale backup sidecar should fail");
    assert!(matches!(sidecar_error, Error::Conflict { .. }));
    assert!(!backup_path.exists());
    cleanup_store_sidecars(&backup_path);

    let report = backup_store(
        &path,
        &BackupRequest {
            output_path: backup_path.clone(),
            format: "sqlite".to_string(),
        },
    )
    .expect("backup succeeds");
    assert!(backup_path.exists());
    assert_private_file_mode(&backup_path);
    assert!(!sidecar_path(&backup_path, "-wal").exists());
    assert!(!sidecar_path(&backup_path, "-shm").exists());
    assert!(!sidecar_path(&backup_path, "-journal").exists());
    let backup_bytes = fs::read(&backup_path).expect("read backup");
    assert_eq!(report.bytes, backup_bytes.len() as u64);
    assert_eq!(report.sha256, sha256_hex(&backup_bytes));
    assert!(report.page_count > 0);

    let stats = store_stats(&backup_path, true).expect("backup is initialized memkeeper store");
    assert_eq!(stats.memory_count, 1);
    assert_eq!(stats.indexes.expect("indexes").fts_memory_rows, 1);
    let overwrite_error = backup_store(
        &path,
        &BackupRequest {
            output_path: backup_path.clone(),
            format: "sqlite".to_string(),
        },
    )
    .expect_err("existing backup output should fail");
    assert!(matches!(overwrite_error, Error::Conflict { .. }));

    cleanup_store(&path);
    cleanup_store(&backup_path);
}

#[test]
fn batch_search_and_pack_reject_invalid_requests() {
    let path = temp_store_path("batch_search_and_pack_reject_invalid_requests");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let error = batch_search_memories(
        &path,
        &BatchSearchRequest {
            queries: Vec::new(),
            common_filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
        },
    )
    .expect_err("empty batch should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    let error = build_pack(
        &path,
        &PackRequest {
            title: "bad".to_string(),
            queries: vec!["sqlite".to_string()],
            filters: SearchFilters::default(),
            max_memories: 1,
            max_chars: 10,
            format: "json".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .expect_err("bad format should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    cleanup_store(&path);
}

#[test]
fn search_rejects_invalid_request() {
    let path = temp_store_path("search_rejects_invalid_request");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let error = search_memories(
        &path,
        &SearchRequest {
            query: "!!!".to_string(),
            filters: SearchFilters::default(),
            limit: 0,
            offset: 0,
            snippet_chars: 0,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect_err("invalid search should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    let error = search_memories(
        &path,
        &SearchRequest {
            query: "needle".to_string(),
            filters: SearchFilters::default(),
            limit: MAX_SEARCH_LIMIT + 1,
            offset: 0,
            snippet_chars: 0,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect_err("oversized limit should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    let error = search_memories(
        &path,
        &SearchRequest {
            query: "needle".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 0,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "noisy".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect_err("unknown lexical fallback should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    cleanup_store(&path);
}

#[test]
fn remember_dry_run_rolls_back() {
    let path = temp_store_path("remember_dry_run_rolls_back");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let request = RememberRequest {
        space: None,
        silo: None,
        scope: None,
        project_key: None,
        kind: Some("fact".to_string()),
        content: "dry run only".to_string(),
        summary: None,
        tags: Vec::new(),
        entity_key: None,
        claim_key: None,
        confidence: 1.0,
        observed_at: None,
        valid_from: None,
        valid_to: None,
        expires_at: None,
        source_ref_json: None,
        metadata_json: None,
        source_episode_id: None,
        pinned: false,
        supersedes: Vec::new(),
        contradicts: Vec::new(),
        embedding: None,
        embedding_model_id: None,
        token_embedding: None,
        token_embedding_model_id: None,
        dry_run: true,
        mode: "auto".to_string(),
    };

    let report = remember_memory(&path, &request).expect("dry run succeeds");
    assert_eq!(report.processing_status, "dry_run");
    let error = get_memory(
        &path,
        &report.memory.id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: false,
        },
    )
    .expect_err("dry run should not persist");
    assert!(matches!(error, Error::NotFound { .. }));
    assert_eq!(store_stats(&path, true).expect("stats").memory_count, 0);

    cleanup_store(&path);
}

#[test]
#[allow(clippy::too_many_lines)]
fn forget_tombstones_memory_and_history_hides_source_when_requested() {
    let path = temp_store_path("forget_tombstones_memory_and_history_hides_source_when_requested");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut request = basic_request("decision: forget keeps audit history");
    request.source_ref_json = Some("{\"type\":\"manual\",\"adapter\":\"host\"}".to_string());
    let remembered = remember_memory(&path, &request).expect("remember succeeds");

    let report = forget_memory(
        &path,
        &ForgetRequest {
            id: remembered.memory.id.clone(),
            reason: Some("no longer wanted".to_string()),
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect("forget succeeds");
    assert_eq!(report.memory_id, remembered.memory.id);
    assert_eq!(report.old_status, "active");
    assert_eq!(report.new_status, "tombstoned");
    assert!(!report.dry_run);

    let fetched = get_memory(
        &path,
        &remembered.memory.id,
        GetOptions {
            include_history: true,
            include_links: false,
            include_source: false,
        },
    )
    .expect("get tombstoned succeeds");
    assert_eq!(fetched.status, "tombstoned");
    assert!(fetched.deleted_at.is_some());
    assert!(fetched.source_ref_json.is_none());
    assert!(fetched.versions.as_ref().expect("versions")[0]
        .source_ref_json
        .is_none());
    assert_eq!(fetched.events.as_ref().expect("events").len(), 2);

    let default_search = search_memories(
        &path,
        &SearchRequest {
            query: "forget audit".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("default search succeeds");
    assert!(default_search.results.is_empty());

    let tombstoned_search = search_memories(
        &path,
        &SearchRequest {
            query: "forget audit".to_string(),
            filters: SearchFilters {
                statuses: vec!["tombstoned".to_string()],
                ..SearchFilters::default()
            },
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("tombstoned search succeeds");
    assert_eq!(tombstoned_search.results.len(), 1);
    assert_eq!(tombstoned_search.results[0].memory_id, remembered.memory.id);

    let history = memory_history(
        &path,
        &remembered.memory.id,
        HistoryOptions {
            limit: 10,
            include_source: false,
        },
    )
    .expect("history succeeds");
    assert_eq!(history.current_status, "tombstoned");
    assert!(!history.truncated);
    assert_eq!(history.events.len(), 2);
    assert_eq!(history.events[0].event_type, "remember");
    assert_eq!(history.events[1].event_type, "forget");
    assert_eq!(
        history.events[1].reason.as_deref(),
        Some("no longer wanted")
    );
    assert_eq!(history.versions.len(), 1);
    assert!(history.versions[0].source_ref_json.is_none());

    let stats = store_stats(&path, true).expect("stats succeeds");
    assert_eq!(stats.memory_count, 1);
    assert_eq!(stats.active_count, 0);
    assert_eq!(stats.indexes.expect("indexes").fts_memory_rows, 1);

    cleanup_store(&path);
}

#[test]
fn tombstoned_memory_cannot_be_superseded() {
    let path = temp_store_path("tombstoned_memory_cannot_be_superseded");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let old = remember_memory(&path, &basic_request("decision: old tombstone target"))
        .expect("remember old");
    forget_memory(
        &path,
        &ForgetRequest {
            id: old.memory.id.clone(),
            reason: Some("user forgot it".to_string()),
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect("forget succeeds");

    let mut replacement = basic_request("decision: replacement should not revive tombstone");
    replacement.supersedes = vec![old.memory.id.clone()];
    let error =
        remember_memory(&path, &replacement).expect_err("superseding tombstone should fail");
    assert!(matches!(error, Error::Conflict { .. }));

    let history = memory_history(
        &path,
        &old.memory.id,
        HistoryOptions {
            limit: 10,
            include_source: false,
        },
    )
    .expect("history succeeds");
    assert_eq!(history.current_status, "tombstoned");

    cleanup_store(&path);
}

#[test]
fn forget_dry_run_rolls_back_and_rejects_repeat() {
    let path = temp_store_path("forget_dry_run_rolls_back_and_rejects_repeat");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let remembered = remember_memory(&path, &basic_request("dry run tombstone")).expect("remember");

    let dry_run = forget_memory(
        &path,
        &ForgetRequest {
            id: remembered.memory.id.clone(),
            reason: None,
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: true,
        },
    )
    .expect("dry-run forget succeeds");
    assert!(dry_run.dry_run);
    let fetched = get_memory(
        &path,
        &remembered.memory.id,
        GetOptions {
            include_history: true,
            include_links: false,
            include_source: false,
        },
    )
    .expect("get active");
    assert_eq!(fetched.status, "active");
    assert_eq!(fetched.events.expect("events").len(), 1);

    forget_memory(
        &path,
        &ForgetRequest {
            id: remembered.memory.id.clone(),
            reason: None,
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect("forget succeeds");
    let error = forget_memory(
        &path,
        &ForgetRequest {
            id: remembered.memory.id.clone(),
            reason: None,
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect_err("repeat tombstone should fail");
    assert!(matches!(error, Error::Conflict { .. }));

    cleanup_store(&path);
}

#[test]
fn forget_correct_mode_records_correction_signal() {
    let path = temp_store_path("forget_correct_mode_records_correction_signal");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // The wrong memory and the replacement that holds the right answer.
    let wrong = remember_memory(&path, &basic_request("fact: the 03:00 cron is enabled"))
        .expect("remember wrong");
    let right = remember_memory(&path, &basic_request("fact: the 03:00 cron is disabled"))
        .expect("remember right");

    let report = forget_memory(
        &path,
        &ForgetRequest {
            id: wrong.memory.id.clone(),
            reason: Some("user says the cron is disabled, not enabled".to_string()),
            mode: "correct".to_string(),
            corrected_by: Some(right.memory.id.clone()),
            dry_run: false,
        },
    )
    .expect("correct succeeds");
    assert_eq!(report.new_status, "tombstoned");

    let fetched = get_memory(
        &path,
        &wrong.memory.id,
        GetOptions {
            include_history: true,
            include_links: true,
            include_source: false,
        },
    )
    .expect("get corrected memory");
    assert_eq!(fetched.status, "tombstoned");

    // The audit event is a distinct `correct`, not a routine `forget` -- this is
    // the explicit, queryable correction signal.
    let events = fetched.events.expect("events");
    assert!(
        events.iter().any(|event| event.event_type == "correct"),
        "expected a correct event, got {events:?}"
    );
    assert!(
        !events.iter().any(|event| event.event_type == "forget"),
        "a correction must not also emit a routine forget event"
    );

    // A `contradicts` edge points from the replacement to the wrong memory.
    let links = fetched.links.expect("links");
    let contradicts = links
        .iter()
        .find(|link| link.link_type == "contradicts")
        .expect("contradicts link recorded");
    assert_eq!(contradicts.src_memory_id, right.memory.id);
    assert_eq!(contradicts.dst_memory_id, wrong.memory.id);
    assert_eq!(contradicts.status, "active");

    cleanup_store(&path);
}

#[test]
fn correct_mode_rejects_corrected_by_self_and_plain_forget() {
    let path = temp_store_path("correct_mode_rejects_corrected_by_self_and_plain_forget");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let remembered =
        remember_memory(&path, &basic_request("fact: correction guard checks")).expect("remember");

    // corrected_by may not reference the memory being corrected.
    let error = forget_memory(
        &path,
        &ForgetRequest {
            id: remembered.memory.id.clone(),
            reason: None,
            mode: "correct".to_string(),
            corrected_by: Some(remembered.memory.id.clone()),
            dry_run: false,
        },
    )
    .expect_err("self correction should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    // corrected_by is meaningless outside correct mode.
    let error = forget_memory(
        &path,
        &ForgetRequest {
            id: remembered.memory.id.clone(),
            reason: None,
            mode: "tombstone".to_string(),
            corrected_by: Some("mem_other".to_string()),
            dry_run: false,
        },
    )
    .expect_err("corrected_by in tombstone mode should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    cleanup_store(&path);
}

#[test]
fn correction_event_data_json_captures_provenance() {
    use super::correction_event_data_json;

    let tags = vec![
        "synthesis-derived".to_string(),
        "session:2026-06-18_abcd1234".to_string(),
    ];
    let json = correction_event_data_json(false, "correct", Some("mem_replacement"), &tags);
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid json object");
    assert_eq!(value["mode"], "correct");
    assert_eq!(value["dry_run"], false);
    assert_eq!(value["corrected_by"], "mem_replacement");
    assert_eq!(value["synthesis_derived"], true);
    assert_eq!(value["session"], "2026-06-18_abcd1234");

    // No provenance tags and no replacement -> nulls and false, still valid.
    let bare = correction_event_data_json(false, "correct", None, &[]);
    let bare_value: serde_json::Value = serde_json::from_str(&bare).expect("valid json object");
    assert_eq!(bare_value["synthesis_derived"], false);
    assert!(bare_value["corrected_by"].is_null());
    assert!(bare_value["session"].is_null());
}

#[test]
fn forget_and_history_reject_invalid_requests() {
    let path = temp_store_path("forget_and_history_reject_invalid_requests");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let remembered =
        remember_memory(&path, &basic_request("invalid forget checks")).expect("remember");

    let error = forget_memory(
        &path,
        &ForgetRequest {
            id: remembered.memory.id.clone(),
            reason: None,
            mode: "hard_delete".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect_err("hard delete mode should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));
    let error = memory_history(
        &path,
        &remembered.memory.id,
        HistoryOptions {
            limit: 0,
            include_source: false,
        },
    )
    .expect_err("zero history limit should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

    cleanup_store(&path);
}

#[test]
fn remember_auto_supersedes_older_same_claim_operational_memory() {
    let path = temp_store_path("remember_auto_supersedes_older_same_claim_operational_memory");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut old = basic_request("preference: old Nora boundary guidance");
    old.entity_key = Some("project:nora".to_string());
    old.claim_key = Some("style.boundaries".to_string());
    old.observed_at = Some("2026-03-01T00:00:00Z".to_string());
    let old_report = remember_memory(&path, &old).expect("old remember");

    let mut new = basic_request("preference: new Nora boundary guidance");
    new.entity_key = Some("project:nora".to_string());
    new.claim_key = Some("style.boundaries".to_string());
    new.observed_at = Some("2026-04-01T00:00:00Z".to_string());
    let new_report = remember_memory(&path, &new).expect("new remember");

    assert_eq!(
        new_report.auto_superseded,
        vec![old_report.memory.id.clone()]
    );
    assert!(new_report.conflict_candidates.is_empty());
    let old_fetched = get_memory(
        &path,
        &old_report.memory.id,
        GetOptions {
            include_history: false,
            include_links: true,
            include_source: false,
        },
    )
    .expect("get old");
    assert_eq!(old_fetched.status, "superseded");
    assert!(old_fetched
        .links
        .expect("links")
        .iter()
        .any(
            |link| link.dst_memory_id == new_report.memory.id && link.link_type == "superseded_by"
        ));

    cleanup_store(&path);
}

#[test]
fn continuity_same_claim_returns_conflict_candidate_without_superseding() {
    let path =
        temp_store_path("continuity_same_claim_returns_conflict_candidate_without_superseding");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut old = basic_request("continuity: Elspeth's coat is green");
    old.kind = Some("continuity".to_string());
    old.entity_key = Some("character:elspeth".to_string());
    old.claim_key = Some("wardrobe.coat_color".to_string());
    old.observed_at = Some("2026-03-01T00:00:00Z".to_string());
    let old_report = remember_memory(&path, &old).expect("old continuity");

    let mut new = basic_request("continuity: Elspeth's coat is blue");
    new.kind = Some("continuity".to_string());
    new.entity_key = Some("character:elspeth".to_string());
    new.claim_key = Some("wardrobe.coat_color".to_string());
    new.observed_at = Some("2026-04-01T00:00:00Z".to_string());
    let new_report = remember_memory(&path, &new).expect("new continuity");

    assert!(new_report.auto_superseded.is_empty());
    assert_eq!(new_report.conflict_candidates.len(), 1);
    assert_eq!(
        new_report.conflict_candidates[0].memory_id,
        old_report.memory.id
    );
    let old_fetched = get_memory(
        &path,
        &new_report.conflict_candidates[0].memory_id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: false,
        },
    )
    .expect("get old");
    assert_eq!(old_fetched.status, "active");
    assert_eq!(store_stats(&path, true).expect("stats").active_count, 2);

    cleanup_store(&path);
}

#[test]
fn remember_supersedes_existing_memory() {
    let path = temp_store_path("remember_supersedes_existing_memory");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let old = remember_memory(&path, &basic_request("old fact")).expect("old remember");
    let mut request = basic_request("new fact");
    request.supersedes = vec![old.memory.id.clone()];
    let new = remember_memory(&path, &request).expect("new remember");

    let old_fetched = get_memory(
        &path,
        &old.memory.id,
        GetOptions {
            include_history: true,
            include_links: true,
            include_source: false,
        },
    )
    .expect("get old");
    assert_eq!(old_fetched.status, "superseded");
    assert!(old_fetched
        .links
        .expect("links")
        .iter()
        .any(|link| link.dst_memory_id == new.memory.id && link.link_type == "superseded_by"));
    assert_eq!(store_stats(&path, true).expect("stats").active_count, 1);

    cleanup_store(&path);
}

#[test]
fn sha256_matches_known_vector() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn init_creates_parent_directories() {
    let dir = temp_store_dir("init_creates_parent_directories");
    let path = dir.join("nested").join("store.sqlite");
    if dir.exists() {
        fs::remove_dir_all(&dir).expect("remove stale test dir");
    }

    let report = init_store(&path).expect("init succeeds");

    assert!(report.created);
    assert!(path.exists());
    fs::remove_dir_all(&dir).expect("remove test dir");
}

#[test]
fn stats_missing_store_is_not_initialized() {
    let path = temp_store_path("stats_missing_store_is_not_initialized");
    cleanup_store(&path);

    let error = store_stats(&path, true).expect_err("stats should fail");
    assert!(matches!(error, Error::NotInitialized { .. }));
}

#[test]
fn stats_empty_database_is_not_initialized() {
    let path = temp_store_path("stats_empty_database_is_not_initialized");
    cleanup_store(&path);
    Connection::open(&path).expect("create empty sqlite database");

    let error = store_stats(&path, true).expect_err("stats should fail");
    assert!(matches!(error, Error::NotInitialized { .. }));

    cleanup_store(&path);
}

#[test]
fn stats_partial_v1_database_is_not_initialized() {
    let path = temp_store_path("stats_partial_v1_database_is_not_initialized");
    cleanup_store(&path);
    let connection = Connection::open(&path).expect("create sqlite database");
    connection
        .execute_batch("PRAGMA user_version = 1;")
        .expect("set partial schema");
    drop(connection);

    let error = store_stats(&path, true).expect_err("stats should fail");
    assert!(matches!(error, Error::NotInitialized { .. }));

    cleanup_store(&path);
}

#[test]
fn stats_newer_schema_is_mismatch() {
    let path = temp_store_path("stats_newer_schema_is_mismatch");
    cleanup_store(&path);
    let connection = Connection::open(&path).expect("create sqlite database");
    connection
        .execute_batch("PRAGMA user_version = 99;")
        .expect("set future schema");

    let error = store_stats(&path, true).expect_err("stats should fail");
    assert!(matches!(
        error,
        Error::SchemaMismatch {
            expected: SCHEMA_VERSION,
            actual: 99
        }
    ));

    cleanup_store(&path);
}

fn assert_score_components_add_up(result: &SearchResult) {
    let expected = result.scores.fts
        + result.scores.metadata
        + result.scores.recency
        + result.scores.scope
        + result.scores.status
        + result.scores.pin
        + result.scores.source_tier;
    assert!(
        (result.score - expected).abs() < 1e-12,
        "score should equal score components: {result:?}"
    );
}

fn entity_upsert_request(entity_key: &str, canonical_name: &str) -> EntityUpsertRequest {
    EntityUpsertRequest {
        space: None,
        entity_key: entity_key.to_string(),
        entity_type: None,
        canonical_name: canonical_name.to_string(),
        aliases: Vec::new(),
        status: None,
        confidence: 1.0,
        source_episode_id: None,
        metadata_json: None,
        include_source: false,
    }
}

fn dream_request_defaults() -> DreamRequest {
    DreamRequest {
        space: None,
        silos: Vec::new(),
        tasks: Vec::new(),
        max_memories: 1_000,
        dry_run: true,
        include_pinned: false,
        promote_threshold: DEFAULT_PROMOTE_THRESHOLD,
        promote_score_floor: DEFAULT_PROMOTE_SCORE_FLOOR,
        promote_rank_cap: DEFAULT_PROMOTE_RANK_CAP,
    }
}

fn candidate_submit_request(content: &str) -> CandidateSubmitRequest {
    CandidateSubmitRequest {
        space: None,
        silo: None,
        scope: None,
        project: None,
        kind: None,
        content: content.to_string(),
        summary: None,
        rationale: None,
        tags: Vec::new(),
        entity_key: None,
        claim_key: None,
        confidence: 1.0,
        source_type: None,
        source_json: None,
        sensitivity: None,
        supersedes: Vec::new(),
        dry_run: false,
    }
}

/// A remember request keyed to a fixed entity/claim so supersession modes
/// have a same-key target to resolve against.
fn keyed_request(content: &str) -> RememberRequest {
    let mut request = remember_request(content);
    request.entity_key = Some("embed".to_string());
    request.claim_key = Some("provider".to_string());
    request.kind = Some("decision".to_string());
    request
}

fn active_count(path: &Path) -> i64 {
    store_stats(path, false).expect("stats").active_count
}

#[test]
fn candidate_submit_list_approve_promotes_to_memory() {
    let path = temp_store_path("candidate_submit_list_approve");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = candidate_submit_request("prefer tabs over spaces in code");
    request.kind = Some("preference".to_string());
    request.source_type = Some("explicit-user".to_string());
    request.rationale = Some("Stated directly in conversation.".to_string());
    let submitted = submit_candidate(&path, &request).expect("submit succeeds");
    assert_eq!(submitted.candidate.status, "pending");
    assert_eq!(submitted.candidate.source_type, "explicit-user");
    assert!(submitted.candidate.resulting_memory_id.is_none());
    let candidate_id = submitted.candidate.id.clone();

    let pending = list_candidates(
        &path,
        &CandidateListRequest {
            status: Some("pending".to_string()),
            space: None,
            limit: 50,
            offset: 0,
        },
    )
    .expect("list pending");
    assert_eq!(pending.total, 1);
    assert_eq!(pending.candidates[0].id, candidate_id);

    let approved = approve_candidate(
        &path,
        &CandidateApproveRequest {
            id: candidate_id.clone(),
            embedding: None,
            embedding_model_id: None,
            dry_run: false,
        },
    )
    .expect("approve succeeds");
    assert_eq!(approved.candidate.status, "approved");
    assert_eq!(approved.memory.status, "active");
    assert_eq!(approved.memory.kind, "preference");
    assert_eq!(
        approved.candidate.resulting_memory_id.as_deref(),
        Some(approved.memory.id.as_str())
    );

    let memory = get_memory(
        &path,
        &approved.memory.id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: true,
        },
    )
    .expect("get succeeds");
    assert_eq!(memory.content, "prefer tabs over spaces in code");
    assert!(memory
        .source_ref_json
        .as_deref()
        .unwrap_or_default()
        .contains("\"source_type\":\"explicit-user\""));

    let pending_after = list_candidates(
        &path,
        &CandidateListRequest {
            status: Some("pending".to_string()),
            space: None,
            limit: 50,
            offset: 0,
        },
    )
    .expect("list pending after");
    assert_eq!(pending_after.total, 0);

    cleanup_store(&path);
}

#[test]
fn candidate_reject_marks_rejected_with_reason() {
    let path = temp_store_path("candidate_reject");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let submitted =
        submit_candidate(&path, &candidate_submit_request("a noisy guess")).expect("submit");
    let rejected = reject_candidate(
        &path,
        &CandidateRejectRequest {
            id: submitted.candidate.id.clone(),
            reason: Some("duplicate".to_string()),
            dry_run: false,
        },
    )
    .expect("reject succeeds");
    assert_eq!(rejected.candidate.status, "rejected");
    assert_eq!(
        rejected.candidate.decided_reason.as_deref(),
        Some("duplicate")
    );
    assert!(rejected.candidate.decided_at.is_some());

    cleanup_store(&path);
}

#[test]
fn candidate_decision_rejects_non_pending() {
    let path = temp_store_path("candidate_non_pending");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let submitted =
        submit_candidate(&path, &candidate_submit_request("approve me once")).expect("submit");
    let id = submitted.candidate.id.clone();
    approve_candidate(
        &path,
        &CandidateApproveRequest {
            id: id.clone(),
            embedding: None,
            embedding_model_id: None,
            dry_run: false,
        },
    )
    .expect("first approve succeeds");

    let second = approve_candidate(
        &path,
        &CandidateApproveRequest {
            id: id.clone(),
            embedding: None,
            embedding_model_id: None,
            dry_run: false,
        },
    );
    assert!(matches!(second, Err(Error::InvalidRequest { .. })));
    let reject = reject_candidate(
        &path,
        &CandidateRejectRequest {
            id,
            reason: None,
            dry_run: false,
        },
    );
    assert!(matches!(reject, Err(Error::InvalidRequest { .. })));

    cleanup_store(&path);
}

#[test]
fn candidate_submit_validates_enums_and_dry_run() {
    let path = temp_store_path("candidate_validate");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut bad_source = candidate_submit_request("bad source");
    bad_source.source_type = Some("rumor".to_string());
    assert!(matches!(
        submit_candidate(&path, &bad_source),
        Err(Error::InvalidRequest { .. })
    ));

    let mut bad_sensitivity = candidate_submit_request("bad sensitivity");
    bad_sensitivity.sensitivity = Some("top-secret".to_string());
    assert!(matches!(
        submit_candidate(&path, &bad_sensitivity),
        Err(Error::InvalidRequest { .. })
    ));

    let mut dry = candidate_submit_request("not persisted");
    dry.dry_run = true;
    let report = submit_candidate(&path, &dry).expect("dry-run submit succeeds");
    assert!(report.dry_run);
    let all = list_candidates(
        &path,
        &CandidateListRequest {
            status: None,
            space: None,
            limit: 50,
            offset: 0,
        },
    )
    .expect("list all");
    assert_eq!(all.total, 0);

    cleanup_store(&path);
}

#[test]
fn remember_mode_append_coexists() {
    let path = temp_store_path("remember_mode_append");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    remember_memory(&path, &keyed_request("decision: use library A")).expect("a");
    let mut b = keyed_request("decision: use library B");
    b.mode = "append".to_string();
    let report = remember_memory(&path, &b).expect("append write");
    assert!(report.auto_superseded.is_empty());
    assert_eq!(active_count(&path), 2);
    cleanup_store(&path);
}

#[test]
fn remember_mode_supersede_force_retires_same_key() {
    let path = temp_store_path("remember_mode_supersede");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    remember_memory(&path, &keyed_request("decision: use library A")).expect("a");
    let mut b = keyed_request("decision: use library B");
    b.mode = "supersede".to_string();
    let report = remember_memory(&path, &b).expect("supersede write");
    assert_eq!(report.auto_superseded.len(), 1);
    assert_eq!(active_count(&path), 1);
    cleanup_store(&path);
}

#[test]
fn remember_mode_suggest_previews_without_mutation() {
    let path = temp_store_path("remember_mode_suggest");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let a = remember_memory(&path, &keyed_request("decision: use library A")).expect("a");
    let mut b = keyed_request("decision: use library B");
    b.mode = "suggest".to_string();
    let report = remember_memory(&path, &b).expect("suggest write");
    assert_eq!(report.supersede_suggestions, vec![a.memory.id]);
    assert!(report.auto_superseded.is_empty());
    assert_eq!(active_count(&path), 2, "suggest must not mutate");
    cleanup_store(&path);
}

#[test]
fn remember_mode_conflict_opens_conflict() {
    let path = temp_store_path("remember_mode_conflict");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    remember_memory(&path, &keyed_request("decision: use library A")).expect("a");
    let mut b = keyed_request("decision: use library B");
    b.mode = "conflict".to_string();
    let report = remember_memory(&path, &b).expect("conflict write");
    assert!(report.auto_superseded.is_empty());
    assert_eq!(report.conflict_candidates.len(), 1);
    assert_eq!(active_count(&path), 2, "conflict must not supersede");
    cleanup_store(&path);
}

#[test]
fn remember_rejects_unknown_mode_and_invalid_source_provenance() {
    let path = temp_store_path("remember_mode_validation");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut bad_mode = remember_request("fact: x");
    bad_mode.mode = "bogus".to_string();
    assert!(matches!(
        remember_memory(&path, &bad_mode),
        Err(Error::InvalidRequest { .. })
    ));

    let mut bad_source = remember_request("fact: y");
    bad_source.source_ref_json = Some(r#"{"source_type":"rumor"}"#.to_string());
    assert!(matches!(
        remember_memory(&path, &bad_source),
        Err(Error::InvalidRequest { .. })
    ));

    let mut good = remember_request("fact: z");
    good.source_ref_json =
        Some(r#"{"source_type":"explicit-user","sensitivity":"sensitive"}"#.to_string());
    remember_memory(&path, &good).expect("valid provenance accepted");

    cleanup_store(&path);
}

fn remember_request(content: &str) -> RememberRequest {
    RememberRequest {
        space: None,
        silo: None,
        scope: None,
        project_key: None,
        kind: None,
        content: content.to_string(),
        summary: None,
        tags: Vec::new(),
        entity_key: None,
        claim_key: None,
        confidence: 1.0,
        observed_at: None,
        valid_from: None,
        valid_to: None,
        expires_at: None,
        source_ref_json: None,
        metadata_json: None,
        source_episode_id: None,
        pinned: false,
        supersedes: Vec::new(),
        contradicts: Vec::new(),
        embedding: None,
        embedding_model_id: None,
        token_embedding: None,
        token_embedding_model_id: None,
        dry_run: false,
        mode: "auto".to_string(),
    }
}

fn relationship_upsert_request_defaults() -> RelationshipUpsertRequest {
    RelationshipUpsertRequest {
        space: None,
        subject_entity_id: None,
        subject_entity_key: None,
        relation_type: String::new(),
        object_entity_id: None,
        object_entity_key: None,
        memory_id: None,
        source_episode_id: None,
        status: None,
        confidence: 1.0,
        observed_at: None,
        valid_from: None,
        valid_to: None,
        metadata_json: None,
        include_source: false,
    }
}

fn entity_search_defaults() -> EntitySearchRequest {
    EntitySearchRequest {
        space: None,
        query: None,
        entity_key: None,
        entity_types: Vec::new(),
        statuses: Vec::new(),
        limit: 20,
        offset: 0,
        include_source: false,
    }
}

fn graph_neighbors_defaults() -> GraphNeighborsRequest {
    GraphNeighborsRequest {
        space: None,
        entity_id: None,
        entity_key: None,
        depth: 1,
        relation_types: Vec::new(),
        statuses: Vec::new(),
        max_edges: 50,
        include_tombstoned: false,
        include_source: false,
    }
}

fn graph_context_defaults() -> GraphContextRequest {
    GraphContextRequest {
        space: None,
        entity_id: None,
        entity_key: None,
        depth: 1,
        relation_types: Vec::new(),
        statuses: Vec::new(),
        max_edges: 50,
        max_memories: 10,
        max_chars: 4_000,
        include_tombstoned: false,
        include_source: false,
    }
}

fn entity_id_for_key(path: &Path, entity_key: &str) -> String {
    search_entities(
        path,
        &EntitySearchRequest {
            entity_key: Some(entity_key.to_string()),
            limit: 10,
            ..entity_search_defaults()
        },
    )
    .expect("entity search succeeds")
    .results
    .into_iter()
    .next()
    .expect("entity exists")
    .entity
    .id
}

fn basic_request(content: &str) -> RememberRequest {
    RememberRequest {
        space: None,
        silo: None,
        scope: None,
        project_key: None,
        kind: Some("fact".to_string()),
        content: content.to_string(),
        summary: None,
        tags: Vec::new(),
        entity_key: None,
        claim_key: None,
        confidence: 1.0,
        observed_at: None,
        valid_from: None,
        valid_to: None,
        expires_at: None,
        source_ref_json: None,
        metadata_json: None,
        source_episode_id: None,
        pinned: false,
        supersedes: Vec::new(),
        contradicts: Vec::new(),
        embedding: None,
        embedding_model_id: None,
        token_embedding: None,
        token_embedding_model_id: None,
        dry_run: false,
        mode: "auto".to_string(),
    }
}

fn temp_store_path(test_name: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "memkeeper-{test_name}-{}-{}.sqlite",
        process::id(),
        unique_nanos()
    ))
}

fn temp_store_dir(test_name: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "memkeeper-{test_name}-{}-{}",
        process::id(),
        unique_nanos()
    ))
}

fn unique_nanos() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("time after epoch")
        .as_nanos()
}

fn import_request_for(path: &Path, dry_run: bool) -> ImportRequest {
    ImportRequest {
        input_path: path.to_path_buf(),
        format: "jsonl".to_string(),
        dry_run,
        conflict_policy: "fail_if_exists".to_string(),
    }
}

fn cleanup_store(path: &Path) {
    let _ = fs::remove_file(path);
    cleanup_store_sidecars(path);
}

fn cleanup_store_sidecars(path: &Path) {
    let _ = fs::remove_file(sidecar_path(path, "-wal"));
    let _ = fs::remove_file(sidecar_path(path, "-shm"));
    let _ = fs::remove_file(sidecar_path(path, "-journal"));
}

#[cfg(unix)]
fn assert_private_file_mode(path: &Path) {
    use std::os::unix::fs::PermissionsExt;
    let mode = fs::metadata(path).expect("metadata").permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
}

#[cfg(not(unix))]
fn assert_private_file_mode(_path: &Path) {}

#[test]
#[cfg(feature = "semantic")]
fn schema_v4_uses_1024_dim_vec_table() {
    use crate::SEMANTIC_TABLE_SQL;
    let store_path = temp_store_path("schema_v4_uses_1024_dim_vec_table");
    cleanup_store(&store_path);
    init_store(&store_path).expect("init store");
    // The vec0 table is created lazily on first use; apply it explicitly here
    // to verify the constant uses the correct table name and dimensions.
    let conn = rusqlite::Connection::open(&store_path).expect("open conn");
    conn.execute_batch(SEMANTIC_TABLE_SQL)
        .expect("create vec table");
    let has_1024: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_vec_1024'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let has_1536: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='memory_vec_1536'",
            [],
            |r| r.get(0),
        )
        .unwrap_or(0);
    let version: i32 = conn
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .expect("user_version");
    assert_eq!(version, 5, "schema version should be 5");
    assert!(has_1024 > 0, "memory_vec_1024 table should exist");
    assert_eq!(has_1536, 0, "memory_vec_1536 table should not exist");
}

#[test]
fn verify_stamps_verified_at_and_preserves_pointer() {
    let path = temp_store_path("verify_stamps");
    cleanup_store(&path);
    init_store(&path).unwrap();
    let mut req = remember_request("the gate is 0.62");
    req.silo = Some("short-term".to_string());
    req.metadata_json = Some(r#"{"verified_against":"~/.zshrc:GATE"}"#.to_string());
    let report = remember_memory(&path, &req).unwrap();

    let verify_req = VerifyRequest {
        memory_id: report.memory.id.clone(),
        verified_against: None,
        now: Some("2026-06-08T20:00:00Z".to_string()),
    };
    let verify_report = verify_memory(&path, &verify_req).unwrap();
    assert_eq!(verify_report.memory_id, report.memory.id);

    let got = get_memory(
        &path,
        &report.memory.id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: false,
        },
    )
    .unwrap();
    let md: serde_json::Value =
        serde_json::from_str(got.metadata_json.as_deref().unwrap()).unwrap();
    assert_eq!(md["verified_at"], "2026-06-08T20:00:00.000Z");
    assert_eq!(md["verified_against"], "~/.zshrc:GATE");
    cleanup_store(&path);
}

#[test]
fn verify_creates_metadata_when_absent_and_sets_verified_against() {
    let path = temp_store_path("verify_no_prior_metadata");
    cleanup_store(&path);
    init_store(&path).unwrap();
    let req = remember_request("the threshold is 0.8");
    let report = remember_memory(&path, &req).unwrap();

    let verify_req = VerifyRequest {
        memory_id: report.memory.id.clone(),
        verified_against: Some("~/.config/app.toml:threshold".to_string()),
        now: Some("2026-06-08T21:00:00Z".to_string()),
    };
    let verify_report = verify_memory(&path, &verify_req).unwrap();
    assert_eq!(verify_report.memory_id, report.memory.id);
    assert_eq!(verify_report.verified_at, "2026-06-08T21:00:00.000Z");

    let got = get_memory(
        &path,
        &report.memory.id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: false,
        },
    )
    .unwrap();
    let md: serde_json::Value =
        serde_json::from_str(got.metadata_json.as_deref().unwrap()).unwrap();
    assert_eq!(md["verified_at"], "2026-06-08T21:00:00.000Z");
    assert_eq!(md["verified_against"], "~/.config/app.toml:threshold");
    cleanup_store(&path);
}

#[test]
fn last_synthesis_run_returns_latest_succeeded() {
    let path = temp_store_path("last_synth_run");
    cleanup_store(&path);
    init_store(&path).unwrap();
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "INSERT INTO dream_runs (id, space_name, status, started_at, finished_at) \
             VALUES ('d1','workspace-memory','succeeded','2026-06-01T03:00:00Z','2026-06-01T03:05:00Z'),\
                    ('d2','workspace-memory','succeeded','2026-06-08T03:00:00Z','2026-06-08T03:04:00Z'),\
                    ('d3','workspace-memory','running','2026-06-09T03:00:00Z',NULL)",
        )
        .unwrap();
    }
    let got = last_synthesis_run(&path).unwrap();
    assert_eq!(got.as_deref(), Some("2026-06-08T03:00:00Z"));
    cleanup_store(&path);
}

#[test]
fn pack_marks_stale_volatile_external_state_memory() {
    let path = temp_store_path("pack_freshness_stale");
    cleanup_store(&path);
    init_store(&path).unwrap();
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "INSERT INTO dream_runs (id, space_name, status, started_at, finished_at) \
             VALUES ('d1','workspace-memory','succeeded','2026-06-08T03:00:00Z','2026-06-08T03:05:00Z')",
        )
        .unwrap();
    }
    let mut req = basic_request("the cosine gate is 0.62");
    req.silo = Some("short-term".to_string());
    req.metadata_json = Some(r#"{"verified_against":"~/.zshrc:GATE"}"#.to_string());
    remember_memory(&path, &req).unwrap();

    let report = build_pack(
        &path,
        &PackRequest {
            title: "cosine gate".to_string(),
            queries: vec!["cosine gate".to_string()],
            filters: SearchFilters::default(),
            max_memories: 10,
            max_chars: 2_000,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .unwrap();
    let content = report.content;
    assert!(
        content.contains("VERIFY vs ~/.zshrc:GATE"),
        "got: {content}"
    );
    cleanup_store(&path);
}

#[test]
fn pack_marks_fresh_volatile_external_state_memory() {
    let path = temp_store_path("pack_freshness_fresh");
    cleanup_store(&path);
    init_store(&path).unwrap();
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "INSERT INTO dream_runs (id, space_name, status, started_at, finished_at) \
             VALUES ('d1','workspace-memory','succeeded','2026-06-08T03:00:00Z','2026-06-08T03:05:00Z')",
        )
        .unwrap();
    }
    let mut req = basic_request("the cosine gate is 0.62");
    req.silo = Some("short-term".to_string());
    req.metadata_json = Some(
        r#"{"verified_against":"~/.zshrc:GATE","verified_at":"2026-06-08T04:00:00Z"}"#.to_string(),
    );
    remember_memory(&path, &req).unwrap();

    let report = build_pack(
        &path,
        &PackRequest {
            title: "cosine gate".to_string(),
            queries: vec!["cosine gate".to_string()],
            filters: SearchFilters::default(),
            max_memories: 10,
            max_chars: 2_000,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .unwrap();
    let content = report.content;
    assert!(
        content.contains("[confirmed 2026-06-08T04:00:00Z vs ~/.zshrc:GATE]"),
        "got: {content}"
    );
    assert!(
        !content.contains("VERIFY"),
        "should not contain VERIFY, got: {content}"
    );
    cleanup_store(&path);
}

#[test]
fn pack_durable_memory_has_no_freshness_marker() {
    let path = temp_store_path("pack_freshness_durable");
    cleanup_store(&path);
    init_store(&path).unwrap();
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch(
            "INSERT INTO dream_runs (id, space_name, status, started_at, finished_at) \
             VALUES ('d1','workspace-memory','succeeded','2026-06-08T03:00:00Z','2026-06-08T03:05:00Z')",
        )
        .unwrap();
    }
    // durable silo (default) with verified_against metadata
    let mut req = basic_request("the cosine gate is 0.62");
    req.metadata_json = Some(r#"{"verified_against":"~/.zshrc:GATE"}"#.to_string());
    remember_memory(&path, &req).unwrap();

    let report = build_pack(
        &path,
        &PackRequest {
            title: "cosine gate".to_string(),
            queries: vec!["cosine gate".to_string()],
            filters: SearchFilters::default(),
            max_memories: 10,
            max_chars: 2_000,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
    )
    .unwrap();
    let content = report.content;
    assert!(
        !content.contains("[VERIFY"),
        "durable should have no VERIFY marker, got: {content}"
    );
    assert!(
        !content.contains("[confirmed"),
        "durable should have no confirmed marker, got: {content}"
    );
    cleanup_store(&path);
}

#[test]
fn volatile_recency_outweighs_durable_recency_for_recent_memory() {
    let now_jd = now_julian_day();
    let recent_jd = Some(now_jd - 1.0);
    let durable = recency_score_for_silo(recent_jd, "durable", now_jd);
    let volatile = recency_score_for_silo(recent_jd, "short-term", now_jd);
    assert!(
        volatile > durable,
        "volatile {volatile} should exceed durable {durable}"
    );
    let old_jd = Some(now_jd - 365.0);
    assert!(recency_score_for_silo(old_jd, "short-term", now_jd) < volatile);
}

#[test]
fn recency_half_life_decays_decisively_at_realistic_timescales() {
    let now_jd = now_julian_day();
    // A volatile claim from yesterday vs one from three months ago must
    // differ by more than any single metadata boost (max 0.10), so
    // recency is decisive between otherwise-equal volatile claims.
    let fresh = recency_score_for_silo(Some(now_jd - 1.0), "short-term", now_jd);
    let stale = recency_score_for_silo(Some(now_jd - 90.0), "short-term", now_jd);
    assert!(
        fresh - stale > 0.10,
        "volatile decay too shallow: fresh {fresh} stale {stale}"
    );
    // Durable decay stays gentle: under half the boost lost in 90 days.
    let durable_fresh = recency_score_for_silo(Some(now_jd - 1.0), "durable", now_jd);
    let durable_aged = recency_score_for_silo(Some(now_jd - 90.0), "durable", now_jd);
    assert!(durable_aged > durable_fresh * 0.5);
    // Future timestamps clamp to the full boost instead of exceeding it.
    let future = recency_score_for_silo(Some(now_jd + 10.0), "short-term", now_jd);
    assert!((future - VOLATILE_MAX_RECENCY_SCORE).abs() < 1e-12);
    // Missing or non-finite timestamps score zero.
    assert!(recency_score_for_silo(None, "short-term", now_jd) == 0.0);
    assert!(recency_score_for_silo(Some(f64::NAN), "durable", now_jd) == 0.0);
}

#[test]
fn timestamps_normalize_to_millisecond_precision_on_write() {
    assert_eq!(
        normalize_utc_timestamp("2026-06-09T12:00:00Z"),
        "2026-06-09T12:00:00.000Z"
    );
    assert_eq!(
        normalize_utc_timestamp("2026-06-09T12:00:00.5Z"),
        "2026-06-09T12:00:00.500Z"
    );
    assert_eq!(
        normalize_utc_timestamp("2026-06-09T12:00:00.123456Z"),
        "2026-06-09T12:00:00.123Z"
    );

    let path = temp_store_path("timestamps_normalize_on_write");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut request = basic_request("normalization probe content");
    request.observed_at = Some("2026-06-01T08:30:00Z".to_string());
    let report = remember_memory(&path, &request).expect("remember succeeds");
    assert_eq!(report.memory.observed_at, "2026-06-01T08:30:00.000Z");
    cleanup_store(&path);
}

#[test]
fn verify_rejects_corrupt_metadata_and_invalid_now() {
    let path = temp_store_path("verify_rejects_corrupt_metadata");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let report =
        remember_memory(&path, &basic_request("verify corruption probe")).expect("remember");
    let memory_id = report.memory.id.clone();

    // Non-UTC / malformed `now` is rejected before touching the store.
    let invalid_now = verify_memory(
        &path,
        &VerifyRequest {
            memory_id: memory_id.clone(),
            verified_against: None,
            now: Some("2026-06-09T12:00:00+02:00".to_string()),
        },
    );
    assert!(matches!(invalid_now, Err(Error::InvalidRequest { .. })));

    // Corrupt stored metadata_json errors instead of being overwritten.
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "UPDATE memories SET metadata_json = '{not json' WHERE id = ?1",
            params![&memory_id],
        )
        .unwrap();
    }
    let corrupt = verify_memory(
        &path,
        &VerifyRequest {
            memory_id,
            verified_against: None,
            now: None,
        },
    );
    assert!(matches!(corrupt, Err(Error::Conflict { .. })));
    cleanup_store(&path);
}

/// End-to-end test: a recent volatile memory must rank above an older durable
/// memory with equivalent text content (equal bm25), proving the widened SQL
/// candidate pool allows the silo-aware recency boost to surface in Rust scoring.
///
/// If the pool were not widened (limit+1), the volatile memory could be pushed
/// below the SQL LIMIT when bm25 is tied and SQL recency is equal, and the 4x
/// Rust boost would never run.
#[test]
fn widened_pool_surfaces_recent_volatile_above_older_durable() {
    let path = temp_store_path("widened_pool_volatile_recency");
    cleanup_store(&path);
    init_store(&path).unwrap();

    // Insert an older durable memory.
    let mut durable_req = basic_request("project configuration alpha setting");
    durable_req.silo = None; // durable (default)
    durable_req.kind = None;
    let durable_report = remember_memory(&path, &durable_req).unwrap();

    // Insert a newer volatile (short-term) memory with identical content.
    let mut volatile_req = basic_request("project configuration alpha setting");
    volatile_req.silo = Some("short-term".to_string());
    volatile_req.kind = None;
    let volatile_report = remember_memory(&path, &volatile_req).unwrap();

    // Backdate the durable memory to ~25 years ago so it has very low recency.
    // The volatile memory stays at "now" (created_at default).
    // observed_at drives recency_jd via the SQL expression.
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute(
            "UPDATE memories SET observed_at = ?1 WHERE id = ?2",
            rusqlite::params!["2001-01-01T00:00:00Z", durable_report.memory.id],
        )
        .unwrap();
        // Give the volatile memory a very recent observed_at so it has high recency.
        conn.execute(
            "UPDATE memories SET observed_at = ?1 WHERE id = ?2",
            rusqlite::params!["2026-06-08T12:00:00Z", volatile_report.memory.id],
        )
        .unwrap();
    }

    // Run a real search with a small limit. The widened pool ensures both
    // candidates are fetched from SQL so Rust scoring can reorder them.
    let report = search_memories(
        &path,
        &SearchRequest {
            query: "project configuration alpha setting".to_string(),
            filters: SearchFilters::default(),
            limit: 5,
            offset: 0,
            snippet_chars: 0,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .unwrap();

    assert!(
        report.results.len() >= 2,
        "expected at least 2 results, got {}",
        report.results.len()
    );

    // The first result must be the volatile (recent) memory.
    assert_eq!(
        report.results[0].memory_id,
        volatile_report.memory.id,
        "volatile (recent) memory should rank first; got silo={:?} at rank 0, durable at rank {}",
        report.results[0].silo,
        report
            .results
            .iter()
            .position(|r| r.memory_id == durable_report.memory.id)
            .unwrap_or(99)
    );

    cleanup_store(&path);
}

#[test]
fn merge_rerank_pools_interleaves_ann_first_dedupes_and_caps() {
    use super::{merge_rerank_pools, PackPoolItem};
    let item = |id: &str, score: f64| PackPoolItem {
        memory_id: id.to_string(),
        score,
    };
    let ann = vec![item("a", 0.9), item("b", 0.8), item("c", 0.7)];
    let bm25 = vec![item("b", 0.5), item("d", 0.4)];

    // Interleaved by rank, ANN first at each rank, dedup by id: a,b | (b),d | c.
    let merged = merge_rerank_pools(&ann, &bm25, 10);
    let ids: Vec<&str> = merged.iter().map(|i| i.memory_id.as_str()).collect();
    assert_eq!(ids, ["a", "b", "d", "c"]);

    // The cap stops the merge mid-interleave.
    let capped = merge_rerank_pools(&ann, &bm25, 2);
    let ids: Vec<&str> = capped.iter().map(|i| i.memory_id.as_str()).collect();
    assert_eq!(ids, ["a", "b"]);

    // Without a BM25 net the ANN order is preserved as-is.
    let ann_only = merge_rerank_pools(&ann, &[], 10);
    let ids: Vec<&str> = ann_only.iter().map(|i| i.memory_id.as_str()).collect();
    assert_eq!(ids, ["a", "b", "c"]);
}

#[test]
fn build_hybrid_rerank_pool_fetches_contents_on_one_snapshot() {
    use super::build_hybrid_rerank_pool;
    let path = temp_store_path("hybrid_rerank_pool_contents");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let first = remember_memory(&path, &basic_request("alpha retrieval probe content"))
        .expect("remember first");
    remember_memory(&path, &basic_request("unrelated beta filler text")).expect("remember second");

    let pool = build_hybrid_rerank_pool(
        &path,
        &PackRequest {
            title: "rerank pool".to_string(),
            queries: vec!["alpha retrieval".to_string()],
            filters: SearchFilters::default(),
            max_memories: 5,
            max_chars: 2_000,
            format: "markdown".to_string(),
            min_score: 0.9, // must be ignored: the pool applies no precision floor
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
        },
        5,
    )
    .expect("pool builds");

    assert!(!pool.candidates.is_empty());
    assert_eq!(pool.candidates[0].memory_id, first.memory.id);
    assert_eq!(pool.candidates[0].content, "alpha retrieval probe content");
    assert!(pool.cos_top > f64::MIN);
    cleanup_store(&path);
}

#[test]
fn hybrid_rerank_pool_late_interaction_splits_summary_from_content() {
    // LI mode must NOT concatenate summary into the rerank/pack text: on
    // prefix-duplicate summaries the concat corrupts cross-encoder ordering
    // (2026-06-12 adversarial diagnosis, -0.066 LoCoMo MRR). The summary is
    // carried as a separate alternate rerank text instead; flag-off candidates
    // carry no summary so legacy behavior stays byte-identical.
    use super::build_hybrid_rerank_pool;
    let path = temp_store_path("hybrid_rerank_pool_li_split");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut with_summary = basic_request("gamma retrieval probe content");
    with_summary.summary = Some("gamma summary line".to_string());
    let first = remember_memory(&path, &with_summary).expect("remember with summary");
    let second = remember_memory(&path, &basic_request("delta retrieval probe content"))
        .expect("remember without summary");

    let model = "colbert-li-split-test";
    {
        let connection = Connection::open(&path).expect("open");
        let vecs: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        upsert_memory_token_embedding(&connection, &first.memory.id, model, &vecs)
            .expect("upsert tokens first");
        upsert_memory_token_embedding(&connection, &second.memory.id, model, &vecs)
            .expect("upsert tokens second");
    }

    let request = |li: bool| PackRequest {
        title: "rerank pool".to_string(),
        queries: vec!["gamma retrieval".to_string()],
        filters: SearchFilters::default(),
        max_memories: 5,
        max_chars: 2_000,
        format: "markdown".to_string(),
        min_score: 0.0,
        rerank_candidates: 0,
        query_embeddings: None,
        query_token_embeddings: li.then(|| vec![vec![vec![1.0, 0.0]]]),
        token_model_id: li.then(|| model.to_string()),
    };

    let pool = build_hybrid_rerank_pool(&path, &request(true), 5).expect("LI pool builds");
    let by_id = |id: &str| {
        pool.candidates
            .iter()
            .find(|c| c.memory_id == id)
            .unwrap_or_else(|| panic!("candidate {id} present"))
    };
    let cand = by_id(&first.memory.id);
    assert_eq!(
        cand.content, "gamma retrieval probe content",
        "LI content must stay content-only (no summary concat)"
    );
    assert_eq!(
        cand.summary.as_deref(),
        Some("gamma summary line"),
        "LI candidate carries the summary as an alternate rerank text"
    );
    assert_eq!(
        by_id(&second.memory.id).summary,
        None,
        "empty summary must not produce an alternate text"
    );

    let legacy = build_hybrid_rerank_pool(&path, &request(false), 5).expect("legacy pool builds");
    assert!(
        legacy.candidates.iter().all(|c| c.summary.is_none()),
        "flag-off candidates must carry no summary"
    );
    cleanup_store(&path);
}

#[test]
fn record_recall_logs_events_and_touches_accessed_at() {
    let path = temp_store_path("record_recall_events");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let report = remember_memory(&path, &basic_request("recall target content")).expect("remember");
    let memory_id = report.memory.id.clone();

    let log = record_recall(
        &path,
        &RecallLogRequest {
            source: Some("unit-test".to_string()),
            session_id: None,
            events: vec![
                RecallEvent {
                    memory_id: memory_id.clone(),
                    kind: "surfaced".to_string(),
                    query: Some("recall target".to_string()),
                    rank: Some(1),
                    score: Some(0.42),
                },
                RecallEvent {
                    memory_id: memory_id.clone(),
                    kind: "retrieved".to_string(),
                    query: None,
                    rank: None,
                    score: None,
                },
            ],
            touch_accessed: true,
        },
    )
    .expect("recall-log succeeds");
    assert_eq!(log.recorded, 2);
    assert_eq!(log.touched, 1);

    let conn = Connection::open(&path).unwrap();
    let events: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM recall_events WHERE memory_id = ?1",
            [&memory_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(events, 2);
    let accessed: Option<String> = conn
        .query_row(
            "SELECT accessed_at FROM memories WHERE id = ?1",
            [&memory_id],
            |row| row.get(0),
        )
        .unwrap();
    assert!(accessed.is_some(), "accessed_at touched for retrieved");

    // Unsupported kinds and empty event lists are rejected.
    let bad_kind = record_recall(
        &path,
        &RecallLogRequest {
            source: None,
            session_id: None,
            events: vec![RecallEvent {
                memory_id,
                kind: "viewed".to_string(),
                query: None,
                rank: None,
                score: None,
            }],
            touch_accessed: false,
        },
    );
    assert!(matches!(bad_kind, Err(Error::InvalidRequest { .. })));
    let empty = record_recall(
        &path,
        &RecallLogRequest {
            source: None,
            session_id: None,
            events: Vec::new(),
            touch_accessed: false,
        },
    );
    assert!(matches!(empty, Err(Error::InvalidRequest { .. })));
    cleanup_store(&path);
}

/// Framing-level malformed JSONL must fail with a clean validation error and
/// never create the import target (row-level invariants are covered by the
/// `import_rejects_*` tests above; this covers the record protocol itself).
#[test]
#[allow(clippy::too_many_lines)]
fn import_rejects_malformed_jsonl_framing() {
    let source = temp_store_path("import_malformed_framing_source");
    cleanup_store(&source);
    init_store(&source).expect("init source");
    remember_memory(&source, &basic_request("framing probe content")).expect("seed");

    let dir = temp_store_dir("import_malformed_framing");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let export_path = dir.join("export.jsonl");
    export_store(
        &source,
        &ExportRequest {
            output_path: export_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export succeeds");
    let original = fs::read_to_string(&export_path).expect("read export");
    let lines: Vec<String> = original.lines().map(str::to_string).collect();
    let header = lines[0].clone();
    let footer = lines.last().expect("footer line").clone();
    let first_row = lines
        .iter()
        .find(|line| line.contains("\"type\":\"row\""))
        .expect("export has at least one row")
        .clone();

    let with = |mutate: &dyn Fn(&mut Vec<String>)| {
        let mut mutated = lines.clone();
        mutate(&mut mutated);
        mutated.join("\n") + "\n"
    };
    let deep_value = format!("{}1{}", "[".repeat(80), "]".repeat(80));
    let cases: Vec<(&str, String)> = vec![
        ("blank line", with(&|v| v.insert(1, String::new()))),
        (
            "garbage line",
            with(&|v| v.insert(1, "not json".to_string())),
        ),
        (
            "non-object record",
            with(&|v| v.insert(1, "[1,2]".to_string())),
        ),
        (
            "unknown record type",
            with(&|v| v.insert(1, r#"{"type":"mystery"}"#.to_string())),
        ),
        ("duplicate header", with(&|v| v.insert(1, header.clone()))),
        (
            "row before header",
            with(&|v| v.insert(0, first_row.clone())),
        ),
        (
            "missing footer",
            with(&|v| {
                v.pop();
            }),
        ),
        ("record after footer", with(&|v| v.push(first_row.clone()))),
        (
            "footer count mismatch",
            with(&|v| {
                let last = v.last_mut().expect("footer");
                *last = footer.replace("\"row_count\":", "\"row_count\":9");
            }),
        ),
        (
            "json depth bomb",
            with(&|v| {
                v.insert(
                    1,
                    format!(r#"{{"type":"row","table":"memories","data":{{"x":{deep_value}}}}}"#),
                );
            }),
        ),
    ];

    for (index, (label, content)) in cases.iter().enumerate() {
        let input = dir.join(format!("case-{index}.jsonl"));
        fs::write(&input, content).expect("write case");
        let target = dir.join(format!("target-{index}.sqlite"));
        let result = import_store(
            &target,
            &ImportRequest {
                input_path: input,
                format: "jsonl".to_string(),
                dry_run: true,
                conflict_policy: "fail_if_exists".to_string(),
            },
        );
        let Err(error) = result else {
            panic!("case '{label}' unexpectedly succeeded")
        };
        assert!(
            matches!(error, Error::InvalidRequest { .. }),
            "case '{label}': unexpected error {error}"
        );
        assert!(
            !target.exists(),
            "case '{label}' must not create the target"
        );
    }

    cleanup_store(&source);
    let _ = fs::remove_dir_all(&dir);
}

/// Build a `retrieved` recall event for a memory id.
fn retrieved_event(memory_id: &str) -> RecallEvent {
    RecallEvent {
        memory_id: memory_id.to_string(),
        kind: "retrieved".to_string(),
        query: None,
        rank: None,
        score: None,
    }
}

fn used_event(memory_id: &str, rank: usize, score: f64) -> RecallEvent {
    RecallEvent {
        memory_id: memory_id.to_string(),
        kind: "retrieved".to_string(),
        query: None,
        rank: Some(rank),
        score: Some(score),
    }
}

fn log_used(path: &std::path::Path, memory_id: &str, session: &str, rank: usize, score: f64) {
    record_recall(
        path,
        &RecallLogRequest {
            source: Some("test".to_string()),
            session_id: Some(session.to_string()),
            events: vec![used_event(memory_id, rank, score)],
            touch_accessed: true,
        },
    )
    .expect("record used");
}

fn promote_request(threshold: usize, floor: f64, cap: usize, dry_run: bool) -> DreamRequest {
    DreamRequest {
        space: Some(DEFAULT_SPACE.to_string()),
        silos: Vec::new(),
        tasks: vec!["promote".to_string()],
        max_memories: 10,
        dry_run,
        include_pinned: false,
        promote_threshold: threshold,
        promote_score_floor: floor,
        promote_rank_cap: cap,
    }
}

fn short_term_memory(path: &std::path::Path, content: &str) -> String {
    let mut req = basic_request(content);
    req.silo = Some("short-term".to_string());
    remember_memory(path, &req).expect("remember").memory.id
}

#[test]
fn promote_counts_distinct_sessions_not_burst() {
    let path = temp_store_path("promote_counts_distinct_sessions_not_burst");
    cleanup_store(&path);
    init_store(&path).expect("init");
    let id = short_term_memory(&path, "decision: burst vs distinct sessions");
    for _ in 0..5 {
        log_used(&path, &id, "sess-A", 1, 0.9); // 5 events, ONE session
    }
    let report = dream_store(&path, &promote_request(3, 0.75, 3, false)).expect("promote");
    assert_eq!(
        report.promote.promoted, 0,
        "one session must not satisfy threshold 3"
    );
}

#[test]
fn promote_fires_across_three_distinct_sessions() {
    let path = temp_store_path("promote_fires_across_three_distinct_sessions");
    cleanup_store(&path);
    init_store(&path).expect("init");
    let id = short_term_memory(&path, "decision: useful across conversations");
    log_used(&path, &id, "sess-A", 1, 0.9);
    log_used(&path, &id, "sess-B", 2, 0.85);
    log_used(&path, &id, "sess-C", 3, 0.8);
    let report = dream_store(&path, &promote_request(3, 0.75, 3, false)).expect("promote");
    assert_eq!(report.promote.promoted, 1);
    assert_eq!(report.promote.memory_ids, vec![id]);
}

#[test]
fn promote_excludes_below_floor_above_cap_and_null_session() {
    let path = temp_store_path("promote_excludes_below_floor_above_cap_and_null_session");
    cleanup_store(&path);
    init_store(&path).expect("init");
    let low_score = short_term_memory(&path, "decision: weak score across sessions");
    let high_rank = short_term_memory(&path, "decision: deep rank across sessions");
    let no_session = short_term_memory(&path, "decision: legacy null session");
    for s in ["s1", "s2", "s3"] {
        log_used(&path, &low_score, s, 1, 0.60); // below 0.75 floor
        log_used(&path, &high_rank, s, 9, 0.95); // above rank cap 3
    }
    for _ in 0..5 {
        record_recall(
            &path,
            &RecallLogRequest {
                source: Some("test".to_string()),
                session_id: None, // legacy/unattributed
                events: vec![used_event(&no_session, 1, 0.95)],
                touch_accessed: true,
            },
        )
        .expect("record");
    }
    let report = dream_store(&path, &promote_request(3, 0.75, 3, false)).expect("promote");
    assert_eq!(report.promote.promoted, 0, "none qualify");
}

#[test]
fn promote_knob_overrides_flip_outcome() {
    let path = temp_store_path("promote_knob_overrides_flip_outcome");
    cleanup_store(&path);
    init_store(&path).expect("init");
    let id = short_term_memory(&path, "decision: tunable promotion");
    log_used(&path, &id, "s1", 5, 0.65);
    log_used(&path, &id, "s2", 5, 0.65);
    assert_eq!(
        dream_store(&path, &promote_request(3, 0.75, 3, false))
            .unwrap()
            .promote
            .promoted,
        0
    );
    assert_eq!(
        dream_store(&path, &promote_request(2, 0.6, 5, false))
            .unwrap()
            .promote
            .promoted,
        1
    );
}

#[test]
fn promote_rejects_invalid_knobs() {
    let path = temp_store_path("promote_rejects_invalid_knobs");
    cleanup_store(&path);
    init_store(&path).expect("init");
    assert!(
        dream_store(&path, &promote_request(3, 0.75, 0, false)).is_err(),
        "rank cap 0 rejected"
    );
    assert!(
        dream_store(&path, &promote_request(3, -1.0, 3, false)).is_err(),
        "negative floor rejected"
    );
}

#[test]
fn record_recall_stores_session_id_per_event() {
    let path = temp_store_path("record_recall_stores_session_id_per_event");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let m =
        remember_memory(&path, &basic_request("fact: session-tagged recall")).expect("remember");
    record_recall(
        &path,
        &RecallLogRequest {
            source: Some("test".to_string()),
            session_id: Some("sess-123".to_string()),
            events: vec![retrieved_event(&m.memory.id)],
            touch_accessed: true,
        },
    )
    .expect("record recall");

    let conn = Connection::open(&path).expect("open");
    let session: Option<String> = conn
        .query_row(
            "SELECT session_id FROM recall_events WHERE memory_id = ?1",
            params![&m.memory.id],
            |row| row.get(0),
        )
        .expect("query session_id");
    assert_eq!(session.as_deref(), Some("sess-123"));
}

#[test]
fn dream_promote_graduates_short_term_after_threshold_recalls() {
    let path = temp_store_path("dream_promote_graduates_short_term_after_threshold_recalls");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // A short-term memory used across 3 distinct sessions (meets default threshold).
    let hot_id = short_term_memory(&path, "decision: hot short-term memory worth keeping");

    // A short-term memory used in only 2 distinct sessions (below threshold).
    let cool_id = short_term_memory(&path, "decision: cool short-term memory recalled rarely");

    log_used(&path, &hot_id, "sess-A", 1, 0.9);
    log_used(&path, &hot_id, "sess-B", 2, 0.85);
    log_used(&path, &hot_id, "sess-C", 3, 0.8);
    log_used(&path, &cool_id, "sess-A", 1, 0.9);
    log_used(&path, &cool_id, "sess-B", 2, 0.85);

    let report = dream_store(
        &path,
        &promote_request(DEFAULT_PROMOTE_THRESHOLD, 0.75, 3, false),
    )
    .expect("promote dream succeeds");

    assert!(report.promote.attempted);
    assert_eq!(report.promote.promoted, 1);
    assert_eq!(report.promote.memory_ids, vec![hot_id.clone()]);

    let opts = GetOptions {
        include_history: false,
        include_links: false,
        include_source: false,
    };
    assert_eq!(
        get_memory(&path, &hot_id, opts).expect("hot memory").silo,
        "durable"
    );
    assert_eq!(
        get_memory(&path, &cool_id, opts).expect("cool memory").silo,
        "short-term"
    );
}

#[test]
fn dream_promote_honors_silo_scope() {
    let path = temp_store_path("dream_promote_honors_silo_scope");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let hot_id = short_term_memory(&path, "decision: short-term reinforced under silo scope");

    log_used(&path, &hot_id, "sess-A", 1, 0.9);
    log_used(&path, &hot_id, "sess-B", 2, 0.85);
    log_used(&path, &hot_id, "sess-C", 3, 0.8);

    let opts = GetOptions {
        include_history: false,
        include_links: false,
        include_source: false,
    };

    // Scoped to durable: short-term is out of scope -> no promotion.
    let mut durable_scope = promote_request(3, 0.75, 3, false);
    durable_scope.silos = vec!["durable".to_string()];
    let scoped_out = dream_store(&path, &durable_scope).expect("durable-scoped promote");
    assert_eq!(scoped_out.promote.promoted, 0);
    assert_eq!(
        get_memory(&path, &hot_id, opts)
            .expect("still short-term")
            .silo,
        "short-term"
    );

    // Scoped to short-term: in scope -> promoted.
    let mut short_scope = durable_scope.clone();
    short_scope.silos = vec!["short-term".to_string()];
    let scoped_in = dream_store(&path, &short_scope).expect("short-term-scoped promote");
    assert_eq!(scoped_in.promote.promoted, 1);
    assert_eq!(
        get_memory(&path, &hot_id, opts).expect("now durable").silo,
        "durable"
    );
}

#[test]
fn dream_promote_clears_ttl_is_dry_runnable_and_idempotent() {
    let path = temp_store_path("dream_promote_clears_ttl_is_dry_runnable_and_idempotent");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // Short-term memory with a future TTL, used across 3 distinct sessions.
    let mut hot = basic_request("decision: short-term with a ttl recalled often");
    hot.silo = Some("short-term".to_string());
    hot.expires_at = Some("2999-01-01T00:00:00Z".to_string());
    let hot_id = remember_memory(&path, &hot)
        .expect("remember hot")
        .memory
        .id;

    log_used(&path, &hot_id, "sess-A", 1, 0.9);
    log_used(&path, &hot_id, "sess-B", 2, 0.85);
    log_used(&path, &hot_id, "sess-C", 3, 0.8);

    let base = promote_request(3, 0.75, 3, true);

    // Dry run reports the candidate but mutates nothing.
    let dry = dream_store(&path, &base).expect("dry-run promote");
    assert_eq!(dry.promote.promoted, 1);
    assert!(dry.dry_run);
    assert!(!dry.journaled);
    let opts = GetOptions {
        include_history: false,
        include_links: false,
        include_source: false,
    };
    assert_eq!(
        get_memory(&path, &hot_id, opts)
            .expect("still short-term after dry run")
            .silo,
        "short-term"
    );

    // Commit run promotes and clears the TTL.
    let mut commit = base.clone();
    commit.dry_run = false;
    let committed = dream_store(&path, &commit).expect("commit promote");
    assert_eq!(committed.promote.promoted, 1);
    let promoted = get_memory(&path, &hot_id, opts).expect("promoted memory");
    assert_eq!(promoted.silo, "durable");
    assert_eq!(promoted.expires_at, None);

    // Second commit run is a no-op: the memory is durable now, not short-term.
    let again = dream_store(&path, &commit).expect("second promote run");
    assert_eq!(again.promote.scanned, 0);
    assert_eq!(again.promote.promoted, 0);
}

#[test]
fn dream_promote_respects_threshold_and_empty_recall_history() {
    let path = temp_store_path("dream_promote_respects_threshold_and_empty_recall_history");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // Short-term memory with zero recalls.
    let never_id = short_term_memory(&path, "decision: short-term never recalled");

    // Short-term memory used across two distinct sessions.
    let twice_id = short_term_memory(&path, "decision: short-term recalled twice");
    log_used(&path, &twice_id, "sess-A", 1, 0.9);
    log_used(&path, &twice_id, "sess-B", 2, 0.85);

    let mut request = promote_request(3, 0.75, 3, false);

    // Threshold 3: nothing qualifies (never=0, twice=2).
    let none = dream_store(&path, &request).expect("promote threshold 3");
    assert_eq!(none.promote.promoted, 0);
    assert!(none.promote.attempted);

    // Lower threshold to 2: the twice-recalled memory qualifies, the never one does not.
    request.promote_threshold = 2;
    let some = dream_store(&path, &request).expect("promote threshold 2");
    assert_eq!(some.promote.promoted, 1);
    assert_eq!(some.promote.memory_ids, vec![twice_id.clone()]);

    let opts = GetOptions {
        include_history: false,
        include_links: false,
        include_source: false,
    };
    assert_eq!(
        get_memory(&path, &never_id, opts)
            .expect("never memory")
            .silo,
        "short-term"
    );
}

#[test]
fn dream_promote_runs_before_expire_and_promotes_pinned() {
    let path = temp_store_path("dream_promote_runs_before_expire_and_promotes_pinned");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // Pinned short-term memory whose TTL has already lapsed, but used across 3 sessions.
    // Promote runs first, clears the TTL, so expire (same run) must NOT reap it.
    let mut hot = basic_request("decision: pinned short-term lapsed ttl but reinforced");
    hot.silo = Some("short-term".to_string());
    hot.pinned = true;
    hot.expires_at = Some("2000-01-01T00:00:00Z".to_string());
    let hot_id = remember_memory(&path, &hot)
        .expect("remember hot")
        .memory
        .id;

    log_used(&path, &hot_id, "sess-A", 1, 0.9);
    log_used(&path, &hot_id, "sess-B", 2, 0.85);
    log_used(&path, &hot_id, "sess-C", 3, 0.8);

    // Run promote AND expire together; include_pinned=true so expire would reap
    // a lapsed pinned memory if promote had not already rescued it.
    let mut request = promote_request(3, 0.75, 3, false);
    request.tasks = vec!["promote".to_string(), "expire".to_string()];
    request.include_pinned = true;
    let report = dream_store(&path, &request).expect("promote+expire run");

    // Promoted (pinned does not block promotion), and NOT expired.
    assert_eq!(report.promote.promoted, 1);
    assert_eq!(report.expire.expired, 0);

    let opts = GetOptions {
        include_history: false,
        include_links: false,
        include_source: false,
    };
    let memory = get_memory(&path, &hot_id, opts).expect("rescued memory");
    assert_eq!(memory.silo, "durable");
    assert_eq!(memory.status, "active");
    assert_eq!(memory.expires_at, None);
}

#[test]
fn dream_promote_ignores_surfaced_recall_events() {
    let path = temp_store_path("dream_promote_ignores_surfaced_recall_events");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mem_id = short_term_memory(&path, "decision: short-term mostly surfaced not retrieved");

    // Qualifying retrieved event in ONE session; 4 surfaced events (kind="surfaced",
    // in distinct sessions) that must be excluded from the distinct-session count.
    let surfaced = |session: &str| RecallLogRequest {
        source: Some("test".to_string()),
        session_id: Some(session.to_string()),
        events: vec![RecallEvent {
            memory_id: mem_id.clone(),
            kind: "surfaced".to_string(),
            query: None,
            rank: Some(1),
            score: Some(0.9),
        }],
        touch_accessed: true,
    };
    log_used(&path, &mem_id, "sess-A", 1, 0.9);
    for s in ["surf-1", "surf-2", "surf-3", "surf-4"] {
        record_recall(&path, &surfaced(s)).expect("record surfaced");
    }

    // Threshold 2: only 1 distinct retrieved session, surfaced must not count -> no promotion.
    let result = dream_store(&path, &promote_request(2, 0.75, 3, false)).expect("promote run");
    assert_eq!(result.promote.promoted, 0);

    let opts = GetOptions {
        include_history: false,
        include_links: false,
        include_source: false,
    };
    assert_eq!(
        get_memory(&path, &mem_id, opts).expect("memory").silo,
        "short-term"
    );
}

#[test]
fn dream_promote_threshold_zero_is_rejected() {
    let path = temp_store_path("dream_promote_threshold_zero_is_rejected");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    assert!(dream_store(&path, &promote_request(0, 0.75, 3, false)).is_err());
}

#[test]
fn schema_v5_has_token_table() {
    let path = temp_store_path("token-schema-v5");
    init_store(&path).expect("init");
    let connection = Connection::open(&path).expect("open");
    let n: i64 = connection
        .query_row("SELECT count(*) FROM memory_token_embeddings", [], |r| {
            r.get(0)
        })
        .expect("token table exists");
    assert_eq!(n, 0);
}

#[test]
fn token_embedding_roundtrip_and_active_filter() {
    let path = temp_store_path("token-roundtrip");
    init_store(&path).expect("init");
    let report = remember_memory(
        &path,
        &basic_request("fact: token embedding roundtrip subject"),
    )
    .expect("remember");
    let memory_id = report.memory.id.clone();
    let connection = Connection::open(&path).expect("open");
    let vecs: Vec<Vec<f32>> = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
    upsert_memory_token_embedding(&connection, &memory_id, "colbert-test", &vecs).expect("upsert");
    let rows = load_token_embeddings(&connection, "colbert-test").expect("load");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].0, memory_id);
    assert_eq!(rows[0].1.len(), 2);
    assert!((rows[0].1[1][2] - 0.6).abs() < 1e-6);
    upsert_memory_token_embedding(&connection, "mem_ghost", "colbert-test", &vecs)
        .expect("upsert ghost");
    let rows = load_token_embeddings(&connection, "colbert-test").expect("load 2");
    assert_eq!(rows.len(), 1, "ghost (no memories row) must be filtered");
}

#[test]
fn maxsim_scores_rank_correctly() {
    // query: 2 tokens in 2-d; doc_a aligned with both, doc_b anti-aligned.
    let query: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let doc_a: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.7, 0.7]];
    let doc_b: Vec<Vec<f32>> = vec![vec![-1.0, 0.0]];
    let a = maxsim_score(&query, &doc_a);
    let b = maxsim_score(&query, &doc_b);
    assert!((a - 2.0).abs() < 1e-6, "doc_a = 1.0 + 1.0, got {a}");
    assert!(a > b);
}

#[test]
fn token_cache_invalidates_on_write() {
    let path = temp_store_path("token-cache");
    init_store(&path).expect("init");
    let report_a = remember_memory(&path, &basic_request("fact: token cache memory alpha"))
        .expect("remember a");
    let report_b = remember_memory(&path, &basic_request("fact: token cache memory beta"))
        .expect("remember b");
    let connection = Connection::open(&path).expect("open");
    // Unique model id: the cache is process-global across tests.
    let model = "colbert-cache-test";
    let vecs: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    upsert_memory_token_embedding(&connection, &report_a.memory.id, model, &vecs)
        .expect("upsert a");
    assert_eq!(
        load_token_embeddings_cached(&connection, model)
            .expect("load 1")
            .len(),
        1
    );
    assert_eq!(
        load_token_embeddings_cached(&connection, model)
            .expect("load 2 (hit)")
            .len(),
        1
    );
    upsert_memory_token_embedding(&connection, &report_b.memory.id, model, &vecs)
        .expect("upsert b");
    assert_eq!(
        load_token_embeddings_cached(&connection, model)
            .expect("load 3 (invalidated)")
            .len(),
        2,
        "cache must invalidate when the row count changes"
    );
}

fn ingest_request(chunks: &[&str]) -> IngestRequest {
    IngestRequest {
        space: None,
        source_type: None,
        source_path: Some("notes/example.md".to_string()),
        source_uri: None,
        source_description: Some("Example note".to_string()),
        metadata_json: None,
        chunks: chunks.iter().map(|c| (*c).to_string()).collect(),
        embeddings: None,
        embedding_model_id: None,
        dry_run: false,
    }
}

#[test]
fn ingest_source_writes_chunks_into_documents_space() {
    let path = temp_store_path("ingest_source_writes_chunks");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let report = ingest_source(&path, &ingest_request(&["first chunk", "second chunk"]))
        .expect("ingest succeeds");

    assert_eq!(report.space, DOCUMENTS_SPACE);
    assert_eq!(report.chunk_count, 2);
    assert_eq!(report.created.len(), 2);
    assert_eq!(report.skipped, 0);
    assert!(
        report.created_space,
        "first ingest seeds the documents space"
    );
    assert!(!report.dry_run);

    cleanup_store(&path);
}

#[test]
fn ingest_source_dedupes_identical_content() {
    let path = temp_store_path("ingest_source_dedupes");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    ingest_source(&path, &ingest_request(&["alpha", "beta"])).expect("first ingest");
    let second =
        ingest_source(&path, &ingest_request(&["alpha", "beta", "gamma"])).expect("second ingest");

    assert_eq!(second.created.len(), 1, "only the new chunk is written");
    assert_eq!(second.skipped, 2, "identical chunks are skipped");
    assert!(!second.created_space, "the documents space already existed");

    cleanup_store(&path);
}

#[test]
fn ingest_source_keeps_independent_duplicates_across_paths() {
    let path = temp_store_path("ingest_source_independent_dupes");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    ingest_source(&path, &ingest_request(&["shared chunk"])).expect("first ingest");

    // Identical content under a *different* source path is an independent chunk,
    // not a dedup collision (to be surfaced as a duplicate later, not silently
    // dropped).
    let mut other = ingest_request(&["shared chunk"]);
    other.source_path = Some("notes/other.md".to_string());
    let second = ingest_source(&path, &other).expect("second ingest");

    assert_eq!(
        second.created.len(),
        1,
        "different path is kept, not skipped"
    );
    assert_eq!(second.skipped, 0);

    let connection = Connection::open(&path).expect("open store");
    let rows: i64 = connection
        .query_row("SELECT COUNT(*) FROM source_episodes", [], |row| row.get(0))
        .expect("count rows");
    assert_eq!(rows, 2, "both paths persist as independent chunks");

    cleanup_store(&path);
}

#[test]
fn ingest_source_repairs_provenance_on_resync() {
    let path = temp_store_path("ingest_source_resync_provenance");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    ingest_source(&path, &ingest_request(&["stable chunk"])).expect("first ingest");

    // Re-sync the same content at the same path with changed provenance/metadata
    // (e.g. a URI added, description/metadata updated). The chunk must not be
    // duplicated, but its mutable provenance must be repaired in place so
    // citations stay accurate.
    let mut resync = ingest_request(&["stable chunk"]);
    resync.source_uri = Some("https://example.com/stable".to_string());
    resync.source_description = Some("Updated description".to_string());
    resync.metadata_json = Some(r#"{"rev":2}"#.to_string());
    let report = ingest_source(&path, &resync).expect("re-sync ingest");

    assert_eq!(report.created.len(), 0, "no new row on same-path re-sync");
    assert_eq!(report.skipped, 1);

    let connection = Connection::open(&path).expect("open store");
    let (uri, description, metadata): (Option<String>, Option<String>, Option<String>) = connection
        .query_row(
            "SELECT source_uri, source_description, metadata_json FROM source_episodes LIMIT 1",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("read repaired row");
    assert_eq!(uri.as_deref(), Some("https://example.com/stable"));
    assert_eq!(description.as_deref(), Some("Updated description"));
    assert_eq!(metadata.as_deref(), Some(r#"{"rev":2}"#));

    // FTS metadata is repaired too (searchable by the new description term).
    let fts_hits: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM source_episode_fts WHERE source_episode_fts MATCH 'Updated'",
            [],
            |row| row.get(0),
        )
        .expect("query fts");
    assert_eq!(fts_hits, 1, "fts row reflects the repaired description");

    cleanup_store(&path);
}

#[test]
fn ingest_source_rejects_malformed_metadata_json() {
    let path = temp_store_path("ingest_source_bad_metadata");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // A non-object (array) and non-JSON garbage are both rejected, matching the
    // import invariant so ingest can never create a store that export/import
    // would later refuse.
    let mut array_meta = ingest_request(&["chunk"]);
    array_meta.metadata_json = Some("[1,2,3]".to_string());
    assert!(matches!(
        ingest_source(&path, &array_meta),
        Err(Error::InvalidRequest { .. })
    ));

    let mut garbage_meta = ingest_request(&["chunk"]);
    garbage_meta.metadata_json = Some("not json".to_string());
    assert!(matches!(
        ingest_source(&path, &garbage_meta),
        Err(Error::InvalidRequest { .. })
    ));

    cleanup_store(&path);
}

#[test]
fn document_duplicates_surfaces_cross_path_clusters() {
    let path = temp_store_path("document_duplicates");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // "shared" appears under two different paths (an independent-duplicate
    // cluster); "unique" appears once and must not surface.
    ingest_source(&path, &ingest_request(&["shared body", "unique body"])).expect("first ingest");
    let mut other = ingest_request(&["shared body"]);
    other.source_path = Some("notes/other.md".to_string());
    ingest_source(&path, &other).expect("second ingest");

    let report = document_duplicates(
        &path,
        &DocumentDuplicatesRequest {
            space: None,
            limit: 0,
            snippet_chars: 80,
        },
    )
    .expect("duplicates scan succeeds");

    assert_eq!(report.clusters.len(), 1, "only the shared content clusters");
    let cluster = &report.clusters[0];
    assert_eq!(cluster.member_count, 2);
    assert_eq!(cluster.members.len(), 2);
    assert_eq!(cluster.snippet, "shared body");
    let paths: Vec<Option<&str>> = cluster
        .members
        .iter()
        .map(|m| m.source_path.as_deref())
        .collect();
    assert!(paths.contains(&Some("notes/example.md")));
    assert!(paths.contains(&Some("notes/other.md")));

    cleanup_store(&path);
}

#[test]
fn document_duplicates_empty_when_all_unique() {
    let path = temp_store_path("document_duplicates_empty");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    ingest_source(&path, &ingest_request(&["alpha", "beta", "gamma"])).expect("ingest");

    let report = document_duplicates(&path, &DocumentDuplicatesRequest::default())
        .expect("duplicates scan succeeds");
    assert!(
        report.clusters.is_empty(),
        "distinct content has no clusters"
    );

    cleanup_store(&path);
}

#[test]
fn prune_documents_removes_chosen_chunk_and_derived_rows() {
    let path = temp_store_path("prune_documents");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // One duplicate cluster: identical content under two paths.
    ingest_source(&path, &ingest_request(&["dup body"])).expect("first ingest");
    let mut other = ingest_request(&["dup body"]);
    other.source_path = Some("notes/other.md".to_string());
    ingest_source(&path, &other).expect("second ingest");

    let before = document_duplicates(&path, &DocumentDuplicatesRequest::default())
        .expect("scan before prune");
    assert_eq!(before.clusters.len(), 1);
    let victim = before.clusters[0].members[0].source_episode_id.clone();

    // Dry-run reports the deletion but persists nothing.
    let preview = prune_documents(
        &path,
        &DocumentPruneRequest {
            space: None,
            source_episode_ids: vec![victim.clone()],
            dry_run: true,
        },
    )
    .expect("dry-run prune succeeds");
    assert!(preview.dry_run);
    assert_eq!(preview.deleted, vec![victim.clone()]);

    let connection = Connection::open(&path).expect("open store");
    let still_there: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM source_episodes WHERE id = ?1",
            [&victim],
            |row| row.get(0),
        )
        .expect("count after dry-run");
    assert_eq!(still_there, 1, "dry-run must not delete");

    // Real prune removes the chunk and its FTS + embedding rows.
    let report = prune_documents(
        &path,
        &DocumentPruneRequest {
            space: None,
            source_episode_ids: vec![victim.clone(), "src_does_not_exist".to_string()],
            dry_run: false,
        },
    )
    .expect("prune succeeds");
    assert_eq!(report.requested, 2);
    assert_eq!(
        report.deleted,
        vec![victim.clone()],
        "only the real id deleted"
    );

    let rows: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM source_episodes WHERE id = ?1",
            [&victim],
            |row| row.get(0),
        )
        .expect("count after prune");
    assert_eq!(rows, 0, "chunk row removed");
    let fts: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM source_episode_fts WHERE source_episode_id = ?1",
            [&victim],
            |row| row.get(0),
        )
        .expect("count fts after prune");
    assert_eq!(fts, 0, "fts row removed");

    // The cluster is gone now that only one copy remains.
    let after = document_duplicates(&path, &DocumentDuplicatesRequest::default())
        .expect("scan after prune");
    assert!(after.clusters.is_empty(), "no duplicates remain");

    cleanup_store(&path);
}

#[test]
fn prune_documents_rejects_empty_ids() {
    let path = temp_store_path("prune_documents_empty");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    assert!(matches!(
        prune_documents(&path, &DocumentPruneRequest::default()),
        Err(Error::InvalidRequest { .. })
    ));

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn prune_documents_clears_vector_index_rows() {
    // Regression: a chunk ingested *with* an embedding has a vec0 ANN row (plus
    // vec0 shadow tables). Prune must delete the real vec row without tripping
    // over the shadow tables (which carry no source_episode_id column).
    let path = temp_store_path("prune_documents_vectors");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let dims = crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS;
    let mut request = ingest_request(&["vector body"]);
    request.embeddings = Some(vec![vec![0.1_f32; dims]]);
    request.embedding_model_id = Some("test-model".to_string());
    let report = ingest_source(&path, &request).expect("ingest with embedding");
    let id = report.created[0].clone();

    let connection = Connection::open(&path).expect("open store");
    let table = format!("source_episode_vec_{dims}");
    let before: i64 = connection
        .query_row(
            &format!("SELECT COUNT(*) FROM {table} WHERE source_episode_id = ?1"),
            [&id],
            |row| row.get(0),
        )
        .expect("count vec before");
    assert_eq!(before, 1, "embedding produced a vec index row");

    prune_documents(
        &path,
        &DocumentPruneRequest {
            space: None,
            source_episode_ids: vec![id.clone()],
            dry_run: false,
        },
    )
    .expect("prune with vectors succeeds");

    let vec_after: i64 = connection
        .query_row(
            &format!("SELECT COUNT(*) FROM {table} WHERE source_episode_id = ?1"),
            [&id],
            |row| row.get(0),
        )
        .expect("count vec after");
    assert_eq!(vec_after, 0, "vec index row removed");
    let emb_after: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM embeddings WHERE source_episode_id = ?1",
            [&id],
            |row| row.get(0),
        )
        .expect("count embeddings after");
    assert_eq!(emb_after, 0, "canonical embedding removed");

    cleanup_store(&path);
}

#[test]
fn ingest_source_dry_run_persists_nothing() {
    let path = temp_store_path("ingest_source_dry_run");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = ingest_request(&["preview only"]);
    request.dry_run = true;
    let preview = ingest_source(&path, &request).expect("dry-run ingest succeeds");
    assert!(preview.dry_run);
    assert!(!preview.created_space);

    // A real ingest of the same content must still create it (proving nothing
    // persisted on the dry run).
    let real = ingest_source(&path, &ingest_request(&["preview only"])).expect("real ingest");
    assert_eq!(real.created.len(), 1);
    assert_eq!(real.skipped, 0);

    cleanup_store(&path);
}

#[test]
fn ingest_source_rejects_empty_request() {
    let path = temp_store_path("ingest_source_rejects_empty");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let empty = IngestRequest {
        chunks: Vec::new(),
        ..IngestRequest::default()
    };
    assert!(matches!(
        ingest_source(&path, &empty),
        Err(Error::InvalidRequest { .. })
    ));

    let blank = ingest_request(&["   "]);
    assert!(matches!(
        ingest_source(&path, &blank),
        Err(Error::InvalidRequest { .. })
    ));

    cleanup_store(&path);
}

#[test]
fn ingest_source_validates_embeddings_shape() {
    let path = temp_store_path("ingest_source_validates_embeddings");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // Ragged: fewer embeddings than chunks.
    let mut ragged = ingest_request(&["alpha", "beta"]);
    ragged.embeddings = Some(vec![vec![0.1_f32, 0.2]]);
    ragged.embedding_model_id = Some("test-model".to_string());
    assert!(matches!(
        ingest_source(&path, &ragged),
        Err(Error::InvalidRequest { .. })
    ));

    // Embeddings supplied without a model id.
    let mut no_model = ingest_request(&["alpha"]);
    no_model.embeddings = Some(vec![vec![0.1_f32, 0.2]]);
    no_model.embedding_model_id = None;
    assert!(matches!(
        ingest_source(&path, &no_model),
        Err(Error::InvalidRequest { .. })
    ));

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn ingest_source_writes_chunk_embeddings_isolated_from_memories() {
    let path = temp_store_path("ingest_source_chunk_embeddings");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let dims = crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS;
    let mut request = ingest_request(&["alpha chunk", "beta chunk"]);
    request.embeddings = Some(vec![vec![0.1_f32; dims], vec![0.2_f32; dims]]);
    request.embedding_model_id = Some("test-model".to_string());

    let report = ingest_source(&path, &request).expect("ingest with embeddings");
    assert_eq!(report.created.len(), 2);

    let connection = Connection::open(&path).expect("open store");
    let chunk_embeddings: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM embeddings WHERE source_episode_id IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .expect("count chunk embeddings");
    assert_eq!(chunk_embeddings, 2, "one embedding row per created chunk");

    // No memory embeddings written: chunk vectors stay isolated from the
    // curated tier.
    let memory_embeddings: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM embeddings WHERE memory_id IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .expect("count memory embeddings");
    assert_eq!(memory_embeddings, 0);

    // Re-ingesting identical content skips dedup'd chunks and writes no new
    // embeddings.
    let again = ingest_source(&path, &request).expect("re-ingest");
    assert_eq!(again.created.len(), 0);
    assert_eq!(again.skipped, 2);
    let after: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM embeddings WHERE source_episode_id IS NOT NULL",
            [],
            |row| row.get(0),
        )
        .expect("recount chunk embeddings");
    assert_eq!(after, 2, "dedup'd chunks add no embeddings");

    cleanup_store(&path);
}

#[test]
fn search_documents_lexical_finds_matching_chunk() {
    let path = temp_store_path("search_documents_lexical");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    ingest_source(
        &path,
        &ingest_request(&[
            "The capital of France is Paris.",
            "Rust prevents data races.",
        ]),
    )
    .expect("ingest");

    let report = search_documents(
        &path,
        &DocumentSearchRequest {
            query: "Paris".to_string(),
            space: None,
            limit: 5,
            include_content: false,
            snippet_chars: 80,
            embedding: None,
            skip_recall_log: true,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.strategy, "lexical_only_v0");
    assert!(!report.semantic_attempted);
    assert_eq!(report.results.len(), 1);
    assert_eq!(report.results[0].chunk_index, 0);
    assert_eq!(report.results[0].match_type, "lexical");
    assert_eq!(
        report.results[0].source_path.as_deref(),
        Some("notes/example.md")
    );
    assert_eq!(report.results[0].chunk_count, 2);

    cleanup_store(&path);
}

#[test]
fn search_documents_only_returns_requested_space() {
    let path = temp_store_path("search_documents_space_scope");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    ingest_source(&path, &ingest_request(&["Paris is in France."])).expect("ingest");

    // A different space holds the same term but must not leak into the default
    // documents-space search.
    let mut other = ingest_request(&["Paris is also a city."]);
    other.space = Some("other-docs".to_string());
    ingest_source(&path, &other).expect("ingest other space");

    let report = search_documents(
        &path,
        &DocumentSearchRequest {
            query: "Paris".to_string(),
            space: None,
            limit: 10,
            include_content: false,
            snippet_chars: 80,
            embedding: None,
            skip_recall_log: true,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.space, DOCUMENTS_SPACE);
    assert_eq!(report.results.len(), 1, "only the documents-space chunk");
    assert_eq!(report.results[0].space, DOCUMENTS_SPACE);

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn search_documents_semantic_ranks_nearest_chunk() {
    let path = temp_store_path("search_documents_semantic");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut request = ingest_request(&["alpha content", "beta content"]);
    request.embeddings = Some(vec![
        vec![1.0_f32, 0.0, 0.0, 0.0],
        vec![0.0_f32, 1.0, 0.0, 0.0],
    ]);
    request.embedding_model_id = Some("test-model".to_string());
    ingest_source(&path, &request).expect("ingest with embeddings");

    let report = search_documents(
        &path,
        &DocumentSearchRequest {
            query: String::new(),
            space: None,
            limit: 5,
            include_content: false,
            snippet_chars: 40,
            embedding: Some(vec![0.9_f32, 0.1, 0.0, 0.0]),
            skip_recall_log: true,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.strategy, "hybrid_rrf_v0");
    assert!(report.semantic_attempted);
    assert!(!report.results.is_empty());
    assert_eq!(
        report.results[0].chunk_index, 0,
        "nearest chunk ranks first"
    );
    assert_eq!(report.results[0].match_type, "semantic");

    cleanup_store(&path);
}

#[test]
fn search_documents_records_retrieval_events() {
    let path = temp_store_path("search_documents_records_retrievals");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    ingest_source(
        &path,
        &ingest_request(&[
            "The capital of France is Paris.",
            "Rust prevents data races.",
        ]),
    )
    .expect("ingest");

    let request = DocumentSearchRequest {
        query: "Paris".to_string(),
        space: None,
        limit: 5,
        include_content: false,
        snippet_chars: 80,
        embedding: None,
        skip_recall_log: false,
    };
    let report = search_documents(&path, &request).expect("search succeeds");
    assert_eq!(report.results.len(), 1);
    let hit_id = report.results[0].source_episode_id.clone();

    let connection = Connection::open(&path).expect("open store");
    let (event_count, recorded_id, recorded_space, recorded_match): (i64, String, String, String) =
        connection
            .query_row(
                "SELECT COUNT(*), source_episode_id, space_name, match_type
                 FROM source_episode_recall_events",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .expect("recall event recorded");
    assert_eq!(event_count, 1, "one event per returned chunk");
    assert_eq!(recorded_id, hit_id);
    assert_eq!(recorded_space, DOCUMENTS_SPACE);
    assert_eq!(recorded_match, "lexical");

    // skip_recall_log suppresses instrumentation.
    let mut quiet = request;
    quiet.skip_recall_log = true;
    search_documents(&path, &quiet).expect("second search succeeds");
    let after: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM source_episode_recall_events",
            [],
            |row| row.get(0),
        )
        .expect("count events");
    assert_eq!(after, 1, "skip_recall_log adds no events");

    cleanup_store(&path);
}

#[test]
fn get_document_by_path_returns_chunks_in_order() {
    let path = temp_store_path("get_document_by_path");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    ingest_source(
        &path,
        &ingest_request(&["first chunk", "second chunk", "third chunk"]),
    )
    .expect("ingest");

    let report = get_document(
        &path,
        &DocumentGetRequest {
            source_path: Some("notes/example.md".to_string()),
            source_episode_id: None,
            space: None,
            include_content: true,
            limit: 0,
        },
    )
    .expect("get_document succeeds");

    assert_eq!(report.space, DOCUMENTS_SPACE);
    assert_eq!(report.chunks.len(), 3);
    assert_eq!(report.chunks[0].chunk_index, 0);
    assert_eq!(report.chunks[2].chunk_index, 2);
    assert_eq!(report.chunks[0].content.as_deref(), Some("first chunk"));
    assert_eq!(report.chunks[0].chunk_count, 3);
    assert_eq!(report.chunks[0].ingest_status, "indexed");
}

#[test]
fn get_document_by_id_returns_single_chunk() {
    let path = temp_store_path("get_document_by_id");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let report = ingest_source(&path, &ingest_request(&["only chunk"])).expect("ingest");
    let id = report.created[0].clone();

    let got = get_document(
        &path,
        &DocumentGetRequest {
            source_path: None,
            source_episode_id: Some(id.clone()),
            space: None,
            include_content: true,
            limit: 0,
        },
    )
    .expect("get_document succeeds");

    assert_eq!(got.chunks.len(), 1);
    assert_eq!(got.chunks[0].source_episode_id, id);
}

#[test]
fn get_document_requires_a_selector() {
    let path = temp_store_path("get_document_requires_selector");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    assert!(matches!(
        get_document(&path, &DocumentGetRequest::default()),
        Err(Error::InvalidRequest { .. })
    ));
}

#[test]
fn promotion_candidates_rank_by_retrieval_signal() {
    let path = temp_store_path("promotion_candidates_rank");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    ingest_source(
        &path,
        &ingest_request(&["alpha apple orchard", "beta banana grove"]),
    )
    .expect("ingest");

    // Two distinct queries both retrieve the apple chunk; the banana chunk is
    // never searched.
    for query in ["apple", "orchard apple"] {
        search_documents(
            &path,
            &DocumentSearchRequest {
                query: query.to_string(),
                space: None,
                limit: 5,
                include_content: false,
                snippet_chars: 40,
                embedding: None,
                skip_recall_log: false,
            },
        )
        .expect("search succeeds");
    }

    let report = promotion_candidates(
        &path,
        &PromotionCandidatesRequest {
            space: None,
            min_hits: 1,
            min_distinct_queries: 1,
            limit: 10,
            include_content: true,
            include_extracted: false,
        },
    )
    .expect("promotion_candidates succeeds");

    assert_eq!(
        report.candidates.len(),
        1,
        "only the searched chunk qualifies"
    );
    let candidate = &report.candidates[0];
    assert_eq!(candidate.hits, 2);
    assert_eq!(candidate.distinct_queries, 2);
    assert!(candidate
        .content
        .as_deref()
        .is_some_and(|c| c.contains("apple")));
}

#[test]
fn promotion_candidates_empty_without_searches() {
    let path = temp_store_path("promotion_candidates_empty");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    ingest_source(&path, &ingest_request(&["unsearched chunk"])).expect("ingest");

    let report = promotion_candidates(&path, &PromotionCandidatesRequest::default())
        .expect("promotion_candidates succeeds");
    assert!(report.candidates.is_empty());
}

#[test]
fn mark_extracted_hides_chunk_from_promotion_candidates() {
    let path = temp_store_path("mark_extracted_hides");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let report = ingest_source(&path, &ingest_request(&["apple orchard notes"])).expect("ingest");
    let id = report.created[0].clone();
    for query in ["apple", "orchard"] {
        search_documents(
            &path,
            &DocumentSearchRequest {
                query: query.to_string(),
                space: None,
                limit: 5,
                include_content: false,
                snippet_chars: 40,
                embedding: None,
                skip_recall_log: false,
            },
        )
        .expect("search succeeds");
    }

    let request = PromotionCandidatesRequest {
        space: None,
        min_hits: 1,
        min_distinct_queries: 1,
        limit: 10,
        include_content: false,
        include_extracted: false,
    };
    assert_eq!(
        promotion_candidates(&path, &request)
            .expect("before")
            .candidates
            .len(),
        1
    );

    let marked = mark_source_episodes_extracted(
        &path,
        &MarkExtractedRequest {
            space: None,
            source_episode_ids: vec![id.clone()],
        },
    )
    .expect("mark succeeds");
    assert_eq!(marked.updated, 1);

    assert!(
        promotion_candidates(&path, &request)
            .expect("after")
            .candidates
            .is_empty(),
        "extracted chunk is hidden from promotion candidates"
    );

    let include_extracted = PromotionCandidatesRequest {
        include_extracted: true,
        ..request
    };
    assert_eq!(
        promotion_candidates(&path, &include_extracted)
            .expect("include")
            .candidates
            .len(),
        1,
        "include_extracted surfaces it again"
    );
}

#[test]
fn mark_extracted_rejects_empty_request() {
    let path = temp_store_path("mark_extracted_empty");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    assert!(matches!(
        mark_source_episodes_extracted(&path, &MarkExtractedRequest::default()),
        Err(Error::InvalidRequest { .. })
    ));
}

#[test]
fn query_alias_shingles_builds_ordered_ngrams() {
    let shingles = super::query_alias_shingles("Replay items from the DLQ pipeline");
    // single tokens (lowercased)
    assert!(shingles.contains("dlq"));
    assert!(shingles.contains("pipeline"));
    // contiguous multi-word shingles, order-preserving
    assert!(shingles.contains("dlq pipeline"));
    assert!(shingles.contains("from the dlq"));
    // not a contiguous span -> absent
    assert!(!shingles.contains("replay dlq"));
    // capped at MAX_ALIAS_SHINGLE_WORDS (3) -> no 4-word shingle
    assert!(!shingles.contains("replay items from the"));
}

#[test]
fn alias_tag_boost_outranks_topical_neighbor() {
    let path = temp_store_path("alias_tag_boost_outranks_topical_neighbor");
    init_store(&path).expect("init succeeds");

    // A is reachable by its alias "k8s" only via the reserved alias:: tag;
    // B is a topical neighbor that shares the "pods" token but has no alias tag.
    let mut a = remember_request("Container orchestration scheduling pods across nodes.");
    a.tags = vec![format!("{}k8s", super::ALIAS_TAG_PREFIX)];
    let a_id = remember_memory(&path, &a).expect("remember A").memory.id;

    let b = remember_request("Container orchestration scheduling pods across cluster nodes.");
    let b_id = remember_memory(&path, &b).expect("remember B").memory.id;

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "k8s pods".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 20,
            include_content: false,
            include_source: false,
            semantic_fallback: "disabled".to_string(),
            lexical_fallback: "disabled".to_string(),
            embedding: None,
            query_token_embedding: None,
            token_model_id: None,
        },
    )
    .expect("search succeeds");

    let a_res = report
        .results
        .iter()
        .find(|r| r.memory_id == a_id)
        .expect("A retrieved");
    let b_res = report.results.iter().find(|r| r.memory_id == b_id);

    // The alias boost must actually fire on A's metadata component. If a future
    // edit drops the boost (or the alias:: tag convention), this fails loudly
    // rather than silently degrading retrieval precision.
    assert!(
        a_res.scores.metadata >= super::ALIAS_MATCH_BOOST,
        "alias-tagged A should carry the alias-match boost in its metadata score, got {}",
        a_res.scores.metadata
    );
    // And the boost must lift A above the topical-only neighbor B.
    if let Some(b_res) = b_res {
        assert!(
            a_res.score > b_res.score,
            "alias-matched A ({}) should outrank topical neighbor B ({})",
            a_res.score,
            b_res.score
        );
    }
}
