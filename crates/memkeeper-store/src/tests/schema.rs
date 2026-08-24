//! Tests for schema operations.

use super::*;

#[test]
fn schema_embeds_required_objects() {
    assert!(schema_mentions_required_objects());
}

#[test]
fn schema_changing_sql_is_owned_by_schema_module() {
    let prohibited = [
        "CREATE TABLE",
        "CREATE TEMP TABLE",
        "CREATE TEMPORARY TABLE",
        "CREATE VIRTUAL TABLE",
        "CREATE INDEX",
        "CREATE UNIQUE INDEX",
        "CREATE TRIGGER",
        "CREATE VIEW",
        "ALTER TABLE",
        "DROP TABLE",
        "DROP INDEX",
        "DROP TRIGGER",
        "DROP VIEW",
        "PRAGMA USER_VERSION =",
        "PRAGMA USER_VERSION(",
        "INSERT INTO SCHEMA_MIGRATIONS",
        "INSERT OR IGNORE INTO SCHEMA_MIGRATIONS",
        "INSERT OR REPLACE INTO SCHEMA_MIGRATIONS",
        "REPLACE INTO SCHEMA_MIGRATIONS",
        "UPDATE SCHEMA_MIGRATIONS",
        "DELETE FROM SCHEMA_MIGRATIONS",
    ];
    let mut violations = Vec::new();

    for (path, normalized) in normalized_production_rust_sources("schema.rs") {
        for token in prohibited {
            if normalized.contains(token) {
                violations.push(format!("{} contains {token}", path.display()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "schema-changing SQL must live in schema.rs:\n{}",
        violations.join("\n")
    );
}

#[test]
fn connection_lifecycle_is_owned_by_connection_module() {
    let owned_functions = [
        "init_store",
        "open_initialized_read_fast",
        "open_initialized_write",
        "configure_connection",
        "inspect_on_copy",
        "claim_or_preflight_init_path",
        "inspect_existing_store",
    ];
    let mut violations = Vec::new();

    for (path, normalized) in normalized_production_rust_sources("connection.rs") {
        for function in owned_functions {
            let definition = format!("FN {}", function.to_ascii_uppercase());
            if normalized.contains(&definition) {
                violations.push(format!("{} defines {function}", path.display()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "connection lifecycle must live in connection.rs:\n{}",
        violations.join("\n")
    );
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
        retrieval_representation: None,
        tags: Vec::new(),
        entity_key: keys.map(|(entity, _)| entity.to_string()),
        claim_key: keys.map(|(_, claim)| claim.to_string()),
        graph: None,
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
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute(
            "DELETE FROM entities WHERE space_name = 'workspace-memory' AND entity_key = 'entity:a'",
            [],
        )
        .expect("delete entity projection to simulate drift");
    drop(connection);

    // Default stats omits the rollup; the opt-in variant includes it.
    assert!(store_stats(&path, false).expect("stats").health.is_none());
    let health = store_stats_with_health(&path, false)
        .expect("health stats")
        .health
        .expect("health present");

    assert_eq!(health.active, 2);
    assert_eq!(health.tombstoned, 0);
    assert_eq!(health.active_without_keys, 1);
    assert_eq!(health.active_missing_entity_projection, 1);
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
            maxsim_shortlist: 0,
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
fn schema_v6_has_version_representations_and_retrieval_text_fts() {
    let path = temp_store_path("schema_v6_representation");
    cleanup_store(&path);
    init_store(&path).expect("init");
    let connection = Connection::open(&path).expect("open");
    let version: i32 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .expect("version");
    assert_eq!(version, 6);
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name='memory_representations'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("table count"),
        1
    );
    for table in ["memory_fts", "memory_fts_public"] {
        let fts_sql: String = connection
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name=?1",
                [table],
                |row| row.get(0),
            )
            .expect("fts sql");
        assert!(fts_sql.contains("retrieval_text"));
        assert!(!fts_sql.contains("summary"));
    }
    drop(connection);
    cleanup_store(&path);
}

#[test]
fn migrate_v5_representation_preserves_identity_observables() {
    let path = temp_store_path("migrate_v5_representation");
    cleanup_store(&path);
    init_store(&path).expect("init");
    let mut request = basic_request("fact: representation migration marker remains searchable");
    request.summary = Some("representation migration marker".to_string());
    request.tags = vec!["migration-fixture".to_string()];
    request.entity_key = Some("test:representation-migration".to_string());
    request.claim_key = Some("test.representation-migration".to_string());
    let remembered = remember_memory(&path, &request).expect("remember fixture");

    let before_search = representation_migration_search_ids(&path);
    let before_pack = representation_migration_pack_ids(&path);
    let connection = Connection::open(&path).expect("open fixture");
    let count_queries = [
        "SELECT count(*) FROM memories",
        "SELECT count(*) FROM memory_versions",
        "SELECT count(*) FROM memory_events",
        "SELECT count(*) FROM memory_tags",
        "SELECT count(*) FROM source_episodes",
        "SELECT count(*) FROM entities",
        "SELECT count(*) FROM relationships",
        "SELECT count(*) FROM embeddings",
        "SELECT count(*) FROM memory_token_embeddings",
    ];
    let before_counts: Vec<i64> = count_queries
        .iter()
        .map(|sql| {
            connection
                .query_row(sql, [], |row| row.get(0))
                .expect("fixture count")
        })
        .collect();
    let before_hash: String = connection
        .query_row(
            "SELECT content_sha256 FROM memory_versions WHERE id=?1",
            [&remembered.memory.version_id],
            |row| row.get(0),
        )
        .expect("content hash");
    downgrade_representation_fixture_to_v5(&connection);
    drop(connection);

    init_store(&path).expect("migrate v5 to v6");
    let connection = Connection::open(&path).expect("open migrated fixture");
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i32>(0))
            .expect("schema version"),
        6
    );
    assert_eq!(
        connection
            .query_row("SELECT count(*) FROM memory_representations", [], |row| {
                row.get::<_, i64>(0)
            })
            .expect("representation count"),
        0
    );
    let after_counts: Vec<i64> = count_queries
        .iter()
        .map(|sql| {
            connection
                .query_row(sql, [], |row| row.get(0))
                .expect("migrated count")
        })
        .collect();
    let after_hash: String = connection
        .query_row(
            "SELECT content_sha256 FROM memory_versions WHERE id=?1",
            [&remembered.memory.version_id],
            |row| row.get(0),
        )
        .expect("migrated content hash");
    drop(connection);

    assert_eq!(after_counts, before_counts);
    assert_eq!(after_hash, before_hash);
    assert_eq!(representation_migration_search_ids(&path), before_search);
    assert_eq!(representation_migration_pack_ids(&path), before_pack);
    cleanup_store(&path);
}

#[test]
fn open_initialized_write_migrates_v5_representation_schema() {
    let path = temp_store_path("open_write_migrates_v5_representation");
    cleanup_store(&path);
    init_store(&path).expect("init");
    let connection = Connection::open(&path).expect("open fixture");
    downgrade_representation_fixture_to_v5(&connection);
    drop(connection);

    let connection = open_initialized_write(&path).expect("write open migrates v5");
    assert_eq!(
        connection
            .query_row("PRAGMA user_version", [], |row| row.get::<_, i32>(0))
            .expect("schema version"),
        6
    );
    assert_eq!(
        connection
            .query_row("SELECT count(*) FROM memory_representations", [], |row| row
                .get::<_, i64>(0),)
            .expect("representation table"),
        0
    );
    drop(connection);
    cleanup_store(&path);
}

#[test]
fn query_alias_shingles_builds_ordered_ngrams() {
    let shingles = crate::query_alias_shingles("Replay items from the DLQ pipeline");
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
    a.tags = vec![format!("{}k8s", crate::ALIAS_TAG_PREFIX)];
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
            maxsim_shortlist: 0,
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
        a_res.scores.metadata >= crate::ALIAS_MATCH_BOOST,
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

#[test]
fn resolve_space_filter_maps_default_named_and_all_scopes() {
    // Empty stays back-compatible: the default space.
    assert_eq!(
        crate::resolve_space_filter(&[]),
        vec![DEFAULT_SPACE.to_string()]
    );
    // The "*" sentinel => all spaces: an empty vector, which the predicate
    // builder renders as no `space_name` filter at all.
    assert!(crate::resolve_space_filter(&["*".to_string()]).is_empty());
    // A stray whitespace-padded sentinel still resolves to all spaces.
    assert!(crate::resolve_space_filter(&[" * ".to_string()]).is_empty());
    // The sentinel dominates a mixed list (union beats a named subset).
    assert!(crate::resolve_space_filter(&["*".to_string(), "trading".to_string()]).is_empty());
    // Named spaces pass through unchanged.
    assert_eq!(
        crate::resolve_space_filter(&["trading".to_string()]),
        vec!["trading".to_string()]
    );
}

#[test]
fn cross_space_search_scopes_default_named_and_all() {
    let (path, default_id, trading_id) = seed_two_space_store("cross_space_search");

    // Default (empty) => only the default-space memory.
    assert_eq!(
        search_ids_for_spaces(&path, vec![]),
        vec![default_id.clone()]
    );

    // Named => only the trading-space memory.
    assert_eq!(
        search_ids_for_spaces(&path, vec!["trading".to_string()]),
        vec![trading_id.clone()]
    );

    // "*" => the union of both spaces.
    let mut expected = vec![default_id, trading_id];
    expected.sort();
    assert_eq!(
        search_ids_for_spaces(&path, vec!["*".to_string()]),
        expected
    );

    cleanup_store(&path);
}

#[test]
fn cross_space_memory_list_scopes_default_named_and_all() {
    let (path, _default_id, _trading_id) = seed_two_space_store("cross_space_list");

    let count_for = |spaces: Vec<String>| -> usize {
        list_memories(
            &path,
            &MemoryListRequest {
                filters: SearchFilters {
                    spaces,
                    ..SearchFilters::default()
                },
                limit: 50,
                offset: 0,
                snippet_chars: 0,
                include_content: false,
                include_source: false,
                order: "updated_desc".to_string(),
            },
        )
        .expect("list succeeds")
        .results
        .len()
    };

    assert_eq!(count_for(vec![]), 1, "default space holds one memory");
    assert_eq!(
        count_for(vec!["trading".to_string()]),
        1,
        "trading space holds one memory"
    );
    assert_eq!(
        count_for(vec!["*".to_string()]),
        2,
        "all spaces unions both"
    );

    cleanup_store(&path);
}

#[test]
fn cross_space_pack_pool_unions_all_spaces() {
    let (path, default_id, trading_id) = seed_two_space_store("cross_space_pack");

    let pool_ids = |spaces: Vec<String>| -> Vec<String> {
        let mut ids: Vec<String> = crate::build_pack_pool(
            &path,
            &PackRequest {
                title: "cross-space".to_string(),
                queries: vec!["retrieval marker".to_string()],
                filters: SearchFilters {
                    spaces,
                    ..SearchFilters::default()
                },
                max_memories: 10,
                max_chars: 4000,
                format: "markdown".to_string(),
                min_score: 0.0,
                rerank_candidates: 0,
                query_embeddings: None,
                query_token_embeddings: None,
                token_model_id: None,
                maxsim_shortlist: 0,
            },
        )
        .expect("pack pool")
        .into_iter()
        .map(|item| item.memory_id)
        .collect();
        ids.sort();
        ids
    };

    // Default pack sees only the default space; "*" unions both.
    assert_eq!(pool_ids(vec![]), vec![default_id.clone()]);
    let mut expected = vec![default_id, trading_id];
    expected.sort();
    assert_eq!(pool_ids(vec!["*".to_string()]), expected);

    cleanup_store(&path);
}

#[test]
fn write_paths_reject_all_spaces_sentinel() {
    let path = temp_store_path("write_paths_reject_all_spaces_sentinel");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    // remember into the sentinel space is rejected (would otherwise auto-create
    // a space named "*" that collides with the union scope).
    let mut mem = basic_request("decision: must not be writable to the sentinel space");
    mem.space = Some("*".to_string());
    let err = remember_memory(&path, &mem).expect_err("remember into * must fail");
    assert!(matches!(err, Error::InvalidRequest { .. }), "got {err:?}");

    // Creating a space literally named "*" is rejected.
    let err = create_space(
        &path,
        &SpaceCreateRequest {
            name: "*".to_string(),
            display_name: None,
            description: None,
            default_silo: None,
            ontology: None,
            config_json: None,
            if_not_exists: false,
        },
    )
    .expect_err("create_space(*) must fail");
    assert!(matches!(err, Error::InvalidRequest { .. }), "got {err:?}");

    cleanup_store(&path);
}

