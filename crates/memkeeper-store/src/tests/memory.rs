//! Tests for memory operations.

use super::*;

#[test]
fn retrieval_representation_contract_is_bounded_and_versioned() {
    let valid = RetrievalRepresentationInput {
        kind: "contextual-card-v1".to_string(),
        text: "2026-07-14. Release owner requested the verified branch".to_string(),
    };
    validate_retrieval_representation(&valid).expect("valid card");
    assert_eq!(
        retrieval_companion(Some("summary"), Some(&valid)),
        Some(valid.text.as_str())
    );
    assert_eq!(retrieval_companion(Some("summary"), None), Some("summary"));

    for invalid in [
        RetrievalRepresentationInput {
            kind: "contextual-card-v2".into(),
            text: "ok".into(),
        },
        RetrievalRepresentationInput {
            kind: "contextual-card-v1".into(),
            text: "   ".into(),
        },
        RetrievalRepresentationInput {
            kind: "contextual-card-v1".into(),
            text: "x".repeat(513),
        },
    ] {
        assert!(validate_retrieval_representation(&invalid).is_err());
    }
}

#[test]
fn representation_document_uses_one_companion() {
    assert_eq!(
        representation_document("content", Some("card")),
        "card\n\ncontent"
    );
    assert_eq!(representation_document("content", None), "content");
}

#[test]
fn remember_persists_retrieval_representation() {
    let path = temp_store_path("remember_persists_retrieval_representation");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let card = "2026-07-14. Release owner requested canonical deployment memory";

    let report = remember_memory(
        &path,
        &represented_request("fact: canonical deployment memory", card),
    )
    .expect("represented remember succeeds");
    let stored = report
        .memory
        .retrieval_representation
        .as_ref()
        .expect("active representation");
    assert_eq!(stored.version_id, report.memory.version_id);
    assert_eq!(stored.kind, "contextual-card-v1");
    assert_eq!(stored.text, card);
    assert_eq!(stored.text_sha256, sha256_hex(card.as_bytes()));

    let status = report.representation.as_ref().expect("write status");
    assert_eq!(status.kind, stored.kind);
    assert_eq!(status.text_sha256, stored.text_sha256);
    assert!(status.fts_indexed);
    assert!(!status.semantic_indexed);
    assert_eq!(status.status, "lexical_only");

    let loaded = get_memory(
        &path,
        &report.memory.id,
        GetOptions {
            include_history: true,
            include_links: false,
            include_source: false,
        },
    )
    .expect("represented memory loads");
    assert_eq!(loaded.retrieval_representation.as_ref(), Some(stored));
    assert_eq!(
        loaded.versions.as_ref().expect("history")[0]
            .retrieval_representation
            .as_ref(),
        Some(stored)
    );

    let connection = Connection::open(&path).expect("open store");
    let representation_rows: i64 = connection
        .query_row("SELECT COUNT(*) FROM memory_representations", [], |row| {
            row.get(0)
        })
        .expect("representation count");
    let fts_rows: i64 = connection
        .query_row("SELECT COUNT(*) FROM memory_fts", [], |row| row.get(0))
        .expect("fts count");
    assert_eq!(representation_rows, 1);
    assert_eq!(fts_rows, 1);
    cleanup_store(&path);
}

#[test]
fn representation_dry_run_rolls_back() {
    let path = temp_store_path("representation_dry_run_rolls_back");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut request = represented_request(
        "fact: canonical deployment memory",
        "2026-07-14. Release owner requested canonical deployment memory",
    );
    request.dry_run = true;
    remember_memory(&path, &request).expect("dry run succeeds");

    let connection = Connection::open(&path).expect("open store");
    for table in ["memories", "memory_versions", "memory_representations"] {
        let count: i64 = connection
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .expect("count rows");
        assert_eq!(count, 0, "{table} must roll back");
    }
    cleanup_store(&path);
}

#[test]
fn representation_sqlite_failure_rolls_back() {
    let path = temp_store_path("representation_sqlite_failure_rolls_back");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    Connection::open(&path)
        .expect("open store")
        .execute_batch(
            "CREATE TRIGGER fail_remember_event BEFORE INSERT ON memory_events
             BEGIN SELECT RAISE(ABORT, 'injected remember failure'); END;",
        )
        .expect("create failure trigger");

    let error = remember_memory(
        &path,
        &represented_request(
            "fact: canonical deployment memory",
            "2026-07-14. Release owner requested canonical deployment memory",
        ),
    )
    .expect_err("injected failure aborts remember");
    assert!(error.to_string().contains("injected remember failure"));

    let connection = Connection::open(&path).expect("open store");
    for table in [
        "memories",
        "memory_versions",
        "memory_representations",
        "memory_events",
        "memory_fts",
        "memory_fts_public",
    ] {
        let count: i64 = connection
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .expect("count rows");
        assert_eq!(count, 0, "{table} must roll back");
    }
    connection
        .execute_batch("DROP TRIGGER fail_remember_event;")
        .expect("drop failure trigger");
    cleanup_store(&path);
}

#[test]
fn representation_survives_lifecycle_changes() {
    let path = temp_store_path("representation_survives_lifecycle_changes");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut old = represented_request("fact: deployment target is blue", "old blue card");
    old.entity_key = Some("deployment".to_string());
    old.claim_key = Some("target".to_string());
    old.observed_at = Some("2026-07-13T12:00:00Z".to_string());
    old.source_ref_json = Some(r#"{"type":"manual","path":"/private/old"}"#.to_string());
    let old = remember_memory(&path, &old).expect("remember old");

    let mut active = represented_request("fact: deployment target is green", "new green card");
    active.entity_key = Some("deployment".to_string());
    active.claim_key = Some("target".to_string());
    active.observed_at = Some("2026-07-14T12:00:00Z".to_string());
    active.source_ref_json = Some(r#"{"type":"manual","path":"/private/new"}"#.to_string());
    let active = remember_memory(&path, &active).expect("remember replacement");
    assert_eq!(active.auto_superseded, vec![old.memory.id.clone()]);

    let mut conflict =
        represented_request("continuity: deployment target is red", "red conflict card");
    conflict.kind = Some("continuity".to_string());
    conflict.entity_key = Some("deployment".to_string());
    conflict.claim_key = Some("target".to_string());
    let conflict = remember_memory(&path, &conflict).expect("remember conflict");
    assert!(!conflict.conflict_candidates.is_empty());

    forget_memory(
        &path,
        &ForgetRequest {
            id: active.memory.id.clone(),
            reason: Some("lifecycle test".to_string()),
            mode: "tombstone".to_string(),
            corrected_by: None,
            dry_run: false,
        },
    )
    .expect("tombstone represented memory");

    for (memory_id, expected_text) in [
        (&old.memory.id, "old blue card"),
        (&active.memory.id, "new green card"),
        (&conflict.memory.id, "red conflict card"),
    ] {
        for include_source in [false, true] {
            let loaded = get_memory(
                &path,
                memory_id,
                GetOptions {
                    include_history: true,
                    include_links: true,
                    include_source,
                },
            )
            .expect("load lifecycle memory");
            let representation = loaded
                .retrieval_representation
                .as_ref()
                .expect("active version representation");
            assert_eq!(representation.text, expected_text);
            assert_eq!(
                representation.text_sha256,
                sha256_hex(expected_text.as_bytes())
            );
            assert_eq!(
                loaded.versions.as_ref().expect("versions")[0]
                    .retrieval_representation
                    .as_ref(),
                Some(representation)
            );
        }
    }
    let rows: i64 = Connection::open(&path)
        .expect("open store")
        .query_row("SELECT COUNT(*) FROM memory_representations", [], |row| {
            row.get(0)
        })
        .expect("representation count");
    assert_eq!(rows, 3);
    cleanup_store(&path);
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
            retrieval_representation: None,
            tags: vec!["memory".to_string(), "sqlite".to_string()],
            entity_key: Some("project:memkeeper".to_string()),
            claim_key: Some("mvp.boundary".to_string()),
            graph: None,
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
        retrieval_representation: None,
        tags: Vec::new(),
        entity_key: None,
        claim_key: None,
        confidence: 1.0,
        observed_at: None,
        valid_from: None,
        valid_to: None,
        graph: None,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
    use crate::correction_event_data_json;

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
fn now_julian_day_matches_sqlite_and_agrees_with_stored_recency() {
    let connection = Connection::open_in_memory().expect("open in-memory connection");
    let now_jd = now_julian_day(&connection).expect("read julian day");

    // Same clock SQLite stamps rows with, so the Rust re-scoring pass and the
    // SQL-side recency ordering cannot drift apart.
    let sqlite_jd: f64 = connection
        .query_row("SELECT julianday('now')", [], |row| row.get(0))
        .expect("sqlite julianday");
    assert!(
        (now_jd - sqlite_jd).abs() < 1e-4,
        "now_julian_day {now_jd} should match sqlite julianday {sqlite_jd}"
    );

    // Guards against a malformed query silently yielding 0.0 or an epoch-offset
    // value: 2020-01-01 is JD 2458849.5, 2100-01-01 is JD 2488069.5.
    assert!(
        (2_458_849.5..2_488_069.5).contains(&now_jd),
        "now_julian_day {now_jd} is not a plausible current Julian day"
    );

    // An age computed against a SQLite-produced timestamp must be non-negative,
    // which is what `recency_score_for_silo` relies on for its clamp.
    let stored_jd: f64 = connection
        .query_row("SELECT julianday('now', '-1 day')", [], |row| row.get(0))
        .expect("sqlite julianday offset");
    let age_days = now_jd - stored_jd;
    assert!(
        (age_days - 1.0).abs() < 1e-3,
        "a row stamped one day ago should read as ~1 day old, got {age_days}"
    );
}

#[test]
fn volatile_recency_outweighs_durable_recency_for_recent_memory() {
    // Fixed reference point: these assertions are about the shape of the decay
    // curve relative to `now_jd`, not about the absolute clock.
    let now_jd = FIXED_TEST_JULIAN_DAY;
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
    let now_jd = FIXED_TEST_JULIAN_DAY;
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
    assert!(recency_score_for_silo(None, "short-term", now_jd).abs() < f64::EPSILON);
    assert!(recency_score_for_silo(Some(f64::NAN), "durable", now_jd).abs() < f64::EPSILON);
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
            batch_id: None,
            latency_ms: None,
            latency_source: None,
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
            batch_id: None,
            latency_ms: None,
            latency_source: None,
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
            batch_id: None,
            latency_ms: None,
            latency_source: None,
            events: Vec::new(),
            touch_accessed: false,
        },
    );
    assert!(matches!(empty, Err(Error::InvalidRequest { .. })));
    cleanup_store(&path);
}
