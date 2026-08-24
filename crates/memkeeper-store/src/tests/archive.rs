//! Tests for archive operations.

use super::*;

#[cfg(feature = "semantic")]
#[test]
fn import_accepts_archive_carrying_foreign_protocol_version() {
    let source = temp_store_path("foreign_protocol_source");
    let target = temp_store_path("foreign_protocol_target");
    let export_path = temp_store_path("foreign_protocol_export").with_extension("jsonl");
    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&export_path);
    init_store(&source).expect("init source");

    let mut request = basic_request("decision: archive spans every exported table");
    request.embedding = Some(vec![0.25_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
    request.embedding_model_id = Some("round-trip-model".to_string());
    remember_memory(&source, &request).expect("remember source");

    upsert_entity(
        &source,
        &entity_upsert_request("project:memkeeper", "Memkeeper"),
    )
    .expect("subject entity");
    upsert_entity(
        &source,
        &entity_upsert_request("component:sqlite", "SQLite"),
    )
    .expect("object entity");
    upsert_relationship(
        &source,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("project:memkeeper".to_string()),
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:sqlite".to_string()),
            confidence: 0.7,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("relationship upsert");

    export_store(
        &source,
        &ExportRequest {
            output_path: export_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export source");

    // Rewrite only the stored row, exactly as an archive from the other-named
    // build carries it. The header stays valid for this build, which is the case
    // that reached the failing validation instead of being rejected at parse.
    // The substituted value is deliberately not either build's own name: this
    // file is rebranded when the public tree is generated, and a value carrying
    // a real brand would be rewritten into the value it is supposed to differ
    // from, silently turning this replace into a no-op there.
    let archive = std::fs::read_to_string(&export_path).expect("read archive");
    let rewritten = archive.replace(
        r#"{"key":"protocol_version","value":"memkeeper.v0.1""#,
        r#"{"key":"protocol_version","value":"otherbuild.v0.1""#,
    );
    assert_ne!(
        archive, rewritten,
        "archive must carry a protocol_version config row"
    );
    std::fs::write(&export_path, rewritten).expect("write archive");

    let report = import_store(
        &target,
        &ImportRequest {
            input_path: export_path.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect("import must accept an archive from the other-named build");
    assert_eq!(report.schema_version, SCHEMA_VERSION);

    let connection = Connection::open(&target).expect("open imported store");
    for (table, expected) in [
        ("memories", 1_i64),
        ("memory_versions", 1),
        ("entities", 2),
        ("relationships", 1),
        ("embeddings", 1),
    ] {
        let actual: i64 = connection
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap_or_else(|error| panic!("count {table}: {error}"));
        assert_eq!(actual, expected, "{table} rows after import");
    }
    let protocol: String = connection
        .query_row(
            "SELECT value FROM config_kv WHERE key = 'protocol_version'",
            [],
            |row| row.get(0),
        )
        .expect("protocol_version row");
    assert_eq!(protocol, "memkeeper.v0.1");
    drop(connection);

    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&export_path);
}

#[cfg(feature = "semantic")]
#[test]
fn import_accepts_archive_without_schema_or_config_metadata() {
    let source = temp_store_path("metadata_light_source");
    let target = temp_store_path("metadata_light_target");
    let export_path = temp_store_path("metadata_light_export").with_extension("jsonl");
    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&export_path);
    init_store(&source).expect("init source");

    let mut request = basic_request("decision: metadata-light archive still restores");
    request.embedding = Some(vec![0.4_f32; crate::DEFAULT_SEMANTIC_EMBEDDING_DIMS]);
    request.embedding_model_id = Some("metadata-light-model".to_string());
    let remembered = remember_memory(&source, &request).expect("remember source");

    upsert_entity(&source, &entity_upsert_request("project:hosted", "Hosted"))
        .expect("subject entity");
    upsert_entity(&source, &entity_upsert_request("component:store", "Store"))
        .expect("object entity");
    upsert_relationship(
        &source,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("project:hosted".to_string()),
            relation_type: "uses".to_string(),
            object_entity_key: Some("component:store".to_string()),
            confidence: 0.6,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("relationship upsert");

    export_store(
        &source,
        &ExportRequest {
            output_path: export_path.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export source");

    // Strip both metadata tables and reconcile the footer, reproducing the shape
    // a hosted export actually has: the header still declares both tables.
    strip_metadata_rows(&export_path);

    let report = import_store(
        &target,
        &ImportRequest {
            input_path: export_path.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect("import must accept an archive with no schema/config metadata");
    assert_eq!(report.schema_version, SCHEMA_VERSION);

    // The store must be usable, not merely loaded: identity metadata restored,
    // content intact, and retrieval working on the recovered rows.
    assert_store_identity_metadata(&target);
    let connection = Connection::open(&target).expect("open imported store");
    for (table, expected) in [
        ("memories", 1_i64),
        ("memory_versions", 1),
        ("entities", 2),
        ("relationships", 1),
        ("embeddings", 1),
    ] {
        let actual: i64 = connection
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap_or_else(|error| panic!("count {table}: {error}"));
        assert_eq!(actual, expected, "{table} rows after import");
    }
    drop(connection);

    let search = search_memories(
        &target,
        &SearchRequest {
            query: "metadata-light archive".to_string(),
            filters: SearchFilters::default(),
            limit: 5,
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
    .expect("search the restored store");
    assert_eq!(search.results[0].memory_id, remembered.memory.id);

    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&export_path);
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
#[allow(clippy::too_many_lines)]
fn export_import_preserves_retrieval_representation() {
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
    request.summary = Some("persisted import summary".to_string());
    request.retrieval_representation = Some(RetrievalRepresentationInput {
        kind: "contextual-card-v1".to_string(),
        text: "archive card contains zephyr-archive-term".to_string(),
    });
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
    assert_eq!(
        loaded
            .retrieval_representation
            .as_ref()
            .expect("imported representation"),
        remembered
            .memory
            .retrieval_representation
            .as_ref()
            .expect("source representation")
    );
    let search = search_memories(
        &import_path,
        &SearchRequest {
            query: "zephyr-archive-term".to_string(),
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
            maxsim_shortlist: 0,
        },
    )
    .expect("search imported card");
    assert_eq!(search.results[0].memory_id, remembered.memory.id);

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
#[allow(clippy::too_many_lines)]
fn import_round_trip_preserves_custom_long_term_default() {
    let path = temp_store_path("import_custom_long_term_source");
    let import_path = temp_store_path("import_custom_long_term_target");
    let export_a = temp_store_path("import_custom_long_term_a").with_extension("jsonl");
    let export_b = temp_store_path("import_custom_long_term_b").with_extension("jsonl");
    cleanup_store(&path);
    cleanup_store(&import_path);
    cleanup_store(&export_a);
    cleanup_store(&export_b);

    init_store(&path).expect("init succeeds");
    create_space(
        &path,
        &SpaceCreateRequest {
            name: "hosted-validation".to_string(),
            display_name: Some("Hosted Validation".to_string()),
            description: Some("Synthetic hosted contract validation".to_string()),
            default_silo: Some("long-term".to_string()),
            ontology: None,
            config_json: None,
            if_not_exists: false,
        },
    )
    .expect("custom space create succeeds");

    init_store(&path).expect("re-init preserves empty custom long-term silo");
    let reinitialized_source_space = list_spaces(&path)
        .expect("list reinitialized source spaces")
        .spaces
        .into_iter()
        .find(|space| space.name == "hosted-validation")
        .expect("custom reinitialized source space exists");
    assert_eq!(reinitialized_source_space.default_silo, "long-term");
    let reinitialized_source_silos = list_silos(
        &path,
        &SiloListRequest {
            space: Some("hosted-validation".to_string()),
        },
    )
    .expect("list reinitialized source silos");
    assert!(reinitialized_source_silos
        .silos
        .iter()
        .any(|silo| silo.name == "long-term" && silo.is_default));

    let card = "archive card contains custom-long-term-schema-six";
    let mut request =
        represented_request("preference: synthetic hosted contract uses long-term", card);
    request.space = Some("hosted-validation".to_string());
    request.silo = None;
    let remembered = remember_memory(&path, &request).expect("remember in custom space");
    assert_eq!(remembered.memory.silo, "long-term");
    assert_eq!(
        remembered
            .memory
            .retrieval_representation
            .as_ref()
            .expect("source representation")
            .text,
        card
    );

    export_store(
        &path,
        &ExportRequest {
            output_path: export_a.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("source export succeeds");

    let source_bytes = fs::read(&export_a).expect("read source export");
    let source_text = std::str::from_utf8(&source_bytes).expect("source export is utf8");
    let records = source_text
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("parse export row"))
        .collect::<Vec<_>>();
    let header = records
        .iter()
        .find(|record| record["type"] == "header")
        .expect("archive header");
    assert_eq!(header["schema_version"], SCHEMA_VERSION);
    assert_eq!(SCHEMA_VERSION, 6);
    assert_eq!(
        records
            .iter()
            .filter(|record| {
                record["type"] == "row" && record["table"] == "memory_representations"
            })
            .count(),
        1
    );
    let footer = records
        .iter()
        .find(|record| record["type"] == "footer")
        .expect("archive footer");
    assert_eq!(footer["table_counts"]["memory_representations"], 1);

    import_store(
        &import_path,
        &ImportRequest {
            input_path: export_a.clone(),
            format: "jsonl".to_string(),
            dry_run: false,
            conflict_policy: "fail_if_exists".to_string(),
        },
    )
    .expect("strict import succeeds");
    export_store(
        &import_path,
        &ExportRequest {
            output_path: export_b.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("round-trip export succeeds");

    assert_eq!(
        source_bytes,
        fs::read(&export_b).expect("read round-trip export"),
        "schema-6 custom Space archive must round-trip byte-identically"
    );
    let imported_space = list_spaces(&import_path)
        .expect("list imported spaces")
        .spaces
        .into_iter()
        .find(|space| space.name == "hosted-validation")
        .expect("custom imported space exists");
    assert_eq!(imported_space.default_silo, "long-term");
    let imported = get_memory(
        &import_path,
        &remembered.memory.id,
        GetOptions {
            include_history: false,
            include_links: false,
            include_source: false,
        },
    )
    .expect("get imported represented memory");
    assert_eq!(
        imported
            .retrieval_representation
            .as_ref()
            .expect("imported representation")
            .text,
        card
    );

    init_store(&import_path).expect("re-init imported store");
    let reinitialized_space = list_spaces(&import_path)
        .expect("list reinitialized spaces")
        .spaces
        .into_iter()
        .find(|space| space.name == "hosted-validation")
        .expect("custom reinitialized space exists");
    assert_eq!(reinitialized_space.default_silo, "long-term");

    cleanup_store(&path);
    cleanup_store(&import_path);
    cleanup_store(&export_a);
    cleanup_store(&export_b);
}

#[test]
fn import_rejects_representation_hash_mismatch() {
    let source = temp_store_path("representation_hash_source");
    let target = temp_store_path("representation_hash_target");
    let archive = temp_store_path("representation_hash_archive").with_extension("jsonl");
    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&archive);
    init_store(&source).expect("init source");
    let remembered = remember_memory(
        &source,
        &represented_request("fact: archive hash target", "archive hash card"),
    )
    .expect("remember represented");
    export_store(
        &source,
        &ExportRequest {
            output_path: archive.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export represented store");
    let hash = remembered
        .memory
        .retrieval_representation
        .as_ref()
        .expect("representation")
        .text_sha256
        .clone();
    let contents = fs::read_to_string(&archive)
        .expect("read archive")
        .replacen(&hash, &"0".repeat(64), 1);
    fs::write(&archive, contents).expect("write corrupted archive");

    let error = import_store(&target, &import_request_for(&archive, false))
        .expect_err("hash mismatch must fail");
    assert!(error
        .to_string()
        .contains("import archive has representation hash mismatch"));
    assert!(!target.exists());
    cleanup_store(&source);
    cleanup_store(&target);
    cleanup_store(&archive);
}

#[test]
#[allow(clippy::too_many_lines)]
fn import_accepts_v5_archive_with_identity_fallback() {
    let source = temp_store_path("legacy_v5_archive_source");
    let target = temp_store_path("legacy_v5_archive_target");
    let current = temp_store_path("legacy_v5_archive_current").with_extension("jsonl");
    let legacy = temp_store_path("legacy_v5_archive_legacy").with_extension("jsonl");
    for path in [&source, &target, &current, &legacy] {
        cleanup_store(path);
    }
    init_store(&source).expect("init source");
    let mut first_request = basic_request("decision: legacy archive alpha target");
    first_request.summary = Some("legacy alpha summary".to_string());
    first_request.tags = vec!["legacy".to_string()];
    let first = remember_memory(&source, &first_request).expect("remember first");
    remember_memory(&source, &basic_request("fact: legacy archive beta filler"))
        .expect("remember second");
    export_store(
        &source,
        &ExportRequest {
            output_path: current.clone(),
            format: "jsonl".to_string(),
        },
    )
    .expect("export current archive");

    let mut legacy_lines = Vec::new();
    for line in fs::read_to_string(&current)
        .expect("read current archive")
        .lines()
    {
        let mut value: serde_json::Value = serde_json::from_str(line).expect("parse archive row");
        match value["type"].as_str().expect("record type") {
            "header" => {
                value["schema_version"] = serde_json::json!(5);
                value["tables"]
                    .as_array_mut()
                    .expect("tables")
                    .retain(|table| table != "memory_representations");
            }
            "row" if value["table"] == "memory_representations" => continue,
            "row" if value["table"] == "schema_migrations" && value["data"]["version"] == 6 => {
                continue;
            }
            "row" if value["table"] == "config_kv" && value["data"]["key"] == "schema_version" => {
                value["data"]["value"] = serde_json::json!("5");
            }
            "footer" => {
                let counts = value["table_counts"].as_object_mut().expect("table counts");
                counts.remove("memory_representations");
                let migrations = counts["schema_migrations"]
                    .as_u64()
                    .expect("migration count");
                counts.insert(
                    "schema_migrations".to_string(),
                    serde_json::json!(migrations - 1),
                );
                let rows = value["row_count"].as_u64().expect("row count");
                value["row_count"] = serde_json::json!(rows - 1);
            }
            _ => {}
        }
        legacy_lines.push(serde_json::to_string(&value).expect("serialize legacy row"));
    }
    fs::write(&legacy, legacy_lines.join("\n") + "\n").expect("write legacy archive");

    let report = import_store(&target, &import_request_for(&legacy, false)).expect("import v5");
    assert_eq!(report.schema_version, 5);
    assert_eq!(
        store_stats(&target, true)
            .expect("target stats")
            .schema_version,
        6
    );
    let connection = Connection::open(&target).expect("open target");
    let representations: i64 = connection
        .query_row("SELECT COUNT(*) FROM memory_representations", [], |row| {
            row.get(0)
        })
        .expect("representation count");
    assert_eq!(representations, 0);
    drop(connection);
    let loaded = get_memory(
        &target,
        &first.memory.id,
        GetOptions {
            include_history: true,
            include_links: false,
            include_source: false,
        },
    )
    .expect("load imported identity memory");
    assert_eq!(loaded.content_sha256, first.memory.content_sha256);
    assert!(loaded.retrieval_representation.is_none());
    let source_search = search_memories(
        &source,
        &SearchRequest {
            query: "legacy archive".to_string(),
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
            maxsim_shortlist: 0,
        },
    )
    .expect("source search");
    let target_search = search_memories(
        &target,
        &SearchRequest {
            query: "legacy archive".to_string(),
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
            maxsim_shortlist: 0,
        },
    )
    .expect("target search");
    assert_eq!(
        source_search
            .results
            .iter()
            .map(|row| row.memory_id.as_str())
            .collect::<Vec<_>>(),
        target_search
            .results
            .iter()
            .map(|row| row.memory_id.as_str())
            .collect::<Vec<_>>()
    );

    for path in [&source, &target, &current, &legacy] {
        cleanup_store(path);
    }
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
