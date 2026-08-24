//! Tests for graph operations.

use super::*;

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
        "Memkeeper".to_string(),
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
fn entity_upsert_mirrors_canonical_name_as_alias() {
    let path = temp_store_path("entity_upsert_mirrors_canonical_name_as_alias");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let first = upsert_entity(
        &path,
        &entity_upsert_request("person:steve", "Steve Edwards"),
    )
    .expect("entity upsert succeeds");
    assert_eq!(first.entity.aliases, vec!["Steve Edwards".to_string()]);

    let second = upsert_entity(
        &path,
        &entity_upsert_request("person:steve", "Steve Edwards"),
    )
    .expect("replay succeeds");
    assert_eq!(second.entity.aliases, first.entity.aliases);

    let connection = Connection::open(&path).expect("open store");
    let alias_count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM entity_aliases
              WHERE entity_id = ?1 AND normalized_alias = 'steve edwards'",
            [&first.entity.id],
            |row| row.get(0),
        )
        .expect("count canonical alias");
    assert_eq!(alias_count, 1);

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
fn relationship_confidence_rises_with_evidence_multiplicity() {
    // Saturating, monotonic, in (0,1): one link is moderate, more links stronger.
    assert!((relationship_confidence_from_evidence(1) - 0.5).abs() < 1e-9);
    assert!((relationship_confidence_from_evidence(3) - 0.75).abs() < 1e-9);
    assert!(
        relationship_confidence_from_evidence(5) > relationship_confidence_from_evidence(1),
        "more supporting links => higher confidence"
    );
    assert!(
        relationship_confidence_from_evidence(100) < 1.0,
        "stays below 1.0"
    );
}

#[test]
fn remember_graph_capture_uses_one_memory_as_atomic_routing_evidence() {
    let path = temp_store_path("remember_graph_capture");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let graph = employment_graph(
        "person:steve",
        "Steve",
        &["Stephen", "I"],
        "org:memkeeper",
        "1",
    );
    let mut request = basic_request("Steve works at Memkeeper.");
    request.graph = Some(graph.clone());

    let report = remember_memory(&path, &request).expect("graph capture succeeds");
    assert_eq!(
        report.graph_capture.as_ref().map(|status| (
            status.routing_contract,
            status.entities,
            status.relationships
        )),
        Some(("evidence_join_v2", 2, 1))
    );

    let stephen = search_entities(
        &path,
        &EntitySearchRequest {
            query: Some("Stephen".to_string()),
            ..entity_search_defaults()
        },
    )
    .expect("alias search succeeds");
    assert_eq!(stephen.results.len(), 1);
    assert_eq!(stephen.results[0].entity.entity_key, "person:steve");

    let connection = Connection::open(&path).expect("open store");
    let (memory_id, metadata_json): (String, String) = connection
        .query_row(
            "SELECT memory_id, metadata_json FROM relationships",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .expect("relationship evidence");
    assert_eq!(memory_id, report.memory.id);
    let metadata: serde_json::Value =
        serde_json::from_str(&metadata_json).expect("valid relationship metadata");
    assert_eq!(metadata["routing_contract"], "evidence_join_v2");
    assert_eq!(metadata["routing_contract_version"], 2);
    assert!(metadata.get("object_memory_id").is_none());
    drop(connection);

    let pool = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &PackRequest {
            title: "alias route".to_string(),
            queries: vec!["Where is Stephen employed?".to_string()],
            filters: SearchFilters::default(),
            max_memories: 5,
            max_chars: 4_000,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: None,
            token_model_id: None,
            maxsim_shortlist: 0,
        },
        5,
        EvidenceJoinOptions::default(),
    )
    .expect("alias-seeded evidence join");
    let candidate = pool
        .candidates
        .iter()
        .find(|candidate| candidate.memory_id == report.memory.id)
        .expect("single evidence memory recovered");
    assert!(candidate.admissions.iter().any(|admission| {
        admission
            .graph_route
            .as_ref()
            .is_some_and(|route| route.matched_query_span.as_deref() == Some("stephen"))
    }));

    let mut dry_run = basic_request("Steve founded Memkeeper.");
    dry_run.dry_run = true;
    dry_run.graph = Some(GraphCapture {
        relationships: vec![CapturedRelationship {
            relation_type: "founded".to_string(),
            ..graph.relationships[0].clone()
        }],
        ..graph
    });
    let dry_report = remember_memory(&path, &dry_run).expect("dry-run graph validates");
    assert!(dry_report.dry_run);
    let connection = Connection::open(&path).expect("open store");
    let memory_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
        .expect("memory count");
    let relationship_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM relationships", [], |row| row.get(0))
        .expect("relationship count");
    assert_eq!(memory_count, 1);
    assert_eq!(relationship_count, 1);
    drop(connection);

    cleanup_store(&path);
}

#[test]
fn remember_graph_capture_reuses_entities_found_by_exact_alias() {
    let path = temp_store_path("remember_graph_alias_reuse");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut initial = basic_request("Steve works at Memkeeper.");
    initial.graph = Some(employment_graph(
        "person:steve",
        "Steve",
        &["Stephen", "I"],
        "org:memkeeper",
        "1",
    ));
    remember_memory(&path, &initial).expect("initial graph capture succeeds");

    let mut alias_reuse = basic_request("Stephen remains employed by Memkeeper.");
    alias_reuse.graph = Some(employment_graph(
        "person:stephen",
        "Stephen",
        &["Steve"],
        "org:memkeeper-proposed",
        "2",
    ));
    remember_memory(&path, &alias_reuse).expect("exact aliases reuse canonical entities");
    let connection = Connection::open(&path).expect("open store");
    let entity_count: i64 = connection
        .query_row("SELECT COUNT(*) FROM entities", [], |row| row.get(0))
        .expect("entity count");
    let distinct_subjects: i64 = connection
        .query_row(
            "SELECT COUNT(DISTINCT subject_entity_id) FROM relationships",
            [],
            |row| row.get(0),
        )
        .expect("subject count");
    assert_eq!(
        entity_count, 2,
        "alias reuse must not create parallel nodes"
    );
    assert_eq!(distinct_subjects, 1);
    drop(connection);

    cleanup_store(&path);
}

#[test]
fn remember_graph_capture_prefers_exact_entity_key_over_ambiguous_name() {
    let path = temp_store_path("remember_graph_exact_key_precedence");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    for (entity_key, canonical_name) in [
        ("system:memkeeper", "Memkeeper"),
        ("project:memkeeper", "Memkeeper"),
        ("org:workspace", "Workspace"),
    ] {
        upsert_entity(&path, &entity_upsert_request(entity_key, canonical_name))
            .expect("entity upsert succeeds");
    }

    let mut request = basic_request("Memkeeper is part of the Workspace.");
    request.graph = Some(GraphCapture {
        extractor: "test-extractor".to_string(),
        extractor_version: Some("1".to_string()),
        entities: vec![
            CapturedEntity {
                entity_key: "system:memkeeper".to_string(),
                entity_type: "system".to_string(),
                canonical_name: "Memkeeper".to_string(),
                aliases: Vec::new(),
            },
            CapturedEntity {
                entity_key: "org:workspace".to_string(),
                entity_type: "organization".to_string(),
                canonical_name: "Workspace".to_string(),
                aliases: Vec::new(),
            },
        ],
        relationships: vec![CapturedRelationship {
            subject_entity_key: "system:memkeeper".to_string(),
            relation_type: "part_of".to_string(),
            object_entity_key: "org:workspace".to_string(),
            confidence: 0.98,
        }],
    });

    remember_memory(&path, &request).expect("exact entity key resolves");

    let connection = Connection::open(&path).expect("open store");
    let subject_key: String = connection
        .query_row(
            "SELECT subject.entity_key
               FROM relationships relationship
               JOIN entities subject ON subject.id = relationship.subject_entity_id",
            [],
            |row| row.get(0),
        )
        .expect("relationship subject");
    assert_eq!(subject_key, "system:memkeeper");
    drop(connection);

    let mut ambiguous = basic_request("Memkeeper is still part of the Workspace.");
    ambiguous.dry_run = true;
    ambiguous.graph = Some(GraphCapture {
        extractor: "test-extractor".to_string(),
        extractor_version: Some("1".to_string()),
        entities: vec![
            CapturedEntity {
                entity_key: "proposed:memkeeper".to_string(),
                entity_type: "system".to_string(),
                canonical_name: "Memkeeper".to_string(),
                aliases: Vec::new(),
            },
            CapturedEntity {
                entity_key: "org:workspace".to_string(),
                entity_type: "organization".to_string(),
                canonical_name: "Workspace".to_string(),
                aliases: Vec::new(),
            },
        ],
        relationships: vec![CapturedRelationship {
            subject_entity_key: "proposed:memkeeper".to_string(),
            relation_type: "part_of".to_string(),
            object_entity_key: "org:workspace".to_string(),
            confidence: 0.98,
        }],
    });
    let error = remember_memory(&path, &ambiguous).expect_err("ambiguous fallback must abstain");
    assert!(matches!(
        error,
        Error::InvalidRequest { message }
            if message == "graph entity proposed:memkeeper matches multiple canonical entities"
    ));

    cleanup_store(&path);
}
