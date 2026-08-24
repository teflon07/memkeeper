//! Tests for dream operations.

use super::*;

#[test]
fn dream_link_bridges_tag_sharing_memories_into_the_graph() {
    let path = temp_store_path("dream_link_tag_bridge");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    for (key, name) in [("ent:a", "A"), ("ent:b", "B"), ("ent:c", "C")] {
        upsert_entity(&path, &entity_upsert_request(key, name)).expect("entity");
    }
    let tagged = |content: &str, ekey: &str, tags: &[&str]| RememberRequest {
        entity_key: Some(ekey.to_string()),
        tags: tags.iter().map(|t| (*t).to_string()).collect(),
        ..remember_request(content)
    };
    // a and b share two discriminative topic tags across entities.
    remember_memory(
        &path,
        &tagged("alpha note", "ent:a", &["zephyr-topic", "cobalt-topic"]),
    )
    .expect("a");
    remember_memory(
        &path,
        &tagged("beta note", "ent:b", &["zephyr-topic", "cobalt-topic"]),
    )
    .expect("b");
    // c shares only one tag with a/b, below the unattended-link evidence floor.
    remember_memory(&path, &tagged("gamma note", "ent:c", &["zephyr-topic"])).expect("c");

    let report = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["link".to_string(), "graph".to_string()],
            max_memories: 100,
            dry_run: false,
            ..dream_request_defaults()
        },
    )
    .expect("link+graph dream");
    assert!(report.link.attempted, "link task ran");
    assert_eq!(
        report.link.links_written, 1,
        "one cross-entity shared-tag link written"
    );

    let connection = Connection::open(&path).expect("open store");
    // The graph projection bridged ent:a <-> ent:b (relationship in either direction).
    let bridged: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM relationships r \
             JOIN entities s ON s.id = r.subject_entity_id \
             JOIN entities o ON o.id = r.object_entity_id \
             WHERE r.status='active' \
               AND ((s.entity_key='ent:a' AND o.entity_key='ent:b') \
                 OR (s.entity_key='ent:b' AND o.entity_key='ent:a'))",
            [],
            |row| row.get(0),
        )
        .expect("query");
    assert_eq!(
        bridged, 1,
        "tag-sharing memories bridged their entities in the graph"
    );
    // ent:c (no shared tag) is not bridged to anything.
    let c_edges: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM relationships r \
             JOIN entities s ON s.id = r.subject_entity_id \
             JOIN entities o ON o.id = r.object_entity_id \
             WHERE r.status='active' AND (s.entity_key='ent:c' OR o.entity_key='ent:c')",
            [],
            |row| row.get(0),
        )
        .expect("query c");
    assert_eq!(c_edges, 0, "a memory sharing no tag stays unlinked");
    drop(connection);

    // Dry run writes nothing but still reports candidates.
    let dry = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["link".to_string()],
            max_memories: 100,
            dry_run: true,
            ..dream_request_defaults()
        },
    )
    .expect("dry link dream");
    assert_eq!(dry.link.links_written, 0, "dry run writes no links");

    cleanup_store(&path);
}

#[test]
fn dream_link_filters_noise_and_resumes_bounded_batches() {
    let path = temp_store_path("dream_link_resumable_batches");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    for key in ["ent:a", "ent:b", "ent:c", "ent:d", "ent:e"] {
        upsert_entity(&path, &entity_upsert_request(key, key)).expect("entity");
    }
    let tagged = |content: &str, ekey: &str, tags: &[&str]| RememberRequest {
        entity_key: Some(ekey.to_string()),
        tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
        ..remember_request(content)
    };
    for (content, entity) in [("alpha", "ent:a"), ("beta", "ent:b"), ("gamma", "ent:c")] {
        remember_memory(
            &path,
            &tagged(content, entity, &["zephyr-topic", "cobalt-topic"]),
        )
        .expect("topic memory");
    }
    // After structural tags are removed, d/e share only one eligible topic tag.
    for (content, entity) in [("delta", "ent:d"), ("epsilon", "ent:e")] {
        remember_memory(
            &path,
            &tagged(
                content,
                entity,
                &["single-topic", "session:test-session", "decision", "status"],
            ),
        )
        .expect("noise memory");
    }

    let first = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["link".to_string()],
            max_memories: 2,
            dry_run: false,
            ..dream_request_defaults()
        },
    )
    .expect("first bounded link run");
    assert_eq!(first.link.candidates, 2);
    assert_eq!(first.link.links_written, 2);
    assert!(first.link.truncated);
    assert_dream_link_evidence(&path, 2);

    let second = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["link".to_string()],
            max_memories: 2,
            dry_run: false,
            ..dream_request_defaults()
        },
    )
    .expect("second bounded link run");
    assert_eq!(second.link.candidates, 1);
    assert_eq!(second.link.links_written, 1);
    assert!(!second.link.truncated);

    let drained = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["link".to_string()],
            max_memories: 2,
            dry_run: true,
            ..dream_request_defaults()
        },
    )
    .expect("drained link probe");
    assert_eq!(drained.link.candidates, 0);
    assert_eq!(drained.link.links_written, 0);
    assert!(!drained.link.truncated);

    let connection = Connection::open(&path).expect("open store");
    let total: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM memory_links WHERE link_type='related_to'",
            [],
            |row| row.get(0),
        )
        .expect("count links");
    assert_eq!(total, 3, "only the three a/b/c topic pairs are linked");
    drop(connection);
    cleanup_store(&path);
}

#[test]
fn dream_link_respects_silo_scope_for_both_endpoints() {
    let path = temp_store_path("dream_link_silo_scope");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let tagged = |content: &str, entity: &str, silo: &str| RememberRequest {
        entity_key: Some(entity.to_string()),
        silo: Some(silo.to_string()),
        tags: vec!["cobalt-topic".to_string(), "zephyr-topic".to_string()],
        ..remember_request(content)
    };
    for (content, entity) in [
        ("durable alpha", "ent:durable-a"),
        ("durable beta", "ent:durable-b"),
    ] {
        remember_memory(&path, &tagged(content, entity, "durable")).expect("durable memory");
    }
    for (content, entity) in [
        ("short alpha", "ent:short-a"),
        ("short beta", "ent:short-b"),
    ] {
        remember_memory(&path, &tagged(content, entity, "short-term")).expect("short-term memory");
    }

    let report = dream_store(
        &path,
        &DreamRequest {
            space: Some(DEFAULT_SPACE.to_string()),
            silos: vec!["short-term".to_string()],
            tasks: vec!["link".to_string()],
            max_memories: 100,
            dry_run: false,
            ..dream_request_defaults()
        },
    )
    .expect("silo-scoped link dream");
    assert_eq!(report.link.candidates, 1);
    assert_eq!(report.link.links_written, 1);

    let connection = Connection::open(&path).expect("open store");
    let linked_silos: Vec<(String, String)> = connection
        .prepare(
            "SELECT src.silo_name, dst.silo_name
               FROM memory_links link
               JOIN memories src ON src.id = link.src_memory_id
               JOIN memories dst ON dst.id = link.dst_memory_id
              WHERE link.link_type = 'related_to'",
        )
        .expect("prepare linked silos")
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .expect("query linked silos")
        .collect::<std::result::Result<Vec<_>, _>>()
        .expect("collect linked silos");
    assert_eq!(
        linked_silos,
        vec![("short-term".to_string(), "short-term".to_string())]
    );
    drop(connection);
    cleanup_store(&path);
}

#[test]
fn dream_link_discriminative_tag_frequency_is_space_local() {
    let path = temp_store_path("dream_link_space_local_frequency");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    create_space(
        &path,
        &SpaceCreateRequest {
            name: "other-space".to_string(),
            display_name: None,
            description: None,
            default_silo: None,
            ontology: None,
            config_json: None,
            if_not_exists: false,
        },
    )
    .expect("create other space");
    let tagged = |content: String, entity: String, space: Option<&str>| RememberRequest {
        space: space.map(str::to_string),
        entity_key: Some(entity),
        tags: vec!["cobalt-topic".to_string(), "zephyr-topic".to_string()],
        ..remember_request(&content)
    };
    for (content, entity) in [
        ("target alpha", "ent:target-a"),
        ("target beta", "ent:target-b"),
    ] {
        remember_memory(
            &path,
            &tagged(content.to_string(), entity.to_string(), None),
        )
        .expect("target memory");
    }
    // With the two target memories, these 24 rows push a global tag count over
    // the discriminative threshold of 25. They must not suppress another space.
    for index in 0..24 {
        remember_memory(
            &path,
            &tagged(
                format!("other space memory {index}"),
                format!("ent:other-{index}"),
                Some("other-space"),
            ),
        )
        .expect("other-space memory");
    }

    let report = dream_store(
        &path,
        &DreamRequest {
            space: Some(DEFAULT_SPACE.to_string()),
            tasks: vec!["link".to_string()],
            max_memories: 100,
            dry_run: true,
            ..dream_request_defaults()
        },
    )
    .expect("space-scoped link dream");
    assert_eq!(report.link.candidates, 1);
    assert_eq!(report.link.links_written, 0);
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
fn dream_graph_reports_and_repairs_missing_entity_projections() {
    let path = temp_store_path("dream_graph_repairs_missing_entity_projections");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let remembered = remember_memory(
        &path,
        &RememberRequest {
            entity_key: Some("entity:projection-drift".to_string()),
            claim_key: Some("claim:projection-drift".to_string()),
            ..remember_request("fact: graph projection drift")
        },
    )
    .expect("remember succeeds");

    let connection = Connection::open(&path).expect("open store");
    connection
        .execute(
            "DELETE FROM entities WHERE space_name = ?1 AND entity_key = ?2",
            params![remembered.memory.space, "entity:projection-drift"],
        )
        .expect("delete entity projection");
    drop(connection);

    let dry_run = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["graph".to_string()],
            max_memories: 10,
            dry_run: true,
            ..dream_request_defaults()
        },
    )
    .expect("graph dry-run succeeds");
    assert_eq!(dry_run.graph.missing_entity_projections.len(), 1);
    assert_eq!(
        dry_run.graph.missing_entity_projections[0].entity_key,
        "entity:projection-drift"
    );
    assert!(
        search_entities(
            &path,
            &EntitySearchRequest {
                entity_key: Some("entity:projection-drift".to_string()),
                limit: 10,
                ..entity_search_defaults()
            },
        )
        .expect("search after dry-run")
        .results
        .is_empty(),
        "dry-run must not recreate the entity projection"
    );

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
    assert_eq!(applied.graph.missing_entity_projections.len(), 1);
    assert_eq!(
        search_entities(
            &path,
            &EntitySearchRequest {
                entity_key: Some("entity:projection-drift".to_string()),
                limit: 10,
                ..entity_search_defaults()
            },
        )
        .expect("search after apply")
        .results
        .len(),
        1
    );

    let idempotent = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["graph".to_string()],
            max_memories: 10,
            dry_run: false,
            ..dream_request_defaults()
        },
    )
    .expect("second graph apply succeeds");
    assert!(idempotent.graph.missing_entity_projections.is_empty());

    cleanup_store(&path);
}

#[test]
fn dream_graph_preserves_intentionally_merged_entity_status() {
    let path = temp_store_path("dream_graph_preserves_merged_entity_status");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    remember_memory(
        &path,
        &RememberRequest {
            entity_key: Some("file:variant".to_string()),
            claim_key: Some("claim:variant".to_string()),
            ..remember_request("fact: variant memory remains historical evidence")
        },
    )
    .expect("variant memory succeeds");
    remember_memory(
        &path,
        &RememberRequest {
            entity_key: Some("file:canon".to_string()),
            claim_key: Some("claim:canon".to_string()),
            ..remember_request("fact: canonical memory backs the merged entity")
        },
    )
    .expect("canonical memory succeeds");
    merge_entity(
        &path,
        &EntityMergeRequest {
            from_entity_key: Some("file:variant".to_string()),
            into_entity_key: Some("file:canon".to_string()),
            ..merge_request_defaults()
        },
    )
    .expect("merge succeeds");

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
    assert!(report.graph.missing_entity_projections.is_empty());
    let connection = Connection::open(&path).expect("open store");
    assert_eq!(entity_status(&connection, "file:variant"), "tombstoned");
    assert_eq!(entity_status(&connection, "file:canon"), "active");
    drop(connection);
    let health = store_stats_with_health(&path, false)
        .expect("health stats")
        .health
        .expect("health present");
    assert_eq!(health.active_missing_entity_projection, 0);

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
    connection
        .execute(
            "DELETE FROM entities WHERE space_name = ?1 AND entity_key = 'entity:object'",
            params![object_memory.memory.space],
        )
        .expect("delete object projection");
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
    assert_eq!(report.graph.missing_entity_projections.len(), 1);
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
fn dream_graph_bounds_relationship_proposals_to_staged_projection_repairs() {
    let path = temp_store_path("dream_graph_bounds_projection_repairs");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let mut memory_ids = Vec::new();
    for key in ["entity:a", "entity:b", "entity:c"] {
        let remembered = remember_memory(
            &path,
            &RememberRequest {
                entity_key: Some(key.to_string()),
                claim_key: Some(format!("claim:{key}")),
                ..remember_request(&format!("fact: bounded projection {key}"))
            },
        )
        .expect("remember succeeds");
        memory_ids.push(remembered.memory.id);
    }
    let connection = Connection::open(&path).expect("open store");
    connection
        .execute(
            "INSERT INTO memory_links (
                src_memory_id, dst_memory_id, link_type, status, confidence, created_at
             ) VALUES (?1, ?2, 'supports', 'active', 1.0, CURRENT_TIMESTAMP)",
            params![memory_ids[1], memory_ids[2]],
        )
        .expect("insert link outside first repair batch");
    connection
        .execute("DELETE FROM entities", [])
        .expect("delete projections");
    drop(connection);

    let report = dream_store(
        &path,
        &DreamRequest {
            tasks: vec!["graph".to_string()],
            max_memories: 1,
            dry_run: false,
            ..dream_request_defaults()
        },
    )
    .expect("bounded graph apply succeeds");
    assert!(report.graph.truncated);
    assert_eq!(report.graph.missing_entity_projections.len(), 1);
    assert_eq!(
        report.graph.missing_entity_projections[0].entity_key,
        "entity:a"
    );
    assert!(report.graph.relationship_proposals.is_empty());
    assert_eq!(
        search_entities(
            &path,
            &EntitySearchRequest {
                limit: 10,
                ..entity_search_defaults()
            },
        )
        .expect("search repaired projections")
        .results
        .len(),
        1
    );

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
                batch_id: None,
                latency_ms: None,
                latency_source: None,
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
            batch_id: None,
            latency_ms: None,
            latency_source: None,
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
fn record_recall_stores_batch_latency_metadata() {
    let path = temp_store_path("record_recall_stores_batch_latency_metadata");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let m =
        remember_memory(&path, &basic_request("fact: latency-tagged recall")).expect("remember");
    record_recall(
        &path,
        &RecallLogRequest {
            source: Some("test".to_string()),
            session_id: Some("sess-latency".to_string()),
            batch_id: Some("batch-123".to_string()),
            latency_ms: Some(12.5),
            latency_source: Some("unit-test".to_string()),
            events: vec![retrieved_event(&m.memory.id)],
            touch_accessed: true,
        },
    )
    .expect("record recall");

    let conn = Connection::open(&path).expect("open");
    let (batch_id, latency_ms, latency_source): (Option<String>, Option<f64>, Option<String>) =
        conn.query_row(
            "SELECT batch_id, latency_ms, latency_source FROM recall_events WHERE memory_id = ?1",
            params![&m.memory.id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("query latency metadata");
    assert_eq!(batch_id.as_deref(), Some("batch-123"));
    assert_eq!(latency_ms, Some(12.5));
    assert_eq!(latency_source.as_deref(), Some("unit-test"));

    let invalid = record_recall(
        &path,
        &RecallLogRequest {
            source: Some("test".to_string()),
            session_id: None,
            batch_id: None,
            latency_ms: Some(-1.0),
            latency_source: Some("unit-test".to_string()),
            events: vec![retrieved_event(&m.memory.id)],
            touch_accessed: false,
        },
    );
    assert!(matches!(invalid, Err(Error::InvalidRequest { .. })));
    cleanup_store(&path);
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
        batch_id: None,
        latency_ms: None,
        latency_source: None,
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
