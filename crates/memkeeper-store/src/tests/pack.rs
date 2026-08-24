//! Tests for pack operations.

use super::*;

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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
        report.content.contains("- [Observed at:"),
        "the truncated entry keeps its leading source-time marker"
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
            maxsim_shortlist: 0,
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
        maxsim_shortlist: 0,
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
fn pack_pool_retains_query_variant_overlap_for_admitted_candidate() {
    use crate::build_pack_pool;
    let path = temp_store_path("pack_pool_query_overlap");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    let memory =
        remember_memory(&path, &basic_request("alpha overlap memory")).expect("remember succeeds");
    let request = PackRequest {
        title: "query overlap".to_string(),
        queries: vec!["alpha overlap".to_string(), "overlap memory".to_string()],
        filters: SearchFilters::default(),
        max_memories: 1,
        max_chars: 1_000,
        format: "markdown".to_string(),
        min_score: 0.0,
        rerank_candidates: 0,
        query_embeddings: None,
        query_token_embeddings: None,
        token_model_id: None,
        maxsim_shortlist: 0,
    };

    let pool = build_pack_pool(&path, &request).expect("pool builds");
    let candidate = pool
        .iter()
        .find(|candidate| candidate.memory_id == memory.memory.id)
        .expect("candidate admitted");
    assert_eq!(
        candidate
            .admissions
            .iter()
            .map(|observation| observation.query_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    cleanup_store(&path);
}

#[test]
fn evidence_graph_join_entity_span_and_seed_bounds_are_deterministic() {
    let query = (0..40)
        .map(|index| format!("token{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    let spans = crate::evidence_query_spans(&[query]);
    assert_eq!(spans.len(), crate::MAX_EVIDENCE_ENTITY_SPANS);
    assert_eq!(spans[0].normalized, "token0 token1 token2 token3 token4");
    assert_eq!(spans[0].end - spans[0].start, 5);
    assert!(
        crate::evidence_query_spans(&["where who Steve".to_string()])
            .iter()
            .all(|span| span.normalized != "where" && span.normalized != "who")
    );

    let memory_seed = crate::EvidenceGraphSeed {
        source: crate::GraphSeedSource::Memory,
        entity_id: "memory-entity".to_string(),
        space_name: DEFAULT_SPACE.to_string(),
        score: 0.8,
        memory_id: Some("memory-seed".to_string()),
        matched_query_index: None,
        matched_query_span: None,
    };
    let entity_seed = |index: usize| crate::EvidenceGraphSeed {
        source: crate::GraphSeedSource::Entity,
        entity_id: format!("entity-{index}"),
        space_name: DEFAULT_SPACE.to_string(),
        score: 1.0,
        memory_id: None,
        matched_query_index: Some(0),
        matched_query_span: Some(format!("entity {index}")),
    };
    let entity_seeds = [entity_seed(1), entity_seed(2), entity_seed(3)];
    let allocated = crate::allocate_evidence_seeds(&[memory_seed], &entity_seeds, 2);
    assert_eq!(allocated.len(), 2);
    assert_eq!(allocated[0].source, crate::GraphSeedSource::Memory);
    assert_eq!(allocated[1].source, crate::GraphSeedSource::Entity);

    let path = temp_store_path("evidence_entity_seed_bound");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    for (key, name) in [
        ("entity:aster", "Aster"),
        ("entity:beryl", "Beryl"),
        ("entity:cedar", "Cedar"),
    ] {
        upsert_entity(&path, &entity_upsert_request(key, name)).expect("entity upsert");
    }
    let request = PackRequest {
        title: "seed bound".to_string(),
        queries: vec!["Aster Beryl Cedar".to_string()],
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
    };
    let filters = crate::evidence_join_filters(&request).expect("filters");
    let connection = Connection::open(&path).expect("open store");
    let entity_seeds =
        crate::evidence_entity_seeds(&connection, &request, &filters).expect("entity seeds");
    assert_eq!(entity_seeds.len(), 3);
    cleanup_store(&path);
}

#[test]
fn evidence_graph_join_candidate_allocation_uses_evidence_strength() {
    let route = |source, depth, relationship: &str| crate::GraphRouteObservation {
        seed_source: source,
        seed_memory_id: (source == crate::GraphSeedSource::Memory)
            .then(|| "memory-seed".to_string()),
        seed_entity_id: format!("seed-{source:?}"),
        matched_query_index: (source == crate::GraphSeedSource::Entity).then_some(0),
        matched_query_span: (source == crate::GraphSeedSource::Entity).then(|| "steve".to_string()),
        hop_depth: depth,
        relationship_ids: vec![relationship.to_string()],
        predicate_names: vec!["routes_to".to_string()],
        traversal_directions: vec!["forward".to_string()],
        evidence_class: crate::GraphEvidenceClass::EndpointSupport,
        route_outcome: "active".to_string(),
    };
    let mut both_sources = crate::EvidenceCandidateRoutes::new();
    for (memory_id, source, activation) in [
        ("memory-best", crate::GraphSeedSource::Memory, 0.9),
        ("memory-next", crate::GraphSeedSource::Memory, 0.8),
        ("entity-best", crate::GraphSeedSource::Entity, 0.2),
    ] {
        crate::record_evidence_candidate(
            &mut both_sources,
            memory_id,
            activation,
            route(source, 1, memory_id),
        );
    }
    let selected = crate::evidence_candidate_order(&both_sources);
    assert_eq!(&selected[..2], ["memory-best", "memory-next"]);

    let mut depth_pressure = crate::EvidenceCandidateRoutes::new();
    for index in 0..8 {
        let memory_id = format!("depth-one-{index}");
        crate::record_evidence_candidate(
            &mut depth_pressure,
            &memory_id,
            1.0 - f64::from(index) / 100.0,
            route(crate::GraphSeedSource::Memory, 1, &memory_id),
        );
    }
    crate::record_evidence_candidate(
        &mut depth_pressure,
        "depth-two",
        0.1,
        route(crate::GraphSeedSource::Memory, 2, "depth-two"),
    );
    let selected = crate::evidence_candidate_order(&depth_pressure);
    assert_eq!(selected[0], "depth-one-0");
    assert!(selected.iter().any(|memory_id| memory_id == "depth-two"));

    crate::record_evidence_candidate(
        &mut depth_pressure,
        "parallel",
        0.2,
        route(crate::GraphSeedSource::Memory, 2, "weaker-path"),
    );
    crate::record_evidence_candidate(
        &mut depth_pressure,
        "parallel",
        0.7,
        route(crate::GraphSeedSource::Memory, 1, "better-path"),
    );
    let parallel = &depth_pressure["parallel"];
    assert_eq!(parallel.len(), 1);
    assert_eq!(
        parallel
            .values()
            .next()
            .expect("one route from the shared memory seed")
            .route
            .relationship_ids,
        vec!["better-path"]
    );
}

#[test]
fn evidence_graph_join_preserves_distinct_memory_seed_routes_for_depth_allocation() {
    let route = |seed_memory_id: &str, depth, relationship: &str| crate::GraphRouteObservation {
        seed_source: crate::GraphSeedSource::Memory,
        seed_memory_id: Some(seed_memory_id.to_string()),
        seed_entity_id: format!("entity-{seed_memory_id}"),
        matched_query_index: None,
        matched_query_span: None,
        hop_depth: depth,
        relationship_ids: vec![relationship.to_string()],
        predicate_names: vec!["routes_to".to_string()],
        traversal_directions: vec!["forward".to_string()],
        evidence_class: crate::GraphEvidenceClass::EndpointSupport,
        route_outcome: "active".to_string(),
    };
    let mut candidates = crate::EvidenceCandidateRoutes::new();
    crate::record_evidence_candidate(
        &mut candidates,
        "gold",
        0.8,
        route("strongest-seed", 2, "strongest-two-hop"),
    );
    crate::record_evidence_candidate(
        &mut candidates,
        "gold",
        0.9,
        route("middle-seed", 1, "middle-one-hop"),
    );
    crate::record_evidence_candidate(
        &mut candidates,
        "unrelated",
        0.7,
        route("unrelated-seed", 2, "unrelated-two-hop"),
    );

    assert_eq!(
        candidates["gold"].len(),
        2,
        "routes from distinct semantic seeds must remain independently visible"
    );
    let selected = crate::evidence_candidate_order(&candidates);
    assert_eq!(
        selected[0], "gold",
        "the candidate's best one-hop route must determine its priority"
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn evidence_graph_join_exact_entity_seed_recovers_endpoint_support() {
    let path = temp_store_path("evidence_graph_join_exact_entity_seed");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut steve_entity = entity_upsert_request("person:steve", "Steve");
    steve_entity.aliases = vec!["Stephen".to_string(), "I".to_string()];
    upsert_entity(&path, &steve_entity).expect("Steve entity with aliases");
    upsert_entity(&path, &entity_upsert_request("org:acme", "Acme Labs")).expect("Acme entity");

    let mut subject = basic_request("Profile record alpha.");
    subject.entity_key = Some("person:steve".to_string());
    let subject_id = remember_memory(&path, &subject)
        .expect("subject support")
        .memory
        .id;
    let mut object = basic_request("Acme Labs employs its principal architect.");
    object.entity_key = Some("org:acme".to_string());
    let object_id = remember_memory(&path, &object)
        .expect("object support")
        .memory
        .id;

    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("person:steve".to_string()),
            relation_type: "works_at".to_string(),
            object_entity_key: Some("org:acme".to_string()),
            memory_id: Some(subject_id),
            metadata_json: Some(routing_metadata()),
            confidence: 1.0,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("routing relationship");

    let request = PackRequest {
        title: "entity route".to_string(),
        queries: vec!["Where does Steve work?".to_string()],
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
    };
    let off = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &request,
        10,
        EvidenceJoinOptions {
            max_graph_neighbors: 0,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect("graph off");
    assert!(!off
        .candidates
        .iter()
        .any(|candidate| candidate.memory_id == object_id));

    let on = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &request,
        10,
        EvidenceJoinOptions {
            max_graph_seeds: 4,
            max_graph_neighbors: 4,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect("evidence join");
    let target = on
        .candidates
        .iter()
        .find(|candidate| candidate.memory_id == object_id)
        .expect("exact endpoint support recovered");
    assert_eq!(on.graph_outcome.as_deref(), Some("active"));
    let route = target
        .admissions
        .iter()
        .filter_map(|admission| admission.graph_route.as_ref())
        .find(|route| route.seed_source == crate::GraphSeedSource::Entity)
        .expect("entity graph route observation");
    assert_eq!(route.seed_source, crate::GraphSeedSource::Entity);
    assert_eq!(route.hop_depth, 1);
    assert_eq!(route.matched_query_span.as_deref(), Some("steve"));

    let mut unaligned_request = request.clone();
    unaligned_request.queries = vec!["Where does Steve live?".to_string()];
    let unaligned_pool = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &unaligned_request,
        10,
        EvidenceJoinOptions {
            max_graph_seeds: 4,
            max_graph_neighbors: 4,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect("unaligned evidence join");
    let unaligned = unaligned_pool
        .candidates
        .iter()
        .find(|candidate| candidate.memory_id == object_id)
        .expect("unaligned endpoint remains available to semantic reranking");
    assert!(
        unaligned
            .admissions
            .iter()
            .any(|admission| admission.source == crate::AdmissionSource::Graph),
        "graph evidence remains in the unified pool for the cross-encoder to judge"
    );

    for (query, expected_span) in [
        ("Where does Stephen work?", "stephen"),
        ("Where do I work?", "i"),
    ] {
        let mut alias_request = request.clone();
        alias_request.queries = vec![query.to_string()];
        let alias_pool = build_hybrid_rerank_pool_with_evidence_options(
            &path,
            &alias_request,
            10,
            EvidenceJoinOptions {
                max_graph_seeds: 4,
                max_graph_neighbors: 4,
                ..EvidenceJoinOptions::default()
            },
        )
        .expect("alias evidence join");
        let alias_route = alias_pool
            .candidates
            .iter()
            .find(|candidate| candidate.memory_id == object_id)
            .and_then(|candidate| {
                candidate
                    .admissions
                    .iter()
                    .filter_map(|admission| admission.graph_route.as_ref())
                    .find(|route| route.seed_source == crate::GraphSeedSource::Entity)
            })
            .expect("alias resolves to the same entity route");
        assert_eq!(
            alias_route.seed_entity_id, route.seed_entity_id,
            "aliases must not create duplicate entities"
        );
        assert_eq!(
            alias_route.matched_query_span.as_deref(),
            Some(expected_span)
        );
    }
    let connection = Connection::open(&path).expect("open store");
    let steve_entities: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM entities WHERE entity_key = 'person:steve'",
            [],
            |row| row.get(0),
        )
        .expect("entity count");
    assert_eq!(steve_entities, 1);

    cleanup_store(&path);
}

#[test]
#[allow(clippy::too_many_lines)]
fn evidence_graph_join_semantic_seed_recovers_two_hop_endpoint_support() {
    let path = temp_store_path("evidence_graph_join_semantic_seed_two_hop");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    for (key, name) in [
        ("node:alpha", "Alpha Node"),
        ("node:beta", "Beta Node"),
        ("node:gamma", "Gamma Node"),
    ] {
        upsert_entity(&path, &entity_upsert_request(key, name)).expect("entity upsert");
    }

    let mut alpha = basic_request("orchard deployment anchor");
    // The canonical source memory is not owned by the captured route entity.
    // Its relationship support id must still provide the graph join foothold.
    alpha.entity_key = Some("conversation:canonical".to_string());
    let alpha_id = remember_memory(&path, &alpha)
        .expect("alpha support")
        .memory
        .id;
    let mut beta = basic_request("intermediate cobalt bridge");
    beta.entity_key = Some("node:beta".to_string());
    let beta_id = remember_memory(&path, &beta)
        .expect("beta support")
        .memory
        .id;
    let mut gamma = basic_request("terminal capybara evidence");
    gamma.entity_key = Some("node:gamma".to_string());
    let gamma_id = remember_memory(&path, &gamma)
        .expect("gamma support")
        .memory
        .id;

    for (subject, relation, object, memory_id) in [
        ("node:alpha", "routes_to", "node:beta", &alpha_id),
        ("node:beta", "supports", "node:gamma", &beta_id),
    ] {
        upsert_relationship(
            &path,
            &RelationshipUpsertRequest {
                subject_entity_key: Some(subject.to_string()),
                relation_type: relation.to_string(),
                object_entity_key: Some(object.to_string()),
                memory_id: Some(memory_id.clone()),
                metadata_json: Some(routing_metadata()),
                confidence: 1.0,
                ..relationship_upsert_request_defaults()
            },
        )
        .expect("routing relationship");
    }

    let request = PackRequest {
        title: "semantic bridge".to_string(),
        queries: vec!["orchard deployment anchor".to_string()],
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
    };
    let on = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &request,
        1,
        EvidenceJoinOptions {
            max_graph_seeds: 2,
            max_graph_neighbors: 4,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect("evidence join");
    let target = on
        .candidates
        .iter()
        .find(|candidate| candidate.memory_id == gamma_id)
        .expect("two-hop endpoint recovered");
    let route = target
        .admissions
        .iter()
        .find_map(|admission| admission.graph_route.as_ref())
        .expect("graph route observation");
    assert_eq!(route.seed_source, crate::GraphSeedSource::Memory);
    assert_eq!(route.seed_entity_id, entity_id_for_key(&path, "node:alpha"));
    assert_eq!(route.hop_depth, 2);

    let trace = build_hybrid_rerank_pool_trace_with_evidence_options(
        &path,
        &request,
        1,
        EvidenceJoinOptions {
            max_graph_seeds: 2,
            max_graph_neighbors: 1,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect("capped evidence trace");
    let capped = trace
        .observed
        .iter()
        .find(|candidate| candidate.memory_id == gamma_id)
        .expect("two-hop endpoint remains observable after graph cap");
    assert_eq!(capped.dropped_at.as_deref(), Some("graph_cap"));
    assert!(capped
        .admissions
        .iter()
        .filter_map(|admission| admission.graph_route.as_ref())
        .any(|route| route.hop_depth == 2));

    cleanup_store(&path);
}

#[test]
fn evidence_graph_join_ambiguous_entity_span_abstains() {
    let path = temp_store_path("evidence_graph_join_ambiguous_entity_span");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    for (key, name) in [
        ("person:one", "Alex"),
        ("person:two", "Alex"),
        ("org:target", "Target Org"),
    ] {
        upsert_entity(&path, &entity_upsert_request(key, name)).expect("entity upsert");
    }
    let mut subject = basic_request("alpha profile record");
    subject.entity_key = Some("person:one".to_string());
    let subject_id = remember_memory(&path, &subject)
        .expect("subject support")
        .memory
        .id;
    let mut object = basic_request("capybara endpoint record");
    object.entity_key = Some("org:target".to_string());
    let object_id = remember_memory(&path, &object)
        .expect("object support")
        .memory
        .id;
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("person:one".to_string()),
            relation_type: "works_at".to_string(),
            object_entity_key: Some("org:target".to_string()),
            memory_id: Some(subject_id),
            metadata_json: Some(routing_metadata()),
            confidence: 1.0,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("routing relationship");

    let request = PackRequest {
        title: "ambiguous entity".to_string(),
        queries: vec!["Where does Alex work?".to_string()],
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
    };
    let off = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &request,
        1,
        EvidenceJoinOptions {
            max_graph_neighbors: 0,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect("graph off");
    let on = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &request,
        1,
        EvidenceJoinOptions {
            max_graph_seeds: 2,
            max_graph_neighbors: 2,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect("ambiguous span abstains");
    assert_eq!(on.graph_outcome.as_deref(), Some("no_eligible_seed_route"));
    assert_eq!(off.graph_outcome.as_deref(), Some("no_eligible_seed_route"));
    assert_eq!(on.cos_top.to_bits(), off.cos_top.to_bits());
    assert_eq!(on.candidates, off.candidates);
    assert_eq!(on.observed, off.observed);
    assert_eq!(on.pool_width, off.pool_width);
    assert!(!on
        .candidates
        .iter()
        .any(|candidate| candidate.memory_id == object_id));
    cleanup_store(&path);
}

#[test]
fn evidence_graph_join_entity_seed_traverses_reverse_and_ignores_unmarked_routes() {
    let path = temp_store_path("evidence_graph_join_reverse");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    for (key, name) in [
        ("person:steve", "Steve"),
        ("org:acme", "Acme Labs"),
        ("org:noise", "Noise Org"),
    ] {
        upsert_entity(&path, &entity_upsert_request(key, name)).expect("entity upsert");
    }
    let mut subject = basic_request("principal architect profile");
    subject.entity_key = Some("person:steve".to_string());
    let subject_id = remember_memory(&path, &subject)
        .expect("subject support")
        .memory
        .id;
    let mut object = basic_request("Acme Labs organization record");
    object.entity_key = Some("org:acme".to_string());
    let object_id = remember_memory(&path, &object)
        .expect("object support")
        .memory
        .id;
    let mut noise = basic_request("unmarked noise endpoint");
    noise.entity_key = Some("org:noise".to_string());
    let noise_id = remember_memory(&path, &noise)
        .expect("noise support")
        .memory
        .id;
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("person:steve".to_string()),
            relation_type: "works_at".to_string(),
            object_entity_key: Some("org:acme".to_string()),
            memory_id: Some(subject_id.clone()),
            metadata_json: Some(routing_metadata()),
            confidence: 1.0,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("routing relationship");
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("org:acme".to_string()),
            relation_type: "has_noise".to_string(),
            object_entity_key: Some("org:noise".to_string()),
            memory_id: Some(object_id),
            metadata_json: Some(serde_json::json!({"object_memory_id": noise_id}).to_string()),
            confidence: 1.0,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("unmarked relationship");

    let request = PackRequest {
        title: "reverse entity route".to_string(),
        queries: vec!["Who works at Acme Labs?".to_string()],
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
    };
    let on = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &request,
        2,
        EvidenceJoinOptions {
            max_graph_seeds: 4,
            max_graph_neighbors: 4,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect("reverse evidence join");
    let subject = on
        .candidates
        .iter()
        .find(|candidate| candidate.memory_id == subject_id)
        .expect("reverse endpoint recovered");
    let route = subject
        .admissions
        .iter()
        .filter_map(|admission| admission.graph_route.as_ref())
        .find(|route| route.seed_source == crate::GraphSeedSource::Entity)
        .expect("entity route");
    assert_eq!(route.traversal_directions, vec!["reverse"]);
    assert!(!on
        .candidates
        .iter()
        .any(|candidate| candidate.memory_id == noise_id));
    cleanup_store(&path);
}

#[test]
fn evidence_graph_join_routing_relationship_lifecycle_is_evidence_specific() {
    let path = temp_store_path("evidence_graph_join_relationship_lifecycle");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(&path, &entity_upsert_request("person:steve", "Steve")).expect("Steve entity");
    upsert_entity(&path, &entity_upsert_request("org:acme", "Acme")).expect("Acme entity");

    let remember_for = |content: &str, entity_key: &str| {
        let mut request = basic_request(content);
        request.entity_key = Some(entity_key.to_string());
        remember_memory(&path, &request)
            .expect("support memory")
            .memory
            .id
    };
    let subject_one = remember_for("Steve profile one", "person:steve");
    let subject_two = remember_for("Steve profile two", "person:steve");
    let subject_three = remember_for("Steve profile three", "person:steve");
    let relationship_request = |relation: &str, evidence_memory: &str| RelationshipUpsertRequest {
        subject_entity_key: Some("person:steve".to_string()),
        relation_type: relation.to_string(),
        object_entity_key: Some("org:acme".to_string()),
        memory_id: Some(evidence_memory.to_string()),
        metadata_json: Some(routing_metadata()),
        confidence: 1.0,
        ..relationship_upsert_request_defaults()
    };

    let first = upsert_relationship(&path, &relationship_request("works_at", &subject_one))
        .expect("first routing relationship");
    assert!(first.created);
    let replay = upsert_relationship(&path, &relationship_request("works_at", &subject_one))
        .expect("exact replay");
    assert!(!replay.created);
    assert_eq!(replay.relationship.id, first.relationship.id);

    let connection = Connection::open(&path).expect("open store");
    let metadata: String = connection
        .query_row(
            "SELECT metadata_json FROM relationships WHERE id = ?1",
            [&first.relationship.id],
            |row| row.get(0),
        )
        .expect("routing metadata");
    let metadata = serde_json::from_str::<serde_json::Value>(&metadata).expect("metadata json");
    assert_eq!(metadata["routing_contract"], "evidence_join_v2");
    assert!(metadata.get("object_memory_id").is_none());

    let new_evidence = upsert_relationship(&path, &relationship_request("works_at", &subject_two))
        .expect("new evidence support");
    assert!(new_evidence.created);
    assert_ne!(new_evidence.relationship.id, first.relationship.id);

    let unmarked = upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("person:steve".to_string()),
            relation_type: "advises".to_string(),
            object_entity_key: Some("org:acme".to_string()),
            memory_id: Some(subject_three.clone()),
            confidence: 1.0,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("unmarked relationship");
    let upgraded = upsert_relationship(&path, &relationship_request("advises", &subject_three))
        .expect("same-identity routing upgrade");
    assert!(!upgraded.created);
    assert_eq!(upgraded.relationship.id, unmarked.relationship.id);

    drop(connection);
    cleanup_store(&path);
}

#[test]
fn evidence_graph_join_rejects_structural_inactive_expired_and_filtered_routes() {
    let path = temp_store_path("evidence_graph_join_route_eligibility");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    for (key, name) in [
        ("person:steve", "Steve"),
        ("org:generic", "Generic Org"),
        ("org:inactive", "Inactive Org"),
        ("org:expired", "Expired Org"),
        ("org:filtered", "Filtered Org"),
    ] {
        upsert_entity(&path, &entity_upsert_request(key, name)).expect("entity upsert");
    }
    let remember_for = |content: &str, entity_key: &str, project: &str| {
        let mut request = basic_request(content);
        request.entity_key = Some(entity_key.to_string());
        request.project_key = Some(project.to_string());
        remember_memory(&path, &request)
            .expect("support memory")
            .memory
            .id
    };
    let target_specs = [
        ("org:generic", "generic endpoint", "eligible"),
        ("org:inactive", "inactive endpoint", "eligible"),
        ("org:expired", "expired endpoint", "eligible"),
        ("org:filtered", "filtered endpoint", "other-project"),
    ];
    let targets = target_specs
        .iter()
        .map(|(entity, content, project)| (*entity, remember_for(content, entity, project)))
        .collect::<Vec<_>>();
    for (index, (entity, target)) in targets.iter().enumerate() {
        let (relation_type, status, valid_to) = match index {
            0 => ("related_to", None, None),
            1 => ("works_at", Some("tombstoned".to_string()), None),
            2 => (
                "works_at",
                None,
                Some("2000-01-01T00:00:00.000Z".to_string()),
            ),
            _ => ("works_at", None, None),
        };
        upsert_relationship(
            &path,
            &RelationshipUpsertRequest {
                subject_entity_key: Some("person:steve".to_string()),
                relation_type: relation_type.to_string(),
                object_entity_key: Some((*entity).to_string()),
                memory_id: Some(target.clone()),
                status,
                valid_to,
                metadata_json: Some(routing_metadata()),
                confidence: 1.0,
                ..relationship_upsert_request_defaults()
            },
        )
        .expect("relationship upsert");
    }

    let request = PackRequest {
        title: "route eligibility".to_string(),
        queries: vec!["Where does Steve work?".to_string()],
        filters: SearchFilters {
            projects: vec!["eligible".to_string()],
            ..SearchFilters::default()
        },
        max_memories: 5,
        max_chars: 4_000,
        format: "markdown".to_string(),
        min_score: 0.0,
        rerank_candidates: 0,
        query_embeddings: None,
        query_token_embeddings: None,
        token_model_id: None,
        maxsim_shortlist: 0,
    };
    let pool = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &request,
        5,
        EvidenceJoinOptions {
            max_graph_seeds: 4,
            max_graph_neighbors: 4,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect("ineligible routes abstain");
    assert_eq!(
        pool.graph_outcome.as_deref(),
        Some("no_eligible_seed_route")
    );
    for (_, target) in &targets {
        assert!(!pool
            .candidates
            .iter()
            .any(|candidate| candidate.memory_id == *target));
    }
    cleanup_store(&path);
}

#[test]
fn evidence_graph_join_invalid_routing_record_fails_visibly() {
    let path = temp_store_path("evidence_graph_join_invalid_route");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    upsert_entity(&path, &entity_upsert_request("person:steve", "Steve")).expect("Steve entity");
    upsert_entity(&path, &entity_upsert_request("org:acme", "Acme")).expect("Acme entity");
    let mut subject = basic_request("Steve profile support");
    subject.entity_key = Some("person:steve".to_string());
    let subject_id = remember_memory(&path, &subject)
        .expect("subject support")
        .memory
        .id;
    upsert_relationship(
        &path,
        &RelationshipUpsertRequest {
            subject_entity_key: Some("person:steve".to_string()),
            relation_type: "works_at".to_string(),
            object_entity_key: Some("org:acme".to_string()),
            memory_id: Some(subject_id),
            metadata_json: Some(serde_json::json!({"routing": true}).to_string()),
            confidence: 1.0,
            ..relationship_upsert_request_defaults()
        },
    )
    .expect("malformed routing relationship is storable for degradation test");
    let request = PackRequest {
        title: "invalid route".to_string(),
        queries: vec!["Where does Steve work?".to_string()],
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
    };
    let error = build_hybrid_rerank_pool_with_evidence_options(
        &path,
        &request,
        5,
        EvidenceJoinOptions {
            ..EvidenceJoinOptions::default()
        },
    )
    .expect_err("invalid routing record must not silently degrade");
    assert!(error.to_string().contains("evidence_join_v2"), "{error}");
    cleanup_store(&path);
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
    let report = assemble_reranked_pack(&request, &candidates);
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
    let report = assemble_reranked_pack(&request, &candidates);
    assert_eq!(report.memory_ids, vec!["b".to_string(), "c".to_string()]);
    assert_eq!(
        report.content,
        "- [Observed at: 2026-07-19T00:00:00.000Z] high\n\
         - [Observed at: 2026-07-19T00:00:00.000Z] mid\n"
    );
    assert!(report.truncated, "3 candidates, 2 injected => truncated");
}

#[test]
fn assemble_reranked_pack_uses_activation_only_for_exact_rerank_ties() {
    let request = rerank_pack_request(3, 10_000, 0.0);
    let candidates = vec![
        rc_reachable("low_activation", "low", 0.50, 0.20),
        rc_reachable("high_activation", "high", 0.50, 0.80),
        rc("higher_rerank", "higher", 0.51),
    ];

    let report = assemble_reranked_pack(&request, &candidates);
    assert_eq!(
        report.memory_ids,
        vec!["higher_rerank", "high_activation", "low_activation"],
        "activation breaks equal rerank scores but cannot outrank a higher score"
    );
}

#[test]
fn assemble_reranked_pack_uses_consensus_only_for_exact_rerank_ties() {
    let request = rerank_pack_request(3, 10_000, 0.0);
    let mut agreed = rc_reachable("agreed", "both routes agree", 0.50, 0.40);
    agreed.consensus = true;
    let candidates = vec![
        rc("higher", "higher score", 0.51),
        rc("direct_tie", "direct only", 0.50),
        agreed,
    ];

    let report = assemble_reranked_pack(&request, &candidates);
    assert_eq!(
        report.memory_ids,
        vec!["higher", "agreed", "direct_tie"],
        "consensus breaks exact reranker ties but never overrides a higher score"
    );
    assert_eq!(
        report.scores,
        vec![
            f64::from(0.51_f32),
            f64::from(0.50_f32),
            f64::from(0.50_f32)
        ],
        "reported scores remain raw reranker output"
    );
}

#[test]
fn assemble_reranked_pack_top_score_gate_blocks_off_topic() {
    let request = rerank_pack_request(5, 10_000, 0.2);
    let candidates = vec![rc("a", "x", 0.05), rc("b", "y", 0.10)];
    // The top rerank score (0.10) is below min_score (0.2), so the whole pack abstains.
    let report = assemble_reranked_pack(&request, &candidates);
    assert!(report.memory_ids.is_empty());
    assert!(report.content.is_empty());
    assert!(!report.truncated);
}

#[test]
fn assemble_reranked_pack_top_score_gate_emits_on_confidence() {
    let request = rerank_pack_request(5, 10_000, 0.4);
    let candidates = vec![rc("a", "x", 0.45), rc("b", "y", 0.10)];
    // The top rerank score (0.45) clears min_score (0.4), so the whole ordered
    // pack remains eligible.
    let report = assemble_reranked_pack(&request, &candidates);
    assert_eq!(report.memory_ids, vec!["a".to_string(), "b".to_string()]);
}

#[test]
fn assemble_reranked_pack_min_score_is_a_pack_gate_not_an_item_floor() {
    let request = rerank_pack_request(5, 10_000, 0.4);
    let candidates = vec![rc("a", "x", 0.9), rc("b", "y", 0.5), rc("c", "z", 0.2)];
    // The top score clears the pack gate, so lower-ranked evidence remains
    // eligible instead of being filtered a second time.
    let report = assemble_reranked_pack(&request, &candidates);
    assert_eq!(
        report.memory_ids,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

#[test]
fn assemble_reranked_pack_respects_char_budget() {
    let first = "- [Observed at: 2026-07-19T00:00:00.000Z] high\n";
    // Budget fits exactly the first timestamped entry; the second would exceed it.
    let request = rerank_pack_request(5, first.len(), 0.0);
    let candidates = vec![rc("a", "high", 0.9), rc("b", "more", 0.5)];
    let report = assemble_reranked_pack(&request, &candidates);
    assert_eq!(report.memory_ids, vec!["a".to_string()]);
    assert_eq!(report.content, first);
    assert!(report.truncated);
}

#[test]
fn assemble_reranked_pack_sets_top_score_to_max_rerank() {
    let request = rerank_pack_request(5, 4000, 0.05);
    let candidates = vec![
        RerankCandidate {
            memory_id: "a".into(),
            content: "alpha".into(),
            observed_at: "2026-07-19T00:00:00.000Z".into(),
            rerank_score: 0.20,
            activation: None,
            consensus: false,
        },
        RerankCandidate {
            memory_id: "b".into(),
            content: "beta".into(),
            observed_at: "2026-07-19T00:00:00.000Z".into(),
            rerank_score: 0.70,
            activation: None,
            consensus: false,
        },
        RerankCandidate {
            memory_id: "c".into(),
            content: "gamma".into(),
            observed_at: "2026-07-19T00:00:00.000Z".into(),
            rerank_score: 0.40,
            activation: None,
            consensus: false,
        },
    ];
    let report = assemble_reranked_pack(&request, &candidates);
    assert_eq!(report.top_score, Some(f64::from(0.70_f32)));
}

#[test]
fn graph_candidates_compete_in_the_unified_rerank_pool() {
    let request = rerank_pack_request(3, 4000, 0.0);
    let candidates = vec![
        rc("a", "alpha", 0.90),
        rc("b", "beta", 0.80),
        rc("c", "gamma", 0.70),
        rc_reachable("graph_only", "graph-only evidence", 0.85, 0.40),
    ];

    let report = assemble_reranked_pack(&request, &candidates);

    assert_eq!(
        report.memory_ids,
        vec!["a", "graph_only", "b"],
        "graph-added evidence receives the same cross-encoder ordering as every other candidate"
    );
}

#[test]
fn graph_activation_does_not_reserve_or_force_a_pack_slot() {
    let request = rerank_pack_request(3, 4000, 0.0);
    let candidates = vec![
        rc("semantic_high", "semantic high", 0.90),
        rc("semantic_mid", "semantic mid", 0.80),
        rc("semantic_low", "semantic low", 0.70),
        rc_reachable("graph_low", "graph evidence", 0.10, 100.0),
    ];

    let report = assemble_reranked_pack(&request, &candidates);

    assert_eq!(
        report.memory_ids,
        vec!["semantic_high", "semantic_mid", "semantic_low"],
        "activation cannot override a higher cross-encoder score"
    );
}

#[test]
fn empty_pack_has_no_top_score() {
    let request = rerank_pack_request(5, 4000, 0.05);
    assert_eq!(empty_pack(&request).top_score, None);
}

#[test]
fn empty_pack_preserves_request_shape() {
    let request = rerank_pack_request(5, 10_000, 0.0);
    let empty = empty_pack(&request);
    assert!(empty.memory_ids.is_empty());
    assert!(empty.content.is_empty());
    assert_eq!(empty.title, "rr");
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
        maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
fn graph_cap_drops_remain_visible_without_changing_admitted_pool() {
    let mut merged = vec![crate::PackPoolItem {
        memory_id: "selected".to_string(),
        score: 0.9,
        admissions: Vec::new(),
    }];
    let activations = BTreeMap::from([("selected".to_string(), 0.9), ("dropped".to_string(), 0.8)]);
    let mut observed = Vec::new();

    let allocation_ranks =
        BTreeMap::from([("selected".to_string(), 1), ("dropped".to_string(), 2)]);
    apply_graph_admission_observations(
        &mut merged,
        &activations,
        &allocation_ranks,
        &BTreeMap::new(),
        &mut observed,
    );

    assert_eq!(
        merged
            .iter()
            .map(|item| item.memory_id.as_str())
            .collect::<Vec<_>>(),
        vec!["selected"],
        "trace instrumentation must not mutate reranker input membership"
    );
    let dropped = observed
        .iter()
        .find(|candidate| candidate.memory_id == "dropped")
        .expect("graph-cap drop remains observable");
    assert!(!dropped.admitted);
    assert_eq!(dropped.dropped_at.as_deref(), Some("graph_cap"));
    assert_eq!(dropped.graph_allocation_rank, Some(2));
    assert_eq!(dropped.admissions.len(), 1);
    assert_eq!(dropped.admissions[0].source, crate::AdmissionSource::Graph);
    assert_eq!(dropped.admissions[0].source_rank, 2);
    assert_eq!(dropped.admissions[0].activation, Some(0.8));
}

#[test]
fn hybrid_dropped_candidate_retains_graph_activation_observation() {
    let mut merged = Vec::new();
    let activations = BTreeMap::from([("shared".to_string(), 0.8)]);
    let mut observed = vec![crate::RerankPoolObservedCandidate {
        memory_id: "shared".to_string(),
        merged_rank: 6,
        admitted: false,
        dropped_at: Some("hybrid_cap".to_string()),
        graph_allocation_rank: None,
        admissions: vec![crate::AdmissionObservation {
            source: crate::AdmissionSource::Bm25,
            query_index: 0,
            source_rank: 6,
            seed_memory_id: None,
            activation: None,
            graph_route: None,
        }],
    }];

    let allocation_ranks = BTreeMap::from([("shared".to_string(), 1)]);
    apply_graph_admission_observations(
        &mut merged,
        &activations,
        &allocation_ranks,
        &BTreeMap::new(),
        &mut observed,
    );

    assert_eq!(
        observed.len(),
        1,
        "the existing observation is enriched in place"
    );
    assert_eq!(observed[0].dropped_at.as_deref(), Some("hybrid_cap"));
    assert_eq!(observed[0].graph_allocation_rank, Some(1));
    let graph = observed[0]
        .admissions
        .iter()
        .find(|admission| admission.source == crate::AdmissionSource::Graph)
        .expect("graph activation is retained alongside the direct source");
    assert_eq!(graph.source_rank, 1);
    assert_eq!(graph.activation, Some(0.8));
}

#[test]
fn production_observation_mode_does_not_materialize_graph_cap_drops() {
    let mut merged = vec![crate::PackPoolItem {
        memory_id: "selected".to_string(),
        score: 0.9,
        admissions: Vec::new(),
    }];
    let activations = BTreeMap::from([("selected".to_string(), 0.9), ("dropped".to_string(), 0.8)]);
    let admitted_before = merged
        .iter()
        .map(|item| item.memory_id.clone())
        .collect::<Vec<_>>();
    let mut observed = Vec::new();

    apply_graph_admission_observations(
        &mut merged,
        &activations,
        &BTreeMap::new(),
        &BTreeMap::new(),
        &mut observed,
    );

    assert_eq!(
        merged
            .iter()
            .map(|item| item.memory_id.clone())
            .collect::<Vec<_>>(),
        admitted_before,
        "observation mode must never change admitted membership"
    );
    assert!(
        observed
            .iter()
            .all(|candidate| candidate.memory_id != "dropped"),
        "normal pack retrieval retains the prior bounded observation payload"
    );
}

#[test]
fn pool_trace_rejects_unbounded_evidence_join_limits() {
    let error = build_hybrid_rerank_pool_trace_with_evidence_options(
        temp_store_path("trace_graph_bounds"),
        &rerank_pack_request(5, 4_000, 0.0),
        10,
        EvidenceJoinOptions {
            max_graph_seeds: crate::MAX_PACK_MEMORIES + 1,
            ..EvidenceJoinOptions::default()
        },
    )
    .expect_err("trace graph limits must fail before opening the store");

    assert!(matches!(error, crate::Error::InvalidRequest { .. }));
    assert!(error.to_string().contains("must not exceed 50"));
}

#[test]
fn canonical_graph_observation_replaces_provisional_rank() {
    let mut merged = vec![crate::PackPoolItem {
        memory_id: "selected".to_string(),
        score: 0.9,
        admissions: vec![crate::AdmissionObservation {
            source: crate::AdmissionSource::Graph,
            query_index: 0,
            source_rank: 99,
            seed_memory_id: None,
            activation: Some(0.9),
            graph_route: None,
        }],
    }];
    let activations = BTreeMap::from([("selected".to_string(), 0.9)]);
    let allocation_ranks = BTreeMap::from([("selected".to_string(), 1)]);
    let mut observed = Vec::new();

    apply_graph_admission_observations(
        &mut merged,
        &activations,
        &allocation_ranks,
        &BTreeMap::new(),
        &mut observed,
    );

    let graph_sources = merged[0]
        .admissions
        .iter()
        .filter(|source| source.source == crate::AdmissionSource::Graph)
        .collect::<Vec<_>>();
    assert_eq!(graph_sources.len(), 1);
    assert_eq!(graph_sources[0].source_rank, 1);
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
            maxsim_shortlist: 0,
        },
    )
    .expect_err("bad format should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
    use crate::{merge_rerank_pools, PackPoolItem};
    let item = |id: &str, score: f64| PackPoolItem {
        memory_id: id.to_string(),
        score,
        admissions: Vec::new(),
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
fn merge_rerank_pools_retains_overlapping_admission_sources() {
    use crate::{merge_rerank_pools, AdmissionSource, PackPoolItem};

    let ann = vec![PackPoolItem::direct_candidate(
        "shared".to_string(),
        0.9,
        AdmissionSource::Ann,
        0,
        1,
    )];
    let bm25 = vec![PackPoolItem::direct_candidate(
        "shared".to_string(),
        0.8,
        AdmissionSource::Bm25,
        0,
        1,
    )];

    let merged = merge_rerank_pools(&ann, &bm25, 1);

    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].memory_id, "shared");
    assert!((merged[0].score - 0.9).abs() < f64::EPSILON);
    assert_eq!(merged[0].admissions.len(), 2);
    assert_eq!(merged[0].admissions[0].source, AdmissionSource::Ann);
    assert_eq!(merged[0].admissions[1].source, AdmissionSource::Bm25);
}

#[test]
fn interleave_pools_retains_later_query_observations_at_cap() {
    use crate::{interleave_pools, AdmissionSource, PackPoolItem};

    let first_query = vec![PackPoolItem::direct_candidate(
        "shared".to_string(),
        0.9,
        AdmissionSource::Maxsim,
        0,
        1,
    )];
    let second_query = vec![PackPoolItem::direct_candidate(
        "shared".to_string(),
        0.8,
        AdmissionSource::Maxsim,
        1,
        1,
    )];

    let merged = interleave_pools(&[first_query, second_query], 1);

    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].memory_id, "shared");
    assert_eq!(
        merged[0]
            .admissions
            .iter()
            .map(|observation| observation.query_index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
}

#[test]
fn merge_rerank_pools_traces_candidates_displaced_by_hybrid_cap() {
    use crate::{merge_rerank_pools_with_trace, AdmissionSource, PackPoolItem};

    let ann = vec![
        PackPoolItem::direct_candidate("ann-1".to_string(), 0.9, AdmissionSource::Ann, 0, 1),
        PackPoolItem::direct_candidate("ann-2".to_string(), 0.8, AdmissionSource::Ann, 0, 2),
    ];
    let bm25 = vec![
        PackPoolItem::direct_candidate("bm25-1".to_string(), 0.7, AdmissionSource::Bm25, 0, 1),
        PackPoolItem::direct_candidate("bm25-2".to_string(), 0.6, AdmissionSource::Bm25, 0, 2),
    ];

    let (admitted, observed) = merge_rerank_pools_with_trace(&ann, &bm25, 2);

    assert_eq!(
        admitted
            .iter()
            .map(|item| item.memory_id.as_str())
            .collect::<Vec<_>>(),
        vec!["ann-1", "bm25-1"]
    );
    assert_eq!(observed.len(), 4);
    assert!(observed[0].admitted);
    assert!(observed[1].admitted);
    assert_eq!(observed[2].dropped_at.as_deref(), Some("hybrid_cap"));
    assert_eq!(observed[3].dropped_at.as_deref(), Some("hybrid_cap"));
}

#[test]
fn build_hybrid_rerank_pool_fetches_contents_on_one_snapshot() {
    use crate::{build_hybrid_rerank_pool, AdmissionSource};
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
            maxsim_shortlist: 0,
        },
        5,
    )
    .expect("pool builds");

    assert!(!pool.candidates.is_empty());
    assert_eq!(pool.candidates[0].memory_id, first.memory.id);
    assert_eq!(pool.candidates[0].content, "alpha retrieval probe content");
    assert_eq!(pool.candidates[0].observed_at, first.memory.observed_at);
    assert_eq!(pool.candidates[0].admissions.len(), 1);
    assert_eq!(
        pool.candidates[0].admissions[0].source,
        AdmissionSource::Bm25
    );
    assert!(pool.cos_top > f64::MIN);
    cleanup_store(&path);
}

#[test]
fn rerank_pool_always_materializes_canonical_content() {
    // Contextual representations and summaries may improve candidate admission,
    // but the cross-encoder always receives canonical evidence-bearing content.
    use crate::build_hybrid_rerank_pool;
    let path = temp_store_path("hybrid_rerank_pool_li_split");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut with_summary = basic_request("gamma retrieval probe content");
    with_summary.summary = Some("gamma summary line".to_string());
    with_summary.embedding = Some(vec![1.0, 0.0]);
    with_summary.embedding_model_id = Some("dense-li-split-test".to_string());
    with_summary.retrieval_representation = Some(RetrievalRepresentationInput {
        kind: "contextual-card-v1".to_string(),
        text: "gamma contextual card".to_string(),
    });
    let first = remember_memory(&path, &with_summary).expect("remember with summary");
    let mut identity = basic_request("delta retrieval probe content");
    identity.summary = Some("delta identity summary".to_string());
    identity.embedding = Some(vec![0.0, 1.0]);
    identity.embedding_model_id = Some("dense-li-split-test".to_string());
    let second = remember_memory(&path, &identity).expect("remember identity summary");

    let model = "colbert-li-split-test";
    {
        let connection = Connection::open(&path).expect("open");
        let vecs: Vec<Vec<f32>> = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        upsert_memory_token_embedding(&connection, &first.memory.id, model, &vecs)
            .expect("upsert tokens first");
        upsert_memory_token_embedding(&connection, &second.memory.id, model, &vecs)
            .expect("upsert tokens second");
    }

    let request = |li: bool, dense: bool| PackRequest {
        title: "rerank pool".to_string(),
        queries: vec!["gamma retrieval".to_string()],
        filters: SearchFilters::default(),
        max_memories: 5,
        max_chars: 2_000,
        format: "markdown".to_string(),
        min_score: 0.0,
        rerank_candidates: 0,
        query_embeddings: dense.then(|| vec![vec![1.0, 0.0]]),
        query_token_embeddings: li.then(|| vec![vec![vec![1.0, 0.0]]]),
        token_model_id: li.then(|| model.to_string()),
        maxsim_shortlist: 0,
    };

    let pool = build_hybrid_rerank_pool(&path, &request(true, true), 5).expect("LI pool builds");
    let by_id = |id: &str| {
        pool.candidates
            .iter()
            .find(|c| c.memory_id == id)
            .unwrap_or_else(|| panic!("candidate {id} present"))
    };
    let cand = by_id(&first.memory.id);
    assert!(cand
        .admissions
        .iter()
        .any(|observation| observation.source == crate::AdmissionSource::Maxsim));
    assert_eq!(
        cand.content, "gamma retrieval probe content",
        "LI content must stay content-only (no summary concat)"
    );
    assert_eq!(
        by_id(&second.memory.id).content,
        "delta retrieval probe content"
    );

    let optimized = build_hybrid_rerank_pool(&path, &request(true, false), 5)
        .expect("optimized LI pool builds");
    assert_eq!(
        optimized.candidates, pool.candidates,
        "late interaction without a cosine gate must not need dense ANN to preserve MaxSim plus BM25 candidates"
    );
    assert!(
        optimized.cos_top.is_nan(),
        "the optimized path has no dense cosine statistic"
    );

    let lexical =
        build_hybrid_rerank_pool(&path, &request(false, false), 5).expect("lexical pool builds");
    assert!(lexical
        .candidates
        .iter()
        .all(|candidate| !candidate.content.contains("summary")));
    cleanup_store(&path);
}
