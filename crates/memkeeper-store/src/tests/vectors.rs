//! Tests for vectors operations.

use super::*;

#[test]
fn token_backfill_uses_retrieval_representation() {
    let path = temp_store_path("token_backfill_uses_retrieval_representation");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut identity = basic_request("identity content");
    identity.summary = Some("identity summary".to_string());
    let identity = remember_memory(&path, &identity).expect("remember identity");
    let mut represented = represented_request("represented content", "context card");
    represented.summary = Some("persisted summary".to_string());
    let represented = remember_memory(&path, &represented).expect("remember represented");

    let targets = crate::collect_token_backfill_targets(&path, true).expect("collect targets");
    let by_id = |id: &str| {
        targets
            .iter()
            .find(|(memory_id, _)| memory_id == id)
            .map(|(_, text)| text.as_str())
            .expect("target exists")
    };
    assert_eq!(
        by_id(&identity.memory.id),
        "identity summary\n\nidentity content"
    );
    assert_eq!(
        by_id(&represented.memory.id),
        "context card\n\nrepresented content"
    );

    #[cfg(feature = "semantic")]
    {
        let dense = crate::collect_reembed_targets(&path).expect("collect dense targets");
        let represented_dense = dense
            .iter()
            .find(|target| target.memory_id == represented.memory.id)
            .expect("dense target exists");
        assert_eq!(represented_dense.content, "represented content");
    }
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
        },
    )
    .expect("search");
    assert_eq!(report.strategy, "semantic_primary_v0");
    assert_eq!(report.results.len(), 2);

    cleanup_store(&path);
}

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
    assert_eq!(
        version, SCHEMA_VERSION,
        "fresh store should use the current schema version"
    );
    assert!(has_1024 > 0, "memory_vec_1024 table should exist");
    assert_eq!(has_1536, 0, "memory_vec_1536 table should not exist");
}

#[test]
#[cfg(feature = "semantic")]
fn drop_all_vector_tables_survives_shadows_listed_first() {
    use crate::schema::drop_all_vector_tables;
    use crate::{ensure_memory_vector_table, SEMANTIC_TABLE_SQL};
    let store_path = temp_store_path("drop_all_vector_tables_shadows_first");
    cleanup_store(&store_path);
    init_store(&store_path).expect("init store");
    let conn = open_initialized_write(&store_path).expect("open conn");
    conn.execute_batch(SEMANTIC_TABLE_SQL)
        .expect("create vec table");
    conn.execute(
        "INSERT INTO memory_vec_1024 (memory_id, embedding) VALUES ('mem_a', ?1)",
        [crate::embedding_json(&vec![0.0_f32; 1024]).expect("json")],
    )
    .expect("seed vector");
    // Reproduce the production store: sqlite_master lists the vec0 shadow tables
    // before the virtual table itself, so a naive drop hits a shadow first.
    conn.execute_batch(
        "PRAGMA writable_schema=ON;
         UPDATE sqlite_master SET rowid = (SELECT MAX(rowid) + 1 FROM sqlite_master)
         WHERE name = 'memory_vec_1024';
         PRAGMA writable_schema=OFF;",
    )
    .expect("reorder schema");
    drop(conn);

    let mut conn = open_initialized_write(&store_path).expect("reopen conn");
    let first: String = conn
        .query_row(
            "SELECT name FROM sqlite_master WHERE name LIKE 'memory_vec_%' ORDER BY rowid LIMIT 1",
            [],
            |r| r.get(0),
        )
        .expect("first vec table");
    assert_ne!(
        first, "memory_vec_1024",
        "test setup should list a shadow table first"
    );

    let tx = conn.transaction().expect("tx");
    drop_all_vector_tables(&tx).expect("drop vector tables");
    // The whole point: the table has to come back and accept writes afterwards.
    let table = ensure_memory_vector_table(&tx, 1024).expect("recreate vec table");
    tx.execute(
        &format!("INSERT INTO {table} (memory_id, embedding) VALUES ('mem_b', ?1)"),
        [crate::embedding_json(&vec![0.0_f32; 1024]).expect("json")],
    )
    .expect("insert after drop");
    tx.commit().expect("commit");
}

#[test]
fn hybrid_maxsim_applies_entity_filter_before_selection() {
    let path = temp_store_path("hybrid_maxsim_prefilters_entity");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut target_request = basic_request("target needle belongs to the requested entity");
    target_request.entity_key = Some("conversation-target".to_string());
    let target = remember_memory(&path, &target_request)
        .expect("remember target")
        .memory
        .id;

    let mut distractor_request = basic_request("unrelated globally stronger memory");
    distractor_request.entity_key = Some("conversation-distractor".to_string());
    let distractor = remember_memory(&path, &distractor_request)
        .expect("remember distractor")
        .memory
        .id;

    let model = "colbert-hybrid-prefilter-test";
    let connection = Connection::open(&path).expect("open store");
    upsert_memory_token_embedding(&connection, &target, model, &[vec![0.8, 0.0]])
        .expect("upsert target tokens");
    upsert_memory_token_embedding(&connection, &distractor, model, &[vec![1.0, 0.0]])
        .expect("upsert distractor tokens");
    drop(connection);

    let pool = crate::build_hybrid_rerank_pool(
        &path,
        &PackRequest {
            title: "filtered maxsim".to_string(),
            queries: vec!["target needle".to_string()],
            filters: SearchFilters {
                entity_keys: vec!["conversation-target".to_string()],
                ..SearchFilters::default()
            },
            max_memories: 1,
            max_chars: 2_000,
            format: "markdown".to_string(),
            min_score: 0.0,
            rerank_candidates: 0,
            query_embeddings: None,
            query_token_embeddings: Some(vec![vec![vec![1.0, 0.0]]]),
            token_model_id: Some(model.to_string()),
            maxsim_shortlist: 0,
        },
        1,
    )
    .expect("hybrid pool builds");

    assert_eq!(pool.candidates.len(), 1);
    assert_eq!(
        pool.candidates[0].memory_id, target,
        "MaxSim width must be applied inside the requested entity scope"
    );
    assert!(pool.candidates[0]
        .admissions
        .iter()
        .any(|observation| observation.source == crate::AdmissionSource::Maxsim));
    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn semantic_maxsim_applies_entity_filter_before_selection() {
    let path = temp_store_path("semantic_maxsim_prefilters_entity");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut target_request = basic_request("semantic target content");
    target_request.entity_key = Some("conversation-target".to_string());
    target_request.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
    target_request.embedding_model_id = Some("semantic-prefilter-test".to_string());
    let target = remember_memory(&path, &target_request)
        .expect("remember target")
        .memory
        .id;

    let mut distractor_request = basic_request("semantic distractor content");
    distractor_request.entity_key = Some("conversation-distractor".to_string());
    distractor_request.embedding = Some(vec![1.0, 0.0, 0.0, 0.0]);
    distractor_request.embedding_model_id = Some("semantic-prefilter-test".to_string());
    let distractor = remember_memory(&path, &distractor_request)
        .expect("remember distractor")
        .memory
        .id;

    let token_model = "colbert-semantic-prefilter-test";
    let connection = Connection::open(&path).expect("open store");
    upsert_memory_token_embedding(&connection, &target, token_model, &[vec![0.8, 0.0]])
        .expect("upsert target tokens");
    upsert_memory_token_embedding(&connection, &distractor, token_model, &[vec![1.0, 0.0]])
        .expect("upsert distractor tokens");
    drop(connection);

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "lexically unmatched probe".to_string(),
            filters: SearchFilters {
                entity_keys: vec!["conversation-target".to_string()],
                ..SearchFilters::default()
            },
            limit: 1,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "fallback".to_string(),
            lexical_fallback: "disabled".to_string(),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            query_token_embedding: Some(vec![vec![1.0, 0.0]]),
            token_model_id: Some(token_model.to_string()),
            maxsim_shortlist: 0,
        },
    )
    .expect("semantic search succeeds");

    assert_eq!(report.strategy, "semantic_primary_v0");
    assert_eq!(report.results.len(), 1);
    assert_eq!(
        report.results[0].memory_id, target,
        "MaxSim selection must retain the requested entity before SQL projection"
    );
    cleanup_store(&path);
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

#[cfg(feature = "semantic")]
#[test]
fn search_semantic_prefilters_scope_before_global_top_k() {
    let path = temp_store_path("search_semantic_prefilters_scope_before_global_top_k");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");
    create_space(
        &path,
        &SpaceCreateRequest {
            name: "other-notes".to_string(),
            display_name: None,
            description: None,
            default_silo: None,
            ontology: None,
            config_json: None,
            if_not_exists: false,
        },
    )
    .expect("space create succeeds");

    // 50 out-of-scope memories nearly identical to the query vector: more
    // than the inflated candidate pool for limit=10 (42), so a global top-k
    // is filled entirely by them.
    for index in 0..50 {
        let mut noise = basic_request(&format!("noise memory {index} in other space"));
        noise.space = Some("other-notes".to_string());
        noise.embedding = Some(vec![1.0, 0.000_1 * index_f32(index)]);
        noise.embedding_model_id = Some("dense-prefilter-test".to_string());
        remember_memory(&path, &noise).expect("remember noise memory");
    }
    let mut in_scope = basic_request("decision: the only in-scope semantic memory");
    in_scope.embedding = Some(vec![0.6, 0.0]);
    in_scope.embedding_model_id = Some("dense-prefilter-test".to_string());
    let remembered = remember_memory(&path, &in_scope).expect("remember in-scope memory");

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "zzz unmatched lexical tokens".to_string(),
            filters: SearchFilters {
                spaces: vec![DEFAULT_SPACE.to_string()],
                ..SearchFilters::default()
            },
            limit: 10,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "fallback".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: Some(vec![1.0, 0.0]),
            query_token_embedding: None,
            token_model_id: None,
            maxsim_shortlist: 0,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.strategy, "semantic_primary_v0");
    assert_eq!(report.results.len(), 1, "in-scope memory must survive");
    assert_eq!(report.results[0].memory_id, remembered.memory.id);

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn maxsim_shortlist_bounds_late_interaction_scan() {
    let path = temp_store_path("maxsim_shortlist_bounds_late_interaction_scan");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let model = "colbert-shortlist-test";
    let mut near = basic_request("alpha memory near the dense query");
    near.embedding = Some(vec![1.0, 0.0]);
    near.embedding_model_id = Some("dense-shortlist-test".to_string());
    let near = remember_memory(&path, &near).expect("remember near");
    let mut close = basic_request("beta memory close to the dense query");
    close.embedding = Some(vec![0.9, 0.1]);
    close.embedding_model_id = Some("dense-shortlist-test".to_string());
    let close = remember_memory(&path, &close).expect("remember close");
    let mut far = basic_request("gamma memory far from the dense query");
    far.embedding = Some(vec![0.0, 1.0]);
    far.embedding_model_id = Some("dense-shortlist-test".to_string());
    let far = remember_memory(&path, &far).expect("remember far");

    {
        let connection = Connection::open(&path).expect("open");
        upsert_memory_token_embedding(&connection, &near.memory.id, model, &[vec![1.0, 0.0]])
            .expect("tokens near");
        upsert_memory_token_embedding(&connection, &close.memory.id, model, &[vec![1.0, 0.0]])
            .expect("tokens close");
        // The far memory's tokens match the query tokens exactly: exhaustive
        // MaxSim always selects it.
        upsert_memory_token_embedding(&connection, &far.memory.id, model, &[vec![0.0, 1.0]])
            .expect("tokens far");
    }

    let search = |maxsim_shortlist: usize| {
        search_memories(
            &path,
            &SearchRequest {
                query: "zzz unmatched lexical tokens".to_string(),
                filters: SearchFilters::default(),
                limit: 2,
                offset: 0,
                snippet_chars: 80,
                include_content: false,
                include_source: false,
                semantic_fallback: "fallback".to_string(),
                lexical_fallback: "conservative".to_string(),
                embedding: Some(vec![1.0, 0.0]),
                query_token_embedding: Some(vec![vec![0.0, 1.0]]),
                token_model_id: Some(model.to_string()),
                maxsim_shortlist,
            },
        )
        .expect("search succeeds")
    };

    let exhaustive = search(0);
    assert_eq!(exhaustive.strategy, "semantic_primary_v0");
    assert!(
        exhaustive
            .results
            .iter()
            .any(|result| result.memory_id == far.memory.id),
        "exhaustive MaxSim must select the token-matching memory"
    );

    let bounded = search(2);
    assert_eq!(bounded.strategy, "semantic_primary_v0");
    assert!(
        bounded
            .results
            .iter()
            .all(|result| result.memory_id != far.memory.id),
        "shortlist of 2 must exclude the memory outside the dense top-2"
    );
    assert_eq!(bounded.results.len(), 2);

    cleanup_store(&path);
}

#[cfg(feature = "semantic")]
#[test]
fn maxsim_shortlist_retains_memories_without_single_vectors() {
    let path = temp_store_path("maxsim_shortlist_retains_memories_without_single_vectors");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let model = "colbert-vectorless-test";
    let mut near = basic_request("alpha memory near the dense query");
    near.embedding = Some(vec![1.0, 0.0]);
    near.embedding_model_id = Some("dense-vectorless-test".to_string());
    let near = remember_memory(&path, &near).expect("remember near");
    // A second vectored memory so the eligible set (3) exceeds the cap (1).
    let mut close = basic_request("beta memory close to the dense query");
    close.embedding = Some(vec![0.9, 0.1]);
    close.embedding_model_id = Some("dense-vectorless-test".to_string());
    let close = remember_memory(&path, &close).expect("remember close");
    let close_id = close.memory.id;
    let vectorless = remember_memory(
        &path,
        &basic_request("delta memory without a single vector"),
    )
    .expect("remember vectorless");

    {
        let connection = Connection::open(&path).expect("open");
        upsert_memory_token_embedding(&connection, &near.memory.id, model, &[vec![1.0, 0.0]])
            .expect("tokens near");
        upsert_memory_token_embedding(&connection, &close_id, model, &[vec![1.0, 0.0]])
            .expect("tokens close");
        upsert_memory_token_embedding(&connection, &vectorless.memory.id, model, &[vec![0.0, 1.0]])
            .expect("tokens vectorless");
    }

    let report = search_memories(
        &path,
        &SearchRequest {
            query: "zzz unmatched lexical tokens".to_string(),
            filters: SearchFilters::default(),
            limit: 1,
            offset: 0,
            snippet_chars: 80,
            include_content: false,
            include_source: false,
            semantic_fallback: "fallback".to_string(),
            lexical_fallback: "conservative".to_string(),
            embedding: Some(vec![1.0, 0.0]),
            query_token_embedding: Some(vec![vec![0.0, 1.0]]),
            token_model_id: Some(model.to_string()),
            maxsim_shortlist: 1,
        },
    )
    .expect("search succeeds");

    assert_eq!(report.strategy, "semantic_primary_v0");
    assert_eq!(report.results.len(), 1);
    assert_eq!(
        report.results[0].memory_id, vectorless.memory.id,
        "the vectorless memory must remain MaxSim-eligible under the cap"
    );

    cleanup_store(&path);
}
