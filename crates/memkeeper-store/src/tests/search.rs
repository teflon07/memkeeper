//! Tests for search operations.

use super::*;

#[test]
fn fts_uses_retrieval_representation_instead_of_summary() {
    let path = temp_store_path("fts_uses_retrieval_representation");
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut represented = represented_request(
        "fact: represented lexical target",
        "context includes zephyr-card-term",
    );
    represented.summary = Some("shared summary-only-term".to_string());
    let represented = remember_memory(&path, &represented).expect("remember represented");
    let mut identity = basic_request("fact: identity lexical target");
    identity.summary = Some("shared summary-only-term".to_string());
    let identity = remember_memory(&path, &identity).expect("remember identity");

    let connection = Connection::open(&path).expect("open store");
    let transaction = connection
        .unchecked_transaction()
        .expect("start rebuild transaction");
    rebuild_fts(&transaction).expect("rebuild fts");
    transaction.commit().expect("commit rebuild");

    let search = |query: &str| {
        search_memories(
            &path,
            &SearchRequest {
                query: query.to_string(),
                filters: SearchFilters::default(),
                limit: 10,
                offset: 0,
                snippet_chars: 80,
                include_content: true,
                include_source: false,
                semantic_fallback: "disabled".to_string(),
                lexical_fallback: "disabled".to_string(),
                embedding: None,
                query_token_embedding: None,
                token_model_id: None,
                maxsim_shortlist: 0,
            },
        )
        .expect("search succeeds")
    };
    let card_hit = search("zephyr-card-term");
    assert_eq!(card_hit.results[0].memory_id, represented.memory.id);

    let hidden_summary = search("summary-only-term");
    assert!(!hidden_summary
        .results
        .iter()
        .any(|row| row.memory_id == represented.memory.id));
    assert!(hidden_summary
        .results
        .iter()
        .any(|row| row.memory_id == identity.memory.id));
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
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
            maxsim_shortlist: 0,
        },
    )
    .expect_err("unknown lexical fallback should fail");
    assert!(matches!(error, Error::InvalidRequest { .. }));

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

