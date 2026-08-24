//! Tests for documents operations.

use super::*;

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

