//! Unit tests for the memkeeper store, split across submodules for navigability.
//! Private access to the parent module is preserved because this is still
//! `mod tests` inside the crate root.

use super::{
    apply_graph_admission_observations, approve_candidate, assemble_reranked_pack, backup_store,
    batch_search_memories, build_hybrid_rerank_pool_trace_with_evidence_options,
    build_hybrid_rerank_pool_with_evidence_options, build_pack, create_space, document_duplicates,
    dream_store, empty_pack, export_store, forget_memory, fts_score, get_document, get_memory,
    graph_context, graph_neighbors, import_store, ingest_source, init_store, last_synthesis_run,
    list_candidates, list_memories, list_silos, list_spaces, load_token_embeddings,
    load_token_embeddings_cached, mark_source_episodes_extracted, maxsim_score, memory_history,
    merge_entity, normalize_utc_timestamp, now_julian_day, open_initialized_write,
    promotion_candidates, prune_documents, rebuild_fts, recency_score_for_silo, record_recall,
    reject_candidate, relationship_confidence_from_evidence, remember_memory,
    representation_document, retrieval_companion, schema_mentions_required_objects,
    search_documents, search_entities, search_memories, sha256_hex, sidecar_path,
    source_tier_score, store_stats, store_stats_with_health, submit_candidate, upsert_entity,
    upsert_memory_token_embedding, upsert_relationship, validate_retrieval_representation,
    verify_memory, BackupRequest, BatchSearchQuery, BatchSearchRequest, CandidateApproveRequest,
    CandidateListRequest, CandidateRejectRequest, CandidateSubmitRequest, CapturedEntity,
    CapturedRelationship, DocumentDuplicatesRequest, DocumentGetRequest, DocumentPruneRequest,
    DocumentSearchRequest, DreamRequest, EntityMergeRequest, EntitySearchRequest,
    EntityUpsertRequest, Error, EvidenceJoinOptions, ExportRequest, ForgetRequest, GetOptions,
    GraphCapture, GraphContextRequest, GraphNeighborsRequest, HistoryOptions, ImportRequest,
    IngestRequest, MarkExtractedRequest, MemoryListRequest, PackRequest,
    PromotionCandidatesRequest, RecallEvent, RecallLogRequest, RelationshipUpsertRequest,
    RememberRequest, RerankCandidate, RetrievalRepresentationInput, SearchFilters, SearchRequest,
    SearchResult, SiloListRequest, SpaceCreateRequest, VerifyRequest, DEFAULT_PROMOTE_RANK_CAP,
    DEFAULT_PROMOTE_SCORE_FLOOR, DEFAULT_PROMOTE_THRESHOLD, DOCUMENTS_SPACE, MAX_CONTENT_CHARS,
    MAX_HISTORY_LIMIT, MAX_MEMORY_LINKS, MAX_METADATA_VALUE_CHARS, MAX_SEARCH_LIMIT,
    MAX_TIMESTAMP_CHARS, PROJECT_STORE_RELATIVE_PATH, SCHEMA_SQL, SCHEMA_VERSION,
    USER_STORE_PATH_HINT, VOLATILE_MAX_RECENCY_SCORE,
};
use memkeeper_core::DEFAULT_SPACE;
use rusqlite::{params, Connection};

/// Stable Julian day (2023-02-24) for recency-curve tests. The curve
/// assertions are relative to this reference, so pinning it keeps them
/// independent of the wall clock.
const FIXED_TEST_JULIAN_DAY: f64 = 2_460_000.0;
use std::{collections::BTreeMap, env, fs, path::Path, path::PathBuf, process, time::SystemTime};

fn represented_request(content: &str, card: &str) -> RememberRequest {
    let mut request = basic_request(content);
    request.summary = Some("deployment summary".to_string());
    request.retrieval_representation = Some(RetrievalRepresentationInput {
        kind: "contextual-card-v1".to_string(),
        text: card.to_string(),
    });
    request
}
fn normalized_production_rust_sources(owner_file: &str) -> Vec<(PathBuf, String)> {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let tests_dir = src.join("tests");
    let tests_file = src.join("tests.rs");
    let owner_file = src.join(owner_file);
    let mut pending = vec![src];
    let mut sources = Vec::new();

    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory).expect("read memkeeper-store source directory") {
            let path = entry.expect("read source entry").path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if path.extension().and_then(|value| value.to_str()) != Some("rs")
                || path == tests_file
                || path.starts_with(&tests_dir)
                || path == owner_file
            {
                continue;
            }
            let source = fs::read_to_string(&path).expect("read production Rust source");
            let uncommented = source
                .lines()
                .map(|line| line.split_once("//").map_or(line, |(code, _)| code))
                .collect::<Vec<_>>()
                .join("\n");
            let normalized = uncommented
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_ascii_uppercase();
            sources.push((path, normalized));
        }
    }

    sources.sort_by(|left, right| left.0.cmp(&right.0));
    sources
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
fn assert_dream_link_evidence(path: &Path, expected_links: usize) {
    let connection = Connection::open(path).expect("open store");
    let written: Vec<(f64, String)> = connection
        .prepare(
            "SELECT confidence, metadata_json FROM memory_links \
             WHERE link_type='related_to' ORDER BY src_memory_id, dst_memory_id",
        )
        .expect("prepare links")
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .expect("query links")
        .collect::<std::result::Result<Vec<_>, _>>()
        .expect("collect links");
    assert_eq!(written.len(), expected_links);
    for (confidence, metadata_json) in written {
        assert!((confidence - (2.0 / 3.0)).abs() < 1e-12);
        let metadata: serde_json::Value =
            serde_json::from_str(&metadata_json).expect("valid link metadata");
        assert_eq!(metadata["source"], "dream_tag_link");
        assert_eq!(metadata["shared_tag_count"], 2);
        assert_eq!(
            metadata["shared_tags"],
            serde_json::json!(["cobalt-topic", "zephyr-topic"])
        );
    }
}
fn routing_metadata() -> String {
    serde_json::json!({
        "routing": true,
        "origin": "automatic_capture",
        "routing_contract": "evidence_join_v2",
        "routing_contract_version": 2,
    })
    .to_string()
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
        maxsim_shortlist: 0,
    }
}
fn rc(id: &str, content: &str, score: f32) -> RerankCandidate {
    RerankCandidate {
        memory_id: id.to_string(),
        content: content.to_string(),
        observed_at: "2026-07-19T00:00:00.000Z".to_string(),
        rerank_score: score,
        activation: None,
        consensus: false,
    }
}
fn rc_reachable(id: &str, content: &str, score: f32, activation: f64) -> RerankCandidate {
    RerankCandidate {
        memory_id: id.to_string(),
        content: content.to_string(),
        observed_at: "2026-07-19T00:00:00.000Z".to_string(),
        rerank_score: score,
        activation: Some(activation),
        consensus: false,
    }
}
fn employment_graph(
    person_key: &str,
    person_name: &str,
    person_aliases: &[&str],
    organization_key: &str,
    extractor_version: &str,
) -> GraphCapture {
    GraphCapture {
        extractor: "test-extractor".to_string(),
        extractor_version: Some(extractor_version.to_string()),
        entities: vec![
            CapturedEntity {
                entity_key: person_key.to_string(),
                entity_type: "person".to_string(),
                canonical_name: person_name.to_string(),
                aliases: person_aliases
                    .iter()
                    .map(|alias| (*alias).to_string())
                    .collect(),
            },
            CapturedEntity {
                entity_key: organization_key.to_string(),
                entity_type: "organization".to_string(),
                canonical_name: "Memkeeper".to_string(),
                aliases: Vec::new(),
            },
        ],
        relationships: vec![CapturedRelationship {
            subject_entity_key: person_key.to_string(),
            relation_type: "works_at".to_string(),
            object_entity_key: organization_key.to_string(),
            confidence: 0.98,
        }],
    }
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
#[cfg(feature = "semantic")]
fn assert_store_identity_metadata(store_path: &Path) {
    let connection = Connection::open(store_path).expect("open imported store");
    let applied: i64 = connection
        .query_row(
            &format!("SELECT COUNT(*) FROM schema_migrations WHERE version = {SCHEMA_VERSION}"),
            [],
            |row| row.get(0),
        )
        .expect("schema_migrations count");
    assert_eq!(applied, 1, "the current migration row must be re-asserted");
    for (key, expected) in [
        ("schema_version", SCHEMA_VERSION.to_string()),
        ("protocol_version", "memkeeper.v0.1".to_string()),
        ("default_space", DEFAULT_SPACE.to_string()),
    ] {
        let actual: String = connection
            .query_row("SELECT value FROM config_kv WHERE key = ?1", [key], |row| {
                row.get(0)
            })
            .unwrap_or_else(|error| panic!("config_kv {key}: {error}"));
        assert_eq!(actual, expected, "config_kv {key}");
    }
}
#[cfg(feature = "semantic")]
fn strip_metadata_rows(archive_path: &Path) {
    let archive = std::fs::read_to_string(archive_path).expect("read archive");
    let mut kept = Vec::new();
    let mut dropped = 0_u64;
    for line in archive.lines() {
        let mut value: serde_json::Value = serde_json::from_str(line).expect("archive line");
        match value.get("type").and_then(serde_json::Value::as_str) {
            Some("row") => {
                let table = value.get("table").and_then(serde_json::Value::as_str);
                if matches!(table, Some("schema_migrations" | "config_kv")) {
                    dropped += 1;
                    continue;
                }
            }
            Some("footer") => {
                let count = value["row_count"].as_u64().expect("footer row_count") - dropped;
                value["row_count"] = serde_json::json!(count);
                value["table_counts"]["schema_migrations"] = serde_json::json!(0);
                value["table_counts"]["config_kv"] = serde_json::json!(0);
            }
            _ => {}
        }
        kept.push(serde_json::to_string(&value).expect("reserialize"));
    }
    assert!(
        dropped > 0,
        "source archive must carry metadata rows to strip"
    );
    std::fs::write(archive_path, kept.join("\n") + "\n").expect("write archive");
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
fn remember_request(content: &str) -> RememberRequest {
    RememberRequest {
        space: None,
        silo: None,
        scope: None,
        project_key: None,
        kind: None,
        content: content.to_string(),
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
            batch_id: None,
            latency_ms: None,
            latency_source: None,
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
fn downgrade_representation_fixture_to_v5(connection: &Connection) {
    connection
        .execute_batch(
            "DROP TABLE IF EXISTS memory_representations;
             CREATE VIRTUAL TABLE memory_fts_v5 USING fts5(
               memory_id UNINDEXED, version_id UNINDEXED, space_name UNINDEXED,
               silo_name UNINDEXED, status UNINDEXED, kind UNINDEXED,
               content, summary, tags, source_text, metadata_text,
               tokenize = 'unicode61 remove_diacritics 2'
             );
             INSERT INTO memory_fts_v5 SELECT * FROM memory_fts;
             DROP TABLE memory_fts;
             CREATE VIRTUAL TABLE memory_fts USING fts5(
               memory_id UNINDEXED, version_id UNINDEXED, space_name UNINDEXED,
               silo_name UNINDEXED, status UNINDEXED, kind UNINDEXED,
               content, summary, tags, source_text, metadata_text,
               tokenize = 'unicode61 remove_diacritics 2'
             );
             INSERT INTO memory_fts SELECT * FROM memory_fts_v5;
             DROP TABLE memory_fts_v5;
             CREATE VIRTUAL TABLE memory_fts_public_v5 USING fts5(
               memory_id UNINDEXED, version_id UNINDEXED, space_name UNINDEXED,
               silo_name UNINDEXED, status UNINDEXED, kind UNINDEXED,
               content, summary, tags, metadata_text,
               tokenize = 'unicode61 remove_diacritics 2'
             );
             INSERT INTO memory_fts_public_v5 SELECT * FROM memory_fts_public;
             DROP TABLE memory_fts_public;
             CREATE VIRTUAL TABLE memory_fts_public USING fts5(
               memory_id UNINDEXED, version_id UNINDEXED, space_name UNINDEXED,
               silo_name UNINDEXED, status UNINDEXED, kind UNINDEXED,
               content, summary, tags, metadata_text,
               tokenize = 'unicode61 remove_diacritics 2'
             );
             INSERT INTO memory_fts_public SELECT * FROM memory_fts_public_v5;
             DROP TABLE memory_fts_public_v5;
             DELETE FROM schema_migrations WHERE version = 6;
             UPDATE config_kv SET value = '5' WHERE key = 'schema_version';
             PRAGMA user_version = 5;",
        )
        .expect("downgrade fixture to v5");
}
fn representation_migration_search_ids(path: &Path) -> Vec<String> {
    search_memories(
        path,
        &SearchRequest {
            query: "representation migration marker".to_string(),
            filters: SearchFilters::default(),
            limit: 10,
            offset: 0,
            snippet_chars: 240,
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
    .expect("lexical search")
    .results
    .into_iter()
    .map(|row| row.memory_id)
    .collect()
}
fn representation_migration_pack_ids(path: &Path) -> Vec<String> {
    build_pack(
        path,
        &PackRequest {
            title: "representation migration".to_string(),
            queries: vec!["representation migration marker".to_string()],
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
    .expect("pack")
    .memory_ids
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
fn seed_two_space_store(test_name: &str) -> (PathBuf, String, String) {
    let path = temp_store_path(test_name);
    cleanup_store(&path);
    init_store(&path).expect("init succeeds");

    let mut default_mem = basic_request("decision: retrieval marker alpha in the default space");
    default_mem.kind = None;
    let default_id = remember_memory(&path, &default_mem)
        .expect("remember default")
        .memory
        .id;

    create_space(
        &path,
        &SpaceCreateRequest {
            name: "trading".to_string(),
            display_name: None,
            description: None,
            default_silo: None,
            ontology: None,
            config_json: None,
            if_not_exists: true,
        },
    )
    .expect("create trading space");

    let mut trading_mem = basic_request("decision: retrieval marker bravo in the trading space");
    trading_mem.kind = None;
    trading_mem.space = Some("trading".to_string());
    let trading_id = remember_memory(&path, &trading_mem)
        .expect("remember trading")
        .memory
        .id;

    (path, default_id, trading_id)
}
fn search_ids_for_spaces(path: &Path, spaces: Vec<String>) -> Vec<String> {
    let report = search_memories(
        path,
        &SearchRequest {
            query: "retrieval marker".to_string(),
            filters: SearchFilters {
                spaces,
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
    let mut ids: Vec<String> = report.results.into_iter().map(|r| r.memory_id).collect();
    ids.sort();
    ids
}
#[cfg(feature = "semantic")]
fn index_f32(index: usize) -> f32 {
    u16::try_from(index).map_or(0.0, f32::from)
}

mod archive;
mod candidates;
mod documents;
mod dream;
mod graph;
mod memory;
mod pack;
mod schema;
mod search;
mod vectors;
