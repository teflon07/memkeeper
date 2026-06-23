#![forbid(unsafe_code)]

//! Ignored benchmark/acceptance harness for latency-sensitive prompt-time paths.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process,
    time::{Duration, Instant, SystemTime},
};

use memkeeper_store::{
    batch_search_memories, build_pack, graph_context, graph_neighbors, init_store, search_entities,
    search_memories, BatchSearchQuery, BatchSearchRequest, EntitySearchRequest, GraphContextReport,
    GraphContextRequest, GraphNeighborsReport, GraphNeighborsRequest, PackRequest, SearchFilters,
    SearchReport, SearchRequest, SearchResult,
};
use rusqlite::{params, Connection, Statement, Transaction};

const DEFAULT_FIXTURE_MEMORIES: usize = 10_000;
const DEFAULT_RUNS: usize = 40;
const DEFAULT_SEARCH_P95_MS: f64 = 50.0;
const DEFAULT_BATCH_P95_MS: f64 = 100.0;
const DEFAULT_PACK_P95_MS: f64 = 100.0;
const DEFAULT_ENTITY_SEARCH_P95_MS: f64 = 50.0;
const DEFAULT_GRAPH_NEIGHBORS_P95_MS: f64 = 100.0;
const DEFAULT_GRAPH_CONTEXT_P95_MS: f64 = 100.0;
const SEARCH_LIMIT: usize = 10;
const SNIPPET_CHARS: usize = 240;
const PACK_MAX_MEMORIES: usize = 10;
const PACK_MAX_CHARS: usize = 2_000;

#[test]
#[ignore = "release-oriented benchmark/acceptance harness; run explicitly"]
fn search_batch_and_pack_latency_acceptance() {
    let config = AcceptanceConfig::from_env();
    let path = temp_store_path("acceptance-latency");
    cleanup_store(&path);

    init_store(&path).expect("init succeeds");
    populate_fixture(&path, config.fixture_memories).expect("populate fixture");

    let source_hidden =
        search_memories(&path, &source_hidden_request()).expect("source-hidden search succeeds");
    assert!(
        source_hidden.results.is_empty(),
        "include_source=false must not match provenance-only fixture terms"
    );

    let search_request = search_request_for("memkeeper deterministic topic3");
    let batch_request = batch_request();
    let pack_request = pack_request();

    // Warm the OS page cache and SQLite VM paths outside measured samples.
    assert_search_report_bounds(
        &search_memories(&path, &search_request).expect("warm search"),
        SEARCH_LIMIT,
        SNIPPET_CHARS,
    );
    assert_batch_report_bounds(
        &batch_search_memories(&path, &batch_request).expect("warm batch"),
        batch_request.queries.len(),
        batch_request.limit,
        batch_request.snippet_chars,
    );
    assert_pack_bounds(
        &build_pack(&path, &pack_request).expect("warm pack"),
        &pack_request,
    );

    let mut search_samples = Vec::with_capacity(config.runs);
    let mut batch_samples = Vec::with_capacity(config.runs);
    let mut pack_samples = Vec::with_capacity(config.runs);

    for _ in 0..config.runs {
        let started = Instant::now();
        let report = search_memories(&path, &search_request).expect("search succeeds");
        search_samples.push(started.elapsed());
        assert_search_report_bounds(&report, SEARCH_LIMIT, SNIPPET_CHARS);

        let started = Instant::now();
        let batch = batch_search_memories(&path, &batch_request).expect("batch succeeds");
        batch_samples.push(started.elapsed());
        assert_batch_report_bounds(
            &batch,
            batch_request.queries.len(),
            batch_request.limit,
            batch_request.snippet_chars,
        );

        let started = Instant::now();
        let pack = build_pack(&path, &pack_request).expect("pack succeeds");
        pack_samples.push(started.elapsed());
        assert_pack_bounds(&pack, &pack_request);
    }

    let search_p95 = p95_ms(&search_samples);
    let batch_p95 = p95_ms(&batch_samples);
    let pack_p95 = p95_ms(&pack_samples);

    eprintln!(
        "memkeeper acceptance: fixture_memories={} runs={} search_p95_ms={search_p95:.3} batch_p95_ms={batch_p95:.3} pack_p95_ms={pack_p95:.3} thresholds=({:.3},{:.3},{:.3})",
        config.fixture_memories,
        config.runs,
        config.search_p95_ms,
        config.batch_p95_ms,
        config.pack_p95_ms,
    );

    assert!(
        search_p95 <= config.search_p95_ms,
        "search p95 {search_p95:.3}ms exceeded threshold {:.3}ms",
        config.search_p95_ms
    );
    assert!(
        batch_p95 <= config.batch_p95_ms,
        "batch-search p95 {batch_p95:.3}ms exceeded threshold {:.3}ms",
        config.batch_p95_ms
    );
    assert!(
        pack_p95 <= config.pack_p95_ms,
        "pack p95 {pack_p95:.3}ms exceeded threshold {:.3}ms",
        config.pack_p95_ms
    );

    cleanup_store(&path);
}

#[test]
#[ignore = "release-oriented graph benchmark/acceptance harness; run explicitly"]
fn graph_latency_acceptance() {
    let config = GraphAcceptanceConfig::from_env();
    let path = temp_store_path("acceptance-graph-latency");
    cleanup_store(&path);

    init_store(&path).expect("init succeeds");
    populate_fixture(&path, config.fixture_memories).expect("populate fixture");
    populate_graph_fixture(&path).expect("populate graph fixture");

    let entity_request = graph_entity_search_request();
    let neighbors_request = graph_neighbors_request();
    let context_request = graph_context_request();

    assert_graph_entity_search_bounds(
        &search_entities(&path, &entity_request).expect("warm entity search"),
        entity_request.limit,
    );
    assert_graph_neighbors_bounds(
        &graph_neighbors(&path, &neighbors_request).expect("warm graph neighbors"),
        &neighbors_request,
    );
    assert_graph_context_bounds(
        &graph_context(&path, &context_request).expect("warm graph context"),
        &context_request,
    );

    let mut entity_samples = Vec::with_capacity(config.runs);
    let mut neighbors_samples = Vec::with_capacity(config.runs);
    let mut context_samples = Vec::with_capacity(config.runs);

    for _ in 0..config.runs {
        let started = Instant::now();
        let report = search_entities(&path, &entity_request).expect("entity search succeeds");
        entity_samples.push(started.elapsed());
        assert_graph_entity_search_bounds(&report, entity_request.limit);

        let started = Instant::now();
        let neighbors = graph_neighbors(&path, &neighbors_request).expect("neighbors succeeds");
        neighbors_samples.push(started.elapsed());
        assert_graph_neighbors_bounds(&neighbors, &neighbors_request);

        let started = Instant::now();
        let context = graph_context(&path, &context_request).expect("context succeeds");
        context_samples.push(started.elapsed());
        assert_graph_context_bounds(&context, &context_request);
    }

    let entity_p95 = p95_ms(&entity_samples);
    let neighbors_p95 = p95_ms(&neighbors_samples);
    let context_p95 = p95_ms(&context_samples);

    eprintln!(
        "memkeeper graph acceptance: fixture_memories={} graph_entities=15 graph_relationships=20 runs={} entity_search_p95_ms={entity_p95:.3} graph_neighbors_p95_ms={neighbors_p95:.3} graph_context_p95_ms={context_p95:.3} thresholds=({:.3},{:.3},{:.3})",
        config.fixture_memories,
        config.runs,
        config.entity_search_p95_ms,
        config.graph_neighbors_p95_ms,
        config.graph_context_p95_ms,
    );

    assert!(
        entity_p95 <= config.entity_search_p95_ms,
        "entity-search p95 {entity_p95:.3}ms exceeded threshold {:.3}ms",
        config.entity_search_p95_ms
    );
    assert!(
        neighbors_p95 <= config.graph_neighbors_p95_ms,
        "graph-neighbors p95 {neighbors_p95:.3}ms exceeded threshold {:.3}ms",
        config.graph_neighbors_p95_ms
    );
    assert!(
        context_p95 <= config.graph_context_p95_ms,
        "graph-context p95 {context_p95:.3}ms exceeded threshold {:.3}ms",
        config.graph_context_p95_ms
    );

    cleanup_store(&path);
}

#[derive(Debug, Clone, Copy)]
struct AcceptanceConfig {
    fixture_memories: usize,
    runs: usize,
    search_p95_ms: f64,
    batch_p95_ms: f64,
    pack_p95_ms: f64,
}

impl AcceptanceConfig {
    fn from_env() -> Self {
        Self {
            fixture_memories: env_usize("MEMKEEPER_ACCEPTANCE_MEMORIES", DEFAULT_FIXTURE_MEMORIES)
                .max(1),
            runs: env_usize("MEMKEEPER_ACCEPTANCE_RUNS", DEFAULT_RUNS).max(1),
            search_p95_ms: env_f64("MEMKEEPER_ACCEPTANCE_SEARCH_P95_MS", DEFAULT_SEARCH_P95_MS),
            batch_p95_ms: env_f64("MEMKEEPER_ACCEPTANCE_BATCH_P95_MS", DEFAULT_BATCH_P95_MS),
            pack_p95_ms: env_f64("MEMKEEPER_ACCEPTANCE_PACK_P95_MS", DEFAULT_PACK_P95_MS),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct GraphAcceptanceConfig {
    fixture_memories: usize,
    runs: usize,
    entity_search_p95_ms: f64,
    graph_neighbors_p95_ms: f64,
    graph_context_p95_ms: f64,
}

impl GraphAcceptanceConfig {
    fn from_env() -> Self {
        Self {
            fixture_memories: env_usize(
                "MEMKEEPER_GRAPH_ACCEPTANCE_MEMORIES",
                DEFAULT_FIXTURE_MEMORIES,
            )
            .max(1),
            runs: env_usize("MEMKEEPER_GRAPH_ACCEPTANCE_RUNS", DEFAULT_RUNS).max(1),
            entity_search_p95_ms: env_f64(
                "MEMKEEPER_GRAPH_ENTITY_SEARCH_P95_MS",
                DEFAULT_ENTITY_SEARCH_P95_MS,
            ),
            graph_neighbors_p95_ms: env_f64(
                "MEMKEEPER_GRAPH_NEIGHBORS_P95_MS",
                DEFAULT_GRAPH_NEIGHBORS_P95_MS,
            ),
            graph_context_p95_ms: env_f64(
                "MEMKEEPER_GRAPH_CONTEXT_P95_MS",
                DEFAULT_GRAPH_CONTEXT_P95_MS,
            ),
        }
    }
}

fn populate_fixture(path: &Path, count: usize) -> rusqlite::Result<()> {
    let mut connection = Connection::open(path)?;
    let transaction = connection.transaction()?;
    {
        let mut inserter = FixtureInserter::new(&transaction)?;
        for index in 0..count {
            inserter.insert(&FixtureRow::new(index))?;
        }
    }
    transaction.commit()
}

fn populate_graph_fixture(path: &Path) -> rusqlite::Result<()> {
    let mut connection = Connection::open(path)?;
    let transaction = connection.transaction()?;
    for topic in 0..10 {
        let entity_id = format!("ent_accept_topic_{topic:02}");
        let entity_key = format!("entity:topic{topic}");
        transaction.execute(
            "INSERT OR IGNORE INTO entities (
                id, space_name, entity_key, entity_type, canonical_name, status, confidence,
                created_at, updated_at
             ) VALUES (?1, 'workspace-memory', ?2, 'Topic', ?3, 'active', 1.0,
                '2026-05-01T00:00:00.000Z', '2026-05-01T00:00:00.000Z')",
            params![entity_id, entity_key, format!("Topic {topic}")],
        )?;
        transaction.execute(
            "INSERT OR IGNORE INTO entity_aliases (entity_id, alias, normalized_alias, created_at)
             VALUES (?1, ?2, ?3, '2026-05-01T00:00:00.000Z')",
            params![
                entity_id,
                format!("memkeeper topic {topic}"),
                format!("memkeeper topic {topic}")
            ],
        )?;
    }
    for project in 0..5 {
        transaction.execute(
            "INSERT OR IGNORE INTO entities (
                id, space_name, entity_key, entity_type, canonical_name, status, confidence,
                created_at, updated_at
             ) VALUES (?1, 'workspace-memory', ?2, 'Project', ?3, 'active', 1.0,
                '2026-05-01T00:00:00.000Z', '2026-05-01T00:00:00.000Z')",
            params![
                format!("ent_accept_project_{project:02}"),
                format!("project:{project}"),
                format!("Project {project}")
            ],
        )?;
    }
    for topic in 0..10 {
        let next = (topic + 1) % 10;
        let memory_id = format!("mem_accept_{:06}", topic * 100);
        transaction.execute(
            "INSERT OR IGNORE INTO relationships (
                id, space_name, subject_entity_id, relation_type, object_entity_id, memory_id,
                status, confidence, observed_at, created_at, updated_at
             ) VALUES (?1, 'workspace-memory', ?2, 'related_to', ?3, ?4, 'active', 1.0,
                '2026-05-01T00:00:00.000Z', '2026-05-01T00:00:00.000Z', '2026-05-01T00:00:00.000Z')",
            params![
                format!("rel_accept_topic_{topic:02}_next"),
                format!("ent_accept_topic_{topic:02}"),
                format!("ent_accept_topic_{next:02}"),
                memory_id,
            ],
        )?;
        transaction.execute(
            "INSERT OR IGNORE INTO relationships (
                id, space_name, subject_entity_id, relation_type, object_entity_id, memory_id,
                status, confidence, observed_at, created_at, updated_at
             ) VALUES (?1, 'workspace-memory', ?2, 'owned_by', ?3, ?4, 'active', 1.0,
                '2026-05-01T00:00:00.000Z', '2026-05-01T00:00:00.000Z', '2026-05-01T00:00:00.000Z')",
            params![
                format!("rel_accept_topic_{topic:02}_project"),
                format!("ent_accept_topic_{topic:02}"),
                format!("ent_accept_project_{:02}", topic % 5),
                format!("mem_accept_{:06}", topic * 100 + 1),
            ],
        )?;
    }
    transaction.commit()
}

struct FixtureInserter<'transaction> {
    memory: Statement<'transaction>,
    version: Statement<'transaction>,
    event: Statement<'transaction>,
    tag: Statement<'transaction>,
    fts: Statement<'transaction>,
    public_fts: Statement<'transaction>,
}

impl<'transaction> FixtureInserter<'transaction> {
    fn new(transaction: &'transaction Transaction<'_>) -> rusqlite::Result<Self> {
        Ok(Self {
            memory: transaction.prepare(
                "INSERT INTO memories (
                    id, space_name, silo_name, scope, project_key, kind, entity_key, claim_key,
                    status, active_version_id, confidence, pinned, observed_at, created_at,
                    updated_at, metadata_json
                 ) VALUES (?1, 'workspace-memory', 'durable', 'workspace', ?2, ?3, ?4, ?5,
                    'active', ?6, ?7, ?8, ?9, ?9, ?10, ?11)",
            )?,
            version: transaction.prepare(
                "INSERT INTO memory_versions (
                    id, memory_id, version_num, content, summary, content_sha256,
                    source_ref_json, created_at, created_by
                 ) VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7, 'acceptance-harness')",
            )?,
            event: transaction.prepare(
                "INSERT INTO memory_events (
                    id, memory_id, event_type, new_status, actor, created_at
                 ) VALUES (?1, ?2, 'remember', 'active', 'acceptance-harness', ?3)",
            )?,
            tag: transaction.prepare(
                "INSERT INTO memory_tags (memory_id, tag, created_at) VALUES (?1, ?2, ?3)",
            )?,
            fts: transaction.prepare(
                "INSERT INTO memory_fts (
                    memory_id, version_id, space_name, silo_name, status, kind, content, summary,
                    tags, source_text, metadata_text
                 ) VALUES (?1, ?2, 'workspace-memory', 'durable', 'active', ?3, ?4, ?5, ?6, ?7, ?8)",
            )?,
            public_fts: transaction.prepare(
                "INSERT INTO memory_fts_public (
                    memory_id, version_id, space_name, silo_name, status, kind, content, summary,
                    tags, metadata_text
                 ) VALUES (?1, ?2, 'workspace-memory', 'durable', 'active', ?3, ?4, ?5, ?6, ?7)",
            )?,
        })
    }

    fn insert(&mut self, row: &FixtureRow) -> rusqlite::Result<()> {
        self.memory.execute(params![
            &row.memory_id,
            &row.project,
            row.kind,
            &row.entity_key,
            &row.claim_key,
            &row.version_id,
            row.confidence,
            row.pinned,
            &row.observed_at,
            &row.updated_at,
            &row.metadata_json,
        ])?;
        self.version.execute(params![
            &row.version_id,
            &row.memory_id,
            &row.content,
            &row.summary,
            &row.content_hash,
            &row.source_ref,
            &row.observed_at,
        ])?;
        self.event
            .execute(params![&row.event_id, &row.memory_id, &row.observed_at])?;
        self.tag
            .execute(params![&row.memory_id, "memkeeper", &row.observed_at])?;
        self.tag
            .execute(params![&row.memory_id, &row.topic, &row.observed_at])?;
        self.fts.execute(params![
            &row.memory_id,
            &row.version_id,
            row.kind,
            &row.content,
            &row.summary,
            &row.tags,
            &row.source_ref,
            &row.metadata_text,
        ])?;
        self.public_fts.execute(params![
            &row.memory_id,
            &row.version_id,
            row.kind,
            &row.content,
            &row.summary,
            &row.tags,
            &row.metadata_text,
        ])?;
        Ok(())
    }
}

struct FixtureRow {
    memory_id: String,
    version_id: String,
    event_id: String,
    topic: String,
    project: String,
    entity_key: String,
    claim_key: String,
    kind: &'static str,
    confidence: f64,
    pinned: i64,
    observed_at: String,
    updated_at: String,
    summary: String,
    content: String,
    source_ref: String,
    metadata_json: String,
    tags: String,
    metadata_text: String,
    content_hash: String,
}

impl FixtureRow {
    fn new(index: usize) -> Self {
        let topic = format!("topic{}", index % 10);
        let marker = format!("marker{}", index % 100);
        let project = format!("project{}", index % 5);
        let entity_key = format!("entity:{topic}");
        let claim_key = format!("claim:{topic}:{marker}");
        let day = (index % 28) + 1;
        let updated_day = ((index + 7) % 28) + 1;
        let confidence_offset = u8::try_from(index % 50).expect("bounded confidence offset");
        Self {
            memory_id: format!("mem_accept_{index:06}"),
            version_id: format!("ver_accept_{index:06}"),
            event_id: format!("evt_accept_{index:06}"),
            kind: if index.is_multiple_of(3) {
                "decision"
            } else {
                "fact"
            },
            confidence: 0.5 + f64::from(confidence_offset) / 100.0,
            pinned: i64::from(index.is_multiple_of(97)),
            observed_at: format!("2026-05-{day:02}T12:00:00.000Z"),
            updated_at: format!("2026-06-{updated_day:02}T12:00:00.000Z"),
            summary: format!("memkeeper deterministic {topic} summary {marker}"),
            content: format!(
                "decision: memkeeper deterministic sqlite memory retrieval {topic} {marker} workspace durable local-first benchmark row {index}"
            ),
            source_ref: format!(
                "{{\"type\":\"manual\",\"path\":\"/private/provenance-only-acceptance-{index}\"}}"
            ),
            metadata_json: format!("{{\"fixture_index\":{index}}}"),
            tags: format!("memkeeper {topic} {marker}"),
            metadata_text: format!("{project} {entity_key} {claim_key}"),
            content_hash: format!("{index:064x}"),
            topic,
            project,
            entity_key,
            claim_key,
        }
    }
}

fn search_request_for(query: &str) -> SearchRequest {
    SearchRequest {
        query: query.to_string(),
        filters: SearchFilters::default(),
        limit: SEARCH_LIMIT,
        offset: 0,
        snippet_chars: SNIPPET_CHARS,
        include_content: false,
        include_source: false,
        semantic_fallback: "disabled".to_string(),
        lexical_fallback: "conservative".to_string(),
        embedding: None,
        query_token_embedding: None,
        token_model_id: None,
    }
}

fn source_hidden_request() -> SearchRequest {
    SearchRequest {
        query: "provenance only acceptance".to_string(),
        filters: SearchFilters::default(),
        limit: SEARCH_LIMIT,
        offset: 0,
        snippet_chars: SNIPPET_CHARS,
        include_content: false,
        include_source: false,
        semantic_fallback: "disabled".to_string(),
        lexical_fallback: "conservative".to_string(),
        embedding: None,
        query_token_embedding: None,
        token_model_id: None,
    }
}

fn batch_request() -> BatchSearchRequest {
    BatchSearchRequest {
        queries: (0..5)
            .map(|index| BatchSearchQuery {
                name: Some(format!("topic{index}")),
                query: format!("memkeeper deterministic topic{index}"),
                limit: Some(SEARCH_LIMIT),
            })
            .collect(),
        common_filters: SearchFilters::default(),
        limit: SEARCH_LIMIT,
        offset: 0,
        snippet_chars: SNIPPET_CHARS,
        include_content: false,
        include_source: false,
        semantic_fallback: "disabled".to_string(),
    }
}

fn pack_request() -> PackRequest {
    PackRequest {
        title: "memkeeper acceptance pack".to_string(),
        queries: (0..5)
            .map(|index| format!("memkeeper deterministic topic{index}"))
            .collect(),
        filters: SearchFilters::default(),
        max_memories: PACK_MAX_MEMORIES,
        max_chars: PACK_MAX_CHARS,
        format: "markdown".to_string(),
        min_score: 0.0,
        rerank_candidates: 0,
        query_embeddings: None,
        query_token_embeddings: None,
        token_model_id: None,
    }
}

fn graph_entity_search_request() -> EntitySearchRequest {
    EntitySearchRequest {
        space: None,
        query: Some("memkeeper topic 3".to_string()),
        entity_key: None,
        entity_types: Vec::new(),
        statuses: Vec::new(),
        limit: 10,
        offset: 0,
        include_source: false,
    }
}

fn graph_neighbors_request() -> GraphNeighborsRequest {
    GraphNeighborsRequest {
        space: None,
        entity_id: None,
        entity_key: Some("entity:topic3".to_string()),
        depth: 2,
        relation_types: Vec::new(),
        statuses: Vec::new(),
        max_edges: 50,
        include_tombstoned: false,
        include_source: false,
    }
}

fn graph_context_request() -> GraphContextRequest {
    GraphContextRequest {
        space: None,
        entity_id: None,
        entity_key: Some("entity:topic3".to_string()),
        depth: 2,
        relation_types: Vec::new(),
        statuses: Vec::new(),
        max_edges: 50,
        max_memories: PACK_MAX_MEMORIES,
        max_chars: PACK_MAX_CHARS,
        include_tombstoned: false,
        include_source: false,
    }
}

fn assert_graph_entity_search_bounds(report: &memkeeper_store::EntitySearchReport, limit: usize) {
    assert_eq!(report.strategy, "deterministic_entity_search_v0");
    assert!(report.results.len() <= limit);
    assert!(
        !report.results.is_empty(),
        "entity fixture query should match rows"
    );
    for result in &report.results {
        assert!(result.entity.source_episode_id.is_none());
        assert_eq!(result.entity.space, "workspace-memory");
        assert_ne!(result.entity.status, "tombstoned");
    }
}

fn assert_graph_neighbors_bounds(report: &GraphNeighborsReport, request: &GraphNeighborsRequest) {
    assert_eq!(report.strategy, "deterministic_graph_neighbors_v0");
    assert_eq!(report.depth, request.depth);
    assert_eq!(report.max_edges, request.max_edges);
    assert!(report.relationships.len() <= request.max_edges);
    assert!(!report.entities.is_empty());
    assert!(!report.relationships.is_empty());
    for entity in &report.entities {
        assert!(entity.entity.source_episode_id.is_none());
        assert_eq!(entity.entity.space, "workspace-memory");
    }
    for relationship in &report.relationships {
        assert!(relationship.relationship.source_episode_id.is_none());
        assert_eq!(relationship.relationship.space, "workspace-memory");
        assert_eq!(relationship.relationship.status, "active");
    }
}

fn assert_graph_context_bounds(report: &GraphContextReport, request: &GraphContextRequest) {
    assert_eq!(report.strategy, "deterministic_graph_context_v0");
    assert!(report.graph.relationships.len() <= request.max_edges);
    assert!(report.pack.memory_ids.len() <= request.max_memories);
    assert!(report.pack.content.chars().count() <= request.max_chars);
    assert!(report.pack.content.contains("Retrieved Memory"));
    assert!(!report.pack.memory_ids.is_empty());
    assert!(
        !report
            .pack
            .content
            .contains("/private/provenance-only-acceptance"),
        "graph-context pack output must not include source/provenance text"
    );
}

fn assert_batch_report_bounds(
    report: &memkeeper_store::BatchSearchReport,
    expected_queries: usize,
    limit: usize,
    snippet_chars: usize,
) {
    assert_eq!(report.results.len(), expected_queries);
    for item in &report.results {
        assert_search_report_bounds(&item.report, limit, snippet_chars);
    }
}

fn assert_search_report_bounds(report: &SearchReport, limit: usize, snippet_chars: usize) {
    assert_eq!(report.strategy, "deterministic_fts_v0");
    assert!(!report.semantic_attempted);
    assert_eq!(report.semantic_reason, "disabled_v0_1");
    assert!(report.results.len() <= limit);
    assert!(
        !report.results.is_empty(),
        "fixture query should match rows"
    );
    for (index, result) in report.results.iter().enumerate() {
        assert_eq!(result.rank, index + 1);
        assert_result_bounds(result, snippet_chars);
    }
}

fn assert_result_bounds(result: &SearchResult, snippet_chars: usize) {
    assert!(result.score.is_finite());
    assert!(result.scores.fts.is_finite());
    assert!(result.scores.metadata.is_finite());
    assert!(result.scores.recency.is_finite());
    assert!(result.scores.scope.is_finite());
    assert!(result.scores.status.is_finite());
    assert!(result.scores.pin.is_finite());
    assert!(result.snippet.chars().count() <= snippet_chars);
    assert!(result.content.is_none());
    assert!(result.source_ref_json.is_none());
    assert_eq!(result.space, "workspace-memory");
    assert_eq!(result.status, "active");
}

fn assert_pack_bounds(report: &memkeeper_store::PackReport, request: &PackRequest) {
    assert_eq!(report.format, "markdown");
    assert!(report.content.chars().count() <= request.max_chars);
    assert!(report.memory_ids.len() <= request.max_memories);
    assert!(report.content.contains("Retrieved Memory"));
    assert!(
        !report
            .content
            .contains("/private/provenance-only-acceptance"),
        "pack output must not include source/provenance text"
    );
}

fn p95_ms(samples: &[Duration]) -> f64 {
    assert!(!samples.is_empty());
    let mut millis = samples
        .iter()
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect::<Vec<_>>();
    millis.sort_by(|left, right| left.partial_cmp(right).expect("finite duration"));
    let index = millis
        .len()
        .saturating_mul(95)
        .div_ceil(100)
        .saturating_sub(1)
        .min(millis.len() - 1);
    millis[index]
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(default)
}

fn temp_store_path(test_name: &str) -> PathBuf {
    env::temp_dir().join(format!(
        "memkeeper-{test_name}-{}-{}.sqlite",
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

fn cleanup_store(path: &Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(sidecar_path(path, "-wal"));
    let _ = fs::remove_file(sidecar_path(path, "-shm"));
    let _ = fs::remove_file(sidecar_path(path, "-journal"));
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    PathBuf::from(format!("{}{}", path.display(), suffix))
}
