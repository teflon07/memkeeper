#![forbid(unsafe_code)]

//! Ralph-loop behaviour-preserving optimization oracle for the deterministic
//! retrieval hot path (search / batch / pack).
//!
//! This is a FROZEN oracle: the optimization loop MAY edit the production
//! retrieval code in `src/lib.rs`, but MUST NOT edit this file, the baseline
//! JSON it reads, or any other `test_*`/`tests/*` file. The loop is allowed to
//! make retrieval faster; it is NOT allowed to change any ranking the engine
//! produces.
//!
//! ## Why this is deterministic (and time-invariant)
//! The corpus is generated purely from row index (no randomness). Every row
//! shares ONE `observed_at`/`updated_at`, so the wall-clock-dependent recency term
//! is an identical constant across all rows and therefore drops out of the
//! ranking order. The fingerprint deliberately EXCLUDES the time-varying
//! `recency` sub-score and the overall `score` (which folds recency in), and
//! the pack `top_score`. It captures everything that a pure-speed change must
//! keep byte-identical: result ordering, identities, the time-invariant
//! sub-scores (fts/metadata/scope/status/pin), snippets, and rendered pack
//! content. A change that reorders results or perturbs the scoring math (even
//! at ULP scale, which can flip an order) flips the fingerprint -> DRIFT.
//!
//! ## Running
//! Write the baseline once:
//! ```text
//! MEMKEEPER_BENCH_WRITE_BASELINE=1 cargo test --release -p memkeeper-store \
//!   --test bench_oracle -- --ignored --nocapture
//! ```
//! Verify behaviour + measure latency (loop runs this every iteration):
//! ```text
//! cargo test --release -p memkeeper-store --test bench_oracle -- --ignored --nocapture
//! ```
//! A behaviour change panics the test (DRIFT). The timing line is printed to
//! stderr as `bench_oracle: ...` for the loop to read.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process,
    time::{Duration, Instant, SystemTime},
};

use memkeeper_store::{
    batch_search_memories, build_pack, init_store, search_memories, BatchSearchQuery,
    BatchSearchReport, BatchSearchRequest, PackReport, PackRequest, SearchFilters, SearchReport,
    SearchRequest,
};
use rusqlite::{params, Connection, Statement, Transaction};

// Frozen oracle parameters. The loop cannot edit this file, so these are fixed.
const FIXTURE_MEMORIES: usize = 6_000;
const WARMUP_RUNS: usize = 5;
const TIMED_RUNS: usize = 60;
const SEARCH_LIMIT: usize = 10;
const SNIPPET_CHARS: usize = 240;
const PACK_MAX_MEMORIES: usize = 10;
const PACK_MAX_CHARS: usize = 2_000;
// One fixed timestamp for every row: recency becomes a uniform constant that
// cancels out of ranking order, making the fingerprint reproducible forever.
const FIXED_TS: &str = "2026-01-01T12:00:00.000Z";

#[test]
#[ignore = "ralph optimization oracle; run explicitly with --release"]
fn retrieval_behaviour_and_latency_oracle() {
    let path = temp_store_path("bench-oracle");
    cleanup_store(&path);

    init_store(&path).expect("init succeeds");
    populate_fixture(&path, FIXTURE_MEMORIES).expect("populate fixture");

    let search_request = search_request_for("memkeeper deterministic topic3");
    let batch_request = batch_request();
    let pack_request = pack_request();

    // Warm OS page cache + SQLite VM paths outside the measured samples and
    // capture the behaviour fingerprint from the warm (steady-state) results.
    let mut search_report = search_memories(&path, &search_request).expect("warm search");
    let mut batch_report = batch_search_memories(&path, &batch_request).expect("warm batch");
    let mut pack_report = build_pack(&path, &pack_request).expect("warm pack");
    for _ in 1..WARMUP_RUNS {
        search_report = search_memories(&path, &search_request).expect("warm search");
        batch_report = batch_search_memories(&path, &batch_request).expect("warm batch");
        pack_report = build_pack(&path, &pack_request).expect("warm pack");
    }

    assert!(
        !search_report.results.is_empty(),
        "fixture search must match rows"
    );
    assert!(
        !pack_report.memory_ids.is_empty(),
        "fixture pack must include memories"
    );

    let fingerprint = fingerprint_behaviour(&search_report, &batch_report, &pack_report);

    let mut search_samples = Vec::with_capacity(TIMED_RUNS);
    let mut batch_samples = Vec::with_capacity(TIMED_RUNS);
    let mut pack_samples = Vec::with_capacity(TIMED_RUNS);
    for _ in 0..TIMED_RUNS {
        let started = Instant::now();
        let report = search_memories(&path, &search_request).expect("search succeeds");
        search_samples.push(started.elapsed());
        assert_eq!(
            fingerprint_search(&report),
            fingerprint_search(&search_report),
            "search ranking drifted between runs (non-determinism)"
        );

        let started = Instant::now();
        let _batch = batch_search_memories(&path, &batch_request).expect("batch succeeds");
        batch_samples.push(started.elapsed());

        let started = Instant::now();
        let pack = build_pack(&path, &pack_request).expect("pack succeeds");
        pack_samples.push(started.elapsed());
        assert_eq!(
            fingerprint_pack(&pack),
            fingerprint_pack(&pack_report),
            "pack output drifted between runs (non-determinism)"
        );
    }

    let timing = Timing {
        search_median_ms: median_ms(&search_samples),
        search_p95_ms: p95_ms(&search_samples),
        batch_median_ms: median_ms(&batch_samples),
        batch_p95_ms: p95_ms(&batch_samples),
        pack_median_ms: median_ms(&pack_samples),
        pack_p95_ms: p95_ms(&pack_samples),
    };

    eprintln!(
        "bench_oracle: fixture={FIXTURE_MEMORIES} runs={TIMED_RUNS} fingerprint={fingerprint:016x} \
         search_median_ms={:.3} search_p95_ms={:.3} \
         batch_median_ms={:.3} batch_p95_ms={:.3} \
         pack_median_ms={:.3} pack_p95_ms={:.3}",
        timing.search_median_ms,
        timing.search_p95_ms,
        timing.batch_median_ms,
        timing.batch_p95_ms,
        timing.pack_median_ms,
        timing.pack_p95_ms,
    );

    let baseline_path = baseline_path();
    if env::var("MEMKEEPER_BENCH_WRITE_BASELINE").is_ok() {
        write_baseline(&baseline_path, fingerprint, &timing);
        eprintln!(
            "bench_oracle: wrote baseline to {} (fingerprint {fingerprint:016x})",
            baseline_path.display()
        );
    } else {
        let baseline = read_baseline(&baseline_path);
        if baseline.fingerprint == fingerprint {
            eprintln!(
                "bench_oracle: fingerprint MATCH ({fingerprint:016x}) — behaviour preserved. \
                 pack_median {:.3} -> {:.3} ms ({:+.1}%)",
                baseline.pack_median_ms,
                timing.pack_median_ms,
                pct_delta(baseline.pack_median_ms, timing.pack_median_ms),
            );
        } else {
            cleanup_store(&path);
            panic!(
                "bench_oracle: fingerprint DRIFT — behaviour changed. baseline={:016x} now={:016x}. \
                 Ranking/scoring is NOT byte-identical; revert this change.",
                baseline.fingerprint, fingerprint
            );
        }
    }

    cleanup_store(&path);
}

#[allow(clippy::struct_field_names)]
struct Timing {
    search_median_ms: f64,
    search_p95_ms: f64,
    batch_median_ms: f64,
    batch_p95_ms: f64,
    pack_median_ms: f64,
    pack_p95_ms: f64,
}

struct Baseline {
    fingerprint: u64,
    pack_median_ms: f64,
}

fn pct_delta(before: f64, after: f64) -> f64 {
    if before <= 0.0 {
        0.0
    } else {
        (after - before) / before * 100.0
    }
}

// ---- Behaviour fingerprint (FNV-1a 64; dependency-free, frozen) ------------

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn mix_bytes(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn mix_str(&mut self, value: &str) {
        self.mix_bytes(value.as_bytes());
        self.mix_bytes(&[0x1f]); // field separator
    }
    fn mix_usize(&mut self, value: usize) {
        self.mix_bytes(&(value as u64).to_le_bytes());
    }
    // Time-invariant sub-scores are captured by exact bit pattern: any ULP
    // change in the scoring math flips the fingerprint.
    fn mix_f64_bits(&mut self, value: f64) {
        self.mix_bytes(&value.to_bits().to_le_bytes());
    }
    fn finish(self) -> u64 {
        self.0
    }
}

fn fingerprint_behaviour(
    search: &SearchReport,
    batch: &BatchSearchReport,
    pack: &PackReport,
) -> u64 {
    let mut fnv = Fnv::new();
    fnv.mix_str("search");
    mix_search_report(&mut fnv, search);
    fnv.mix_str("batch");
    fnv.mix_usize(batch.results.len());
    for item in &batch.results {
        fnv.mix_str(item.query.as_str());
        mix_search_report(&mut fnv, &item.report);
    }
    fnv.mix_str("pack");
    mix_pack_report(&mut fnv, pack);
    fnv.finish()
}

fn fingerprint_search(report: &SearchReport) -> u64 {
    let mut fnv = Fnv::new();
    mix_search_report(&mut fnv, report);
    fnv.finish()
}

fn fingerprint_pack(report: &PackReport) -> u64 {
    let mut fnv = Fnv::new();
    mix_pack_report(&mut fnv, report);
    fnv.finish()
}

fn mix_search_report(fnv: &mut Fnv, report: &SearchReport) {
    fnv.mix_str(report.strategy.as_str());
    fnv.mix_usize(report.results.len());
    for result in &report.results {
        fnv.mix_usize(result.rank);
        fnv.mix_str(result.memory_id.as_str());
        fnv.mix_str(result.version_id.as_str());
        fnv.mix_str(result.kind.as_str());
        fnv.mix_str(result.status.as_str());
        fnv.mix_str(result.snippet.as_str());
        // Time-invariant sub-scores only (recency + overall score excluded).
        fnv.mix_f64_bits(result.scores.fts);
        fnv.mix_f64_bits(result.scores.metadata);
        fnv.mix_f64_bits(result.scores.scope);
        fnv.mix_f64_bits(result.scores.status);
        fnv.mix_f64_bits(result.scores.pin);
        fnv.mix_usize(result.tags.len());
        for tag in &result.tags {
            fnv.mix_str(tag.as_str());
        }
    }
}

fn mix_pack_report(fnv: &mut Fnv, report: &PackReport) {
    fnv.mix_str(report.format.as_str());
    fnv.mix_str(report.content.as_str());
    fnv.mix_usize(report.memory_ids.len());
    for id in &report.memory_ids {
        fnv.mix_str(id.as_str());
    }
    fnv.mix_bytes(&[u8::from(report.truncated)]);
}

// ---- Baseline persistence (tiny hand-rolled JSON; no serde dep) ------------

fn baseline_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("bench_oracle_baseline.json")
}

fn write_baseline(path: &Path, fingerprint: u64, timing: &Timing) {
    let json = format!(
        "{{\n  \"fingerprint\": \"{fingerprint:016x}\",\n  \"fixture_memories\": {FIXTURE_MEMORIES},\n  \"timed_runs\": {TIMED_RUNS},\n  \"search_median_ms\": {:.3},\n  \"batch_median_ms\": {:.3},\n  \"pack_median_ms\": {:.3}\n}}\n",
        timing.search_median_ms, timing.batch_median_ms, timing.pack_median_ms,
    );
    fs::write(path, json).expect("write baseline");
}

fn read_baseline(path: &Path) -> Baseline {
    let text = fs::read_to_string(path).unwrap_or_else(|err| {
        panic!(
            "bench_oracle: missing baseline {} ({err}). Run once with \
             MEMKEEPER_BENCH_WRITE_BASELINE=1 to create it.",
            path.display()
        )
    });
    Baseline {
        fingerprint: u64::from_str_radix(&extract_json_string(&text, "fingerprint"), 16)
            .expect("hex fingerprint"),
        pack_median_ms: extract_json_number(&text, "pack_median_ms"),
    }
}

fn extract_json_string(text: &str, key: &str) -> String {
    let needle = format!("\"{key}\"");
    let start = text.find(&needle).expect("key present") + needle.len();
    let rest = &text[start..];
    let open = rest.find('"').expect("opening quote") + 1;
    let close = rest[open..].find('"').expect("closing quote") + open;
    rest[open..close].to_string()
}

fn extract_json_number(text: &str, key: &str) -> f64 {
    let needle = format!("\"{key}\"");
    let start = text.find(&needle).expect("key present") + needle.len();
    let rest = text[start..].trim_start_matches([':', ' ']);
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(rest.len());
    rest[..end].parse().expect("numeric value")
}

// ---- Latency stats ---------------------------------------------------------

fn sorted_millis(samples: &[Duration]) -> Vec<f64> {
    let mut millis = samples
        .iter()
        .map(|duration| duration.as_secs_f64() * 1_000.0)
        .collect::<Vec<_>>();
    millis.sort_by(|left, right| left.partial_cmp(right).expect("finite duration"));
    millis
}

#[allow(clippy::manual_is_multiple_of, clippy::manual_midpoint)]
fn median_ms(samples: &[Duration]) -> f64 {
    let millis = sorted_millis(samples);
    assert!(!millis.is_empty());
    let mid = millis.len() / 2;
    if millis.len() % 2 == 0 {
        (millis[mid - 1] + millis[mid]) / 2.0
    } else {
        millis[mid]
    }
}

fn p95_ms(samples: &[Duration]) -> f64 {
    let millis = sorted_millis(samples);
    let index = millis
        .len()
        .saturating_mul(95)
        .div_ceil(100)
        .saturating_sub(1)
        .min(millis.len() - 1);
    millis[index]
}

// ---- Requests (fixed query set) --------------------------------------------

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
        title: "memkeeper bench oracle pack".to_string(),
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

// ---- Deterministic fixture (uniform dates) ---------------------------------

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
                 ) VALUES (?1, ?2, 1, ?3, ?4, ?5, ?6, ?7, 'bench-oracle')",
            )?,
            event: transaction.prepare(
                "INSERT INTO memory_events (
                    id, memory_id, event_type, new_status, actor, created_at
                 ) VALUES (?1, ?2, 'remember', 'active', 'bench-oracle', ?3)",
            )?,
            tag: transaction.prepare(
                "INSERT INTO memory_tags (memory_id, tag, created_at) VALUES (?1, ?2, ?3)",
            )?,
            fts: transaction.prepare(
                "INSERT INTO memory_fts (
                    memory_id, version_id, space_name, silo_name, status, kind, content, retrieval_text,
                    tags, source_text, metadata_text
                 ) VALUES (?1, ?2, 'workspace-memory', 'durable', 'active', ?3, ?4, ?5, ?6, ?7, ?8)",
            )?,
            public_fts: transaction.prepare(
                "INSERT INTO memory_fts_public (
                    memory_id, version_id, space_name, silo_name, status, kind, content, retrieval_text,
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
            FIXED_TS,
            FIXED_TS,
            &row.metadata_json,
        ])?;
        self.version.execute(params![
            &row.version_id,
            &row.memory_id,
            &row.content,
            &row.summary,
            &row.content_hash,
            &row.source_ref,
            FIXED_TS,
        ])?;
        self.event
            .execute(params![&row.event_id, &row.memory_id, FIXED_TS])?;
        self.tag
            .execute(params![&row.memory_id, "memkeeper", FIXED_TS])?;
        self.tag
            .execute(params![&row.memory_id, &row.topic, FIXED_TS])?;
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
        let confidence_offset = u8::try_from(index % 50).expect("bounded confidence offset");
        Self {
            memory_id: format!("mem_bench_{index:06}"),
            version_id: format!("ver_bench_{index:06}"),
            event_id: format!("evt_bench_{index:06}"),
            kind: if index.is_multiple_of(3) {
                "decision"
            } else {
                "fact"
            },
            confidence: 0.5 + f64::from(confidence_offset) / 100.0,
            pinned: i64::from(index.is_multiple_of(97)),
            summary: format!("memkeeper deterministic {topic} summary {marker}"),
            content: format!(
                "decision: memkeeper deterministic sqlite memory retrieval {topic} {marker} workspace durable local-first benchmark row {index}"
            ),
            source_ref: format!(
                "{{\"type\":\"manual\",\"path\":\"/private/provenance-only-bench-{index}\"}}"
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

// ---- Store path helpers ----------------------------------------------------

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
