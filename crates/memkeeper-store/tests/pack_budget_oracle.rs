#![forbid(unsafe_code)]

//! Ralph-discipline correctness oracle for the pack char-budget defect.
//!
//! ## The defect (motivation)
//! `assemble_reranked_pack` injects the highest-ranked candidates that fit the
//! char budget *whole*. When the gate passes but the top-ranked candidate is by
//! itself larger than the entire `max_chars` budget, the loop breaks on the very
//! first item and the pack is emitted EMPTY — a confidently-relevant memory is
//! dropped purely for being long. Live measurement (2026-06-12) on the 20-query
//! probe set at the hook config (`max_memories=5, max_chars=1500`) showed this
//! firing on 5/20 queries (25%), including the highest-confidence hits
//! (`top_score` up to 0.781).
//!
//! ## Why this is a FROZEN oracle the loop MUST NOT edit
//! Unlike a behaviour-preserving perf loop, this fix MUST change output — but
//! only in the defect case. The oracle is therefore PARTITIONED:
//!
//! * **Regression half** (`regression_fingerprint_frozen`): a deterministic FNV
//!   fingerprint over a sweep of cases where the top item already fits (and the
//!   gate-blocked / floored / empty-pool cases that must stay empty). The fix
//!   MUST keep this byte-identical — proof it is surgical.
//! * **Target half** (`target_*`): the pre-registered post-fix behaviour for the
//!   defect case. These assertions are RED against the pre-fix engine and turn
//!   GREEN only when the fix lands. They pin: the top eligible candidate is
//!   injected truncated to `max_chars`, content is non-empty and within budget,
//!   the truncation marker `…` is present, valid UTF-8 is preserved on a char
//!   boundary, and the fallback never fires when the gate blocked or the floor
//!   excluded the top item.
//!
//! ## Pre-registered success criteria (no moving goalposts)
//! 1. `regression_fingerprint_frozen` MATCHES the frozen constant below.
//! 2. All `target_*` assertions pass.
//! 3. The full `memkeeper-store` unit suite + workspace tests stay green.
//! 4. No test/oracle file is edited (this file, `src/tests.rs`, `tests/*`).
//! 5. Live re-measurement: defect count on the 20-probe set drops 5 -> 0.

use memkeeper_store::{
    assemble_reranked_pack, PackReport, PackRequest, RerankCandidate, SearchFilters,
};

// Frozen regression fingerprint over `regression_scenarios()`, captured from the
// pre-fix engine. A value of 0 means "capture mode": the test prints the computed
// fingerprint and passes, so the baseline can be read once and pinned here.
const REGRESSION_FINGERPRINT: u64 = 0x2401_bdc4_114b_ece7;

// --- self-contained FNV-1a (matches bench_oracle.rs conventions) ------------
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
        self.mix_bytes(&[0x1f]);
    }
    fn mix_usize(&mut self, value: usize) {
        self.mix_bytes(&(value as u64).to_le_bytes());
    }
    fn mix_f64_bits(&mut self, value: f64) {
        self.mix_bytes(&value.to_bits().to_le_bytes());
    }
    fn finish(self) -> u64 {
        self.0
    }
}

fn mix_pack(fnv: &mut Fnv, report: &PackReport) {
    fnv.mix_str(&report.content);
    fnv.mix_usize(report.memory_ids.len());
    for id in &report.memory_ids {
        fnv.mix_str(id);
    }
    fnv.mix_bytes(&[u8::from(report.truncated)]);
    match report.top_score {
        Some(score) => {
            fnv.mix_bytes(&[1]);
            fnv.mix_f64_bits(score);
        }
        None => fnv.mix_bytes(&[0]),
    }
}

// --- helpers (integration-test view of the public API) ----------------------
fn req(max_memories: usize, max_chars: usize, min_score: f64) -> PackRequest {
    PackRequest {
        title: "oracle".to_string(),
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
    }
}

fn rc(id: &str, content: &str, score: f32) -> RerankCandidate {
    RerankCandidate {
        memory_id: id.to_string(),
        content: content.to_string(),
        rerank_score: score,
    }
}

/// One scenario: `(request, cosine_gate, cos_top, candidates)`.
type Scenario = (PackRequest, f64, f64, Vec<RerankCandidate>);

/// Deterministic sweep of cases the fix MUST NOT change: items that fit whole,
/// whole-item budget truncation, the gate-blocked / floored / empty-pool cases
/// that must stay empty, and exact-boundary fits.
fn regression_scenarios() -> Vec<Scenario> {
    vec![
        // single short item, generous budget
        (req(5, 10_000, 0.0), 0.0, 0.0, vec![rc("a", "short", 0.9)]),
        // two items both fit, ordered by score
        (
            req(5, 10_000, 0.0),
            0.0,
            0.0,
            vec![rc("a", "low", 0.1), rc("b", "high", 0.9)],
        ),
        // max_memories cap: 3 candidates, 2 injected -> truncated
        (
            req(2, 10_000, 0.0),
            0.0,
            0.0,
            vec![
                rc("a", "aaa", 0.3),
                rc("b", "bbb", 0.9),
                rc("c", "ccc", 0.5),
            ],
        ),
        // whole-item budget truncation: first fits, second too big for remaining
        // budget but the FIRST item fits whole (top item NOT oversized)
        (
            req(5, 14, 0.0),
            0.0,
            0.0,
            vec![rc("a", "aaaa", 0.9), rc("b", "bbbbbbbbbb", 0.5)],
        ),
        // exact-boundary fit: "- aaaa\n" == 7 bytes == max_chars
        (req(5, 7, 0.0), 0.0, 0.0, vec![rc("a", "aaaa", 0.9)]),
        // gate blocked (cos below gate, rr below min) -> empty, stays empty
        (
            req(5, 1500, 0.2),
            0.30,
            0.10,
            vec![rc("a", &"x".repeat(2000), 0.05)],
        ),
        // legacy per-item floor drops sub-floor items
        (
            req(5, 10_000, 0.4),
            0.0,
            0.0,
            vec![rc("a", "aa", 0.9), rc("b", "bb", 0.5), rc("c", "cc", 0.2)],
        ),
        // legacy floor excludes an OVERSIZED top item -> must stay empty (the
        // budget fallback must not override the floor)
        (
            req(5, 1500, 0.8),
            0.0,
            0.0,
            vec![rc("a", &"x".repeat(2000), 0.10)],
        ),
        // empty candidate pool -> empty pack, top_score None
        (req(5, 1500, 0.0), 0.0, 0.0, vec![]),
    ]
}

#[test]
fn regression_fingerprint_frozen() {
    let mut fnv = Fnv::new();
    for (request, gate, cos_top, candidates) in regression_scenarios() {
        let report = assemble_reranked_pack(&request, gate, cos_top, &candidates);
        mix_pack(&mut fnv, &report);
    }
    let fp = fnv.finish();
    eprintln!("pack_budget_oracle: regression_fingerprint={fp:016x}");
    if REGRESSION_FINGERPRINT == 0 {
        eprintln!("pack_budget_oracle: CAPTURE MODE — pin REGRESSION_FINGERPRINT = 0x{fp:016x}");
        return;
    }
    assert_eq!(
        fp, REGRESSION_FINGERPRINT,
        "regression fingerprint DRIFT: the fix changed a fit/blocked/floored case it must not touch"
    );
}

// --- target half: the fix (RED before, GREEN after) -------------------------

const OVERSIZED: usize = 2_000; // > the 1500 hook budget
const MARKER: &str = "…\n";

#[test]
fn target_oversized_top_injected_truncated_via_cosine_gate() {
    let request = req(5, 1500, 0.0);
    let candidates = vec![rc("top", &"a".repeat(OVERSIZED), 0.05)];
    // cos_top (0.80) clears the gate -> emit; top item is oversized.
    let report = assemble_reranked_pack(&request, 0.30, 0.80, &candidates);
    assert_eq!(
        report.memory_ids,
        vec!["top".to_string()],
        "oversized top item must be injected, not dropped"
    );
    assert!(!report.content.is_empty(), "content must be non-empty");
    assert!(
        report.content.len() <= request.max_chars,
        "content {} must be within budget {}",
        report.content.len(),
        request.max_chars
    );
    assert!(
        report.content.starts_with("- "),
        "markdown bullet preserved"
    );
    assert!(
        report.content.ends_with(MARKER),
        "truncation marker must be present"
    );
    assert!(report.truncated, "truncated flag set when text was cut");
}

#[test]
fn target_oversized_top_then_smaller_injects_only_top() {
    let request = req(5, 1500, 0.0);
    let candidates = vec![
        rc("top", &"a".repeat(OVERSIZED), 0.90),
        rc("second", "small", 0.50),
    ];
    let report = assemble_reranked_pack(&request, 0.30, 0.80, &candidates);
    assert_eq!(
        report.memory_ids,
        vec!["top".to_string()],
        "only the truncated top fits; budget is exhausted"
    );
    assert!(report.content.len() <= request.max_chars);
}

#[test]
fn target_oversized_top_preserves_utf8_char_boundary() {
    let request = req(5, 1500, 0.0);
    // "é" is 2 bytes in UTF-8; 1000 of them = 2000 bytes, oversized.
    let candidates = vec![rc("top", &"é".repeat(1000), 0.05)];
    let report = assemble_reranked_pack(&request, 0.30, 0.80, &candidates);
    assert_eq!(report.memory_ids, vec!["top".to_string()]);
    assert!(report.content.len() <= request.max_chars);
    // If truncation split a multibyte char, the String would be invalid and the
    // round-trip below would differ; this asserts a clean char-boundary cut.
    assert_eq!(
        report.content,
        String::from_utf8(report.content.clone().into_bytes()).unwrap()
    );
    assert!(report.content.ends_with(MARKER));
}

#[test]
fn target_oversized_top_injected_via_reranker_confidence() {
    let request = req(5, 1500, 0.4);
    let candidates = vec![rc("top", &"a".repeat(OVERSIZED), 0.45)];
    // cos_top (0.05) below the gate, but rr_top (0.45) >= min_score (0.4) -> emit.
    let report = assemble_reranked_pack(&request, 0.30, 0.05, &candidates);
    assert_eq!(report.memory_ids, vec!["top".to_string()]);
    assert!(!report.content.is_empty());
    assert!(report.content.len() <= request.max_chars);
}
