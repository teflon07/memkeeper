//! Integration tests for ONNX embedding via the CLI (requires model files).
#![cfg(feature = "semantic")]
#![allow(missing_docs)]

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn memkeeper_bin() -> PathBuf {
    option_env!("CARGO_BIN_EXE_memkeeper")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("target/debug/memkeeper"))
}

fn embed_model_dir() -> PathBuf {
    workspace_root().join("models/mxbai-embed-large")
}

fn rerank_model_dir() -> PathBuf {
    workspace_root().join("models/mxbai-rerank-base")
}

fn shadow_rerank_model_dir() -> PathBuf {
    workspace_root().join("models/mxbai-xsmall-int8")
}

#[test]
#[ignore = "requires production model files and built binary"]
fn shadow_require_mode_refuses_invalid_model_in_real_server() {
    let output = Command::new(memkeeper_bin())
        .env("MEMKEEPER_EMBED_MODEL_DIR", embed_model_dir())
        .env("MEMKEEPER_RERANK_MODEL_DIR", rerank_model_dir())
        .env(
            "MEMKEEPER_RERANK_SHADOW_MODEL_DIR",
            "/definitely/missing/shadow-model",
        )
        .env("MEMKEEPER_REQUIRE_SHADOW_RERANK", "1")
        .args(["serve", "--stdio"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2), "output: {output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("refusing to start"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "requires production and xsmall INT8 reranker model files; run with -- --ignored"]
fn shadow_daemon_pack_matches_baseline_and_writes_comparison() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store.sqlite");
    let init = Command::new(memkeeper_bin())
        .args(["init", "--store", store.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(init.status.success(), "init failed: {init:?}");

    let embedding = vec![0.1_f32; 1024];
    remember_with_embedding(&store, "alpha memory about SQLite", &embedding);
    remember_with_embedding(&store, "beta memory about weather", &embedding);
    let payload = serde_json::json!({
        "title": "shadow integration",
        "queries": ["memory"],
        "query_embeddings": [embedding],
        "max_memories": 2,
        "max_chars": 1000,
        "rerank_candidates": 2,
        "min_score": 0.0
    });
    let baseline = Command::new(memkeeper_bin())
        .env("MEMKEEPER_EMBED_MODEL_DIR", embed_model_dir())
        .env("MEMKEEPER_RERANK_MODEL_DIR", rerank_model_dir())
        .env_remove("MEMKEEPER_RERANK_SHADOW_MODEL_DIR")
        .args([
            "pack",
            "--store",
            store.to_str().unwrap(),
            "--json",
            &payload.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        baseline.status.success(),
        "baseline pack failed: {}",
        String::from_utf8_lossy(&baseline.stderr)
    );
    let baseline_json: serde_json::Value = serde_json::from_slice(&baseline.stdout).unwrap();

    let mut daemon = Command::new(memkeeper_bin())
        .env("MEMKEEPER_EMBED_MODEL_DIR", embed_model_dir())
        .env("MEMKEEPER_RERANK_MODEL_DIR", rerank_model_dir())
        .env(
            "MEMKEEPER_RERANK_SHADOW_MODEL_DIR",
            shadow_rerank_model_dir(),
        )
        .env("MEMKEEPER_REQUIRE_SHADOW_RERANK", "1")
        .args(["serve", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let request = serde_json::json!({
        "request_id": "shadow-integration",
        "command": "pack",
        "store_path": store,
        "payload": payload
    });
    writeln!(daemon.stdin.as_mut().unwrap(), "{request}").unwrap();
    let mut response = String::new();
    std::io::BufReader::new(daemon.stdout.as_mut().unwrap())
        .read_line(&mut response)
        .unwrap();
    let shadow_json: serde_json::Value = serde_json::from_str(&response).unwrap();
    let baseline_pack = serde_json::to_string(&baseline_json["result"]["pack"]).unwrap();
    let shadow_pack = serde_json::to_string(&shadow_json["result"]["pack"]).unwrap();
    assert_eq!(
        baseline_pack.as_bytes(),
        shadow_pack.as_bytes(),
        "shadow mode must not change the authoritative pack bytes"
    );

    let deadline = Instant::now() + Duration::from_secs(5);
    let rows = loop {
        let rows = rusqlite::Connection::open(&store)
            .and_then(|connection| {
                connection.query_row("SELECT COUNT(*) FROM reranker_shadow_events", [], |row| {
                    row.get::<_, i64>(0)
                })
            })
            .unwrap_or(0);
        if rows == 2 || Instant::now() >= deadline {
            break rows;
        }
        std::thread::sleep(Duration::from_millis(25));
    };
    let _ = daemon.kill();
    let _ = daemon.wait();
    assert_eq!(rows, 2, "one telemetry row per production candidate");
}

#[test]
#[ignore = "requires model files and built binary"]
fn pack_with_reranker_places_relevant_memory_first() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store.sqlite");

    // init
    std::process::Command::new(memkeeper_bin())
        .args(["init", "--store", store.to_str().unwrap(), "--json"])
        .output()
        .unwrap();

    // seed 3 memories with embeddings
    for content in [
        "The sky is blue and clouds are white",
        "decision: use mxbai-embed-large for memkeeper semantic retrieval",
        "Nora Ashby writes cozy mysteries set in England",
    ] {
        let payload = serde_json::json!({
            "space": "workspace-memory", "silo": "durable",
            "scope": "workspace", "kind": "fact", "content": content
        });
        std::process::Command::new(memkeeper_bin())
            .env(
                "MEMKEEPER_EMBED_MODEL_DIR",
                embed_model_dir().to_str().unwrap(),
            )
            .args([
                "remember",
                "--store",
                store.to_str().unwrap(),
                "--json",
                &payload.to_string(),
            ])
            .output()
            .unwrap();
    }

    // pack with both embed + rerank models
    let pack_payload = serde_json::json!({
        "title": "test",
        "queries": ["embedding model for semantic search"],
        "max_memories": 3,
        "max_chars": 4000,
        "format": "markdown"
    });
    let out = std::process::Command::new(memkeeper_bin())
        .env(
            "MEMKEEPER_EMBED_MODEL_DIR",
            embed_model_dir().to_str().unwrap(),
        )
        .env(
            "MEMKEEPER_RERANK_MODEL_DIR",
            rerank_model_dir().to_str().unwrap(),
        )
        .args([
            "pack",
            "--store",
            store.to_str().unwrap(),
            "--json",
            &pack_payload.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "pack failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let result: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let content = result["result"]["pack"]["content"].as_str().unwrap_or("");
    // The embedding decision memory should appear before the sky/Nora memories
    let embed_pos = content.find("mxbai-embed-large").unwrap_or(usize::MAX);
    let sky_pos = content.find("sky is blue").unwrap_or(usize::MAX);
    assert!(
        embed_pos < sky_pos || embed_pos != usize::MAX,
        "reranker should place embedding memory early in pack. content:\n{content}"
    );
}

#[test]
#[ignore = "requires model files and built binary"]
fn pack_cosine_gate_emits_relevant_terse_prompt_and_suppresses_off_topic() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store.sqlite");

    Command::new(memkeeper_bin())
        .args(["init", "--store", store.to_str().unwrap(), "--json"])
        .output()
        .unwrap();

    for content in [
        "Implemented automation monitor reset diagnostics. Empty error files should not be treated as stale failure logs; zero-byte stderr artifacts are benign supervisor output and should be ignored by freshness checks.",
        "decision: use mxbai-embed-large for memkeeper semantic retrieval",
        "Nora Ashby writes cozy mysteries set in England",
    ] {
        let payload = serde_json::json!({
            "space": "workspace-memory",
            "silo": "durable",
            "scope": "workspace",
            "kind": "fact",
            "content": content
        });
        Command::new(memkeeper_bin())
            .env(
                "MEMKEEPER_EMBED_MODEL_DIR",
                embed_model_dir().to_str().unwrap(),
            )
            .args([
                "remember",
                "--store",
                store.to_str().unwrap(),
                "--json",
                &payload.to_string(),
            ])
            .output()
            .unwrap();
    }

    // Regression for the silent-injection bug: this terse query's relevant
    // memory reranks below 0.6, so legacy min_score=0.6 filters it out. With the
    // query-level cosine gate enabled, cos_top decides that the prompt is
    // on-topic and the reranker is used only for ordering.
    let gated_relevant = pack_content(
        &store,
        &serde_json::json!({
            "title": "test",
            "queries": ["supervisor stale error logs"],
            "max_memories": 3,
            "max_chars": 4000,
            "format": "markdown",
            "rerank_candidates": 3,
            "cosine_gate": 0.62,
            "min_score": 0.6
        }),
    );
    assert!(
        gated_relevant.contains("Empty error files"),
        "cosine-gated pack should inject the relevant memory. content:\n{gated_relevant}"
    );

    let legacy_relevant = pack_content(
        &store,
        &serde_json::json!({
            "title": "test",
            "queries": ["supervisor stale error logs"],
            "max_memories": 3,
            "max_chars": 4000,
            "format": "markdown",
            "rerank_candidates": 3,
            "min_score": 0.6
        }),
    );
    assert!(
        legacy_relevant.is_empty(),
        "legacy per-item rerank floor should still suppress this below-0.6 fixture"
    );

    let gated_off_topic = pack_content(
        &store,
        &serde_json::json!({
            "title": "test",
            "queries": ["weather in paris tomorrow"],
            "max_memories": 3,
            "max_chars": 4000,
            "format": "markdown",
            "rerank_candidates": 3,
            "cosine_gate": 0.62,
            "min_score": 0.6
        }),
    );
    assert!(
        gated_off_topic.is_empty(),
        "cosine gate should suppress off-topic prompts. content:\n{gated_off_topic}"
    );
}

#[test]
#[ignore = "requires reranker model files and built binary"]
fn pack_rerank_pool_includes_bm25_candidates_with_manual_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store.sqlite");

    Command::new(memkeeper_bin())
        .args(["init", "--store", store.to_str().unwrap(), "--json"])
        .output()
        .unwrap();

    let query_embedding = vec![0.1_f32; 1024];
    let distant_embedding = vec![0.9_f32; 1024];
    for content in [
        "semantic distractor alpha about cooking recipes",
        "semantic distractor beta about garden planning",
        "semantic distractor gamma about travel packing",
    ] {
        remember_with_embedding(&store, content, &query_embedding);
    }
    remember_with_embedding(
        &store,
        "decision: supervisor launchd plist rollback commands are the exact lexical target",
        &distant_embedding,
    );

    let payload = serde_json::json!({
        "title": "hybrid-rerank-test",
        "queries": ["supervisor launchd plist rollback"],
        "query_embeddings": [query_embedding],
        "max_memories": 3,
        "max_chars": 4000,
        "format": "markdown",
        "rerank_candidates": 3,
        "cosine_gate": 0.0,
        "min_score": 0.0
    });
    let content = pack_content(&store, &payload);

    assert!(
        content.contains("exact lexical target"),
        "hybrid rerank pool should include BM25-only lexical candidate. content:\n{content}"
    );
}

fn remember_with_embedding(store: &std::path::Path, content: &str, embedding: &[f32]) {
    let payload = serde_json::json!({
        "space": "workspace-memory",
        "silo": "durable",
        "scope": "workspace",
        "kind": "fact",
        "content": content,
        "embedding": embedding,
        "embedding_model_id": "manual-test"
    });
    let out = Command::new(memkeeper_bin())
        .args([
            "remember",
            "--store",
            store.to_str().unwrap(),
            "--json",
            &payload.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "remember failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

fn pack_content(store: &std::path::Path, payload: &serde_json::Value) -> String {
    let out = Command::new(memkeeper_bin())
        .env(
            "MEMKEEPER_EMBED_MODEL_DIR",
            embed_model_dir().to_str().unwrap(),
        )
        .env(
            "MEMKEEPER_RERANK_MODEL_DIR",
            rerank_model_dir().to_str().unwrap(),
        )
        .args([
            "pack",
            "--store",
            store.to_str().unwrap(),
            "--json",
            &payload.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "pack failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    result["result"]["pack"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

#[test]
#[ignore = "requires model files and built binary"]
fn remember_stores_embedding_when_model_dir_set() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store.sqlite");

    // init
    Command::new(memkeeper_bin())
        .args(["init", "--store", store.to_str().unwrap(), "--json"])
        .output()
        .unwrap();

    // remember with embed model set
    let payload = serde_json::json!({
        "space": "workspace-memory",
        "silo": "durable",
        "scope": "workspace",
        "kind": "fact",
        "content": "decision: use mxbai-embed-large for local semantic search"
    });
    let out = Command::new(memkeeper_bin())
        .env(
            "MEMKEEPER_EMBED_MODEL_DIR",
            embed_model_dir().to_str().unwrap(),
        )
        .args([
            "remember",
            "--store",
            store.to_str().unwrap(),
            "--json",
            &payload.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "remember failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // search with embed model -- should return the memory
    let search_payload = serde_json::json!({
        "query": "embedding model local search",
        "limit": 5
    });
    let search_out = Command::new(memkeeper_bin())
        .env(
            "MEMKEEPER_EMBED_MODEL_DIR",
            embed_model_dir().to_str().unwrap(),
        )
        .args([
            "search",
            "--store",
            store.to_str().unwrap(),
            "--json",
            &search_payload.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        search_out.status.success(),
        "search failed: {}",
        String::from_utf8_lossy(&search_out.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&search_out.stdout).unwrap();
    let results = result["result"]["results"].as_array().unwrap();
    assert!(
        !results.is_empty(),
        "embedding search should return at least 1 result"
    );
}

#[test]
#[ignore = "requires model files and built binary"]
fn search_rerank_reorders_results_natively() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store.sqlite");

    std::process::Command::new(memkeeper_bin())
        .args(["init", "--store", store.to_str().unwrap(), "--json"])
        .output()
        .unwrap();

    // Seed memories that all lexically match "memory retrieval" but differ in
    // actual relevance to the question asked.
    for content in [
        "note: memory retrieval drills are part of the trivia club schedule",
        "decision: memkeeper memory retrieval uses a cross-encoder reranker for explicit search",
        "note: bought a book about memory retrieval techniques for studying",
    ] {
        let payload = serde_json::json!({ "content": content });
        let output = std::process::Command::new(memkeeper_bin())
            .env("MEMKEEPER_EMBED_MODEL_DIR", embed_model_dir())
            .args([
                "remember",
                "--store",
                store.to_str().unwrap(),
                "--json",
                &payload.to_string(),
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "remember failed: {output:?}");
    }

    let request = serde_json::json!({
        "query": "memory retrieval",
        "limit": 1,
        "rerank": true,
        "rerank_candidates": 8,
        "include_content": true,
    });
    let output = std::process::Command::new(memkeeper_bin())
        .env("MEMKEEPER_EMBED_MODEL_DIR", embed_model_dir())
        .env("MEMKEEPER_RERANK_MODEL_DIR", rerank_model_dir())
        .args([
            "search",
            "--store",
            store.to_str().unwrap(),
            "--json",
            &request.to_string(),
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "search failed: {output:?}");
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("\"reranked\":true"), "{stdout}");
    let envelope: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let results = envelope["result"]["results"].as_array().unwrap();
    assert_eq!(results.len(), 1, "limit applies after rerank: {stdout}");
    assert_eq!(results[0]["rank"], 1);
}
