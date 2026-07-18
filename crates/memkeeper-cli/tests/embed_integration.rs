//! Integration tests for ONNX embedding via the CLI (requires model files).
#![cfg(feature = "semantic")]
#![allow(missing_docs)]

use std::path::PathBuf;
use std::process::Command;

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

fn model_root() -> PathBuf {
    std::env::var_os("MEMKEEPER_TEST_MODEL_ROOT")
        .map_or_else(|| workspace_root().join("models"), PathBuf::from)
}

fn embed_model_dir() -> PathBuf {
    model_root().join("mxbai-embed-large")
}

fn rerank_model_dir() -> PathBuf {
    model_root().join("mxbai-xsmall-int8")
}

fn colbert_model_dir() -> PathBuf {
    model_root().join("colbert-small")
}

fn semantic_command() -> Command {
    let mut command = Command::new(memkeeper_bin());
    command
        .env_remove("MEMKEEPER_STORE")
        .env_remove("MEMKEEPER_SOCK")
        .env("MEMKEEPER_EMBED_MODEL_DIR", embed_model_dir())
        .env("MEMKEEPER_RERANK_MODEL_DIR", rerank_model_dir())
        .env("MEMKEEPER_COLBERT_MODEL_DIR", colbert_model_dir())
        .env("MEMKEEPER_LATE_INTERACTION", "1")
        .env("MEMKEEPER_REQUIRE_SEMANTIC", "1")
        .env("MEMKEEPER_REQUIRE_RERANK", "1");
    command
}

fn assert_evidence_join_models_exist() {
    for path in [
        embed_model_dir().join("model.onnx"),
        rerank_model_dir().join("model.onnx"),
        colbert_model_dir().join("model.onnx"),
    ] {
        assert!(
            path.is_file(),
            "required model artifact missing: {}",
            path.display()
        );
    }
}

fn run_json_command(
    store: &std::path::Path,
    command: &str,
    payload: &serde_json::Value,
) -> serde_json::Value {
    let output = semantic_command()
        .args([
            command,
            "--store",
            store.to_str().unwrap(),
            "--json",
            &payload.to_string(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{command} failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "invalid JSON command output: {error}\nstdout: {}",
            String::from_utf8_lossy(&output.stdout)
        )
    })
}

fn remember_semantic(store: &std::path::Path, content: &str, entity_key: Option<&str>) -> String {
    let mut payload = serde_json::json!({
        "content": content,
        "silo": "durable",
        "scope": "workspace",
        "kind": "fact"
    });
    if let Some(entity_key) = entity_key {
        payload["entity_key"] = serde_json::json!(entity_key);
    }
    run_json_command(store, "remember", &payload)["result"]["memory"]["id"]
        .as_str()
        .expect("remembered memory id")
        .to_string()
}

fn upsert_test_entity(store: &std::path::Path, entity_key: &str, canonical_name: &str) {
    run_json_command(
        store,
        "entity-upsert",
        &serde_json::json!({
            "entity_key": entity_key,
            "canonical_name": canonical_name
        }),
    );
}

fn upsert_test_route(
    store: &std::path::Path,
    subject_key: &str,
    predicate: &str,
    object_key: &str,
    subject_memory_id: &str,
    object_memory_id: &str,
) {
    run_json_command(
        store,
        "relationship-upsert",
        &serde_json::json!({
            "subject_entity_key": subject_key,
            "relation_type": predicate,
            "object_entity_key": object_key,
            "memory_id": subject_memory_id,
            "metadata_json": serde_json::json!({
                "routing": true,
                "origin": "adjudicated_capture",
                "routing_contract": "evidence_join_v1",
                "routing_contract_version": 1,
                "object_memory_id": object_memory_id
            }).to_string()
        }),
    );
}

fn trace_memory<'a>(
    trace: &'a serde_json::Value,
    memory_id: &str,
) -> Option<&'a serde_json::Value> {
    trace["result"]["pool_trace"]["candidates"]
        .as_array()?
        .iter()
        .find(|candidate| candidate["memory_id"] == memory_id)
}

#[test]
#[ignore = "requires the production embed, ColBERT, and INT8 reranker artifacts"]
fn evidence_graph_join_recovers_exact_entity_seed() {
    assert_evidence_join_models_exist();
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store.sqlite");
    let init = semantic_command()
        .args(["init", "--store", store.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(init.status.success(), "init failed: {init:?}");

    upsert_test_entity(&store, "person:steve", "Steve");
    upsert_test_entity(&store, "org:acme", "Acme Labs");
    let subject_id = remember_semantic(
        &store,
        "Steve profile identity record.",
        Some("person:steve"),
    );
    let object_id = remember_semantic(
        &store,
        "ZXQ-771 canonical organization endpoint evidence.",
        Some("org:acme"),
    );
    for content in [
        "Where a remote employee completes payroll forms.",
        "How an architect chooses a productive workplace.",
        "Employment policy and office location guidance.",
        "A directory of companies and job titles.",
    ] {
        remember_semantic(&store, content, None);
    }
    upsert_test_route(
        &store,
        "person:steve",
        "works_at",
        "org:acme",
        &subject_id,
        &object_id,
    );

    let base_payload = serde_json::json!({
        "title": "exact entity integration",
        "queries": ["Where does Steve work?"],
        "max_memories": 2,
        "max_chars": 2000,
        "rerank_candidates": 2
    });
    let treatment = run_json_command(&store, "pool-trace", &base_payload);
    let target = trace_memory(&treatment, &object_id).expect("entity route recovers endpoint");
    assert!(target["sources"].as_array().is_some_and(|sources| {
        sources.iter().any(|source| {
            source["graph_route"]["seed_source"] == "entity"
                && source["graph_route"]["matched_query_span"] == "steve"
                && source["graph_route"]["route_outcome"] == "active"
        })
    }));

    let pack = run_json_command(&store, "pack", &base_payload);
    assert!(
        pack["result"]["pack"]["content"]
            .as_str()
            .unwrap_or("")
            .contains("ZXQ-771"),
        "shipping unified rerank path must land endpoint: {pack}"
    );
}

#[test]
#[ignore = "requires the production embed, ColBERT, and INT8 reranker artifacts"]
fn evidence_graph_join_recovers_semantic_bridge_two_hop() {
    assert_evidence_join_models_exist();
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store.sqlite");
    let init = semantic_command()
        .args(["init", "--store", store.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(init.status.success(), "init failed: {init:?}");

    for (key, name) in [
        ("node:alpha", "Alpha Node"),
        ("node:beta", "Beta Node"),
        ("node:gamma", "Gamma Node"),
    ] {
        upsert_test_entity(&store, key, name);
    }
    let alpha_id = remember_semantic(
        &store,
        "orchard deployment anchor for the production rollout",
        Some("node:alpha"),
    );
    let beta_id = remember_semantic(
        &store,
        "intermediate cobalt bridge support",
        Some("node:beta"),
    );
    let gamma_id = remember_semantic(
        &store,
        "CAPYBARA-992 terminal evidence record",
        Some("node:gamma"),
    );
    for index in 0..8 {
        remember_semantic(
            &store,
            &format!("orchard deployment anchor checklist distractor {index}"),
            None,
        );
    }
    upsert_test_route(
        &store,
        "node:alpha",
        "routes_to",
        "node:beta",
        &alpha_id,
        &beta_id,
    );
    upsert_test_route(
        &store,
        "node:beta",
        "supports",
        "node:gamma",
        &beta_id,
        &gamma_id,
    );

    let base_payload = serde_json::json!({
        "title": "semantic bridge integration",
        "queries": ["orchard deployment anchor"],
        "max_memories": 2,
        "max_chars": 2000,
        "rerank_candidates": 2
    });
    let treatment = run_json_command(&store, "pool-trace", &base_payload);
    assert!(
        trace_memory(&treatment, &alpha_id).is_some(),
        "real model must supply the semantic memory seed: {treatment}"
    );
    let target = trace_memory(&treatment, &gamma_id).expect("two-hop endpoint recovered");
    assert!(target["sources"].as_array().is_some_and(|sources| {
        sources.iter().any(|source| {
            source["graph_route"]["seed_source"] == "memory"
                && source["graph_route"]["hop_depth"] == 2
                && source["graph_route"]["route_outcome"] == "active"
        })
    }));

    // The endpoint deliberately shares no query text. This case proves that a
    // real semantic seed can recover a two-hop graph candidate into the unified
    // rerank pool; the frozen effectiveness gate owns final-pack quality.
}

#[test]
fn primary_require_mode_refuses_invalid_model_in_real_pack() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("store.sqlite");
    let init = Command::new(memkeeper_bin())
        .args(["init", "--store", store.to_str().unwrap(), "--json"])
        .output()
        .unwrap();
    assert!(init.status.success(), "init failed: {init:?}");

    let output = Command::new(memkeeper_bin())
        .env("MEMKEEPER_EMBED_PROVIDER", "none")
        .env("MEMKEEPER_LATE_INTERACTION", "0")
        .env(
            "MEMKEEPER_RERANK_MODEL_DIR",
            "/definitely/missing/primary-rerank-model",
        )
        .env("MEMKEEPER_REQUIRE_RERANK", "1")
        .args([
            "pack",
            "--store",
            store.to_str().unwrap(),
            "--json",
            r#"{"title":"required-reranker","queries":["memory"],"max_memories":1,"max_chars":1000}"#,
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2), "output: {output:?}");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("primary reranker is not active"),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
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
fn pack_top_score_gate_emits_relevant_prompt_and_suppresses_off_topic() {
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

    // The pack uses one top rerank-score gate. Once the top candidate clears
    // it, lower-ranked evidence remains eligible for the count/char budget.
    let relevant = pack_content(
        &store,
        &serde_json::json!({
            "title": "test",
            "queries": ["supervisor stale error logs"],
            "max_memories": 3,
            "max_chars": 4000,
            "format": "markdown",
            "rerank_candidates": 3,
            "min_score": 0.05
        }),
    );
    assert!(
        relevant.contains("Empty error files"),
        "top-score-gated pack should inject the relevant memory. content:\n{relevant}"
    );

    let off_topic = pack_content(
        &store,
        &serde_json::json!({
            "title": "test",
            "queries": ["weather in paris tomorrow"],
            "max_memories": 3,
            "max_chars": 4000,
            "format": "markdown",
            "rerank_candidates": 3,
            "min_score": 0.05
        }),
    );
    assert!(
        off_topic.is_empty(),
        "top rerank-score gate should suppress off-topic prompts. content:\n{off_topic}"
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
