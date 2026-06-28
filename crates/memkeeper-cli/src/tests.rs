//! Unit tests for the memkeeper CLI binary.

use super::{
    backup_request_from_json, batch_search_request_from_json, diagnostic_store_candidate_from,
    dream_request_from_json, entity_search_request_from_json, entity_upsert_request_from_json,
    export_request_from_json, forget_request_from_json, get_request_from_json,
    graph_context_request_from_json, graph_neighbors_request_from_json, history_request_from_json,
    hook_query_text, import_request_from_json, json_string, memory_list_request_from_json,
    pack_expansion_options_from_json, pack_request_from_json, parse_backup_args,
    parse_batch_search_args, parse_doctor_args, parse_dream_args, parse_entity_search_args,
    parse_entity_upsert_args, parse_export_args, parse_forget_args, parse_graph_context_args,
    parse_graph_neighbors_args, parse_history_args, parse_hook_flags, parse_import_args,
    parse_memory_list_args, parse_pack_args, parse_relationship_upsert_args, parse_remember_args,
    parse_serve_args, parse_serve_request, parse_silo_list_args, parse_space_create_args,
    parse_space_list_args, parse_stats_args, parse_store_args,
    relationship_upsert_request_from_json, remember_request_from_json, rerank_request_from_json,
    search_request_from_json, serve_line_response, silo_list_request_from_json,
    space_create_request_from_json, space_record_json, CliError, SemanticModels, SpaceRecord,
    DEFAULT_PROMOTE_THRESHOLD, PROJECT_STORE_RELATIVE_PATH,
};
use std::path::PathBuf;

#[test]
fn semantic_unavailable_error_is_retryable_with_stable_code() {
    let err = CliError::SemanticUnavailable("query embedding failed: 429".to_string());
    assert_eq!(err.code().as_str(), "semantic_unavailable");
    assert!(
        err.retryable(),
        "transient embedder failures should be retryable"
    );
    assert_eq!(err.exit_code(), 1);
}

#[test]
fn hook_query_text_truncates_on_char_boundary() {
    // Short prompts pass through unchanged.
    assert_eq!(hook_query_text("hello"), "hello");

    // A long multibyte prompt must truncate to 500 chars without panicking;
    // a raw byte slice at index 500 would split a 4-byte code point.
    let multibyte = "\u{1f600}".repeat(600);
    let truncated = hook_query_text(&multibyte);
    assert_eq!(truncated.chars().count(), 500);
    assert!(truncated.chars().all(|c| c == '\u{1f600}'));
}

#[test]
fn parse_hook_flags_rejects_missing_and_flaglike_values() {
    assert_eq!(parse_hook_flags(&[]).unwrap(), (None, None));
    assert_eq!(
        parse_hook_flags(&[
            "--store".to_string(),
            "/tmp/s.sqlite".to_string(),
            "--sock".to_string(),
            "/tmp/d.sock".to_string(),
        ])
        .unwrap(),
        (
            Some("/tmp/s.sqlite".to_string()),
            Some("/tmp/d.sock".to_string())
        )
    );
    // `--store` must not swallow the following `--sock` flag as its value.
    assert!(parse_hook_flags(&["--store".to_string(), "--sock".to_string()]).is_err());
    assert!(parse_hook_flags(&["--store".to_string()]).is_err());
    assert!(parse_hook_flags(&["--bogus".to_string()]).is_err());
}

#[test]
fn json_string_escapes_control_characters() {
    assert_eq!(json_string("a\"b\\c\n"), "\"a\\\"b\\\\c\\n\"");
}

#[test]
fn parse_init_defaults_store_when_absent() {
    // `--store` is optional: a bare `memkeeper init` resolves the store from
    // MEMKEEPER_STORE or the ~/.memkeeper default instead of erroring. The exact
    // path depends on ambient env, so it is intentionally not asserted here
    // (mirrors the parse_doctor_args env-resolution case above).
    parse_store_args(&["--json".to_string()]).expect("defaults when --store absent");

    // An explicit --store flag still wins over any env/default resolution.
    let parsed = parse_store_args(&["--store".to_string(), "custom.sqlite".to_string()])
        .expect("explicit flag parses");
    assert_eq!(parsed.store, PathBuf::from("custom.sqlite"));
}

#[test]
fn parse_serve_supports_stdio_and_strict_envelopes() {
    parse_serve_args(&["--stdio".to_string()]).expect("serve parses");
    let missing = parse_serve_args(&[]).expect_err("a mode is required");
    assert!(missing.to_string().contains("serve requires one of"));

    let request = parse_serve_request(
        r#"{"protocol_version":"memkeeper.v0.1","request_id":"r1","command":"search","store_path":"store.sqlite","payload":{"query":"deterministic"}}"#,
    )
    .expect("serve request parses");
    assert_eq!(request.request_id.as_deref(), Some("r1"));
    assert_eq!(request.command_name, "search");
    assert_eq!(request.store_path, Some(PathBuf::from("store.sqlite")));
    assert!(request.payload_json.contains("deterministic"));

    let unknown = parse_serve_request(r#"{"command":"search","payload":{},"extra":1}"#)
        .expect_err("unknown field rejected");
    assert!(unknown.to_string().contains("unknown field"));

    let failure = serve_line_response(
        r#"{"request_id":"bad1","command":"search","payload":[]}"#,
        std::time::Instant::now(),
        &SemanticModels::for_serve(),
    );
    assert!(failure.contains("\"ok\":false"));
    assert!(failure.contains("\"request_id\":\"bad1\""));
    assert!(failure.contains("\"command\":\"search\""));
}

#[test]
fn parse_doctor_supports_optional_store_and_no_indexes() {
    // Store resolution precedence is env-independent, so it is tested via the
    // pure resolver to keep the suite hermetic regardless of ambient
    // MEMKEEPER_STORE / PI_MEMKEEPER_STORE.
    assert_eq!(
        diagnostic_store_candidate_from(None, None),
        (
            PathBuf::from(PROJECT_STORE_RELATIVE_PATH),
            "project_hint".to_string()
        )
    );
    assert_eq!(
        diagnostic_store_candidate_from(Some("@/tmp/store.sqlite"), None),
        (
            PathBuf::from("/tmp/store.sqlite"),
            "MEMKEEPER_STORE".to_string()
        )
    );
    assert_eq!(
        diagnostic_store_candidate_from(None, Some("/tmp/pi.sqlite")).1,
        "PI_MEMKEEPER_STORE"
    );

    // Parsing without --store succeeds; the resolved source depends on
    // ambient env, so it is intentionally not asserted here.
    let parsed = parse_doctor_args(&["--json".to_string()]).expect("parse succeeds");
    assert!(!parsed.include_indexes);

    // An explicit --store flag overrides env resolution.
    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--include-indexes".to_string(),
    ];
    let parsed = parse_doctor_args(&args).expect("parse succeeds");
    assert_eq!(parsed.store, PathBuf::from("store.sqlite"));
    assert_eq!(parsed.store_source, "flag");
    assert!(parsed.include_indexes);
}

#[test]
fn doctor_reports_semantic_models_check() {
    // The semantic-readiness check is the onboarding signal (does the user need
    // `pull-models`?). Guard that doctor emits it so a refactor can't silently
    // drop it. Present in both semantic and lexical-only builds.
    let args = parse_doctor_args(&[
        "--store".to_string(),
        "/nonexistent/doctor-semantic-check.sqlite".to_string(),
    ])
    .expect("parse succeeds");
    let (json, _) = crate::output::doctor_result_json(&args);
    assert!(
        json.contains("semantic.models"),
        "doctor output must include the semantic.models check: {json}"
    );
}

#[test]
fn parse_stats_supports_no_indexes() {
    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--no-indexes".to_string(),
    ];
    let parsed = parse_stats_args(&args).expect("parse succeeds");
    assert!(!parsed.include_indexes);
}

#[test]
fn space_list_and_silo_list_parse_flags() {
    let space_args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
    ];
    let space_list = parse_space_list_args(&space_args).expect("space-list parses");
    assert_eq!(space_list.store.to_string_lossy(), "store.sqlite");

    let silo_args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--space".to_string(),
        "project-notes".to_string(),
        "--json".to_string(),
    ];
    let silo_list = parse_silo_list_args(&silo_args).expect("silo-list parses");
    assert_eq!(silo_list.request.space.as_deref(), Some("project-notes"));

    let json_silo_args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
        r#"{"space":"workspace-memory"}"#.to_string(),
    ];
    let json_silo = parse_silo_list_args(&json_silo_args).expect("silo-list json parses");
    assert_eq!(json_silo.request.space.as_deref(), Some("workspace-memory"));
}

#[test]
fn space_create_json_and_flags_parse() {
    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
        r#"{"name":"project-notes","display_name":"Project Notes","description":"Project notes","default_silo":"long-term","ontology":"notes","config":{"owner":"test"},"if_not_exists":true}"#.to_string(),
    ];
    let parsed = parse_space_create_args(&args).expect("space-create json parses");
    assert_eq!(parsed.request.name, "project-notes");
    assert_eq!(parsed.request.default_silo.as_deref(), Some("long-term"));
    assert!(parsed
        .request
        .config_json
        .expect("config")
        .contains("owner"));
    assert!(parsed.request.if_not_exists);

    let flag_args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--name".to_string(),
        "flag-space".to_string(),
        "--default-silo".to_string(),
        "durable".to_string(),
        "--if-not-exists".to_string(),
    ];
    let flags = parse_space_create_args(&flag_args).expect("space-create flags parse");
    assert_eq!(flags.request.name, "flag-space");
    assert_eq!(flags.request.default_silo.as_deref(), Some("durable"));
    assert!(flags.request.if_not_exists);
}

#[test]
fn space_record_json_bounds_large_fields_and_preserves_json_validity() {
    let record = SpaceRecord {
        name: "large-space".to_string(),
        display_name: Some("Large Space".to_string()),
        description: Some("d".repeat(8_000)),
        default_silo: "durable".to_string(),
        ontology: Some("o".repeat(32_000)),
        config_json: Some(format!("{{\"blob\":\"{}\"}}", "x".repeat(3_000))),
        created_at: "2026-05-25T00:00:00.000Z".to_string(),
        updated_at: "2026-05-25T00:00:00.000Z".to_string(),
        memory_count: 0,
        active_count: 0,
        silo_count: 3,
    };
    let json = space_record_json(&record, Some(true));
    assert!(
        json.len() < 5_000,
        "space JSON was too large: {}",
        json.len()
    );
    assert!(json.contains("\"description_truncated\":true"));
    assert!(json.contains("\"ontology_truncated\":true"));
    assert!(json.contains("\"config_truncated\":true"));
    assert!(super::parse_json(&json).is_ok());

    let invalid = SpaceRecord {
        config_json: Some("not-json".to_string()),
        ..record
    };
    let invalid_json = space_record_json(&invalid, None);
    assert!(invalid_json.contains("\"config\":null"));
    assert!(super::parse_json(&invalid_json).is_ok());
}

#[test]
fn space_and_silo_json_reject_unknown_fields() {
    let space_error = space_create_request_from_json(r#"{"name":"x","unknown":1}"#)
        .expect_err("unknown rejected");
    assert!(space_error.to_string().contains("unknown field"));
    let silo_error =
        silo_list_request_from_json(r#"{"space":"x","unknown":1}"#).expect_err("unknown rejected");
    assert!(silo_error.to_string().contains("unknown field"));
}

#[test]
fn remember_json_parses_source_and_arrays() {
    let request = remember_request_from_json(
        r#"{"content":"lesson: test parser","tags":["a","b"],"source":{"type":"manual","source_episode_id":"src1"},"dry_run":true}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.content, "lesson: test parser");
    assert_eq!(request.tags, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(request.source_episode_id.as_deref(), Some("src1"));
    assert!(request
        .source_ref_json
        .expect("source")
        .contains("\"manual\""));
    assert!(request.dry_run);
}

#[test]
fn json_parses_text_embedding_3_small_sized_arrays() {
    let embedding = std::iter::repeat_n("0.0", 1_536)
        .collect::<Vec<_>>()
        .join(",");
    let remember = remember_request_from_json(&format!(
        r#"{{"content":"fact: embedding parser smoke","embedding":[{embedding}]}}"#
    ))
    .expect("1536-dim remember embedding parses");
    assert_eq!(remember.embedding.expect("embedding").len(), 1_536);

    let search = search_request_from_json(&format!(
        r#"{{"query":"embedding parser smoke","semantic_fallback":"fallback","embedding":[{embedding}]}}"#
    ))
    .expect("1536-dim search embedding parses");
    assert_eq!(search.embedding.expect("embedding").len(), 1_536);
}

#[test]
fn remember_json_derive_keys_fills_missing_keys() {
    let request = remember_request_from_json(
        r#"{"content":"decision: use SQLite as the canonical store","derive_keys":true}"#,
    )
    .expect("parse succeeds");
    assert_eq!(
        request.entity_key.as_deref(),
        Some("auto:use-sqlite-canonical-store")
    );
    assert!(request
        .claim_key
        .as_deref()
        .is_some_and(|claim| claim.starts_with("auto:")));
}

#[test]
fn remember_json_derive_keys_does_not_override_caller_keys() {
    let request = remember_request_from_json(
        r#"{"content":"fact: voyage trains on free-tier data","derive_keys":true,"entity_key":"voyage","claim_key":"voyage-policy"}"#,
    )
    .expect("parse succeeds");
    // Caller-supplied keys always win over derivation.
    assert_eq!(request.entity_key.as_deref(), Some("voyage"));
    assert_eq!(request.claim_key.as_deref(), Some("voyage-policy"));
}

#[test]
fn remember_json_without_derive_keys_leaves_keys_unset() {
    let request =
        remember_request_from_json(r#"{"content":"fact: no keys here"}"#).expect("parse succeeds");
    assert_eq!(request.entity_key, None);
    assert_eq!(request.claim_key, None);
}

#[test]
fn remember_json_rejects_unknown_fields() {
    let error = remember_request_from_json(r#"{"content":"ok","sourcee":{"type":"manual"}}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn remember_json_rejects_duplicate_keys_and_excessive_depth() {
    let duplicate = remember_request_from_json(r#"{"content":"ok","content":"again"}"#)
        .expect_err("duplicate key should fail");
    assert!(duplicate.to_string().contains("duplicate JSON object key"));

    let mut nested = String::new();
    for _ in 0..70 {
        nested.push('[');
    }
    nested.push_str("null");
    for _ in 0..70 {
        nested.push(']');
    }
    let too_deep = remember_request_from_json(&format!(r#"{{"content":"ok","tags":{nested}}}"#))
        .expect_err("excessive depth should fail");
    assert!(too_deep.to_string().contains("nesting depth"));
}

#[test]
fn remember_json_parses_surrogate_pairs() {
    let request = remember_request_from_json(r#"{"content":"emoji \ud83d\ude00"}"#)
        .expect("surrogate pair should parse");
    assert_eq!(request.content, "emoji 😀");
}

#[test]
fn search_json_parses_filters_and_bounds() {
    let request = search_request_from_json(
        r#"{"query":"sqlite memory","filters":{"spaces":["workspace-memory"],"kinds":["decision"],"tags":["search"]},"limit":5,"offset":1,"snippet_chars":42,"include_content":true,"include_source":false,"semantic_fallback":"disabled","lexical_fallback":"disabled"}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.query, "sqlite memory");
    assert_eq!(request.filters.kinds, vec!["decision".to_string()]);
    assert_eq!(request.limit, 5);
    assert_eq!(request.offset, 1);
    assert_eq!(request.snippet_chars, 42);
    assert!(request.include_content);
    assert!(!request.include_source);
    assert_eq!(request.semantic_fallback, "disabled");
    assert_eq!(request.lexical_fallback, "disabled");

    let default_source = search_request_from_json(r#"{"query":"sqlite memory"}"#)
        .expect("parse default source succeeds");
    assert!(!default_source.include_source);
    assert_eq!(default_source.lexical_fallback, "conservative");
}

#[test]
fn search_json_rejects_unknown_filter() {
    let error = search_request_from_json(r#"{"query":"sqlite","filters":{"surprise":["x"]}}"#)
        .expect_err("unknown filter should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn entity_upsert_json_and_flags_parse() {
    let request = entity_upsert_request_from_json(
        r#"{"space":"workspace-memory","entity_key":"project:memkeeper","entity_type":"Project","canonical_name":"Memkeeper","aliases":["fm"],"status":"active","confidence":0.75,"source_episode_id":"src1","metadata":{"owner":"cli"},"include_source":false}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.space.as_deref(), Some("workspace-memory"));
    assert_eq!(request.entity_key, "project:memkeeper");
    assert_eq!(request.entity_type.as_deref(), Some("Project"));
    assert_eq!(request.canonical_name, "Memkeeper");
    assert_eq!(request.aliases, vec!["fm".to_string()]);
    assert_eq!(request.status.as_deref(), Some("active"));
    assert!((request.confidence - 0.75).abs() < f64::EPSILON);
    assert_eq!(request.source_episode_id.as_deref(), Some("src1"));
    assert!(request
        .metadata_json
        .expect("metadata")
        .contains("\"owner\""));
    assert!(!request.include_source);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
        r#"{"entity_key":"project:memkeeper","canonical_name":"Memkeeper"}"#.to_string(),
    ];
    let parsed = parse_entity_upsert_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.entity_key, "project:memkeeper");
    assert_eq!(parsed.request.canonical_name, "Memkeeper");
}

#[test]
fn relationship_upsert_json_and_flags_parse() {
    let request = relationship_upsert_request_from_json(
        r#"{"space":"workspace-memory","subject_entity_key":"project:memkeeper","relation_type":"uses","object_entity_id":"ent_sqlite","memory_id":"mem1","source_episode_id":"src1","status":"active","confidence":0.6,"observed_at":"2026-05-28T00:00:00Z","valid_from":"2026-05-28T00:00:00Z","valid_to":"2026-12-31T00:00:00Z","metadata":{"owner":"cli"},"include_source":true}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.space.as_deref(), Some("workspace-memory"));
    assert_eq!(
        request.subject_entity_key.as_deref(),
        Some("project:memkeeper")
    );
    assert_eq!(request.relation_type, "uses");
    assert_eq!(request.object_entity_id.as_deref(), Some("ent_sqlite"));
    assert_eq!(request.memory_id.as_deref(), Some("mem1"));
    assert_eq!(request.source_episode_id.as_deref(), Some("src1"));
    assert_eq!(request.status.as_deref(), Some("active"));
    assert!((request.confidence - 0.6).abs() < f64::EPSILON);
    assert_eq!(request.observed_at.as_deref(), Some("2026-05-28T00:00:00Z"));
    assert!(request.metadata_json.expect("metadata").contains("owner"));
    assert!(request.include_source);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
        r#"{"subject_entity_key":"project:memkeeper","relation_type":"uses","object_entity_key":"component:sqlite"}"#.to_string(),
    ];
    let parsed = parse_relationship_upsert_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.relation_type, "uses");
    assert_eq!(
        parsed.request.object_entity_key.as_deref(),
        Some("component:sqlite")
    );
}

#[test]
fn entity_search_json_and_flags_parse() {
    let request = entity_search_request_from_json(
        r#"{"space":"workspace-memory","query":"memkeeper","entity_key":"project:memkeeper","entity_types":["MemorySubject"],"statuses":["active"],"limit":5,"offset":1,"include_source":false}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.space.as_deref(), Some("workspace-memory"));
    assert_eq!(request.query.as_deref(), Some("memkeeper"));
    assert_eq!(request.entity_key.as_deref(), Some("project:memkeeper"));
    assert_eq!(request.entity_types, vec!["MemorySubject".to_string()]);
    assert_eq!(request.statuses, vec!["active".to_string()]);
    assert_eq!(request.limit, 5);
    assert_eq!(request.offset, 1);
    assert!(!request.include_source);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
        r#"{"query":"memkeeper"}"#.to_string(),
    ];
    let parsed = parse_entity_search_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.query.as_deref(), Some("memkeeper"));
}

#[test]
fn graph_neighbors_json_and_flags_parse() {
    let request = graph_neighbors_request_from_json(
        r#"{"space":"workspace-memory","entity_key":"project:memkeeper","depth":2,"relation_types":["related_to"],"statuses":["active"],"max_edges":25,"include_tombstoned":false,"include_source":false}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.space.as_deref(), Some("workspace-memory"));
    assert_eq!(request.entity_key.as_deref(), Some("project:memkeeper"));
    assert_eq!(request.depth, 2);
    assert_eq!(request.relation_types, vec!["related_to".to_string()]);
    assert_eq!(request.statuses, vec!["active".to_string()]);
    assert_eq!(request.max_edges, 25);
    assert!(!request.include_tombstoned);
    assert!(!request.include_source);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
        r#"{"entity_id":"ent_1"}"#.to_string(),
    ];
    let parsed = parse_graph_neighbors_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.entity_id.as_deref(), Some("ent_1"));
}

#[test]
fn graph_context_json_and_flags_parse() {
    let request = graph_context_request_from_json(
        r#"{"space":"workspace-memory","entity_key":"project:memkeeper","depth":2,"relation_types":["uses"],"statuses":["active"],"max_edges":25,"max_memories":8,"max_chars":2000,"include_tombstoned":false,"include_source":true}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.space.as_deref(), Some("workspace-memory"));
    assert_eq!(request.entity_key.as_deref(), Some("project:memkeeper"));
    assert_eq!(request.depth, 2);
    assert_eq!(request.relation_types, vec!["uses".to_string()]);
    assert_eq!(request.statuses, vec!["active".to_string()]);
    assert_eq!(request.max_edges, 25);
    assert_eq!(request.max_memories, 8);
    assert_eq!(request.max_chars, 2000);
    assert!(!request.include_tombstoned);
    assert!(request.include_source);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
        r#"{"entity_id":"ent_1","max_memories":3}"#.to_string(),
    ];
    let parsed = parse_graph_context_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.entity_id.as_deref(), Some("ent_1"));
    assert_eq!(parsed.request.max_memories, 3);
}

#[test]
fn graph_json_rejects_unknown_fields() {
    let error = entity_upsert_request_from_json(
        r#"{"entity_key":"project:x","canonical_name":"X","surprise":true}"#,
    )
    .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
    let error = relationship_upsert_request_from_json(
        r#"{"subject_entity_key":"project:x","relation_type":"uses","object_entity_key":"project:y","surprise":true}"#,
    )
    .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
    let error = entity_search_request_from_json(r#"{"query":"memkeeper","surprise":true}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
    let error = graph_neighbors_request_from_json(r#"{"entity_key":"project:x","extra":1}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
    let error = graph_context_request_from_json(r#"{"entity_key":"project:x","extra":1}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn memory_list_json_and_flags_parse() {
    let request = memory_list_request_from_json(
        r#"{"filters":{"statuses":["active"],"tags":["memory"]},"limit":25,"offset":2,"snippet_chars":80,"include_content":true,"include_source":false,"order":"observed_desc"}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.filters.statuses, vec!["active".to_string()]);
    assert_eq!(request.filters.tags, vec!["memory".to_string()]);
    assert_eq!(request.limit, 25);
    assert_eq!(request.offset, 2);
    assert_eq!(request.snippet_chars, 80);
    assert!(request.include_content);
    assert!(!request.include_source);
    assert_eq!(request.order, "observed_desc");

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--limit".to_string(),
        "12".to_string(),
        "--include-source".to_string(),
        "--order".to_string(),
        "created_desc".to_string(),
        "--json".to_string(),
    ];
    let parsed = parse_memory_list_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.limit, 12);
    assert!(parsed.request.include_source);
    assert_eq!(parsed.request.order, "created_desc");
}

#[test]
fn memory_list_json_rejects_unknown_fields() {
    let error = memory_list_request_from_json(r#"{"limit":10,"surprise":true}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn batch_search_json_and_flags_parse() {
    let request = batch_search_request_from_json(
        r#"{"queries":[{"name":"storage","query":"sqlite storage","limit":3}],"common_filters":{"statuses":["active"]},"limit":5,"include_source":false}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.queries.len(), 1);
    assert_eq!(request.queries[0].name.as_deref(), Some("storage"));
    assert_eq!(request.queries[0].limit, Some(3));
    assert_eq!(request.common_filters.statuses, vec!["active".to_string()]);
    assert_eq!(request.limit, 5);
    assert!(!request.include_source);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
        r#"{"queries":[{"query":"sqlite"}]}"#.to_string(),
    ];
    let parsed = parse_batch_search_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.queries[0].query, "sqlite");
    assert!(!parsed.request.include_source);
}

#[test]
fn batch_search_json_rejects_unknown_fields() {
    let error = batch_search_request_from_json(r#"{"queries":[{"query":"sqlite","extra":true}]}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn pack_json_and_flags_parse() {
    let request = pack_request_from_json(
        r#"{"title":"impl","queries":["sqlite","search"],"filters":{"kinds":["decision"]},"max_memories":4,"max_chars":1000,"format":"markdown","query_expansion":true,"thread_expansion":true}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.title, "impl");
    assert_eq!(
        request.queries,
        vec!["sqlite".to_string(), "search".to_string()]
    );
    assert_eq!(request.filters.kinds, vec!["decision".to_string()]);
    assert_eq!(request.max_memories, 4);
    assert_eq!(request.max_chars, 1000);
    let expansion = pack_expansion_options_from_json(
        r#"{"title":"impl","queries":["sqlite"],"query_expansion":true,"thread_expansion":true,"max_query_variants":5,"max_thread_seeds":2,"max_thread_neighbors":4}"#,
    )
    .expect("expansion parses");
    assert!(expansion.query_expansion);
    assert!(expansion.thread_expansion);
    assert_eq!(expansion.max_query_variants, 5);
    assert_eq!(expansion.max_thread_seeds, 2);
    assert_eq!(expansion.max_thread_neighbors, 4);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--json".to_string(),
        r#"{"title":"impl","queries":["sqlite"],"query_expansion":true}"#.to_string(),
    ];
    let parsed = parse_pack_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.title, "impl");
    assert_eq!(parsed.request.max_memories, 10);
    assert!(parsed.expansion.query_expansion);
}

#[test]
fn pack_json_rejects_unknown_fields() {
    let error = pack_request_from_json(r#"{"title":"impl","queries":["sqlite"],"surprise":true}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn rerank_json_parses_query_and_documents() {
    let request = rerank_request_from_json(
        r#"{"query":"rust async","documents":["Tokio 1.40 released","Best coffee in Seattle"]}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.query, "rust async");
    assert_eq!(
        request.documents,
        vec![
            "Tokio 1.40 released".to_string(),
            "Best coffee in Seattle".to_string()
        ]
    );
}

#[test]
fn rerank_json_allows_empty_documents() {
    let request = rerank_request_from_json(r#"{"query":"q","documents":[]}"#)
        .expect("empty documents are valid");
    assert!(request.documents.is_empty());
}

#[test]
fn rerank_json_requires_query() {
    let error =
        rerank_request_from_json(r#"{"documents":["a"]}"#).expect_err("missing query should fail");
    assert!(error.to_string().contains("query"));
}

#[test]
fn rerank_json_rejects_unknown_fields() {
    let error = rerank_request_from_json(r#"{"query":"q","documents":["a"],"surprise":true}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn get_json_parses_options() {
    let request = get_request_from_json(
        r#"{"id":"mem_1","include_history":true,"include_links":false,"include_source":false}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.id, "mem_1");
    assert!(request.options.include_history);
    assert!(!request.options.include_links);
    assert!(!request.options.include_source);

    let default_source =
        get_request_from_json(r#"{"id":"mem_1"}"#).expect("parse default source succeeds");
    assert!(!default_source.options.include_source);
}

#[test]
fn get_json_rejects_unknown_fields() {
    let error = get_request_from_json(r#"{"id":"mem_1","surprise":true}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn forget_json_and_flags_parse() {
    let request = forget_request_from_json(
        r#"{"id":"mem_1","reason":"stale","mode":"tombstone","dry_run":true}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.id, "mem_1");
    assert_eq!(request.reason.as_deref(), Some("stale"));
    assert_eq!(request.mode, "tombstone");
    assert!(request.dry_run);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--id".to_string(),
        "mem_2".to_string(),
        "--reason".to_string(),
        "done".to_string(),
        "--dry-run".to_string(),
        "--json".to_string(),
    ];
    let parsed = parse_forget_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.id, "mem_2");
    assert_eq!(parsed.request.reason.as_deref(), Some("done"));
    assert!(parsed.request.dry_run);
}

#[test]
fn forget_json_rejects_unknown_fields() {
    let error = forget_request_from_json(r#"{"id":"mem_1","surprise":true}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn history_json_and_flags_parse() {
    let request = history_request_from_json(r#"{"id":"mem_1","limit":25,"include_source":false}"#)
        .expect("parse succeeds");
    assert_eq!(request.id, "mem_1");
    assert_eq!(request.options.limit, 25);
    assert!(!request.options.include_source);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--id".to_string(),
        "mem_2".to_string(),
        "--limit".to_string(),
        "10".to_string(),
        "--no-source".to_string(),
        "--json".to_string(),
    ];
    let parsed = parse_history_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.id, "mem_2");
    assert_eq!(parsed.options.limit, 10);
    assert!(!parsed.options.include_source);

    let default_source =
        history_request_from_json(r#"{"id":"mem_1"}"#).expect("parse default source succeeds");
    assert!(!default_source.options.include_source);
}

#[test]
fn history_json_rejects_unknown_fields() {
    let error = history_request_from_json(r#"{"id":"mem_1","source":false}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn export_json_and_flags_parse() {
    let request = export_request_from_json(r#"{"output":"export.jsonl","format":"jsonl"}"#)
        .expect("parse succeeds");
    assert_eq!(request.output_path.to_string_lossy(), "export.jsonl");
    assert_eq!(request.format, "jsonl");

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--output".to_string(),
        "export.jsonl".to_string(),
        "--json".to_string(),
    ];
    let parsed = parse_export_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.output_path.to_string_lossy(), "export.jsonl");
    assert_eq!(parsed.request.format, "jsonl");
}

#[test]
fn export_json_rejects_unknown_fields() {
    let error = export_request_from_json(r#"{"output":"export.jsonl","filters":{}}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn import_json_and_flags_parse() {
    let request = import_request_from_json(
        r#"{"in":"export.jsonl","format":"jsonl","dry_run":true,"conflict_policy":"fail_if_exists"}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.input_path.to_string_lossy(), "export.jsonl");
    assert_eq!(request.format, "jsonl");
    assert!(request.dry_run);
    assert_eq!(request.conflict_policy, "fail_if_exists");

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--input".to_string(),
        "export.jsonl".to_string(),
        "--dry-run".to_string(),
        "--json".to_string(),
    ];
    let parsed = parse_import_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.input_path.to_string_lossy(), "export.jsonl");
    assert!(parsed.request.dry_run);
}

#[test]
fn import_json_rejects_unknown_fields() {
    let error = import_request_from_json(r#"{"input":"export.jsonl","mode":"merge"}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn dream_json_and_flags_parse() {
    let request = dream_request_from_json(
        r#"{"space":"workspace-memory","silos":["durable"],"tasks":["expire","dedupe","graph"],"max_memories":42,"dry_run":true,"include_pinned":true}"#,
    )
    .expect("parse succeeds");
    assert_eq!(request.space.as_deref(), Some("workspace-memory"));
    assert_eq!(request.silos, vec!["durable"]);
    assert_eq!(request.tasks, vec!["expire", "dedupe", "graph"]);
    assert_eq!(request.max_memories, 42);
    assert!(request.dry_run);
    assert!(request.include_pinned);

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--space".to_string(),
        "workspace-memory".to_string(),
        "--silo".to_string(),
        "durable,long-term".to_string(),
        "--tasks".to_string(),
        "expire,reindex".to_string(),
        "--max-memories".to_string(),
        "25".to_string(),
        "--dry-run".to_string(),
        "--include-pinned".to_string(),
        "--json".to_string(),
    ];
    let parsed = parse_dream_args(&args).expect("parse flags succeeds");
    assert_eq!(parsed.request.space.as_deref(), Some("workspace-memory"));
    assert_eq!(parsed.request.silos, vec!["durable", "long-term"]);
    assert_eq!(parsed.request.tasks, vec!["expire", "reindex"]);
    assert_eq!(parsed.request.max_memories, 25);
    assert!(parsed.request.dry_run);
    assert!(parsed.request.include_pinned);
}

#[test]
fn dream_request_parses_promote_threshold() {
    let parsed =
        dream_request_from_json(r#"{"tasks":["promote"],"promote_threshold":5}"#).expect("parse");
    assert_eq!(parsed.promote_threshold, 5);

    let defaulted = dream_request_from_json(r#"{"tasks":["promote"]}"#).expect("parse default");
    assert_eq!(defaulted.promote_threshold, DEFAULT_PROMOTE_THRESHOLD);
}

#[test]
fn dream_json_rejects_unknown_fields() {
    let error = dream_request_from_json(r#"{"tasks":["expire"],"output":"x"}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn backup_json_and_flags_parse() {
    let request = backup_request_from_json(r#"{"out":"backup.sqlite"}"#).expect("parse succeeds");
    assert_eq!(request.output_path.to_string_lossy(), "backup.sqlite");
    assert_eq!(request.format, "sqlite");

    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--output".to_string(),
        "backup.sqlite".to_string(),
        "--json".to_string(),
    ];
    let parsed = parse_backup_args(&args).expect("parse flags succeeds");
    assert_eq!(
        parsed.request.output_path.to_string_lossy(),
        "backup.sqlite"
    );
    assert_eq!(parsed.request.format, "sqlite");
}

#[test]
fn backup_json_rejects_unknown_fields() {
    let error = backup_request_from_json(r#"{"output":"backup.sqlite","out_dir":"dir"}"#)
        .expect_err("unknown field should fail");
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn parse_remember_requires_request_json() {
    let args = ["--store".to_string(), "store.sqlite".to_string()];
    let error = parse_remember_args(&args).expect_err("missing request");
    assert!(error.to_string().contains("missing remember request JSON"));
}

#[test]
fn max_combine_rerank_scores_takes_per_candidate_max() {
    use super::max_combine_rerank_scores;
    // Candidates 0 and 2 have summaries; candidate 0's summary wins, 2's loses.
    let combined = max_combine_rerank_scores(&[0.2, 0.9, 0.5], &[0.8, 0.1], &[0, 2]);
    assert_eq!(combined, vec![0.8, 0.9, 0.5]);
    // No summaries: content scores pass through untouched.
    let untouched = max_combine_rerank_scores(&[0.3, 0.4], &[], &[]);
    assert_eq!(untouched, vec![0.3, 0.4]);
}

#[test]
fn default_build_includes_semantic() {
    // Guard against the silent-FTS footgun: a plain `cargo build`/`cargo test`
    // must produce a semantic binary. If `semantic` is ever dropped from the
    // crate's default features, this fails loudly here instead of degrading
    // production retrieval to FTS-only at runtime.
    //
    // A compile-time cfg guard, not `assert!(cfg!(...))`: under the (correct)
    // semantic build the feature is a constant `true`, which `clippy -D warnings`
    // rejects as `assertions_on_constants`. With `embed` on, the body compiles
    // away to nothing and passes; without it, the test panics loudly.
    #[cfg(not(feature = "embed"))]
    panic!(
        "default build must include semantic features (embed). \
         Re-add `semantic` to [features] default in memkeeper-cli/Cargo.toml."
    );
}

#[test]
fn semantic_guard_decides_serve_posture() {
    use super::serve::{semantic_guard, SemanticGuard};
    assert_eq!(semantic_guard(true, false), SemanticGuard::Ok);
    assert_eq!(semantic_guard(true, true), SemanticGuard::Ok);
    assert_eq!(semantic_guard(false, false), SemanticGuard::Degraded);
    assert_eq!(semantic_guard(false, true), SemanticGuard::Refuse);
}

#[test]
fn one_shot_search_fails_closed_only_when_required_and_unembeddable() {
    use super::must_refuse_search_without_semantic;
    // Refuse: require_semantic set, no caller embedding, no active embedder.
    // This is the gap the external review found in one-shot CLI search.
    assert!(must_refuse_search_without_semantic(false, false, true));
    // Allowed: a caller-supplied embedding means semantic search can still run.
    assert!(!must_refuse_search_without_semantic(true, false, true));
    // Allowed: an active embedder (local OR remote, e.g. the hosted api build)
    // will embed the query, so the require flag is satisfied.
    assert!(!must_refuse_search_without_semantic(false, true, true));
    // Allowed: without the require flag, fall back to BM25 as before.
    assert!(!must_refuse_search_without_semantic(false, false, false));
}

/// Parse a JSON payload through the real request parser for `command`.
/// Returns Ok(()) on success so schema docs can be validated against the
/// actual parsers (any drift between the two fails these tests).
fn parse_payload_for(command: &str, json: &str) -> Result<(), CliError> {
    use crate::requests::*;
    match command {
        "space-create" => space_create_request_from_json(json).map(|_| ()),
        "silo-list" => silo_list_request_from_json(json).map(|_| ()),
        "remember" => remember_request_from_json(json).map(|_| ()),
        "search" => search_request_from_json(json).map(|_| ()),
        "entity-upsert" => entity_upsert_request_from_json(json).map(|_| ()),
        "relationship-upsert" => relationship_upsert_request_from_json(json).map(|_| ()),
        "entity-merge" => entity_merge_request_from_json(json).map(|_| ()),
        "entity-search" => entity_search_request_from_json(json).map(|_| ()),
        "graph-neighbors" => graph_neighbors_request_from_json(json).map(|_| ()),
        "graph-context" => graph_context_request_from_json(json).map(|_| ()),
        "memory-list" => memory_list_request_from_json(json).map(|_| ()),
        "batch-search" => batch_search_request_from_json(json).map(|_| ()),
        "pack" => pack_request_from_json(json).map(|_| ()),
        "get" => get_request_from_json(json).map(|_| ()),
        "forget" => forget_request_from_json(json).map(|_| ()),
        "verify" => verify_request_from_json(json).map(|_| ()),
        "recall-log" => recall_log_request_from_json(json).map(|_| ()),
        "history" => history_request_from_json(json).map(|_| ()),
        "export" => export_request_from_json(json).map(|_| ()),
        "import" => import_request_from_json(json).map(|_| ()),
        "dream" => dream_request_from_json(json).map(|_| ()),
        "backup" => backup_request_from_json(json).map(|_| ()),
        "candidate-submit" => candidate_submit_request_from_json(json).map(|_| ()),
        "candidate-list" => candidate_list_request_from_json(json).map(|_| ()),
        "candidate-approve" => candidate_approve_request_from_json(json).map(|_| ()),
        "candidate-reject" => candidate_reject_request_from_json(json).map(|_| ()),
        "ingest" => ingest_request_from_json(json).map(|_| ()),
        "document-search" => document_search_request_from_json(json).map(|_| ()),
        "document-get" => document_get_request_from_json(json).map(|_| ()),
        "document-duplicates" => document_duplicates_request_from_json(json).map(|_| ()),
        "promotion-candidates" => promotion_candidates_request_from_json(json).map(|_| ()),
        "mark-extracted" => mark_extracted_request_from_json(json).map(|_| ()),
        "document-prune" => document_prune_request_from_json(json).map(|_| ()),
        other => panic!("schema documents command '{other}' with no test parser mapping"),
    }
}

/// A type-appropriate dummy JSON value for building an all-fields payload.
fn dummy_value(field: &crate::schema::FieldDoc) -> &'static str {
    match field.ty {
        "string" => "\"x\"",
        "string[]" => "[\"x\"]",
        "int" => "1",
        "float" => "1.0",
        "bool" => "true",
        "object" => "{}",
        "float[]" => "[0.1]",
        "float[][]" => "[[0.1]]",
        "object[]" => match field.name {
            "queries" => "[{\"query\":\"x\"}]",
            "events" => "[{\"memory_id\":\"x\",\"kind\":\"x\"}]",
            _ => "[{}]",
        },
        other => panic!("unknown schema field type '{other}'"),
    }
}

#[test]
fn schema_examples_parse_through_real_parsers() {
    // Every documented example must be a valid payload for its command.
    // If a parser's required fields or accepted keys change, the example
    // stops parsing and this test fails — catching doc drift.
    for schema in crate::schema::COMMAND_SCHEMAS {
        parse_payload_for(schema.command, schema.example).unwrap_or_else(|err| {
            panic!(
                "schema example for '{}' failed to parse: {err}\n  example: {}",
                schema.command, schema.example
            )
        });
    }
}

#[test]
fn schema_documents_only_accepted_fields() {
    // Build a payload containing every documented field (dummy values) and
    // parse it. A documented-but-rejected field (e.g. renamed in the parser)
    // surfaces as an "unknown field" error here.
    for schema in crate::schema::COMMAND_SCHEMAS {
        let body = schema
            .fields
            .iter()
            .map(|f| format!("\"{}\":{}", f.name, dummy_value(f)))
            .collect::<Vec<_>>()
            .join(",");
        let payload = format!("{{{body}}}");
        parse_payload_for(schema.command, &payload).unwrap_or_else(|err| {
            panic!(
                "all-fields payload for '{}' failed to parse: {err}\n  payload: {payload}",
                schema.command
            )
        });
    }
}

#[test]
fn schema_index_renders_valid_json() {
    // The hand-rolled JSON serializer must emit parseable JSON for every command.
    for schema in crate::schema::COMMAND_SCHEMAS {
        let json = crate::schema::render_schema_json_for_test(schema);
        crate::json::parse_json(&json)
            .unwrap_or_else(|err| panic!("schema JSON for '{}' is invalid: {err}", schema.command));
    }
    // Reviewer-critical commands must be documented.
    for command in ["remember", "search", "pack", "dream", "batch-search"] {
        assert!(
            crate::schema::schema_for(command).is_some(),
            "command '{command}' must have a documented schema"
        );
    }
}

#[test]
fn pack_result_json_contract_is_stable() {
    // The pack wire format is a documented contract (docs/operations.md).
    // This pins the top-level shape so a serializer change can't silently
    // break consumers that inject packs into Codex/Claude/etc.
    let report = super::PackReport {
        title: "context".to_string(),
        format: "markdown".to_string(),
        content: "- a\n- b".to_string(),
        memory_ids: vec!["mem_1".to_string(), "mem_2".to_string()],
        scores: vec![0.9, 0.5],
        truncated: false,
        top_score: Some(0.9),
    };
    let json = super::pack_result_json(&report);
    let value = crate::json::parse_json(&json).expect("pack json parses");
    let object = value.as_object().expect("top-level object");
    let pack = object
        .get("pack")
        .and_then(|v| v.as_object())
        .expect("pack key is an object");
    for field in [
        "title",
        "format",
        "content",
        "memory_ids",
        "scores",
        "truncated",
        "top_score",
    ] {
        assert!(
            pack.get(field).is_some(),
            "pack output must contain '{field}'"
        );
    }
}

#[test]
fn apply_flag_forces_commit() {
    // `--apply` is the explicit opposite of `--dry-run` (alias of --commit):
    // a later --apply wins, and for dream it also overrides a JSON dry_run:true.
    let dream = parse_dream_args(&[
        "--apply".to_string(),
        "--json".to_string(),
        r#"{"tasks":["dedupe"],"dry_run":true}"#.to_string(),
    ])
    .expect("dream parse");
    assert!(
        !dream.request.dry_run,
        "--apply must override JSON dry_run:true"
    );

    // forget flag path: --dry-run then --apply ends committed.
    let forget = parse_forget_args(&[
        "--id".to_string(),
        "mem_x".to_string(),
        "--dry-run".to_string(),
        "--apply".to_string(),
    ])
    .expect("forget parse");
    assert!(!forget.request.dry_run);

    // import flag path: same precedence.
    let import = parse_import_args(&[
        "--input".to_string(),
        "export.jsonl".to_string(),
        "--dry-run".to_string(),
        "--apply".to_string(),
    ])
    .expect("import parse");
    assert!(!import.request.dry_run);

    // Sanity: --dry-run alone still previews.
    let preview = parse_forget_args(&[
        "--id".to_string(),
        "mem_x".to_string(),
        "--dry-run".to_string(),
    ])
    .expect("forget parse");
    assert!(preview.request.dry_run);
}

#[test]
fn dream_flags_override_json_payload() {
    // Regression: `dream --task dedupe --dry-run --json "{}"` must honor the flags
    // rather than silently discarding them in favor of the (empty) JSON payload.
    // Dry-run trust depends on `--dry-run` always winning.
    let args = [
        "--store".to_string(),
        "store.sqlite".to_string(),
        "--task".to_string(),
        "dedupe".to_string(),
        "--dry-run".to_string(),
        "--json".to_string(),
        "{}".to_string(),
    ];
    let parsed = parse_dream_args(&args).expect("parse succeeds");
    assert_eq!(parsed.request.tasks, vec!["dedupe"]);
    assert!(parsed.request.dry_run);

    // Fields not set by a flag fall back to the JSON payload's values.
    let args = [
        "--dry-run".to_string(),
        "--json".to_string(),
        r#"{"tasks":["promote"],"max_memories":7}"#.to_string(),
    ];
    let parsed = parse_dream_args(&args).expect("parse succeeds");
    assert_eq!(parsed.request.tasks, vec!["promote"]);
    assert_eq!(parsed.request.max_memories, 7);
    assert!(parsed.request.dry_run);

    // `--commit` explicitly overrides a payload's dry_run:true back to false.
    let args = [
        "--commit".to_string(),
        "--json".to_string(),
        r#"{"tasks":["expire"],"dry_run":true}"#.to_string(),
    ];
    let parsed = parse_dream_args(&args).expect("parse succeeds");
    assert!(!parsed.request.dry_run);
}

#[test]
fn resolve_json_args_inlines_file_and_passes_through_literals() {
    use std::io::Write;

    // A literal JSON payload is left untouched.
    let literal = super::resolve_json_args(vec![
        "remember".to_string(),
        "--json".to_string(),
        r#"{"content":"x"}"#.to_string(),
    ])
    .expect("literal passes through");
    assert_eq!(literal[2], r#"{"content":"x"}"#);

    // `--json=@<file>` inlines the file contents (and the `=` form is preserved).
    let mut path = std::env::temp_dir();
    path.push(format!("memkeeper-json-arg-{}.json", std::process::id()));
    write!(
        std::fs::File::create(&path).expect("create temp"),
        r#"{{"content":"from file"}}"#
    )
    .expect("write temp");
    let from_file = super::resolve_json_args(vec![
        "remember".to_string(),
        format!("--json=@{}", path.display()),
    ])
    .expect("file resolves");
    assert_eq!(from_file[1], r#"--json={"content":"from file"}"#);
    let _ = std::fs::remove_file(&path);

    // A missing `@<file>` is a loud error, not a silent literal.
    assert!(super::resolve_json_args(vec![
        "remember".to_string(),
        "--json".to_string(),
        "@/no/such/memkeeper/file.json".to_string(),
    ])
    .is_err());
}
