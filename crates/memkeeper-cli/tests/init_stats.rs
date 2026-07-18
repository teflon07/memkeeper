#![forbid(unsafe_code)]

//! CLI smoke tests for deterministic store initialization and stats.

use std::{
    fs,
    io::Write,
    path::Path,
    path::PathBuf,
    process::{Command, Stdio},
    time::SystemTime,
};

use rusqlite::Connection;

#[test]
fn init_then_stats_returns_json_success() {
    let path = temp_store_path("init_then_stats_returns_json_success");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");
    let init_stdout = String::from_utf8(init.stdout).expect("valid utf8");
    assert!(init_stdout.contains("\"ok\":true"));
    assert!(init_stdout.contains("\"command\":\"init\""));
    assert!(init_stdout.contains("\"created\":true"));
    assert!(init_stdout.contains("\"spaces\":[\"workspace-memory\"]"));

    let stats = memkeeper_command()
        .args(["stats", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run stats");
    assert!(stats.status.success(), "stats failed: {stats:?}");
    let stats_stdout = String::from_utf8(stats.stdout).expect("valid utf8");
    assert!(stats_stdout.contains("\"ok\":true"));
    assert!(stats_stdout.contains("\"command\":\"stats\""));
    assert!(stats_stdout.contains("\"schema_version\":6"));
    assert!(stats_stdout.contains("\"space_count\":1"));
    assert!(stats_stdout.contains("\"silo_count\":2"));
    assert!(stats_stdout.contains("\"memory_count\":0"));
    assert!(stats_stdout.contains("\"indexes\":{"));

    let stats_without_indexes = memkeeper_command()
        .args(["stats", "--store"])
        .arg(&path)
        .args(["--json", "--no-indexes"])
        .output()
        .expect("run stats without indexes");
    assert!(stats_without_indexes.status.success());
    let stats_without_indexes_stdout =
        String::from_utf8(stats_without_indexes.stdout).expect("valid utf8");
    assert!(stats_without_indexes_stdout.contains("\"indexes\":null"));

    cleanup_store(&path);
}

#[test]
fn doctor_reports_missing_and_initialized_store() {
    let path = temp_store_path("doctor_reports_missing_and_initialized_store");
    cleanup_store(&path);

    let missing = memkeeper_command()
        .args(["doctor", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run doctor missing");
    assert!(missing.status.success(), "doctor failed: {missing:?}");
    let missing_stdout = String::from_utf8(missing.stdout).expect("valid utf8");
    assert!(missing_stdout.contains("\"ok\":true"));
    assert!(missing_stdout.contains("\"command\":\"doctor\""));
    assert!(missing_stdout.contains("\"status\":\"warning\""));
    assert!(missing_stdout.contains("\"state\":\"missing\""));
    assert!(missing_stdout.contains("\"code\":\"store_not_initialized\""));
    assert!(missing_stdout.contains("\"mutating\":false"));

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");

    let ready = memkeeper_command()
        .args(["doctor", "--store"])
        .arg(&path)
        .args(["--json", "--include-indexes"])
        .output()
        .expect("run doctor ready");
    assert!(ready.status.success(), "doctor failed: {ready:?}");
    let ready_stdout = String::from_utf8(ready.stdout).expect("valid utf8");
    assert!(ready_stdout.contains("\"status\":\"ok\""));
    assert!(ready_stdout.contains("\"state\":\"initialized\""));
    assert!(ready_stdout.contains("\"embedded_schema_required_objects\":true"));
    assert!(ready_stdout.contains("\"journal_mode\":\"wal\""));
    assert!(ready_stdout.contains("\"indexes\":{"));

    cleanup_store(&path);
}

#[test]
fn doctor_does_not_create_sqlite_sidecars() {
    let path = temp_store_path("doctor_does_not_create_sqlite_sidecars");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");

    let wal = sqlite_sidecar_path(&path, "-wal");
    let shm = sqlite_sidecar_path(&path, "-shm");
    assert!(!wal.exists(), "init fixture unexpectedly left WAL sidecar");
    assert!(!shm.exists(), "init fixture unexpectedly left SHM sidecar");

    let doctor = memkeeper_command()
        .args(["doctor", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run doctor");
    assert!(doctor.status.success(), "doctor failed: {doctor:?}");
    assert!(!wal.exists(), "doctor created WAL sidecar");
    assert!(!shm.exists(), "doctor created SHM sidecar");

    cleanup_store(&path);
}

#[test]
fn serve_stdio_handles_multiple_json_requests() {
    let path = temp_store_path("serve_stdio_handles_multiple_json_requests");
    cleanup_store(&path);

    let mut child = memkeeper_command()
        .args(["serve", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn serve");
    {
        let stdin = child.stdin.as_mut().expect("stdin available");
        writeln!(
            stdin,
            r#"{{"request_id":"serve-init","command":"init","store_path":"{}","payload":{{}}}}"#,
            path.display()
        )
        .expect("write init request");
        writeln!(
            stdin,
            r#"{{"request_id":"serve-remember","command":"remember","store_path":"{}","payload":{{"content":"decision: serve stdio works","entity_key":"project:serve-stdio"}}}}"#,
            path.display()
        )
        .expect("write remember request");
        writeln!(
            stdin,
            r#"{{"request_id":"serve-entity-upsert","command":"entity-upsert","store_path":"{}","payload":{{"entity_key":"project:serve-stdio","entity_type":"Project","canonical_name":"Serve stdio","aliases":["stdio smoke"]}}}}"#,
            path.display()
        )
        .expect("write entity-upsert request");
        writeln!(
            stdin,
            r#"{{"request_id":"serve-object-upsert","command":"entity-upsert","store_path":"{}","payload":{{"entity_key":"component:stdio-transport","entity_type":"Component","canonical_name":"stdio transport"}}}}"#,
            path.display()
        )
        .expect("write object entity-upsert request");
        writeln!(
            stdin,
            r#"{{"request_id":"serve-relationship-upsert","command":"relationship-upsert","store_path":"{}","payload":{{"subject_entity_key":"project:serve-stdio","relation_type":"uses","object_entity_key":"component:stdio-transport"}}}}"#,
            path.display()
        )
        .expect("write relationship-upsert request");
        writeln!(
            stdin,
            r#"{{"request_id":"serve-entity-search","command":"entity-search","store_path":"{}","payload":{{"entity_key":"project:serve-stdio","limit":5}}}}"#,
            path.display()
        )
        .expect("write entity-search request");
        writeln!(
            stdin,
            r#"{{"request_id":"serve-graph-neighbors","command":"graph-neighbors","store_path":"{}","payload":{{"entity_key":"project:serve-stdio","depth":1,"max_edges":10}}}}"#,
            path.display()
        )
        .expect("write graph-neighbors request");
        writeln!(
            stdin,
            r#"{{"request_id":"serve-graph-context","command":"graph-context","store_path":"{}","payload":{{"entity_key":"project:serve-stdio","depth":1,"max_edges":10,"max_memories":5,"max_chars":1000}}}}"#,
            path.display()
        )
        .expect("write graph-context request");
    }
    let output = child.wait_with_output().expect("wait serve");
    assert!(output.status.success(), "serve failed: {output:?}");
    let stdout = String::from_utf8(output.stdout).expect("valid utf8");
    let lines = stdout.lines().collect::<Vec<_>>();
    assert_eq!(lines.len(), 8, "stdout was {stdout}");
    assert!(lines[0].contains("\"request_id\":\"serve-init\""));
    assert!(lines[0].contains("\"command\":\"init\""));
    assert!(lines[0].contains("\"ok\":true"));
    assert!(lines[0].contains("\"created\":true"));
    assert!(lines[1].contains("\"request_id\":\"serve-remember\""));
    assert!(lines[1].contains("\"command\":\"remember\""));
    assert!(lines[1].contains("serve stdio works"));
    assert!(lines[2].contains("\"request_id\":\"serve-entity-upsert\""));
    assert!(lines[2].contains("\"command\":\"entity-upsert\""));
    assert!(lines[2].contains("\"created\":false"));
    assert!(lines[2].contains("stdio smoke"));
    assert!(lines[3].contains("\"request_id\":\"serve-object-upsert\""));
    assert!(lines[3].contains("\"command\":\"entity-upsert\""));
    assert!(lines[3].contains("component:stdio-transport"));
    assert!(lines[4].contains("\"request_id\":\"serve-relationship-upsert\""));
    assert!(lines[4].contains("\"command\":\"relationship-upsert\""));
    assert!(lines[4].contains("\"relation_type\":\"uses\""));
    assert!(lines[5].contains("\"request_id\":\"serve-entity-search\""));
    assert!(lines[5].contains("\"command\":\"entity-search\""));
    assert!(lines[5].contains("project:serve-stdio"));
    assert!(lines[6].contains("\"request_id\":\"serve-graph-neighbors\""));
    assert!(lines[6].contains("\"command\":\"graph-neighbors\""));
    assert!(lines[6].contains("\"relationships\":["));
    assert!(lines[7].contains("\"request_id\":\"serve-graph-context\""));
    assert!(lines[7].contains("\"command\":\"graph-context\""));
    assert!(lines[7].contains("\"graph_context\""));
    assert!(lines[7].contains("serve stdio works"));

    cleanup_store(&path);
}

#[test]
fn serve_startup_notes_capture_adjudication_require_mode() {
    // With MEMKEEPER_CAPTURE_REQUIRE_ADJUDICATION set, serve must surface the fail-closed posture
    // once at startup on stderr (never stdout — that would corrupt the JSON-RPC stream). stdin is
    // piped and closed by wait_with_output (EOF), so serve exits cleanly.
    let child = memkeeper_command()
        .env("MEMKEEPER_CAPTURE_REQUIRE_ADJUDICATION", "1")
        .args(["serve", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve");
    let output = child.wait_with_output().expect("wait serve");
    assert!(output.status.success(), "serve failed: {output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("capture-adjudication require-mode ACTIVE"),
        "startup NOTE missing from stderr: {stderr}"
    );
    // The NOTE goes to stderr only; stdout stays clean (no request sent -> empty JSON stream).
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        !stdout.contains("require-mode"),
        "startup NOTE leaked into stdout JSON stream: {stdout}"
    );
}

#[test]
fn serve_startup_silent_on_capture_adjudication_by_default() {
    // Without the env var (permissive default), serve must NOT emit the adjudication NOTE — the
    // capture write-path is opt-in, so quiet is correct (no crying wolf on every serve).
    let child = memkeeper_command()
        .args(["serve", "--stdio"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn serve");
    let output = child.wait_with_output().expect("wait serve");
    assert!(output.status.success(), "serve failed: {output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("capture-adjudication require-mode"),
        "unexpected adjudication NOTE when require-mode is off: {stderr}"
    );
}

#[test]
fn space_and_silo_commands_manage_custom_space() {
    let path = temp_store_path("space_and_silo_commands_manage_custom_space");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success());

    let create = memkeeper_command()
        .args(["space-create", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"name":"project-notes","display_name":"Project Notes","description":"Project notes","default_silo":"long-term","config":{"owner":"cli"},"if_not_exists":true}"#,
        ])
        .output()
        .expect("run space-create");
    assert!(create.status.success(), "space-create failed: {create:?}");
    let create_stdout = String::from_utf8(create.stdout).expect("valid utf8");
    assert!(create_stdout.contains("\"command\":\"space-create\""));
    assert!(create_stdout.contains("\"name\":\"project-notes\""));
    assert!(create_stdout.contains("\"default_silo\":\"long-term\""));
    assert!(create_stdout.contains("\"created\":true"));

    let idempotent = memkeeper_command()
        .args(["space-create", "--store"])
        .arg(&path)
        .args(["--json", r#"{"name":"project-notes","if_not_exists":true}"#])
        .output()
        .expect("run idempotent space-create");
    assert!(idempotent.status.success());
    let idempotent_stdout = String::from_utf8(idempotent.stdout).expect("valid utf8");
    assert!(idempotent_stdout.contains("\"created\":false"));

    let space_list = memkeeper_command()
        .args(["space-list", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run space-list");
    assert!(
        space_list.status.success(),
        "space-list failed: {space_list:?}"
    );
    let space_stdout = String::from_utf8(space_list.stdout).expect("valid utf8");
    assert!(space_stdout.contains("\"command\":\"space-list\""));
    assert!(space_stdout.contains("\"name\":\"project-notes\""));
    assert!(space_stdout.contains("\"silo_count\":3"));

    let silo_list = memkeeper_command()
        .args(["silo-list", "--store"])
        .arg(&path)
        .args(["--space", "project-notes", "--json"])
        .output()
        .expect("run silo-list");
    assert!(
        silo_list.status.success(),
        "silo-list failed: {silo_list:?}"
    );
    let silo_stdout = String::from_utf8(silo_list.stdout).expect("valid utf8");
    assert!(silo_stdout.contains("\"command\":\"silo-list\""));
    assert!(silo_stdout.contains("\"space\":\"project-notes\""));
    assert!(silo_stdout.contains("\"name\":\"short-term\""));
    assert!(silo_stdout.contains("\"name\":\"long-term\""));
    assert!(silo_stdout.contains("\"is_default\":true"));

    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"space":"project-notes","content":"decision: custom space defaults to long-term"}"#,
        ])
        .output()
        .expect("run remember custom space");
    assert!(remember.status.success(), "remember failed: {remember:?}");
    let remember_stdout = String::from_utf8(remember.stdout).expect("valid utf8");
    assert!(remember_stdout.contains("\"space\":\"project-notes\""));
    assert!(remember_stdout.contains("\"silo\":\"long-term\""));

    cleanup_store(&path);
}

#[test]
fn init_refuses_unrelated_existing_database() {
    let path = temp_store_path("init_refuses_unrelated_existing_database");
    cleanup_store(&path);
    let connection = Connection::open(&path).expect("create unrelated sqlite database");
    connection
        .execute_batch("CREATE TABLE unrelated (id INTEGER PRIMARY KEY);")
        .expect("create unrelated table");
    drop(connection);

    let output = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("valid utf8");
    assert!(stdout.contains("\"ok\":false"));
    assert!(stdout.contains("\"code\":\"conflict\""));
    assert!(stdout.contains("non-memkeeper existing database"));

    cleanup_store(&path);
}

#[test]
fn init_rejects_sqlite_uri_path() {
    let path = temp_store_path("init_rejects_sqlite_uri_path");
    cleanup_store(&path);
    let uri = format!("file:{}", path.display());

    let output = memkeeper_command()
        .args(["init", "--store", &uri, "--json"])
        .output()
        .expect("run init");

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("valid utf8");
    assert!(stdout.contains("\"ok\":false"));
    assert!(stdout.contains("\"code\":\"invalid_request\""));
    assert!(!Path::new("file:").exists());

    cleanup_store(&path);
}

#[test]
#[allow(clippy::too_many_lines)]
fn remember_then_get_returns_memory_and_updates_stats() {
    let path = temp_store_path("remember_then_get_returns_memory_and_updates_stats");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success());

    let request = r#"{"content":"decision: remember get works","summary":"remember get works","tags":["memory","cli"],"source":{"type":"manual","adapter":"host"}}"#;
    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args(["--json", request])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");
    let remember_stdout = String::from_utf8(remember.stdout).expect("valid utf8");
    assert!(remember_stdout.contains("\"ok\":true"));
    assert!(remember_stdout.contains("\"command\":\"remember\""));
    assert!(remember_stdout.contains("\"kind\":\"decision\""));
    assert!(remember_stdout.contains("\"processing_status\":\"indexed\""));
    assert!(!remember_stdout.contains("\"adapter\":\"host\""));
    let memory_id = extract_json_string_after(&remember_stdout, "\"id\":\"");

    assert_duplicate_remember_candidate(&path);

    let get = memkeeper_command()
        .args(["get", "--store"])
        .arg(&path)
        .args(["--id", &memory_id, "--json", "--include-history"])
        .output()
        .expect("run get");
    assert!(get.status.success(), "get failed: {get:?}");
    let get_stdout = String::from_utf8(get.stdout).expect("valid utf8");
    assert!(get_stdout.contains("\"ok\":true"));
    assert!(get_stdout.contains("\"command\":\"get\""));
    assert!(get_stdout.contains("decision: remember get works"));
    assert!(get_stdout.contains("\"versions\":["));
    assert!(get_stdout.contains("\"events\":["));
    assert!(!get_stdout.contains("\"adapter\":\"host\""));

    let get_no_source = memkeeper_command()
        .args(["get", "--store"])
        .arg(&path)
        .args([
            "--id",
            &memory_id,
            "--json",
            "--include-history",
            "--no-source",
        ])
        .output()
        .expect("run get no source");
    assert!(get_no_source.status.success());
    let get_no_source_stdout = String::from_utf8(get_no_source.stdout).expect("valid utf8");
    assert!(!get_no_source_stdout.contains("\"adapter\":\"host\""));

    let search = memkeeper_command()
        .args(["search", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"query":"remember works","filters":{"kinds":["decision"]},"limit":5,"include_content":true,"include_source":false}"#,
        ])
        .output()
        .expect("run search");
    assert!(search.status.success(), "search failed: {search:?}");
    let search_stdout = String::from_utf8(search.stdout).expect("valid utf8");
    assert!(search_stdout.contains("\"command\":\"search\""));
    assert!(search_stdout.contains("\"strategy\":\"deterministic_fts_v0\""));
    assert!(search_stdout.contains("decision: remember get works"));
    assert!(!search_stdout.contains("\"adapter\":\"host\""));

    let memory_list = memkeeper_command()
        .args(["memory-list", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"filters":{"tags":["memory"]},"limit":5,"include_content":false,"include_source":false}"#,
        ])
        .output()
        .expect("run memory-list");
    assert!(
        memory_list.status.success(),
        "memory-list failed: {memory_list:?}"
    );
    let memory_list_stdout = String::from_utf8(memory_list.stdout).expect("valid utf8");
    assert!(memory_list_stdout.contains("\"command\":\"memory-list\""));
    assert!(memory_list_stdout.contains("\"strategy\":\"deterministic_list_v0\""));
    assert!(memory_list_stdout.contains(&memory_id));
    assert!(memory_list_stdout.contains("\"summary\":\"remember get works\""));
    assert!(!memory_list_stdout.contains("decision: remember get works"));
    assert!(!memory_list_stdout.contains("\"adapter\":\"host\""));

    let stats = memkeeper_command()
        .args(["stats", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run stats");
    assert!(stats.status.success());
    let stats_stdout = String::from_utf8(stats.stdout).expect("valid utf8");
    assert!(stats_stdout.contains("\"memory_count\":1"));
    assert!(stats_stdout.contains("\"active_count\":1"));
    assert!(stats_stdout.contains("\"fts_memory_rows\":1"));

    cleanup_store(&path);
}

fn assert_duplicate_remember_candidate(path: &Path) {
    let duplicate = memkeeper_command()
        .args(["remember", "--store"])
        .arg(path)
        .args([
            "--json",
            r#"{"content":"decision: remember get works","dry_run":true}"#,
        ])
        .output()
        .expect("run duplicate remember dry-run");
    assert!(
        duplicate.status.success(),
        "duplicate remember failed: {duplicate:?}"
    );
    let duplicate_stdout = String::from_utf8(duplicate.stdout).expect("valid utf8");
    assert!(duplicate_stdout.contains("\"dry_run\":true"));
    assert!(duplicate_stdout.contains("\"candidates\":[{"));
    assert!(duplicate_stdout.contains("\"relationship\":\"duplicate\""));
    assert!(duplicate_stdout.contains("\"content_sha256\""));
    assert!(duplicate_stdout.contains("\"candidates_truncated\":false"));
    assert!(!duplicate_stdout.contains("\"adapter\":\"host\""));
}

#[test]
#[allow(clippy::too_many_lines)]
fn dream_expires_reindexes_and_reports_duplicates() {
    let path = temp_store_path("dream_expires_reindexes_and_reports_duplicates");
    cleanup_store(&path);
    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success());

    let expired = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"content":"temporary cli dream expiry","expires_at":"2000-01-01T00:00:00.000Z"}"#,
        ])
        .output()
        .expect("run remember expired");
    assert!(
        expired.status.success(),
        "remember expired failed: {expired:?}"
    );
    let expired_stdout = String::from_utf8(expired.stdout).expect("valid utf8");
    let expired_id = extract_json_string_after(&expired_stdout, "\"id\":\"");

    for content in ["duplicate cli dream", "duplicate cli dream"] {
        let request = format!(
            r#"{{"content":{},"derive_keys":false}}"#,
            json_string(content)
        );
        let remember = memkeeper_command()
            .args(["remember", "--store"])
            .arg(&path)
            .args(["--json", &request])
            .output()
            .expect("run remember duplicate");
        assert!(
            remember.status.success(),
            "remember duplicate failed: {remember:?}"
        );
    }

    let dry_run = memkeeper_command()
        .args(["dream", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"tasks":["expire","reindex","dedupe"],"dry_run":true,"max_memories":10}"#,
        ])
        .output()
        .expect("run dream dry-run");
    assert!(
        dry_run.status.success(),
        "dream dry-run failed: {dry_run:?}"
    );
    let dry_run_stdout = String::from_utf8(dry_run.stdout).expect("valid utf8");
    assert!(dry_run_stdout.contains("\"command\":\"dream\""));
    assert!(dry_run_stdout.contains("\"dry_run\":true"));
    assert!(dry_run_stdout.contains("\"journaled\":false"));
    assert!(dry_run_stdout.contains("\"expired\":1"));
    assert!(dry_run_stdout.contains("\"memory_rows\":3"));
    assert!(dry_run_stdout.contains("\"proposals\":[{"));

    let get_before_commit = memkeeper_command()
        .args(["get", "--store"])
        .arg(&path)
        .args(["--id", &expired_id, "--json"])
        .output()
        .expect("get before commit");
    assert!(get_before_commit.status.success());
    let before_stdout = String::from_utf8(get_before_commit.stdout).expect("valid utf8");
    assert!(before_stdout.contains("\"status\":\"active\""));

    let committed = memkeeper_command()
        .args(["dream", "--store"])
        .arg(&path)
        .args([
            "--tasks",
            "expire,reindex,dedupe",
            "--max-memories",
            "10",
            "--json",
        ])
        .output()
        .expect("run dream commit");
    assert!(
        committed.status.success(),
        "dream commit failed: {committed:?}"
    );
    let committed_stdout = String::from_utf8(committed.stdout).expect("valid utf8");
    assert!(committed_stdout.contains("\"dry_run\":false"));
    assert!(committed_stdout.contains("\"journaled\":true"));
    assert!(committed_stdout.contains("\"expired\":1"));
    assert!(committed_stdout.contains("\"memory_rows\":3"));
    assert!(committed_stdout.contains("\"total_count\":2"));

    let get_after_commit = memkeeper_command()
        .args(["get", "--store"])
        .arg(&path)
        .args(["--id", &expired_id, "--json", "--include-history"])
        .output()
        .expect("get after commit");
    assert!(get_after_commit.status.success());
    let after_stdout = String::from_utf8(get_after_commit.stdout).expect("valid utf8");
    assert!(after_stdout.contains("\"status\":\"expired\""));
    assert!(after_stdout.contains("\"type\":\"dream\""));

    cleanup_store(&path);
}

fn json_string(value: &str) -> String {
    serde_json::to_string(value).expect("JSON string")
}

#[test]
fn batch_search_and_pack_return_compact_context() {
    let path = temp_store_path("batch_search_and_pack_return_compact_context");
    cleanup_store(&path);
    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success());

    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"content":"decision: remember get works","summary":"remember get works","tags":["memory"],"source":{"type":"manual","adapter":"host"}}"#,
        ])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");
    let memory_id = extract_json_string_after(
        &String::from_utf8(remember.stdout).expect("valid utf8"),
        "\"id\":\"",
    );

    let batch = memkeeper_command()
        .args(["batch-search", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"queries":[{"name":"decision","query":"remember works","limit":3}],"common_filters":{"tags":["memory"]},"include_source":false}"#,
        ])
        .output()
        .expect("run batch-search");
    assert!(batch.status.success(), "batch-search failed: {batch:?}");
    let batch_stdout = String::from_utf8(batch.stdout).expect("valid utf8");
    assert!(batch_stdout.contains("\"command\":\"batch-search\""));
    assert!(batch_stdout.contains("\"name\":\"decision\""));
    assert!(batch_stdout.contains(&memory_id));
    assert!(!batch_stdout.contains("\"adapter\":\"host\""));

    let pack = memkeeper_command()
        .args(["pack", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"title":"cli memory","queries":["remember works"],"filters":{"tags":["memory"]},"max_memories":5,"max_chars":1000}"#,
        ])
        .output()
        .expect("run pack");
    assert!(pack.status.success(), "pack failed: {pack:?}");
    let pack_stdout = String::from_utf8(pack.stdout).expect("valid utf8");
    assert!(pack_stdout.contains("\"command\":\"pack\""));
    assert!(
        pack_stdout.contains("## Retrieved Memory"),
        "pack stdout: {pack_stdout}"
    );
    assert!(pack_stdout.contains(&memory_id));
    assert!(!pack_stdout.contains("\"adapter\":\"host\""));

    cleanup_store(&path);
}

#[test]
fn forget_then_history_tombstones_memory() {
    let path = temp_store_path("forget_then_history_tombstones_memory");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success());

    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"content":"decision: cli forget works","source":{"type":"manual","adapter":"host"}}"#,
        ])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");
    let remember_stdout = String::from_utf8(remember.stdout).expect("valid utf8");
    let memory_id = extract_json_string_after(&remember_stdout, "\"id\":\"");

    let forget = memkeeper_command()
        .args(["forget", "--store"])
        .arg(&path)
        .args(["--id", &memory_id, "--reason", "stale", "--json"])
        .output()
        .expect("run forget");
    assert!(forget.status.success(), "forget failed: {forget:?}");
    let forget_stdout = String::from_utf8(forget.stdout).expect("valid utf8");
    assert!(forget_stdout.contains("\"command\":\"forget\""));
    assert!(forget_stdout.contains("\"old_status\":\"active\""));
    assert!(forget_stdout.contains("\"new_status\":\"tombstoned\""));

    let search = memkeeper_command()
        .args(["search", "--store"])
        .arg(&path)
        .args(["--json", r#"{"query":"forget works","limit":5}"#])
        .output()
        .expect("run default search");
    assert!(search.status.success());
    let search_stdout = String::from_utf8(search.stdout).expect("valid utf8");
    assert!(search_stdout.contains("\"results\":[]"));

    let history = memkeeper_command()
        .args(["history", "--store"])
        .arg(&path)
        .args(["--id", &memory_id, "--json", "--no-source"])
        .output()
        .expect("run history");
    assert!(history.status.success(), "history failed: {history:?}");
    let history_stdout = String::from_utf8(history.stdout).expect("valid utf8");
    assert!(history_stdout.contains("\"command\":\"history\""));
    assert!(history_stdout.contains("\"current_status\":\"tombstoned\""));
    assert!(history_stdout.contains("\"type\":\"forget\""));
    assert!(!history_stdout.contains("\"adapter\":\"host\""));

    let stats = memkeeper_command()
        .args(["stats", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run stats");
    assert!(stats.status.success());
    let stats_stdout = String::from_utf8(stats.stdout).expect("valid utf8");
    assert!(stats_stdout.contains("\"memory_count\":1"));
    assert!(stats_stdout.contains("\"active_count\":0"));

    cleanup_store(&path);
}

#[test]
fn export_and_backup_write_files() {
    let path = temp_store_path("export_and_backup_write_files");
    let export_path =
        temp_store_path("export_and_backup_write_files_export").with_extension("jsonl");
    let backup_path = temp_store_path("export_and_backup_write_files_backup");
    let import_path = temp_store_path("export_and_backup_write_files_import");
    cleanup_store(&path);
    cleanup_store(&export_path);
    cleanup_store(&backup_path);
    cleanup_store(&import_path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success());

    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"content":"decision: cli export backup works","source":{"type":"manual","adapter":"host"}}"#,
        ])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");

    let export = memkeeper_command()
        .args(["export", "--store"])
        .arg(&path)
        .args(["--output"])
        .arg(&export_path)
        .arg("--json")
        .output()
        .expect("run export");
    assert!(export.status.success(), "export failed: {export:?}");
    let export_stdout = String::from_utf8(export.stdout).expect("valid utf8");
    assert!(export_stdout.contains("\"command\":\"export\""));
    assert!(export_stdout.contains("\"format\":\"jsonl\""));
    assert!(export_stdout.contains("\"sha256\":"));
    let export_text = fs::read_to_string(&export_path).expect("read export");
    assert!(export_text.contains("\"type\":\"header\""));
    assert!(export_text.contains("\"table\":\"memories\""));
    assert!(export_text.contains("adapter"));

    let backup = memkeeper_command()
        .args(["backup", "--store"])
        .arg(&path)
        .args(["--output"])
        .arg(&backup_path)
        .arg("--json")
        .output()
        .expect("run backup");
    assert!(backup.status.success(), "backup failed: {backup:?}");
    let backup_stdout = String::from_utf8(backup.stdout).expect("valid utf8");
    assert!(backup_stdout.contains("\"command\":\"backup\""));
    assert!(backup_stdout.contains("\"format\":\"sqlite\""));
    assert!(backup_stdout.contains("\"sha256\":"));
    assert!(backup_path.exists());
    assert!(!PathBuf::from(format!("{}-wal", backup_path.display())).exists());
    assert!(!PathBuf::from(format!("{}-shm", backup_path.display())).exists());
    assert!(!PathBuf::from(format!("{}-journal", backup_path.display())).exists());

    let backup_stats = memkeeper_command()
        .args(["stats", "--store"])
        .arg(&backup_path)
        .arg("--json")
        .output()
        .expect("run backup stats");
    assert!(
        backup_stats.status.success(),
        "backup stats failed: {backup_stats:?}"
    );
    let backup_stats_stdout = String::from_utf8(backup_stats.stdout).expect("valid utf8");
    assert!(backup_stats_stdout.contains("\"memory_count\":1"));

    assert_cli_imports_export(&export_path, &import_path);

    cleanup_store(&path);
    cleanup_store(&export_path);
    cleanup_store(&backup_path);
    cleanup_store(&import_path);
}

fn assert_cli_imports_export(export_path: &Path, import_path: &Path) {
    let import = memkeeper_command()
        .args(["import", "--store"])
        .arg(import_path)
        .args(["--input"])
        .arg(export_path)
        .arg("--json")
        .output()
        .expect("run import");
    assert!(import.status.success(), "import failed: {import:?}");
    let import_stdout = String::from_utf8(import.stdout).expect("valid utf8");
    assert!(import_stdout.contains("\"command\":\"import\""));
    assert!(import_stdout.contains("\"format\":\"jsonl\""));
    assert!(import_stdout.contains("\"dry_run\":false"));
    assert!(import_stdout.contains("\"fts_memory_rows\":1"));

    let import_stats = memkeeper_command()
        .args(["stats", "--store"])
        .arg(import_path)
        .arg("--json")
        .output()
        .expect("run import stats");
    assert!(
        import_stats.status.success(),
        "import stats failed: {import_stats:?}"
    );
    let import_stats_stdout = String::from_utf8(import_stats.stdout).expect("valid utf8");
    assert!(import_stats_stdout.contains("\"memory_count\":1"));
    assert!(import_stats_stdout.contains("\"fts_memory_rows\":1"));
}

#[test]
fn stats_missing_store_returns_protocol_error() {
    let path = temp_store_path("stats_missing_store_returns_protocol_error");
    cleanup_store(&path);

    let output = memkeeper_command()
        .args(["stats", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run stats");

    assert!(!output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("valid utf8");
    assert!(stdout.contains("\"ok\":false"));
    assert!(stdout.contains("\"code\":\"store_not_initialized\""));
    assert!(stdout.contains("memkeeper init --store <path> --json"));
}

#[test]
fn verify_stamps_metadata_via_cli() {
    let path = temp_store_path("verify_stamps_metadata_via_cli");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");

    // Remember a memory with metadata_json as a JSON string containing a verified_against pointer.
    let remember_req = r#"{"content":"fact: the gate is 0.62","metadata_json":"{\"verified_against\":\"~/.zshrc:GATE\"}"}"#;
    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args(["--json", remember_req])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");
    let remember_stdout = String::from_utf8(remember.stdout).expect("valid utf8");
    let memory_id = extract_json_string_after(&remember_stdout, "\"id\":\"");

    // Verify the memory with an explicit now timestamp.
    let verify_req = format!(r#"{{"memory_id":"{memory_id}","now":"2026-06-08T20:00:00Z"}}"#);
    let verify = memkeeper_command()
        .args(["verify", "--store"])
        .arg(&path)
        .args(["--json", &verify_req])
        .output()
        .expect("run verify");
    assert!(verify.status.success(), "verify failed: {verify:?}");
    let verify_stdout = String::from_utf8(verify.stdout).expect("valid utf8");
    assert!(
        verify_stdout.contains("\"ok\":true"),
        "verify ok: {verify_stdout}"
    );
    assert!(
        verify_stdout.contains("\"command\":\"verify\""),
        "command field: {verify_stdout}"
    );
    assert!(
        verify_stdout.contains("\"verified_at\":\"2026-06-08T20:00:00.000Z\""),
        "verified_at in response: {verify_stdout}"
    );

    cleanup_store(&path);
}

#[test]
fn search_result_includes_verified_against_from_metadata_json() {
    let path = temp_store_path("search_result_includes_verified_against_from_metadata_json");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");

    // Remember a memory with metadata_json containing verified_against.
    let remember_req = r#"{"content":"fact: the gate value is 0.62","metadata_json":"{\"verified_against\":\"~/.zshrc:GATE\"}"}"#;
    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args(["--json", remember_req])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");
    let remember_stdout = String::from_utf8(remember.stdout).expect("valid utf8");
    let memory_id = extract_json_string_after(&remember_stdout, "\"id\":\"");

    // Search for the memory and assert verified_against appears in output.
    let search = memkeeper_command()
        .args(["search", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"query":"gate value","limit":5,"include_content":true}"#,
        ])
        .output()
        .expect("run search");
    assert!(search.status.success(), "search failed: {search:?}");
    let search_stdout = String::from_utf8(search.stdout).expect("valid utf8");
    assert!(
        search_stdout.contains(&memory_id),
        "memory_id in search results: {search_stdout}"
    );
    assert!(
        search_stdout.contains("\"verified_against\":\"~/.zshrc:GATE\""),
        "verified_against in search output: {search_stdout}"
    );
    // Sanity-check that other fields are not shifted (kind is still present and correct).
    assert!(
        search_stdout.contains("\"kind\":\"fact\""),
        "kind field correct (not shifted): {search_stdout}"
    );
    assert!(
        search_stdout.contains("\"silo\":"),
        "silo field present (not shifted): {search_stdout}"
    );

    cleanup_store(&path);
}

#[test]
fn search_result_includes_freshness_label_for_volatile_with_verified_against() {
    let path = temp_store_path("search_freshness_volatile");
    cleanup_store(&path);

    // Init the store.
    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");

    // Insert a succeeded dream_run so last_synthesis_run returns a known timestamp.
    {
        let conn = Connection::open(&path).expect("open db");
        conn.execute_batch(
            "INSERT INTO dream_runs (id, space_name, status, started_at, finished_at) \
             VALUES ('d1','workspace-memory','succeeded','2026-06-08T03:00:00Z','2026-06-08T03:05:00Z')",
        )
        .expect("insert dream_run");
    }

    // Remember a volatile (short-term) memory with verified_against but NO verified_at → stale.
    let remember_req = r#"{"content":"fact: the gate is 0.62","silo":"short-term","metadata_json":"{\"verified_against\":\"~/.zshrc:GATE\"}"}"#;
    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args(["--json", remember_req])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");

    // Search and assert freshness:"verify" is present.
    let search = memkeeper_command()
        .args(["search", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"query":"gate","limit":5,"include_content":true}"#,
        ])
        .output()
        .expect("run search");
    assert!(search.status.success(), "search failed: {search:?}");
    let search_stdout = String::from_utf8(search.stdout).expect("valid utf8");
    assert!(
        search_stdout.contains("\"freshness\":\"verify\""),
        "expected freshness:verify in search output: {search_stdout}"
    );

    cleanup_store(&path);
}

#[test]
fn search_result_no_freshness_for_durable_even_with_verified_against() {
    let path = temp_store_path("search_freshness_durable");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");

    // Insert a succeeded dream_run.
    {
        let conn = Connection::open(&path).expect("open db");
        conn.execute_batch(
            "INSERT INTO dream_runs (id, space_name, status, started_at, finished_at) \
             VALUES ('d1','workspace-memory','succeeded','2026-06-08T03:00:00Z','2026-06-08T03:05:00Z')",
        )
        .expect("insert dream_run");
    }

    // Remember a durable memory with verified_against — freshness must NOT appear.
    let remember_req = r#"{"content":"fact: the gate is 0.62","silo":"durable","metadata_json":"{\"verified_against\":\"~/.zshrc:GATE\"}"}"#;
    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args(["--json", remember_req])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");

    let search = memkeeper_command()
        .args(["search", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"query":"gate","limit":5,"include_content":true}"#,
        ])
        .output()
        .expect("run search");
    assert!(search.status.success(), "search failed: {search:?}");
    let search_stdout = String::from_utf8(search.stdout).expect("valid utf8");
    assert!(
        !search_stdout.contains("\"freshness\""),
        "durable memory must NOT have freshness field: {search_stdout}"
    );

    cleanup_store(&path);
}

fn extract_json_string_after(output: &str, marker: &str) -> String {
    let start = output.find(marker).expect("marker exists") + marker.len();
    let rest = &output[start..];
    let end = rest.find('"').expect("closing quote exists");
    rest[..end].to_string()
}

#[test]
fn pack_without_reranker_returns_content_but_empty_scores() {
    // BM25-fallback safety: this default build has no cross-encoder, so the pack
    // must inject content (memory_ids present) but expose NO scores -- the
    // auto-retrieve hook then logs NULL scores and the rerank-scale promote
    // floor excludes them (no over-promotion on the wrong score scale).
    let path = temp_store_path("pack_without_reranker_returns_content_but_empty_scores");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");

    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"content":"decision: sqlite retrieval uses deterministic bm25 indexes","kind":"decision"}"#,
        ])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");

    let pack = memkeeper_command()
        .args(["pack", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"title":"t","queries":["sqlite retrieval bm25"],"max_memories":5,"rerank_candidates":16}"#,
        ])
        .output()
        .expect("run pack");
    assert!(pack.status.success(), "pack failed: {pack:?}");
    let pack_stdout = String::from_utf8(pack.stdout).expect("valid utf8");
    // Content injected (a memory matched) ...
    assert!(
        pack_stdout.contains("\"memory_ids\":[\""),
        "expected non-empty memory_ids: {pack_stdout}"
    );
    // ... but scores cleared (no rerank-scale scores available in this build).
    assert!(
        pack_stdout.contains("\"scores\":[]"),
        "expected empty scores without a reranker: {pack_stdout}"
    );

    cleanup_store(&path);
}

/// Spawn the memkeeper binary with model discovery pinned to cargo's empty
/// integration-test tmpdir. These tests assert the no-models posture (lexical
/// search, `reranked:false`, empty pack scores); without the pin, model-dir
/// discovery finds a populated `models/` checkout near the workspace and
/// silently activates semantic + rerank, so the suite passes in CI but fails
/// on any machine that has pulled models.
fn memkeeper_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_memkeeper"));
    command.env("MEMKEEPER_MODELS_DIR", env!("CARGO_TARGET_TMPDIR"));
    command
}

fn temp_store_path(test_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "memkeeper-cli-{test_name}-{}-{}.sqlite",
        std::process::id(),
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
    let _ = fs::remove_file(path.with_extension("sqlite-wal"));
    let _ = fs::remove_file(path.with_extension("sqlite-shm"));
    let _ = fs::remove_file(sqlite_sidecar_path(path, "-wal"));
    let _ = fs::remove_file(sqlite_sidecar_path(path, "-shm"));
    let _ = fs::remove_file(sqlite_sidecar_path(path, "-journal"));
}

fn sqlite_sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

#[test]
fn recall_log_and_rerank_flag_via_cli() {
    let path = temp_store_path("recall_log_and_rerank_flag_via_cli");
    cleanup_store(&path);

    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");

    let remember = memkeeper_command()
        .args(["remember", "--store"])
        .arg(&path)
        .args(["--json", r#"{"content":"fact: recall telemetry probe"}"#])
        .output()
        .expect("run remember");
    assert!(remember.status.success(), "remember failed: {remember:?}");
    let remember_stdout = String::from_utf8(remember.stdout).expect("valid utf8");
    let memory_id = extract_json_string_after(&remember_stdout, "\"id\":\"");

    // recall-log records events and touches accessed_at for retrieved.
    let recall_req = format!(
        r#"{{"source":"cli-test","events":[{{"memory_id":"{memory_id}","kind":"surfaced","query":"telemetry probe","rank":1,"score":0.5}},{{"memory_id":"{memory_id}","kind":"retrieved"}}]}}"#
    );
    let recall = memkeeper_command()
        .args(["recall-log", "--store"])
        .arg(&path)
        .args(["--json", &recall_req])
        .output()
        .expect("run recall-log");
    assert!(recall.status.success(), "recall-log failed: {recall:?}");
    let recall_stdout = String::from_utf8(recall.stdout).expect("valid utf8");
    assert!(recall_stdout.contains("\"ok\":true"), "{recall_stdout}");
    assert!(
        recall_stdout.contains("\"command\":\"recall-log\""),
        "{recall_stdout}"
    );
    assert!(recall_stdout.contains("\"recorded\":2"), "{recall_stdout}");
    assert!(recall_stdout.contains("\"touched\":1"), "{recall_stdout}");

    // Unknown kinds are rejected with invalid_request.
    let bad = memkeeper_command()
        .args(["recall-log", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"events":[{"memory_id":"mem-x","kind":"viewed"}]}"#,
        ])
        .output()
        .expect("run recall-log invalid");
    let bad_stdout = String::from_utf8(bad.stdout).expect("valid utf8");
    assert!(
        bad_stdout.contains("\"code\":\"invalid_request\""),
        "{bad_stdout}"
    );

    // search accepts the rerank flag and reports reranked:false without models.
    let search = memkeeper_command()
        .args(["search", "--store"])
        .arg(&path)
        .args([
            "--json",
            r#"{"query":"telemetry probe","limit":5,"rerank":true,"rerank_candidates":8}"#,
        ])
        .output()
        .expect("run search");
    assert!(search.status.success(), "search failed: {search:?}");
    let search_stdout = String::from_utf8(search.stdout).expect("valid utf8");
    assert!(search_stdout.contains("\"ok\":true"), "{search_stdout}");
    assert!(
        search_stdout.contains("\"reranked\":false"),
        "{search_stdout}"
    );
    assert!(search_stdout.contains(&memory_id), "{search_stdout}");

    cleanup_store(&path);
}

#[cfg(unix)]
#[test]
fn serve_socket_round_trips_requests() {
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;

    let path = temp_store_path("serve_socket_round_trips_requests");
    cleanup_store(&path);
    let init = memkeeper_command()
        .args(["init", "--store"])
        .arg(&path)
        .arg("--json")
        .output()
        .expect("run init");
    assert!(init.status.success(), "init failed: {init:?}");

    let sock_dir = tempfile::tempdir().expect("socket dir");
    let sock_path = sock_dir.path().join("serve-test.sock");
    let mut server = memkeeper_command()
        .args(["serve", "--socket"])
        .arg(&sock_path)
        .spawn()
        .expect("spawn serve --socket");

    // Wait for the socket to come up.
    let mut stream = None;
    for _ in 0..100 {
        match UnixStream::connect(&sock_path) {
            Ok(connected) => {
                stream = Some(connected);
                break;
            }
            Err(_) => std::thread::sleep(std::time::Duration::from_millis(50)),
        }
    }
    let stream = stream.expect("socket never came up");

    let request = format!(
        "{{\"command\":\"stats\",\"request_id\":\"sock-1\",\"store_path\":{:?},\"payload\":{{\"include_indexes\":false}}}}\n",
        path.to_string_lossy()
    );
    let mut writer = stream.try_clone().expect("clone stream");
    writer.write_all(request.as_bytes()).expect("send request");
    let mut reader = BufReader::new(stream);
    let mut response = String::new();
    reader.read_line(&mut response).expect("read response");
    assert!(response.contains("\"ok\":true"), "{response}");
    assert!(response.contains("\"request_id\":\"sock-1\""), "{response}");
    assert!(response.contains("\"command\":\"stats\""), "{response}");

    // A second request on the same connection also works (persistent client).
    writer.write_all(request.as_bytes()).expect("send second");
    let mut second = String::new();
    reader.read_line(&mut second).expect("read second");
    assert!(second.contains("\"ok\":true"), "{second}");

    server.kill().expect("kill server");
    let _ = server.wait();
    cleanup_store(&path);
}
