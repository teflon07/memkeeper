//! Claude Code hook subcommands (capture + retrieve via the warm daemon).

#[allow(clippy::wildcard_imports)]
use super::*;
#[allow(clippy::wildcard_imports)]
use crate::output::*;

pub(crate) fn run_hook(args: &[String]) -> i32 {
    let subcommand = args.first().map_or("", String::as_str);
    let rest = if args.is_empty() { &[][..] } else { &args[1..] };
    if subcommand == "retrieve" {
        run_hook_retrieve(rest)
    } else {
        eprintln!("Unknown hook subcommand '{subcommand}'. Supported: retrieve");
        2
    }
}

/// Parse the optional `--store` / `--sock` hook flags, rejecting missing or
/// flag-like values instead of silently consuming the next flag as a value.
pub(crate) fn parse_hook_flags(
    args: &[String],
) -> Result<(Option<String>, Option<String>), String> {
    let mut store_flag = None;
    let mut sock_flag = None;
    let mut parser = ArgParser::new(args);
    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store_flag = Some(hook_flag_value(&mut parser, "--store")?),
            "--sock" => sock_flag = Some(hook_flag_value(&mut parser, "--sock")?),
            other => return Err(format!("unknown flag '{other}'")),
        }
    }
    Ok((store_flag, sock_flag))
}

pub(crate) fn hook_flag_value(parser: &mut ArgParser<'_>, flag: &str) -> Result<String, String> {
    match parser.next() {
        Some(value) if !value.starts_with("--") => Ok(value),
        _ => Err(format!("{flag} requires a value")),
    }
}

/// Resolve the hook's store path: explicit flag, then `MEMKEEPER_STORE` /
/// `PI_MEMKEEPER_STORE`, then the project hint anchored to the hook's reported
/// cwd (not this process's cwd).
pub(crate) fn hook_store_path(store_flag: Option<String>, cwd: &str) -> String {
    store_flag.unwrap_or_else(|| {
        let (resolved, _) = diagnostic_store_candidate_from(
            non_empty_env("MEMKEEPER_STORE").as_deref(),
            non_empty_env("PI_MEMKEEPER_STORE").as_deref(),
        );
        if resolved.as_path() == Path::new(PROJECT_STORE_RELATIVE_PATH) && !cwd.is_empty() {
            Path::new(cwd)
                .join(PROJECT_STORE_RELATIVE_PATH)
                .to_string_lossy()
                .into_owned()
        } else {
            resolved.to_string_lossy().into_owned()
        }
    })
}

/// `UserPromptSubmit` hook: send fire-and-forget capture then pack, print `additionalContext` JSON.
/// Reads Claude Code hook envelope from stdin: `{"prompt","session_id","cwd"}`.
/// Exits 0 silently on any error — must never block or crash the hook chain.
pub(crate) fn run_hook_retrieve(args: &[String]) -> i32 {
    use std::io::Read;
    use std::time::Duration;

    let (store_flag, sock_flag) = match parse_hook_flags(args) {
        Ok(flags) => flags,
        Err(message) => {
            eprintln!("hook retrieve: {message}");
            return 0;
        }
    };
    let sock_path = sock_flag
        .or_else(|| non_empty_env("MEMKEEPER_SOCK"))
        .unwrap_or_else(|| "/tmp/memkeeper_daemon.sock".to_string());

    // --- Read hook envelope from stdin ---
    let mut stdin_buf = String::new();
    if io::stdin().read_to_string(&mut stdin_buf).is_err() {
        return 0;
    }
    let envelope: serde_json::Value = match serde_json::from_str(&stdin_buf) {
        Ok(v) => v,
        Err(_) => return 0,
    };

    let prompt = envelope["prompt"].as_str().unwrap_or("");
    if prompt.len() < 20 {
        return 0;
    }
    let cwd = envelope["cwd"].as_str().unwrap_or("");

    let store_path = hook_store_path(store_flag, cwd);

    // --- Pack request: wait for response ---
    let query = hook_query_text(prompt);
    // Rerank a larger candidate pool before injecting the top few. Tunable on the
    // hot path via env; capped at the protocol max. 0 disables the extra pool.
    let rerank_candidates = non_empty_env("MEMKEEPER_HOOK_RERANK_CANDIDATES")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(50)
        .min(memkeeper_store::MAX_PACK_MEMORIES);
    let pack_req = format!(
        "{{\"protocol_version\":\"memkeeper.v0.1\",\"request_id\":\"hook-retrieve\",\
         \"command\":\"pack\",\"store_path\":{},\"payload\":{{\"title\":\"hook-retrieve\",\
         \"queries\":[{}],\"max_memories\":5,\"max_chars\":1500,\"format\":\"markdown\",\
         \"rerank_candidates\":{}}}}}\n",
        json_string(&store_path),
        json_string(&query),
        rerank_candidates,
    );

    let started = std::time::Instant::now();
    let Ok(response_str) = hook_unix_request(&sock_path, &pack_req, Duration::from_millis(2500))
    else {
        return 0;
    };
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;

    let response: serde_json::Value = match serde_json::from_str(&response_str) {
        Ok(v) => v,
        Err(_) => return 0,
    };

    if !response["ok"].as_bool().unwrap_or(false) {
        return 0;
    }

    let content = response["result"]["pack"]["content"]
        .as_str()
        .unwrap_or("")
        .trim();
    if content.is_empty() {
        return 0;
    }

    println!(
        "{{\"hookSpecificOutput\":{{\"hookEventName\":\"UserPromptSubmit\",\"additionalContext\":{}}}}}",
        json_string(&format!("Relevant memories from memkeeper:\n{content}"))
    );

    // Record what we injected as recall telemetry (real query -> injected memory
    // pairs) for later mining as in-distribution reranker data. Best-effort.
    record_injected_recall(
        &sock_path,
        &store_path,
        &query,
        envelope["session_id"].as_str().unwrap_or(""),
        &response["result"]["pack"],
        latency_ms,
    );
    0
}

/// Best-effort recall telemetry for the memories the retrieve hook injected:
/// real `query -> injected memory` pairs (rank + rerank score), tagged
/// `source:"hook"` so they can later be mined as in-distribution reranker
/// training/eval data. Fire-and-forget with a short timeout; any error is
/// ignored so it never delays or fails the hook chain.
fn record_injected_recall(
    sock_path: &str,
    store_path: &str,
    query: &str,
    session_id: &str,
    pack: &serde_json::Value,
    latency_ms: f64,
) {
    use std::time::Duration;

    if let Some(req) =
        build_injected_recall_request(store_path, query, session_id, pack, Some(latency_ms))
    {
        if let Ok(response) = hook_unix_request(sock_path, &req, Duration::from_millis(500)) {
            if recall_log_response_ok(&response) {
                return;
            }
        }
        if let Some(legacy_req) =
            build_injected_recall_request(store_path, query, session_id, pack, None)
        {
            let _ = hook_unix_request(sock_path, &legacy_req, Duration::from_millis(500));
        }
    }
}

/// Build the `recall-log` daemon request that records the injected memories as
/// `surfaced` events (rank + rerank score) under `source:"hook"`. Returns `None`
/// when the pack injected nothing, so the caller skips the socket round-trip.
fn build_injected_recall_request(
    store_path: &str,
    query: &str,
    session_id: &str,
    pack: &serde_json::Value,
    latency_ms: Option<f64>,
) -> Option<String> {
    use std::fmt::Write;

    let ids = pack["memory_ids"].as_array()?;
    let scores = pack["scores"].as_array();
    let mut events = String::new();
    for (i, id) in ids.iter().enumerate() {
        let Some(id) = id.as_str() else { continue };
        if !events.is_empty() {
            events.push(',');
        }
        let score = scores
            .and_then(|s| s.get(i))
            .and_then(serde_json::Value::as_f64)
            .filter(|s| s.is_finite());
        let _ = match score {
            Some(score) => write!(
                events,
                "{{\"memory_id\":{},\"kind\":\"surfaced\",\"query\":{},\"rank\":{},\"score\":{score}}}",
                json_string(id),
                json_string(query),
                i + 1,
            ),
            None => write!(
                events,
                "{{\"memory_id\":{},\"kind\":\"surfaced\",\"query\":{},\"rank\":{}}}",
                json_string(id),
                json_string(query),
                i + 1,
            ),
        };
    }
    if events.is_empty() {
        return None;
    }
    let latency_fields = latency_ms
        .filter(|latency| latency.is_finite() && *latency >= 0.0)
        .map(|latency| {
            format!(
                ",\"batch_id\":{},\"latency_ms\":{latency},\"latency_source\":{}",
                json_string(&hook_recall_batch_id()),
                json_string("memkeeper_rust_hook")
            )
        })
        .unwrap_or_default();
    Some(format!(
        "{{\"protocol_version\":\"memkeeper.v0.1\",\"request_id\":\"hook-recall-log\",\
         \"command\":\"recall-log\",\"store_path\":{},\"payload\":{{\"source\":\"hook\",\
         \"session_id\":{}{latency_fields},\"touch_accessed\":false,\"events\":[{events}]}}}}\n",
        json_string(store_path),
        json_string(session_id),
    ))
}

fn recall_log_response_ok(response: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(response)
        .ok()
        .and_then(|value| value["ok"].as_bool())
        .unwrap_or(false)
}

fn hook_recall_batch_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos());
    format!("hook-{nanos}")
}

/// First 500 characters of the prompt, truncated on a char boundary so a
/// multibyte code point near the limit can never panic the hook.
pub(crate) fn hook_query_text(prompt: &str) -> String {
    prompt.chars().take(500).collect()
}

#[cfg(unix)]
pub(crate) fn hook_unix_request(
    sock_path: &str,
    request: &str,
    timeout: std::time::Duration,
) -> Result<String, std::io::Error> {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;

    let mut stream = UnixStream::connect(sock_path)?;
    stream.set_write_timeout(Some(timeout))?;
    stream.set_read_timeout(Some(timeout))?;
    stream.write_all(request.as_bytes())?;

    let mut response: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; 65536];
    loop {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        response.extend_from_slice(&buf[..n]);
        if response.contains(&b'\n') {
            break;
        }
    }
    let line_end = response
        .iter()
        .position(|&byte| byte == b'\n')
        .unwrap_or(response.len());
    Ok(String::from_utf8_lossy(&response[..line_end]).into_owned())
}

/// Unix-socket daemon routing is unavailable on non-Unix platforms (Rust's
/// `UnixStream` is Unix-only). Every caller already handles an `Err` by
/// degrading (direct dispatch / in-process models), so report it as unsupported.
#[cfg(not(unix))]
pub(crate) fn hook_unix_request(
    _sock_path: &str,
    _request: &str,
    _timeout: std::time::Duration,
) -> Result<String, std::io::Error> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "Unix-socket daemon is not supported on this platform",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn builds_well_formed_recall_log_for_injected_memories() {
        let pack = json!({"memory_ids": ["mem_a", "mem_b"], "scores": [0.91, 0.42]});
        let req = build_injected_recall_request(
            "/tmp/s.sqlite",
            "what did we decide",
            "sess-1",
            &pack,
            None,
        )
        .expect("request built");
        // Must be a single well-formed JSON line: malformed payload = silent data loss.
        assert!(req.ends_with('\n'));
        let v: serde_json::Value = serde_json::from_str(req.trim()).expect("valid json");
        assert_eq!(v["command"], "recall-log");
        assert_eq!(v["payload"]["source"], "hook");
        assert_eq!(v["payload"]["session_id"], "sess-1");
        assert_eq!(v["payload"]["touch_accessed"], false);
        let events = v["payload"]["events"].as_array().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0]["memory_id"], "mem_a");
        assert_eq!(events[0]["kind"], "surfaced");
        assert_eq!(events[0]["query"], "what did we decide");
        assert_eq!(events[0]["rank"], 1);
        assert!((events[0]["score"].as_f64().unwrap() - 0.91).abs() < 1e-9);
        assert_eq!(events[1]["rank"], 2);
    }

    #[test]
    fn recall_log_can_include_latency_metadata() {
        let pack = json!({"memory_ids": ["mem_a"], "scores": [0.91]});
        let req = build_injected_recall_request("/tmp/s.sqlite", "q", "sess-1", &pack, Some(12.5))
            .expect("request built");
        let v: serde_json::Value = serde_json::from_str(req.trim()).expect("valid json");
        assert!(v["payload"]["batch_id"]
            .as_str()
            .unwrap()
            .starts_with("hook-"));
        assert_eq!(v["payload"]["latency_ms"], 12.5);
        assert_eq!(v["payload"]["latency_source"], "memkeeper_rust_hook");
    }

    #[test]
    fn no_request_when_pack_injected_nothing() {
        assert!(
            build_injected_recall_request("/s", "q", "s", &json!({"memory_ids": []}), None)
                .is_none()
        );
        assert!(build_injected_recall_request("/s", "q", "s", &json!({}), None).is_none());
    }

    #[test]
    fn omits_score_when_absent_or_nonfinite() {
        // missing scores array
        let req =
            build_injected_recall_request("/s", "q", "sid", &json!({"memory_ids": ["m1"]}), None)
                .expect("built");
        let v: serde_json::Value = serde_json::from_str(req.trim()).expect("valid json");
        assert!(v["payload"]["events"][0].get("score").is_none());
        assert_eq!(v["payload"]["events"][0]["rank"], 1);
    }

    #[test]
    fn escapes_quotes_in_query() {
        let pack = json!({"memory_ids": ["m1"], "scores": [0.5]});
        let req = build_injected_recall_request("/s", "why \"pi\" name?", "sid", &pack, None)
            .expect("built");
        let v: serde_json::Value =
            serde_json::from_str(req.trim()).expect("valid json despite quotes");
        assert_eq!(v["payload"]["events"][0]["query"], "why \"pi\" name?");
    }
}
