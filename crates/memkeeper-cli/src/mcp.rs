//! Native MCP-over-stdio server.
//!
//! `memkeeper mcp` speaks the Model Context Protocol directly (JSON-RPC 2.0 over
//! newline-delimited stdio), so any MCP client (Claude Code/Desktop, Cursor, …)
//! can talk to a local store with no Python shim. It is a thin transport: every
//! `tools/call` is translated into a serve request envelope and dispatched
//! through the SAME [`execute_serve_request`] used by `serve --stdio`/`--socket`,
//! so the tool surface, payload validation, and degradation behavior are shared
//! with the rest of the engine (no duplicated request logic).
//!
//! Transport is deliberately a thin hand-rolled JSON-RPC loop rather than an
//! async SDK crate: MCP stdio framing is just newline-delimited JSON-RPC, which
//! mirrors the existing sync serve loop, so we add zero dependencies (`serde_json`
//! is already pulled in) and keep the binary lean. The mapped tool surface
//! mirrors `adapters/mcp/memkeeper_mcp.py` exactly (candidate approve/reject are
//! intentionally omitted — those stay human-review-only via CLI/dashboard).

#[allow(clippy::wildcard_imports)]
use super::*;

use serde_json::{json, Map, Value};

/// MCP protocol version this server defaults to when the client does not request
/// one. We echo the client's requested version when present (forward/backward
/// compatible), falling back to this widely-supported revision.
const DEFAULT_MCP_PROTOCOL_VERSION: &str = "2025-06-18";

/// `source.adapter` recorded on MCP writes. Overridable so multiple MCP clients
/// can distinguish their provenance; mirrors the Python adapter's default.
fn mcp_adapter() -> String {
    non_empty_env("MEMKEEPER_MCP_ADAPTER").unwrap_or_else(|| "generic-mcp".to_string())
}

/// `source.source_description` recorded on MCP writes / recall telemetry.
fn mcp_source_description() -> String {
    non_empty_env("MEMKEEPER_MCP_SOURCE_DESCRIPTION").unwrap_or_else(|| "memkeeper MCP".to_string())
}

/// How `memkeeper mcp` dispatches each serve request.
///
/// When `MEMKEEPER_SOCK` points at a reachable `memkeeper serve --socket` daemon we
/// route every request over that warm socket and load NO models in this process
/// (the daemon already owns them) — keeping per-session startup instant and
/// avoiding a second multi-GB resident copy of the ONNX models. Otherwise we
/// load the embed/rerank models in-process and dispatch locally, as before.
enum McpBackend {
    /// Round-trip requests to a warm daemon at this Unix socket path.
    Daemon(String),
    /// Dispatch in-process against locally loaded models (boxed: the models are
    /// large and would otherwise bloat every value of this enum).
    InProcess(Box<SemanticModels>),
}

/// Resolve the warm-daemon socket to route through, if any.
///
/// Returns `Some(path)` only when `MEMKEEPER_SOCK` is set AND the socket accepts a
/// connection right now. When the var is set but the daemon is unreachable we
/// emit a loud stderr warning and return `None` (degrade to in-process) rather
/// than fail silently — a misconfigured socket should be visible, not masked.
#[cfg(unix)]
fn daemon_socket_target() -> Option<String> {
    use std::os::unix::net::UnixStream;
    let sock = non_empty_env("MEMKEEPER_SOCK")?;
    match UnixStream::connect(&sock) {
        Ok(_) => Some(sock),
        Err(error) => {
            eprintln!(
                "[memkeeper] mcp: WARNING MEMKEEPER_SOCK={sock} set but daemon unreachable ({error}); \
                 falling back to in-process models"
            );
            None
        }
    }
}

/// Warm-daemon socket routing is Unix-only (`UnixStream`). On other platforms
/// always use in-process models, but stay loud if the operator set the var.
#[cfg(not(unix))]
fn daemon_socket_target() -> Option<String> {
    if non_empty_env("MEMKEEPER_SOCK").is_some() {
        eprintln!(
            "[memkeeper] mcp: WARNING MEMKEEPER_SOCK is set but Unix-socket daemon routing is not \
             supported on this platform; using in-process models"
        );
    }
    None
}

pub(crate) fn run_mcp(args: &[String]) -> i32 {
    let store = match parse_store_args(args) {
        Ok(parsed) => parsed.store,
        Err(error) => {
            eprintln!("[memkeeper] mcp: {error}");
            return error.exit_code();
        }
    };
    let backend = if let Some(sock) = daemon_socket_target() {
        eprintln!(
            "[memkeeper] mcp server ready (store {}) — routing to warm daemon at {sock}",
            store.display()
        );
        McpBackend::Daemon(sock)
    } else {
        let models = match serve_semantic_models_or_refuse() {
            Ok(models) => models,
            Err(code) => return code,
        };
        eprintln!(
            "[memkeeper] mcp server ready (store {}) — models loaded in-process, speaking MCP over stdio",
            store.display()
        );
        McpBackend::InProcess(Box::new(models))
    };
    run_mcp_stdio(&store, &backend)
}

/// Dispatch one serve request via the active backend, returning the serve
/// response envelope as a JSON string (the same shape either path produces).
fn dispatch_serve(backend: &McpBackend, envelope: &ServeRequestEnvelope) -> String {
    match backend {
        McpBackend::InProcess(models) => execute_serve_request(envelope, Instant::now(), models),
        McpBackend::Daemon(sock) => dispatch_via_daemon(sock, envelope),
    }
}

/// Serialize `envelope` to the daemon wire format and round-trip it over the
/// Unix socket. On any transport error we synthesize an `ok:false` envelope so
/// the failure surfaces to the MCP client as a tool error, never a silent empty
/// result.
fn dispatch_via_daemon(sock: &str, envelope: &ServeRequestEnvelope) -> String {
    use std::time::Duration;
    let line = serialize_serve_request_line(envelope);
    match hook_unix_request(sock, &line, Duration::from_secs(30)) {
        Ok(response) => response,
        Err(error) => daemon_failure_envelope(&envelope.command_name, &error.to_string()),
    }
}

/// Build the newline-terminated daemon request line for `envelope`. The payload
/// is already a JSON object string, so it is embedded verbatim; the result is
/// exactly the envelope `parse_serve_request` accepts (round-tripped in tests).
fn serialize_serve_request_line(envelope: &ServeRequestEnvelope) -> String {
    let request_id = envelope.request_id.as_deref().unwrap_or("");
    let store_path = envelope
        .store_path
        .as_ref()
        .map(|path| path.to_string_lossy().into_owned())
        .unwrap_or_default();
    format!(
        "{{\"protocol_version\":\"memkeeper.v0.1\",\"request_id\":{},\"command\":{},\
         \"store_path\":{},\"payload\":{}}}\n",
        json_string(request_id),
        json_string(&envelope.command_name),
        json_string(&store_path),
        envelope.payload_json,
    )
}

/// Minimal `ok:false` serve envelope for a daemon transport failure.
fn daemon_failure_envelope(command_name: &str, message: &str) -> String {
    format!(
        "{{\"protocol_version\":\"memkeeper.v0.1\",\"ok\":false,\"command\":{},\
         \"error\":{{\"code\":\"daemon_unreachable\",\"message\":{}}}}}",
        json_string(command_name),
        json_string(message),
    )
}

/// Read one JSON-RPC message per line; write one response line per request that
/// carries an `id`. Notifications (no `id`) are processed for side effects but
/// never answered, per JSON-RPC 2.0.
fn run_mcp_stdio(store: &Path, backend: &McpBackend) -> i32 {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(line) => line,
            Err(error) => {
                eprintln!("[memkeeper] mcp: failed to read stdin: {error}");
                return 1;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let Some(response) = handle_mcp_line(&line, store, backend) else {
            continue; // notification or unparseable-without-id: nothing to send
        };
        let serialized = match serde_json::to_string(&response) {
            Ok(serialized) => serialized,
            Err(error) => {
                eprintln!("[memkeeper] mcp: failed to serialize response: {error}");
                continue;
            }
        };
        if writeln!(stdout, "{serialized}").is_err() || stdout.flush().is_err() {
            return 1;
        }
    }
    0
}

/// Parse and dispatch one JSON-RPC line. Returns `Some(response)` when the
/// message expects a reply, `None` for notifications.
fn handle_mcp_line(line: &str, store: &Path, backend: &McpBackend) -> Option<Value> {
    let request: Value = match serde_json::from_str(line) {
        Ok(value) => value,
        // Parse error: reply only if we cannot tell it was a notification.
        Err(_) => return Some(jsonrpc_error(&Value::Null, -32700, "parse error")),
    };
    let id = request.get("id").cloned();
    let method = request.get("method").and_then(Value::as_str).unwrap_or("");
    let params = request.get("params").cloned().unwrap_or(Value::Null);
    let is_notification = id.is_none();

    // Notifications (initialized, cancelled, …) carry no id and get no response.
    if is_notification {
        return None;
    }
    let id = id.unwrap_or(Value::Null);

    let result = match method {
        "initialize" => Ok(initialize_result(&params)),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => return Some(handle_tools_call(&id, &params, store, backend)),
        "ping" => Ok(json!({})),
        other => Err((-32601, format!("method not found: {other}"))),
    };
    Some(match result {
        Ok(result) => jsonrpc_result(&id, &result),
        Err((code, message)) => jsonrpc_error(&id, code, &message),
    })
}

fn initialize_result(params: &Value) -> Value {
    let protocol_version = params
        .get("protocolVersion")
        .and_then(Value::as_str)
        .unwrap_or(DEFAULT_MCP_PROTOCOL_VERSION)
        .to_string();
    json!({
        "protocolVersion": protocol_version,
        "capabilities": { "tools": {} },
        "serverInfo": { "name": "memkeeper", "version": env!("CARGO_PKG_VERSION") },
        "instructions": MCP_INSTRUCTIONS,
    })
}

const MCP_INSTRUCTIONS: &str = "Access to your local memkeeper store. Memories remain the source of truth; \
graph rows are rebuildable projections. Source/provenance is hidden unless an explicit include_source \
argument is provided and the user asked for provenance. Write only concise, durable, non-secret \
facts/decisions/preferences/lessons/actions/continuity notes. Remember responses may include \
auto_superseded ids or conflict_candidates; surface continuity conflicts to the user when relevant. \
Do not dump transcripts, secrets, noisy command output, or temporary task state. Use forget to \
tombstone a specific memory id; when retiring a memory because it is WRONG (e.g. a recalled fact \
the user contradicted), use forget with mode='correct' (and corrected_by when you have the right \
answer) so the correction is captured explicitly. For plausible-but-unverified inferences, prefer \
candidate_submit (enqueues for human review) over remember.";

fn handle_tools_call(id: &Value, params: &Value, store: &Path, backend: &McpBackend) -> Value {
    let Some(name) = params.get("name").and_then(Value::as_str) else {
        return jsonrpc_error(id, -32602, "tools/call missing required field: name");
    };
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let arguments = match arguments {
        Value::Object(map) => map,
        Value::Null => Map::new(),
        _ => return jsonrpc_error(id, -32602, "tools/call arguments must be an object"),
    };

    let (command_name, payload_json) = match build_serve_call(name, &arguments) {
        Ok(call) => call,
        Err(message) => return tool_error_result(id, &message),
    };
    let Some(command) = Command::parse(command_name) else {
        return tool_error_result(
            id,
            &format!("internal: unknown serve command {command_name}"),
        );
    };
    let envelope = ServeRequestEnvelope {
        request_id: Some(format!("mcp-{command_name}")),
        command,
        command_name: command_name.to_string(),
        store_path: Some(store.to_path_buf()),
        payload_json,
    };
    let response = dispatch_serve(backend, &envelope);

    // Best-effort recall telemetry, matching the Python adapter: surfaced on
    // search, retrieved on get. Failures are swallowed (never break recall).
    record_mcp_recall(name, &response, &arguments, store, backend);

    // Surface the serve envelope as the tool's text content. An engine-level
    // failure (ok:false) is reported as an MCP tool error so the client sees it.
    let is_error = !envelope_ok(&response);
    jsonrpc_result(
        id,
        &json!({
            "content": [ { "type": "text", "text": response } ],
            "isError": is_error,
        }),
    )
}

/// Whether a serve envelope string reports success (`"ok": true`).
fn envelope_ok(response: &str) -> bool {
    serde_json::from_str::<Value>(response)
        .ok()
        .and_then(|value| value.get("ok").and_then(Value::as_bool))
        .unwrap_or(false)
}

fn jsonrpc_result(id: &Value, result: &Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn jsonrpc_error(id: &Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// A tool-level (not protocol-level) error: a successful JSON-RPC response whose
/// result is flagged `isError`, per the MCP tools spec.
fn tool_error_result(id: &Value, message: &str) -> Value {
    jsonrpc_result(
        id,
        &json!({
            "content": [ { "type": "text", "text": message } ],
            "isError": true,
        }),
    )
}

// --------------------------------------------------------------------------
// Argument → serve payload mapping. Mirrors adapters/mcp/memkeeper_mcp.py: the
// payload dicts here are byte-for-byte the serve payloads that adapter sends, so
// they pass the engine's strict `*_request_from_json` validators unchanged.
// --------------------------------------------------------------------------

fn arg_bool(args: &Map<String, Value>, key: &str, default: bool) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(default)
}

fn arg_i64(args: &Map<String, Value>, key: &str, default: i64) -> i64 {
    args.get(key).and_then(Value::as_i64).unwrap_or(default)
}

fn arg_f64(args: &Map<String, Value>, key: &str, default: f64) -> f64 {
    args.get(key).and_then(Value::as_f64).unwrap_or(default)
}

fn arg_str<'a>(args: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}

/// Require a non-empty string argument, returning a client-facing error message.
fn require_str(args: &Map<String, Value>, key: &str) -> Result<String, String> {
    match args.get(key).and_then(Value::as_str) {
        Some(value) if !value.is_empty() => Ok(value.to_string()),
        _ => Err(format!("missing required argument: {key}")),
    }
}

/// Copy `key` from args into `payload` verbatim if present and not null.
fn copy_opt(payload: &mut Map<String, Value>, args: &Map<String, Value>, key: &str) {
    if let Some(value) = args.get(key) {
        if !value.is_null() {
            payload.insert(key.to_string(), value.clone());
        }
    }
}

/// Build a single-key string-array filter entry (`spaces`, `tags`, `entity_keys`, …).
fn string_filter(
    args: &Map<String, Value>,
    arg_key: &str,
    filter_key: &str,
) -> Option<(String, Value)> {
    match args.get(arg_key) {
        Some(Value::String(value)) if !value.is_empty() => {
            Some((filter_key.to_string(), json!([value])))
        }
        _ => None,
    }
}

fn to_payload(map: Map<String, Value>) -> String {
    Value::Object(map).to_string()
}

/// Map an MCP tool name + arguments to `(serve_command, payload_json)`.
#[allow(clippy::too_many_lines)]
fn build_serve_call(
    name: &str,
    args: &Map<String, Value>,
) -> Result<(&'static str, String), String> {
    let mut payload = Map::new();
    match name {
        "stats" => {
            payload.insert(
                "include_indexes".into(),
                json!(arg_bool(args, "include_indexes", false)),
            );
            payload.insert(
                "include_health".into(),
                json!(arg_bool(args, "include_health", false)),
            );
            Ok(("stats", to_payload(payload)))
        }
        "search" => {
            let query = require_str(args, "query")?;
            let limit = arg_i64(args, "limit", 10);
            payload.insert("query".into(), json!(query));
            payload.insert("limit".into(), json!(limit));
            payload.insert(
                "include_content".into(),
                json!(arg_bool(args, "include_content", false)),
            );
            payload.insert(
                "include_source".into(),
                json!(arg_bool(args, "include_source", false)),
            );
            let rerank = arg_bool(args, "rerank", true) && limit > 0;
            if rerank {
                payload.insert("rerank".into(), json!(true));
                payload.insert("rerank_candidates".into(), json!(16));
            }
            // semantic_fallback overrides semantic_enabled when provided.
            let semantic_enabled = args
                .get("semantic_fallback")
                .and_then(Value::as_bool)
                .unwrap_or_else(|| arg_bool(args, "semantic_enabled", true));
            payload.insert(
                "semantic_fallback".into(),
                json!(if semantic_enabled {
                    "fallback"
                } else {
                    "disabled"
                }),
            );
            let mut filters = Map::new();
            filters.extend(string_filter(args, "space", "spaces"));
            if let Some(Value::Array(tags)) = args.get("tags") {
                if !tags.is_empty() {
                    filters.insert("tags".into(), Value::Array(tags.clone()));
                }
            }
            filters.extend(string_filter(args, "entity_key", "entity_keys"));
            if !filters.is_empty() {
                payload.insert("filters".into(), Value::Object(filters));
            }
            Ok(("search", to_payload(payload)))
        }
        "get" => {
            payload.insert("id".into(), json!(require_str(args, "memory_id")?));
            payload.insert(
                "include_history".into(),
                json!(arg_bool(args, "include_history", false)),
            );
            payload.insert(
                "include_source".into(),
                json!(arg_bool(args, "include_source", false)),
            );
            Ok(("get", to_payload(payload)))
        }
        "memory_list" => {
            payload.insert("limit".into(), json!(arg_i64(args, "limit", 20)));
            payload.insert(
                "include_content".into(),
                json!(arg_bool(args, "include_content", false)),
            );
            payload.insert(
                "include_source".into(),
                json!(arg_bool(args, "include_source", false)),
            );
            let mut filters = Map::new();
            filters.extend(string_filter(args, "status", "statuses"));
            filters.extend(string_filter(args, "space", "spaces"));
            filters.extend(string_filter(args, "entity_key", "entity_keys"));
            if !filters.is_empty() {
                payload.insert("filters".into(), Value::Object(filters));
            }
            Ok(("memory-list", to_payload(payload)))
        }
        "entity_search" => {
            payload.insert("limit".into(), json!(arg_i64(args, "limit", 10)));
            payload.insert(
                "include_source".into(),
                json!(arg_bool(args, "include_source", false)),
            );
            copy_opt(&mut payload, args, "query");
            copy_opt(&mut payload, args, "entity_key");
            if let Some(entity_type) = arg_str(args, "entity_type") {
                payload.insert("entity_types".into(), json!([entity_type]));
            }
            Ok(("entity-search", to_payload(payload)))
        }
        "graph_neighbors" => {
            payload.insert("entity_key".into(), json!(require_str(args, "entity_key")?));
            payload.insert("depth".into(), json!(arg_i64(args, "depth", 1)));
            payload.insert("max_edges".into(), json!(arg_i64(args, "max_edges", 50)));
            payload.insert(
                "include_tombstoned".into(),
                json!(arg_bool(args, "include_tombstoned", false)),
            );
            payload.insert(
                "include_source".into(),
                json!(arg_bool(args, "include_source", false)),
            );
            Ok(("graph-neighbors", to_payload(payload)))
        }
        "graph_context" => {
            payload.insert("entity_key".into(), json!(require_str(args, "entity_key")?));
            payload.insert("depth".into(), json!(arg_i64(args, "depth", 1)));
            payload.insert("max_edges".into(), json!(arg_i64(args, "max_edges", 50)));
            payload.insert(
                "max_memories".into(),
                json!(arg_i64(args, "max_memories", 10)),
            );
            payload.insert("max_chars".into(), json!(arg_i64(args, "max_chars", 4000)));
            payload.insert(
                "include_source".into(),
                json!(arg_bool(args, "include_source", false)),
            );
            Ok(("graph-context", to_payload(payload)))
        }
        "dream_graph" => {
            payload.insert("tasks".into(), json!(["graph"]));
            payload.insert(
                "max_memories".into(),
                json!(arg_i64(args, "max_memories", 1000)),
            );
            payload.insert("dry_run".into(), json!(true));
            copy_opt(&mut payload, args, "space");
            Ok(("dream", to_payload(payload)))
        }
        "remember" => {
            let content = require_str(args, "content")?;
            let mut source = Map::new();
            source.insert("type".into(), json!("mcp"));
            source.insert("adapter".into(), json!(mcp_adapter()));
            source.insert("source_description".into(), json!(mcp_source_description()));
            source.insert(
                "source_type".into(),
                json!(arg_str(args, "source_type").unwrap_or("assistant-inference")),
            );
            if let Some(sensitivity) = arg_str(args, "sensitivity") {
                source.insert("sensitivity".into(), json!(sensitivity));
            }
            payload.insert("content".into(), json!(content));
            payload.insert("confidence".into(), json!(arg_f64(args, "confidence", 1.0)));
            payload.insert("pinned".into(), json!(arg_bool(args, "pinned", false)));
            payload.insert(
                "derive_keys".into(),
                json!(arg_bool(args, "derive_keys", true)),
            );
            payload.insert(
                "mode".into(),
                json!(arg_str(args, "mode").unwrap_or("auto")),
            );
            payload.insert("dry_run".into(), json!(arg_bool(args, "dry_run", false)));
            payload.insert("source".into(), Value::Object(source));
            for key in [
                "space",
                "silo",
                "scope",
                "project",
                "kind",
                "summary",
                "tags",
                "entity_key",
                "claim_key",
                "observed_at",
                "valid_from",
                "valid_to",
                "expires_at",
                "supersedes",
                "contradicts",
            ] {
                copy_opt(&mut payload, args, key);
            }
            if let Some(verified_against) = arg_str(args, "verified_against") {
                payload.insert(
                    "metadata_json".into(),
                    json!(json!({ "verified_against": verified_against }).to_string()),
                );
            }
            Ok(("remember", to_payload(payload)))
        }
        "forget" => {
            payload.insert("id".into(), json!(require_str(args, "memory_id")?));
            payload.insert("dry_run".into(), json!(arg_bool(args, "dry_run", false)));
            copy_opt(&mut payload, args, "reason");
            copy_opt(&mut payload, args, "mode");
            copy_opt(&mut payload, args, "corrected_by");
            Ok(("forget", to_payload(payload)))
        }
        "entity_upsert" => {
            payload.insert("entity_key".into(), json!(require_str(args, "entity_key")?));
            payload.insert(
                "canonical_name".into(),
                json!(require_str(args, "canonical_name")?),
            );
            payload.insert("confidence".into(), json!(arg_f64(args, "confidence", 1.0)));
            payload.insert(
                "include_source".into(),
                json!(arg_bool(args, "include_source", false)),
            );
            for key in [
                "entity_type",
                "aliases",
                "space",
                "status",
                "source_episode_id",
                "metadata",
            ] {
                copy_opt(&mut payload, args, key);
            }
            Ok(("entity-upsert", to_payload(payload)))
        }
        "relationship_upsert" => {
            payload.insert(
                "relation_type".into(),
                json!(require_str(args, "relation_type")?),
            );
            payload.insert("confidence".into(), json!(arg_f64(args, "confidence", 1.0)));
            payload.insert(
                "include_source".into(),
                json!(arg_bool(args, "include_source", false)),
            );
            for key in [
                "subject_entity_key",
                "object_entity_key",
                "subject_entity_id",
                "object_entity_id",
                "memory_id",
                "space",
                "source_episode_id",
                "status",
                "observed_at",
                "valid_from",
                "valid_to",
                "metadata",
            ] {
                copy_opt(&mut payload, args, key);
            }
            Ok(("relationship-upsert", to_payload(payload)))
        }
        "verify" => {
            payload.insert("memory_id".into(), json!(require_str(args, "memory_id")?));
            copy_opt(&mut payload, args, "verified_against");
            Ok(("verify", to_payload(payload)))
        }
        "pack" => {
            let queries = match args.get("queries") {
                Some(Value::Array(queries)) if !queries.is_empty() => Value::Array(queries.clone()),
                _ => return Err("missing required argument: queries (non-empty array)".to_string()),
            };
            payload.insert(
                "title".into(),
                json!(arg_str(args, "title").unwrap_or("context")),
            );
            payload.insert("queries".into(), queries);
            payload.insert(
                "max_memories".into(),
                json!(arg_i64(args, "max_memories", 10)),
            );
            payload.insert("max_chars".into(), json!(arg_i64(args, "max_chars", 6000)));
            payload.insert("min_score".into(), json!(arg_f64(args, "min_score", 0.0)));
            for key in [
                "query_expansion",
                "thread_expansion",
                "max_query_variants",
                "max_thread_seeds",
                "max_thread_neighbors",
            ] {
                copy_opt(&mut payload, args, key);
            }
            let mut filters = Map::new();
            filters.extend(string_filter(args, "space", "spaces"));
            if let Some(Value::Array(tags)) = args.get("tags") {
                if !tags.is_empty() {
                    filters.insert("tags".into(), Value::Array(tags.clone()));
                }
            }
            if !filters.is_empty() {
                payload.insert("filters".into(), Value::Object(filters));
            }
            Ok(("pack", to_payload(payload)))
        }
        "candidate_submit" => {
            payload.insert("content".into(), json!(require_str(args, "content")?));
            payload.insert("confidence".into(), json!(arg_f64(args, "confidence", 1.0)));
            payload.insert(
                "source_type".into(),
                json!(arg_str(args, "source_type").unwrap_or("assistant-inference")),
            );
            payload.insert("dry_run".into(), json!(arg_bool(args, "dry_run", false)));
            for key in [
                "rationale",
                "kind",
                "summary",
                "tags",
                "entity_key",
                "claim_key",
                "sensitivity",
                "space",
                "silo",
                "scope",
                "project",
                "supersedes",
            ] {
                copy_opt(&mut payload, args, key);
            }
            Ok(("candidate-submit", to_payload(payload)))
        }
        "candidate_list" => {
            payload.insert("limit".into(), json!(arg_i64(args, "limit", 50)));
            payload.insert(
                "status".into(),
                json!(arg_str(args, "status").unwrap_or("pending")),
            );
            copy_opt(&mut payload, args, "space");
            Ok(("candidate-list", to_payload(payload)))
        }
        other => Err(format!("unknown tool: {other}")),
    }
}

// --------------------------------------------------------------------------
// Recall telemetry (best-effort, mirrors the Python adapter). Logged through
// the engine's `recall-log` command so the recall_events table + accessed_at
// touch stay engine-owned. Any failure here must never break recall.
// --------------------------------------------------------------------------
fn record_mcp_recall(
    tool: &str,
    response: &str,
    args: &Map<String, Value>,
    store: &Path,
    backend: &McpBackend,
) {
    let events = match tool {
        "search" => surfaced_events(response, arg_str(args, "query").unwrap_or("")),
        "get" => retrieved_events(response),
        _ => return,
    };
    if events.is_empty() {
        return;
    }
    let payload = json!({
        "source": mcp_source_description(),
        "touch_accessed": true,
        "events": events,
    })
    .to_string();
    let Some(command) = Command::parse("recall-log") else {
        return;
    };
    let envelope = ServeRequestEnvelope {
        request_id: Some("mcp-recall-log".to_string()),
        command,
        command_name: "recall-log".to_string(),
        store_path: Some(store.to_path_buf()),
        payload_json: payload,
    };
    let _ = dispatch_serve(backend, &envelope);
}

fn surfaced_events(response: &str, query: &str) -> Vec<Value> {
    let Ok(parsed) = serde_json::from_str::<Value>(response) else {
        return Vec::new();
    };
    let results = parsed
        .get("result")
        .and_then(|result| result.get("results"))
        .and_then(Value::as_array);
    let Some(results) = results else {
        return Vec::new();
    };
    results
        .iter()
        .filter_map(|result| {
            let memory_id = result.get("memory_id").and_then(Value::as_str)?;
            let mut event = Map::new();
            event.insert("memory_id".into(), json!(memory_id));
            event.insert("kind".into(), json!("surfaced"));
            if !query.is_empty() {
                event.insert("query".into(), json!(query));
            }
            if let Some(rank) = result.get("rank") {
                event.insert("rank".into(), rank.clone());
            }
            if let Some(score) = result.get("score") {
                event.insert("score".into(), score.clone());
            }
            Some(Value::Object(event))
        })
        .collect()
}

fn retrieved_events(response: &str) -> Vec<Value> {
    let Ok(parsed) = serde_json::from_str::<Value>(response) else {
        return Vec::new();
    };
    let memory_id = parsed
        .get("result")
        .and_then(|result| result.get("memory"))
        .and_then(|memory| memory.get("id"))
        .and_then(Value::as_str);
    match memory_id {
        Some(memory_id) => vec![json!({ "memory_id": memory_id, "kind": "retrieved" })],
        None => Vec::new(),
    }
}

// --------------------------------------------------------------------------
// Tool definitions for tools/list. Names and ergonomics mirror the Python
// adapter (adapters/mcp/memkeeper_mcp.py). candidate-approve/reject are
// intentionally absent (human-review actions stay CLI/dashboard-only).
// --------------------------------------------------------------------------
#[allow(clippy::needless_pass_by_value)] // builder embeds `properties` into the schema
fn tool(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    })
}

#[allow(clippy::too_many_lines)]
fn tool_definitions() -> Vec<Value> {
    vec![
        tool(
            "stats",
            "Report store statistics: total/active memory counts, breakdowns by space, silo, status, and kind, schema version, and database size. Read-only; no side effects. Use to inspect the store's overall state and health, not to retrieve memories (use `search` or `pack` for that).",
            json!({
                "include_indexes": { "type": "boolean", "description": "If true, add per-index row counts. Default false." },
                "include_health": { "type": "boolean", "description": "If true, add the governance/health rollup (counts of stale, expiring, and low-confidence memories). Default false." },
            }),
            &[],
        ),
        tool(
            "search",
            "Find individual memories ranked by relevance to a query. Semantic-primary when embedding models are loaded, falling back to deterministic BM25/FTS keyword search otherwise; cross-encoder reranked by default. Read-only. Returns scored, individual memory records (with ids) — use this to locate or inspect specific memories. To assemble a prompt-ready context block, use `pack` instead; to browse recent memories without a query, use `memory_list`.",
            json!({
                "query": { "type": "string", "description": "Natural-language search query. Required." },
                "limit": { "type": "integer", "description": "Maximum number of memories to return. Default 10." },
                "space": { "type": "string", "description": "Restrict to a single memory space (namespace). Omit to search the default space." },
                "tags": { "type": "array", "items": { "type": "string" }, "description": "Restrict to memories carrying these tags." },
                "entity_key": { "type": "string", "description": "Restrict to memories linked to this entity key." },
                "include_content": { "type": "boolean", "description": "If true, return each memory's full text instead of a snippet. Default false." },
                "include_source": { "type": "boolean", "description": "If true, reveal provenance/source metadata. Default false." },
                "semantic_enabled": { "type": "boolean", "description": "Force semantic retrieval on or off. Default: on when embedding models are available, else lexical." },
                "rerank": { "type": "boolean", "description": "Apply the cross-encoder reranker to the candidate pool. Default true." },
            }),
            &["query"],
        ),
        tool(
            "get",
            "Fetch one memory by its exact id (for example, an id returned by `search` or `memory_list`). Read-only. Use when you already have the id and want the full record; use `search` to find a memory by its content.",
            json!({
                "memory_id": { "type": "string", "description": "The memory's id. Required." },
                "include_history": { "type": "boolean", "description": "If true, include the memory's version/change history. Default false." },
                "include_source": { "type": "boolean", "description": "If true, reveal provenance/source metadata. Default false." },
            }),
            &["memory_id"],
        ),
        tool(
            "memory_list",
            "List recent memories in reverse-chronological order for review or cleanup, optionally filtered. Read-only. Use to browse or audit what is stored (including stale or superseded entries); use `search` or `pack` for relevance-ranked retrieval against a query.",
            json!({
                "limit": { "type": "integer", "description": "Maximum number of memories to return. Default 20." },
                "status": { "type": "string", "description": "Filter by lifecycle status (e.g. active, superseded, tombstoned). Omit for active memories." },
                "space": { "type": "string", "description": "Restrict to a single memory space (namespace)." },
                "entity_key": { "type": "string", "description": "Restrict to memories linked to this entity key." },
                "include_content": { "type": "boolean", "description": "If true, return each memory's full text instead of a snippet. Default false." },
                "include_source": { "type": "boolean", "description": "If true, reveal provenance/source metadata. Default false." },
            }),
            &[],
        ),
        tool(
            "entity_search",
            "Search the entity graph by key, canonical name, alias, or type (substring match). Read-only. Returns entity records, not memories — use it to resolve an entity_key or canonical name from a partial term. For memory content use `search`; to traverse outward from a known entity use `graph_neighbors` or `graph_context`.",
            json!({
                "query": { "type": "string", "description": "Substring matched against entity key, canonical name, and aliases." },
                "entity_key": { "type": "string", "description": "Filter to an exact entity key." },
                "entity_type": { "type": "string", "description": "Filter by entity type (e.g. person, project, concept)." },
                "limit": { "type": "integer", "description": "Maximum number of entities to return. Default 10." },
                "include_source": { "type": "boolean", "description": "If true, reveal provenance/source metadata. Default false." },
            }),
            &[],
        ),
        tool(
            "graph_neighbors",
            "Traverse the entity graph outward from a starting entity, returning connected entities and the relationships between them up to a bounded depth. Read-only. Use to explore how an entity connects to others (raw graph structure); use `graph_context` if you want a prose, prompt-ready context pack instead of edges.",
            json!({
                "entity_key": { "type": "string", "description": "Entity key to start the traversal from. Required." },
                "depth": { "type": "integer", "description": "Number of relationship hops to follow. Default 1." },
                "max_edges": { "type": "integer", "description": "Maximum relationships to return (bounds the traversal). Default 50." },
                "include_tombstoned": { "type": "boolean", "description": "If true, include tombstoned (soft-deleted) entities/edges. Default false." },
                "include_source": { "type": "boolean", "description": "If true, reveal provenance/source metadata. Default false." },
            }),
            &["entity_key"],
        ),
        tool(
            "graph_context",
            "Build a compact, prompt-ready context pack centered on an entity: the entity, its graph neighbors, and the most relevant linked memories, budgeted to a character limit. Read-only. Use when an agent needs ready-to-inject context about one specific entity; use `pack` for query-driven context, or `graph_neighbors` for raw graph edges.",
            json!({
                "entity_key": { "type": "string", "description": "Entity key the context pack is centered on. Required." },
                "depth": { "type": "integer", "description": "Number of relationship hops to include. Default 1." },
                "max_edges": { "type": "integer", "description": "Maximum relationships to include. Default 50." },
                "max_memories": { "type": "integer", "description": "Maximum linked memories to include. Default 10." },
                "max_chars": { "type": "integer", "description": "Character budget for the assembled pack. Default 4000." },
                "include_source": { "type": "boolean", "description": "If true, reveal provenance/source metadata. Default false." },
            }),
            &["entity_key"],
        ),
        tool(
            "dream_graph",
            "Preview the graph-maintenance pass in dry-run (proposal-only) mode: surfaces the entity and relationship extractions and merges the nightly `dream` job would make, without writing anything. Read-only; no side effects. Use to inspect what graph changes are pending before they are applied.",
            json!({
                "max_memories": { "type": "integer", "description": "How many recent memories to analyze for proposals. Default 1000." },
                "space": { "type": "string", "description": "Restrict the analysis to a single memory space (namespace)." },
            }),
            &[],
        ),
        tool(
            "remember",
            "Write one durable memory the agent should be able to recall later. Mutating: persists a memory (set dry_run to validate without writing). Store exactly one atomic, self-contained fact, decision, preference, or lesson per call — include enough context that it stands alone (\"the user deploys from the release branch, never main\", not just \"release branch\"). Do not store secrets or raw transcripts. For a plausible-but-unverified inference, use `candidate_submit` instead so a human approves it first.",
            json!({
                "content": { "type": "string", "description": "The memory text: one atomic, self-contained claim with enough context to stand on its own. Required." },
                "space": { "type": "string", "description": "Memory space (namespace) to write into. Omit for the default space." },
                "silo": { "type": "string", "description": "Retention tier (e.g. short-term, durable). Omit to use the space default." },
                "scope": { "type": "string", "description": "Visibility scope: global, workspace, project, session, or custom." },
                "project": { "type": "string", "description": "Free-form project key this memory belongs to." },
                "kind": { "type": "string", "description": "Memory kind (fact, decision, preference, lesson, action, ...). Inferred from the content prefix when omitted." },
                "summary": { "type": "string", "description": "Optional shorter summary of the content." },
                "tags": { "type": "array", "items": { "type": "string" }, "description": "Free-form tags for filtering and retrieval boosts." },
                "entity_key": { "type": "string", "description": "Stable key of the entity this memory is about (groups related memories in the graph)." },
                "claim_key": { "type": "string", "description": "Stable key identifying the claim, used to group versions for supersession." },
                "derive_keys": { "type": "boolean", "description": "Auto-derive entity_key/claim_key from the content when not provided. Default true." },
                "confidence": { "type": "number", "description": "Confidence in the memory, 0.0–1.0. Default 1.0." },
                "observed_at": { "type": "string", "description": "RFC 3339 timestamp of when this was observed. Defaults to now." },
                "valid_from": { "type": "string", "description": "RFC 3339 timestamp the fact starts being true." },
                "valid_to": { "type": "string", "description": "RFC 3339 timestamp the fact stops being true (past values are excluded from recall)." },
                "expires_at": { "type": "string", "description": "RFC 3339 timestamp after which the memory is dropped from recall." },
                "pinned": { "type": "boolean", "description": "If true, exempt from automatic eviction. Default false." },
                "supersedes": { "type": "array", "items": { "type": "string" }, "description": "Memory ids this memory replaces (they become superseded)." },
                "contradicts": { "type": "array", "items": { "type": "string" }, "description": "Memory ids this memory conflicts with." },
                "verified_against": { "type": "string", "description": "What this memory was checked against, if any." },
                "mode": { "type": "string", "enum": ["auto", "append", "supersede", "suggest", "conflict"], "description": "How to resolve against existing memories sharing the same entity/claim key. Default auto." },
                "source_type": { "type": "string", "description": "Provenance: assistant-inference (default) when the agent inferred it, or explicit-user when the user stated it directly." },
                "sensitivity": { "type": "string", "enum": ["normal", "sensitive"], "description": "Mark sensitive to flag the memory for stricter handling. Default normal." },
                "dry_run": { "type": "boolean", "description": "If true, validate and return what would be written without persisting. Default false." },
            }),
            &["content"],
        ),
        tool(
            "forget",
            "Retire one specific memory by id. Mutating: tombstones the memory (a soft delete that preserves audit history), so it stops surfacing in recall; it is not a hard delete. \
Set mode='correct' when retiring a memory because it is WRONG (e.g. a surfaced/recalled \
fact the user contradicted), as opposed to routine cleanup: this records a distinct \
`correct` event with the memory's provenance, and if you pass corrected_by (the id of the \
memory holding the right answer) it also records a `contradicts` link. Use mode='correct' \
for factual corrections so the signal is captured explicitly rather than inferred later.",
            json!({
                "memory_id": { "type": "string", "description": "Id of the memory to retire. Required." },
                "reason": { "type": "string", "description": "Why the memory is being retired (recorded in the audit trail)." },
                "mode": { "type": "string", "enum": ["tombstone", "correct"], "description": "tombstone (default) for routine cleanup; correct when the memory was factually wrong (records a correction signal)." },
                "corrected_by": { "type": "string", "description": "With mode='correct', the id of the memory holding the right answer (records a contradicts link)." },
                "dry_run": { "type": "boolean", "description": "If true, validate without retiring. Default false." },
            }),
            &["memory_id"],
        ),
        tool(
            "entity_upsert",
            "Create or update one entity in the graph projection (register it, rename it, or add aliases). Mutating. The graph is a rebuildable projection over memories, which remain the source of truth — use this to curate entity identity, not to store facts (use `remember` for facts).",
            json!({
                "entity_key": { "type": "string", "description": "Stable, unique key identifying the entity. Required." },
                "canonical_name": { "type": "string", "description": "Primary display name for the entity. Required." },
                "entity_type": { "type": "string", "description": "Type of entity (e.g. person, project, concept, tool)." },
                "aliases": { "type": "array", "items": { "type": "string" }, "description": "Alternate names/surface forms that should resolve to this entity." },
                "space": { "type": "string", "description": "Memory space (namespace) the entity belongs to." },
                "status": { "type": "string", "description": "Lifecycle status (e.g. active, tombstoned)." },
                "confidence": { "type": "number", "description": "Confidence in the entity, 0.0–1.0." },
                "source_episode_id": { "type": "string", "description": "Id of the source episode this entity was derived from, if any." },
                "metadata": { "type": "object", "description": "Arbitrary key/value attributes to attach to the entity." },
                "include_source": { "type": "boolean", "description": "If true, reveal provenance/source metadata in the response. Default false." },
            }),
            &["entity_key", "canonical_name"],
        ),
        tool(
            "relationship_upsert",
            "Create or update one directed relationship in the graph: subject --relation_type--> object. Mutating. Identify each endpoint by entity_key (preferred) or internal entity_id. The graph is a rebuildable projection over memories — curate structure here, store facts with `remember`.",
            json!({
                "relation_type": { "type": "string", "description": "The relationship type/predicate (e.g. depends_on, works_with, part_of). Required." },
                "subject_entity_key": { "type": "string", "description": "Entity key of the subject (source) endpoint. Preferred over subject_entity_id." },
                "object_entity_key": { "type": "string", "description": "Entity key of the object (target) endpoint. Preferred over object_entity_id." },
                "subject_entity_id": { "type": "string", "description": "Internal id of the subject endpoint (alternative to subject_entity_key)." },
                "object_entity_id": { "type": "string", "description": "Internal id of the object endpoint (alternative to object_entity_key)." },
                "memory_id": { "type": "string", "description": "Id of the memory this relationship was derived from, if any." },
                "space": { "type": "string", "description": "Memory space (namespace) the relationship belongs to." },
                "source_episode_id": { "type": "string", "description": "Id of the source episode this relationship was derived from, if any." },
                "status": { "type": "string", "description": "Lifecycle status (e.g. active, tombstoned)." },
                "confidence": { "type": "number", "description": "Confidence in the relationship, 0.0–1.0." },
                "observed_at": { "type": "string", "description": "RFC 3339 timestamp of when this was observed." },
                "valid_from": { "type": "string", "description": "RFC 3339 timestamp the relationship starts being valid." },
                "valid_to": { "type": "string", "description": "RFC 3339 timestamp the relationship stops being valid." },
                "metadata": { "type": "object", "description": "Arbitrary key/value attributes to attach to the relationship." },
                "include_source": { "type": "boolean", "description": "If true, reveal provenance/source metadata in the response. Default false." },
            }),
            &["relation_type"],
        ),
        tool(
            "verify",
            "Re-confirm that an existing memory is still accurate as of now, stamping its last-verified time. Mutating: updates verification metadata only — it does NOT change the memory's content or promote it to a durable tier. If the value has CHANGED, do not verify; write a new memory with `remember` and supersede the old one instead.",
            json!({
                "memory_id": { "type": "string", "description": "Id of the memory being re-confirmed. Required." },
                "verified_against": { "type": "string", "description": "The source or ground truth the memory was checked against." },
            }),
            &["memory_id"],
        ),
        tool(
            "pack",
            "Assemble a compact, prompt-ready context block from one or more queries: retrieves, reranks, and budgets the top memories into injectable text. Read-only. This is the retrieval path for putting memory into an agent's prompt; use `search` instead when you want individual scored records rather than an assembled block.",
            json!({
                "queries": { "type": "array", "items": { "type": "string" }, "description": "One or more natural-language queries to retrieve and merge into the pack. Required." },
                "title": { "type": "string", "description": "Heading for the assembled pack. Default \"context\"." },
                "max_memories": { "type": "integer", "description": "Maximum memories to include in the pack. Default 10." },
                "max_chars": { "type": "integer", "description": "Character budget for the assembled pack. Default 6000." },
                "min_score": { "type": "number", "description": "Drop memories scoring below this threshold; the pack abstains (returns empty) when nothing clears it. Default 0 (no floor)." },
                "query_expansion": { "type": "boolean", "description": "Default false. Deterministically add subqueries before retrieval." },
                "thread_expansion": { "type": "boolean", "description": "Default false. Add same-entity/same-claim neighbors to the rerank pool." },
                "max_query_variants": { "type": "integer", "description": "Default engine maximum." },
                "max_thread_seeds": { "type": "integer", "description": "Default 3." },
                "max_thread_neighbors": { "type": "integer", "description": "Default 3." },
                "space": { "type": "string", "description": "Restrict retrieval to a single memory space (namespace)." },
                "tags": { "type": "array", "items": { "type": "string" }, "description": "Restrict retrieval to memories carrying these tags." },
            }),
            &["queries"],
        ),
        tool(
            "candidate_submit",
            "Queue a proposed memory for human review instead of writing it to recall directly. Mutating: adds an item to the review queue (it does not enter recall until a human approves it via CLI/dashboard). Use this for plausible-but-unverified inferences; use `remember` when the fact is confirmed and should be recallable immediately.",
            json!({
                "content": { "type": "string", "description": "The proposed memory text: one atomic, self-contained claim. Required." },
                "rationale": { "type": "string", "description": "Why you are proposing this (evidence/reasoning) to help the human reviewer decide." },
                "kind": { "type": "string", "description": "Memory kind (fact, decision, preference, lesson, ...)." },
                "summary": { "type": "string", "description": "Optional shorter summary of the content." },
                "tags": { "type": "array", "items": { "type": "string" }, "description": "Free-form tags." },
                "entity_key": { "type": "string", "description": "Stable key of the entity this memory is about." },
                "claim_key": { "type": "string", "description": "Stable key identifying the claim." },
                "confidence": { "type": "number", "description": "Confidence in the proposed memory, 0.0–1.0." },
                "source_type": { "type": "string", "description": "Provenance: assistant-inference (default) or explicit-user." },
                "sensitivity": { "type": "string", "description": "normal (default) or sensitive." },
                "space": { "type": "string", "description": "Memory space (namespace) the candidate targets." },
                "silo": { "type": "string", "description": "Retention tier the candidate targets (e.g. short-term, durable)." },
                "scope": { "type": "string", "description": "Visibility scope: global, workspace, project, session, or custom." },
                "project": { "type": "string", "description": "Free-form project key." },
                "supersedes": { "type": "array", "items": { "type": "string" }, "description": "Memory ids this candidate would replace if approved." },
                "dry_run": { "type": "boolean", "description": "If true, validate without enqueuing. Default false." },
            }),
            &["content"],
        ),
        tool(
            "candidate_list",
            "List memories in the human-review queue, filtered by review status. Read-only. Use to see what has been proposed via `candidate_submit` and its disposition; approving or rejecting candidates is a human action in the CLI/dashboard.",
            json!({
                "status": { "type": "string", "enum": ["pending", "approved", "rejected"], "description": "Which queue to list. Default pending." },
                "space": { "type": "string", "description": "Restrict to a single memory space (namespace)." },
                "limit": { "type": "integer", "description": "Maximum candidates to return. Default 50." },
            }),
            &[],
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            _ => Map::new(),
        }
    }

    fn payload(name: &str, value: Value) -> Value {
        let (_, json) = build_serve_call(name, &args(value)).expect("call builds");
        serde_json::from_str(&json).expect("payload is json")
    }

    #[test]
    fn tool_surface_matches_python_adapter() {
        let definitions = tool_definitions();
        let names: Vec<&str> = definitions
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        let expected = [
            "stats",
            "search",
            "get",
            "memory_list",
            "entity_search",
            "graph_neighbors",
            "graph_context",
            "dream_graph",
            "remember",
            "forget",
            "entity_upsert",
            "relationship_upsert",
            "verify",
            "pack",
            "candidate_submit",
            "candidate_list",
        ];
        assert_eq!(names, expected);
        // candidate approve/reject are intentionally NOT exposed over MCP.
        assert!(!names.contains(&"candidate_approve"));
        assert!(!names.contains(&"candidate_reject"));
    }

    #[test]
    fn every_tool_maps_to_a_real_serve_command() {
        for tool in tool_definitions() {
            let name = tool["name"].as_str().unwrap();
            // Supply the required args so the mapping does not error on missing fields.
            let mut filled = Map::new();
            for required in tool["inputSchema"]["required"].as_array().unwrap() {
                let key = required.as_str().unwrap();
                let value = if key == "queries" {
                    json!(["q"])
                } else {
                    json!("x")
                };
                filled.insert(key.to_string(), value);
            }
            let (command, _payload) =
                build_serve_call(name, &filled).expect("required-args mapping builds");
            assert!(
                Command::parse(command).is_some(),
                "tool {name} maps to unknown serve command {command}"
            );
        }
    }

    #[test]
    fn search_builds_reranked_semantic_payload_with_filters() {
        let value = payload(
            "search",
            json!({ "query": "hello", "space": "workspace-memory", "tags": ["a"], "entity_key": "e1" }),
        );
        assert_eq!(value["query"], json!("hello"));
        assert_eq!(value["limit"], json!(10));
        assert_eq!(value["rerank"], json!(true));
        assert_eq!(value["rerank_candidates"], json!(16));
        assert_eq!(value["semantic_fallback"], json!("fallback"));
        assert_eq!(value["filters"]["spaces"], json!(["workspace-memory"]));
        assert_eq!(value["filters"]["tags"], json!(["a"]));
        assert_eq!(value["filters"]["entity_keys"], json!(["e1"]));
    }

    #[test]
    fn search_disabled_semantic_and_zero_limit_skips_rerank() {
        let value = payload(
            "search",
            json!({ "query": "x", "limit": 0, "semantic_fallback": false }),
        );
        assert_eq!(value["semantic_fallback"], json!("disabled"));
        assert!(value.get("rerank").is_none());
    }

    #[test]
    fn remember_carries_mcp_source_and_mode() {
        let value = payload(
            "remember",
            json!({ "content": "fact", "source_type": "explicit-user", "mode": "supersede", "tags": ["t"] }),
        );
        assert_eq!(value["content"], json!("fact"));
        assert_eq!(value["mode"], json!("supersede"));
        assert_eq!(value["derive_keys"], json!(true));
        assert_eq!(value["source"]["type"], json!("mcp"));
        assert_eq!(value["source"]["source_type"], json!("explicit-user"));
        assert_eq!(value["tags"], json!(["t"]));
    }

    #[test]
    fn remember_verified_against_becomes_metadata_json() {
        let value = payload(
            "remember",
            json!({ "content": "f", "verified_against": "env:X" }),
        );
        let metadata: Value =
            serde_json::from_str(value["metadata_json"].as_str().unwrap()).unwrap();
        assert_eq!(metadata["verified_against"], json!("env:X"));
    }

    #[test]
    fn get_and_verify_use_correct_id_fields() {
        assert_eq!(
            payload("get", json!({ "memory_id": "m1" }))["id"],
            json!("m1")
        );
        assert_eq!(
            payload("verify", json!({ "memory_id": "m1" }))["memory_id"],
            json!("m1")
        );
        assert_eq!(
            payload("forget", json!({ "memory_id": "m1" }))["id"],
            json!("m1")
        );
    }

    #[test]
    fn dream_graph_wraps_dream_dry_run() {
        let (command, json) = build_serve_call("dream_graph", &args(json!({}))).unwrap();
        assert_eq!(command, "dream");
        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(value["tasks"], json!(["graph"]));
        assert_eq!(value["dry_run"], json!(true));
        assert_eq!(value["max_memories"], json!(1000));
    }

    #[test]
    fn candidate_list_defaults_to_pending() {
        let value = payload("candidate_list", json!({}));
        assert_eq!(value["status"], json!("pending"));
        assert_eq!(value["limit"], json!(50));
    }

    #[test]
    fn missing_required_argument_is_a_client_error() {
        let error = build_serve_call("search", &args(json!({}))).unwrap_err();
        assert!(error.contains("query"));
        let unknown = build_serve_call("does_not_exist", &args(json!({}))).unwrap_err();
        assert!(unknown.contains("unknown tool"));
    }

    #[test]
    fn serialized_daemon_request_round_trips_through_parser() {
        let envelope = ServeRequestEnvelope {
            request_id: Some("mcp-search".to_string()),
            command: Command::Search,
            command_name: "search".to_string(),
            store_path: Some(std::path::PathBuf::from("/tmp/store.sqlite")),
            payload_json: r#"{"query":"hello","limit":5}"#.to_string(),
        };
        let line = serialize_serve_request_line(&envelope);
        // Exactly one newline-terminated JSON line: the daemon reads up to '\n'.
        assert!(line.ends_with('\n'));
        assert_eq!(line.matches('\n').count(), 1);
        // The engine's own parser must accept what the client emits — this is the
        // contract that keeps the native client and the daemon in lockstep.
        let parsed = parse_serve_request(line.trim()).expect("daemon accepts request line");
        assert_eq!(parsed.command_name, "search");
        assert_eq!(parsed.request_id.as_deref(), Some("mcp-search"));
        assert_eq!(
            parsed.store_path,
            Some(std::path::PathBuf::from("/tmp/store.sqlite"))
        );
        let payload: Value = serde_json::from_str(&parsed.payload_json).unwrap();
        assert_eq!(payload["query"], json!("hello"));
        assert_eq!(payload["limit"], json!(5));
    }

    #[test]
    fn daemon_failure_envelope_reports_not_ok() {
        let envelope = daemon_failure_envelope("search", "connection refused");
        // A transport failure must surface as ok:false so the MCP layer flags
        // isError — never a silently-empty success.
        assert!(!envelope_ok(&envelope));
        let value: Value = serde_json::from_str(&envelope).expect("valid JSON envelope");
        assert_eq!(value["command"], json!("search"));
        assert_eq!(value["error"]["code"], json!("daemon_unreachable"));
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("connection refused"));
    }

    #[test]
    fn initialize_echoes_requested_protocol_version() {
        let result = initialize_result(&json!({ "protocolVersion": "2024-11-05" }));
        assert_eq!(result["protocolVersion"], json!("2024-11-05"));
        assert_eq!(result["serverInfo"]["name"], json!("memkeeper"));
        assert_eq!(result["capabilities"]["tools"], json!({}));
    }

    #[test]
    fn initialize_falls_back_to_default_protocol_version() {
        let result = initialize_result(&json!({}));
        assert_eq!(
            result["protocolVersion"],
            json!(DEFAULT_MCP_PROTOCOL_VERSION)
        );
    }

    #[test]
    fn notifications_get_no_response() {
        let store = Path::new("/tmp/does-not-matter.sqlite");
        let backend = McpBackend::InProcess(Box::new(SemanticModels::for_serve()));
        let response = handle_mcp_line(
            r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
            store,
            &backend,
        );
        assert!(response.is_none());
    }

    #[test]
    fn unknown_method_returns_method_not_found() {
        let store = Path::new("/tmp/does-not-matter.sqlite");
        let backend = McpBackend::InProcess(Box::new(SemanticModels::for_serve()));
        let response = handle_mcp_line(
            r#"{"jsonrpc":"2.0","id":7,"method":"bogus/method"}"#,
            store,
            &backend,
        )
        .expect("request with id gets a response");
        assert_eq!(response["error"]["code"], json!(-32601));
        assert_eq!(response["id"], json!(7));
    }

    #[test]
    fn tools_list_is_a_well_formed_response() {
        let store = Path::new("/tmp/does-not-matter.sqlite");
        let backend = McpBackend::InProcess(Box::new(SemanticModels::for_serve()));
        let response = handle_mcp_line(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
            store,
            &backend,
        )
        .expect("response");
        assert_eq!(response["jsonrpc"], json!("2.0"));
        assert_eq!(response["result"]["tools"].as_array().unwrap().len(), 16);
    }

    #[test]
    fn envelope_ok_detects_success_and_failure() {
        assert!(envelope_ok(r#"{"ok":true}"#));
        assert!(!envelope_ok(r#"{"ok":false}"#));
        assert!(!envelope_ok("not json"));
    }

    #[test]
    fn surfaced_events_parse_search_results() {
        let response =
            r#"{"ok":true,"result":{"results":[{"memory_id":"m1","rank":1,"score":0.5}]}}"#;
        let events = surfaced_events(response, "q");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["memory_id"], json!("m1"));
        assert_eq!(events[0]["kind"], json!("surfaced"));
        assert_eq!(events[0]["query"], json!("q"));
    }
}
