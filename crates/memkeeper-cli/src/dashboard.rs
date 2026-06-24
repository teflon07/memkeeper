//! Local read-only HTTP dashboard for `memkeeper serve --http`.
//!
//! A tiny std-only HTTP/1.1 server (no async runtime, no extra dependencies,
//! matching the Unix-socket serve idiom) that serves an embedded single-page
//! app and proxies a read-only subset of the serve commands as JSON. The SPA
//! never learns the store path: the server resolves the default store once and
//! injects it into every request, and a command allowlist rejects all writes.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
use std::time::{Duration, Instant};

use memkeeper_protocol::Command;

use crate::json::{parse_json, JsonValue};
use crate::{
    execute_serve_request, extract_serve_identity, parse_serve_request, resolve_store_default,
    serve_failure_envelope, CliError, SemanticModels,
};

/// Default bind address: loopback only, so the dashboard never touches the
/// network unless the operator explicitly binds elsewhere.
pub(crate) const DEFAULT_HTTP_ADDR: &str = "127.0.0.1:7777";

/// Reject request bodies larger than this (matches the CLI's `MAX_JSON_BYTES`).
const MAX_BODY_BYTES: usize = 1_048_576;

// Embedded SPA assets. Bundling them into the binary keeps the dashboard
// self-contained (no asset directory to ship) and lets the extractor carry
// them through as part of the crate source tree.
const INDEX_HTML: &str = include_str!("dashboard/index.html");
const APP_JS: &str = include_str!("dashboard/app.js");
const STYLE_CSS: &str = include_str!("dashboard/style.css");
const CYTOSCAPE_JS: &str = include_str!("dashboard/cytoscape.min.js");

/// Commands the dashboard may invoke. Read-only browsing/diagnostics only;
/// everything else (writes, maintenance) is denied. New `Command` variants
/// default to denied (fail closed).
fn is_read_only(command: Command) -> bool {
    matches!(
        command,
        Command::Search
            | Command::MemoryList
            | Command::Get
            | Command::Stats
            | Command::EntitySearch
            | Command::GraphNeighbors
            | Command::GraphContext
            | Command::GraphFull
            | Command::History
            | Command::SpaceList
            | Command::SiloList
            | Command::Doctor
            | Command::DocumentSearch
            | Command::DocumentGet
            | Command::PromotionCandidates
            | Command::DocumentDuplicates
    )
}

/// Commands permitted over HTTP when (and only when) a valid write token is
/// presented. Kept minimal: the hosted document-ingestion path. Everything else
/// stays denied over HTTP regardless of auth.
fn is_auth_writable(command: Command) -> bool {
    matches!(command, Command::Ingest | Command::DocumentPrune)
}

/// Constant-time byte comparison, to avoid leaking the token via timing. The
/// length check leaks only the token length, which is not secret.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Authorize a write: a non-empty token must be configured AND the request must
/// carry a matching `Authorization: Bearer <token>`. Fails closed.
fn authorize_write(write_token: Option<&str>, authorization: Option<&str>) -> bool {
    let Some(token) = write_token.filter(|token| !token.is_empty()) else {
        return false;
    };
    let Some(header) = authorization else {
        return false;
    };
    let presented = header
        .strip_prefix("Bearer ")
        .or_else(|| header.strip_prefix("bearer "))
        .unwrap_or("")
        .trim();
    constant_time_eq(presented.as_bytes(), token.as_bytes())
}

/// Resolve a static asset path to `(content_type, bytes)`. Returns `None` for
/// any unknown path — there is no filesystem lookup, so path traversal is moot.
fn asset(path: &str) -> Option<(&'static str, &'static [u8])> {
    match path {
        "/assets/app.js" => Some(("text/javascript; charset=utf-8", APP_JS.as_bytes())),
        "/assets/style.css" => Some(("text/css; charset=utf-8", STYLE_CSS.as_bytes())),
        "/assets/cytoscape.min.js" => {
            Some(("text/javascript; charset=utf-8", CYTOSCAPE_JS.as_bytes()))
        }
        _ => None,
    }
}

/// Parse, authorize, and dispatch one `/api` request body, returning the serve
/// envelope JSON (success or failure). Store path defaults are injected here.
fn dispatch_api(
    body: &str,
    default_store: &Path,
    models: &SemanticModels,
    write_token: Option<&str>,
    authorization: Option<&str>,
) -> String {
    let started = Instant::now();
    let mut request = match parse_serve_request(body) {
        Ok(request) => request,
        Err(error) => {
            let (request_id, command_name) = extract_serve_identity(body);
            return serve_failure_envelope(
                request_id.as_deref(),
                command_name.as_deref().unwrap_or("unknown"),
                &error,
                started,
            );
        }
    };
    let permitted = is_read_only(request.command)
        || (is_auth_writable(request.command) && authorize_write(write_token, authorization));
    if !permitted {
        let error = CliError::InvalidRequest(format!(
            "command '{}' is not permitted over the dashboard (writes require a valid Authorization token)",
            request.command_name
        ));
        return serve_failure_envelope(
            request.request_id.as_deref(),
            &request.command_name,
            &error,
            started,
        );
    }
    if request.store_path.is_none() {
        request.store_path = Some(default_store.to_path_buf());
    }
    // The dashboard is read-only: document search must not write recall
    // telemetry (which feeds usage-based promotion), so force `skip_recall_log`
    // on regardless of what the client sent.
    if request.command == Command::DocumentSearch {
        request.payload_json = force_skip_recall_log(&request.payload_json);
    }
    execute_serve_request(&request, started, models)
}

/// Force `skip_recall_log: true` onto a document-search payload so the read-only
/// dashboard never records retrieval telemetry. Leaves the payload untouched if
/// it does not parse as an object (the downstream parser will reject it).
fn force_skip_recall_log(payload_json: &str) -> String {
    match parse_json(payload_json) {
        Ok(JsonValue::Object(mut object)) => {
            object.set("skip_recall_log", JsonValue::Bool(true));
            JsonValue::Object(object).to_json()
        }
        _ => payload_json.to_string(),
    }
}

/// Compute the response for one request. Asset/SPA routes are static; `/api`
/// dispatches to the read-only command path.
#[allow(clippy::too_many_arguments)]
fn route(
    method: &str,
    path: &str,
    body: &str,
    store: &Path,
    models: &SemanticModels,
    write_token: Option<&str>,
    authorization: Option<&str>,
) -> (&'static str, &'static str, Vec<u8>) {
    if method == "GET" {
        if path == "/" || path == "/index.html" {
            return (
                "200 OK",
                "text/html; charset=utf-8",
                INDEX_HTML.as_bytes().to_vec(),
            );
        }
        if let Some((content_type, bytes)) = asset(path) {
            return ("200 OK", content_type, bytes.to_vec());
        }
        if path == "/healthz" {
            return ("200 OK", "text/plain; charset=utf-8", b"ok".to_vec());
        }
    }
    if method == "POST" && path == "/api" {
        return (
            "200 OK",
            "application/json; charset=utf-8",
            dispatch_api(body, store, models, write_token, authorization).into_bytes(),
        );
    }
    (
        "404 Not Found",
        "text/plain; charset=utf-8",
        b"not found".to_vec(),
    )
}

fn write_response(writer: &mut impl Write, status: &str, content_type: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Connection: close\r\nCache-Control: no-store\r\n\r\n",
        body.len()
    );
    let _ = writer.write_all(header.as_bytes());
    let _ = writer.write_all(body);
    let _ = writer.flush();
}

/// Handle one HTTP/1.1 connection: parse the request line + headers, read the
/// body by `Content-Length`, route, and write one `Connection: close` response.
fn handle_connection(
    stream: TcpStream,
    store: &Path,
    models: &SemanticModels,
    write_token: Option<&str>,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
    let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(stream);

    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() || request_line.trim().is_empty() {
        return;
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("");
    let path = target.split('?').next().unwrap_or("").to_string();

    let mut content_length = 0usize;
    let mut authorization: Option<String> = None;
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Err(_) => return,
            Ok(_) => {}
        }
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((name, value)) = trimmed.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                content_length = value.trim().parse().unwrap_or(0);
            } else if name.trim().eq_ignore_ascii_case("authorization") {
                authorization = Some(value.trim().to_string());
            }
        }
    }

    let body = if method == "POST" && content_length > 0 {
        if content_length > MAX_BODY_BYTES {
            write_response(
                &mut writer,
                "413 Payload Too Large",
                "text/plain",
                b"too large",
            );
            return;
        }
        let mut buf = vec![0u8; content_length];
        if reader.read_exact(&mut buf).is_err() {
            return;
        }
        String::from_utf8(buf).unwrap_or_default()
    } else {
        String::new()
    };

    let (status, content_type, payload) = route(
        &method,
        &path,
        &body,
        store,
        models,
        write_token,
        authorization.as_deref(),
    );
    write_response(&mut writer, status, content_type, &payload);
}

/// Whether `addr` (a `host:port` string) resolves to a loopback host. A
/// hostname we cannot prove is loopback returns `false` so the bind guard fails
/// closed (treats it as remote unless explicitly overridden).
fn addr_is_loopback(addr: &str) -> bool {
    let host = addr.rsplit_once(':').map_or(addr, |(host, _port)| host);
    let host = host.trim_start_matches('[').trim_end_matches(']');
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    host.parse::<std::net::IpAddr>()
        .is_ok_and(|ip| ip.is_loopback())
}

/// Serve the read-only dashboard over HTTP until interrupted. One thread per
/// connection (store opened per request, models behind mutexes — same safety
/// model as the Unix-socket serve mode).
pub(crate) fn run_serve_http(
    addr: &str,
    models: &SemanticModels,
    store_override: Option<&Path>,
) -> i32 {
    // Binding off-loopback publishes the entire read-only store (search, get,
    // history, stats, full graph, doctor config) to the network with no auth.
    // Fail closed: refuse unless the operator explicitly opts in, and warn
    // loudly when they do — silent network exposure violates the loud-failure
    // floor.
    if !addr_is_loopback(addr) {
        if crate::serve::env_flag_enabled("MEMKEEPER_ALLOW_REMOTE") {
            eprintln!(
                "[memkeeper] WARNING: dashboard bound to non-loopback address {addr}; the read-only store is exposed to the network with NO authentication (MEMKEEPER_ALLOW_REMOTE override active)."
            );
        } else {
            eprintln!(
                "[memkeeper] refusing to bind dashboard to non-loopback address {addr}: this exposes the entire read-only memory store to the network with no authentication."
            );
            eprintln!(
                "[memkeeper] set MEMKEEPER_ALLOW_REMOTE=1 to override (anyone who can reach {addr} can then read all stored memories)."
            );
            return 1;
        }
    }
    // Optional authenticated write path: when MEMKEEPER_HTTP_WRITE_TOKEN is set,
    // `ingest` is accepted with a matching `Authorization: Bearer <token>`.
    // Unset/blank => all writes stay denied (fail closed).
    let write_token = std::env::var("MEMKEEPER_HTTP_WRITE_TOKEN")
        .ok()
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty());
    if write_token.is_some() {
        eprintln!("[memkeeper] authenticated ingest enabled (MEMKEEPER_HTTP_WRITE_TOKEN set).");
        if !addr_is_loopback(addr) {
            eprintln!(
                "[memkeeper] WARNING: the write token travels in cleartext over plain HTTP; put a TLS-terminating proxy in front of {addr} for remote use."
            );
        }
    }

    // `--store` overrides the default resolution (env `MEMKEEPER_STORE`, then the
    // user/project default). The dashboard never learns this from the client.
    let store = store_override.map_or_else(resolve_store_default, Path::to_path_buf);
    let listener = match TcpListener::bind(addr) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("[memkeeper] dashboard failed to bind {addr}: {error}");
            return 1;
        }
    };
    eprintln!(
        "[memkeeper] dashboard serving on http://{addr}  (store: {})",
        store.display()
    );
    let write_token = write_token.as_deref();
    std::thread::scope(|scope| {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let store = store.as_path();
            scope.spawn(move || handle_connection(stream, store, models, write_token));
        }
    });
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_allowlist_permits_browsing_and_blocks_writes() {
        assert!(is_read_only(Command::Search));
        assert!(is_read_only(Command::MemoryList));
        assert!(is_read_only(Command::Get));
        assert!(is_read_only(Command::Stats));
        assert!(is_read_only(Command::EntitySearch));
        assert!(is_read_only(Command::GraphNeighbors));
        assert!(!is_read_only(Command::Remember));
        assert!(!is_read_only(Command::Forget));
        assert!(!is_read_only(Command::Dream));
        assert!(!is_read_only(Command::Import));
        assert!(!is_read_only(Command::EntityUpsert));
        // Document reads are browsable; ingest is a write (never read-only).
        assert!(is_read_only(Command::DocumentSearch));
        assert!(is_read_only(Command::DocumentGet));
        assert!(is_read_only(Command::PromotionCandidates));
        assert!(is_read_only(Command::DocumentDuplicates));
        assert!(!is_read_only(Command::Ingest));
    }

    #[test]
    fn only_document_writes_are_auth_writable() {
        assert!(is_auth_writable(Command::Ingest));
        assert!(is_auth_writable(Command::DocumentPrune));
        assert!(!is_auth_writable(Command::Remember));
        assert!(!is_auth_writable(Command::MarkExtracted));
        assert!(!is_auth_writable(Command::Forget));
        assert!(!is_auth_writable(Command::Search));
        // Prune is a write: it must NOT be in the read-only allowlist.
        assert!(!is_read_only(Command::DocumentPrune));
    }

    #[test]
    fn force_skip_recall_log_overrides_client_value() {
        // Read-only dashboard must suppress recall telemetry regardless of the
        // client-supplied flag.
        let forced = force_skip_recall_log(r#"{"query":"x","skip_recall_log":false}"#);
        let object = parse_json(&forced).unwrap();
        let object = object.as_object().unwrap();
        assert_eq!(object.get("skip_recall_log"), Some(&JsonValue::Bool(true)));
        // Absent flag is added.
        let added = force_skip_recall_log(r#"{"query":"x"}"#);
        let added = parse_json(&added).unwrap();
        assert_eq!(
            added.as_object().unwrap().get("skip_recall_log"),
            Some(&JsonValue::Bool(true))
        );
    }

    #[test]
    fn constant_time_eq_matches_only_identical_bytes() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secre"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn authorize_write_fails_closed() {
        // No token configured => never authorized, even with a header.
        assert!(!authorize_write(None, Some("Bearer anything")));
        assert!(!authorize_write(Some(""), Some("Bearer anything")));
        // Token configured but missing/blank/wrong header.
        assert!(!authorize_write(Some("tok"), None));
        assert!(!authorize_write(Some("tok"), Some("")));
        assert!(!authorize_write(Some("tok"), Some("Bearer wrong")));
        assert!(!authorize_write(Some("tok"), Some("tok"))); // missing Bearer prefix
                                                             // Correct bearer token (either case prefix).
        assert!(authorize_write(Some("tok"), Some("Bearer tok")));
        assert!(authorize_write(Some("tok"), Some("bearer tok")));
    }

    #[test]
    fn loopback_detection_fails_closed_for_remote_and_unknown_hosts() {
        // Loopback hosts the bind guard must allow without an opt-in.
        assert!(addr_is_loopback("127.0.0.1:7777"));
        assert!(addr_is_loopback("127.0.0.1:0"));
        assert!(addr_is_loopback("[::1]:7777"));
        assert!(addr_is_loopback("localhost:7777"));
        assert!(addr_is_loopback("LocalHost:7777"));
        // Non-loopback or unprovable hosts must be treated as remote (guarded).
        assert!(!addr_is_loopback("0.0.0.0:8080"));
        assert!(!addr_is_loopback("192.168.1.10:8080"));
        assert!(!addr_is_loopback("[::]:8080"));
        assert!(!addr_is_loopback("example.com:8080"));
    }

    #[test]
    fn asset_routes_only_known_paths() {
        assert!(asset("/assets/app.js").is_some());
        assert!(asset("/assets/style.css").is_some());
        assert!(asset("/assets/cytoscape.min.js").is_some());
        assert!(asset("/assets/../Cargo.toml").is_none());
        assert!(asset("/etc/passwd").is_none());
        assert!(asset("/nope").is_none());
    }

    #[test]
    fn dispatch_rejects_non_read_command_before_touching_store() {
        let models = SemanticModels::for_serve();
        let store = Path::new("/nonexistent/store.sqlite");
        let response = dispatch_api(
            r#"{"command":"remember","payload":{}}"#,
            store,
            &models,
            None,
            None,
        );
        assert!(response.contains("\"ok\":false"));
        assert!(response.contains("not permitted"));
    }

    #[test]
    fn dispatch_denies_ingest_without_token_and_opens_with_token() {
        let models = SemanticModels::for_serve();
        let store = Path::new("/nonexistent/store.sqlite");
        let body = r#"{"command":"ingest","payload":{"chunks":["x"]}}"#;
        // No token => denied at the gate (never reaches the store).
        let denied = dispatch_api(body, store, &models, None, Some("Bearer t"));
        assert!(denied.contains("\"ok\":false"));
        assert!(denied.contains("not permitted"));
        // Valid token => passes the gate (then fails at the missing store, which
        // is a different error, proving authorization succeeded).
        let opened = dispatch_api(body, store, &models, Some("t"), Some("Bearer t"));
        assert!(!opened.contains("not permitted"));
    }

    #[test]
    fn dispatch_reports_parse_error_as_failure_envelope() {
        let models = SemanticModels::for_serve();
        let store = Path::new("/nonexistent/store.sqlite");
        let response = dispatch_api("{not valid json", store, &models, None, None);
        assert!(response.contains("\"ok\":false"));
    }

    #[test]
    fn unknown_route_is_404() {
        let models = SemanticModels::for_serve();
        let store = Path::new("/nonexistent/store.sqlite");
        let (status, _, _) = route("GET", "/wat", "", store, &models, None, None);
        assert_eq!(status, "404 Not Found");
        let (status, content_type, body) = route("GET", "/", "", store, &models, None, None);
        assert_eq!(status, "200 OK");
        assert_eq!(content_type, "text/html; charset=utf-8");
        assert!(!body.is_empty());
    }
}
