//! Long-lived serve mode: stdio/Unix-socket request loops, envelope parsing, dispatch.

#[allow(clippy::wildcard_imports)]
use super::*;
#[allow(clippy::wildcard_imports)]
use crate::json::*;
#[allow(clippy::wildcard_imports)]
use crate::output::*;
#[allow(clippy::wildcard_imports)]
use crate::requests::*;

/// Serve posture when the binary's semantic capability is checked at startup.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SemanticGuard {
    /// Built with semantic features — serve normally.
    Ok,
    /// Non-semantic build, no hard requirement — serve FTS-only but warn loudly.
    Degraded,
    /// Non-semantic build and `MEMKEEPER_REQUIRE_SEMANTIC` set — refuse to serve.
    Refuse,
}

/// Pure decision: given whether this binary was built with semantic features
/// and whether the operator requires semantic, what should `serve` do? Kept
/// pure so it is unit-testable without spawning a daemon.
pub(crate) fn semantic_guard(semantic_built: bool, require_semantic: bool) -> SemanticGuard {
    if semantic_built {
        SemanticGuard::Ok
    } else if require_semantic {
        SemanticGuard::Refuse
    } else {
        SemanticGuard::Degraded
    }
}

/// Serve posture *after* models are loaded. A semantic build whose model files
/// are missing at runtime (`MEMKEEPER_EMBED_MODEL_DIR` unset, or the files gone)
/// silently produces no vectors and degrades retrieval to BM25 — the same silent
/// failure the build-time guard exists to prevent, one layer deeper. This catches
/// the runtime case so the daemon can refuse (or at least shout) instead of
/// quietly serving worse results.
#[cfg(feature = "embed")]
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RuntimeGuard {
    /// Embedder loaded — serve normally.
    Ok,
    /// Models absent, no hard requirement — serve FTS-only but warn loudly.
    Degraded,
    /// Models absent and `MEMKEEPER_REQUIRE_SEMANTIC` set — refuse to serve.
    Refuse,
}

/// Pure decision mirroring `semantic_guard`, but keyed on whether the embedder
/// actually loaded rather than on the build features.
#[cfg(feature = "embed")]
pub(crate) fn runtime_semantic_guard(embed_active: bool, require_semantic: bool) -> RuntimeGuard {
    if embed_active {
        RuntimeGuard::Ok
    } else if require_semantic {
        RuntimeGuard::Refuse
    } else {
        RuntimeGuard::Degraded
    }
}

/// Surface the retrieval mode when a non-semantic (lexical-only) binary serves.
/// Semantic is the default and preferred build; a lexical-only daemon is a
/// supported but lesser mode, so state it visibly without crying wolf — a calm
/// NOTE, no desktop alarm. When the operator has *demanded* semantic via
/// `MEMKEEPER_REQUIRE_SEMANTIC`, serving lexical-only is a genuine failure: escalate
/// to a loud ERROR plus a best-effort macOS desktop notification, and the caller
/// refuses to serve. This keeps the loud-failure floor intact (the require path
/// still fails closed) without mislabeling the expected lexical-only mode.
fn warn_non_semantic(refusing: bool) {
    if refusing {
        eprintln!(
            "[memkeeper] ERROR: MEMKEEPER_REQUIRE_SEMANTIC is set, but this binary was built \
             WITHOUT semantic features — embeddings, rerank, and late-interaction are OFF. \
             Refusing to serve. Rebuild with `cargo build --release` (semantic is the default)."
        );
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("osascript")
                .args([
                    "-e",
                    "display notification \"required semantic, but this is a lexical-only build\" \
                     with title \"memkeeper: refusing to serve\"",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn();
        }
        return;
    }
    eprintln!(
        "[memkeeper] NOTE: this is a lexical-only (BM25/FTS) build — semantic embeddings and \
         rerank are OFF. Semantic retrieval is the default and preferred mode; build from \
         source with `cargo build --release` to enable it. Set MEMKEEPER_REQUIRE_SEMANTIC=1 to \
         refuse lexical-only rather than serve it."
    );
}

/// Surface that a semantic-capable build is serving with no embedder loaded. The
/// hazard (silently falling back to BM25) and tone depend on the backend the
/// binary was built for:
///
/// - **Local build, model files missing** is a setup gap for a binary that
///   *intends* local semantic — keep it loud (ERROR + desktop alarm) and point at
///   `pull-models`, matching `warn_non_semantic`'s posture.
/// - **Off-device (api-only) build with no provider configured** is the expected
///   default of the prebuilt binary (bring-a-key is opt-in) — a calm NOTE that
///   points at the provider/key, no alarm, same as the lexical-only binary.
///
/// Either way, `MEMKEEPER_REQUIRE_SEMANTIC` escalates to a loud ERROR and refuses to
/// serve, so the fail-closed floor holds regardless of backend.
#[cfg(feature = "embed")]
fn warn_models_absent(refusing: bool) {
    // Off-device build (api compiled, local not): the embedder is a remote API,
    // not local model files. Point at the provider/key and stay calm when degraded.
    #[cfg(all(feature = "api", not(feature = "semantic")))]
    {
        if refusing {
            eprintln!(
                "[memkeeper] ERROR: MEMKEEPER_REQUIRE_SEMANTIC is set, but no embeddings provider \
                 is configured — embeddings and rerank are OFF. Refusing to serve. Set \
                 MEMKEEPER_EMBED_PROVIDER=openai with an embeddings API key (e.g. OpenRouter)."
            );
            #[cfg(target_os = "macos")]
            {
                let _ = std::process::Command::new("osascript")
                    .args([
                        "-e",
                        "display notification \"required semantic, but no embeddings API key set\" \
                         with title \"memkeeper: refusing to serve\"",
                    ])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn();
            }
            return;
        }
        eprintln!(
            "[memkeeper] NOTE: serving lexical (BM25/FTS) — no embeddings provider configured. \
             This build does off-device semantic: set MEMKEEPER_EMBED_PROVIDER=openai with an \
             embeddings API key (e.g. OpenRouter) to enable it. Set MEMKEEPER_REQUIRE_SEMANTIC=1 \
             to refuse lexical-only rather than serve it."
        );
    }

    // Local-model build (the `semantic` feature): a binary that intends on-device
    // semantic whose model files did not load at runtime. Keep loud.
    #[cfg(feature = "semantic")]
    {
        let tail = if refusing {
            "MEMKEEPER_REQUIRE_SEMANTIC is set — refusing to serve."
        } else {
            "Serving FTS-only (degraded). Set MEMKEEPER_EMBED_MODEL_DIR (try `memkeeper pull-models`)."
        };
        eprintln!(
            "[memkeeper] ERROR: semantic build but the embed model did not load \
             (MEMKEEPER_EMBED_MODEL_DIR unset or model files missing) — embeddings and \
             rerank are OFF. {tail}"
        );
        #[cfg(target_os = "macos")]
        {
            let _ = std::process::Command::new("osascript")
                .args([
                    "-e",
                    "display notification \"daemon is FTS-only — embed model missing\" \
                     with title \"memkeeper degraded\"",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn();
        }
    }
}

/// Whether the named env var is set truthy (`1`/`true`). Single source of truth
/// for boolean opt-in/opt-out flags across the CLI.
pub(crate) fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

/// Whether `MEMKEEPER_REQUIRE_SEMANTIC` is set truthy (`1`/`true`). Single source
/// of truth for the fail-closed semantic flag, shared by the serve startup
/// guard and the per-request search path.
pub(crate) fn require_semantic_env() -> bool {
    env_flag_enabled("MEMKEEPER_REQUIRE_SEMANTIC")
}

/// Load the warm semantic models for a long-lived loop (serve or mcp), applying
/// both fail-loud guards: a non-semantic build, and a semantic build whose model
/// files are missing at runtime. Returns `Err(exit_code)` when the operator
/// requires semantic and it is unavailable (fail closed). Shared so the native
/// MCP server gets byte-identical degradation behavior to `serve`.
pub(crate) fn serve_semantic_models_or_refuse() -> Result<SemanticModels, i32> {
    let require_semantic = require_semantic_env();
    match semantic_guard(cfg!(feature = "embed"), require_semantic) {
        SemanticGuard::Ok => {}
        SemanticGuard::Degraded => warn_non_semantic(false),
        SemanticGuard::Refuse => {
            warn_non_semantic(true);
            return Err(2);
        }
    }

    let semantic_models = SemanticModels::for_serve();
    // Second guard, one layer deeper: a semantic build whose model files are
    // missing at runtime would silently serve BM25. Refuse (or shout) instead.
    #[cfg(feature = "embed")]
    match runtime_semantic_guard(semantic_models.embed_active(), require_semantic) {
        RuntimeGuard::Ok => {}
        RuntimeGuard::Degraded => warn_models_absent(false),
        RuntimeGuard::Refuse => {
            warn_models_absent(true);
            return Err(2);
        }
    }

    Ok(semantic_models)
}

pub(crate) fn run_serve(args: &[String]) -> i32 {
    let require_semantic = require_semantic_env();
    match semantic_guard(cfg!(feature = "embed"), require_semantic) {
        SemanticGuard::Ok => {}
        SemanticGuard::Degraded => warn_non_semantic(false),
        SemanticGuard::Refuse => {
            warn_non_semantic(true);
            return 2;
        }
    }

    let parsed = match parse_serve_args(args) {
        Ok(parsed) => parsed,
        Err(error) => {
            println!(
                "{}",
                serve_failure_envelope(None, "serve", &error, Instant::now())
            );
            return error.exit_code();
        }
    };
    let ServeArgs { mode, store_path } = parsed;

    let semantic_models = SemanticModels::for_serve();
    // Second guard, one layer deeper: a semantic build whose model files are
    // missing at runtime would silently serve BM25. Refuse (or shout) instead.
    #[cfg(feature = "embed")]
    match runtime_semantic_guard(semantic_models.embed_active(), require_semantic) {
        RuntimeGuard::Ok => {}
        RuntimeGuard::Degraded => warn_models_absent(false),
        RuntimeGuard::Refuse => {
            warn_models_absent(true);
            return 2;
        }
    }

    match mode {
        ServeMode::Stdio => run_serve_stdio(&semantic_models),
        ServeMode::Socket(path) => run_serve_socket(&path, &semantic_models),
        ServeMode::Http(addr) => {
            crate::dashboard::run_serve_http(&addr, &semantic_models, store_path.as_deref())
        }
    }
}

fn run_serve_stdio(semantic_models: &SemanticModels) -> i32 {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let started = Instant::now();
        let response = match line {
            Ok(line) if line.trim().is_empty() => continue,
            Ok(line) => serve_line_response(&line, started, semantic_models),
            Err(error) => serve_failure_envelope(
                None,
                "unknown",
                &CliError::InvalidRequest(format!("failed to read stdio request: {error}")),
                started,
            ),
        };
        if writeln!(stdout, "{response}").is_err() || stdout.flush().is_err() {
            return 1;
        }
    }
    0
}

/// Serve newline-delimited JSON requests on a Unix domain socket, keeping the
/// models warm across requests. One response line per request line; each
/// connection is handled on its own thread (store access opens per-request
/// connections and the models are behind mutexes, so this is safe).
#[cfg(unix)]
fn run_serve_socket(path: &Path, semantic_models: &SemanticModels) -> i32 {
    use std::os::unix::fs::PermissionsExt;
    use std::os::unix::net::{UnixListener, UnixStream};
    use std::time::Duration;

    // Refuse to clobber a live server; remove only a stale socket file.
    if UnixStream::connect(path).is_ok() {
        eprintln!(
            "[memkeeper] another server is already listening on {}",
            path.display()
        );
        return 1;
    }
    let _ = fs::remove_file(path);
    let listener = match UnixListener::bind(path) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("[memkeeper] failed to bind {}: {error}", path.display());
            return 1;
        }
    };
    if let Err(error) = fs::set_permissions(path, fs::Permissions::from_mode(0o600)) {
        eprintln!(
            "[memkeeper] failed to restrict {} permissions: {error}",
            path.display()
        );
        return 1;
    }
    eprintln!("[memkeeper] serving on {}", path.display());

    std::thread::scope(|scope| {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            scope.spawn(|| {
                let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                let _ = stream.set_write_timeout(Some(Duration::from_secs(30)));
                serve_socket_connection(stream, semantic_models);
            });
        }
    });
    0
}

#[cfg(not(unix))]
fn run_serve_socket(_path: &Path, _semantic_models: &SemanticModels) -> i32 {
    eprintln!("[memkeeper] serve --socket is only supported on Unix platforms");
    1
}

#[cfg(unix)]
fn serve_socket_connection(stream: std::os::unix::net::UnixStream, models: &SemanticModels) {
    let Ok(read_half) = stream.try_clone() else {
        return;
    };
    let reader = io::BufReader::new(read_half);
    let mut writer = io::BufWriter::new(stream);
    for line in reader.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let started = Instant::now();
        let response = serve_line_response(&line, started, models);
        if writeln!(writer, "{response}").is_err() || writer.flush().is_err() {
            break;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ServeMode {
    /// Newline-delimited JSON over stdin/stdout (one client, e.g. a parent process).
    Stdio,
    /// Newline-delimited JSON over a Unix domain socket (many short-lived clients).
    Socket(PathBuf),
    /// Read-only HTTP dashboard (browser SPA + JSON read endpoints).
    Http(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServeArgs {
    pub(crate) mode: ServeMode,
    /// Default store for the HTTP dashboard (`--store`). Unused by stdio/socket
    /// modes, where each request carries its own `store_path`.
    pub(crate) store_path: Option<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ServeRequestEnvelope {
    pub(crate) request_id: Option<String>,
    pub(crate) command: Command,
    pub(crate) command_name: String,
    pub(crate) store_path: Option<PathBuf>,
    pub(crate) payload_json: String,
}

pub(crate) fn parse_serve_args(args: &[String]) -> Result<ServeArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut stdio = false;
    let mut socket: Option<PathBuf> = None;
    let mut http: Option<String> = None;
    let mut store: Option<PathBuf> = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--stdio" => stdio = true,
            "--store" => store = Some(parser.required_value("--store")?),
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            "--socket" => socket = Some(parser.required_value("--socket")?),
            "--http" => {
                // Optional address: `--http` alone binds the loopback default;
                // `--http <addr>` overrides it (but not another flag).
                http = Some(match parser.peek() {
                    Some(value) if !value.starts_with('-') => parser.required_string("--http")?,
                    _ => crate::dashboard::DEFAULT_HTTP_ADDR.to_string(),
                });
            }
            "--json" => {}
            value if value.starts_with("--socket=") => {
                socket = Some(PathBuf::from(value.trim_start_matches("--socket=")));
            }
            value if value.starts_with("--http=") => {
                http = Some(value.trim_start_matches("--http=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported serve flag: {unknown}"
                )));
            }
        }
    }

    let modes = usize::from(stdio) + usize::from(socket.is_some()) + usize::from(http.is_some());
    if modes > 1 {
        return Err(CliError::InvalidRequest(
            "serve accepts exactly one of --stdio, --socket <path>, or --http [addr]".to_string(),
        ));
    }
    if stdio {
        return Ok(ServeArgs {
            mode: ServeMode::Stdio,
            store_path: store,
        });
    }
    if let Some(path) = socket {
        return Ok(ServeArgs {
            mode: ServeMode::Socket(path),
            store_path: store,
        });
    }
    if let Some(addr) = http {
        return Ok(ServeArgs {
            mode: ServeMode::Http(addr),
            store_path: store,
        });
    }
    Err(CliError::InvalidRequest(
        "serve requires one of --stdio, --socket <path>, or --http [addr]".to_string(),
    ))
}

pub(crate) fn serve_line_response(
    line: &str,
    started: Instant,
    semantic_models: &SemanticModels,
) -> String {
    match parse_serve_request(line) {
        Ok(request) => execute_serve_request(&request, started, semantic_models),
        Err(error) => {
            let (request_id, command_name) = extract_serve_identity(line);
            serve_failure_envelope(
                request_id.as_deref(),
                command_name.as_deref().unwrap_or("unknown"),
                &error,
                started,
            )
        }
    }
}

pub(crate) fn extract_serve_identity(input: &str) -> (Option<String>, Option<String>) {
    let Ok(value) = parse_json(input) else {
        return (None, None);
    };
    let Some(object) = value.as_object() else {
        return (None, None);
    };
    let request_id = match object.get("request_id") {
        Some(JsonValue::String(value)) => Some(value.clone()),
        None | Some(_) => None,
    };
    let command_name = match object.get("command") {
        Some(JsonValue::String(value)) => Some(value.clone()),
        None | Some(_) => None,
    };
    (request_id, command_name)
}

pub(crate) fn parse_serve_request(input: &str) -> Result<ServeRequestEnvelope, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("serve request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "protocol_version",
            "request_id",
            "command",
            "store_path",
            "cwd",
            "payload",
        ],
    )?;
    if let Some(protocol_version) = optional_string_field(object, "protocol_version")? {
        if protocol_version != PROTOCOL_VERSION {
            return Err(CliError::InvalidRequest(format!(
                "unsupported protocol_version: {protocol_version}"
            )));
        }
    }
    let command_name = required_string_field(object, "command")?;
    let command = Command::parse(&command_name).ok_or_else(|| {
        CliError::InvalidRequest(format!("unsupported serve command: {command_name}"))
    })?;
    let store_path = optional_string_field(object, "store_path")?.map(PathBuf::from);
    let payload_json = match object.get("payload") {
        None | Some(JsonValue::Null) => "{}".to_string(),
        Some(JsonValue::Object(payload)) => payload.to_json(),
        Some(_) => {
            return Err(CliError::InvalidRequest(
                "field payload must be a JSON object".to_string(),
            ));
        }
    };
    Ok(ServeRequestEnvelope {
        request_id: optional_string_field(object, "request_id")?,
        command,
        command_name,
        store_path,
        payload_json,
    })
}

pub(crate) fn execute_serve_request(
    request: &ServeRequestEnvelope,
    started: Instant,
    semantic_models: &SemanticModels,
) -> String {
    let result = execute_serve_request_result(request, started, semantic_models);
    match result {
        Ok((store_path, schema_version, result_json)) => serve_success_envelope(
            request.request_id.as_deref(),
            &request.command_name,
            &store_path,
            schema_version,
            &result_json,
            started,
        ),
        Err(error) => serve_failure_envelope(
            request.request_id.as_deref(),
            &request.command_name,
            &error,
            started,
        ),
    }
}

#[allow(clippy::too_many_lines)]
pub(crate) fn execute_serve_request_result(
    request: &ServeRequestEnvelope,
    _started: Instant,
    semantic_models: &SemanticModels,
) -> Result<(PathBuf, Option<i32>, String), CliError> {
    match request.command {
        Command::Reindex => Err(CliError::InvalidRequest(
            "reindex is not available over the serve protocol; run `memkeeper reindex --store <path>`"
                .to_string(),
        )),
        Command::Init => {
            expect_empty_payload(&request.payload_json, "init")?;
            let store = required_serve_store_path(request)?;
            let report = init_store(&store)?;
            Ok((
                store,
                Some(report.schema_version),
                init_result_json(&report),
            ))
        }
        Command::Doctor => {
            let (store, store_source) = request
                .store_path
                .clone()
                .map_or_else(diagnostic_store_candidate, |store| {
                    (store, "request".to_string())
                });
            let options = DoctorArgs {
                store,
                store_source,
                include_indexes: include_indexes_payload(&request.payload_json, "doctor")?,
            };
            let (result_json, schema_version) = doctor_result_json(&options);
            Ok((options.store, schema_version, result_json))
        }
        Command::SpaceList => {
            expect_empty_payload(&request.payload_json, "space-list")?;
            let store = required_serve_store_path(request)?;
            let report = list_spaces(&store)?;
            Ok((store, Some(SCHEMA_VERSION), space_list_result_json(&report)))
        }
        Command::SpaceCreate => {
            let store = required_serve_store_path(request)?;
            let report = create_space(
                &store,
                &space_create_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                space_create_result_json(&report),
            ))
        }
        Command::SiloList => {
            let store = required_serve_store_path(request)?;
            let report = list_silos(&store, &silo_list_request_from_json(&request.payload_json)?)?;
            Ok((store, Some(SCHEMA_VERSION), silo_list_result_json(&report)))
        }
        Command::Remember => {
            let store = required_serve_store_path(request)?;
            let mut remember_request = remember_request_from_json(&request.payload_json)?;
            maybe_embed_remember_request(&mut remember_request, semantic_models);
            maybe_colbert_embed_remember_request(&mut remember_request, semantic_models);
            let report = remember_memory(&store, &remember_request)?;
            Ok((store, Some(SCHEMA_VERSION), remember_result_json(&report)))
        }
        Command::Search => {
            let store = required_serve_store_path(request)?;
            let search_request = search_request_from_json(&request.payload_json)?;
            let (rerank, rerank_candidates) =
                search_rerank_options_from_json(&request.payload_json)?;
            let result_json = execute_search(
                &store,
                search_request,
                rerank,
                rerank_candidates,
                semantic_models,
            )?;
            Ok((store, Some(SCHEMA_VERSION), result_json))
        }
        Command::EntityUpsert => {
            let store = required_serve_store_path(request)?;
            let report = upsert_entity(
                &store,
                &entity_upsert_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                entity_upsert_result_json(&report),
            ))
        }
        Command::RelationshipUpsert => {
            let store = required_serve_store_path(request)?;
            let report = upsert_relationship(
                &store,
                &relationship_upsert_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                relationship_upsert_result_json(&report),
            ))
        }
        Command::EntityMerge => {
            let store = required_serve_store_path(request)?;
            let report = merge_entity(
                &store,
                &entity_merge_request_from_json(&request.payload_json)?,
            )?;
            Ok((store, Some(SCHEMA_VERSION), entity_merge_result_json(&report)))
        }
        Command::EntitySearch => {
            let store = required_serve_store_path(request)?;
            let report = search_entities(
                &store,
                &entity_search_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                entity_search_result_json(&report),
            ))
        }
        Command::GraphNeighbors => {
            let store = required_serve_store_path(request)?;
            let report = graph_neighbors(
                &store,
                &graph_neighbors_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                graph_neighbors_result_json(&report),
            ))
        }
        Command::GraphContext => {
            let store = required_serve_store_path(request)?;
            let report = graph_context(
                &store,
                &graph_context_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                graph_context_result_json(&report),
            ))
        }
        Command::GraphFull => {
            let store = required_serve_store_path(request)?;
            let report = graph_full(&store)?;
            Ok((store, Some(SCHEMA_VERSION), graph_full_result_json(&report)))
        }
        Command::MemoryList => {
            let store = required_serve_store_path(request)?;
            let report = list_memories(
                &store,
                &memory_list_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                memory_list_result_json(&report),
            ))
        }
        Command::BatchSearch => {
            let store = required_serve_store_path(request)?;
            let report = batch_search_memories(
                &store,
                &batch_search_request_from_json(&request.payload_json)?,
            )?;
            let last_synth = last_synthesis_run(&store).unwrap_or(None);
            Ok((
                store,
                Some(SCHEMA_VERSION),
                batch_search_result_json(&report, last_synth.as_deref()),
            ))
        }
        Command::Pack => {
            let store = required_serve_store_path(request)?;
            let mut pack_request = pack_request_from_json(&request.payload_json)?;
            let mut expansion = pack_expansion_options_from_json(&request.payload_json)?;
            let cosine_gate = pack_cosine_gate_from_json(&request.payload_json)?;
            apply_pack_query_expansion(&mut pack_request, &mut expansion);
            maybe_embed_pack_request(&mut pack_request, semantic_models);
            maybe_colbert_embed_pack_request(&mut pack_request, semantic_models);
            let mut report = build_pack(&store, &pack_request)?;
            maybe_rerank_pack_report(
                &store,
                &pack_request,
                cosine_gate,
                expansion,
                &mut report,
                semantic_models,
            );
            Ok((store, Some(SCHEMA_VERSION), pack_result_json(&report)))
        }
        Command::Get => {
            let store = required_serve_store_path(request)?;
            let parsed = get_request_from_json(&request.payload_json)?;
            let memory = get_memory(&store, &parsed.id, parsed.options)?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                format!("{{\"memory\":{}}}", memory_json(&memory, parsed.options)),
            ))
        }
        Command::Forget => {
            let store = required_serve_store_path(request)?;
            let report = forget_memory(&store, &forget_request_from_json(&request.payload_json)?)?;
            Ok((store, Some(SCHEMA_VERSION), forget_result_json(&report)))
        }
        Command::Verify => {
            let store = required_serve_store_path(request)?;
            let report =
                verify_memory(&store, &verify_request_from_json(&request.payload_json)?)?;
            Ok((store, Some(SCHEMA_VERSION), verify_result_json(&report)))
        }
        Command::RecallLog => {
            let store = required_serve_store_path(request)?;
            let report = record_recall(
                &store,
                &recall_log_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                recall_log_result_json(&report),
            ))
        }
        Command::Ingest => {
            let store = required_serve_store_path(request)?;
            let mut ingest_request = ingest_request_from_json(&request.payload_json)?;
            maybe_embed_ingest_request(&mut ingest_request, semantic_models);
            let report = ingest_source(&store, &ingest_request)?;
            Ok((store, Some(SCHEMA_VERSION), ingest_result_json(&report)))
        }
        Command::DocumentSearch => {
            let store = required_serve_store_path(request)?;
            let mut document_request = document_search_request_from_json(&request.payload_json)?;
            maybe_embed_document_search_request(&mut document_request, semantic_models);
            let report = search_documents(&store, &document_request)?;
            Ok((store, Some(SCHEMA_VERSION), document_search_result_json(&report)))
        }
        Command::DocumentGet => {
            let store = required_serve_store_path(request)?;
            let report = get_document(&store, &document_get_request_from_json(&request.payload_json)?)?;
            Ok((store, Some(SCHEMA_VERSION), document_get_result_json(&report)))
        }
        Command::PromotionCandidates => {
            let store = required_serve_store_path(request)?;
            let report = promotion_candidates(
                &store,
                &promotion_candidates_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                promotion_candidates_result_json(&report),
            ))
        }
        Command::DocumentDuplicates => {
            let store = required_serve_store_path(request)?;
            let report = document_duplicates(
                &store,
                &document_duplicates_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                document_duplicates_result_json(&report),
            ))
        }
        Command::DocumentPrune => {
            let store = required_serve_store_path(request)?;
            let report = prune_documents(
                &store,
                &document_prune_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                document_prune_result_json(&report),
            ))
        }
        Command::MarkExtracted => {
            let store = required_serve_store_path(request)?;
            let report = mark_source_episodes_extracted(
                &store,
                &mark_extracted_request_from_json(&request.payload_json)?,
            )?;
            Ok((store, Some(SCHEMA_VERSION), mark_extracted_result_json(&report)))
        }
        Command::CandidateSubmit => {
            let store = required_serve_store_path(request)?;
            let report = submit_candidate(
                &store,
                &candidate_submit_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                candidate_submit_result_json(&report),
            ))
        }
        Command::CandidateList => {
            let store = required_serve_store_path(request)?;
            let report = list_candidates(
                &store,
                &candidate_list_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                candidate_list_result_json(&report),
            ))
        }
        Command::CandidateApprove => {
            let store = required_serve_store_path(request)?;
            let report = approve_candidate(
                &store,
                &candidate_approve_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                candidate_approve_result_json(&report),
            ))
        }
        Command::CandidateReject => {
            let store = required_serve_store_path(request)?;
            let report = reject_candidate(
                &store,
                &candidate_reject_request_from_json(&request.payload_json)?,
            )?;
            Ok((
                store,
                Some(SCHEMA_VERSION),
                candidate_reject_result_json(&report),
            ))
        }
        Command::History => {
            let store = required_serve_store_path(request)?;
            let parsed = history_request_from_json(&request.payload_json)?;
            let report = memory_history(&store, &parsed.id, parsed.options)?;
            Ok((store, Some(SCHEMA_VERSION), history_result_json(&report)))
        }
        Command::Stats => {
            let store = required_serve_store_path(request)?;
            let (include_indexes, include_health) = stats_payload_options(&request.payload_json)?;
            let stats = if include_health {
                store_stats_with_health(&store, include_indexes)?
            } else {
                store_stats(&store, include_indexes)?
            };
            Ok((store, Some(stats.schema_version), stats_result_json(&stats)))
        }
        Command::Export => {
            let store = required_serve_store_path(request)?;
            let report = export_store(&store, &export_request_from_json(&request.payload_json)?)?;
            Ok((
                store,
                Some(report.schema_version),
                export_result_json(&report),
            ))
        }
        Command::Import => {
            let store = required_serve_store_path(request)?;
            let report = import_store(&store, &import_request_from_json(&request.payload_json)?)?;
            Ok((
                store,
                Some(report.schema_version),
                import_result_json(&report),
            ))
        }
        Command::Dream => {
            let store = required_serve_store_path(request)?;
            let report = dream_store(&store, &dream_request_from_json(&request.payload_json)?)?;
            Ok((store, Some(SCHEMA_VERSION), dream_result_json(&report)))
        }
        Command::Backup => {
            let store = required_serve_store_path(request)?;
            let report = backup_store(&store, &backup_request_from_json(&request.payload_json)?)?;
            Ok((
                store,
                Some(report.schema_version),
                backup_result_json(&report),
            ))
        }
        Command::Rerank => {
            // No store: documents are supplied inline in the payload.
            let rerank_request = rerank_request_from_json(&request.payload_json)?;
            let result_json = run_rerank_payload(&rerank_request, semantic_models)?;
            Ok((PathBuf::new(), Some(SCHEMA_VERSION), result_json))
        }
    }
}

pub(crate) fn required_serve_store_path(
    request: &ServeRequestEnvelope,
) -> Result<PathBuf, CliError> {
    request.store_path.clone().ok_or_else(|| {
        CliError::InvalidRequest("missing required store_path for serve request".to_string())
    })
}

pub(crate) fn expect_empty_payload(input: &str, command: &str) -> Result<(), CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest(format!("{command} payload must be a JSON object"))
    })?;
    reject_unknown_fields(object, &[])
}

pub(crate) fn include_indexes_payload(input: &str, command: &str) -> Result<bool, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest(format!("{command} payload must be a JSON object"))
    })?;
    reject_unknown_fields(object, &["include_indexes"])?;
    optional_bool_field(object, "include_indexes").map(|value| value.unwrap_or(false))
}

/// Parse the `stats` serve payload into `(include_indexes, include_health)`.
/// `include_health` opts into the memory-governance rollup (default false).
pub(crate) fn stats_payload_options(input: &str) -> Result<(bool, bool), CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("stats payload must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["include_indexes", "include_health"])?;
    let include_indexes = optional_bool_field(object, "include_indexes")?.unwrap_or(false);
    let include_health = optional_bool_field(object, "include_health")?.unwrap_or(false);
    Ok((include_indexes, include_health))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "embed")]
    #[test]
    fn runtime_guard_serves_when_embedder_loaded() {
        assert_eq!(runtime_semantic_guard(true, false), RuntimeGuard::Ok);
        assert_eq!(runtime_semantic_guard(true, true), RuntimeGuard::Ok);
    }

    #[cfg(feature = "embed")]
    #[test]
    fn runtime_guard_refuses_when_required_and_models_absent() {
        assert_eq!(runtime_semantic_guard(false, true), RuntimeGuard::Refuse);
    }

    #[cfg(feature = "embed")]
    #[test]
    fn runtime_guard_degrades_loud_when_models_absent_and_not_required() {
        assert_eq!(runtime_semantic_guard(false, false), RuntimeGuard::Degraded);
    }

    #[test]
    fn stats_payload_options_parses_indexes_and_health() {
        assert_eq!(stats_payload_options("{}").expect("empty"), (false, false));
        assert_eq!(
            stats_payload_options(r#"{"include_indexes":true,"include_health":true}"#)
                .expect("both"),
            (true, true)
        );
        assert_eq!(
            stats_payload_options(r#"{"include_health":true}"#).expect("health only"),
            (false, true)
        );
        // Unknown fields are rejected, matching the other serve payload parsers.
        assert!(stats_payload_options(r#"{"bogus":1}"#).is_err());
    }

    #[test]
    fn parse_serve_request_accepts_minimal_envelope() {
        let request = parse_serve_request(
            r#"{"command":"stats","request_id":"req-1","store_path":"/tmp/s.sqlite"}"#,
        )
        .expect("minimal envelope parses");
        assert_eq!(request.command, Command::Stats);
        assert_eq!(request.command_name, "stats");
        assert_eq!(request.request_id.as_deref(), Some("req-1"));
        assert_eq!(request.payload_json, "{}");
    }

    #[test]
    fn parse_serve_request_rejects_protocol_version_mismatch() {
        let error = parse_serve_request(r#"{"command":"stats","protocol_version":"memkeeper.v9"}"#)
            .expect_err("mismatched protocol must fail");
        assert!(error.to_string().contains("unsupported protocol_version"));
    }

    #[test]
    fn parse_serve_request_rejects_unknown_command_and_fields() {
        let unknown_command =
            parse_serve_request(r#"{"command":"explode"}"#).expect_err("unknown command must fail");
        assert!(unknown_command
            .to_string()
            .contains("unsupported serve command"));

        let unknown_field = parse_serve_request(r#"{"command":"stats","shard":3}"#)
            .expect_err("unknown envelope field must fail");
        assert!(unknown_field.to_string().contains("shard"));
    }

    #[test]
    fn parse_serve_request_rejects_non_object_payload() {
        let error = parse_serve_request(r#"{"command":"stats","payload":[1,2]}"#)
            .expect_err("array payload must fail");
        assert!(error.to_string().contains("payload must be a JSON object"));
    }

    #[test]
    fn required_serve_store_path_errors_without_store() {
        let request = parse_serve_request(r#"{"command":"stats"}"#).expect("parses");
        let error = required_serve_store_path(&request).expect_err("missing store must fail");
        assert!(error.to_string().contains("store_path"));
    }

    #[test]
    fn parse_serve_args_supports_stdio_and_socket_modes() {
        let stdio = parse_serve_args(&["--stdio".to_string()]).expect("stdio parses");
        assert_eq!(stdio.mode, ServeMode::Stdio);

        let socket = parse_serve_args(&["--socket".to_string(), "/tmp/x.sock".to_string()])
            .expect("socket parses");
        assert_eq!(socket.mode, ServeMode::Socket(PathBuf::from("/tmp/x.sock")));

        let eq_form = parse_serve_args(&["--socket=/tmp/y.sock".to_string()]).expect("eq form");
        assert_eq!(
            eq_form.mode,
            ServeMode::Socket(PathBuf::from("/tmp/y.sock"))
        );

        assert!(parse_serve_args(&[]).is_err(), "a mode is required");
        assert!(
            parse_serve_args(&["--stdio".to_string(), "--socket=/tmp/z.sock".to_string()]).is_err(),
            "modes are mutually exclusive"
        );
        assert!(
            parse_serve_args(&["--socket".to_string()]).is_err(),
            "--socket requires a path"
        );
    }

    #[test]
    fn parse_serve_args_supports_http_mode() {
        let default = parse_serve_args(&["--http".to_string()]).expect("bare --http");
        assert_eq!(
            default.mode,
            ServeMode::Http(crate::dashboard::DEFAULT_HTTP_ADDR.to_string())
        );

        let explicit = parse_serve_args(&["--http".to_string(), "127.0.0.1:9000".to_string()])
            .expect("--http addr");
        assert_eq!(explicit.mode, ServeMode::Http("127.0.0.1:9000".to_string()));

        let eq_form = parse_serve_args(&["--http=0.0.0.0:8080".to_string()]).expect("eq form");
        assert_eq!(eq_form.mode, ServeMode::Http("0.0.0.0:8080".to_string()));

        assert!(
            parse_serve_args(&["--http".to_string(), "--stdio".to_string()]).is_err(),
            "--http and --stdio are mutually exclusive"
        );
        // `--http` followed by another flag uses the default and the flag errors
        // as a duplicate mode — either way it must not silently swallow --stdio.
    }

    #[test]
    fn parse_serve_args_accepts_store_for_dashboard() {
        // `--store` sets the dashboard's default store; `--http` before it binds
        // the default address (the next token starts with `-`).
        let spaced = parse_serve_args(&[
            "--http".to_string(),
            "--store".to_string(),
            "/tmp/dash.sqlite".to_string(),
        ])
        .expect("http + store parses");
        assert_eq!(
            spaced.mode,
            ServeMode::Http(crate::dashboard::DEFAULT_HTTP_ADDR.to_string())
        );
        assert_eq!(spaced.store_path, Some(PathBuf::from("/tmp/dash.sqlite")));

        let eq_form =
            parse_serve_args(&["--http".to_string(), "--store=/tmp/x.sqlite".to_string()])
                .expect("store eq form");
        assert_eq!(eq_form.store_path, Some(PathBuf::from("/tmp/x.sqlite")));

        // No `--store` leaves it unset (server falls back to default resolution).
        let none = parse_serve_args(&["--stdio".to_string()]).expect("stdio");
        assert_eq!(none.store_path, None);
    }

    #[test]
    fn serve_line_failure_envelope_echoes_request_identity() {
        let response = serve_line_response(
            r#"{"command":"stats","request_id":"req-9","unknown_field":1}"#,
            std::time::Instant::now(),
            &SemanticModels::for_serve(),
        );
        assert!(response.contains("\"ok\":false"));
        assert!(response.contains("\"request_id\":\"req-9\""));
        assert!(response.contains("\"command\":\"stats\""));
    }
}
