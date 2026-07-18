//! Response envelopes and result-JSON builders (deterministic, hand-built
//! for byte-stable output).

#[allow(clippy::wildcard_imports)]
use super::*;
#[allow(clippy::wildcard_imports)]
use crate::json::*;

pub(crate) fn doctor_success_envelope(
    store_path: &Path,
    schema_version: Option<i32>,
    result_json: &str,
    started: Instant,
) -> String {
    let elapsed_ms = started.elapsed().as_millis();
    let schema_json = schema_version.map_or_else(|| "null".to_string(), |value| value.to_string());
    format!(
        "{{\"protocol_version\":{},\"ok\":true,\"command\":\"doctor\",\"store\":{{\"path\":{},\"schema_version\":{schema_json}}},\"result\":{result_json},\"warnings\":[],\"elapsed_ms\":{elapsed_ms}}}",
        json_string(PROTOCOL_VERSION),
        json_path(store_path)
    )
}

pub(crate) fn success_envelope(
    command: Command,
    store_path: &Path,
    schema_version: i32,
    result_json: &str,
    started: Instant,
) -> String {
    let elapsed_ms = started.elapsed().as_millis();
    format!(
        "{{\"protocol_version\":{},\"ok\":true,\"command\":{},\"store\":{{\"path\":{},\"schema_version\":{schema_version}}},\"result\":{result_json},\"warnings\":[],\"elapsed_ms\":{elapsed_ms}}}",
        json_string(PROTOCOL_VERSION),
        json_string(command.as_str()),
        json_path(store_path)
    )
}

pub(crate) fn failure_envelope(command: Command, error: &CliError, started: Instant) -> String {
    serve_failure_envelope(None, command.as_str(), error, started)
}

pub(crate) fn serve_success_envelope(
    request_id: Option<&str>,
    command_name: &str,
    store_path: &Path,
    schema_version: Option<i32>,
    result_json: &str,
    started: Instant,
) -> String {
    let elapsed_ms = started.elapsed().as_millis();
    let schema_json = schema_version.map_or_else(|| "null".to_string(), |value| value.to_string());
    format!(
        "{{\"protocol_version\":{},\"request_id\":{},\"ok\":true,\"command\":{},\"store\":{{\"path\":{},\"schema_version\":{schema_json}}},\"result\":{result_json},\"warnings\":[],\"elapsed_ms\":{elapsed_ms}}}",
        json_string(PROTOCOL_VERSION),
        optional_string_json(request_id),
        json_string(command_name),
        json_path(store_path)
    )
}

pub(crate) fn serve_failure_envelope(
    request_id: Option<&str>,
    command_name: &str,
    error: &CliError,
    started: Instant,
) -> String {
    let elapsed_ms = started.elapsed().as_millis();
    let code = error.code().as_str();
    let message = error.to_string();
    let retryable = error.retryable();
    let hint = error.hint();
    let details = error.details_json();
    format!(
        "{{\"protocol_version\":{},\"request_id\":{},\"ok\":false,\"command\":{},\"error\":{{\"code\":{},\"message\":{},\"details\":{details},\"retryable\":{retryable},\"hint\":{}}},\"warnings\":[],\"elapsed_ms\":{elapsed_ms}}}",
        json_string(PROTOCOL_VERSION),
        optional_string_json(request_id),
        json_string(command_name),
        json_string(code),
        json_string(&message),
        json_string(hint)
    )
}

pub(crate) fn init_result_json(report: &InitReport) -> String {
    format!(
        "{{\"initialized\":{},\"created\":{},\"schema_version\":{},\"protocol_version\":{},\"sqlite_version\":{},\"journal_mode\":{},\"spaces\":{},\"default_space\":{}}}",
        report.initialized,
        report.created,
        report.schema_version,
        json_string(&report.protocol_version),
        json_string(&report.sqlite_version),
        json_string(&report.journal_mode),
        string_array_json(&report.spaces),
        json_string(&report.default_space)
    )
}

pub(crate) fn space_list_result_json(report: &SpaceListReport) -> String {
    format!(
        "{{\"spaces\":{},\"truncated\":{}}}",
        spaces_json(&report.spaces, None),
        report.truncated
    )
}

pub(crate) fn space_create_result_json(report: &SpaceCreateReport) -> String {
    format!(
        "{{\"space\":{}}}",
        space_record_json(&report.space, Some(report.created))
    )
}

pub(crate) fn silo_list_result_json(report: &SiloListReport) -> String {
    format!(
        "{{\"space\":{},\"silos\":{},\"truncated\":{}}}",
        json_string(&report.space),
        silos_json(&report.silos),
        report.truncated
    )
}

pub(crate) fn spaces_json(spaces: &[SpaceRecord], created: Option<bool>) -> String {
    let mut output = String::from("[");
    for (index, space) in spaces.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&space_record_json(space, created));
    }
    output.push(']');
    output
}

pub(crate) fn space_record_json(space: &SpaceRecord, created: Option<bool>) -> String {
    let (description, description_truncated) =
        optional_bounded_string_json(space.description.as_deref(), MAX_SPACE_TEXT_OUTPUT_CHARS);
    let (ontology, ontology_truncated) =
        optional_bounded_string_json(space.ontology.as_deref(), MAX_SPACE_TEXT_OUTPUT_CHARS);
    let (config, config_truncated) = bounded_raw_json_object_or_null(
        space.config_json.as_deref(),
        MAX_SPACE_CONFIG_OUTPUT_CHARS,
    );
    let mut output = format!(
        "{{\"name\":{},\"display_name\":{},\"description\":{},\"description_truncated\":{},\"default_silo\":{},\"ontology\":{},\"ontology_truncated\":{},\"config\":{},\"config_truncated\":{},\"created_at\":{},\"updated_at\":{},\"memory_count\":{},\"active_count\":{},\"silo_count\":{}",
        json_string(&space.name),
        optional_bounded_string_json(space.display_name.as_deref(), MAX_SPACE_TEXT_OUTPUT_CHARS).0,
        description,
        description_truncated,
        json_string(&space.default_silo),
        ontology,
        ontology_truncated,
        config,
        config_truncated,
        json_string(&bounded_char_prefix(&space.created_at, MAX_SPACE_TEXT_OUTPUT_CHARS)),
        json_string(&bounded_char_prefix(&space.updated_at, MAX_SPACE_TEXT_OUTPUT_CHARS)),
        space.memory_count,
        space.active_count,
        space.silo_count
    );
    if let Some(created) = created {
        let _ = write!(output, ",\"created\":{created}");
    }
    output.push('}');
    output
}

pub(crate) fn silos_json(silos: &[SiloRecord]) -> String {
    let mut output = String::from("[");
    for (index, silo) in silos.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&silo_record_json(silo));
    }
    output.push(']');
    output
}

pub(crate) fn silo_record_json(silo: &SiloRecord) -> String {
    let (description, description_truncated) =
        optional_bounded_string_json(silo.description.as_deref(), MAX_SPACE_TEXT_OUTPUT_CHARS);
    let (config, config_truncated) =
        bounded_raw_json_object_or_null(silo.config_json.as_deref(), MAX_SPACE_CONFIG_OUTPUT_CHARS);
    format!(
        "{{\"space\":{},\"name\":{},\"description\":{},\"description_truncated\":{},\"retention\":{},\"default_scope\":{},\"config\":{},\"config_truncated\":{},\"created_at\":{},\"updated_at\":{},\"memory_count\":{},\"active_count\":{},\"is_default\":{}}}",
        json_string(&silo.space),
        json_string(&silo.name),
        description,
        description_truncated,
        json_string(&silo.retention_policy),
        json_string(&silo.default_scope),
        config,
        config_truncated,
        json_string(&bounded_char_prefix(&silo.created_at, MAX_SPACE_TEXT_OUTPUT_CHARS)),
        json_string(&bounded_char_prefix(&silo.updated_at, MAX_SPACE_TEXT_OUTPUT_CHARS)),
        silo.memory_count,
        silo.active_count,
        silo.is_default
    )
}

pub(crate) fn optional_bounded_string_json(
    value: Option<&str>,
    max_chars: usize,
) -> (String, bool) {
    match value {
        None => ("null".to_string(), false),
        Some(value) => {
            let truncated = value.chars().count() > max_chars;
            (
                json_string(&bounded_char_prefix(value, max_chars)),
                truncated,
            )
        }
    }
}

pub(crate) fn bounded_raw_json_object_or_null(
    value: Option<&str>,
    max_chars: usize,
) -> (String, bool) {
    let Some(value) = value else {
        return ("null".to_string(), false);
    };
    let truncated = value.chars().count() > max_chars;
    if truncated {
        return ("null".to_string(), true);
    }
    match parse_json(value).ok() {
        Some(JsonValue::Object(_)) => (value.to_string(), false),
        Some(_) | None => ("null".to_string(), true),
    }
}

pub(crate) fn bounded_char_prefix(value: &str, max_chars: usize) -> String {
    value.chars().take(max_chars).collect()
}

pub(crate) fn diagnostic_store_candidate() -> (PathBuf, String) {
    diagnostic_store_candidate_from(
        non_empty_env("MEMKEEPER_STORE").as_deref(),
        non_empty_env("PI_MEMKEEPER_STORE").as_deref(),
    )
}

pub(crate) fn diagnostic_store_candidate_from(
    memkeeper_store: Option<&str>,
    pi_memkeeper_store: Option<&str>,
) -> (PathBuf, String) {
    if let Some(value) = memkeeper_store {
        return (
            PathBuf::from(strip_at_prefix(value)),
            "MEMKEEPER_STORE".to_string(),
        );
    }
    if let Some(value) = pi_memkeeper_store {
        return (
            PathBuf::from(strip_at_prefix(value)),
            "PI_MEMKEEPER_STORE".to_string(),
        );
    }
    (
        PathBuf::from(PROJECT_STORE_RELATIVE_PATH),
        "project_hint".to_string(),
    )
}

pub(crate) fn non_empty_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

pub(crate) fn strip_at_prefix(value: &str) -> &str {
    value.strip_prefix('@').unwrap_or(value)
}

pub(crate) fn doctor_result_json(options: &DoctorArgs) -> (String, Option<i32>) {
    let schema_ok = schema_mentions_required_objects();
    let current_exe = env::current_exe().ok();
    let metadata = fs::symlink_metadata(&options.store);
    let stats_result = inspect_store_stats(&options.store, options.include_indexes);
    let schema_version = stats_result.as_ref().ok().map(|stats| stats.schema_version);
    let store_state = doctor_store_state(&metadata, &stats_result);
    let doctor_status = doctor_overall_status(schema_ok, store_state);
    let binary = doctor_binary_json(schema_ok, current_exe.as_deref());
    let config = doctor_config_json(options);
    let store = doctor_store_json(options, &metadata, &stats_result, store_state);
    let checks = doctor_checks_json(schema_ok, &metadata, &stats_result, store_state);

    (
        format!(
            "{{\"doctor\":{{\"status\":{},\"mutating\":false}},\"binary\":{binary},\"config\":{config},\"store\":{store},\"checks\":{checks}}}",
            json_string(doctor_status)
        ),
        schema_version,
    )
}

pub(crate) fn doctor_overall_status(schema_ok: bool, store_state: &str) -> &'static str {
    if !schema_ok {
        return "error";
    }
    match store_state {
        "initialized" => "ok",
        "missing" | "not_initialized" => "warning",
        _ => "error",
    }
}

pub(crate) fn doctor_store_state(
    metadata: &std::io::Result<fs::Metadata>,
    stats: &std::result::Result<Stats, StoreError>,
) -> &'static str {
    match stats {
        Ok(_) => "initialized",
        Err(StoreError::NotInitialized { .. }) => {
            if metadata.is_ok() {
                "not_initialized"
            } else {
                "missing"
            }
        }
        Err(StoreError::SchemaMismatch { .. }) => "schema_mismatch",
        Err(StoreError::WalUnavailable { .. }) => "wal_unavailable",
        Err(StoreError::UnsafeExistingDatabase { .. }) => "unrecognized_database",
        Err(StoreError::InvalidPath { .. }) => "invalid_path",
        Err(error) if error.is_locked() => "locked",
        Err(StoreError::Io(_)) => "io_error",
        Err(StoreError::Database(_)) => "database_error",
        Err(StoreError::InvalidRequest { .. }) => "invalid_request",
        Err(StoreError::NotFound { .. }) => "not_found",
        Err(StoreError::Conflict { .. }) => "conflict",
    }
}

pub(crate) fn doctor_binary_json(schema_ok: bool, current_exe: Option<&Path>) -> String {
    format!(
        "{{\"protocol_version\":{},\"schema_version\":{},\"current_exe\":{},\"embedded_schema_required_objects\":{schema_ok}}}",
        json_string(PROTOCOL_VERSION),
        SCHEMA_VERSION,
        optional_path_json(current_exe)
    )
}

pub(crate) fn doctor_config_json(options: &DoctorArgs) -> String {
    let prefix_capture_enabled = prefix_capture_enabled_from_env();
    format!(
        "{{\"store_resolution\":{{\"source\":{},\"path\":{}}},\"user_store_path_hint\":{},\"project_store_relative_path\":{},\"env\":{},\"pi_adapter\":{{\"package_path\":\"adapters/pi-extension\",\"prefix_capture_enabled\":{prefix_capture_enabled}}}}}",
        json_string(&options.store_source),
        json_path(&options.store),
        json_string(USER_STORE_PATH_HINT),
        json_string(PROJECT_STORE_RELATIVE_PATH),
        doctor_env_json()
    )
}

pub(crate) fn doctor_env_json() -> String {
    let names = [
        "MEMKEEPER_STORE",
        "PI_MEMKEEPER_STORE",
        "MEMKEEPER_BIN",
        "PI_MEMKEEPER_BIN",
        "MEMKEEPER_ROOT",
        "PI_MEMKEEPER_ROOT",
        "MEMKEEPER_TIMEOUT_MS",
        "MEMKEEPER_PREFIX_CAPTURE",
        "PI_MEMKEEPER_PREFIX_CAPTURE",
    ];
    let mut output = String::from("{");
    for (index, name) in names.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(output, "{}:{}", json_string(name), env_value_json(name));
    }
    output.push('}');
    output
}

pub(crate) fn env_value_json(name: &str) -> String {
    match non_empty_env(name) {
        Some(value) => {
            let (bounded, truncated) = bounded_env_value(&value);
            format!(
                "{{\"set\":true,\"value\":{},\"truncated\":{truncated}}}",
                json_string(&bounded)
            )
        }
        None => "{\"set\":false,\"value\":null,\"truncated\":false}".to_string(),
    }
}

pub(crate) fn bounded_env_value(value: &str) -> (String, bool) {
    let max_chars = 1_024;
    let truncated = value.chars().count() > max_chars;
    (value.chars().take(max_chars).collect(), truncated)
}

pub(crate) fn prefix_capture_enabled_from_env() -> bool {
    ["MEMKEEPER_PREFIX_CAPTURE", "PI_MEMKEEPER_PREFIX_CAPTURE"]
        .iter()
        .filter_map(|name| non_empty_env(name))
        .map(|value| value.to_ascii_lowercase())
        .all(|value| !matches!(value.as_str(), "0" | "false" | "off" | "no"))
}

pub(crate) fn doctor_store_json(
    options: &DoctorArgs,
    metadata: &std::io::Result<fs::Metadata>,
    stats_result: &std::result::Result<Stats, StoreError>,
    store_state: &str,
) -> String {
    let (exists, is_file, is_symlink, bytes) = metadata_summary(metadata);
    let stats_json = stats_result
        .as_ref()
        .map_or_else(|_| "null".to_string(), doctor_stats_json);
    let error_json = stats_result
        .as_ref()
        .map_or_else(doctor_store_error_json, |_| "null".to_string());
    format!(
        "{{\"path\":{},\"path_source\":{},\"state\":{},\"exists\":{exists},\"is_file\":{is_file},\"is_symlink\":{is_symlink},\"database_bytes\":{},\"stats\":{stats_json},\"error\":{error_json}}}",
        json_path(&options.store),
        json_string(&options.store_source),
        json_string(store_state),
        optional_u64_json(bytes)
    )
}

pub(crate) fn metadata_summary(
    metadata: &std::io::Result<fs::Metadata>,
) -> (bool, bool, bool, Option<u64>) {
    match metadata {
        Ok(metadata) => (
            true,
            metadata.file_type().is_file(),
            metadata.file_type().is_symlink(),
            Some(metadata.len()),
        ),
        Err(_) => (false, false, false, None),
    }
}

pub(crate) fn doctor_stats_json(stats: &Stats) -> String {
    let indexes = stats
        .indexes
        .as_ref()
        .map_or_else(|| "null".to_string(), index_stats_json);
    format!(
        "{{\"schema_version\":{},\"protocol_version\":{},\"sqlite_version\":{},\"journal_mode\":{},\"database_bytes\":{},\"memory_count\":{},\"active_count\":{},\"source_episode_count\":{},\"space_count\":{},\"silo_count\":{},\"indexes\":{indexes}}}",
        stats.schema_version,
        json_string(&stats.protocol_version),
        json_string(&stats.sqlite_version),
        json_string(&stats.journal_mode),
        stats.database_bytes,
        stats.memory_count,
        stats.active_count,
        stats.source_episode_count,
        stats.space_count,
        stats.silo_count
    )
}

pub(crate) fn doctor_store_error_json(error: &StoreError) -> String {
    format!(
        "{{\"code\":{},\"message\":{},\"retryable\":{}}}",
        json_string(store_error_code(error).as_str()),
        json_string(&bounded_char_prefix(&error.to_string(), 1_000)),
        error.is_retryable()
    )
}

pub(crate) fn store_error_code(error: &StoreError) -> ErrorCode {
    match error {
        StoreError::InvalidRequest { .. } | StoreError::InvalidPath { .. } => {
            ErrorCode::InvalidRequest
        }
        StoreError::NotInitialized { .. } => ErrorCode::StoreNotInitialized,
        StoreError::NotFound { .. } => ErrorCode::NotFound,
        StoreError::UnsafeExistingDatabase { .. }
        | StoreError::WalUnavailable { .. }
        | StoreError::Conflict { .. } => ErrorCode::Conflict,
        StoreError::SchemaMismatch { .. } => ErrorCode::SchemaMismatch,
        StoreError::Io(_) => ErrorCode::IoError,
        error if error.is_locked() => ErrorCode::Locked,
        StoreError::Database(_) => ErrorCode::InternalError,
    }
}

pub(crate) fn doctor_checks_json(
    schema_ok: bool,
    metadata: &std::io::Result<fs::Metadata>,
    stats: &std::result::Result<Stats, StoreError>,
    store_state: &str,
) -> String {
    let mut checks = vec![doctor_check_json(
        "binary.embedded_schema",
        if schema_ok { "ok" } else { "error" },
        "embedded schema includes required canonical objects",
    )];
    let (exists, is_file, is_symlink, _) = metadata_summary(metadata);
    let path_status = if exists && is_file && !is_symlink {
        "ok"
    } else if exists {
        "error"
    } else {
        "warning"
    };
    checks.push(doctor_check_json(
        "store.path",
        path_status,
        if exists {
            "store path exists"
        } else {
            "store path does not exist yet"
        },
    ));
    checks.push(doctor_check_json(
        "store.initialized",
        doctor_store_check_status(store_state),
        doctor_store_check_message(store_state),
    ));
    if let Ok(stats) = stats {
        checks.push(doctor_check_json(
            "store.wal",
            if stats.journal_mode == "wal" {
                "ok"
            } else {
                "error"
            },
            "initialized store reports journal mode",
        ));
    }
    // Semantic readiness: the onboarding-critical signal. A semantic-capable
    // binary with no models pulled silently serves lexical, so surface it here
    // and point at `pull-models` rather than leaving the user to infer it.
    #[cfg(feature = "semantic")]
    {
        let model_status = memkeeper_embed::local_model_status();
        let (level, message) = if model_status.embed_present {
            (
                "ok",
                format!(
                    "semantic ready (embed model at {})",
                    model_status.embed_dir.display()
                ),
            )
        } else {
            (
                "warning",
                format!(
                    "semantic available but models not downloaded — run `memkeeper pull-models` (looked in {})",
                    model_status.embed_dir.display()
                ),
            )
        };
        checks.push(doctor_check_json("semantic.models", level, &message));
    }
    #[cfg(not(feature = "semantic"))]
    {
        checks.push(doctor_check_json(
            "semantic.models",
            "ok",
            "lexical-only build (BM25/FTS); semantic not compiled in",
        ));
    }
    format!("[{}]", checks.join(","))
}

pub(crate) fn doctor_store_check_status(state: &str) -> &'static str {
    match state {
        "initialized" => "ok",
        "missing" | "not_initialized" => "warning",
        _ => "error",
    }
}

pub(crate) fn doctor_store_check_message(state: &str) -> &'static str {
    match state {
        "initialized" => "store is initialized and compatible",
        "missing" => "run memkeeper init --store <path> --json before using memory tools",
        "not_initialized" => "path exists but is not an initialized memkeeper store",
        "schema_mismatch" => "store schema is not compatible with this binary",
        "locked" => "store is currently locked by another writer",
        _ => "store readiness check failed",
    }
}

pub(crate) fn doctor_check_json(name: &str, status: &str, message: &str) -> String {
    format!(
        "{{\"name\":{},\"status\":{},\"message\":{}}}",
        json_string(name),
        json_string(status),
        json_string(message)
    )
}

pub(crate) fn optional_path_json(path: Option<&Path>) -> String {
    path.map_or_else(|| "null".to_string(), json_path)
}

pub(crate) fn optional_u64_json(value: Option<u64>) -> String {
    value.map_or_else(|| "null".to_string(), |value| value.to_string())
}

pub(crate) fn stats_result_json(stats: &Stats) -> String {
    let indexes = stats
        .indexes
        .as_ref()
        .map_or_else(|| "null".to_string(), index_stats_json);
    // Opt-in `health` rollup is appended only when present, so default `stats`
    // output stays byte-identical for existing consumers.
    let health = stats.health.as_ref().map_or_else(String::new, |health| {
        format!(",\"health\":{}", health_stats_json(health))
    });
    format!(
        "{{\"schema_version\":{},\"protocol_version\":{},\"sqlite_version\":{},\"journal_mode\":{},\"database_bytes\":{},\"memory_count\":{},\"active_count\":{},\"source_episode_count\":{},\"space_count\":{},\"silo_count\":{},\"spaces\":{},\"indexes\":{indexes}{health}}}",
        stats.schema_version,
        json_string(&stats.protocol_version),
        json_string(&stats.sqlite_version),
        json_string(&stats.journal_mode),
        stats.database_bytes,
        stats.memory_count,
        stats.active_count,
        stats.source_episode_count,
        stats.space_count,
        stats.silo_count,
        space_stats_json(stats),
    )
}

pub(crate) fn batch_search_result_json(
    report: &BatchSearchReport,
    last_synth: Option<&str>,
) -> String {
    let mut output = String::from("{\"results\":[");
    for (index, item) in report.results.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&batch_search_item_json(item, last_synth));
    }
    output.push_str("]}");
    output
}

pub(crate) fn batch_search_item_json(
    item: &BatchSearchItemReport,
    last_synth: Option<&str>,
) -> String {
    format!(
        "{{\"name\":{},\"query\":{},\"search\":{{\"strategy\":{},\"semantic\":{{\"attempted\":{},\"reason\":{}}},\"total_estimate\":{},\"truncated\":{}}},\"results\":{}}}",
        optional_string_json(item.name.as_deref()),
        json_string(&item.query),
        json_string(&item.report.strategy),
        item.report.semantic_attempted,
        json_string(&item.report.semantic_reason),
        item.report.total_estimate,
        item.report.truncated,
        search_results_json(&item.report.results, last_synth)
    )
}

pub(crate) fn pack_result_json(report: &PackReport) -> String {
    format!("{{\"pack\":{}}}", pack_payload_json(report))
}

#[cfg(any(feature = "embed", test))]
pub(crate) fn pool_trace_result_json(pool: &memkeeper_store::RerankPool) -> String {
    let mut candidates = String::from("[");
    for (candidate_index, candidate) in pool.observed.iter().enumerate() {
        if candidate_index > 0 {
            candidates.push(',');
        }
        let mut sources = String::from("[");
        for (source_index, observation) in candidate.admissions.iter().enumerate() {
            if source_index > 0 {
                sources.push(',');
            }
            let source = match observation.source {
                memkeeper_store::AdmissionSource::Ann => "ann",
                memkeeper_store::AdmissionSource::Maxsim => "maxsim",
                memkeeper_store::AdmissionSource::Bm25 => "bm25",
                memkeeper_store::AdmissionSource::Graph => "graph",
            };
            let activation = observation.activation.map_or_else(
                || "null".to_string(),
                |value| {
                    if value.is_finite() {
                        value.to_string()
                    } else {
                        "null".to_string()
                    }
                },
            );
            let route = observation.graph_route.as_ref().map_or_else(
                || "null".to_string(),
                |route| {
                    let seed_source = match route.seed_source {
                        memkeeper_store::GraphSeedSource::Memory => "memory",
                        memkeeper_store::GraphSeedSource::Entity => "entity",
                    };
                    let evidence_class = match route.evidence_class {
                        memkeeper_store::GraphEvidenceClass::EndpointSupport => "endpoint_support",
                        memkeeper_store::GraphEvidenceClass::EntityFallback => "entity_fallback",
                    };
                    format!(
                        "{{\"seed_source\":{},\"seed_memory_id\":{},\"seed_entity_id\":{},\"matched_query_index\":{},\"matched_query_span\":{},\"hop_depth\":{},\"relationship_ids\":{},\"predicate_names\":{},\"traversal_directions\":{},\"evidence_class\":{},\"route_outcome\":{}}}",
                        json_string(seed_source),
                        optional_string_json(route.seed_memory_id.as_deref()),
                        json_string(&route.seed_entity_id),
                        route.matched_query_index.map_or_else(
                            || "null".to_string(),
                            |index| index.to_string(),
                        ),
                        optional_string_json(route.matched_query_span.as_deref()),
                        route.hop_depth,
                        string_array_json(&route.relationship_ids),
                        string_array_json(&route.predicate_names),
                        string_array_json(&route.traversal_directions),
                        json_string(evidence_class),
                        json_string(&route.route_outcome),
                    )
                },
            );
            let _ = write!(
                sources,
                "{{\"source\":{},\"query_index\":{},\"source_rank\":{},\"seed_memory_id\":{},\"activation\":{},\"graph_route\":{}}}",
                json_string(source),
                observation.query_index,
                observation.source_rank,
                optional_string_json(observation.seed_memory_id.as_deref()),
                activation,
                route,
            );
        }
        sources.push(']');
        let _ = write!(
            candidates,
            "{{\"memory_id\":{},\"merged_rank\":{},\"admitted\":{},\"dropped_at\":{},\"graph_allocation_rank\":{},\"sources\":{}}}",
            json_string(&candidate.memory_id),
            candidate.merged_rank,
            candidate.admitted,
            optional_string_json(candidate.dropped_at.as_deref()),
            candidate.graph_allocation_rank.map_or_else(
                || "null".to_string(),
                |rank| rank.to_string(),
            ),
            sources,
        );
    }
    candidates.push(']');
    let cos_top = if pool.cos_top.is_finite() {
        pool.cos_top.to_string()
    } else {
        "null".to_string()
    };
    format!(
        "{{\"pool_trace\":{{\"pool_width\":{},\"cos_top\":{},\"candidate_count\":{},\"observed_count\":{},\"graph_outcome\":{},\"candidates\":{}}}}}",
        pool.pool_width,
        cos_top,
        pool.candidates.len(),
        pool.observed.len(),
        optional_string_json(pool.graph_outcome.as_deref()),
        candidates,
    )
}

pub(crate) fn pack_payload_json(report: &PackReport) -> String {
    let scores = {
        let mut s = String::from("[");
        for (i, v) in report.scores.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            if v.is_finite() {
                let _ = write!(s, "{v}");
            } else {
                s.push_str("null");
            }
        }
        s.push(']');
        s
    };
    format!(
        "{{\"title\":{},\"format\":{},\"content\":{},\"memory_ids\":{},\"scores\":{},\"truncated\":{},\"top_score\":{}}}",
        json_string(&report.title),
        json_string(&report.format),
        json_string(&report.content),
        string_array_json(&report.memory_ids),
        scores,
        report.truncated,
        match report.top_score {
            Some(s) if s.is_finite() => format!("{s}"),
            _ => "null".to_string(),
        }
    )
}

pub(crate) fn mark_extracted_result_json(report: &MarkExtractedReport) -> String {
    format!(
        "{{\"space\":{},\"updated\":{}}}",
        json_string(&report.space),
        report.updated,
    )
}

pub(crate) fn document_get_result_json(report: &DocumentGetReport) -> String {
    let chunks = report
        .chunks
        .iter()
        .map(document_chunk_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"document\":{{\"space\":{}}},\"chunks\":[{chunks}]}}",
        json_string(&report.space),
    )
}

fn document_chunk_json(chunk: &DocumentChunk) -> String {
    let mut output = format!(
        "{{\"source_episode_id\":{},\"space\":{},\"source_type\":{},\"source_path\":{},\"source_uri\":{},\"chunk_index\":{},\"chunk_count\":{},\"content_sha256\":{},\"ingest_status\":{}",
        json_string(&chunk.source_episode_id),
        json_string(&chunk.space),
        json_string(&chunk.source_type),
        optional_string_json(chunk.source_path.as_deref()),
        optional_string_json(chunk.source_uri.as_deref()),
        chunk.chunk_index,
        chunk.chunk_count,
        optional_string_json(chunk.content_sha256.as_deref()),
        json_string(&chunk.ingest_status),
    );
    if let Some(content) = &chunk.content {
        output.push_str(",\"content\":");
        output.push_str(&json_string(content));
    }
    output.push('}');
    output
}

pub(crate) fn promotion_candidates_result_json(report: &PromotionCandidatesReport) -> String {
    let candidates = report
        .candidates
        .iter()
        .map(promotion_candidate_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"promotion\":{{\"space\":{}}},\"candidates\":[{candidates}]}}",
        json_string(&report.space),
    )
}

fn promotion_candidate_json(candidate: &PromotionCandidate) -> String {
    let mut output = format!(
        "{{\"source_episode_id\":{},\"space\":{},\"source_path\":{},\"source_uri\":{},\"chunk_index\":{},\"chunk_count\":{},\"hits\":{},\"distinct_queries\":{},\"last_hit\":{}",
        json_string(&candidate.source_episode_id),
        json_string(&candidate.space),
        optional_string_json(candidate.source_path.as_deref()),
        optional_string_json(candidate.source_uri.as_deref()),
        candidate.chunk_index,
        candidate.chunk_count,
        candidate.hits,
        candidate.distinct_queries,
        json_string(&candidate.last_hit),
    );
    if let Some(content) = &candidate.content {
        output.push_str(",\"content\":");
        output.push_str(&json_string(content));
    }
    output.push('}');
    output
}

pub(crate) fn document_prune_result_json(report: &DocumentPruneReport) -> String {
    let deleted = report
        .deleted
        .iter()
        .map(|id| json_string(id))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"document_prune\":{{\"space\":{},\"requested\":{},\"deleted_count\":{},\"dry_run\":{}}},\"deleted\":[{deleted}]}}",
        json_string(&report.space),
        report.requested,
        report.deleted.len(),
        report.dry_run,
    )
}

pub(crate) fn document_duplicates_result_json(report: &DocumentDuplicatesReport) -> String {
    let clusters = report
        .clusters
        .iter()
        .map(duplicate_cluster_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"document_duplicates\":{{\"space\":{}}},\"clusters\":[{clusters}]}}",
        json_string(&report.space),
    )
}

fn duplicate_cluster_json(cluster: &DuplicateChunkCluster) -> String {
    let members = cluster
        .members
        .iter()
        .map(duplicate_member_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"content_sha256\":{},\"member_count\":{},\"snippet\":{},\"members\":[{members}]}}",
        json_string(&cluster.content_sha256),
        cluster.member_count,
        json_string(&cluster.snippet),
    )
}

fn duplicate_member_json(member: &DuplicateChunkMember) -> String {
    format!(
        "{{\"source_episode_id\":{},\"source_path\":{},\"source_uri\":{},\"chunk_index\":{},\"chunk_count\":{},\"ingest_status\":{},\"ingested_at\":{}}}",
        json_string(&member.source_episode_id),
        optional_string_json(member.source_path.as_deref()),
        optional_string_json(member.source_uri.as_deref()),
        member.chunk_index,
        member.chunk_count,
        json_string(&member.ingest_status),
        json_string(&member.ingested_at),
    )
}

pub(crate) fn document_search_result_json(report: &DocumentSearchReport) -> String {
    let results = report
        .results
        .iter()
        .map(document_search_result_item_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"document_search\":{{\"strategy\":{},\"semantic_attempted\":{},\"space\":{}}},\"results\":[{results}]}}",
        json_string(&report.strategy),
        report.semantic_attempted,
        json_string(&report.space),
    )
}

fn document_search_result_item_json(result: &DocumentSearchResult) -> String {
    let mut output = format!(
        "{{\"rank\":{},\"source_episode_id\":{},\"space\":{},\"source_type\":{},\"source_path\":{},\"source_uri\":{},\"chunk_index\":{},\"chunk_count\":{},\"score\":{},\"match_type\":{},\"snippet\":{}",
        result.rank,
        json_string(&result.source_episode_id),
        json_string(&result.space),
        json_string(&result.source_type),
        optional_string_json(result.source_path.as_deref()),
        optional_string_json(result.source_uri.as_deref()),
        result.chunk_index,
        result.chunk_count,
        finite_number_json(result.score),
        json_string(&result.match_type),
        json_string(&result.snippet),
    );
    if let Some(content) = &result.content {
        output.push_str(",\"content\":");
        output.push_str(&json_string(content));
    }
    output.push('}');
    output
}

pub(crate) fn search_result_json(
    report: &SearchReport,
    last_synth: Option<&str>,
    reranked: bool,
) -> String {
    format!(
        "{{\"search\":{{\"strategy\":{},\"semantic\":{{\"attempted\":{},\"reason\":{}}},\"reranked\":{reranked},\"total_estimate\":{},\"truncated\":{}}},\"results\":{}}}",
        json_string(&report.strategy),
        report.semantic_attempted,
        json_string(&report.semantic_reason),
        report.total_estimate,
        report.truncated,
        search_results_json(&report.results, last_synth)
    )
}

pub(crate) fn entity_upsert_result_json(report: &EntityUpsertReport) -> String {
    format!(
        "{{\"entity_upsert\":{{\"strategy\":{},\"created\":{}}},\"entity\":{}}}",
        json_string(&report.strategy),
        report.created,
        entity_record_json(&report.entity)
    )
}

pub(crate) fn relationship_upsert_result_json(report: &RelationshipUpsertReport) -> String {
    format!(
        "{{\"relationship_upsert\":{{\"strategy\":{},\"created\":{}}},\"relationship\":{}}}",
        json_string(&report.strategy),
        report.created,
        relationship_record_json(&report.relationship)
    )
}

pub(crate) fn entity_merge_result_json(report: &EntityMergeReport) -> String {
    format!(
        "{{\"entity_merge\":{{\"strategy\":{},\"dry_run\":{},\"from_entity_key\":{},\"into_entity_key\":{},\"relationships_repointed\":{},\"relationships_tombstoned_duplicate\":{},\"relationships_tombstoned_self_loop\":{},\"aliases_added\":{},\"from_tombstoned\":{}}},\"into\":{}}}",
        json_string(&report.strategy),
        report.dry_run,
        json_string(&report.from_entity_key),
        json_string(&report.into_entity_key),
        report.relationships_repointed,
        report.relationships_tombstoned_duplicate,
        report.relationships_tombstoned_self_loop,
        report.aliases_added,
        report.from_tombstoned,
        entity_record_json(&report.into)
    )
}

pub(crate) fn entity_search_result_json(report: &EntitySearchReport) -> String {
    format!(
        "{{\"entity_search\":{{\"strategy\":{},\"space\":{},\"total_estimate\":{},\"truncated\":{}}},\"results\":{}}}",
        json_string(&report.strategy),
        json_string(&report.space),
        report.total_estimate,
        report.truncated,
        entity_search_results_json(&report.results)
    )
}

pub(crate) fn graph_full_result_json(report: &GraphFullReport) -> String {
    let nodes = report
        .nodes
        .iter()
        .map(|n| {
            format!(
                "{{\"entity_key\":{},\"canonical_name\":{},\"entity_type\":{},\"degree\":{}}}",
                json_string(&n.entity_key),
                json_string(&n.canonical_name),
                optional_string_json(n.entity_type.as_deref()),
                n.degree,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let links = report
        .links
        .iter()
        .map(|l| {
            format!(
                "{{\"source\":{},\"target\":{},\"weight\":{}}}",
                json_string(&l.source),
                json_string(&l.target),
                l.weight,
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"nodes\":[{nodes}],\"links\":[{links}]}}")
}

pub(crate) fn graph_neighbors_result_json(report: &GraphNeighborsReport) -> String {
    format!(
        "{{\"graph_neighbors\":{{\"strategy\":{},\"depth\":{},\"max_edges\":{},\"truncated\":{}}},\"seed\":{},\"entities\":{},\"relationships\":{}}}",
        json_string(&report.strategy),
        report.depth,
        report.max_edges,
        report.truncated,
        entity_record_json(&report.seed),
        graph_entities_json(&report.entities),
        graph_relationships_json(&report.relationships)
    )
}

pub(crate) fn entity_search_results_json(results: &[EntitySearchResult]) -> String {
    let mut output = String::from("[");
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"rank\":{},\"entity\":{},\"matched_aliases\":{}}}",
            result.rank,
            entity_record_json(&result.entity),
            string_array_json(&result.matched_aliases)
        );
    }
    output.push(']');
    output
}

pub(crate) fn graph_entities_json(entities: &[GraphEntityRecord]) -> String {
    let mut output = String::from("[");
    for (index, entity) in entities.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"depth\":{},\"entity\":{}}}",
            entity.depth,
            entity_record_json(&entity.entity)
        );
    }
    output.push(']');
    output
}

pub(crate) fn graph_relationships_json(relationships: &[GraphRelationshipRecord]) -> String {
    let mut output = String::from("[");
    for (index, relationship) in relationships.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"subject_depth\":{},\"object_depth\":{},\"relationship\":{}}}",
            relationship.subject_depth,
            relationship.object_depth,
            relationship_record_json(&relationship.relationship)
        );
    }
    output.push(']');
    output
}

pub(crate) fn entity_record_json(entity: &EntityRecord) -> String {
    let mut output = format!(
        "{{\"id\":{},\"space\":{},\"entity_key\":{},\"entity_type\":{},\"canonical_name\":{},\"status\":{},\"confidence\":{},\"aliases\":{},\"created_at\":{},\"updated_at\":{}",
        json_string(&entity.id),
        json_string(&entity.space),
        json_string(&entity.entity_key),
        json_string(&entity.entity_type),
        json_string(&entity.canonical_name),
        json_string(&entity.status),
        finite_number_json(entity.confidence),
        string_array_json(&entity.aliases),
        json_string(&entity.created_at),
        json_string(&entity.updated_at)
    );
    if let Some(source_episode_id) = &entity.source_episode_id {
        output.push_str(",\"source_episode_id\":");
        output.push_str(&json_string(source_episode_id));
    }
    output.push('}');
    output
}

pub(crate) fn relationship_record_json(relationship: &RelationshipRecord) -> String {
    let mut output = format!(
        "{{\"id\":{},\"space\":{},\"subject_entity_id\":{},\"relation_type\":{},\"object_entity_id\":{},\"memory_id\":{},\"status\":{},\"confidence\":{},\"observed_at\":{},\"valid_from\":{},\"valid_to\":{},\"created_at\":{},\"updated_at\":{}",
        json_string(&relationship.id),
        json_string(&relationship.space),
        json_string(&relationship.subject_entity_id),
        json_string(&relationship.relation_type),
        json_string(&relationship.object_entity_id),
        optional_string_json(relationship.memory_id.as_deref()),
        json_string(&relationship.status),
        finite_number_json(relationship.confidence),
        optional_string_json(relationship.observed_at.as_deref()),
        optional_string_json(relationship.valid_from.as_deref()),
        optional_string_json(relationship.valid_to.as_deref()),
        json_string(&relationship.created_at),
        json_string(&relationship.updated_at)
    );
    if let Some(source_episode_id) = &relationship.source_episode_id {
        output.push_str(",\"source_episode_id\":");
        output.push_str(&json_string(source_episode_id));
    }
    output.push('}');
    output
}

pub(crate) fn graph_context_result_json(report: &GraphContextReport) -> String {
    format!(
        "{{\"graph_context\":{{\"strategy\":{}}},\"graph\":{},\"pack\":{},\"evidence_memory_ids\":{},\"entity_memory_ids\":{}}}",
        json_string(&report.strategy),
        graph_neighbors_result_json(&report.graph),
        pack_payload_json(&report.pack),
        string_array_json(&report.evidence_memory_ids),
        string_array_json(&report.entity_memory_ids)
    )
}

pub(crate) fn memory_list_result_json(report: &MemoryListReport) -> String {
    format!(
        "{{\"review\":{{\"strategy\":{},\"total_estimate\":{},\"truncated\":{}}},\"results\":{}}}",
        json_string(&report.strategy),
        report.total_estimate,
        report.truncated,
        memory_list_items_json(&report.results)
    )
}

pub(crate) fn memory_list_items_json(results: &[MemoryListItem]) -> String {
    let mut output = String::from("[");
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&memory_list_item_json(result));
    }
    output.push(']');
    output
}

pub(crate) fn memory_list_item_json(result: &MemoryListItem) -> String {
    let mut output = format!(
        "{{\"rank\":{},\"memory_id\":{},\"version_id\":{},\"space\":{},\"silo\":{},\"scope\":{},\"project\":{},\"kind\":{},\"status\":{},\"confidence\":{},\"pinned\":{},\"summary\":{},\"snippet\":{},\"tags\":{},\"entity_key\":{},\"claim_key\":{},\"observed_at\":{},\"created_at\":{},\"updated_at\":{}",
        result.rank,
        json_string(&result.memory_id),
        json_string(&result.version_id),
        json_string(&result.space),
        json_string(&result.silo),
        json_string(&result.scope),
        optional_string_json(result.project_key.as_deref()),
        json_string(&result.kind),
        json_string(&result.status),
        finite_number_json(result.confidence),
        result.pinned,
        optional_string_json(result.summary.as_deref()),
        json_string(&result.snippet),
        string_array_json(&result.tags),
        optional_string_json(result.entity_key.as_deref()),
        optional_string_json(result.claim_key.as_deref()),
        json_string(&result.observed_at),
        json_string(&result.created_at),
        json_string(&result.updated_at)
    );
    if let Some(content) = &result.content {
        output.push_str(",\"content\":");
        output.push_str(&json_string(content));
    }
    if let Some(source) = &result.source_ref_json {
        output.push_str(",\"source\":");
        output.push_str(source);
    }
    output.push('}');
    output
}

pub(crate) fn search_results_json(results: &[SearchResult], last_synth: Option<&str>) -> String {
    let mut output = String::from("[");
    for (index, result) in results.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&search_result_item_json(result, last_synth));
    }
    output.push(']');
    output
}

pub(crate) fn search_result_item_json(result: &SearchResult, last_synth: Option<&str>) -> String {
    let mut output = format!(
        "{{\"rank\":{},\"memory_id\":{},\"version_id\":{},\"score\":{},\"scores\":{{\"fts\":{},\"metadata\":{},\"recency\":{},\"scope\":{},\"status\":{},\"pin\":{},\"source_tier\":{}}},\"space\":{},\"silo\":{},\"scope\":{},\"project\":{},\"kind\":{},\"status\":{},\"summary\":{},\"snippet\":{},\"tags\":{},\"entity_key\":{},\"claim_key\":{},\"observed_at\":{}",
        result.rank,
        json_string(&result.memory_id),
        json_string(&result.version_id),
        finite_number_json(result.score),
        finite_number_json(result.scores.fts),
        finite_number_json(result.scores.metadata),
        finite_number_json(result.scores.recency),
        finite_number_json(result.scores.scope),
        finite_number_json(result.scores.status),
        finite_number_json(result.scores.pin),
        finite_number_json(result.scores.source_tier),
        json_string(&result.space),
        json_string(&result.silo),
        json_string(&result.scope),
        optional_string_json(result.project_key.as_deref()),
        json_string(&result.kind),
        json_string(&result.status),
        optional_string_json(result.summary.as_deref()),
        json_string(&result.snippet),
        string_array_json(&result.tags),
        optional_string_json(result.entity_key.as_deref()),
        optional_string_json(result.claim_key.as_deref()),
        json_string(&result.observed_at)
    );
    if let Some(content) = &result.content {
        output.push_str(",\"content\":");
        output.push_str(&json_string(content));
    }
    if let Some(source) = &result.source_ref_json {
        output.push_str(",\"source\":");
        output.push_str(source);
    }
    let verified_against = metadata_field(result.metadata_json.as_deref(), "verified_against");
    let verified_at = metadata_field(result.metadata_json.as_deref(), "verified_at");
    if let Some(ref va) = verified_against {
        output.push_str(",\"verified_against\":");
        output.push_str(&json_string(va));
    }
    if let Some(ref vt) = verified_at {
        output.push_str(",\"verified_at\":");
        output.push_str(&json_string(vt));
    }
    if let Some(label) = freshness_label(
        &result.silo,
        verified_at.as_deref(),
        verified_against.as_deref(),
        last_synth,
    ) {
        output.push_str(",\"freshness\":");
        output.push_str(&json_string(label));
    }
    output.push('}');
    output
}

pub(crate) fn freshness_label(
    silo: &str,
    verified_at: Option<&str>,
    verified_against: Option<&str>,
    last_synth: Option<&str>,
) -> Option<&'static str> {
    if silo == "durable" {
        return None;
    }
    verified_against?; // only the checkable subset gets a label
    let fresh = matches!((verified_at, last_synth), (Some(v), Some(ls)) if v >= ls);
    Some(if fresh { "fresh" } else { "verify" })
}

pub(crate) fn metadata_field(metadata_json: Option<&str>, key: &str) -> Option<String> {
    metadata_json
        .and_then(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .and_then(|v| v.get(key).and_then(|x| x.as_str()).map(str::to_string))
}

pub(crate) fn remember_result_json(report: &RememberReport) -> String {
    let mut memory = report.memory.clone();
    memory.retrieval_representation = None;
    let mut output = format!(
        "{{\"memory\":{},\"event\":{{\"id\":{},\"type\":\"remember\"}},\"processing_status\":{},\"candidates\":{},\"candidates_truncated\":{},\"auto_superseded\":{},\"conflict_candidates\":{},\"supersede_suggestions\":{},\"dry_run\":{}}}",
        memory_json(
            &memory,
            GetOptions {
                include_history: false,
                include_links: true,
                include_source: false,
            }
        ),
        json_string(&report.event_id),
        json_string(&report.processing_status),
        remember_candidates_json(&report.candidates),
        report.candidates_truncated,
        string_array_json(&report.auto_superseded),
        remember_conflict_candidates_json(&report.conflict_candidates),
        string_array_json(&report.supersede_suggestions),
        report.dry_run
    );
    if let Some(representation) = &report.representation {
        output.pop();
        output.push_str(",\"representation\":");
        output.push_str(&representation_status_json(representation));
        output.push('}');
    }
    output
}

fn representation_status_json(value: &RepresentationWriteStatus) -> String {
    format!(
        "{{\"kind\":{},\"text_sha256\":{},\"fts_indexed\":{},\"semantic_indexed\":{},\"status\":{}}}",
        json_string(&value.kind),
        json_string(&value.text_sha256),
        value.fts_indexed,
        value.semantic_indexed,
        json_string(&value.status),
    )
}

fn memory_representation_json(value: &MemoryRepresentationRecord) -> String {
    format!(
        "{{\"version_id\":{},\"kind\":{},\"text\":{},\"text_sha256\":{},\"created_at\":{}}}",
        json_string(&value.version_id),
        json_string(&value.kind),
        json_string(&value.text),
        json_string(&value.text_sha256),
        json_string(&value.created_at),
    )
}

pub(crate) fn remember_conflict_candidates_json(
    candidates: &[RememberConflictCandidate],
) -> String {
    let mut output = String::from("[");
    for (index, candidate) in candidates.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"memory_id\":{},\"kind\":{},\"observed_at\":{},\"snippet\":{}}}",
            json_string(&candidate.memory_id),
            json_string(&candidate.kind),
            json_string(&candidate.observed_at),
            json_string(&candidate.snippet)
        );
    }
    output.push(']');
    output
}

pub(crate) fn remember_candidates_json(candidates: &[RememberCandidate]) -> String {
    let mut output = String::from("[");
    for (index, candidate) in candidates.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"memory_id\":{},\"relationship\":{},\"score\":{},\"matched_on\":{},\"space\":{},\"silo\":{},\"kind\":{},\"status\":{},\"summary\":{},\"snippet\":{},\"content_sha256\":{},\"entity_key\":{},\"claim_key\":{}}}",
            json_string(&candidate.memory_id),
            json_string(&candidate.relationship),
            finite_number_json(candidate.score),
            string_array_json(&candidate.matched_on),
            json_string(&candidate.space),
            json_string(&candidate.silo),
            json_string(&candidate.kind),
            json_string(&candidate.status),
            optional_string_json(candidate.summary.as_deref()),
            json_string(&candidate.snippet),
            json_string(&candidate.content_sha256),
            optional_string_json(candidate.entity_key.as_deref()),
            optional_string_json(candidate.claim_key.as_deref())
        );
    }
    output.push(']');
    output
}

pub(crate) fn forget_result_json(report: &ForgetReport) -> String {
    format!(
        "{{\"memory_id\":{},\"old_status\":{},\"new_status\":{},\"event\":{{\"id\":{},\"type\":\"forget\"}},\"dry_run\":{}}}",
        json_string(&report.memory_id),
        json_string(&report.old_status),
        json_string(&report.new_status),
        json_string(&report.event_id),
        report.dry_run
    )
}

pub(crate) fn verify_result_json(report: &VerifyReport) -> String {
    format!(
        "{{\"memory_id\":{},\"verified_at\":{}}}",
        json_string(&report.memory_id),
        json_string(&report.verified_at),
    )
}

pub(crate) fn recall_log_result_json(report: &RecallLogReport) -> String {
    format!(
        "{{\"recorded\":{},\"touched\":{}}}",
        report.recorded, report.touched
    )
}

pub(crate) fn history_result_json(report: &HistoryReport) -> String {
    format!(
        "{{\"memory_id\":{},\"current_status\":{},\"events\":{},\"versions\":{},\"truncated\":{}}}",
        json_string(&report.memory_id),
        json_string(&report.current_status),
        events_json(&report.events),
        versions_json(&report.versions),
        report.truncated
    )
}

pub(crate) fn export_result_json(report: &ExportReport) -> String {
    format!(
        "{{\"output\":{},\"format\":{},\"schema_version\":{},\"tables\":{},\"row_count\":{},\"bytes\":{},\"sha256\":{}}}",
        json_path(&report.output_path),
        json_string(&report.format),
        report.schema_version,
        export_tables_json(&report.tables),
        report.row_count,
        report.bytes,
        json_string(&report.sha256)
    )
}

pub(crate) fn import_result_json(report: &ImportReport) -> String {
    format!(
        "{{\"input\":{},\"format\":{},\"schema_version\":{},\"dry_run\":{},\"tables\":{},\"row_count\":{},\"bytes\":{},\"sha256\":{},\"fts_memory_rows\":{},\"fts_source_episode_rows\":{}}}",
        json_path(&report.input_path),
        json_string(&report.format),
        report.schema_version,
        report.dry_run,
        export_tables_json(&report.tables),
        report.row_count,
        report.bytes,
        json_string(&report.sha256),
        report.fts_memory_rows,
        report.fts_source_episode_rows
    )
}

pub(crate) fn dream_result_json(report: &DreamReport) -> String {
    format!(
        "{{\"dream\":{{\"id\":{},\"status\":{},\"space\":{},\"silos\":{},\"tasks\":{},\"started_at\":{},\"finished_at\":{},\"dry_run\":{},\"journaled\":{},\"max_memories\":{}}},\"promote\":{},\"expire\":{},\"reindex\":{},\"dedupe\":{},\"graph\":{},\"link\":{}}}",
        json_string(&report.run_id),
        json_string(&report.status),
        optional_string_json(report.space.as_deref()),
        string_array_json(&report.silos),
        string_array_json(&report.tasks),
        json_string(&report.started_at),
        json_string(&report.finished_at),
        report.dry_run,
        report.journaled,
        report.max_memories,
        dream_promote_json(&report.promote),
        dream_expire_json(&report.expire),
        dream_reindex_json(&report.reindex),
        dream_dedupe_json(&report.dedupe),
        dream_graph_json(&report.graph),
        dream_link_json(&report.link)
    )
}

pub(crate) fn dream_link_json(report: &memkeeper_store::DreamLinkReport) -> String {
    format!(
        "{{\"attempted\":{},\"candidates\":{},\"links_written\":{},\"truncated\":{}}}",
        report.attempted, report.candidates, report.links_written, report.truncated
    )
}

pub(crate) fn dream_promote_json(report: &DreamPromoteReport) -> String {
    format!(
        "{{\"attempted\":{},\"scanned\":{},\"promoted\":{},\"truncated\":{},\"memory_ids\":{}}}",
        report.attempted,
        report.scanned,
        report.promoted,
        report.truncated,
        string_array_json(&report.memory_ids)
    )
}

pub(crate) fn dream_expire_json(report: &DreamExpireReport) -> String {
    format!(
        "{{\"attempted\":{},\"scanned\":{},\"expired\":{},\"skipped_pinned\":{},\"truncated\":{},\"memory_ids\":{},\"skipped_pinned_ids\":{}}}",
        report.attempted,
        report.scanned,
        report.expired,
        report.skipped_pinned,
        report.truncated,
        string_array_json(&report.memory_ids),
        string_array_json(&report.skipped_pinned_ids)
    )
}

pub(crate) fn dream_reindex_json(report: &DreamReindexReport) -> String {
    format!(
        "{{\"attempted\":{},\"memory_rows\":{},\"source_episode_rows\":{}}}",
        report.attempted, report.memory_rows, report.source_episode_rows
    )
}

pub(crate) fn dream_dedupe_json(report: &DreamDedupeReport) -> String {
    let mut output = format!(
        "{{\"attempted\":{},\"truncated\":{},\"proposals\":[",
        report.attempted, report.truncated
    );
    for (index, proposal) in report.proposals.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&dream_duplicate_proposal_json(proposal));
    }
    output.push_str("]}");
    output
}

pub(crate) fn dream_graph_json(report: &DreamGraphReport) -> String {
    format!(
        "{{\"attempted\":{},\"orphan_entities\":{},\"dangling_relationships\":{},\"inactive_evidence_relationships\":{},\"missing_entity_projections\":{},\"relationship_proposals\":{},\"truncated\":{},\"orphan_entity_ids\":{},\"dangling_relationship_ids\":{},\"inactive_evidence_relationship_ids\":{}}}",
        report.attempted,
        report.orphan_entities,
        report.dangling_relationships,
        report.inactive_evidence_relationships,
        dream_entity_projections_json(&report.missing_entity_projections),
        dream_relationship_proposals_json(&report.relationship_proposals),
        report.truncated,
        string_array_json(&report.orphan_entity_ids),
        string_array_json(&report.dangling_relationship_ids),
        string_array_json(&report.inactive_evidence_relationship_ids)
    )
}

pub(crate) fn dream_entity_projections_json(projections: &[DreamEntityProjection]) -> String {
    format!(
        "[{}]",
        projections
            .iter()
            .map(|projection| format!(
                "{{\"space\":{},\"entity_key\":{}}}",
                json_string(&projection.space),
                json_string(&projection.entity_key)
            ))
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn dream_relationship_proposals_json(proposals: &[DreamRelationshipProposal]) -> String {
    format!(
        "[{}]",
        proposals
            .iter()
            .map(dream_relationship_proposal_json)
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn dream_relationship_proposal_json(proposal: &DreamRelationshipProposal) -> String {
    format!(
        "{{\"space\":{},\"src_memory_id\":{},\"dst_memory_id\":{},\"link_type\":{},\"subject_entity_key\":{},\"relation_type\":{},\"object_entity_key\":{}}}",
        json_string(&proposal.space),
        json_string(&proposal.src_memory_id),
        json_string(&proposal.dst_memory_id),
        json_string(&proposal.link_type),
        json_string(&proposal.subject_entity_key),
        json_string(&proposal.relation_type),
        json_string(&proposal.object_entity_key)
    )
}

pub(crate) fn dream_duplicate_proposal_json(proposal: &DreamDuplicateProposal) -> String {
    format!(
        "{{\"space\":{},\"content_sha256\":{},\"canonical_memory_id\":{},\"duplicate_memory_ids\":{},\"total_count\":{},\"pinned_count\":{},\"duplicate_ids_truncated\":{}}}",
        json_string(&proposal.space),
        json_string(&proposal.content_sha256),
        json_string(&proposal.canonical_memory_id),
        string_array_json(&proposal.duplicate_memory_ids),
        proposal.total_count,
        proposal.pinned_count,
        proposal.duplicate_ids_truncated
    )
}

pub(crate) fn backup_result_json(report: &BackupReport) -> String {
    format!(
        "{{\"output\":{},\"format\":{},\"schema_version\":{},\"page_count\":{},\"bytes\":{},\"sha256\":{}}}",
        json_path(&report.output_path),
        json_string(&report.format),
        report.schema_version,
        report.page_count,
        report.bytes,
        json_string(&report.sha256)
    )
}

pub(crate) fn export_tables_json(tables: &[ExportTableReport]) -> String {
    let mut output = String::from("[");
    for (index, table) in tables.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"name\":{},\"rows\":{}}}",
            json_string(&table.name),
            table.rows
        );
    }
    output.push(']');
    output
}

pub(crate) fn memory_json(memory: &MemoryRecord, options: GetOptions) -> String {
    let mut output = format!(
        "{{\"id\":{},\"version_id\":{},\"space\":{},\"silo\":{},\"scope\":{},\"project\":{},\"kind\":{},\"entity_key\":{},\"claim_key\":{},\"status\":{},\"confidence\":{},\"pinned\":{},\"content\":{},\"summary\":{},\"content_sha256\":{},\"tags\":{},\"observed_at\":{},\"created_at\":{},\"updated_at\":{},\"valid_from\":{},\"valid_to\":{},\"expires_at\":{},\"deleted_at\":{}",
        json_string(&memory.id),
        json_string(&memory.version_id),
        json_string(&memory.space),
        json_string(&memory.silo),
        json_string(&memory.scope),
        optional_string_json(memory.project_key.as_deref()),
        json_string(&memory.kind),
        optional_string_json(memory.entity_key.as_deref()),
        optional_string_json(memory.claim_key.as_deref()),
        json_string(&memory.status),
        finite_number_json(memory.confidence),
        memory.pinned,
        json_string(&memory.content),
        optional_string_json(memory.summary.as_deref()),
        json_string(&memory.content_sha256),
        string_array_json(&memory.tags),
        json_string(&memory.observed_at),
        json_string(&memory.created_at),
        json_string(&memory.updated_at),
        optional_string_json(memory.valid_from.as_deref()),
        optional_string_json(memory.valid_to.as_deref()),
        optional_string_json(memory.expires_at.as_deref()),
        optional_string_json(memory.deleted_at.as_deref())
    );
    if options.include_source {
        output.push_str(",\"source_episode_id\":");
        output.push_str(&optional_string_json(memory.source_episode_id.as_deref()));
        output.push_str(",\"source\":");
        output.push_str(memory.source_ref_json.as_deref().unwrap_or("null"));
    }
    if let Some(representation) = &memory.retrieval_representation {
        output.push_str(",\"retrieval_representation\":");
        output.push_str(&memory_representation_json(representation));
    }
    if options.include_history {
        output.push_str(",\"versions\":");
        output.push_str(&versions_json(memory.versions.as_deref().unwrap_or(&[])));
        output.push_str(",\"events\":");
        output.push_str(&events_json(memory.events.as_deref().unwrap_or(&[])));
    }
    if options.include_links {
        output.push_str(",\"links\":");
        output.push_str(&links_json(memory.links.as_deref().unwrap_or(&[])));
    }
    output.push('}');
    output
}

pub(crate) fn versions_json(versions: &[MemoryVersionRecord]) -> String {
    let mut output = String::from("[");
    for (index, version) in versions.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let mut item = format!(
            "{{\"id\":{},\"version_num\":{},\"content\":{},\"summary\":{},\"content_sha256\":{},\"created_at\":{},\"source\":{}}}",
            json_string(&version.id),
            version.version_num,
            json_string(&version.content),
            optional_string_json(version.summary.as_deref()),
            json_string(&version.content_sha256),
            json_string(&version.created_at),
            version.source_ref_json.as_deref().unwrap_or("null")
        );
        if let Some(representation) = &version.retrieval_representation {
            item.pop();
            item.push_str(",\"retrieval_representation\":");
            item.push_str(&memory_representation_json(representation));
            item.push('}');
        }
        output.push_str(&item);
    }
    output.push(']');
    output
}

pub(crate) fn events_json(events: &[MemoryEventRecord]) -> String {
    let mut output = String::from("[");
    for (index, event) in events.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"id\":{},\"type\":{},\"old_status\":{},\"new_status\":{},\"reason\":{},\"created_at\":{}}}",
            json_string(&event.id),
            json_string(&event.event_type),
            optional_string_json(event.old_status.as_deref()),
            optional_string_json(event.new_status.as_deref()),
            optional_string_json(event.reason.as_deref()),
            json_string(&event.created_at)
        );
    }
    output.push(']');
    output
}

pub(crate) fn links_json(links: &[MemoryLinkRecord]) -> String {
    let mut output = String::from("[");
    for (index, link) in links.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"src_memory_id\":{},\"dst_memory_id\":{},\"link_type\":{},\"status\":{},\"confidence\":{}}}",
            json_string(&link.src_memory_id),
            json_string(&link.dst_memory_id),
            json_string(&link.link_type),
            json_string(&link.status),
            optional_f64_json(link.confidence)
        );
    }
    output.push(']');
    output
}

pub(crate) fn optional_string_json(value: Option<&str>) -> String {
    value.map_or_else(|| "null".to_string(), json_string)
}

pub(crate) fn optional_f64_json(value: Option<f64>) -> String {
    value.map_or_else(|| "null".to_string(), finite_number_json)
}

pub(crate) fn finite_number_json(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".to_string()
    }
}

pub(crate) fn index_stats_json(stats: &IndexStats) -> String {
    format!(
        "{{\"fts_memory_rows\":{},\"fts_source_episode_rows\":{},\"document_duplicate_clusters\":{},\"pending_jobs\":{}}}",
        stats.fts_memory_rows,
        stats.fts_source_episode_rows,
        stats.document_duplicate_clusters,
        stats.pending_jobs
    )
}

pub(crate) fn health_stats_json(stats: &HealthStats) -> String {
    format!(
        "{{\"active\":{},\"superseded\":{},\"conflicted\":{},\"tombstoned\":{},\"expired\":{},\"active_without_keys\":{},\"active_missing_entity_projection\":{},\"duplicate_key_groups\":{},\"short_term_active\":{},\"active_past_valid_to\":{},\"active_without_embedding\":{},\"last_embedding_at\":{},\"candidates\":{{\"pending\":{},\"approved\":{},\"rejected\":{}}}}}",
        stats.active,
        stats.superseded,
        stats.conflicted,
        stats.tombstoned,
        stats.expired,
        stats.active_without_keys,
        stats.active_missing_entity_projection,
        stats.duplicate_key_groups,
        stats.short_term_active,
        stats.active_past_valid_to,
        stats.active_without_embedding,
        optional_string_json(stats.last_embedding_at.as_deref()),
        stats.candidates_pending,
        stats.candidates_approved,
        stats.candidates_rejected,
    )
}

pub(crate) fn space_stats_json(stats: &Stats) -> String {
    let mut output = String::from("[");
    for (index, space) in stats.spaces.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        let _ = write!(
            output,
            "{{\"name\":{},\"memory_count\":{},\"active_count\":{}}}",
            json_string(&space.name),
            space.memory_count,
            space.active_count
        );
    }
    output.push(']');
    output
}

pub(crate) fn string_array_json(values: &[String]) -> String {
    let mut output = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push_str(&json_string(value));
    }
    output.push(']');
    output
}

pub(crate) fn candidate_record_json(record: &CandidateRecord) -> String {
    format!(
        "{{\"id\":{},\"status\":{},\"space\":{},\"silo\":{},\"scope\":{},\"project\":{},\
         \"kind\":{},\"content\":{},\"summary\":{},\"rationale\":{},\"tags\":{},\
         \"entity_key\":{},\"claim_key\":{},\"confidence\":{},\"source_type\":{},\
         \"source\":{},\"sensitivity\":{},\"supersedes\":{},\"created_at\":{},\
         \"decided_at\":{},\"decided_reason\":{},\"resulting_memory_id\":{}}}",
        json_string(&record.id),
        json_string(&record.status),
        optional_string_json(record.space.as_deref()),
        optional_string_json(record.silo.as_deref()),
        optional_string_json(record.scope.as_deref()),
        optional_string_json(record.project.as_deref()),
        optional_string_json(record.kind.as_deref()),
        json_string(&record.content),
        optional_string_json(record.summary.as_deref()),
        optional_string_json(record.rationale.as_deref()),
        string_array_json(&record.tags),
        optional_string_json(record.entity_key.as_deref()),
        optional_string_json(record.claim_key.as_deref()),
        finite_number_json(record.confidence),
        json_string(&record.source_type),
        // source_json is already a validated JSON object string; embed raw.
        record.source_json.as_deref().unwrap_or("null"),
        json_string(&record.sensitivity),
        string_array_json(&record.supersedes),
        json_string(&record.created_at),
        optional_string_json(record.decided_at.as_deref()),
        optional_string_json(record.decided_reason.as_deref()),
        optional_string_json(record.resulting_memory_id.as_deref()),
    )
}

pub(crate) fn ingest_result_json(report: &IngestReport) -> String {
    format!(
        "{{\"space\":{},\"source_path\":{},\"chunk_count\":{},\"created\":{},\"created_count\":{},\"skipped\":{},\"created_space\":{},\"dry_run\":{}}}",
        json_string(&report.space),
        optional_string_json(report.source_path.as_deref()),
        report.chunk_count,
        string_array_json(&report.created),
        report.created.len(),
        report.skipped,
        report.created_space,
        report.dry_run,
    )
}

pub(crate) fn candidate_submit_result_json(report: &CandidateSubmitReport) -> String {
    format!(
        "{{\"candidate\":{},\"dry_run\":{}}}",
        candidate_record_json(&report.candidate),
        report.dry_run
    )
}

pub(crate) fn candidate_list_result_json(report: &CandidateListReport) -> String {
    let items = report
        .candidates
        .iter()
        .map(candidate_record_json)
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"candidates\":[{items}],\"total\":{}}}", report.total)
}

pub(crate) fn candidate_approve_result_json(report: &CandidateApproveReport) -> String {
    format!(
        "{{\"candidate\":{},\"memory\":{},\"dry_run\":{}}}",
        candidate_record_json(&report.candidate),
        memory_json(
            &report.memory,
            GetOptions {
                include_history: false,
                include_links: true,
                include_source: false,
            }
        ),
        report.dry_run
    )
}

pub(crate) fn candidate_reject_result_json(report: &CandidateRejectReport) -> String {
    format!(
        "{{\"candidate\":{},\"dry_run\":{}}}",
        candidate_record_json(&report.candidate),
        report.dry_run
    )
}

pub(crate) fn candidate_quarantine_result_json(report: &CandidateQuarantineReport) -> String {
    format!(
        "{{\"candidate\":{},\"dry_run\":{}}}",
        candidate_record_json(&report.candidate),
        report.dry_run
    )
}

pub(crate) fn json_path(path: &Path) -> String {
    json_string(&path.to_string_lossy())
}

pub(crate) fn json_string(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}
