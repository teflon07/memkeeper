//! Shared request validators extracted from `lib.rs` (pure code movement).

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

use memkeeper_core::{infer_kind_from_prefix, kind, scope, ALL_SPACES, status};

use crate::{
    is_supported_kind, is_supported_scope, is_supported_status, normalized_tags,
    reject_sqlite_sidecar_symlinks, sidecar_path, validate_graph_capture,
    validate_retrieval_representation, CANDIDATE_SENSITIVITIES, CANDIDATE_SOURCE_TYPES,
    Error, ForgetRequest, HistoryOptions, JsonValidator, RememberRequest, Result,
    REMEMBER_SUPERSEDE_MODES, SearchFilters,
    MAX_BATCH_QUERIES, MAX_BATCH_QUERY_LIMIT, MAX_CONTENT_CHARS, MAX_FORGET_REASON_CHARS,
    MAX_HISTORY_LIMIT, MAX_METADATA_VALUE_CHARS, MAX_PACK_CHARS, MAX_PACK_MEMORIES,
    MAX_PACK_TITLE_CHARS, MAX_SEARCH_OFFSET, MAX_SNIPPET_CHARS, MAX_SOURCE_REF_JSON_CHARS,
    MAX_SUMMARY_CHARS, MAX_TAG_CHARS, MAX_TAGS, MAX_TIMESTAMP_CHARS, MAX_MEMORY_LINKS,
    MAX_SEMANTIC_EMBEDDING_DIMS, BackupRequest, BatchSearchRequest,
    ExportRequest, PackRequest,
};

pub(crate) fn validate_export_request(request: &ExportRequest) -> Result<()> {
    if request.format != "jsonl" {
        return Err(Error::InvalidRequest {
            message: "export format must be jsonl in v0.1".to_string(),
        });
    }
    validate_output_path(&request.output_path)
}

pub(crate) fn validate_backup_request(request: &BackupRequest) -> Result<()> {
    if request.format != "sqlite" {
        return Err(Error::InvalidRequest {
            message: "backup format must be sqlite in v0.1".to_string(),
        });
    }
    validate_output_path(&request.output_path)?;
    reject_existing_output_sidecars(&request.output_path)
}
pub(crate) fn validate_batch_search_request(request: &BatchSearchRequest) -> Result<()> {
    if request.queries.is_empty() || request.queries.len() > MAX_BATCH_QUERIES {
        return Err(Error::InvalidRequest {
            message: format!("queries must contain between 1 and {MAX_BATCH_QUERIES} entries"),
        });
    }
    if request.limit == 0 || request.limit > MAX_BATCH_QUERY_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("batch limit must be between 1 and {MAX_BATCH_QUERY_LIMIT}"),
        });
    }
    if request.offset > MAX_SEARCH_OFFSET {
        return Err(Error::InvalidRequest {
            message: format!("offset must be at most {MAX_SEARCH_OFFSET}"),
        });
    }
    if request.snippet_chars > MAX_SNIPPET_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("snippet_chars must be at most {MAX_SNIPPET_CHARS}"),
        });
    }
    if request.semantic_fallback != "disabled" {
        return Err(Error::InvalidRequest {
            message: "semantic_fallback must be disabled in v0.1".to_string(),
        });
    }
    for query in &request.queries {
        if query
            .name
            .as_deref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err(Error::InvalidRequest {
                message: "batch query name must not be empty".to_string(),
            });
        }
        if query
            .name
            .as_deref()
            .is_some_and(|name| name.chars().count() > MAX_TAG_CHARS)
        {
            return Err(Error::InvalidRequest {
                message: format!("batch query name must be at most {MAX_TAG_CHARS} characters"),
            });
        }
        if query
            .limit
            .is_some_and(|limit| limit == 0 || limit > MAX_BATCH_QUERY_LIMIT)
        {
            return Err(Error::InvalidRequest {
                message: format!("batch query limit must be between 1 and {MAX_BATCH_QUERY_LIMIT}"),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_pack_request(request: &PackRequest) -> Result<()> {
    if request.title.trim().is_empty() || request.title.chars().count() > MAX_PACK_TITLE_CHARS {
        return Err(Error::InvalidRequest {
            message: format!(
                "title must be non-empty and at most {MAX_PACK_TITLE_CHARS} characters"
            ),
        });
    }
    if request.queries.is_empty() || request.queries.len() > MAX_BATCH_QUERIES {
        return Err(Error::InvalidRequest {
            message: format!("queries must contain between 1 and {MAX_BATCH_QUERIES} entries"),
        });
    }
    if request.max_memories == 0 || request.max_memories > MAX_PACK_MEMORIES {
        return Err(Error::InvalidRequest {
            message: format!("max_memories must be between 1 and {MAX_PACK_MEMORIES}"),
        });
    }
    if request.max_chars == 0 || request.max_chars > MAX_PACK_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("max_chars must be between 1 and {MAX_PACK_CHARS}"),
        });
    }
    if !request.min_score.is_finite() || request.min_score < 0.0 {
        return Err(Error::InvalidRequest {
            message: "min_score must be a finite value >= 0.0".to_string(),
        });
    }
    if request.rerank_candidates > MAX_PACK_MEMORIES {
        return Err(Error::InvalidRequest {
            message: format!("rerank_candidates must be between 0 and {MAX_PACK_MEMORIES}"),
        });
    }
    if request.format != "markdown" {
        return Err(Error::InvalidRequest {
            message: "pack format must be markdown in v0.1".to_string(),
        });
    }
    Ok(())
}
pub(crate) fn reject_all_spaces_sentinel(space: Option<&str>) -> Result<()> {
    if space.is_some_and(|value| value.trim() == ALL_SPACES) {
        return Err(Error::InvalidRequest {
            message: "space must not be the reserved all-spaces sentinel \"*\"".to_string(),
        });
    }
    Ok(())
}

pub(crate) fn normalize_search_filters(mut filters: SearchFilters) -> Result<SearchFilters> {
    filters.spaces = normalize_filter_values(&filters.spaces)?;
    filters.silos = normalize_filter_values(&filters.silos)?;
    filters.scopes = normalize_filter_values(&filters.scopes)?;
    filters.projects = normalize_filter_values(&filters.projects)?;
    filters.kinds = normalize_filter_values(&filters.kinds)?;
    filters.statuses = normalize_filter_values(&filters.statuses)?;
    filters.entity_keys = normalize_filter_values(&filters.entity_keys)?;
    filters.claim_keys = normalize_filter_values(&filters.claim_keys)?;
    filters.tags = normalized_tags(&filters.tags)?;
    Ok(filters)
}
pub(crate) fn normalize_filter_values(values: &[String]) -> Result<Vec<String>> {
    if values.len() > MAX_TAGS {
        return Err(Error::InvalidRequest {
            message: "filter has too many values".to_string(),
        });
    }
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(values.len());
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(Error::InvalidRequest {
                message: "filter contains an invalid value".to_string(),
            });
        }
        if !seen.insert(trimmed.to_string()) {
            return Err(Error::InvalidRequest {
                message: format!("filter contains duplicate value: {trimmed}"),
            });
        }
        normalized.push(trimmed.to_string());
    }
    Ok(normalized)
}

pub(crate) fn validate_search_filters(filters: &SearchFilters) -> Result<()> {
    validate_filter_values("spaces", &filters.spaces)?;
    validate_filter_values("silos", &filters.silos)?;
    validate_filter_values("scopes", &filters.scopes)?;
    validate_filter_values("projects", &filters.projects)?;
    validate_filter_values("kinds", &filters.kinds)?;
    validate_filter_values("statuses", &filters.statuses)?;
    validate_filter_values("entity_keys", &filters.entity_keys)?;
    validate_filter_values("claim_keys", &filters.claim_keys)?;
    let _ = normalized_tags(&filters.tags)?;
    for scope_value in &filters.scopes {
        if !is_supported_scope(scope_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported scope: {scope_value}"),
            });
        }
    }
    for kind_value in &filters.kinds {
        if !is_supported_kind(kind_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported kind: {kind_value}"),
            });
        }
    }
    for status_value in &filters.statuses {
        if !is_supported_status(status_value) {
            return Err(Error::InvalidRequest {
                message: format!("unsupported status: {status_value}"),
            });
        }
    }
    Ok(())
}

fn validate_filter_values(name: &str, values: &[String]) -> Result<()> {
    if values.len() > MAX_TAGS {
        return Err(Error::InvalidRequest {
            message: format!("filter {name} has too many values"),
        });
    }
    let mut seen = BTreeSet::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(Error::InvalidRequest {
                message: format!("filter {name} contains an invalid value"),
            });
        }
        if !seen.insert(trimmed) {
            return Err(Error::InvalidRequest {
                message: format!("filter {name} contains duplicate value: {trimmed}"),
            });
        }
    }
    Ok(())
}
pub(crate) fn validate_forget_request(request: &ForgetRequest) -> Result<()> {
    if request.id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "memory id must not be empty".to_string(),
        });
    }
    if request.mode != "tombstone" && request.mode != "correct" {
        return Err(Error::InvalidRequest {
            message: "forget mode must be tombstone or correct".to_string(),
        });
    }
    if request.mode != "correct" && request.corrected_by.is_some() {
        return Err(Error::InvalidRequest {
            message: "corrected_by is only valid in correct mode".to_string(),
        });
    }
    if let Some(corrected_by) = &request.corrected_by {
        if corrected_by.trim().is_empty() {
            return Err(Error::InvalidRequest {
                message: "corrected_by must not be empty".to_string(),
            });
        }
        if corrected_by == &request.id {
            return Err(Error::InvalidRequest {
                message: "corrected_by must differ from the corrected memory id".to_string(),
            });
        }
    }
    if let Some(reason) = &request.reason {
        if reason.trim().is_empty() || reason.chars().count() > MAX_FORGET_REASON_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "reason must be non-empty and at most {MAX_FORGET_REASON_CHARS} characters"
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_history_request(id: &str, options: HistoryOptions) -> Result<()> {
    if id.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "memory id must not be empty".to_string(),
        });
    }
    if options.limit == 0 || options.limit > MAX_HISTORY_LIMIT {
        return Err(Error::InvalidRequest {
            message: format!("history limit must be between 1 and {MAX_HISTORY_LIMIT}"),
        });
    }
    Ok(())
}


pub(crate) fn validate_remember_request(request: &RememberRequest) -> Result<()> {
    let content = request.content.trim();
    if content.is_empty() {
        return Err(Error::InvalidRequest {
            message: "content must not be empty".to_string(),
        });
    }
    if request.content.chars().count() > MAX_CONTENT_CHARS {
        return Err(Error::InvalidRequest {
            message: format!("content must be at most {MAX_CONTENT_CHARS} characters"),
        });
    }
    if request
        .summary
        .as_deref()
        .is_some_and(|summary| summary.chars().count() > MAX_SUMMARY_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: format!("summary must be at most {MAX_SUMMARY_CHARS} characters"),
        });
    }
    if let Some(representation) = &request.retrieval_representation {
        validate_retrieval_representation(representation)?;
    }
    validate_optional_metadata_value("space", request.space.as_deref())?;
    reject_all_spaces_sentinel(request.space.as_deref())?;
    validate_optional_metadata_value("silo", request.silo.as_deref())?;
    validate_optional_metadata_value("scope", request.scope.as_deref())?;
    validate_optional_metadata_value("project", request.project_key.as_deref())?;
    validate_optional_metadata_value("kind", request.kind.as_deref())?;
    validate_optional_metadata_value("entity_key", request.entity_key.as_deref())?;
    validate_optional_metadata_value("claim_key", request.claim_key.as_deref())?;
    validate_graph_capture(request.graph.as_ref())?;
    validate_optional_metadata_value("source_episode_id", request.source_episode_id.as_deref())?;
    validate_optional_timestamp("observed_at", request.observed_at.as_deref())?;
    validate_optional_timestamp("valid_from", request.valid_from.as_deref())?;
    validate_optional_timestamp("valid_to", request.valid_to.as_deref())?;
    validate_optional_timestamp("expires_at", request.expires_at.as_deref())?;
    validate_memory_link_ids("supersedes", &request.supersedes)?;
    validate_memory_link_ids("contradicts", &request.contradicts)?;
    if !(0.0..=1.0).contains(&request.confidence) {
        return Err(Error::InvalidRequest {
            message: "confidence must be between 0.0 and 1.0".to_string(),
        });
    }
    validate_optional_embedding("embedding", request.embedding.as_deref())?;
    let _ = normalized_tags(&request.tags)?;
    if let Some(source_ref_json) = &request.source_ref_json {
        if source_ref_json.chars().count() > MAX_SOURCE_REF_JSON_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "source JSON must be at most {MAX_SOURCE_REF_JSON_CHARS} characters"
                ),
            });
        }
        if !JsonValidator::is_object(source_ref_json) {
            return Err(Error::InvalidRequest {
                message: "source JSON must be a valid JSON object".to_string(),
            });
        }
        // Provenance/trust keys recognized inside the source object: when
        // present, source_type and sensitivity must use the shared vocabularies
        // (the same ones candidates validate), so explicit-vs-harvested writes
        // and sensitivity stay consistent across the remember and candidate paths.
        validate_source_object_provenance(source_ref_json)?;
    }
    if !REMEMBER_SUPERSEDE_MODES.contains(&request.mode.as_str()) {
        return Err(Error::InvalidRequest {
            message: format!(
                "unsupported mode: {} (expected one of {})",
                request.mode,
                REMEMBER_SUPERSEDE_MODES.join(", ")
            ),
        });
    }
    let scope = request.scope.as_deref().unwrap_or(scope::WORKSPACE);
    if !is_supported_scope(scope) {
        return Err(Error::InvalidRequest {
            message: format!("unsupported scope: {scope}"),
        });
    }
    let inferred_kind = infer_kind_from_prefix(&request.content);
    let kind = request
        .kind
        .as_deref()
        .or(inferred_kind)
        .unwrap_or(kind::FACT);
    if !is_supported_kind(kind) {
        return Err(Error::InvalidRequest {
            message: format!("unsupported kind: {kind}"),
        });
    }
    Ok(())
}

const MAX_CAPTURE_ENTITIES: usize = 32;
const MAX_CAPTURE_RELATIONSHIPS: usize = 64;

/// Validate the optional `source_type` / `sensitivity` keys inside a source
/// provenance object against the shared candidate vocabularies. Absent keys are
/// fine; only present-but-invalid values error.
fn validate_source_object_provenance(source_ref_json: &str) -> Result<()> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(source_ref_json) else {
        // Shape was already validated as a JSON object above; nothing to do.
        return Ok(());
    };
    let Some(object) = value.as_object() else {
        return Ok(());
    };
    if let Some(source_type) = object.get("source_type").and_then(|v| v.as_str()) {
        if !CANDIDATE_SOURCE_TYPES.contains(&source_type) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported source.source_type: {source_type} (expected one of {})",
                    CANDIDATE_SOURCE_TYPES.join(", ")
                ),
            });
        }
    }
    if let Some(sensitivity) = object.get("sensitivity").and_then(|v| v.as_str()) {
        if !CANDIDATE_SENSITIVITIES.contains(&sensitivity) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "unsupported source.sensitivity: {sensitivity} (expected one of {})",
                    CANDIDATE_SENSITIVITIES.join(", ")
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_optional_metadata_value(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "{name} must be non-empty and at most {MAX_METADATA_VALUE_CHARS} characters"
                ),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_optional_timestamp(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.chars().count() > MAX_TIMESTAMP_CHARS || !is_utc_rfc3339_like(value) {
            return Err(Error::InvalidRequest {
                message: format!("{name} must be a UTC RFC3339 timestamp ending in Z"),
            });
        }
    }
    Ok(())
}

/// Normalize a validated UTC RFC3339 timestamp to fixed millisecond
/// precision (`YYYY-MM-DDTHH:MM:SS.fffZ`) so lexical string comparison
/// matches temporal order. Shorter fractions pad with zeros; longer
/// fractions truncate (sub-millisecond precision is not preserved).
/// Callers must validate with `is_utc_rfc3339_like` first.
pub(crate) fn normalize_utc_timestamp(value: &str) -> String {
    let body = &value[..19];
    let digits = value[19..value.len() - 1].strip_prefix('.').unwrap_or("");
    let mut millis = String::with_capacity(3);
    for index in 0..3 {
        millis.push(
            digits
                .as_bytes()
                .get(index)
                .copied()
                .map_or('0', char::from),
        );
    }
    format!("{body}.{millis}Z")
}

pub(crate) fn validate_optional_embedding(name: &str, embedding: Option<&[f32]>) -> Result<()> {
    if let Some(embedding) = embedding {
        let dims = embedding.len();
        if !(1..=MAX_SEMANTIC_EMBEDDING_DIMS).contains(&dims) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "{name} dimension {dims} is not supported (expected 1..={MAX_SEMANTIC_EMBEDDING_DIMS})"
                ),
            });
        }
        if embedding.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidRequest {
                message: format!("{name} must contain only finite floats"),
            });
        }
    }
    Ok(())
}

pub(crate) fn is_utc_rfc3339_like(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 20 || !value.ends_with('Z') {
        return false;
    }
    if !matches!(bytes.get(4), Some(b'-'))
        || !matches!(bytes.get(7), Some(b'-'))
        || !matches!(bytes.get(10), Some(b'T'))
        || !matches!(bytes.get(13), Some(b':'))
        || !matches!(bytes.get(16), Some(b':'))
        || !timestamp_parts_are_valid(bytes, 0, 5, 8, 11, 14, 17)
    {
        return false;
    }
    match bytes.len() {
        20 => true,
        len if len > 21 && bytes[19] == b'.' => bytes[20..len - 1].iter().all(u8::is_ascii_digit),
        _ => false,
    }
}

pub(crate) fn timestamp_parts_are_valid(
    bytes: &[u8],
    year_start: usize,
    month_start: usize,
    day_start: usize,
    hour_start: usize,
    minute_start: usize,
    second_start: usize,
) -> bool {
    let Some(year) = parse_ascii_digits(bytes, year_start, 4) else {
        return false;
    };
    let Some(month) = parse_ascii_digits(bytes, month_start, 2) else {
        return false;
    };
    let Some(day) = parse_ascii_digits(bytes, day_start, 2) else {
        return false;
    };
    let Some(hour) = parse_ascii_digits(bytes, hour_start, 2) else {
        return false;
    };
    let Some(minute) = parse_ascii_digits(bytes, minute_start, 2) else {
        return false;
    };
    let Some(second) = parse_ascii_digits(bytes, second_start, 2) else {
        return false;
    };

    (1..=12).contains(&month)
        && day >= 1
        && day <= days_in_month(year, month)
        && hour <= 23
        && minute <= 59
        && second <= 59
}

fn parse_ascii_digits(bytes: &[u8], start: usize, len: usize) -> Option<u32> {
    let end = start.checked_add(len)?;
    let slice = bytes.get(start..end)?;
    let mut value = 0_u32;
    for byte in slice {
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value * 10 + u32::from(byte - b'0');
    }
    Some(value)
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: u32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

pub(crate) fn validate_memory_link_ids(name: &str, values: &[String]) -> Result<()> {
    if values.len() > MAX_MEMORY_LINKS {
        return Err(Error::InvalidRequest {
            message: format!("{name} must contain at most {MAX_MEMORY_LINKS} memory ids"),
        });
    }
    let mut seen = BTreeSet::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "{name} ids must be non-empty and at most {MAX_METADATA_VALUE_CHARS} characters"
                ),
            });
        }
        if !seen.insert(trimmed) {
            return Err(Error::InvalidRequest {
                message: format!("{name} contains duplicate memory id: {trimmed}"),
            });
        }
    }
    Ok(())
}
pub(crate) fn validate_output_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "output path must not be empty",
        });
    }
    if path
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "output path must not contain parent directory components",
        });
    }
    let display_path = path.to_string_lossy();
    if path == Path::new(":memory:") || display_path.starts_with("file:") {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "output paths must use plain filesystem paths",
        });
    }
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(Error::InvalidPath {
                    path: path.to_path_buf(),
                    reason: "output path must not be a symlink",
                });
            }
            if metadata.is_dir() {
                return Err(Error::InvalidPath {
                    path: path.to_path_buf(),
                    reason: "output path points to a directory",
                });
            }
            return Err(Error::Conflict {
                message: format!("output path already exists: {}", path.display()),
            });
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(Error::Io(error)),
    }
    reject_sqlite_sidecar_symlinks(path)?;
    Ok(())
}
pub(crate) fn reject_existing_output_sidecars(path: &Path) -> Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = sidecar_path(path, suffix);
        match fs::symlink_metadata(&sidecar) {
            Ok(_) => {
                return Err(Error::Conflict {
                    message: format!(
                        "output SQLite sidecar already exists: {}",
                        sidecar.display()
                    ),
                });
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(Error::Io(error)),
        }
    }
    Ok(())
}
