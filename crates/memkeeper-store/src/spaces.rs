//! `spaces` helpers extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::path::Path;

use memkeeper_core::{scope, ALL_SPACES, DEFAULT_DURABLE_SILO, DEFAULT_SPACE};
use rusqlite::{params, Connection, OptionalExtension, Row, Transaction};

use crate::{
    collect_rows, limit_i64, now_timestamp, open_initialized_read_fast, open_initialized_write,
    Error, JsonValidator, Result, SiloListReport, SiloListRequest, SiloRecord, SpaceCreateReport,
    SpaceCreateRequest, SpaceListReport, SpaceRecord, MAX_METADATA_VALUE_CHARS,
    MAX_SILO_LIST_LIMIT, MAX_SOURCE_REF_JSON_CHARS, MAX_SPACE_LIST_LIMIT, MAX_SUMMARY_CHARS,
};

/// List configured spaces with deterministic counts.
///
/// # Errors
///
/// Returns an error when the store is missing, is not initialized, or cannot be queried.
pub fn list_spaces(path: impl AsRef<Path>) -> Result<SpaceListReport> {
    let connection = open_initialized_read_fast(path.as_ref())?;
    let mut spaces = load_spaces(&connection, MAX_SPACE_LIST_LIMIT + 1)?;
    let truncated = spaces.len() > MAX_SPACE_LIST_LIMIT;
    spaces.truncate(MAX_SPACE_LIST_LIMIT);
    Ok(SpaceListReport { spaces, truncated })
}
/// Create a top-level space and seed usable v0.1 silos.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is invalid,
/// the space already exists and `if_not_exists` is false, or `SQLite` rejects the write.
pub fn create_space(
    path: impl AsRef<Path>,
    request: &SpaceCreateRequest,
) -> Result<SpaceCreateReport> {
    let normalized = validate_space_create_request(request)?;
    let mut connection = open_initialized_write(path.as_ref())?;
    let transaction = connection.transaction()?;

    if space_exists(&transaction, &normalized.name)? {
        if normalized.if_not_exists {
            let space =
                load_space(&transaction, &normalized.name)?.ok_or_else(|| Error::NotFound {
                    entity: "space",
                    id: normalized.name.clone(),
                })?;
            transaction.commit()?;
            return Ok(SpaceCreateReport {
                space,
                created: false,
            });
        }
        return Err(Error::Conflict {
            message: format!("space already exists: {}", normalized.name),
        });
    }

    let now = now_timestamp(&transaction)?;
    transaction.execute(
        "INSERT INTO spaces (
            name, display_name, description, default_silo, ontology, config_json, created_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)",
        params![
            &normalized.name,
            normalized.display_name.as_deref(),
            normalized.description.as_deref(),
            &normalized.default_silo,
            normalized.ontology.as_deref(),
            normalized.config_json.as_deref(),
            &now,
        ],
    )?;
    seed_standard_silos(
        &transaction,
        &normalized.name,
        &normalized.default_silo,
        &now,
    )?;
    let space = load_space(&transaction, &normalized.name)?.ok_or_else(|| Error::NotFound {
        entity: "space",
        id: normalized.name.clone(),
    })?;
    transaction.commit()?;
    Ok(SpaceCreateReport {
        space,
        created: true,
    })
}
/// List silos for one configured space.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is invalid,
/// the space does not exist, or `SQLite` rejects the query.
pub fn list_silos(path: impl AsRef<Path>, request: &SiloListRequest) -> Result<SiloListReport> {
    let space = normalize_space_name(request.space.as_deref())?
        .unwrap_or_else(|| DEFAULT_SPACE.to_string());
    let connection = open_initialized_read_fast(path.as_ref())?;
    ensure_space_exists(&connection, &space)?;
    let mut silos = load_silos(&connection, &space, MAX_SILO_LIST_LIMIT + 1)?;
    let truncated = silos.len() > MAX_SILO_LIST_LIMIT;
    silos.truncate(MAX_SILO_LIST_LIMIT);
    Ok(SiloListReport {
        space: space.clone(),
        silos,
        truncated,
    })
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NormalizedSpaceCreateRequest {
    name: String,
    display_name: Option<String>,
    description: Option<String>,
    default_silo: String,
    ontology: Option<String>,
    config_json: Option<String>,
    if_not_exists: bool,
}
pub(crate) fn validate_space_create_request(
    request: &SpaceCreateRequest,
) -> Result<NormalizedSpaceCreateRequest> {
    let name = normalize_required_space_component("space", &request.name)?;
    let default_silo = normalize_space_name(request.default_silo.as_deref())?
        .unwrap_or_else(|| DEFAULT_DURABLE_SILO.to_string());
    let display_name = normalize_optional_human_text(
        "display_name",
        request.display_name.as_deref(),
        MAX_METADATA_VALUE_CHARS,
    )?;
    let description = normalize_optional_human_text(
        "description",
        request.description.as_deref(),
        MAX_SUMMARY_CHARS,
    )?;
    let ontology = normalize_optional_human_text(
        "ontology",
        request.ontology.as_deref(),
        MAX_SOURCE_REF_JSON_CHARS,
    )?;
    if let Some(config_json) = request.config_json.as_deref() {
        if config_json.chars().count() > MAX_SOURCE_REF_JSON_CHARS
            || !JsonValidator::is_object(config_json)
        {
            return Err(Error::InvalidRequest {
                message: "config_json must be a valid JSON object".to_string(),
            });
        }
    }
    Ok(NormalizedSpaceCreateRequest {
        name,
        display_name,
        description,
        default_silo,
        ontology,
        config_json: request.config_json.clone(),
        if_not_exists: request.if_not_exists,
    })
}
pub(crate) fn normalize_required_space_component(name: &str, value: &str) -> Result<String> {
    let trimmed = value.trim();
    if trimmed != value || trimmed.is_empty() || trimmed.chars().count() > MAX_METADATA_VALUE_CHARS
    {
        return Err(Error::InvalidRequest {
            message: format!(
                "{name} must be canonical, non-empty, and at most {MAX_METADATA_VALUE_CHARS} characters"
            ),
        });
    }
    // The `*` sentinel is the read-side "all spaces" marker; it must never be a
    // real space (or silo) name, or it would collide with the union scope. Reject
    // it on every write/create path (this is the single chokepoint both traverse).
    if trimmed == ALL_SPACES {
        return Err(Error::InvalidRequest {
            message: format!("{name} must not be the reserved all-spaces sentinel \"*\""),
        });
    }
    Ok(trimmed.to_string())
}
pub(crate) fn normalize_space_name(value: Option<&str>) -> Result<Option<String>> {
    value
        .map(|value| normalize_required_space_component("space", value))
        .transpose()
}
pub(crate) fn normalize_optional_human_text(
    name: &str,
    value: Option<&str>,
    max_chars: usize,
) -> Result<Option<String>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let trimmed = value.trim();
    if trimmed != value || trimmed.is_empty() || trimmed.chars().count() > max_chars {
        return Err(Error::InvalidRequest {
            message: format!(
                "{name} must be canonical, non-empty, and at most {max_chars} characters"
            ),
        });
    }
    Ok(Some(trimmed.to_string()))
}
pub(crate) fn space_exists(connection: &Connection, space: &str) -> Result<bool> {
    let exists: i64 = connection.query_row(
        "SELECT COUNT(*) FROM spaces WHERE name = ?1",
        [space],
        |row| row.get(0),
    )?;
    Ok(exists != 0)
}
pub(crate) fn ensure_space_exists(connection: &Connection, space: &str) -> Result<()> {
    if space_exists(connection, space)? {
        Ok(())
    } else {
        Err(Error::NotFound {
            entity: "space",
            id: space.to_string(),
        })
    }
}
/// Remove the vestigial `long-term` silo from the legacy default Space, but only
/// when no memory references it. Idempotent; preserves custom Space configuration.
pub(crate) fn cleanup_vestigial_long_term_silo(connection: &Connection) -> Result<()> {
    // Repoint the legacy default Space so default-silo writes don't strand on a
    // missing silo after the delete below. Custom Spaces may use `long-term`.
    connection.execute(
        "UPDATE spaces
         SET default_silo = ?1
         WHERE name = ?2 AND default_silo = 'long-term'",
        params![DEFAULT_DURABLE_SILO, DEFAULT_SPACE],
    )?;
    connection.execute(
        "DELETE FROM silos
         WHERE space_name = ?1
           AND name = 'long-term'
           AND NOT EXISTS (
               SELECT 1 FROM memories m
               WHERE m.space_name = silos.space_name AND m.silo_name = 'long-term')",
        [DEFAULT_SPACE],
    )?;
    Ok(())
}
pub(crate) fn seed_standard_silos(
    transaction: &Transaction<'_>,
    space: &str,
    default_silo: &str,
    now: &str,
) -> Result<()> {
    let standard = [
        (
            "short-term",
            "Working/session memory.",
            "ttl",
            scope::SESSION,
        ),
        (
            DEFAULT_DURABLE_SILO,
            "High-value decisions and commitments.",
            "keep",
            scope::WORKSPACE,
        ),
    ];
    for (name, description, retention_policy, default_scope) in standard {
        transaction.execute(
            "INSERT OR IGNORE INTO silos (
                space_name, name, description, retention_policy, default_scope, created_at, updated_at
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
            params![space, name, description, retention_policy, default_scope, now],
        )?;
    }
    if !standard.iter().any(|(name, _, _, _)| *name == default_silo) {
        transaction.execute(
            "INSERT OR IGNORE INTO silos (
                space_name, name, retention_policy, default_scope, created_at, updated_at
             ) VALUES (?1, ?2, 'keep', ?3, ?4, ?4)",
            params![space, default_silo, scope::WORKSPACE, now],
        )?;
    }
    Ok(())
}
pub(crate) fn load_spaces(connection: &Connection, limit: usize) -> Result<Vec<SpaceRecord>> {
    let mut statement = connection.prepare(
        "SELECT
            s.name,
            s.display_name,
            s.description,
            s.default_silo,
            s.ontology,
            s.config_json,
            s.created_at,
            s.updated_at,
            (SELECT COUNT(*) FROM memories m WHERE m.space_name = s.name),
            (SELECT COUNT(*) FROM memories m WHERE m.space_name = s.name AND m.status = 'active'),
            (SELECT COUNT(*) FROM silos si WHERE si.space_name = s.name)
         FROM spaces s
         ORDER BY s.name
         LIMIT ?1",
    )?;
    let rows = statement.query_map([limit_i64(limit)?], space_record_from_row)?;
    collect_rows(rows)
}
pub(crate) fn load_space(connection: &Connection, space: &str) -> Result<Option<SpaceRecord>> {
    connection
        .query_row(
            "SELECT
                s.name,
                s.display_name,
                s.description,
                s.default_silo,
                s.ontology,
                s.config_json,
                s.created_at,
                s.updated_at,
                (SELECT COUNT(*) FROM memories m WHERE m.space_name = s.name),
                (SELECT COUNT(*) FROM memories m WHERE m.space_name = s.name AND m.status = 'active'),
                (SELECT COUNT(*) FROM silos si WHERE si.space_name = s.name)
             FROM spaces s
             WHERE s.name = ?1",
            [space],
            space_record_from_row,
        )
        .optional()
        .map_err(Into::into)
}
pub(crate) fn space_record_from_row(row: &Row<'_>) -> rusqlite::Result<SpaceRecord> {
    Ok(SpaceRecord {
        name: row.get(0)?,
        display_name: row.get(1)?,
        description: row.get(2)?,
        default_silo: row.get(3)?,
        ontology: row.get(4)?,
        config_json: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
        memory_count: row.get(8)?,
        active_count: row.get(9)?,
        silo_count: row.get(10)?,
    })
}
pub(crate) fn load_silos(
    connection: &Connection,
    space: &str,
    limit: usize,
) -> Result<Vec<SiloRecord>> {
    let mut statement = connection.prepare(
        "SELECT
            si.space_name,
            si.name,
            si.description,
            si.retention_policy,
            si.default_scope,
            si.config_json,
            si.created_at,
            si.updated_at,
            (SELECT COUNT(*) FROM memories m WHERE m.space_name = si.space_name AND m.silo_name = si.name),
            (SELECT COUNT(*) FROM memories m WHERE m.space_name = si.space_name AND m.silo_name = si.name AND m.status = 'active'),
            CASE WHEN sp.default_silo = si.name THEN 1 ELSE 0 END
         FROM silos si
         JOIN spaces sp ON sp.name = si.space_name
         WHERE si.space_name = ?1
         ORDER BY CASE si.name WHEN 'short-term' THEN 0 WHEN 'durable' THEN 1 ELSE 2 END,
                  si.name
         LIMIT ?2",
    )?;
    let rows = statement.query_map(params![space, limit_i64(limit)?], |row| {
        Ok(SiloRecord {
            space: row.get(0)?,
            name: row.get(1)?,
            description: row.get(2)?,
            retention_policy: row.get(3)?,
            default_scope: row.get(4)?,
            config_json: row.get(5)?,
            created_at: row.get(6)?,
            updated_at: row.get(7)?,
            memory_count: row.get(8)?,
            active_count: row.get(9)?,
            is_default: row.get::<_, i64>(10)? == 1,
        })
    })?;
    collect_rows(rows)
}
pub(crate) fn default_silo(connection: &Connection, space: &str) -> Result<Option<String>> {
    connection
        .query_row(
            "SELECT default_silo FROM spaces WHERE name = ?1",
            [space],
            |row| row.get(0),
        )
        .optional()
        .map_err(Into::into)
}
pub(crate) fn ensure_silo_exists(connection: &Connection, space: &str, silo: &str) -> Result<()> {
    let exists: i64 = connection.query_row(
        "SELECT COUNT(*) FROM silos WHERE space_name = ?1 AND name = ?2",
        params![space, silo],
        |row| row.get(0),
    )?;
    if exists == 0 {
        let space_exists: i64 = connection.query_row(
            "SELECT COUNT(*) FROM spaces WHERE name = ?1",
            [space],
            |row| row.get(0),
        )?;
        return if space_exists == 0 {
            Err(Error::NotFound {
                entity: "space",
                id: space.to_string(),
            })
        } else {
            Err(Error::NotFound {
                entity: "silo",
                id: format!("{space}/{silo}"),
            })
        };
    }
    Ok(())
}
