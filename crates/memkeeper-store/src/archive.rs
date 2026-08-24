//! Store import/export/backup extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::fs::OpenOptions;
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::Ordering;
use std::time::Duration;

use rusqlite::{
    backup, params, params_from_iter,
    types::{Value, ValueRef},
    Connection, OpenFlags, OptionalExtension, Row, Transaction,
};

use memkeeper_core::status;

use crate::{
    apply_schema, cleanup_inspection_copy, collect_rows, configure_connection, count,
    create_new_private_file, create_parent_dirs, create_temp_output_file, cleanup_temp_output,
    enable_wal, ensure_memory_candidates, ensure_recall_events, ensure_source_episode_recall_events,
    init_store, inspection_copy_path, is_supported_kind, is_supported_scope, is_supported_status,
    limit_i64, memory_fts_metadata_text, next_id, normalize_imported_schema_metadata, normalized_tags,
    now_timestamp, open_initialized_read_fast, open_initialized_write, publish_temp_output,
    register_sqlite_vec_extension, reject_existing_output_sidecars, reject_output_sidecar_files,
    reject_source_sidecar_output, reject_sqlite_sidecar_symlinks, required_config_value,
    schema_mentions_required_objects, sha256_hex, sha256_path, sha256_text, sidecar_path, table_exists,
    unique_nanos, user_version, validate_export_request, validate_backup_request,
    validate_memory_link_ids, validate_optional_metadata_value, validate_optional_timestamp,
    validate_output_path, validate_remember_request, validate_retrieval_representation,
    validate_initialized, validate_store_path, with_read_snapshot, is_utc_rfc3339_like,
    timestamp_parts_are_valid, EXPORT_TABLES, ExportTableReport,
    ExportTableSpec, Error, BackupReport, BackupRequest, ExportReport, ExportRequest, ImportReport,
    ImportRequest, Result, RetrievalRepresentationInput, Sha256, ID_COUNTER, REMEMBER_MODE_AUTO,
    SCHEMA_VERSION, MAX_CONTENT_CHARS, MAX_IMPORT_JSON_ARRAY_ITEMS, MAX_IMPORT_JSON_DEPTH,
    MAX_IMPORT_JSON_OBJECT_FIELDS, MAX_IMPORT_LINE_BYTES, MAX_METADATA_VALUE_CHARS,
    MAX_SOURCE_REF_JSON_CHARS, MAX_SUMMARY_CHARS, MAX_TAGS, MAX_TIMESTAMP_CHARS,
};

#[cfg(feature = "semantic")]
use crate::rebuild_vector_index;

pub fn export_store(path: impl AsRef<Path>, request: &ExportRequest) -> Result<ExportReport> {
    validate_export_request(request)?;
    let path = path.as_ref();
    let connection = open_initialized_read_fast(path)?;
    reject_source_sidecar_output(path, &request.output_path)?;
    with_read_snapshot(&connection, |connection| {
        export_store_on_connection(connection, request)
    })
}

/// Create a consistent physical `SQLite` backup of an initialized store.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is invalid,
/// the output path already exists, backup work fails, or the backup cannot be
/// validated as a self-contained memkeeper database.
pub fn backup_store(path: impl AsRef<Path>, request: &BackupRequest) -> Result<BackupReport> {
    validate_backup_request(request)?;
    let path = path.as_ref();
    let connection = open_initialized_read_fast(path)?;
    reject_source_sidecar_output(path, &request.output_path)?;
    with_read_snapshot(&connection, |connection| {
        backup_store_on_connection(connection, request)
    })
}

/// Import a deterministic logical JSONL export into a new initialized store.
///
/// # Errors
///
/// Returns an error when the request/archive is invalid, the target store already
/// exists, the schema is unsupported, or import/index rebuild validation fails.
pub fn import_store(path: impl AsRef<Path>, request: &ImportRequest) -> Result<ImportReport> {
    validate_import_request(request)?;
    let path = path.as_ref();
    validate_store_path(path)?;
    if request.dry_run {
        return import_store_dry_run(request);
    }
    import_store_create(path, request)
}

/// Run explicit bounded maintenance/dream tasks in the Rust core.
///
/// # Errors
///
/// Returns an error when the store is missing/incompatible, the request is invalid,
fn export_store_on_connection(
    connection: &Connection,
    request: &ExportRequest,
) -> Result<ExportReport> {
    let (temp_path, file) = create_temp_output_file(&request.output_path)?;
    let export_result = write_export_file(connection, request, file);
    match export_result {
        Ok(report) => {
            publish_temp_output(&temp_path, &request.output_path)?;
            Ok(report)
        }
        Err(error) => {
            cleanup_temp_output(&temp_path);
            Err(error)
        }
    }
}

fn write_export_file(
    connection: &Connection,
    request: &ExportRequest,
    file: File,
) -> Result<ExportReport> {
    let schema_version = user_version(connection)?;
    let mut writer = HashingFileWriter::new(file);
    writer.write_bytes(b"{\"type\":\"header\",\"format\":")?;
    write_json_string_to(&mut writer, "memkeeper.export.v0.1")?;
    writer.write_bytes(b",\"protocol_version\":\"memkeeper.v0.1\",\"schema_version\":")?;
    writer.write_bytes(schema_version.to_string().as_bytes())?;
    writer.write_bytes(b",\"tables\":")?;
    write_string_array_to(&mut writer, EXPORT_TABLES.iter().map(|table| table.name))?;
    writer.write_bytes(b",\"rebuildable_omitted\":[\"memory_fts\",\"memory_fts_public\",\"source_episode_fts\"]}\n")?;

    let mut tables = Vec::with_capacity(EXPORT_TABLES.len());
    let mut row_count = 0_u64;
    for table in EXPORT_TABLES {
        let rows = export_table(connection, &mut writer, table)?;
        row_count = row_count.saturating_add(rows);
        tables.push(ExportTableReport {
            name: table.name.to_string(),
            rows,
        });
    }

    writer.write_bytes(b"{\"type\":\"footer\",\"row_count\":")?;
    writer.write_bytes(row_count.to_string().as_bytes())?;
    writer.write_bytes(b",\"table_counts\":{")?;
    for (index, table) in tables.iter().enumerate() {
        if index > 0 {
            writer.write_bytes(b",")?;
        }
        write_json_string_to(&mut writer, &table.name)?;
        writer.write_bytes(b":")?;
        writer.write_bytes(table.rows.to_string().as_bytes())?;
    }
    writer.write_bytes(b"}}\n")?;
    let (bytes, sha256) = writer.finish()?;

    Ok(ExportReport {
        output_path: request.output_path.clone(),
        format: request.format.clone(),
        schema_version,
        tables,
        row_count,
        bytes,
        sha256,
    })
}

fn export_table(
    connection: &Connection,
    writer: &mut HashingFileWriter,
    table: &ExportTableSpec,
) -> Result<u64> {
    let sql = format!(
        "SELECT {} FROM {} ORDER BY {}",
        table.columns.join(", "),
        table.name,
        table.order_by
    );
    let mut statement = connection.prepare(&sql)?;
    let mut rows = statement.query([])?;
    let mut row_count = 0_u64;

    while let Some(row) = rows.next()? {
        writer.write_bytes(b"{\"type\":\"row\",\"table\":")?;
        write_json_string_to(writer, table.name)?;
        writer.write_bytes(b",\"data\":{")?;
        for (index, column) in table.columns.iter().enumerate() {
            if index > 0 {
                writer.write_bytes(b",")?;
            }
            write_json_string_to(writer, column)?;
            writer.write_bytes(b":")?;
            write_sql_value_json(writer, row.get_ref(index)?)?;
        }
        writer.write_bytes(b"}}\n")?;
        row_count = row_count.saturating_add(1);
    }

    Ok(row_count)
}

fn backup_store_on_connection(
    connection: &Connection,
    request: &BackupRequest,
) -> Result<BackupReport> {
    let (temp_path, file) = create_temp_output_file(&request.output_path)?;
    drop(file);

    let backup_result = backup_to_temp(connection, &temp_path);
    match backup_result {
        Ok(page_count) => {
            let finalize = (|| -> Result<BackupReport> {
                reject_output_sidecar_files(&temp_path)?;
                let bytes = fs::metadata(&temp_path)?.len();
                let sha256 = sha256_path(&temp_path)?;
                let schema_version = user_version(connection)?;
                reject_existing_output_sidecars(&request.output_path)?;
                publish_temp_output(&temp_path, &request.output_path)?;
                if let Err(error) = reject_existing_output_sidecars(&request.output_path) {
                    let _ = fs::remove_file(&request.output_path);
                    return Err(error);
                }
                Ok(BackupReport {
                    output_path: request.output_path.clone(),
                    format: request.format.clone(),
                    schema_version,
                    page_count,
                    bytes,
                    sha256,
                })
            })();
            if finalize.is_err() {
                cleanup_temp_output(&temp_path);
            }
            finalize
        }
        Err(error) => {
            cleanup_temp_output(&temp_path);
            Err(error)
        }
    }
}

fn backup_to_temp(connection: &Connection, temp_path: &Path) -> Result<i64> {
    register_sqlite_vec_extension()?;
    let mut destination =
        Connection::open_with_flags(temp_path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    configure_connection(&destination)?;
    {
        let backup = backup::Backup::new(connection, &mut destination)?;
        backup.run_to_completion(128, Duration::from_millis(0), None::<fn(backup::Progress)>)?;
    }
    let _journal_mode: String =
        destination.query_row("PRAGMA journal_mode = DELETE", [], |row| row.get(0))?;
    validate_initialized(temp_path, &destination)?;
    count(&destination, "PRAGMA page_count")
}

fn import_store_dry_run(request: &ImportRequest) -> Result<ImportReport> {
    let temp_path = inspection_copy_path()?;
    let result = (|| -> Result<ImportReport> {
        let file = create_new_private_file(&temp_path, false)?;
        drop(file);
        import_store_into_path(&temp_path, request, true, false)
    })();
    cleanup_inspection_copy(&temp_path);
    result
}

fn import_store_create(path: &Path, request: &ImportRequest) -> Result<ImportReport> {
    reject_existing_output_sidecars(path)?;
    let (temp_path, file) = create_temp_output_file(path)?;
    drop(file);

    let result = (|| -> Result<ImportReport> {
        let report = import_store_into_path(&temp_path, request, false, false)?;
        reject_output_sidecar_files(&temp_path)?;
        reject_existing_output_sidecars(path)?;
        publish_temp_output(&temp_path, path)?;
        if let Err(error) = reject_existing_output_sidecars(path) {
            cleanup_import_target(path);
            return Err(error);
        }
        if let Err(error) = init_store(path) {
            cleanup_import_target(path);
            return Err(error);
        }
        Ok(report)
    })();

    if result.is_err() {
        cleanup_temp_output(&temp_path);
    }
    result
}

fn import_store_into_path(
    path: &Path,
    request: &ImportRequest,
    dry_run: bool,
    wal: bool,
) -> Result<ImportReport> {
    let open_path = fs::canonicalize(path)?;
    register_sqlite_vec_extension()?;
    let mut connection = Connection::open_with_flags(
        &open_path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NOFOLLOW,
    )?;
    configure_connection(&connection)?;
    if wal {
        let journal_mode = enable_wal(&connection)?;
        if journal_mode != "wal" {
            return Err(Error::WalUnavailable {
                path: path.to_path_buf(),
                journal_mode,
            });
        }
    } else {
        let _journal_mode: String =
            connection.query_row("PRAGMA journal_mode = DELETE", [], |row| row.get(0))?;
    }
    apply_schema(&connection)?;
    let report = import_archive_into_connection(&mut connection, request, dry_run)?;
    validate_initialized(path, &connection)?;
    Ok(report)
}

fn import_archive_into_connection(
    connection: &mut Connection,
    request: &ImportRequest,
    dry_run: bool,
) -> Result<ImportReport> {
    let transaction = connection.transaction()?;
    clear_import_tables(&transaction)?;
    let (parse, fts_memory_rows, fts_source_episode_rows) = {
        let parse = parse_import_file(&transaction, request)?;
        normalize_imported_schema_metadata(&transaction, parse.schema_version)?;
        validate_import_integrity(&transaction)?;
        let (fts_memory_rows, fts_source_episode_rows) = rebuild_fts(&transaction)?;
        #[cfg(feature = "semantic")]
        rebuild_vector_index(&transaction)?;
        (parse, fts_memory_rows, fts_source_episode_rows)
    };
    transaction.commit()?;

    Ok(ImportReport {
        input_path: request.input_path.clone(),
        format: request.format.clone(),
        schema_version: parse.schema_version,
        dry_run,
        tables: parse.tables,
        row_count: parse.row_count,
        bytes: parse.bytes,
        sha256: parse.sha256,
        fts_memory_rows,
        fts_source_episode_rows,
    })
}

fn clear_import_tables(transaction: &Transaction<'_>) -> Result<()> {
    transaction.execute("DELETE FROM memory_fts", [])?;
    transaction.execute("DELETE FROM memory_fts_public", [])?;
    transaction.execute("DELETE FROM source_episode_fts", [])?;
    for table in EXPORT_TABLES.iter().rev() {
        let sql = format!("DELETE FROM {}", table.name);
        transaction.execute(&sql, [])?;
    }
    Ok(())
}

fn validate_import_integrity(transaction: &Transaction<'_>) -> Result<()> {
    let invalid_active_version: Option<String> = transaction
        .query_row(
            "SELECT m.id
             FROM memories m
             LEFT JOIN memory_versions v
                ON v.id = m.active_version_id AND v.memory_id = m.id
             WHERE m.active_version_id IS NULL OR v.id IS NULL
             ORDER BY m.id
             LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(memory_id) = invalid_active_version {
        return Err(Error::InvalidRequest {
            message: format!("import archive has invalid active_version_id for memory {memory_id}"),
        });
    }
    validate_import_spaces_and_silos(transaction)?;
    validate_import_source_refs(transaction)?;
    validate_import_source_episodes(transaction)?;
    validate_import_memory_invariants(transaction)?;
    validate_import_representations(transaction)?;
    validate_import_space_isolation(transaction)?;
    Ok(())
}

fn validate_import_representations(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare(
        "SELECT r.version_id, r.kind, r.text, r.text_sha256, v.id
         FROM memory_representations r
         LEFT JOIN memory_versions v ON v.id = r.version_id
         ORDER BY r.version_id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
        ))
    })?;
    for row in rows {
        let (version_id, kind, text, text_sha256, version_exists) = row?;
        if version_exists.is_none() {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive has representation for missing version {version_id}"
                ),
            });
        }
        validate_retrieval_representation(&RetrievalRepresentationInput {
            kind,
            text: text.clone(),
        })?;
        if sha256_hex(text.as_bytes()) != text_sha256 {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive has representation hash mismatch for version {version_id}"
                ),
            });
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn validate_import_spaces_and_silos(transaction: &Transaction<'_>) -> Result<()> {
    let mut spaces = transaction.prepare(
        "SELECT name, display_name, description, default_silo, ontology, config_json,
                created_at, updated_at
         FROM spaces
         ORDER BY name",
    )?;
    let space_rows = spaces.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, Option<String>>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, Option<String>>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, String>(7)?,
        ))
    })?;
    for row in space_rows {
        let (
            name,
            display_name,
            description,
            default_silo,
            ontology,
            config_json,
            created_at,
            updated_at,
        ) = row?;
        validate_required_import_metadata_value("space name", &name)?;
        validate_optional_import_metadata_value("space display_name", display_name.as_deref())?;
        validate_optional_import_text_value(
            "space description",
            description.as_deref(),
            MAX_SUMMARY_CHARS,
        )?;
        validate_required_import_metadata_value("space default_silo", &default_silo)?;
        validate_optional_import_text_value(
            "space ontology",
            ontology.as_deref(),
            MAX_SOURCE_REF_JSON_CHARS,
        )?;
        validate_optional_import_json_object("space config_json", config_json.as_deref())?;
        validate_required_import_timestamp("space created_at", &created_at)?;
        validate_required_import_timestamp("space updated_at", &updated_at)?;
    }

    let mut silos = transaction.prepare(
        "SELECT space_name, name, description, retention_policy, default_scope, config_json,
                created_at, updated_at
         FROM silos
         ORDER BY space_name, name",
    )?;
    let silo_rows = silos.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
            row.get::<_, Option<String>>(5)?,
            row.get::<_, String>(6)?,
            row.get::<_, String>(7)?,
        ))
    })?;
    for row in silo_rows {
        let (
            space_name,
            name,
            description,
            retention_policy,
            default_scope,
            config_json,
            created_at,
            updated_at,
        ) = row?;
        validate_required_import_metadata_value("silo space", &space_name)?;
        validate_required_import_metadata_value("silo name", &name)?;
        validate_optional_import_text_value(
            "silo description",
            description.as_deref(),
            MAX_SUMMARY_CHARS,
        )?;
        validate_required_import_metadata_value("silo retention_policy", &retention_policy)?;
        validate_required_import_metadata_value("silo default_scope", &default_scope)?;
        if !is_supported_scope(&default_scope) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive has unsupported silo default_scope: {default_scope}"
                ),
            });
        }
        validate_optional_import_json_object("silo config_json", config_json.as_deref())?;
        validate_required_import_timestamp("silo created_at", &created_at)?;
        validate_required_import_timestamp("silo updated_at", &updated_at)?;
    }

    let missing_default_silo: Option<String> = transaction
        .query_row(
            "SELECT s.name
             FROM spaces s
             LEFT JOIN silos si
                ON si.space_name = s.name AND si.name = s.default_silo
             WHERE si.name IS NULL
             ORDER BY s.name
             LIMIT 1",
            [],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(space) = missing_default_silo {
        return Err(Error::InvalidRequest {
            message: format!("import archive has missing default silo for space {space}"),
        });
    }
    Ok(())
}

fn validate_optional_import_text_value(
    name: &str,
    value: Option<&str>,
    max_chars: usize,
) -> Result<()> {
    if let Some(value) = value {
        if value.trim() != value || value.is_empty() || value.chars().count() > max_chars {
            return Err(Error::InvalidRequest {
                message: format!("import archive {name} is not canonical or is too long"),
            });
        }
    }
    Ok(())
}

fn validate_optional_import_json_object(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value) {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid {name}"),
            });
        }
    }
    Ok(())
}

fn validate_required_import_timestamp(name: &str, value: &str) -> Result<()> {
    if value.chars().count() > MAX_TIMESTAMP_CHARS
        || !(is_utc_rfc3339_like(value) || is_sqlite_current_timestamp_like(value))
    {
        return Err(Error::InvalidRequest {
            message: format!("{name} must be a bounded UTC timestamp"),
        });
    }
    Ok(())
}

fn is_sqlite_current_timestamp_like(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 19
        && matches!(bytes.get(4), Some(b'-'))
        && matches!(bytes.get(7), Some(b'-'))
        && matches!(bytes.get(10), Some(b' '))
        && matches!(bytes.get(13), Some(b':'))
        && matches!(bytes.get(16), Some(b':'))
        && timestamp_parts_are_valid(bytes, 0, 5, 8, 11, 14, 17)
}

fn validate_import_source_refs(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare(
        "SELECT id, source_ref_json
         FROM memory_versions
         WHERE source_ref_json IS NOT NULL
         ORDER BY memory_id, version_num, id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    for row in rows {
        let (version_id, source_ref_json) = row?;
        if source_ref_json.chars().count() > MAX_SOURCE_REF_JSON_CHARS {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive source_ref_json exceeds maximum size for version {version_id}"
                ),
            });
        }
        if !JsonValidator::is_object(&source_ref_json) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive has invalid source_ref_json for version {version_id}"
                ),
            });
        }
    }
    Ok(())
}

/// Ingest one document source as isolated, embedded-ready chunks.
///
/// Chunks are written as `source_episodes` rows in a dedicated space (default
/// [`DOCUMENTS_SPACE`]) so they never receive the curated memory tier's
/// supersession/dedup/graph/promotion treatment. Re-ingesting identical content
/// is idempotent: a chunk whose `content_sha256` already exists in the space is
/// skipped rather than duplicated.
///
/// # Errors
/// Returns [`Error::InvalidRequest`] when the request has no chunks or a chunk is
/// empty/too large, or an I/O/storage error when the store cannot be written.
fn validate_import_source_episodes(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare(
        "SELECT id, space_name, source_type, source_path, source_description, content,
            content_sha256, chunk_index, chunk_count, metadata_json
         FROM source_episodes
         ORDER BY id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(ImportSourceEpisodeInvariantRow {
            id: row.get(0)?,
            space_name: row.get(1)?,
            source_type: row.get(2)?,
            source_path: row.get(3)?,
            source_description: row.get(4)?,
            content: row.get(5)?,
            content_sha256: row.get(6)?,
            chunk_index: row.get(7)?,
            chunk_count: row.get(8)?,
            metadata_json: row.get(9)?,
        })
    })?;
    for row in rows {
        validate_import_source_episode_row(&row?)?;
    }
    Ok(())
}

struct ImportSourceEpisodeInvariantRow {
    id: String,
    space_name: String,
    source_type: String,
    source_path: Option<String>,
    source_description: Option<String>,
    content: Option<String>,
    content_sha256: Option<String>,
    chunk_index: i64,
    chunk_count: i64,
    metadata_json: Option<String>,
}

fn validate_import_source_episode_row(row: &ImportSourceEpisodeInvariantRow) -> Result<()> {
    validate_required_import_metadata_value("source episode id", &row.id)?;
    validate_required_import_metadata_value("source episode space", &row.space_name)?;
    validate_required_import_metadata_value("source episode type", &row.source_type)?;
    validate_optional_import_long_text("source episode path", row.source_path.as_deref())?;
    validate_optional_import_long_text(
        "source episode description",
        row.source_description.as_deref(),
    )?;
    if row.chunk_count < 1 || row.chunk_index < 0 || row.chunk_index >= row.chunk_count {
        return Err(Error::InvalidRequest {
            message: format!(
                "import archive has invalid chunk indexes for source {}",
                row.id
            ),
        });
    }
    if row
        .content
        .as_deref()
        .is_some_and(|content| content.chars().count() > MAX_CONTENT_CHARS)
    {
        return Err(Error::InvalidRequest {
            message: format!(
                "import archive source content is too large for source {}",
                row.id
            ),
        });
    }
    if let Some(hash) = row.content_sha256.as_deref() {
        validate_required_import_metadata_value("source episode content_sha256", hash)?;
        if row.content.as_deref().map(sha256_text) != Some(hash.to_string()) {
            return Err(Error::InvalidRequest {
                message: format!(
                    "import archive has content hash mismatch for source {}",
                    row.id
                ),
            });
        }
    }
    validate_import_source_metadata_json(&row.id, row.metadata_json.as_deref())
}

fn validate_optional_import_long_text(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.trim() != value
            || value.is_empty()
            || value.chars().count() > MAX_SOURCE_REF_JSON_CHARS
        {
            return Err(Error::InvalidRequest {
                message: format!("import archive {name} is not canonical or is too long"),
            });
        }
    }
    Ok(())
}

fn validate_import_source_metadata_json(source_id: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value) {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid metadata_json for source {source_id}"),
            });
        }
    }
    Ok(())
}

fn validate_import_memory_invariants(transaction: &Transaction<'_>) -> Result<()> {
    validate_import_memories(transaction)?;
    validate_import_memory_versions(transaction)?;
    validate_import_memory_tags(transaction)?;
    Ok(())
}

fn validate_import_memories(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare(
        "SELECT id, space_name, silo_name, scope, project_key, kind, entity_key,
            claim_key, status, confidence, source_episode_id, valid_from, valid_to,
            observed_at, created_at, updated_at, accessed_at, expires_at, deleted_at,
            metadata_json
         FROM memories
         ORDER BY id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok(ImportMemoryInvariantRow {
            id: row.get(0)?,
            space_name: row.get(1)?,
            silo_name: row.get(2)?,
            scope: row.get(3)?,
            project_key: row.get(4)?,
            kind: row.get(5)?,
            entity_key: row.get(6)?,
            claim_key: row.get(7)?,
            status: row.get(8)?,
            confidence: row.get(9)?,
            source_episode_id: row.get(10)?,
            valid_from: row.get(11)?,
            valid_to: row.get(12)?,
            observed_at: row.get(13)?,
            created_at: row.get(14)?,
            updated_at: row.get(15)?,
            accessed_at: row.get(16)?,
            expires_at: row.get(17)?,
            deleted_at: row.get(18)?,
            metadata_json: row.get(19)?,
        })
    })?;
    for row in rows {
        validate_import_memory_row(&row?)?;
    }
    Ok(())
}

struct ImportMemoryInvariantRow {
    id: String,
    space_name: String,
    silo_name: String,
    scope: String,
    project_key: Option<String>,
    kind: String,
    entity_key: Option<String>,
    claim_key: Option<String>,
    status: String,
    confidence: f64,
    source_episode_id: Option<String>,
    valid_from: Option<String>,
    valid_to: Option<String>,
    observed_at: String,
    created_at: String,
    updated_at: String,
    accessed_at: Option<String>,
    expires_at: Option<String>,
    deleted_at: Option<String>,
    metadata_json: Option<String>,
}

fn validate_import_memory_row(row: &ImportMemoryInvariantRow) -> Result<()> {
    validate_required_import_metadata_value("memory id", &row.id)?;
    validate_required_import_metadata_value("memory space", &row.space_name)?;
    validate_required_import_metadata_value("memory silo", &row.silo_name)?;
    validate_required_import_metadata_value("memory scope", &row.scope)?;
    validate_optional_import_metadata_value("memory project", row.project_key.as_deref())?;
    validate_required_import_metadata_value("memory kind", &row.kind)?;
    validate_optional_import_metadata_value("memory entity_key", row.entity_key.as_deref())?;
    validate_optional_import_metadata_value("memory claim_key", row.claim_key.as_deref())?;
    validate_required_import_metadata_value("memory status", &row.status)?;
    validate_optional_import_metadata_value(
        "memory source_episode_id",
        row.source_episode_id.as_deref(),
    )?;
    if !is_supported_scope(&row.scope) {
        return Err(Error::InvalidRequest {
            message: format!("import archive has unsupported memory scope: {}", row.scope),
        });
    }
    if !is_supported_kind(&row.kind) {
        return Err(Error::InvalidRequest {
            message: format!("import archive has unsupported memory kind: {}", row.kind),
        });
    }
    if !is_supported_status(&row.status) || !(0.0..=1.0).contains(&row.confidence) {
        return Err(Error::InvalidRequest {
            message: format!(
                "import archive has invalid status/confidence for memory {}",
                row.id
            ),
        });
    }
    validate_import_memory_timestamps(row)?;
    validate_import_metadata_json(&row.id, row.metadata_json.as_deref())
}

fn validate_required_import_metadata_value(name: &str, value: &str) -> Result<()> {
    validate_optional_import_metadata_value(name, Some(value))
}

fn validate_optional_import_metadata_value(name: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.trim() != value {
            return Err(Error::InvalidRequest {
                message: format!("import archive {name} must not need trimming"),
            });
        }
        validate_optional_metadata_value(name, Some(value))?;
    }
    Ok(())
}

fn validate_import_memory_timestamps(row: &ImportMemoryInvariantRow) -> Result<()> {
    validate_required_timestamp("memory observed_at", &row.observed_at)?;
    validate_required_timestamp("memory created_at", &row.created_at)?;
    validate_required_timestamp("memory updated_at", &row.updated_at)?;
    validate_optional_timestamp("memory valid_from", row.valid_from.as_deref())?;
    validate_optional_timestamp("memory valid_to", row.valid_to.as_deref())?;
    validate_optional_timestamp("memory accessed_at", row.accessed_at.as_deref())?;
    validate_optional_timestamp("memory expires_at", row.expires_at.as_deref())?;
    validate_optional_timestamp("memory deleted_at", row.deleted_at.as_deref())
}

fn validate_required_timestamp(name: &str, value: &str) -> Result<()> {
    validate_optional_timestamp(name, Some(value))
}

fn validate_import_metadata_json(memory_id: &str, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        if value.chars().count() > MAX_SOURCE_REF_JSON_CHARS || !JsonValidator::is_object(value) {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid metadata_json for memory {memory_id}"),
            });
        }
    }
    Ok(())
}

fn validate_import_memory_versions(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement = transaction.prepare(
        "SELECT id, content, summary, content_sha256
         FROM memory_versions
         ORDER BY memory_id, version_num, id",
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    for row in rows {
        let (id, content, summary, content_sha256) = row?;
        validate_required_import_metadata_value("memory version id", &id)?;
        if content.trim().is_empty() || content.chars().count() > MAX_CONTENT_CHARS {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid content length for version {id}"),
            });
        }
        if summary
            .as_deref()
            .is_some_and(|summary| summary.chars().count() > MAX_SUMMARY_CHARS)
        {
            return Err(Error::InvalidRequest {
                message: format!("import archive has invalid summary length for version {id}"),
            });
        }
        if content_sha256 != sha256_hex(content.as_bytes()) {
            return Err(Error::InvalidRequest {
                message: format!("import archive has content hash mismatch for version {id}"),
            });
        }
    }
    Ok(())
}

fn validate_import_memory_tags(transaction: &Transaction<'_>) -> Result<()> {
    let mut statement =
        transaction.prepare("SELECT memory_id, tag FROM memory_tags ORDER BY memory_id, tag")?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let mut tags_by_memory: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for row in rows {
        let (memory_id, tag) = row?;
        tags_by_memory.entry(memory_id).or_default().push(tag);
    }
    for (memory_id, tags) in tags_by_memory {
        let normalized = normalized_tags(&tags).map_err(|_| Error::InvalidRequest {
            message: format!("import archive has invalid tags for memory {memory_id}"),
        })?;
        if normalized != tags {
            return Err(Error::InvalidRequest {
                message: format!("import archive has non-canonical tags for memory {memory_id}"),
            });
        }
    }
    Ok(())
}

fn validate_import_space_isolation(transaction: &Transaction<'_>) -> Result<()> {
    reject_import_space_mismatch(
        transaction,
        "memory source_episode_id crosses spaces",
        "SELECT m.id
         FROM memories m
         JOIN source_episodes s ON s.id = m.source_episode_id
         WHERE s.space_name != m.space_name
         ORDER BY m.id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "memory version source_episode_id crosses spaces",
        "SELECT v.id
         FROM memory_versions v
         JOIN memories m ON m.id = v.memory_id
         JOIN source_episodes s ON s.id = v.source_episode_id
         WHERE s.space_name != m.space_name
         ORDER BY v.id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "memory link crosses spaces",
        "SELECT l.src_memory_id
         FROM memory_links l
         JOIN memories src ON src.id = l.src_memory_id
         JOIN memories dst ON dst.id = l.dst_memory_id
         WHERE src.space_name != dst.space_name
         ORDER BY l.src_memory_id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "conflict memory crosses spaces",
        "SELECT c.id
         FROM conflicts c
         JOIN memories a ON a.id = c.memory_a_id
         JOIN memories b ON b.id = c.memory_b_id
         WHERE c.space_name != a.space_name OR c.space_name != b.space_name
         ORDER BY c.id
         LIMIT 1",
    )?;
    validate_import_entity_space_isolation(transaction)
}

fn validate_import_entity_space_isolation(transaction: &Transaction<'_>) -> Result<()> {
    reject_import_space_mismatch(
        transaction,
        "entity source_episode_id crosses spaces",
        "SELECT e.id
         FROM entities e
         JOIN source_episodes s ON s.id = e.source_episode_id
         WHERE s.space_name != e.space_name
         ORDER BY e.id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "entity alias source_episode_id crosses spaces",
        "SELECT a.entity_id
         FROM entity_aliases a
         JOIN entities e ON e.id = a.entity_id
         JOIN source_episodes s ON s.id = a.source_episode_id
         WHERE s.space_name != e.space_name
         ORDER BY a.entity_id
         LIMIT 1",
    )?;
    reject_import_space_mismatch(
        transaction,
        "relationship crosses spaces",
        "SELECT r.id
         FROM relationships r
         JOIN entities subject ON subject.id = r.subject_entity_id
         JOIN entities object ON object.id = r.object_entity_id
         LEFT JOIN memories m ON m.id = r.memory_id
         LEFT JOIN source_episodes s ON s.id = r.source_episode_id
         WHERE r.space_name != subject.space_name
            OR r.space_name != object.space_name
            OR (m.id IS NOT NULL AND r.space_name != m.space_name)
            OR (s.id IS NOT NULL AND r.space_name != s.space_name)
         ORDER BY r.id
         LIMIT 1",
    )
}

fn reject_import_space_mismatch(
    transaction: &Transaction<'_>,
    reason: &'static str,
    sql: &str,
) -> Result<()> {
    let invalid_id: Option<String> = transaction
        .query_row(sql, [], |row| row.get(0))
        .optional()?;
    if let Some(id) = invalid_id {
        return Err(Error::InvalidRequest {
            message: format!("import archive {reason}: {id}"),
        });
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct FtsMemoryProjection {
    memory_id: String,
    version_id: String,
    space_name: String,
    silo_name: String,
    status: String,
    kind: String,
    content: String,
    retrieval_text: Option<String>,
    summary: Option<String>,
    tags: String,
    source_text: Option<String>,
    project_key: Option<String>,
    entity_key: Option<String>,
    claim_key: Option<String>,
}

fn fts_memory_projection_from_row(row: &Row<'_>) -> rusqlite::Result<FtsMemoryProjection> {
    Ok(FtsMemoryProjection {
        memory_id: row.get(0)?,
        version_id: row.get(1)?,
        space_name: row.get(2)?,
        silo_name: row.get(3)?,
        status: row.get(4)?,
        kind: row.get(5)?,
        content: row.get(6)?,
        retrieval_text: row.get(7)?,
        summary: row.get(8)?,
        tags: row.get(9)?,
        source_text: row.get(10)?,
        project_key: row.get(11)?,
        entity_key: row.get(12)?,
        claim_key: row.get(13)?,
    })
}

pub(crate) fn rebuild_fts(transaction: &Transaction<'_>) -> Result<(i64, i64)> {
    transaction.execute("DELETE FROM memory_fts", [])?;
    transaction.execute("DELETE FROM memory_fts_public", [])?;
    transaction.execute("DELETE FROM source_episode_fts", [])?;

    let memory_rows = {
        let mut statement = transaction.prepare(
            "SELECT
                m.id,
                v.id,
                m.space_name,
                m.silo_name,
                m.status,
                m.kind,
                v.content,
                COALESCE(r.text, v.summary),
                v.summary,
                COALESCE((
                    SELECT group_concat(tag, ' ')
                    FROM (SELECT tag FROM memory_tags WHERE memory_id = m.id ORDER BY tag)
                ), ''),
                v.source_ref_json,
                m.project_key,
                m.entity_key,
                m.claim_key
             FROM memories m
             JOIN memory_versions v ON v.id = m.active_version_id
             LEFT JOIN memory_representations r ON r.version_id = v.id",
        )?;
        let rows = statement.query_map([], fts_memory_projection_from_row)?;
        collect_rows(rows)?
    };

    for row in memory_rows {
        let metadata_text = memory_fts_metadata_text(
            row.project_key.as_deref(),
            row.entity_key.as_deref(),
            row.claim_key.as_deref(),
            &row.content,
            row.summary.as_deref(),
            &row.tags,
        );
        transaction.execute(
            "INSERT INTO memory_fts (
                memory_id, version_id, space_name, silo_name, status, kind, content, retrieval_text,
                tags, source_text, metadata_text
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                &row.memory_id,
                &row.version_id,
                &row.space_name,
                &row.silo_name,
                &row.status,
                &row.kind,
                &row.content,
                row.retrieval_text.as_deref(),
                &row.tags,
                row.source_text.as_deref(),
                &metadata_text,
            ],
        )?;
        transaction.execute(
            "INSERT INTO memory_fts_public (
                memory_id, version_id, space_name, silo_name, status, kind, content, retrieval_text,
                tags, metadata_text
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                &row.memory_id,
                &row.version_id,
                &row.space_name,
                &row.silo_name,
                &row.status,
                &row.kind,
                &row.content,
                row.retrieval_text.as_deref(),
                &row.tags,
                &metadata_text,
            ],
        )?;
    }
    transaction.execute(
        "INSERT INTO source_episode_fts (
            source_episode_id, space_name, source_type, source_path, source_description,
            content, metadata_text
         )
         SELECT id, space_name, source_type, source_path, source_description,
            content, metadata_json
         FROM source_episodes",
        [],
    )?;
    Ok((
        count(transaction, "SELECT COUNT(*) FROM memory_fts")?,
        count(transaction, "SELECT COUNT(*) FROM source_episode_fts")?,
    ))
}

struct ParsedImportFile {
    schema_version: i32,
    tables: Vec<ExportTableReport>,
    row_count: u64,
    bytes: u64,
    sha256: String,
}

fn parse_import_file(
    transaction: &Transaction<'_>,
    request: &ImportRequest,
) -> Result<ParsedImportFile> {
    let file = open_import_input_file(&request.input_path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut bytes = 0_u64;
    let mut line = Vec::new();
    let mut line_number = 0_usize;
    let mut state = ImportState::new();

    loop {
        let read = read_limited_jsonl_record(&mut reader, &mut line, line_number + 1)?;
        if read == 0 {
            break;
        }
        hasher.update(&line);
        bytes = bytes.saturating_add(read as u64);
        line_number = line_number.saturating_add(1);
        process_import_line(transaction, &mut state, &line, line_number)?;
    }

    if !state.footer_seen {
        return Err(Error::InvalidRequest {
            message: "import file missing footer".to_string(),
        });
    }

    Ok(ParsedImportFile {
        schema_version: state.schema_version,
        tables: state.table_reports(),
        row_count: state.row_count,
        bytes,
        sha256: hasher.finish_hex(),
    })
}

fn read_limited_jsonl_record<R: BufRead>(
    reader: &mut R,
    buffer: &mut Vec<u8>,
    line_number: usize,
) -> Result<usize> {
    buffer.clear();
    let mut total = 0_usize;
    loop {
        let available = reader.fill_buf()?;
        if available.is_empty() {
            return Ok(total);
        }
        let end = available
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(available.len(), |position| position + 1);
        if total.saturating_add(end) > MAX_IMPORT_LINE_BYTES {
            return import_invalid(line_number, "import JSONL record exceeds maximum size");
        }
        buffer.extend_from_slice(&available[..end]);
        reader.consume(end);
        total = total.saturating_add(end);
        if buffer.ends_with(b"\n") {
            return Ok(total);
        }
    }
}

fn process_import_line(
    transaction: &Transaction<'_>,
    state: &mut ImportState,
    raw_line: &[u8],
    line_number: usize,
) -> Result<()> {
    let line = trim_jsonl_newline(raw_line);
    if line.iter().all(u8::is_ascii_whitespace) {
        return import_invalid(line_number, "import JSONL records must not be blank");
    }
    let text = std::str::from_utf8(line).map_err(|_| Error::InvalidRequest {
        message: format!("import JSONL line {line_number} is not valid UTF-8"),
    })?;
    let value = parse_import_json_line(text, line_number)?;
    let object = import_object(&value, line_number, "record")?;
    let record_type = import_string_field(object, "type", line_number)?;
    match record_type {
        "header" => state.accept_header(object, line_number),
        "row" => state.accept_row(transaction, object, line_number),
        "footer" => state.accept_footer(object, line_number),
        _ => import_invalid(line_number, "unknown import record type"),
    }
}

fn trim_jsonl_newline(mut line: &[u8]) -> &[u8] {
    if line.ends_with(b"\n") {
        line = &line[..line.len() - 1];
    }
    if line.ends_with(b"\r") {
        line = &line[..line.len() - 1];
    }
    line
}

struct ImportState {
    header_seen: bool,
    footer_seen: bool,
    current_table_index: usize,
    schema_version: i32,
    tables: Vec<&'static ExportTableSpec>,
    row_count: u64,
    table_counts: Vec<u64>,
}

impl ImportState {
    fn new() -> Self {
        Self {
            header_seen: false,
            footer_seen: false,
            current_table_index: 0,
            schema_version: 0,
            tables: Vec::new(),
            row_count: 0,
            table_counts: Vec::new(),
        }
    }

    fn accept_header(
        &mut self,
        object: &BTreeMap<String, ImportJsonValue>,
        line_number: usize,
    ) -> Result<()> {
        if self.header_seen {
            return import_invalid(line_number, "duplicate import header");
        }
        if self.footer_seen || self.row_count > 0 {
            return import_invalid(line_number, "import header must be the first record");
        }
        import_reject_unknown_fields(
            object,
            &[
                "type",
                "format",
                "protocol_version",
                "schema_version",
                "tables",
                "rebuildable_omitted",
            ],
            line_number,
        )?;
        let format = import_string_field(object, "format", line_number)?;
        if format != "memkeeper.export.v0.1" {
            return import_invalid(line_number, "import format must be memkeeper.export.v0.1");
        }
        let protocol = import_string_field(object, "protocol_version", line_number)?;
        if protocol != "memkeeper.v0.1" {
            return import_invalid(
                line_number,
                "import protocol_version must be memkeeper.v0.1",
            );
        }
        let schema_version = import_i64_field(object, "schema_version", line_number)?;
        let schema_version = i32::try_from(schema_version).unwrap_or_default();
        let tables = import_tables_for_schema(schema_version)?;
        validate_import_table_list(object, &tables, line_number)?;
        validate_rebuildable_omitted(object, line_number)?;
        self.schema_version = schema_version;
        self.table_counts = vec![0; tables.len()];
        self.tables = tables;
        self.header_seen = true;
        Ok(())
    }

    fn accept_row(
        &mut self,
        transaction: &Transaction<'_>,
        object: &BTreeMap<String, ImportJsonValue>,
        line_number: usize,
    ) -> Result<()> {
        self.require_open_rows(line_number)?;
        import_reject_unknown_fields(object, &["type", "table", "data"], line_number)?;
        let table_name = import_string_field(object, "table", line_number)?;
        let table_index = self
            .tables
            .iter()
            .position(|table| table.name == table_name)
            .ok_or_else(|| Error::InvalidRequest {
                message: format!(
                    "import JSONL line {line_number}: unknown export table {table_name}"
                ),
            })?;
        if table_index < self.current_table_index {
            return import_invalid(line_number, "import rows are not in export table order");
        }
        self.current_table_index = table_index;
        let data = import_object_field(object, "data", line_number)?;
        insert_import_row(transaction, self.tables[table_index], data, line_number)?;
        self.table_counts[table_index] = self.table_counts[table_index].saturating_add(1);
        self.row_count = self.row_count.saturating_add(1);
        Ok(())
    }

    fn accept_footer(
        &mut self,
        object: &BTreeMap<String, ImportJsonValue>,
        line_number: usize,
    ) -> Result<()> {
        self.require_open_rows(line_number)?;
        import_reject_unknown_fields(object, &["type", "row_count", "table_counts"], line_number)?;
        let row_count = import_i64_field(object, "row_count", line_number)?;
        if row_count < 0 || u64::try_from(row_count).ok() != Some(self.row_count) {
            return import_invalid(line_number, "import footer row_count mismatch");
        }
        let table_counts = import_object_field(object, "table_counts", line_number)?;
        self.validate_footer_table_counts(table_counts, line_number)?;
        self.footer_seen = true;
        Ok(())
    }

    fn require_open_rows(&self, line_number: usize) -> Result<()> {
        if !self.header_seen {
            return import_invalid(line_number, "import row appeared before header");
        }
        if self.footer_seen {
            return import_invalid(line_number, "import record appeared after footer");
        }
        Ok(())
    }

    fn validate_footer_table_counts(
        &self,
        object: &BTreeMap<String, ImportJsonValue>,
        line_number: usize,
    ) -> Result<()> {
        if object.len() != self.tables.len() {
            return import_invalid(line_number, "import footer table_counts mismatch");
        }
        for (index, table) in self.tables.iter().enumerate() {
            let count = import_i64_field(object, table.name, line_number)?;
            if count < 0 || u64::try_from(count).ok() != Some(self.table_counts[index]) {
                return import_invalid(line_number, "import footer table count mismatch");
            }
        }
        Ok(())
    }

    fn table_reports(&self) -> Vec<ExportTableReport> {
        self.tables
            .iter()
            .zip(&self.table_counts)
            .map(|(table, rows)| ExportTableReport {
                name: table.name.to_string(),
                rows: *rows,
            })
            .collect()
    }
}

fn validate_import_table_list(
    object: &BTreeMap<String, ImportJsonValue>,
    expected_tables: &[&ExportTableSpec],
    line_number: usize,
) -> Result<()> {
    let tables = import_array_field(object, "tables", line_number)?;
    if tables.len() != expected_tables.len() {
        return import_invalid(line_number, "import header table list mismatch");
    }
    for (value, table) in tables.iter().zip(expected_tables) {
        if import_json_string(value) != Some(table.name) {
            return import_invalid(line_number, "import header table list mismatch");
        }
    }
    Ok(())
}

fn import_tables_for_schema(schema_version: i32) -> Result<Vec<&'static ExportTableSpec>> {
    match schema_version {
        SCHEMA_VERSION => Ok(EXPORT_TABLES.iter().collect()),
        5 => Ok(EXPORT_TABLES
            .iter()
            .filter(|table| table.name != "memory_representations")
            .collect()),
        actual => Err(Error::SchemaMismatch {
            expected: SCHEMA_VERSION,
            actual,
        }),
    }
}

fn validate_rebuildable_omitted(
    object: &BTreeMap<String, ImportJsonValue>,
    line_number: usize,
) -> Result<()> {
    let omitted = import_array_field(object, "rebuildable_omitted", line_number)?;
    let expected = ["memory_fts", "memory_fts_public", "source_episode_fts"];
    if omitted.len() != expected.len() {
        return import_invalid(line_number, "import header rebuildable_omitted mismatch");
    }
    for (value, expected) in omitted.iter().zip(expected) {
        if import_json_string(value) != Some(expected) {
            return import_invalid(line_number, "import header rebuildable_omitted mismatch");
        }
    }
    Ok(())
}

fn insert_import_row(
    transaction: &Transaction<'_>,
    table: &ExportTableSpec,
    data: &BTreeMap<String, ImportJsonValue>,
    line_number: usize,
) -> Result<()> {
    validate_import_row_columns(table, data, line_number)?;
    let values = table
        .columns
        .iter()
        .map(|column| {
            import_sql_value(
                table,
                column,
                data.get(*column).expect("validated column"),
                line_number,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let placeholders = (1..=table.columns.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "INSERT INTO {} ({}) VALUES ({})",
        table.name,
        table.columns.join(", "),
        placeholders
    );
    transaction.execute(&sql, params_from_iter(values.iter()))?;
    Ok(())
}

fn validate_import_row_columns(
    table: &ExportTableSpec,
    data: &BTreeMap<String, ImportJsonValue>,
    line_number: usize,
) -> Result<()> {
    if data.len() != table.columns.len() {
        return import_invalid(line_number, "import row column set mismatch");
    }
    for column in table.columns {
        if !data.contains_key(*column) {
            return import_invalid(line_number, "import row missing expected column");
        }
    }
    Ok(())
}

fn import_sql_value(
    table: &ExportTableSpec,
    column: &str,
    value: &ImportJsonValue,
    line_number: usize,
) -> Result<Value> {
    match (import_column_type(table.name, column), value) {
        (_, ImportJsonValue::Null) => Ok(Value::Null),
        (ImportColumnType::Integer, ImportJsonValue::Integer(value)) => Ok(Value::Integer(*value)),
        (ImportColumnType::Real, ImportJsonValue::Integer(value)) => {
            let real = value
                .to_string()
                .parse::<f64>()
                .map_err(|_| Error::InvalidRequest {
                    message: format!(
                        "import JSON line {line_number}: integer for real column is out of range"
                    ),
                })?;
            Ok(Value::Real(real))
        }
        (ImportColumnType::Real, ImportJsonValue::Real(value)) => Ok(Value::Real(*value)),
        (ImportColumnType::Text, ImportJsonValue::String(value)) => Ok(Value::Text(value.clone())),
        (ImportColumnType::Blob, ImportJsonValue::Object(object)) => {
            import_blob_value(object, line_number)
        }
        _ => Err(Error::InvalidRequest {
            message: format!(
                "import JSON line {line_number}: column {}.{} has invalid SQLite storage class",
                table.name, column
            ),
        }),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ImportColumnType {
    Integer,
    Real,
    Text,
    Blob,
}

fn import_column_type(table: &str, column: &str) -> ImportColumnType {
    match (table, column) {
        ("schema_migrations", "version")
        | (
            _,
            "chunk_index" | "chunk_count" | "version_num" | "pinned" | "priority" | "attempts"
            | "max_attempts" | "dimensions",
        ) => ImportColumnType::Integer,
        (_, "confidence") => ImportColumnType::Real,
        ("embeddings", "vector_blob") => ImportColumnType::Blob,
        _ => ImportColumnType::Text,
    }
}

fn import_blob_value(
    object: &BTreeMap<String, ImportJsonValue>,
    line_number: usize,
) -> Result<Value> {
    if object.len() != 1 {
        return import_invalid(
            line_number,
            "import BLOB wrapper must have exactly one field",
        );
    }
    let hex = import_string_field(object, "$memkeeper_blob_hex", line_number)?;
    Ok(Value::Blob(decode_hex_blob(hex, line_number)?))
}

fn decode_hex_blob(value: &str, line_number: usize) -> Result<Vec<u8>> {
    if value.len() % 2 != 0 {
        return import_invalid(line_number, "import BLOB hex length must be even");
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|chunk| {
            let high = hex_nibble(chunk[0], line_number)?;
            let low = hex_nibble(chunk[1], line_number)?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn hex_nibble(byte: u8, line_number: usize) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => import_invalid(line_number, "import BLOB hex contains non-hex character"),
    }
}

fn validate_import_request(request: &ImportRequest) -> Result<()> {
    if request.format != "jsonl" {
        return Err(Error::InvalidRequest {
            message: "import format must be jsonl in v0.1".to_string(),
        });
    }
    if request.conflict_policy != "fail_if_exists" {
        return Err(Error::InvalidRequest {
            message: "import conflict_policy must be fail_if_exists in v0.1".to_string(),
        });
    }
    Ok(())
}

fn open_import_input_file(path: &Path) -> Result<File> {
    validate_import_input_path_shape(path)?;
    let file = open_import_input_file_inner(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "input path must be a regular file",
        });
    }
    Ok(file)
}

fn validate_import_input_path_shape(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "input path must not be empty",
        });
    }
    let display_path = path.to_string_lossy();
    if path == Path::new(":memory:") || display_path.starts_with("file:") {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "input paths must use plain filesystem paths",
        });
    }
    Ok(())
}

#[cfg(unix)]
fn open_import_input_file_inner(path: &Path) -> Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(unix_import_input_open_flags());
    options.open(path).map_err(Into::into)
}

#[cfg(not(unix))]
fn open_import_input_file_inner(path: &Path) -> Result<File> {
    File::open(path).map_err(Into::into)
}

#[cfg(target_os = "linux")]
const fn unix_import_input_open_flags() -> i32 {
    0o400_000 | 0o4_000
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "openbsd",
    target_os = "netbsd",
    target_os = "dragonfly"
))]
const fn unix_import_input_open_flags() -> i32 {
    0x0100 | 0x0004
}

#[cfg(all(
    unix,
    not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "ios",
        target_os = "freebsd",
        target_os = "openbsd",
        target_os = "netbsd",
        target_os = "dragonfly"
    ))
))]
const fn unix_import_input_open_flags() -> i32 {
    0
}

fn cleanup_import_target(path: &Path) {
    let _ = fs::remove_file(path);
    cleanup_temp_output(path);
}

#[derive(Debug, Clone, PartialEq)]
enum ImportJsonValue {
    Null,
    Bool(bool),
    Integer(i64),
    Real(f64),
    String(String),
    Array(Vec<ImportJsonValue>),
    Object(BTreeMap<String, ImportJsonValue>),
}

impl<'de> serde::Deserialize<'de> for ImportJsonValue {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(ImportJsonValueVisitor)
    }
}

struct ImportJsonValueVisitor;

impl<'de> serde::de::Visitor<'de> for ImportJsonValueVisitor {
    type Value = ImportJsonValue;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a bounded JSON value")
    }

    fn visit_unit<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::Null)
    }

    fn visit_none<E>(self) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::Null)
    }

    fn visit_bool<E>(self, value: bool) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::Integer(value))
    }

    fn visit_u64<E>(self, value: u64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        i64::try_from(value)
            .map(ImportJsonValue::Integer)
            .map_err(|_| E::custom("integer out of range"))
    }

    fn visit_f64<E>(self, value: f64) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if value.is_finite() {
            Ok(ImportJsonValue::Real(value))
        } else {
            Err(E::custom("number must be finite"))
        }
    }

    fn visit_str<E>(self, value: &str) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> std::result::Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(ImportJsonValue::String(value))
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = seq.next_element::<ImportJsonValue>()? {
            if values.len() >= MAX_IMPORT_JSON_ARRAY_ITEMS {
                return Err(serde::de::Error::custom("JSON array has too many items"));
            }
            values.push(value);
        }
        Ok(ImportJsonValue::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut object = BTreeMap::new();
        while let Some(key) = map.next_key::<String>()? {
            if object.len() >= MAX_IMPORT_JSON_OBJECT_FIELDS {
                return Err(serde::de::Error::custom("JSON object has too many fields"));
            }
            let value = map.next_value::<ImportJsonValue>()?;
            if object.insert(key, value).is_some() {
                return Err(serde::de::Error::custom("duplicate field in JSON object"));
            }
        }
        Ok(ImportJsonValue::Object(object))
    }
}

fn parse_import_json_line(text: &str, line_number: usize) -> Result<ImportJsonValue> {
    let value =
        serde_json::from_str::<ImportJsonValue>(text).map_err(|error| Error::InvalidRequest {
            message: format!("import JSON line {line_number}: {error}"),
        })?;
    validate_import_json_depth(&value, 0, line_number)?;
    Ok(value)
}

fn validate_import_json_depth(
    value: &ImportJsonValue,
    depth: usize,
    line_number: usize,
) -> Result<()> {
    if depth > MAX_IMPORT_JSON_DEPTH {
        return import_invalid(line_number, "import JSON record exceeds maximum depth");
    }
    match value {
        ImportJsonValue::Array(values) => {
            for value in values {
                validate_import_json_depth(value, depth + 1, line_number)?;
            }
        }
        ImportJsonValue::Object(object) => {
            for value in object.values() {
                validate_import_json_depth(value, depth + 1, line_number)?;
            }
        }
        ImportJsonValue::Null
        | ImportJsonValue::Bool(_)
        | ImportJsonValue::Integer(_)
        | ImportJsonValue::Real(_)
        | ImportJsonValue::String(_) => {}
    }
    Ok(())
}

fn import_json_object_is_valid(input: &str) -> bool {
    let Ok(value) = serde_json::from_str::<ImportJsonValue>(input) else {
        return false;
    };
    matches!(value, ImportJsonValue::Object(_)) && validate_import_json_depth(&value, 0, 0).is_ok()
}

fn import_object<'a>(
    value: &'a ImportJsonValue,
    line_number: usize,
    label: &str,
) -> Result<&'a BTreeMap<String, ImportJsonValue>> {
    match value {
        ImportJsonValue::Object(object) => Ok(object),
        _ => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: {label} must be an object"),
        }),
    }
}

fn import_object_field<'a>(
    object: &'a BTreeMap<String, ImportJsonValue>,
    key: &str,
    line_number: usize,
) -> Result<&'a BTreeMap<String, ImportJsonValue>> {
    let value = object.get(key).ok_or_else(|| Error::InvalidRequest {
        message: format!("import JSON line {line_number}: missing field {key}"),
    })?;
    import_object(value, line_number, key)
}

fn import_array_field<'a>(
    object: &'a BTreeMap<String, ImportJsonValue>,
    key: &str,
    line_number: usize,
) -> Result<&'a [ImportJsonValue]> {
    match object.get(key) {
        Some(ImportJsonValue::Array(values)) => Ok(values),
        Some(_) => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: field {key} must be an array"),
        }),
        None => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: missing field {key}"),
        }),
    }
}

fn import_string_field<'a>(
    object: &'a BTreeMap<String, ImportJsonValue>,
    key: &str,
    line_number: usize,
) -> Result<&'a str> {
    match object.get(key) {
        Some(ImportJsonValue::String(value)) => Ok(value),
        Some(_) => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: field {key} must be a string"),
        }),
        None => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: missing field {key}"),
        }),
    }
}

fn import_i64_field(
    object: &BTreeMap<String, ImportJsonValue>,
    key: &str,
    line_number: usize,
) -> Result<i64> {
    match object.get(key) {
        Some(ImportJsonValue::Integer(value)) => Ok(*value),
        Some(_) => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: field {key} must be an integer"),
        }),
        None => Err(Error::InvalidRequest {
            message: format!("import JSON line {line_number}: missing field {key}"),
        }),
    }
}

fn import_json_string(value: &ImportJsonValue) -> Option<&str> {
    match value {
        ImportJsonValue::String(value) => Some(value),
        _ => None,
    }
}

fn import_reject_unknown_fields(
    object: &BTreeMap<String, ImportJsonValue>,
    allowed: &[&str],
    line_number: usize,
) -> Result<()> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(Error::InvalidRequest {
                message: format!("import JSON line {line_number}: unknown field {key}"),
            });
        }
    }
    Ok(())
}

fn import_invalid<T>(line_number: usize, message: &str) -> Result<T> {
    Err(Error::InvalidRequest {
        message: format!("import JSON line {line_number}: {message}"),
    })
}

struct HashingFileWriter {
    inner: BufWriter<File>,
    hasher: Sha256,
    bytes: u64,
}

impl HashingFileWriter {
    fn new(file: File) -> Self {
        Self {
            inner: BufWriter::new(file),
            hasher: Sha256::new(),
            bytes: 0,
        }
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        self.inner.write_all(bytes)?;
        self.hasher.update(bytes);
        self.bytes = self.bytes.saturating_add(bytes.len() as u64);
        Ok(())
    }

    fn finish(mut self) -> Result<(u64, String)> {
        self.inner.flush()?;
        self.inner.get_ref().sync_all()?;
        Ok((self.bytes, self.hasher.finish_hex()))
    }
}

fn write_json_string_to(writer: &mut HashingFileWriter, value: &str) -> Result<()> {
    let encoded = serde_json::to_string(value).map_err(|error| Error::InvalidRequest {
        message: format!("could not encode JSON string: {error}"),
    })?;
    writer.write_bytes(encoded.as_bytes())
}

fn write_string_array_to<'a>(
    writer: &mut HashingFileWriter,
    values: impl Iterator<Item = &'a str>,
) -> Result<()> {
    writer.write_bytes(b"[")?;
    for (index, value) in values.enumerate() {
        if index > 0 {
            writer.write_bytes(b",")?;
        }
        write_json_string_to(writer, value)?;
    }
    writer.write_bytes(b"]")
}

fn write_sql_value_json(writer: &mut HashingFileWriter, value: ValueRef<'_>) -> Result<()> {
    match value {
        ValueRef::Null => writer.write_bytes(b"null"),
        ValueRef::Integer(value) => writer.write_bytes(value.to_string().as_bytes()),
        ValueRef::Real(value) => writer.write_bytes(finite_json_number(value).as_bytes()),
        ValueRef::Text(value) => {
            let text = std::str::from_utf8(value).map_err(|_| Error::InvalidRequest {
                message: "database text value is not valid UTF-8".to_string(),
            })?;
            write_json_string_to(writer, text)
        }
        ValueRef::Blob(value) => {
            writer.write_bytes(b"{\"$memkeeper_blob_hex\":\"")?;
            write_hex_bytes(writer, value)?;
            writer.write_bytes(b"\"}")
        }
    }
}

fn finite_json_number(value: f64) -> String {
    if value.is_finite() {
        value.to_string()
    } else {
        "null".to_string()
    }
}

fn write_hex_bytes(writer: &mut HashingFileWriter, bytes: &[u8]) -> Result<()> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        writer.write_bytes(&[HEX[usize::from(byte >> 4)], HEX[usize::from(byte & 0x0f)]])?;
    }
    Ok(())
}

pub(crate) struct JsonValidator;

impl JsonValidator {
    pub(crate) fn is_object(input: &str) -> bool {
        import_json_object_is_valid(input)
    }
}
