//! Initialized-store connection lifecycle and safe compatibility inspection.

use std::{
    env,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process,
    sync::atomic::Ordering,
    time::Duration,
};

use rusqlite::{Connection, OpenFlags};

#[cfg(feature = "semantic")]
use rusqlite::auto_extension::{register_auto_extension, RawAutoExtension};

use crate::{
    apply_schema, cleanup_vestigial_long_term_silo, create_new_private_file, create_parent_dirs,
    older_schema_is_memkeeper, required_config_value, space_names, unique_nanos,
    validate_initialized, Error, InitReport, Result, BUSY_TIMEOUT_MS, ID_COUNTER, SCHEMA_VERSION,
};

#[cfg(feature = "semantic")]
static SQLITE_VEC_EXTENSION_REGISTERED: std::sync::OnceLock<std::result::Result<(), String>> =
    std::sync::OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExistingStoreInspection {
    Compatible,
    OlderSchema(i32),
    Unrecognized,
    FutureSchema(i32),
}

/// Initialize a local memkeeper store at `path`.
///
/// The operation creates parent directories, applies the v0.1 schema
/// idempotently, seeds the default workspace space/silos, and validates the
/// resulting schema version.
///
/// # Errors
///
/// Returns an error when the path is not durable, the parent directory cannot
/// be created, an existing non-memkeeper database would be mutated, the existing
/// schema is newer than this binary supports, WAL cannot be enabled, or
/// `SQLite` rejects the schema batch.
pub fn init_store(path: impl AsRef<Path>) -> Result<InitReport> {
    let path = path.as_ref();
    validate_store_path(path)?;
    create_parent_dirs(path)?;
    let created = claim_or_preflight_init_path(path)?;

    reject_sqlite_sidecar_symlinks(path)?;
    register_sqlite_vec_extension()?;
    let connection = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_CREATE,
    )?;
    configure_connection(&connection)?;
    let enabled_journal_mode = enable_wal(&connection)?;
    if enabled_journal_mode != "wal" {
        return Err(Error::WalUnavailable {
            path: path.to_path_buf(),
            journal_mode: enabled_journal_mode,
        });
    }

    apply_schema(&connection)?;
    validate_initialized(path, &connection)?;
    cleanup_vestigial_long_term_silo(&connection)?;

    Ok(InitReport {
        created,
        initialized: true,
        schema_version: user_version(&connection)?,
        protocol_version: required_config_value(path, &connection, "protocol_version")?,
        sqlite_version: sqlite_version(&connection)?,
        journal_mode: journal_mode(&connection)?,
        spaces: space_names(&connection)?,
        default_space: required_config_value(path, &connection, "default_space")?,
    })
}

pub(crate) fn open_initialized_read_fast(path: &Path) -> Result<Connection> {
    validate_store_path(path)?;
    if !path.exists() {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    match inspect_existing_store_immutable(path)? {
        ExistingStoreInspection::Compatible => {}
        ExistingStoreInspection::OlderSchema(actual)
        | ExistingStoreInspection::FutureSchema(actual) => {
            return Err(Error::SchemaMismatch {
                expected: SCHEMA_VERSION,
                actual,
            });
        }
        ExistingStoreInspection::Unrecognized => match inspect_existing_store(path)? {
            ExistingStoreInspection::Compatible => {}
            ExistingStoreInspection::OlderSchema(actual)
            | ExistingStoreInspection::FutureSchema(actual) => {
                return Err(Error::SchemaMismatch {
                    expected: SCHEMA_VERSION,
                    actual,
                });
            }
            ExistingStoreInspection::Unrecognized => {
                return Err(Error::NotInitialized {
                    path: path.to_path_buf(),
                });
            }
        },
    }

    reject_sqlite_sidecar_symlinks(path)?;
    register_sqlite_vec_extension()?;
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    configure_connection(&connection)?;
    validate_initialized(path, &connection)?;
    Ok(connection)
}

pub(crate) fn open_initialized_write(path: &Path) -> Result<Connection> {
    validate_store_path(path)?;
    if !path.exists() {
        return Err(Error::NotInitialized {
            path: path.to_path_buf(),
        });
    }

    let needs_migration = match inspect_existing_store_immutable(path)? {
        ExistingStoreInspection::Compatible => false,
        ExistingStoreInspection::OlderSchema(_) => true,
        ExistingStoreInspection::Unrecognized => match inspect_existing_store(path)? {
            ExistingStoreInspection::Compatible => false,
            ExistingStoreInspection::OlderSchema(_) => true,
            ExistingStoreInspection::Unrecognized => {
                return Err(Error::NotInitialized {
                    path: path.to_path_buf(),
                });
            }
            ExistingStoreInspection::FutureSchema(actual) => {
                return Err(Error::SchemaMismatch {
                    expected: SCHEMA_VERSION,
                    actual,
                });
            }
        },
        ExistingStoreInspection::FutureSchema(actual) => {
            return Err(Error::SchemaMismatch {
                expected: SCHEMA_VERSION,
                actual,
            });
        }
    };

    reject_sqlite_sidecar_symlinks(path)?;
    register_sqlite_vec_extension()?;
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_WRITE)?;
    configure_connection(&connection)?;
    if needs_migration {
        apply_schema(&connection)?;
    }
    validate_initialized(path, &connection)?;
    Ok(connection)
}

fn reject_symlink_path(path: &Path) -> Result<()> {
    if fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "path must not be a symlink",
        });
    }
    Ok(())
}

pub(crate) fn reject_sqlite_sidecar_symlinks(path: &Path) -> Result<()> {
    for suffix in ["-wal", "-shm", "-journal"] {
        let sidecar = sidecar_path(path, suffix);
        if fs::symlink_metadata(&sidecar).is_ok_and(|metadata| metadata.file_type().is_symlink()) {
            return Err(Error::InvalidPath {
                path: sidecar,
                reason: "SQLite sidecar path must not be a symlink",
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_store_path(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "path must not be empty",
        });
    }
    let display_path = path.to_string_lossy();
    if path == Path::new(":memory:") || display_path.starts_with("file:") {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason:
                "durable stores must use plain filesystem paths, not SQLite URI or memory paths",
        });
    }
    if let Ok(metadata) = fs::symlink_metadata(path) {
        if metadata.file_type().is_symlink() {
            return Err(Error::InvalidPath {
                path: path.to_path_buf(),
                reason: "store path must not be a symlink",
            });
        }
        if metadata.is_dir() {
            return Err(Error::InvalidPath {
                path: path.to_path_buf(),
                reason: "path points to a directory",
            });
        }
    } else if path.is_dir() {
        return Err(Error::InvalidPath {
            path: path.to_path_buf(),
            reason: "path points to a directory",
        });
    }
    reject_sqlite_sidecar_symlinks(path)?;
    Ok(())
}

fn claim_or_preflight_init_path(path: &Path) -> Result<bool> {
    if !path.exists() {
        match create_new_private_file(path, false) {
            Ok(_) => return Ok(true),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(Error::Io(error)),
        }
    }
    preflight_init_path(path)
}

fn preflight_init_path(path: &Path) -> Result<bool> {
    validate_store_path(path)?;
    if !path.exists() {
        return Ok(true);
    }

    let metadata = fs::metadata(path)?;
    if metadata.len() == 0 {
        return Ok(false);
    }

    match inspect_existing_store(path)? {
        ExistingStoreInspection::Compatible | ExistingStoreInspection::OlderSchema(_) => Ok(false),
        ExistingStoreInspection::Unrecognized => Err(Error::UnsafeExistingDatabase {
            path: path.to_path_buf(),
        }),
        ExistingStoreInspection::FutureSchema(actual) => Err(Error::SchemaMismatch {
            expected: SCHEMA_VERSION,
            actual,
        }),
    }
}

fn inspect_existing_store(path: &Path) -> Result<ExistingStoreInspection> {
    match inspect_existing_store_copy(path) {
        Ok(inspection) => Ok(inspection),
        Err(Error::Database(_)) => Ok(ExistingStoreInspection::Unrecognized),
        Err(error) => Err(error),
    }
}

fn inspect_existing_store_immutable(path: &Path) -> Result<ExistingStoreInspection> {
    match inspect_existing_store_immutable_inner(path) {
        Ok(inspection) => Ok(inspection),
        Err(Error::Database(_)) => Ok(ExistingStoreInspection::Unrecognized),
        Err(error) => Err(error),
    }
}

fn inspect_existing_store_immutable_inner(path: &Path) -> Result<ExistingStoreInspection> {
    let uri = sqlite_immutable_uri(path)?;
    register_sqlite_vec_extension()?;
    let connection = Connection::open_with_flags(
        &uri,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_URI,
    )?;
    configure_connection(&connection)?;
    inspect_opened_store(path, &connection)
}

fn sqlite_immutable_uri(path: &Path) -> Result<String> {
    let absolute = fs::canonicalize(path)?;
    Ok(format!(
        "file:{}?mode=ro&immutable=1",
        percent_encode_path(&absolute.to_string_lossy())
    ))
}

fn percent_encode_path(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for byte in value.as_bytes() {
        match *byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'/' | b'-' | b'.' | b'_' | b'~' => {
                output.push(char::from(*byte));
            }
            byte => {
                let _ = write!(output, "%{byte:02X}");
            }
        }
    }
    output
}

fn inspect_existing_store_copy(path: &Path) -> Result<ExistingStoreInspection> {
    inspect_on_copy(path, |connection| {
        configure_connection(connection)?;
        inspect_opened_store(path, connection)
    })
}

fn inspect_opened_store(path: &Path, connection: &Connection) -> Result<ExistingStoreInspection> {
    let actual = user_version(connection)?;
    if actual > SCHEMA_VERSION {
        return Ok(ExistingStoreInspection::FutureSchema(actual));
    }
    if actual < SCHEMA_VERSION {
        return if older_schema_is_memkeeper(connection, actual)? {
            Ok(ExistingStoreInspection::OlderSchema(actual))
        } else {
            Ok(ExistingStoreInspection::Unrecognized)
        };
    }

    match validate_initialized(path, connection) {
        Ok(()) => Ok(ExistingStoreInspection::Compatible),
        Err(Error::NotInitialized { .. }) => Ok(ExistingStoreInspection::Unrecognized),
        Err(Error::SchemaMismatch { actual, .. }) => {
            Ok(ExistingStoreInspection::FutureSchema(actual))
        }
        Err(error) => Err(error),
    }
}

pub(crate) fn inspect_on_copy<T>(
    path: &Path,
    inspect: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    reject_symlink_path(path)?;
    let copy_path = inspection_copy_path()?;
    fs::copy(path, &copy_path)?;
    copy_sidecar_if_present(path, &copy_path, "-wal")?;
    copy_sidecar_if_present(path, &copy_path, "-shm")?;
    copy_sidecar_if_present(path, &copy_path, "-journal")?;

    register_sqlite_vec_extension()?;
    let result = Connection::open_with_flags(&copy_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .map_err(Error::Database)
        .and_then(|connection| inspect(&connection));

    cleanup_inspection_copy(&copy_path);
    result
}

pub(crate) fn inspection_copy_path() -> Result<PathBuf> {
    for _ in 0..16 {
        let dir = env::temp_dir().join(format!(
            "memkeeper-inspect-{}-{}-{}",
            process::id(),
            unique_nanos(),
            ID_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        match create_private_dir(&dir) {
            Ok(()) => return Ok(dir.join("store.sqlite")),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(Error::Io(error)),
        }
    }
    Err(Error::InvalidPath {
        path: env::temp_dir(),
        reason: "could not create private inspection directory",
    })
}

#[cfg(unix)]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    fs::DirBuilder::new().mode(0o700).create(path)
}

#[cfg(not(unix))]
fn create_private_dir(path: &Path) -> std::io::Result<()> {
    fs::create_dir(path)
}

fn copy_sidecar_if_present(source: &Path, copy: &Path, suffix: &str) -> Result<()> {
    let source_sidecar = sidecar_path(source, suffix);
    match fs::symlink_metadata(&source_sidecar) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(Error::InvalidPath {
                    path: source_sidecar,
                    reason: "SQLite sidecar path must not be a symlink",
                });
            }
            fs::copy(source_sidecar, sidecar_path(copy, suffix))?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(Error::Io(error)),
    }
    Ok(())
}

pub(crate) fn cleanup_inspection_copy(path: &Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(sidecar_path(path, "-wal"));
    let _ = fs::remove_file(sidecar_path(path, "-shm"));
    let _ = fs::remove_file(sidecar_path(path, "-journal"));
    if let Some(parent) = path.parent() {
        let _ = fs::remove_dir(parent);
    }
}

pub(crate) fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

#[cfg(feature = "semantic")]
pub(crate) fn register_sqlite_vec_extension() -> Result<()> {
    // sqlite-vec exposes SQLite's raw extension entry point; rusqlite requires
    // registering that entry point before opening connections that use vec0. The
    // registration is process-global, so do it once to avoid adding duplicate
    // auto-extension hooks on repeated short-lived CLI/store calls.
    match SQLITE_VEC_EXTENSION_REGISTERED.get_or_init(|| {
        let entry: RawAutoExtension =
            unsafe { std::mem::transmute(sqlite_vec::sqlite3_vec_init as *const ()) };
        unsafe { register_auto_extension(entry) }.map_err(|error| error.to_string())
    }) {
        Ok(()) => Ok(()),
        Err(message) => Err(Error::InvalidRequest {
            message: format!("failed to register sqlite-vec extension: {message}"),
        }),
    }
}

#[cfg(not(feature = "semantic"))]
#[allow(clippy::unnecessary_wraps)]
pub(crate) fn register_sqlite_vec_extension() -> Result<()> {
    Ok(())
}

pub(crate) fn configure_connection(connection: &Connection) -> Result<()> {
    connection.busy_timeout(Duration::from_millis(BUSY_TIMEOUT_MS))?;
    connection.execute_batch("PRAGMA foreign_keys = ON;")?;
    Ok(())
}

pub(crate) fn enable_wal(connection: &Connection) -> Result<String> {
    connection
        .query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0))
        .map_err(Into::into)
}

pub(crate) fn journal_mode(connection: &Connection) -> Result<String> {
    connection
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .map_err(Into::into)
}

pub(crate) fn user_version(connection: &Connection) -> Result<i32> {
    connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(Into::into)
}

pub(crate) fn sqlite_version(connection: &Connection) -> Result<String> {
    connection
        .query_row("SELECT sqlite_version()", [], |row| row.get(0))
        .map_err(Into::into)
}
