//! Read-snapshot transaction helper extracted from `lib.rs` (pure code movement).

use rusqlite::Connection;

use crate::{Error, Result};

/// Run `read` inside a deferred read transaction so all queries see one
/// consistent snapshot, then commit or roll back.
pub(crate) fn with_read_snapshot<T>(
    connection: &Connection,
    read: impl FnOnce(&Connection) -> Result<T>,
) -> Result<T> {
    connection.execute_batch("BEGIN DEFERRED TRANSACTION")?;
    let result = read(connection);
    match result {
        Ok(value) => {
            connection.execute_batch("COMMIT")?;
            Ok(value)
        }
        Err(error) => {
            let _ = connection.execute_batch("ROLLBACK");
            Err(error)
        }
    }
}
