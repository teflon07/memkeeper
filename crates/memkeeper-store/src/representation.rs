use crate::{Error, MemoryRepresentationRecord, Result, RetrievalRepresentationInput};
use rusqlite::{params, Connection, OptionalExtension, Transaction};

pub(crate) const CONTEXTUAL_CARD_V1: &str = "contextual-card-v1";
pub(crate) const MAX_RETRIEVAL_REPRESENTATION_CHARS: usize = 512;

pub(crate) fn validate_retrieval_representation(
    input: &RetrievalRepresentationInput,
) -> Result<()> {
    if input.kind != CONTEXTUAL_CARD_V1 {
        return Err(Error::InvalidRequest {
            message: format!("unsupported retrieval representation kind: {}", input.kind),
        });
    }
    if input.text.trim().is_empty() {
        return Err(Error::InvalidRequest {
            message: "retrieval representation text must not be blank".to_string(),
        });
    }
    let chars = input.text.chars().count();
    if chars > MAX_RETRIEVAL_REPRESENTATION_CHARS {
        return Err(Error::InvalidRequest {
            message: format!(
                "retrieval representation text must not exceed {MAX_RETRIEVAL_REPRESENTATION_CHARS} characters"
            ),
        });
    }
    Ok(())
}

/// Selects the one alternate text used by lexical and late-interaction retrieval.
#[must_use]
pub fn retrieval_companion<'a>(
    summary: Option<&'a str>,
    representation: Option<&'a RetrievalRepresentationInput>,
) -> Option<&'a str> {
    representation.map(|value| value.text.as_str()).or(summary)
}

/// Composes the bounded alternate text with canonical content for token retrieval.
#[must_use]
pub fn representation_document(content: &str, companion: Option<&str>) -> String {
    companion.filter(|value| !value.is_empty()).map_or_else(
        || content.to_string(),
        |value| format!("{value}\n\n{content}"),
    )
}

pub(crate) fn insert_representation(
    transaction: &Transaction<'_>,
    version_id: &str,
    input: &RetrievalRepresentationInput,
    now: &str,
) -> Result<MemoryRepresentationRecord> {
    validate_retrieval_representation(input)?;
    let text_sha256 = crate::sha256_hex(input.text.as_bytes());
    transaction.execute(
        "INSERT INTO memory_representations
         (version_id, kind, text, text_sha256, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![version_id, &input.kind, &input.text, &text_sha256, now],
    )?;
    Ok(MemoryRepresentationRecord {
        version_id: version_id.to_string(),
        kind: input.kind.clone(),
        text: input.text.clone(),
        text_sha256,
        created_at: now.to_string(),
    })
}

pub(crate) fn load_representation(
    connection: &Connection,
    version_id: &str,
) -> Result<Option<MemoryRepresentationRecord>> {
    connection
        .query_row(
            "SELECT version_id, kind, text, text_sha256, created_at
             FROM memory_representations WHERE version_id = ?1",
            [version_id],
            |row| {
                Ok(MemoryRepresentationRecord {
                    version_id: row.get(0)?,
                    kind: row.get(1)?,
                    text: row.get(2)?,
                    text_sha256: row.get(3)?,
                    created_at: row.get(4)?,
                })
            },
        )
        .optional()
        .map_err(Into::into)
}
