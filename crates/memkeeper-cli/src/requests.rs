//! Request payload parsers: JSON object -> memkeeper-store request types.

#[allow(clippy::wildcard_imports)]
use super::*;
#[allow(clippy::wildcard_imports)]
use crate::json::*;

#[derive(Debug)]
pub(crate) struct ParsedGetRequest {
    pub(crate) id: String,
    pub(crate) options: GetOptions,
}

#[derive(Debug)]
pub(crate) struct ParsedHistoryRequest {
    pub(crate) id: String,
    pub(crate) options: HistoryOptions,
}

pub(crate) fn ingest_request_from_json(input: &str) -> Result<IngestRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("ingest request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "source_type",
            "source_path",
            "source_uri",
            "source_description",
            "metadata_json",
            "chunks",
            "dry_run",
        ],
    )?;
    Ok(IngestRequest {
        space: optional_string_field(object, "space")?,
        source_type: optional_string_field(object, "source_type")?,
        source_path: optional_string_field(object, "source_path")?,
        source_uri: optional_string_field(object, "source_uri")?,
        source_description: optional_string_field(object, "source_description")?,
        metadata_json: optional_string_field(object, "metadata_json")?,
        chunks: required_string_array_field(object, "chunks")?,
        // Embeddings are computed by the CLI/daemon embed step, never accepted
        // from the wire payload.
        embeddings: None,
        embedding_model_id: None,
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
    })
}

pub(crate) fn document_search_request_from_json(
    input: &str,
) -> Result<DocumentSearchRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("document-search request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "query",
            "space",
            "limit",
            "include_content",
            "snippet_chars",
            "skip_recall_log",
        ],
    )?;
    Ok(DocumentSearchRequest {
        query: required_string_field(object, "query")?,
        space: optional_string_field(object, "space")?,
        limit: optional_usize_field(object, "limit")?.unwrap_or(0),
        include_content: optional_bool_field(object, "include_content")?.unwrap_or(false),
        snippet_chars: optional_usize_field(object, "snippet_chars")?.unwrap_or(240),
        skip_recall_log: optional_bool_field(object, "skip_recall_log")?.unwrap_or(false),
        // Query embedding is computed by the CLI/daemon embed step, never read
        // from the wire payload.
        embedding: None,
    })
}

pub(crate) fn document_get_request_from_json(input: &str) -> Result<DocumentGetRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("document-get request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "source_path",
            "source_episode_id",
            "space",
            "include_content",
            "limit",
        ],
    )?;
    Ok(DocumentGetRequest {
        source_path: optional_string_field(object, "source_path")?,
        source_episode_id: optional_string_field(object, "source_episode_id")?,
        space: optional_string_field(object, "space")?,
        include_content: optional_bool_field(object, "include_content")?.unwrap_or(true),
        limit: optional_usize_field(object, "limit")?.unwrap_or(0),
    })
}

pub(crate) fn promotion_candidates_request_from_json(
    input: &str,
) -> Result<PromotionCandidatesRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("promotion-candidates request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "min_hits",
            "min_distinct_queries",
            "limit",
            "include_content",
            "include_extracted",
        ],
    )?;
    Ok(PromotionCandidatesRequest {
        space: optional_string_field(object, "space")?,
        min_hits: optional_usize_field(object, "min_hits")?.unwrap_or(1),
        min_distinct_queries: optional_usize_field(object, "min_distinct_queries")?.unwrap_or(1),
        limit: optional_usize_field(object, "limit")?.unwrap_or(0),
        include_content: optional_bool_field(object, "include_content")?.unwrap_or(true),
        include_extracted: optional_bool_field(object, "include_extracted")?.unwrap_or(false),
    })
}

pub(crate) fn document_duplicates_request_from_json(
    input: &str,
) -> Result<DocumentDuplicatesRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("document-duplicates request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["space", "limit", "snippet_chars"])?;
    Ok(DocumentDuplicatesRequest {
        space: optional_string_field(object, "space")?,
        limit: optional_usize_field(object, "limit")?.unwrap_or(0),
        snippet_chars: optional_usize_field(object, "snippet_chars")?.unwrap_or(160),
    })
}

pub(crate) fn document_prune_request_from_json(
    input: &str,
) -> Result<DocumentPruneRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("document-prune request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["space", "source_episode_ids", "dry_run"])?;
    Ok(DocumentPruneRequest {
        space: optional_string_field(object, "space")?,
        source_episode_ids: required_string_array_field(object, "source_episode_ids")?,
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
    })
}

pub(crate) fn mark_extracted_request_from_json(
    input: &str,
) -> Result<MarkExtractedRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("mark-extracted request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["space", "source_episode_ids"])?;
    Ok(MarkExtractedRequest {
        space: optional_string_field(object, "space")?,
        source_episode_ids: required_string_array_field(object, "source_episode_ids")?,
    })
}

pub(crate) fn space_create_request_from_json(input: &str) -> Result<SpaceCreateRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("space-create request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "name",
            "display_name",
            "description",
            "default_silo",
            "ontology",
            "config",
            "config_json",
            "if_not_exists",
        ],
    )?;
    Ok(SpaceCreateRequest {
        name: required_string_field(object, "name")?,
        display_name: optional_string_field(object, "display_name")?,
        description: optional_string_field(object, "description")?,
        default_silo: optional_string_field(object, "default_silo")?,
        ontology: optional_string_field(object, "ontology")?,
        config_json: optional_raw_json_alias_field(object, "config", "config_json")?,
        if_not_exists: optional_bool_field(object, "if_not_exists")?.unwrap_or(false),
    })
}

pub(crate) fn silo_list_request_from_json(input: &str) -> Result<SiloListRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("silo-list request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["space"])?;
    Ok(SiloListRequest {
        space: optional_string_field(object, "space")?,
    })
}

pub(crate) fn remember_request_from_json(input: &str) -> Result<RememberRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("remember request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "silo",
            "scope",
            "project",
            "kind",
            "content",
            "summary",
            "tags",
            "entity_key",
            "claim_key",
            "derive_keys",
            "confidence",
            "observed_at",
            "valid_from",
            "valid_to",
            "expires_at",
            "source",
            "metadata_json",
            "pinned",
            "supersedes",
            "contradicts",
            "embedding",
            "embedding_model_id",
            "dry_run",
            "mode",
        ],
    )?;
    let content = required_string_field(object, "content")?;
    let source_object = match object.get("source") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Object(source)) => Some(source),
        Some(_) => {
            return Err(CliError::InvalidRequest(
                "field source must be an object".to_string(),
            ));
        }
    };
    let source = source_object.map(JsonObject::to_json);
    let source_episode_id = source_object
        .map(|source| optional_string_field(source, "source_episode_id"))
        .transpose()?
        .flatten();

    let summary = optional_string_field(object, "summary")?;
    let mut entity_key = optional_string_field(object, "entity_key")?;
    let mut claim_key = optional_string_field(object, "claim_key")?;
    // `derive_keys` fabricates deterministic keys from the memory text for any
    // key the caller left unset, so keyless writes still participate in
    // entity/claim grouping and exact-duplicate supersession. Caller-supplied
    // keys always win.
    if optional_bool_field(object, "derive_keys")?.unwrap_or(false) {
        let (derived_entity, derived_claim) =
            memkeeper_core::derive::keys(&content, summary.as_deref());
        if entity_key.is_none() {
            entity_key = derived_entity;
        }
        if claim_key.is_none() {
            claim_key = derived_claim;
        }
    }

    Ok(RememberRequest {
        space: optional_string_field(object, "space")?,
        silo: optional_string_field(object, "silo")?,
        scope: optional_string_field(object, "scope")?,
        project_key: optional_string_field(object, "project")?,
        kind: optional_string_field(object, "kind")?,
        content,
        summary,
        tags: optional_string_array_field(object, "tags")?.unwrap_or_default(),
        entity_key,
        claim_key,
        confidence: optional_number_field(object, "confidence")?.unwrap_or(1.0),
        observed_at: optional_string_field(object, "observed_at")?,
        valid_from: optional_string_field(object, "valid_from")?,
        valid_to: optional_string_field(object, "valid_to")?,
        expires_at: optional_string_field(object, "expires_at")?,
        source_ref_json: source,
        metadata_json: optional_string_field(object, "metadata_json")?,
        source_episode_id,
        pinned: optional_bool_field(object, "pinned")?.unwrap_or(false),
        supersedes: optional_string_array_field(object, "supersedes")?.unwrap_or_default(),
        contradicts: optional_string_array_field(object, "contradicts")?.unwrap_or_default(),
        embedding: optional_number_array_field(object, "embedding")?,
        embedding_model_id: optional_string_field(object, "embedding_model_id")?,
        // Token embeddings are computed CLI-side (never accepted over the wire).
        token_embedding: None,
        token_embedding_model_id: None,
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
        mode: optional_string_field(object, "mode")?.unwrap_or_else(|| "auto".to_string()),
    })
}

/// Search rerank orchestration options carried in the same request JSON:
/// `(rerank, rerank_candidates)`. Defaults: `(false, 16)`.
pub(crate) fn search_rerank_options_from_json(input: &str) -> Result<(bool, usize), CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("search request must be a JSON object".to_string())
    })?;
    let rerank = optional_bool_field(object, "rerank")?.unwrap_or(false);
    let rerank_candidates = optional_usize_field(object, "rerank_candidates")?.unwrap_or(16);
    Ok((rerank, rerank_candidates))
}

pub(crate) fn search_request_from_json(input: &str) -> Result<SearchRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("search request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "query",
            "filters",
            "limit",
            "offset",
            "snippet_chars",
            "include_content",
            "include_source",
            "semantic_fallback",
            "lexical_fallback",
            "embedding",
            "rerank",
            "rerank_candidates",
        ],
    )?;
    let filters = match object.get("filters") {
        None | Some(JsonValue::Null) => SearchFilters::default(),
        Some(JsonValue::Object(filters)) => search_filters_from_json(filters)?,
        Some(_) => {
            return Err(CliError::InvalidRequest(
                "field filters must be an object".to_string(),
            ));
        }
    };
    Ok(SearchRequest {
        query: required_string_field(object, "query")?,
        filters,
        limit: optional_usize_field(object, "limit")?.unwrap_or(10),
        offset: optional_usize_field(object, "offset")?.unwrap_or(0),
        snippet_chars: optional_usize_field(object, "snippet_chars")?.unwrap_or(240),
        include_content: optional_bool_field(object, "include_content")?.unwrap_or(false),
        include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
        // Default to semantic-primary: an auto-embedded query ranks by semantic
        // relevance, with BM25/FTS as graceful degradation. Pass "disabled" to
        // force pure lexical search.
        semantic_fallback: optional_string_field(object, "semantic_fallback")?
            .unwrap_or_else(|| "fallback".to_string()),
        lexical_fallback: optional_string_field(object, "lexical_fallback")?
            .unwrap_or_else(|| "conservative".to_string()),
        embedding: optional_number_array_field(object, "embedding")?,
        // Token embeddings are computed CLI-side (never accepted over the wire).
        query_token_embedding: None,
        token_model_id: None,
    })
}

pub(crate) fn entity_upsert_request_from_json(
    input: &str,
) -> Result<EntityUpsertRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("entity-upsert request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "entity_key",
            "entity_type",
            "canonical_name",
            "aliases",
            "status",
            "confidence",
            "source_episode_id",
            "metadata",
            "metadata_json",
            "include_source",
        ],
    )?;
    Ok(EntityUpsertRequest {
        space: optional_string_field(object, "space")?,
        entity_key: required_string_field(object, "entity_key")?,
        entity_type: optional_string_field(object, "entity_type")?,
        canonical_name: required_string_field(object, "canonical_name")?,
        aliases: optional_string_array_field(object, "aliases")?.unwrap_or_default(),
        status: optional_string_field(object, "status")?,
        confidence: optional_number_field(object, "confidence")?.unwrap_or(1.0),
        source_episode_id: optional_string_field(object, "source_episode_id")?,
        metadata_json: optional_raw_json_alias_field(object, "metadata", "metadata_json")?,
        include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
    })
}

pub(crate) fn relationship_upsert_request_from_json(
    input: &str,
) -> Result<RelationshipUpsertRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("relationship-upsert request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "subject_entity_id",
            "subject_entity_key",
            "relation_type",
            "object_entity_id",
            "object_entity_key",
            "memory_id",
            "source_episode_id",
            "status",
            "confidence",
            "observed_at",
            "valid_from",
            "valid_to",
            "metadata",
            "metadata_json",
            "include_source",
        ],
    )?;
    Ok(RelationshipUpsertRequest {
        space: optional_string_field(object, "space")?,
        subject_entity_id: optional_string_field(object, "subject_entity_id")?,
        subject_entity_key: optional_string_field(object, "subject_entity_key")?,
        relation_type: required_string_field(object, "relation_type")?,
        object_entity_id: optional_string_field(object, "object_entity_id")?,
        object_entity_key: optional_string_field(object, "object_entity_key")?,
        memory_id: optional_string_field(object, "memory_id")?,
        source_episode_id: optional_string_field(object, "source_episode_id")?,
        status: optional_string_field(object, "status")?,
        confidence: optional_number_field(object, "confidence")?.unwrap_or(1.0),
        observed_at: optional_string_field(object, "observed_at")?,
        valid_from: optional_string_field(object, "valid_from")?,
        valid_to: optional_string_field(object, "valid_to")?,
        metadata_json: optional_raw_json_alias_field(object, "metadata", "metadata_json")?,
        include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
    })
}

pub(crate) fn entity_merge_request_from_json(input: &str) -> Result<EntityMergeRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("entity-merge request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "from_entity_id",
            "from_entity_key",
            "into_entity_id",
            "into_entity_key",
            "dry_run",
            "include_source",
        ],
    )?;
    Ok(EntityMergeRequest {
        space: optional_string_field(object, "space")?,
        from_entity_id: optional_string_field(object, "from_entity_id")?,
        from_entity_key: optional_string_field(object, "from_entity_key")?,
        into_entity_id: optional_string_field(object, "into_entity_id")?,
        into_entity_key: optional_string_field(object, "into_entity_key")?,
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
        include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
    })
}

pub(crate) fn entity_search_request_from_json(
    input: &str,
) -> Result<EntitySearchRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("entity-search request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "query",
            "entity_key",
            "entity_types",
            "statuses",
            "limit",
            "offset",
            "include_source",
        ],
    )?;
    Ok(EntitySearchRequest {
        space: optional_string_field(object, "space")?,
        query: optional_string_field(object, "query")?,
        entity_key: optional_string_field(object, "entity_key")?,
        entity_types: optional_string_array_field(object, "entity_types")?.unwrap_or_default(),
        statuses: optional_string_array_field(object, "statuses")?.unwrap_or_default(),
        limit: optional_usize_field(object, "limit")?.unwrap_or(20),
        offset: optional_usize_field(object, "offset")?.unwrap_or(0),
        include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
    })
}

pub(crate) fn graph_neighbors_request_from_json(
    input: &str,
) -> Result<GraphNeighborsRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("graph-neighbors request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "entity_id",
            "entity_key",
            "depth",
            "relation_types",
            "statuses",
            "max_edges",
            "include_tombstoned",
            "include_source",
        ],
    )?;
    Ok(GraphNeighborsRequest {
        space: optional_string_field(object, "space")?,
        entity_id: optional_string_field(object, "entity_id")?,
        entity_key: optional_string_field(object, "entity_key")?,
        depth: optional_usize_field(object, "depth")?.unwrap_or(1),
        relation_types: optional_string_array_field(object, "relation_types")?.unwrap_or_default(),
        statuses: optional_string_array_field(object, "statuses")?.unwrap_or_default(),
        max_edges: optional_usize_field(object, "max_edges")?.unwrap_or(50),
        include_tombstoned: optional_bool_field(object, "include_tombstoned")?.unwrap_or(false),
        include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
    })
}

pub(crate) fn graph_context_request_from_json(
    input: &str,
) -> Result<GraphContextRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("graph-context request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "entity_id",
            "entity_key",
            "depth",
            "relation_types",
            "statuses",
            "max_edges",
            "max_memories",
            "max_chars",
            "include_tombstoned",
            "include_source",
        ],
    )?;
    Ok(GraphContextRequest {
        space: optional_string_field(object, "space")?,
        entity_id: optional_string_field(object, "entity_id")?,
        entity_key: optional_string_field(object, "entity_key")?,
        depth: optional_usize_field(object, "depth")?.unwrap_or(1),
        relation_types: optional_string_array_field(object, "relation_types")?.unwrap_or_default(),
        statuses: optional_string_array_field(object, "statuses")?.unwrap_or_default(),
        max_edges: optional_usize_field(object, "max_edges")?.unwrap_or(50),
        max_memories: optional_usize_field(object, "max_memories")?.unwrap_or(10),
        max_chars: optional_usize_field(object, "max_chars")?.unwrap_or(4000),
        include_tombstoned: optional_bool_field(object, "include_tombstoned")?.unwrap_or(false),
        include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
    })
}

pub(crate) fn memory_list_request_from_json(input: &str) -> Result<MemoryListRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("memory-list request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "filters",
            "limit",
            "offset",
            "snippet_chars",
            "include_content",
            "include_source",
            "order",
        ],
    )?;
    let filters = match object.get("filters") {
        None | Some(JsonValue::Null) => SearchFilters::default(),
        Some(JsonValue::Object(filters)) => search_filters_from_json(filters)?,
        Some(_) => {
            return Err(CliError::InvalidRequest(
                "field filters must be an object".to_string(),
            ));
        }
    };
    Ok(MemoryListRequest {
        filters,
        limit: optional_usize_field(object, "limit")?.unwrap_or(20),
        offset: optional_usize_field(object, "offset")?.unwrap_or(0),
        snippet_chars: optional_usize_field(object, "snippet_chars")?.unwrap_or(240),
        include_content: optional_bool_field(object, "include_content")?.unwrap_or(false),
        include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
        order: optional_string_field(object, "order")?
            .unwrap_or_else(|| "updated_desc".to_string()),
    })
}

pub(crate) fn search_filters_from_json(object: &JsonObject) -> Result<SearchFilters, CliError> {
    reject_unknown_fields(
        object,
        &[
            "spaces",
            "silos",
            "scopes",
            "projects",
            "kinds",
            "statuses",
            "tags",
            "entity_keys",
            "claim_keys",
        ],
    )?;
    Ok(SearchFilters {
        spaces: optional_string_array_field(object, "spaces")?.unwrap_or_default(),
        silos: optional_string_array_field(object, "silos")?.unwrap_or_default(),
        scopes: optional_string_array_field(object, "scopes")?.unwrap_or_default(),
        projects: optional_string_array_field(object, "projects")?.unwrap_or_default(),
        kinds: optional_string_array_field(object, "kinds")?.unwrap_or_default(),
        statuses: optional_string_array_field(object, "statuses")?.unwrap_or_default(),
        tags: optional_string_array_field(object, "tags")?.unwrap_or_default(),
        entity_keys: optional_string_array_field(object, "entity_keys")?.unwrap_or_default(),
        claim_keys: optional_string_array_field(object, "claim_keys")?.unwrap_or_default(),
    })
}

pub(crate) fn batch_search_request_from_json(input: &str) -> Result<BatchSearchRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("batch-search request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "queries",
            "common_filters",
            "limit",
            "offset",
            "snippet_chars",
            "include_content",
            "include_source",
            "semantic_fallback",
        ],
    )?;
    let common_filters = match object.get("common_filters") {
        None | Some(JsonValue::Null) => SearchFilters::default(),
        Some(JsonValue::Object(filters)) => search_filters_from_json(filters)?,
        Some(_) => {
            return Err(CliError::InvalidRequest(
                "field common_filters must be an object".to_string(),
            ));
        }
    };
    Ok(BatchSearchRequest {
        queries: batch_queries_from_json(object)?,
        common_filters,
        limit: optional_usize_field(object, "limit")?.unwrap_or(10),
        offset: optional_usize_field(object, "offset")?.unwrap_or(0),
        snippet_chars: optional_usize_field(object, "snippet_chars")?.unwrap_or(240),
        include_content: optional_bool_field(object, "include_content")?.unwrap_or(false),
        include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
        semantic_fallback: optional_string_field(object, "semantic_fallback")?
            .unwrap_or_else(|| "disabled".to_string()),
    })
}

pub(crate) fn batch_queries_from_json(
    object: &JsonObject,
) -> Result<Vec<BatchSearchQuery>, CliError> {
    match object.get("queries") {
        Some(JsonValue::Array(values)) => values
            .iter()
            .map(|value| {
                let object = value.as_object().ok_or_else(|| {
                    CliError::InvalidRequest("batch queries must be objects".to_string())
                })?;
                reject_unknown_fields(object, &["name", "query", "limit"])?;
                Ok(BatchSearchQuery {
                    name: optional_string_field(object, "name")?,
                    query: required_string_field(object, "query")?,
                    limit: optional_usize_field(object, "limit")?,
                })
            })
            .collect(),
        Some(_) => Err(CliError::InvalidRequest(
            "field queries must be an array".to_string(),
        )),
        None => Err(CliError::InvalidRequest(
            "missing required array field: queries".to_string(),
        )),
    }
}

/// Parse the optional query-level cosine OR-gate from a pack request payload.
/// `0.0` (the default) preserves legacy per-item rerank-floor behavior.
pub(crate) fn pack_cosine_gate_from_json(input: &str) -> Result<f64, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("pack request must be a JSON object".to_string())
    })?;
    Ok(optional_number_field(object, "cosine_gate")?.unwrap_or(0.0))
}

pub(crate) fn pack_expansion_options_from_json(
    input: &str,
) -> Result<PackExpansionOptions, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("pack request must be a JSON object".to_string())
    })?;
    Ok(PackExpansionOptions {
        query_expansion: optional_bool_field(object, "query_expansion")?.unwrap_or(false),
        thread_expansion: optional_bool_field(object, "thread_expansion")?.unwrap_or(false),
        max_query_variants: optional_usize_field(object, "max_query_variants")?
            .unwrap_or(PackExpansionOptions::default().max_query_variants),
        max_thread_seeds: optional_usize_field(object, "max_thread_seeds")?
            .unwrap_or(PackExpansionOptions::default().max_thread_seeds),
        max_thread_neighbors: optional_usize_field(object, "max_thread_neighbors")?
            .unwrap_or(PackExpansionOptions::default().max_thread_neighbors),
    })
}

pub(crate) fn pack_request_from_json(input: &str) -> Result<PackRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("pack request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "title",
            "queries",
            "filters",
            "max_memories",
            "max_chars",
            "format",
            "min_score",
            "cosine_gate",
            "rerank_candidates",
            "query_embeddings",
            "query_expansion",
            "thread_expansion",
            "max_query_variants",
            "max_thread_seeds",
            "max_thread_neighbors",
        ],
    )?;
    let filters = match object.get("filters") {
        None | Some(JsonValue::Null) => SearchFilters::default(),
        Some(JsonValue::Object(filters)) => search_filters_from_json(filters)?,
        Some(_) => {
            return Err(CliError::InvalidRequest(
                "field filters must be an object".to_string(),
            ));
        }
    };
    let query_embeddings: Option<Vec<Vec<f32>>> = match object.get("query_embeddings") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Array(rows)) => {
            let mut result = Vec::with_capacity(rows.len());
            for row in rows {
                match row {
                    JsonValue::Array(nums) => {
                        let vec = nums
                            .iter()
                            .map(|n| match n {
                                JsonValue::Number(s) => s.parse::<f32>().map_err(|_| {
                                    CliError::InvalidRequest(
                                        "query_embeddings must contain finite numbers".to_string(),
                                    )
                                }),
                                _ => Err(CliError::InvalidRequest(
                                    "query_embeddings inner arrays must contain numbers"
                                        .to_string(),
                                )),
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        result.push(vec);
                    }
                    _ => {
                        return Err(CliError::InvalidRequest(
                            "query_embeddings must be an array of arrays".to_string(),
                        ))
                    }
                }
            }
            Some(result)
        }
        Some(_) => {
            return Err(CliError::InvalidRequest(
                "query_embeddings must be an array".to_string(),
            ))
        }
    };
    Ok(PackRequest {
        title: required_string_field(object, "title")?,
        queries: required_string_array_field(object, "queries")?,
        filters,
        max_memories: optional_usize_field(object, "max_memories")?.unwrap_or(10),
        max_chars: optional_usize_field(object, "max_chars")?.unwrap_or(6_000),
        format: optional_string_field(object, "format")?.unwrap_or_else(|| "markdown".to_string()),
        min_score: optional_number_field(object, "min_score")?.unwrap_or(0.0),
        rerank_candidates: optional_usize_field(object, "rerank_candidates")?.unwrap_or(0),
        query_embeddings,
        // Token embeddings are computed CLI-side (never accepted over the wire).
        query_token_embeddings: None,
        token_model_id: None,
    })
}

pub(crate) fn get_request_from_json(input: &str) -> Result<ParsedGetRequest, CliError> {
    let value = parse_json(input)?;
    let object = value
        .as_object()
        .ok_or_else(|| CliError::InvalidRequest("get request must be a JSON object".to_string()))?;
    reject_unknown_fields(
        object,
        &["id", "include_history", "include_links", "include_source"],
    )?;
    Ok(ParsedGetRequest {
        id: required_string_field(object, "id")?,
        options: GetOptions {
            include_history: optional_bool_field(object, "include_history")?.unwrap_or(false),
            include_links: optional_bool_field(object, "include_links")?.unwrap_or(true),
            include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
        },
    })
}

pub(crate) fn forget_request_from_json(input: &str) -> Result<ForgetRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("forget request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["id", "reason", "mode", "corrected_by", "dry_run"])?;
    Ok(ForgetRequest {
        id: required_string_field(object, "id")?,
        reason: optional_string_field(object, "reason")?,
        mode: optional_string_field(object, "mode")?.unwrap_or_else(|| "tombstone".to_string()),
        corrected_by: optional_string_field(object, "corrected_by")?,
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
    })
}

pub(crate) fn verify_request_from_json(input: &str) -> Result<VerifyRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("verify request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["memory_id", "verified_against", "now"])?;
    Ok(VerifyRequest {
        memory_id: required_string_field(object, "memory_id")?,
        verified_against: optional_string_field(object, "verified_against")?,
        now: optional_string_field(object, "now")?,
    })
}

pub(crate) fn recall_log_request_from_json(input: &str) -> Result<RecallLogRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("recall-log request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &["source", "session_id", "events", "touch_accessed"],
    )?;
    let events_value = object.get("events").ok_or_else(|| {
        CliError::InvalidRequest("recall-log request requires an events array".to_string())
    })?;
    let JsonValue::Array(raw_events) = events_value else {
        return Err(CliError::InvalidRequest(
            "field events must be an array".to_string(),
        ));
    };
    let mut events = Vec::with_capacity(raw_events.len());
    for raw in raw_events {
        let event = raw.as_object().ok_or_else(|| {
            CliError::InvalidRequest("each recall event must be a JSON object".to_string())
        })?;
        reject_unknown_fields(event, &["memory_id", "kind", "query", "rank", "score"])?;
        events.push(RecallEvent {
            memory_id: required_string_field(event, "memory_id")?,
            kind: required_string_field(event, "kind")?,
            query: optional_string_field(event, "query")?,
            rank: optional_usize_field(event, "rank")?,
            score: optional_number_field(event, "score")?,
        });
    }
    Ok(RecallLogRequest {
        source: optional_string_field(object, "source")?,
        session_id: optional_string_field(object, "session_id")?,
        events,
        touch_accessed: optional_bool_field(object, "touch_accessed")?.unwrap_or(true),
    })
}

pub(crate) fn history_request_from_json(input: &str) -> Result<ParsedHistoryRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("history request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["id", "limit", "include_source"])?;
    Ok(ParsedHistoryRequest {
        id: required_string_field(object, "id")?,
        options: HistoryOptions {
            limit: optional_usize_field(object, "limit")?.unwrap_or(50),
            include_source: optional_bool_field(object, "include_source")?.unwrap_or(false),
        },
    })
}

pub(crate) fn export_request_from_json(input: &str) -> Result<ExportRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("export request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["output", "out", "format"])?;
    Ok(ExportRequest {
        output_path: output_path_field(object)?,
        format: optional_string_field(object, "format")?.unwrap_or_else(|| "jsonl".to_string()),
    })
}

pub(crate) fn import_request_from_json(input: &str) -> Result<ImportRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("import request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &["input", "in", "format", "dry_run", "conflict_policy"],
    )?;
    Ok(ImportRequest {
        input_path: input_path_field(object)?,
        format: optional_string_field(object, "format")?.unwrap_or_else(|| "jsonl".to_string()),
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
        conflict_policy: optional_string_field(object, "conflict_policy")?
            .unwrap_or_else(|| "fail_if_exists".to_string()),
    })
}

pub(crate) fn dream_request_from_json(input: &str) -> Result<DreamRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("dream request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "silos",
            "tasks",
            "max_memories",
            "dry_run",
            "include_pinned",
            "promote_threshold",
            "promote_score_floor",
            "promote_rank_cap",
        ],
    )?;
    Ok(DreamRequest {
        space: optional_string_field(object, "space")?,
        silos: optional_string_array_field(object, "silos")?.unwrap_or_default(),
        tasks: optional_string_array_field(object, "tasks")?.unwrap_or_default(),
        max_memories: optional_usize_field(object, "max_memories")?
            .unwrap_or(DEFAULT_DREAM_MAX_MEMORIES),
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
        include_pinned: optional_bool_field(object, "include_pinned")?.unwrap_or(false),
        promote_threshold: optional_usize_field(object, "promote_threshold")?
            .unwrap_or(DEFAULT_PROMOTE_THRESHOLD),
        promote_score_floor: optional_number_field(object, "promote_score_floor")?
            .unwrap_or(DEFAULT_PROMOTE_SCORE_FLOOR),
        promote_rank_cap: optional_usize_field(object, "promote_rank_cap")?
            .unwrap_or(DEFAULT_PROMOTE_RANK_CAP),
    })
}

pub(crate) fn backup_request_from_json(input: &str) -> Result<BackupRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("backup request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["output", "out", "format"])?;
    Ok(BackupRequest {
        output_path: output_path_field(object)?,
        format: optional_string_field(object, "format")?.unwrap_or_else(|| "sqlite".to_string()),
    })
}

pub(crate) fn candidate_submit_request_from_json(
    input: &str,
) -> Result<CandidateSubmitRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("candidate-submit request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &[
            "space",
            "silo",
            "scope",
            "project",
            "kind",
            "content",
            "summary",
            "rationale",
            "tags",
            "entity_key",
            "claim_key",
            "confidence",
            "source_type",
            "source",
            "sensitivity",
            "supersedes",
            "dry_run",
        ],
    )?;
    let source_object = match object.get("source") {
        None | Some(JsonValue::Null) => None,
        Some(JsonValue::Object(source)) => Some(source),
        Some(_) => {
            return Err(CliError::InvalidRequest(
                "field source must be an object".to_string(),
            ));
        }
    };
    Ok(CandidateSubmitRequest {
        space: optional_string_field(object, "space")?,
        silo: optional_string_field(object, "silo")?,
        scope: optional_string_field(object, "scope")?,
        project: optional_string_field(object, "project")?,
        kind: optional_string_field(object, "kind")?,
        content: required_string_field(object, "content")?,
        summary: optional_string_field(object, "summary")?,
        rationale: optional_string_field(object, "rationale")?,
        tags: optional_string_array_field(object, "tags")?.unwrap_or_default(),
        entity_key: optional_string_field(object, "entity_key")?,
        claim_key: optional_string_field(object, "claim_key")?,
        confidence: optional_number_field(object, "confidence")?.unwrap_or(1.0),
        source_type: optional_string_field(object, "source_type")?,
        source_json: source_object.map(JsonObject::to_json),
        sensitivity: optional_string_field(object, "sensitivity")?,
        supersedes: optional_string_array_field(object, "supersedes")?.unwrap_or_default(),
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
    })
}

pub(crate) fn candidate_list_request_from_json(
    input: &str,
) -> Result<CandidateListRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("candidate-list request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["status", "space", "limit", "offset"])?;
    Ok(CandidateListRequest {
        status: optional_string_field(object, "status")?,
        space: optional_string_field(object, "space")?,
        limit: optional_usize_field(object, "limit")?.unwrap_or(50),
        offset: optional_usize_field(object, "offset")?.unwrap_or(0),
    })
}

pub(crate) fn candidate_approve_request_from_json(
    input: &str,
) -> Result<CandidateApproveRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("candidate-approve request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(
        object,
        &["id", "embedding", "embedding_model_id", "dry_run"],
    )?;
    Ok(CandidateApproveRequest {
        id: required_string_field(object, "id")?,
        embedding: optional_number_array_field(object, "embedding")?,
        embedding_model_id: optional_string_field(object, "embedding_model_id")?,
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
    })
}

pub(crate) fn candidate_reject_request_from_json(
    input: &str,
) -> Result<CandidateRejectRequest, CliError> {
    let value = parse_json(input)?;
    let object = value.as_object().ok_or_else(|| {
        CliError::InvalidRequest("candidate-reject request must be a JSON object".to_string())
    })?;
    reject_unknown_fields(object, &["id", "reason", "dry_run"])?;
    Ok(CandidateRejectRequest {
        id: required_string_field(object, "id")?,
        reason: optional_string_field(object, "reason")?,
        dry_run: optional_bool_field(object, "dry_run")?.unwrap_or(false),
    })
}
