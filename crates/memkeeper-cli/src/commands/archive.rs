//! `archive` command handlers.

#![allow(clippy::wildcard_imports)]

use std::{
    env,
    fmt::Write as _,
    fs,
    io::{self, BufRead, Write as IoWrite},
    path::{Path, PathBuf},
    process,
    time::Instant,
};

use memkeeper_protocol::Command;
#[cfg(feature = "embed")]
use memkeeper_store::build_hybrid_rerank_pool_trace_with_evidence_options;
use memkeeper_store::*;
use std::result::Result;

use super::{
    append_csv_values, maybe_colbert_embed_remember_request,
    maybe_colbert_embed_remember_request_with_requirement, maybe_colbert_embed_search_request,
    maybe_embed_document_search_request, maybe_embed_ingest_request, maybe_embed_remember_request,
    maybe_embed_search_request, parse_bool, parse_f64_arg, parse_json_command_args,
    parse_usize_arg, print_result, resolve_store_default, ArgParser, SemanticModels,
};
use crate::{hook::*, json::*, output::*, requests::*, serve::*, CliError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExportArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: ExportRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ImportArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: ImportRequest,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DreamArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: DreamRequest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackupArgs {
    pub(crate) store: PathBuf,
    pub(crate) request: BackupRequest,
}

pub(crate) fn run_export(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Export;
    let result = parse_export_args(args)
        .and_then(|options| {
            export_store(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                report.schema_version,
                &export_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

pub(crate) fn run_import(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Import;
    let result = parse_import_args(args)
        .and_then(|options| {
            import_store(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                report.schema_version,
                &import_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

pub(crate) fn run_dream(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Dream;
    let result = parse_dream_args(args)
        .and_then(|options| {
            dream_store(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                SCHEMA_VERSION,
                &dream_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

pub(crate) fn run_backup(args: &[String]) -> i32 {
    let started = Instant::now();
    let command = Command::Backup;
    let result = parse_backup_args(args)
        .and_then(|options| {
            backup_store(&options.store, &options.request)
                .map(|report| (options.store, report))
                .map_err(Into::into)
        })
        .map(|(path, report)| {
            success_envelope(
                command,
                &path,
                report.schema_version,
                &backup_result_json(&report),
                started,
            )
        });
    print_result(command, started, result)
}

pub(crate) fn run_reindex(args: &[String]) -> i32 {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "Usage: memkeeper reindex [--store <path>] [--embed] [--tokens] [--force] [--provider <local|openai>] [--embed-model <name>] [--base-url <url>] [--dims <n>]\n\n  \
             Backfill derived data for memories already in the store. Needs a build with\n  \
             the 'semantic' (or 'api') feature; a lexical-only binary reports an error.\n\n  \
             --embed   recompute embedding vectors (run once after `pull-models`, so\n            \
             memories stored before the models were present become semantically searchable)\n  \
             --tokens  backfill token counts only (no model required beyond tokenizer)\n  \
             --force   re-run even for entries that already have up-to-date vectors\n  \
             --provider/--embed-model/--base-url/--dims  configure the embedder for this\n\
                        reindex only; API keys stay in MEMKEEPER_EMBED_API_KEY\n  \
             --store   store path (defaults to the usual user/project resolution)"
        );
        return 0;
    }
    let started = Instant::now();
    let command = Command::Reindex;
    let result = parse_reindex_args(args).and_then(|parsed| {
        let (schema, count) = if parsed.tokens {
            token_backfill_store(&parsed.store, parsed.force)?
        } else if parsed.embed {
            apply_reindex_embed_overrides(&parsed);
            reembed_store(&parsed.store)?
        } else {
            reindex_count(&parsed.store)?
        };
        Ok(success_envelope(
            command,
            &parsed.store,
            schema,
            &format!("{{\"reindexed\":{count}}}"),
            started,
        ))
    });
    print_result(command, started, result)
}

pub(crate) struct ReindexArgs {
    pub(crate) store: PathBuf,
    pub(crate) embed: bool,
    pub(crate) tokens: bool,
    pub(crate) force: bool,
    pub(crate) provider: Option<String>,
    pub(crate) embed_model: Option<String>,
    pub(crate) base_url: Option<String>,
    pub(crate) dims: Option<usize>,
}

pub(crate) fn parse_reindex_args(args: &[String]) -> Result<ReindexArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut embed = false;
    let mut tokens = false;
    let mut force = false;
    let mut provider = None;
    let mut embed_model = None;
    let mut base_url = None;
    let mut dims = None;
    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            "--embed" => embed = true,
            "--tokens" => tokens = true,
            "--force" => force = true,
            "--provider" => {
                provider = Some(
                    parser
                        .required_value("--provider")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            "--embed-model" => {
                embed_model = Some(
                    parser
                        .required_value("--embed-model")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            "--base-url" => {
                base_url = Some(
                    parser
                        .required_value("--base-url")?
                        .to_string_lossy()
                        .into_owned(),
                );
            }
            "--dims" => {
                let value = parser.required_value("--dims")?;
                dims = Some(value.to_string_lossy().parse::<usize>().map_err(|_| {
                    CliError::InvalidRequest("--dims must be a positive integer".to_string())
                })?);
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported reindex flag: {unknown}"
                )));
            }
        }
    }
    if dims == Some(0) {
        return Err(CliError::InvalidRequest(
            "--dims must be a positive integer".to_string(),
        ));
    }
    if !embed
        && (provider.is_some() || embed_model.is_some() || base_url.is_some() || dims.is_some())
    {
        return Err(CliError::InvalidRequest(
            "embedding override flags require --embed".to_string(),
        ));
    }
    let store = store.unwrap_or_else(resolve_store_default);
    Ok(ReindexArgs {
        store,
        embed,
        tokens,
        force,
        provider,
        embed_model,
        base_url,
        dims,
    })
}

fn apply_reindex_embed_overrides(args: &ReindexArgs) {
    if let Some(value) = &args.provider {
        std::env::set_var("MEMKEEPER_EMBED_PROVIDER", value);
    }
    if let Some(value) = &args.embed_model {
        std::env::set_var("MEMKEEPER_EMBED_MODEL", value);
    }
    if let Some(value) = &args.base_url {
        std::env::set_var("MEMKEEPER_EMBED_BASE_URL", value);
    }
    if let Some(value) = args.dims {
        std::env::set_var("MEMKEEPER_EMBED_DIMS", value.to_string());
    }
}

#[cfg(feature = "embed")]
fn reindex_count(store: &Path) -> Result<(i32, usize), CliError> {
    let count = memkeeper_store::reindex_vectors(store).map_err(CliError::from)?;
    Ok((SCHEMA_VERSION, count))
}

#[cfg(not(feature = "embed"))]
#[allow(clippy::unnecessary_wraps)]
fn reindex_count(_store: &Path) -> Result<(i32, usize), CliError> {
    Err(CliError::InvalidRequest(
        "reindex requires a build with the 'semantic' or 'api' feature".to_string(),
    ))
}

#[cfg(feature = "embed")]
fn reembed_store(store: &Path) -> Result<(i32, usize), CliError> {
    let mut embedder = memkeeper_embed::embedder_from_env().ok_or_else(|| {
        CliError::InvalidRequest(
            "no embedding provider configured; set MEMKEEPER_EMBED_PROVIDER and model env"
                .to_string(),
        )
    })?;
    let targets = memkeeper_store::collect_reembed_targets(store).map_err(CliError::from)?;
    let model_id = embedder.model_id().to_string();
    let dims = embedder.dims();
    let mut vectors: Vec<(String, String, Vec<f32>)> = Vec::with_capacity(targets.len());
    for chunk in targets.chunks(32) {
        let texts: Vec<&str> = chunk.iter().map(|target| target.content.as_str()).collect();
        let embeddings = embedder
            .embed(&texts)
            .map_err(|error| CliError::InvalidRequest(format!("re-embed failed: {error}")))?;
        for (target, embedding) in chunk.iter().zip(embeddings) {
            vectors.push((
                target.memory_id.clone(),
                target.version_id.clone(),
                embedding,
            ));
        }
    }
    let count =
        memkeeper_store::apply_reembed(store, &model_id, dims, &vectors).map_err(CliError::from)?;
    Ok((SCHEMA_VERSION, count))
}

#[cfg(not(feature = "embed"))]
#[allow(clippy::unnecessary_wraps)]
fn reembed_store(_store: &Path) -> Result<(i32, usize), CliError> {
    Err(CliError::InvalidRequest(
        "reindex --embed requires a build with the 'semantic' or 'api' feature".to_string(),
    ))
}

#[cfg(feature = "embed")]
fn token_backfill_store(store: &Path, force: bool) -> Result<(i32, usize), CliError> {
    let mut model = memkeeper_embed::colbert_from_env().ok_or_else(|| {
        CliError::InvalidRequest(
            "no colbert model configured; set MEMKEEPER_COLBERT_MODEL_DIR (and build with the 'semantic' feature)"
                .to_string(),
        )
    })?;
    let targets =
        memkeeper_store::collect_token_backfill_targets(store, force).map_err(CliError::from)?;
    let model_id = model.model_id().to_string();
    let total = targets.len();
    let mut rows: Vec<(String, Vec<Vec<f32>>)> = Vec::with_capacity(total);
    for chunk in targets.chunks(32) {
        let texts: Vec<&str> = chunk.iter().map(|(_, text)| text.as_str()).collect();
        let encoded = model
            .encode_docs(&texts)
            .map_err(|error| CliError::InvalidRequest(format!("token encode failed: {error}")))?;
        for ((memory_id, _), vecs) in chunk.iter().zip(encoded) {
            rows.push((memory_id.clone(), vecs));
        }
        eprintln!("[memkeeper] token backfill encoded {}/{total}", rows.len());
    }
    let count = memkeeper_store::apply_token_embeddings(store, &model_id, &rows, force)
        .map_err(CliError::from)?;
    Ok((SCHEMA_VERSION, count))
}

#[cfg(not(feature = "embed"))]
#[allow(clippy::unnecessary_wraps)]
fn token_backfill_store(_store: &Path, _force: bool) -> Result<(i32, usize), CliError> {
    Err(CliError::InvalidRequest(
        "reindex --tokens requires a build with the 'semantic' feature".to_string(),
    ))
}

pub(crate) fn parse_export_args(args: &[String]) -> Result<ExportArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut output = None;
    let mut format = "jsonl".to_string();
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--output" | "--out" => output = Some(parser.required_value(arg.as_str())?),
            "--format" => format = parser.required_string("--format")?,
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--output=") => {
                output = Some(PathBuf::from(value.trim_start_matches("--output=")));
            }
            value if value.starts_with("--out=") => {
                output = Some(PathBuf::from(value.trim_start_matches("--out=")));
            }
            value if value.starts_with("--format=") => {
                format = value.trim_start_matches("--format=").to_string();
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported export flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        export_request_from_json(&request_json)?
    } else {
        ExportRequest {
            output_path: output.ok_or_else(|| {
                CliError::InvalidRequest("missing required --output <path>".to_string())
            })?,
            format,
        }
    };

    Ok(ExportArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

pub(crate) fn parse_import_args(args: &[String]) -> Result<ImportArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut input = None;
    let mut format = "jsonl".to_string();
    let mut dry_run = false;
    let mut conflict_policy = "fail_if_exists".to_string();
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--input" | "--in" => input = Some(parser.required_value(arg.as_str())?),
            "--format" => format = parser.required_string("--format")?,
            "--dry-run" => dry_run = true,
            "--commit" | "--apply" => dry_run = false,
            "--conflict-policy" => conflict_policy = parser.required_string("--conflict-policy")?,
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--input=") => {
                input = Some(PathBuf::from(value.trim_start_matches("--input=")));
            }
            value if value.starts_with("--in=") => {
                input = Some(PathBuf::from(value.trim_start_matches("--in=")));
            }
            value if value.starts_with("--format=") => {
                format = value.trim_start_matches("--format=").to_string();
            }
            value if value.starts_with("--dry-run=") => {
                dry_run = parse_bool(value.trim_start_matches("--dry-run="))?;
            }
            value if value.starts_with("--conflict-policy=") => {
                conflict_policy = value.trim_start_matches("--conflict-policy=").to_string();
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported import flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        import_request_from_json(&request_json)?
    } else {
        ImportRequest {
            input_path: input.ok_or_else(|| {
                CliError::InvalidRequest("missing required --input <path>".to_string())
            })?,
            format,
            dry_run,
            conflict_policy,
        }
    };

    Ok(ImportArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

#[allow(clippy::too_many_lines)]
pub(crate) fn parse_dream_args(args: &[String]) -> Result<DreamArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut space = None;
    let mut silos = Vec::new();
    let mut tasks = Vec::new();
    // Track flags as Option so explicitly-passed values can override a --json payload.
    // (None means "not supplied on the command line".)
    let mut max_memories = None;
    let mut dry_run = None;
    let mut include_pinned = None;
    let mut promote_threshold = None;
    let mut promote_score_floor = None;
    let mut promote_rank_cap = None;
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--space" => space = Some(parser.required_string("--space")?),
            "--silo" => append_csv_values(&mut silos, &parser.required_string("--silo")?),
            "--task" => append_csv_values(&mut tasks, &parser.required_string("--task")?),
            "--tasks" => append_csv_values(&mut tasks, &parser.required_string("--tasks")?),
            "--max-memories" => {
                max_memories = Some(parse_usize_arg(
                    "--max-memories",
                    &parser.required_string("--max-memories")?,
                )?);
            }
            "--promote-threshold" => {
                promote_threshold = Some(parse_usize_arg(
                    "--promote-threshold",
                    &parser.required_string("--promote-threshold")?,
                )?);
            }
            "--promote-score-floor" => {
                promote_score_floor = Some(parse_f64_arg(
                    "--promote-score-floor",
                    &parser.required_string("--promote-score-floor")?,
                )?);
            }
            "--promote-rank-cap" => {
                promote_rank_cap = Some(parse_usize_arg(
                    "--promote-rank-cap",
                    &parser.required_string("--promote-rank-cap")?,
                )?);
            }
            "--dry-run" => dry_run = Some(true),
            "--commit" | "--apply" => dry_run = Some(false),
            "--include-pinned" => include_pinned = Some(true),
            "--exclude-pinned" => include_pinned = Some(false),
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--space=") => {
                space = Some(value.trim_start_matches("--space=").to_string());
            }
            value if value.starts_with("--silo=") => {
                append_csv_values(&mut silos, value.trim_start_matches("--silo="));
            }
            value if value.starts_with("--task=") => {
                append_csv_values(&mut tasks, value.trim_start_matches("--task="));
            }
            value if value.starts_with("--tasks=") => {
                append_csv_values(&mut tasks, value.trim_start_matches("--tasks="));
            }
            value if value.starts_with("--max-memories=") => {
                max_memories = Some(parse_usize_arg(
                    "--max-memories",
                    value.trim_start_matches("--max-memories="),
                )?);
            }
            value if value.starts_with("--promote-threshold=") => {
                promote_threshold = Some(parse_usize_arg(
                    "--promote-threshold",
                    value.trim_start_matches("--promote-threshold="),
                )?);
            }
            value if value.starts_with("--promote-score-floor=") => {
                promote_score_floor = Some(parse_f64_arg(
                    "--promote-score-floor",
                    value.trim_start_matches("--promote-score-floor="),
                )?);
            }
            value if value.starts_with("--promote-rank-cap=") => {
                promote_rank_cap = Some(parse_usize_arg(
                    "--promote-rank-cap",
                    value.trim_start_matches("--promote-rank-cap="),
                )?);
            }
            value if value.starts_with("--dry-run=") => {
                dry_run = Some(parse_bool(value.trim_start_matches("--dry-run="))?);
            }
            value if value.starts_with("--include-pinned=") => {
                include_pinned = Some(parse_bool(value.trim_start_matches("--include-pinned="))?);
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported dream flag: {unknown}"
                )));
            }
        }
    }

    // Start from the --json payload when supplied, otherwise from defaults.
    let mut request = if let Some(request_json) = request_json {
        dream_request_from_json(&request_json)?
    } else {
        DreamRequest {
            space: None,
            silos: Vec::new(),
            tasks: Vec::new(),
            max_memories: DEFAULT_DREAM_MAX_MEMORIES,
            dry_run: false,
            include_pinned: false,
            promote_threshold: DEFAULT_PROMOTE_THRESHOLD,
            promote_score_floor: DEFAULT_PROMOTE_SCORE_FLOOR,
            promote_rank_cap: DEFAULT_PROMOTE_RANK_CAP,
        }
    };

    // Explicit CLI flags are the user's direct intent and override the payload.
    // This keeps dry-run trust intact: `--dry-run` always wins, even alongside `--json`.
    if space.is_some() {
        request.space = space;
    }
    if !silos.is_empty() {
        request.silos = silos;
    }
    if !tasks.is_empty() {
        request.tasks = tasks;
    }
    if let Some(value) = max_memories {
        request.max_memories = value;
    }
    if let Some(value) = dry_run {
        request.dry_run = value;
    }
    if let Some(value) = include_pinned {
        request.include_pinned = value;
    }
    if let Some(value) = promote_threshold {
        request.promote_threshold = value;
    }
    if let Some(value) = promote_score_floor {
        request.promote_score_floor = value;
    }
    if let Some(value) = promote_rank_cap {
        request.promote_rank_cap = value;
    }

    Ok(DreamArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}

pub(crate) fn parse_backup_args(args: &[String]) -> Result<BackupArgs, CliError> {
    let mut parser = ArgParser::new(args);
    let mut store = None;
    let mut output = None;
    let mut format = "sqlite".to_string();
    let mut request_json = None;

    while let Some(arg) = parser.next() {
        match arg.as_str() {
            "--store" => store = Some(parser.required_value("--store")?),
            "--output" | "--out" => output = Some(parser.required_value(arg.as_str())?),
            "--format" => format = parser.required_string("--format")?,
            "--json" => {
                if let Some(next) = parser
                    .peek()
                    .filter(|value| value.trim_start().starts_with('{'))
                {
                    request_json = Some(next.to_string());
                    let _ = parser.next();
                }
            }
            "--request-json" | "--request" => {
                request_json = Some(parser.required_string(arg.as_str())?);
            }
            value if value.starts_with("--store=") => {
                store = Some(PathBuf::from(value.trim_start_matches("--store=")));
            }
            value if value.starts_with("--output=") => {
                output = Some(PathBuf::from(value.trim_start_matches("--output=")));
            }
            value if value.starts_with("--out=") => {
                output = Some(PathBuf::from(value.trim_start_matches("--out=")));
            }
            value if value.starts_with("--format=") => {
                format = value.trim_start_matches("--format=").to_string();
            }
            value if value.starts_with("--json=") => {
                request_json = Some(value.trim_start_matches("--json=").to_string());
            }
            unknown => {
                return Err(CliError::InvalidRequest(format!(
                    "unsupported backup flag: {unknown}"
                )));
            }
        }
    }

    let request = if let Some(request_json) = request_json {
        backup_request_from_json(&request_json)?
    } else {
        BackupRequest {
            output_path: output.ok_or_else(|| {
                CliError::InvalidRequest("missing required --output <path>".to_string())
            })?,
            format,
        }
    };

    Ok(BackupArgs {
        store: store.unwrap_or_else(resolve_store_default),
        request,
    })
}
