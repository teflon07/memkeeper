//! `models` command handlers.

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

use memkeeper_protocol::{Command, PROTOCOL_VERSION};
use memkeeper_store::*;
use std::result::Result;
#[cfg(feature = "embed")]
use memkeeper_store::build_hybrid_rerank_pool_trace_with_evidence_options;

use crate::{CliError, hook::*, json::*, output::*, requests::*, serve::*};
use super::{ArgParser, 
    append_csv_values, maybe_colbert_embed_remember_request,
    maybe_colbert_embed_remember_request_with_requirement, maybe_colbert_embed_search_request,
    maybe_embed_document_search_request, maybe_embed_ingest_request, maybe_embed_remember_request,
    maybe_embed_search_request, parse_bool, parse_f64_arg, parse_json_command_args,
    parse_usize_arg, print_result, SemanticModels,
};

pub(crate) fn run_pull_models(args: &[String]) -> i32 {
    let mut quantized = false;
    let mut dir: Option<PathBuf> = None;
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--quantized" => quantized = true,
            "--dir" => {
                let Some(value) = iter.next() else {
                    eprintln!("pull-models: --dir needs a path");
                    return 2;
                };
                dir = Some(PathBuf::from(value));
            }
            "-h" | "--help" => {
                println!(
                    "Usage: memkeeper pull-models [--quantized] [--dir DIR]\n\n  \
                     --quantized  fetch smaller INT8 models (~0.6GB) instead of fp32 (~2.1GB);\n               \
                     recall drifts from the fp32 baseline, so prefer fp32 for parity.\n  \
                     --dir DIR    install root (default: $MEMKEEPER_MODELS_DIR or ~/.memkeeper/models)"
                );
                return 0;
            }
            other => {
                eprintln!("pull-models: unknown argument: {other}");
                return 2;
            }
        }
    }

    // curl is the one external dependency, same as scripts/fetch-models.sh.
    let curl_ok = process::Command::new("curl")
        .arg("--version")
        .stdout(process::Stdio::null())
        .stderr(process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if !curl_ok {
        eprintln!("pull-models: curl is required but was not found on PATH");
        return 1;
    }

    let dir = dir.unwrap_or_else(|| {
        env::var_os("MEMKEEPER_MODELS_DIR").map_or_else(
            || {
                let home = env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from);
                home.join(".memkeeper").join("models")
            },
            PathBuf::from,
        )
    });

    let onnx = if quantized {
        "model_quantized.onnx"
    } else {
        "model.onnx"
    };
    let hf = "https://huggingface.co";
    // (HF repo, local subdir). The HF repos carry the `-v1` suffix; the local
    // subdirs intentionally omit it to match MEMKEEPER_EMBED_MODEL_DIR/
    // MEMKEEPER_RERANK_MODEL_DIR's documented defaults.
    let models = [
        ("mixedbread-ai/mxbai-embed-large-v1", "mxbai-embed-large"),
        ("mixedbread-ai/mxbai-rerank-base-v1", "mxbai-rerank-base"),
    ];

    for (repo, subdir) in models {
        let dest = dir.join(subdir);
        if let Err(error) = fs::create_dir_all(&dest) {
            eprintln!("pull-models: cannot create {}: {error}", dest.display());
            return 1;
        }
        println!("==> {repo}  ->  {}  ({onnx})", dest.display());
        let downloads = [
            (
                format!("{hf}/{repo}/resolve/main/onnx/{onnx}"),
                dest.join("model.onnx"),
            ),
            (
                format!("{hf}/{repo}/resolve/main/tokenizer.json"),
                dest.join("tokenizer.json"),
            ),
        ];
        for (url, out) in downloads {
            if !curl_download(&url, &out) {
                eprintln!("pull-models: FAILED downloading {url}");
                return 1;
            }
        }
    }

    let embed = dir.join("mxbai-embed-large");
    let rerank = dir.join("mxbai-rerank-base");
    print_models_env_hint(
        &dir.display().to_string(),
        &embed.display().to_string(),
        &rerank.display().to_string(),
    );
    0
}

/// Print the post-download env-setup hint in the host shell's dialect (PowerShell
/// on Windows, POSIX `export` elsewhere) so a copy-paste just works.
fn print_models_env_hint(root: &str, embed_path: &str, rerank_path: &str) {
    if cfg!(windows) {
        println!(
            "\nDone. Models installed under: {root}\n\
             Point the daemon at them — in PowerShell:\n\n  \
             $env:MEMKEEPER_EMBED_MODEL_DIR = \"{embed_path}\"\n  \
             $env:MEMKEEPER_RERANK_MODEL_DIR = \"{rerank_path}\"\n\n  \
             # persist across sessions (new shells):\n  \
             setx MEMKEEPER_EMBED_MODEL_DIR \"{embed_path}\"\n  \
             setx MEMKEEPER_RERANK_MODEL_DIR \"{rerank_path}\"\n\n\
             Then `memkeeper serve` runs with semantics on. To require it (fail closed if\n\
             the models go missing), also set MEMKEEPER_REQUIRE_SEMANTIC=1.",
        );
    } else {
        println!(
            "\nDone. Models installed under: {root}\n\
             Point the daemon at them (add to your shell profile or memkeeper launch env):\n\n  \
             export MEMKEEPER_EMBED_MODEL_DIR=\"{embed_path}\"\n  \
             export MEMKEEPER_RERANK_MODEL_DIR=\"{rerank_path}\"\n\n\
             Then `memkeeper serve` runs with semantics on. To require it (fail closed if\n\
             the models go missing), also set MEMKEEPER_REQUIRE_SEMANTIC=1.",
        );
    }
}

/// `curl -fL --retry 3 --proto =https --tlsv1.2 -o <out> <url>` — fail loud on
/// any HTTP error, follow CDN redirects, https-only. Returns true on success.
fn curl_download(url: &str, out: &Path) -> bool {
    process::Command::new("curl")
        .args([
            "-fL",
            "--retry",
            "3",
            "--proto",
            "=https",
            "--tlsv1.2",
            "-o",
        ])
        .arg(out)
        .arg(url)
        .status()
        .is_ok_and(|status| status.success())
}

pub(crate) fn print_help() {
    println!(
        "memkeeper {PROTOCOL_VERSION} schema {SCHEMA_VERSION}\n\n\
         Commands:\n\
           init --store <path> --json             Initialize a local store.\n\
           doctor [--store <path>] --json         Diagnose binary/config/store readiness.\n\
           stats --store <path> --json            Show deterministic store stats.\n\
           local-usage                              Show local model inference totals.\n\
           space-list --store <path> --json       List configured spaces.\n\
           space-create --store <path> --json '{{}}' Create a space and default silos.\n\
           silo-list --store <path> [--space <name>] --json List silos.\n\
           remember --store <path> --json '{{}}'   Store one explicit memory.\n\
           search --store <path> --json '{{}}'     Search memories with FTS5/BM25.\n\
           entity-upsert --store <path> --json '{{}}' Create/update a graph entity.\n\
           relationship-upsert --store <path> --json '{{}}' Create/update a graph relationship.\n\
           entity-merge --store <path> --json '{{}}' Merge a graph entity into another (relink + tombstone).\n\
           entity-search --store <path> --json '{{}}' Search graph entities.\n\
           graph-neighbors --store <path> --json '{{}}' Traverse graph neighbors.\n\
           graph-context --store <path> --json '{{}}' Build graph-centered memory context.\n\
           memory-list --store <path> --json '{{}}' List recent memories for review.\n\
           batch-search --store <path> --json '{{}}' Run multiple searches.\n\
           ingest --store <path> --json '{{}}'     Ingest a document as isolated, embedded chunks (RAG store).\n\
           document-search --store <path> --json '{{}}' Hybrid search over ingested document chunks.\n\
           document-get --store <path> --json '{{}}' Fetch a document's chunks by path or chunk id.\n\
           document-duplicates --store <path> --json '{{}}' List exact-content duplicate chunks (clusters).\n\
           document-prune --store <path> --json '{{}}' Delete chosen document chunks by id (--dry-run via JSON).\n\
           promotion-candidates --store <path> --json '{{}}' Rank document chunks that earned retrieval traffic.\n\
           mark-extracted --store <path> --json '{{}}' Mark document chunks extracted (promoted).\n\
           candidate-submit --store <path> --json '{{}}' Submit a candidate memory for review.\n\
           candidate-list --store <path> --json '{{}}' List candidate memories (filter by status).\n\
           candidate-approve --store <path> --json '{{}}' Approve a candidate, promoting it to a memory.\n\
           candidate-reject --store <path> --json '{{}}' Reject a candidate memory.\n\
           candidate-quarantine --store <path> --json '{{}}' Quarantine a candidate memory.\n\
           pack --store <path> --json '{{\"title\":<str>,\"queries\":[<str>,...]}}' Build a compact memory pack. Optional: max_memories, max_chars, min_score, filters, format.\n\
           pool-trace --store <path> --json '{{\"title\":<str>,\"queries\":[<str>,...]}}' Diagnose the ID-only pre-rerank pool (requires semantic embedder).\n\
           get --store <path> --id <id> --json    Fetch one memory.\n\
           forget --store <path> --id <id> --json Tombstone one memory.\n\
           history --store <path> --id <id> --json Show memory versions/events.\n\
           export --store <path> --output <path> --json Write logical JSONL export.\n\
           import --store <path> --input <path> --json  Import logical JSONL into a new store.\n\
           dream --store <path> [--task promote|expire|reindex|dedupe|link|graph] [--promote-threshold <N>] [--promote-score-floor <F>] [--promote-rank-cap <N>] [--dry-run|--apply] --json '{{}}' Run bounded maintenance tasks.\n\
           (--dry-run previews; --apply/--commit mutates. forget/import/dream accept both; mutating commands always report dry_run + changed ids.)\n\
           backup --store <path> --output <path> --json Create physical SQLite backup.\n\
           reindex --store <path> [--embed] [--tokens] [--force] Backfill semantic vectors for memories written before the models were present (needs a semantic build). `reindex --help` for details.\n\
           hook retrieve [--store <path>] [--sock <path>]  Claude Code UserPromptSubmit hook client.\n\
           serve --stdio | --socket <path>       Serve newline-delimited JSON requests (stdio or Unix socket).\n\
           mcp [--store <path>]                  Speak MCP (JSON-RPC 2.0) over stdio for any MCP client.\n\
           serve --http [addr] [--store <path>]  Serve the read-only local dashboard (default 127.0.0.1:7777).\n\
           schema [command] [--json]              Show accepted JSON payload fields for a command (or list all).\n\
           schema-status                          Check embedded schema metadata.\n\
           pull-models [--quantized] [--dir <path>] Download the ONNX embed+rerank models (needs curl).\n\
           --help, help                           Show this help.\n\
           --version, version                     Show protocol/schema version.\n\n\
         Per-command request/response shapes: run `memkeeper schema <command>`.\n\
         A `--json` value may be `@<file>` (read the payload from a file) or `-` (read it from stdin),\n\
         which avoids inline-quoting pitfalls (e.g. in Windows PowerShell).\n\
         CLI commands are store-path explicit except read-only doctor diagnostics; hints are user={USER_STORE_PATH_HINT} project={PROJECT_STORE_RELATIVE_PATH}"
    );
}

pub(crate) fn print_schema_status() {
    let schema_mentions_required_objects = schema_mentions_required_objects();
    println!(
        "{{\"protocol_version\":\"{PROTOCOL_VERSION}\",\"schema_version\":{SCHEMA_VERSION},\"user_store_path_hint\":\"{USER_STORE_PATH_HINT}\",\"project_store_relative_path\":\"{PROJECT_STORE_RELATIVE_PATH}\",\"schema_mentions_required_objects\":{schema_mentions_required_objects},\"scaffold_only\":false}}"
    );
}

