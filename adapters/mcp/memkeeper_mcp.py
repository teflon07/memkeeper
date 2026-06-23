#!/usr/bin/env python3
"""MCP bridge from any MCP-capable agent (Claude Code, Cursor, ...) to a local memkeeper store.

This exposes source-hidden read tools plus explicit write tools for curated memories
and graph rows. Mutating tools are intentionally narrow: use remember for durable
facts/decisions/preferences/lessons/actions, forget to tombstone a specific memory id,
and entity/relationship upserts for explicit graph projection maintenance.

Configuration is entirely via environment variables with portable defaults under
``~/.memkeeper`` (the directory ``memkeeper init`` creates). No machine-specific paths
are baked in:

    MEMKEEPER_HOME              base dir for the defaults below (default: ~/.memkeeper)
    MEMKEEPER_STORE             sqlite store path        (default: $MEMKEEPER_HOME/store.sqlite)
    MEMKEEPER_BIN               memkeeper binary           (default: `memkeeper` on $PATH)
    MEMKEEPER_EMBED_MODEL_DIR   local embed model dir    (default: $MEMKEEPER_HOME/models/mxbai-embed-large)
    MEMKEEPER_RERANK_MODEL_DIR  local rerank model dir   (default: $MEMKEEPER_HOME/models/mxbai-rerank-base)
    MEMKEEPER_SOCK              warm-daemon socket path  (default: /tmp/memkeeper_daemon.sock)

Without the model dirs present, search degrades to deterministic BM25 (still works);
with them, search is semantic + cross-encoder rerank.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

# Base dir for every default below. `memkeeper init` creates this directory.
MEMKEEPER_HOME = Path(os.environ.get("MEMKEEPER_HOME", str(Path.home() / ".memkeeper"))).resolve()
STORE = Path(os.environ.get("MEMKEEPER_STORE", str(MEMKEEPER_HOME / "store.sqlite"))).resolve()
# Optional: only affects the subprocess working directory (the CLI always gets
# an explicit --store, so cwd does not change which store is used). Defaults to
# the user's home so no repo checkout is assumed.
WORKSPACE = Path(os.environ.get("MEMKEEPER_WORKSPACE", str(Path.home()))).resolve()
# Binary resolution: explicit MEMKEEPER_BIN wins; otherwise look up `memkeeper` on
# PATH; otherwise fall back to the bare name so _run_memkeeper surfaces a clear
# "not found" error instead of a confusing absolute path.
_bin_env = os.environ.get("MEMKEEPER_BIN")
if _bin_env:
    MEMKEEPER_BIN = Path(_bin_env).resolve()
else:
    _which = shutil.which("memkeeper")
    MEMKEEPER_BIN = Path(_which).resolve() if _which else Path("memkeeper")
# Local embedding model dir. The semantic CLI self-embeds remember/search at
# 1024-dim (mxbai); the bridge never embeds itself, so there is no second
# provider that could write mismatched-dimension vectors into the store.
EMBED_MODEL_DIR = Path(
    os.environ.get("MEMKEEPER_EMBED_MODEL_DIR", str(MEMKEEPER_HOME / "models" / "mxbai-embed-large"))
).resolve()
# Cross-encoder reranker for explicit search. The engine reranks natively when
# the search request carries rerank=true (one process, one retrieval); any
# rerank failure inside the engine falls back to plain order, so recall can
# never regress. If this dir is absent, rerank is skipped (BM25/embed order).
RERANK_MODEL_DIR = Path(
    os.environ.get("MEMKEEPER_RERANK_MODEL_DIR", str(MEMKEEPER_HOME / "models" / "mxbai-rerank-base"))
).resolve()
# Warm daemon socket (memkeeper serve --socket). When present, requests go to
# the warm engine (models already loaded) instead of spawning a cold process
# per call. Pre-send failures fall back to the subprocess path; failures after
# a request has been sent raise instead of re-running (a retry could double-
# apply a write).
MEMKEEPER_SOCK = os.environ.get("MEMKEEPER_SOCK", "/tmp/memkeeper_daemon.sock")
RERANK_SEARCH = os.environ.get("MEMKEEPER_MCP_RERANK", "1").strip().lower() not in ("0", "false", "no", "off")
RERANK_CANDIDATES = int(os.environ.get("MEMKEEPER_MCP_RERANK_CANDIDATES", "16"))
MCP_ADAPTER = os.environ.get("MEMKEEPER_MCP_ADAPTER", "generic-mcp")
MCP_SOURCE_DESCRIPTION = os.environ.get("MEMKEEPER_MCP_SOURCE_DESCRIPTION", "memkeeper MCP")
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"

mcp = FastMCP(
    "memkeeper",
    instructions=(
        "Access to your local memkeeper store. Memories remain the source of truth; "
        "graph rows are rebuildable projections. Source/provenance is hidden unless an explicit "
        "include_source argument is provided and the user asked for provenance. "
        "Write only concise, durable, non-secret facts/decisions/preferences/lessons/actions/continuity notes. "
        "Remember responses may include auto_superseded ids or conflict_candidates; surface continuity conflicts to the user when relevant. "
        "Do not dump transcripts, secrets, noisy command output, or temporary task state. "
        "Use forget only to tombstone a specific memory id. "
        "For plausible-but-unverified inferences, prefer candidate_submit (enqueues for human review) over remember."
    ),
)


def _keychain_password(service: str, account: str | None = None) -> str | None:
    """Read one macOS Keychain password without exposing it in logs.

    On non-macOS hosts (no `security` binary) this returns None gracefully; it
    only does anything when MEMKEEPER_EMBED_KEYCHAIN_SERVICE is set.
    """
    if not service:
        return None
    args = ["security", "find-generic-password", "-w", "-s", service]
    if account:
        args.extend(["-a", account])
    try:
        proc = subprocess.run(
            args,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip()
    return value or None


def _embedding_env() -> dict[str, str]:
    """Normalize embedding env for the current Rust memkeeper CLI.

    Current memkeeper accepts providers `local` and `openai` (the latter covers
    any OpenAI-compatible endpoint, including OpenRouter, via a custom base URL).
    Legacy configs may still set provider `openrouter`; treat it as an
    OpenAI-compatible endpoint instead of letting the CLI disable embeddings.
    """
    env = {**os.environ}
    provider = env.get("MEMKEEPER_EMBED_PROVIDER", "local").strip().lower() or "local"

    if provider == "openrouter":
        env["MEMKEEPER_EMBED_PROVIDER"] = "openai"
        env.setdefault("MEMKEEPER_EMBED_BASE_URL", OPENROUTER_EMBEDDINGS_URL)
        if env.get("MEMKEEPER_EMBED_MODEL", "").startswith("openai/"):
            env["MEMKEEPER_EMBED_MODEL"] = env["MEMKEEPER_EMBED_MODEL"].split("/", 1)[1]
    else:
        env["MEMKEEPER_EMBED_PROVIDER"] = provider

    if env["MEMKEEPER_EMBED_PROVIDER"] == "local":
        env.setdefault("MEMKEEPER_EMBED_MODEL_DIR", str(EMBED_MODEL_DIR))

    if not env.get("MEMKEEPER_EMBED_API_KEY"):
        service = env.get("MEMKEEPER_EMBED_KEYCHAIN_SERVICE", "")
        account = env.get("MEMKEEPER_EMBED_KEYCHAIN_ACCOUNT") or None
        password = _keychain_password(service, account)
        if password:
            env["MEMKEEPER_EMBED_API_KEY"] = password

    # Only set the key when a real value resolves; assigning "" would mask the
    # missing-credential case (the CLI's own empty-key guard logs and disables
    # API embedding, so let it see the var as absent rather than blank).
    if not env.get("MEMKEEPER_EMBED_API_KEY") and env.get("MEMKEEPER_EMBED_PROVIDER") == "openai":
        key = env.get("OPENAI_API_KEY") or env.get("OPENROUTER_API_KEY")
        if key:
            env["MEMKEEPER_EMBED_API_KEY"] = key

    return env


def _socket_request(command: str, payload: dict[str, Any] | None, extra_args: list[str] | None) -> str | None:
    """Send one serve-protocol request to the warm daemon socket.

    Returns the response envelope, or None when the socket is unavailable
    BEFORE the request is sent (caller falls back to a subprocess). Failures
    after the request was sent raise: silently re-running could double-apply
    a write.
    """
    serve_payload: dict[str, Any] | None = payload
    if command == "stats":
        stats_flags = extra_args or []
        serve_payload = {
            "include_indexes": "--no-indexes" not in stats_flags,
            "include_health": "--health" in stats_flags,
        }
    elif payload is None and extra_args:
        return None  # flag-style invocation with no serve payload mapping
    request = {
        "protocol_version": "memkeeper.v0.1",
        "request_id": f"mcp-{command}",
        "command": command,
        "store_path": str(STORE),
        "payload": serve_payload or {},
    }
    line = json.dumps(request, ensure_ascii=False, separators=(",", ":")) + "\n"
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(30.0)
        try:
            sock.connect(MEMKEEPER_SOCK)
            sock.sendall(line.encode("utf-8"))
        except OSError:
            return None  # daemon down/unreachable: safe to fall back
        data = b""
        while not data.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            data += chunk
    finally:
        sock.close()
    if not data:
        raise RuntimeError(f"memkeeper daemon returned no response for {command}")
    envelope = json.loads(data)
    if not isinstance(envelope, dict) or "ok" not in envelope:
        raise RuntimeError(f"memkeeper daemon returned a malformed envelope for {command}")
    if envelope.get("ok") is False:
        error = envelope.get("error") or {}
        raise RuntimeError(
            f"memkeeper {command} failed ({error.get('code', 'unknown')}): {error.get('message', '')}"
        )
    return data.decode("utf-8").rstrip("\n")


def _run_memkeeper(
    command: str,
    payload: dict[str, Any] | None = None,
    extra_args: list[str] | None = None,
    env_overrides: dict[str, str] | None = None,
) -> str:
    if os.path.exists(MEMKEEPER_SOCK):
        response = _socket_request(command, payload, extra_args)
        if response is not None:
            return response
    if not MEMKEEPER_BIN.exists():
        raise RuntimeError(
            f"memkeeper binary not found ({MEMKEEPER_BIN}). Install memkeeper (cargo install --git ...) "
            "or set MEMKEEPER_BIN to its path."
        )
    args = [str(MEMKEEPER_BIN), command, "--store", str(STORE)]
    if payload is not None:
        args.extend(["--json", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))])
    elif extra_args:
        args.extend(extra_args)
    else:
        args.append("--json")
    env = _embedding_env()
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        args,
        cwd=str(WORKSPACE),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"memkeeper {command} failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout.strip()


# --------------------------------------------------------------------------
# Recall telemetry. Recorded through the engine's `recall-log` command, which
# owns the recall_events table and the memories.accessed_at touch. Two kinds:
#   surfaced  -- a memory appeared in a search result (with rank/score/query)
#   retrieved -- a memory was explicitly fetched via get (true use)
# Telemetry is strictly best-effort: a logging failure must never break recall.
# --------------------------------------------------------------------------
def _record_recall(events: list[dict[str, Any]]) -> None:
    """Send recall events to the engine. Best-effort: failures are swallowed."""
    if not events:
        return
    try:
        _run_memkeeper(
            "recall-log",
            {"source": MCP_SOURCE_DESCRIPTION, "touch_accessed": True, "events": events},
        )
    except Exception:
        pass


def _log_surfaced(query: str, raw: str) -> None:
    """Parse a search envelope and log one 'surfaced' event per returned memory."""
    try:
        results = (json.loads(raw).get("result") or {}).get("results") or []
        events = [
            {
                "memory_id": r["memory_id"],
                "kind": "surfaced",
                "query": query,
                "rank": r.get("rank"),
                "score": r.get("score"),
            }
            for r in results
            if r.get("memory_id")
        ]
        _record_recall([{k: v for k, v in e.items() if v is not None} for e in events])
    except Exception:
        pass


def _log_retrieved(raw: str) -> None:
    """Parse a get envelope and log a 'retrieved' event (touches accessed_at)."""
    try:
        mem = (json.loads(raw).get("result") or {}).get("memory") or {}
        mid = mem.get("id")
        if mid:
            _record_recall([{"memory_id": mid, "kind": "retrieved"}])
    except Exception:
        pass


@mcp.tool()
def stats(include_indexes: bool = False, include_health: bool = False) -> str:
    """Return memkeeper store statistics. Read-only.

    include_health adds a memory-governance rollup: lifecycle status counts,
    active memories missing keys, duplicate (entity,claim) groups, short-term
    promotion backlog, logically-stale (past valid_to) memories, embedding
    coverage, and the last embedding timestamp.
    """
    args = ["--json"]
    if not include_indexes:
        args.append("--no-indexes")
    if include_health:
        args.append("--health")
    return _run_memkeeper("stats", extra_args=args)


@mcp.tool()
def search(
    query: str,
    limit: int = 10,
    space: str | None = None,
    tags: list[str] | None = None,
    entity_key: str | None = None,
    include_content: bool = False,
    include_source: bool = False,
    semantic_enabled: bool = True,
    semantic_fallback: bool | None = None,
    rerank: bool = True,
) -> str:
    """Search memories: semantic-primary when embeddings are available, with deterministic BM25/FTS degradation.

    By default the candidate pool is reranked with a cross-encoder (matching the
    passive auto-retrieve hooks); pass rerank=False for the raw embed/BM25 order.
    """
    filters: dict[str, Any] = {}
    if space:
        filters["spaces"] = [space]
    if tags:
        filters["tags"] = tags
    if entity_key:
        filters["entity_keys"] = [entity_key]
    do_rerank = rerank and RERANK_SEARCH and limit > 0 and RERANK_MODEL_DIR.exists()
    payload: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "include_content": include_content,
        "include_source": include_source,
    }
    if do_rerank:
        # Native rerank: the engine widens the pool, runs the cross-encoder,
        # reorders, and truncates back to `limit` -- one process, one retrieval.
        payload["rerank"] = True
        payload["rerank_candidates"] = RERANK_CANDIDATES
    if semantic_fallback is not None:
        semantic_enabled = semantic_fallback
    payload["semantic_fallback"] = "fallback" if semantic_enabled else "disabled"
    if filters:
        payload["filters"] = filters
    overrides = (
        {"MEMKEEPER_RERANK_PROVIDER": "local", "MEMKEEPER_RERANK_MODEL_DIR": str(RERANK_MODEL_DIR)}
        if do_rerank
        else None
    )
    raw = _run_memkeeper("search", payload, env_overrides=overrides)
    _log_surfaced(query, raw)
    return raw


@mcp.tool()
def get(memory_id: str, include_history: bool = False, include_source: bool = False) -> str:
    """Get one memory by id. Source hidden by default."""
    raw = _run_memkeeper("get", {"id": memory_id, "include_history": include_history, "include_source": include_source})
    _log_retrieved(raw)
    return raw


@mcp.tool()
def memory_list(
    limit: int = 20,
    status: str | None = None,
    space: str | None = None,
    entity_key: str | None = None,
    include_content: bool = False,
    include_source: bool = False,
) -> str:
    """List recent memories for review/cleanup. Source hidden by default."""
    filters: dict[str, Any] = {}
    if status:
        filters["statuses"] = [status]
    if space:
        filters["spaces"] = [space]
    if entity_key:
        filters["entity_keys"] = [entity_key]
    payload: dict[str, Any] = {"limit": limit, "include_content": include_content, "include_source": include_source}
    if filters:
        payload["filters"] = filters
    return _run_memkeeper("memory-list", payload)


@mcp.tool()
def entity_search(
    query: str | None = None,
    entity_key: str | None = None,
    entity_type: str | None = None,
    limit: int = 10,
    include_source: bool = False,
) -> str:
    """Search graph entities by key/name/alias/type. Source hidden by default."""
    payload: dict[str, Any] = {"limit": limit, "include_source": include_source}
    if query:
        payload["query"] = query
    if entity_key:
        payload["entity_key"] = entity_key
    if entity_type:
        payload["entity_types"] = [entity_type]
    return _run_memkeeper("entity-search", payload)


@mcp.tool()
def graph_neighbors(
    entity_key: str,
    depth: int = 1,
    max_edges: int = 50,
    include_tombstoned: bool = False,
    include_source: bool = False,
) -> str:
    """Traverse bounded graph neighbors from an entity key. Source hidden by default."""
    return _run_memkeeper(
        "graph-neighbors",
        {
            "entity_key": entity_key,
            "depth": depth,
            "max_edges": max_edges,
            "include_tombstoned": include_tombstoned,
            "include_source": include_source,
        },
    )


@mcp.tool()
def graph_context(
    entity_key: str,
    depth: int = 1,
    max_edges: int = 50,
    max_memories: int = 10,
    max_chars: int = 4000,
    include_source: bool = False,
) -> str:
    """Build a compact graph-centered context pack around an entity key. Source hidden by default."""
    return _run_memkeeper(
        "graph-context",
        {
            "entity_key": entity_key,
            "depth": depth,
            "max_edges": max_edges,
            "max_memories": max_memories,
            "max_chars": max_chars,
            "include_source": include_source,
        },
    )


@mcp.tool()
def dream_graph(max_memories: int = 1000, space: str | None = None) -> str:
    """Run read-only/proposal-only graph diagnostics via dream dry-run."""
    payload: dict[str, Any] = {"tasks": ["graph"], "max_memories": max_memories, "dry_run": True}
    if space:
        payload["space"] = space
    return _run_memkeeper("dream", payload)


@mcp.tool()
def remember(
    content: str,
    space: str | None = None,
    silo: str | None = None,
    scope: str | None = None,
    project: str | None = None,
    kind: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    entity_key: str | None = None,
    claim_key: str | None = None,
    derive_keys: bool = True,
    confidence: float = 1.0,
    observed_at: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    expires_at: str | None = None,
    pinned: bool = False,
    supersedes: list[str] | None = None,
    contradicts: list[str] | None = None,
    verified_against: str | None = None,
    mode: str = "auto",
    source_type: str = "assistant-inference",
    sensitivity: str | None = None,
    dry_run: bool = False,
) -> str:
    """Store one atomic, self-contained fact per call.

    Rules:
    - One claim per call. If content contains multiple facts, make multiple remember calls.
    - Each memory must be independently retrievable without context from other memories.
    - Do not store secrets, transcripts, noisy command output, or temporary task state.
    - Prefer specific, concrete language ("use British English" not "has language rules").
    - derive_keys (default true) auto-fills entity_key/claim_key from the content when
      you do not supply them, so repeated/updated facts group and supersede cleanly.
      Any entity_key/claim_key you pass explicitly always wins; set derive_keys=False
      to store a one-off memory with no derived keys.
    - source_type marks provenance/trust; retrieval ranks higher-trust sources above
      inferred ones. Default "assistant-inference" (you inferred it). Pass
      "explicit-user" when the user directly stated the fact/preference. sensitivity
      is "normal" (default) or "sensitive".
    - mode controls how this write resolves against active memories sharing its
      entity/claim key: "auto" (default; retire older same-key of eligible kinds),
      "append" (coexist), "supersede" (force-retire all same-key), "suggest"
      (return what WOULD be retired, mutate nothing), "conflict" (open a conflict
      for review instead of retiring).
    """
    source_obj: dict[str, Any] = {
        "type": "mcp",
        "adapter": MCP_ADAPTER,
        "source_description": MCP_SOURCE_DESCRIPTION,
        "source_type": source_type,
    }
    if sensitivity:
        source_obj["sensitivity"] = sensitivity
    payload: dict[str, Any] = {
        "content": content,
        "confidence": confidence,
        "pinned": pinned,
        "derive_keys": derive_keys,
        "mode": mode,
        "dry_run": dry_run,
        "source": source_obj,
    }
    optional_values: dict[str, Any] = {
        "space": space,
        "silo": silo,
        "scope": scope,
        "project": project,
        "kind": kind,
        "summary": summary,
        "tags": tags,
        "entity_key": entity_key,
        "claim_key": claim_key,
        "observed_at": observed_at,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "expires_at": expires_at,
        "supersedes": supersedes,
        "contradicts": contradicts,
    }
    payload.update({key: value for key, value in optional_values.items() if value is not None})
    if verified_against:
        payload["metadata_json"] = json.dumps({"verified_against": verified_against})
    return _run_memkeeper("remember", payload)


@mcp.tool()
def forget(memory_id: str, reason: str | None = None, dry_run: bool = False) -> str:
    """Tombstone one specific memory id. This preserves audit history; it is not a hard delete."""
    payload: dict[str, Any] = {"id": memory_id, "dry_run": dry_run}
    if reason:
        payload["reason"] = reason
    return _run_memkeeper("forget", payload)


@mcp.tool()
def entity_upsert(
    entity_key: str,
    canonical_name: str,
    entity_type: str | None = None,
    aliases: list[str] | None = None,
    space: str | None = None,
    status: str | None = None,
    confidence: float = 1.0,
    source_episode_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    include_source: bool = False,
) -> str:
    """Create or update a graph entity projection. Memories remain the source of truth."""
    payload: dict[str, Any] = {
        "entity_key": entity_key,
        "canonical_name": canonical_name,
        "confidence": confidence,
        "include_source": include_source,
    }
    optional_values: dict[str, Any] = {
        "entity_type": entity_type,
        "aliases": aliases,
        "space": space,
        "status": status,
        "source_episode_id": source_episode_id,
        "metadata": metadata,
    }
    payload.update({key: value for key, value in optional_values.items() if value is not None})
    return _run_memkeeper("entity-upsert", payload)


@mcp.tool()
def relationship_upsert(
    relation_type: str,
    subject_entity_key: str | None = None,
    object_entity_key: str | None = None,
    subject_entity_id: str | None = None,
    object_entity_id: str | None = None,
    memory_id: str | None = None,
    space: str | None = None,
    source_episode_id: str | None = None,
    status: str | None = None,
    confidence: float = 1.0,
    observed_at: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    metadata: dict[str, Any] | None = None,
    include_source: bool = False,
) -> str:
    """Create or update a graph relationship projection between two entities."""
    payload: dict[str, Any] = {"relation_type": relation_type, "confidence": confidence, "include_source": include_source}
    optional_values: dict[str, Any] = {
        "subject_entity_key": subject_entity_key,
        "object_entity_key": object_entity_key,
        "subject_entity_id": subject_entity_id,
        "object_entity_id": object_entity_id,
        "memory_id": memory_id,
        "space": space,
        "source_episode_id": source_episode_id,
        "status": status,
        "observed_at": observed_at,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "metadata": metadata,
    }
    payload.update({key: value for key, value in optional_values.items() if value is not None})
    return _run_memkeeper("relationship-upsert", payload)


@mcp.tool()
def verify(memory_id: str, verified_against: str | None = None) -> str:
    """Stamp a volatile memory as re-confirmed against ground truth NOW.

    Use after checking a volatile memory's claim against its source (file/env/config).
    Sets verified_at=now (resetting the freshness clock until the next synthesis run).
    Optionally (re)sets verified_against to the ground-truth pointer.
    Does NOT promote the memory to durable. If the value CHANGED, do not verify --
    write a new memory and supersede the old one instead.
    """
    payload: dict[str, Any] = {"memory_id": memory_id}
    if verified_against:
        payload["verified_against"] = verified_against
    return _run_memkeeper("verify", payload)


@mcp.tool()
def pack(
    queries: list[str],
    title: str = "context",
    max_memories: int = 10,
    max_chars: int = 6000,
    min_score: float = 0.0,
    space: str | None = None,
    tags: list[str] | None = None,
) -> str:
    """Build a compact, ready-to-inject context pack from one or more queries.

    Returns bounded markdown of the top memories across the queries — ideal for
    seeding an agent's working context in one call instead of many searches.
    Read-only.
    """
    payload: dict[str, Any] = {
        "title": title,
        "queries": queries,
        "max_memories": max_memories,
        "max_chars": max_chars,
        "min_score": min_score,
    }
    filters: dict[str, Any] = {}
    if space:
        filters["spaces"] = [space]
    if tags:
        filters["tags"] = tags
    if filters:
        payload["filters"] = filters
    return _run_memkeeper("pack", payload)


@mcp.tool()
def candidate_submit(
    content: str,
    rationale: str | None = None,
    kind: str | None = None,
    summary: str | None = None,
    tags: list[str] | None = None,
    entity_key: str | None = None,
    claim_key: str | None = None,
    confidence: float = 1.0,
    source_type: str = "assistant-inference",
    sensitivity: str | None = None,
    space: str | None = None,
    silo: str | None = None,
    scope: str | None = None,
    project: str | None = None,
    supersedes: list[str] | None = None,
    dry_run: bool = False,
) -> str:
    """Submit a candidate memory for human review instead of writing it directly.

    Use this for plausible-but-unverified inferences you want approved before they
    enter the store. It only enqueues — approval/rejection is a human action (CLI
    or dashboard). Include `rationale` to explain why it is worth keeping. For
    high-confidence, user-stated facts, prefer `remember` directly.
    """
    payload: dict[str, Any] = {
        "content": content,
        "confidence": confidence,
        "source_type": source_type,
        "dry_run": dry_run,
    }
    optional_values: dict[str, Any] = {
        "rationale": rationale,
        "kind": kind,
        "summary": summary,
        "tags": tags,
        "entity_key": entity_key,
        "claim_key": claim_key,
        "sensitivity": sensitivity,
        "space": space,
        "silo": silo,
        "scope": scope,
        "project": project,
        "supersedes": supersedes,
    }
    payload.update({key: value for key, value in optional_values.items() if value is not None})
    return _run_memkeeper("candidate-submit", payload)


@mcp.tool()
def candidate_list(status: str = "pending", space: str | None = None, limit: int = 50) -> str:
    """List candidate memories for review. status: pending | approved | rejected (default pending)."""
    payload: dict[str, Any] = {"limit": limit}
    if status:
        payload["status"] = status
    if space:
        payload["space"] = space
    return _run_memkeeper("candidate-list", payload)


if __name__ == "__main__":
    mcp.run()
