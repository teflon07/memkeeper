#!/usr/bin/env python3
"""LoCoMo retrieval-only benchmark harness for memkeeper recall.

This harness evaluates whether memkeeper's bounded `pack` retrieval returns the
LoCoMo dialogue turns annotated as evidence for each QA item. It does not call
an answering model or judge. The goal is a deterministic external recall signal
that mirrors the host retrieval path: question -> query bundle ->
`memkeeper pack` -> bounded prompt context.

Dataset input is a local LoCoMo JSON file. The original SNAP release is
`data/locomo10.json` from https://github.com/snap-research/locomo and is
licensed separately (CC BY-NC 4.0). This script intentionally does not vendor
or auto-download the dataset.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

WORKSPACE = Path(__file__).resolve().parents[3]
# Crate root holding target/: memory/memkeeper in the monorepo, the repo root in
# the standalone OSS layout. Derived from the script's own location so it is
# correct in both (scripts/ always sits one level under the crate root).
MEMKEEPER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BIN = MEMKEEPER_DIR / "target" / "release" / "memkeeper"
SOURCE_MARKERS = [
    # Machine/provenance fields that should not appear in source-hidden packs.
    '"source_episode_id"',
    '"source_ref"',
    '"source_description"',
    '"adapter"',
    '"session_id"',
    '"cwd"',
    "source_ref_json",
]
CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "before",
    "can",
    "did",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "him",
    "his",
    "how",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}


@dataclass(frozen=True)
class DialogueTurn:
    sample_id: str
    session_key: str
    session_index: int
    session_datetime: str
    turn_index: int
    dia_id: str
    speaker: str
    text: str
    blip_caption: str = ""
    query: str = ""


@dataclass(frozen=True)
class TurnRef:
    sample_id: str
    dia_id: str


@dataclass(frozen=True)
class RetrievalScore:
    evidence_turn_ids: list[str]
    retrieved_dia_ids: list[str]
    retrieved_turn_ids: list[str]
    hit: bool
    evidence_recall: float
    reciprocal_rank: float


@dataclass(frozen=True)
class QAItem:
    qa_id: str
    sample_id: str
    question: str
    answer: Any
    category: int | str
    evidence: list[str]


@dataclass
class PreparedSample:
    sample_id: str
    conversation: dict[str, Any]
    qa: list[QAItem]


@dataclass
class RetrievalResult:
    qa_id: str
    sample_id: str
    question: str
    category: str
    evidence: list[str]
    evidence_turn_ids: list[str]
    retrieved_dia_ids: list[str]
    retrieved_turn_ids: list[str]
    retrieved_memory_ids: list[str]
    hit: bool
    evidence_recall: float
    reciprocal_rank: float
    elapsed_ms: float
    chars: int
    char_budget_usage: float
    truncated: bool
    source_leaks: list[str]
    semantic_attempted: bool | None = None
    semantic_reasons: list[str] = field(default_factory=list)


@dataclass
class AggregateBucket:
    total: int = 0
    with_evidence: int = 0
    hit: int = 0
    evidence_found: int = 0
    evidence_total: int = 0
    reciprocal_rank_sum: float = 0.0
    latency_values: list[float] = field(default_factory=list)
    char_values: list[int] = field(default_factory=list)
    budget_values: list[float] = field(default_factory=list)
    truncated: int = 0
    source_leak_count: int = 0
    semantic_attempted: int = 0
    semantic_inspected: int = 0


def percentile(values: list[float], pct: float) -> float:
    """Return nearest-rank percentile for a non-empty list, or 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def category_name(category: int | str) -> str:
    """Return a stable category label."""
    try:
        value = int(category)
    except (TypeError, ValueError):
        text = str(category).strip() or "unknown"
        return text.lower().replace(" ", "_")
    return CATEGORY_NAMES.get(value, f"category_{value}")


def normalize_evidence(values: Any) -> list[str]:
    """Normalize LoCoMo evidence IDs."""
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    result: list[str] = []
    for value in values:
        text = str(value).strip().strip("()")
        if text:
            result.append(text)
    return result


def parse_json_maybe(value: Any) -> Any:
    """Parse JSON strings in Hugging Face flattened exports when needed."""
    if isinstance(value, str):
        text = value.strip()
        if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
            return json.loads(text)
    return value


def qa_from_raw(sample_id: str, index: int, payload: dict[str, Any]) -> QAItem:
    """Build a QA item from original or flattened LoCoMo fields."""
    qa_id = str(payload.get("qa_id") or f"{sample_id}#q{index:04d}")
    evidence = parse_json_maybe(payload.get("evidence_json", payload.get("evidence", [])))
    return QAItem(
        qa_id=qa_id,
        sample_id=sample_id,
        question=str(payload.get("question") or ""),
        answer=payload.get("answer"),
        category=payload.get("category", "unknown"),
        evidence=normalize_evidence(evidence),
    )


def load_locomo_dataset(path: Path) -> list[PreparedSample]:
    """Load original LoCoMo JSON or common flattened JSON/JSONL exports."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    if not isinstance(raw, list):
        raise ValueError("LoCoMo dataset must be a JSON list or JSONL records")

    # Original schema: one row per conversation with conversation + qa.
    if raw and isinstance(raw[0], dict) and "conversation" in raw[0] and "qa" in raw[0]:
        samples = []
        for sample in raw:
            sample_id = str(sample.get("sample_id") or f"sample_{len(samples):04d}")
            qa = [qa_from_raw(sample_id, idx, item) for idx, item in enumerate(sample.get("qa") or [])]
            samples.append(PreparedSample(sample_id=sample_id, conversation=sample["conversation"], qa=qa))
        return samples

    # Hugging Face / table schema: one row per QA with repeated conversation_json.
    grouped: dict[str, dict[str, Any]] = {}
    qa_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        if not isinstance(row, dict):
            continue
        sample_id = str(row.get("sample_id") or row.get("conversation_id") or "sample")
        if sample_id not in grouped:
            conversation = parse_json_maybe(row.get("conversation_json", row.get("conversation")))
            if not isinstance(conversation, dict):
                raise ValueError(f"missing conversation JSON for sample {sample_id}")
            grouped[sample_id] = conversation
        qa_rows[sample_id].append(row)
    samples = []
    for sample_id, rows in sorted(qa_rows.items()):
        qa = [qa_from_raw(sample_id, int(row.get("qa_index", idx)), row) for idx, row in enumerate(rows)]
        samples.append(PreparedSample(sample_id=sample_id, conversation=grouped[sample_id], qa=qa))
    return samples


def session_index_from_key(key: str) -> int:
    match = re.search(r"session_(\d+)$", key)
    return int(match.group(1)) if match else 0


def iter_dialogue_turns(sample: PreparedSample) -> list[DialogueTurn]:
    """Return dialogue turns in chronological session/turn order."""
    turns: list[DialogueTurn] = []
    session_keys = [key for key, value in sample.conversation.items() if key.startswith("session_") and isinstance(value, list)]
    for session_key in sorted(session_keys, key=session_index_from_key):
        session_index = session_index_from_key(session_key)
        session_datetime = str(sample.conversation.get(f"{session_key}_date_time") or "")
        for turn_index, turn in enumerate(sample.conversation.get(session_key) or [], start=1):
            if not isinstance(turn, dict):
                continue
            dia_id = str(turn.get("dia_id") or f"D{session_index}:{turn_index}")
            text = str(turn.get("text") or "").strip()
            if not text:
                continue
            turns.append(
                DialogueTurn(
                    sample_id=sample.sample_id,
                    session_key=session_key,
                    session_index=session_index,
                    session_datetime=session_datetime,
                    turn_index=turn_index,
                    dia_id=dia_id,
                    speaker=str(turn.get("speaker") or "unknown"),
                    text=text,
                    blip_caption=str(turn.get("blip_caption") or ""),
                    query=str(turn.get("query") or ""),
                )
            )
    return turns


def observed_at_for_turn(turn: DialogueTurn) -> str:
    """Use deterministic synthetic timestamps so benchmark ordering is repeatable."""
    base = datetime(2023, 1, 1, tzinfo=timezone.utc)
    stamp = base + timedelta(days=turn.session_index, seconds=turn.turn_index)
    return stamp.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def turn_content(turn: DialogueTurn) -> str:
    """Build the memory text for a dialogue turn."""
    parts = [
        f"fact: LoCoMo dialogue turn {turn.sample_id} {turn.dia_id}.",
        f"Session {turn.session_index} date/time: {turn.session_datetime}.",
        f"{turn.speaker}: {turn.text}",
    ]
    if turn.blip_caption:
        parts.append(f"Image caption: {turn.blip_caption}.")
    if turn.query:
        parts.append(f"Image search query: {turn.query}.")
    return " ".join(parts)


def _socket_request(sock_path: str, store: Path, command: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    """Send one protocol envelope to a warm `serve --socket` daemon and return the response."""
    import socket

    envelope = {
        "protocol_version": "memkeeper.v0.1",
        "request_id": f"bench-{command}",
        "command": command,
        "store_path": str(store),
        "payload": payload or {},
    }
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(30.0)
        sock.connect(sock_path)
        sock.sendall(json.dumps(envelope, separators=(",", ":")).encode() + b"\n")
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
    response = json.loads(buf.split(b"\n", 1)[0].decode())
    if not response.get("ok", False):
        error = response.get("error") or {}
        raise RuntimeError(f"daemon {command} failed: {error.get('code')}: {error.get('message')}")
    return response


def run_memkeeper(binary: Path, store: Path, command: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run one memkeeper command: via the warm daemon socket when MEMKEEPER_BENCH_SOCK
    is set (avoids per-call model loads), else by spawning the CLI binary."""
    sock_path = os.environ.get("MEMKEEPER_BENCH_SOCK", "")
    if sock_path:
        return _socket_request(sock_path, store, command, payload)
    args = [str(binary), command, "--store", str(store)]
    if payload is None:
        args.append("--json")
    else:
        args.extend(["--json", json.dumps(payload, separators=(",", ":"))])
    proc = subprocess.run(
        args,
        cwd=str(WORKSPACE),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def qualified_dia_id(sample_id: str, dia_id: str) -> str:
    """Return a sample-qualified dialogue ID for scoring across conversations."""
    return f"{sample_id}::{dia_id}"


def seed_store(binary: Path, store: Path, samples: list[PreparedSample]) -> tuple[dict[str, TurnRef], dict[str, str]]:
    """Initialize store and remember one memory per dialogue turn.

    Returns (memory_id_to_turn_ref, qualified_dia_id_to_memory_id). LoCoMo dialogue IDs
    repeat across conversations, so scoring must use sample-qualified IDs.
    """
    run_memkeeper(binary, store, "init")
    memory_to_turn: dict[str, TurnRef] = {}
    dia_to_memory: dict[str, str] = {}
    for sample in samples:
        for turn in iter_dialogue_turns(sample):
            payload = {
                "space": "workspace-memory",
                "silo": "durable",
                "scope": "workspace",
                "project": "LoCoMo",
                "kind": "fact",
                "content": turn_content(turn),
                "summary": f"LoCoMo {turn.sample_id} {turn.dia_id}: {turn.speaker}: {turn.text[:160]}",
                "tags": ["locomo", turn.sample_id, f"session-{turn.session_index}"],
                "entity_key": f"locomo:{turn.sample_id}",
                "claim_key": f"locomo.{turn.sample_id}.{turn.dia_id.replace(':', '_')}",
                "confidence": 1.0,
                "observed_at": observed_at_for_turn(turn),
                "source": {
                    "type": "benchmark",
                    "adapter": "memkeeper-locomo-benchmark",
                    "source_description": f"{turn.sample_id} {turn.dia_id}",
                },
            }
            response = run_memkeeper(binary, store, "remember", payload)
            memory_id = response["result"]["memory"]["id"]
            memory_to_turn[memory_id] = TurnRef(sample_id=turn.sample_id, dia_id=turn.dia_id)
            dia_to_memory[qualified_dia_id(turn.sample_id, turn.dia_id)] = memory_id
    return memory_to_turn, dia_to_memory


def keyword_query(text: str, limit: int = 10) -> str:
    """Build a simple deterministic keyword query from a question."""
    words = [w.strip(".,?!:;()[]{}\"'`)./\\").lower() for w in text.split()]
    keywords = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    return " ".join(keywords[:limit])


def query_bundle(question: str, category: str, mode: str) -> list[str]:
    """Build the query list used by `pack`. `bundle` is multi-query expansion
    (the retrieval shape hosts use for pack); `hermes` is a deprecated alias."""
    if mode == "question":
        return [question]
    queries = [question]
    keywords = keyword_query(question, limit=10)
    if keywords:
        queries.append(keywords)
    if category == "temporal":
        queries.append(f"{keywords} date time when conversation".strip())
    if category == "adversarial":
        queries.append(f"{keywords} no information available".strip())
    seen: set[str] = set()
    result = []
    for candidate in queries:
        candidate = candidate.strip()
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result[:4]


def sample_filters(sample_id: str, mode: str) -> dict[str, list[str]]:
    """Build optional filters that keep independent LoCoMo conversations isolated."""
    if mode == "none":
        return {}
    if mode == "tag":
        return {"tags": [sample_id]}
    return {"entity_keys": [f"locomo:{sample_id}"]}


def run_pack(
    binary: Path,
    store: Path,
    qa: QAItem,
    *,
    max_memories: int,
    max_chars: int,
    query_mode: str,
    sample_filter: str,
    rerank_candidates: int = 0,
) -> tuple[dict[str, Any], float]:
    category = category_name(qa.category)
    payload = {
        "title": f"LoCoMo retrieval: {qa.qa_id}",
        "queries": query_bundle(qa.question, category, query_mode),
        "max_memories": max_memories,
        "max_chars": max_chars,
        "format": "markdown",
    }
    if rerank_candidates > 0:
        payload["rerank_candidates"] = rerank_candidates
    filters = sample_filters(qa.sample_id, sample_filter)
    if filters:
        payload["filters"] = filters
    started = time.perf_counter()
    data = run_memkeeper(binary, store, "pack", payload)
    elapsed_ms = (time.perf_counter() - started) * 1000
    pack = (((data.get("result") or {}).get("pack")) or {}) if isinstance(data, dict) else {}
    return pack, elapsed_ms


def inspect_semantic(
    binary: Path,
    store: Path,
    qa: QAItem,
    query_mode: str,
    limit: int,
    sample_filter: str,
) -> tuple[bool, list[str]]:
    """Inspect semantic fallback metadata using batch-search over the same query bundle."""
    category = category_name(qa.category)
    queries = [
        {"name": f"q{index}", "query": query, "limit": limit}
        for index, query in enumerate(query_bundle(qa.question, category, query_mode), start=1)
    ]
    payload: dict[str, Any] = {"queries": queries}
    filters = sample_filters(qa.sample_id, sample_filter)
    if filters:
        payload["common_filters"] = filters
    data = run_memkeeper(binary, store, "batch-search", payload)
    attempted = False
    reasons: list[str] = []
    for item in ((data.get("result") or {}).get("results") or []):
        semantic = ((item.get("search") or {}).get("semantic") or {})
        if semantic:
            attempted = attempted or bool(semantic.get("attempted"))
            reason = str(semantic.get("reason") or "")
            if reason:
                reasons.append(reason)
    return attempted, sorted(set(reasons))


def score_retrieval(qa: QAItem, retrieved_turns: list[TurnRef]) -> RetrievalScore:
    """Score retrieved turns against sample-qualified LoCoMo evidence IDs."""
    evidence_turn_ids = [qualified_dia_id(qa.sample_id, dia_id) for dia_id in qa.evidence]
    retrieved_dia_ids = [turn.dia_id for turn in retrieved_turns]
    retrieved_turn_ids = [qualified_dia_id(turn.sample_id, turn.dia_id) for turn in retrieved_turns]
    evidence_set = set(evidence_turn_ids)
    found = [turn_id for turn_id in retrieved_turn_ids if turn_id in evidence_set]
    first_rank = next((index + 1 for index, turn_id in enumerate(retrieved_turn_ids) if turn_id in evidence_set), None)
    return RetrievalScore(
        evidence_turn_ids=evidence_turn_ids,
        retrieved_dia_ids=retrieved_dia_ids,
        retrieved_turn_ids=retrieved_turn_ids,
        hit=bool(found),
        evidence_recall=(len(set(found)) / len(evidence_set)) if evidence_set else 0.0,
        reciprocal_rank=(1.0 / first_rank) if first_rank else 0.0,
    )


def evaluate_one(
    binary: Path,
    store: Path,
    qa: QAItem,
    memory_to_turn: dict[str, TurnRef],
    *,
    max_memories: int,
    max_chars: int,
    query_mode: str,
    sample_filter: str,
    inspect_semantic_flag: bool,
    rerank_candidates: int = 0,
) -> RetrievalResult:
    """Run pack and compute evidence recall metrics for one QA item."""
    pack, elapsed_ms = run_pack(
        binary,
        store,
        qa,
        max_memories=max_memories,
        max_chars=max_chars,
        query_mode=query_mode,
        sample_filter=sample_filter,
        rerank_candidates=rerank_candidates,
    )
    content = str(pack.get("content") or "")
    memory_ids = list(pack.get("memory_ids") or [])
    retrieved_turns = [memory_to_turn[mid] for mid in memory_ids if mid in memory_to_turn]
    score = score_retrieval(qa, retrieved_turns)
    semantic_attempted: bool | None = None
    semantic_reasons: list[str] = []
    if inspect_semantic_flag:
        semantic_attempted, semantic_reasons = inspect_semantic(binary, store, qa, query_mode, max_memories, sample_filter)
    return RetrievalResult(
        qa_id=qa.qa_id,
        sample_id=qa.sample_id,
        question=qa.question,
        category=category_name(qa.category),
        evidence=qa.evidence,
        evidence_turn_ids=score.evidence_turn_ids,
        retrieved_dia_ids=score.retrieved_dia_ids,
        retrieved_turn_ids=score.retrieved_turn_ids,
        retrieved_memory_ids=memory_ids,
        hit=score.hit,
        evidence_recall=score.evidence_recall,
        reciprocal_rank=score.reciprocal_rank,
        elapsed_ms=round(elapsed_ms, 3),
        chars=len(content),
        char_budget_usage=(len(content) / max_chars) if max_chars else 0.0,
        truncated=bool(pack.get("truncated")),
        source_leaks=[marker for marker in SOURCE_MARKERS if marker.lower() in content.lower()],
        semantic_attempted=semantic_attempted,
        semantic_reasons=semantic_reasons,
    )


def add_to_bucket(bucket: AggregateBucket, result: RetrievalResult) -> None:
    bucket.total += 1
    if result.evidence_turn_ids:
        bucket.with_evidence += 1
        bucket.evidence_total += len(set(result.evidence_turn_ids))
        bucket.evidence_found += round(result.evidence_recall * len(set(result.evidence_turn_ids)))
        bucket.hit += int(result.hit)
        bucket.reciprocal_rank_sum += result.reciprocal_rank
    bucket.latency_values.append(result.elapsed_ms)
    bucket.char_values.append(result.chars)
    bucket.budget_values.append(result.char_budget_usage)
    bucket.truncated += int(result.truncated)
    bucket.source_leak_count += int(bool(result.source_leaks))
    if result.semantic_attempted is not None:
        bucket.semantic_inspected += 1
        bucket.semantic_attempted += int(result.semantic_attempted)


# Rough LLM-token approximation. No model tokenizer is bundled, so context cost
# is reported as an estimate of chars / 4 (typical English ratio). Labeled
# `_est` everywhere it surfaces so it is never mistaken for an exact count.
CHARS_PER_TOKEN = 4


def bucket_summary(bucket: AggregateBucket) -> dict[str, Any]:
    latencies = bucket.latency_values
    char_avg = (sum(bucket.char_values) / len(bucket.char_values)) if bucket.char_values else 0.0
    char_max = max(bucket.char_values, default=0)
    return {
        "total": bucket.total,
        "with_evidence": bucket.with_evidence,
        "hit_at_k": (bucket.hit / bucket.with_evidence) if bucket.with_evidence else 0.0,
        "evidence_recall_at_k": (bucket.evidence_found / bucket.evidence_total) if bucket.evidence_total else 0.0,
        "mrr": (bucket.reciprocal_rank_sum / bucket.with_evidence) if bucket.with_evidence else 0.0,
        "pack_latency_ms": {
            "avg": (sum(latencies) / len(latencies)) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies, default=0.0),
        },
        "pack_chars": {
            "avg": char_avg,
            "max": char_max,
            "avg_budget_usage": (sum(bucket.budget_values) / len(bucket.budget_values)) if bucket.budget_values else 0.0,
        },
        "pack_tokens_est": {
            "avg": char_avg / CHARS_PER_TOKEN,
            "max": char_max / CHARS_PER_TOKEN,
        },
        "truncation_rate": (bucket.truncated / bucket.total) if bucket.total else 0.0,
        "source_leak_rate": (bucket.source_leak_count / bucket.total) if bucket.total else 0.0,
        "semantic_fallback_usage_rate": (
            bucket.semantic_attempted / bucket.semantic_inspected if bucket.semantic_inspected else None
        ),
    }


def summarize(results: list[RetrievalResult]) -> dict[str, Any]:
    overall = AggregateBucket()
    by_category: dict[str, AggregateBucket] = defaultdict(AggregateBucket)
    semantic_reasons = Counter()
    for result in results:
        add_to_bucket(overall, result)
        add_to_bucket(by_category[result.category], result)
        semantic_reasons.update(result.semantic_reasons)
    return {
        "overall": bucket_summary(overall),
        "by_category": {name: bucket_summary(bucket) for name, bucket in sorted(by_category.items())},
        "semantic_reasons": dict(sorted(semantic_reasons.items())),
    }


def select_samples_and_questions(
    samples: list[PreparedSample],
    *,
    mode: str,
    max_samples: int | None,
    max_questions: int | None,
    include_adversarial: bool,
) -> list[PreparedSample]:
    """Apply sample/dev-loop limits without randomization."""
    selected = samples[:]
    if mode == "sample" and max_samples is None:
        max_samples = 1
    if mode == "sample" and max_questions is None:
        max_questions = 20
    if max_samples is not None:
        selected = selected[: max(0, max_samples)]
    output = []
    remaining = max_questions
    for sample in selected:
        qa_items = [qa for qa in sample.qa if include_adversarial or category_name(qa.category) != "adversarial"]
        if remaining is not None:
            qa_items = qa_items[: max(0, remaining)]
            remaining -= len(qa_items)
        output.append(PreparedSample(sample.sample_id, sample.conversation, qa_items))
        if remaining is not None and remaining <= 0:
            break
    return output


def git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(WORKSPACE),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""

    return {
        "commit": git("rev-parse", "HEAD"),
        "short_commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    binary = args.binary.resolve()
    if not binary.exists():
        raise SystemExit(f"memkeeper binary not found: {binary}")
    all_samples = load_locomo_dataset(args.dataset)
    samples = select_samples_and_questions(
        all_samples,
        mode=args.mode,
        max_samples=args.max_samples,
        max_questions=args.max_questions,
        include_adversarial=not args.exclude_adversarial,
    )
    if not samples or not any(sample.qa for sample in samples):
        raise SystemExit("no LoCoMo samples/questions selected")

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.store:
        store = args.store
        if store.exists() and not args.reuse_store:
            raise SystemExit(f"store exists; pass --reuse-store to reuse it: {store}")
        store.parent.mkdir(parents=True, exist_ok=True)
        if not args.reuse_store:
            memory_to_turn, _dia_to_memory = seed_store(binary, store, samples)
        else:
            raise SystemExit("--reuse-store is reserved for a future cached mapping implementation")
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="memkeeper-locomo-")
        store = Path(temp_dir.name) / "locomo.sqlite"
        memory_to_turn, _dia_to_memory = seed_store(binary, store, samples)

    qa_items = [qa for sample in samples for qa in sample.qa]
    if args.category:
        qa_items = [qa for qa in qa_items if category_name(qa.category) == args.category]
        if not qa_items:
            raise SystemExit(f"no questions in category: {args.category}")
    eval_kwargs = dict(
        max_memories=args.max_memories,
        max_chars=args.max_chars,
        query_mode=args.query_mode,
        sample_filter=args.sample_filter,
        inspect_semantic_flag=not args.no_semantic_inspect,
        rerank_candidates=args.rerank_candidates,
    )
    results = []
    try:
        if args.workers <= 1:
            for qa in qa_items:
                results.append(evaluate_one(binary, store, qa, memory_to_turn, **eval_kwargs))
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {
                    pool.submit(evaluate_one, binary, store, qa, memory_to_turn, **eval_kwargs): qa
                    for qa in qa_items
                }
                for fut in as_completed(futs):
                    results.append(fut.result())
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    summary = summarize(results)
    failures = [
        asdict(result)
        for result in results
        if result.evidence and (not result.hit or result.evidence_recall < 1.0 or result.source_leaks)
    ][: args.max_failures]
    ok = summary["overall"]["evidence_recall_at_k"] >= args.fail_under_recall and summary["overall"]["mrr"] >= args.fail_under_mrr
    report = {
        "ok": ok,
        "benchmark": "memkeeper-locomo-retrieval-v0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_info(),
        "settings": {
            "dataset": str(args.dataset),
            "mode": args.mode,
            "query_mode": args.query_mode,
            "sample_filter": args.sample_filter,
            "sample_count": len(samples),
            "question_count": len(results),
            "max_memories": args.max_memories,
            "max_chars": args.max_chars,
            "rerank_candidates": args.rerank_candidates,
            "exclude_adversarial": args.exclude_adversarial,
            "category": args.category,
            "semantic_inspect": not args.no_semantic_inspect,
            "fail_under_recall": args.fail_under_recall,
            "fail_under_mrr": args.fail_under_mrr,
            "workers": args.workers,
        },
        "store": {"temporary": args.store is None, "turn_memories": len(memory_to_turn)},
        "summary": summary,
        "failures": failures,
    }
    if args.emit_results:
        report["results"] = [asdict(result) for result in results]
    return report


def print_text_report(report: dict[str, Any]) -> None:
    settings = report["settings"]
    overall = report["summary"]["overall"]
    print(
        "memkeeper LoCoMo retrieval: "
        f"questions={settings['question_count']} samples={settings['sample_count']} "
        f"recall@{settings['max_memories']}={overall['evidence_recall_at_k']:.3f} "
        f"hit@{settings['max_memories']}={overall['hit_at_k']:.3f} "
        f"mrr={overall['mrr']:.3f} "
        f"p95={overall['pack_latency_ms']['p95']:.3f}ms "
        f"trunc={overall['truncation_rate']:.1%}"
    )
    print("category metrics:")
    for name, bucket in report["summary"]["by_category"].items():
        print(
            f"  {name:<12} n={bucket['total']:<4} "
            f"recall={bucket['evidence_recall_at_k']:.3f} "
            f"hit={bucket['hit_at_k']:.3f} mrr={bucket['mrr']:.3f} "
            f"p95={bucket['pack_latency_ms']['p95']:.3f}ms"
        )
    semantic_rate = overall["semantic_fallback_usage_rate"]
    if semantic_rate is not None:
        print(f"semantic fallback usage: {semantic_rate:.1%}; reasons={report['summary']['semantic_reasons']}")
    if report["failures"]:
        print("sample failures:")
        for failure in report["failures"][:5]:
            print(
                f"  {failure['qa_id']} {failure['category']} "
                f"evidence={failure['evidence_turn_ids']} retrieved={failure['retrieved_turn_ids']} "
                f"q={failure['question'][:100]}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Path to LoCoMo locomo10.json or flattened JSON/JSONL")
    parser.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--mode", choices=["retrieval", "sample"], default="retrieval")
    parser.add_argument(
        "--query-mode",
        choices=["bundle", "question", "hermes"],
        default="bundle",
        help="bundle = multi-query expansion (hermes is a deprecated alias)",
    )
    parser.add_argument(
        "--sample-filter",
        choices=["entity", "tag", "none"],
        default="entity",
        help="Filter retrieval to the QA sample conversation; use none to measure cross-sample contamination",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Limit conversations for dev runs")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit QA items for dev runs")
    parser.add_argument("--max-memories", type=int, default=20, help="Pack retrieval k")
    parser.add_argument("--max-chars", type=int, default=8000, help="Pack character budget")
    parser.add_argument(
        "--rerank-candidates",
        type=int,
        default=0,
        help=(
            "Cross-encoder rerank pool width for pack. 0 (default) omits the key and the binary "
            "still reranks over a narrow max-memories pool — the worst config. Use 50 for "
            "production parity. There is no no-rerank flag; unset MEMKEEPER_RERANK_MODEL_DIR instead."
        ),
    )
    parser.add_argument("--exclude-adversarial", action="store_true", help="Skip LoCoMo category 5")
    parser.add_argument("--category", default=None, help="Only run questions whose category name matches (e.g. adversarial)")
    parser.add_argument("--emit-results", action="store_true", help="Include every per-question record in the JSON report")
    parser.add_argument("--no-semantic-inspect", action="store_true", help="Skip batch-search semantic metadata check")
    parser.add_argument("--store", type=Path, default=None, help="Optional explicit output store; default is temp store")
    parser.add_argument("--reuse-store", action="store_true", help="Reserved for future cached-store runs")
    parser.add_argument("--fail-under-recall", type=float, default=0.0)
    parser.add_argument("--fail-under-mrr", type=float, default=0.0)
    parser.add_argument("--max-failures", type=int, default=20)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent pack requests to the daemon (default 1 = serial). "
             "2-6 is useful with a warm MEMKEEPER_BENCH_SOCK daemon; higher values "
             "queue at the model mutexes and add no throughput.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON report")
    args = parser.parse_args(argv)

    report = run_benchmark(args)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
