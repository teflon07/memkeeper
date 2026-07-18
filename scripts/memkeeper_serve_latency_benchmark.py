#!/usr/bin/env python3
"""Compare memkeeper semantic query latency for CLI vs persistent serve --stdio.

The benchmark needs a semantic-enabled memkeeper binary and local model files:

    cargo build --release --features semantic
    python3 memory/memkeeper/scripts/memkeeper_serve_latency_benchmark.py \
        --embed-model-dir memory/memkeeper/models/mxbai-embed-large

It seeds a temporary store through one serve process, then measures repeated
semantic search requests through fresh CLI invocations and one persistent
serve --stdio process. The expected optimization is that serve pays ONNX model
load once at process startup instead of once per query.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import harness_lib

WORKSPACE = Path(__file__).resolve().parents[3]
DEFAULT_BIN = WORKSPACE / "memory" / "memkeeper" / "target" / "release" / "memkeeper"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, round((len(values) - 1) * pct))
    return values[index]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": round(min(values), 1),
        "p50_ms": round(statistics.median(values), 1),
        "p95_ms": round(percentile(values, 0.95), 1),
        "max_ms": round(max(values), 1),
        "avg_ms": round(sum(values) / len(values), 1),
    }


def run_cli(binary: Path, store: Path, command: str, payload: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    args = [str(binary), command, "--store", str(store), "--json"]
    if payload:
        args.append(json.dumps(payload, separators=(",", ":")))
    result = subprocess.run(
        args,
        cwd=WORKSPACE,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return json.loads(result.stdout)


class ServeClient:
    def __init__(self, binary: Path, env: dict[str, str]) -> None:
        self.proc = subprocess.Popen(
            [str(binary), "serve", "--stdio"],
            cwd=WORKSPACE,
            env=env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.next_id = 0

    def request(self, command: str, store: Path, payload: dict[str, Any]) -> dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("serve process pipes are unavailable")
        self.next_id += 1
        request = {
            "request_id": f"bench-{self.next_id}",
            "command": command,
            "store_path": str(store),
            "payload": payload,
        }
        self.proc.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr is not None else ""
            raise RuntimeError(f"serve process stopped without a response: {stderr}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(json.dumps(response, indent=2))
        return response

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)


def seed_store(binary: Path, store: Path, env: dict[str, str], memories: int) -> None:
    client = ServeClient(binary, env)
    try:
        client.request("init", store, {})
        for index in range(memories):
            topic = "semantic retrieval latency" if index % 3 == 0 else "workspace memory operations"
            client.request(
                "remember",
                store,
                {
                    "space": "workspace-memory",
                    "silo": "durable",
                    "scope": "workspace",
                    "kind": "fact",
                    "content": f"fact: benchmark memory {index} about {topic} and serve stdio reuse.",
                    "entity_key": "benchmark:serve-latency",
                },
            )
    finally:
        client.close()


def search_payload(query: str, rerank: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"query": query, "limit": 5, "semantic_fallback": "fallback"}
    if rerank:
        payload["rerank"] = True
    return payload


def measure_cli(
    binary: Path,
    store: Path,
    env: dict[str, str],
    queries: list[str],
    runs: int,
    rerank: bool,
) -> list[float]:
    latencies = []
    for index in range(runs):
        payload = search_payload(queries[index % len(queries)], rerank)
        started = time.perf_counter()
        run_cli(binary, store, "search", payload, env)
        latencies.append((time.perf_counter() - started) * 1000.0)
    return latencies


def measure_serve(
    binary: Path,
    store: Path,
    env: dict[str, str],
    queries: list[str],
    runs: int,
    warmup: int,
    rerank: bool,
) -> list[float]:
    client = ServeClient(binary, env)
    try:
        for index in range(warmup):
            client.request("search", store, search_payload(queries[index % len(queries)], rerank))
        latencies = []
        for index in range(runs):
            payload = search_payload(queries[index % len(queries)], rerank)
            started = time.perf_counter()
            client.request("search", store, payload)
            latencies.append((time.perf_counter() - started) * 1000.0)
        return latencies
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=DEFAULT_BIN)
    parser.add_argument(
        "--embed-model-dir",
        type=Path,
        default=None,
        help="Local ONNX embed model dir; omit to use MEMKEEPER_EMBED_* provider env (e.g. MEMKEEPER_EMBED_PROVIDER=openai)",
    )
    parser.add_argument("--rerank-model-dir", type=Path)
    parser.add_argument("--rerank", action="store_true", help="Set rerank:true on each search request")
    parser.add_argument("--memories", type=int, default=30)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--results", type=Path, default=None, help="JSONL checkpoint file (one record per arm)")
    parser.add_argument("--resume", action="store_true", help="skip arms already recorded in --results")
    args = parser.parse_args()

    env = os.environ.copy()
    if args.embed_model_dir:
        env["MEMKEEPER_EMBED_MODEL_DIR"] = str(args.embed_model_dir)
    elif not env.get("MEMKEEPER_EMBED_PROVIDER"):
        print(
            "ERROR: no --embed-model-dir and MEMKEEPER_EMBED_PROVIDER is unset; "
            "refusing to run a lexical-only latency benchmark",
            file=sys.stderr,
        )
        return 2
    if args.rerank_model_dir:
        env["MEMKEEPER_RERANK_MODEL_DIR"] = str(args.rerank_model_dir)

    queries = [
        "semantic retrieval latency",
        "serve stdio model reuse",
        "workspace memory operations",
    ]

    done: set[str] = set()
    prior: dict[str, dict] = {}
    if args.results is not None:
        if args.results.exists() and not args.resume:
            raise SystemExit(
                f"results file exists; pass --resume to continue it or use a fresh --results path: {args.results}"
            )
        if args.resume:
            prior = harness_lib.latest_records(harness_lib.iter_jsonl_records(args.results), key="arm")
            done = {arm for arm, record in prior.items() if record.get("status") == "ok"}
    elif args.resume:
        raise SystemExit("--resume requires --results")

    cli_latencies = prior["cli"]["latencies_ms"] if "cli" in done else None
    serve_latencies = prior["serve"]["latencies_ms"] if "serve" in done else None

    if cli_latencies is None or serve_latencies is None:
        with tempfile.TemporaryDirectory(prefix="memkeeper_serve_latency_") as tmpdir:
            store = Path(tmpdir) / "store.sqlite"
            seed_store(args.binary, store, env, args.memories)
            handle = args.results.open("a", encoding="utf-8") if args.results else None
            try:
                if cli_latencies is None:
                    cli_latencies = measure_cli(args.binary, store, env, queries, args.runs, args.rerank)
                    if handle is not None:
                        harness_lib.append_result(handle, {"arm": "cli", "status": "ok", "latencies_ms": cli_latencies})
                if serve_latencies is None:
                    serve_latencies = measure_serve(args.binary, store, env, queries, args.runs, args.warmup, args.rerank)
                    if handle is not None:
                        harness_lib.append_result(handle, {"arm": "serve", "status": "ok", "latencies_ms": serve_latencies})
            finally:
                if handle is not None:
                    handle.close()

    result = {
        "binary": str(args.binary),
        "runs": args.runs,
        "memories": args.memories,
        "rerank": args.rerank,
        "cli_search": stats(cli_latencies),
        "serve_stdio_search": stats(serve_latencies),
        "p50_speedup": round(statistics.median(cli_latencies) / statistics.median(serve_latencies), 2),
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"CLI search p50: {result['cli_search']['p50_ms']} ms")
        print(f"serve --stdio search p50: {result['serve_stdio_search']['p50_ms']} ms")
        print(f"p50 speedup: {result['p50_speedup']}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
