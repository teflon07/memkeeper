#!/usr/bin/env python3
"""Build pairwise reranker training examples from Memkeeper retrieval traces."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return rows


def teacher_score_missing(rows: list[dict[str, Any]], model_name: str) -> None:
    model = CrossEncoder(model_name)
    pairs: list[tuple[str, str]] = []
    refs: list[dict[str, Any]] = []
    for row in rows:
        query = row["query"]
        for candidate in row.get("candidates", []):
            if "label" not in candidate:
                pairs.append((query, candidate["text"]))
                refs.append(candidate)
    if not pairs:
        return
    scores = model.predict(pairs, batch_size=32, show_progress_bar=True)
    for candidate, score in zip(refs, scores, strict=True):
        candidate["teacher_score"] = float(score)


def candidate_strength(candidate: dict[str, Any]) -> float:
    if "label" in candidate:
        return float(candidate["label"])
    return float(candidate.get("teacher_score", 0.0))


def build_pairs(
    rows: list[dict[str, Any]], negatives_per_positive: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output: list[dict[str, Any]] = []

    for row in rows:
        query = row["query"]
        query_id = row.get("query_id")
        candidates = row.get("candidates", [])
        positives = [c for c in candidates if candidate_strength(c) >= 2.0]
        negatives = [c for c in candidates if candidate_strength(c) <= 1.0]

        # Hardest negatives first: high retriever score, then explicit misleading label.
        negatives.sort(
            key=lambda c: (
                float(c.get("retrieval_score", 0.0)),
                1.0 if c.get("label") == -1 else 0.0,
            ),
            reverse=True,
        )

        for positive in positives:
            pool = negatives[: max(negatives_per_positive * 3, negatives_per_positive)]
            chosen = rng.sample(pool, k=min(negatives_per_positive, len(pool))) if pool else []
            for negative in chosen:
                output.append(
                    {
                        "query_id": query_id,
                        "query": query,
                        "positive_id": positive.get("memory_id"),
                        "positive": positive["text"],
                        "negative_id": negative.get("memory_id"),
                        "negative": negative["text"],
                        "positive_score": candidate_strength(positive),
                        "negative_score": candidate_strength(negative),
                        "negative_metadata": negative.get("metadata", {}),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--teacher", default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    teacher_score_missing(rows, args.teacher)
    pairs = build_pairs(rows, args.negatives_per_positive, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"wrote {len(pairs)} training pairs to {args.output}")


if __name__ == "__main__":
    main()
