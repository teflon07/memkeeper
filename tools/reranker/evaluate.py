#!/usr/bin/env python3
"""Evaluate a reranker on query-grouped Memkeeper traces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dcg(labels: list[float], k: int) -> float:
    return sum((2.0**rel - 1.0) / math.log2(index + 2.0) for index, rel in enumerate(labels[:k]))


def ndcg(labels: list[float], k: int) -> float:
    ideal = dcg(sorted(labels, reverse=True), k)
    return dcg(labels, k) / ideal if ideal else 0.0


def score_model(model: CrossEncoder, rows: list[dict[str, Any]]) -> dict[str, float]:
    ndcgs: list[float] = []
    reciprocal_ranks: list[float] = []
    recall20: list[float] = []
    stale_top5 = wrong_project_top5 = superseded_top5 = 0
    top5_total = 0

    for row in rows:
        candidates = row.get("candidates", [])
        if not candidates:
            continue
        pairs = [(row["query"], candidate["text"]) for candidate in candidates]
        scores = model.predict(pairs, batch_size=64, show_progress_bar=False)
        ranked = [candidate for _, candidate in sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)]
        labels = [float(candidate.get("label", 0)) for candidate in ranked]
        relevant_total = sum(label >= 2 for label in labels)
        relevant_top20 = sum(label >= 2 for label in labels[:20])

        ndcgs.append(ndcg(labels, 10))
        first_relevant = next((index + 1 for index, label in enumerate(labels) if label >= 2), None)
        reciprocal_ranks.append(1.0 / first_relevant if first_relevant else 0.0)
        recall20.append(relevant_top20 / relevant_total if relevant_total else 1.0)

        for candidate in ranked[:5]:
            metadata = candidate.get("metadata", {})
            stale_top5 += int(metadata.get("status") in {"stale", "obsolete"})
            wrong_project_top5 += int(bool(metadata.get("wrong_project", False)))
            superseded_top5 += int(bool(metadata.get("superseded", False)))
            top5_total += 1

    denom = max(top5_total, 1)
    return {
        "nDCG@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "MRR": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
        "Recall@20": float(np.mean(recall20)) if recall20 else 0.0,
        "stale_top5_rate": stale_top5 / denom,
        "wrong_project_top5_rate": wrong_project_top5 / denom,
        "superseded_top5_rate": superseded_top5 / denom,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    results = {"candidate": score_model(CrossEncoder(args.model), rows)}
    if args.baseline:
        results["baseline"] = score_model(CrossEncoder(args.baseline), rows)
        results["delta"] = {
            key: results["candidate"][key] - results["baseline"][key]
            for key in results["candidate"]
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
