# Memkeeper compact reranker pipeline

This directory trains a small cross-encoder specifically for Memkeeper retrieval. The design goal is a reranker that is smaller and faster than a general-purpose model while improving accuracy on memory-specific distinctions such as supersession, project scope, provenance, durable-vs-temporary state, and rejected alternatives.

## Pipeline

1. Export retrieval traces as JSONL using the schema below.
2. Build pairwise examples and hard negatives.
3. Distill scores from a stronger teacher reranker when explicit labels are absent.
4. Fine-tune a compact cross-encoder.
5. Evaluate against the current production reranker.
6. Export to ONNX and optionally quantize to INT8.

The first target should be a 6-layer MiniLM-class encoder. Do not train from scratch. Distill and fine-tune an existing compact checkpoint.

## Input JSONL

Each line represents one query and its candidate set:

```json
{
  "query_id": "q-001",
  "query": "Why did we reject Zep?",
  "candidates": [
    {
      "memory_id": "m-123",
      "text": "Zep was rejected because the hosted plan cost $125 per month.",
      "label": 3,
      "retrieval_score": 0.71,
      "metadata": {
        "project": "memkeeper",
        "memory_type": "decision",
        "status": "current",
        "has_provenance": true,
        "superseded": false
      }
    }
  ]
}
```

Labels:

- `3`: directly answers the query
- `2`: useful supporting context
- `1`: related but not useful enough
- `0`: irrelevant
- `-1`: actively misleading, stale, superseded, or wrong-scope

If `label` is omitted, `build_dataset.py` can use teacher scores. Explicit human labels always win.

## Install

```bash
cd tools/reranker
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build training data

```bash
python build_dataset.py \
  --input traces.jsonl \
  --output data/pairs.jsonl \
  --teacher cross-encoder/ms-marco-MiniLM-L-12-v2 \
  --negatives-per-positive 4
```

Hard negatives are selected from candidates that rank highly under the current retriever but have low or negative relevance labels. This is more valuable than random negatives for Memkeeper.

## Train

```bash
python train.py \
  --train data/pairs.jsonl \
  --base-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --output artifacts/memkeeper-reranker \
  --epochs 3 \
  --batch-size 32 \
  --max-length 384
```

## Evaluate

```bash
python evaluate.py \
  --input heldout.jsonl \
  --model artifacts/memkeeper-reranker \
  --baseline cross-encoder/ms-marco-MiniLM-L-6-v2
```

Primary metrics are `nDCG@10`, `MRR`, `Recall@20`, stale-memory rate, wrong-project rate, and superseded-memory rate. Promote a model only when it improves ranking quality without increasing policy errors.

## Export and quantize

```bash
python export_onnx.py \
  --model artifacts/memkeeper-reranker \
  --output artifacts/memkeeper-reranker-onnx \
  --quantize
```

The export produces a tokenizer directory plus `model.onnx` or `model.int8.onnx`. Benchmark both. INT8 may improve CPU latency substantially, but promotion should be based on measured quality and latency rather than model size alone.

## Recommended experiment order

1. Keep the existing embedding model fixed.
2. Train only the reranker.
3. Compare 6-layer, 4-layer, and distilled 2-layer variants.
4. Quantize the best accuracy/latency candidate.
5. Fine-tune embeddings only if candidate recall remains the bottleneck.

Use query-grouped train/validation/test splits. Never place candidates from the same query in different splits, because that leaks query-specific wording and inflates results.
