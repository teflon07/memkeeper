# memkeeper benchmarks

Point measurements on a single machine with the semantic build
(`cargo build --release`), the default `mxbai-embed-large` embedder and
`mxbai-rerank-base` cross-encoder reranker. These are reproducible single-machine
numbers, not a controlled cross-system comparison.

## Retrieval quality — LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) (CC BY-NC 4.0) is a long-term
conversational QA benchmark. We seed every dialogue turn as a memory, then for
each question run memkeeper's bounded `pack` retrieval and score whether the
annotated evidence turns come back in the top 20 (recall@20 / hit@20 / MRR).

Full `locomo10` set: 10 dialogues, 5,882 turn-memories, 1,982 evidence-bearing
questions. Config: `--max-memories 20 --max-chars 8000 --rerank-candidates 50`.

| Config | recall@20 | hit@20 | MRR |
|---|---|---|---|
| Default (semantic + rerank) | 0.768 | 0.880 | 0.668 |
| + late-interaction (ColBERT) | 0.784 | 0.894 | 0.666 |

The **default** row is the out-of-the-box configuration: what `cargo build
--release` plus `memkeeper pull-models` gives you. Late-interaction adds a ColBERT
MaxSim pass on top; it is off by default and its model is not fetched by
`pull-models` (see `scripts/fetch-models.sh`), so treat the late-interaction row
as an optional upgrade rather than the baseline.

## Latency

Semantic search through a warm `serve` daemon (ONNX models loaded once at startup)
versus a cold per-call binary that reloads the models on every query:

| Path | p50 | p95 |
|---|---|---|
| Warm `serve` search | 24.9 ms | 25.5 ms |
| Cold per-call CLI search | 799 ms | 815 ms |

(30-memory store, 30 runs.) The warm daemon is ~32× faster — run memkeeper as a
persistent `serve` process for prompt-time retrieval. The heavier semantic `pack`
path (a 50-candidate rerank over the full LoCoMo store) runs roughly 4 s per query
sequentially; it is distinct from the lightweight search measured above.

## Reproduce

1. Build the semantic binary and fetch the models:
   ```sh
   cargo build --release
   scripts/fetch-models.sh          # mxbai embed + rerank, ~2.1GB
   ```
2. Get the dataset (not vendored; CC BY-NC 4.0): download `locomo10.json` from
   <https://github.com/snap-research/locomo>.
3. Start a warm daemon so the models load once, then drive the harness over its
   socket:
   ```sh
   export MEMKEEPER_EMBED_MODEL_DIR=~/.memkeeper/models/mxbai-embed-large
   export MEMKEEPER_RERANK_MODEL_DIR=~/.memkeeper/models/mxbai-rerank-base
   ./target/release/memkeeper serve --socket /tmp/mk-bench.sock &

   MEMKEEPER_BENCH_SOCK=/tmp/mk-bench.sock \
     python3 scripts/memkeeper_locomo_benchmark.py \
       --dataset path/to/locomo10.json \
       --binary ./target/release/memkeeper \
       --max-memories 20 --max-chars 8000 --rerank-candidates 50 --json
   ```
   Without a warm `MEMKEEPER_BENCH_SOCK` daemon the binary reloads the ONNX models
   on every query and a full run takes hours instead of minutes.

The harness is retrieval-only: it scores whether the evidence turns are retrieved,
and does not call an answering model or judge.
