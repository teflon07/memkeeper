<p align="center">
  <img src="assets/logo.png" alt="memkeeper logo" width="180" height="180" />
</p>

<p align="center"><em>Most software has the memory of a goldfish. This one doesn&rsquo;t.</em></p>

# memkeeper

Local-first memory for AI agents. A fast, embeddable memory engine that stores,
ranks, and retrieves an agent's durable context, entirely on your machine, with
no required network or LLM calls.

> ℹ️ **Generated release mirror.** This repo is generated from a private
> development repo and published as releases. The `main` branch may be
> regenerated, so **pin to tagged releases** (or the release artifacts) rather
> than to arbitrary `main` commits — tagged releases are stable. See
> [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute; issues, security
> reports, and design feedback are the best paths today.

- **Local-first.** A single SQLite database. No server, no cloud, no telemetry.
- **Fast at prompt time.** Deterministic BM25/FTS retrieval with optional ONNX
  semantic embeddings and a cross-encoder reranker.
- **Durable by design.** Atomic writes, schema-versioned storage, and a
  retention model that promotes recurring, high-signal memories to a durable tier.

> Status: pre-release (v0.2). APIs and the wire protocol may change before 1.0.

## Prerequisites

- **Rust toolchain** (stable, via [rustup](https://rustup.rs)) — provides `cargo`,
  which builds the CLI. The crates are edition 2021, so Rust 1.56 or newer.
- **A C toolchain**, for the native dependencies a default build compiles
  (bundled SQLite, plus the ONNX runtime for semantic search). On macOS: Xcode
  Command Line Tools (`xcode-select --install`); on Debian/Ubuntu: `build-essential`.
- **No network, LLM, or API key is needed at runtime.** Building fetches crate
  dependencies from crates.io the first time, like any Rust project; after that a
  clean build is offline. The ONNX models for semantic search are fetched
  separately (see below), and `pull-models` needs `curl`. A `--no-default-features`
  build is lexical-only and needs neither the models nor `curl`.

## Quickstart

```sh
# Build the CLI. Semantic search + rerank is ON by default and needs the ONNX
# models (see below); for a lexical-only binary use `--no-default-features`.
cargo build --release

# Initialize a store and remember something
./target/release/memkeeper init --store ~/.memkeeper/store.sqlite --json
./target/release/memkeeper remember --store ~/.memkeeper/store.sqlite \
  --json '{"content":"memkeeper stores memories in a local SQLite database"}'

# Search it back
./target/release/memkeeper search --store ~/.memkeeper/store.sqlite \
  --json '{"query":"where are memories stored","limit":3}'
```

The store defaults to `~/.memkeeper/store.sqlite` when `--store` is omitted.

## What to store

memkeeper holds **self-contained memories**: facts, decisions, preferences,
lessons. Each `remember` is one memory written to stand on its own, with its
context and intent intact (store "the user likes pineapple on pizza," not just
"pineapple"). Atomic means one idea per memory, not a stripped keyword.
Retrieval, dedup, supersession, and the entity graph all work best at that grain.

Two ends to avoid:

- **Too small:** a bare keyword or fragment that drops the point.
- **Too large:** a whole document. The curated memory tier has no chunking, and
  the embedder sees only the first ~512 tokens of an entry, so loading long files
  (for example, an entire markdown library) gives weak semantic recall on those
  entries (lexical BM25 still indexes the full text). To bring whole documents in,
  don't store them as memories — use the [document store](#document-store-rag),
  memkeeper's separate RAG tier that chunks and embeds files into an isolated
  space (the [`memkeeper-ingest`](https://github.com/teflon07/memkeeper-ingest)
  add-on imports whole folders this way). Or distill the document down to its
  takeaways and store those as memories.

## Capturing memories

memkeeper is **curated** memory you populate deliberately — not an automatic
transcript logger. Memories get in two ways:

- **Directly** — `memkeeper remember --json '{"content":"…"}'`, from the CLI or a
  script.
- **From an agent** — the [MCP bridge](adapters/mcp) lets an MCP client (Claude and
  other agents) call `remember` during a session, so durable facts are captured as
  they come up.

On the *retrieval* side, `memkeeper hook retrieve` is a Claude Code
UserPromptSubmit hook client that injects relevant memories into the prompt — so an
agent recalls without an explicit search. It retrieves; capture stays a deliberate
`remember`.

## Semantic retrieval (default)

memkeeper has three retrieval modes. **Local semantic is the default and the
recommended, fully on-device mode.** Pick one up front — the embedding backend is
recorded in the store, so changing it means re-embedding (`reindex --embed`), not a
flip.

| Mode | Network | Setup |
|---|---|---|
| **Local semantic** (default) | none | build from source, `pull-models` |
| **Lexical only** | none | no models; `--no-default-features`, or just skip model setup |
| **Off-device semantic** | embeds via an API | set `MEMKEEPER_EMBED_PROVIDER=openai` + base URL + key |

> **Privacy:** off-device semantic sends your memory **text** to the embeddings
> provider to be vectorized. Use it only where that is acceptable; the two on-device
> modes never send memory content anywhere.

### Local semantic (default)

Semantic embeddings + cross-encoder reranking are enabled by default, so a plain
`cargo build --release` produces a semantic binary. It needs the ONNX models,
which are not downloaded automatically. Fetch them once with `pull-models` or
the script.

`cargo build --release` does not put `memkeeper` on your `PATH`; the binary lands
at `./target/release/memkeeper`. Either call it by that path (as below) or install
it to your `PATH` with `cargo install --path crates/memkeeper-cli`, after which a
bare `memkeeper` works.

```sh
# Fetch the embed + rerank models (needs curl; ~2.1GB, or --quantized for ~0.6GB).
# Either the built-in subcommand:
./target/release/memkeeper pull-models
# ...or, equivalently, the bundled script if you have the repo checked out:
scripts/fetch-models.sh

# Both print the two env vars to export (MEMKEEPER_EMBED_MODEL_DIR /
# MEMKEEPER_RERANK_MODEL_DIR); set them so `memkeeper serve` runs with semantics on.
```

If a semantic build runs with the models missing, memkeeper does not degrade
*silently*: it logs a loud WARNING and marks results as semantic-unavailable
(e.g. `"semantic":{"attempted":false,"reason":"missing_embedding"}`). By default
it still **falls back to lexical (BM25/FTS)** retrieval so search keeps working.
Set `MEMKEEPER_REQUIRE_SEMANTIC=1` to **fail closed** instead — refuse the request
rather than serve degraded results. Use that in any deployment that must never
silently run lexical-only.

Embeddings are computed when a memory is **written**. Memories you stored before
the models were present (for example, the one from the Quickstart above) are
lexical-only and won't appear in semantic-only queries until they are embedded.
After setting the model env vars, backfill existing memories once with:

```sh
./target/release/memkeeper reindex --embed --store ~/.memkeeper/store.sqlite
```

New memories written with the models in place are embedded automatically.

### Lexical only

To build a deterministic, model-free **lexical-only** (BM25/FTS) binary, disable
the default feature:

```sh
cargo build --release --no-default-features
```

### Off-device semantic (no model download)

To get semantic quality without downloading the ONNX models, point memkeeper at an
OpenAI-compatible embeddings API (OpenAI, OpenRouter, or any compatible proxy). This
mode embeds and reranks over the network instead of loading local models, so it
needs no `pull-models` and carries no ONNX runtime:

```sh
# Embeddings (required for semantic): any OpenAI-compatible /embeddings endpoint.
export MEMKEEPER_EMBED_PROVIDER=openai     # "openai" = the OpenAI-compatible API dialect
export MEMKEEPER_EMBED_BASE_URL=https://api.openai.com/v1/embeddings   # or your provider, e.g. OpenRouter
export MEMKEEPER_EMBED_API_KEY=sk-...
export MEMKEEPER_EMBED_MODEL=text-embedding-3-small
export MEMKEEPER_EMBED_DIMS=1536

# Reranking (optional, recommended): Cohere /rerank dialect, which OpenRouter speaks.
export MEMKEEPER_RERANK_PROVIDER=openrouter
export MEMKEEPER_RERANK_API_KEY=sk-...
export MEMKEEPER_RERANK_MODEL=cohere/rerank-v3.5
```

The **prebuilt release binaries are built this way** (`--features api`): with a key
configured they do off-device semantic; with no key they serve lexical (BM25/FTS),
and `MEMKEEPER_REQUIRE_SEMANTIC=1` makes them refuse rather than serve degraded.
They carry no local model runtime, so for fully on-device semantic, build from
source (`cargo build --release`) and use local models as above.

Prebuilt binaries are published for **macOS (Apple Silicon)** and **Linux x86_64**.
**Windows is experimental** — there's no prebuilt binary, but it builds and runs
from source; see [docs/windows.md](docs/windows.md). (`serve --socket` is Unix-only
there; the http dashboard and stdio serve are cross-platform.)

### Switching the embedding model

The embedding backend is recorded per store, and memkeeper refuses to mix vectors
from different models (they live in different vector spaces). To switch — local↔
off-device, or between models — change the embedding env vars, then re-embed every
memory under the new model in one step:

```sh
./target/release/memkeeper reindex --embed --store ~/.memkeeper/store.sqlite
```

This wipes the old vectors, records the new active model, and re-embeds all active
memories in one transaction. It is the supported way to change models; there is no
partial mix.

## Document store (RAG)

Alongside curated memories, memkeeper can hold a separate tier of ingested
document chunks for retrieval-augmented use. Chunks live in their own space
(default `documents`), isolated from the curated memory tier, so they never
receive supersession, dedup, graph, or promotion treatment.

- `ingest` — store a document source as embedded, isolated chunks. Re-ingesting
  the same `source_path` repairs that chunk's provenance in place; identical
  content under a different path is kept as an independent chunk.
- `document-search` — hybrid (BM25 + vector) search over the chunks, with a
  citation back to `source_path` and chunk index.
- `document-get` — fetch a document's chunks by path, or one chunk by id.
- `document-duplicates` — surface exact-content duplicate chunks (the same
  content held under different sources) as clusters. `stats` also reports a
  `document_duplicate_clusters` count so you know when there are duplicates worth
  reviewing.
- `document-prune` — delete the specific chunks you choose (supports `dry_run`).
  Deletion is always explicit: review duplicates, decide which copies to keep,
  then prune the rest.
- `promotion-candidates` / `mark-extracted` — rank chunks that earned retrieval
  traffic, and mark a chunk extracted once it has been promoted into a memory.

Run `memkeeper schema <command>` for each command's accepted JSON fields. Over
`serve --http`, reads (search/get/duplicates) are available on the read-only
dashboard. Writes (`ingest`, `document-prune`) are disabled unless you set a
write token: start the server with `MEMKEEPER_HTTP_WRITE_TOKEN=<secret>` in the
environment, then send it on write requests as `Authorization: Bearer <secret>`.
With no token set, the HTTP server is read-only.

## The dashboard

`memkeeper serve --http` starts a read-only local dashboard (default
`http://127.0.0.1:7777`) for browsing memories and the entity graph.

**A fresh store starts empty — that's expected.** Two views, populated
differently:

- The **memory list** fills as you `remember`.
- The **graph** visualizes *entities and relationships*, which are a separate layer
  from raw memories. They accrue from memories that carry an entity, from the
  `dream` synthesis pipeline extracting them over time, or from explicit
  `entity-upsert` / `relationship-upsert`. So a handful of plain memories fill the
  list but leave the **Graph** tab empty until entities and links exist — add a few
  connected memories (or run `dream`) and it builds out.

## Benchmarks

On [LoCoMo](https://github.com/snap-research/locomo) (10 multi-session dialogues,
1,982 evidence-bearing questions), memkeeper's default semantic retrieval scores:

| Metric | Score |
|---|---|
| recall@20 | 0.768 |
| hit@20 | 0.880 |
| MRR | 0.668 |

Prompt-time search on a warm `serve` daemon (ONNX models loaded once) runs in
**~25 ms** p50/p95, about 32× faster than a cold per-call binary that reloads the
models on every query.

Full methodology, per-config results (including the late-interaction upgrade), and
a reproduction script are in [docs/benchmarks.md](docs/benchmarks.md).

## Workspace layout

| Crate | Role |
|---|---|
| `memkeeper-core` | Core types and retrieval policy |
| `memkeeper-store` | SQLite storage, schema, indexing, promotion |
| `memkeeper-embed` | ONNX embeddings + cross-encoder reranker |
| `memkeeper-protocol` | Wire protocol (`memkeeper.v0.1`) |
| `memkeeper-cli` | The `memkeeper` binary (CLI + daemon) |

Editor/agent integrations live under `adapters/` (an MCP bridge and a thin
extension client).

## Memkeeper family

[Warden](https://github.com/teflon07/memkeeper-warden) is a companion capability
broker and execution gate: it decides whether an agent's requested action (a
shell command, a file read/write) is allowed by a declared, auditable policy, and
logs every decision. memkeeper remembers; Warden guards.

## License

Dual-licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions require signing the project
[Contributor License Agreement](docs/CLA.md) — the CLA bot prompts you on your
first pull request. You keep the copyright to your contributions.
