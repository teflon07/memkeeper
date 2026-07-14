# memkeeper Pi extension

Thin Pi adapter over the `memkeeper` Rust CLI, with support for the local `memkeeper serve --stdio` transport. The extension keeps durable memory semantics in Rust and registers explicit memory tools plus an explicit-prefix input hook; it does **not** auto-ingest sessions, auto-initialize stores, or call remote services.

When the bundled local ONNX models are present (`<memkeeper root>/models/mxbai-embed-large` and `.../mxbai-rerank-base`), the adapter self-provisions `MEMKEEPER_EMBED_MODEL_DIR` / `MEMKEEPER_RERANK_MODEL_DIR` and runs prompt-time auto-retrieve on the **embed → ANN + BM25 candidate fusion → cross-encoder rerank** path over a warm `serve --stdio` child (models load once per session; the child is warmed at startup). Without those model dirs it transparently falls back to BM25. All embedding/reranking is local ONNX — no remote calls.

## Quickstart

Run these from the repository root.

```bash
# Build the Rust CLI used by the adapter.
cargo build -p memkeeper-cli

# Initialize the store explicitly. The adapter will not create it for you.
target/debug/memkeeper init --store .memkeeper/store.sqlite --json

# Try the package for this Pi run only.
pi -e ./adapters/pi-extension
```

Once Pi starts, use `memory_doctor` or `memory_stats` first to verify the binary/store setup, then `memory_review`, `memory_search`, `memory_entity_search`, `memory_graph_neighbors`, `memory_remember`, `memory_get`, `memory_history`, or `memory_forget` as needed. Default routing is project-local store, `workspace-memory` space, `durable` silo, and `workspace` scope unless you explicitly override it. The adapter also performs prompt-time local retrieval by default and injects one transient source-hidden memory context message when relevant memories are found. You can explicitly capture a short memory by starting an input with `remember:`, `fact:`, `decision:`, `preference:`, `lesson:`, `action:`, or `revert:`.

## Install as a local Pi package

This directory is a local Pi package (`package.json` points Pi at `./index.ts`). To install it in project settings from the repository root:

```bash
pi install -l ./adapters/pi-extension
```

To install it in user settings instead, omit `-l`:

```bash
pi install ./adapters/pi-extension
```

Pi local paths are referenced in settings rather than copied. If you do copy `index.ts` or this package elsewhere, also set `MEMKEEPER_BIN`/`PI_MEMKEEPER_BIN` or `MEMKEEPER_ROOT`/`PI_MEMKEEPER_ROOT` so the adapter can find the Rust binary.

## Configuration

The adapter resolves the store path in this order:

1. per-tool `store` argument,
2. `MEMKEEPER_STORE`,
3. `PI_MEMKEEPER_STORE`,
4. `.memkeeper/store.sqlite` under Pi's current working directory.

The adapter resolves the binary in this order:

1. `MEMKEEPER_BIN`,
2. `PI_MEMKEEPER_BIN`,
3. `~/.local/libexec/memkeeper/current/memkeeper`,
4. `$MEMKEEPER_ROOT/target/release/memkeeper` or `$PI_MEMKEEPER_ROOT/target/release/memkeeper`,
5. `$MEMKEEPER_ROOT/target/debug/memkeeper` or `$PI_MEMKEEPER_ROOT/target/debug/memkeeper`,
6. `target/release/memkeeper` or `target/debug/memkeeper` under the versioned adapter's `memory/memkeeper` root,
7. `memkeeper` on `PATH`.

Optional timeout override for explicit tools:

```bash
export MEMKEEPER_TIMEOUT_MS=5000
```

Prompt-time retrieval is enabled by default. It calls local `memkeeper pack` before each sufficiently long prompt and injects one transient context message through Pi's `context` event; it does not write memory and does not persist the injected message in the session. It fails open on setup errors, timeouts, or no matches.

Disable prompt-time retrieval:

```bash
export MEMKEEPER_AUTO_RETRIEVE=0
# or
export PI_MEMKEEPER_AUTO_RETRIEVE=0
```

Prompt-time retrieval knobs:

| Var | Default | Description |
|-----|---------|-------------|
| `MEMKEEPER_HOOK_TIMEOUT_MS` | `4000` | Per-prompt retrieval timeout. Sized for warm embed+rerank of realistic memories; cold model load is masked by startup warmup. |
| `MEMKEEPER_HOOK_MAX_MEMORIES` | `5` | Max memories in the injected pack. |
| `MEMKEEPER_HOOK_MAX_CHARS` | `3000` | Max characters in the injected pack. |
| `MEMKEEPER_HOOK_RERANK_CANDIDATES` | `12` | Candidate pool the cross-encoder reranks after ANN+BM25 fusion (capped at 50). ~100ms/doc on CPU; 12 keeps warm retrieval ~1.5–2s while preserving quality. |
| `MEMKEEPER_HOOK_QUERY_EXPANSION` | `true` | Deterministically adds bounded subqueries before embedding/retrieval; set `0` to disable. |
| `MEMKEEPER_HOOK_THREAD_EXPANSION` | `true` | Adds same-entity/same-claim neighbors from top anchors into the rerank pool; set `0` to disable. |
| `MEMKEEPER_HOOK_MAX_QUERY_VARIANTS` | `8` | Cap on expanded query variants. |
| `MEMKEEPER_HOOK_MAX_THREAD_SEEDS` | `3` | Anchor count for same-thread expansion. |
| `MEMKEEPER_HOOK_MAX_THREAD_NEIGHBORS` | `3` | Neighbor count per anchor. |
| `MEMKEEPER_EMBED_MODEL_DIR` | bundled `models/mxbai-embed-large` if present | Local embedder dir. Set empty / `MEMKEEPER_EMBED_PROVIDER=none` to force BM25. |
| `MEMKEEPER_RERANK_MODEL_DIR` | bundled `models/mxbai-rerank-base` if present | Local cross-encoder reranker dir. |
| `MEMKEEPER_HOOK_MIN_PROMPT_CHARS` | `20` | Skip retrieval for shorter prompts. |
| `MEMKEEPER_HOOK_MAX_QUERY_CHARS` | `500` | Query prefix length from the submitted prompt. |
| `MEMKEEPER_HOOK_MIN_SCORE` | unset | Optional precision floor for `pack`. Leave unset by default: the floor applies to ANN/embedding scores on the rerank path (rerank itself only reorders, it does not threshold), and an over-tight floor drops real recall. When a floor yields no matches, the adapter skips injection entirely. |

Each var also accepts a `PI_` prefix variant, for example `PI_MEMKEEPER_HOOK_MAX_CHARS`.

Transport: when local models are available the adapter defaults to the persistent newline-JSON `stdio` transport (so the embedder/reranker load once per session instead of ~5s per CLI call). Override explicitly:

```bash
export MEMKEEPER_TRANSPORT=cli     # force one process per call (cold models each time)
export MEMKEEPER_TRANSPORT=stdio   # force the persistent serve child
# PI_-prefixed variants are also accepted
```

The stdio transport starts one local `memkeeper serve --stdio` child process owned by the Pi adapter, sends one JSON envelope per line, requires `request_id` echo, and falls back to the CLI transport on transport-level failures. Valid memkeeper error envelopes are not retried. If a mutating request has already been written to stdio and the final state is unknown, the adapter does not retry it through CLI; this avoids duplicating writes while preserving Rust-owned memory semantics. Disable stdio fallback with `MEMKEEPER_STDIO_FALLBACK=0` or `PI_MEMKEEPER_STDIO_FALLBACK=0` when debugging transport failures.

Prefix capture is enabled by default because it requires an explicit leading prefix. Disable it with either env var set to `0`, `false`, `off`, or `no`:

```bash
export MEMKEEPER_PREFIX_CAPTURE=0
# or
export PI_MEMKEEPER_PREFIX_CAPTURE=0
```

## Registered tools

- `memory_doctor` - read-only setup diagnostics for binary/config/store readiness.
- `memory_search` - deterministic SQLite FTS/BM25 search. `include_source` defaults to `false`.
- `memory_review` - list recent memories with ids for review/cleanup. Full content and source are hidden by default.
- `memory_entity_search` - search projected graph entities by key/name/alias/type/status. Source hidden by default.
- `memory_graph_neighbors` - traverse bounded projected graph neighbors from one entity id/key. Source hidden by default.
- `memory_remember` - explicit user-approved memory write with duplicate/update candidate reporting and `entity_key` projection. No auto-ingest.
- `memory_get` - retrieve one memory by id. History/source hidden by default.
- `memory_history` - inspect audit events and versions for one memory id. Source hidden by default.
- `memory_forget` - tombstone one memory by id; audit history is preserved. Use `dry_run=true` to preview.
- `memory_stats` - inspect store health/counts. Index details are omitted unless `include_indexes=true`.

By default, all tools spawn the Rust CLI with argv arrays. With `MEMKEEPER_TRANSPORT=stdio`, tools use a persistent local `memkeeper serve --stdio` child process and retain CLI as fallback for transport failures. Both transports parse the strict JSON envelope, return compact bounded text to the model, and keep the full envelope in tool `details` for UI/session inspection. When `include_source=true`, those details may include provenance/source metadata and local paths; leave source disabled unless explicitly requested. If `memory_remember` reports candidate memories, inspect them before deciding whether a later write should use explicit `supersedes` links; v0.1 does not auto-merge or auto-supersede candidates.

## Explicit prefix capture

When an input begins with one of these exact lowercase prefixes, the adapter captures exactly one text memory by calling the Rust `remember` command and then handles the input without sending it to the LLM:

- `remember:` / `fact:` → `kind=fact`
- `decision:` / `revert:` → `kind=decision`
- `preference:` → `kind=preference`
- `lesson:` → `kind=lesson`
- `action:` → `kind=task`

Prefix capture is intentionally conservative: max 4,096 characters, text-only, no embeddings/LLM/network, no auto-init, one memory per input, and common secrets-looking text is rejected rather than stored. The memory receives a `prefix-capture` tag and source metadata identifying the capture mechanism; source remains hidden unless `include_source=true`.

## Cleanup workflow

1. Use `memory_review` to list recent active memories and copy the exact id.
2. Use `memory_get` for full current content or `memory_history` for audit events/versions. Leave `include_source=false` unless the user explicitly asks for provenance.
3. Use `memory_forget` only after an explicit user request for a specific id. `dry_run=true` previews the tombstone result; a committed forget preserves audit history and is not a hard delete.

## Safety defaults

- No broad automatic writes: `memory_remember` only runs when Pi chooses or the user explicitly asks for a memory write, and prefix capture only runs for explicit leading memory prefixes.
- No automatic initialization: create the SQLite store with `memkeeper init` first.
- Prompt-time retrieval is read-only, bounded, source-hidden, and fail-open; disable with `MEMKEEPER_AUTO_RETRIEVE=0` if a run should have no automatic recall.
- No prompt-time extraction/dream/embedding writes: search/get/stats/retrieval use bounded local CLI calls.
- Source privacy: `include_source=false` hides provenance and uses source-free search ranking in the Rust core.
- Forget means tombstone in v0.1; audit/history is preserved.
