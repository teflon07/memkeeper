# Changelog

All notable changes to memkeeper are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches
1.0. Until then, minor releases may include breaking changes to the storage
schema and wire protocol.

## [0.2.5] - 2026-06-28

### Added
- **Alias-exact-match retrieval boost.** A query token that exactly matches a
  memory's reserved `alias::<normalized>` tag now adds a fixed boost to that
  memory's retrieval score, lifting exact alias hits (e.g. "k8s" → the Kubernetes
  memory) above semantically-similar neighbors near the abstention floor. Matching
  uses normalized 1–3-word query shingles, so multi-word aliases ("dead letter
  queue") resolve too. Reuses tag storage; no schema or wire-protocol change.

## [0.2.4] - 2026-06-27

### Changed
- **Validity-aware retrieval.** Search now excludes logically stale facts: a
  memory whose `valid_to` has passed, or whose `expires_at` is reached, no longer
  surfaces in recall, even before the `dream` expire task deletes it. This changes
  default search behavior. `memory-list` still returns stale memories so they stay
  visible for review and cleanup.

### Added
- **Estimated context-token reporting** in the LoCoMo retrieval harness. It reports
  `pack_tokens_est` per query (estimated as characters / 4, since no model
  tokenizer is bundled), pairing recall accuracy with its context cost.

## [0.2.3] - 2026-06-24

Cross-platform usability fixes (surfaced testing on Windows, but general).

### Added
- **`--json @<file>` / `--json -` (stdin).** Any command's `--json` payload can be
  read from a file or stdin instead of an inline string, avoiding shell-quoting
  pitfalls (notably Windows PowerShell 5.1, which mangles inline JSON).
- **`serve --http --store <path>`.** The dashboard now takes `--store` like every
  other command, instead of requiring the `MEMKEEPER_STORE` env var.

### Changed
- **`pull-models`** prints env-setup lines in the host shell's dialect — PowerShell
  (`$env:... = "..."` / `setx`) on Windows, POSIX `export` elsewhere.
- **Windows docs** now reflect that the default on-device semantic build works on
  Windows (verified: local embeddings + reranker + warm-daemon HTTP), with an
  expanded troubleshooting section (`reindex --embed` after configuring models,
  dashboard `--store`, inline-JSON in PowerShell, "Access is denied" on rebuild,
  git "dubious ownership").

## [0.2.2] - 2026-06-24

### Added
- **Windows (experimental)** — verified to build and run from source. New
  [docs/windows.md](docs/windows.md) covers prerequisites (rustup + MSVC C++ build
  tools), the build, and the two common Rust-on-Windows blockers (`link.exe`, the
  Schannel `CARGO_HTTP_CHECK_REVOKE` workaround). CI now compiles + smoke-tests on
  Windows to catch regressions. No prebuilt Windows binary — build from source;
  Windows is best-effort, not a first-class platform.
- **"Capturing memories"** README section — memkeeper is curated memory you
  populate deliberately (`remember`, or the MCP bridge from an agent), not an
  automatic transcript logger; the `hook retrieve` client is the retrieval half.
- **"The dashboard"** README section — explains that a fresh store starts empty
  and that the **graph** is built from entities/relationships (which accrue from
  `dream` synthesis or explicit upserts), distinct from the memory list.

### Changed
- The dashboard's empty-graph message is now actionable (how to populate the graph)
  instead of just noting a fresh store may be empty.

## [0.2.1] - 2026-06-24

### Changed
- Re-published the macOS (Apple Silicon) and Linux x86_64 binaries with a clean
  build (the v0.2.0 release workflow's Windows job failed; macOS/Linux were
  unaffected).

### In progress
- **Windows support.** The Unix-socket code paths are now gated for non-Unix
  targets (a prerequisite for compiling on Windows), but a Windows binary is not
  yet published — pending build verification and a Windows CI job. When it lands,
  `serve --socket` will be Unix-only; the http dashboard and stdio serve are
  cross-platform.

## [0.2.0] - 2026-06-24

Semantic stays the default, now with two opt-in alternatives and a download path
that can reach semantic without building from source.

### Added
- **Off-device semantic retrieval.** Point memkeeper at any OpenAI-compatible
  embeddings API (OpenRouter, OpenAI, …) with `MEMKEEPER_EMBED_PROVIDER=openai` +
  base URL + key for embeddings, and a Cohere `/rerank` dialect for reranking — no
  local model download. The README documents three modes: local semantic (default),
  lexical-only, and off-device.
- **Prebuilt binaries now ship off-device support** (built `--features api`):
  lexical out of the box, or set an embeddings API key for semantic, with no 2GB
  model download. `MEMKEEPER_REQUIRE_SEMANTIC=1` still fails closed.
- Prebuilt binaries for macOS (Apple Silicon) and Linux x86_64.
- Documented the embedding-model **switch** path (`reindex --embed` atomically
  re-embeds the whole store under a new model), and made `reindex` discoverable in
  `--help` with its own `reindex --help`.

### Changed
- The prebuilt binary moved from lexical-only to off-device-capable (`api`).
- A lexical-only build's startup line is now a calm NOTE that frames semantic as the
  default, rather than an ERROR; the loud ERROR + desktop alarm is reserved for the
  `MEMKEEPER_REQUIRE_SEMANTIC` fail-closed path. The message is backend-aware (local
  models vs off-device key).
- `resolve_store_default` falls back to `%USERPROFILE%` on Windows where `$HOME` is
  unset.

### Fixed
- README no longer claims "no network is needed to build" (a fresh build still
  fetches crate dependencies); clarified to "no network/API key at runtime".
- Semantic-setup docs use the `./target/release/memkeeper` path instead of assuming
  the binary is on `PATH`; MCP install instructions pin a release tag.
- A `--no-default-features` build no longer emits an unused-import warning; CI now
  lints the non-default feature configs so config-specific warnings can't go latent.

### Notes
- Local-model (on-device) semantic on Windows is untested; build from source.

## [0.1.0] - 2026-06-23

Initial public release. A local-first memory engine for AI agents.

### Added
- **Local SQLite store** with atomic writes, schema versioning, and an FTS5/BM25
  retrieval path that requires no models or network.
- **Semantic retrieval, on by default** (the `semantic` feature is in the default
  build; disable with `--no-default-features`): ONNX embeddings plus a
  cross-encoder reranker. If models are missing, the engine **fails loud** — it
  refuses to start when `MEMKEEPER_REQUIRE_SEMANTIC=1`, and otherwise logs a loud
  WARNING and falls back to BM25 rather than silently degrading.
- **Two-tier retention** (durable / volatile short-term) with a promotion task
  that graduates recurring, high-signal memories to the durable tier based on
  distinct-session recall, a rerank score floor, and a rank cap.
- **Graph projection**: entity/relationship upsert, neighbor traversal, and
  graph-centered context packs.
- **CLI + daemon** (`memkeeper`): init, remember, search, pack, dream
  (maintenance), import/export, backup, and a Unix-socket serve mode.
- **Adapters**: an MCP bridge and a thin extension client.
- Dual-licensed **MIT OR Apache-2.0**.

[0.2.5]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.5
[0.2.4]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.4
[0.2.3]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.3
[0.2.2]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.2
[0.2.1]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.1
[0.2.0]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.0
[0.1.0]: https://github.com/teflon07/memkeeper/releases/tag/v0.1.0
