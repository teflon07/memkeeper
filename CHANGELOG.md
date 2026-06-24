# Changelog

All notable changes to memkeeper are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches
1.0. Until then, minor releases may include breaking changes to the storage
schema and wire protocol.

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
- **Windows x86_64** release binaries (`.zip`), alongside macOS (Apple Silicon) and
  Linux x86_64. `serve --http` and `--stdio` work; `serve --socket` is Unix-only.
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

[0.2.0]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.0
[0.1.0]: https://github.com/teflon07/memkeeper/releases/tag/v0.1.0
