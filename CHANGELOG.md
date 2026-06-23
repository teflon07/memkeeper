# Changelog

All notable changes to memkeeper are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches
1.0. Until then, minor releases may include breaking changes to the storage
schema and wire protocol.

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

[0.1.0]: https://github.com/teflon07/memkeeper/releases/tag/v0.1.0
