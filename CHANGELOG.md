# Changelog

All notable changes to memkeeper are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches
1.0. Until then, minor releases may include breaking changes to the storage
schema and wire protocol.

## [0.5.1] - 2026-07-21

### Fixed
- **Import accepts an archive written by a differently named build.** Import
  replaces `config_kv` wholesale from the archive, so the stored
  `protocol_version` survived from whichever build wrote the archive. A build
  whose own wire name differed then failed its own initialization check against
  a store it had just written, reporting `store is not initialized` for an
  internal temporary path. The value is now reconciled during import, and only
  when it actually differs, so a byte-identical export/import/export round trip
  still holds.
- **`reindex --embed` no longer fails after a vector-table rebuild.**
  `drop_all_vector_tables` dropped `memory_vec_%` tables in `sqlite_master`
  order, which put the vec0 shadow tables ahead of the virtual table that owns
  them and corrupted it. Every subsequent `reindex --embed` died with
  `SQL logic error`.
- **Recency no longer reads two different clocks.** `now_julian_day`
  reimplemented `julianday('now')` in Rust, so SQL-side recency ordering and the
  authoritative re-scoring pass disagreed by construction. It now queries SQLite.
- **Graph capture no longer creates colliding aliases.** Generated slugs already
  owned by another canonical entity are filtered, and aliases are deduplicated
  case-insensitively and never repeat the entity key or canonical name.
- **`schema remember` documents `graph` and `derive_keys`.** Both were accepted
  but neither was listed, leaving atomic graph capture undiscoverable from the
  CLI that advertises it.

### Changed
- Split the graph subsystem into `graph.rs` and dream consolidation into
  `dream.rs`. No behavior change.
- Cache the entity lookup and the entity-alias statement in `load_entity`.
- Strip the release binary, 32 MB down to 23 MB.
- Gate the retrieval oracle baseline in CI.

## [0.5.0] - 2026-07-19

### Added
- **Atomic evidence-backed graph capture.** Native MCP `remember` accepts bounded
  entities, exact aliases, and typed relationships alongside one confirmed
  memory. Memkeeper validates and commits the memory and graph projection in one
  transaction, and the canonical memory ID is the relationship's sole evidence.
- **Source time in context packs.** Every injected memory line includes its
  stored `observed_at` timestamp. The timestamp is not sent to the reranker, so
  it improves temporal answerability without changing ranking.

### Changed
- `evidence_join_v2` is the normal semantic pack path. Exact entity and alias
  matches join semantic and lexical seeds on canonical memory IDs before one
  shared rerank.
- Direct and graph agreement breaks exact reranker ties only. It never promotes
  a lower-scoring candidate over a higher cross-encoder score.
- Generic `dream graph` `related_to` rows remain available for visualization and
  graph browsing, but only typed relationships backed by a memory participate
  in evidence retrieval.

### Compatibility
- No storage schema, embedding model, reranker model, provider default, or local
  model requirement changed in this release. Graph structure is supplied by the
  MCP host agent in the existing `remember` call; Memkeeper does not add a
  separate LLM or extraction service.

## [0.4.0] - 2026-07-18

### Changed
- **One evidence-join retrieval path.** Semantic and lexical results seed a
  bounded evidence-backed graph traversal, then every canonical memory ID
  competes in the same cross-encoder rerank pool. Graph candidates receive no
  reserved slots or automatic demotion.
- `min_score` now gates the whole pack on its top reranker score. It no longer
  removes lower-ranked supporting evidence after the best candidate clears the
  gate.
- Pack execution builds one candidate pool and reranks it once. Late
  interaction supplies semantic seeds without a redundant dense query
  embedding when available.
- Normal `pack` requests no longer accept query expansion, thread expansion,
  cosine-gate, or graph-tuning controls. Graph bounds remain available only on
  the diagnostic `pool-trace` command.

### Fixed
- `MEMKEEPER_REQUIRE_RERANK=1` now fails closed when the reranker fails during a
  request, not only when the model is absent at startup. Optional fallback is
  visible and packs the same unified semantic, lexical, and graph pool in
  retrieval order.

### Compatibility
- No storage schema, embedding model, reranker model, or provider default
  changed in this release.

## [0.3.1] - 2026-07-14

### Fixed
- **Custom Space defaults survive initialization and archive restore.** The
  vestigial `long-term` cleanup is limited to the legacy default
  `workspace-memory` Space again. Custom Spaces intentionally configured with a
  `long-term` default now preserve it across initialization and schema-6 logical
  export/import/export, including archives with retrieval representations.

## [0.3.0] - 2026-07-14

### Added
- **Version-owned retrieval representations.** CLI and `serve` callers can attach
  an optional `contextual-card-v1` companion of up to 512 characters when they
  save a memory. Memkeeper uses that companion for lexical retrieval,
  late-interaction scoring, and reranking while preserving the canonical memory
  text returned to agents.
- **Representation lifecycle support.** `get`, `history`, logical
  export/import, physical backup, and token reindexing now preserve or rebuild
  the representation with its owning memory version. Legacy schema-5 archives
  remain importable.
- **Pre-rerank pool diagnostics.** The new semantic-only `pool-trace` command
  replays a `pack` request and returns memory IDs, retrieval routes, admission
  state, drop stages, and graph-allocation ranks without returning memory text.
- **Required-reranker mode.** Set `MEMKEEPER_REQUIRE_RERANK=1` to make `pack`,
  reranked one-shot `search`, `serve`, and native MCP fail closed when the
  primary cross-encoder is unavailable instead of serving plain retrieval
  order.
- `stats --health` now reports active memories that have an entity key but no
  matching entity projection.

### Changed
- **Storage schema 6.** Existing schema-5 stores migrate on the first write by
  adding `memory_representations` and the representation-aware FTS projection.
  The migration preserves canonical content, summaries, embeddings, and memory
  identity.
- `remember` now derives missing entity and claim keys by default. Set
  `derive_keys:false` to preserve an intentionally keyless write; explicit
  caller-provided keys are never replaced.
- The generic MCP `remember` tool intentionally does not expose the new field.
  Formation remains an explicit CLI or `serve` adapter responsibility until a
  separate MCP formation contract is reviewed.
- The Pi extension now checks the managed Memkeeper runtime at
  `~/.local/libexec/memkeeper/current/memkeeper` before falling back to source
  build paths or `PATH`.

### Experimental
- Graph expansion can select one memory per reached entity using first-query
  late-interaction MaxSim by setting `graph_within_entity_maxsim`. It remains
  off by default and requires late-interaction tokens.

### Fixed
- Non-default feature builds remain clean under strict Clippy checks when local
  semantic components are disabled.

### Internal
- Schema mutation and initialized-connection lifecycle now live in focused
  store modules. Structural tests guard those ownership boundaries.
- The LoCoMo harness can capture the exact pre-rerank pool alongside retrieval
  results for admission and miss analysis.

## [0.2.15] - 2026-07-14

### Fixed
- **Custom Space defaults survive restart and restore.** Initialization now
  limits the legacy `long-term` cleanup to the default `workspace-memory` Space,
  preserving explicit `long-term` defaults and silos in custom Spaces across
  restart and logical export/import/export.

## [0.2.14] - 2026-07-10

### Added
- **Smithery MCPB packaging.** `scripts/package-smithery-mcpb.sh` builds and
  validates a portable MCP bundle from the release assets, with usage and
  publishing instructions in `smithery/README.md`.

### Fixed
- **Docker catalog compatibility.** The published container now uses a catalog-
  compatible base image.

## [0.2.13] - 2026-06-30

### Fixed
- **CI clippy `too_many_lines`.** The v0.2.12 rerank-report function exceeded the
  100-line pedantic cap under `-D warnings`; the candidate-building block is now a
  small helper. Internal only — no functional or behavioral change from 0.2.12.

## [0.2.12] - 2026-06-30

### Performance
- **`dream link` tag-link query rewritten as CTEs.** The shared-tag cross-entity
  link pass used a correlated tag-frequency subquery that ran for minutes on a
  realistic store; materializing the discriminative-tag set and eligible rows as
  CTEs brings it down to ~1.4s. Same result set and ordering.

### Experimental (off by default)
- **Associative-recall graph path.** Behind `MEMKEEPER_ASSOCIATIVE_RECALL`
  (default off, byte-identical when off), pack retrieval can graph-expand the
  rerank candidate pool one hop and reserve a bounded slot for a hop-reached
  memory. A relative gate quarantines below-pool graph additions to that reserved
  slot, so enabling it adds at most one bounded swap per query rather than
  reordering results. This is experimental: it has not shown a measurable recall
  gain on labeled benchmarks and is not recommended for production use yet.

## [0.2.11] - 2026-06-30

### Removed
- **The legacy Python MCP bridge (`adapters/mcp`) is gone.** The `memkeeper`
  binary speaks MCP over stdio natively (`memkeeper mcp`) with the same tool
  surface, so the Python `fastmcp` adapter was redundant. Point your MCP client
  at the native binary: `{ "command": "memkeeper", "args": ["mcp"] }`. The
  `pi-extension` adapter is unaffected.

## [0.2.10] - 2026-06-29

### Changed
- **Reranker and late-interaction (ColBERT) degradation is now loud, not silent.**
  When the embedder loads but the reranker model is missing, `serve`/`mcp` now log
  a NOTE (an ERROR under `MEMKEEPER_REQUIRE_SEMANTIC`) and continue with plain
  retrieval order, instead of silently skipping rerank. When
  `MEMKEEPER_LATE_INTERACTION=1` is set but the ColBERT model is absent, startup
  warns loudly and refuses to serve under `MEMKEEPER_REQUIRE_SEMANTIC` rather than
  silently disabling late-interaction. Only the embedder was guarded before.

### Internal
- Split the large `memkeeper-store` `lib.rs` into focused modules (`types`,
  `common`, `archive_spec`, `recall`, `stats`, `spaces`) — pure code movement, no
  API or behavior change.
- Added a drift test that ties the MCP tool `inputSchema` to the real request
  parsers, so the advertised tool schema cannot silently diverge from what the
  engine accepts.

## [0.2.9] - 2026-06-28

### Changed
- **Calmer startup when on-device models aren't present.** A semantic-capable
  build (the default release binary) running before `pull-models` now logs a calm
  NOTE and serves lexical BM25/FTS, instead of an ERROR plus a desktop alarm.
  Lexical-only is a supported default; the loud ERROR + alarm + refuse-to-serve is
  now reserved for `MEMKEEPER_REQUIRE_SEMANTIC=1` (the fail-closed path).

## [0.2.8] - 2026-06-28

### Changed
- **Richer MCP tool definitions.** All 16 tools exposed over `memkeeper mcp` now
  carry full, agent-facing descriptions: each tool states its purpose, when to
  use it versus similar tools, and whether it is read-only or mutating, and every
  parameter documents its type, default, and constraints. No change to the tool
  surface (names/arguments) — better guidance for agents (and higher
  tool-definition-quality scores).

## [0.2.7] - 2026-06-28

### Added
- **`doctor` reports semantic readiness.** `memkeeper doctor` now includes a
  `semantic.models` check: `ok` when the local embed model is present, a
  `warning` pointing at `pull-models` (and naming the resolved model dir) when a
  semantic-capable build has no models yet, or a note for lexical-only builds.
  Overall doctor status is unchanged — lexical still works — so it's guidance,
  not a failure.

## [0.2.6] - 2026-06-28

### Changed
- **Release binaries are now self-contained and semantic-capable.** The published
  macOS/Linux binaries bundle the ONNX runtime (built `--features semantic,api`),
  so on-device semantic search works with no rebuild — run `memkeeper pull-models`
  once to fetch the models. Previously the release binary was lexical-only and
  local semantic required building from source.
- **Zero-config local models.** memkeeper now looks for the embed/rerank models in
  the `pull-models` default location (`$MEMKEEPER_MODELS_DIR`, else
  `~/.memkeeper/models`) when `MEMKEEPER_EMBED_MODEL_DIR` /
  `MEMKEEPER_RERANK_MODEL_DIR` are unset — so `pull-models` then `search` turns on
  semantics with no environment variables. A missing model dir now prints an
  actionable `pull-models` hint instead of silently degrading.

### Added
- **`install.sh`** — one-line installer: downloads the release binary for your
  platform, verifies its SHA-256 (fail closed), and installs it to `~/.local/bin`.

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

[0.5.1]: https://github.com/teflon07/memkeeper/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/teflon07/memkeeper/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/teflon07/memkeeper/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/teflon07/memkeeper/compare/v0.3.0...v0.3.1
[0.2.11]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.11
[0.2.10]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.10
[0.2.9]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.9
[0.2.8]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.8
[0.2.7]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.7
[0.2.6]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.6
[0.2.5]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.5
[0.2.4]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.4
[0.2.3]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.3
[0.2.2]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.2
[0.2.1]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.1
[0.2.0]: https://github.com/teflon07/memkeeper/releases/tag/v0.2.0
[0.1.0]: https://github.com/teflon07/memkeeper/releases/tag/v0.1.0
