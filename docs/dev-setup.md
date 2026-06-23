# Contributor setup

This is developer/contributor information. End users should follow the
Quickstart in the top-level `README.md` instead.

## Toolchain

memkeeper builds with stable Rust (developed against 1.95.0). Install via
[rustup](https://rustup.rs). If `cargo` is not already on your `PATH`, add it
for the session:

```bash
# rustup's default location
export PATH="$HOME/.cargo/bin:$PATH"

# Homebrew rustup (macOS) also needs its toolchain shims:
export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"
```

Interactive shells that source `~/.cargo/env` pick this up automatically.

## Validation

The workspace is expected to pass:

```bash
cargo fmt --check --all
cargo check --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
npm run check --prefix adapters/pi-extension
```

## CLI smoke walk

A full command walk against a throwaway store. `memkeeper` resolves the store from
`--store` > `MEMKEEPER_STORE` > `~/.memkeeper/store.sqlite`; this walk passes an
explicit `--store` so it never touches a real store:

```bash
STORE=/tmp/memkeeper-smoke/store.sqlite
cargo run --quiet -- schema-status
cargo run --quiet -- --version
cargo run --quiet -- init --store "$STORE" --json
cargo run --quiet -- doctor --store "$STORE" --json
printf '%s\n' "{\"request_id\":\"smoke\",\"command\":\"stats\",\"store_path\":\"$STORE\",\"payload\":{}}" | cargo run --quiet -- serve --stdio
cargo run --quiet -- space-list --store "$STORE" --json
cargo run --quiet -- space-create --store "$STORE" --json '{"name":"project-notes","description":"Project notes","default_silo":"long-term","if_not_exists":true}'
cargo run --quiet -- silo-list --store "$STORE" --space project-notes --json
cargo run --quiet -- remember --store "$STORE" --json '{"content":"decision: keep memkeeper deterministic","entity_key":"project:memkeeper"}'
cargo run --quiet -- entity-upsert --store "$STORE" --json '{"entity_key":"project:memkeeper","entity_type":"Project","canonical_name":"Memkeeper","aliases":["memkeeper memory"]}'
cargo run --quiet -- entity-upsert --store "$STORE" --json '{"entity_key":"component:sqlite","entity_type":"Component","canonical_name":"SQLite"}'
cargo run --quiet -- relationship-upsert --store "$STORE" --json '{"subject_entity_key":"project:memkeeper","relation_type":"uses","object_entity_key":"component:sqlite"}'
cargo run --quiet -- search --store "$STORE" --json '{"query":"memkeeper deterministic","limit":5}'
cargo run --quiet -- entity-search --store "$STORE" --json '{"entity_key":"project:memkeeper","limit":5}'
cargo run --quiet -- graph-neighbors --store "$STORE" --json '{"entity_key":"project:memkeeper","depth":1,"max_edges":10}'
cargo run --quiet -- graph-context --store "$STORE" --json '{"entity_key":"project:memkeeper","depth":1,"max_edges":10,"max_memories":5,"max_chars":1000}'
cargo run --quiet -- memory-list --store "$STORE" --json '{"limit":20}'
cargo run --quiet -- batch-search --store "$STORE" --json '{"queries":[{"name":"determinism","query":"memkeeper deterministic","limit":3}]}'
cargo run --quiet -- pack --store "$STORE" --json '{"title":"memkeeper smoke","queries":["memkeeper deterministic"],"max_memories":5,"max_chars":1000}'
cargo run --quiet -- get --store "$STORE" --id <memory-id> --json
cargo run --quiet -- history --store "$STORE" --id <memory-id> --json --no-source
cargo run --quiet -- export --store "$STORE" --output /tmp/memkeeper-smoke/export.jsonl --json
cargo run --quiet -- import --store /tmp/memkeeper-smoke/imported.sqlite --input /tmp/memkeeper-smoke/export.jsonl --json
cargo run --quiet -- dream --store "$STORE" --dry-run --tasks expire,reindex,dedupe,graph --max-memories 1000 --json
cargo run --quiet -- backup --store "$STORE" --output /tmp/memkeeper-smoke/backup.sqlite --json
cargo run --quiet -- forget --store "$STORE" --id <memory-id> --reason stale --json
cargo run --quiet -- stats --store "$STORE" --json
```

Benchmark and acceptance harnesses are documented in the top-level `README.md`.
