# memkeeper operations

Operational runbook for a memkeeper store: backups, the restore drill, and the
stable `pack` output contract. Commands assume the `memkeeper` binary is on
`PATH` and `<store>` is the SQLite store path.

## Backups

`backup` writes a consistent physical SQLite snapshot (it refuses to clobber
an existing file or write next to the live store's sidecars):

```
memkeeper backup --store <store> --json '{"output":"/backups/memkeeper-YYYYMMDD.sqlite"}'
```

The report includes `bytes`, `sha256`, and `page_count`. Record the `sha256`
so the snapshot's integrity can be verified later.

A logical alternative is `export` (JSONL), re-loadable with `import`. Use the
physical backup for disaster recovery and the logical export for portability
or schema-migration round-trips.

## Restore drill

A backup is only trustworthy if it reopens as a healthy, *searchable* store.
Run this drill periodically (and after any backup-pipeline change). It uses a
throwaway copy and never touches the live store:

1. **Restore** — copy the snapshot to a scratch path (do not point at the live
   store):
   ```
   cp /backups/memkeeper-YYYYMMDD.sqlite /tmp/restore-check.sqlite
   ```
2. **Health** — confirm the store opens and the rollup looks sane:
   ```
   memkeeper stats --store /tmp/restore-check.sqlite --health --json
   ```
   Expect a matching `active` count and no surprises in `tombstoned` /
   `expired` / `duplicate_key_groups`.
3. **Doctor** (optional) — deeper readiness check:
   ```
   memkeeper doctor --store /tmp/restore-check.sqlite --json
   ```
4. **Search smoke** — confirm retrieval actually works, not just that rows exist:
   ```
   memkeeper search --store /tmp/restore-check.sqlite --json '{"query":"<a phrase you know is stored>","limit":5}'
   ```
   At least one expected memory should come back.
5. **Clean up** — remove the scratch copy and its sidecars:
   ```
   rm -f /tmp/restore-check.sqlite /tmp/restore-check.sqlite-*
   ```

A store-level test (`backup_creates_restorable_sqlite_snapshot`) also exercises
backup → reopen → stats in CI, so a regression in restorability fails
`cargo test`.

## Scheduling nightly maintenance (dream)

`dream` runs bounded maintenance over the store: `expire` (drop TTL'd
short-term memories), `reindex` (rebuild FTS), `dedupe` (report exact-duplicate
proposals), and `graph` (reconcile projection drift **and** materialize
link-derived relationship edges into the entity graph). It is **dry-run by
default**; pass `--apply` (or `--commit`) to write.

Preview, then apply:

```
memkeeper dream --store <store> --tasks expire,reindex,dedupe,graph --dry-run --json
memkeeper dream --store <store> --tasks expire,reindex,dedupe,graph --apply   --json
```

A dry run reports the proposed counts (`graph_relationship_proposals`, orphan
entities, etc.) without mutating; the graph task only writes on `--apply`.

The engine ships the `dream` command but **no scheduler** — wire one yourself to
dream nightly. Run the dream *before* your backup so the snapshot captures the
maintained state. The maintenance tasks are deterministic and do not require the
embedding models.

### cron

```
# 02:00 daily (leave margin before any backup job)
0 2 * * *  memkeeper dream --store /path/to/store.sqlite --tasks expire,reindex,dedupe,graph --apply --json >> /var/log/memkeeper-dream.log 2>&1
```

### systemd timer

`/etc/systemd/system/memkeeper-dream.service`:

```
[Unit]
Description=memkeeper nightly dream (maintenance + graph apply)

[Service]
Type=oneshot
Environment=MEMKEEPER_STORE=/path/to/store.sqlite
ExecStart=/usr/local/bin/memkeeper dream --tasks expire,reindex,dedupe,graph --apply --json
```

`/etc/systemd/system/memkeeper-dream.timer`:

```
[Unit]
Description=Run memkeeper nightly dream

[Timer]
OnCalendar=*-*-* 02:00:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

Enable it:

```
systemctl daemon-reload && systemctl enable --now memkeeper-dream.timer
```

## Pack output contract

`pack` produces context for injection into a host model. Its JSON output is a
stable contract; consumers may rely on these top-level fields. The shape is
pinned by the CLI test `pack_result_json_contract_is_stable`.

```json
{
  "pack": {
    "title":      "string  — the requested pack title",
    "format":     "string  — output format (e.g. \"markdown\")",
    "content":    "string  — the bounded, ready-to-inject pack text",
    "memory_ids": ["string — ids included in content, in display order"],
    "scores":     [0.0, "  — per-memory ranking score, aligned 1:1 with memory_ids"],
    "truncated":  false,
    "top_score":  0.0
  }
}
```

Notes:
- `scores` is aligned 1:1 with `memory_ids` (same order). Entries are the
  cross-encoder rerank score on the rerank path, otherwise the retrieval score.
- `top_score` is the highest relevance score across the *retrieved candidate
  pool* (before the score floor / char budget trimmed the pack); `null` for an
  empty pack.
- `truncated` is `true` when memories or text were dropped to honor limits.

## Shadow reranker telemetry

Set `MEMKEEPER_RERANK_SHADOW_MODEL_DIR` on a long-lived server to score each
production rerank candidate set with a second local model. Shadow scoring runs
after the authoritative pack is assembled on a bounded background worker, so
it cannot alter or delay the served pack. Failures and queue saturation are
logged and the production response continues.

Set `MEMKEEPER_REQUIRE_SHADOW_RERANK=1` when the server must refuse startup unless
the shadow model loaded and its worker started. This requirement is enforced in
semantic and lexical-only builds.

Comparisons are stored in the local `reranker_shadow_events` table, one row per
candidate and one batch per pack. Rows include the query, production and shadow
model ids, scores, and ranks. The table is operational telemetry and is excluded
from export/import.

## Supersession modes

`remember` takes an optional `mode` that governs how a write resolves against
active memories sharing its `entity_key` + `claim_key`:

- `auto` (default) — retire older same-key memories of eligible kinds.
- `append` — coexist; supersede nothing.
- `supersede` — force-retire all non-pinned same-key actives, any kind.
- `suggest` — mutate nothing; return the would-be set in `supersede_suggestions`.
- `conflict` — mutate nothing; open a `conflicts` row per same-key active.
