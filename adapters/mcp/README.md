# memkeeper MCP adapter

An [MCP](https://modelcontextprotocol.io) bridge that exposes a local memkeeper
store to any MCP-capable agent (Claude Code, Cursor, and others). It surfaces
source-hidden **read** tools (`search`, `get`, `memory_list`, `entity_search`,
`graph_neighbors`, `graph_context`, `dream_graph`, `stats`) plus narrow
**write** tools (`remember`, `forget`, `entity_upsert`, `relationship_upsert`,
`verify`).

Provenance is hidden by default: every read defaults `include_source=False`, so
the agent sees the memory, not where it came from, unless it explicitly asks.

## Native mode (no Python — preferred)

The `memkeeper` binary speaks MCP directly, exposing the **same tool surface** as
this Python bridge with no shim to install or keep in sync:

```bash
memkeeper mcp                 # uses $MEMKEEPER_STORE or ~/.memkeeper/store.sqlite
memkeeper mcp --store /path/to/store.sqlite
```

Register it in Claude Code (or any MCP client) by pointing at the binary instead
of `python3`:

```json
{
  "mcpServers": {
    "memkeeper": {
      "command": "memkeeper",
      "args": ["mcp"],
      "env": { "MEMKEEPER_MCP_SOURCE_DESCRIPTION": "Claude Code memkeeper MCP" }
    }
  }
}
```

Native mode loads the embed/rerank models in-process (semantic-primary when the
model dirs are present, deterministic BM25 otherwise) and honors
`MEMKEEPER_REQUIRE_SEMANTIC=1` to fail closed — identical degradation behavior to
`memkeeper serve`. The Python bridge below remains supported as a fallback for
environments where running the binary directly is inconvenient.

## Prerequisites

1. The `memkeeper` binary on your `PATH` (`cargo install --git <repo-url> memkeeper-cli`),
   or point `MEMKEEPER_BIN` at it.
2. An initialized store: `memkeeper init` creates `~/.memkeeper/store.sqlite`.
3. Python 3.10+ with the `mcp` package. Either `pip install mcp` into the
   environment you run the bridge from, or run it with [`uv`](https://docs.astral.sh/uv/)
   (`uvx --from . memkeeper-mcp`), which resolves the dependency for you.

Semantic search and rerank are optional: without the ONNX model dirs present,
search degrades to deterministic BM25 (still works). With them, search is
semantic + cross-encoder rerank. See the main memkeeper README for fetching the
mxbai models.

## Configuration

All configuration is environment variables with portable defaults under
`~/.memkeeper` — no machine-specific paths are baked in.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEMKEEPER_HOME` | `~/.memkeeper` | Base dir for the defaults below |
| `MEMKEEPER_STORE` | `$MEMKEEPER_HOME/store.sqlite` | SQLite store path |
| `MEMKEEPER_BIN` | `memkeeper` on `$PATH` | memkeeper binary |
| `MEMKEEPER_EMBED_MODEL_DIR` | `$MEMKEEPER_HOME/models/mxbai-embed-large` | Local embed model (optional) |
| `MEMKEEPER_RERANK_MODEL_DIR` | `$MEMKEEPER_HOME/models/mxbai-rerank-base` | Local rerank model (optional) |
| `MEMKEEPER_SOCK` | `/tmp/memkeeper_daemon.sock` | Warm-daemon socket (`memkeeper serve --socket`); falls back to a cold subprocess if absent |
| `MEMKEEPER_MCP_SOURCE_DESCRIPTION` | `memkeeper MCP` | Label recorded on writes/recall telemetry |

## Claude Code

Add the server with `claude mcp add` (adjust the path to where you cloned the
repo):

```bash
claude mcp add memkeeper -- \
  python3 /path/to/memkeeper/adapters/mcp/memkeeper_mcp.py
```

Or, equivalently, in your MCP config JSON:

```json
{
  "mcpServers": {
    "memkeeper": {
      "command": "python3",
      "args": ["/path/to/memkeeper/adapters/mcp/memkeeper_mcp.py"],
      "env": {
        "MEMKEEPER_MCP_SOURCE_DESCRIPTION": "Claude Code memkeeper MCP"
      }
    }
  }
}
```

With `uv` (no manual `pip install mcp` needed):

```json
{
  "mcpServers": {
    "memkeeper": {
      "command": "uvx",
      "args": ["--from", "/path/to/memkeeper/adapters/mcp", "memkeeper-mcp"]
    }
  }
}
```

## Cursor

Add to `~/.cursor/mcp.json` (or the project-local `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "memkeeper": {
      "command": "python3",
      "args": ["/path/to/memkeeper/adapters/mcp/memkeeper_mcp.py"]
    }
  }
}
```

## Notes

- The bridge never embeds text itself; the memkeeper CLI self-embeds `remember`
  and `search` at the model's native dimension, so there is no second provider
  that could write mismatched-dimension vectors into the store.
- Writes are intentionally narrow. `remember` stores one atomic fact per call;
  `forget` tombstones (preserving audit history, not a hard delete). Graph
  upserts maintain the rebuildable projection — memories remain the source of
  truth.
- Recall telemetry (which memories were surfaced/retrieved) is best-effort: a
  logging failure never breaks a read.

Licensed under either of MIT or Apache-2.0 at your option.
