# Smithery packaging

Memkeeper is a local-first stdio MCP server, so Smithery distribution should use
an MCPB bundle rather than URL publishing.

Build the bundle from the current release metadata:

```sh
scripts/package-smithery-mcpb.sh
```

Build a specific release:

```sh
scripts/package-smithery-mcpb.sh v0.2.13
```

The script downloads the GitHub release binaries, verifies the published SHA256
files, validates the strict MCPB structure, adds the live MCP tool schemas
required by Smithery publishing, and writes the bundle to `dist/`.

Install and authenticate the Smithery CLI:

```sh
npm install -g smithery@latest
smithery auth login
```

Publish to Smithery:

```sh
smithery mcp publish dist/memkeeper-v0.2.13.mcpb -n teflon07/memkeeper
```

Supported bundled platforms:

- macOS Apple Silicon (`aarch64-apple-darwin`)
- Linux x86_64 (`x86_64-unknown-linux-gnu`)

The bundle intentionally does not include the ONNX model files. Lexical retrieval
works out of the box; semantic and rerank models can be installed through the
standard memkeeper CLI model-fetch flow.
