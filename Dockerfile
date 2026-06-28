# Containerized memkeeper MCP server (stdio JSON-RPC) — for Glama and any MCP
# client that prefers running the server in a container. Introspection works out
# of the box; retrieval is lexical (BM25/FTS) until you run `memkeeper
# pull-models` for on-device semantic. Build for linux/amd64 (the published
# release target).
FROM debian:bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates tar \
 && rm -rf /var/lib/apt/lists/*

# Install the latest self-contained release binary (the semantic ONNX runtime is
# statically bundled). install.sh detects the platform and verifies the SHA-256.
RUN curl -fsSL https://raw.githubusercontent.com/teflon07/memkeeper/main/install.sh \
      | MEMKEEPER_INSTALL_DIR=/usr/local/bin bash

# Persist the store (and any pulled models) by mounting a volume here.
VOLUME ["/root/.memkeeper"]

# Serve MCP over stdio (the agent connects to the container's stdin/stdout).
ENTRYPOINT ["memkeeper", "mcp"]
