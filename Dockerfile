# Containerized memkeeper MCP server (stdio JSON-RPC) — for Glama and any MCP
# client that prefers running the server in a container. Introspection works out
# of the box; retrieval is lexical (BM25/FTS) until you run `memkeeper
# pull-models` for on-device semantic. Build for linux/amd64 (the published
# release target).
FROM --platform=linux/amd64 ubuntu:24.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates tar \
 && rm -rf /var/lib/apt/lists/*

# Install the self-contained release binary (the semantic ONNX runtime is
# statically bundled). install.sh detects the platform and verifies the SHA-256.
# MEMKEEPER_VERSION pins a specific release (e.g. v0.2.9) — pass it via
# `--build-arg` to make the image reproducible; empty (the default, e.g. Glama
# builds) installs the latest release.
# Ownership label the official MCP registry checks: its value MUST equal the
# `name` in server.json, or `mcp-publisher publish` fails OCI verification.
LABEL io.modelcontextprotocol.server.name="io.github.teflon07/memkeeper"

ARG MEMKEEPER_VERSION=""
RUN curl -fsSL https://raw.githubusercontent.com/teflon07/memkeeper/main/install.sh \
      | MEMKEEPER_INSTALL_DIR=/usr/local/bin MEMKEEPER_VERSION="$MEMKEEPER_VERSION" bash

# Persist the store (and any pulled models) by mounting a volume here.
VOLUME ["/root/.memkeeper"]

# Serve MCP over stdio (the agent connects to the container's stdin/stdout).
ENTRYPOINT ["memkeeper", "mcp"]
