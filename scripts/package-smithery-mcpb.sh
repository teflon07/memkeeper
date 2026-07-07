#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="${1:-}"
if [[ -z "$version" ]]; then
  version="$(python3 - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("server.json").read_text())["version"])
PY
)"
fi
version="${version#v}"
tag="v${version}"

release_base="https://github.com/teflon07/memkeeper/releases/download/${tag}"
work_dir="${repo_root}/target/smithery/memkeeper-${version}"
cache_dir="${repo_root}/target/smithery/cache/${tag}"
bundle_dir="${work_dir}/bundle"
dist_dir="${repo_root}/dist"
bundle_path="${dist_dir}/memkeeper-${tag}.mcpb"

rm -rf "$work_dir"
mkdir -p "$bundle_dir/server/bin" "$bundle_dir/assets" "$cache_dir" "$dist_dir"

download_and_verify() {
  local archive="$1"
  local url="${release_base}/${archive}"

  curl -fL --retry 3 --retry-delay 2 -o "${cache_dir}/${archive}" "$url"
  curl -fL --retry 3 --retry-delay 2 -o "${cache_dir}/${archive}.sha256" "${url}.sha256"
  (cd "$cache_dir" && shasum -a 256 -c "${archive}.sha256")
}

copy_binary_from_archive() {
  local archive="$1"
  local dest="$2"
  local extract_dir="${work_dir}/extract-${dest}"

  rm -rf "$extract_dir"
  mkdir -p "$extract_dir"
  tar -xzf "${cache_dir}/${archive}" -C "$extract_dir"

  local binary
  binary="$(find "$extract_dir" -type f -name memkeeper | head -n 1)"
  if [[ -z "$binary" ]]; then
    echo "No memkeeper binary found in ${archive}" >&2
    exit 1
  fi

  cp "$binary" "${bundle_dir}/server/bin/${dest}"
  chmod 755 "${bundle_dir}/server/bin/${dest}"
}

darwin_archive="memkeeper-${tag}-aarch64-apple-darwin.tar.gz"
linux_archive="memkeeper-${tag}-x86_64-unknown-linux-gnu.tar.gz"

download_and_verify "$darwin_archive"
download_and_verify "$linux_archive"
copy_binary_from_archive "$darwin_archive" "memkeeper-darwin-arm64"
copy_binary_from_archive "$linux_archive" "memkeeper-linux-x86_64"

cp assets/logo.png "${bundle_dir}/assets/logo.png"

cat > "${bundle_dir}/server/memkeeper-mcp" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

server_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
os="$(uname -s)"
arch="$(uname -m)"

case "${os}:${arch}" in
  Darwin:arm64|Darwin:aarch64)
    bin="${server_dir}/bin/memkeeper-darwin-arm64"
    ;;
  Linux:x86_64|Linux:amd64)
    bin="${server_dir}/bin/memkeeper-linux-x86_64"
    ;;
  *)
    echo "memkeeper MCPB does not include a binary for ${os}/${arch}." >&2
    echo "Install from https://github.com/teflon07/memkeeper for this platform." >&2
    exit 78
    ;;
esac

export MEMKEEPER_MCP_ADAPTER="${MEMKEEPER_MCP_ADAPTER:-smithery-mcpb}"
export MEMKEEPER_MCP_SOURCE_DESCRIPTION="${MEMKEEPER_MCP_SOURCE_DESCRIPTION:-Smithery MCPB}"

exec "$bin" mcp "$@"
SH
chmod 755 "${bundle_dir}/server/memkeeper-mcp"

python3 - "$version" "$bundle_dir" "$work_dir" > "${bundle_dir}/manifest.json" <<'PY'
import json
import os
from pathlib import Path
import subprocess
import sys

version = sys.argv[1]
bundle_dir = Path(sys.argv[2])
work_dir = Path(sys.argv[3])

manifest_home = work_dir / "manifest-home"
manifest_home.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
env["HOME"] = str(manifest_home)
env["MEMKEEPER_MCP_ADAPTER"] = "smithery-mcpb-manifest"
env["MEMKEEPER_MCP_SOURCE_DESCRIPTION"] = "Smithery MCPB manifest generation"

requests = "\n".join(
    [
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "mcpb-manifest", "version": version},
                },
            }
        ),
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
        "",
    ]
)
server = bundle_dir / "server" / "memkeeper-mcp"
result = subprocess.run(
    [str(server)],
    input=requests,
    text=True,
    capture_output=True,
    timeout=20,
    check=False,
    env=env,
)
if result.returncode != 0:
    print(result.stderr, file=sys.stderr)
    raise SystemExit(f"failed to query packaged MCP tools: exit {result.returncode}")

tools = []
for line in result.stdout.splitlines():
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        continue
    if message.get("id") == 2:
        tools = message.get("result", {}).get("tools", [])
        break
if not tools:
    print(result.stderr, file=sys.stderr)
    raise SystemExit("packaged MCP server did not return tools/list data")

manifest = {
    "manifest_version": "0.3",
    "name": "memkeeper",
    "display_name": "memkeeper",
    "version": version,
    "description": "Local-first memory for AI agents: on-device hybrid retrieval over a single SQLite file.",
    "long_description": (
        "memkeeper gives MCP clients durable local memory backed by one SQLite file. "
        "It supports memory writes, relevance search, prompt-ready context packs, "
        "entity graph traversal, human-reviewed candidates, and provenance-aware cleanup. "
        "The Smithery bundle runs locally over stdio and does not require a hosted service. "
        "Lexical retrieval works out of the box; semantic and rerank models can be installed "
        "with the standard memkeeper CLI model-fetch flow."
    ),
    "author": {
        "name": "teflon07",
        "url": "https://github.com/teflon07",
    },
    "repository": {
        "type": "git",
        "url": "https://github.com/teflon07/memkeeper.git",
    },
    "homepage": "https://memkeeper.ai",
    "documentation": "https://github.com/teflon07/memkeeper#readme",
    "support": "https://github.com/teflon07/memkeeper/issues",
    "icon": "assets/logo.png",
    "server": {
        "type": "binary",
        "entry_point": "server/memkeeper-mcp",
        "mcp_config": {
            "command": "${__dirname}/server/memkeeper-mcp",
            "args": [],
            "env": {
                "MEMKEEPER_MCP_ADAPTER": "smithery-mcpb",
                "MEMKEEPER_MCP_SOURCE_DESCRIPTION": "Smithery MCPB",
            },
        },
    },
    "tools": tools,
    "tools_generated": True,
    "keywords": [
        "mcp",
        "memory",
        "local-first",
        "ai-agents",
        "sqlite",
        "retrieval",
        "semantic-search",
        "knowledge-graph",
        "rust",
    ],
    "license": "MIT OR Apache-2.0",
    "compatibility": {
        "platforms": ["darwin", "linux"],
    },
}
json.dump(manifest, sys.stdout, indent=2)
sys.stdout.write("\n")
PY

validation_dir="${work_dir}/validation-bundle"
rm -rf "$validation_dir"
cp -R "$bundle_dir" "$validation_dir"
python3 - "${validation_dir}/manifest.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
manifest = json.loads(path.read_text())
for tool in manifest.get("tools", []):
    tool.pop("inputSchema", None)
path.write_text(json.dumps(manifest, indent=2) + "\n")
PY

npx -y @anthropic-ai/mcpb validate "$validation_dir"
rm -f "$bundle_path"
(cd "$bundle_dir" && /usr/bin/zip -qr "$bundle_path" .)

bundle_size="$(wc -c < "$bundle_path" | tr -d '[:space:]')"
if (( bundle_size > 25 * 1024 * 1024 )); then
  echo "Bundle exceeds Smithery's 25 MB limit: ${bundle_size} bytes" >&2
  exit 1
fi

echo "Built ${bundle_path}"
echo "Publish with: smithery mcp publish ${bundle_path} -n teflon07/memkeeper"
