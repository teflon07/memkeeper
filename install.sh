#!/usr/bin/env bash
# memkeeper installer — downloads the latest (or a pinned) release binary for
# your platform, verifies its SHA-256, and installs it.
#
# The release binary is self-contained: the semantic ONNX runtime is statically
# bundled, so there is nothing else to install. It serves lexical BM25/FTS out of
# the box; run `memkeeper pull-models` once to enable on-device semantic search.
#
#   curl -fsSL https://raw.githubusercontent.com/teflon07/memkeeper/main/install.sh | bash
#
# Environment:
#   MEMKEEPER_VERSION      release tag to install (default: latest)
#   MEMKEEPER_INSTALL_DIR  install directory (default: ~/.local/bin)
set -euo pipefail

REPO="teflon07/memkeeper"
INSTALL_DIR="${MEMKEEPER_INSTALL_DIR:-$HOME/.local/bin}"

err() { echo "memkeeper-install: $*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || err "curl is required but was not found on PATH"
command -v tar  >/dev/null 2>&1 || err "tar is required but was not found on PATH"

# --- platform -> release target triple ---------------------------------------
os="$(uname -s)"; arch="$(uname -m)"
case "$os-$arch" in
  Darwin-arm64)  target="aarch64-apple-darwin" ;;
  Linux-x86_64)  target="x86_64-unknown-linux-gnu" ;;
  *) err "unsupported platform: $os-$arch (published builds: macOS arm64, Linux x86_64).
   Build from source instead: https://github.com/$REPO" ;;
esac

# --- resolve version (latest unless pinned) ----------------------------------
version="${MEMKEEPER_VERSION:-}"
if [ -z "$version" ]; then
  # Follow the /releases/latest redirect to its tagged URL. Avoids the GitHub
  # API (rate limits) and needs no JSON parser.
  loc="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest")" \
    || err "could not resolve the latest release"
  version="${loc##*/tag/}"
  { [ -n "$version" ] && [ "$version" != "$loc" ]; } || err "could not parse latest version from: $loc"
fi

asset="memkeeper-${version}-${target}.tar.gz"
base="https://github.com/$REPO/releases/download/$version"

tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

echo "==> Downloading $asset ($version)"
curl -fSL "$base/$asset"        -o "$tmp/$asset"        || err "download failed: $base/$asset"
curl -fSL "$base/$asset.sha256" -o "$tmp/$asset.sha256" || err "checksum download failed: $base/$asset.sha256"

# --- verify checksum (fail closed) -------------------------------------------
echo "==> Verifying checksum"
(
  cd "$tmp"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 -c "$asset.sha256"
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c "$asset.sha256"
  else
    err "need 'shasum' or 'sha256sum' to verify the download"
  fi
) || err "CHECKSUM MISMATCH — refusing to install $asset"

# --- extract + install -------------------------------------------------------
tar xzf "$tmp/$asset" -C "$tmp"
bin="$(find "$tmp" -type f -name memkeeper | head -1)"
[ -n "$bin" ] || err "memkeeper binary not found in $asset"
mkdir -p "$INSTALL_DIR"
install -m 0755 "$bin" "$INSTALL_DIR/memkeeper"
echo "==> Installed memkeeper $version -> $INSTALL_DIR/memkeeper"

# --- next steps --------------------------------------------------------------
# Use a bare `memkeeper` in the printed steps only when the install dir is
# already on PATH; otherwise use the full path so the commands copy-paste and run
# on a fresh shell, and show how to add it to PATH permanently.
case ":$PATH:" in
  *":$INSTALL_DIR:"*)
    cmd="memkeeper"
    ;;
  *)
    cmd="$INSTALL_DIR/memkeeper"
    echo "    Note: $INSTALL_DIR is not on your PATH yet. Add it for a bare \`memkeeper\`:"
    echo "      export PATH=\"$INSTALL_DIR:\$PATH\""
    ;;
esac

cat <<EOF

Next steps:
  $cmd pull-models    # one-time: fetch on-device embed + rerank models (semantic search)
  $cmd doctor         # verify your setup
  $cmd init           # create a store
  $cmd remember --json '{"content":"memkeeper remembers this across sessions"}'
  $cmd search   --json '{"query":"what does memkeeper remember","limit":5}'

Docs: https://github.com/teflon07/memkeeper
EOF
