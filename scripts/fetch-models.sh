#!/usr/bin/env bash
# Fetch the ONNX models memkeeper's semantic path needs: the embedder and the
# reranker. Both are Apache-2.0, downloaded from their official HuggingFace
# repos. Without these, the daemon runs FTS-only (BM25) and `serve` will say so
# loudly (or refuse, under MEMKEEPER_REQUIRE_SEMANTIC=1).
#
#   scripts/fetch-models.sh [--quantized] [--dir DIR]
#
#   --quantized   fetch the smaller INT8 models (~0.6GB total) instead of the
#                 full fp32 models (~2.1GB). Note: recall drifts from the
#                 fp32-benchmarked baseline, so prefer fp32 for parity.
#   --dir DIR     install root (default: $MEMKEEPER_MODELS_DIR or ~/.memkeeper/models)
#
# Late-interaction (ColBERT) is intentionally NOT fetched here: it is off by
# default (needs MEMKEEPER_LATE_INTERACTION=1) and the upstream repo ships its
# config as `artifact.metadata` rather than the `colbert_config.json` the loader
# expects, so it needs manual setup. See docs.
set -euo pipefail

QUANT=0
DIR="${MEMKEEPER_MODELS_DIR:-$HOME/.memkeeper/models}"

while [ $# -gt 0 ]; do
  case "$1" in
    --quantized) QUANT=1 ;;
    --dir) shift; [ $# -gt 0 ] || { echo "fetch-models: --dir needs a path" >&2; exit 2; }; DIR="$1" ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "fetch-models: unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

command -v curl >/dev/null 2>&1 || { echo "fetch-models: curl is required but not found" >&2; exit 1; }

HF="https://huggingface.co"
ONNX="model.onnx"
[ "$QUANT" -eq 1 ] && ONNX="model_quantized.onnx"

# _dl <url> <dest> — fail loud on any HTTP error (-f), follow CDN redirects (-L).
_dl() {
  local url="$1" out="$2"
  if ! curl -fL --retry 3 --proto '=https' --tlsv1.2 -o "$out" "$url"; then
    echo "fetch-models: FAILED downloading $url" >&2
    exit 1
  fi
}

# fetch_model <repo> <dest-subdir> — saves <onnx variant> as model.onnx + tokenizer.json
fetch_model() {
  local repo="$1" dest="$DIR/$2"
  mkdir -p "$dest"
  echo "==> $repo  ->  $dest  ($ONNX)"
  _dl "$HF/$repo/resolve/main/onnx/$ONNX" "$dest/model.onnx"
  _dl "$HF/$repo/resolve/main/tokenizer.json" "$dest/tokenizer.json"
}

# HF repos carry the `-v1` suffix; local subdirs omit it to match the
# MEMKEEPER_EMBED_MODEL_DIR/MEMKEEPER_RERANK_MODEL_DIR defaults (and `memkeeper
# pull-models`), so a no-arg fetch lands exactly where the adapter looks.
fetch_model "mixedbread-ai/mxbai-embed-large-v1" "mxbai-embed-large"
fetch_model "mixedbread-ai/mxbai-rerank-base-v1" "mxbai-rerank-base"

cat <<EOF

Done. Models installed under: $DIR
Point the daemon at them (add to your shell profile or memkeeper launch env):

  export MEMKEEPER_EMBED_MODEL_DIR="$DIR/mxbai-embed-large"
  export MEMKEEPER_RERANK_MODEL_DIR="$DIR/mxbai-rerank-base"

Then \`memkeeper serve\` runs with semantics on. To require it (fail closed if the
models go missing), also set MEMKEEPER_REQUIRE_SEMANTIC=1.
EOF
