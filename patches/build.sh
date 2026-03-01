#!/usr/bin/env bash
# build.sh — Build patched files into .build/ for volume mounting.
# Usage: ./build.sh [v23|v11|all]
#   Defaults to "all" if no argument provided.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$REPO_ROOT/.build"

build_version() {
  local VER="$1"
  local PATCH_DIR="$SCRIPT_DIR/vllm/$VER"
  local OUT="$BUILD_DIR/$VER"
  echo "==> Building $VER..."
  mkdir -p "$OUT/entrypoints" "$OUT/tool_parsers"

  patch -o "$OUT/entrypoints/chat_utils.py" \
    "$PATCH_DIR/entrypoints/chat_utils.py" \
    "$PATCH_DIR/entrypoints/chat_utils.patch"

  patch -o "$OUT/tool_parsers/qwen3coder_tool_parser.py" \
    "$PATCH_DIR/tool_parsers/qwen3coder_tool_parser.py" \
    "$PATCH_DIR/tool_parsers/qwen3coder_tool_parser.patch"

  echo "    -> $OUT/"
}

TARGET="${1:-all}"
case "$TARGET" in
  v23) build_version v23 ;;
  v11) build_version v11 ;;
  all) build_version v23; build_version v11 ;;
  *) echo "Usage: $0 [v23|v11|all]"; exit 1 ;;
esac
echo "Done."
