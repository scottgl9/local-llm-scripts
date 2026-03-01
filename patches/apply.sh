#!/usr/bin/env bash
# apply.sh — Apply vLLM tool-call patches to a running container by volume-mounting patched files.
#
# Usage:
#   ./apply.sh <container_name> [v23|v11]
#
# The script:
#   1. Determines the correct image version (v23 or v11) if not specified.
#   2. Patches the original source files into .build/<version>/ using `patch`.
#   3. Prints the -v flags to add to your `docker run` command.
#
# Note: This script does NOT restart the container.  Add the printed -v flags to
# your run script and restart the container to pick up the patches.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$REPO_ROOT/.build"

CONTAINER="${1:-}"
VERSION="${2:-}"

usage() {
  echo "Usage: $0 <container_name> [v23|v11]"
  exit 1
}

[[ -z "$CONTAINER" ]] && usage

# Auto-detect version from running container image if not provided
if [[ -z "$VERSION" ]]; then
  IMAGE=$(docker inspect --format='{{.Config.Image}}' "$CONTAINER" 2>/dev/null || true)
  if [[ "$IMAGE" == *"v23"* ]]; then
    VERSION="v23"
  elif [[ "$IMAGE" == *"v11"* ]]; then
    VERSION="v11"
  else
    echo "ERROR: Could not auto-detect version from image '$IMAGE'. Pass v23 or v11 explicitly."
    exit 1
  fi
fi

echo "==> Applying patches for $VERSION into $BUILD_DIR/$VERSION/"
mkdir -p "$BUILD_DIR/$VERSION/entrypoints"
mkdir -p "$BUILD_DIR/$VERSION/tool_parsers"

PATCH_DIR="$SCRIPT_DIR/vllm/$VERSION"

# Apply each patch: patch -o output original patchfile
patch -o "$BUILD_DIR/$VERSION/entrypoints/chat_utils.py" \
  "$PATCH_DIR/entrypoints/chat_utils.py" \
  "$PATCH_DIR/entrypoints/chat_utils.patch"

patch -o "$BUILD_DIR/$VERSION/tool_parsers/qwen3coder_tool_parser.py" \
  "$PATCH_DIR/tool_parsers/qwen3coder_tool_parser.py" \
  "$PATCH_DIR/tool_parsers/qwen3coder_tool_parser.patch"

echo "==> Patched files written to $BUILD_DIR/$VERSION/"

# Print the -v flags for docker run
if [[ "$VERSION" == "v23" ]]; then
  VLLM_BASE="/app/vllm/vllm"
else
  VLLM_BASE="/opt/venv/lib/python3.12/site-packages/vllm"
fi

echo ""
echo "==> Add these -v flags to your docker run command:"
echo "    -v \"$BUILD_DIR/$VERSION/entrypoints/chat_utils.py:$VLLM_BASE/entrypoints/chat_utils.py:ro\""
echo "    -v \"$BUILD_DIR/$VERSION/tool_parsers/qwen3coder_tool_parser.py:$VLLM_BASE/tool_parsers/qwen3coder_tool_parser.py:ro\""
echo ""
echo "==> Or copy into running container (no restart needed for Python files — requires container restart to reload):"
echo "    docker cp \"$BUILD_DIR/$VERSION/entrypoints/chat_utils.py\" \"$CONTAINER:$VLLM_BASE/entrypoints/chat_utils.py\""
echo "    docker cp \"$BUILD_DIR/$VERSION/tool_parsers/qwen3coder_tool_parser.py\" \"$CONTAINER:$VLLM_BASE/tool_parsers/qwen3coder_tool_parser.py\""
echo "    docker exec \"$CONTAINER\" find $VLLM_BASE -name '*.pyc' -delete"
