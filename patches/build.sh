#!/usr/bin/env bash
# build.sh — Build patched files into .build/ for volume mounting.
# Usage: ./build.sh [v23|v11|nightly|all]
#   Defaults to "all" if no argument provided.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$REPO_ROOT/.build"

apply_patch() {
  local ORIG="$1"
  local PATCHFILE="$2"
  local OUT="$3"
  mkdir -p "$(dirname "$OUT")"
  patch -o "$OUT" "$ORIG" "$PATCHFILE"
}

build_version() {
  local VER="$1"
  local PATCH_DIR="$SCRIPT_DIR/vllm/$VER"
  local OUT="$BUILD_DIR/$VER"
  echo "==> Building $VER..."

  if [[ -f "$PATCH_DIR/entrypoints/chat_utils.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/entrypoints/chat_utils.py" \
      "$PATCH_DIR/entrypoints/chat_utils.patch" \
      "$OUT/entrypoints/chat_utils.py"
  fi

  if [[ -f "$PATCH_DIR/tool_parsers/qwen3coder_tool_parser.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/tool_parsers/qwen3coder_tool_parser.py" \
      "$PATCH_DIR/tool_parsers/qwen3coder_tool_parser.patch" \
      "$OUT/tool_parsers/qwen3coder_tool_parser.py"
  fi

  # Optional patches
  if [[ -f "$PATCH_DIR/model_executor/layers/quantization/modelopt.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/model_executor/layers/quantization/modelopt.py" \
      "$PATCH_DIR/model_executor/layers/quantization/modelopt.patch" \
      "$OUT/model_executor/layers/quantization/modelopt.py"
  fi

  if [[ -f "$PATCH_DIR/model_executor/models/qwen3_5_mtp.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/model_executor/models/qwen3_5_mtp.py" \
      "$PATCH_DIR/model_executor/models/qwen3_5_mtp.patch" \
      "$OUT/model_executor/models/qwen3_5_mtp.py"
  fi

  if [[ -f "$PATCH_DIR/reasoning/qwen3_reasoning_parser.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/reasoning/qwen3_reasoning_parser.py" \
      "$PATCH_DIR/reasoning/qwen3_reasoning_parser.patch" \
      "$OUT/reasoning/qwen3_reasoning_parser.py"
  fi

  if [[ -f "$PATCH_DIR/entrypoints/openai/chat_completion/serving.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/entrypoints/openai/chat_completion/serving.py" \
      "$PATCH_DIR/entrypoints/openai/chat_completion/serving.patch" \
      "$OUT/entrypoints/openai/chat_completion/serving.py"
  fi

  if [[ -f "$PATCH_DIR/flashinfer/cutlass/include/cutlass/float_subbyte.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/flashinfer/cutlass/include/cutlass/float_subbyte.h" \
      "$PATCH_DIR/flashinfer/cutlass/include/cutlass/float_subbyte.patch" \
      "$OUT/flashinfer/cutlass/include/cutlass/float_subbyte.h"
  fi

  if [[ -f "$PATCH_DIR/quantization/compressed_tensors/compressed_tensors_moe.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/quantization/compressed_tensors/compressed_tensors_moe.py" \
      "$PATCH_DIR/quantization/compressed_tensors/compressed_tensors_moe.patch" \
      "$OUT/quantization/compressed_tensors/compressed_tensors_moe.py"
  fi

  if [[ -f "$PATCH_DIR/model_executor/models/qwen3_5.patch" ]]; then
    apply_patch \
      "$PATCH_DIR/model_executor/models/qwen3_5.py" \
      "$PATCH_DIR/model_executor/models/qwen3_5.patch" \
      "$OUT/model_executor/models/qwen3_5.py"
  fi

  echo "    -> $OUT/"
}

TARGET="${1:-all}"
case "$TARGET" in
  v23) build_version v23 ;;
  v11) build_version v11 ;;
  nightly) build_version nightly ;;
  all) build_version v23; build_version v11; build_version nightly ;;
  *) echo "Usage: $0 [v23|v11|nightly|all]"; exit 1 ;;
esac
echo "Done."
