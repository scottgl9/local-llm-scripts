#!/usr/bin/env bash
# run-v11.sh — Launch qwen3-coder-next-fp8 server using avarok/vllm-dgx-spark:v11
# Applies vLLM tool-call patches via volume mounts.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTAINER_NAME="qwen3-fp8-server"
IMAGE="avarok/vllm-dgx-spark:v11"
VERSION="v11"
VLLM_BASE="/opt/venv/lib/python3.12/site-packages/vllm"
BUILD_DIR="$REPO_ROOT/.build/$VERSION"

# Build patched files if not already done
if [[ ! -f "$BUILD_DIR/entrypoints/chat_utils.py" || \
      ! -f "$BUILD_DIR/tool_parsers/qwen3coder_tool_parser.py" ]]; then
  echo "==> Building patched vLLM files..."
  bash "$REPO_ROOT/patches/build.sh" "$VERSION"
fi

# Remove old container if exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "==> Removing existing container '$CONTAINER_NAME'..."
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

mkdir -p ~/.cache/huggingface

echo "==> Starting $CONTAINER_NAME ($IMAGE)..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --shm-size=32g \
  --ipc=host \
  --privileged \
  --memory 115g \
  --memory-swap 125g \
  --oom-score-adj 800 \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$BUILD_DIR/entrypoints/chat_utils.py:$VLLM_BASE/entrypoints/chat_utils.py:ro" \
  -v "$BUILD_DIR/tool_parsers/qwen3coder_tool_parser.py:$VLLM_BASE/tool_parsers/qwen3coder_tool_parser.py:ro" \
  -e VLLM_USE_DEEP_GEMM=0 \
  "$IMAGE" \
  serve Qwen/Qwen3-Coder-Next-FP8 \
  --served-model-name qwen3-coder-next \
  --host 0.0.0.0 --port 8000 \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8_e4m3 \
  --stream-interval 5 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --max-model-len 131072

echo "==> Container started. Logs:"
echo "    docker logs -f $CONTAINER_NAME"
