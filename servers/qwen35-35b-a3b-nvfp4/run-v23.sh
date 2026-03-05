#!/usr/bin/env bash
# run-v23.sh — Launch Sehyo/Qwen3.5-35B-A3B-NVFP4 on avarok/dgx-vllm-nvfp4-kernel:v23
# Applies 6 volume-mounted patches: chat_utils, qwen3coder_tool_parser, modelopt,
# qwen3_5_mtp (OOB clamp + gate_up_proj fix), qwen3_reasoning_parser, serving.py.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTAINER_NAME="qwen35-35b-server"
IMAGE="avarok/dgx-vllm-nvfp4-kernel:v23"
VERSION="v23"
VLLM_BASE="/app/vllm/vllm"
BUILD_DIR="$REPO_ROOT/.build/$VERSION"

# Build all patched files if any are missing
if [[ ! -f "$BUILD_DIR/entrypoints/chat_utils.py" || \
      ! -f "$BUILD_DIR/tool_parsers/qwen3coder_tool_parser.py" || \
      ! -f "$BUILD_DIR/model_executor/layers/quantization/modelopt.py" || \
      ! -f "$BUILD_DIR/model_executor/models/qwen3_5_mtp.py" || \
      ! -f "$BUILD_DIR/reasoning/qwen3_reasoning_parser.py" || \
      ! -f "$BUILD_DIR/entrypoints/openai/chat_completion/serving.py" ]]; then
  echo "==> Building patched vLLM files..."
  bash "$REPO_ROOT/patches/build.sh" "$VERSION"
fi

# Remove old container if exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "==> Removing existing container '$CONTAINER_NAME'..."
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo "==> Starting $CONTAINER_NAME ($IMAGE)..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --gpus all \
  --ipc=host \
  --oom-score-adj 800 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$BUILD_DIR/entrypoints/chat_utils.py:$VLLM_BASE/entrypoints/chat_utils.py:ro" \
  -v "$BUILD_DIR/tool_parsers/qwen3coder_tool_parser.py:$VLLM_BASE/tool_parsers/qwen3coder_tool_parser.py:ro" \
  -v "$BUILD_DIR/model_executor/layers/quantization/modelopt.py:$VLLM_BASE/model_executor/layers/quantization/modelopt.py:ro" \
  -v "$BUILD_DIR/model_executor/models/qwen3_5_mtp.py:$VLLM_BASE/model_executor/models/qwen3_5_mtp.py:ro" \
  -v "$BUILD_DIR/reasoning/qwen3_reasoning_parser.py:$VLLM_BASE/reasoning/qwen3_reasoning_parser.py:ro" \
  -v "$BUILD_DIR/entrypoints/openai/chat_completion/serving.py:$VLLM_BASE/entrypoints/openai/chat_completion/serving.py:ro" \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MODEL=Sehyo/Qwen3.5-35B-A3B-NVFP4 \
  -e PORT=8000 \
  -e GPU_MEMORY_UTIL=0.9 \
  -e MAX_MODEL_LEN=131072 \
  -e MAX_NUM_SEQS=4 \
  -e VLLM_EXTRA_ARGS="--speculative-config.method qwen3_next_mtp --speculative-config.num_speculative_tokens 3 --attention-backend flashinfer --kv-cache-dtype fp8 --no-enable-chunked-prefill --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --served-model-name qwen35-35b" \
  ${IMAGE} \
  serve

echo "==> Container started. Logs:"
echo "    docker logs -f $CONTAINER_NAME"
