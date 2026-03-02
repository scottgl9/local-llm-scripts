#!/usr/bin/env bash
# run-v23.sh — Launch qwen3-coder-next-fp8 server using avarok/dgx-vllm-nvfp4-kernel:v23
# Applies vLLM tool-call patches via volume mounts.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTAINER_NAME="qwen3-fp8-server"
IMAGE="avarok/dgx-vllm-nvfp4-kernel:v23"
VERSION="v23"
VLLM_BASE="/app/vllm/vllm"
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

# Clear system caches
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Ensure host cache directories exist so Docker doesn't create them as root
mkdir -p ~/.cache/huggingface
mkdir -p ~/.cache/vllm_compilers/nv
mkdir -p ~/.cache/vllm_compilers/triton
mkdir -p ~/.cache/vllm_compilers/flashinfer
mkdir -p ~/.cache/vllm_compilers/torch

echo "==> Starting $CONTAINER_NAME ($IMAGE)..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --gpus all \
  --shm-size=32g \
  --ipc=host \
  --memory 115g \
  --memory-swap 125g \
  --oom-score-adj 800 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm_compilers/triton:/root/.triton \
  -v ~/.cache/vllm_compilers/nv:/root/.nv \
  -v ~/.cache/vllm_compilers/flashinfer:/root/.cache/flashinfer \
  -v ~/.cache/vllm_compilers/torch:/root/.cache/torch \
  -v "$BUILD_DIR/entrypoints/chat_utils.py:$VLLM_BASE/entrypoints/chat_utils.py:ro" \
  -v "$BUILD_DIR/tool_parsers/qwen3coder_tool_parser.py:$VLLM_BASE/tool_parsers/qwen3coder_tool_parser.py:ro" \
  -e MAX_JOBS=4 \
  -e TORCH_COMPILE_THREADS=4 \
  -e TORCHINDUCTOR_COMPILE_THREADS=4 \
  -e CUDA_NVCC_FLAGS="--threads 4" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e CUDA_CACHE_PATH=/root/.nv/ComputeCache \
  -e CUDA_CACHE_MAXSIZE=4294967296 \
  -e FLASHINFER_WORKSPACE_DIR=/root/.cache/flashinfer \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor \
  -e PYTHONWARNINGS="ignore::UserWarning:vllm.model_executor.layers.fla.ops" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e MODEL=Qwen/Qwen3-Coder-Next-FP8 \
  -e PORT=8000 \
  -e GPU_MEMORY_UTIL=0.85 \
  -e MAX_MODEL_LEN=131072 \
  -e MAX_NUM_SEQS=3 \
  -e VLLM_EXTRA_ARGS="--served-model-name qwen3-coder-next --kv-cache-dtype fp8_e4m3 --stream-interval 5 --enable-chunked-prefill --enable-prefix-caching --max-num-batched-tokens 8192 --enable-auto-tool-choice --tool-call-parser qwen3_coder" \
  "${IMAGE}" \
  serve

echo "==> Container started. Logs:"
echo "    docker logs -f $CONTAINER_NAME"
