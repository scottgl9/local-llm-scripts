#!/usr/bin/env bash
# run.sh — Launch qwen3-next-nvfp4 server (nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4)
# Uses dgx-vllm-mtp-ready:v23 with MTP speculative decoding.

set -euo pipefail

CONTAINER_NAME="qwen3-nvfp4-server"
IMAGE="docker.io/library/dgx-vllm-mtp-ready:v23"

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
  --memory 119g \
  --memory-swap 125g \
  --oom-score-adj 800 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MODEL=nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4 \
  -e PORT=8000 \
  -e GPU_MEMORY_UTIL=0.85 \
  -e MAX_MODEL_LEN=65536 \
  -e MAX_NUM_SEQS=32 \
  -e VLLM_EXTRA_ARGS="--attention-backend flashinfer --kv-cache-dtype fp8 --speculative-config.method qwen3_next_mtp --speculative-config.num_speculative_tokens 2 --no-enable-chunked-prefill --served-model-name qwen3-coder-next --enable-auto-tool-choice --tool-call-parser qwen3_coder" \
  ${IMAGE} \
  serve

echo "==> Container started. Logs:"
echo "    docker logs -f $CONTAINER_NAME"
