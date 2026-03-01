#!/usr/bin/env bash
# run.sh — Launch qwen3-coder-next-nvfp4 server using dgx-vllm-mtp-ready:v23 (local build)
# Serves Sehyo/Qwen3.5-122B-A10B-NVFP4 with MTP speculative decoding.

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
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm_patches/modelopt.py:/app/vllm/vllm/model_executor/layers/quantization/modelopt.py:ro \
  -v ~/.cache/vllm_patches/qwen3_5_mtp.py:/app/vllm/vllm/model_executor/models/qwen3_5_mtp.py:ro \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MODEL=Sehyo/Qwen3.5-122B-A10B-NVFP4 \
  -e PORT=8000 \
  -e GPU_MEMORY_UTIL=0.84 \
  -e MAX_MODEL_LEN=65536 \
  -e MAX_NUM_SEQS=8 \
  -e VLLM_EXTRA_ARGS="--attention-backend flashinfer --kv-cache-dtype fp8 --speculative-config.method qwen3_next_mtp --speculative-config.num_speculative_tokens 2 --no-enable-chunked-prefill --served-model-name qwen3-coder-next --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only" \
  ${IMAGE} \
  serve

echo "==> Container started. Logs:"
echo "    docker logs -f $CONTAINER_NAME"
