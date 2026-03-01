#!/usr/bin/env bash
# run.sh — Launch qwen3.5 NVFP4 server (vincentzed-hf/Qwen3-Coder-Next-NVFP4 or similar)
# Uses avarok/vllm-dgx-spark:v11.

set -euo pipefail

CONTAINER_NAME="qwen3-nvfp4-server"
IMAGE="avarok/vllm-dgx-spark:v11"

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
  -p 8000:8000 \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e USE_FASTSAFETENSOR=true \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "$IMAGE" \
  serve vincentzed-hf/Qwen3-Coder-Next-NVFP4 \
  --served-model-name qwen3-coder-next \
  --host 0.0.0.0 --port 8000 \
  --no-async-scheduling \
  --gpu-memory-utilization 0.8 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-prefix-caching \
  --max-num-seqs 8 \
  --max-num-batched-tokens 1024 \
  --enable-auto-tool-choice \
  --quantization modelopt \
  --tool-call-parser qwen3_coder \
  --max-model-len 16384

echo "==> Container started. Logs:"
echo "    docker logs -f $CONTAINER_NAME"
