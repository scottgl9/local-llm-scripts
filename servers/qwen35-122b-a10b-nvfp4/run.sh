#!/usr/bin/env bash
# run.sh — Launch Sehyo/Qwen3.5-122B-A10B-NVFP4 server using vllm nightly
# Based on community guidance: use qwen3_next_mtp speculative decoding with 3 tokens.
# Applies patches via volume mount:
#   - modelopt.py: MTP NVFP4 exclusion fix
#   - float_subbyte.h: SM121 (GB10) CUDA PTX FP4 conversion fix
#   - qwen3_5_mtp.py: Clamp OOB token IDs from MTP draft sampling (illegal memory access fix)
#   - compressed_tensors_moe.py: GB10 (SM121) workaround — clone MoE weight tensors to prevent
#     Marlin kernel NaN from CUDA unified memory address issues (vllm PR #36183)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTAINER_NAME="qwen35-122b-server"
IMAGE="vllm/vllm-openai:cu130-nightly"
MODEL="Sehyo/Qwen3.5-122B-A10B-NVFP4"
VLLM_BASE="/usr/local/lib/python3.12/dist-packages/vllm"
BUILD_DIR="$REPO_ROOT/.build/nightly"

# Remove old container if exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "==> Removing existing container '$CONTAINER_NAME'..."
  docker stop "$CONTAINER_NAME" 2>/dev/null || true
  docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

FI_BASE="/usr/local/lib/python3.12/dist-packages/flashinfer"

# Build patched files if not already done
if [[ ! -f "$BUILD_DIR/quantization/modelopt.py" || \
      ! -f "$BUILD_DIR/flashinfer/cutlass/include/cutlass/float_subbyte.h" || \
      ! -f "$BUILD_DIR/model_executor/models/qwen3_5_mtp.py" || \
      ! -f "$BUILD_DIR/quantization/compressed_tensors/compressed_tensors_moe.py" ]]; then
  echo "==> Building patched vLLM files..."
  bash "$REPO_ROOT/patches/build.sh" nightly
fi

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

mkdir -p ~/.cache/huggingface
mkdir -p ~/.cache/vllm_compilers/nv
mkdir -p ~/.cache/vllm_compilers/triton
mkdir -p ~/.cache/vllm_compilers/flashinfer
mkdir -p ~/.cache/vllm_compilers/torch

echo "==> Starting $CONTAINER_NAME ($IMAGE)..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --network host \
  --oom-score-adj 800 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/.cache/vllm_compilers/triton:/root/.triton \
  -v ~/.cache/vllm_compilers/nv:/root/.nv \
  -v ~/.cache/vllm_compilers/flashinfer:/root/.cache/flashinfer \
  -v ~/.cache/vllm_compilers/torch:/root/.cache/torch \
  -v "$BUILD_DIR/quantization/modelopt.py:$VLLM_BASE/model_executor/layers/quantization/modelopt.py:ro" \
  -v "$BUILD_DIR/flashinfer/cutlass/include/cutlass/float_subbyte.h:$FI_BASE/data/cutlass/include/cutlass/float_subbyte.h:ro" \
  -v "$BUILD_DIR/model_executor/models/qwen3_5_mtp.py:$VLLM_BASE/model_executor/models/qwen3_5_mtp.py:ro" \
  -v "$BUILD_DIR/quantization/compressed_tensors/compressed_tensors_moe.py:$VLLM_BASE/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py:ro" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MAX_JOBS=4 \
  -e TORCH_COMPILE_THREADS=4 \
  -e TORCHINDUCTOR_COMPILE_THREADS=4 \
  -e CUDA_NVCC_FLAGS="--threads 4" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CUDA_CACHE_PATH=/root/.nv/ComputeCache \
  -e CUDA_CACHE_MAXSIZE=4294967296 \
  -e FLASHINFER_WORKSPACE_DIR=/root/.cache/flashinfer \
  -e TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$IMAGE" \
  --model "$MODEL" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --max-model-len 131072 \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8 \
  --no-enable-chunked-prefill \
  --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 3}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --served-model-name qwen35-122b

echo "==> Container started. Logs:"
echo "    docker logs -f $CONTAINER_NAME"
