#!/bin/bash
# vllm.sh — Build and launch vLLM on DGX GB10 (SM 12.1 / CUDA 13.0)
#
# Commands:
#   build                          Compile vLLM into .venv-gb10/ (run once)
#   launch [args]                  Start the OpenAI-compatible API server (raw args)
#   shell                          Drop into an activated venv shell
#   Qwen3.5-NVFP4 [args]          Qwen3.5-122B MoE NVFP4 + speculative decoding
#   Qwen3.5-35B-NVFP4 [args]      Qwen3.5-35B-A3B MoE NVFP4 + speculative decoding
#   Qwen3-Coder-Next-NVFP4 [args] GadflyII Qwen3-Coder-Next NVFP4
#   Qwen3-Coder-Next-FP8 [args]   Qwen/Qwen3-Coder-Next dense FP8
#   minimax [args]                 MiniMax M2.5 REAP 139B NVFP4
#
# Context window (default 65536 — override with MAX_MODEL_LEN):
#   MAX_MODEL_LEN=32768 ./vllm.sh Qwen3.5-NVFP4
#
# Build:
#   ./vllm.sh build
#   rm -rf .venv-gb10 && ./vllm.sh build   # clean rebuild
#
# Model path overrides:
#   QWEN35_MODEL=/path/to/snapshot               ./vllm.sh Qwen3.5-NVFP4
#   QWEN35_35B_MODEL=Sehyo/Qwen3.5-35B-A3B-NVFP4 ./vllm.sh Qwen3.5-35B-NVFP4
#   QWEN3_CODER_NVFP4_MODEL=GadflyII/...         ./vllm.sh Qwen3-Coder-Next-NVFP4
#   QWEN3_CODER_MODEL=Qwen/Qwen3-Coder-Next-FP8  ./vllm.sh Qwen3-Coder-Next-FP8
#   MINIMAX_MODEL=/path/to/model                  ./vllm.sh minimax
#
# Key environment overrides:
#   MAX_MODEL_LEN              Context window tokens (default: 65536)
#   VLLM_QUANTIZE_LM_HEAD=nvfp4  Post-quantize lm_head to NVFP4 (~15% speedup)
#   VLLM_NVFP4_GEMM_BACKEND    marlin (default) | cutlass | flashinfer-cutlass
#
# Slow model loading on GB10 (safetensors mmap is slow due to ARM SMMU):
#   --safetensors-load-strategy eager   ~6x faster than mmap (8m41s → 1m28s)
#   --load-format fastsafetensors       Fastest; verify FP4/FP8 compat first
#   sudo bash -c "echo 8192 > /sys/block/nvme0n1/queue/read_ahead_kb"
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
VLLM_DIR="$(pwd)"
VENV_DIR="${VLLM_DIR}/.venv-gb10"
PYTHON="python3.12"

# ── Context window default ────────────────────────────────────────────────────
# Override before calling: MAX_MODEL_LEN=32768 ./vllm.sh <preset>
#MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# ── Shared launch arg groups ──────────────────────────────────────────────────

# Standard server binding (all presets)
SERVER_ARGS=(--host 0.0.0.0 --port 8000)

# Common to all Qwen3 presets: model name + tool calling
QWEN3_ARGS=(
    --served-model-name qwen3-coder-next
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
)

# Common to Qwen3-Coder-Next presets: chunked prefill + prefix caching
QWEN3_CODER_ARGS=(
    --enable-chunked-prefill
    --enable-prefix-caching
    --max-num-batched-tokens 8192
)

# ── Build-time env vars ───────────────────────────────────────────────────────
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="12.1a"
# Do NOT set -arch here: cmake handles arch via TORCH_CUDA_ARCH_LIST.
# Adding -arch=sm_121a conflicts with cmake's own sm_121 gencode flags.
# ENABLE_NVFP4_SM120=1: required for scaled_fp4_quant on SM120/SM121.
# cmake sets this automatically via FP4_ARCHS when TORCH_CUDA_ARCH_LIST="12.1a",
# but we also set it here via NVCC_PREPEND_FLAGS as a safety net.
export NVCC_PREPEND_FLAGS="-DENABLE_TCGEN05_HARDWARE=1 -DENABLE_NVFP4_SM120=1"
export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"
# Set MAX_JOBS before calling this script to override (default: 8 for GB10).
export MAX_JOBS="${MAX_JOBS:-8}"
export CUDA_NVCC_FLAGS="${CUDA_NVCC_FLAGS:---threads 4}"

# ── Runtime env vars ──────────────────────────────────────────────────────────
setup_runtime_env() {
    export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    export TRITON_PTXAS_PATH="${CUDA_HOME}/bin/ptxas"

    # CUDA kernel cache — persists JIT-compiled kernels across restarts
    export CUDA_CACHE_PATH="${HOME}/.nv/ComputeCache"
    export CUDA_CACHE_MAXSIZE=4294967296   # 4 GB

    # FlashInfer + inductor cache dirs
    export FLASHINFER_WORKSPACE_DIR="${HOME}/.cache/flashinfer"
    export TORCHINDUCTOR_CACHE_DIR="${HOME}/.cache/torch/inductor"

    # Faster safetensors weight loading via GPU pinned memory
    export SAFETENSORS_FAST_GPU=1

    # torch.compile / inductor thread count (avoids saturating CPUs during JIT)
    export TORCH_COMPILE_THREADS=4
    export TORCHINDUCTOR_COMPILE_THREADS=4

    # NVFP4 GEMM via Marlin (fastest on SM121); override with VLLM_NVFP4_GEMM_BACKEND
    export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-marlin}"
    # Marlin atomic-add reduction (improves throughput on GB10)
    export VLLM_MARLIN_USE_ATOMIC_ADD=1
    # Force Marlin FP8 GEMM on SM121 (avoids CUTLASS FP8 issues on GB10).
    export VLLM_TEST_FORCE_FP8_MARLIN=1
    # FlashInfer TRTLLM MoE FP4 is NOT supported on SM121 (GB10).
    # The TRTLLM kernel has a hardcoded C++ ICHECK_EQ(major, 10) that rejects SM12x
    # at runtime even after JIT compilation succeeds. Use Marlin MoE instead.
    export VLLM_USE_FLASHINFER_MOE_FP4=0
    # Disable DeepGEMM (not supported on SM121)
    export VLLM_USE_DEEP_GEMM=0
    # spawn avoids fork-related CUDA issues with multiprocessing
    export VLLM_WORKER_MULTIPROC_METHOD=spawn

    # Expandable segments avoids allocator fragmentation on large models
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo -e "\033[1;34m[vllm]\033[0m $*"; }
success() { echo -e "\033[1;32m[vllm]\033[0m $*"; }
warn()    { echo -e "\033[1;33m[vllm]\033[0m $*"; }
die()     { echo -e "\033[1;31m[vllm]\033[0m ERROR: $*" >&2; exit 1; }

# ── Subcommands ───────────────────────────────────────────────────────────────

cmd_build() {
    info "Building vLLM on GB10 (SM 12.1, CUDA 13.0)"
    info "  Source : ${VLLM_DIR}"
    info "  Venv   : ${VENV_DIR}"
    echo ""

    [[ -d "${VLLM_DIR}" ]] || die "vllm source not found at ${VLLM_DIR}"
    [[ -x "${CUDA_HOME}/bin/nvcc" ]] || die "nvcc not found at ${CUDA_HOME}/bin/nvcc. Is CUDA installed?"

    if [[ ! -d "${VENV_DIR}" ]]; then
        info "Creating venv with ${PYTHON}..."
        "${PYTHON}" -m venv "${VENV_DIR}"
    else
        warn "Venv already exists — reusing (delete ${VENV_DIR} to rebuild from scratch)"
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    # Pin the pytorch whl index for ALL subsequent pip calls in this shell.
    # Without this, pip install -r requirements/build.txt re-resolves torch from
    # PyPI and pulls the CPU wheel, overwriting the cu130 build.
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu130"

    info "Upgrading pip..."
    pip install -q --upgrade pip

    info "Installing PyTorch 2.10 for CUDA 13.0 (cu130, aarch64)..."
    pip install "torch==2.10.0+cu130" --index-url https://download.pytorch.org/whl/cu130

    info "Installing Triton >=3.3.0 (SM121a ptxas support)..."
    # triton <3.3.0 produces a fatal ptxas error on sm_121a
    pip install "triton>=3.3.0"

    info "Installing FlashInfer (pre-release, SM120/121 NVFP4 kernels)..."
    pip install flashinfer-python --pre

    info "Installing vLLM build requirements..."
    pip install -r "${VLLM_DIR}/requirements/build.txt"

    # Force-reinstall cu130 before cmake: requirements/build.txt has bare
    # torch==2.10.0 which pip may resolve to the CPU wheel from PyPI.
    info "Re-pinning torch+cu130 before CUDA compilation..."
    pip install --force-reinstall --no-deps "torch==2.10.0+cu130" \
        --index-url https://download.pytorch.org/whl/cu130

    # GB10-specific source modifications on this branch:
    #   CMakeLists.txt: disabled scaled_mm_blockwise_sm121_fp8.cu (CUTLASS macro
    #     fails under CUDA 13.0) and grouped_mm_gb10_native_v109.cu (missing symbol).
    #   scaled_mm_entry.cu: SM121 dispatch added before SM120 branch.
    #   scaled_mm_c3x_sm121.cu: fixed to call cutlass_scaled_mm_sm121_fp8().
    #   scaled_mm_helper.hpp: nullptr guard for blockwise_func on SM121.
    #   vllm_flash_attn/__init__.py: warn instead of raise on missing FA2/FA3.
    #   setup.py: FA2/FA3 CMakeExtension marked optional=True.
    info "Compiling vLLM (editable install, no-build-isolation)..."
    info "  TORCH_CUDA_ARCH_LIST = ${TORCH_CUDA_ARCH_LIST}"
    info "  NVCC_PREPEND_FLAGS   = ${NVCC_PREPEND_FLAGS}"
    info "  MAX_JOBS             = ${MAX_JOBS}"
    pushd "${VLLM_DIR}" > /dev/null
    pip install -e . --no-build-isolation -v 2>&1 | tee /tmp/vllm-gb10-build.log
    popd > /dev/null

    # pip install -e . can drag in CPU torch via transitive deps; re-pin last.
    info "Pinning torch+cu130 (final reinstall)..."
    pip install --force-reinstall --no-deps "torch==2.10.0+cu130" \
        --index-url https://download.pytorch.org/whl/cu130

    info "Verifying installation..."
    TORCH_VER=$(python -c "import torch; print(torch.__version__)")
    TORCH_CUDA=$(python -c "import torch; print(torch.version.cuda or 'None')")
    VLLM_VER=$(python -c "import vllm; print(getattr(vllm, '__version__', 'ok'))" 2>/dev/null || echo "ok")
    PLATFORM=$(python -c "from vllm.platforms import current_platform; print(current_platform.get_device_capability())" 2>/dev/null || echo "(GPU not visible in this shell)")

    echo ""
    echo "  torch              : ${TORCH_VER}"
    echo "  torch.version.cuda : ${TORCH_CUDA}"
    echo "  vllm               : ${VLLM_VER}"
    echo "  platform cap       : ${PLATFORM}"
    echo ""

    # Verify NVFP4 SM120/SM121 kernels were compiled in.
    # If ENABLE_NVFP4_SM120=1 was not active during compilation, scaled_fp4_quant
    # falls through to TORCH_CHECK_NOT_IMPLEMENTED and produces "!!!!" output on GB10.
    #
    # Use nm -D (dynamic symbol table) via process substitution rather than
    # "strings ... | grep -q".  With set -o pipefail, grep -q exits 0 as soon as
    # it finds a match on the large .so, which sends SIGPIPE (exit 141) to the
    # strings process.  pipefail then returns 141 — a false negative even when the
    # symbol IS present.  Process substitution avoids that: nm runs as an
    # independent process, so only grep's exit code reaches the `if`.
    local so_path="${VLLM_DIR}/vllm/_C.abi3.so"
    if grep -qF "scaled_fp4_quant_sm1xxa" <(nm -D "${so_path}" 2>/dev/null); then
        success "NVFP4 SM120/SM121 kernels present in _C.abi3.so"
    else
        warn "WARNING: NVFP4 SM120/SM121 kernels NOT found in _C.abi3.so"
        warn "  ENABLE_NVFP4_SM120=1 was not compiled in — Qwen3.5-NVFP4 will output '!!!!'"
        warn "  Ensure TORCH_CUDA_ARCH_LIST='12.1a' is set and rebuild."
    fi

    if [[ "${TORCH_CUDA}" == "None" || "${TORCH_CUDA}" == "" ]]; then
        warn "torch.version.cuda is None — CPU-only torch was installed."
        warn "Fix: pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130"
    else
        success "Build complete. Log: /tmp/vllm-gb10-build.log"
    fi
}

cmd_launch() {
    [[ -d "${VENV_DIR}" ]] || die "Venv not found at ${VENV_DIR}. Run: ./vllm.sh build"

    # Kill any existing vLLM server and wait for GPU memory to drain.
    # Launching while the previous server still holds VRAM risks GPU OOM
    # which can cascade into a full system hang on GB10 (unified memory).
    #
    # We kill both the API server (matches "vllm.entrypoints.openai") and any
    # orphaned EngineCore subprocesses (which rename themselves to "VLLM::EngineCore"
    # and are invisible to pgrep -f "vllm.entrypoints.openai").
    local existing engine_cores
    existing=$(pgrep -f "vllm.entrypoints.openai" 2>/dev/null || true)
    engine_cores=$(pgrep -x "VLLM::EngineCore" 2>/dev/null || true)
    local all_pids="${existing} ${engine_cores}"
    all_pids="${all_pids## }"  # trim leading space
    all_pids="${all_pids%% }"  # trim trailing space
    if [[ -n "${all_pids// /}" ]]; then
        info "Stopping existing vLLM server (PIDs: ${all_pids})..."
        # shellcheck disable=SC2086
        kill ${all_pids} 2>/dev/null || true
        local waited=0
        while pgrep -f "vllm.entrypoints.openai" > /dev/null 2>&1 \
              || pgrep -x "VLLM::EngineCore" > /dev/null 2>&1; do
            sleep 1
            (( waited++ )) || true
            if (( waited >= 30 )); then
                info "  Sending SIGKILL after ${waited}s..."
                pkill -9 -f "vllm.entrypoints.openai" 2>/dev/null || true
                pkill -9 -x "VLLM::EngineCore" 2>/dev/null || true
                break
            fi
        done
        # Give the NVIDIA driver time to reclaim GPU memory
        info "  Waiting 5s for GPU memory to be released..."
        sleep 5
    fi

    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    setup_runtime_env

    info "Launching vLLM OpenAI-compatible server"
    info "  VLLM_NVFP4_GEMM_BACKEND = ${VLLM_NVFP4_GEMM_BACKEND}"
    info "  VLLM_QUANTIZE_LM_HEAD   = ${VLLM_QUANTIZE_LM_HEAD:-<unset>}"
    info "  SAFETENSORS_FAST_GPU    = ${SAFETENSORS_FAST_GPU}"
    echo ""

    # cd away from the source dir: if CWD contains a directory named 'vllm',
    # Python adds it to sys.path and shadows the editable install.
    cd /tmp
    exec python -m vllm.entrypoints.openai.api_server "$@"
}

cmd_qwen35_nvfp4() {
    # Auto-detect latest snapshot under the Sehyo HF cache, or use QWEN35_MODEL
    local base="${HOME}/.cache/huggingface/hub/models--Sehyo--Qwen3.5-122B-A10B-NVFP4/snapshots"
    local model="${QWEN35_MODEL:-}"
    if [[ -z "${model}" ]]; then
        model=$(ls -td "${base}"/*/  2>/dev/null | head -1)
        model="${model%/}"
        [[ -n "${model}" ]] || die "Qwen3.5-NVFP4 not found at ${base}. Set QWEN35_MODEL=/path/to/snapshot"
    fi

    local spec_args=()
    if [[ "${DISABLE_MTP:-}" != "1" ]]; then
        spec_args=(--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":3}')
        info "Preset: Qwen3.5-122B-A10B-NVFP4 (compressed-tensors, speculative qwen3_next_mtp)"
    else
        info "Preset: Qwen3.5-122B-A10B-NVFP4 (compressed-tensors, MTP DISABLED)"
    fi
    info "  Model : ${model}"
    info "  MaxLen: ${MAX_MODEL_LEN}"

    cmd_launch \
        --model "${model}" \
        --quantization compressed-tensors \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization 0.88 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs 3 \
        --attention-backend flashinfer \
        "${spec_args[@]}" \
        --no-enable-chunked-prefill \
        --reasoning-parser qwen3 \
        --language-model-only \
        --trust-remote-code \
        "${SERVER_ARGS[@]}" \
        "${QWEN3_ARGS[@]}" \
        "$@"
}

cmd_qwen35_35b_nvfp4() {
    local model="${QWEN35_35B_MODEL:-Sehyo/Qwen3.5-35B-A3B-NVFP4}"

    local spec_args=()
    if [[ "${DISABLE_MTP:-}" != "1" ]]; then
        spec_args=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
        info "Preset: Qwen3.5-35B-A3B-NVFP4 (compressed-tensors, speculative mtp)"
    else
        info "Preset: Qwen3.5-35B-A3B-NVFP4 (compressed-tensors, MTP DISABLED)"
    fi
    info "  Model : ${model}"
    info "  MaxLen: ${MAX_MODEL_LEN}"

    cmd_launch \
        --model "${model}" \
        --quantization compressed-tensors \
        --kv-cache-dtype fp8 \
        --gpu-memory-utilization 0.95 \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs 8 \
        --attention-backend flashinfer \
        "${spec_args[@]}" \
        --enable-chunked-prefill \
        --max-num-batched-tokens 8192 \
        --enable-prefix-caching \
        --swap-space 0 \
        --safetensors-load-strategy eager \
        --language-model-only \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        "${SERVER_ARGS[@]}" \
        "${QWEN3_ARGS[@]}" \
        "$@"
}

cmd_qwen3_coder_next_nvfp4() {
    local model="${QWEN3_CODER_NVFP4_MODEL:-GadflyII/Qwen3-Coder-Next-NVFP4}"
    local ctx="${MAX_MODEL_LEN:-131072}"

    info "Preset: Qwen3-Coder-Next-NVFP4 (GadflyII, compressed-tensors, chunked-prefill)"
    info "  Model : ${model}"
    info "  MaxLen: ${ctx}"

    cmd_launch \
        --model "${model}" \
        --quantization compressed-tensors \
        --kv-cache-dtype fp8_e4m3 \
        --gpu-memory-utilization 0.8 \
        --max-num-seqs 64 \
        --max-model-len "${ctx}" \
        "${SERVER_ARGS[@]}" \
        "${QWEN3_ARGS[@]}" \
        "${QWEN3_CODER_ARGS[@]}" \
        "$@"
}

cmd_qwen3_coder_next_fp8() {
    local model="${QWEN3_CODER_MODEL:-Qwen/Qwen3-Coder-Next-FP8}"
    local ctx="${MAX_MODEL_LEN:-131072}"

    info "Preset: Qwen3-Coder-Next-FP8 (dense FP8, chunked-prefill, prefix-caching)"
    info "  Model : ${model}"
    info "  MaxLen: ${ctx}"

    cmd_launch \
        --model "${model}" \
        --kv-cache-dtype fp8_e4m3 \
        --gpu-memory-utilization 0.86 \
        --max-num-seqs 4 \
        --max-model-len "${ctx}" \
        "${SERVER_ARGS[@]}" \
        "${QWEN3_ARGS[@]}" \
        "${QWEN3_CODER_ARGS[@]}" \
        "$@"
}

cmd_minimax() {
    local model="${MINIMAX_MODEL:-${HOME}/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4}"
    export VLLM_QUANTIZE_LM_HEAD="${VLLM_QUANTIZE_LM_HEAD:-nvfp4}"

    info "Preset: MiniMax M2.5 REAP 139B NVFP4 (modelopt_fp4, lm_head=${VLLM_QUANTIZE_LM_HEAD})"
    info "  Model : ${model}"

    cmd_launch \
        --model "${model}" \
        --quantization modelopt_fp4 \
        --kv-cache-dtype fp8 \
        --max-model-len 4096 \
        --enforce-eager \
        "$@"
}

cmd_shell() {
    [[ -d "${VENV_DIR}" ]] || die "Venv not found at ${VENV_DIR}. Run: ./vllm.sh build"
    info "Activating vLLM venv — type 'deactivate' to exit"
    setup_runtime_env
    exec bash --rcfile <(echo "source '${VENV_DIR}/bin/activate'; PS1='(vllm-gb10) \u@\h:\w\$ '")
}

usage() {
    cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  build                          Compile vLLM into .venv-gb10/
  launch [vllm args]             Start the OpenAI-compatible API server
  shell                          Drop into an activated venv shell

  Qwen3.5-NVFP4 [args]          Qwen3.5-122B MoE NVFP4, speculative decoding
  Qwen3.5-35B-NVFP4 [args]      Qwen3.5-35B-A3B MoE NVFP4, speculative decoding
  Qwen3-Coder-Next-NVFP4 [args] GadflyII/Qwen3-Coder-Next-NVFP4
  Qwen3-Coder-Next-FP8 [args]   Qwen/Qwen3-Coder-Next-FP8
  minimax [args]                 MiniMax M2.5 REAP 139B NVFP4

Context window (default: ${MAX_MODEL_LEN}):
  MAX_MODEL_LEN=32768 ./vllm.sh Qwen3.5-NVFP4

Model path overrides:
  QWEN35_MODEL=<path>              Override Qwen3.5-NVFP4 snapshot path
  QWEN35_35B_MODEL=<path>          Override Qwen3.5-35B-NVFP4 model
  QWEN3_CODER_NVFP4_MODEL=<path>  Override Qwen3-Coder-Next-NVFP4 model
  QWEN3_CODER_MODEL=<path>        Override Qwen3-Coder-Next-FP8 model
  MINIMAX_MODEL=<path>             Override MiniMax model path

Environment overrides:
  MAX_MODEL_LEN=N              Context window tokens (default: 65536)
  VLLM_QUANTIZE_LM_HEAD=nvfp4  Post-quantize lm_head to NVFP4 (~15% speedup)
  VLLM_NVFP4_GEMM_BACKEND=...  marlin (default) | cutlass | flashinfer-cutlass

Examples:
  ./vllm.sh Qwen3.5-NVFP4
  MAX_MODEL_LEN=32768 ./vllm.sh Qwen3.5-NVFP4
  QWEN35_MODEL=/path/to/snapshot ./vllm.sh Qwen3.5-NVFP4
  ./vllm.sh minimax --max-model-len 8192

EOF
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
CMD="${1:-}"
shift || true

case "${CMD}" in
    build)   cmd_build ;;
    launch)  cmd_launch "$@" ;;
    shell)   cmd_shell ;;
    Qwen3.5-NVFP4|qwen3.5-nvfp4|qwen35-nvfp4) cmd_qwen35_nvfp4 "$@" ;;
    Qwen3.5-35B-NVFP4|qwen35-35b-nvfp4) cmd_qwen35_35b_nvfp4 "$@" ;;
    Qwen3-Coder-Next-NVFP4|qwen3-coder-next-nvfp4) cmd_qwen3_coder_next_nvfp4 "$@" ;;
    Qwen3-Coder-Next-FP8|qwen3-coder-next-fp8) cmd_qwen3_coder_next_fp8 "$@" ;;
    minimax|MiniMax) cmd_minimax "$@" ;;
    ""|help|-h|--help) usage ;;
    *) die "Unknown command: ${CMD}. Run './vllm.sh help' for usage." ;;
esac
