# GB10 vLLM Launch Arguments Reference

Optimized vLLM arguments for Qwen3.5-122B-A10B-NVFP4 on NVIDIA DGX Spark (GB10).

## Hardware Summary

| Spec | Value |
|------|-------|
| GPU | NVIDIA GB10 (Blackwell, SM121 / cuda capability 12.1) |
| Memory | 128 GiB unified (CPU + GPU shared) |
| GPUs | 1 (single-GPU only) |
| IOMMU | ARM SMMU (makes mmap-based weight loading slow) |

## Model Architecture

Qwen3.5-122B-A10B-NVFP4 is a Mixture-of-Experts model with hybrid attention:

| Property | Value |
|----------|-------|
| Total layers | 48 |
| Full attention layers | 12 (standard KV cache) |
| GDN layers | 36 (linear attention, fixed-size state per request) |
| Experts | 256 total, 10 active per token |
| KV heads | 2 per layer |
| Head dimension | 256 |
| Quantization | NVFP4 (weights), FP8 (KV cache) |

**Key insight:** Only 12/48 layers use standard KV cache. The 36 GDN layers use fixed-size state regardless of sequence length. This makes long context very memory-efficient.

## Memory Budget

| Component | Size |
|-----------|------|
| Model weights (NVFP4) | ~76 GiB |
| MTP drafter (qwen3_next_mtp) | ~3 GiB |
| Total model | ~79 GiB |
| Available at 0.90 util | 128 × 0.90 = 115.2 GiB |
| Headroom for KV cache + activations | ~36 GiB |

### KV Cache Cost (FP8)

Per-token KV cache = 12 layers × 2 (K+V) × 2 KV heads × 256 head_dim × 1 byte (FP8) = **12,288 bytes ≈ 12 KiB/token**

At 131K context: 131,072 × 12 KiB ≈ 1.5 GiB per sequence. With 4 max sequences: worst case ~6 GiB. Well within the ~36 GiB headroom.

## Recommended Arguments

| Argument | Value | Default | Rationale |
|----------|-------|---------|-----------|
| `--gpu-memory-utilization` | `0.90` | 0.9 | Reclaims ~2.5 GiB vs 0.88 for KV cache; safe on GB10 with no other GPU consumers |
| `--max-model-len` | `131072` | model config | Full 131K context; feasible because only 12 layers use KV cache |
| `--max-num-seqs` | `4` | 256 | MoE is cheap per-sequence; 4 allows batching without excessive memory |
| `--kv-cache-dtype` | `fp8` | auto | Halves KV cache memory vs FP16; negligible quality impact |
| `--enable-chunked-prefill` | (flag) | off | **Required.** Without it, profiling allocates max_model_len tokens → OOM |
| `--max-num-batched-tokens` | `8192` | max_model_len | Controls profiling memory; 8192 prevents OOM during profile_run() |
| `--enable-prefix-caching` | (flag) | off | Reuses KV cache for shared prefixes (system prompts); saves recomputation |
| `--prefix-caching-hash-algo` | `xxhash` | sha256 | Faster hashing; sha256 is unnecessary for single-tenant |
| `--performance-mode` | `throughput` | balanced | Tunes scheduler for maximum tokens/sec |
| `--swap-space` | `0` | 4 (GiB) | Swap is useless on unified memory; frees 4 GiB for KV cache |
| `--safetensors-load-strategy` | `eager` | mmap | 6x faster model loading on GB10 (8m41s → 1m28s) due to ARM SMMU overhead |
| `--language-model-only` | (flag) | off | Skips vision encoder loading; saves memory |
| `--attention-backend` | `flashinfer` | auto | Required for FP8 KV cache + prefix caching on SM121 |
| `--trust-remote-code` | (flag) | off | Required for some model configs |
| `--speculative-config` | `method=qwen3_next_mtp, num_speculative_tokens=3` | off | MTP speculative decoding; 3 tokens is proven stable |

## Required Environment Variables

| Variable | Value | Rationale |
|----------|-------|-----------|
| `VLLM_NVFP4_GEMM_BACKEND` | `marlin` | Fastest NVFP4 GEMM on SM121 |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `1` | Improves Marlin MoE throughput on GB10 |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | TRTLLM MoE kernel rejects SM12x at runtime (see Backend Compatibility) |
| `VLLM_USE_DEEP_GEMM` | `0` | DeepGEMM not supported on SM121 |
| `VLLM_TEST_FORCE_FP8_MARLIN` | `1` | Forces Marlin for FP8 (Docker only; native build auto-selects CUTLASS) |
| `SAFETENSORS_FAST_GPU` | `1` | Enables pinned memory for faster weight loading |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Avoids allocator fragmentation on large models |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Avoids fork-related CUDA issues |
| `CUDA_CACHE_MAXSIZE` | `4294967296` | 4 GB cache for JIT-compiled kernels |

## Rejected Arguments

| Argument | Why Not |
|----------|---------|
| `--calculate-kv-scales` | Adds compute overhead per attention op for negligible quality gain |
| `--optimization-level O3` | O2 default is correct; O3 is less tested and may cause instability |
| `--block-size N` | Let FlashInfer auto-select optimal block size for the hardware |
| `--moe-backend marlin` | "auto" correctly selects Marlin via `VLLM_NVFP4_GEMM_BACKEND` env var |
| `--num-scheduler-steps N` | Removed from vLLM; no longer exists |

## Backend Compatibility

| Backend | SM121 Support | Notes |
|---------|---------------|-------|
| **Marlin MoE** | Supported | Recommended. Set via `VLLM_NVFP4_GEMM_BACKEND=marlin` |
| **FlashInfer** | Supported | Used for attention. Requires `ptxas-blackwell`; first compile takes 30+ min |
| **TensorRT-LLM MoE** | NOT supported | Hardcoded `ICHECK_EQ(major, 10)` in C++ kernel rejects SM12x at runtime |
| **DeepGEMM** | NOT supported | Disable with `VLLM_USE_DEEP_GEMM=0` |
| **FlashInfer MOE FP4** | NOT supported | Uses TRTLLM kernel internally; same SM12x rejection |
| **CUTLASS FP8** | Supported | Works natively since PR #35568 recognized SM121 as SM120 family |

## Troubleshooting

### OOM during profiling (startup crash)

**Symptom:** Container killed by OOM during `profile_run()` before serving any requests.

**Cause:** Without chunked prefill, profiling runs a dummy forward pass with `max_model_len` tokens. At 131K tokens this requires ~305 GiB virtual memory.

**Fix:** Always use `--enable-chunked-prefill --max-num-batched-tokens 8192`. The 8192 cap limits profiling memory to a manageable size.

### OOM at runtime

**Symptom:** Server crashes under load after successful startup.

**Fix (in order):**
1. Reduce `--gpu-memory-utilization` from 0.90 to 0.88
2. Reduce `--max-model-len` from 131072 to 98304 or 65536
3. Reduce `--max-num-seqs` from 4 to 2

### Slow model loading (8+ minutes)

**Symptom:** Server takes 8-10 minutes to load weights before serving.

**Cause:** Default `mmap` strategy is slow on GB10 due to ARM SMMU translation overhead.

**Fix:** Use `--safetensors-load-strategy eager` (loads in ~1.5 minutes). Alternatively, set host read-ahead: `sudo bash -c "echo 8192 > /sys/block/nvme0n1/queue/read_ahead_kb"`

### First startup takes 30+ minutes

**Symptom:** First launch after a fresh install takes 30-45 minutes before serving.

**Cause:** Triton/FlashInfer kernels must be JIT-compiled for SM121 via `ptxas-blackwell`.

**Fix:** Mount compiler cache directories to persist across restarts:
- `~/.cache/vllm_compilers/triton` → `/root/.triton`
- `~/.cache/vllm_compilers/flashinfer` → `/root/.cache/flashinfer`
- `~/.cache/vllm_compilers/torch` → `/root/.cache/torch`
