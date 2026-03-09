# Performance Analysis: Qwen3.5-122B-A10B-NVFP4 on DGX GB10 Spark

## Hardware Profile

| Property | Value |
|----------|-------|
| GPU | NVIDIA GB10 (DGX GB10 Spark) |
| SM architecture | SM121 (Compute Capability 12.1) |
| Memory | 119 GiB unified (CPU+GPU shared) |
| Memory bandwidth | ~200 GB/s (LPDDR5X unified memory) |
| CUDA version | 13.0 |
| Tensor cores | FP4, FP8, FP16, BF16 MMA supported |
| Key limitation | No `cvt.rn.satfinite.e2m1x2.f32` PTX (software E2M1 path) |

## 122B Model Architecture

Qwen3.5-122B-A10B-NVFP4 is a Mixture-of-Experts model with the following structure:

| Parameter | Value |
|-----------|-------|
| Total parameters | ~122B |
| Active parameters per token | ~10B |
| Hidden size | 4096 |
| Num layers | 48 (alternating linear/full attention) |
| Num experts | 256 |
| Experts per token (top-k) | 8 |
| Expert intermediate size (N) | 1024 |
| Shared expert intermediate size | 4096 |
| MTP modules | 1 |
| Quantization | NVFP4 (compressed-tensors) |

### Comparison with 35B variant

| Property | 35B (A3B) | 122B (A10B) | Ratio |
|----------|-----------|-------------|-------|
| Active params/token | ~3B | ~10B | 3.3x |
| Expert intermediate size (N) | 512 | 1024 | 2x |
| Experts per token | 8 | 8 | 1x |
| Num experts | 256 | 256 | 1x |
| Num layers | 36 | 48 | 1.33x |
| Hidden size | 2560 | 4096 | 1.6x |
| Observed decode speed | ~70.5 tps | ~33 tps | 0.47x |

## Why 33 tps Is the Expected Baseline

The 122B model has **3.3x more active parameters** per token than the 35B. On a bandwidth-limited device like GB10 (~200 GB/s), decode performance scales roughly inversely with active parameter count.

Expected: `70.5 / 3.3 = ~21 tps` (pure parameter scaling)

The observed 33 tps is **better** than pure scaling would predict, because:
1. NVFP4 quantization compresses weights 8x vs FP32 (4 bits/param)
2. Some overhead is fixed (attention, routing, embeddings) and doesn't scale with N
3. `VLLM_MTP_MOE_FP8=1` post-quantizes MTP expert weights to FP8, reducing MTP draft cost

## MoE Backend Status

### Marlin NVFP4 (current default)
- **Status**: Works on SM121
- **Performance**: Functional but not optimal. Marlin kernels were designed for A100/H100 GEMM shapes. The expert GEMM shapes (small M, medium N/K) are not Marlin's strong suit.
- **Used for**: Main model MoE layers (NVFP4 weights)

### CUTLASS MoE
- **Status**: Produces all-zero output on SM121
- **Workaround**: `VLLM_NVFP4_MOE_MARLIN=1` forces Marlin (set in vllm.sh)
- **Root cause**: Unknown — likely a CUTLASS kernel selection bug for SM121

### TRTLLM Fused MoE (FlashInfer)
- **Status**: Blocked by hardcoded `ICHECK_EQ(major, 10)` checks
- **Potential**: Highest-performance backend — fuses routing + expert GEMM + reduction
- **Fix**: Patch runtime checks to allow major=12 (Step 2 of optimization plan)
- **Risk**: Medium. CUTLASS underneath uses `ArchTag::kMinComputeCapability >= 100` which includes SM121

### Triton Fused MoE
- **Status**: Works (used for MTP MoE with `VLLM_MTP_MOE_FP8=1`)
- **Performance**: Using default kernel params — no tuned config for GB10
- **Fix**: Run autotuning for E=256,N=1024 (Step 3 of optimization plan)

## Per-Component Bottleneck Breakdown (122B Decode)

For single-token decode at 33 tps → ~30ms per token:

| Component | Est. time/token | % of total | Notes |
|-----------|----------------|------------|-------|
| MoE expert GEMM (48 layers) | ~18ms | 60% | 8 experts x 2 GEMMs x 48 layers, NVFP4 via Marlin |
| Attention (24 full + 24 linear) | ~5ms | 17% | MLA with FP8 KV cache, Triton MLA backend |
| Shared expert GEMM | ~3ms | 10% | N=4096 dense GEMM per layer |
| MTP draft (1 layer) | ~2ms | 7% | MoE + attention + LM head, FP8 with VLLM_MTP_MOE_FP8=1 |
| Routing + reduction | ~1ms | 3% | Top-8 selection from 256 experts |
| Other (embed, norm, lm_head) | ~1ms | 3% | |

**Primary bottleneck**: MoE expert GEMMs dominate at ~60% of decode time.

## Optimization Opportunities (Ranked by Impact)

### 1. TRTLLM Fused MoE on SM121 (HIGH impact)
- **Expected gain**: 15-30% MoE speedup → 5-10 tps
- **Effort**: Patch 2 lines in CUDA source + rebuild JIT cache
- **Risk**: Medium — kernel may not produce correct output on SM121
- **Approach**: Patch `gb10_compat.py` to modify runtime checks, test with `VLLM_USE_FLASHINFER_MOE_FP4=1`

### 2. Triton MoE Autotuning for 122B (MEDIUM impact)
- **Expected gain**: 5-15% MTP MoE improvement → 1-3 tps
- **Effort**: ~4-5 hours GPU time (automated)
- **Risk**: Low — purely finding better kernel parameters
- **Approach**: Run `benchmark_moe.py --tune` for E=256,N=1024

### 3. Triton MoE Autotuning for 35B (LOW impact for 122B)
- **Expected gain**: Only affects 35B model
- **Effort**: ~4-5 hours GPU time (automated)
- **Approach**: Run `benchmark_moe.py --tune` for E=256,N=512

### 4. CUTLASS MoE Fix (HIGH impact but hard)
- **Expected gain**: Could replace Marlin as default backend
- **Effort**: Requires deep debugging of CUTLASS kernel output
- **Risk**: High — root cause unknown
- **Status**: Deferred — Marlin workaround is functional

### 5. Attention Optimization
- **Expected gain**: Small (attention is ~17% of total)
- **Options**: FlashInfer MLA backend (if SM121 compatible), prefix caching
- **Status**: Triton MLA already in use, working well

## Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `VLLM_NVFP4_GEMM_BACKEND` | `marlin` | Force Marlin for NVFP4 (avoid CUTLASS zero-output bug) |
| `VLLM_NVFP4_MOE_MARLIN` | `1` | Force Marlin for MoE (same reason) |
| `VLLM_MTP_MOE_FP8` | `1` | Post-quantize MTP MoE weights to FP8 |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` (default) | TRTLLM MoE — blocked on SM121 until patched |
| `TORCH_CUDA_ARCH_LIST` | `12.1a` | Build for SM121 |

## Files

| File | Purpose |
|------|---------|
| `vllm/utils/gb10_compat.py` | SM121 FlashInfer patches |
| `vllm/model_executor/layers/fused_moe/configs/` | Triton MoE tuned configs |
| `benchmarks/kernels/benchmark_moe.py` | Autotuning script |
| `vllm.sh` | Launch script with SM121 env vars |
