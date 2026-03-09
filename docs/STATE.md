# Session State — SGLang GB10 Optimization (2026-03-09)

## Current Status

**Server**: RUNNING — Qwen3.5-35B-A3B-NVFP4 with Marlin FP4 dense GEMM + GDN post-quant + MTP
**Log**: `tail -f /tmp/sglang.log`
**Model**: `Sehyo/Qwen3.5-35B-A3B-NVFP4` on DGX GB10 (SM121, 119GB unified memory)

## Session 2026-03-09 — Summary

### Problem
Qwen3.5-35B-A3B-NVFP4 on GB10 (SM121) produced garbage output ("!!!!") because
CUTLASS FP4 GEMM produces corrupt output (all zeros/NaN) on SM121. This affects
ALL FP4 operations: MoE experts, attention projections, shared experts, and GDN
post-quantization.

### Root Causes Identified & Fixed

#### 1. Marlin MoE kernel: `dequant_fp8_scales` commented out
**File**: `python/sglang/jit_kernel/csrc/gemm/marlin_moe/marlin_template.h`
- The `dequant_fp8_scales` code block for FP4 weight types was commented out in
  sglang's MoE fork (but active in the non-MoE kernel and in vLLM)
- Without runtime S0E5M3→BF16 scale conversion, scale bytes were misinterpreted
  as raw BF16 values → SiLU output overflow to Inf

#### 2. `moe_sum_reduce` missing 3rd argument
**File**: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4_moe.py`
- `moe_sum_reduce` requires 3 args: (input, output, routed_scaling_factor)
- Was called with only 2 → all-zero output
- Added `1.0` as the third argument

#### 3. NVFP4 scale processing for Marlin MoE
**File**: `python/sglang/srt/layers/quantization/marlin_utils.py`
- Added `nvfp4_marlin_process_scales()` — converts E4M3 scales to S0E5M3 format
- Added `nvfp4_marlin_process_global_scale()` — applies exponent bias correction
- Added `nvfp4_marlin_interleave_scales()` — byte-interleaves adjacent K-group
  scale rows so Marlin kernel's `warp_row % 2` selects correct per-16 scale
- Ported from vLLM's `marlin_utils_fp4.py`

#### 4. GDN post-quantization via Marlin FP4 on SM121
**File**: `python/sglang/srt/layers/quantization/nvfp4_post_quant.py`
- GDN projections (in_proj_qkv, in_proj_z) are BF16 in the checkpoint
- On SM120, they get post-quantized to NVFP4 using CUTLASS FP4 GEMM
- CUTLASS FP4 is broken on SM121 → previously skipped post-quant entirely
- **Fix**: Added `_quantize_bf16_to_raw_nvfp4()` — pure Python BF16→NVFP4
  quantization producing checkpoint-compatible format (not CUTLASS-swizzled)
- Added `_convert_to_marlin_fp4()` — repacks to Marlin tile layout with
  interleaved scales
- Added `MarlinFp4PostQuantLinearMethod` — inference via `gptq_marlin_gemm`
- Result: 60 GDN layers now post-quantized on SM121, +13.5% decode speedup

**File**: `python/sglang/srt/models/qwen3_5.py`
- Removed SM121 skip guard — post-quant now runs on all SM120+ GPUs
- Applied to both `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`

#### 5. CUTLASS FP4 dense GEMM broken on SM121 (non-MoE linear layers)
**File**: `python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`
- Attention qkv_proj, o_proj, and shared expert gate_up/down are NVFP4-quantized
  (NOT in the model's ignore list) and use `CompressedTensorsW4A4Fp4` + `fp4_gemm`
  (CUTLASS) → all-zero output on SM121
- **v1 fix** (superseded): BF16 dequant fallback — 4x memory expansion, ~60 TPS
- **v2 fix** (broken — garbage output): Marlin FP4 dense GEMM without scale interleaving.
  The Marlin kernel's FP4 path divides `s_tb_groups` by 2, causing adjacent K-groups
  (16 elements each) to share one scale instead of having distinct scales.
- **v3 fix** (current): Marlin FP4 dense GEMM with **byte-interleaved scales**.
  The kernel loads two adjacent K-group scale rows per tile and uses `warp_row % 2`
  to select between even/odd int2 (8-byte) chunks. By interleaving the fp8 scale
  rows in 8-byte chunks, each K-group correctly gets its own scale.
  - Weight repack: `gptq_marlin_repack()` → Marlin tile layout
  - Scales: `marlin_permute_scales()` → `nvfp4_marlin_process_scales()` (S0E5M3)
    → `nvfp4_marlin_interleave_scales()` (8-byte chunk interleaving)
  - Global scale: invert (1/scale) → `nvfp4_marlin_process_global_scale()` (bias correction)
  - Same interleaving applied to MoE scales for consistency
  - BF16 dequant kept as fallback for layers with N%64!=0 or K%128!=0

### Debugging Journey (chronological)
1. MoE Marlin GEMM output had SiLU overflow → fixed `dequant_fp8_scales`
2. MoE output was all zeros → fixed `moe_sum_reduce` missing argument
3. NaN from GDN layer 2 (linear_attention) → skipped post-quant on SM121
4. NaN from layer 27 self_attention → traced to qkv_proj outputting NaN
5. ALL attention layers output abs_max=0 from qkv_proj → CUTLASS FP4 broken
6. Added BF16 dequant fallback → no more NaN, but all zeros + residual exploded to 1.21e+23
7. Residual explosion traced to shared expert using wrong dequant formula
   (was multiplying by global_scale, should divide) → fixed
8. All layers now produce correct non-zero output → coherent text generation
9. Replaced BF16 dequant with Marlin FP4 dense GEMM → faster but garbage output
10. Traced to Marlin kernel applying same scale to adjacent K-group pairs (groups 0,1
    share scale[0], groups 2,3 share scale[2], etc.) due to `/2` on `s_tb_groups`
11. Fixed by byte-interleaving adjacent fp8 scale rows in 8-byte chunks so kernel's
    `warp_row % 2` selects correct scale → cos_sim 0.71→0.9998, coherent output
12. Added GDN NVFP4 post-quant via Marlin FP4 — pure Python quantization to avoid
    CUTLASS-swizzled format, then Marlin repack with interleaved scales → +13.5% speedup

### Files Modified This Session
| File | Change |
|------|--------|
| `marlin_moe/marlin_template.h` | Uncommented `dequant_fp8_scales` for FP4 |
| `compressed_tensors_w4a4_nvfp4.py` | Marlin FP4 dense GEMM + interleaved scales for SM121 |
| `compressed_tensors_w4a4_nvfp4_moe.py` | moe_sum_reduce fix + NVFP4 scale processing + interleaving |
| `marlin_utils.py` | `nvfp4_marlin_process_scales` + `nvfp4_marlin_process_global_scale` + `nvfp4_marlin_interleave_scales` |
| `nvfp4_post_quant.py` | Marlin FP4 post-quant path for SM121 (pure Python quantization + Marlin repack) |
| `qwen3_5.py` | Enable GDN post-quant on SM121 (removed skip guard) + FP8 post-quant for GDN layers |
| `compressed_tensors.py` | Route ParallelLMHead through Fp8LinearMethod for FP8 lm_head |
| `eagle_worker_v2.py` | Share FP8 weight_scale from target to draft lm_head |
| `eagle_worker.py` | Same weight_scale sharing for v1 eagle worker |
| `logits_processor.py` | Route FP8-quantized lm_head through quant_method.apply() |
| `fp8_post_quant.py` | CUTLASS FP8 post-quant for BF16 GDN layers |
| `sglang.sh` | SM121 auto-detection, flashinfer attention backend |

### Performance Results (Qwen3.5-35B-A3B-NVFP4 on GB10)
| Metric | BF16 Dequant (v1) | Marlin FP4 + Interleaved (v3) | + GDN Post-Quant (v4) |
|--------|-------------------|-------------------------------|----------------------|
| Decode tok/s (with MTP) | **~60** | **~67** (+12%) | **~76** (+27%) |
| E2E tok/s (with MTP) | ~58 | ~66 (+14%) | **~75** (+29%) |
| TTFT | ~123ms | ~118ms | **~107ms** |
| Model mem (main) | ~29 GB (est.) | ~22 GB (-25%) | **~22 GB** |
| Inference test | PASS | PASS | **PASS** |
| Tool call test | PASS | PASS | **PASS** |

### GDN Post-Quantization Accuracy (BF16 → NVFP4 round-trip, 60 layers)
| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Cosine Similarity | **0.9987** | 0.9956 | 1.0027 |
| SNR (dB) | **20.51** | 20.43 | 20.56 |
| Relative Error | 0.190 | 0.177 | 0.200 |

### Launch Configuration

#### Qwen3.5-35B-A3B-NVFP4

```bash
# Launch command (SM121 env vars auto-detected by sglang.sh)
source .sglang/bin/activate
SGLANG_MTP_FP8=1 CUDA_VISIBLE_DEVICES=0 bash sglang.sh Qwen3.5-35B-NVFP4
```

**Preset args** (`cmd_qwen35_35b_nvfp4` in `sglang.sh`):
| Argument | Value |
|----------|-------|
| `--model-path` | `Sehyo/Qwen3.5-35B-A3B-NVFP4` |
| `--quantization` | `compressed-tensors` |
| `--kv-cache-dtype` | `auto` (resolved to `fp8_e4m3` by server) |
| `--mem-fraction-static` | `0.75` |
| `--context-length` | `65536` |
| `--max-running-requests` | `3` |
| `--attention-backend` | `flashinfer` |
| `--linear-attn-prefill-backend` | `triton` |
| `--chunked-prefill-size` | `-1` (disabled) |
| `--disable-multimodal` | yes |
| `--served-model-name` | `qwen3-coder-next` |
| `--tool-call-parser` | `qwen3_coder` |
| `--reasoning-parser` | `qwen3` |
| `--trust-remote-code` | yes |

**MTP speculative decoding** (enabled by default):
| Argument | Value |
|----------|-------|
| `--speculative-algorithm` | `NEXTN` |
| `--speculative-num-steps` | `2` |
| `--speculative-eagle-topk` | `1` |
| `--speculative-num-draft-tokens` | `2` (auto-adjusted to `3`) |
| `--mamba-scheduler-strategy` | `extra_buffer` |
| `SGLANG_ENABLE_SPEC_V2` | `1` |

**Environment** (auto-detected by `setup_runtime_env()` for SM121):
| Variable | Value | Reason |
|----------|-------|--------|
| `KV_CACHE_DTYPE` | `auto` → `fp8_e4m3` | Server resolves auto to model's native dtype |
| `SGLANG_ENABLE_JIT_DEEPGEMM` | `0` | DeepGEMM JIT fails on SM121 |
| `SGLANG_MTP_FP8` | `1` | FP8 post-quant for MTP draft layers (enabled) |
| `SGLANG_QUANTIZE_LM_HEAD_FP8` | `1` | FP8 lm_head (enabled) |

**Auto-detected by `server_args.py`**:
| Setting | Value | Reason |
|---------|-------|--------|
| `--moe-runner-backend` | `marlin` | SM121: CUTLASS/TRT-LLM broken |
| `--speculative-moe-runner-backend` | `marlin` | Same |
| `--speculative-attention-mode` | `decode` | Auto for SM100+/SM120+ with FlashInfer |

#### Qwen3.5-122B-A10B-NVFP4

```bash
source .sglang/bin/activate
SGLANG_MTP_FP8=1 CUDA_VISIBLE_DEVICES=0 nohup bash sglang.sh Qwen3.5-NVFP4 \
> /tmp/sglang.log 2>&1 &
```

### Git
- **Branch**: `gb10-optimization`
- **Latest commit**: `b965fd461` — "Feat: CUTLASS FP8 post-quant for BF16 GDN layers + lm_head FP8 routing on SM121"
- **Pushed** to `fork/gb10-optimization`

## Key Architecture Notes

### Qwen3.5 NVFP4 Quantization Config
- **Quantized (NVFP4 W4A4)**: MoE expert weights, attention qkv_proj/o_proj, shared expert
- **Post-quantized (BF16→NVFP4)**: GDN projections (in_proj_qkv, in_proj_z) — 60 layers (35B), 72 layers (122B)
- **Unquantized (BF16)**: gates, lm_head (FP8 post-quant), in_proj_b/a, o_proj for GDN
- Layer pattern: `[linear_attention×3, full_attention]×10` = 40 layers total

### SM121 NVFP4 Execution Paths
| Layer Type | Backend | Status |
|------------|---------|--------|
| MoE experts (256 experts, topk=8) | Marlin FP4 MoE kernel | Working |
| Attention qkv_proj, o_proj (100 layers) | Marlin FP4 dense GEMM | Working |
| Shared expert gate_up/down | Marlin FP4 dense GEMM | Working |
| GDN projections (60 layers) | Marlin FP4 dense GEMM (post-quant) | Working |
| CUTLASS FP4 GEMM | N/A | **BROKEN on SM121** — not used |

### Performance Results (Qwen3.5-122B-A10B-NVFP4 on GB10)
| Metric | Baseline (NVFP4+GDN post-quant) | + FP8 GDN (broken lm_head) | + FP8 lm_head + scale sharing |
|--------|--------------------------------|----------------------------|-------------------------------|
| Decode tok/s | **~36.9** | **~37.9** (+3%) | **~44.3** (+20%) |
| E2E tok/s | **~36.3** | **~37.2** (+2%) | **~43.5** (+20%) |
| TTFT | ~250ms | **~245ms** | **~224ms** |
| MTP accept rate | ~0.97 | ~0.97 | **~0.88-0.97** |
| MTP accept len | ~2.9 | ~2.9 | **~2.4-2.9** |
| Inference test | PASS | **PASS** | **PASS** |
| GDN BF16→FP8 layers | 0 | **108** | **108** |
| lm_head FP8 | BF16 matmul | bypass (BF16 matmul*) | **CUTLASS FP8 GEMM** |
| CUDA graphs | bs=1,2,3 | bs=1,2,3 | bs=1,2,3 |

*lm_head FP8 routing was broken: `ParallelLMHead` extends `VocabParallelEmbedding` (not `LinearBase`),
so `compressed_tensors.get_quant_method()` never matched it for `Fp8LinearMethod`.

## Kernel Tuning Analysis

### Torch Profiler Results (122B, 20 decode steps)

**Trace files**: `/tmp/1773089728.4829538-TP-0-{DECODE,EXTEND}.trace.json.gz`

#### GPU Compute Time Breakdown (per decode step = main verify + 2 draft)
| Category | ms/step | % | Calls/step | Notes |
|----------|---------|---|------------|-------|
| **CUTLASS BF16 GEMM** | **26.2** | **33.0%** | 164 | BF16 layers: GDN in_proj_b, in_proj_a, o_proj (~54/fwd) |
| **Marlin MoE FP4** | **20.7** | **26.0%** | 96 | MoE expert GEMM (topk=8, 256 experts) |
| **Marlin Dense FP4** | **10.2** | **12.9%** | 192 | Attention qkv/o_proj + shared expert + GDN post-quant |
| **cuBLAS GEMV** | **9.6** | **12.1%** | 6 | lm_head (main + MTP draft) |
| GDN/Mamba kernels | 3.5 | 4.3% | 76 | delta_rule_update, causal_conv1d |
| Other GPU | 2.6 | 3.3% | 727 | Elementwise ops (sigmoid, multiply, etc.) |
| Triton fused_moe | 2.5 | 3.2% | 4 | MoE gating/routing |
| cuBLAS reduce ops | 1.9 | 2.3% | 148 | GEMV split-K reductions |
| FlashInfer | 0.9 | 1.1% | 267 | Attention + RMSNorm + SiLU |
| MoE routing | 0.6 | 0.8% | 246 | topkGating, align_block, count_sort |
| **TOTAL compute** | **79.5** | | | |

#### Host-Side Sync Overhead
| Source | ms/step | Calls/step | Pattern |
|--------|---------|------------|---------|
| `aten::item()` (speculative decode) | **63.4** | 2 | Alternating ~10ms + ~53ms |
| cudaMemcpyAsync | 4.1 | 53 | |
| cudaGraphLaunch | 1.2 | 2 | |
| **TOTAL sync** | **~68.7** | | |

The ~10ms item() is in `eagle_info_v2.py:140` (`batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()`)
with a `FIXME(lsyin): make this sync optional` comment. The ~53ms item() waits for the full
forward pass to complete before reading acceptance results.

#### Key Insights
1. **BF16 GEMM is the #1 compute cost (33%)** — GDN layers `in_proj_b`, `in_proj_a`, `o_proj`
   remain BF16. Post-quantizing to FP8 would halve bandwidth → ~13ms/step savings → **~16% speedup**
2. **Marlin MoE (26%)** — Kernel-level tuning (pipe_stages, block size) could help here
3. **cuBLAS GEMV (12%)** — lm_head already FP8, but GEMV is suboptimal for M=1; Marlin GEMM may be faster
4. **Host sync (68.7ms/step)** — Speculative decoding tree building forces GPU→CPU synchronization
   between forward passes. This is architectural, not easily fixable without upstream changes.

### Config Tuning Results — TESTED (no improvement for single-request)
| Config | Decode TPS | vs Baseline (36.9) | Notes |
|--------|-----------|---------------------|-------|
| `speculative_num_steps=3` + `continuous_decode=2` | **35.9** | -2.7% | Extra draft cost > benefit |
| `speculative_num_steps=2` + `continuous_decode=2` | **36.3** | -1.6% | Multi-request optimization |
| **Baseline** (`steps=2`, `continuous=1`) | **36.9** | — | Already optimal |

### FP8 Post-Quant for GDN BF16 Layers — Results

Applied FP8 post-quantization to 108 GDN BF16 layers (`in_proj_a`, `in_proj_b`, `out_proj`)
using `apply_fp8_post_quant_linear_base()` with `Fp8PostQuantLinearMethod` (torch._scaled_mm).

**Files modified**:
- `python/sglang/srt/layers/quantization/fp8_post_quant.py` — Added `Fp8PostQuantLinearMethod` class + `apply_fp8_post_quant_linear_base()` function
- `python/sglang/srt/models/qwen3_5.py` — Applied FP8 post-quant in both model classes

**Profile comparison (normalized to per-step-unit)**:

| Kernel Category | Baseline (ms) | FP8 (ms) | Change |
|---|---|---|---|
| Marlin MoE | 207.0 | 209.5 | same |
| **CUTLASS BF16 WMMA (128x1)** | **145.2** | **128.9** | **-11%** |
| **CUTLASS BF16 WMMA (128x2)** | **115.4** | **14.9** | **-87%** |
| Marlin FP4 dense | 102.5 | 100.2 | same |
| cuBLAS GEMV | 95.9 | 95.8 | same |
| **FP8 scaled_mm (NEW)** | 0 | **76.7** | **new** |
| **Total GPU** | **794.6** | **739.8** | **-7%** |

**Key finding**: FP8 IS working — `sm89_xmma_gemm_e4m3bf16` replaced most CUTLASS BF16 128x2
kernels. But the **net savings are only ~55ms (~7%)** because torch._scaled_mm overhead is
significant at M=1 decode. Throughput unchanged: **~36.75 TPS** (was 36.9).

### Critical Bug Found: lm_head FP8 Not Actually Used

**Root cause**: `LogitsProcessor._compute_lm_head()` checks `hasattr(lm_head, "weight")` first
(line 862), which is True for `ParallelLMHead`. This sends it to the `torch.matmul` path (line 881)
instead of `quant_method.apply()`, **completely bypassing FP8 GEMM**.

The remaining **CUTLASS BF16 128x1** kernel (128.9ms, 20 calls at 6.4ms each) IS the lm_head:
- lm_head weight = [151936, 5120] at BF16 = 1.49 GB
- At 232 GB/s bandwidth: 1.49 GB / 232 GB/s = 6.4ms ← **exact match**
- 20 calls = 2 per decode step (2 speculative draft steps)

**Fix applied**: Added FP8 weight dtype check in `logits_processor.py:_compute_lm_head()` to
route FP8-quantized lm_head through `quant_method.apply()` for proper CUTLASS FP8 GEMM.

### Critical Finding: torch._scaled_mm is SLOW on SM121, CUTLASS FP8 is FAST

**Microbenchmark results** (M=1 decode, SM121):

| Operation | BF16 matmul | torch._scaled_mm FP8 | CUTLASS fp8_scaled_mm | Speedup |
|-----------|------------|----------------------|----------------------|---------|
| out_proj [1,8192]×[8192,3072] | 0.223ms | ~0.15ms* | **0.110ms** | **2.04x** |
| lm_head [1,5120]×[5120,151936] | 9.628ms | ~6.5ms* | **3.504ms** | **2.75x** |

*estimated from profile (sm89_xmma kernel)

The initial FP8 attempt used `torch._scaled_mm` with per-tensor scales → dispatched to
`sm89_xmma_gemm_e4m3bf16` kernel, which is **not optimized for SM121**.

**Fix**: Switched to CUTLASS FP8 (`fp8_scaled_mm` from sgl-kernel) with per-channel weight
scales via `sglang_per_token_quant_fp8()`. This uses the native CUTLASS FP8 GEMM which is
2-3x faster than BF16 on SM121.

**Files modified**:
- `fp8_post_quant.py`: `Fp8PostQuantLinearMethod.apply()` now uses `fp8_scaled_mm` + `sglang_per_token_quant_fp8`
- `fp8_post_quant.py`: `apply_fp8_post_quant_linear_base()` now does per-channel FP8 quantization
- `logits_processor.py`: Routes FP8 lm_head through `quant_method.apply()` instead of `torch.matmul`

**Result**: 36.9 → **37.9 TPS** (+3% mean, best run 39.4 TPS)

### Fix: lm_head FP8 Routing for ParallelLMHead

**Problem**: `SGLANG_QUANTIZE_LM_HEAD_FP8=1` was completely broken. The `compressed_tensors.get_quant_method()` only checked `isinstance(layer, LinearBase)` before applying `Fp8LinearMethod`. But `ParallelLMHead` extends `VocabParallelEmbedding` (NOT `LinearBase`), so the FP8 config was never applied. The lm_head stayed BF16 and used `torch.matmul`.

**Fix** (`compressed_tensors.py`): Added a `ParallelLMHead` check before the `LinearBase` branch in `get_quant_method()`. Now when `lm_head_fp8_config` is set, `ParallelLMHead` gets `Fp8LinearMethod` which:
1. Creates BF16 weight buffer during `create_weights()`
2. Quantizes to FP8 per-channel during `process_weights_after_loading()` (stores as [K,N] transposed)
3. Uses CUTLASS `fp8_scaled_mm` during `apply()` (2.75x faster than BF16 for lm_head)

### Fix: MTP Draft Model weight_scale Sharing

**Problem**: After the lm_head FP8 fix, MTP acceptance rate dropped from ~97% to 33% (zero draft tokens accepted). Root cause: the MTP draft model's lm_head was also quantized to FP8 by `Fp8LinearMethod`, but it had NO checkpoint weights (`mtp.lm_head.weight` absent from checkpoint). So `process_weights_after_loading()` quantized random/uninitialized data → meaningless `weight_scale`. When `set_embed_and_head()` shared the main model's correct FP8 weight, the draft model used the WRONG scale → garbage logits.

**Fix** (`eagle_worker_v2.py`, `eagle_worker.py`): After `set_embed_and_head()` shares the lm_head weight, also share `weight_scale` and `input_scale` from target to draft model. This ensures the draft model uses the correct FP8 scales matching the shared weight.

**Result**: 37.9 → **44.3 TPS** (+17% from lm_head FP8, +20% total over baseline)

### Remaining Tuning Opportunities (ordered by expected impact)

#### 1. Marlin Kernel Tuning (MEDIUM — requires sgl-kernel recompile)
| Parameter | Current | File | Notes |
|-----------|---------|------|-------|
| `pipe_stages` | 4 | `marlin.cuh:15` | SM121 has 228KB shared mem — try 5-6 stages |
| Thread block configs | Auto | `gptq_marlin.cuh:115-129` | Profile specific configs for model's GEMM sizes |

#### 2. MoE Block Size (MEDIUM)
- Auto from `[8, 16, 32, 48, 64]` in `fused_marlin_moe.py`
- For 122B (256 experts, topk=8), may not be optimal

#### 3. Host sync overhead (MEDIUM — architectural)
- `aten::item()` in `eagle_info_v2.py:140` forces GPU→CPU sync between draft steps
- ~63ms/step overhead from speculative decode tree building
- Has upstream `FIXME(lsyin)` comment — not easily fixable locally

## Pending
- [x] ~~Verify server reaches "The server is ready"~~ ✓
- [x] ~~Run speed test (35B)~~ ✓ 76 tok/s with MTP + GDN post-quant
- [x] ~~Test inference and tool calls~~ ✓ Both PASS
- [x] ~~Test 122B model~~ ✓ Loads and generates coherent text
- [x] ~~Run speed test on 122B~~ ✓ ~27 tok/s with MTP
- [x] ~~GDN NVFP4 post-quant via Marlin FP4~~ ✓ +13.5% speedup (35B), cos_sim 0.9987
- [x] ~~Re-test 122B with GDN post-quant + MTP FP8~~ ✓ **~37 tok/s** (+36%)
- [x] ~~Kernel tuning: quick config wins~~ ✓ No improvement — system is bandwidth-bound
- [x] ~~Kernel tuning: profile hotspots with torch profiler~~ ✓ BF16 GEMM=33%, Marlin MoE=26%, Marlin dense=13%, GEMV=12%
- [x] ~~FP8 post-quant for GDN in_proj_b/in_proj_a/o_proj~~ ✓ 108 layers, CUTLASS FP8 (2x faster than BF16)
- [x] ~~Diagnose lm_head FP8 bypass~~ ✓ LogitsProcessor bypasses quant_method.apply()
- [x] ~~Fix FP8 to use CUTLASS~~ ✓ torch._scaled_mm → fp8_scaled_mm, per-channel scales
- [x] ~~Test CUTLASS FP8~~ ✓ **37.9 TPS** (+3% over baseline, best 39.4)
- [x] ~~Fix lm_head FP8 routing~~ ✓ ParallelLMHead now matched by get_quant_method()
- [x] ~~Fix MTP weight_scale sharing~~ ✓ Draft model uses target's FP8 scales
- [x] ~~Test FP8 lm_head + scale sharing~~ ✓ **44.3 TPS** (+20% over baseline)
- [ ] Kernel tuning: Marlin pipe_stages (requires recompile)
- [ ] Run accuracy benchmarks (lm-evaluation-harness)
