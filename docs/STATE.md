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
| `qwen3_5.py` | Enable GDN post-quant on SM121 (removed skip guard) |
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
- **Latest commit**: `07ce8c871` — "Feat: GDN NVFP4 post-quantization via Marlin FP4 on SM121 (GB10)"
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
| Metric | Previous (no GDN post-quant, MTP_FP8=0) | Current (GDN post-quant + MTP_FP8=1) |
|--------|----------------------------------------|---------------------------------------|
| Decode tok/s | ~27.2 | **~36.9** (+36%) |
| E2E tok/s | ~26.7 | **~36.3** (+36%) |
| TTFT | ~302ms | **~250ms** (-17%) |
| Inference test | PASS | **PASS** |
| Tool call test | PASS | **PASS** |
| GDN layers post-quantized | 0 | **72** |
| Available KV cache | ~14 GB | TBD |
| CUDA graphs | bs=1,2,3 only | TBD |

## Kernel Tuning Analysis

### Bottleneck Analysis
Decode is **memory-bandwidth bound** — each token loads 8/256 expert weight matrices per
MoE layer + attention/shared-expert/GDN weights through Marlin FP4 GEMM. MTP accept rate
~0.90-0.95 with accept length ~2.7 tokens.

### Tuning Opportunities (ordered by expected impact)

#### 1. Quick Config Wins (no code changes, server restart only)
| Parameter | Current | Proposed | Rationale |
|-----------|---------|----------|-----------|
| `num_continuous_decode_steps` | 1 | 2-4 | Batch multiple decode steps, reduce scheduling overhead |
| `speculative_num_steps` | 2 | 3 | More tokens/forward pass (~2.7→~3.5 accept len if draft accurate enough) |
| `stream_interval` | 1 | 2 | Less HTTP streaming overhead per token |

#### 2. Marlin Kernel Tuning (requires sgl-kernel recompile)
| Parameter | Current | File | Notes |
|-----------|---------|------|-------|
| `pipe_stages` | 4 | `marlin.cuh:15` | SM121 has 228KB shared mem — could fit 5-6 stages for better latency hiding |
| Thread block configs | Auto-select | `gptq_marlin.cuh:115-129` | Priority-ordered list; could profile specific configs for this model's GEMM sizes |
| `max_thread_m_blocks` | 4 | `gptq_marlin.cuh:614` | Controls M-dimension tiling; decode uses m=1 so less impact |

#### 3. MoE Block Size
- Auto-selected from `[8, 16, 32, 48, 64]` in `fused_marlin_moe.py`
- Heuristic: `if M * topk / E / block_size_m < 0.9: break`
- For 122B (256 experts, topk=8), optimal block size may differ from default

#### 4. FlashInfer Attention
| Parameter | Current | Env Var |
|-----------|---------|---------|
| Decode split tile size | 2048 | `SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE` |
| Prefill split tile size | 4096 | `SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE` |
| Workspace size | 512 MB (Qwen) | `SGLANG_FLASHINFER_WORKSPACE_SIZE` |

#### 5. Deeper Kernel Work (high effort)
- Marlin shared memory bank conflict avoidance (padding)
- Custom fused kernels (RMSNorm + GEMM, etc.)
- Weight layout optimization for SM121 cache hierarchy

## Pending
- [x] ~~Verify server reaches "The server is ready"~~ ✓
- [x] ~~Run speed test (35B)~~ ✓ 76 tok/s with MTP + GDN post-quant
- [x] ~~Test inference and tool calls~~ ✓ Both PASS
- [x] ~~Test 122B model~~ ✓ Loads and generates coherent text
- [x] ~~Run speed test on 122B~~ ✓ ~27 tok/s with MTP
- [x] ~~GDN NVFP4 post-quant via Marlin FP4~~ ✓ +13.5% speedup (35B), cos_sim 0.9987
- [x] ~~Re-test 122B with GDN post-quant + MTP FP8~~ ✓ **~37 tok/s** (+36%)
- [ ] **Kernel tuning: quick config wins** (num_continuous_decode_steps, speculative_num_steps)
- [ ] Kernel tuning: Marlin pipe_stages (requires recompile)
- [ ] Kernel tuning: profile hotspots with torch profiler
- [ ] Run accuracy benchmarks (lm-evaluation-harness)
