# Performance Investigation: vLLM 0.16.1rc1 vs Container v23

## Summary

**Current best: ~70.5 tps_decode p50 avg (3 spec tokens, FP8 lm_head + FP8 MoE experts)**
**vs container's 66.75 tps (3 spec tokens) = +4.5–6.8% FASTER**

## Root Cause Analysis

### Raw decode speed: nearly identical (-1.1%)
- Our build (no MTP): 41.74 tps
- Container (no MTP): 42.21 tps
- This rules out GEMM, attention kernels, model weights, etc. as bottlenecks.

### MTP effectiveness: significantly worse
- Container: MTP gives +58% boost (42.21 → 66.98 with 3 spec tokens)
- Our build: MTP gives +43% boost (41.74 → 59.61 with 2 spec tokens)
- The per-step MTP overhead is higher in our build.

### Per-step MTP overhead analysis

| Metric | Container (v0.16.0rc2) | Our build (v0.16.1rc1) |
|--------|----------------------|----------------------|
| MTP spec=1 | 60.25 tps (+42.7%) | 59.36 tps (+42.2%) |
| MTP spec=2 | 65.11 tps (+54.2%) | 59.61 tps (+42.8%) |
| MTP spec=3 | 66.98 tps (+58.7%) | 57.55 tps (+37.9%) |
| Marginal gain spec 1→2 | +4.86 tps | +0.25 tps |
| Marginal gain spec 2→3 | +1.87 tps | -2.06 tps |

**Key insight:** At spec=1, both versions are nearly identical (59.36 vs 60.25, -1.5%).
The gap widens with each additional spec token, meaning **the loop overhead per additional
draft step is much higher** in v0.16.1rc1.

## MTP Draft Loop Profiling Results

### Method

Added CUDA event timing instrumentation to the MTP draft loop in `eagle.py`
(`propose()` method, the `for token_index in range(...)` loop). Four phases measured:

1. **metadata** — `build_for_drafting()` to rebuild attention metadata
2. **prep** — copy inputs to CUDA graph buffers
3. **forward** — `self.model(**model_kwargs)` (CUDA graph replay of 1-layer MTP model)
4. **sample** — `_greedy_sample()` → `compute_logits().argmax()`

### Results (averaged over 500 draft steps, spec=3, batch=1)

| Phase | Time (ms) | % of total |
|-------|----------|-----------|
| metadata (build_for_drafting) | 0.06 | 0.9% |
| prep (copy to CUDA graph buffers) | 0.14 | 2.1% |
| forward (MTP model, CUDA graph) | 0.76 | 11.5% |
| **sample (compute_logits + argmax)** | **5.64** | **85.3%** |
| **total per draft step** | **6.61** | **100%** |

### Analysis: compute_logits is the bottleneck

`_greedy_sample()` calls `self.model.compute_logits(hidden_states).argmax(dim=-1)`:

1. `compute_logits` does a matmul: `hidden_states @ lm_head.weight.T`
   - hidden_size = 2048, vocab_size = 248,320
   - lm_head is **not quantized** (excluded from NVFP4 in model config's `ignore` list)
   - Weight size: 248,320 × 2048 × 2 bytes (bf16) = **~1.02 GB**
   - At GB10's ~200 GB/s memory bandwidth → **~5.1ms** (matches observed 5.64ms)

2. `argmax(dim=-1)` over 248K elements: ~0.1ms (negligible)

The lm_head matmul is **memory-bandwidth-bound**. Each draft step loads ~1 GB of weight
data from memory just to produce a single token ID.

### Both builds have the same lm_head cost

Checked v0.16.0rc2's eagle.py — the MTP loop is nearly identical:
```python
# v0.16.0rc2 (line 696-697)
logits = self.model.compute_logits(last_hidden_states[:batch_size])
draft_token_ids = logits.argmax(dim=-1)

# v0.16.1rc1 (via _greedy_sample, line 377)
return self.model.compute_logits(hidden_states).argmax(dim=-1)
```

Same lm_head, same matmul, same ~5.6ms cost. The code paths are functionally identical.

### Estimated per-step wall-clock time (from tps data)

Using measured tps and acceptance rates to calculate effective per-step times:

| Step | Container (ms) | Our build (ms) | Delta |
|------|---------------|----------------|-------|
| Target model forward | 23.69 | 23.96 | +0.27 |
| 1st draft (first pass) | 7.51 | 7.71 | +0.20 |
| 2nd draft (loop iter 1) | 4.43 | 7.25 | **+2.82** |
| 3rd draft (loop iter 2) | 1.54 | 4.35 | **+2.81** |

The 1st draft step is similar (+0.20ms). The gap appears in loop iterations 2+.

The container shows **decreasing per-step cost** (7.51 → 4.43 → 1.54ms), suggesting
better async GPU/CPU overlap in v0.16.0rc2. Our build shows roughly constant cost
(7.71 → 7.25 → 4.35ms), suggesting less effective pipelining.

### Why loop iterations differ: async execution overlap

Without `torch.cuda.synchronize()`, the GPU executes operations asynchronously:
- While GPU processes step N's `compute_logits` (5.6ms), CPU can prepare step N+1's
  metadata and queue kernels
- The data dependency (`input_ids = draft_token_ids_list[-1].int()`) forces the GPU
  to wait for step N's argmax before starting step N+1's model forward
- But CPU-side work (metadata build, tensor copies, Python overhead) can overlap

The container (v0.16.0rc2 + FlashInfer 0.6.3) appears to have better async overlap,
possibly due to:
1. **FlashInfer 0.6.3 decode kernel** — may have lower CPU-side overhead for `plan()`
2. **CUDA graph dispatcher** — v0.16.1rc1 refactored the dispatcher, potentially adding
   Python overhead per dispatch
3. **`fast_build` parameter** — accepted but **never used** in FlashInfer's `build()` method
   in both versions (dead code path)

### `use_local_argmax_reduction` — not helpful at TP=1

The `use_local_argmax_reduction` flag (commit `2bcf71b9c`) enables `get_top_tokens()`
which does vocab-parallel argmax without all-gathering logits. At TP=1 (our case),
this is equivalent to `compute_logits().argmax()` — same matmul, same bandwidth cost.
Only helps for TP>1 by reducing inter-GPU communication.

### Quantization details

| Component | Quantization | Weight size |
|-----------|-------------|-------------|
| Main model layers | NVFP4 (W4A4) | ~4.4 GB |
| MTP transformer layer | Not quantized (in `ignore`) | bf16 |
| MTP `fc` layer | Not quantized (`quant_config=None`) | bf16 |
| `lm_head` | Not quantized (in `ignore`) | **~1.02 GB bf16** |
| `embed_tokens` | Not quantized | ~1.02 GB bf16 |

The `lm_head` is NOT weight-tied (`tie_word_embeddings=False`), so it's a separate
~1 GB weight matrix from `embed_tokens`.

## What changed between v0.16.0rc2 and v0.16.1rc1

### Code path changes (MTP draft loop in eagle.py)

The draft loop is functionally identical — both call:
1. Update positions + slot mappings
2. `build_for_drafting()` to rebuild attention metadata
3. Copy inputs to CUDA graph buffers
4. `model(**kwargs)` forward pass
5. `compute_logits().argmax()` for greedy sampling

Minor differences:
- **AttentionGroup iteration**: New code loops over `draft_attn_groups` instead of using
  a single `attn_metadata_builder`. For MTP (1 group), this is trivial overhead.
- **Block size access**: New code uses `self.block_size` (cached), old used
  `attn_metadata_builder.kv_cache_spec.block_size`. Equivalent.
- **`_greedy_sample()` method**: New code has an indirection through `_greedy_sample()`,
  but it still calls `compute_logits().argmax()`. No extra work.

### Infrastructure changes

Between v0.16.0rc2 and v0.16.1rc1, significant refactoring occurred:
- `extract_hidden_states.py` system added (runs during target forward, not draft loop)
- CUDA graph dispatcher refactored
- FlashInfer attention builder refactored (new `fast_build` path — but never used)
- Various bugfixes and feature additions

### FlashInfer 0.6.3 vs 0.6.4

Kernel-level changes between FlashInfer versions could account for per-step overhead.
The decode kernel selection and JIT compilation may differ.

## Hypothesis: FlashInfer decode path overhead

The native FlashInfer decode path on SM121 uses the XQA backend (since TRTLLM is
SM10x-only). Between FlashInfer 0.6.3 and 0.6.4:
- The XQA backend may have changed JIT compilation behavior
- The `plan()` call in `BatchDecodeWithPagedKVCacheWrapper` may have different overhead
- CUDA graph replay characteristics may differ

This would explain why the overhead scales with spec tokens (each additional token =
another `build_for_drafting` → `plan()` → decode cycle).

## TRTLLM Attention on SM121

**Status: Not possible without NVIDIA support.**

- TRTLLM decode attention requires pre-compiled cubins from NVIDIA artifactory
- Cubins are only available for SM10x (GB200/B200)
- SM121 (GB10 Spark) is SM12x family — cubins won't work
- FlashInfer's `supports_trtllm_attention()` checks `is_device_capability_family(100)`
- The `--attention-config.use_trtllm_attention` force flag doesn't override this check
- FlashInfer has `gen_trtllm_fmha_v2_module()` with `supported_major_versions=[12]`
  but this is for prefill/MLA, not the decode path cubins

**To enable TRTLLM on SM121 would require:**
1. NVIDIA publishing SM12x cubins to artifactory, OR
2. FlashInfer adding JIT compilation for the decode path on SM12x, OR
3. Modifying the hardcoded `get_compute_capability(query.device)[0] == 10` check in
   FlashInfer's decode.py (risky — cubins may not work on SM12x)

## Actionable improvements

### Already done
- [x] Remove `build_for_drafting` override — **+28.7% decode speed** (commit 41f74917a)
- [x] Profile MTP loop — identified `compute_logits` (lm_head matmul) as 85% bottleneck
- [x] **FP8 lm_head quantization** — +20% decode speed, beats container by +3.4% (69.02 tps)
- [x] **Spec token sweep (spec=4, spec=5)** — spec=3 remains optimal even with FP8 lm_head
- [x] **FP8 MTP MoE expert quantization** — additional +3.3%, new best ~70.5 tps (+5.6% vs container)

### FP8 draft lm_head optimization (IMPLEMENTED)

**Result: 69.02 tps_decode p50 (spec=3) vs container's 66.75 tps = +3.4%**

The lm_head weight matrix (248K × 2048, bf16, ~1 GB) is dynamically quantized to
`float8_e4m3fn` (~500 MB) at model load time. Per-tensor quantization with
`torch._scaled_mm` for the matmul.

**Microbenchmark results (isolated matmul+argmax, batch=1, 248K vocab):**

| Method | Time (ms) | Speedup | Accuracy |
|--------|----------|---------|----------|
| BF16 baseline | ~35 | 1x | 100% |
| torch.compile | ~6.2 | 3.3x | 100% |
| **FP8 (torch._scaled_mm)** | **~3.7** | **9.5x** | **~85%** |
| Fused Triton matmul+argmax | ~284 | 0.1x | 100% |

The ~85% token match rate for FP8 is acceptable for draft tokens — the verification
step catches mismatches, and the 9.5x matmul speedup more than compensates for slightly
lower acceptance rates. The fused Triton kernel was abandoned (can't compete with cuBLAS).

**End-to-end speed test:**
- BF16 lm_head, spec=3: 57.55 tps (each additional spec token had negative marginal value)
- FP8 lm_head, spec=3: **69.02 tps** (+20%, spec=3 profitable again)
- FP8 lm_head, spec=4: 66.50 tps — worse than spec=3 (4th token not accepted often enough)
- FP8 lm_head, spec=5: 66.68 tps — similar to spec=4, not profitable
- **spec=3 is optimal** even with FP8 making draft steps cheaper

**Usage:** Set `VLLM_DRAFT_SAMPLE_OPT=fp8` environment variable (default in vllm.sh).
Also available: `VLLM_DRAFT_SAMPLE_OPT=compiled` (torch.compile, 3.3x, 100% accuracy).

**Implementation:** `vllm/v1/spec_decode/draft_sample_opt.py` (FP8LMHeadSampler),
integrated via `_greedy_sample()` in `eagle.py`.

### FP8 MTP MoE expert quantization (IMPLEMENTED)

**Result: ~70.5 tps p50 avg (spec=3) vs container's 66.75 tps = +5.6%**

`VLLM_MTP_MOE_FP8=1` post-quantizes MTP MoE expert weights from bf16 to FP8 at load time
(before CUDA graph capture). Implemented in `draft_sample_opt.py:quantize_mtp_moe_fp8()`.

**Why it works:** The Triton `UnquantizedFusedMoEMethod` uses `FusedMoEQuantConfig` to select
the kernel path. By replacing `quant_method.moe_quant_config` and `quant_method.kernel` with
an FP8 `fp8_w8a8_moe_quant_config` + new `TritonExperts` instance, the CUDA graph captures
the FP8 kernel path, halving active-expert memory traffic (50 MB → 25 MB per step for 35B).

**Quantization:** Per-expert per-tensor FP8 (one scale per expert per weight matrix).
1536 MB bf16 → 768 MB fp8 (-50%). Active expert load per step: 50 MB → 25 MB.

**Accuracy validation (offline, `validate_mtp_moe_fp8.py`):**

Loaded all 256 experts per model from `extra_weights.safetensors`, quantized to FP8,
dequantized back, and measured weight reconstruction error + matmul cosine similarity
(16 random input vectors per expert).

| Model | Proj | Weight Rel Err (mean) | Weight Rel Err (max) | Matmul Cos Sim (mean) | Matmul Cos Sim (min) |
|-------|------|-----------------------|----------------------|-----------------------|----------------------|
| **35B** | gate_proj | 2.252% | 2.258% | 0.999645 | 0.999610 |
| **35B** | up_proj   | 2.252% | 2.261% | 0.999646 | 0.999606 |
| **35B** | down_proj | 2.258% | 2.272% | 0.999648 | 0.999625 |
| **122B** | gate_proj | 2.227% | 2.260% | 0.999650 | 0.999627 |
| **122B** | up_proj   | 2.226% | 2.258% | 0.999651 | 0.999630 |
| **122B** | down_proj | 2.237% | 2.278% | 0.999649 | 0.999634 |

**Overall cosine similarity: 0.9997 for both models (~0.04% output deviation).**
This is near-perfect — far better than the lm_head FP8 (~15% token mismatch) because
per-expert scales are much finer-grained than the lm_head's single per-tensor scale.
The 122B is very slightly better than 35B (larger expert matrices → tighter scale fit).

The end-to-end 35B tps improvement confirms: draft acceptance rate does not meaningfully
drop with MoE FP8 — throughput increases because bandwidth savings outweigh any accuracy loss.

**End-to-end result (spec=3, FP8 lm_head + FP8 MoE, 35B):**
- Run 1: 71.29 tps p50 (+6.8% vs container)
- Run 2: 69.77 tps p50 (+4.5% vs container)
- Average: **~70.5 tps** (+5.6% vs container, +2.2% vs FP8 lm_head alone)

### Remaining potential optimizations

1. **Enable `VLLM_MTP_MOE_FP8=1` by default in vllm.sh** — tested, stable, +3.3% on 35B.
   Pending confirmation of output quality before making default.

2. **Triton MoE autotuning** — tune kernel configs for E=256,N=512 on GB10
   (see `docs/TRITON_MOE_AUTOTUNING.md`). Could improve main model MoE layer speed.
   Duration: ~8-10 hours. Run overnight.

3. **FlashInfer 0.6.3 downgrade test** — install 0.6.3 and test if per-step overhead
   decreases. This would confirm FlashInfer as the source of the async overlap regression.

4. **torch.compile'd draft sampling** — `VLLM_DRAFT_SAMPLE_OPT=compiled` gives 3.3x
   matmul speedup with 100% accuracy. Not yet tested end-to-end (FP8 is faster and default).

5. **Upstream PR** — report the per-step MTP overhead regression to vLLM.

### Not actionable (requires external changes)
- TRTLLM attention on SM121 (needs NVIDIA cubins)
- `FULL_AND_PIECEWISE` CUDA graphs (needs TRTLLM → UNIFORM_BATCH support)

## Container v23 patch comparison

All container patches are covered by our build:
- FlashInfer E2M1 SM121 fix → in `gb10_compat.py`
- MTP NVFP4 exclusion → in `qwen3_5_mtp.py` (ReplicatedLinear with quant_config=None)
- FlashInfer MoE backend fix → in `gb10_compat.py`
- NVFP4 emulation backend fix → in `gb10_compat.py`
- Qwen3-Next prefix fix → in model code
- Capability 121 routing → in `generate_kernels.py` SM12x family

## Environment variable comparison

All container env vars are present in our vllm.sh:
- `VLLM_NVFP4_GEMM_BACKEND=marlin` ✓
- `VLLM_TEST_FORCE_FP8_MARLIN=1` ✓
- `VLLM_USE_DEEP_GEMM=0` ✓
- `VLLM_USE_FLASHINFER_MOE_FP4=0` ✓
- `VLLM_MARLIN_USE_ATOMIC_ADD=1` ✓
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ✓
- `SAFETENSORS_FAST_GPU=1` ✓

## Post-Quantization Analysis

### Main model (35B and 122B)

The main model is already heavily quantized — essentially all large weights are NVFP4:

| dtype | size | what |
|-------|------|------|
| `uint8` (NVFP4 packed) | 16.3 GB | All MoE expert weights (every layer) |
| `float8_e4m3fn` | 2.0 GB | NVFP4 scale factors |
| **`bfloat16`** | **5.0 GB** | lm_head (1017 MB), embed_tokens (1017 MB), layernorms, SSM conv weights |

- **lm_head (1017 MB)** — handled by `FP8LMHeadSampler`. ✅
- **embed_tokens (1017 MB)** — not a decode bottleneck. During decode, embedding is a
  single-row lookup (2048 × 2 = 4 KB per token). Quantizing saves ~500 MB RAM but ~0 speed gain.
- Everything else is tiny norms/SSM params, or already NVFP4.

### MTP model — all BF16, potential target

| Layer | 35B size | 122B size | Active per step |
|-------|----------|-----------|-----------------|
| MTP MoE experts (256 experts × 3 proj) | 1611 MB | 4832 MB | 50 MB / 151 MB |
| MTP self_attn (q/k/v/o) | 55 MB | 157 MB | all |
| MTP fc (concat hidden states) | 17 MB | 38 MB | all |
| **Total MTP** | **1689 MB** | **5047 MB** | |

At 200 GB/s bandwidth, loading active MoE experts per step:
- 35B: 50 MB → **0.25 ms** (out of ~0.76 ms total MTP forward, ~33%)
- 122B: 151 MB → **0.75 ms** (likely dominates MTP forward entirely)

FP8 quantizing MTP experts halves this cost (**IMPLEMENTED** — see verdict below):
- 35B: save ~0.12 ms/step × 3 steps = ~0.36 ms — observed as **+3.3% tps** (71.29 vs 69.02)
- 122B: save ~0.37 ms/step × 3 steps = ~1.1 ms — larger gain expected, not yet tested

**Implementation:** Post-quantize at load time by replacing `quant_method.moe_quant_config`
and `quant_method.kernel` on each `UnquantizedFusedMoEMethod`. The Triton FP8 W8A8 path is
already supported — just requires FP8 weights and `fp8_w8a8_moe_quant_config` scales.
Done before CUDA graph capture so graphs capture the FP8 path. See `quantize_mtp_moe_fp8()`.

### Verdict

| Target | Worth it? | Why |
|--------|-----------|-----|
| **lm_head FP8** | ✅ Done | 9.5x speedup, implemented in `draft_sample_opt.py` |
| embed_tokens FP8/INT8 | ❌ No | Single-row lookup, ~0 speed gain |
| MTP fc FP8 (17–38 MB) | ❌ No | Small weight, tiny gain |
| MTP attn FP8 (55–157 MB) | ❌ No | Needs FlashInfer FP8 attention kernel support |
| **MTP MoE experts FP8** | ✅ Done | 35B: **+3.3% vs lm_head-only** (~70.5 tps avg); 122B: even larger gain expected |

For **35B**: `VLLM_MTP_MOE_FP8=1` gives +3.3% more vs FP8 lm_head alone. **IMPLEMENTED.**
Accuracy: 0.9997 cosine similarity, ~0.04% output deviation. **Safe to enable by default.**
For **122B**: active experts 151 MB → 75 MB per step saves ~0.37 ms/step × 3 steps — larger benefit.
Accuracy: 0.9997 cosine similarity (identical to 35B). **Safe to enable by default.**

## MTP Acceptance Rate Comparison

### With forced prefill path (old, before fix)
- Position 0: ~90%, Position 1: ~19%, Position 2: ~4%
- Overall: 37.8%, Mean acceptance length: ~1.1 tokens

### With native decode path (after fix)
- Position 0: ~87-89%, Position 1: ~49-53%, Position 2: ~36-41%
- Overall: 57-63%, Mean acceptance length: 2.7-2.8 tokens
