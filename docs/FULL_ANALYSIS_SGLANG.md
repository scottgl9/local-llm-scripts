# SGLang NVFP4 Optimization Analysis — Qwen3.5-122B-A10B on GB10

**Hardware:** NVIDIA GB10, SM12.1 (Blackwell), 128 GiB unified LPDDR5X memory
**Model:** `Sehyo/Qwen3.5-122B-A10B-NVFP4` (compressed-tensors NVFP4)
**Baseline:** ~33 tok/sec decode

---

## 1. Hardware Baseline — Why 33 tok/sec

| Property | Value |
|---|---|
| Memory bandwidth | ~221 GB/s (LPDDR5X unified) |
| Shared memory per SM | 99 KB (vs 228 KB on Hopper/SM100) |
| Tensor cores | FP4, FP8, BF16 all supported |
| Total memory | 128 GiB unified (CPU+GPU) |
| Architecture | SM12.1 (Blackwell, different from GB200 SM90) |

**Decode bottleneck math:**

Decode is memory-bandwidth-bound: every token requires reading all relevant weight tiles from DRAM.
At 221 GB/s, with ~6.6 GB of BF16/FP8/FP4 data accessed per step:

```
6.6 GB / 221 GB/s ≈ 30 ms/token → ~33 tok/sec
```

This is a hard bandwidth ceiling. Closing the gap means reducing total bytes read per token — either by:
1. Quantizing more layers (especially `lm_head`)
2. Making kernels more bandwidth-efficient (better pipeline utilization)
3. Choosing kernel backends that have native NVFP4 support

---

## 2. lm_head FP8 Quantization — HIGH IMPACT (answer to your question)

**Yes, you can quantize `lm_head` to FP8. Here's the full picture:**

### Current state

`lm_head` appears in the `ignore` list of the model's `quantization_config`. In SGLang,
`should_ignore_layer()` is called at `compressed_tensors.py:808`:

```python
# compressed_tensors.py line 808
if should_ignore_layer(
    layer_name, ignore=self.ignore, fused_mapping=self.packed_modules_mapping
):
    return None  # → falls through to UnquantizedLinearMethod() → BF16
```

`UnquantizedLinearMethod` is imported at `compressed_tensors.py:66` and used at line 173.
The `ignore` list is loaded from the HuggingFace config at line 217:

```python
ignore: List[str] = cast(List[str], config.get("ignore", []))
```

### Size impact

For Qwen3.5-122B-A10B: `vocab_size × hidden_dim` parameters.
- **BF16 lm_head**: full 2-byte-per-param read every decode step (every predicted token)
- **FP8 lm_head**: 1 byte per param → **50% reduction** in lm_head bandwidth cost

### Infrastructure exists in SGLang

- `Fp8LinearMethod` — `python/sglang/srt/layers/quantization/fp8.py`
- `ParallelLMHead` in `python/sglang/srt/layers/vocab_parallel_embedding.py` already supports quantized linear methods
- FP8 dynamic activation quantization is well-tested across Hopper and Blackwell

### How to enable it

**Option A (patch model config before loading):**
Remove `"lm_head"` from the `ignore` list in the NVFP4 model's `config.json` under
`quantization_config.ignore` and add an appropriate `linear_fp8_config` with
`activation_scheme: dynamic`. This tells the compressed-tensors parser to apply
`CompressedTensorsW8A8Fp8` to `lm_head`.

**Option B (SGLang server-side override):**
In `compressed_tensors.py`, after parsing the ignore list (line 217), add a hook:

```python
# After line 217: ignore = cast(List[str], config.get("ignore", []))
if server_args.quantize_lm_head == "fp8":
    ignore = [layer for layer in ignore if "lm_head" not in layer]
```

Then wire `--quantize-lm-head fp8` through `server_args.py`. This is a ~10-line change.

### Files to modify

| File | Change |
|---|---|
| `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py` | Post-parse override to remove `lm_head` from `ignore` list (line ~217) |
| `python/sglang/srt/server_args.py` | Add `--quantize-lm-head` server arg |

---

## 3. Triton num_stages=1 for SM12.x — HIGH IMPACT (~20–30%)

**File:** `python/sglang/srt/layers/quantization/fp8_kernel.py:1300`

### Current code (confirmed)

```python
# fp8_kernel.py line 1299-1300
if num_stages is None:
    num_stages = 1 if is_sm120_supported() else (4 if is_sm100_supported() else 1)
```

The docstring at line 1296-1297 reads:
```
num_stages: Number of pipeline stages. If None, auto-selects based on GPU:
    SM120: 1, SM100: 4.
```

### Why this matters

`num_stages` controls Triton's software pipeline depth — how many iterations ahead the GPU
prefetches DRAM tiles before the compute unit needs them. With `num_stages=1`:
- No pipelining: tensor cores stall waiting for each DRAM load to complete
- Memory latency is exposed rather than hidden
- GPU utilization is low during memory-bound GEMMs

SM100 (Hopper) gets 4 stages and uses its 228 KB shared memory to buffer 4× the tile data.
SM12.x (Blackwell GB10) has 99 KB shared memory — less headroom, but still enough for 2–4 stages
at the tile sizes used for MoE (1024 intermediate dim, group_size=16).

### Fix

Start with `num_stages=2` and benchmark:

```python
# fp8_kernel.py line 1300
num_stages = 2 if is_sm120_supported() else (4 if is_sm100_supported() else 1)
```

Then sweep `{2, 3, 4}` — the optimal value depends on tile size vs. available SMEM.

**Estimated impact:** 20–30% throughput improvement on all quantized linear and MoE GEMM kernels
on SM12.x, by overlapping DRAM loads with tensor core compute.

---

## 4. MoE Runner Backend — Missing Optimal Kernel Path

### What auto selects on SM120 for Qwen3.5 + compressed-tensors

The server_args logic at lines 1757–1810 handles Qwen3.5 model architectures. For SM100 it
sets `flashinfer_trtllm` — but **only when** `quantization in ("fp8", "modelopt_fp4", None)`.

Since Qwen3.5-NVFP4 uses `compressed-tensors` quantization (not `"fp8"` or `"modelopt_fp4"`),
the SM100 fast path is skipped. For SM120 there is no Qwen3.5-specific override at all,
so `moe_runner_backend` stays at `"auto"`.

The GptOss-specific SM120+MXFP4 path (lines 1568–1572) does set `triton_kernel`, but that
only triggers when `quant_method == "mxfp4"` in a GptOss model — not Qwen3.5 compressed-tensors.

### Available backends

`MoeRunnerBackend` enum in `python/sglang/srt/layers/moe/utils.py:65`:

```python
FLASHINFER_MXFP4 = "flashinfer_mxfp4"
```

The kernel itself exists: `sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu`

The backend is used in `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:236`:
```python
self.use_flashinfer_mxfp4_moe = get_moe_runner_backend().is_flashinfer_mxfp4()
```

### What to try

```bash
--moe-runner-backend flashinfer_mxfp4
```

This routes MoE GEMMs through the NVFP4-native CUTLASS kernel with TMA support,
which should outperform generic Triton on NVFP4 tensors.

**Risk:** Medium — verify that the kernel handles the Qwen3.5 MoE dimensions correctly
(num_experts, hidden_dim, intermediate_size). Check with a short prompt first.

---

## 5. Linear Attention Backend for GatedDeltaNet — MEDIUM IMPACT

### Model structure

Qwen3.5-122B-A10B is a **hybrid model**: 24 standard attention layers + 24 GatedDeltaNet
(linear attention) layers. These alternate throughout the model depth.

### Current state

The default linear attention backend is `triton` (line 541 of `server_args.py`):
```python
linear_attn_backend: str = "triton"
```

### Available options

```python
LINEAR_ATTN_KERNEL_BACKEND_CHOICES = ["triton", "cutedsl", "flashinfer"]
```

FlashInfer's linear attention kernels may have SM12.x-specific optimizations. Worth benchmarking:

```bash
--linear-attn-backend flashinfer
```

Or set prefill/decode separately:
```bash
--linear-attn-decode-backend flashinfer
--linear-attn-prefill-backend flashinfer
```

The `cutedsl` option is also worth trying — it uses CuteDSL which has direct Blackwell tensor
core support.

**Files:**
- `python/sglang/srt/layers/attention/fla/` — FLA kernel backends
- Backend selection: `--linear-attn-backend` arg → `server_args.py:541`

---

## 6. Chunked Prefill Size — MINOR

The GB10 has 128 GB unified memory. The default `chunked_prefill_size` for large-memory GPUs
is 16K tokens. Increasing to 32K reduces kernel launch overhead for long prompts:

```bash
--chunked-prefill-size 32768
```

At 128 GB, this is safe. It reduces prefill latency by batching more tokens per kernel launch.

---

## 7. CUDA Graph Profiling — Identify True Bottleneck

Before or after any change, collect a profiling trace to confirm what is actually dominant:

```bash
--enable-profile-cuda-graph
```

This enables `torch.profiler` recording during CUDA graph replay. The output shows top ops by
`cuda_time_total`. Use this to confirm whether MoE GEMM or GatedDeltaNet or attention is the
dominant cost, which determines which optimization gives the most return.

---

## 8. Recommended Optimized Launch Config

```bash
python -m sglang.launch_server \
  --model-path ~/.cache/huggingface/hub/models--Sehyo--Qwen3.5-122B-A10B-NVFP4/snapshots/<hash>/ \
  --quantization compressed-tensors \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.94 \
  --max-model-len 32768 \
  --moe-runner-backend flashinfer_mxfp4 \
  --linear-attn-backend flashinfer \
  --chunked-prefill-size 32768 \
  --mem-fraction-static 0.85 \
  --trust-remote-code
```

Add `--enable-profile-cuda-graph` for a diagnostic run.

---

## 9. Code Changes — Ordered by Expected Impact

| # | Change | Location | Impact | Risk |
|---|--------|----------|--------|------|
| 1 | Triton `num_stages=2` for SM12.x | `fp8_kernel.py:1300` | HIGH (~20–30%) | Low — tune up from 1 |
| 2 | `lm_head` FP8 via ignore-list override | `compressed_tensors.py:~217` | HIGH (~5–10%) | Low — FP8 well-tested |
| 3 | `--moe-runner-backend flashinfer_mxfp4` | Launch flag | MEDIUM | Medium — verify dims |
| 4 | `--linear-attn-backend flashinfer` | Launch flag | MEDIUM | Low — just a flag |
| 5 | `--chunked-prefill-size 32768` | Launch flag | LOW | Low — memory safe |
| 6 | `--enable-profile-cuda-graph` | Launch flag | Diagnostic | None |

---

## 10. Theoretical Performance Ceiling

| Scenario | Estimated tok/sec |
|---|---|
| Current (all defaults) | ~33 |
| + Triton num_stages=2 | ~40–43 |
| + lm_head FP8 | +1–2 on top |
| + flashinfer_mxfp4 backend | unknown until tested |
| Hard bandwidth ceiling (221 GB/s, full FP4/FP8) | ~50 |

The 221 GB/s bandwidth ceiling is the ultimate limit. To approach 50 tok/sec requires:
- All weight reads in FP4 or FP8 (no BF16 stragglers like `lm_head`)
- Near-100% memory bandwidth utilization (requires good pipeline depth → num_stages fix)
- Minimal kernel launch overhead (chunked prefill, CUDA graph)

**Realistic target with all changes above: 40–45 tok/sec.**

---

## Implementation Order

1. **Start with `num_stages=2`** — one-line change, high impact, no flag needed
2. **Benchmark** with `python -m sglang.bench_serving` before and after
3. **Try `--moe-runner-backend flashinfer_mxfp4`** — no code change, just a flag
4. **Try `--linear-attn-backend flashinfer`** — no code change, just a flag
5. **Implement lm_head FP8 override** — requires ~10-line code change + server arg

## Key Files

| File | Relevance |
|---|---|
| `python/sglang/srt/layers/quantization/fp8_kernel.py:1300` | `num_stages` for SM12.x |
| `python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py:217` | `ignore` list parsing; lm_head override point |
| `python/sglang/srt/layers/quantization/compressed_tensors/utils.py` | `should_ignore_layer()` implementation |
| `python/sglang/srt/layers/quantization/fp8.py` | `Fp8LinearMethod` (target for lm_head) |
| `python/sglang/srt/layers/moe/utils.py:65` | `MoeRunnerBackend` enum, `FLASHINFER_MXFP4` |
| `python/sglang/srt/layers/moe/fused_moe_triton/layer.py:236` | `use_flashinfer_mxfp4_moe` flag |
| `sgl-kernel/csrc/moe/nvfp4_blockwise_moe.cu` | NVFP4 MoE CUDA kernel |
| `python/sglang/srt/server_args.py:1757–1810` | Qwen3.5 model-specific backend logic |
| `python/sglang/srt/server_args.py:541` | `linear_attn_backend` default |
