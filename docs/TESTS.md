# Qwen3.5-35B-A3B-NVFP4 Speed Tests on GB10

Target: Beat container v23's **66.75 tps_decode p50** by >5% (target: ~70+ tps)

## Test Parameters
- Speed test: `python3 ~/llm_speed_test.py --runs 8 --warmup 2 --max-tokens 512 --skip-tests`
- GPU: NVIDIA GB10, SM121, CUDA 13.0, 119 GiB unified memory
- vLLM: 0.16.1rc1, FlashInfer 0.6.4
- Container baseline (v23): vLLM 0.16.0rc2, FlashInfer 0.6.3 — **66.75 tps_decode p50**

## Results Summary

| Experiment | Spec tokens | tps_decode p50 | Delta vs baseline | Notes |
|------------|-------------|---------------|-------------------|-------|
| Baseline (forced prefill MTP) | 3 | 44.83 | — | build_for_drafting forced prefill path |
| enforce-eager | 3 | 34.96 | -22.0% | torch.compile essential |
| CUTLASS GEMM | 3 | 45.01 | +0.4% | ~same as Marlin |
| performance-mode throughput | 3 | 44.45 | -0.8% | no difference |
| num_speculative_tokens=5 | 5 | 31.08 | -30.6% | poor acceptance (20%) |
| **Remove build_for_drafting** | 3 | **57.55** | **+28.4%** | native decode path works! |
| Native decode, spec=2 | 2 | 59.61 | +33.0% | prev best |
| Native decode, spec=1 | 1 | 59.36 | +32.4% | ~same as spec=2 |
| Native decode, no MTP | 0 | 41.74 | -6.9% | raw decode speed |
| **FP8 draft lm_head, spec=3** | 3 | **69.02** | **+54.0%** | **BEATS container (+3.4%)** |
| FP8 draft lm_head (run 2) | 3 | 69.92 | +56.0% | confirmed |
| FP8 draft lm_head, spec=4 | 4 | 66.50 | +48.3% | worse than spec=3 |
| FP8 draft lm_head, spec=5 | 5 | 66.68 | +48.7% | worse than spec=3 |
| **FP8 lm_head + FP8 MoE experts** | 3 | **71.29** | **+59.6%** | **VLLM_MTP_MOE_FP8=1 (+6.8% vs container)** |
| FP8 lm_head + FP8 MoE (run 2) | 3 | 69.77 | +56.6% | confirmed, avg ~70.5 tps |
| Container v23 (target) | 3 | **66.75** | — | vLLM 0.16.0rc2 + FI 0.6.3 |

## Detailed Results

### Phase 1: Baseline with forced prefill MTP drafting

#### Baseline (matching container args, flashinfer attention)
- Date: 2026-03-06
- tps_decode: p50=44.83 mean=44.56 p95=45.53
- tps_e2e: p50=44.35 mean=44.09 p95=45.04
- TTFT: p50=0.123 mean=0.123
- Notes: Significantly below container v23 (66.75). MTP spec method, Marlin GEMM+MoE.
  The `build_for_drafting` override forced all MTP draft steps through the prefill path.

#### enforce-eager (no torch.compile / CUDA graphs)
- tps_decode: p50=34.96 mean=35.13
- Notes: 28% slower than compiled baseline. Compilation+CUDA graphs essential.

#### GEMM Backend: cutlass
- tps_decode: p50=45.01 mean=44.88
- Notes: ~same as Marlin. Needs gpu-memory-utilization 0.93 (uses slightly more VRAM).

#### Performance Mode: throughput
- tps_decode: p50=44.45 mean=44.33
- Notes: ~same as baseline

#### num_speculative_tokens=5
- tps_decode: p50=31.08 mean=30.97
- Notes: Much worse. Acceptance rate only 20% (1283/6365).

#### flash_attn backend
- Notes: FAILED — flash_attn does not support fp8 kv_cache_dtype

### Phase 2: Remove build_for_drafting override (KEY FIX)

**Commit:** `41f74917a` — perf: remove forced prefill path for MTP drafting on SM121

The `build_for_drafting` override in `flashinfer.py` forced MTP draft steps through the
prefill path to avoid a suspected crash in `BatchDecodeWithPagedKVCacheWrapper`. Testing
confirms the native FlashInfer decode path works correctly on SM121 with FlashInfer 0.6.4.

#### 3 speculative tokens (native decode)
- tps_decode: p50=57.55 mean=57.78 p95=59.32
- tps_e2e: p50=56.77 mean=56.98 p95=58.50
- TTFT: p50=0.124 mean=0.123
- MTP acceptance: pos0=87-89%, pos1=49-53%, pos2=36-41%, overall=57-63%
- Mean acceptance length: 2.7-2.8 tokens

#### 2 speculative tokens (native decode) — BEST
- tps_decode: p50=59.61 mean=59.58 p95=60.96
- tps_e2e: p50=58.85 mean=58.82 p95=60.17
- TTFT: p50=0.111 mean=0.111

#### 1 speculative token (native decode)
- tps_decode: p50=59.36 mean=59.68 p95=61.03
- tps_e2e: p50=58.70 mean=59.01 p95=60.34
- TTFT: p50=0.097 mean=0.097

#### No MTP (raw decode baseline)
- tps_decode: p50=41.74 mean=41.70 p95=41.75
- tps_e2e: p50=41.53 mean=41.47 p95=41.53
- TTFT: p50=0.063 mean=0.067
- Notes: Raw decode is very close to container's no-MTP result (42.21).

### Speculative Token Comparison (our build vs container)

| Spec tokens | Our build (tps p50) | Container v23 (tps p50) | Gap |
|-------------|--------------------|-----------------------|-----|
| 0 (no MTP) | 41.74 | 42.21 | -1.1% |
| 1 | 59.36 | 60.25 | -1.5% |
| 2 | 59.61 | 65.11 | -8.4% |
| 3 | 57.55 | 66.98 | -14.1% |

**Key insight:** At 1 spec token our build nearly matches the container (-1.5%). The gap
grows with more spec tokens, suggesting MTP per-step overhead is higher in our build
(vLLM 0.16.1rc1 vs 0.16.0rc2). Likely causes:
1. `extract_hidden_states` refactor in 0.16.1rc1 (new MTP code path)
2. CUDA graph mode: PIECEWISE only (both builds, no TRTLLM on SM121)
3. Missing Triton MoE tuned config for E=256,N=512 on GB10

### MTP Acceptance Rate Analysis

#### With forced prefill path (old)
- Position 0: ~90%, Position 1: ~19%, Position 2: ~4%
- Overall: 37.8%, Mean acceptance length: ~1.1 tokens
- Notes: Prefill path produced bad draft quality at positions 1+

#### With native decode path (new)
- Position 0: ~87-89%, Position 1: ~49-53%, Position 2: ~36-41%
- Overall: 57-63%, Mean acceptance length: 2.7-2.8 tokens
- Notes: Much better draft quality. The decode path runs MTP properly.

### Phase 3: FP8 lm_head draft sampling (KEY OPTIMIZATION)

**Env var:** `VLLM_DRAFT_SAMPLE_OPT=fp8`

The MTP model's `lm_head` (248K vocab × 2048 hidden, bf16) is the dominant per-step
cost (~85%, ~5.6ms per draft step). By quantizing to FP8 (float8_e4m3fn), the weight
matrix shrinks from ~1 GB to ~500 MB, halving memory bandwidth.

Microbenchmark results (isolated matmul+argmax, batch=1):
- BF16: ~35ms per call (without CUDA graph context)
- FP8: ~3.7ms per call — **9.5x faster**
- torch.compile: ~6.2ms — **3.3x faster**
- Fused Triton kernel: ~284ms — 10x slower (abandoned)
- Token match rate FP8 vs BF16: ~85% (per-tensor quant, random data)

#### FP8 draft lm_head, spec=3 — **NEW BEST, BEATS CONTAINER**
- Date: 2026-03-06
- tps_decode: p50=69.02 mean=69.55 p95=73.54 (run 1)
- tps_decode: p50=69.92 mean=69.45 p95=70.93 (run 2)
- tps_e2e: p50=68.00 mean=68.51 p95=72.38 (run 1)
- tps_e2e: p50=68.87 mean=68.42 p95=69.88 (run 2)
- TTFT: p50=0.111 mean=0.112
- **vs container v23: +3.4% (69.02 vs 66.75)**
- **vs previous best (native decode, spec=2): +15.8% (69.02 vs 59.61)**
- Notes: `VLLM_DRAFT_SAMPLE_OPT=fp8` with 3 spec tokens.
  The FP8 lm_head reduces per-step MTP overhead enough that spec=3 is now
  profitable again (was -2.06 tps marginal gain, now positive).

#### FP8 draft lm_head, spec=4
- Date: 2026-03-06
- tps_decode: p50=66.50 mean=66.51 p95=68.74
- tps_e2e: p50=65.48 mean=65.48 p95=67.65
- TTFT: p50=0.120 mean=0.120
- **vs spec=3: -2.5% (66.50 vs 69.02)** — worse despite FP8 making draft steps cheaper
- Notes: 4th spec token not accepted frequently enough to offset 4th MTP forward pass cost.

#### FP8 draft lm_head, spec=5
- Date: 2026-03-06
- tps_decode: p50=66.68 mean=67.50 p95=71.80
- tps_e2e: p50=65.59 mean=66.39 p95=70.53
- TTFT: p50=0.127 mean=0.127
- **vs spec=3: -2.3% (66.68 vs 69.02)** — similar to spec=4, not profitable
- Notes: Higher variance (64.01–73.16 tps range). **spec=3 remains optimal.**

### Phase 4: FP8 MTP MoE expert quantization (VLLM_MTP_MOE_FP8=1)

#### FP8 lm_head + FP8 MoE experts, spec=3 — **NEW BEST**
- Date: 2026-03-06
- tps_decode: p50=71.29 mean=71.02 p95=73.38 (run 1)
- tps_decode: p50=69.77 mean=69.59 p95=72.09 (run 2)
- Average p50: **~70.5 tps**
- tps_e2e: p50=70.22 mean=69.95 (run 1)
- TTFT: p50=0.110 mean=0.110
- **vs container v23: +6.8% / +4.5% (71.29 / 69.77 vs 66.75)**
- **vs FP8 lm_head only: +3.3% / +1.1% (71.29 / 69.77 vs 69.02)**
- Notes: `VLLM_MTP_MOE_FP8=1` post-quantizes MTP MoE expert weights from bf16 to
  FP8 at load time (before CUDA graph capture). 256 experts × [gate_up, down_proj]
  quantized per-expert to float8_e4m3fn with per-expert per-tensor scales.
  1536 MB bf16 → 768 MB fp8 (-50%). Active expert load per step: 50 MB → 25 MB.
  Higher run-to-run variance than lm_head-only (~3 tps range vs ~1.5 tps range).
  Implemented in `draft_sample_opt.py:quantize_mtp_moe_fp8()`.

### Step 5: Triton MoE Autotuned Config (E=256,N=512)

**Status:** Not yet completed. See `docs/TRITON_MOE_AUTOTUNING.md` for instructions.

**How to run:**
```bash
cd ~/sandbox/vllm && source .venv-gb10/bin/activate

# Required fix: benchmark_moe.py didn't recognize Qwen3_5MoeForConditionalGeneration.
# Added it to the architecture list in get_model_params() alongside Qwen3VLMoeForConditionalGeneration.

python benchmarks/kernels/benchmark_moe.py \
    --model Sehyo/Qwen3.5-35B-A3B-NVFP4 \
    --dtype fp8_w8a8 \
    --tp-size 1 \
    --tune \
    --trust-remote-code \
    --save-dir vllm/model_executor/layers/fused_moe/configs/
```

Output: `vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json`
Duration: ~8-10 hours (all 18 batch sizes) or ~4-5 hours (decode-only batch sizes).

- tps_decode: p50= mean= (pending)
- Notes: No tuned config for GB10 exists yet. Default kernel params used.

## Remaining Performance Gap Analysis

**Current best: 59.61 tps (2 spec tokens) vs container's 66.75 tps (3 spec tokens) = -10.7%**

### Why our build is slower with 2-3 spec tokens

The raw decode speed (no MTP) is nearly identical: 41.74 vs 42.21 (-1.1%). But MTP
gives the container +59% while only giving us +43% (at spec=2). The per-step MTP
overhead is higher in our build.

Potential causes:
1. **vLLM 0.16.1rc1 MTP code path changes** — the `extract_hidden_states` system was
   refactored between 0.16.0rc2 and 0.16.1rc1. This new code path may have more overhead.
2. **FlashInfer 0.6.3 vs 0.6.4** — kernel performance differences
3. **Missing Triton MoE tuned config** — affects MoE layer speed during MTP draft steps
4. **CUDA graph behavior** — both builds use PIECEWISE mode, but the graph capture/replay
   characteristics may differ between vLLM versions

### Next steps to close the gap
1. Complete Triton MoE autotuning (see `docs/TRITON_MOE_AUTOTUNING.md`)
2. Profile MTP draft step overhead (nsys or torch profiler)
3. Compare `extract_hidden_states` code path vs v0.16.0rc2 eagle.py MTP path
4. Test with FlashInfer 0.6.3 to isolate version impact
5. Investigate TRTLLM attention on SM121 (cubins are SM10x only; see investigation notes)

## Prior test results (from QWEN35_35B_TESTS.md on container v23)
- See `/home/scottgl/spark-llm-scripts/docs/QWEN35_35B_TESTS.md` for full results
- Spec tokens: 1→60.25, 2→baseline, 3→66.98 (winner), 4→62.86, none→42.21
- GPU_MEMORY_UTIL: 0.95→67.76 (+1.2%)
- enforce-eager: 60.42 (-10.8%)
- KV cache fp8 vs auto: no significant difference
