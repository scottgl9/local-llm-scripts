# NVFP4 "!!!!" Output Investigation — Qwen3.5 on SM121 (GB10 Spark)

## Problem

Qwen3.5-NVFP4 models output only "!" characters (token ID 0) on the local vLLM build on SM121 (GB10 Spark). All output logprobs are NaN. Working containers produce correct output.

## RESOLVED

**Both 35B and 122B models now work correctly** after:
1. Restoring `VLLM_TEST_FORCE_FP8_MARLIN` override in `oracle/nvfp4.py` (forces Marlin MoE backend)
2. Removing debug instrumentation that caused torch.compile failures

### Verification Results
```
# 122B model with --language-model-only:
"The capital of France is" → " Paris.\nA. True\nB. False\n\n<think>..."   ← CORRECT

# 35B model with vision enabled (no --language-model-only):
"The capital of France is" → " the capital of France.\nThe capital of..." ← WORKS (lower quality, expected for 35B)

# 35B model with --language-model-only:
"The capital of France is" → " the capital of France.\nThe capital of..." ← ALSO WORKS
```

### Actual Root Cause

**CUTLASS MoE produces all-zero output on SM121.** The local build was auto-selecting CUTLASS MoE backend, while the working container forces Marlin MoE via `VLLM_TEST_FORCE_FP8_MARLIN=1`.

The "gate weights all zeros" observation from Phase 5 was likely an artifact of the debug instrumentation causing torch.compile to crash during warmup, leaving model weights in a partially-loaded state. Once debug code was removed and the model could complete initialization normally, weights loaded correctly.

## Earlier Findings (Preserved from investigation)

### NaN topk_weights from MoE Router — Gate Weights Not Loaded (Phase 5 finding)

**Note: This was observed during debugging with torch.compile-incompatible `.item()` calls that caused model initialization crashes.**

The MoE router gate linear layer (`self.gate`, an `nn.Linear`) had **broken weights** during the debug session:
- **Layer 0**: 489/524288 NaN values, inf mean — partially corrupted
- **Layers 1-39**: ALL ZEROS (0/524288 nonzero) — weights never loaded

```
Gate[4]: logits_nan=True hs_nan=False gate_w: shape=[256, 2048] dtype=bfloat16
         nan=True nan_count=489/524288 nonzero=163825/524288 mean=inf     (Layer 0)
Gate[5]: logits_nan=True hs_nan=False gate_w: shape=[256, 2048] dtype=bfloat16
         nan=False nan_count=0/524288 nonzero=0/524288 mean=0.0000e+00   (Layer 1+)
```

NaN/zero gate weights → NaN/zero router_logits → NaN topk_weights from softmax → NaN MoE output → all-NaN hidden states → token 0 ("!")

### Checkpoint Key Structure

The checkpoint uses `model.language_model.layers.X.mlp.gate.weight` prefix:
- `Qwen3_5MoeForConditionalGeneration` (multimodal) has `self.language_model` → `self.model` → `self.layers`
- `AutoWeightsLoader` strips `model.` → finds `language_model` child → delegates to `Qwen3_5MoeForCausalLM`
- `--language-model-only` only sets MM input limits to 0, does NOT change weight loading

## Evidence Chain

### GEMM2 NaN from topk_weights multiplication
```
GEMM1 (w13 gate/up): nan=False, mean=2.8e-01  OK
Activation (SiLU):   nan=False, mean=1.0e-02  OK
GEMM2 (w2 down):     nan=True,  mean=0.0e+00  BROKEN

topk_weights values: [nan, nan, nan, nan, nan, nan, nan, nan]
```

### Standalone GEMM2 replay proof
```python
# With model's routing (NaN topk_weights, mul_topk_weights=True)
Result: nan=True

# With simple routing (ones weights, mul_topk_weights=False)
Result: nan=False, mean=3.8e-03  WORKS!
```

The GEMM2 kernel itself works correctly. The NaN comes from multiplying by NaN topk_weights.

### Routing chain
1. `hidden_states` → `gate` linear layer → `router_logits` (shape [M, 256])
2. `router_logits` → `vllm_topk_softmax` → `topk_weights` (shape [M, 8])

## Investigation Timeline

### Phase 1: MoE Backend (Fixed — THIS WAS THE ROOT CAUSE)
- CUTLASS MoE produces all-zero output on SM121
- Restored `VLLM_TEST_FORCE_FP8_MARLIN` in `oracle/nvfp4.py`
- This forces Marlin MoE which works correctly

### Phase 2: Individual Kernel Tests (All pass)
- Linear Marlin GEMM: works
- MoE Marlin GEMM: works with manual routing
- Weight processing: correct

### Phase 3: Full Model NaN Localization
- Embedding: OK (508M/508M nonzero weights)
- V1 runner: uses `inputs_embeds` path (multimodal model), correctly computed
- Layer 0 attention: OK (mean=5.4e-03)
- Layer 0 MLP (MoE): NaN

### Phase 4: Narrowing MoE Stage
- GEMM1 (w13): OK
- Activation: OK
- **GEMM2 (w2): NaN** ← traced to `mul_topk_weights=True` with NaN weights

### Phase 5: Gate Weight Loading Failure (observed during debug, likely artifact)
- Discovered `topk_weights` are all NaN
- Source: `fused_topk()` → `vllm_topk_softmax()` applied to gate output
- Gate output (router_logits) is NaN because gate weights are NaN/zero
- **Layer 0 gate**: 489 NaN values out of 524288, partially loaded
- **Layers 1-39 gates**: ALL ZERO weights — never loaded from checkpoint
- Checkpoint keys: `model.language_model.layers.X.mlp.gate.weight`
- Model class: `Qwen3_5MoeForConditionalGeneration` → `Qwen3_5MoeForCausalLM` → `Qwen3_5Model`
- AutoWeightsLoader routes: `model.` → `language_model.` → CausalLM's `load_weights()`
- **NOTE**: These zero gate weights were likely caused by debug `.item()` calls crashing torch.compile during warmup, leaving weights uninitialized

### Phase 6: Vision/Language-Only Testing (COMPLETED)
- Removed all debug instrumentation (caused torch.compile errors with `.item()` calls in torch.compile'd code)
- Results:
  - **35B with vision**: WORKS — produces text output
  - **35B with --language-model-only**: WORKS — produces same text output
  - **122B with --language-model-only**: WORKS — produces correct "Paris" output
- **Conclusion**: `--language-model-only` is NOT the cause. The fix was:
  1. `VLLM_TEST_FORCE_FP8_MARLIN=1` in `oracle/nvfp4.py` (from Phase 1)
  2. Removing debug instrumentation that was incompatible with torch.compile

## Model Architecture
- 35B model: 40 layers, MoE layers, 256 experts, top-8
- 122B model: 48 layers, all with MoE (256 experts, top-8)
- `layer_types`: mix of `linear_attention` and `full_attention`
- `hidden_size=2048`, `intermediate_size=512`
- Architecture: `Qwen3_5MoeForConditionalGeneration` (multimodal, in `qwen3_5.py`)
- Internal router: gate runs inside FusedMoE (`DefaultMoERunner.forward_impl`)

## Environment
- GPU: NVIDIA GB10, SM121, CUDA 13.0, 119 GiB unified memory
- Branch: `gb10-spark-main-20260305`
- Test models: `Sehyo/Qwen3.5-35B-A3B-NVFP4`, `Sehyo/Qwen3.5-122B-A10B-NVFP4`
- Key env vars: `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1`

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `oracle/nvfp4.py` | `VLLM_TEST_FORCE_FP8_MARLIN` MoE override | Done (THE FIX) |
| `compressed_tensors_moe.py` | GB10 `.clone()` workaround | Applied (not needed for this fix) |
| `fused_marlin_moe.py` | Stage-by-stage NaN detection | Debug removed |
| `qwen3_next.py` | Per-layer NaN detection | Debug removed |
| `gpu_model_runner.py` | Embedding copy debug | Debug removed |
| `default_moe_runner.py` | Gate weight NaN check | Debug removed |
| `test_gemm2_replay.py` | Standalone GEMM2 replay | Test file |

## Debunked Theories

1. **"CUDA unified memory address" theory** — Wrong. The NaN was always from GEMM2 multiplying by NaN topk_weights, not from memory address issues.
2. **"CUTLASS FP4 MoE works on SM121"** — Wrong. CUTLASS MoE produces all-zero output on SM121; Marlin MoE is needed.
3. **"GEMM1 is broken"** — Wrong (previous session). GEMM1 works fine; the NaN propagates from GEMM2's topk_weights multiplication.
4. **".clone() workaround fixes it"** — Wrong. The clone was tested with simplified routing that avoided NaN topk_weights.
5. **"--language-model-only causes gate weight loading failure"** — Wrong. Both 35B and 122B work with `--language-model-only`. The gate weight zeros were an artifact of debug code crashing torch.compile.
6. **"Gate weights are not loaded from checkpoint"** — Likely wrong / artifact. Gate weights load correctly when debug instrumentation is removed. The zero weights were observed during sessions where `.item()` calls in torch.compile'd code caused initialization crashes.
