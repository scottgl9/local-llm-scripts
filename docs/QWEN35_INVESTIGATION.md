# NVFP4 "!!!!" Output Investigation — Qwen3.5 on SM121 (GB10 Spark)

## Problem

Qwen3.5-NVFP4 models output only "!" characters (token ID 0) on the local vLLM build on SM121 (GB10 Spark). All output logprobs are NaN. Working containers produce correct output.

## Root Cause: NaN topk_weights from MoE Router

**The `topk_weights` (router softmax weights) are all NaN.** When GEMM2 (w2 down projection) multiplies its output by these NaN weights (`mul_topk_weights=True`), the result is all NaN.

### Evidence
```
GEMM1 (w13 gate/up): nan=False, mean=2.8e-01  OK
Activation (SiLU):   nan=False, mean=1.0e-02  OK
GEMM2 (w2 down):     nan=True,  mean=0.0e+00  BROKEN

topk_weights values: [nan, nan, nan, nan, nan, nan, nan, nan]
```

### Proof: Standalone replay
```python
# With model's routing (NaN topk_weights, mul_topk_weights=True)
Result: nan=True

# With simple routing (ones weights, mul_topk_weights=False)
Result: nan=False, mean=3.8e-03  WORKS!
```

The GEMM2 kernel itself works correctly. The NaN comes from multiplying by NaN topk_weights.

## Why topk_weights are NaN

The routing chain:
1. `hidden_states` → `gate` linear layer → `router_logits` (shape [M, 256])
2. `router_logits` → `vllm_topk_softmax` → `topk_weights` (shape [M, 8])

If the gate linear layer produces inf or NaN in the logits, softmax will produce NaN weights.

The gate layer is a regular `nn.Linear` (not quantized). Possible causes:
- Gate weight loading issue (wrong values)
- Gate weight dtype issue
- Gate computation issue on SM121
- The `vllm_topk_softmax` CUDA kernel issue on SM121

## Investigation Timeline

### Phase 1: MoE Backend (Fixed, not root cause)
- CUTLASS MoE produces all-zero output on SM121
- Restored `VLLM_TEST_FORCE_FP8_MARLIN` in `oracle/nvfp4.py`

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

### Phase 5: topk_weights (Current)
- Discovered `topk_weights` are all NaN
- Source: `fused_topk()` → `vllm_topk_softmax()` applied to gate output
- Need to check: gate linear layer output (router_logits)

## Next Steps

1. Add NaN check on `router_logits` output from gate layer
2. Check if `vllm_topk_softmax` kernel works on SM121
3. Check gate weight values
4. Compare gate output with working container

## Model Architecture
- 40 layers, all with MoE (256 experts, top-8)
- `layer_types`: mix of `linear_attention` and `full_attention`
- `hidden_size=2048`, `intermediate_size=512`
- Internal router: gate runs inside FusedMoE

## Environment
- GPU: NVIDIA GB10, SM121, CUDA 13.0, 119 GiB unified memory
- Branch: `gb10-spark-main-20260305`
- Test model: `Sehyo/Qwen3.5-35B-A3B-NVFP4`
- Key env vars: `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1`

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `oracle/nvfp4.py` | `VLLM_TEST_FORCE_FP8_MARLIN` MoE override | Done |
| `compressed_tensors_moe.py` | GB10 `.clone()` workaround | Applied (not the fix) |
| `fused_marlin_moe.py` | Stage-by-stage NaN detection | Temporary debug |
| `qwen3_next.py` | Per-layer NaN detection | Temporary debug |
| `gpu_model_runner.py` | Embedding copy debug | Temporary debug |
| `test_gemm2_replay.py` | Standalone GEMM2 replay | Test file |
