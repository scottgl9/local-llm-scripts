# NVFP4 "!!!!" Output Investigation — Qwen3.5 on SM121 (GB10 Spark)

## Problem

Qwen3.5-NVFP4 models (both 122B and 35B) output only "!" characters (token ID 0) when running on the local vLLM build on SM121 (GB10 Spark). All output logprobs are NaN. The `vllm/vllm-openai:cu130-nightly` and `avarok/dgx-vllm-nvfp4-kernel:v23` containers produce correct output with the same model and similar flags.

## Environment

- **GPU**: NVIDIA GB10, SM121, CUDA 13.0, 119 GiB unified memory
- **Branch**: `gb10-spark-main-20260305`
- **Test model**: `Sehyo/Qwen3.5-35B-A3B-NVFP4` (smaller, same architecture as 122B)
- **Quantization**: compressed-tensors (W4A4 NVFP4)
- **Key env vars**: `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1`, `VLLM_USE_FLASHINFER_MOE_FP4=0`

## Root Cause: CUDA Unified Memory Address Issue on GB10

The NaN originates from `moe_wna16_marlin_gemm` reading MoE weight tensors from certain unified memory addresses. GB10 has 119 GiB **unified** (CPU/GPU shared) memory. Weights loaded by the model loading pipeline end up at memory addresses that the Marlin MoE CUDA kernel cannot read correctly, producing all-NaN output.

**Key evidence:**
- `.clone()` of the same weight tensor at a different memory address produces correct results
- `.contiguous()` also produces correct results (allocates new memory)
- Tensor metadata is identical: same shape, stride, contiguity, storage_offset, `torch.equal()` returns True
- The only difference is the **data_ptr** address (e.g. `0xfdfe40000000` vs `0xfdf7c0000000`)

**Fix:** Force `.clone()` on MoE weight tensors after loading to ensure they reside in kernel-accessible memory regions.

## Findings (Chronological)

### 1. MoE Backend Override Restored (Necessary but not sufficient)

**File**: `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` (line 239)

The local commit `d074a89ee` had removed the `VLLM_TEST_FORCE_FP8_MARLIN` check from MoE backend selection, allowing CUTLASS MoE to be auto-selected. CUTLASS MoE produces all-zero output on SM121. Restored the override to force Marlin MoE.

**Status**: Fixed, but "!!!!" output persisted — necessary but not the root cause.

### 2. Linear Marlin GEMM Works on SM121

**Test**: `test_marlin_kernel.py` — loads real q_proj weights from the 35B model, repacks, runs Marlin GEMM.

```
output: mean=1.29e-02, nan=False, inf=False
```

### 3. MoE Marlin GEMM Works on SM121

**Test**: `test_moe_kernel.py` — loads real gate_proj weights, builds stacked tensors for 256 experts, calls `ops.moe_wna16_marlin_gemm` with M=8192, top_k=8.

```
MoE GEMM result: mean=1.50e-02, nan=False, inf=False
```

The MoE Marlin GEMM works correctly when tensors are allocated fresh by the test harness.

### 4. Weight Processing is Correct

- CT checkpoint divisor inversion (`1/raw_value`) produces correct FP32 values
- BF16 conversion preserves the values (no underflow)
- `nvfp4_marlin_process_global_scale` produces ~2e31 (BF16), representable
- No inf or NaN after processing

### 5. FP16 Underflow Theory — DEBUNKED

FP16 subnormals preserve the global_scale value. Reverted all `marlin_utils_fp4.py` changes.

### 6. NaN Localized to MoE Layer 0, Routed Expert Output

Per-layer NaN detection added to `qwen3_next.py`:
- Layer 0 forward → attention output: OK
- Layer 0 forward → MLP (MoE) → shared expert output: OK
- Layer 0 forward → MLP (MoE) → **fused (routed) expert output: NaN**
- First `moe_wna16_marlin_gemm` call (w13 gate/up projection) produces all-NaN

### 7. Saved Tensors Replay Works — Problem is Memory-Location Dependent

Tensors saved to `/tmp/moe_gemm1_tensors.pt` from the failing call and replayed in a standalone script produce correct results. The kernel code is correct; the problem is where the tensors reside in memory.

### 8. BREAKTHROUGH: `.clone()` Fixes NaN

Debug retry code in `_fused_marlin_moe()` (fused_marlin_moe.py):

| Retry | What Changed | Result |
|-------|-------------|--------|
| RETRY (same tensors) | Fresh output buffer only | NaN |
| RETRY2 (tiny M=4) | Same weights, small input | NaN |
| **RETRY3 (.clone())** | **Cloned w1, w1_scale, global_scale1** | **OK (mean=3.39e-02)** |
| **RETRY4 (.contiguous())** | **w1.contiguous(), etc.** | **OK** |

Tensor properties comparison:
```
w1:  contiguous=True stride=(262144, 2048, 1) storage_offset=0 data_ptr=fdfe40000000
cw:  contiguous=True stride=(262144, 2048, 1) storage_offset=0 data_ptr=fdf7c0000000
torch.equal(w1, cw) = True  (data identical)
```

**Conclusion**: The Marlin MoE CUDA kernel fails when reading from certain CUDA unified memory addresses on GB10. Copying to fresh allocations places them at accessible addresses.

## Container Comparison

### Working Container (122B)
- Image: `vllm/vllm-openai:cu130-nightly`
- Flags: `VLLM_TEST_FORCE_FP8_MARLIN=1`, `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_USE_FLASHINFER_MOE_FP4=0`
- Extra: `--kv-cache-dtype fp8`, `--attention-backend flashinfer`, `--no-enable-chunked-prefill`

### Working Container (35B)
- Image: `avarok/dgx-vllm-nvfp4-kernel:v23`
- Same env vars and runtime flags

### Local Build (BROKEN → FIXED)
- Custom vLLM build with SM121 patches
- Fix: force `.clone()` on MoE weights after processing

## Fix Applied

**File**: `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` (line ~548)

After `convert_to_nvfp4_moe_kernel_format()` returns processed weights, force `.clone()` on the weight and scale tensors to move them to fresh memory allocations:

```python
# GB10 workaround: CUDA unified memory address issue causes NaN
# in moe_wna16_marlin_gemm. Cloning forces new allocations.
w13 = w13.clone()
w13_scale = w13_scale.clone()
w2 = w2.clone()
w2_scale = w2_scale.clone()
```

## Files Modified

| File | Change | Status |
|------|--------|--------|
| `oracle/nvfp4.py` | Restored `VLLM_TEST_FORCE_FP8_MARLIN` MoE override | Done |
| `compressed_tensors_moe.py` | `.clone()` MoE weights after processing (GB10 fix) | Done |
| `fused_marlin_moe.py` | Debug logging + retry tests | Temporary (remove) |
| `qwen3_next.py` | Per-layer NaN detection | Temporary (remove) |
| `marlin_utils_fp4.py` | Debug logging | Temporary (remove) |
| `compressed_tensors_w4a4_nvfp4.py` | Debug logging | Temporary (remove) |
| `test_marlin_kernel.py` | Standalone linear GEMM test | Test file |
| `test_marlin_debug.py` | Full model inference test | Test file |
| `test_moe_kernel.py` | Standalone MoE GEMM test | Test file |
| `test_nan_detector.py` | Full model NaN detection test | Test file |
