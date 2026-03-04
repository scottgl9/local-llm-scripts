#!/usr/bin/env python3
"""
patch_trtllm_minimax.py
=======================
Patch TRT-LLM 1.3.0rc6 to serve MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10
(llm-compressor compressed-tensors NVFP4 format).

Run this INSIDE the container before starting trtllm-serve:
  python3 /patches/patch_trtllm_minimax.py

Key discoveries:
  - Model is NVFP4 (FP4 weights, group_size=16), llm-compressor compressed-tensors format
  - Must use QuantAlgo.NVFP4 (nvfp4_gemm kernel), NOT W4A8_NVFP4_FP8
    (fp4_fp8_gemm_trtllmgen requires SM 10.0a/10.3a; GB10 DGX Spark is SM 12.1a)
  - llm-compressor key names differ from TRT-LLM's expected names
  - Scale conventions differ between llm-compressor and TRT-LLM

Key name mapping (llm-compressor -> TRT-LLM):
  weight_packed        -> weight
  weight_global_scale  -> weight_scale_2  (with conversion: 1/(6*weight_global_scale) = amax_weight/2688)
  input_global_scale   -> input_scale     (with conversion: 1/(6*input_global_scale) = amax_input/2688;
                                           TRT-LLM's load_weight_scales inverts this to get 2688/amax_input)

Scale convention (llm-compressor stores quantization factors):
  input_global_scale  = FP8_MAX / amax_input  = 448 / amax_input
  weight_global_scale = FP8_MAX / amax_weight = 448 / amax_weight

NVFP4 linear scale convention (from create_weights comments in TRT-LLM rc6):
  llm-compressor stores:
    input_global_scale  = FP8_MAX / amax_input  = 448 / amax_input  (e.g., 163 for amax≈2.75)
    weight_global_scale = FP8_MAX / amax_weight = 448 / amax_weight (e.g., 8512 for amax≈0.053)
  TRT-LLM NVFP4 expects in checkpoints (confirmed from create_weights docstrings):
    input_scale    = amax_input/(FP8_MAX*E2M1_MAX) = 1/(6*igs)   [small, e.g. ~1.02e-3]
    weight_scale_2 = amax_weight/(FP8_MAX*E2M1_MAX) = 1/(6*wgs)  [small, e.g. ~1.96e-5]
  TRT-LLM load_weight_scales then:
    INVERTS input_scale:  module.input_scale  = 2688/amax_input = 6*igs   [large, ~978]
    keeps weight_scale_2: module.weight_scale_2 = amax_weight/2688 = 1/(6*wgs) [small, ~1.96e-5]
    computes alpha = pre_inv_input_scale * weight_scale_2
                   = (amax_input/2688) * (amax_weight/2688)
                   = amax_input * amax_weight / 2688^2             [~2.02e-8]

MoE NVFP4 CUTLASS (load_expert_fc31_alpha_nvfp4):
  tmp_fc31_input_scale[expert] = input_scale_stored = 1/(6*igs) = amax_input/2688
  module.fc31_input_scale = 1/max(tmp) = 2688/max_amax_input  [inverted in load_quant_scales]
  w3_w1_weight_scale_2 = 1.0 / weight_scale_2 = 6*wgs = 2688/amax_weight
  fc31_alpha = 1/(fc31_input_scale * w3_w1_weight_scale_2) = max_amax_input * amax_weight / 2688^2
"""

import sys

MODEL_CONFIG   = '/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/model_config.py'
LINEAR         = '/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/modules/linear.py'
MOE_CUTLASS    = '/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py'
QUANT          = '/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/modules/fused_moe/quantization.py'
MINIMAXM2      = '/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/models/modeling_minimaxm2.py'


def apply_patch(filepath, description, old, new):
    with open(filepath, 'r') as f:
        src = f.read()
    if old in src:
        with open(filepath, 'w') as f:
            f.write(src.replace(old, new, 1))
        print(f"  OK: {description}")
    elif new.strip()[:80] in src or (len(new) > 100 and new[:80] in src):
        print(f"  Already patched: {description}")
    else:
        print(f"  ERROR: target not found for: {description}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Patch 1: model_config.py — add W4A8_NVFP4_FP8 for compressed-tensors 4-bit
# ---------------------------------------------------------------------------
print("=== Patch 1: model_config.py (W4A8_NVFP4_FP8 for 4-bit compressed-tensors) ===")
apply_patch(
    MODEL_CONFIG,
    'compressed-tensors: 4-bit -> NVFP4 (nvfp4_gemm, SM12.1 compatible)',
    '            else:\n'
    '                raise ValueError(\n'
    '                    f"Unsupported quant_bits: {weights_quant_config[\'num_bits\']}. "\n'
    '                    "Supported: 8.")\n'
    '\n'
    '            quant_config.exclude_modules = hf_quant_config.get("ignore", [])',
    '            elif weights_quant_config["num_bits"] == 4:\n'
    '                fmt = hf_quant_config.get("format", "")\n'
    '                if "nvfp4" in fmt:\n'
    '                    # Use NVFP4 (nvfp4_gemm, SM12.1/GB10 compatible).\n'
    '                    # fp4_fp8_gemm_trtllmgen (W4A8) requires SM10.0a and does NOT run on GB10.\n'
    '                    quant_config.quant_algo = QuantAlgo.NVFP4\n'
    '                    quant_config.group_size = weights_quant_config.get("group_size", 16)\n'
    '                else:\n'
    '                    raise ValueError(f"Unsupported 4-bit format: {fmt}. Supported: nvfp4-pack-quantized.")\n'
    '            else:\n'
    '                raise ValueError(\n'
    '                    f"Unsupported quant_bits: {weights_quant_config[\'num_bits\']}. "\n'
    '                    "Supported: 4, 8.")\n'
    '\n'
    '            quant_config.exclude_modules = hf_quant_config.get("ignore", [])'
)

# ---------------------------------------------------------------------------
# Patch 2a: fused_moe_cutlass.py — _get_quant_method: add w4a8_nvfp4_fp8
# ---------------------------------------------------------------------------
print("=== Patch 2a: fused_moe_cutlass.py (_get_quant_method) ===")
apply_patch(
    MOE_CUTLASS,
    '_get_quant_method: add w4a8_nvfp4_fp8',
    '            elif self.quant_config.layer_quant_mode.has_nvfp4():\n'
    '                return NVFP4CutlassFusedMoEMethod()',
    '            elif (self.quant_config.layer_quant_mode.has_nvfp4() or\n'
    '                    self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8()):\n'
    '                return NVFP4CutlassFusedMoEMethod()'
)

# ---------------------------------------------------------------------------
# Patch 2b: fused_moe_cutlass.py — _check_configs: add w4a8_nvfp4_fp8
# ---------------------------------------------------------------------------
print("=== Patch 2b: fused_moe_cutlass.py (_check_configs) ===")
apply_patch(
    MOE_CUTLASS,
    '_check_configs: add w4a8_nvfp4_fp8',
    '                    | self.quant_config.quant_mode.has_w4a8_mxfp4_mxfp8()):',
    '                    | self.quant_config.quant_mode.has_w4a8_mxfp4_mxfp8()\n'
    '                    | self.quant_config.quant_mode.has_w4a8_nvfp4_fp8()):'
)

# ---------------------------------------------------------------------------
# Patch 3: linear.py NVFP4LinearMethod.load_weight_scales — normalize keys
#           AND invert input_global_scale -> 1/v for linear (W4A8 convention)
# ---------------------------------------------------------------------------
print("=== Patch 3: linear.py (load_weight_scales key normalization + scale inversion) ===")
apply_patch(
    LINEAR,
    'NVFP4LinearMethod.load_weight_scales: llm-compressor keys + NVFP4 scale conversion',
    '        for w in weights:\n'
    '            if "input_scale" in w:\n'
    '                if input_scale is None:\n'
    '                    input_scale = w["input_scale"][...]\n'
    '                else:\n'
    '                    assert input_scale == w["input_scale"][\n'
    '                        ...], "The input_scale should be same for all the weights"\n'
    '            if "weight_scale" in w:\n'
    '                ws = load_weight_shard(w["weight_scale"],',
    '        for w in weights:\n'
    '            # Normalize llm-compressor key names to TRT-LLM NVFP4 format.\n'
    '            # Scale convention: llm-compressor stores quantization factors:\n'
    '            #   input_global_scale  = FP8_MAX / amax_input  = 448 / amax_input\n'
    '            #   weight_global_scale = FP8_MAX / amax_weight = 448 / amax_weight\n'
    '            # NVFP4 (W4A4 fp4_quantize) expects:\n'
    '            #   input_scale  = 1/(6*igs) = amax_input/2688  [inverted by load_weight_scales to 2688/amax_input]\n'
    '            #   weight_scale_2 = 1/wgs  = amax_weight/448   [NOT inverted; used as-is for alpha]\n'
    '            #   alpha = pre_inv_input_scale * weight_scale_2 = amax_input*amax_weight/(2688*448)\n'
    '            if "weight_packed" in w and "weight" not in w:\n'
    '                w["weight"] = w["weight_packed"]\n'
    '            if "input_global_scale" in w and "input_scale" not in w:\n'
    '                import torch as _t; v = w["input_global_scale"][...]\n'
    '                v32 = v.to(_t.float32) if isinstance(v, _t.Tensor) else _t.tensor(float(v))\n'
    '                # input_global_scale = FP8_MAX/amax_input = 448/amax_input\n'
    '                # TRT-LLM expects: input_scale = 1/(6*igs) = amax_input/2688  [SMALL]\n'
    '                # load_weight_scales INVERTS this -> module.input_scale = 2688/amax_input  [LARGE, for fp4_quantize]\n'
    '                w["input_scale"] = 1.0 / (6.0 * v32)  # = amax_input / 2688\n'
    '            if "weight_global_scale" in w and "weight_scale_2" not in w:\n'
    '                import torch as _t; v = w["weight_global_scale"][...]\n'
    '                v32 = v.to(_t.float32) if isinstance(v, _t.Tensor) else _t.tensor(float(v))\n'
    '                # weight_global_scale = FP8_MAX/amax_weight = 448/amax_weight\n'
    '                # TRT-LLM expects: weight_scale_2 = amax_weight/(FP8_MAX*E2M1_MAX) = 1/(6*wgs)  [NOT inverted]\n'
    '                # module.alpha = pre_inv_input_scale * weight_scale_2 = amax_input*amax_weight/2688^2\n'
    '                w["weight_scale_2"] = 1.0 / (6.0 * v32)  # = amax_weight / 2688\n'
    '            if "input_scale" in w:\n'
    '                if input_scale is None:\n'
    '                    input_scale = w["input_scale"][...]\n'
    '                else:\n'
    '                    assert input_scale == w["input_scale"][\n'
    '                        ...], "The input_scale should be same for all the weights"\n'
    '            if "weight_scale" in w:\n'
    '                ws = load_weight_shard(w["weight_scale"],'
)

# ---------------------------------------------------------------------------
# Patch 4a: quantization.py — load_expert_weights: fallback to weight_packed
# ---------------------------------------------------------------------------
print("=== Patch 4a: quantization.py (load_expert_weights weight_packed fallback) ===")
apply_patch(
    QUANT,
    'load_expert_weights: fallback to weight_packed',
    '                w1_weight = weights[\n'
    '                    f"{expert_id}.w1.weight"] if f"{expert_id}.w1.weight" in weights else None\n'
    '                w3_weight = weights[\n'
    '                    f"{expert_id}.w3.weight"] if f"{expert_id}.w3.weight" in weights else None\n'
    '                w2_weight = weights[\n'
    '                    f"{expert_id}.w2.weight"] if f"{expert_id}.w2.weight" in weights else None',
    '                def _gw(k, fb=None): return weights[k] if k in weights else (weights[fb] if fb and fb in weights else None)\n'
    '                w1_weight = _gw(f"{expert_id}.w1.weight", f"{expert_id}.w1.weight_packed")\n'
    '                w3_weight = _gw(f"{expert_id}.w3.weight", f"{expert_id}.w3.weight_packed")\n'
    '                w2_weight = _gw(f"{expert_id}.w2.weight", f"{expert_id}.w2.weight_packed")'
)

# ---------------------------------------------------------------------------
# Patch 4b: quantization.py — load_quant_scales: accept weight_global_scale,
#            converting to 1/(6*weight_global_scale) = amax_weight/2688.
#            MoE load_expert_fc31_alpha_nvfp4 does:
#              w3_w1_weight_scale_2 = 1.0 / w1_weight_scale_2
#              fc31_alpha = 1.0 / (fc31_input_scale * w3_w1_weight_scale_2)
#            With w1_weight_scale_2 = amax_weight/2688 and
#                 fc31_input_scale = 2688/amax_input (from patch 4c),
#            we get fc31_alpha = amax_input * amax_weight / 2688^2 (correct).
# ---------------------------------------------------------------------------
print("=== Patch 4b: quantization.py (load_quant_scales weight_global_scale fallback + conversion) ===")
apply_patch(
    QUANT,
    'load_quant_scales: accept weight_global_scale with 1/v conversion',
    '                w1_weight_scale_2 = weights[f"{expert_id}.w1.weight_scale_2"]\n'
    '                w3_weight_scale_2 = weights[f"{expert_id}.w3.weight_scale_2"]\n'
    '                w2_weight_scale_2 = weights[f"{expert_id}.w2.weight_scale_2"]',
    '                import torch as _torch\n'
    '                def _ws2(n):\n'
    '                    k = f"{expert_id}.{n}.weight_scale_2"\n'
    '                    fb = f"{expert_id}.{n}.weight_global_scale"\n'
    '                    if k in weights: return weights[k]\n'
    '                    if fb not in weights: return None\n'
    '                    v = weights[fb]\n'
    '                    v32 = v.to(_torch.float32) if isinstance(v, _torch.Tensor) else _torch.tensor(float(v))\n'
    '                    return _torch.reciprocal(v32 * 6.0)  # 1/(6*wgs) = amax_weight/2688\n'
    '                w1_weight_scale_2 = _ws2("w1")\n'
    '                w3_weight_scale_2 = _ws2("w3")\n'
    '                w2_weight_scale_2 = _ws2("w2")'
)

# ---------------------------------------------------------------------------
# Patch 4c: quantization.py — load_quant_scales: accept input_global_scale,
#            converting to 1/(6*input_global_scale) = amax_input/2688.
#            MoE load_expert_fc31_alpha_nvfp4 does:
#              fc31_input_scale = 1.0 / max(input_scale_values)
#            With input_scale = amax_input/2688 → fc31_input_scale = 2688/amax_input (correct).
# ---------------------------------------------------------------------------
print("=== Patch 4c: quantization.py (load_quant_scales input_global_scale fallback + conversion) ===")
apply_patch(
    QUANT,
    'load_quant_scales: accept input_global_scale with 1/(6*v) conversion',
    '                w1_input_scale = weights[f"{expert_id}.w1.input_scale"]\n'
    '                w3_input_scale = weights[f"{expert_id}.w3.input_scale"]\n'
    '                w2_input_scale = weights[f"{expert_id}.w2.input_scale"]',
    '                import torch as _torch\n'
    '                def _isc(n):\n'
    '                    k = f"{expert_id}.{n}.input_scale"\n'
    '                    fb = f"{expert_id}.{n}.input_global_scale"\n'
    '                    if k in weights: return weights[k]\n'
    '                    if fb not in weights: return None\n'
    '                    v = weights[fb]\n'
    '                    v32 = v.to(_torch.float32) if isinstance(v, _torch.Tensor) else _torch.tensor(float(v))\n'
    '                    if v32.isnan().item() or v32.item() < 1e-6:  # dead/NaN/subnormal expert\n'
    '                        return _torch.tensor(0.0)  # contributes 0 to max; fc31_input_scale uses live experts only\n'
    '                    return _torch.reciprocal(v32 * 6.0)  # 1/(6*igs) = amax_input/2688\n'
    '                w1_input_scale = _isc("w1")\n'
    '                w3_input_scale = _isc("w3")\n'
    '                w2_input_scale = _isc("w2")'
)

# ---------------------------------------------------------------------------
# Patch 4d: quantization.py — update _isc NaN guard from == 0.0 to isnan()||<1e-6
#            Patch 4c may have already run (previous container launch) leaving the
#            old zero-only guard in place.  This patch upgrades it to also catch
#            NaN and near-zero subnormal input_global_scale values from the checkpoint
#            (layer 61 has 8 experts with NaN; layer 57 has 2 with ~6.45e-27).
# ---------------------------------------------------------------------------
print("=== Patch 4d: quantization.py (_isc NaN/subnormal guard upgrade) ===")
apply_patch(
    QUANT,
    '_isc: upgrade zero-only guard to catch NaN and near-zero subnormals',
    '                    if v32.item() == 0.0:  # dead/uncalibrated expert (amax_input=0)\n'
    '                        return _torch.tensor(0.0)  # contributes 0 to max; fc31_input_scale uses live experts only',
    '                    if v32.isnan().item() or v32.item() < 1e-6:  # dead/NaN/subnormal expert\n'
    '                        return _torch.tensor(0.0)  # contributes 0 to max; fc31_input_scale uses live experts only'
)

# ---------------------------------------------------------------------------
# Patch 5: linear.py — load_weights_vanilla: normalize keys before helper
# ---------------------------------------------------------------------------
print("=== Patch 5: linear.py (load_weights_vanilla key normalization) ===")
apply_patch(
    LINEAR,
    'NVFP4LinearMethod.load_weights_vanilla: normalize keys before helper',
    '    def load_weights_vanilla(self, module: Linear, weights: List[Dict]) -> None:\n'
    '        load_weights_vanilla_helper(module, weights)\n'
    '\n'
    '        input_scale, weight_scale, weight_scale_2, alpha = self.load_weight_scales(',
    '    def load_weights_vanilla(self, module: Linear, weights: List[Dict]) -> None:\n'
    '        # Normalize llm-compressor key names for NVFP4 (key renaming only; scale values fixed in load_weight_scales)\n'
    '        for w in weights:\n'
    '            if "weight_packed" in w and "weight" not in w: w["weight"] = w["weight_packed"]\n'
    '        load_weights_vanilla_helper(module, weights)\n'
    '\n'
    '        input_scale, weight_scale, weight_scale_2, alpha = self.load_weight_scales('
)

# ---------------------------------------------------------------------------
# Patch 6: linear.py — load_weights_fused_qkv_linear: normalize keys
# ---------------------------------------------------------------------------
print("=== Patch 6: linear.py (load_weights_fused_qkv_linear key normalization) ===")
apply_patch(
    LINEAR,
    'NVFP4LinearMethod.load_weights_fused_qkv_linear: normalize keys',
    '    def load_weights_fused_qkv_linear(self, module: Linear,\n'
    '                                      weights: List[Dict]) -> None:\n'
    '        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(',
    '    def load_weights_fused_qkv_linear(self, module: Linear,\n'
    '                                      weights: List[Dict]) -> None:\n'
    '        # Normalize llm-compressor key names (scale values fixed in load_weight_scales)\n'
    '        for w in weights:\n'
    '            if "weight_packed" in w and "weight" not in w: w["weight"] = w["weight_packed"]\n'
    '        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper('
)

# ---------------------------------------------------------------------------
# Patch 7: linear.py — load_weights_fused_gate_up_linear: normalize keys
# ---------------------------------------------------------------------------
print("=== Patch 7: linear.py (load_weights_fused_gate_up_linear key normalization) ===")
apply_patch(
    LINEAR,
    'NVFP4LinearMethod.load_weights_fused_gate_up_linear: normalize keys',
    '    def load_weights_fused_gate_up_linear(self, module: Linear,\n'
    '                                          weights: List[Dict]) -> None:\n'
    '        gate_weight, up_weight = load_weights_fused_gate_up_helper(',
    '    def load_weights_fused_gate_up_linear(self, module: Linear,\n'
    '                                          weights: List[Dict]) -> None:\n'
    '        # Normalize llm-compressor key names (scale values fixed in load_weight_scales)\n'
    '        for w in weights:\n'
    '            if "weight_packed" in w and "weight" not in w: w["weight"] = w["weight_packed"]\n'
    '        gate_weight, up_weight = load_weights_fused_gate_up_helper('
)

# ---------------------------------------------------------------------------
# Patch 8: linear.py — load_weight_scales: max input_scale instead of assert
#           (q/k/v projections each have their own input_global_scale)
# ---------------------------------------------------------------------------
print("=== Patch 8: linear.py (max input_scale for fused QKV) ===")
apply_patch(
    LINEAR,
    'NVFP4LinearMethod.load_weight_scales: max input_scale instead of assert',
    '                else:\n'
    '                    assert input_scale == w["input_scale"][\n'
    '                        ...], "The input_scale should be same for all the weights"',
    '                else:\n'
    '                    input_scale = torch.max(input_scale, w["input_scale"][...])'
)

# ---------------------------------------------------------------------------
# Patch 8b: linear.py — NVFP4LinearMethod.load_weight_scales:
#            replace weight_scale_2 equality assertion with max
#            (different experts/heads may have different weight_global_scale)
# ---------------------------------------------------------------------------
print("=== Patch 8b: linear.py (NVFP4 load_weight_scales: max weight_scale_2 instead of assert) ===")
apply_patch(
    LINEAR,
    'NVFP4LinearMethod.load_weight_scales: max weight_scale_2 instead of assert',
    '            if "weight_scale_2" in w:\n'
    '                if weight_scale_2 is None:\n'
    '                    weight_scale_2 = w["weight_scale_2"][...]\n'
    '                else:\n'
    '                    assert weight_scale_2 == w["weight_scale_2"][\n'
    '                        ...], "The weight_scale_2 should be same for all the weights"',
    '            if "weight_scale_2" in w:\n'
    '                if weight_scale_2 is None:\n'
    '                    weight_scale_2 = w["weight_scale_2"][...]\n'
    '                else:\n'
    '                    weight_scale_2 = torch.max(weight_scale_2, w["weight_scale_2"][...])'
)

# ---------------------------------------------------------------------------
# Patch 9: linear.py — W4A8NVFP4FP8LinearMethod.load_weights_vanilla:
#           normalize weight_packed->weight and add scale inversions
# ---------------------------------------------------------------------------
print("=== Patch 9: linear.py (W4A8 load_weights_vanilla key normalization) ===")
apply_patch(
    LINEAR,
    'W4A8NVFP4FP8LinearMethod.load_weights_vanilla: normalize keys + invert scales',
    '    def load_weights_vanilla(self, module: Linear, weights: List[Dict]) -> None:\n'
    '        # FIXME: this depends on the kernel internals\n'
    '        load_weights_vanilla_helper(\n'
    '            module, weights,\n'
    '            lambda w: fp4_utils.shuffle_matrix_a(w, module.epilogue_tile_m))',
    '    def load_weights_vanilla(self, module: Linear, weights: List[Dict]) -> None:\n'
    '        # Normalize llm-compressor key names; invert scale convention for W4A8\n'
    '        # input_scale must be scalar [] to match module.inv_input_scale shape\n'
    '        for w in weights:\n'
    '            if "weight_packed" in w and "weight" not in w: w["weight"] = w["weight_packed"]\n'
    '            if "input_global_scale" in w and "input_scale" not in w:\n'
    '                import torch as _t; v = w["input_global_scale"][...]\n'
    '                w["input_scale"] = _t.reciprocal(v.to(torch.float32).reshape([])) if isinstance(v, _t.Tensor) else torch.tensor(1.0/float(v))\n'
    '            if "weight_global_scale" in w and "weight_scale_2" not in w:\n'
    '                import torch as _t; v = w["weight_global_scale"][...]\n'
    '                w["weight_scale_2"] = _t.reciprocal(v.to(torch.float32).reshape([])) if isinstance(v, _t.Tensor) else torch.tensor(1.0/float(v))\n'
    '        # FIXME: this depends on the kernel internals\n'
    '        load_weights_vanilla_helper(\n'
    '            module, weights,\n'
    '            lambda w: fp4_utils.shuffle_matrix_a(w, module.epilogue_tile_m))'
)

# ---------------------------------------------------------------------------
# Patch 10: linear.py — W4A8NVFP4FP8LinearMethod.load_weights_fused_qkv_linear:
#            normalize weight_packed->weight and scale keys
# ---------------------------------------------------------------------------
print("=== Patch 10: linear.py (W4A8 load_weights_fused_qkv_linear key normalization) ===")
apply_patch(
    LINEAR,
    'W4A8NVFP4FP8LinearMethod.load_weights_fused_qkv_linear: normalize keys',
    '    def load_weights_fused_qkv_linear(self, module: Linear,\n'
    '                                      weights: List[Dict]) -> None:\n'
    '        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(\n'
    '            module, weights)',
    '    def load_weights_fused_qkv_linear(self, module: Linear,\n'
    '                                      weights: List[Dict]) -> None:\n'
    '        for w in weights:\n'
    '            if "weight_packed" in w and "weight" not in w: w["weight"] = w["weight_packed"]\n'
    '            if "input_global_scale" in w and "input_scale" not in w:\n'
    '                import torch as _t; v = w["input_global_scale"][...]\n'
    '                w["input_scale"] = _t.reciprocal(v.to(torch.float32).reshape([])) if isinstance(v, _t.Tensor) else torch.tensor(1.0/float(v))\n'
    '            if "weight_global_scale" in w and "weight_scale_2" not in w:\n'
    '                import torch as _t; v = w["weight_global_scale"][...]\n'
    '                w["weight_scale_2"] = _t.reciprocal(v.to(torch.float32).reshape([])) if isinstance(v, _t.Tensor) else torch.tensor(1.0/float(v))\n'
    '        q_weight, k_weight, v_weight = load_weights_fused_qkv_helper(\n'
    '            module, weights)'
)

# ---------------------------------------------------------------------------
# Patch 11: linear.py — W4A8NVFP4FP8LinearMethod.load_weights_fused_gate_up_linear:
#            normalize keys
# ---------------------------------------------------------------------------
print("=== Patch 11: linear.py (W4A8 load_weights_fused_gate_up_linear key normalization) ===")
apply_patch(
    LINEAR,
    'W4A8NVFP4FP8LinearMethod.load_weights_fused_gate_up_linear: normalize keys',
    '    def load_weights_fused_gate_up_linear(self, module: Linear,\n'
    '                                          weights: List[Dict]) -> None:\n'
    '        gate_weight, up_weight = load_weights_fused_gate_up_helper(\n'
    '            module, weights)\n'
    '        fused_weight = torch.cat((gate_weight, up_weight))\n'
    '        fused_weight = fp4_utils.shuffle_matrix_a(fused_weight,\n'
    '                                                  module.epilogue_tile_m)\n'
    '        copy_weight(module.weight, fused_weight)\n'
    '\n'
    '        input_scale, weight_scales, weight_scale_2, alpha = self.load_weight_scales(',
    '    def load_weights_fused_gate_up_linear(self, module: Linear,\n'
    '                                          weights: List[Dict]) -> None:\n'
    '        for w in weights:\n'
    '            if "weight_packed" in w and "weight" not in w: w["weight"] = w["weight_packed"]\n'
    '            if "input_global_scale" in w and "input_scale" not in w:\n'
    '                import torch as _t; v = w["input_global_scale"][...]\n'
    '                w["input_scale"] = _t.reciprocal(v.to(torch.float32).reshape([])) if isinstance(v, _t.Tensor) else torch.tensor(1.0/float(v))\n'
    '            if "weight_global_scale" in w and "weight_scale_2" not in w:\n'
    '                import torch as _t; v = w["weight_global_scale"][...]\n'
    '                w["weight_scale_2"] = _t.reciprocal(v.to(torch.float32).reshape([])) if isinstance(v, _t.Tensor) else torch.tensor(1.0/float(v))\n'
    '        gate_weight, up_weight = load_weights_fused_gate_up_helper(\n'
    '            module, weights)\n'
    '        fused_weight = torch.cat((gate_weight, up_weight))\n'
    '        fused_weight = fp4_utils.shuffle_matrix_a(fused_weight,\n'
    '                                                  module.epilogue_tile_m)\n'
    '        copy_weight(module.weight, fused_weight)\n'
    '\n'
    '        input_scale, weight_scales, weight_scale_2, alpha = self.load_weight_scales('
)

# ---------------------------------------------------------------------------
# Patch 12: linear.py — W4A8NVFP4FP8LinearMethod.load_weight_scales:
#            replace equality assertions with max (fused QKV has different per-head scales)
# ---------------------------------------------------------------------------
print("=== Patch 12: linear.py (W4A8 load_weight_scales: max instead of assert for input_scale) ===")
apply_patch(
    LINEAR,
    'W4A8NVFP4FP8LinearMethod.load_weight_scales: max input_scale instead of assert',
    '            if "input_scale" in w:\n'
    '                if input_scale is None:\n'
    '                    input_scale = w["input_scale"][...]\n'
    '                else:\n'
    '                    assert input_scale == w["input_scale"][\n'
    '                        ...], "The input_scale should be same for all the weights"',
    '            if "input_scale" in w:\n'
    '                if input_scale is None:\n'
    '                    input_scale = w["input_scale"][...]\n'
    '                else:\n'
    '                    input_scale = torch.max(input_scale, w["input_scale"][...])'
)

# ---------------------------------------------------------------------------
# Patch 12b: linear.py — W4A8NVFP4FP8LinearMethod.load_weight_scales:
#             replace weight_scale_2 equality assertion with max too
# ---------------------------------------------------------------------------
print("=== Patch 12b: linear.py (W4A8 load_weight_scales: max instead of assert for weight_scale_2) ===")
apply_patch(
    LINEAR,
    'W4A8NVFP4FP8LinearMethod.load_weight_scales: max weight_scale_2 instead of assert',
    '            if "weight_scale_2" in w:\n'
    '                if weight_scale_2 is None:\n'
    '                    weight_scale_2 = w["weight_scale_2"][...]\n'
    '                else:\n'
    '                    assert weight_scale_2 == w["weight_scale_2"][...], (\n'
    '                        f"The weight_scale_2 should be same for all the weights: {weight_scale_2} vs. {w[\'weight_scale_2\']}*6"\n'
    '                    )',
    '            if "weight_scale_2" in w:\n'
    '                if weight_scale_2 is None:\n'
    '                    weight_scale_2 = w["weight_scale_2"][...]\n'
    '                else:\n'
    '                    weight_scale_2 = torch.max(weight_scale_2, w["weight_scale_2"][...])'
)

# ---------------------------------------------------------------------------
# Patch 13: linear.py — W4A8NVFP4FP8LinearMethod.load_weight_scales:
#            downsample group_size=16 weight_scale to group_size=32 by max-pooling pairs.
#            The fp4_fp8_gemm_trtllmgen kernel requires scaling_vector_size=32 (cannot change).
#            We keep scaling_vector_size=32 in create_weights and merge adjacent FP8 scales.
# ---------------------------------------------------------------------------
print("=== Patch 13: linear.py (W4A8 load_weight_scales: downsample 16->32 group scales) ===")
apply_patch(
    LINEAR,
    'W4A8NVFP4FP8LinearMethod.load_weight_scales: merge gs16->gs32 by max-pooling',
    '            if "weight_scale" in w:\n'
    '                ws = load_weight_shard(w["weight_scale"],\n'
    '                                       tp_size,\n'
    '                                       tp_rank,\n'
    '                                       tp_mode,\n'
    '                                       device=device).contiguous()\n'
    '                assert ws.dtype == torch.float8_e4m3fn\n'
    '                weight_scale.append(ws.view(dtype=fp4_utils.float4_sf_dtype))',
    '            if "weight_scale" in w:\n'
    '                ws = load_weight_shard(w["weight_scale"],\n'
    '                                       tp_size,\n'
    '                                       tp_rank,\n'
    '                                       tp_mode,\n'
    '                                       device=device).contiguous()\n'
    '                assert ws.dtype == torch.float8_e4m3fn\n'
    '                # Downsample from group_size=16 to group_size=32 (kernel requires 32)\n'
    '                # by max-pooling adjacent FP8 scale pairs (conservative, slight accuracy loss)\n'
    '                if ws.dim() == 2 and ws.shape[-1] % 2 == 0:\n'
    '                    M, K16 = ws.shape\n'
    '                    ws = ws.to(torch.float32).reshape(M, K16 // 2, 2).max(dim=2).values.to(torch.float8_e4m3fn).contiguous()\n'
    '                elif ws.dim() == 1 and ws.numel() % 2 == 0 and "weight_packed" in w:\n'
    '                    # Flat shape: infer from weight_packed\n'
    '                    M = w["weight_packed"].shape[0] if hasattr(w["weight_packed"], "shape") else ws.numel()\n'
    '                    K16 = ws.numel() // M\n'
    '                    if K16 % 2 == 0:\n'
    '                        ws = ws.reshape(M, K16).to(torch.float32).reshape(M, K16 // 2, 2).max(dim=2).values.to(torch.float8_e4m3fn).contiguous()\n'
    '                weight_scale.append(ws.view(dtype=fp4_utils.float4_sf_dtype))'
)

# ---------------------------------------------------------------------------
# Patch 14: quantization.py — relax w1==w3 weight_scale_2 assertion in MoE
#            (llm-compressor stores distinct weight_global_scale per expert gate)
# ---------------------------------------------------------------------------
print("=== Patch 14: quantization.py (relax w1==w3 weight_scale_2 assertion) ===")
apply_patch(
    QUANT,
    'load_expert_fc31_alpha_nvfp4: relax w1==w3 assertion to allow different values',
    '        assert torch.allclose(\n'
    '            w1_weight_scale_2,\n'
    '            w3_weight_scale_2), "w1_weight_scale_2 != w3_weight_scale_2"\n'
    '\n'
    '        w3_w1_weight_scale_2 = 1.0 / w1_weight_scale_2',
    '        # llm-compressor stores distinct weight_global_scale per gate; use max\n'
    '        if not torch.allclose(w1_weight_scale_2, w3_weight_scale_2):\n'
    '            w1_weight_scale_2 = torch.max(w1_weight_scale_2, w3_weight_scale_2)\n'
    '\n'
    '        w3_w1_weight_scale_2 = 1.0 / w1_weight_scale_2'
)

# ---------------------------------------------------------------------------
# Patch 16: quantization.py — load_expert_fc31_input_scale_nvfp4:
#            relax w1_input_scale == w3_input_scale assertion.
#            llm-compressor stores distinct input_global_scale per gate (w1 vs w3).
#            If the assertion fires, dst_fc31_input_scale stays at 1.0 (tensor(1.)),
#            which causes fp4_quantize to clip everything to FP4 max → NaN.
#            Fix: use max(w1_input_scale, w3_input_scale) instead.
# ---------------------------------------------------------------------------
print("=== Patch 16: quantization.py (relax w1==w3 input_scale assertion in MoE) ===")
apply_patch(
    QUANT,
    'load_expert_fc31_input_scale_nvfp4: relax w1==w3 assertion for input_scale',
    '        assert torch.allclose(\n'
    '            w1_input_scale, w3_input_scale), "w1_input_scale != w3_input_scale"\n'
    '        dst_fc31_input_scale.copy_(w1_input_scale)',
    '        # llm-compressor stores distinct input_global_scale per gate (w1 vs w3); use max\n'
    '        combined_input_scale = torch.max(w1_input_scale, w3_input_scale)\n'
    '        dst_fc31_input_scale.copy_(combined_input_scale)'
)

# ---------------------------------------------------------------------------
# Patch 15: linear.py — NVFP4LinearMethod._input_prepare: debug hook
#            Print alpha, input_scale, and output magnitude on first call.
# ---------------------------------------------------------------------------
print("=== Patch 15: linear.py (NVFP4LinearMethod debug hook) ===")
apply_patch(
    LINEAR,
    'NVFP4LinearMethod._input_prepare: debug hook for scale values',
    '            # Dynamic vs static quantization\n'
    '            if module.input_scale is None or module.force_dynamic_quantization:\n'
    '                # Dynamic mode: compute input_scale and alpha from current input\n'
    '                FP8_MAX, E2M1_MAX = 448.0, 6.0',
    '            # DEBUG: print first-call scale values\n'
    '            if not getattr(module, "_debug_printed", False):\n'
    '                module._debug_printed = True\n'
    '                a = module.alpha.item() if module.alpha is not None else "None"\n'
    '                ws2 = module.weight_scale_2.item() if module.weight_scale_2 is not None else "None"\n'
    '                isc = module.input_scale.item() if module.input_scale is not None else "None"\n'
    '                print(f"[NVFP4_DBG] input_scale={isc} alpha={a} ws2={ws2} in_mag={input.abs().mean().item():.4g}", flush=True)\n'
    '            # Dynamic vs static quantization\n'
    '            if module.input_scale is None or module.force_dynamic_quantization:\n'
    '                # Dynamic mode: compute input_scale and alpha from current input\n'
    '                FP8_MAX, E2M1_MAX = 448.0, 6.0'
)

# ---------------------------------------------------------------------------
# Patch 17: fused_moe_cutlass.py — quantize_input: MoE debug hook
#            Print scale values and check for NaN before/after fp4_quantize
#            and before/after fused_moe kernel on first call.
# ---------------------------------------------------------------------------
print("=== Patch 17: fused_moe_cutlass.py (MoE NVFP4 debug hook) ===")
MOE_CUTLASS_FWD = '/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cutlass.py'
apply_patch(
    MOE_CUTLASS_FWD,
    'MoE NVFP4 quantize_input: debug hook',
    '            elif self.has_nvfp4:\n'
    '                if hasattr(\n'
    '                        self,\n'
    '                        \'fc31_act_scale\') and self.fc31_act_scale is not None:\n'
    '                    assert not isinstance(\n'
    '                        x, Fp4QuantizedTensor\n'
    '                    ), "Fp4QuantizedTensor is not expected for AWQ quantization."\n'
    '                    x = x * self.fc31_act_scale',
    '            elif self.has_nvfp4:\n'
    '                # DEBUG: MoE NVFP4 scale check on first call\n'
    '                if not getattr(self, "_moe_dbg_printed", False):\n'
    '                    self._moe_dbg_printed = True\n'
    '                    isc = self.fc31_input_scale.item()\n'
    '                    al = self.fc31_alpha.float() if hasattr(self.fc31_alpha, "float") else self.fc31_alpha\n'
    '                    al_min, al_max = (al.min().item(), al.max().item()) if hasattr(al, "min") else (al, al)\n'
    '                    fc2_isc = self.fc2_input_scale.item()\n'
    '                    print(f"[MOE_DBG] fc31_input_scale={isc:.4g}  fc31_alpha min={al_min:.4g} max={al_max:.4g}  fc2_input_scale={fc2_isc:.4g}  x_mag={x.float().abs().mean().item():.4g}  x_has_nan={x.isnan().any().item()}", flush=True)\n'
    '                if hasattr(\n'
    '                        self,\n'
    '                        \'fc31_act_scale\') and self.fc31_act_scale is not None:\n'
    '                    assert not isinstance(\n'
    '                        x, Fp4QuantizedTensor\n'
    '                    ), "Fp4QuantizedTensor is not expected for AWQ quantization."\n'
    '                    x = x * self.fc31_act_scale'
)

# ---------------------------------------------------------------------------
# Patch 18: modeling_minimaxm2.py — MiniMaxM2MoE.load_weights:
#            Also load expert weights from the snapshot weight dict.
#
# Root cause: _load_weights_impl uses ConsumableWeightsDict + run_concurrently
# (ThreadPoolExecutor).  After MiniMaxM2MoE.load_weights returns, the caller
# calls weights.mark_consumed('model.layers.N.block_sparse_moe'), which
# DELETES every key beginning with 'model.layers.N.block_sparse_moe.' —
# including all expert weight keys.  A concurrent thread that was assigned
# 'model.layers.N.block_sparse_moe.experts' may not have run yet; when it
# does, filter_weights returns an empty dict → CutlassFusedMoE.load_weights
# is skipped → fc31_input_scale / fc31_alpha stay at initialisation values
# (1.0), producing wildly wrong MoE output.
#
# Fix: inside MiniMaxM2MoE.load_weights, extract the 'experts.*' sub-dict
# from the snapshot and call self.experts.load_weights() directly.  The
# snapshot is a plain dict captured before mark_consumed runs, so it is
# unaffected by concurrent deletions.  If the concurrent thread happened to
# run first (rare), CutlassFusedMoE._weights_loaded is True and we skip,
# avoiding a redundant second load.
# ---------------------------------------------------------------------------
print("=== Patch 18: modeling_minimaxm2.py (MiniMaxM2MoE: load expert weights to avoid race) ===")
apply_patch(
    MINIMAXM2,
    'MiniMaxM2MoE.load_weights: load expert weights from snapshot to avoid ConsumableWeightsDict race',
    '    def load_weights(self, weights: List[Dict]):\n'
    '        assert len(weights) == 1\n'
    '\n'
    '        self.e_score_correction_bias.copy_(\n'
    '            weights[0]["e_score_correction_bias"][:].to(self.e_score_correction_bias.dtype)\n'
    '        )',
    '    def load_weights(self, weights: List[Dict]):\n'
    '        assert len(weights) == 1\n'
    '\n'
    '        self.e_score_correction_bias.copy_(\n'
    '            weights[0]["e_score_correction_bias"][:].to(self.e_score_correction_bias.dtype)\n'
    '        )\n'
    '        # Load expert weights while we still have the snapshot.\n'
    '        # ConsumableWeightsDict.mark_consumed() is called by the weight-loading\n'
    '        # loop AFTER this method returns and deletes all keys under\n'
    '        # \'block_sparse_moe.\', so the concurrent thread for\n'
    '        # \'block_sparse_moe.experts\' often receives an empty dict.\n'
    '        if not getattr(self.experts, "_weights_loaded_by_parent", False):\n'
    '            _pfx = "experts."\n'
    '            _ew = {k[len(_pfx):]: v for k, v in weights[0].items()\n'
    '                   if k.startswith(_pfx)}\n'
    '            if _ew:\n'
    '                self.experts.load_weights(weights=[_ew])\n'
    '                self.experts._weights_loaded_by_parent = True'
)

# ---------------------------------------------------------------------------
# Patch 19: sampling_utils_flashinfer.py — add NaN debug print before assert
#            When logits contain NaN, print shape/stats then fire the assert.
#            This lets us distinguish "logits are NaN" vs "something else".
# ---------------------------------------------------------------------------
print("=== Patch 19: sampling_utils_flashinfer.py (NaN debug print before assert) ===")
SAMPLING_FI = '/usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/pyexecutor/sampling_utils_flashinfer.py'
apply_patch(
    SAMPLING_FI,
    'sampling_utils_flashinfer: print logit stats before NaN assert',
    '            torch._assert_async(~torch.any(torch.isnan(inputs)))\n'
    '\n'
    '            return False',
    '            _has_nan = torch.any(torch.isnan(inputs))\n'
    '            if _has_nan:\n'
    '                print(f"[NAN_DBG] logits have NaN! shape={inputs.shape} dtype={inputs.dtype} '
    'min={inputs.float().min().item():.4g} max={inputs.float().max().item():.4g} '
    'nan_frac={inputs.isnan().float().mean().item():.4g}", flush=True)\n'
    '            torch._assert_async(~_has_nan)\n'
    '\n'
    '            return False'
)

# ---------------------------------------------------------------------------
# Patch 20: modeling_minimaxm2.py — MiniMaxM2Model.forward: add NaN check
#            after each decoder layer to identify the first NaN-producing layer.
# ---------------------------------------------------------------------------
print("=== Patch 20: modeling_minimaxm2.py (NaN check per decoder layer in forward) ===")
apply_patch(
    MINIMAXM2,
    'MiniMaxM2Model.forward: check for NaN after each decoder layer',
    '        for decoder_layer in self.layers:\n'
    '            hidden_states, residual = decoder_layer(\n'
    '                position_ids=position_ids,\n'
    '                hidden_states=hidden_states,\n'
    '                attn_metadata=attn_metadata,\n'
    '                residual=residual,\n'
    '            )',
    '        for _layer_idx, decoder_layer in enumerate(self.layers):\n'
    '            hidden_states, residual = decoder_layer(\n'
    '                position_ids=position_ids,\n'
    '                hidden_states=hidden_states,\n'
    '                attn_metadata=attn_metadata,\n'
    '                residual=residual,\n'
    '            )\n'
    '            if not getattr(self, "_nan_layer_found", False):\n'
    '                _hs_nan = hidden_states.isnan().any().item()\n'
    '                _res_nan = (residual.isnan().any().item() if residual is not None else False)\n'
    '                if _hs_nan or _res_nan:\n'
    '                    self._nan_layer_found = True\n'
    '                    print(f"[NAN_LAYER] First NaN at layer {_layer_idx}: hs_nan={_hs_nan} res_nan={_res_nan} hs_mag={hidden_states.float().abs().mean().item():.4g}", flush=True)'
)

print("\n=== All patches applied successfully! ===")
