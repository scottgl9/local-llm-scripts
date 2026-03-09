# Triton MoE Kernel Autotuning for GB10

## Background

vLLM uses Triton fused MoE (Mixture of Experts) kernels for models like Qwen3.5-35B-A3B-NVFP4. These kernels have tunable parameters (block sizes, warps, stages, etc.) that significantly affect performance. vLLM ships pre-tuned configs for common GPU/model combinations, but **no config existed for the GB10 (SM121) + E=256,N=512 combination**.

Without a tuned config, vLLM falls back to default kernel parameters, which may be far from optimal for the specific GPU architecture.

### Model MoE dimensions (Qwen3.5-35B-A3B-NVFP4)

| Parameter | Value | Config key |
|-----------|-------|------------|
| Experts | 256 | `num_experts` |
| Expert intermediate size | 512 | `moe_intermediate_size` |
| Experts per token (topk) | 8 | `num_experts_per_tok` |
| Quantization | FP8 W8A8 | `compressed-tensors` |

## Quick Start (run overnight)

**Before running:** Make sure the GPU is free — kill any running vLLM server first.

```bash
# Kill any running vLLM processes
pkill -9 -f "vllm.entrypoints" 2>/dev/null; pkill -9 -x "VLLM::EngineCore" 2>/dev/null
sleep 3

# Activate venv and run
cd ~/sandbox/vllm
source .venv-gb10/bin/activate

# Full tune (all 18 batch sizes, ~8-10 hours)
nohup python benchmarks/kernels/benchmark_moe.py \
    --model Sehyo/Qwen3.5-35B-A3B-NVFP4 \
    --dtype fp8_w8a8 \
    --tp-size 1 \
    --tune \
    --trust-remote-code \
    --save-dir vllm/model_executor/layers/fused_moe/configs/ \
    > /tmp/moe-autotune.log 2>&1 &

echo "PID: $!"
echo "Log: /tmp/moe-autotune.log"
echo "Output: vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json"
```

### Faster alternative (decode-relevant batch sizes only, ~4-5 hours)

For single-user decode (max_num_seqs=8, topk=8 -> max 64 tokens per MoE call):

```bash
pkill -9 -f "vllm.entrypoints" 2>/dev/null; pkill -9 -x "VLLM::EngineCore" 2>/dev/null
sleep 3

cd ~/sandbox/vllm
source .venv-gb10/bin/activate

nohup python benchmarks/kernels/benchmark_moe.py \
    --model Sehyo/Qwen3.5-35B-A3B-NVFP4 \
    --dtype fp8_w8a8 \
    --tp-size 1 \
    --tune \
    --trust-remote-code \
    --batch-size 1 2 4 8 16 24 32 48 64 \
    --save-dir vllm/model_executor/layers/fused_moe/configs/ \
    > /tmp/moe-autotune.log 2>&1 &

echo "PID: $!"
```

### Check progress

```bash
# Quick progress check (shows current pass progress)
tail -1 /tmp/ray/session_latest/logs/worker-*.out 2>/dev/null \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{d[\"x\"]}/{d[\"total\"]} ({100*d[\"x\"]/d[\"total\"]:.1f}%)')"

# Check which pass (batch size) it's on
python3 << 'EOF'
import json, glob
for f in sorted(glob.glob("/tmp/ray/session_latest/logs/worker-*.out")):
    uuids = []
    with open(f) as fh:
        for line in fh:
            try:
                d = json.loads(line)
                u = d.get("uuid", "")[:8]
                if not uuids or uuids[-1] != u:
                    uuids.append(u)
            except: pass
    if uuids:
        print(f"Pass {len(uuids)} of 18 (or 9 if using --batch-size)")
        break
EOF

# Check if the process is still running
pgrep -f benchmark_moe && echo "Still running" || echo "Finished (or crashed — check log)"

# Check log for errors
tail -20 /tmp/moe-autotune.log
```

### Check if it finished successfully

```bash
ls -la ~/sandbox/vllm/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json
```

If the file exists, autotuning completed. You can inspect it:
```bash
python3 -m json.tool ~/sandbox/vllm/vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json
```

## How it works

1. Loads the model config to extract MoE dimensions (E, N, topk, hidden_size)
2. Generates a search space of **1920 Triton kernel configurations** per batch size:
   - Block sizes (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
   - Number of warps
   - Number of pipeline stages
   - Group sizes
3. For each batch size, benchmarks all 1920 configs and picks the fastest
4. Saves the best config per batch size to a JSON file

### Default batch sizes (18 total)

1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096

Total: 18 x 1920 = **34,560 kernel evaluations**.

### Resource usage

- ~4-5 GB GPU memory (via Ray worker)
- 1 CPU core for the Ray driver + 1 for the worker
- GPU must be free of other workloads

## Prerequisite: Code fix

The `benchmark_moe.py` script didn't recognize `Qwen3_5MoeForConditionalGeneration`. This was fixed by adding it to `get_model_params()` in `benchmarks/kernels/benchmark_moe.py` (~line 782):

```python
elif config.architectures[0] in (
    "Qwen3VLMoeForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",  # <-- added
):
    text_config = config.get_text_config()
    E = text_config.num_experts
    topk = text_config.num_experts_per_tok
    intermediate_size = text_config.moe_intermediate_size
    hidden_size = text_config.hidden_size
```

This fix is already applied on the `gb10-spark-main-20260305` branch.

## Output file

Location:
```
vllm/model_executor/layers/fused_moe/configs/E=256,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json
```

Format — JSON mapping batch sizes to optimal kernel configs:
```json
{
  "1": {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "num_stages": 4
  },
  "2": { ... },
  ...
}
```

## How vLLM uses the config at runtime

`vllm/model_executor/layers/fused_moe/fused_moe.py` looks up the config file matching `(E, N, device_name, dtype)`. The naming convention is:
```
E={E},N={N},device_name={GPU_NAME},dtype={dtype}.json
```

If no matching config exists, vLLM uses hardcoded defaults. The tuned config is picked up automatically — no code changes or flags needed, just restart the server.

## Testing the tuned config

After the config file is created, restart vLLM and run the speed test:

```bash
# Start server
cd ~/sandbox/vllm
bash vllm.sh Qwen3.5-35B-NVFP4

# In another terminal, once server is ready:
python3 ~/llm_speed_test.py --runs 8 --warmup 2 --max-tokens 512 --skip-tests
```

Compare tps_decode p50 against the baseline (44.83 tps without tuned config).

## Existing configs in the repo

```bash
ls vllm/model_executor/layers/fused_moe/configs/ | grep "E=256,N=512"
```

As of 2026-03-06, configs exist for H100, B200, MI300X, MI325 — but not GB10.

---

## Applicability to Other Models

### Does the 35B tune (E=256,N=512) apply to 122B or MiniMax?

**No.** The config filename encodes `E` and `N` exactly. A config tuned for one
`(E, N)` pair is never used for another. Each model needs its own tune.

### Model MoE dimensions comparison

| Model | Architecture | E (experts) | N (intermediate) | topk | Config needed |
|-------|-------------|-------------|-----------------|------|--------------|
| **Qwen3.5-35B-A3B** | Qwen3_5MoeForConditionalGeneration | 256 | 512 | 8 | `E=256,N=512,...` ← existing plan |
| **Qwen3.5-122B-A10B** | Qwen3_5MoeForConditionalGeneration | 256 | **1024** | 8 | `E=256,N=1024,...` |
| **MiniMax-M2.5-139B** | MiniMaxM2ForCausalLM | **154** | **1536** | 8 | `E=154,N=1536,...` |

---

## Qwen3.5-122B-A10B Tune

**Config file:** `E=256,N=1024,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json`

Same architecture as 35B — `benchmark_moe.py` already supports it via the
`Qwen3_5MoeForConditionalGeneration` branch. Just point at the 122B model.

The tune applies to the MTP MoE layer after `VLLM_MTP_MOE_FP8=1` post-quantization
(bf16 → fp8_w8a8). The main model MoE uses Marlin NVFP4, not Triton.

```bash
pkill -9 -f "vllm.entrypoints" 2>/dev/null; pkill -9 -x "VLLM::EngineCore" 2>/dev/null
sleep 3

cd ~/sandbox/vllm
source .venv-gb10/bin/activate

nohup python benchmarks/kernels/benchmark_moe.py \
    --model Sehyo/Qwen3.5-122B-A10B-NVFP4 \
    --dtype fp8_w8a8 \
    --tp-size 1 \
    --tune \
    --trust-remote-code \
    --batch-size 1 2 4 8 16 24 32 48 64 \
    --save-dir vllm/model_executor/layers/fused_moe/configs/ \
    > /tmp/moe-autotune-122b.log 2>&1 &

echo "PID: $!"
echo "Output: vllm/model_executor/layers/fused_moe/configs/E=256,N=1024,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json"
```

**Expected benefit:** The 122B MTP MoE layer is 3× larger than 35B's (N=1024 vs 512),
and is already the dominant cost in the MTP forward pass. A tuned Triton config
compounds the `VLLM_MTP_MOE_FP8=1` bandwidth reduction with better compute efficiency.

---

## MiniMax-M2.5-REAP-139B-A10B Tune

**Config file:** `E=154,N=1536,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json`

**Model key facts (from `saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10`):**

| Field | Value |
|-------|-------|
| Architecture | `MiniMaxM2ForCausalLM` |
| `num_local_experts` | 154 |
| `intermediate_size` | 1536 |
| `num_experts_per_tok` | 8 |
| `hidden_size` | 3072 |
| `num_hidden_layers` | 62 |
| `mtp_transformer_layers` / `num_mtp_modules` | 3 (vs 1 for Qwen3.5) |
| `shared_intermediate_size` | 0 (no shared expert) |
| Main model quantization | NVFP4 (compressed-tensors) |

**MTP has 3 layers** — `VLLM_MTP_MOE_FP8=1` would quantize all 3, tripling the
bandwidth savings vs a single-layer MTP model.

### benchmark_moe.py support for MiniMaxM2ForCausalLM

`MiniMaxM2ForCausalLM` is not explicitly handled in `get_model_params()`. It falls
into the `else` branch (line 806), which calls `config.get_text_config()` then reads
`num_local_experts`, `num_experts_per_tok`, `intermediate_size`, `hidden_size` — all
of which exist at the top level of the MiniMax config. However, if `get_text_config()`
raises (some flat configs do not implement it), the script will crash.

**Recommended fix:** Add an explicit case to `get_model_params()` in
`benchmarks/kernels/benchmark_moe.py`:

```python
elif config.architectures[0] == "MiniMaxM2ForCausalLM":
    E = config.num_local_experts
    topk = config.num_experts_per_tok
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
```

Add this before the `else` branch. Then run:

```bash
pkill -9 -f "vllm.entrypoints" 2>/dev/null; pkill -9 -x "VLLM::EngineCore" 2>/dev/null
sleep 3

cd ~/sandbox/vllm
source .venv-gb10/bin/activate

MINIMAX_SNAP="$HOME/.cache/huggingface/hub/models--saricles--MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10/snapshots/bfdccfb01a260ccbbb93581600ad1c65ac0dfea0"

nohup python benchmarks/kernels/benchmark_moe.py \
    --model "${MINIMAX_SNAP}" \
    --dtype fp8_w8a8 \
    --tp-size 1 \
    --tune \
    --trust-remote-code \
    --batch-size 1 2 4 8 16 24 32 48 64 \
    --save-dir vllm/model_executor/layers/fused_moe/configs/ \
    > /tmp/moe-autotune-minimax.log 2>&1 &

echo "PID: $!"
echo "Output: vllm/model_executor/layers/fused_moe/configs/E=154,N=1536,device_name=NVIDIA_GB10,dtype=fp8_w8a8.json"
```

### MiniMax VLLM_MTP_MOE_FP8=1 applicability

`quantize_mtp_moe_fp8()` looks for `UnquantizedFusedMoEMethod` with bf16 weights.
MiniMax uses a different MoE layer class (`block_sparse_moe`). Whether
`VLLM_MTP_MOE_FP8=1` applies needs to be verified at runtime by checking whether
the MTP layers are detected and quantized (the log line
`quantize_mtp_moe_fp8: ... num_experts=154` will appear if it works).

### Order of tunes (recommended)

Run one at a time — each occupies the full GPU:
1. **122B tune first** (`E=256,N=1024`) — highest priority, no code changes needed
2. **MiniMax tune second** (`E=154,N=1536`) — after adding the `MiniMaxM2ForCausalLM` code fix above
3. **35B tune third** (`E=256,N=512`) — lowest priority; 35B is already well-optimized
