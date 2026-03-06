# DEBUG — Active Issues

## Container & Patch Setup

**Container name:** `qwen35-122b-server`
**Image:** `avarok/dgx-vllm-nvfp4-kernel:v23`
**Run script:** `servers/qwen35-122b-a10b-nvfp4/run-v23.sh`
**vLLM path inside container:** `/app/vllm/vllm/`

### How patches work

`patches/build.sh v23` applies each `.patch` file (unified diff) against its corresponding original
source file in `patches/vllm/v23/`, writing the patched output to `.build/v23/`. The run script
volume-mounts those built files into the container at startup (`:ro`).

Patched files for v23:

| Patch | Container path |
|-------|---------------|
| `entrypoints/chat_utils.patch` | `/app/vllm/vllm/entrypoints/chat_utils.py` |
| `entrypoints/openai/chat_completion/serving.patch` | `/app/vllm/vllm/entrypoints/openai/chat_completion/serving.py` |
| `model_executor/layers/quantization/modelopt.patch` | `/app/vllm/vllm/model_executor/layers/quantization/modelopt.py` |
| `model_executor/models/qwen3_5_mtp.patch` | `/app/vllm/vllm/model_executor/models/qwen3_5_mtp.py` |
| `model_executor/models/qwen3_5.patch` | `/app/vllm/vllm/model_executor/models/qwen3_5.py` |
| `quantization/compressed_tensors/compressed_tensors_moe.patch` | `/app/vllm/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` |
| `reasoning/qwen3_reasoning_parser.patch` | `/app/vllm/vllm/reasoning/qwen3_reasoning_parser.py` |
| `tool_parsers/qwen3coder_tool_parser.patch` | `/app/vllm/vllm/tool_parsers/qwen3coder_tool_parser.py` |

### Useful commands

```bash
# Start / restart container (rebuilds patched files if missing)
bash servers/qwen35-122b-a10b-nvfp4/run-v23.sh

# Tail live logs
docker logs -f qwen35-122b-server

# Rebuild all patched files manually
bash patches/build.sh v23

# Copy a file into the running container (bypasses volume mount for hot-patching)
docker cp .build/v23/tool_parsers/qwen3coder_tool_parser.py \
  qwen35-122b-server:/app/vllm/vllm/tool_parsers/qwen3coder_tool_parser.py
```

---

## Issue 1: OOM During Profiling Step (FIXED)

**First observed:** 2026-03-06
**Status:** Fixed in run-v23.sh

### Symptom

APIServer exits with:
```
RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

EngineCore process (`pid=138`) logs cut off immediately after the `fused_moe` config warning — no
profiling, no CUDA graph capture, no explicit exception printed.

### Root Cause

**Linux OOM killer killed the EngineCore process.** Confirmed via `dmesg`:
```
oom-kill: task=VLLM::EngineCor, pid=18448, total-vm:320248764kB (~305 GiB)
Out of memory: Killed process 18448 (VLLM::EngineCor)
```

The DGX Spark GB10 has 128 GiB unified (CPU+GPU) memory. The model uses 76 GiB. After torch.compile,
vLLM runs a profiling step (`profile_run()` in `gpu_model_runner.py`) that executes a dummy forward
pass with `max_num_batched_tokens` tokens to measure peak activation memory.

With `--no-enable-chunked-prefill`, `max_num_batched_tokens` defaults to `max_model_len` (131072).
A forward pass with 131072 tokens on a 76 GiB model overflows the remaining ~52 GiB of unified
memory, causing the kernel OOM killer to fire.

**Why `Failed core proc(s): {}` is empty:** The process was killed externally by SIGKILL from the
OOM killer, not by a Python exception. The parent process detected the child died but couldn't
retrieve an error message.

### Timeline from logs

| Time | Event |
|------|-------|
| 01:59:33 | Main model loading begins |
| 02:07:44 | Main model loaded (486 s, 76.0 GiB) |
| 02:07:49 | Drafter (MTP) model loading begins |
| 02:09:02 | Drafter loaded (72 s) |
| 02:09:15 | torch.compile backbone starts |
| 02:32:55 | torch.compile backbone done (1419 s) |
| 02:33:05 | torch.compile eagle_head starts |
| 02:36:08 | torch.compile eagle_head done (177 s) |
| 02:36:11 | AOT compiled function saved |
| 02:36:15 | **WARNING** `fused_moe.py:1089` — missing MoE config for `NVIDIA_GB10` |
| 02:36:15 | EngineCore logs stop — OOM kill occurs during `profile_run()` |
| — | APIServer crashes with the generic `RuntimeError` above |

### Fix

Changed `--no-enable-chunked-prefill` to `--enable-chunked-prefill --max-num-batched-tokens 8192`
in `run-v23.sh`. This limits the profiling dummy forward pass to 8192 tokens instead of 131072,
fitting within the remaining ~52 GiB of unified memory.

Also added compiler cache volume mounts and thread-limiting env vars (matching the working
qwen3-coder-next-fp8 config) for faster restarts and lower memory pressure during compilation.

### Key vLLM internals (for future reference)

- `profile_run()` → `_dummy_run(self.max_num_tokens, is_profile=True)`
- `self.max_num_tokens` = `scheduler_config.max_num_batched_tokens`
- With `--no-enable-chunked-prefill`: `max_num_batched_tokens = max_model_len`
- With `--enable-chunked-prefill`: `max_num_batched_tokens` can be set independently
- The missing MoE config `E=256,N=1024,device_name=NVIDIA_GB10.json` is a performance-only
  warning — it causes sub-optimal kernel selection, not a crash

### How to diagnose similar OOM crashes in the future

1. Check `sudo dmesg | grep -i "killed process\|oom"` for kernel OOM kills
2. Look for `Failed core proc(s): {}` — empty dict = external kill (OOM), not Python exception
3. Check the last EngineCore log line — if it's just before profiling/warmup, OOM is likely
4. Fix by reducing `max_num_batched_tokens` (controls profiling memory) or `gpu_memory_utilization`
