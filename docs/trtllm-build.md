# trtllm-build — What It Is and Why We Don't Use It

## What trtllm-build does

`trtllm-build` compiles a TRT-LLM checkpoint into a **TensorRT engine file** — a serialized,
GPU-optimized inference graph. It is step 2 in the classical TRT-LLM workflow:

```
Step 1: convert_checkpoint.py
        Converts HuggingFace weights → TRT-LLM checkpoint format (JSON config + weight shards)

Step 2: trtllm-build
        Compiles the checkpoint → optimized .engine file for a specific GPU / batch / seq-len

Step 3: trtllm-serve --engine_dir <dir>
        Serves the compiled engine (no Python overhead, kernel-fused inference)
```

### What the engine compilation buys you

The TensorRT engine is ahead-of-time compiled for a fixed configuration:

- **Kernel fusion**: adjacent ops merged into single CUDA kernels
- **Precision selection**: FP8 / FP4 kernels chosen and tuned at build time
- **Memory planning**: activation buffers pre-allocated; no dynamic allocation at runtime
- **Shape specialization**: padded and optimal paths selected for the target batch/seq-len

Result: lower latency and higher throughput than PyTorch inference for the same model.

### TRT-LLM 1.3.0rc6 — two backends

| Backend | Entry point | Workflow | Performance |
|---------|------------|---------|-------------|
| **TRT engine** | `trtllm-build` → `trtllm-serve --engine_dir` | convert → build → serve | Highest (kernel-fused) |
| **`_torch` (PyTorch)** | `trtllm-serve <hf_model_dir>` | serve directly from HF checkpoint | Lower, but still uses CUTLASS kernels |

`trtllm-build` is only for the TRT engine path. Models that only have a `_torch`
implementation cannot be compiled with `trtllm-build`.

### Getting convert_checkpoint.py

`convert_checkpoint.py` is **not installed by `pip install tensorrt-llm`** — it lives in
the GitHub source repo under `examples/<model>/`. To get it:

```bash
git clone https://github.com/NVIDIA/TensorRT-LLM.git \
  --branch v0.13.0rc6 --depth 1 ~/tensorrt-llm-src
# then e.g.:
python ~/tensorrt-llm-src/examples/qwen/convert_checkpoint.py --help
```

---

## Models in this repo — does trtllm-build apply?

**No.** Investigated for all models; none can use `trtllm-build` in TRT-LLM 1.3.0rc6:

| Model | HF architecture | Runtime | trtllm-build? | Reason |
|-------|----------------|---------|--------------|--------|
| `Qwen/Qwen3-Coder-Next-FP8` | `Qwen3MoeForCausalLM` | vLLM | No | Different runtime entirely |
| `Sehyo/Qwen3.5-122B-A10B-NVFP4` | `Qwen3_5MoeForConditionalGeneration` | vLLM | No | See below |
| `GadflyII/Qwen3-Coder-Next-NVFP4` | `Qwen3NextForCausalLM` | vLLM | No | See below |
| `nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` | `Qwen3NextForCausalLM` | vLLM | No | See below |
| `saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10` | `MiniMaxM2ForCausalLM` | TRT-LLM `_torch` | No | See below |

---

## Why trtllm-build won't work for the Qwen NVFP4 models

### Architecture not in the C++ engine path

The two NVFP4 Qwen variants in this repo use architectures that are only implemented in
the `_torch` (PyTorch) backend in 1.3.0rc6 — there is no C++ TRT engine for them:

| Architecture | C++ engine (`trtllm-build`) | `_torch` (`trtllm-serve`) |
|---|---|---|
| `Qwen3ForCausalLM` | ✅ | ✅ |
| `Qwen3MoeForCausalLM` | ✅ | ✅ |
| `Qwen3NextForCausalLM` | ❌ **not registered** | ✅ |
| `Qwen3_5MoeForConditionalGeneration` | ❌ **not registered** | ❌ (only `auto_deploy`) |

`Qwen3NextForCausalLM` — the architecture of `GadflyII/Qwen3-Coder-Next-NVFP4` and
`nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4` — exists only in `_torch`. There is no C++
engine implementation to compile to.

`Qwen3_5MoeForConditionalGeneration` — the architecture of `Sehyo/Qwen3.5-122B-A10B-NVFP4` —
is not even in the standard `_torch` registry; it appears only in `auto_deploy` (a
separate experimental path).

### The C++ Qwen engine has no NVFP4 / compressed-tensors support

Even for architectures that *are* in the C++ engine path (`Qwen3MoeForCausalLM`), the
`examples/qwen/convert_checkpoint.py` script has no support for NVFP4 or
compressed-tensors checkpoints. It only handles int8/int4 weight-only quantization
applied during conversion from a **full-precision** (BF16) source. A pre-quantized
NVFP4 checkpoint from llm-compressor cannot be converted via this path.

### vLLM is already the right runtime for these models

These FP4 Qwen models are well-supported by vLLM on GB10 (46–70 tok/s), with the
patches in this repo fixing the remaining bugs. Replacing that with an unvalidated
TRT engine path would require significant effort for uncertain gain.

---

## Why trtllm-build won't work for MiniMax

`MiniMaxM2ForCausalLM` is only implemented as a `_torch` model in 1.3.0rc6
(`tensorrt_llm/_torch/models/modeling_minimaxm2.py`). There is no C++ TRT engine
implementation, so `trtllm-build` cannot be used.

`convert_checkpoint.py` also has no MiniMax support — there is no `--model_type minimax`
option.

The `_torch` path is not meaningfully slower for MiniMax: the bottleneck is memory
bandwidth (139B parameters), and CUTLASS `nvfp4_gemm` kernels are dispatched either way.

```bash
# Correct way to serve MiniMax — no build step needed
~/trtllm-venv/bin/trtllm-serve saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10 \
  --port 8000 --max_seq_len 131072 --trust_remote_code
```

---

## When you would use trtllm-build

Use `trtllm-build` for models with a C++ TRT engine implementation:

- **LLaMA / Mistral / Qwen2 / Qwen3 (non-Next, non-3.5)** — have both
  `convert_checkpoint.py` and C++ engine support
- **Fixed-batch, latency-critical deployments** where build time (minutes to hours for
  large models) is worth the inference speedup
- **Multi-GPU tensor-parallel engines** — `--tp_size N` is a `trtllm-build` flag

For new architectures, pre-quantized NVFP4 checkpoints, and models without C++
implementations — use the `_torch` backend (`trtllm-serve <hf_dir>`) or vLLM.
