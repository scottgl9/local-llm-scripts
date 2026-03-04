# TRT-LLM 1.3.0rc6 — Local venv (No Container)

Running `trtllm-serve` and `trtllm-build` natively in a Python venv on the DGX Spark /
GB10 host, without the Docker container.

---

## Environment

| Item | Value |
|------|-------|
| Host | NVIDIA GB10 (Grace Blackwell), SM 12.1, 120 GB unified memory |
| OS | Ubuntu 24.04 LTS, aarch64 |
| CUDA | 13.0 (`nvcc --version`, driver 580.126.09) |
| Python | `/usr/bin/python3.12` (system) — **not** the linuxbrew 3.14 |
| venv path | `~/trtllm-venv` |
| TRT-LLM | 1.3.0rc6 (from PyPI `--pre`) |
| PyTorch | 2.9.1+cu130 (from `download.pytorch.org/whl/cu130`) |

---

## Why Not Use the System Python?

`/home/linuxbrew/.linuxbrew/bin/python3` is version 3.14.
TRT-LLM 1.3.0rc6 supports Python 3.10–3.13 only.
Use `/usr/bin/python3.12` to create the venv.

---

## Setup Steps

### 1. Create the venv

```bash
/usr/bin/python3.12 -m venv ~/trtllm-venv
~/trtllm-venv/bin/pip install --upgrade pip
```

### 2. Install TRT-LLM 1.3.0rc6 with all dependencies

```bash
~/trtllm-venv/bin/pip install --pre "tensorrt-llm==1.3.0rc6"
```

This installs ~100 packages and builds several wheels from source (takes ~10 min).
Key packages pulled in: `torch 2.9.1` (CPU-only from PyPI), `tensorrt 10.14.x`, `flashinfer 0.6.4`, etc.

### 3. Replace CPU torch with CUDA-enabled build

The default PyPI `torch 2.9.1` is CPU-only. Replace it:

```bash
~/trtllm-venv/bin/pip install --force-reinstall "torch==2.9.1+cu130" \
  --index-url https://download.pytorch.org/whl/cu130 --no-deps
```

Also replace torchvision:

```bash
~/trtllm-venv/bin/pip install --force-reinstall "torchvision==0.24.1" --no-deps
```

### 4. Install missing system packages

OpenMPI is required by TRT-LLM's MPI bindings:

```bash
sudo apt-get install -y openmpi-bin libopenmpi-dev
```

### 5. Install missing CUDA libraries

The CUDA torch depends on additional NVIDIA packages not pulled automatically:

```bash
~/trtllm-venv/bin/pip install nvidia-cudnn-cu13 nvidia-cusparselt-cu13 --no-deps
```

### 6. Build and deploy the ABI shim

See [ABI Compatibility](#abi-compatibility-shim) below for full explanation.

```bash
TORCH_LIB=~/trtllm-venv/lib/python3.12/site-packages/torch/lib
TRTLLM_LIBS=~/trtllm-venv/lib/python3.12/site-packages/tensorrt_llm/libs

gcc -O2 -shared -fPIC \
  -o "$TORCH_LIB/pyobjectslot_shim.so" \
  "$TORCH_LIB/pyobjectslot_shim.c" \
  -L "$TORCH_LIB" -lc10_cuda \
  -Wl,-rpath,"$TORCH_LIB"

cp "$TORCH_LIB/pyobjectslot_shim.so" "$TRTLLM_LIBS/"
~/trtllm-venv/bin/patchelf --add-needed pyobjectslot_shim.so \
  "$TRTLLM_LIBS/libth_common.so"
```

### 7. Apply MiniMax patches

```bash
~/trtllm-venv/bin/apply-minimax-patches
```

Or equivalently:

```bash
~/trtllm-venv/bin/python3 ~/spark-llm-scripts/patch_trtllm_minimax.py
```

(The patch script hardcodes container paths; `apply-minimax-patches` rewrites them to
the venv paths automatically.)

### 8. Verify

```bash
~/trtllm-venv/bin/python3 -c "
import tensorrt_llm, torch, os
print('TRT-LLM:', tensorrt_llm.__version__)
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
"
```

Expected output (warnings about SM 12.1 range are harmless):
```
TRT-LLM: 1.3.0rc6
torch: 2.9.1+cu130
CUDA available: True
```

---

## Serving the MiniMax Model

```bash
~/trtllm-venv/bin/trtllm-serve saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10 \
  --port 8000 \
  --max_seq_len 131072 \
  --trust_remote_code
```

**Note:** This model uses the **PyTorch `_torch` backend** — `trtllm-serve` loads the
model directly from the HuggingFace checkpoint via Python, with no TensorRT engine build
required. `trtllm-build` is not needed for this model.

### GB10 / SM 12.1 Compatibility

PyTorch warns that SM 12.1 is outside its "tested" range (8.0–12.0), but inference
works. The model uses `QuantAlgo.NVFP4` (CUTLASS `nvfp4_gemm` kernel) which runs on
SM 12.1. The `W4A8_NVFP4_FP8` kernel (`fp4_fp8_gemm_trtllmgen`) would be faster but
requires SM 10.0a/10.3a and does **not** run on GB10.

---

## ABI Compatibility Shim

### Problem

TRT-LLM 1.3.0rc6 ships pre-built C++ binaries (`libth_common.so`) compiled against a
custom NVIDIA-internal torch build. This build falls between the PyPI stable releases
`2.9.1+cu130` and `2.10.0+cu130`, and differs from both in two symbol signatures:

| Symbol | TRT-LLM expects | PyPI torch 2.9.1+cu130 has | Notes |
|--------|----------------|---------------------------|-------|
| `c10::cuda::c10_cuda_check_implementation` | `jb` (4th param `unsigned int`) | `ib` (4th param `int`) | Line number parameter type |
| `c10::impl::PyObjectSlot::PyObjectSlot()` | external symbol | external symbol ✓ | OK in 2.9.1 |
| `c10::impl::PyObjectSlot::~PyObjectSlot()` | external symbol | external symbol ✓ | OK in 2.9.1 |

Summary: torch `2.9.1+cu130` from PyPI has `PyObjectSlot` symbols (good) but the wrong
signature for `c10_cuda_check_implementation` (line number is `int` instead of
`unsigned int`).

### Solution

Build a small shim `.so` that provides the `jb` (unsigned int) variant by forwarding
to the existing `ib` (int) variant in `libc10_cuda.so`. Line numbers are always small
positive integers, so the `unsigned int → int` cast is safe.

The shim source lives at:
```
~/trtllm-venv/lib/python3.12/site-packages/torch/lib/pyobjectslot_shim.c
```

The shim is injected into `libth_common.so`'s `DT_NEEDED` list using `patchelf`, so
it is loaded automatically whenever `libth_common.so` is loaded — no `LD_PRELOAD`
environment variable needed.

### Why not torch 2.10.0+cu130?

torch `2.10.0+cu130` (final release) moved `PyObjectSlot` constructor/destructor to
inline-only, removing the external symbols. TRT-LLM's `libth_common.so` still
references them externally, causing `ImportError` at load time. A 2-symbol shim could
address this too, but then `c10_cuda_check_implementation` would have the correct `jb`
signature from `libc10_cuda.so` — making it theoretically workable, but we'd need to
shim `MessageLogger` constructor as well (added a `bool` param in 2.10.0 final). The
`2.9.1+cu130` approach requires only one shim symbol.

### Re-applying after package updates

If torch is reinstalled (e.g. by `pip install tensorrt-llm` pulling CPU torch), rebuild
and redeploy the shim:

```bash
TORCH_LIB=~/trtllm-venv/lib/python3.12/site-packages/torch/lib
TRTLLM_LIBS=~/trtllm-venv/lib/python3.12/site-packages/tensorrt_llm/libs

# Rebuild
gcc -O2 -shared -fPIC \
  -o "$TORCH_LIB/pyobjectslot_shim.so" \
  "$TORCH_LIB/pyobjectslot_shim.c" \
  -L "$TORCH_LIB" -lc10_cuda \
  -Wl,-rpath,"$TORCH_LIB"

# Redeploy
cp "$TORCH_LIB/pyobjectslot_shim.so" "$TRTLLM_LIBS/"

# Only needed if libth_common.so was replaced (i.e. tensorrt-llm was reinstalled)
readelf -d "$TRTLLM_LIBS/libth_common.so" | grep -q pyobjectslot_shim || \
  ~/trtllm-venv/bin/patchelf --add-needed pyobjectslot_shim.so \
    "$TRTLLM_LIBS/libth_common.so"
```

---

## Package Versions (Key)

| Package | Version |
|---------|---------|
| tensorrt-llm | 1.3.0rc6 |
| torch | 2.9.1+cu130 |
| torchvision | 0.24.1 |
| nvidia-nccl-cu13 | 2.27.7 (pinned by torch 2.9.1) |
| nvidia-cudnn-cu13 | 9.15.1.9 |
| nvidia-cusparselt-cu13 | 0.8.0 |
| tensorrt | 10.14.1.48.post1 |
| flashinfer-python | 0.6.4 |
| transformers | 4.57.1 |
| triton | 3.5.1 |
| nvidia-cutlass-dsl | 4.3.4 |

---

## Install Pitfalls

### Pitfall 1 — pip installs CPU-only torch

Running `pip install tensorrt-llm` pulls `torch` from PyPI default index which gives a
CPU-only build. Always install torch separately from the PyTorch CUDA wheel index:

```bash
pip install "torch==2.9.1+cu130" --index-url https://download.pytorch.org/whl/cu130 --no-deps
```

### Pitfall 2 — Reinstalling tensorrt-llm overwrites CUDA torch

`pip install tensorrt-llm` (without `--no-deps`) resolves torch from PyPI and replaces
the CUDA build. After any `tensorrt-llm` reinstall, always re-run:

```bash
~/trtllm-venv/bin/pip install --force-reinstall "torch==2.9.1+cu130" \
  --index-url https://download.pytorch.org/whl/cu130 --no-deps
~/trtllm-venv/bin/pip install --force-reinstall "torchvision==0.24.1" --no-deps
```

Then rebuild/redeploy the shim (see above).

### Pitfall 3 — MiniMax patches are not persistent

The patches modify files inside the installed `tensorrt_llm` package directory. If
`tensorrt-llm` is reinstalled, the patches are lost. Re-apply with:

```bash
~/trtllm-venv/bin/apply-minimax-patches
```

### Pitfall 4 — Missing libmpi / OpenMPI

TRT-LLM uses MPI for multi-GPU coordination even on single-GPU setups. The MPI
bindings will fail at import time if the runtime library is missing:

```
ImportError: libmpi.so.40: cannot open shared object file
```

Fix: `sudo apt-get install -y openmpi-bin libopenmpi-dev`

### Pitfall 5 — linuxbrew Python 3.14

`which python3` → `/home/linuxbrew/.linuxbrew/bin/python3` (3.14) on this system.
TRT-LLM does not support Python 3.14 yet. Always use `/usr/bin/python3.12` to create
the venv and `~/trtllm-venv/bin/python3` (or `trtllm-serve`) to run it.

---

## Maintenance Script

`~/trtllm-venv/bin/apply-minimax-patches` — idempotent script that translates container
paths to venv paths and runs `patch_trtllm_minimax.py`. Source is at
`~/spark-llm-scripts/patch_trtllm_minimax.py`.

After any `tensorrt-llm` reinstall, the full recovery sequence is:

```bash
# 1. Reinstall CUDA torch
~/trtllm-venv/bin/pip install --force-reinstall "torch==2.9.1+cu130" \
  --index-url https://download.pytorch.org/whl/cu130 --no-deps
~/trtllm-venv/bin/pip install --force-reinstall "torchvision==0.24.1" --no-deps

# 2. Rebuild shim
TORCH_LIB=~/trtllm-venv/lib/python3.12/site-packages/torch/lib
gcc -O2 -shared -fPIC \
  -o "$TORCH_LIB/pyobjectslot_shim.so" \
  "$TORCH_LIB/pyobjectslot_shim.c" \
  -L "$TORCH_LIB" -lc10_cuda -Wl,-rpath,"$TORCH_LIB"
cp "$TORCH_LIB/pyobjectslot_shim.so" \
  ~/trtllm-venv/lib/python3.12/site-packages/tensorrt_llm/libs/
~/trtllm-venv/bin/patchelf --add-needed pyobjectslot_shim.so \
  ~/trtllm-venv/lib/python3.12/site-packages/tensorrt_llm/libs/libth_common.so

# 3. Reapply MiniMax patches
~/trtllm-venv/bin/apply-minimax-patches
```

---

## Relationship to Container Workflow

| Aspect | Container (`serve_minimax.sh`) | Local venv |
|--------|-------------------------------|-----------|
| TRT-LLM version | 1.3.0rc6 | 1.3.0rc6 |
| Python | 3.12 in container | `/usr/bin/python3.12` |
| torch version | NVIDIA internal (between 2.9.1 and 2.10.0) | 2.9.1+cu130 + ABI shim |
| Patches | Applied at container startup | Applied once; persist in venv dir |
| MiniMax support | Yes (after patching) | Yes (after patching) |
| Startup overhead | Docker pull + start + patch (~2 min) | None after first setup |
| Isolation | Full container isolation | Host system |

Both workflows apply the same `patch_trtllm_minimax.py` patches and run the same
`trtllm-serve` command. The local venv is faster to start but requires the ABI shim to
bridge the torch version gap.
