# spark-llm-scripts — Claude Code Guidelines

## Repo Purpose

Scripts and patches for running LLM servers (Qwen3-Coder-Next-FP8, Qwen3.5-122B-A10B-NVFP4, etc.) on NVIDIA DGX Spark (GB10) hardware using vLLM Docker containers.

## Repository Structure

```
patches/          — vLLM bug-fix patches (source originals + diffs)
  build.sh        — Build patched files into .build/ for volume mounting
  apply.sh        — Apply patches to a running container (docker cp)
  vllm/
    v23/          — Patches for avarok/dgx-vllm-nvfp4-kernel:v23
      entrypoints/
        chat_utils.py               ← ORIGINAL from container (unmodified)
        chat_utils.patch            ← unified diff applied at build time
      tool_parsers/
        qwen3coder_tool_parser.py   ← ORIGINAL from container (unmodified)
        qwen3coder_tool_parser.patch
    nightly/      — Patches for vllm/vllm-openai:cu130-nightly
    v11/          — Patches for avarok/vllm-dgx-spark:v11
servers/          — Docker run scripts per model/version
  qwen3-coder-next-fp8/
    run-v23.sh    — Launch container (calls build.sh if needed)
  qwen35-122b-a10b-nvfp4/
    run-v23.sh    — Launch Qwen3.5-122B container
.build/           — Generated output of build.sh (gitignored)
  v23/            — Patched files volume-mounted into container
docs/
  TOOL_CALL_BUGS.md  — Root-cause analysis of all fixed bugs
  DEBUG.md        — Active debugging notes and issue tracking
images/           — Dockerfile for custom images with patches baked in
```

## Patch Workflow

### How patches work

Each `patches/vllm/<version>/` directory contains:
1. **Original source file** (e.g., `qwen3coder_tool_parser.py`) — the UNMODIFIED file extracted directly from the container image. Do NOT edit this.
2. **Patch file** (e.g., `qwen3coder_tool_parser.patch`) — unified diff between the original and the fixed version.

`build.sh` applies the patch to the original to produce `.build/<version>/`, which is then volume-mounted into the container.

### Making changes to a patch

**Always work from `.build/<version>/` as your edit target.**

1. Edit the built file in `.build/<version>/`:
   ```bash
   # Edit the built output
   nano .build/v23/tool_parsers/qwen3coder_tool_parser.py
   ```

2. Regenerate the patch file from the diff:
   ```bash
   diff -u \
     patches/vllm/v23/tool_parsers/qwen3coder_tool_parser.py \
     .build/v23/tool_parsers/qwen3coder_tool_parser.py \
     --label a/qwen3coder_tool_parser.py \
     --label b/qwen3coder_tool_parser.py \
     > patches/vllm/v23/tool_parsers/qwen3coder_tool_parser.patch
   ```
   (exit code 1 from diff means differences were found and the patch was written — this is expected and correct)

3. Restart the container:
   ```bash
   bash servers/qwen35-122b-a10b-nvfp4/run-v23.sh
   ```
   The run script volume-mounts `.build/v23/` files into the container at startup.

### Adding a new patch for a file

1. Extract the original from the container:
   ```bash
   docker cp qwen35-122b-server:/app/vllm/vllm/path/to/some_file.py \
     patches/vllm/v23/path/to/some_file.py
   ```

2. Copy to `.build/` and edit:
   ```bash
   mkdir -p .build/v23/path/to/
   cp patches/vllm/v23/path/to/some_file.py .build/v23/path/to/some_file.py
   nano .build/v23/path/to/some_file.py
   ```

3. Generate the patch:
   ```bash
   diff -u \
     patches/vllm/v23/path/to/some_file.py \
     .build/v23/path/to/some_file.py \
     --label a/some_file.py \
     --label b/some_file.py \
     > patches/vllm/v23/path/to/some_file.patch
   ```

4. Add the volume mount to the run script (`-v "$BUILD_DIR/path/to/some_file.py:$VLLM_BASE/path/to/some_file.py:ro"`).

5. Add the file to the build-check `if` block in the run script.

### vLLM install paths per container

| Container image | vLLM path inside container |
|----------------|---------------------------|
| `avarok/dgx-vllm-nvfp4-kernel:v23` | `/app/vllm/vllm/` |
| `vllm/vllm-openai:cu130-nightly` | `/usr/local/lib/python3.12/dist-packages/vllm/` |
| `avarok/vllm-dgx-spark:v11` | `/opt/venv/lib/python3.12/site-packages/vllm/` |

Verify with:
```bash
docker exec <container> python3 -c \
  "import inspect, vllm.tool_parsers.qwen3coder_tool_parser as m; print(inspect.getfile(m))"
```

## Debugging Container Crashes

When a container crashes, follow this workflow:

### 1. Gather information

```bash
# Check container status
docker ps -a --filter name=<container-name>

# Get full logs (stderr + stdout)
docker logs <container-name> 2>&1 | tail -300

# Check for OOM kills (common on GB10 unified memory)
sudo dmesg | grep -i "killed process\|oom\|out of memory" | tail -20
```

### 2. Identify the failure point

Look at the **last EngineCore log line** before the crash. Common failure points:

| Last log before crash | Likely cause |
|----------------------|--------------|
| After torch.compile, before profiling | OOM during `profile_run()` — reduce `max-num-batched-tokens` |
| During weight loading | Missing weight / wrong quantization config — check `packed_modules_mapping` |
| `device-side assert` | NaN logits from uninitialized weights — check weight loading |
| `Failed core proc(s): {}` (empty) | External kill (OOM killer) — check `dmesg` |
| `Failed core proc(s): {pid: ...}` | Python exception — check EngineCore logs above the traceback |

### 3. Use ~/sandbox/vllm as reference

The `~/sandbox/vllm` directory contains a full vLLM source checkout for code reference:
```bash
# Find the relevant function
grep -rn "profile_run\|determine_num_available_blocks" ~/sandbox/vllm/vllm/v1/worker/
```

### 4. Patch, rebuild, and restart

```bash
# Edit the built file (NOT the original)
nano .build/v23/path/to/file.py

# Regenerate the patch
diff -u patches/vllm/v23/path/to/file.py .build/v23/path/to/file.py \
  --label a/file.py --label b/file.py > patches/vllm/v23/path/to/file.patch

# Restart the container (run script handles stop + rm + rebuild check)
bash servers/qwen35-122b-a10b-nvfp4/run-v23.sh

# Monitor startup
docker logs -f qwen35-122b-server
```

### 5. Document the issue

Update `docs/DEBUG.md` with:
- Symptom (error message, traceback)
- Root cause analysis
- Timeline from logs
- Fix applied
- Key vLLM internals learned

## DGX Spark (GB10) Specifics

- **128 GiB unified memory** — CPU and GPU share the same memory pool
- **SM121 (cuda capability 12.1)** — requires Triton kernels compiled via `ptxas-blackwell`; first startup takes 30+ min
- **Compiler caches** — mount `~/.cache/vllm_compilers/` directories to persist compiled kernels across restarts
- **MoE config missing** — `E=256,N=1024,device_name=NVIDIA_GB10.json` not shipped; performance-only warning, not a crash
- **OOM is the #1 crash cause** — with 76 GiB model, only ~52 GiB remains for activations + KV cache
- **`max_num_batched_tokens`** controls profiling memory, NOT `max_model_len`; set explicitly with `--enable-chunked-prefill --max-num-batched-tokens 8192`

## Known Bugs Fixed

See `docs/TOOL_CALL_BUGS.md` for full details. Summary:

| Bug | File | Symptom |
|-----|------|---------|
| 1 | `chat_utils.py` | `TypeError: Can only get item pairs from a mapping` |
| 2 | `qwen3coder_tool_parser.py` | `IndexError: streamed_args_for_tool` never populated |
| 3 | `qwen3coder_tool_parser.py` | Tool arguments always `{}` empty |
| 4 | `qwen3coder_tool_parser.py` | Doubled JSON arguments (params re-streamed at function close) |
| 5 | `chat_utils.py` | `TemplateError: Unexpected message role` — developer→system conversion |
| 6 | `qwen3_5.py` | `in_proj_ba.weight` not found — missing `packed_modules_mapping` entries |
| 7 | `qwen3_5.py` | `mlp.gate.weight` false match — `gate_proj` substring matches MoE router |

Bug 4 (found 2026-03-01): `closing_frag = args_json[1:]` in the function-end handler resent all params even when some were already streamed individually. Fix: track `json_fragment` in `streamed_args_for_tool` during param loop; compute `closing_frag = args_json[len(already_streamed):]`.
