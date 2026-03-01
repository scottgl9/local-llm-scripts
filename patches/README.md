# vLLM Tool-Call Patches

This directory contains patches for vLLM bugs affecting the `qwen3_coder` tool-call parser.

## Bugs fixed

Three bugs in `vLLM 0.16.0rc2` cause all tool calls to either fail or return empty arguments when using `Qwen3-Coder-Next` models. See [docs/TOOL_CALL_BUGS.md](../docs/TOOL_CALL_BUGS.md) for full root-cause analysis.

| Bug | File | Symptom |
|-----|------|---------|
| 1 | `chat_utils.py` | `TypeError: Can only get item pairs from a mapping` — arguments stay as JSON string instead of dict |
| 2 | `qwen3coder_tool_parser.py` | `IndexError: list index out of range` — `streamed_args_for_tool` never populated |
| 3 | `qwen3coder_tool_parser.py` | Tool arguments always streamed as `{}` empty — `</function>` tag processed before parameter loop |

## Structure

```
patches/
├── apply.sh          — Apply patches to a RUNNING container (docker cp)
├── build.sh          — Generate patched files into .build/ for volume mounting
└── vllm/
    ├── v23/          — Patches for avarok/dgx-vllm-nvfp4-kernel:v23
    │   │              vLLM path inside container: /app/vllm/vllm/
    │   ├── entrypoints/
    │   │   ├── chat_utils.py        (original from image)
    │   │   └── chat_utils.patch     (unified diff to apply)
    │   └── tool_parsers/
    │       ├── qwen3coder_tool_parser.py    (original from image)
    │       └── qwen3coder_tool_parser.patch
    └── v11/          — Patches for avarok/vllm-dgx-spark:v11
        │              vLLM path inside container: /opt/venv/lib/python3.12/site-packages/vllm/
        ├── entrypoints/
        │   ├── chat_utils.py
        │   └── chat_utils.patch
        └── tool_parsers/
            ├── qwen3coder_tool_parser.py
            └── qwen3coder_tool_parser.patch
```

## Usage

### Option 1: Volume-mount at container start (recommended)

```bash
# Build patched files into .build/ once
bash patches/build.sh v23   # or v11, or all

# Then use the appropriate run script:
bash servers/qwen3-coder-next-fp8/run-v23.sh
```

The run scripts call `build.sh` automatically on first run, then pass `-v` flags to mount the patched files read-only.

### Option 2: Apply patches to a running container

```bash
bash patches/apply.sh qwen3-fp8-server v23
```

This copies the patched files into the running container. You still need to restart vLLM for the changes to take effect.

### Option 3: Build a custom Docker image

```bash
# From repo root — embeds patches into image at build time
cd images/dgx-vllm-nvfp4-kernel/v23
docker build -t dgx-vllm-mtp-ready:v23 \
  --build-context repo=../../.. .
```

## Generating patch files (for updating)

If you need to regenerate the `.patch` files after modifying the patched sources:

```bash
diff -u patches/vllm/v23/tool_parsers/qwen3coder_tool_parser.py \
         .build/v23/tool_parsers/qwen3coder_tool_parser.py \
  --label a/qwen3coder_tool_parser.py \
  --label b/qwen3coder_tool_parser.py \
  > patches/vllm/v23/tool_parsers/qwen3coder_tool_parser.patch
```
