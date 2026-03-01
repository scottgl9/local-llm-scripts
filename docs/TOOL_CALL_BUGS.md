# vLLM Tool Call Bug Report — Qwen3-Coder-Next-FP8 + OpenClaw

**Container:** `qwen3-fp8-server` (`avarok/vllm-dgx-spark:v11`)  
**Model:** `Qwen/Qwen3-Coder-Next-FP8`  
**Tool parser:** `qwen3_coder` (`--tool-call-parser qwen3_coder`)  
**Client:** OpenClaw (OpenAI-compatible, strict mode tools)  
**vLLM install path:** `/opt/venv/lib/python3.12/site-packages/vllm/` (NOT `/app/vllm/vllm/`)  
**Date:** 2026-03-01  
**Status:** ✅ All 3 bugs fixed and verified. Tool calls stream correctly with full arguments (`{"location": "San Francisco", "unit": "celsius"}`). Zero errors in server logs.

---

## Bug 1 — `TypeError: Can only get item pairs from a mapping` (Chat Template)

### Symptom
Every tool-bearing request raises a 400 error:
```
jinja2.exceptions.UndefinedError: TypeError: Can only get item pairs from a mapping.
```

### Root Cause
The Qwen3-Coder-Next chat template uses Jinja2's `|items` filter on `tool_call.arguments`:
```jinja2
{%- if tool_call.arguments is defined %}
    {%- for args_name, args_value in tool_call.arguments|items %}
```
`|items` requires a **dict**. The preprocessing function `_postprocess_messages()` in `chat_utils.py` was supposed to convert JSON-string arguments to dicts, but had these bugs:

```python
# BEFORE (buggy)
for item in tool_calls:
    if content := item["function"].get("arguments"):
        if not isinstance(content, (dict, list)):   # ← lets lists pass through unchanged
            item["function"]["arguments"] = json.loads(content)  # ← may return list/None; no try/except
    else:
        item["function"]["arguments"] = {}
```

- `isinstance(content, (dict, list))` skips conversion when args is a list → template crash
- `json.loads()` can return `None` or a list → stored as-is → template crash
- No `try/except` — malformed JSON propagates as unhandled exception
- No guard for missing `"function"` key on non-function tool call types

### Fix
**File:** `/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/chat_utils.py`  
**Function:** `_postprocess_messages()`

```python
# AFTER (patched)
for item in tool_calls:
    func = item.get("function")
    if func is None:
        continue
    content = func.get("arguments")
    if isinstance(content, dict):
        pass  # already correct
    elif not content and content != 0:
        func["arguments"] = {}
    else:
        try:
            parsed = json.loads(content) if isinstance(content, str) else content
            func["arguments"] = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            func["arguments"] = {}
```

### Defense-in-Depth
**File:** `tokenizer_config.json` (model snapshot)  
Added `is mapping` guard in the chat template to skip non-dict arguments silently:
```jinja2
{%- if tool_call.arguments is defined and tool_call.arguments is mapping %}
```

### Notes
- transformers Jinja2 sandbox (v4.57.6) has no `fromjson` filter — template cannot self-repair strings
- OpenClaw sends `strict: true` and `store` fields — these are stripped by vLLM and are **not** the root cause

---

## Bug 2 — `IndexError: list index out of range` (Streaming — Wrong Path)

### Symptom
```
File ".../entrypoints/openai/serving_chat.py", line ~1201
  actual_call = tool_parser.streamed_args_for_tool[index]
IndexError: list index out of range
```
Occurs on every streaming request with a tool call. Persisted even after first patch attempt.

### Root Cause
`serving_chat.py` reads `tool_parser.streamed_args_for_tool[index]` at finish-time to compute what argument tokens are still unstreamed.

In `Qwen3CoderToolParser`, `streamed_args_for_tool` is declared as `[]` in `__init__` but **never populated** anywhere in the class — the list is always empty, so any index access raises `IndexError`. Every other vLLM parser (`hermes`, `olmo3`, `kimi_k2`, etc.) populates this list during streaming.

### First Patch Attempt (WRONG PATH — had no effect)
Patches were written to `/app/vllm/vllm/tool_parsers/qwen3coder_tool_parser.py` and `-v` mounts added to the run scripts pointing there. However, vLLM runs from `/opt/venv/lib/python3.12/site-packages/vllm/` (confirmed via `python3 -c "import inspect, vllm.tool_parsers.qwen3coder_tool_parser as m; print(inspect.getfile(m))"`). The `/app/vllm/` path is not in `sys.path` and is never loaded. The bug persisted across container restart.

### Correct Fix
**File:** `/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py`

1. **`_reset_streaming_state()`** — reset list between requests:
   ```python
   self.streamed_args_for_tool = []
   ```

2. **On new tool header sent** (after `prev_tool_call_arr.append(...)`):
   ```python
   while len(self.streamed_args_for_tool) <= self.current_tool_index:
       self.streamed_args_for_tool.append("")
   ```

3. **Opening brace `{` streamed:**
   ```python
   if self.current_tool_index < len(self.streamed_args_for_tool):
       self.streamed_args_for_tool[self.current_tool_index] += "{"
   ```

4. **Closing fragment streamed** (see Bug 3 below — `}` replaced with full args content):
   ```python
   if self.current_tool_index < len(self.streamed_args_for_tool):
       self.streamed_args_for_tool[self.current_tool_index] += closing_frag
   ```

### Volume Mount Fix
Run scripts (`~/run_qwen3_coder_next_fp8.sh`, `~/scripts/run_qwen3_coder_next_fp8.sh`) updated to mount the correct path:
```bash
# OLD (wrong — not in sys.path)
-v ~/dgx-vllm/patches/vllm/tool_parsers/qwen3coder_tool_parser.py:/app/vllm/vllm/tool_parsers/qwen3coder_tool_parser.py:ro

# NEW (correct)
-v ~/dgx-vllm/patches/vllm/tool_parsers/qwen3coder_tool_parser.py:/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py:ro
```

---

## Bug 3 — Tool Arguments Stream as `{}` (Empty) (Newly Discovered)

### Symptom
After Bug 2 is fixed, tool calls complete without crashing but the streamed arguments are always empty:
```json
{"name": "get_weather", "arguments": ""}  ← header
{"arguments": "{"}                         ← opening brace
{"arguments": "}"}                         ← closing brace immediately — no params!
{"finish_reason": "tool_calls"}            ← finish, delta has no tool_calls
```
The `get_weather` call for "San Francisco" returns `{}` instead of `{"location": "San Francisco"}`.

### Root Cause — Two sub-issues

**3a. Parameters never streamed — function-end check fires first**

In `extract_tool_calls_streaming()`, the check for `</function>` (closing the tool call) runs **before** the parameter streaming loop:
```python
# function_end fires first → returns immediately with "}"
if not self.json_closed and self.function_end_token in tool_text:
    ...
    return DeltaMessage(...arguments="}")   # ← returns here

# parameter loop never reached:
if not self.in_param and self.param_count < len(param_starts):
    ...
```
With `--stream-interval 5`, by the time the parser sees a delta containing parameters, the `</function>` closing tag is often already in `tool_text`. The function-end branch fires immediately, emits `}`, and the parameters are discarded without streaming.

The `remaining_call` mechanism in `serving_chat.py` is designed to fill this gap at finish time:
```python
expected_call = json.dumps(prev_tool_call_arr[index]["arguments"])
actual_call   = streamed_args_for_tool[index]
remaining_call = expected_call.replace(actual_call, "", 1)
# remaining_call replaces the final delta
```
But this mechanism only fires when `delta_message.tool_calls[0].function.arguments is not None` at finish. Since the parser returns `None` on the final token (after already emitting `}`), the condition is False and `remaining_call` is never sent.

**3b. `prev_tool_call_arr[i]["arguments"]` stores a JSON string, not a dict**

`_parse_xml_function_call()` returns a `ToolCall` where `function.arguments` is a JSON-encoded string (e.g. `'{"location": "San Francisco"}'`). The parser stores it directly:
```python
args = parsed_tool.function.arguments       # string: '{"location": "San Francisco"}'
self.prev_tool_call_arr[i]["arguments"] = args
```
But `serving_chat.py` does:
```python
expected_call = json.dumps(prev_tool_call_arr[index].get("arguments", {}))
```
`json.dumps()` on a string double-encodes it: `'"{\\"location\\": \\"San Francisco\\"}"'` — completely wrong. The `remaining_call` diff breaks even when the mechanism does fire.

Also, the placeholder in `prev_tool_call_arr` is initialized as `"arguments": "{}"` (string) instead of `"arguments": {}` (dict), causing `json.dumps("{}")` = `'"{}"'` instead of `'{}'`.

### Fix
**File:** `/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py`

**Change 1 — Placeholder dict, not string:**
```python
# BEFORE
self.prev_tool_call_arr.append({"name": ..., "arguments": "{}"})
# AFTER
self.prev_tool_call_arr.append({"name": ..., "arguments": {}})
```

**Change 2 — Stream full args content when closing function, store dict:**
```python
# BEFORE: only emits "}"
result = DeltaMessage(...arguments="}")

# AFTER: parse complete args, store as dict, stream content+closing-brace
args_dict = json.loads(parsed_tool.function.arguments)
args_json  = parsed_tool.function.arguments          # e.g. '{"location": "San Francisco"}'
self.prev_tool_call_arr[i]["arguments"] = args_dict  # store dict for json.dumps in serving

if self.json_started:
    closing_frag = args_json[1:]   # skip already-streamed "{", send '"location": "San Francisco"}'
else:
    closing_frag = args_json       # "{" never sent; send full JSON
    self.json_started = True

# track in streamed_args_for_tool
self.streamed_args_for_tool[self.current_tool_index] += closing_frag
result = DeltaMessage(...arguments=closing_frag)
```

This way the client receives:
```
{   ← first delta
"location": "San Francisco"}   ← closing delta (contains params + closing brace)
```
Which assembles to `{"location": "San Francisco"}` — correct.

The `remaining_call` mechanism in `serving_chat.py` handles the edge case where `</function>` lands on the same chunk as `finish_reason` (it correctly re-derives and replaces the final delta).

---

## Files to Patch (Correct Paths)

| File | Change |
|------|--------|
| `/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/chat_utils.py` | Bug 1: robust `_postprocess_messages()` |
| `/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py` | Bug 2+3: `streamed_args_for_tool` population; full args streamed at function close; dict in `prev_tool_call_arr` |
| `tokenizer_config.json` (model snapshot, huggingface cache) | Bug 1 defense-in-depth: `is mapping` guard |

## Persistence via Volume Mounts

Host copies in `~/dgx-vllm/patches/vllm/` are mounted `:ro` into the container at startup.  
Run scripts mount to the **correct** installed path:
```bash
-v ~/dgx-vllm/patches/vllm/entrypoints/chat_utils.py:/opt/venv/lib/python3.12/site-packages/vllm/entrypoints/chat_utils.py:ro
-v ~/dgx-vllm/patches/vllm/tool_parsers/qwen3coder_tool_parser.py:/opt/venv/lib/python3.12/site-packages/vllm/tool_parsers/qwen3coder_tool_parser.py:ro
```

> ⚠️ If the image is rebuilt or the container is re-created without the run script, patches are lost. Both bugs should be submitted upstream to `vllm-project/vllm`.
