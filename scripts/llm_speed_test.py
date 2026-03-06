#!/usr/bin/env python3
"""Accurate LLM speed test for OpenAI-compatible chat endpoints.

Measures:
- TTFT (time-to-first-token)
- End-to-end latency
- Decode-only latency (first token -> end)
- Tokens/sec (end-to-end and decode-only)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class RunResult:
    run_idx: int
    ok: bool
    error: str | None
    latency_s: float | None
    ttft_s: float | None
    decode_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    tps_e2e: float | None
    tps_decode: float | None


def fetch_models(base_url: str, api_key: str, timeout: float = 10.0) -> list[str]:
    """Fetch available model names from the /v1/models endpoint."""
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["id"] for m in data.get("data", [])]


def resolve_model(args: argparse.Namespace) -> str:
    """Auto-detect model from server if --model was not explicitly set."""
    if args.model:
        return args.model
    print(f"Fetching models from {args.base_url.rstrip('/')}/models ...")
    try:
        models = fetch_models(args.base_url, args.api_key)
    except Exception as e:
        print(f"ERROR: Could not fetch models: {e}")
        print("Use --model to specify the model name explicitly.")
        sys.exit(1)
    if not models:
        print("ERROR: Server returned no models.")
        sys.exit(1)
    if len(models) == 1:
        print(f"Auto-selected model: {models[0]}")
        return models[0]
    print(f"Multiple models available: {', '.join(models)}")
    print("Use --model to specify which one to test.")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark LLM speed in tokens/sec")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="")
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--runs", type=int, default=8, help="Measured runs")
    p.add_argument("--warmup", type=int, default=2, help="Warmup runs")
    p.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--prompt",
        default=(
            "Write concise Python code for a stable merge sort and then explain the time complexity. "
            "Keep the explanation brief."
        ),
    )
    p.add_argument(
        "--json-out",
        default="",
        help="Optional path to write raw results JSON",
    )
    p.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip inference and tool-call validation tests",
    )
    return p.parse_args()


def _pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def _extract_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    out.append(txt)
        return "".join(out)
    # Some servers expose reasoning fields instead of content.
    for key in ("reasoning_content", "reasoning"):
        val = delta.get(key)
        if isinstance(val, str):
            return val
    return ""


def run_once(args: argparse.Namespace, idx: int) -> RunResult:
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    url = args.base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {args.api_key}",
        },
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta_text = _extract_delta_text(chunk)
                if delta_text and first_token_at is None:
                    first_token_at = time.perf_counter()

                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    total_tokens = usage.get("total_tokens", total_tokens)

        ended = time.perf_counter()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        msg = f"HTTP {e.code}: {body[:400]}"
        return RunResult(idx, False, msg, None, None, None, None, None, None, None, None)
    except Exception as e:
        return RunResult(idx, False, str(e), None, None, None, None, None, None, None, None)

    latency_s = ended - started
    ttft_s = None if first_token_at is None else first_token_at - started
    decode_s = None if first_token_at is None else ended - first_token_at

    tps_e2e = None
    tps_decode = None
    if completion_tokens and completion_tokens > 0:
        tps_e2e = completion_tokens / latency_s if latency_s > 0 else None
        if decode_s and decode_s > 0:
            tps_decode = completion_tokens / decode_s

    return RunResult(
        run_idx=idx,
        ok=True,
        error=None,
        latency_s=latency_s,
        ttft_s=ttft_s,
        decode_s=decode_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        tps_e2e=tps_e2e,
        tps_decode=tps_decode,
    )


def f2(v: float | None) -> str:
    return "-" if v is None else f"{v:.2f}"


def fi(v: int | None) -> str:
    return "-" if v is None else str(v)


def _chat_completion(
    base_url: str,
    model: str,
    api_key: str,
    messages: list[dict],
    timeout: float = 60.0,
    tools: list[dict] | None = None,
) -> dict:
    """Non-streaming chat completion request. Returns the full response JSON."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 256,
    }
    if tools:
        payload["tools"] = tools
    url = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_inference(args: argparse.Namespace) -> bool:
    """Ask 'What is the capital of France?' and verify the answer contains 'paris'."""
    print("\n--- Inference Test ---")
    print("Prompt: What is the capital of France?")
    try:
        resp = _chat_completion(
            args.base_url,
            args.model,
            args.api_key,
            [{"role": "user", "content": "What is the capital of France?"}],
            timeout=args.timeout,
        )
        content = resp["choices"][0]["message"].get("content", "")
        print(f"Response: {content[:300]}")
        if "paris" in content.lower():
            print("Result: PASS")
            return True
        else:
            print("Result: FAIL (response does not contain 'paris')")
            return False
    except Exception as e:
        print(f"Result: FAIL ({e})")
        return False


def test_tool_call(args: argparse.Namespace) -> bool:
    """Request a tool call to list files, verify the model emits the tool call."""
    print("\n--- Tool Call Test ---")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files and directories in a given path",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The directory path to list",
                        }
                    },
                    "required": ["path"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "List the files in the current directory using the list_files tool.",
        }
    ]
    print(f"Prompt: {messages[0]['content']}")
    try:
        resp = _chat_completion(
            args.base_url,
            args.model,
            args.api_key,
            messages,
            timeout=args.timeout,
            tools=tools,
        )
        message = resp["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            content = message.get("content", "")
            print(f"Response (no tool call): {content[:300]}")
            print("Result: FAIL (model did not emit a tool call)")
            return False
        tc = tool_calls[0]
        fn_name = tc["function"]["name"]
        fn_args = tc["function"]["arguments"]
        if isinstance(fn_args, str):
            fn_args = json.loads(fn_args)
        print(f"Tool call: {fn_name}({json.dumps(fn_args)})")
        if fn_name == "list_files" and "path" in fn_args:
            print("Result: PASS")
            return True
        else:
            print(f"Result: FAIL (unexpected function name or missing 'path' arg)")
            return False
    except Exception as e:
        print(f"Result: FAIL ({e})")
        return False


def main() -> int:
    args = parse_args()
    args.model = resolve_model(args)
    total_runs = args.warmup + args.runs
    results: list[RunResult] = []

    print(f"Endpoint: {args.base_url.rstrip('/')}/chat/completions")
    print(f"Model: {args.model}")
    print(f"Warmup: {args.warmup}  Measured: {args.runs}")
    print("")

    for i in range(total_runs):
        phase = "warmup" if i < args.warmup else "measure"
        res = run_once(args, i + 1)
        results.append(res)
        if not res.ok:
            print(f"[{phase} {i + 1}/{total_runs}] ERROR: {res.error}")
            continue
        print(
            f"[{phase} {i + 1}/{total_runs}] "
            f"lat={f2(res.latency_s)}s ttft={f2(res.ttft_s)}s decode={f2(res.decode_s)}s "
            f"ctok={fi(res.completion_tokens)} tps(e2e)={f2(res.tps_e2e)} tps(decode)={f2(res.tps_decode)}"
        )

    measured = [r for r in results[args.warmup :] if r.ok]
    if not measured:
        print("\nNo successful measured runs.")
        return 2

    e2e_vals = [r.tps_e2e for r in measured if r.tps_e2e is not None]
    dec_vals = [r.tps_decode for r in measured if r.tps_decode is not None]
    ttft_vals = [r.ttft_s for r in measured if r.ttft_s is not None]
    lat_vals = [r.latency_s for r in measured if r.latency_s is not None]
    ctok_vals = [r.completion_tokens for r in measured if r.completion_tokens is not None]

    print("\nPer-run measured results:")
    print("run,latency_s,ttft_s,decode_s,prompt_tokens,completion_tokens,total_tokens,tps_e2e,tps_decode")
    for r in measured:
        print(
            f"{r.run_idx},{f2(r.latency_s)},{f2(r.ttft_s)},{f2(r.decode_s)},"
            f"{fi(r.prompt_tokens)},{fi(r.completion_tokens)},{fi(r.total_tokens)},"
            f"{f2(r.tps_e2e)},{f2(r.tps_decode)}"
        )

    print("\nSummary (measured runs only):")
    if ttft_vals:
        print(
            f"TTFT(s): mean={statistics.mean(ttft_vals):.3f} "
            f"p50={statistics.median(ttft_vals):.3f} p95={_pct(sorted(ttft_vals), 0.95):.3f}"
        )
    if lat_vals:
        print(
            f"Latency(s): mean={statistics.mean(lat_vals):.3f} "
            f"p50={statistics.median(lat_vals):.3f} p95={_pct(sorted(lat_vals), 0.95):.3f}"
        )
    if ctok_vals:
        print(f"Completion tokens/run: mean={statistics.mean(ctok_vals):.1f}")
    if e2e_vals:
        print(
            f"Tokens/sec (E2E): mean={statistics.mean(e2e_vals):.2f} "
            f"p50={statistics.median(e2e_vals):.2f} p95={_pct(sorted(e2e_vals), 0.95):.2f}"
        )
    else:
        print("Tokens/sec (E2E): unavailable (server did not return usage.completion_tokens)")
    if dec_vals:
        print(
            f"Tokens/sec (decode): mean={statistics.mean(dec_vals):.2f} "
            f"p50={statistics.median(dec_vals):.2f} p95={_pct(sorted(dec_vals), 0.95):.2f}"
        )
    else:
        print("Tokens/sec (decode): unavailable (missing first token time or usage.completion_tokens)")

    if args.json_out:
        data = {
            "base_url": args.base_url,
            "model": args.model,
            "warmup": args.warmup,
            "runs": args.runs,
            "results": [r.__dict__ for r in results],
        }
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"\nWrote JSON: {args.json_out}")

    if not args.skip_tests:
        inference_ok = test_inference(args)
        tool_ok = test_tool_call(args)
        print("\n--- Test Summary ---")
        print(f"Inference test: {'PASS' if inference_ok else 'FAIL'}")
        print(f"Tool call test: {'PASS' if tool_ok else 'FAIL'}")
        if not (inference_ok and tool_ok):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
