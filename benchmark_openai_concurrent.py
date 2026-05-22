from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional


def make_prompt(target_tokens: int) -> str:
    if target_tokens <= 0:
        return "Explain why the sky is blue in scientific terms."
    seed = (
        "This is controlled benchmark context. "
        "Repeatable text makes prefill measurements comparable across runs. "
        "The answer should ignore this filler and respond to the final instruction. "
    )
    approx_tokens_per_seed = max(1, len(seed.split()) * 4 // 3)
    repeats = max(1, target_tokens // approx_tokens_per_seed)
    return (seed * repeats) + "\nFinal instruction: summarize why the sky is blue."


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((pct / 100) * (len(ordered) - 1)))
    return ordered[index]


def extract_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return (delta.get("content") or "") + (delta.get("reasoning_content") or "")


def post_chat(base_url: str, payload: dict[str, Any], stream: bool) -> dict[str, Any]:
    request_payload = dict(payload)
    request_payload["stream"] = stream
    if stream:
        request_payload["stream_options"] = {"include_usage": True}

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    ttft: Optional[float] = None
    usage: dict[str, Any] = {}
    chunks = 0
    chars = 0

    with urllib.request.urlopen(req, timeout=3600) as resp:
        if not stream:
            body = json.loads(resp.read().decode("utf-8"))
            usage = body.get("usage") or {}
        else:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                text = extract_delta_text(chunk)
                if text:
                    chunks += 1
                    chars += len(text)
                    if ttft is None:
                        ttft = time.perf_counter() - started

    elapsed = time.perf_counter() - started
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if stream and chunks == 0 and not usage:
        raise RuntimeError("stream ended without tokens or usage")
    decode_window = elapsed - ttft if ttft is not None else elapsed

    return {
        "elapsed_seconds": round(elapsed, 3),
        "ttft_seconds": round(ttft, 3) if ttft is not None else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": usage.get("total_tokens"),
        "prefill_tok_s_online": (
            prompt_tokens / ttft if prompt_tokens and ttft and ttft > 0 else None
        ),
        "decode_tok_s_online": (
            completion_tokens / decode_window
            if completion_tokens and decode_window and decode_window > 0
            else None
        ),
        "decode_tok_s_e2e": (
            completion_tokens / elapsed if completion_tokens and elapsed > 0 else None
        ),
        "stream_chunks": chunks if stream else None,
        "stream_chars": chars if stream else None,
    }


def summarize(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {
            "min": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "max": None,
            "mean": None,
            "stdev": None,
        }
    return {
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run_case(args: argparse.Namespace, prompt_tokens: int, max_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [{"role": "user", "content": make_prompt(prompt_tokens)}],
        "temperature": args.temperature,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning_effort": "none",
        "chat_template_kwargs": {"enable_thinking": False},
    }

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(post_chat, args.base_url, payload, args.stream)
            for _ in range(args.concurrency)
        ]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    wall = time.perf_counter() - started

    completion_tokens = sum(result.get("completion_tokens") or 0 for result in results)
    total_tokens = sum(result.get("total_tokens") or 0 for result in results)
    latencies = [result["elapsed_seconds"] for result in results if result.get("elapsed_seconds")]
    ttfts = [result["ttft_seconds"] for result in results if result.get("ttft_seconds") is not None]
    prefill_rates = [
        result["prefill_tok_s_online"]
        for result in results
        if result.get("prefill_tok_s_online") is not None
    ]
    decode_rates = [
        result["decode_tok_s_online"]
        for result in results
        if result.get("decode_tok_s_online") is not None
    ]

    return {
        "prompt_target_tokens": prompt_tokens,
        "max_output_tokens": max_tokens,
        "concurrency": args.concurrency,
        "stream": args.stream,
        "successful_requests": sum(1 for result in results if "error" not in result),
        "failed_requests": sum(1 for result in results if "error" in result),
        "wall_seconds": round(wall, 3),
        "aggregate_completion_tok_s": completion_tokens / wall if wall else None,
        "aggregate_total_tok_s": total_tokens / wall if wall and total_tokens else None,
        "latency": summarize(latencies),
        "ttft": summarize(ttfts),
        "prefill_tok_s_online": summarize(prefill_rates),
        "decode_tok_s_online": summarize(decode_rates),
        "requests": results,
    }


def parse_matrix(raw: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for item in raw.split(","):
        prompt_tokens, max_tokens = item.split("x", 1)
        cases.append((int(prompt_tokens), int(max_tokens)))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark prefill, TTFT, decode, and aggregate throughput for /v1 endpoints."
    )
    parser.add_argument("--base-url", required=True, help="Endpoint base URL ending in /v1")
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--stream", action="store_true", help="Measure streaming TTFT")
    parser.add_argument(
        "--matrix",
        default="512x1,2048x1,8192x1,128x256,1024x256,16384x128",
        help="Comma-separated prompt_tokens x max_output_tokens cases, e.g. 2048x1,128x512",
    )
    args = parser.parse_args()

    results = [run_case(args, prompt_tokens, max_tokens) for prompt_tokens, max_tokens in parse_matrix(args.matrix)]
    print(json.dumps({"cases": results}, indent=2))


if __name__ == "__main__":
    main()
