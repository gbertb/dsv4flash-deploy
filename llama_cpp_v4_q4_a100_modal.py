from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import fmean, median, stdev
from typing import Any

import modal


APP_NAME = os.environ.get("LLAMA_CPP_APP_NAME", "deepseek-v4-flash-llama-cpp-q4-a100")
LLAMA_REPO = os.environ.get("LLAMA_CPP_REPO", "https://github.com/cchuter/llama.cpp.git")
LLAMA_BRANCH = os.environ.get("LLAMA_CPP_BRANCH", "feat/v4-port-cuda")
LLAMA_DIR = "/opt/llama.cpp"
SOURCE_PROBE_REPO = os.environ.get("LLAMA_CPP_SOURCE_PROBE_REPO", LLAMA_REPO)
SOURCE_PROBE_BRANCH = os.environ.get("LLAMA_CPP_SOURCE_PROBE_BRANCH", LLAMA_BRANCH)
SOURCE_PROBE_DIR = "/opt/llama.cpp-source-probe"
MODEL_DIR = "/models"
CUDA_STUBS_DIR = "/usr/local/cuda/targets/x86_64-linux/lib/stubs"

HF_REPO = os.environ.get("LLAMA_CPP_HF_REPO", "teamblobfish/DeepSeek-V4-Flash-GGUF")
HF_SECRET_NAME = os.environ.get("LLAMA_CPP_HF_SECRET", "")
MODEL_QUANT = os.environ.get("LLAMA_CPP_MODEL_QUANT", "Q4_K_M-XL")
MODEL_SHARDS = {
    "IQ1_M-XL": 2,
    "IQ1_M": 2,
    "IQ1_S-XL": 2,
    "IQ2_XS-XL": 2,
    "IQ2_XXS-XL": 2,
    "Q2_K-XL": 3,
    "Q4_K_M-XL": 4,
    "Q8_0": 7,
}
if MODEL_QUANT not in MODEL_SHARDS:
    raise ValueError(f"LLAMA_CPP_MODEL_QUANT must be one of: {', '.join(MODEL_SHARDS)}")

MODEL_BASENAME = f"DeepSeek-V4-Flash-{MODEL_QUANT}"
MODEL_FIRST_SHARD = (
    f"{MODEL_QUANT}/{MODEL_BASENAME}-00001-of-{MODEL_SHARDS[MODEL_QUANT]:05d}.gguf"
)
MODEL_PATH = f"{MODEL_DIR}/{MODEL_FIRST_SHARD}"

PORT = 8000
MINUTES = 60

GPU_CONFIG = os.environ.get("LLAMA_CPP_GPU", "A100-80GB:4")
CUDA_ARCH = os.environ.get("LLAMA_CPP_CUDA_ARCH", "80")
SCHED_MAX_SPLIT_INPUTS = int(os.environ.get("LLAMA_CPP_SCHED_MAX_SPLIT_INPUTS", "128"))
CMAKE_EXTRA_ARGS = os.environ.get("LLAMA_CPP_CMAKE_EXTRA_ARGS", "")
CPU_COUNT = float(os.environ.get("LLAMA_CPP_CPU", "32"))
MEMORY_MIB = int(os.environ.get("LLAMA_CPP_MEMORY_MIB", "327680"))
EPHEMERAL_DISK_MIB = int(os.environ.get("LLAMA_CPP_EPHEMERAL_DISK_MIB", "1048576"))
STARTUP_TIMEOUT = int(os.environ.get("LLAMA_CPP_STARTUP_TIMEOUT", str(4 * 60 * MINUTES)))

CTX_TOKENS = int(os.environ.get("LLAMA_CPP_CTX", "32768"))
PARALLEL = int(os.environ.get("LLAMA_CPP_PARALLEL", "1"))
N_GPU_LAYERS = os.environ.get("LLAMA_CPP_N_GPU_LAYERS", "999")
SPLIT_MODE = os.environ.get("LLAMA_CPP_SPLIT_MODE", "layer")
TENSOR_SPLIT = os.environ.get("LLAMA_CPP_TENSOR_SPLIT", "")
MAIN_GPU = os.environ.get("LLAMA_CPP_MAIN_GPU", "")
CACHE_TYPE_K = os.environ.get("LLAMA_CPP_CACHE_TYPE_K", "q4_0")
CACHE_TYPE_V = os.environ.get("LLAMA_CPP_CACHE_TYPE_V", "")
EXTRA_SERVER_ARGS = os.environ.get("LLAMA_CPP_EXTRA_SERVER_ARGS", "")
SCALEDOWN_WINDOW = int(os.environ.get("LLAMA_CPP_SCALEDOWN_WINDOW", "60"))

REMOTE_ENV_KEYS = (
    "LLAMA_CPP_REPO",
    "LLAMA_CPP_BRANCH",
    "LLAMA_CPP_SOURCE_PROBE_REPO",
    "LLAMA_CPP_SOURCE_PROBE_BRANCH",
    "LLAMA_CPP_HF_REPO",
    "LLAMA_CPP_HF_SECRET",
    "LLAMA_CPP_MODEL_QUANT",
    "LLAMA_CPP_GPU",
    "LLAMA_CPP_CUDA_ARCH",
    "LLAMA_CPP_SCHED_MAX_SPLIT_INPUTS",
    "LLAMA_CPP_CMAKE_EXTRA_ARGS",
    "LLAMA_CPP_CPU",
    "LLAMA_CPP_MEMORY_MIB",
    "LLAMA_CPP_EPHEMERAL_DISK_MIB",
    "LLAMA_CPP_STARTUP_TIMEOUT",
    "LLAMA_CPP_CTX",
    "LLAMA_CPP_PARALLEL",
    "LLAMA_CPP_N_GPU_LAYERS",
    "LLAMA_CPP_SPLIT_MODE",
    "LLAMA_CPP_TENSOR_SPLIT",
    "LLAMA_CPP_MAIN_GPU",
    "LLAMA_CPP_CACHE_TYPE_K",
    "LLAMA_CPP_CACHE_TYPE_V",
    "LLAMA_CPP_EXTRA_SERVER_ARGS",
    "LLAMA_CPP_SCALEDOWN_WINDOW",
    "LLAMA_CPP_MODEL_VOLUME",
    "LLAMA_CPP_SMOKE_TOKENS",
    "LLAMA_CPP_SMOKE_CTX",
    "LLAMA_CPP_BENCHMARK_TOKENS",
    "CUDA_SCALE_LAUNCH_QUEUES",
    "GGML_CUDA_P2P",
    "GGML_CUDA_FORCE_CUBLAS_COMPUTE_16F",
    "GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F",
)
remote_env = {key: os.environ[key] for key in REMOTE_ENV_KEYS if key in os.environ}

MODEL_VOLUME_NAME = os.environ.get(
    "LLAMA_CPP_MODEL_VOLUME", f"llama-cpp-v4-flash-{MODEL_QUANT.lower()}".replace("_", "-")
)
model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
download_secrets = [modal.Secret.from_name(HF_SECRET_NAME)] if HF_SECRET_NAME else []

llama_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "build-essential",
        "ca-certificates",
        "cmake",
        "curl",
        "git",
        "ninja-build",
    )
    .pip_install("hf_transfer", "huggingface_hub[hf_transfer]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", **remote_env})
    .run_commands(
        (
            f"test -e {CUDA_STUBS_DIR}/libcuda.so && "
            f"ln -sf {CUDA_STUBS_DIR}/libcuda.so {CUDA_STUBS_DIR}/libcuda.so.1"
        ),
        f"git clone --depth=1 --branch {LLAMA_BRANCH} {LLAMA_REPO} {LLAMA_DIR}",
        (
            f"cd {LLAMA_DIR} && LIBRARY_PATH={CUDA_STUBS_DIR}:$LIBRARY_PATH "
            "cmake -B build -G Ninja "
            "-DGGML_CUDA=ON "
            f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH} "
            f"{CMAKE_EXTRA_ARGS} "
            f"-DCMAKE_C_FLAGS='-DGGML_SCHED_MAX_SPLIT_INPUTS={SCHED_MAX_SPLIT_INPUTS}' "
            f"-DCMAKE_CXX_FLAGS='-DGGML_SCHED_MAX_SPLIT_INPUTS={SCHED_MAX_SPLIT_INPUTS}' "
            f"-DCMAKE_CUDA_FLAGS='-DGGML_SCHED_MAX_SPLIT_INPUTS={SCHED_MAX_SPLIT_INPUTS}' "
            f"-DCMAKE_EXE_LINKER_FLAGS='-L{CUDA_STUBS_DIR} -Wl,-rpath-link,{CUDA_STUBS_DIR}' "
            "-DCMAKE_BUILD_TYPE=Release"
        ),
        (
            f"cd {LLAMA_DIR} && LIBRARY_PATH={CUDA_STUBS_DIR}:$LIBRARY_PATH "
            "cmake --build build -j --target llama-server llama-cli llama-bench test-backend-ops"
        ),
        f"LD_LIBRARY_PATH={CUDA_STUBS_DIR}:$LD_LIBRARY_PATH {LLAMA_DIR}/build/bin/llama-server --help | head -80",
    )
)

source_probe_image = (
    modal.Image.from_registry(
        "ubuntu:22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("ca-certificates", "git")
    .env(remote_env)
    .run_commands(
        (
            f"git clone --depth=1 --branch {SOURCE_PROBE_BRANCH} "
            f"{SOURCE_PROBE_REPO} {SOURCE_PROBE_DIR}"
        ),
        f"cd {SOURCE_PROBE_DIR} && git rev-parse HEAD && git status --short",
    )
)

app = modal.App(APP_NAME)


def _model_exists() -> bool:
    path = Path(MODEL_PATH)
    return path.exists() and path.stat().st_size > 1024 * 1024 * 1024


def _download_hint() -> str:
    return (
        f"rtk modal run {Path(__file__).name} --action download "
        f"# {HF_REPO}:{MODEL_QUANT} into {MODEL_VOLUME_NAME}"
    )


def _require_model() -> None:
    if not _model_exists():
        raise RuntimeError(
            f"Missing {MODEL_QUANT} model shards at {MODEL_PATH}. Run: {_download_hint()}"
        )


def _server_cmd() -> list[str]:
    cmd = [
        f"{LLAMA_DIR}/build/bin/llama-server",
        "-m",
        MODEL_PATH,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "-c",
        str(CTX_TOKENS),
        "-ngl",
        N_GPU_LAYERS,
        "--split-mode",
        SPLIT_MODE,
        "--parallel",
        str(PARALLEL),
        "--jinja",
    ]
    if CACHE_TYPE_K:
        cmd.extend(["--cache-type-k", CACHE_TYPE_K])
    if CACHE_TYPE_V:
        cmd.extend(["--cache-type-v", CACHE_TYPE_V])
    if TENSOR_SPLIT:
        cmd.extend(["--tensor-split", TENSOR_SPLIT])
    if MAIN_GPU:
        cmd.extend(["--main-gpu", MAIN_GPU])
    if EXTRA_SERVER_ARGS:
        cmd.extend(EXTRA_SERVER_ARGS.split())

    return cmd


@app.function(
    image=llama_image,
    timeout=12 * 60 * MINUTES,
    cpu=8,
    memory=32768,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={MODEL_DIR: model_volume},
    secrets=download_secrets,
)
def download_model() -> None:
    cmd = [
        "hf",
        "download",
        HF_REPO,
        "--include",
        f"{MODEL_QUANT}/*",
        "--local-dir",
        MODEL_DIR,
    ]
    print("Downloading model shards:", json.dumps(cmd), flush=True)
    subprocess.run(cmd, check=True)

    if not _model_exists():
        raise RuntimeError(f"Expected first GGUF shard was not downloaded: {MODEL_PATH}")

    model_volume.commit()
    print(f"Downloaded {HF_REPO}:{MODEL_QUANT} into {MODEL_DIR}", flush=True)


@app.function(
    image=llama_image,
    timeout=10 * MINUTES,
    volumes={MODEL_DIR: model_volume},
)
def inspect_model_volume() -> None:
    root = Path(MODEL_DIR)
    print(f"Listing {root}", flush=True)
    for path in sorted(root.rglob("*.gguf")):
        print(
            f"{path.relative_to(root)}\t{path.stat().st_size / (1024**3):.2f} GiB",
            flush=True,
        )


@app.function(
    image=llama_image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    timeout=60 * MINUTES,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
)
def backend_ops() -> None:
    cmd = [
        f"{LLAMA_DIR}/build/bin/test-backend-ops",
        "-o",
        ",".join(
            [
                "DSV4_ROPE_TAIL",
                "DSV4_HC_SPLIT_SINKHORN",
                "DSV4_HC_WEIGHTED_SUM",
                "DSV4_HC_EXPAND",
                "DSV4_FP8_KV_QUANTIZE",
            ]
        ),
    ]
    print("Running llama.cpp DSV4 backend op tests:", json.dumps(cmd), flush=True)
    subprocess.run(cmd, cwd=LLAMA_DIR, check=True)


@app.function(
    image=llama_image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    timeout=12 * 60 * MINUTES,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=1,
    volumes={MODEL_DIR: model_volume},
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT)
def serve() -> None:
    _require_model()

    cmd = _server_cmd()
    print("Starting llama-server:", json.dumps(cmd), flush=True)
    subprocess.Popen(cmd, cwd=LLAMA_DIR, stdout=sys.stdout, stderr=sys.stderr)


@app.function(
    image=llama_image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    timeout=12 * 60 * MINUTES,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={MODEL_DIR: model_volume},
)
def debug_server() -> None:
    _require_model()

    cmd = _server_cmd()
    print("Running llama-server in foreground:", json.dumps(cmd), flush=True)
    subprocess.run(cmd, cwd=LLAMA_DIR, check=True)


@app.function(
    image=llama_image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    timeout=12 * 60 * MINUTES,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={MODEL_DIR: model_volume},
)
def cli_smoke(prompt: str = "What is 17*19? Answer briefly.") -> None:
    _require_model()

    cmd = [
        f"{LLAMA_DIR}/build/bin/llama-cli",
        "-m",
        MODEL_PATH,
        "-p",
        prompt,
        "-n",
        os.environ.get("LLAMA_CPP_SMOKE_TOKENS", "32"),
        "-c",
        os.environ.get("LLAMA_CPP_SMOKE_CTX", "2048"),
        "-ngl",
        N_GPU_LAYERS,
        "--split-mode",
        SPLIT_MODE,
        "--jinja",
        "--no-conversation",
    ]
    if CACHE_TYPE_K:
        cmd.extend(["--cache-type-k", CACHE_TYPE_K])
    if CACHE_TYPE_V:
        cmd.extend(["--cache-type-v", CACHE_TYPE_V])
    if TENSOR_SPLIT:
        cmd.extend(["--tensor-split", TENSOR_SPLIT])
    if MAIN_GPU:
        cmd.extend(["--main-gpu", MAIN_GPU])

    print("Running llama-cli smoke:", json.dumps(cmd), flush=True)
    subprocess.run(cmd, cwd=LLAMA_DIR, check=True)


def _extract_content(body: str) -> str:
    parsed = json.loads(body)
    return parsed["choices"][0]["message"].get("content") or ""


def _make_chat_request(url: str, payload: dict[str, Any]) -> urllib.request.Request:
    return urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _extract_delta_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return (delta.get("content") or "") + (delta.get("reasoning_content") or "")


def _post_chat_stream(url: str, payload: dict[str, Any], deadline: float) -> dict[str, Any]:
    request_payload = dict(payload)
    request_payload["stream"] = True
    request_payload["stream_options"] = {"include_usage": True}

    started = time.perf_counter()
    loading_retries = 0
    while True:
        ttft: float | None = None
        chunks = 0
        chars = 0
        usage: dict[str, Any] = {}
        try:
            with urllib.request.urlopen(
                _make_chat_request(url, request_payload), timeout=60 * MINUTES
            ) as resp:
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
                    text = _extract_delta_text(chunk)
                    if text:
                        chunks += 1
                        chars += len(text)
                        if ttft is None:
                            ttft = time.perf_counter() - started
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8")
            if exc.code == 503 and "Loading model" in error_body and time.time() <= deadline:
                loading_retries += 1
                print("server still loading model; retrying stream request", flush=True)
                time.sleep(10)
                continue
            raise urllib.error.HTTPError(
                exc.url,
                exc.code,
                exc.msg,
                exc.headers,
                io.BytesIO(error_body.encode("utf-8")),
            ) from RuntimeError(error_body)

        elapsed = time.perf_counter() - started
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
        if chunks == 0 and not usage:
            raise RuntimeError("stream ended without tokens or usage")
        decode_window = elapsed - ttft if ttft is not None else elapsed
        return {
            "elapsed_seconds": round(elapsed, 3),
            "loading_retries": loading_retries,
            "ttft_seconds": round(ttft, 3) if ttft is not None else None,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
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
            "stream_chunks": chunks,
            "stream_chars": chars,
            "usage": usage,
        }


def _post_chat(url: str, payload: dict[str, Any], deadline: float) -> dict[str, Any]:
    started = time.perf_counter()
    loading_retries = 0
    while True:
        try:
            with urllib.request.urlopen(_make_chat_request(url, payload), timeout=60 * MINUTES) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8")
            if exc.code == 503 and "Loading model" in error_body and time.time() <= deadline:
                loading_retries += 1
                print("server still loading model; retrying request", flush=True)
                time.sleep(10)
                continue
            raise urllib.error.HTTPError(
                exc.url,
                exc.code,
                exc.msg,
                exc.headers,
                io.BytesIO(error_body.encode("utf-8")),
            ) from RuntimeError(error_body)

        elapsed = time.perf_counter() - started
        return {
            "elapsed_seconds": round(elapsed, 3),
            "loading_retries": loading_retries,
            "raw_response": json.loads(body),
            "content": _extract_content(body),
        }


def _benchmark_prompt(target_tokens: int) -> str:
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


def _parse_matrix(raw: str) -> list[tuple[int, int]]:
    cases: list[tuple[int, int]] = []
    for item in raw.split(","):
        prompt_tokens, max_tokens = item.split("x", 1)
        cases.append((int(prompt_tokens), int(max_tokens)))
    return cases


def _summarize_numbers(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "min": None,
            "p50": None,
            "median": None,
            "p90": None,
            "p99": None,
            "max": None,
            "mean": None,
            "stdev": None,
        }
    sorted_values = sorted(values)

    def percentile(q: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        pos = (len(sorted_values) - 1) * q
        lower = int(pos)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = pos - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    return {
        "min": sorted_values[0],
        "p50": percentile(0.50),
        "median": median(sorted_values),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": sorted_values[-1],
        "mean": fmean(values),
        "stdev": stdev(values) if len(values) > 1 else 0.0,
    }


def _benchmark_payloads(max_tokens: int) -> list[tuple[str, dict[str, Any]]]:
    sky_prompt = "In a short summary, explain why the sky is blue in scientific terms"
    return [
        (
            "warmup_arithmetic",
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "What is 17*19? Answer briefly."}],
                "temperature": 0,
                "max_tokens": 16,
                "stream": False,
            },
        ),
        (
            "sky_thinking_off",
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": sky_prompt}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": False,
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        ),
        (
            "sky_thinking_on",
            {
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": sky_prompt}],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": False,
                "reasoning_effort": "medium",
                "chat_template_kwargs": {"enable_thinking": True},
            },
        ),
    ]


def _source_snippets(
    path: Path,
    patterns: list[str],
    context: int = 4,
    root_dir: str = LLAMA_DIR,
) -> list[dict[str, Any]]:
    if not path.exists():
        return [{"path": str(path), "error": "file not found"}]
    lines = path.read_text(errors="replace").splitlines()
    matches = []
    lowered = [(pattern, pattern.lower()) for pattern in patterns]
    for idx, line in enumerate(lines):
        line_lower = line.lower()
        if any(pattern_lower in line_lower for _, pattern_lower in lowered):
            start = max(0, idx - context)
            end = min(len(lines), idx + context + 1)
            matches.append(
                {
                    "path": str(path.relative_to(root_dir)),
                    "line": idx + 1,
                    "match": line.strip(),
                    "snippet": [
                        {"line": line_no + 1, "text": lines[line_no]}
                        for line_no in range(start, end)
                    ],
                }
            )
    return matches


def _source_tree_matches(
    patterns: list[str],
    max_matches: int = 120,
    root_dir: str = LLAMA_DIR,
) -> list[dict[str, Any]]:
    roots = [
        Path(root_dir) / "src",
        Path(root_dir) / "common",
        Path(root_dir) / "tools/server",
    ]
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp"}
    matches = []
    lowered = [(pattern, pattern.lower()) for pattern in patterns]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in suffixes:
                continue
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for idx, line in enumerate(lines):
                line_lower = line.lower()
                hit = [pattern for pattern, pattern_lower in lowered if pattern_lower in line_lower]
                if not hit:
                    continue
                matches.append(
                    {
                        "path": str(path.relative_to(root_dir)),
                        "line": idx + 1,
                        "patterns": hit,
                        "text": line.strip(),
                    }
                )
                if len(matches) >= max_matches:
                    return matches
    return matches


def _source_inspect_result(root_dir: str) -> dict[str, Any]:
    files = [
        (
            Path(root_dir) / "src/llama-context.cpp",
            [
                "const uint32_t n_seqs = model.arch == LLM_ARCH_DEEPSEEK4 ? 1 : cparams.n_seq_max;",
                "DeepSeek4: forcing fp16 KV cache",
                "model->split_mode() == LLAMA_SPLIT_MODE_TENSOR",
            ],
        ),
        (
            Path(root_dir) / "src/models/deepseek4.cpp",
            [
                "n_seq_max",
                "n_comp_visible",
                "n_comp_cache",
                "nextn",
                "cache-type",
                "fp16 KV",
                "f16 KV",
            ],
        ),
        (
            Path(root_dir) / "src/llama-arch.cpp",
            ["LLM_ARCH_DEEPSEEK4"],
        ),
        (
            Path(root_dir) / "src/llama-model.cpp",
            ["LLM_ARCH_DEEPSEEK4", "cache_type", "f16", "force"],
        ),
        (
            Path(root_dir) / "common/speculative.cpp",
            ["deepseek", "nextn", "mtp", "draft", "speculative"],
        ),
        (
            Path(root_dir) / "tools/server/server.cpp",
            ["speculative", "draft", "nextn", "mtp"],
        ),
    ]
    git_commit = subprocess.run(
        ["git", "-C", root_dir, "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()
    result: dict[str, Any] = {
        "llama_dir": root_dir,
        "git_commit": git_commit,
        "tree_matches": _source_tree_matches(
            [
                "supports_tensor_split",
                "LLAMA_SPLIT_MODE_TENSOR",
                "LLM_ARCH_DEEPSEEK4",
                "nextn_predict_layers",
                "deepseek_mtp",
                "deepseek4-mtp",
                "cache_type_k",
                "cache_type_v",
                "forcing fp16 KV",
            ],
            root_dir=root_dir,
        ),
        "files": {},
    }
    for path, patterns in files:
        result["files"][str(path.relative_to(root_dir))] = _source_snippets(
            path,
            patterns,
            context=12,
            root_dir=root_dir,
        )
    return result


@app.function(image=llama_image, timeout=30 * MINUTES)
def source_inspect() -> None:
    result = _source_inspect_result(LLAMA_DIR)
    print("source_inspect_results=" + json.dumps(result, indent=2), flush=True)


@app.function(image=source_probe_image, timeout=30 * MINUTES)
def latest_source_probe() -> None:
    result = _source_inspect_result(SOURCE_PROBE_DIR)
    result["source_probe_repo"] = SOURCE_PROBE_REPO
    result["source_probe_branch"] = SOURCE_PROBE_BRANCH
    print("latest_source_probe_results=" + json.dumps(result, indent=2), flush=True)


@app.function(
    image=llama_image,
    gpu=GPU_CONFIG,
    cpu=CPU_COUNT,
    memory=MEMORY_MIB,
    timeout=12 * 60 * MINUTES,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={MODEL_DIR: model_volume},
)
def offline_bench() -> None:
    _require_model()

    matrix = os.environ.get("LLAMA_CPP_BENCH_MATRIX", "512x1,2048x1,128x256,1024x256")
    repeats = int(os.environ.get("LLAMA_CPP_BENCH_REPEATS", "5"))
    results = []
    for prompt_tokens, gen_tokens in _parse_matrix(matrix):
        cmd = [
            f"{LLAMA_DIR}/build/bin/llama-bench",
            "-m",
            MODEL_PATH,
            "-p",
            str(prompt_tokens),
            "-n",
            str(gen_tokens),
            "-r",
            str(repeats),
            "-b",
            os.environ.get("LLAMA_CPP_BENCH_BATCH", "2048"),
            "-ub",
            os.environ.get("LLAMA_CPP_BENCH_UBATCH", "512"),
            "-ngl",
            N_GPU_LAYERS,
            "-sm",
            SPLIT_MODE,
            "-o",
            "json",
        ]
        if CACHE_TYPE_K:
            cmd.extend(["-ctk", CACHE_TYPE_K])
        if CACHE_TYPE_V:
            cmd.extend(["-ctv", CACHE_TYPE_V])
        if TENSOR_SPLIT:
            cmd.extend(["-ts", TENSOR_SPLIT])
        if MAIN_GPU:
            cmd.extend(["-mg", MAIN_GPU])

        print("Running offline llama-bench:", json.dumps(cmd), flush=True)
        proc = subprocess.run(
            cmd,
            cwd=LLAMA_DIR,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        print(proc.stdout, flush=True)
        record: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens,
            "returncode": proc.returncode,
            "command": cmd,
            "raw_output": proc.stdout,
        }
        if proc.returncode:
            record["error"] = "llama-bench returned non-zero status"
        results.append(record)

    print("offline_bench_results=" + json.dumps(results, indent=2), flush=True)
    if any(result["returncode"] for result in results):
        raise RuntimeError("one or more offline llama-bench cases failed")


@app.local_entrypoint()
async def main(action: str = "url", prompt: str = "What is 17*19? Answer briefly.") -> None:
    if action == "download":
        download_model.remote()
        return

    if action == "inspect":
        inspect_model_volume.remote()
        return

    if action == "backend-ops":
        backend_ops.remote()
        return

    if action == "source-inspect":
        source_inspect.remote()
        return

    if action == "latest-source-probe":
        latest_source_probe.remote()
        return

    if action == "debug":
        debug_server.remote()
        return

    if action == "cli-smoke":
        cli_smoke.remote(prompt)
        return

    if action == "offline-bench":
        offline_bench.remote()
        return

    url = await serve.get_web_url.aio()
    print(f"OpenAI-compatible endpoint: {url}/v1")

    if action not in {"test", "benchmark", "online-matrix"}:
        return

    deadline = time.time() + STARTUP_TIMEOUT
    while True:
        try:
            if action == "test":
                payload: dict[str, Any] = {
                    "model": "deepseek-v4-flash",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64,
                    "stream": False,
                }
                print(json.dumps(_post_chat(url, payload, deadline), indent=2), flush=True)
                return

            if action == "online-matrix":
                matrix = os.environ.get(
                    "LLAMA_CPP_ONLINE_MATRIX",
                    "512x1,2048x1,128x256,1024x256",
                )
                concurrency = int(os.environ.get("LLAMA_CPP_ONLINE_CONCURRENCY", "1"))
                results = []
                for prompt_target_tokens, max_tokens in _parse_matrix(matrix):
                    payload = {
                        "model": "deepseek-v4-flash",
                        "messages": [
                            {
                                "role": "user",
                                "content": _benchmark_prompt(prompt_target_tokens),
                            }
                        ],
                        "temperature": 0.2,
                        "max_tokens": max_tokens,
                        "stream": True,
                        "reasoning_effort": "none",
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                    started = time.perf_counter()
                    request_results = []
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futures = [
                            pool.submit(_post_chat_stream, url, payload, deadline)
                            for _ in range(concurrency)
                        ]
                        for future in as_completed(futures):
                            try:
                                request_results.append(future.result())
                            except Exception as exc:
                                request_results.append(
                                    {
                                        "elapsed_seconds": round(
                                            time.perf_counter() - started, 3
                                        ),
                                        "error_type": type(exc).__name__,
                                        "error": str(exc),
                                    }
                                )
                    wall = time.perf_counter() - started
                    completion_tokens = sum(
                        result.get("completion_tokens") or 0 for result in request_results
                    )
                    total_tokens = sum(result.get("total_tokens") or 0 for result in request_results)
                    decode_rates = [
                        result["decode_tok_s_online"]
                        for result in request_results
                        if result.get("decode_tok_s_online") is not None
                    ]
                    prefill_rates = [
                        result["prefill_tok_s_online"]
                        for result in request_results
                        if result.get("prefill_tok_s_online") is not None
                    ]
                    ttfts = [
                        result["ttft_seconds"]
                        for result in request_results
                        if result.get("ttft_seconds") is not None
                    ]
                    latencies = [
                        result["elapsed_seconds"]
                        for result in request_results
                        if result.get("elapsed_seconds") is not None
                    ]
                    record = {
                        "prompt_target_tokens": prompt_target_tokens,
                        "max_output_tokens": max_tokens,
                        "concurrency": concurrency,
                        "successful_requests": sum(
                            1 for result in request_results if "error" not in result
                        ),
                        "failed_requests": sum(
                            1 for result in request_results if "error" in result
                        ),
                        "wall_seconds": round(wall, 3),
                        "aggregate_completion_tok_s": (
                            completion_tokens / wall if wall > 0 else None
                        ),
                        "aggregate_total_tok_s": (
                            total_tokens / wall if wall > 0 and total_tokens else None
                        ),
                        "completion_tok_s_summary": _summarize_numbers(decode_rates),
                        "prefill_tok_s_online_summary": _summarize_numbers(prefill_rates),
                        "ttft_seconds_summary": _summarize_numbers(ttfts),
                        "latency_seconds_summary": _summarize_numbers(latencies),
                        "requests": request_results,
                    }
                    results.append(record)
                    print(json.dumps(record, indent=2), flush=True)

                completion_rates = [
                    request["decode_tok_s_online"]
                    for result in results
                    for request in result["requests"]
                    if request.get("decode_tok_s_online") is not None
                ]
                prefill_rates = [
                    request["prefill_tok_s_online"]
                    for result in results
                    for request in result["requests"]
                    if request.get("prefill_tok_s_online") is not None
                ]
                summary = {
                    "matrix": matrix,
                    "script": Path(__file__).name,
                    "model_quant": MODEL_QUANT,
                    "ctx_tokens": CTX_TOKENS,
                    "parallel": PARALLEL,
                    "concurrency": concurrency,
                    "split_mode": SPLIT_MODE,
                    "cache_type_k_requested": CACHE_TYPE_K,
                    "completion_tok_s_summary": _summarize_numbers(completion_rates),
                    "prefill_tok_s_online_summary": _summarize_numbers(prefill_rates),
                    "results": results,
                }
                print("online_matrix_results=" + json.dumps(summary, indent=2), flush=True)
                return

            max_tokens = int(os.environ.get("LLAMA_CPP_BENCHMARK_TOKENS", "256"))
            results = []
            for name, payload in _benchmark_payloads(max_tokens):
                try:
                    result = _post_chat(url, payload, deadline)
                    used_payload = payload
                except urllib.error.HTTPError as exc:
                    error_body = exc.read().decode("utf-8")
                    if "chat_template_kwargs" not in json.dumps(payload):
                        print(error_body, flush=True)
                        raise
                    fallback_payload = dict(payload)
                    fallback_payload.pop("chat_template_kwargs", None)
                    fallback_payload.pop("reasoning_effort", None)
                    result = _post_chat(url, fallback_payload, deadline)
                    used_payload = fallback_payload
                    result["thinking_controls_accepted"] = False
                    result["thinking_control_error"] = error_body
                else:
                    result["thinking_controls_accepted"] = "chat_template_kwargs" in payload
                record = {
                    "name": name,
                    "query": used_payload["messages"],
                    "max_tokens": used_payload["max_tokens"],
                    **result,
                }
                results.append(record)
                print(json.dumps(record, indent=2), flush=True)

            print("benchmark_results=" + json.dumps(results, indent=2), flush=True)
            return
        except urllib.error.HTTPError as exc:
            print(exc.read().decode("utf-8"))
            raise
        except urllib.error.URLError:
            if time.time() > deadline:
                raise
            time.sleep(10)
