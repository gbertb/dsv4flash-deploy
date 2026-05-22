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
from typing import Any

import modal


APP_NAME = os.environ.get("LLAMA_CPP_APP_NAME", "deepseek-v4-flash-llama-cpp-q4-a100")
LLAMA_REPO = os.environ.get("LLAMA_CPP_REPO", "https://github.com/cchuter/llama.cpp.git")
LLAMA_BRANCH = os.environ.get("LLAMA_CPP_BRANCH", "feat/v4-port-cuda")
LLAMA_DIR = "/opt/llama.cpp"
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
            "cmake --build build -j --target llama-server llama-cli test-backend-ops"
        ),
        f"LD_LIBRARY_PATH={CUDA_STUBS_DIR}:$LD_LIBRARY_PATH {LLAMA_DIR}/build/bin/llama-server --help | head -80",
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

    if action == "debug":
        debug_server.remote()
        return

    if action == "cli-smoke":
        cli_smoke.remote(prompt)
        return

    url = await serve.get_web_url.aio()
    print(f"OpenAI-compatible endpoint: {url}/v1")

    if action not in {"test", "benchmark"}:
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
