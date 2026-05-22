from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import modal


APP_NAME = "deepseek-v4-flash-llama-cpp"
LLAMA_REPO = os.environ.get("LLAMA_CPP_REPO", "https://github.com/cchuter/llama.cpp.git")
LLAMA_BRANCH = os.environ.get("LLAMA_CPP_BRANCH", "feat/v4-port-cuda")
LLAMA_DIR = "/opt/llama.cpp"
MODEL_DIR = "/models"
CUDA_STUBS_DIR = "/usr/local/cuda/targets/x86_64-linux/lib/stubs"

HF_REPO = os.environ.get("LLAMA_CPP_HF_REPO", "teamblobfish/DeepSeek-V4-Flash-GGUF")
HF_SECRET_NAME = os.environ.get("LLAMA_CPP_HF_SECRET", "")
MODEL_QUANT = os.environ.get("LLAMA_CPP_MODEL_QUANT", "Q2_K-XL")
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

GPU_CONFIG = os.environ.get("LLAMA_CPP_GPU", "H100:2")
CUDA_ARCH = os.environ.get("LLAMA_CPP_CUDA_ARCH", "90")
SCHED_MAX_SPLIT_INPUTS = int(os.environ.get("LLAMA_CPP_SCHED_MAX_SPLIT_INPUTS", "128"))
CPU_COUNT = float(os.environ.get("LLAMA_CPP_CPU", "32"))
MEMORY_MIB = int(os.environ.get("LLAMA_CPP_MEMORY_MIB", "327680"))
EPHEMERAL_DISK_MIB = int(os.environ.get("LLAMA_CPP_EPHEMERAL_DISK_MIB", "1048576"))
STARTUP_TIMEOUT = int(os.environ.get("LLAMA_CPP_STARTUP_TIMEOUT", str(4 * 60 * MINUTES)))

CTX_TOKENS = int(os.environ.get("LLAMA_CPP_CTX", "32768"))
PARALLEL = int(os.environ.get("LLAMA_CPP_PARALLEL", "1"))
N_GPU_LAYERS = os.environ.get("LLAMA_CPP_N_GPU_LAYERS", "999")
TENSOR_SPLIT = os.environ.get("LLAMA_CPP_TENSOR_SPLIT", "")
MAIN_GPU = os.environ.get("LLAMA_CPP_MAIN_GPU", "")
EXTRA_SERVER_ARGS = os.environ.get("LLAMA_CPP_EXTRA_SERVER_ARGS", "")

REMOTE_ENV_KEYS = (
    "LLAMA_CPP_REPO",
    "LLAMA_CPP_BRANCH",
    "LLAMA_CPP_HF_REPO",
    "LLAMA_CPP_HF_SECRET",
    "LLAMA_CPP_MODEL_QUANT",
    "LLAMA_CPP_GPU",
    "LLAMA_CPP_CUDA_ARCH",
    "LLAMA_CPP_SCHED_MAX_SPLIT_INPUTS",
    "LLAMA_CPP_CPU",
    "LLAMA_CPP_MEMORY_MIB",
    "LLAMA_CPP_EPHEMERAL_DISK_MIB",
    "LLAMA_CPP_STARTUP_TIMEOUT",
    "LLAMA_CPP_CTX",
    "LLAMA_CPP_PARALLEL",
    "LLAMA_CPP_N_GPU_LAYERS",
    "LLAMA_CPP_TENSOR_SPLIT",
    "LLAMA_CPP_MAIN_GPU",
    "LLAMA_CPP_EXTRA_SERVER_ARGS",
    "LLAMA_CPP_MODEL_VOLUME",
    "LLAMA_CPP_SMOKE_TOKENS",
    "LLAMA_CPP_SMOKE_CTX",
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
            f"-DCMAKE_C_FLAGS='-DGGML_SCHED_MAX_SPLIT_INPUTS={SCHED_MAX_SPLIT_INPUTS}' "
            f"-DCMAKE_CXX_FLAGS='-DGGML_SCHED_MAX_SPLIT_INPUTS={SCHED_MAX_SPLIT_INPUTS}' "
            f"-DCMAKE_EXE_LINKER_FLAGS='-L{CUDA_STUBS_DIR} -Wl,-rpath-link,{CUDA_STUBS_DIR}' "
            "-DCMAKE_BUILD_TYPE=Release"
        ),
        (
            f"cd {LLAMA_DIR} && LIBRARY_PATH={CUDA_STUBS_DIR}:$LIBRARY_PATH "
            "cmake --build build -j --target llama-server llama-cli"
        ),
        f"LD_LIBRARY_PATH={CUDA_STUBS_DIR}:$LD_LIBRARY_PATH {LLAMA_DIR}/build/bin/llama-server --help | head -80",
    )
)

app = modal.App(APP_NAME)


def _model_exists() -> bool:
    path = Path(MODEL_PATH)
    return path.exists() and path.stat().st_size > 1024 * 1024 * 1024


def _download_hint() -> str:
    env_parts = [
        f"LLAMA_CPP_MODEL_QUANT={MODEL_QUANT}",
        f"LLAMA_CPP_GPU={GPU_CONFIG}",
        f"LLAMA_CPP_CUDA_ARCH={CUDA_ARCH}",
    ]
    if TENSOR_SPLIT:
        env_parts.append(f"LLAMA_CPP_TENSOR_SPLIT={TENSOR_SPLIT}")
    return " ".join(
        [
            *env_parts,
            "rtk",
            "modal",
            "run",
            "llama_cpp_v4_modal.py",
            "--action",
            "download",
        ]
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
        "--parallel",
        str(PARALLEL),
        "--jinja",
    ]

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
    include = f"{MODEL_QUANT}/*"
    cmd = [
        "hf",
        "download",
        HF_REPO,
        "--include",
        include,
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
    timeout=12 * 60 * MINUTES,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    scaledown_window=30 * MINUTES,
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
def cli_smoke(prompt: str = "What is 17*19?") -> None:
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
        "--jinja",
    ]
    if TENSOR_SPLIT:
        cmd.extend(["--tensor-split", TENSOR_SPLIT])
    if MAIN_GPU:
        cmd.extend(["--main-gpu", MAIN_GPU])

    print("Running llama-cli smoke:", json.dumps(cmd), flush=True)
    subprocess.run(cmd, cwd=LLAMA_DIR, check=True)


@app.local_entrypoint()
async def main(action: str = "url", prompt: str = "What is 17*19?") -> None:
    if action == "download":
        download_model.remote()
        return

    if action == "inspect":
        inspect_model_volume.remote()
        return

    if action == "debug":
        debug_server.remote()
        return

    if action == "cli-smoke":
        cli_smoke.remote(prompt)
        return

    url = await serve.get_web_url.aio()
    print(f"OpenAI-compatible endpoint: {url}/v1")

    if action != "test":
        return

    payload: dict[str, object] = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    deadline = time.time() + STARTUP_TIMEOUT
    while True:
        try:
            with urllib.request.urlopen(req, timeout=60 * MINUTES) as resp:
                print(resp.read().decode("utf-8"))
                return
        except urllib.error.HTTPError as exc:
            print(exc.read().decode("utf-8"))
            raise
        except urllib.error.URLError:
            if time.time() > deadline:
                raise
            time.sleep(10)
