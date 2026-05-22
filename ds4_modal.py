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


APP_NAME = "deepseek-v4-flash-ds4"
DS4_REPO = "https://github.com/antirez/ds4.git"
DS4_DIR = "/opt/ds4"

MODEL_FILES = {
    "q2-imatrix": (
        "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-"
        "OutQ8-chat-v2-imatrix.gguf"
    ),
    "q4-imatrix": (
        "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-"
        "F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf"
    ),
    "q2": "DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf",
    "q4": (
        "DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-"
        "F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf"
    ),
}
MODEL_KIND = os.environ.get("DS4_MODEL_KIND", "q2-imatrix")
if MODEL_KIND not in MODEL_FILES:
    raise ValueError(f"DS4_MODEL_KIND must be one of: {', '.join(MODEL_FILES)}")
MODEL_FILE = MODEL_FILES[MODEL_KIND]
MODEL_PATH = f"{DS4_DIR}/gguf/{MODEL_FILE}"

PORT = 8000
MINUTES = 60

# DS4 q4-imatrix is documented for >= 256 GB RAM machines. Keep a little
# headroom by default; lower this only if Modal capacity requires it.
MEMORY_MIB = int(os.environ.get("DS4_MEMORY_MIB", "327680"))
CPU_COUNT = float(os.environ.get("DS4_CPU", "32"))
GPU_CONFIG = os.environ.get("DS4_GPU", "A100-80GB:4")
CUDA_ARCH = os.environ.get("DS4_CUDA_ARCH", "sm_80")
BACKEND = os.environ.get("DS4_BACKEND", "cpu").lower()
if BACKEND not in {"cpu", "cuda"}:
    raise ValueError("DS4_BACKEND must be 'cpu' or 'cuda'")

CTX_TOKENS = int(os.environ.get("DS4_CTX", "32768"))
DEFAULT_TOKENS = int(
    os.environ.get("DS4_DEFAULT_TOKENS", "256" if BACKEND == "cpu" else "4096")
)
SMOKE_CTX = int(os.environ.get("DS4_SMOKE_CTX", "512"))
SMOKE_TOKENS = int(
    os.environ.get("DS4_SMOKE_TOKENS", "4" if BACKEND == "cpu" else "64")
)
KV_DISK_SPACE_MB = int(os.environ.get("DS4_KV_DISK_SPACE_MB", "65536"))
EPHEMERAL_DISK_MIB = int(os.environ.get("DS4_EPHEMERAL_DISK_MIB", "1048576"))
STARTUP_TIMEOUT = int(os.environ.get("DS4_STARTUP_TIMEOUT", str(4 * 60 * MINUTES)))

GGUF_VOLUME_NAME = os.environ.get("DS4_GGUF_VOLUME", f"ds4-{MODEL_KIND}-gguf")
gguf_volume = modal.Volume.from_name(GGUF_VOLUME_NAME, create_if_missing=True)
kv_volume = modal.Volume.from_name("ds4-kv-cache", create_if_missing=True)

remote_env = {
    "DS4_MODEL_KIND": MODEL_KIND,
    "DS4_BACKEND": BACKEND,
    "DS4_GPU": GPU_CONFIG,
    "DS4_CUDA_ARCH": CUDA_ARCH,
    "DS4_CTX": str(CTX_TOKENS),
    "DS4_DEFAULT_TOKENS": str(DEFAULT_TOKENS),
    "DS4_SMOKE_CTX": str(SMOKE_CTX),
    "DS4_SMOKE_TOKENS": str(SMOKE_TOKENS),
    "DS4_KV_DISK_SPACE_MB": str(KV_DISK_SPACE_MB),
    "DS4_EPHEMERAL_DISK_MIB": str(EPHEMERAL_DISK_MIB),
    "DS4_STARTUP_TIMEOUT": str(STARTUP_TIMEOUT),
    "DS4_GGUF_VOLUME": GGUF_VOLUME_NAME,
}

ds4_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .env(remote_env)
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "git",
        "make",
    )
    .run_commands(
        f"git clone --depth=1 {DS4_REPO} {DS4_DIR}",
        f"cd {DS4_DIR} && make cuda CUDA_ARCH={CUDA_ARCH} "
        "NATIVE_CPU_FLAG=-march=x86-64-v3",
        f"cd {DS4_DIR} && ./ds4-server --help | head -80",
    )
)

app = modal.App(APP_NAME)

serve_kwargs: dict[str, object] = {
    "image": ds4_image,
    "cpu": CPU_COUNT,
    "memory": MEMORY_MIB,
    "timeout": 12 * 60 * MINUTES,
    "startup_timeout": STARTUP_TIMEOUT,
    "ephemeral_disk": EPHEMERAL_DISK_MIB,
    "volumes": {
        f"{DS4_DIR}/gguf": gguf_volume,
        f"{DS4_DIR}/kv-cache": kv_volume,
    },
}
if BACKEND == "cuda":
    serve_kwargs["gpu"] = GPU_CONFIG

smoke_kwargs: dict[str, object] = {
    "image": ds4_image,
    "cpu": CPU_COUNT,
    "memory": MEMORY_MIB,
    "timeout": 12 * 60 * MINUTES,
    "startup_timeout": STARTUP_TIMEOUT,
    "ephemeral_disk": EPHEMERAL_DISK_MIB,
    "volumes": {f"{DS4_DIR}/gguf": gguf_volume},
}
if BACKEND == "cuda":
    smoke_kwargs["gpu"] = GPU_CONFIG


def _model_exists() -> bool:
    path = Path(MODEL_PATH)
    return path.exists() and path.stat().st_size > 1024 * 1024 * 1024


def _backend_flag() -> str:
    return "--cuda" if BACKEND == "cuda" else "--cpu"


def _server_cmd() -> list[str]:
    cmd = [
        f"{DS4_DIR}/ds4-server",
        "--model",
        MODEL_PATH,
        _backend_flag(),
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--ctx",
        str(CTX_TOKENS),
        "--tokens",
        str(DEFAULT_TOKENS),
        "--kv-disk-dir",
        f"{DS4_DIR}/kv-cache",
        "--kv-disk-space-mb",
        str(KV_DISK_SPACE_MB),
        "--kv-cache-reject-different-quant",
        "--cors",
    ]

    if os.environ.get("DS4_WARM_WEIGHTS") == "1":
        cmd.append("--warm-weights")

    trace_path = os.environ.get("DS4_TRACE_PATH")
    if trace_path:
        cmd.extend(["--trace", trace_path])

    return cmd


@app.function(
    image=ds4_image,
    timeout=12 * 60 * MINUTES,
    cpu=8,
    memory=32768,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={f"{DS4_DIR}/gguf": gguf_volume},
)
def download_model() -> None:
    """Download antirez/ds4 q4-imatrix GGUF into a Modal Volume."""

    env = os.environ.copy()
    env["DS4_GGUF_DIR"] = f"{DS4_DIR}/gguf"
    cmd = [f"{DS4_DIR}/download_model.sh", MODEL_KIND]
    print("Downloading DS4 model:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=DS4_DIR, env=env, check=True)

    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        raise RuntimeError(f"Expected model was not downloaded: {MODEL_PATH}")

    print(
        f"Downloaded {MODEL_PATH} ({model_path.stat().st_size / (1024**3):.2f} GiB)",
        flush=True,
    )
    gguf_volume.commit()


@app.function(
    image=ds4_image,
    timeout=10 * MINUTES,
    volumes={f"{DS4_DIR}/gguf": gguf_volume},
)
def inspect_model_volume() -> None:
    """Print model files currently present in the GGUF volume."""

    root = Path(f"{DS4_DIR}/gguf")
    print(f"Listing {root}", flush=True)
    for path in sorted(root.glob("*")):
        if path.is_file():
            print(f"{path.name}\t{path.stat().st_size / (1024**3):.2f} GiB", flush=True)


@app.function(**serve_kwargs, scaledown_window=30 * MINUTES, max_containers=1)
@modal.concurrent(max_inputs=8)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT)
def serve() -> None:
    if not _model_exists():
        raise RuntimeError(
            f"Missing {MODEL_PATH}. Run: modal run ds4_modal.py --action download"
        )

    cmd = _server_cmd()
    print("Starting ds4-server:", json.dumps(cmd), flush=True)
    subprocess.Popen(cmd, cwd=DS4_DIR, stdout=sys.stdout, stderr=sys.stderr)


@app.function(**serve_kwargs)
def debug_server() -> None:
    if not _model_exists():
        raise RuntimeError(
            f"Missing {MODEL_PATH}. Run: modal run ds4_modal.py --action download"
        )

    cmd = _server_cmd()
    print("Running ds4-server in foreground:", json.dumps(cmd), flush=True)
    subprocess.run(cmd, cwd=DS4_DIR, check=True)


@app.function(**smoke_kwargs)
def cli_smoke(prompt: str = "Explain Redis streams in one paragraph.") -> None:
    if not _model_exists():
        raise RuntimeError(
            f"Missing {MODEL_PATH}. Run: modal run ds4_modal.py --action download"
        )

    cmd = [
        f"{DS4_DIR}/ds4",
        _backend_flag(),
        "-m",
        MODEL_PATH,
        "--ctx",
        str(SMOKE_CTX),
        "--nothink",
        "-n",
        str(SMOKE_TOKENS),
        "-p",
        prompt,
    ]
    print("Running DS4 CLI smoke:", json.dumps(cmd), flush=True)
    subprocess.run(cmd, cwd=DS4_DIR, check=True)


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
        "reasoning_effort": "none",
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
            if time.time() >= deadline:
                raise
            time.sleep(5)
