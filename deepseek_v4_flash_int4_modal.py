from __future__ import annotations

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


APP_NAME = "deepseek-v4-flash-base-int4"
MODEL_REPO = "EnsueAI/DeepSeek-V4-Flash-Base-INT4"
INFERENCE_REPO = "deepseek-ai/DeepSeek-V4-Flash"
SERVED_MODEL_NAME = "deepseek-v4-flash-base-int4"

N_GPU = 4
GPU_CONFIG = "A100-80GB:4"
PORT = 8000
MINUTES = 60
STARTUP_TIMEOUT = int(os.environ.get("DSV4_INT4_STARTUP_TIMEOUT", str(4 * 60 * MINUTES)))
TIMEOUT = int(os.environ.get("DSV4_INT4_TIMEOUT", str(12 * 60 * MINUTES)))
EPHEMERAL_DISK_MIB = int(os.environ.get("DSV4_INT4_EPHEMERAL_DISK_MIB", "1048576"))
SCALEDOWN_WINDOW = int(os.environ.get("DSV4_INT4_SCALEDOWN_WINDOW", "60"))

VOLUME_ROOT = "/dsv4-int4-volume"
CKPT_DIR = f"{VOLUME_ROOT}/hf"
SRC_DIR = "/opt/deepseek-v4-flash"
SERVER_PATH = "/opt/dsv4_int4_server.py"
CONFIG_PATH = f"{SRC_DIR}/inference/config.json"

MODEL_VOLUME_NAME = os.environ.get("DSV4_INT4_MODEL_VOLUME", "deepseek-v4-flash-base-int4")
HF_SECRET_NAME = os.environ.get("DSV4_INT4_HF_SECRET", "")

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name(HF_SECRET_NAME)] if HF_SECRET_NAME else []

remote_env = {
    "HF_HOME": f"{VOLUME_ROOT}/.cache/huggingface",
    "HF_HUB_CACHE": f"{VOLUME_ROOT}/.cache/huggingface/hub",
    "HF_XET_CACHE": f"{VOLUME_ROOT}/.cache/huggingface/xet",
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

int4_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
        "git",
        "libgomp1",
        "ninja-build",
    )
    .run_commands(
        "python -m pip install --pre --upgrade torch --index-url https://download.pytorch.org/whl/nightly/cu128"
    )
    .pip_install(
        "hf_transfer",
        "huggingface_hub>=1.0.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "pydantic>=2.7.0",
        "safetensors>=0.7.0",
        "transformers>=5.0.0",
        "tilelang==0.1.8",
        "apache-tvm-ffi<0.1.8",
        "tqdm",
    )
    .run_commands(
        "python -m pip install wheel",
        "CC=gcc CXX=g++ MAX_JOBS=8 python -m pip install --no-build-isolation "
        "git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0",
        "python - <<'PY'\n"
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download({INFERENCE_REPO!r}, local_dir={SRC_DIR!r}, "
        "allow_patterns=['inference/*', 'encoding/*', 'tokenizer*'], "
        "local_dir_use_symlinks=False)\n"
        "PY",
    )
    .env(remote_env)
    .add_local_file("dsv4_int4_server.py", SERVER_PATH, copy=True)
)

app = modal.App(APP_NAME)


def _download_patterns() -> list[str]:
    return [
        "README.md",
        "model*-mp4.safetensors",
        "per_linear_quant.json",
        "quant_metadata.json",
        "recipe.yaml",
        "tokenizer.json",
        "tokenizer_config.json",
    ]


def _expected_files() -> list[Path]:
    return [Path(CKPT_DIR) / f"model{i}-mp4.safetensors" for i in range(N_GPU)]


def _checkpoint_exists() -> bool:
    return all(path.exists() and path.stat().st_size > 1024 * 1024 * 1024 for path in _expected_files())


@app.function(
    image=int4_image,
    timeout=8 * MINUTES,
    secrets=secrets,
)
def inspect_remote_model() -> None:
    code = f"""
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info({MODEL_REPO!r}, files_metadata=True)
print("repo", {MODEL_REPO!r})
print("sha", info.sha)
total = 0
for s in info.siblings:
    size = getattr(s, "size", None) or 0
    total += size
    print(s.rfilename, size)
print("total_gib", total / (1024**3))
"""
    subprocess.run(["python", "-c", code], check=True)


@app.function(
    image=int4_image,
    timeout=8 * MINUTES,
    volumes={VOLUME_ROOT: model_volume},
    secrets=secrets,
)
def inspect_volume() -> None:
    for root in (Path(CKPT_DIR), Path(f"{VOLUME_ROOT}/.cache/huggingface")):
        print(f"\n# {root}", flush=True)
        if not root.exists():
            print("missing", flush=True)
            continue
        files = sorted([p for p in root.rglob("*") if p.is_file()])
        total = sum(p.stat().st_size for p in files)
        print(f"{len(files)} files, {total / (1024**3):.2f} GiB", flush=True)
        for path in files[:120]:
            print(f"{path.relative_to(root)}\t{path.stat().st_size / (1024**3):.3f} GiB", flush=True)


@app.function(
    image=int4_image,
    timeout=6 * 60 * MINUTES,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={VOLUME_ROOT: model_volume},
    secrets=secrets,
)
def download_model() -> None:
    code = f"""
from pathlib import Path
from huggingface_hub import snapshot_download

Path({CKPT_DIR!r}).mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id={MODEL_REPO!r},
    local_dir={CKPT_DIR!r},
    allow_patterns={_download_patterns()!r},
    local_dir_use_symlinks=False,
    resume_download=True,
)
"""
    print(f"Downloading or resuming {MODEL_REPO} into {CKPT_DIR}", flush=True)
    subprocess.run(["python", "-c", code], check=True)
    model_volume.commit()


@app.function(
    image=int4_image,
    timeout=30 * MINUTES,
    volumes={VOLUME_ROOT: model_volume},
    secrets=secrets,
)
def inspect_tensors(max_shards: int = 1) -> None:
    code = f"""
from pathlib import Path
from safetensors import safe_open

root = Path({CKPT_DIR!r})
files = sorted(root.glob("model*-mp4.safetensors"))[:{max_shards}]
if not files:
    raise RuntimeError(f"No INT4 safetensors found in {{root}}")
for path in files:
    print(f"\\n# {{path.name}}", flush=True)
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print("tensor_count", len(keys), flush=True)
        for name in keys[:80]:
            t = f.get_tensor(name)
            print(name, tuple(t.shape), t.dtype, flush=True)
        for prefix in (
            "layers.0.ffn.experts.0.w1",
            "layers.0.ffn.shared_experts.w1",
            "layers.0.attn.wq_a",
            "mtp.0.ffn.experts.0.w1",
        ):
            print(f"\\nPREFIX {{prefix}}", flush=True)
            for name in [k for k in keys if k.startswith(prefix)]:
                t = f.get_tensor(name)
                print(name, tuple(t.shape), t.dtype, flush=True)
"""
    subprocess.run(["python", "-c", code], check=True)


@app.function(
    image=int4_image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={VOLUME_ROOT: model_volume},
    secrets=secrets,
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=1,
)
@modal.concurrent(max_inputs=1)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT)
def serve() -> None:
    if not _checkpoint_exists():
        raise RuntimeError(f"Missing INT4 checkpoint in {CKPT_DIR}; run action=download")

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc-per-node",
        str(N_GPU),
        SERVER_PATH,
        "--ckpt-path",
        CKPT_DIR,
        "--config",
        CONFIG_PATH,
        "--inference-root",
        SRC_DIR,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--served-model-name",
        SERVED_MODEL_NAME,
    ]
    print("Starting EnsueAI INT4 DeepSeek V4 server:", json.dumps(cmd), flush=True)
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


@app.function(
    image=int4_image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={VOLUME_ROOT: model_volume},
    secrets=secrets,
)
def debug_start() -> None:
    if not _checkpoint_exists():
        raise RuntimeError(f"Missing INT4 checkpoint in {CKPT_DIR}; run action=download")
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc-per-node",
        str(N_GPU),
        SERVER_PATH,
        "--ckpt-path",
        CKPT_DIR,
        "--config",
        CONFIG_PATH,
        "--inference-root",
        SRC_DIR,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--served-model-name",
        SERVED_MODEL_NAME,
    ]
    print("Running EnsueAI INT4 server in foreground:", json.dumps(cmd), flush=True)
    subprocess.run(cmd, check=True)


@app.local_entrypoint()
async def main(action: str = "url", prompt: str = "What is 17*19?") -> None:
    if action == "remote":
        inspect_remote_model.remote()
        return
    if action == "inspect":
        inspect_volume.remote()
        return
    if action == "tensors":
        inspect_tensors.remote()
        return
    if action == "download":
        download_model.remote()
        return
    if action == "debug":
        debug_start.remote()
        return

    url = await serve.get_web_url.aio()
    print(f"OpenAI-compatible endpoint: {url}/v1")

    if action != "test":
        return

    payload: dict[str, Any] = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 16,
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
            if time.time() >= deadline:
                raise
            time.sleep(5)
