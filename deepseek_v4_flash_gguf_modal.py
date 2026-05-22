from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any

import modal


APP_NAME = "deepseek-v4-flash-gguf-vllm"

MODEL_REPO = "Preyazz/DeepSeek-V4-Flash-Q8_0-GGUF"
MODEL_ALIAS = "Q8_0"
MODEL_NAME = f"{MODEL_REPO}:{MODEL_ALIAS}"
SERVED_MODEL_NAME = "deepseek-v4-flash-gguf"
TOKENIZER_REPO = "deepseek-ai/DeepSeek-V4-Flash"

N_GPU = 4
VLLM_PORT = 8000
MINUTES = 60
VLLM_PYTHON = "/usr/bin/python3"
VLLM_BIN = "/usr/local/bin/vllm"
LARGE_EPHEMERAL_DISK_MIB = 1_572_864  # 1.5 TiB
STARTUP_TIMEOUT = 4 * 60 * MINUTES

# Keep the default context conservative for the first deployment. The full model
# advertises 1M context, but a 302 GB Q8 GGUF leaves very little room on 4x80 GB.
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
GPU_MEMORY_UTILIZATION = os.environ.get("GPU_MEMORY_UTILIZATION", "0.92")

hf_cache_vol = modal.Volume.from_name("deepseek-v4-flash-hf-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("deepseek-v4-flash-vllm-cache", create_if_missing=True)

vllm_image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:deepseekv4-cu130",
        add_python="3.11",
    )
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        f"{VLLM_PYTHON} -m pip install --no-cache-dir --upgrade "
        "huggingface_hub==1.8.0 hf_transfer"
    )
    .run_commands(
        f"{VLLM_PYTHON} -m pip install --no-cache-dir --no-deps --upgrade "
        "git+https://github.com/huggingface/transformers.git"
    )
    .run_commands(
        "TORCH_CUDA_ARCH_LIST=8.0 "
        f"{VLLM_PYTHON} -m pip install --no-cache-dir "
        "--no-build-isolation --no-deps "
        "git+https://github.com/vllm-project/vllm-gguf-plugin.git@v0.0.1"
    )
    .add_local_file(
        "patch_vllm_gguf_plugin.py",
        "/root/patch_vllm_gguf_plugin_20260520_5.py",
        copy=True,
    )
    .run_commands(f"{VLLM_PYTHON} /root/patch_vllm_gguf_plugin_20260520_5.py")
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "HF_HUB_CACHE": "/root/.cache/huggingface/hub",
            "HF_XET_CACHE": "/root/.cache/huggingface/xet",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "TMPDIR": "/tmp",
            "VLLM_ENGINE_READY_TIMEOUT_S": str(STARTUP_TIMEOUT),
            "VLLM_RPC_TIMEOUT": "600000",
            "VLLM_LOG_STATS_INTERVAL": "10",
            "VLLM_LOGGING_LEVEL": "DEBUG",
            "PYTHONFAULTHANDLER": "1",
            "TILELANG_CLEANUP_TEMP_FILES": "1",
        }
    )
)

app = modal.App(APP_NAME)


@app.function(
    image=vllm_image,
    timeout=8 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def inspect_environment() -> None:
    """Print versions and plugin import state for deployment debugging."""

    commands = [
        (
            [VLLM_PYTHON, "-c", "import vllm; print('vllm', vllm.__version__)"],
            True,
        ),
        (
            [
                VLLM_PYTHON,
                "-c",
                "import vllm_gguf_plugin; print('vllm_gguf_plugin import ok')",
            ],
            True,
        ),
        (
            [
                VLLM_PYTHON,
                "-c",
                (
                    "from huggingface_hub import is_offline_mode; "
                    "from transformers.models.deepseek_v4.modeling_deepseek_v4 "
                    "import DeepseekV4ForCausalLM; "
                    "print('transformers deepseek_v4 import ok')"
                ),
            ],
            True,
        ),
        (
            [
                VLLM_PYTHON,
                "-c",
                (
                    "from vllm.transformers_utils.config import HFConfigParser; "
                    f"_, c = HFConfigParser().parse({TOKENIZER_REPO!r}, trust_remote_code=True); "
                    "print('deepseek_v4 config parse ok', c.model_type, "
                    "c.max_position_embeddings, bool(c.rope_parameters))"
                ),
            ],
            True,
        ),
        (
            [
                VLLM_PYTHON,
                "-c",
                (
                    "import vllm_gguf_plugin.quantization.linear; "
                    "import vllm_gguf_plugin.quantization.fused_moe; "
                    "import vllm_gguf_plugin.quantization.vocal_embeds; "
                    "import vllm.model_executor.layers.quantization.gguf; "
                    "print('gguf op duplicate-registration smoke test ok')"
                ),
            ],
            True,
        ),
        ([VLLM_BIN, "--help"], False),
    ]
    for cmd, required in commands:
        print("$", " ".join(cmd), flush=True)
        result = subprocess.run(cmd, check=False)
        if required and result.returncode:
            raise subprocess.CalledProcessError(result.returncode, cmd)


@app.function(
    image=vllm_image,
    timeout=6 * 60 * MINUTES,
    ephemeral_disk=LARGE_EPHEMERAL_DISK_MIB,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def download_model() -> None:
    """Warm the Modal Volume with the GGUF model and tokenizer."""

    download_code = f"""
from pathlib import Path

from huggingface_hub import snapshot_download

Path("/root/.cache/huggingface/tmp").mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id={MODEL_REPO!r},
    allow_patterns=["*.gguf", "*.json", "*.model", "*.txt"],
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id={TOKENIZER_REPO!r},
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "tokenizer*",
        "*.json",
        "*.model",
        "*.txt",
    ],
    local_dir_use_symlinks=False,
)
"""
    subprocess.run(
        [VLLM_PYTHON, "-c", download_code],
        check=True,
    )
    hf_cache_vol.commit()


@app.function(
    image=vllm_image,
    timeout=30 * MINUTES,
    ephemeral_disk=LARGE_EPHEMERAL_DISK_MIB,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def inspect_gguf_tensors() -> None:
    """Print representative GGUF tensor names from the cached model file."""

    code = r"""
from pathlib import Path

import gguf

gguf_paths = sorted(Path("/root/.cache/huggingface").rglob("*.gguf"))
if not gguf_paths:
    raise RuntimeError("No cached GGUF files found under /root/.cache/huggingface")

model_path = str(gguf_paths[0])
print(f"Inspecting GGUF tensor names in {model_path}", flush=True)
reader = gguf.GGUFReader(model_path)

names = [tensor.name for tensor in reader.tensors]
print(f"Tensor count: {len(names)}", flush=True)
for needle in ("attn", "compress", "index", "sink", "hc", "head", "tid", "o_", "kv"):
    print(f"\n===== names containing {needle!r} =====", flush=True)
    for name in [n for n in names if needle in n][:120]:
        print(name, flush=True)
"""
    subprocess.run([VLLM_PYTHON, "-c", code], check=True)


@app.function(image=vllm_image, timeout=8 * MINUTES)
def inspect_installed_source() -> None:
    """Print installed vLLM source snippets needed for GGUF name mapping."""

    code = r"""
from pathlib import Path

paths = [
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py"),
    Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/deepseek_v4_attention.py"),
]

for path in paths:
    print(f"\n===== {path} =====", flush=True)
    lines = path.read_text().splitlines()
    if path.name == "deepseek_v4.py":
        fixed_windows = [(1, 180), (180, 360), (360, 580), (650, 745), (772, 849)]
    else:
        fixed_windows = [(120, 280), (944, 1065)]
    for start, end in fixed_windows:
        start = max(1, start)
        end = min(len(lines), end)
        print(f"\n--- fixed lines {start}-{end} ---", flush=True)
        for line_no in range(start, end + 1):
            print(f"{line_no:5}: {lines[line_no - 1]}", flush=True)

    needles = (
        "def load_weights",
        "self_attn",
        "sinks",
        "q_a_norm",
        "kv_proj",
        "o_a_proj",
        "compressor",
        "attn_hc",
        "ffn_hc",
        "hc_head",
    )
    matches = sorted(
        {
            line_no
            for line_no, line in enumerate(lines, start=1)
            if any(needle in line for needle in needles)
        }
    )
    windows: list[tuple[int, int]] = []
    for line_no in matches:
        start = max(1, line_no - 8)
        end = min(len(lines), line_no + 12)
        if windows and start <= windows[-1][1] + 3:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    for start, end in windows:
        print(f"\n--- lines {start}-{end} ---", flush=True)
        for line_no in range(start, end + 1):
            print(f"{line_no:5}: {lines[line_no - 1]}", flush=True)
"""
    subprocess.run([VLLM_PYTHON, "-c", code], check=True)


@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    timeout=6 * 60 * MINUTES,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=LARGE_EPHEMERAL_DISK_MIB,
    scaledown_window=30 * MINUTES,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def serve() -> None:
    cmd = build_vllm_command()
    print("Starting vLLM server:", json.dumps(cmd), flush=True)
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


def build_vllm_command() -> list[str]:
    cmd = [
        VLLM_BIN,
        "serve",
        MODEL_NAME,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--tokenizer",
        TOKENIZER_REPO,
        "--trust-remote-code",
        "--load-format",
        "gguf",
        "--config-format",
        "gguf",
        "--quantization",
        "gguf",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level",
        "info",
        "--disable-uvicorn-access-log",
        "--tensor-parallel-size",
        str(N_GPU),
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--tokenizer-mode",
        "deepseek_v4",
        "--reasoning-parser",
        "deepseek_v4",
        "--tool-call-parser",
        "deepseek_v4",
        "--enable-auto-tool-choice",
        "--kv-cache-dtype",
        "fp8",
        "--block-size",
        "256",
        "--enforce-eager",
    ]

    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        cmd.extend(["--api-key", api_key])

    return cmd


@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    timeout=6 * 60 * MINUTES,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=LARGE_EPHEMERAL_DISK_MIB,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def debug_start() -> None:
    cmd = build_vllm_command()
    print("Running vLLM server in foreground for debugging:", json.dumps(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    tail: deque[str] = deque(maxlen=500)
    interesting: deque[str] = deque(maxlen=250)
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if any(
            marker in line
            for marker in (
                "Traceback",
                "RuntimeError",
                "Exception",
                "Error",
                "ERROR",
                "Worker",
                "Failed",
                "failed",
                "CUDA",
                "OOM",
                "map GGUF",
                "Unsupported",
            )
        ):
            interesting.append(line)

    returncode = proc.wait()
    if returncode:
        print("\n===== vLLM failure markers =====", flush=True)
        for line in interesting:
            print(line, end="", flush=True)
        print("\n===== vLLM last 500 lines =====", flush=True)
        for line in tail:
            print(line, end="", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd)


@app.local_entrypoint()
async def main(action: str = "url", prompt: str = "What is 17*19?") -> None:
    if action == "inspect":
        inspect_environment.remote()
        return

    if action == "download":
        download_model.remote()
        return

    if action == "gguf-names":
        inspect_gguf_tensors.remote()
        return

    if action == "source":
        inspect_installed_source.remote()
        return

    if action == "debug":
        debug_start.remote()
        return

    url = await serve.get_web_url.aio()
    print(f"OpenAI-compatible endpoint: {url}/v1")

    if action != "test":
        return

    import urllib.error
    import urllib.request

    payload: dict[str, Any] = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 64,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60 * MINUTES) as resp:
            print(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8"))
        raise
