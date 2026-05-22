from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get("VLLM_NATIVE_APP_NAME", "deepseek-v4-flash-vllm-native-a100")
MODEL_REPO = os.environ.get("VLLM_NATIVE_MODEL_REPO", "deepseek-ai/DeepSeek-V4-Flash")
SERVED_MODEL_NAME = os.environ.get("VLLM_NATIVE_SERVED_MODEL_NAME", "deepseek-v4-flash")
N_GPU = int(os.environ.get("VLLM_NATIVE_N_GPU", "4"))
GPU_CONFIG = os.environ.get("VLLM_NATIVE_GPU", f"A100-80GB:{N_GPU}")
PORT = int(os.environ.get("VLLM_NATIVE_PORT", "8000"))
MINUTES = 60
STARTUP_TIMEOUT = int(os.environ.get("VLLM_NATIVE_STARTUP_TIMEOUT", str(4 * 60 * MINUTES)))
TIMEOUT = int(os.environ.get("VLLM_NATIVE_TIMEOUT", str(8 * 60 * MINUTES)))
EPHEMERAL_DISK_MIB = int(os.environ.get("VLLM_NATIVE_EPHEMERAL_DISK_MIB", "1572864"))

VLLM_PYTHON = os.environ.get("VLLM_NATIVE_PYTHON", "/usr/bin/python3")
VLLM_BIN = os.environ.get("VLLM_NATIVE_BIN", "/usr/local/bin/vllm")
MAX_MODEL_LEN = int(os.environ.get("VLLM_NATIVE_MAX_MODEL_LEN", "4096"))
MAX_NUM_SEQS = int(os.environ.get("VLLM_NATIVE_MAX_NUM_SEQS", "4"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("VLLM_NATIVE_MAX_NUM_BATCHED_TOKENS", "4096"))
GPU_MEMORY_UTILIZATION = os.environ.get("VLLM_NATIVE_GPU_MEMORY_UTILIZATION", "0.92")
KV_CACHE_DTYPE = os.environ.get("VLLM_NATIVE_KV_CACHE_DTYPE", "fp8")
BLOCK_SIZE = os.environ.get("VLLM_NATIVE_BLOCK_SIZE", "256")
LOAD_FORMAT = os.environ.get("VLLM_NATIVE_LOAD_FORMAT", "")
SPECULATIVE_CONFIG = os.environ.get("VLLM_NATIVE_SPECULATIVE_CONFIG", "")
EXTRA_ARGS = os.environ.get("VLLM_NATIVE_EXTRA_ARGS", "")

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
        "'huggingface_hub>=0.34.0,<1.0' hf_transfer"
    )
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
            "PYTHONUNBUFFERED": "1",
        }
    )
)

app = modal.App(APP_NAME)


def _split_extra_args(value: str) -> list[str]:
    if not value.strip():
        return []
    import shlex

    return shlex.split(value)


def build_vllm_command(
    load_format: str | None = None,
    speculative_config: str | None = None,
    extra_args: str | None = None,
) -> list[str]:
    effective_load_format = LOAD_FORMAT if load_format is None else load_format
    effective_speculative_config = (
        SPECULATIVE_CONFIG if speculative_config is None else speculative_config
    )
    effective_extra_args = EXTRA_ARGS if extra_args is None else extra_args
    cmd = [
        VLLM_BIN,
        "serve",
        MODEL_REPO,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--uvicorn-log-level",
        "info",
        "--disable-uvicorn-access-log",
        "--tensor-parallel-size",
        str(N_GPU),
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-num-batched-tokens",
        str(MAX_NUM_BATCHED_TOKENS),
        "--tokenizer-mode",
        "deepseek_v4",
        "--reasoning-parser",
        "deepseek_v4",
        "--kv-cache-dtype",
        KV_CACHE_DTYPE,
        "--block-size",
        BLOCK_SIZE,
    ]
    if effective_load_format:
        cmd.extend(["--load-format", effective_load_format])
    if effective_speculative_config:
        cmd.extend(["--speculative-config", effective_speculative_config])
    cmd.extend(_split_extra_args(effective_extra_args))
    return cmd


def _run(cmd: list[str], required: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, text=True, check=False)
    if required and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


@app.function(
    image=vllm_image,
    timeout=12 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def inspect_environment() -> None:
    checks = [
        [VLLM_PYTHON, "-c", "import vllm; print('vllm', vllm.__version__)"],
        [VLLM_PYTHON, "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"],
        [VLLM_BIN, "--help"],
        [VLLM_BIN, "serve", "--help"],
    ]
    for cmd in checks:
        _run(cmd, required=cmd[:2] != [VLLM_BIN, "serve"])

    code = f"""
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained({MODEL_REPO!r}, trust_remote_code=True)
print('config model_type', getattr(cfg, 'model_type', None))
print('architectures', getattr(cfg, 'architectures', None))
print('max_position_embeddings', getattr(cfg, 'max_position_embeddings', None))
print('num_nextn_predict_layers', getattr(cfg, 'num_nextn_predict_layers', None))
print('nextn_predict_layers', getattr(cfg, 'nextn_predict_layers', None))
"""
    _run([VLLM_PYTHON, "-c", code])


@app.function(
    image=vllm_image,
    timeout=12 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def inspect_summary() -> None:
    code = f"""
import json
import subprocess
import sys

import torch
import vllm
from transformers import AutoConfig
from vllm.transformers_utils.config import HFConfigParser

help_text = subprocess.run(
    [{VLLM_BIN!r}, "serve", "--help"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
).stdout
auto_config_error = None
cfg = None
try:
    cfg = AutoConfig.from_pretrained({MODEL_REPO!r}, trust_remote_code=True)
except Exception as exc:
    auto_config_error = type(exc).__name__ + ": " + str(exc)

vllm_config_error = None
vllm_cfg = None
try:
    _, vllm_cfg = HFConfigParser().parse({MODEL_REPO!r}, trust_remote_code=True)
except Exception as exc:
    vllm_config_error = type(exc).__name__ + ": " + str(exc)

effective_cfg = cfg if cfg is not None else vllm_cfg
summary = {{
    "vllm_version": vllm.__version__,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "model_repo": {MODEL_REPO!r},
    "auto_config_error": auto_config_error,
    "vllm_config_error": vllm_config_error,
    "config_model_type": getattr(effective_cfg, "model_type", None),
    "architectures": getattr(effective_cfg, "architectures", None),
    "max_position_embeddings": getattr(effective_cfg, "max_position_embeddings", None),
    "num_nextn_predict_layers": getattr(effective_cfg, "num_nextn_predict_layers", None),
    "nextn_predict_layers": getattr(effective_cfg, "nextn_predict_layers", None),
    "serve_help_flags": {{
        "--speculative-config": "--speculative-config" in help_text,
        "--tokenizer-mode": "--tokenizer-mode" in help_text,
        "--reasoning-parser": "--reasoning-parser" in help_text,
        "--kv-cache-dtype": "--kv-cache-dtype" in help_text,
        "--block-size": "--block-size" in help_text,
        "--max-num-batched-tokens": "--max-num-batched-tokens" in help_text,
    }},
    "command": {build_vllm_command()!r},
}}
print("vllm_native_inspect_summary=" + json.dumps(summary, sort_keys=True))
"""
    _run([VLLM_PYTHON, "-c", code])


@app.function(image=vllm_image, timeout=8 * MINUTES)
def source_probe() -> None:
    code = r"""
from pathlib import Path
import json

roots = [
    Path('/usr/local/lib/python3.12/dist-packages/vllm'),
    Path('/usr/local/lib/python3.11/dist-packages/vllm'),
    Path('/usr/local/lib/python3.10/dist-packages/vllm'),
]
root = next((p for p in roots if p.exists()), None)
if root is None:
    raise RuntimeError('vllm package root not found')

patterns = [
    'DeepseekV4',
    'DeepSeekV4',
    'deepseek_v4',
    'deepseek_mtp',
    'mtp',
    'speculative',
    'num_speculative_tokens',
    'kv_cache_dtype',
]
matches = {pattern: [] for pattern in patterns}
files = list(root.rglob('*.py'))
for path in files:
    rel = str(path.relative_to(root))
    try:
        lines = path.read_text(errors='ignore').splitlines()
    except OSError:
        continue
    for line_no, line in enumerate(lines, start=1):
        lowered = line.lower()
        for pattern in patterns:
            if pattern.lower() in lowered and len(matches[pattern]) < 80:
                matches[pattern].append({'file': rel, 'line': line_no, 'text': line.strip()[:220]})

important = {}
for rel in [
    'model_executor/models/deepseek_v4.py',
    'model_executor/models/deepseek_v4_nvidia.py',
    'model_executor/layers/deepseek_v4_attention.py',
    'model_executor/models/deepseek_v4/nvidia/mtp.py',
]:
    path = root / rel
    important[rel] = path.exists()

result = {
    'root': str(root),
    'python_file_count': len(files),
    'important_files': important,
    'pattern_counts': {pattern: len(rows) for pattern, rows in matches.items()},
    'matches': matches,
}
print('vllm_native_source_probe_results=' + json.dumps(result, sort_keys=True))
"""
    _run([VLLM_PYTHON, "-c", code])


@app.function(image=vllm_image, timeout=8 * MINUTES)
def source_summary() -> None:
    code = r"""
from pathlib import Path
import json

roots = [
    Path('/usr/local/lib/python3.12/dist-packages/vllm'),
    Path('/usr/local/lib/python3.11/dist-packages/vllm'),
    Path('/usr/local/lib/python3.10/dist-packages/vllm'),
]
root = next((p for p in roots if p.exists()), None)
if root is None:
    raise RuntimeError('vllm package root not found')

targets = {
    'config/speculative.py': ['deepseek_mtp', 'DeepSeekV4MTPModel', 'num_speculative_tokens'],
    'model_executor/models/deepseek_v4.py': ['DeepseekV4ForCausalLM', 'MTP', 'nextn'],
    'v1/core/kv_cache_utils.py': ['DeepseekV4', 'MTP attention layer', 'KV cache'],
    'v1/spec_decode/eagle.py': ['DeepseekV4', 'target_hidden_states'],
}
snippets = {}
for rel, needles in targets.items():
    path = root / rel
    if not path.exists():
        snippets[rel] = {'exists': False, 'matches': []}
        continue
    lines = path.read_text(errors='ignore').splitlines()
    matches = []
    for line_no, line in enumerate(lines, start=1):
        if any(needle.lower() in line.lower() for needle in needles):
            matches.append({'line': line_no, 'text': line.strip()[:220]})
    snippets[rel] = {'exists': True, 'matches': matches[:60]}

print('vllm_native_source_summary=' + json.dumps({
    'root': str(root),
    'snippets': snippets,
}, sort_keys=True))
"""
    _run([VLLM_PYTHON, "-c", code])


@app.function(
    image=vllm_image,
    timeout=12 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
)
def remote_model_probe() -> None:
    code = f"""
from huggingface_hub import HfApi
api = HfApi()
info = api.model_info({MODEL_REPO!r}, files_metadata=True)
total = 0
safetensors = 0
sample = []
for sibling in info.siblings:
    size = getattr(sibling, 'size', None) or 0
    if sibling.rfilename.endswith('.safetensors'):
        safetensors += 1
        total += size
    if len(sample) < 80:
        sample.append({{'name': sibling.rfilename, 'size': size}})
print('vllm_native_remote_model_results=' + __import__('json').dumps({{
    'repo': {MODEL_REPO!r},
    'sha': info.sha,
    'safetensors_count': safetensors,
    'safetensors_total_gib': total / (1024 ** 3),
    'sample_files': sample,
}}, sort_keys=True))
"""
    _run([VLLM_PYTHON, "-c", code])


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
def debug_start(
    max_runtime_seconds: int = 0,
    load_format: str | None = None,
    speculative_config: str | None = None,
    extra_args: str | None = None,
) -> None:
    cmd = build_vllm_command(load_format, speculative_config, extra_args)
    print("Running native vLLM server in foreground:", json.dumps(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    started = time.monotonic()
    tail: deque[str] = deque(maxlen=500)
    interesting: deque[str] = deque(maxlen=250)
    ready_markers = (
        "Uvicorn running on",
        "Application startup complete",
        "Started server process",
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail.append(line)
        if any(marker in line for marker in ready_markers):
            print("native vLLM readiness marker observed", flush=True)
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
                "Unsupported",
                "not supported",
                "ValueError",
            )
        ):
            interesting.append(line)
        if max_runtime_seconds and time.monotonic() - started > max_runtime_seconds:
            print(f"Reached max_runtime_seconds={max_runtime_seconds}; terminating probe.", flush=True)
            proc.terminate()
            break

    try:
        returncode = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()

    if returncode:
        print("\n===== native vLLM failure markers =====", flush=True)
        for line in interesting:
            print(line, end="", flush=True)
        print("\n===== native vLLM last 500 lines =====", flush=True)
        for line in tail:
            print(line, end="", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd)


@app.function(
    image=vllm_image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    scaledown_window=30 * MINUTES,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT)
def serve() -> None:
    cmd = build_vllm_command()
    print("Starting native vLLM server:", json.dumps(cmd), flush=True)
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)


@app.local_entrypoint()
async def main(action: str = "command", prompt: str = "What is 17*19?") -> None:
    if action == "inspect":
        inspect_environment.remote()
        return
    if action == "inspect-summary":
        inspect_summary.remote()
        return
    if action == "source-probe":
        source_probe.remote()
        return
    if action == "source-summary":
        source_summary.remote()
        return
    if action == "remote-model":
        remote_model_probe.remote()
        return
    if action == "debug":
        debug_start.remote()
        return
    if action == "debug-short":
        debug_start.remote(20 * MINUTES)
        return
    if action == "debug-dummy-short":
        debug_start.remote(20 * MINUTES, "dummy", None, "--enforce-eager")
        return
    if action == "debug-dummy-mtp-short":
        debug_start.remote(
            20 * MINUTES,
            "dummy",
            '{"method":"deepseek_mtp","num_speculative_tokens":1}',
            "--enforce-eager",
        )
        return
    if action == "command":
        print(json.dumps(build_vllm_command(), indent=2))
        return
    if action == "command-dummy":
        print(json.dumps(build_vllm_command("dummy", None, "--enforce-eager"), indent=2))
        return
    if action == "command-dummy-mtp":
        print(
            json.dumps(
                build_vllm_command(
                    "dummy",
                    '{"method":"deepseek_mtp","num_speculative_tokens":1}',
                    "--enforce-eager",
                ),
                indent=2,
            )
        )
        return

    url = await serve.get_web_url.aio()
    print(f"OpenAI-compatible endpoint: {url}/v1")
    if action != "test":
        return

    payload: dict[str, Any] = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
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
