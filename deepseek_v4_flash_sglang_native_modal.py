from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import modal


APP_NAME = os.environ.get("SGLANG_NATIVE_APP_NAME", "deepseek-v4-flash-sglang-native-a100")
MODEL_REPO = os.environ.get("SGLANG_NATIVE_MODEL_REPO", "deepseek-ai/DeepSeek-V4-Flash")
PORT = int(os.environ.get("SGLANG_NATIVE_PORT", "30000"))
N_GPU = int(os.environ.get("SGLANG_NATIVE_N_GPU", "4"))
GPU_CONFIG = os.environ.get("SGLANG_NATIVE_GPU", f"A100-80GB:{N_GPU}")
MINUTES = 60
STARTUP_TIMEOUT = int(os.environ.get("SGLANG_NATIVE_STARTUP_TIMEOUT", str(4 * 60 * MINUTES)))
TIMEOUT = int(os.environ.get("SGLANG_NATIVE_TIMEOUT", str(8 * 60 * MINUTES)))
EPHEMERAL_DISK_MIB = int(os.environ.get("SGLANG_NATIVE_EPHEMERAL_DISK_MIB", "1572864"))

MAX_CONTEXT_LEN = int(os.environ.get("SGLANG_NATIVE_CONTEXT_LEN", "4096"))
MEM_FRACTION_STATIC = os.environ.get("SGLANG_NATIVE_MEM_FRACTION_STATIC", "0.88")
EXTRA_ARGS = os.environ.get("SGLANG_NATIVE_EXTRA_ARGS", "")
SGLANG_IMAGE = os.environ.get("SGLANG_NATIVE_IMAGE", "lmsysorg/sglang:latest")
SGLANG_PYTHON = os.environ.get("SGLANG_NATIVE_PYTHON", "/usr/bin/python3")

HF_CACHE_MOUNT = "/sglang-hf-cache"
hf_cache_vol = modal.Volume.from_name("deepseek-v4-flash-hf-cache", create_if_missing=True)

sglang_image = (
    modal.Image.from_registry(SGLANG_IMAGE, add_python="3.11")
    .entrypoint([])
    .env(
        {
            "HF_HOME": HF_CACHE_MOUNT,
            "HF_HUB_CACHE": f"{HF_CACHE_MOUNT}/hub",
            "HF_XET_CACHE": f"{HF_CACHE_MOUNT}/xet",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        }
    )
)

app = modal.App(APP_NAME)


def _split_extra_args(value: str) -> list[str]:
    if not value.strip():
        return []
    import shlex

    return shlex.split(value)


def build_sglang_command(
    *,
    dummy: bool = False,
    mtp: bool = False,
    extra_args: str | None = None,
) -> list[str]:
    # SGLang documents both `sglang serve` and `python -m sglang.launch_server`.
    # The module form gives a stable Python entrypoint inside Modal containers.
    cmd = [
        SGLANG_PYTHON,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL_REPO,
        "--host",
        "0.0.0.0",
        "--port",
        str(PORT),
        "--tp",
        str(N_GPU),
        "--context-length",
        str(MAX_CONTEXT_LEN),
        "--mem-fraction-static",
        MEM_FRACTION_STATIC,
        "--trust-remote-code",
        "--reasoning-parser",
        "deepseek-v4",
        "--tool-call-parser",
        "deepseekv4",
        "--attention-backend",
        "dsv4",
        "--prefill-attention-backend",
        "dsv4",
        "--decode-attention-backend",
        "dsv4",
        "--disable-cuda-graph",
    ]
    if dummy:
        cmd.extend(["--load-format", "dummy"])
    if mtp:
        # Expected to be validated against `--help`; SGLang's V4 docs describe
        # MTP/EAGLE and require SGLANG_ENABLE_SPEC_V2=1.
        cmd.extend(["--speculative-algorithm", "EAGLE", "--speculative-num-draft-tokens", "1"])
    cmd.extend(_split_extra_args(EXTRA_ARGS if extra_args is None else extra_args))
    return cmd


def _run(cmd: list[str], required: bool = True) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, text=True, check=False)
    if required and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


@app.function(
    image=sglang_image,
    timeout=12 * MINUTES,
    volumes={HF_CACHE_MOUNT: hf_cache_vol},
)
def inspect_environment() -> None:
    checks = [
        ["which", "python3"],
        ["ls", "-l", "/usr/bin/python3", "/usr/local/bin/python3"],
        [SGLANG_PYTHON, "-c", "import sys; print('python', sys.executable, sys.version)"],
        [SGLANG_PYTHON, "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"],
        [SGLANG_PYTHON, "-c", "import sglang; print('sglang', getattr(sglang, '__version__', 'unknown'))"],
        [SGLANG_PYTHON, "-m", "sglang.launch_server", "--help"],
        ["sglang", "serve", "--help"],
    ]
    for cmd in checks:
        _run(cmd, required=cmd[0] != "sglang")


@app.function(
    image=sglang_image,
    timeout=12 * MINUTES,
    volumes={HF_CACHE_MOUNT: hf_cache_vol},
)
def inspect_summary() -> None:
    code = f"""
import json
import subprocess

import torch
import sglang

help_text = subprocess.run(
    [{SGLANG_PYTHON!r}, "-m", "sglang.launch_server", "--help"],
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    check=False,
).stdout

summary = {{
    "image": {SGLANG_IMAGE!r},
    "python": {SGLANG_PYTHON!r},
    "model_repo": {MODEL_REPO!r},
    "sglang_version": getattr(sglang, "__version__", "unknown"),
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "help_flags": {{
        "--model-path": "--model-path" in help_text,
        "--tp": "--tp" in help_text,
        "--context-length": "--context-length" in help_text,
        "--mem-fraction-static": "--mem-fraction-static" in help_text,
        "--load-format": "--load-format" in help_text,
        "--speculative-algorithm": "--speculative-algorithm" in help_text,
        "--speculative-num-draft-tokens": "--speculative-num-draft-tokens" in help_text,
        "--reasoning-parser": "--reasoning-parser" in help_text,
        "--tool-call-parser": "--tool-call-parser" in help_text,
        "--moe-runner-backend": "--moe-runner-backend" in help_text,
    }},
    "command": {build_sglang_command()!r},
    "dummy_command": {build_sglang_command(dummy=True)!r},
    "dummy_mtp_command": {build_sglang_command(dummy=True, mtp=True)!r},
}}
print("sglang_native_inspect_summary=" + json.dumps(summary, sort_keys=True))
"""
    _run([SGLANG_PYTHON, "-c", code])


@app.function(image=sglang_image, timeout=8 * MINUTES)
def source_summary() -> None:
    code = r"""
from pathlib import Path
import json
import sglang

roots = [Path(sglang.__file__).resolve().parent]
roots.extend([
    Path('/sgl-workspace/sglang/python/sglang'),
    Path('/usr/local/lib/python3.12/site-packages/sglang'),
    Path('/usr/local/lib/python3.11/site-packages/sglang'),
    Path('/usr/local/lib/python3.10/site-packages/sglang'),
    Path('/usr/local/lib/python3.12/dist-packages/sglang'),
    Path('/usr/local/lib/python3.11/dist-packages/sglang'),
    Path('/usr/lib/python3.12/dist-packages/sglang'),
    Path('/usr/lib/python3/dist-packages/sglang'),
])
root = next((p for p in roots if p.exists()), None)
if root is None:
    raise RuntimeError('sglang package root not found')

patterns = [
    'DeepseekV4',
    'DeepSeekV4',
    'deepseek_v4',
    'DeepSeek-V4',
    'mtp',
    'EAGLE',
    'speculative',
    'hicache',
    'shadow',
    'hyper',
    'mhc',
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

print('sglang_native_source_summary=' + json.dumps({
    'root': str(root),
    'python_file_count': len(files),
    'pattern_counts': {pattern: len(rows) for pattern, rows in matches.items()},
    'matches': matches,
}, sort_keys=True))
"""
    _run([SGLANG_PYTHON, "-c", code])


@app.function(
    image=sglang_image,
    timeout=12 * MINUTES,
    volumes={HF_CACHE_MOUNT: hf_cache_vol},
)
def remote_model_probe() -> None:
    code = f"""
import json
from huggingface_hub import HfApi

api = HfApi()
info = api.model_info({MODEL_REPO!r}, files_metadata=True)
total = 0
safetensors = 0
sample = []
for sibling in info.siblings:
    size = getattr(sibling, "size", None) or 0
    if sibling.rfilename.endswith(".safetensors"):
        safetensors += 1
        total += size
    if len(sample) < 80:
        sample.append({{"name": sibling.rfilename, "size": size}})
print("sglang_native_remote_model_results=" + json.dumps({{
    "repo": {MODEL_REPO!r},
    "sha": info.sha,
    "safetensors_count": safetensors,
    "safetensors_total_gib": total / (1024 ** 3),
    "sample_files": sample,
}}, sort_keys=True))
"""
    _run([SGLANG_PYTHON, "-c", code])


@app.function(
    image=sglang_image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    volumes={HF_CACHE_MOUNT: hf_cache_vol},
)
def debug_start(max_runtime_seconds: int = 0, dummy: bool = False, mtp: bool = False) -> None:
    cmd = build_sglang_command(dummy=dummy, mtp=mtp)
    print("Running native SGLang server in foreground:", json.dumps(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    started = time.monotonic()
    ready_at: float | None = None
    requested_terminate = False
    tail: deque[str] = deque(maxlen=500)
    interesting: deque[str] = deque(maxlen=250)
    ready_markers = (
        "The server is fired up",
        "Application startup complete",
        "Uvicorn running on",
        "server started",
    )
    while proc.poll() is None:
        if max_runtime_seconds and time.monotonic() - started > max_runtime_seconds:
            print(f"Reached max_runtime_seconds={max_runtime_seconds}; terminating probe.", flush=True)
            proc.terminate()
            requested_terminate = True
            break
        if ready_at is not None and time.monotonic() - ready_at > 60:
            print("Readiness grace period elapsed; terminating probe.", flush=True)
            proc.terminate()
            requested_terminate = True
            break

        readable, _, _ = select.select([proc.stdout], [], [], 1.0)
        if not readable:
            continue
        line = proc.stdout.readline()
        if not line:
            continue
        print(line, end="", flush=True)
        tail.append(line)
        if any(marker.lower() in line.lower() for marker in ready_markers):
            ready_at = ready_at or time.monotonic()
            print("native SGLang readiness marker observed", flush=True)
        if any(
            marker in line
            for marker in (
                "Traceback",
                "RuntimeError",
                "Exception",
                "Error",
                "ERROR",
                "Failed",
                "failed",
                "CUDA",
                "OOM",
                "Unsupported",
                "unsupported",
                "ValueError",
            )
        ):
            interesting.append(line)

    try:
        returncode = proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        returncode = proc.wait()

    if requested_terminate:
        print(f"Probe terminated intentionally with returncode={returncode}", flush=True)
        return

    if returncode:
        print("\n===== native SGLang failure markers =====", flush=True)
        for line in interesting:
            print(line, end="", flush=True)
        print("\n===== native SGLang last 500 lines =====", flush=True)
        for line in tail:
            print(line, end="", flush=True)
        raise subprocess.CalledProcessError(returncode, cmd)


@app.local_entrypoint()
async def main(action: str = "command") -> None:
    if action == "command":
        print(json.dumps(build_sglang_command(), indent=2))
        return
    if action == "command-dummy":
        print(json.dumps(build_sglang_command(dummy=True), indent=2))
        return
    if action == "command-dummy-mtp":
        print(json.dumps(build_sglang_command(dummy=True, mtp=True), indent=2))
        return
    if action == "inspect":
        inspect_environment.remote()
        return
    if action == "inspect-summary":
        inspect_summary.remote()
        return
    if action == "source-summary":
        source_summary.remote()
        return
    if action == "remote-model":
        remote_model_probe.remote()
        return
    if action == "debug-dummy-short":
        debug_start.remote(20 * MINUTES, True, False)
        return
    if action == "debug-dummy-mtp-short":
        debug_start.remote(20 * MINUTES, True, True)
        return
    if action == "debug-short":
        debug_start.remote(20 * MINUTES, False, False)
        return

    raise ValueError(f"unknown action: {action}")
