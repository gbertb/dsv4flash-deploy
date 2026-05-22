from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import modal


APP_NAME = os.environ.get(
    "LLAMA_CPP_SOURCE_PROBE_APP_NAME",
    "deepseek-v4-flash-llama-cpp-upstream-master-source-probe-compact",
)
REPO = os.environ.get("LLAMA_CPP_SOURCE_PROBE_REPO", "https://github.com/ggml-org/llama.cpp.git")
BRANCH = os.environ.get("LLAMA_CPP_SOURCE_PROBE_BRANCH", "master")
SOURCE_DIR = "/opt/llama.cpp-source-probe"

image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("ca-certificates", "git")
    .run_commands(
        f"git clone --depth=1 --branch {BRANCH} {REPO} {SOURCE_DIR}",
        f"cd {SOURCE_DIR} && git rev-parse HEAD && git status --short",
    )
)

app = modal.App(APP_NAME)


def _find_lines(root_dir: str, patterns: list[str], max_matches: int = 200) -> list[dict[str, Any]]:
    roots = [
        Path(root_dir) / "src",
        Path(root_dir) / "common",
        Path(root_dir) / "tools/server",
    ]
    suffixes = {".c", ".cc", ".cpp", ".h", ".hpp"}
    lowered = [(pattern, pattern.lower()) for pattern in patterns]
    matches = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in suffixes:
                continue
            lines = path.read_text(errors="replace").splitlines()
            for idx, line in enumerate(lines):
                line_lower = line.lower()
                hit = [pattern for pattern, lower in lowered if lower in line_lower]
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


def _exists(path: str) -> bool:
    return (Path(SOURCE_DIR) / path).exists()


@app.function(image=image, timeout=20 * 60)
def probe() -> None:
    commit = subprocess.run(
        ["git", "-C", SOURCE_DIR, "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    ).stdout.strip()
    patterns = [
        "LLM_ARCH_DEEPSEEK4",
        "deepseek4",
        "draft-mtp",
        "COMMON_SPECULATIVE_TYPE_DRAFT_MTP",
        "nextn_predict_layers",
        "kv_only_nextn",
        "LLAMA_SPLIT_MODE_TENSOR",
        "forcing fp16 KV",
        "cache_type_k",
        "cache_type_v",
    ]
    matches = _find_lines(SOURCE_DIR, patterns)
    counts = {pattern: 0 for pattern in patterns}
    for match in matches:
        for pattern in match["patterns"]:
            counts[pattern] += 1

    result = {
        "repo": REPO,
        "branch": BRANCH,
        "commit": commit,
        "files_exist": {
            "src/models/deepseek4.cpp": _exists("src/models/deepseek4.cpp"),
            "common/speculative.cpp": _exists("common/speculative.cpp"),
            "tools/server/server.cpp": _exists("tools/server/server.cpp"),
        },
        "pattern_counts": counts,
        "key_matches": [
            match
            for match in matches
            if any(
                pattern in match["patterns"]
                for pattern in [
                    "LLM_ARCH_DEEPSEEK4",
                    "deepseek4",
                    "draft-mtp",
                    "COMMON_SPECULATIVE_TYPE_DRAFT_MTP",
                    "kv_only_nextn",
                    "forcing fp16 KV",
                    "LLAMA_SPLIT_MODE_TENSOR",
                ]
            )
        ][:80],
    }
    print("compact_source_probe_results=" + json.dumps(result, indent=2), flush=True)


@app.local_entrypoint()
def main() -> None:
    probe.remote()

