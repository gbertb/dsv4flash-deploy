from __future__ import annotations

import os


os.environ.setdefault(
    "LLAMA_CPP_APP_NAME",
    "deepseek-v4-flash-llama-cpp-upstream-master-source-probe",
)
os.environ.setdefault("LLAMA_CPP_SOURCE_PROBE_REPO", "https://github.com/ggml-org/llama.cpp.git")
os.environ.setdefault("LLAMA_CPP_SOURCE_PROBE_BRANCH", "master")

from llama_cpp_v4_q4_a100_modal import app, main  # noqa: E402,F401

