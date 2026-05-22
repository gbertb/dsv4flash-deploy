from __future__ import annotations

import os


os.environ.setdefault("LLAMA_CPP_APP_NAME", "deepseek-v4-flash-llama-cpp-q2-a100")
os.environ.setdefault("LLAMA_CPP_MODEL_QUANT", "Q2_K-XL")
os.environ.setdefault("LLAMA_CPP_GPU", "A100-80GB:4")
os.environ.setdefault("LLAMA_CPP_CUDA_ARCH", "80")

from llama_cpp_v4_q4_a100_modal import app, main  # noqa: E402,F401
