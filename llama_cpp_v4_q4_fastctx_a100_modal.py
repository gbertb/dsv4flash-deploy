from __future__ import annotations

import os


os.environ.setdefault("LLAMA_CPP_APP_NAME", "deepseek-v4-flash-llama-cpp-q4-fastctx-a100")
os.environ.setdefault("LLAMA_CPP_MODEL_QUANT", "Q4_K_M-XL")
os.environ.setdefault("LLAMA_CPP_GPU", "A100-80GB:4")
os.environ.setdefault("LLAMA_CPP_CUDA_ARCH", "80")
os.environ.setdefault("LLAMA_CPP_CTX", "4096")
os.environ.setdefault("LLAMA_CPP_EXTRA_SERVER_ARGS", "--flash-attn off --cache-ram 0 --no-warmup")

from llama_cpp_v4_q4_a100_modal import app, main  # noqa: E402,F401
