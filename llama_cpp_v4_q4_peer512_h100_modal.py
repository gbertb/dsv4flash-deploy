from __future__ import annotations

import os


os.environ.setdefault("LLAMA_CPP_APP_NAME", "deepseek-v4-flash-llama-cpp-q4-peer512-h100")
os.environ.setdefault("LLAMA_CPP_MODEL_QUANT", "Q4_K_M-XL")
os.environ.setdefault("LLAMA_CPP_GPU", "H100:4")
os.environ.setdefault("LLAMA_CPP_CUDA_ARCH", "90")
os.environ.setdefault("LLAMA_CPP_CMAKE_EXTRA_ARGS", "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512")
os.environ.setdefault("LLAMA_CPP_CTX", "4096")
os.environ.setdefault("LLAMA_CPP_SPLIT_MODE", "layer")
os.environ.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4x")
os.environ.setdefault("GGML_CUDA_P2P", "1")
os.environ.setdefault(
    "LLAMA_CPP_EXTRA_SERVER_ARGS",
    "--cache-ram 0 --no-warmup --batch-size 2048 --ubatch-size 512 --poll 100 --poll-batch 1",
)

from llama_cpp_v4_q4_a100_modal import app, main  # noqa: E402,F401
