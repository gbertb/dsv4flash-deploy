from __future__ import annotations

import os


os.environ.setdefault("LLAMA_CPP_APP_NAME", "deepseek-v4-flash-llama-cpp-q4-modelcard-a100")
os.environ.setdefault("LLAMA_CPP_MODEL_QUANT", "Q4_K_M-XL")
os.environ.setdefault("LLAMA_CPP_GPU", "A100-80GB:4")
os.environ.setdefault("LLAMA_CPP_CUDA_ARCH", "80")
os.environ.setdefault("LLAMA_CPP_CMAKE_EXTRA_ARGS", "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512")
os.environ.setdefault("LLAMA_CPP_CTX", "8192")
os.environ.setdefault("LLAMA_CPP_SPLIT_MODE", "layer")
os.environ.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4x")
os.environ.setdefault("GGML_CUDA_P2P", "1")
os.environ.setdefault(
    "LLAMA_CPP_EXTRA_SERVER_ARGS",
    (
        "--reasoning off --flash-attn on --no-repack "
        "--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 "
        "--cache-ram 0 --no-warmup --batch-size 2048 --ubatch-size 512 "
        "--poll 100 --poll-batch 1"
    ),
)

from llama_cpp_v4_q4_a100_modal import app, main  # noqa: E402,F401
