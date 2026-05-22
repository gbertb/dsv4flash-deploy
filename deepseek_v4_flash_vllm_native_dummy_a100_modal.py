from __future__ import annotations

import os


os.environ.setdefault("VLLM_NATIVE_APP_NAME", "deepseek-v4-flash-vllm-native-dummy-a100")
os.environ.setdefault("VLLM_NATIVE_LOAD_FORMAT", "dummy")
os.environ.setdefault("VLLM_NATIVE_EXTRA_ARGS", "--enforce-eager")
os.environ.setdefault("VLLM_NATIVE_TIMEOUT", "3600")

from deepseek_v4_flash_vllm_native_modal import app, main  # noqa: E402,F401
