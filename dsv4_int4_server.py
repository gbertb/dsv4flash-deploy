from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from safetensors import safe_open
from safetensors.torch import load_model
from transformers import PreTrainedTokenizerFast
import uvicorn


A100_COMPAT = os.getenv("DSV4_A100_COMPAT", "1") not in {"0", "false", "False"}
INT4_GROUP_SIZE = 32


def _add_inference_paths(inference_root: str) -> None:
    root = Path(inference_root)
    sys.path.insert(0, str(root / "inference"))
    sys.path.insert(0, str(root / "encoding"))


def _special_token(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        return content if isinstance(content, str) else None
    return None


def _load_tokenizer(ckpt_path: str) -> PreTrainedTokenizerFast:
    ckpt = Path(ckpt_path)
    tokenizer_config_path = ckpt / "tokenizer_config.json"
    tokenizer_kwargs: dict[str, Any] = {}
    if tokenizer_config_path.exists():
        config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        for key in ("bos_token", "eos_token", "pad_token", "unk_token"):
            token = _special_token(config.get(key))
            if token is not None:
                tokenizer_kwargs[key] = token
        if isinstance(config.get("chat_template"), str):
            tokenizer_kwargs["chat_template"] = config["chat_template"]

    return PreTrainedTokenizerFast(
        tokenizer_file=str(ckpt / "tokenizer.json"),
        **tokenizer_kwargs,
    )


class PackedInt4Linear(nn.Module):
    def __init__(self, packed_shape: tuple[int, int], group_size: int = INT4_GROUP_SIZE):
        super().__init__()
        self.out_features = packed_shape[0]
        self.in_features = packed_shape[1] * 2
        self.group_size = group_size
        scale_shape = (self.out_features, self.in_features // group_size)
        self.register_buffer("weight", torch.empty(packed_shape, dtype=torch.uint8))
        self.register_buffer("scale", torch.empty(scale_shape, dtype=torch.float16))
        self.register_buffer("zero_point", torch.empty(scale_shape, dtype=torch.uint8))
        self.register_parameter("bias", None)

    def _dequant_weight(self) -> torch.Tensor:
        packed = self.weight
        q = torch.empty(
            (self.out_features, self.in_features),
            device=packed.device,
            dtype=torch.float16,
        )
        q[:, 0::2] = (packed & 0x0F).to(torch.float16)
        q[:, 1::2] = (packed >> 4).to(torch.float16)
        scale = self.scale.repeat_interleave(self.group_size, dim=1)
        zero_point = self.zero_point.to(torch.float16).repeat_interleave(self.group_size, dim=1)
        return ((q - zero_point) * scale).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequant_weight()
        return F.linear(x.to(weight.dtype), weight, None).to(x.dtype)


def _get_module(root: nn.Module, path: str) -> nn.Module:
    module: nn.Module = root
    for part in path.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    return module


def _set_module(root: nn.Module, path: str, value: nn.Module) -> None:
    parent_path, name = path.rsplit(".", 1)
    parent = _get_module(root, parent_path)
    if name.isdigit():
        parent[int(name)] = value  # type: ignore[index]
    else:
        setattr(parent, name, value)


def _replace_int4_linears(model: nn.Module, shard_path: str) -> int:
    replaced = 0
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for key in sorted(keys):
            if not key.endswith(".weight"):
                continue
            info = f.get_slice(key)
            if info.get_dtype() != "U8":
                continue
            module_name = key.removesuffix(".weight")
            if f"{module_name}.scale" not in keys or f"{module_name}.zero_point" not in keys:
                continue
            try:
                _get_module(model, module_name)
            except (AttributeError, IndexError, TypeError):
                continue
            _set_module(model, module_name, PackedInt4Linear(tuple(info.get_shape())))
            replaced += 1
    return replaced


def _install_a100_compat(model_module: Any) -> None:
    def dequant_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
        scale = getattr(weight, "scale", None)
        if scale is None:
            return weight.to(torch.bfloat16)
        out_dim, in_dim = weight.shape
        block = 128
        if out_dim % block != 0 or in_dim % block != 0:
            expanded = scale.float().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
            return (weight.float() * expanded[:out_dim, :in_dim]).to(torch.bfloat16)
        return (
            weight.float()
            .unflatten(0, (-1, block))
            .unflatten(-1, (-1, block))
            * scale.float()[:, None, :, None]
        ).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)

    def linear_compat(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        assert bias is None
        if weight.dtype == torch.float8_e4m3fn:
            return F.linear(x, dequant_fp8_weight(weight))
        if hasattr(torch, "float4_e2m1fn_x2") and weight.dtype == torch.float4_e2m1fn_x2:
            raise RuntimeError("A100 INT4 path should not instantiate FP4 expert weights")
        return F.linear(x, weight, bias)

    def act_quant_compat(
        x: torch.Tensor,
        block_size: int = 128,
        scale_fmt: str | None = None,
        scale_dtype: torch.dtype = torch.float32,
        inplace: bool = False,
    ) -> Any:
        if inplace:
            return x
        scale_shape = (*x.shape[:-1], (x.shape[-1] + block_size - 1) // block_size)
        return x, torch.ones(scale_shape, dtype=scale_dtype, device=x.device)

    def fp4_act_quant_compat(x: torch.Tensor, block_size: int = 32, inplace: bool = False) -> Any:
        if inplace:
            return x
        scale_shape = (*x.shape[:-1], (x.shape[-1] + block_size - 1) // block_size)
        return x, torch.ones(scale_shape, dtype=torch.float32, device=x.device)

    model_module.linear = linear_compat
    model_module.act_quant = act_quant_compat
    model_module.fp4_act_quant = fp4_act_quant_compat


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    reasoning_content: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0
    stream: bool = False
    thinking_mode: str | None = None
    reasoning_effort: str | None = None


def _resolve_thinking_mode(payload: dict[str, Any]) -> str:
    explicit = payload.get("thinking_mode")
    if isinstance(explicit, str):
        lowered = explicit.lower()
        if lowered in {"thinking", "think", "on", "true", "1"}:
            return "thinking"
        if lowered in {"chat", "off", "false", "0", "none"}:
            return "chat"

    effort = payload.get("reasoning_effort")
    if isinstance(effort, str) and effort.lower() in {"none", "off", "low", "chat"}:
        return "chat"
    return "thinking"


def _load_runtime(ckpt_path: str, config_path: str, inference_root: str) -> dict[str, Any]:
    _add_inference_paths(inference_root)

    from generate import generate
    from encoding_dsv4 import encode_messages, parse_message_from_completion_text
    import model as model_module
    from model import ModelArgs, Transformer

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    if A100_COMPAT:
        _install_a100_compat(model_module)
        if rank == 0:
            print("Installed A100 compatibility patch for FP8/BF16 fallback", flush=True)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["expert_dtype"] = "fp8"
    args = ModelArgs(**config)
    args.max_batch_size = 1

    if rank == 0:
        print(args, flush=True)

    shard_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    with torch.device("cuda"):
        model = Transformer(args)
        replaced = _replace_int4_linears(model, shard_path)
    if rank == 0:
        print(f"Replaced {replaced} packed INT4 linears before loading {shard_path}", flush=True)
    missing, unexpected = load_model(model, shard_path, strict=False)
    if rank == 0:
        print(f"load_model missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        for name in list(missing)[:80]:
            print(f"missing: {name}", flush=True)
        for name in list(unexpected)[:80]:
            print(f"unexpected: {name}", flush=True)
    model.eval()

    tokenizer = _load_tokenizer(ckpt_path)
    torch.set_default_device("cuda")

    return {
        "rank": rank,
        "world_size": world_size,
        "device": f"cuda:{local_rank}",
        "model": model,
        "tokenizer": tokenizer,
        "generate": generate,
        "encode_messages": encode_messages,
        "parse_message": parse_message_from_completion_text,
    }


def _run_generation(runtime: dict[str, Any], payload: dict[str, Any]) -> str | None:
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    generate = runtime["generate"]
    encode_messages = runtime["encode_messages"]
    parse_message = runtime["parse_message"]
    torch.set_default_device(runtime["device"])

    thinking_mode = _resolve_thinking_mode(payload)
    prompt = encode_messages(payload["messages"], thinking_mode=thinking_mode)
    prompt_tokens = tokenizer.encode(prompt)
    completion_tokens = generate(
        model,
        [prompt_tokens],
        int(payload.get("max_tokens", 64)),
        tokenizer.eos_token_id,
        float(payload.get("temperature", 0.0)),
    )
    completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
    try:
        parsed = parse_message(completion, thinking_mode=thinking_mode)
        return parsed.get("content", completion)
    except Exception:
        return completion


def _worker_loop(runtime: dict[str, Any]) -> None:
    while True:
        objects: list[Any] = [None]
        dist.broadcast_object_list(objects, src=0)
        payload = objects[0]
        if payload is None or payload.get("op") == "shutdown":
            return
        _run_generation(runtime, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--inference-root", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="deepseek-v4-flash-base-int4")
    args = parser.parse_args()

    runtime = _load_runtime(args.ckpt_path, args.config, args.inference_root)
    rank = runtime["rank"]
    world_size = runtime["world_size"]

    if rank != 0:
        _worker_loop(runtime)
        return

    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "model": args.served_model_name, "world_size": world_size}

    @app.post("/v1/chat/completions")
    def chat(req: ChatCompletionRequest) -> dict[str, Any]:
        if req.stream:
            raise HTTPException(status_code=400, detail="stream=true is not implemented")
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        if not messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        payload = {
            "op": "generate",
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "thinking_mode": req.thinking_mode,
            "reasoning_effort": req.reasoning_effort,
        }
        if world_size > 1:
            dist.broadcast_object_list([payload], src=0)
        content = _run_generation(runtime, payload)
        return {
            "id": "chatcmpl-dsv4-int4-modal",
            "object": "chat.completion",
            "model": args.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
