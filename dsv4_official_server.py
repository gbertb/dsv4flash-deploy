from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from safetensors.torch import load_model
from transformers import PreTrainedTokenizerFast
import uvicorn


A100_COMPAT = os.getenv("DSV4_A100_COMPAT", "1") not in {"0", "false", "False"}


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


def _install_a100_compat(model_module: Any) -> None:
    import torch.nn.functional as F

    def dequant_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
        scale = getattr(weight, "scale", None)
        if scale is None:
            return weight.to(torch.bfloat16)
        out_dim, in_dim = weight.shape
        block = 128
        if out_dim % block != 0 or in_dim % block != 0:
            scale = scale.float().repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
            return (weight.float() * scale[:out_dim, :in_dim]).to(torch.bfloat16)
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
            raise RuntimeError("A100 compat requires FP4 experts converted to FP8 first")
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
    temperature: float = 1.0
    stream: bool = False
    reasoning_effort: str | None = None


def _load_runtime(ckpt_path: str, config_path: str, inference_root: str) -> dict[str, Any]:
    _add_inference_paths(inference_root)

    from encoding_dsv4 import encode_messages, parse_message_from_completion_text
    from generate import generate
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
            print("Installed A100 compatibility patch: FP8 GEMM and activation quantization disabled", flush=True)

    with open(config_path, "r", encoding="utf-8") as f:
        args = ModelArgs(**json.load(f))
    args.max_batch_size = 1

    if rank == 0:
        print(args, flush=True)

    with torch.device("cuda"):
        model = Transformer(args)

    shard_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if rank == 0:
        print(f"Loading model shard prefix from {ckpt_path}", flush=True)
    load_model(model, shard_path, strict=False)
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
    encode_messages = runtime["encode_messages"]
    parse_message = runtime["parse_message"]
    generate = runtime["generate"]
    torch.set_default_device(runtime["device"])

    messages = payload["messages"]
    max_tokens = int(payload.get("max_tokens", 64))
    temperature = float(payload.get("temperature", 1.0))
    thinking_mode = payload.get("thinking_mode", "chat")

    prompt = encode_messages(messages, thinking_mode=thinking_mode)
    prompt_tokens = tokenizer.encode(prompt)
    completion_tokens = generate(
        model,
        [prompt_tokens],
        max_tokens,
        tokenizer.eos_token_id,
        temperature,
    )
    completion = tokenizer.decode(completion_tokens[0])

    try:
        parsed = parse_message(completion, thinking_mode=thinking_mode)
        content = parsed.get("content", completion)
    except Exception:
        content = completion
    return content


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
    parser.add_argument("--served-model-name", default="deepseek-v4-flash")
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

        thinking_mode = "chat"
        if req.reasoning_effort and req.reasoning_effort.lower() not in {"none", "low"}:
            thinking_mode = "thinking"

        payload = {
            "op": "generate",
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "thinking_mode": thinking_mode,
        }
        if world_size > 1:
            dist.broadcast_object_list([payload], src=0)
        content = _run_generation(runtime, payload)
        return {
            "id": "chatcmpl-dsv4-modal",
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
