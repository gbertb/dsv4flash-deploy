# SGLang Native DeepSeek V4 Flash A100 Probe

Date: 2026-05-22 12:51:09 PDT

## Scope

Probe whether current SGLang native DeepSeek V4 support can run on 4x A100 well
enough to justify a full 148.7 GiB checkpoint benchmark with MTP and cache tests.

This was a startup/kernel probe, not a throughput benchmark. It used
`--load-format dummy` to avoid downloading full weights before validating the
A100 execution path.

## Environment

- Image: `lmsysorg/sglang:latest`
- SGLang: `0.5.12`
- Python used for SGLang: `/usr/bin/python3`
- Torch: `2.11.0+cu130`
- CUDA: `13.0`
- GPU target: `A100-80GB:4`
- Model: `deepseek-ai/DeepSeek-V4-Flash`
- Model revision: `6976c7ff1b30a1b2cb7805021b8ba4684041f136`
- Remote weights: 46 safetensors, `148.655 GiB`

The container also has Modal's injected `/usr/local/bin/python3`, but SGLang is
installed in the base image environment. The probe explicitly uses
`/usr/bin/python3`.

## Command

```bash
/usr/bin/python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 4 \
  --context-length 4096 \
  --mem-fraction-static 0.88 \
  --trust-remote-code \
  --reasoning-parser deepseek-v4 \
  --tool-call-parser deepseekv4 \
  --attention-backend dsv4 \
  --prefill-attention-backend dsv4 \
  --decode-attention-backend dsv4 \
  --disable-cuda-graph \
  --load-format dummy
```

## Source/Feature Evidence

SGLang exposes the required DeepSeek V4 launch knobs:

- `--reasoning-parser deepseek-v4`
- `--tool-call-parser deepseekv4`
- `--attention-backend dsv4`
- `--prefill-attention-backend dsv4`
- `--decode-attention-backend dsv4`
- `--kv-cache-dtype`
- `--speculative-algorithm`
- `--speculative-num-draft-tokens`
- `--moe-runner-backend`
- `--enable-hierarchical-cache`

Source inspection found native DeepSeek V4 modules:

- `srt/models/deepseek_v4.py`
- `srt/models/deepseek_v4_nextn.py`
- `srt/configs/deepseek_v4.py`
- `srt/layers/attention/deepseek_v4_backend.py`
- `srt/mem_cache/deepseek_v4_memory_pool.py`
- `srt/layers/mhc.py`
- `srt/layers/mhc_head.py`
- parser support for `deepseek-v4` and `deepseekv4`

## A100 Dummy Startup Result

The server reached readiness:

- SGLang recognized `DeepseekV4ForCausalLM`.
- It selected the DSV4 attention backend and `page_size=256`.
- It set `max_running_requests=256`.
- It set KV cache dtype to `fp8_e4m3`.
- It initialized 4-way tensor parallel NCCL.
- It detected the fp8 checkpoint and selected A100 fallback behavior:
  `Weight-only FP8 compression will be used leveraging the Marlin kernel`.
- It loaded dummy weights at about `44.23 GB` per GPU.
- It initialized DeepSeek V4 KV pools:
  `full=1628672`, `swa=162816`, `c4=407168`, `c128=12724`.
- It started Uvicorn and served `/model_info`.

The first warmup/forward path then failed in DSV4 metadata top-k planning:

```text
RuntimeError: ninja exited with status 1
nvcc ... -gencode=arch=compute_80,code=sm_80 ...
/sgl-workspace/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh(247):
error: __cluster_dims__ is not supported for this GPU architecture
/sgl-workspace/sglang/python/sglang/jit_kernel/csrc/deepseek_v4/topk_v2.cuh(322):
error: __cluster_dims__ is not supported for this GPU architecture
error: namespace "cooperative_groups" has no member "this_cluster"
```

## Conclusion

Current SGLang native DeepSeek V4 support is real, but the default DSV4 forward
path is not A100-safe. It uses a JIT top-k kernel requiring cluster launch
features unavailable on `sm_80`. A full native checkpoint benchmark is therefore
not justified on A100 until SGLang provides an Ampere-compatible DSV4 top-k
metadata path or a documented fallback.

Do not run MTP benchmarking on this SGLang image yet. MTP would stack on top of
the same DSV4 forward path, and the non-MTP dummy path already fails on A100.

## Modal Cleanup

Post-run checks showed no active Modal containers. The SGLang probe apps were in
`stopped` state.
