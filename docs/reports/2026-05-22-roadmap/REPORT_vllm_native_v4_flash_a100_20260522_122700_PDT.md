# DeepSeek V4 Flash Native vLLM A100 Probe

Date: 2026-05-22 12:27 PDT  
Roadmap phase: Phase 3, native vLLM DeepSeek V4 on A100  
Hardware target: Modal `A100-80GB:4`  
Script added: `deepseek_v4_flash_vllm_native_modal.py`

## Goal

Determine whether native vLLM DeepSeek V4 support can be a viable A100 path
before downloading and benchmarking the full official checkpoint.

The probe deliberately separated:

- source/config support checks
- remote checkpoint metadata
- A100 dummy-load engine startup

This avoids mistaking a documented or source-visible feature for a working A100
runtime.

## Native Command Shape

The generated no-MTP command was:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --uvicorn-log-level info \
  --disable-uvicorn-access-log \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256
```

The A100 dummy-load probe added:

```bash
--load-format dummy --enforce-eager
```

## Checkpoint Metadata

Remote model probe:

```text
repo: deepseek-ai/DeepSeek-V4-Flash
sha: 6976c7ff1b30a1b2cb7805021b8ba4684041f136
safetensors_count: 46
safetensors_total_gib: 148.65505419671535
```

This is small enough that fitting on 4x80 GB is plausible in principle, but the
dummy-load result below blocks native vLLM on A100 before real-weight benchmark
work is justified.

## vLLM Capability Probe

Image:

```text
vllm/vllm-openai:deepseekv4-cu130
```

Observed runtime:

```text
vllm_version: 0.1.dev15833+g62d441ee8
torch_version: 2.11.0+cu130
torch_cuda: 13.0
```

vLLM's internal parser recognized the model:

```text
config_model_type: deepseek_v4
architectures: ["DeepseekV4ForCausalLM"]
max_position_embeddings: 1048576
num_nextn_predict_layers: 1
```

Plain `transformers.AutoConfig` did not recognize `deepseek_v4` in this image,
but `vllm.transformers_utils.config.HFConfigParser` did. That means this image
depends on vLLM's bundled config path for V4 rather than stock Transformers.

## Source Evidence

Installed vLLM source contains native DeepSeek V4 files:

```text
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v4.py
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/deepseek_v4_attention.py
```

Source summary found:

- `config/speculative.py` includes `deepseek_mtp`.
- `config/speculative.py` maps DeepSeek V4 MTP to
  `DeepSeekV4MTPModel`.
- `model_executor/models/deepseek_v4.py` has MTP hidden-state buffering and
  target-model weight loading skips `mtp.` weights.
- `v1/core/kv_cache_utils.py` has DeepSeekV4-specific KV cache handling and
  flags the last layer as the MTP attention layer.

This proves native V4 and MTP code exists in the image. It does not prove A100
runtime viability.

## A100 Dummy-Load Result

Command:

```bash
rtk modal run deepseek_v4_flash_vllm_native_modal.py --action debug-dummy-short
```

The server reached the native DeepSeek V4 engine path:

```text
Resolved architecture: DeepseekV4ForCausalLM
Using max model len 4096
Chunked prefill is enabled with max_num_batched_tokens=4096
Using fp8 data type to store kv cache
Initializing a V1 LLM engine ...
tensor_parallel_size=4
load_format=dummy
quantization=deepseek_v4_fp8
kv_cache_dtype=fp8
```

It initialized 4-way NCCL on A100 and selected fallback-ish A100-compatible
pieces in some places:

```text
SymmMemCommunicator: Device capability 8.0 not supported
Mxfp4 MoE backend 'FLASHINFER_TRTLLM_MXFP4_MXFP8' does not support current device cuda
Mxfp4 MoE backend 'DEEPGEMM_MXFP4' does not support current device cuda
Using 'MARLIN' Mxfp4 MoE backend
Using DeepSeek's fp8_ds_mla KV cache format
Using FP8 indexer cache for Lighening Indexer
Model loading took 37.91 GiB memory and 6.338938 seconds
```

The probe failed during vLLM's dummy/profile run, before the OpenAI server became
ready:

```text
RuntimeError: Assertion error (/workspace/.deps/deepgemm-src/csrc/apis/hyperconnection.hpp:56): Unsupported architecture
```

Stack location:

```text
vllm/model_executor/models/deepseek_v4.py:490 hc_pre
vllm/model_executor/layers/mhc.py:263 mhc_pre
vllm/utils/deep_gemm.py:479 tf32_hc_prenorm_gemm
```

Engine startup then failed:

```text
RuntimeError: Engine core initialization failed. See root cause above.
```

## Interpretation

Native vLLM DeepSeek V4 support is present in this image, including MTP plumbing,
but this exact runtime is not currently a working 4x A100 path. The blocker is
not checkpoint download size, tokenizer parsing, or TP startup; it is an SM80
unsupported-architecture assertion in the DeepGEMM hyperconnection/MHC path.

Because the no-MTP dummy path fails, full native-weight download/benchmark and
MTP benchmarking should be deferred until one of these is true:

- vLLM adds an A100-compatible MHC/hyperconnection fallback.
- A patched Ampere branch disables or replaces the DeepGEMM MHC path.
- A runtime flag exists to force an A100-safe implementation for
  `torch.ops.vllm.mhc_pre`.

## Roadmap Decision

Do not run the expensive full native vLLM checkpoint benchmark on this image yet.
It would likely fail at the same A100 MHC kernel path after downloading real
weights.

Next useful vLLM work:

1. Search or patch for an Ampere-safe `mhc_pre` implementation.
2. Re-run `debug-dummy-short`.
3. Only if dummy startup reaches server readiness, run the full native no-MTP
   load and then the online matrix.
4. Add `--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}'`
   only after no-MTP startup and smoke tests pass.
