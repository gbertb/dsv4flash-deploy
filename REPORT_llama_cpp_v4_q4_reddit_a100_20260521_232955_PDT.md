# DeepSeek V4 Flash Q4 llama.cpp Reddit Follow-Up, A100 Only

Date: 2026-05-21 23:29 PDT

## Summary

This followed up on the Reddit post for `cchuter/llama.cpp` branch `feat/v4-port-cuda`, but kept the target hardware to Modal 4x A100-80GB. The relevant A100 items from the post were:

- Build CUDA for SM80: `-DCMAKE_CUDA_ARCHITECTURES=80`
- Use the V4-aware fork: `cchuter/llama.cpp`, branch `feat/v4-port-cuda`
- For multi-GPU, pass `-DGGML_SCHED_MAX_SPLIT_INPUTS=128` to both CXX and CUDA compiler flags
- Try the recommended Flash Q4 quant: `teamblobfish/DeepSeek-V4-Flash-GGUF`, `Q4_K_M-XL`
- Report prompt/decode t/s and crashes with build config

The new A100 experiments did not beat the existing best. The current best remains:

`llama_cpp_v4_q4_peer512_a100_modal.py`

Best measured decode:

- `12.9950 tok/s` on `sky_thinking_off`
- `12.9415 tok/s` on `sky_thinking_on`

## New Scripts

- `llama_cpp_v4_q4_modelcard_a100_modal.py`
  - A100 adaptation of the model-card-style runtime command
  - `Q4_K_M-XL`, `A100-80GB:4`, SM80, layer split
  - `-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512`
  - `--ctx-size 8192`
  - Forced `--flash-attn on`, `--reasoning off`, `--no-repack`

- `llama_cpp_v4_q4_row_manual_a100_modal.py`
  - Same A100/model-card settings, but with `--split-mode row`
  - Manual tensor split: `--tensor-split 1,1,1,1`
  - Main GPU: `--main-gpu 0`

- `llama_cpp_v4_q4_peer512_h100_modal.py`
  - Created before scope was narrowed to A100-only
  - H100 run was aborted and stopped; no benchmark result recorded

## A100 Results

| Script | Main difference | Prompt tok/s | Generated tok/s | Result |
| --- | --- | ---: | ---: | --- |
| `llama_cpp_v4_q4_peer512_a100_modal.py` | Prior best: layer split, launch queues, P2P, `PEER_MAX_BATCH_SIZE=512`, Flash Attention auto-disabled | 39.76 off, 44.23 on | 12.9950 off, 12.9415 on | Best measured |
| `llama_cpp_v4_q4_modelcard_a100_modal.py` | Reddit/model-card command shape with `--ctx-size 8192`, forced `--flash-attn on`, `--no-repack`, sampling flags | 42.7159 off, 51.9259 on | 10.8784 off, 9.7484 on | Slower |
| `llama_cpp_v4_q4_row_manual_a100_modal.py` | `--split-mode row --tensor-split 1,1,1,1 --main-gpu 0` | N/A | N/A | Failed at startup |

## Model-Card A100 Run

Command shape:

```bash
LLAMA_CPP_CMAKE_EXTRA_ARGS="-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512"
CUDA_SCALE_LAUNCH_QUEUES=4x
GGML_CUDA_P2P=1

llama-server \
  -m /models/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf \
  -c 8192 \
  -ngl 999 \
  --split-mode layer \
  --parallel 1 \
  --jinja \
  --cache-type-k q4_0 \
  --reasoning off \
  --flash-attn on \
  --no-repack \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --cache-ram 0 \
  --no-warmup \
  --batch-size 2048 \
  --ubatch-size 512 \
  --poll 100 \
  --poll-batch 1
```

Build/runtime evidence:

- `CUDA : ARCHS = 800 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 512`
- All layers offloaded across 4x A100-80GB
- DeepSeek4 KV cache forced to fp16 internally
- `flash_attn = enabled`

Measured benchmark:

- Warmup arithmetic: 9 completion tokens, 168.625s, prompt `11.2465 tok/s`, generation `11.2268 tok/s`
- `sky_thinking_off`: 116 completion tokens, 11.643s, prompt `42.7159 tok/s`, generation `10.8784 tok/s`
- `sky_thinking_on`: 203 completion tokens, 21.729s, prompt `51.9259 tok/s`, generation `9.7484 tok/s`

Conclusion: forcing Flash Attention and the model-card command shape is materially slower on A100 than the earlier peer512 run where Flash Attention was auto-disabled.

## Row/Manual A100 Run

Effective command included:

```bash
--split-mode row \
--tensor-split 1,1,1,1 \
--main-gpu 0 \
--reasoning off \
--flash-attn on \
--no-repack
```

The server failed before serving requests. The relevant error was:

```text
/opt/llama.cpp/ggml/src/ggml-backend.cpp:908: pre-allocated tensor (blk.0.attn_output_a.weight (reshaped)) in a buffer (CUDA0_Split) that cannot run the operation (RESHAPE)
```

Conclusion: row/manual split is accepted by the CLI for this run shape, but it is not viable for this DeepSeek4 Q4 configuration on 4x A100 in the cchuter branch. It fails during graph/buffer setup.

## Tensor Parallelism Status

The useful tensor-parallel equivalent in llama.cpp would be:

```bash
--split-mode tensor --tensor-split 1,1,1,1
```

That remains blocked for `LLM_ARCH_DEEPSEEK4` in this fork. The row/manual attempt is not the same as true tensor parallelism and also failed. All successful A100 runs so far use `--split-mode layer`.

## KV Cache Quantization Status

`--cache-type-k q4_0` remains part of the wrapper command for comparability, but this branch forces DeepSeek4 KV cache behavior to fp16 for the compressed/indexed K caches. In practice, the CLI flag does not deliver a working Q4/Q8 K-cache speedup for DeepSeek4 on this branch.

## MTP / NextN Status

The GGUF/model metadata exposes a next-token prediction layer count, but the cchuter branch does not wire DeepSeek4 MTP/NextN into `llama-server` speculative decode. There is no working server flag from this branch that enables MTP for this model today.

## A100 Next Experiments

The remaining useful A100-only experiments are narrower than the earlier sweep:

- Run the V4 custom op test target on A100: `test-backend-ops -o DSV4_ROPE_TAIL,DSV4_HC_SPLIT_SINKHORN,DSV4_HC_WEIGHTED_SUM,DSV4_HC_EXPAND,DSV4_FP8_KV_QUANTIZE`
- Add a deterministic `llama-bench` wrapper for short/medium context decode so flag differences are less prompt-dependent
- Try `Q8_0` only if enough aggregate VRAM plus overhead fits on 4x A100; nominal shard size is about `282 GiB`, so fit is tight after runtime buffers
- Revisit tensor split only if the fork adds DeepSeek4 tensor-split support
- Revisit MTP only if the fork wires DeepSeek4 NextN/MTP into llama.cpp serving

## Sources

- Reddit post: https://www.reddit.com/r/DeepSeek/comments/1tad2h3/deepseek_v4_in_llamacpp_flash_pro_cuda_metal/
- cchuter fork/branch referenced by the post: https://github.com/cchuter/llama.cpp/tree/feat/v4-port-cuda
- Flash GGUF model repo referenced by the post: https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF
- Previous local result report: `REPORT_llama_cpp_v4_q4_additional_flags_a100_20260521_225300_PDT.md`
