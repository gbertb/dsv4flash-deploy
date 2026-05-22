# DeepSeek V4 Flash Q4 llama.cpp Additional Flag Sweep

Date: 2026-05-21 22:53 PDT

## Summary

This continued the Q4_K_M-XL llama.cpp optimization work on Modal 4x A100-80GB without changing context as the primary lever. The new best measured variant is:

`llama_cpp_v4_q4_peer512_a100_modal.py`

It reached:

- `12.995 tok/s` on `sky_thinking_off`
- `12.941 tok/s` on `sky_thinking_on`

That narrowly beats the previous best `llama_cpp_v4_q4_fastpipe_a100_modal.py` result of about `12.88 tok/s`.

## New Scripts

- `llama_cpp_v4_q4_flashon_a100_modal.py`
  - Same fastpipe runtime settings, plus `--flash-attn on`
- `llama_cpp_v4_q4_mmq_a100_modal.py`
  - Same fastpipe runtime settings, plus build flag `-DGGML_CUDA_FORCE_MMQ=ON`
- `llama_cpp_v4_q4_peer512_a100_modal.py`
  - Same fastpipe runtime settings, plus build flag `-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512`

## Results

| Script | Main difference | Prompt tok/s | Generated tok/s | Result |
| --- | --- | ---: | ---: | --- |
| `llama_cpp_v4_q4_fastpipe_a100_modal.py` | Prior best: launch queues, P2P, batch/ubatch/poll tuning | 40.73 off, 66.24 on | 12.8779 off, 12.8854 on | Previous best |
| `llama_cpp_v4_q4_flashon_a100_modal.py` | Forced `--flash-attn on` | 42.00 off, 55.16 on | 10.8410 off, 9.9674 on | Slower |
| `llama_cpp_v4_q4_mmq_a100_modal.py` | Forced CUDA MMQ quant matmul kernels | 39.18 off, 62.72 on | 12.3590 off, 12.5042 on | Slower than fastpipe |
| `llama_cpp_v4_q4_peer512_a100_modal.py` | Raised `GGML_CUDA_PEER_MAX_BATCH_SIZE` from 128 to 512 | 39.76 off, 44.23 on | 12.9950 off, 12.9415 on | Best measured |

## Tensor Parallelism And Single-GPU Fit

llama.cpp tensor parallelism is not a numeric `tensor_parallel_size=4` style flag. The equivalent is:

```bash
--split-mode tensor --tensor-split 1,1,1,1
```

or just `--split-mode tensor` with four visible GPUs if the implementation supports that architecture. This DeepSeek4 branch does not support tensor split for `LLM_ARCH_DEEPSEEK4`, so all successful runs remain `--split-mode layer`, i.e. pipeline parallelism.

The Q4_K_M-XL model cannot fit fully on one A100-80GB. Runtime evidence from llama.cpp:

- Model file size: `162.86 GiB`
- Model params: `284.33 B`
- Device memory: 4x `81152 MiB`
- Layer-split model buffers:
  - CUDA0: `43243.10 MiB`
  - CUDA1: `41122.48 MiB`
  - CUDA2: `41667.69 MiB`
  - CUDA3: `40192.41 MiB`

So a single 80GB A100 cannot hold the full Q4 model. It would need CPU offload or a lower quant, and CPU offload would be much slower.

## Flash Attention

Auto Flash Attention was disabled in the fastpipe/MMQ/peer512 runs:

`Flash Attention was auto, set to disabled`

The log reason was:

`layer 2 is assigned to device CUDA0 but the Flash Attention tensor is assigned to device CPU`

Forcing `--flash-attn on` did work, but it was slower on this workload. It increased compute-buffer memory substantially and dropped decode to roughly 10 tok/s. For this branch/model/run shape, Flash Attention is not a speed win.

## MMQ

`-DGGML_CUDA_FORCE_MMQ=ON` did change the backend path:

`CUDA : ARCHS = 800 | FORCE_MMQ = 1 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128`

It also lowered compute-buffer memory versus the default path, but it did not improve decode speed:

- `12.3590 tok/s` thinking off
- `12.5042 tok/s` thinking on

This suggests the default CUDA kernel choice is better than forced MMQ on A100 for this DeepSeek4 Q4 path.

## Peer Max Batch Size

`-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512` changed the build info as intended:

`CUDA : ARCHS = 800 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 512`

This is the best measured setting so far. It is a small win, but it is a real non-context optimization over the previous best.

## Current Best Command Shape

The best wrapper currently maps to:

```bash
LLAMA_CPP_CMAKE_EXTRA_ARGS="-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512"
CUDA_SCALE_LAUNCH_QUEUES=4x
GGML_CUDA_P2P=1

llama-server \
  -m /models/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf \
  -c 4096 \
  -ngl 999 \
  --split-mode layer \
  --parallel 1 \
  --jinja \
  --cache-type-k q4_0 \
  --cache-ram 0 \
  --no-warmup \
  --batch-size 2048 \
  --ubatch-size 512 \
  --poll 100 \
  --poll-batch 1
```

Note: `--cache-type-k q4_0` is still effectively ignored for DeepSeek4 in this branch because the branch forces f16 KV cache for correctness.

## Updated Conclusion

We have now moved the verified Q4 4x A100 llama.cpp deployment above the original 12.5 tok/s target:

- Previous target: `> 12.5 tok/s`
- Previous best: about `12.88 tok/s`
- Current best: about `13.0 tok/s`

The next likely material improvement is not another simple server flag. The high-leverage items remain:

- DeepSeek4 tensor-split support in llama.cpp
- DeepSeek4 MTP/NextN speculative decode wired into llama-server
- A compatible draft GGUF model for draft-model speculative decoding
- A deterministic `llama-bench` harness for longer, less stochastic comparisons

## Sources

- llama.cpp build docs, CUDA env and performance flags: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- llama.cpp multi-GPU docs, split modes and tensor restrictions: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- llama.cpp speculative decoding docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- Local cchuter DeepSeek4 branch source inspected from a temporary checkout, commit `781e978`.
