# llama.cpp DeepSeek V4 Modal Research Script

This is a research-only Modal script for DeepSeek V4 Flash on a llama.cpp fork.
Do not deploy it until we have validated the image build and a small smoke run.

## Findings

- Stock upstream llama.cpp does not yet have stable DeepSeek V4 Flash support.
- `Preyazz/DeepSeek-V4-Flash-GGUF` says its quants require llama.cpp PR
  `#22378` / `nisparks:wip/deepseek-v4-support`, and lists Q2/Q3/Q4 single-file
  GGUFs.
- A newer CUDA-focused fork is `cchuter/llama.cpp` branch `feat/v4-port-cuda`.
  It is not upstream, but reports CUDA + CPU + Metal support and publishes
  matching `teamblobfish/DeepSeek-V4-Flash-GGUF` quants.
- The safer first target for Modal is `teamblobfish` `Q2_K-XL`, not the DS4
  q4-imatrix file we already downloaded. `Q2_K-XL` is split into three GGUF
  shards and is documented by the fork author as a cleaner sub-Q4 option than
  the IQ-class quants.

## Script

Use:

```bash
rtk modal run llama_cpp_v4_modal.py --action inspect
```

Download the default `Q2_K-XL` shards:

```bash
rtk modal run llama_cpp_v4_modal.py --action download
```

Try a CLI smoke test after download:

```bash
rtk modal run llama_cpp_v4_modal.py --action cli-smoke
```

Start a dev web endpoint:

```bash
rtk modal run llama_cpp_v4_modal.py
```

Deploy only after the smoke test works:

```bash
rtk modal deploy llama_cpp_v4_modal.py
```

## Defaults

- llama.cpp repo: `https://github.com/cchuter/llama.cpp.git`
- branch: `feat/v4-port-cuda`
- HF model repo: `teamblobfish/DeepSeek-V4-Flash-GGUF`
- quant: `Q2_K-XL`
- GPU request: `H100:2`
- CUDA arch: `90`
- Modal volume: `llama-cpp-v4-flash-q2-k-xl`
- model path: `/models/Q2_K-XL/DeepSeek-V4-Flash-Q2_K-XL-00001-of-00003.gguf`

## Overrides

Use a smaller research quant:

```bash
LLAMA_CPP_MODEL_QUANT=IQ2_XXS-XL rtk modal run llama_cpp_v4_modal.py --action download
```

Use Q4:

```bash
LLAMA_CPP_MODEL_QUANT=Q4_K_M-XL LLAMA_CPP_GPU=H100:4 LLAMA_CPP_TENSOR_SPLIT=1,1,1,1 rtk modal run llama_cpp_v4_modal.py --action cli-smoke
```

Use a different llama.cpp fork or branch:

```bash
LLAMA_CPP_REPO=https://github.com/nisparks/llama.cpp.git LLAMA_CPP_BRANCH=wip/deepseek-v4-support rtk modal run llama_cpp_v4_modal.py --action inspect
```

Pass extra `llama-server` flags:

```bash
LLAMA_CPP_EXTRA_SERVER_ARGS="--cache-type-k q8_0 --cache-type-v q8_0" rtk modal run llama_cpp_v4_modal.py
```
