# DeepSeek V4 Flash Deployment Benchmark Report

Timestamp: 2026-05-21 23:44 PDT

## Executive Summary

The best working deployment in this repo is the llama.cpp GGUF path using:

- Engine: `cchuter/llama.cpp`, branch `feat/v4-port-cuda`
- Model: `teamblobfish/DeepSeek-V4-Flash-GGUF`
- Quant: `Q4_K_M-XL`
- Hardware: Modal `A100-80GB:4`
- Split mode: `--split-mode layer`
- Script: `llama_cpp_v4_q4_peer512_a100_modal.py`

Best measured decode speed:

- `12.9950 tok/s` with thinking off
- `12.9415 tok/s` with thinking on

The official DeepSeek inference path and EnsueAI INT4 Base path both ran on 4x A100 only after BF16 compatibility fallbacks, but they were much slower and/or had chat-output quality issues. DS4 CUDA, row/manual split, tensor split, vLLM GGUF, and forced Flash Attention did not produce a better deployment.

## Result Chart

Higher is better. Bars use the best preserved generated-token throughput per script.

```text
llama_cpp_q4_peer512      12.995 tok/s | ##########################
llama_cpp_q4_fastpipe     12.885 tok/s | #########################
llama_cpp_q4_cublas       12.698 tok/s | #########################
llama_cpp_q4_fastctx      12.540 tok/s | #########################
llama_cpp_q4_baseline     12.400 tok/s | ########################
llama_cpp_q4_mmq          12.504 tok/s | #########################
llama_cpp_q4_modelcard    10.878 tok/s | ######################
llama_cpp_q4_flashon      10.841 tok/s | ######################
llama_cpp_q4_spec_ngram   10.989 tok/s | ######################
official_a100_compat      slow         | functional, about 65s short prompt
base_int4_a100_compat     slow         | functional, 4.190s warm short prompt
ds4_cuda_q4               failed       | CUDA prefill failed
row_manual                failed       | CUDA split buffer RESHAPE error
tensor_probe              not run      | DeepSeek4 tensor split unsupported
vllm_gguf_q8              blocked      | exploratory GGUF mapping/plugin path
```

## Main Benchmark Table

| Rank | Script / path | Engine | Model / quant | Hardware | Key config | Result |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `llama_cpp_v4_q4_peer512_a100_modal.py` | llama.cpp fork | `teamblobfish`, `Q4_K_M-XL` | `A100-80GB:4` | `-c 4096`, layer split, P2P, launch queues, batch/ubatch tuning, `GGML_CUDA_PEER_MAX_BATCH_SIZE=512` | Best: `12.9950` off, `12.9415` on |
| 2 | `llama_cpp_v4_q4_fastpipe_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | `-c 4096`, layer split, `CUDA_SCALE_LAUNCH_QUEUES=4x`, `GGML_CUDA_P2P=1`, batch/poll tuning | `12.8779` off, `12.8854` on |
| 3 | `llama_cpp_v4_q4_cublas_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | Fastpipe plus `-DGGML_CUDA_FORCE_CUBLAS=ON` | `12.678` off, `12.698` on |
| 4 | `llama_cpp_v4_q4_fastctx_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | `-c 4096`, `--flash-attn off`, `--cache-ram 0`, `--no-warmup` | `12.45` off, `12.54` on |
| 5 | `llama_cpp_v4_q4_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | `-c 32768`, layer split, `--jinja`, `--cache-type-k q4_0` | `12.25` off, `12.40` on |
| 6 | `llama_cpp_v4_q4_mmq_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | Fastpipe plus `-DGGML_CUDA_FORCE_MMQ=ON` | `12.3590` off, `12.5042` on |
| 7 | `llama_cpp_v4_q4_modelcard_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | `-c 8192`, peer512, forced `--flash-attn on`, `--reasoning off`, `--no-repack` | `10.8784` off, `9.7484` on |
| 8 | `llama_cpp_v4_q4_flashon_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | Fastpipe plus forced `--flash-attn on` | `10.8410` off, `9.9674` on |
| 9 | `llama_cpp_v4_q4_spec_ngram_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | Fastpipe plus `--spec-type ngram-mod` | `10.616` off, `10.989` on; zero drafts |
| - | `llama_cpp_v4_q4_row_manual_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | `--split-mode row --tensor-split 1,1,1,1 --main-gpu 0` | Failed startup: CUDA split buffer cannot run `RESHAPE` |
| - | `llama_cpp_v4_q4_tensor_probe_a100_modal.py` | llama.cpp fork | `Q4_K_M-XL` | `A100-80GB:4` | `--split-mode tensor`, f16 KV | Not run; source inspection showed `LLM_ARCH_DEEPSEEK4` tensor split unsupported |
| - | `llama_cpp_v4_q2_a100_modal.py` | llama.cpp fork | `Q2_K-XL` | `A100-80GB:4` | Same base Modal llama.cpp wrapper, lower quant | Rejected qualitatively: lower quality and throughput than Q4; exact tok/s not preserved in reports |
| - | `deepseek_v4_flash_official_modal.py` | official DeepSeek inference | `deepseek-ai/DeepSeek-V4-Flash`, MP4 converted | `A100-80GB:4` | `torchrun --nproc-per-node 4`, A100 BF16 compatibility patch | Functional but slow; short arithmetic prompt took about 65s |
| - | `deepseek_v4_flash_int4_modal.py` | official inference plus custom INT4 loader | `EnsueAI/DeepSeek-V4-Flash-Base-INT4` | `A100-80GB:4` | TP=4 safetensors, packed INT4 linears dequantized in PyTorch, A100 BF16 fallback | Functional; warm short request `4.190s`, but Base checkpoint produced extra decoded text |
| - | `ds4_modal.py` | `antirez/ds4` | DS4 q4/q2 GGUF variants | CPU or `A100-80GB:4` CUDA | DS4 server, q4/q2 imatrix variants | q4 CUDA failed prefill; q4 CPU failed expert tensor expectation |
| - | `deepseek_v4_flash_gguf_modal.py` | vLLM + GGUF plugin | `Preyazz/DeepSeek-V4-Flash-Q8_0-GGUF` | `A100-80GB:4` | vLLM OpenAI image, patched GGUF plugin, max len 8192 | Exploratory and not the winning path; superseded by teamblobfish + cchuter llama.cpp |

## Best Reproducible Configuration

Use:

```bash
rtk modal run llama_cpp_v4_q4_peer512_a100_modal.py --action benchmark
```

The wrapper maps to:

```bash
LLAMA_CPP_MODEL_QUANT=Q4_K_M-XL
LLAMA_CPP_GPU=A100-80GB:4
LLAMA_CPP_CUDA_ARCH=80
LLAMA_CPP_CTX=4096
LLAMA_CPP_SPLIT_MODE=layer
LLAMA_CPP_CMAKE_EXTRA_ARGS="-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512"
CUDA_SCALE_LAUNCH_QUEUES=4x
GGML_CUDA_P2P=1
```

Effective server shape:

```bash
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

Important caveat: `--cache-type-k q4_0` is accepted but effectively ignored for DeepSeek4 in the tested cchuter branch. The branch forces V4 KV cache dtype to fp16 because the compressed/indexer K-cache implementation requires shared dtype/view behavior.

## Deployment Pattern

All successful benchmark work used Modal through local Python scripts and the `rtk` command wrapper. The repo pattern is:

1. Build a Modal image from CUDA 12.8 or a relevant runtime image.
2. Download model files into a persistent Modal Volume.
3. Run inspect/backend-op/smoke/benchmark actions with `rtk modal run`.
4. Serve OpenAI-compatible endpoints only after a smoke test succeeds.
5. After every Modal action, check:

```bash
rtk modal container list --json
rtk modal app list --json
```

6. Stop lingering containers or apps immediately, especially 4x A100 apps:

```bash
rtk modal container stop <container-id>
rtk modal app stop <app-id-or-name>
```

### llama.cpp Deployment

Base implementation: `llama_cpp_v4_q4_a100_modal.py`.

It clones:

```text
https://github.com/cchuter/llama.cpp.git
branch: feat/v4-port-cuda
```

It builds:

```bash
cmake -B build -G Ninja \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_C_FLAGS='-DGGML_SCHED_MAX_SPLIT_INPUTS=128' \
  -DCMAKE_CXX_FLAGS='-DGGML_SCHED_MAX_SPLIT_INPUTS=128' \
  -DCMAKE_CUDA_FLAGS='-DGGML_SCHED_MAX_SPLIT_INPUTS=128' \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-cli test-backend-ops
```

The important validated test command was:

```bash
rtk modal run llama_cpp_v4_q4_a100_modal.py --action backend-ops
```

Result: all five V4 CUDA custom op groups passed, `19/19` each, on 4x A100.

### Official DeepSeek Deployment

Scripts:

- `deepseek_v4_flash_official_modal.py`
- `dsv4_official_server.py`

Flow:

- Download `deepseek-ai/DeepSeek-V4-Flash`.
- Convert to MP=4 shards.
- Serve with `torchrun --nproc-per-node 4`.
- Patch the official runtime on A100 to avoid unsupported native FP8/FP4 kernels by dequantizing to BF16 and using `F.linear`.

Result: it ran and answered arithmetic correctly, but was too slow for the target deployment.

### Base INT4 Deployment

Scripts:

- `deepseek_v4_flash_int4_modal.py`
- `dsv4_int4_server.py`

Flow:

- Download `EnsueAI/DeepSeek-V4-Flash-Base-INT4`.
- Use official inference code for config/encoding.
- Replace packed INT4 linear layers with a dequantizing PyTorch module.
- Run with 4-way tensor parallelism.

Result: endpoint worked and returned `323`; warm short request was `4.190s`. It was not selected because the Base checkpoint produced extra decoded/template-like text and was poor for chat-shaped serving.

### DS4 Deployment

Script: `ds4_modal.py`.

Flow:

- Clone `antirez/ds4`.
- Build DS4 CUDA with `CUDA_ARCH=sm_80` or run CPU.
- Try q4/q2 imatrix GGUF variants.

Result:

- q4 CUDA failed with `ds4: prompt processing failed: cuda prefill failed`.
- q4 CPU failed with `ds4: expected IQ2_XXS expert tensors`.
- DS4 is not a useful 4x A100 tensor/model-parallel path for this repo as tested.

### vLLM GGUF Deployment

Scripts:

- `deepseek_v4_flash_gguf_modal.py`
- `patch_vllm_gguf_plugin.py`

Flow:

- Use `vllm/vllm-openai:deepseekv4-cu130`.
- Install and patch `vllm-gguf-plugin`.
- Target `Preyazz/DeepSeek-V4-Flash-Q8_0-GGUF`.

Result: exploratory path only. It was superseded by the cchuter llama.cpp fork plus teamblobfish GGUFs. Keep it as historical context if vLLM GGUF support improves.

## Findings By Technique

### Quant Choice

`Q4_K_M-XL` is the best tested quality/speed tradeoff. `Q2_K-XL` was tested enough to reject qualitatively because quality and throughput were worse than Q4. Sub-Q4/IQ quants were not the selected path because the research notes warned of tool-call JSON issues and lower quality.

### Multi-GPU Mode

All successful llama.cpp runs used:

```bash
--split-mode layer
```

This is pipeline/layer splitting, not true tensor parallelism. True tensor split would look like:

```bash
--split-mode tensor --tensor-split 1,1,1,1
```

but the inspected cchuter branch did not support tensor split for `LLM_ARCH_DEEPSEEK4`.

### Single-GPU Fit

The Q4 model does not fit on one A100-80GB:

- Q4_K_M-XL file size: `162.86 GiB`
- Four A100 model buffers in the successful split were roughly `43.2`, `41.1`, `41.7`, and `40.2 GiB`
- A single 80GB A100 would need CPU offload or a lower quant, likely much slower

### Flash Attention

Auto Flash Attention was disabled in the best runs. Forcing it on made performance worse:

- Forced flash-on fastpipe variant: about `10.84` off and `9.97` on
- Model-card forced flash variant: about `10.88` off and `9.75` on

For this A100 + DeepSeek4 + cchuter branch shape, forced Flash Attention is not a speed win.

### KV Cache Quantization

The tested llama.cpp branch forces DeepSeek4 KV cache to fp16. `--cache-type-k q4_0` and q8-style attempts should not be treated as active optimizations until the branch changes its V4 KV-cache implementation.

### Speculative Decode / MTP

Ngram speculative decode ran but generated zero useful drafts and slowed the benchmark. The GGUF metadata indicates NextN/MTP-related data, but the tested cchuter branch did not wire DeepSeek4 MTP into `llama-server`.

### A100 FP8 Reality

A100 is SM80. It does not have the native FP8 path used on Ada/Hopper/Blackwell. The llama.cpp fork builds and passes V4 op tests on A100, but the official DeepSeek FP8 runtime needed BF16 fallback patches and was slow.

## Source Reports In This Repo

- `REPORT_llama_cpp_v4_q4_a100_20260521_212811_PDT.md`
- `REPORT_llama_cpp_v4_q4_fastctx_a100_20260521_215926_PDT.md`
- `REPORT_llama_cpp_v4_q4_optimization_flags_a100_20260521_223340_PDT.md`
- `REPORT_llama_cpp_v4_q4_additional_flags_a100_20260521_225300_PDT.md`
- `REPORT_llama_cpp_v4_q4_reddit_a100_20260521_232955_PDT.md`
- `DSV4_INT4_MODAL.md`
- `CHANGELOG_deepseek_v4_flash_base_int4.md`
- `CHANGELOG_deepseek_v4_flash_official.md`
- `DS4_MODAL.md`
- `LLAMA_CPP_V4_MODAL.md`
- `GGUF_quantization_llama.cpp.md`

## Future Agent Instructions

Use this section as the handoff prompt for a future agent.

```text
You are working from the repository root.

Goal: review the existing DeepSeek V4 Flash A100 deployment experiments, then determine whether newer techniques, engine support, or repo updates make a better deployment possible.

Constraints:
- Target hardware remains Modal A100-80GB:4 unless the user explicitly changes it.
- Do not overwrite existing scripts. Create new timestamped or descriptive scripts.
- Prefix shell commands with rtk.
- After every Modal run/deploy/debug, check:
  rtk modal container list --json
  rtk modal app list --json
  Stop any lingering experiment containers/apps.
- Preserve the current best baseline:
  llama_cpp_v4_q4_peer512_a100_modal.py
  Q4_K_M-XL, 4x A100, cchuter llama.cpp feat/v4-port-cuda, layer split
  best decode about 12.995 tok/s thinking off and 12.941 tok/s thinking on.

Local files to read first:
- BENCHMARK_REPORT_*.md, newest timestamp
- REPORT_llama_cpp_v4_q4_additional_flags_a100_20260521_225300_PDT.md
- REPORT_llama_cpp_v4_q4_reddit_a100_20260521_232955_PDT.md
- llama_cpp_v4_q4_a100_modal.py
- llama_cpp_v4_q4_peer512_a100_modal.py
- DSV4_INT4_MODAL.md
- CHANGELOG_deepseek_v4_flash_official.md
- DS4_MODAL.md

Research current upstream state before changing code:
- cchuter/llama.cpp, branch feat/v4-port-cuda
- ggml-org/llama.cpp, especially DeepSeek V3.2/V4 architecture support, CUDA kernels, split modes, speculative decode, KV cache dtype handling
- vllm-project/vllm DeepSeek V4 support, FP4/FP8 support, A100 compatibility, tensor parallel serving
- vllm-project/vllm-gguf-plugin or any replacement GGUF loading path
- deepseek-ai/DeepSeek-V4-Flash official inference repo
- teamblobfish/DeepSeek-V4-Flash-GGUF model card and new quants
- EnsueAI/DeepSeek-V4-Flash-Base-INT4 and Intel/DeepSeek-V4-Flash-W4A16-AutoRound
- SGLang support for DeepSeek V4, FP4/FP8, GGUF, W4A16, and A100

Specific things to look for:
- llama.cpp adds tensor split support for LLM_ARCH_DEEPSEEK4
- llama.cpp wires DeepSeek4 NextN/MTP into llama-server speculative decoding
- llama.cpp changes DeepSeek4 KV cache dtype handling so q4/q8 cache is actually usable
- llama.cpp improves A100/SM80 FP8 emulation or compressed KV kernels
- vLLM gains stable DeepSeek V4 Flash support on A100 for FP4/FP8 or W4A16
- vLLM or another engine gains stable GGUF DeepSeek V4 loading
- SGLang gains a working DeepSeek V4 Flash A100 path
- new GGUF quants appear with better quality/speed than Q4_K_M-XL
- new draft models or built-in MTP support become usable for speculative decoding

Experiment protocol:
1. Start by reproducing the current best benchmark if needed.
2. Change one major variable per new script.
3. Keep result prompts comparable:
   - warmup arithmetic prompt
   - sky prompt thinking off
   - sky prompt thinking on
4. Record prompt tok/s, generated tok/s, elapsed time, completion tokens, quality notes, and failure traces.
5. Write a new timestamped REPORT_*.md after each sweep.
6. Update the latest BENCHMARK_REPORT_*.md only by creating a new timestamped benchmark report, not by editing old reports.

Do not spend Modal credits on H100/Hopper unless the user explicitly changes the hardware target.
```

## Recommended Next Work

The most useful next A100-only improvement is not another simple flag sweep. Prioritize code or engine changes that unlock one of:

- DeepSeek4 tensor parallelism in llama.cpp
- DeepSeek4 MTP/NextN speculative decode in llama.cpp
- Real KV cache quantization support for DeepSeek4
- A vLLM/SGLang A100 path that supports the native DeepSeek V4 checkpoint or W4A16 without BF16 fallback overhead
- A deterministic `llama-bench` wrapper for this repo, so short-prompt chat timing noise does not dominate small changes
