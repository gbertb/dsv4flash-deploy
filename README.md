# DeepSeek V4 Flash A100 Deployment

This repository tracks experiments for deploying DeepSeek V4 Flash on Modal with 4x NVIDIA A100-80GB GPUs. The goal is to identify a practical, reproducible serving path for Ampere hardware, document failed approaches, and preserve benchmark evidence for future optimization work.

For the full benchmark details, methodology, command history, caveats, and future-agent handoff notes, read [BENCHMARK_REPORT_20260521_234412_PDT.md](BENCHMARK_REPORT_20260521_234412_PDT.md).

## Current Best Path

The best working deployment found so far is the llama.cpp GGUF path:

- Engine: `cchuter/llama.cpp`, branch `feat/v4-port-cuda`
- Model: `teamblobfish/DeepSeek-V4-Flash-GGUF`
- Quantization: `Q4_K_M-XL`
- Hardware: Modal `A100-80GB:4`
- Split mode: `--split-mode layer`
- Script: `llama_cpp_v4_q4_peer512_a100_modal.py`

Best measured decode throughput:

- Thinking off: `12.9950 tok/s`
- Thinking on: `12.9415 tok/s`

Reproduce the current benchmark baseline with:

```bash
rtk modal run llama_cpp_v4_q4_peer512_a100_modal.py --action benchmark
```

## Summary Of Findings

- `Q4_K_M-XL` is the best tested quality and speed tradeoff. Lower `Q2_K-XL` quantization was rejected because both quality and throughput were worse.
- All successful llama.cpp runs used layer splitting across 4x A100s. Tensor splitting was inspected but not usable for `LLM_ARCH_DEEPSEEK4` in the tested branch.
- The Q4 model does not fit on a single A100-80GB without offload or lower quantization. Successful 4-GPU model buffers were roughly 40-43 GiB per GPU.
- Forced Flash Attention reduced performance in these tests. The best runs left auto Flash Attention disabled.
- DeepSeek4 KV cache quantization was not active in practice. The tested llama.cpp branch forced V4 KV cache dtype to fp16 even when `--cache-type-k q4_0` was accepted.
- Ngram speculative decoding did not help. It produced zero useful drafts and slowed the benchmark.
- Official DeepSeek inference could run on A100 only with BF16 fallback patches, but it was too slow for the deployment target.
- The EnsueAI INT4 Base path worked functionally, including a warm short request around `4.190s`, but it was not selected because the Base checkpoint produced extra decoded/template-like text for chat-shaped serving.
- DS4 CUDA, manual row split, tensor split, vLLM GGUF, and forced Flash Attention variants did not beat the llama.cpp GGUF baseline.

## Project Layout

- `llama_cpp_v4_q4_peer512_a100_modal.py`: current best benchmark/deployment script.
- `llama_cpp_v4_q4_*.py`: llama.cpp A100 flag and configuration experiments.
- `deepseek_v4_flash_official_modal.py` and `dsv4_official_server.py`: official DeepSeek runtime experiment with A100 compatibility patches.
- `deepseek_v4_flash_int4_modal.py` and `dsv4_int4_server.py`: INT4 Base checkpoint experiment.
- `ds4_modal.py`: DS4 runtime experiment.
- `deepseek_v4_flash_gguf_modal.py` and `patch_vllm_gguf_plugin.py`: exploratory vLLM GGUF path.
- `REPORT_*.md`, `CHANGELOG_*.md`, and deployment notes: preserved experiment records.

## Next Work

The most valuable next improvements are likely engine or code changes rather than another simple flag sweep. Priorities are:

- DeepSeek4 tensor parallelism in llama.cpp.
- DeepSeek4 MTP/NextN speculative decoding in llama.cpp.
- Real DeepSeek4 KV cache quantization support.
- A stable vLLM or SGLang A100 path for the native checkpoint or W4A16 without BF16 fallback overhead.
- A deterministic `llama-bench` wrapper so short-prompt chat timing noise does not dominate small benchmark deltas.

Modal runs can leave expensive GPU containers active. After every Modal run, deploy, or debugging session, check active containers and apps and stop anything no longer needed.
