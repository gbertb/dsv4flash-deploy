# DeepSeek V4 Flash A100 Deployment

This repository tracks experiments for deploying DeepSeek V4 Flash on Modal with 4x NVIDIA A100-80GB GPUs. The goal is to identify a practical, reproducible serving path for Ampere hardware, document failed approaches, and preserve benchmark evidence for future optimization work.

For the original benchmark details, methodology, command history, caveats, and
future-agent handoff notes, read
[BENCHMARK_REPORT_20260521_234412_PDT.md](docs/benchmarks/BENCHMARK_REPORT_20260521_234412_PDT.md).
For the latest roadmap execution summary, read
[FINAL_EXECUTION_SUMMARY_DEEPSEEK_V4_FLASH_A100_20260522_114343_PDT.md](docs/reports/2026-05-22-roadmap/FINAL_EXECUTION_SUMMARY_DEEPSEEK_V4_FLASH_A100_20260522_114343_PDT.md).

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
- `llama_cpp_v4_q4_a100_modal.py`: shared llama.cpp Modal implementation and
  benchmark harness.
- `benchmark_openai_concurrent.py`: reusable streaming `/v1` benchmark client
  with TTFT, prefill, decode, latency, and aggregate throughput metrics.
- `deepseek_v4_flash_official_modal.py` and `dsv4_official_server.py`: official DeepSeek runtime experiment with A100 compatibility patches.
- `deepseek_v4_flash_int4_modal.py` and `dsv4_int4_server.py`: INT4 Base checkpoint experiment.
- `ds4_modal.py`: DS4 runtime experiment.
- `deepseek_v4_flash_gguf_modal.py` and `patch_vllm_gguf_plugin.py`: exploratory vLLM GGUF path.
- `deepseek_v4_flash_vllm_native_modal.py`: native vLLM DeepSeek V4 probe.
- `deepseek_v4_flash_sglang_native_modal.py`: native SGLang DeepSeek V4 probe.
- `docs/benchmarks/`: baseline benchmark reports.
- `docs/reports/2026-05-21/`: earlier llama.cpp experiment reports.
- `docs/reports/2026-05-22-roadmap/`: roadmap execution reports from the latest
  research session.
- `docs/research/`: research notes and digests.
- `docs/changelog/`: historical changelogs.
- `docs/notes/`: deployment and implementation notes.

See [docs/README.md](docs/README.md) for the documentation index.

## Changelog: 2026-05-22 Roadmap Research

- Added a real benchmark protocol to the roadmap. Future results must separate
  TTFT, prompt/prefill speed, token generation speed, latency percentiles,
  aggregate throughput, raw rows, and engine feature activation evidence.
- Added `offline-bench` and `online-matrix` actions to the llama.cpp wrapper and
  added `benchmark_openai_concurrent.py` for streaming OpenAI-compatible
  endpoint tests.
- Baseline llama.cpp online matrix showed warm streaming decode around
  `9.44-9.87 tok/s`; the older `12.995 tok/s` sky-prompt decode remains the best
  prior single-stream number but is no longer sufficient by itself.
- `llama-bench` produced useful `2048x1` prefill evidence
  (`165.816 tok/s` mean) but hit a DeepSeek4 graph assertion on `512x1`.
- `LLAMA_CPP_PARALLEL=2/4` is not safe on the current cchuter branch. Multi-slot
  runs hit `GGML_ASSERT(n_comp_visible <= n_comp_cache)`, and
  `parallel=1/concurrency=2` serialized through one slot.
- Source inspection found DeepSeek4 is hard-capped to one sequence, KV cache is
  forced to fp16, tensor split is not implemented for `LLM_ARCH_DEEPSEEK4`, and
  MTP/NextN metadata is loaded but not wired to a speculative server path.
- Upstream `ggml-org/llama.cpp` master has generic `draft-mtp` plumbing, but no
  DeepSeek4 model implementation. The practical llama.cpp path is a guarded
  cchuter-based MTP prototype branch.
- Native vLLM DeepSeek V4 support exists, but the A100 dummy path failed in
  DeepGEMM hyperconnection/MHC with an unsupported-architecture path.
- Native SGLang DeepSeek V4 support exists and reached server readiness on A100,
  but its DSV4 top-k JIT kernel requires cluster launch features unavailable on
  `sm_80`.
- TensorRT-LLM is not currently a good A100 target for DeepSeek V4 Flash; current
  public support evidence is DeepSeek R1/V3/V3.2 and Blackwell-oriented V4
  vLLM/SGLang releases, not a direct TensorRT-LLM A100 path.
- New design/prototype docs define the required work for llama.cpp MTP, tensor
  split, and real DeepSeek4 KV quantization before more GPU time is spent.

## Next Work

The most valuable next improvements are code changes rather than another simple
flag sweep. Priorities are:

- DeepSeek4 MTP/NextN speculative decoding in llama.cpp.
- DeepSeek4 tensor parallelism in llama.cpp after MTP/source gates.
- Real DeepSeek4 KV cache quantization support.
- An Ampere-safe vLLM/SGLang native path only after the current unsupported
  architecture blockers are fixed.
- TensorRT-LLM only after direct DeepSeek V4 Flash support is published or found
  in a source/container probe.

Modal runs can leave expensive GPU containers active. After every Modal run, deploy, or debugging session, check active containers and apps and stop anything no longer needed.
