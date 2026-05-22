# Temporary Research Digest - 2026-05-22

Purpose: working notes distilled from local benchmark artifacts, implementation
scripts, `NEW_RESEARCH_05222026.md`, and current primary-source checks. This is
not the final report.

## Local Baseline

- Current best: `llama_cpp_v4_q4_peer512_a100_modal.py`.
- Engine/model: `cchuter/llama.cpp` `feat/v4-port-cuda`,
  `teamblobfish/DeepSeek-V4-Flash-GGUF`, `Q4_K_M-XL`.
- Hardware: Modal `A100-80GB:4`, CUDA arch 80.
- Mode: `--split-mode layer`, not true tensor parallelism.
- Best preserved decode:
  - thinking off: `12.9950 tok/s`
  - thinking on: `12.9415 tok/s`
- Effective winning shape:
  - `-c 4096`
  - `-ngl 999`
  - `--split-mode layer`
  - `--parallel 1`
  - `--jinja`
  - `--cache-type-k q4_0` accepted but ignored for DeepSeek4
  - `--cache-ram 0 --no-warmup`
  - `--batch-size 2048 --ubatch-size 512`
  - `--poll 100 --poll-batch 1`
  - `CUDA_SCALE_LAUNCH_QUEUES=4x`
  - `GGML_CUDA_P2P=1`
  - build extra `-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512`

## Things Already Tried

- Q4 baseline at 32k context: about `12.25-12.40 tok/s`.
- Fast context (`-c 4096`, no warmup/cache RAM): about `12.45-12.54 tok/s`.
- Fastpipe queue/P2P/batch tuning: about `12.88 tok/s`.
- Peer max batch 512: about `12.94-13.00 tok/s`, current best.
- Forced Flash Attention: slower, about `9.75-10.88 tok/s`.
- Forced MMQ: slower, about `12.36-12.50 tok/s`.
- Forced cuBLAS: slower than fastpipe, about `12.68-12.70 tok/s`.
- Ngram speculative decoding: slower, zero useful drafts.
- Row/manual split: startup failure on `RESHAPE` in CUDA split buffer.
- Tensor split: not run because inspected branch rejects
  `LLM_ARCH_DEEPSEEK4` tensor split.
- Q2_K-XL: rejected because quality and throughput were worse than Q4.
- Official DeepSeek runtime: functional only with BF16 A100 fallback, slow
  short prompt around 65s.
- EnsueAI INT4 Base: functional, warm short request 4.190s, but Base checkpoint
  emits extra template-like text and long sky prompts were very slow.
- DS4: CUDA prefill failed; CPU path expected IQ2_XXS expert tensors.
- vLLM GGUF plugin path: exploratory, superseded by cchuter/teamblobfish path.

## Local Constraints/Blockers

- A100 is SM80 and does not have native FP8 tensor cores.
- cchuter branch passes custom DSV4 op tests on A100 but uses software/emulated
  or fallback behavior for FP8-related paths.
- DeepSeek4 KV cache flags are not active in tested llama.cpp branch. The branch
  forces K/V cache dtype to f16 because V4 standard SWA, compressed, and indexer
  K caches share dtype/view behavior.
- GGUF has `deepseek4.nextn_predict_layers = 1`, but tested cchuter
  `llama-server` did not expose wired DeepSeek4 MTP/NextN speculative decoding.
- All successful local llama.cpp tests use pipeline/layer split; true tensor
  split is not available for DeepSeek4 in the tested branch.

## Current External Checks

- vLLM recipe page identifies DeepSeek-V4-Flash as 284B total / 13B active, 1M
  context, FP4+FP8 weights, and MTP. Recipe is framed around H200/B200/B300
  deployments, not A100.
- vLLM blog says DeepSeek-V4-Flash command is runnable on 4xB200/B300 and
  describes V4 KV handling: bf16 KV for prefill and token-wise fp8 for decode,
  with fp4 indexer/fp8 attention cache options.
- vLLM API docs expose a DeepSeek V4 NVIDIA MTP module, so the model-side MTP
  implementation exists in vLLM.
- LMSYS/SGLang blog says DeepSeek-V4 ships a single-layer MTP head and SGLang
  supports it by fusing hybrid-attention metadata preparation into CUDA graphs.
- Hugging Face Transformers docs confirm the architecture: hybrid local +
  compressed sparse/heavily compressed attention and mHC.
- teamblobfish HF model card still says the GGUF quants require the V4-aware
  cchuter fork, not stock upstream llama.cpp.

## Synthesis

Highest-confidence paths beyond 13 tok/s:

1. Better measurement and aggregate throughput benchmark on current peer512.
   Existing tests are single serial chat requests. `--parallel`/continuous
   batching may exceed 13 aggregate tok/s without changing single-stream decode.
2. DeepSeek4 MTP/NextN in llama.cpp. This is the best material improvement that
   preserves the current proven GGUF path. It likely requires code support if no
   newer cchuter/upstream branch wires it.
3. vLLM/SGLang path with native MTP. Feature support exists upstream, but A100 is
   the risk. Test load first, then MTP, then batching.
4. True tensor parallel support in llama.cpp. High potential for per-stream
   latency but large implementation risk because DeepSeek4 tensor split is
   currently blocked.
5. Real DeepSeek4 KV cache quant/fp8/f8 support in llama.cpp. More useful for
   long context and batching than short decode, but stackable.
6. Kernel fusion / graph split reduction for the custom V4 path. Novel and hard;
   probably only worth after MTP/tensor split feasibility is clearer.

