# DeepSeek V4 Flash A100 Performance Roadmap

Date: 2026-05-22  
Target: Modal `A100-80GB:4` unless explicitly changed  
Baseline to beat: `~13 tok/s` single-stream decode from
`llama_cpp_v4_q4_peer512_a100_modal.py`

## Executive Summary

The current repo has already exhausted most low-risk llama.cpp flag tuning. The
best verified path is still:

- `cchuter/llama.cpp`, branch `feat/v4-port-cuda`
- `teamblobfish/DeepSeek-V4-Flash-GGUF`, `Q4_K_M-XL`
- 4x A100-80GB, SM80, `--split-mode layer`
- `GGML_CUDA_PEER_MAX_BATCH_SIZE=512`, P2P, launch queue scaling, tuned
  batch/ubatch/poll settings
- best decode: `12.9950 tok/s` thinking off, `12.9415 tok/s` thinking on

The next material gains are unlikely to come from another simple runtime flag.
The highest-value paths are:

1. Add a deterministic benchmark and concurrent-request harness so we can
   distinguish single-stream latency from aggregate throughput.
2. Enable DeepSeek4 MTP/NextN speculative decoding in the llama.cpp GGUF path.
3. Try vLLM/SGLang native DeepSeek V4 with MTP on A100, treating A100 support as
   the main risk.
4. Implement DeepSeek4 tensor split in llama.cpp if we want a per-stream latency
   breakthrough on the current GGUF path.
5. Implement real DeepSeek4 KV cache quantization or fp8/f8 cache support in
   llama.cpp, mainly to unlock longer-context batching and stack with MTP.
6. Only after those: investigate kernel fusion, graph split reduction, and
   DeepSeek4-specific Flash Attention-style kernels.

## Evidence From Existing Experiments

### Proven Best Baseline

Source files:

- `../../../README.md`
- `../../benchmarks/BENCHMARK_REPORT_20260521_234412_PDT.md`
- `../2026-05-21/REPORT_llama_cpp_v4_q4_additional_flags_a100_20260521_225300_PDT.md`
- `../2026-05-21/REPORT_llama_cpp_v4_q4_reddit_a100_20260521_232955_PDT.md`
- `llama_cpp_v4_q4_peer512_a100_modal.py`

Best command shape:

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

Important caveat: `--cache-type-k q4_0` is accepted but ignored in the tested
DeepSeek4 llama.cpp branch because the implementation forces fp16 KV cache.

### Already Tried And Not Worth Repeating Blindly

| Area | Result |
| --- | --- |
| Q4 32k baseline | Worked, about `12.25-12.40 tok/s` |
| Q4 fast context | Improved short-prompt wall time; decode only `12.45-12.54 tok/s` |
| Fastpipe P2P/queues/batch tuning | Improved to about `12.88 tok/s` |
| Peer max batch 512 | Current best, about `13 tok/s` |
| Forced Flash Attention | Slower, roughly `9.75-10.88 tok/s` |
| Forced MMQ | Slower, roughly `12.36-12.50 tok/s` |
| Forced cuBLAS | Slower than fastpipe, roughly `12.68-12.70 tok/s` |
| Ngram speculative decoding | Slower; zero useful drafts |
| Row/manual split | Failed startup with CUDA split-buffer `RESHAPE` error |
| Tensor split | Blocked by source support for `LLM_ARCH_DEEPSEEK4` |
| Q2_K-XL | Rejected: lower quality and worse throughput than Q4 |
| Official DeepSeek runtime | Functional with BF16 A100 fallback, but too slow |
| EnsueAI INT4 Base | Functional, but poor chat behavior and slow long outputs |
| DS4 | CUDA prefill failed; CPU path expected IQ2_XXS experts |
| vLLM GGUF plugin | Exploratory only; not currently competitive |

## External Research Cross-Check

The new research markdown is directionally useful, but several items need
qualification against local evidence:

- vLLM official recipes list DeepSeek-V4-Flash as 284B total / 13B active, 1M
  context, FP4+FP8 weights, and MTP-capable. The recommended recipes are framed
  around H200/B200/B300, not A100.
- vLLM's DeepSeek V4 blog says the Flash quick command is runnable on 4xB200 or
  4xB300 and describes bf16 KV during prefill plus token-wise fp8 during decode.
  That is encouraging architecturally but not proof of A100 speed.
- vLLM currently exposes a DeepSeek V4 NVIDIA MTP implementation in its docs.
- SGLang's launch post says DeepSeek-V4 has a single-layer MTP head and that
  SGLang fuses hybrid-attention metadata prep into CUDA graphs for speculative
  draft and verify passes.
- Hugging Face Transformers documents the V4 architecture as hybrid local +
  compressed sparse/heavily compressed attention with mHC.
- The teamblobfish GGUF card still ties the quants to the V4-aware cchuter fork,
  not stock upstream llama.cpp.

Sources:

- https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash
- https://vllm-project.github.io/2026/04/24/deepseek-v4.html
- https://docs.vllm.ai/en/latest/api/vllm/models/deepseek_v4/nvidia/mtp/
- https://www.lmsys.org/blog/2026-04-25-deepseek-v4/
- https://huggingface.co/docs/transformers/model_doc/deepseek_v4
- https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF
- https://github.com/cchuter/llama.cpp/tree/feat/v4-port-cuda

## Roadmap

### Phase 0: Measurement Before More Expensive Runs

Goal: make future deltas trustworthy and stop treating two chat responses as a
benchmark. Each candidate should report prefill, token generation, latency, and
aggregate throughput separately.

Experiments:

1. Add an offline `llama-bench`/`llama-cli` action for the current peer512
   script and build the relevant target in the Modal image.
   - Report prompt/prefill tokens per second.
   - Report decode/eval tokens per second.
   - Run at least 5 repeats and record mean, median, min, max, and stdev.
   - Use fixed token counts so quality/verbosity changes cannot masquerade as
     speed changes.
2. Add an online OpenAI-compatible streaming benchmark client.
   - Report time to first token (TTFT).
   - Report end-to-end latency.
   - Report `usage.prompt_tokens`, `usage.completion_tokens`, and total tokens.
   - Report approximate online prefill rate as `prompt_tokens / TTFT` when the
     endpoint provides usage data.
   - Report steady-state decode as `completion_tokens / (elapsed - TTFT)`.
   - Keep raw responses and server timing blocks when the engine exposes them.
   - Treat `tokens/sec` alone as insufficient; every online row must split
     prompt processing, first-token latency, decode generation, and aggregate
     throughput.
   - Keep per-request raw rows so p50/p90/p99 distributions can be recomputed.
3. Use a benchmark matrix instead of two prompts:
   - Prefill-only-ish: 512, 2k, 8k, 32k prompt tokens with `max_tokens=1`.
   - Decode-only-ish: short prompt with 128, 256, 512, 1024 output tokens.
   - Mixed interactive: 1k prompt with 256 output tokens.
   - Long-context continuation: 16k or 32k prompt with 128 output tokens.
   - Reasoning/chat controls: thinking off and thinking on for the same prompt.
4. Add concurrency and batching sweeps.
   - concurrency 1, 2, 4, and 8 when memory allows
   - `LLAMA_CPP_PARALLEL=1/2/4`
   - report per-request tok/s, aggregate completion tok/s, aggregate total
     tok/s, p50/p90/p99 latency, and p50/p90/p99 TTFT
   - include warmup rows, then exclude warmup from the reported steady-state
     summary
   - fail the run if a candidate improves aggregate throughput only by
     regressing single-stream decode or TTFT beyond the noise band
5. Record engine-level evidence in every report.
   - full server command
   - exact git commit/branch
   - GPU model/topology/P2P status
   - peak VRAM per GPU
   - context size, batch size, ubatch size, parallel slots
   - whether Flash Attention, KV quant, tensor split, and MTP are actually
     active or silently ignored
   - backend/kernel fallback decisions, such as fp8 weight-only fallback,
     unsupported architecture paths, or disabled CUDA graphs
6. Keep pass/fail quality gates separate from performance metrics.
   - arithmetic smoke prompt must answer correctly
   - sky/science prompt must stay coherent
   - no template spillover
   - thinking controls must be accepted where tested

Minimum acceptance bar for a benchmark report:

- `>=5` repeated offline samples for prefill and decode microbenchmarks when the
  engine exposes a deterministic runner.
- Online streaming rows for prefill-heavy, decode-heavy, mixed, and
  long-context cases.
- Separate single-stream and concurrent results; aggregate tok/s cannot replace
  per-request decode tok/s.
- Explicit TTFT, prefill tok/s, decode tok/s, latency percentiles, and aggregate
  completion tok/s.
- Raw command, engine commit/image, GPU topology, memory usage, and feature
  activation evidence.
- A clear decision: continue, retry with a targeted fix, or stop.

Recommended metric schema for each run:

```json
{
  "engine": "llama.cpp",
  "script": "llama_cpp_v4_q4_peer512_a100_modal.py",
  "git_commit": "unknown",
  "quant": "Q4_K_M-XL",
  "ctx": 4096,
  "parallel": 1,
  "concurrency": 1,
  "prompt_target_tokens": 2048,
  "max_output_tokens": 256,
  "prompt_tokens": 2048,
  "completion_tokens": 256,
  "ttft_seconds": 3.21,
  "elapsed_seconds": 22.91,
  "prefill_tok_s_online": 638.0,
  "decode_tok_s_online": 12.99,
  "aggregate_completion_tok_s": 12.99,
  "quality_pass": true,
  "notes": "server timing block attached in report"
}
```

Feasibility: very high.  
Expected single-stream gain: none directly.  
Expected insight: high. This identifies whether a candidate improves prefill,
decode, TTFT, or aggregate batching instead of blending all of them into one
wall-time number.

Sample online benchmark client:

Implemented as `benchmark_openai_concurrent.py`. It now runs a matrix of
prompt-size/output-size cases and can use streaming to measure TTFT:

```bash
rtk python3 benchmark_openai_concurrent.py \
  --base-url "$BASE_URL/v1" \
  --model deepseek-v4-flash \
  --stream \
  --concurrency 1 \
  --matrix 512x1,2048x1,8192x1,32768x1,128x256,128x512,1024x256,16384x128
```

Run the same matrix at concurrency 2 and 4 for aggregate throughput. The output
contains per-case `ttft`, latency distribution, online prefill estimate,
online decode estimate, aggregate completion tok/s, aggregate total tok/s, and
per-request raw metric rows.

### Phase 1: Squeeze Current llama.cpp Path Without New Kernels

Goal: verify whether the current server shape has any remaining easy wins.

Experiments:

1. `--parallel` and aggregate batching:
   - `LLAMA_CPP_PARALLEL=2`
   - `LLAMA_CPP_PARALLEL=4`
   - `--batch-size 2048/4096`
   - `--ubatch-size 256/512/1024`
2. Peer and scheduler constants:
   - `GGML_CUDA_PEER_MAX_BATCH_SIZE=256/512/1024`
   - `GGML_SCHED_MAX_SPLIT_INPUTS=128/256`
3. Context and cache shape:
   - `-c 2048/4096/8192`
   - keep `--cache-ram 0 --no-warmup`
4. New cchuter/upstream commit smoke:
   - build latest branch
   - run backend ops
   - source-inspect tensor split, KV cache override, MTP wiring
   - only benchmark if source inspection shows a meaningful change

Feasibility: high.  
Expected single-stream gain: 0-5%.  
Expected aggregate gain: possibly large if continuous batching is effective.  
Stackability: stacks with every later path.

Recommended new wrappers:

Implemented as `llama_cpp_v4_q4_peer512_parallel4_a100_modal.py`.

```python
# llama_cpp_v4_q4_peer512_parallel4_a100_modal.py
from __future__ import annotations

import os

os.environ.setdefault("LLAMA_CPP_APP_NAME", "deepseek-v4-flash-llama-cpp-q4-peer512-par4-a100")
os.environ.setdefault("LLAMA_CPP_MODEL_QUANT", "Q4_K_M-XL")
os.environ.setdefault("LLAMA_CPP_GPU", "A100-80GB:4")
os.environ.setdefault("LLAMA_CPP_CUDA_ARCH", "80")
os.environ.setdefault("LLAMA_CPP_CTX", "4096")
os.environ.setdefault("LLAMA_CPP_SPLIT_MODE", "layer")
os.environ.setdefault("LLAMA_CPP_PARALLEL", "4")
os.environ.setdefault("LLAMA_CPP_CMAKE_EXTRA_ARGS", "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=512")
os.environ.setdefault("CUDA_SCALE_LAUNCH_QUEUES", "4x")
os.environ.setdefault("GGML_CUDA_P2P", "1")
os.environ.setdefault(
    "LLAMA_CPP_EXTRA_SERVER_ARGS",
    "--cache-ram 0 --no-warmup --batch-size 4096 --ubatch-size 512 --poll 100 --poll-batch 1",
)

from llama_cpp_v4_q4_a100_modal import app, main  # noqa: E402,F401
```

### Phase 2: llama.cpp DeepSeek4 MTP/NextN

Goal: unlock the built-in MTP head while keeping the proven GGUF path.

Why this is the best next code investment:

- The GGUF metadata already indicates NextN/MTP-related data.
- The tested branch loads DeepSeek4 metadata but does not expose a working
  DeepSeek4 MTP speculative mode in `llama-server`.
- vLLM and SGLang both treat MTP as a first-class DeepSeek V4 feature, so the
  model architecture supports it.

Experiment ladder:

1. Source-inspect latest cchuter/upstream for MTP before writing code.
2. If available, create a wrapper with the new MTP flags and run:
   - `num_speculative_tokens=1`
   - `num_speculative_tokens=2`
   - temperature 0, 0.2, 0.7
   - thinking off/on
3. If unavailable, implement a llama.cpp speculative type for DeepSeek4 MTP.
4. Add logging:
   - proposed draft tokens
   - accepted draft tokens
   - acceptance rate
   - fallback count
   - generated tok/s

Feasibility: medium.  
Expected gain if accepted drafts are high: roughly 1.3-1.8x single-stream is a
reasonable target, but this must be measured on A100.  
Risk: MTP may regress if metadata prep or hidden-state plumbing becomes the new
bottleneck.  
Novel/community gap: wiring DeepSeek4 MTP into llama.cpp server support may be
novel if current cchuter/upstream still lacks it.

Implementation sketch:

```text
common/speculative:
  add spec type: deepseek4-mtp
  parse --spec-type deepseek4-mtp --spec-n N

model load:
  verify deepseek4.nextn_predict_layers > 0
  keep MTP block weights loaded and addressable

decode loop:
  after target token t is accepted:
    run MTP block using previous hidden state + embedding(t)
    sample draft token(s)
    verify draft token(s) through target model in one forward pass
    accept prefix while logits match speculative acceptance criterion
    fall back to normal target decode on rejection

server metrics:
  spec_draft_tokens_total
  spec_accepted_tokens_total
  spec_acceptance_rate
```

Stackability:

- Stack with peer512/P2P/launch queues.
- Stack with aggregate batching, but acceptance rates must be measured again
  under concurrency.
- Stack later with true tensor split and KV cache improvements.

### Phase 3: vLLM Native DeepSeek V4 On A100

Goal: determine whether vLLM can beat llama.cpp on this hardware by using native
V4 support, MTP, PagedAttention, TP/EP/DP, and continuous batching.

Why it is promising:

- vLLM has official DeepSeek V4 support and an NVIDIA MTP module.
- vLLM offers production server features that llama.cpp does not currently
  expose for this model path.

Why it is risky:

- Official recipes are for H200/B200/B300-class deployments, not A100.
- Local official DeepSeek runtime showed A100 FP8 paths need fallbacks and can be
  very slow.
- The current repo's vLLM GGUF plugin path is not the right baseline for this;
  test native official weights or W4A16/AutoRound instead.

Experiment ladder:

1. Native official weights, no MTP, conservative context:
   - `--tensor-parallel-size 4` or vLLM's recommended DeepSeek V4 DP/EP mode
   - `--kv-cache-dtype fp8`
   - `--tokenizer-mode deepseek_v4`
   - `--reasoning-parser deepseek_v4`
   - `--max-model-len 4096/8192`
2. Add MTP:
   - `--speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}'`
   - then `2`
3. Test concurrency:
   - 1, 2, 4, 8 requests if memory allows
4. Test W4A16/AutoRound only if the checkpoint keeps MTP weights and the engine
   supports the quantization on A100.

Feasibility: medium.  
Expected gain: high if it loads and MTP works; low if A100 fallback dominates.  
Stackability: native MTP + continuous batching + prefix caching can stack.

Minimal Modal command shape to adapt:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --disable-uvicorn-access-log
```

Then MTP:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --max-model-len 8192 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --disable-uvicorn-access-log \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":1}'
```

Abort criteria:

- native FP4/FP8 kernels fail on SM80 without a viable fallback
- load succeeds but warm short prompt is materially slower than llama.cpp
- MTP silently no-ops because the checkpoint/loader drops MTP weights

### Phase 4: SGLang Native DeepSeek V4

Goal: test whether SGLang's DeepSeek V4 MTP and hybrid-attention graph work can
beat llama.cpp for agent/RAG-style workloads, especially shared-prefix workloads.

Why it is promising:

- SGLang documents Day-0 DeepSeek V4 support.
- Its MTP design specifically addresses hybrid-attention metadata becoming a
  launch bottleneck.
- Prefix caching and shared-prefix handling could matter more than single
  prompt tok/s for real usage.

Why it is risky:

- New research and existing ecosystem notes suggest many fast kernels are
  Hopper/Blackwell-oriented.
- A100 fallback quality and performance are unproven in this repo.

Experiment ladder:

1. Load official weights at short context.
2. Run one request without MTP.
3. Enable MTP with 1 and 2 draft tokens.
4. Run shared-prefix benchmark:
   - same 2k-token prefix, varied short suffixes
   - concurrency 4
   - compare aggregate tok/s and latency against llama.cpp peer512

Feasibility: medium-low.  
Expected gain: workload-dependent; most promising for prefix-heavy traffic.  
Stackability: MTP + prefix caching can stack, but engine support decides.

### Phase 5: TensorRT-LLM + Triton

Goal: evaluate NVIDIA's compiled production stack if direct DeepSeek V4 Flash
support is confirmed for the target hardware.

Why it is promising:

- TensorRT-LLM is strong on NVIDIA multi-GPU serving, with TP/PP/EP,
  in-flight batching, paged KV, and Triton serving.
- A100 is generally a supported NVIDIA inference target.

Why it is not first:

- This repo has no successful TensorRT-LLM V4 Flash artifact yet.
- Current web checks did not give stronger local evidence than vLLM/SGLang.
- Build and engine-generation cost is high.

Experiment ladder:

1. Verify direct DeepSeek V4 Flash support in the exact TensorRT-LLM release.
2. Build a minimal short-context TP=4 engine.
3. Add EP if supported for this MoE layout.
4. Enable MTP only after non-MTP decode is correct.
5. Compare Triton aggregate throughput against vLLM and llama.cpp.

Feasibility: medium-low until direct support is verified.  
Expected gain: high if support is real and A100 kernels are optimized.  
Stackability: TP/EP + MTP + in-flight batching.

### Phase 6: llama.cpp Tensor Parallelism For DeepSeek4

Goal: reduce single-stream decode latency by replacing layer/pipeline split with
true tensor split.

Current blocker:

- The tested branch does not include `LLM_ARCH_DEEPSEEK4` in tensor-split
  support.
- Row/manual split failed on a CUDA split-buffer `RESHAPE` operation.

What implementation likely needs:

- Add architecture support only after auditing every DeepSeek4 tensor and custom
  op for split-buffer compatibility.
- Ensure custom ops can run with tensor-split buffers or explicitly gather where
  needed.
- Fix operations like `RESHAPE` that fail on split buffers.
- Validate all five DSV4 custom backend op groups under split mode.

Feasibility: low-medium.  
Expected gain: potentially high for per-stream latency if it works.  
Risk: high; incorrect splits can silently corrupt generation.  
Novel/community gap: likely novel unless upstream adds it first.

Stackability:

- Stack with MTP after both are independently correct.
- Stack with KV cache improvements.
- May change optimal batch/ubatch settings.

### Phase 7: Real DeepSeek4 KV Cache Quantization In llama.cpp

Goal: replace the current forced fp16 KV cache behavior with a correct lower
precision cache path.

Current blocker:

- The tested branch pins DeepSeek4 K/V cache to f16 because standard SWA,
  compressed-attention, and indexer K caches currently share dtype/view
  assumptions.

Implementation directions:

1. Split cache storage by cache kind:
   - SWA K/V
   - compressed CSA/HCA K/V
   - indexer cache
2. Avoid concatenated views that require one shared dtype.
3. Dequantize only at the kernel boundary where needed.
4. Add correctness tests:
   - short deterministic prompts
   - long context with repeated facts
   - compare f16 vs f8/q8/q4 outputs at temperature 0
5. Benchmark:
   - short decode
   - 32k/128k context
   - concurrency 2/4

Feasibility: low-medium.  
Expected single-stream short-context gain: probably modest.  
Expected long-context/concurrency gain: potentially important.  
Novel/community gap: real DeepSeek4 KV quant in llama.cpp appears unsupported in
the tested branch.

### Phase 8: Kernel Fusion, Graph Split Reduction, And DeepSeek4 Attention

Goal: attack the low-level A100 execution bottleneck after higher-level levers
are tested.

Candidate work:

- Reduce graph split count around V4 custom ops and layer boundaries.
- Fuse recurrent metadata/indexer preparation into CUDA graphs, similar in
  spirit to SGLang's MTP work.
- Fuse small DeepSeek4 custom ops where memory traffic dominates launch cost.
- Revisit Flash Attention only as a DeepSeek4-specific hybrid-attention kernel,
  not by forcing the generic llama.cpp flag that already regressed.
- Profile with Nsight Systems/Compute on one decode loop before changing code.

Feasibility: low.  
Expected gain: 10-30% is plausible if graph/launch overhead is dominant, but this
is profile-dependent.  
Stackability: stacks with MTP/tensor split only if each kernel path is profiled
again.

## Stackability Matrix

| Technique | Single-stream tok/s | Aggregate tok/s | Stacks with | Main risk |
| --- | ---: | ---: | --- | --- |
| Better benchmark harness | 0 | 0 | all | none |
| llama.cpp `--parallel`/concurrency | maybe down per stream | likely up | peer512, MTP | only improves aggregate |
| More peer/batch sweeps | 0-5% | 0-10% | all llama.cpp paths | noise, overfitting short prompt |
| llama.cpp MTP/NextN | high | high | peer512, concurrency, tensor split | not wired; acceptance may be low |
| vLLM native + MTP | high if it runs | high | batching, prefix cache | A100 FP4/FP8 fallbacks |
| SGLang + MTP/cache | medium | high for shared prefixes | prefix-heavy apps | A100 kernel maturity |
| TensorRT-LLM | high if supported | high | TP/EP/MTP/Triton | support/build cost |
| llama.cpp tensor split | high | medium | MTP, KV quant | substantial code work |
| DeepSeek4 KV quant | low short-context | medium/high long-context | MTP, batching | correctness complexity |
| Kernel fusion/graph reduction | medium | medium | most code paths | requires profiling and CUDA work |

## Recommended Experiment Order

1. `REPORT_llama_cpp_v4_q4_bench_harness_a100_<timestamp>.md`
   - Build/run deterministic `llama-bench` or equivalent fixed decode.
   - Report prefill tok/s and decode tok/s separately across the Phase 0 matrix.
   - Success: repeatable variance small enough to judge 3-5% changes.

2. `REPORT_llama_cpp_v4_q4_online_matrix_a100_<timestamp>.md`
   - Run `benchmark_openai_concurrent.py` with streaming enabled.
   - Measure TTFT, online prefill estimate, steady decode, latency distribution,
     aggregate completion tok/s, and aggregate total tok/s.
   - Test `LLAMA_CPP_PARALLEL=1/2/4` and concurrency 1/2/4.
   - Success: aggregate tok/s exceeds 13 without quality regression and without
     hiding prefill/decode regressions.

3. `REPORT_llama_cpp_v4_q4_latest_branch_probe_a100_<timestamp>.md`
   - Build latest cchuter/upstream candidate.
   - Inspect MTP, tensor split, KV cache override.
   - Run backend ops before benchmark.

4. `REPORT_llama_cpp_v4_q4_mtp_a100_<timestamp>.md`
   - Use existing MTP if available; otherwise start a code branch.
   - Test speculative tokens 1 and 2.
   - Success: accepted draft rate is high and decode beats 13 tok/s.

5. `REPORT_vllm_native_v4_flash_a100_<timestamp>.md`
   - Native official weights, short context, no MTP first.
   - Add MTP only after correctness.
   - Success: warm single-stream or aggregate throughput beats llama.cpp.

6. `REPORT_sglang_v4_flash_a100_<timestamp>.md`
   - Only after vLLM result or if shared-prefix workload becomes primary.

7. `REPORT_llama_cpp_v4_tensor_split_design_<timestamp>.md`
   - Design/prototype tensor split support.
   - Do not spend GPU time until backend op and source-level correctness risks
     are addressed.

8. `REPORT_llama_cpp_v4_kv_quant_design_<timestamp>.md`
   - Design/prototype separated cache dtypes.
   - Prioritize if long context or concurrency becomes the bottleneck.

9. `REPORT_trtllm_v4_flash_a100_<timestamp>.md`
   - Only after direct release support is confirmed.

## Stop/Continue Criteria

Continue an approach when:

- It beats `12.995 tok/s` single-stream decode, or
- It materially improves aggregate tok/s at acceptable per-request latency, or
- It unlocks a stackable feature such as MTP/tensor split/KV quant.

Stop an approach when:

- It only changes wall time by shortening outputs.
- It regresses decode in repeated deterministic tests.
- It requires Hopper-only kernels for the critical path.
- It produces chat-template spillover or loses thinking-mode controls.
- It holds Modal A100 containers after the run and cannot be cleaned reliably.

## Final Prioritization

The practical next run should be the concurrent/parallel benchmark on the current
peer512 baseline because it is cheap, safe, and may already answer whether the
deployment can exceed 13 aggregate tok/s. The practical next code investment is
DeepSeek4 MTP/NextN in llama.cpp, because it is the most likely way to improve
single-stream speed while preserving the only path that is already verified on
4x A100 in this repo.

The highest-upside but riskiest work is true DeepSeek4 tensor parallelism in
llama.cpp. It should not be attempted before MTP and vLLM native paths are
checked, because it is deeper engine work and the current branch explicitly
blocks the mode.
