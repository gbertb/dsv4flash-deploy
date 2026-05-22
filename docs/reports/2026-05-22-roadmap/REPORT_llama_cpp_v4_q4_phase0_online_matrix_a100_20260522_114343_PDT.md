# DeepSeek V4 Flash Q4 A100 Phase 0/1 Benchmark Report

Date: 2026-05-22 11:43:43 PDT  
Hardware: Modal `A100-80GB:4`  
Model: `teamblobfish/DeepSeek-V4-Flash-GGUF`, `Q4_K_M-XL`  
Engine: `cchuter/llama.cpp`, `feat/v4-port-cuda`, build `b1-781e978`  
Baseline to beat: prior single-stream decode `~12.995 tok/s`

## Scope

This run executed the Phase 0 measurement work from
`FINAL_ROADMAP_DEEPSEEK_V4_FLASH_A100_20260522.md` and started the Phase 1
parallel/concurrency sweep. The goal was to replace the old "two chat prompts
and wall time" benchmark with measurements that separate:

- prompt/prefill speed
- TTFT
- token generation speed
- end-to-end latency
- aggregate throughput under concurrency
- engine-level evidence such as Flash Attention/KV/MTP state

## Code Changes

- `llama_cpp_v4_q4_a100_modal.py`
  - Builds the `llama-bench` target in the Modal image.
  - Adds `--action offline-bench` for fixed-token `llama-bench` cases.
  - Adds `--action online-matrix` for streaming OpenAI-compatible benchmarks.
  - Records TTFT, prompt tokens, completion tokens, total tokens, online prefill
    estimate, online decode estimate, end-to-end decode, per-request latency,
    p50/p90/p99 summaries, aggregate completion tok/s, and aggregate total
    tok/s.
  - Records structured per-request failures so crashed server configurations
    produce usable benchmark rows.
- `benchmark_openai_concurrent.py`
  - Adds a reusable endpoint benchmark client for external `/v1` endpoints.
  - Supports streaming TTFT, matrix cases, concurrency, p50/p90/p99 summaries,
    prefill/decode summaries, and structured request failures.
- `llama_cpp_v4_q4_peer512_parallel4_a100_modal.py`
  - Adds a Phase 1 wrapper for `LLAMA_CPP_PARALLEL=4`,
    `LLAMA_CPP_ONLINE_CONCURRENCY=4`, and larger server batch size.

## Engine Evidence

Baseline online matrix command:

```text
llama-server ... -c 4096 -ngl 999 --split-mode layer --parallel 1 --jinja
  --cache-type-k q4_0 --cache-ram 0 --no-warmup
  --batch-size 2048 --ubatch-size 512 --poll 100 --poll-batch 1
```

Observed server state:

- GPUs: 4x `NVIDIA A100-SXM4-80GB`
- CUDA arch: `800`
- `USE_GRAPHS=1`
- `PEER_MAX_BATCH_SIZE=512`
- split mode: `layer`
- Flash Attention: auto-disabled
- KV cache: DeepSeek4 forced fp16 KV; requested `--cache-type-k q4_0` was ignored
- pipeline parallelism: enabled
- NextN/MTP metadata: `nextn_predict_layers = 1`, but no speculative decode
  implementation was active

## Offline `llama-bench`

The first `llama-bench` attempt exposed an argument mismatch: `llama-bench`
does not accept `-c`. The harness was fixed to use `-p`, `-n`, `-b`, `-ub`,
`-ctk`, and `-sm`.

The corrected run showed partial DeepSeek4 compatibility:

| Case | Result |
| --- | --- |
| `512x1` | Failed in graph construction: `GGML_ASSERT(n_comp_visible <= n_comp_cache)` |
| `2048x1`, prompt pass | `165.816 tok/s` mean, `2.150` stdev, samples `163.377, 166.780, 166.890, 168.294, 163.739` |
| `2048x1`, isolated one-token generation | `1.543 tok/s` mean, not representative of steady decode |

Conclusion: `llama-bench` is useful for controlled prefill evidence, but the
current DeepSeek4 branch is not robust across the full matrix. The online
streaming benchmark is the more reliable path for decode and end-to-end server
behavior.

## Online Matrix: Baseline `parallel=1`, `concurrency=1`

Matrix: `512x1,2048x1,128x256,1024x256`.

| Case | TTFT | Latency | Prompt Tokens | Completion Tokens | Online Prefill | Online Decode | E2E Decode | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `512x1` | `338.936s` | `339.036s` | 465 | 1 | `1.372 tok/s` | `10.009 tok/s` | `0.003 tok/s` | Cold model load; not a valid warm prefill row |
| `2048x1` | `14.429s` | `14.431s` | 1875 | 1 | `129.946 tok/s` | `557.997 tok/s` | `0.069 tok/s` | One-token decode rate is not meaningful |
| `128x256` | `1.656s` | `12.359s` | 105 | 105 | `63.414 tok/s` | `9.810 tok/s` | `8.496 tok/s` | Warm decode row |
| `1024x256` | `7.217s` | `23.107s` | 945 | 150 | `130.946 tok/s` | `9.440 tok/s` | `6.492 tok/s` | Warm mixed row |

Server timing blocks for the warm decode rows:

| Case | Prompt Eval | Decode Eval |
| --- | ---: | ---: |
| `128x256` | 105 tokens in `1016.60 ms` = `103.29 tok/s` | 105 tokens in `10634.54 ms` = `9.87 tok/s` |
| `1024x256` | 945 tokens in `6539.32 ms` = `144.51 tok/s` | 150 tokens in `15890.54 ms` = `9.44 tok/s` |

Interpretation:

- Warm prefill in these server rows is about `103-145 tok/s`.
- Warm streaming decode in these rows is about `9.44-9.87 tok/s`.
- The prior `~12.995 tok/s` result remains the best single-stream decode
  baseline for the older sky-prompt benchmark, but this matrix shows why future
  comparisons must separate prompt shape, output length, cold load, and server
  timing.

## Phase 1 Parallel/Concurrency Attempt

Configuration:

```text
LLAMA_CPP_PARALLEL=4
LLAMA_CPP_ONLINE_CONCURRENCY=4
LLAMA_CPP_ONLINE_MATRIX=128x256,1024x256
--batch-size 4096 --ubatch-size 512
```

Result: failed during DeepSeek4 graph initialization before a valid benchmark
row completed.

Failure:

```text
/opt/llama.cpp/ggml/src/ggml.c:3660:
GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2) failed
```

Modal then returned HTTP 500 errors to the concurrent benchmark requests. The
app stopped cleanly after the exception. No benchmark throughput should be
attributed to this configuration.

Interpretation: current `parallel=4` with `batch-size=4096` is not a safe Phase
1 path on this DeepSeek4 llama.cpp branch. The next sweep should test smaller
steps, starting with `parallel=2`, `concurrency=2`, and the known-good
`batch-size=2048`, then isolate whether the crash is caused by `--parallel`,
batch size, or concurrent prefill.

## Cleanup

After the Modal runs:

- `rtk modal container list --json` returned `[]`
- benchmark apps were stopped with zero tasks:
  - `deepseek-v4-flash-llama-cpp-q4-peer512-a100`
  - `deepseek-v4-flash-llama-cpp-q4-peer512-par4-a100`

## Next Recommended Run

Run a narrower Phase 1 sweep:

```bash
LLAMA_CPP_PARALLEL=2 \
LLAMA_CPP_ONLINE_CONCURRENCY=2 \
LLAMA_CPP_ONLINE_MATRIX=128x256,1024x256 \
rtk modal run llama_cpp_v4_q4_peer512_a100_modal.py --action online-matrix
```

If that succeeds, increase only one variable at a time:

1. `concurrency=4`, keep `parallel=2`, `batch-size=2048`
2. `parallel=4`, keep `batch-size=2048`
3. `batch-size=4096` only after `parallel=4` is stable

