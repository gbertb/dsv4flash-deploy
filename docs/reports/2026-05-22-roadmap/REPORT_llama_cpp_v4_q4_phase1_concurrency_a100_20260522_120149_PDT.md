# DeepSeek V4 Flash Q4 A100 Phase 1 Concurrency Report

Date: 2026-05-22 12:01:49 PDT  
Hardware: Modal `A100-80GB:4`  
Model: `teamblobfish/DeepSeek-V4-Flash-GGUF`, `Q4_K_M-XL`  
Engine: `cchuter/llama.cpp`, `feat/v4-port-cuda`, build `b1-781e978`  
Baseline reference: `REPORT_llama_cpp_v4_q4_phase0_online_matrix_a100_20260522_114343_PDT.md`

## Scope

This run continued Phase 1 of `FINAL_ROADMAP_DEEPSEEK_V4_FLASH_A100_20260522.md`
after the initial `parallel=4/concurrency=4` attempt crashed. The purpose was to
isolate whether the crash and poor aggregate behavior came from `--parallel`,
client concurrency, or the larger `--batch-size`.

## Harness Fix

The online matrix harness was updated after the `parallel=2` run exposed an
edge case: if the server process dies while streaming, the HTTP stream can close
without chunks or usage data. That is now treated as a failed request:

```text
RuntimeError: stream ended without tokens or usage
```

The standalone `benchmark_openai_concurrent.py` client received the same fix.

## Configuration A: `parallel=2`, `concurrency=2`

Script: `llama_cpp_v4_q4_peer512_parallel2_a100_modal.py`

Effective server shape:

```text
--parallel 2
--batch-size 2048
--ubatch-size 512
--split-mode layer
--cache-type-k q4_0
```

Result: failed during DeepSeek4 graph initialization before valid tokens were
generated.

Failure:

```text
/opt/llama.cpp/ggml/src/ggml.c:3660:
GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2) failed
```

The matrix rows closed with zero chunks and no usage data:

| Case | Wall | Tokens | Interpretation |
| --- | ---: | ---: | --- |
| `128x256`, concurrency 2 | `20.591s` | 0 | invalid; server crashed |
| `1024x256`, concurrency 2 | `21.693s` | 0 | invalid; server crashed |

Conclusion: the DeepSeek4 graph assert is not only caused by `parallel=4` or
`batch-size=4096`. `--parallel 2` with the known-good `batch-size=2048` is also
unsafe in this llama.cpp branch.

## Configuration B: `parallel=1`, `concurrency=2`

Script: `llama_cpp_v4_q4_peer512_concurrency2_a100_modal.py`

Effective server shape:

```text
--parallel 1
--batch-size 2048
--ubatch-size 512
--split-mode layer
--cache-type-k q4_0
```

Engine evidence:

- `n_seq_max = 1`
- `n_slots = 1`
- Flash Attention auto-disabled
- DeepSeek4 forced fp16 KV; requested `q4_0` KV ignored
- speculative decoding checkpoint support initialized, but no speculative
  implementation was active

### Results

Matrix: `128x256,1024x256`

| Case | Successful | Failed | Wall | Aggregate Completion | Aggregate Total | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `128x256` | 2 | 0 | `259.625s` | `0.878 tok/s` | `1.687 tok/s` | cold-load polluted; serialized through one slot |
| `1024x256` | 2 | 0 | `59.725s` | `4.588 tok/s` | `36.233 tok/s` | warm row; serialized through one slot |

Warm `1024x256` per-request rows:

| Request | TTFT | Latency | Prompt Tokens | Completion Tokens | Online Prefill | Online Decode | E2E Decode | Cached Prompt Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `9.135s` | `37.572s` | 945 | 137 | `103.454 tok/s` | `4.818 tok/s` | `3.646 tok/s` | 0 |
| 2 | `39.274s` | `59.725s` | 945 | 137 | `24.062 tok/s` | `6.699 tok/s` | `2.294 tok/s` | 813 |

Server timing for warm `1024x256`:

| Request | Prompt Eval | Decode Eval |
| --- | ---: | ---: |
| 1 | 945 tokens in `8513.28 ms` = `111.00 tok/s` | 137 tokens in `28442.32 ms` = `4.82 tok/s` |
| 2 | 132 tokens in `1709.46 ms` = `77.22 tok/s` | 137 tokens in `20354.54 ms` = `6.73 tok/s` |

## Interpretation

- `--parallel 2` and `--parallel 4` both crash with the same DeepSeek4 graph
  assert, so the current branch cannot use llama.cpp multi-slot parallelism for
  this model/configuration.
- Plain client concurrency with `--parallel 1` is stable, but it does not
  improve aggregate decode throughput because the server exposes only one slot.
- The warm `1024x256` aggregate completion rate at concurrency 2 was
  `4.588 tok/s`, worse than the Phase 0 single-request warm row
  (`1024x256`, `150` completion tokens, `6.492` end-to-end completion tok/s and
  `9.44` online decode tok/s).
- The cold `128x256` row should not be used for throughput comparison because
  both requests spent most of their time waiting for model load.

## Decision

Do not continue blind `LLAMA_CPP_PARALLEL` sweeps on this branch. The next useful
work is source inspection of DeepSeek4 graph construction for `n_seq_max > 1`
and MTP/tensor-split availability, then a targeted code change or a native vLLM
/ SGLang probe.

## Cleanup

After the Modal runs:

- `rtk modal container list --json` returned `[]`
- `deepseek-v4-flash-llama-cpp-q4-peer512-par2-a100` was stopped with zero tasks
- `deepseek-v4-flash-llama-cpp-q4-peer512-conc2-a100` was stopped; Modal still
  displayed one completed task, but there were no active containers

