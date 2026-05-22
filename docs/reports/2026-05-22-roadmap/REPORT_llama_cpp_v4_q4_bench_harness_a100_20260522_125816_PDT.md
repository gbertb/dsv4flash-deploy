# DeepSeek V4 Flash Q4 llama.cpp Bench Harness Report

Date: 2026-05-22 12:58:16 PDT  
Roadmap item: `REPORT_llama_cpp_v4_q4_bench_harness_a100_<timestamp>.md`  
Hardware: Modal `A100-80GB:4`  
Engine: cchuter `llama.cpp`, branch `feat/v4-port-cuda`  
Model: `teamblobfish/DeepSeek-V4-Flash-GGUF`, `Q4_K_M-XL`

## Scope

Record the deterministic benchmark harness work required by Phase 0. The harness
was implemented in `llama_cpp_v4_q4_a100_modal.py` by building `llama-bench` and
adding the `offline-bench` action.

Related report with the online matrix and raw serving context:

- `REPORT_llama_cpp_v4_q4_phase0_online_matrix_a100_20260522_114343_PDT.md`

## Implemented Harness

`llama_cpp_v4_q4_a100_modal.py` now builds:

```text
llama-server
llama-cli
llama-bench
test-backend-ops
```

The offline action runs fixed prompt/decode shapes through `llama-bench` so
future changes cannot look faster merely by producing shorter chat responses.

## Result

Successful deterministic row:

```text
2048 prompt tokens, 1 generated token
prompt/prefill mean: 165.816 tok/s
stdev: 2.1496 tok/s
```

Observed samples were in the roughly `163-168 tok/s` range.

Failed deterministic row:

```text
512 prompt tokens, 1 generated token
```

Failure:

```text
GGML_ASSERT(n_comp_visible <= n_comp_cache)
```

The same DeepSeek4 graph/compressed-cache assertion also appeared in later
`--parallel > 1` serving probes. The failure is therefore useful evidence: the
current branch has DeepSeek4 graph/cache shape assumptions that limit which
microbench rows are valid.

## Decision

The deterministic harness exists and produced useful prefill evidence, but it is
not yet a complete stable microbenchmark suite for all requested matrix rows.
Future code branches should keep using `offline-bench`, but they must treat
DeepSeek4 graph asserts as correctness blockers rather than missing data.

Pass/fail interpretation:

- `2048x1` prefill evidence is valid.
- `512x1` is a branch bug/limitation, not a performance result.
- Online streaming rows remain necessary for TTFT and decode-rate evidence.

