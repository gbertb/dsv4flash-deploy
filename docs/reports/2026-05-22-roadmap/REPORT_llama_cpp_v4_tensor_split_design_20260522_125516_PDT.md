# DeepSeek V4 Flash llama.cpp Tensor Split Design

Date: 2026-05-22 12:55:16 PDT  
Roadmap item: `REPORT_llama_cpp_v4_tensor_split_design_<timestamp>.md`  
Target path: cchuter `llama.cpp`, branch `feat/v4-port-cuda`  
Target hardware: Modal `A100-80GB:4`

## Scope

Define the source-level work required before spending more A100 time on
DeepSeek4 tensor split in llama.cpp. This is a design/prototype gate, not a
claim that tensor split is implemented.

Inputs:

- `../../benchmarks/BENCHMARK_REPORT_20260521_234412_PDT.md`
- `REPORT_llama_cpp_v4_q4_source_inspect_a100_20260522_120149_PDT.md`
- `REPORT_llama_cpp_v4_q4_latest_branch_probe_a100_20260522_120802_PDT.md`
- prior row/manual split failure in `llama_cpp_v4_q4_row_manual_a100_modal.py`

## Current Evidence

The current winning path is layer split:

```bash
--split-mode layer
```

It works because each GPU owns whole layer ranges. It does not reduce
single-token latency like true tensor parallelism could.

Tensor split is not a runtime flag away:

- Source inspection found a generic unsupported path for
  `LLAMA_SPLIT_MODE_TENSOR` on architectures without explicit support.
- Runtime history already showed row/manual split failing with a CUDA
  split-buffer `RESHAPE` error.
- DeepSeek4 custom compressed/indexer cache logic already asserts under
  `--parallel > 1`, so split-buffer safety cannot be assumed.
- Upstream `ggml-org/llama.cpp` master has newer generic infrastructure but no
  `LLM_ARCH_DEEPSEEK4` implementation, so upstream cannot directly replace the
  cchuter branch.

## Required Implementation Work

### 1. Audit Every DeepSeek4 Tensor Role

Inventory tensors in `src/models/deepseek4.cpp` by role:

- token embedding and output projection
- attention projections
- compressed attention/indexer tensors
- SWA/local attention tensors
- MoE router tensors
- expert gate/up/down tensors
- shared expert tensors
- normalization tensors
- MTP/NextN tensors

For each tensor, decide whether it is:

- row-sharded
- column-sharded
- replicated
- gathered before a custom op
- illegal to split without a new kernel

Do not start by adding `LLM_ARCH_DEEPSEEK4` to the generic tensor-split allow
list. That would make unsupported split buffers flow into code that has not been
audited.

### 2. Isolate Split-Buffer Unsafe Ops

Known suspicious areas:

- `RESHAPE` on CUDA split buffers from the prior row/manual split failure
- compressed/indexer cache layout math
- DeepSeek4 graph construction around `n_comp_visible <= n_comp_cache`
- any custom CUDA op that assumes contiguous single-device tensors
- views shared across SWA, compressed attention, and indexer caches

Add explicit local tests around these operations before serving:

```bash
rtk modal run <tensor-split-prototype>.py --action backend-ops
```

The backend-op pass condition must be stronger than the current layer-split
`19/19` result: it must run with tensor-split buffers active, not just with the
default layer-split allocation.

### 3. Add Architecture Support Behind A Feature Guard

Add a separate experimental flag or build-time guard, for example:

```text
LLAMA_CPP_DEEPSEEK4_TENSOR_SPLIT_EXPERIMENTAL=1
```

The initial patch should fail closed:

- reject tensor split unless the guard is enabled
- log every tensor-split decision at model load
- refuse unsupported tensor roles explicitly

### 4. Validate Correctness Before Performance

Correctness gates:

- backend ops under tensor split
- arithmetic smoke prompt returns `323`
- sky/science prompt is coherent
- repeated-fact prompt returns the right fact at temperature `0`
- target layer-split and tensor-split outputs match at temperature `0` for
  short prompts, or any divergence is explained by a known sampler difference

Only then run the Phase 0 online benchmark matrix.

## Benchmark Gate

Use the roadmap's benchmark schema. A valid tensor-split report must include:

- exact source commit/diff summary
- exact split mode and tensor split
- GPU topology/P2P status
- peak VRAM per GPU
- offline prefill/decode samples with repeats when `llama-bench` works
- online TTFT, prefill estimate, steady decode, aggregate throughput
- quality pass/fail rows

Pass condition:

- single-stream decode beats the `12.995 tok/s` prior best, or
- tensor split unlocks a stackable feature that passes correctness and does not
  regress decode beyond measurement noise.

## Decision

Do not spend more A100 time on `--split-mode tensor` or row/manual split as a
flag sweep. The next tensor-split artifact should be a prototype branch with
split-buffer backend-op tests and a focused diff summary.
