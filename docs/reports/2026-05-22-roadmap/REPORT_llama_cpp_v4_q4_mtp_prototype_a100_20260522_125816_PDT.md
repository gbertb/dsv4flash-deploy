# DeepSeek V4 Flash Q4 llama.cpp MTP Prototype Start

Date: 2026-05-22 12:58:16 PDT  
Roadmap item: `REPORT_llama_cpp_v4_q4_mtp_a100_<timestamp>.md`  
Status: source-prototype started, not benchmarkable

## Scope

Close the roadmap gap after source inspection showed that DeepSeek4 MTP is not
available as a server flag in the tested cchuter branch.

The roadmap says:

```text
Use existing MTP if available; otherwise start a code branch.
```

Existing MTP was not available. This report records the branch-start artifact
and the remaining gates before any A100 benchmark would be valid.

## Artifact Created

- `PROTOTYPE_llama_cpp_deepseek4_mtp_source_patch_20260522_125816_PDT.md`

That artifact defines:

- the intended branch shape
- the `draft-mtp` speculative type scaffold
- an explicit `LLAMA_CPP_DEEPSEEK4_MTP_EXPERIMENTAL` guard
- required draft/accept/reject metrics
- validation commands and gate order
- non-goals to keep the prototype isolated from tensor split, KV quant, and
  `--parallel > 1`

## Why This Is Not Yet A Benchmark

The current source evidence is strong enough to start the branch but not strong
enough to run MTP:

- cchuter `feat/v4-port-cuda` loads `nextn_predict_layers = 1`, but no active
  DeepSeek4 speculative serving path exists.
- upstream `ggml-org/llama.cpp` master has generic `draft-mtp`, but no
  `LLM_ARCH_DEEPSEEK4` or `src/models/deepseek4.cpp`.
- DeepSeek4 serving is hard-capped to one sequence in the cchuter branch.
- `--parallel > 1` currently crashes in compressed-cache graph assumptions.
- DeepSeek4 KV cache dtype is forced to fp16 and should not be changed in the
  MTP branch.

## Required Next Commit

A real llama.cpp branch should be created outside this deployment repo with:

```text
base: cchuter/llama.cpp feat/v4-port-cuda
reference: ggml-org/llama.cpp master draft-mtp plumbing
```

Minimum commit contents:

1. port generic `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` argument/settings plumbing
2. add `--spec-type draft-mtp --draft-max 1` parsing if missing
3. add DeepSeek4 guard requiring `LLAMA_CPP_DEEPSEEK4_MTP_EXPERIMENTAL=1`
4. expose DeepSeek4 MTP head tensors separately from target verification layers
5. add draft/accepted/rejected/fallback counters
6. keep layer split, fp16 KV, and `parallel=1`

## Benchmark Gate

Do not run the online matrix until all of these pass:

- build: `llama-server`, `llama-cli`, `llama-bench`, `test-backend-ops`
- backend ops: all DeepSeek4 custom op groups pass
- arithmetic smoke: answer `323`
- template quality: no spillover
- deterministic equivalence: target-only and MTP match at temperature `0`, or
  divergence is fully explained by sampler acceptance logic
- metrics: nonzero draft token count and visible accept/reject counters

## Decision

The roadmap MTP item is now executed to the point supported by current evidence:
existing MTP was checked and found unavailable, a branch-start scaffold artifact
was created, and benchmarking is explicitly blocked until a real llama.cpp
worktree produces a guarded compilable diff.

