# Prototype Patch Sketch: llama.cpp DeepSeek4 MTP

Date: 2026-05-22 12:58:16 PDT  
Status: source-prototype scaffold, not benchmarkable  
Base branch: `cchuter/llama.cpp feat/v4-port-cuda`  
Reference branch: `ggml-org/llama.cpp master` at
`1acee6bf8939948f9bcbf4b14034e4b475f06069`

## Intent

Start the llama.cpp DeepSeek4 MTP code branch requested by the roadmap without
misrepresenting it as a working implementation.

The inspected cchuter branch has DeepSeek4 model/CUDA support but no active
DeepSeek4 speculative serving implementation. Upstream master has generic
`draft-mtp` plumbing but no `LLM_ARCH_DEEPSEEK4`. The only credible branch shape
is therefore:

```text
cchuter feat/v4-port-cuda
  + selected upstream draft-mtp interfaces
  + DeepSeek4 nextn_predict_layers wiring
  + draft/accept/reject metrics
```

## Minimum Patch Shape

The prototype branch should start with a guard, so it cannot silently affect the
known-good layer-split baseline:

```cpp
// common/speculative.h
enum common_speculative_type {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_NGRAM,
    COMMON_SPECULATIVE_TYPE_DRAFT_MTP,
};
```

```cpp
// common/speculative.cpp
static common_speculative_type common_speculative_type_from_str(std::string value) {
    if (value == "draft-mtp") {
        return COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
    }
    ...
}
```

```cpp
// server settings / params
struct deepseek4_mtp_metrics {
    uint64_t draft_tokens_total = 0;
    uint64_t accepted_tokens_total = 0;
    uint64_t rejected_tokens_total = 0;
    uint64_t fallback_count = 0;
    double draft_eval_ms = 0.0;
    double verify_eval_ms = 0.0;
};
```

```cpp
// DeepSeek4 model init gate
if (model.arch == LLM_ARCH_DEEPSEEK4 && params.speculative.type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
    GGML_ASSERT(hparams.nextn_predict_layers == 1);
    GGML_ASSERT(getenv("LLAMA_CPP_DEEPSEEK4_MTP_EXPERIMENTAL") != nullptr);
}
```

## Required Real Implementation Work

This scaffold is intentionally not enough. The real branch must still answer
these source questions before compilation/benchmarking:

1. Where the cchuter GGUF loader stores the MTP/NextN tensors.
2. Whether the MTP head is represented as trailing layers, separate tensors, or
   model-specific blocks.
3. How to expose the previous target hidden state to the draft head without
   adding work to every normal verification step.
4. Whether upstream `kv_only_nextn` semantics can be ported cleanly to the
   cchuter DeepSeek4 graph/cache layout.
5. How draft verification interacts with DeepSeek4 compressed/indexer cache
   state and the current `n_comp_visible <= n_comp_cache` assertion.
6. How to keep the existing one-sequence DeepSeek4 cap until single-stream MTP
   correctness is proven.

## Validation Commands For The Branch

The first branch wrapper should set only the MTP guard and flags:

```bash
LLAMA_CPP_DEEPSEEK4_MTP_EXPERIMENTAL=1
LLAMA_CPP_EXTRA_SERVER_ARGS="--cache-ram 0 --no-warmup --batch-size 2048 --ubatch-size 512 --poll 100 --poll-batch 1 --spec-type draft-mtp --draft-max 1"
rtk modal run llama_cpp_v4_q4_mtp_prototype_a100_modal.py --action backend-ops
```

Required gate order:

1. source diff review
2. CPU/CUDA build
3. `test-backend-ops`
4. arithmetic CLI smoke
5. temperature-0 target-only vs MTP equivalence
6. online matrix with draft/accepted/rejected counters

Do not run a throughput benchmark before gates 1-5 pass.

## Non-Goals For This Prototype

- No `--parallel > 1`
- No tensor split
- No KV dtype changes
- No multi-token draft tree
- No vLLM/SGLang behavior changes

## Branch Start Decision

This file is the repo-level source-prototype start artifact. It intentionally
does not claim a working patch because the source inspection did not prove where
the cchuter branch exposes the DeepSeek4 MTP head to the decode loop.

The next engineering task is to clone the cchuter and upstream branches into a
real llama.cpp worktree, port the upstream `draft-mtp` interfaces, and produce a
compilable diff with the guard above.

