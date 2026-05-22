# DeepSeek V4 Flash Q4 llama.cpp Latest Branch Probe

Date: 2026-05-22 12:08:02 PDT  
Roadmap item: `REPORT_llama_cpp_v4_q4_latest_branch_probe_a100_<timestamp>.md`  
Probe type: source-only Modal probe, no GPU allocation

## Scope

This report continues the roadmap after Phase 1 showed that the current
cchuter `feat/v4-port-cuda` branch cannot safely use `--parallel > 1`. The goal
was to check whether a newer upstream llama.cpp branch has useful MTP, tensor
split, KV, or DeepSeek4 changes before spending more A100 time.

## Code Added

- `llama_cpp_v4_q4_a100_modal.py`
  - Added `latest-source-probe`, a source-only probe action.
  - Added configurable source-probe repo/branch variables.
- `llama_cpp_v4_q4_upstream_master_source_probe_modal.py`
  - Wrapper for upstream `ggml-org/llama.cpp` `master`.
- `llama_cpp_upstream_master_source_probe_modal.py`
  - Standalone compact probe that avoids importing the full serving module, so
    it does not trigger the CUDA build.

## Commands

The first source probe completed, but importing the full serving module caused
Modal to build the full CUDA llama.cpp image too. I then added and ran the
standalone compact probe:

```bash
rtk modal run llama_cpp_upstream_master_source_probe_modal.py
```

## Upstream Probe Result

Repository:

```text
https://github.com/ggml-org/llama.cpp.git
branch: master
commit: 1acee6bf8939948f9bcbf4b14034e4b475f06069
```

Compact probe evidence:

```json
{
  "files_exist": {
    "src/models/deepseek4.cpp": false,
    "common/speculative.cpp": true,
    "tools/server/server.cpp": true
  },
  "pattern_counts": {
    "LLM_ARCH_DEEPSEEK4": 0,
    "deepseek4": 0,
    "draft-mtp": 3,
    "COMMON_SPECULATIVE_TYPE_DRAFT_MTP": 11,
    "nextn_predict_layers": 75,
    "kv_only_nextn": 2,
    "LLAMA_SPLIT_MODE_TENSOR": 9,
    "forcing fp16 KV": 0
  }
}
```

Important matches:

- `common/speculative.cpp` exposes `"draft-mtp"`.
- `common/common.h` describes `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` as
  multi-token prediction.
- `tools/server/server-context.cpp` references
  `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`.
- `src/llama-hparams.h` has `kv_only_nextn`, described as trailing MTP-head
  blocks owning KV cache.
- No `LLM_ARCH_DEEPSEEK4` symbols were found.
- No `src/models/deepseek4.cpp` file exists in upstream `master`.

## Interpretation

Upstream `master` has generic MTP/draft-MTP plumbing that is newer than the
tested cchuter branch, but it does not contain the DeepSeek4 architecture/model
implementation needed for the teamblobfish DeepSeek V4 GGUF path.

This means upstream `master` is not directly usable as a replacement serving
branch for the current best deployment. A useful llama.cpp code path would be a
merge/cherry-pick project:

1. Keep cchuter DeepSeek4 model/CUDA support.
2. Port or merge upstream generic `draft-mtp` speculative infrastructure.
3. Wire DeepSeek4 `nextn_predict_layers = 1` into that draft-MTP path.
4. Re-run backend ops and deterministic online matrix before using A100 decode
   benchmarks.

## Decision

Do not benchmark upstream `master` on A100 for this model. It cannot load the
current DeepSeek4 GGUF path without DeepSeek4 architecture support.

The next meaningful llama.cpp work is a source-level integration plan for
DeepSeek4 MTP:

- compare cchuter DeepSeek4 model code against upstream `draft-mtp` interfaces
- identify required MTP tensors and hidden-state inputs
- add server metrics for draft/accepted/rejected tokens
- only then run a short A100 smoke benchmark

## Cleanup

The compact source probe used no GPU allocation. Modal cleanup was checked after
the run.

