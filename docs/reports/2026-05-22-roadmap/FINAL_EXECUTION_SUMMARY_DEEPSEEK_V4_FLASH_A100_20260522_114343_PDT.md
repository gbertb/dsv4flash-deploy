# DeepSeek V4 Flash A100 Execution Summary

Date: 2026-05-22 11:43:43 PDT

## Completed

- Updated the roadmap to require real benchmark coverage: prefill, TTFT,
  decode speed, latency, aggregate throughput, concurrency, and engine evidence.
- Added `offline-bench` and `online-matrix` actions to
  `llama_cpp_v4_q4_a100_modal.py`.
- Added `benchmark_openai_concurrent.py` for reusable `/v1` endpoint testing.
- Added `llama_cpp_v4_q4_peer512_parallel4_a100_modal.py` for Phase 1
  parallel/concurrency testing.
- Ran the baseline online matrix on 4x A100.
- Attempted the Phase 1 `parallel=4`, `concurrency=4` sweep.
- Ran a Phase 1 `parallel=2`, `concurrency=2`, `batch-size=2048` isolation
  sweep.
- Ran a Phase 1 `parallel=1`, `concurrency=2` isolation sweep.
- Added and ran `source-inspect` to inspect the built cchuter llama.cpp source
  for DeepSeek4 parallel, KV, tensor-split, and MTP blockers.
- Added and ran an upstream llama.cpp `master` source probe to check whether
  newer generic MTP support can replace or inform the cchuter DeepSeek4 branch.
- Wrote a DeepSeek4 MTP integration design that maps cchuter DeepSeek4 model
  support to upstream generic `draft-mtp` plumbing and defines validation gates.
- Added a guarded DeepSeek4 MTP source-prototype scaffold artifact to start the
  required llama.cpp code branch without claiming it is benchmarkable.
- Added and ran a native vLLM DeepSeek V4 A100 probe covering source/config
  support, remote checkpoint metadata, and an A100 dummy-load engine startup.
- Added and ran a native SGLang DeepSeek V4 A100 probe covering launch flags,
  source support, remote checkpoint metadata, and an A100 dummy startup/forward
  path.
- Wrote the remaining roadmap design/support artifacts for llama.cpp tensor
  split, llama.cpp DeepSeek4 KV quantization, and TensorRT-LLM support status.
- Verified Modal cleanup: no active benchmark containers remained.

## Key Findings

- Warm baseline prefill from server timing was about `103-145 tok/s`.
- Warm baseline streaming decode from the matrix was about `9.44-9.87 tok/s`.
- The old `~12.995 tok/s` sky-prompt result remains the best prior single-stream
  decode number, but it is not enough by itself for future decisions.
- `llama-bench` produced useful prefill evidence at `2048x1`
  (`165.816 tok/s` mean), but failed on `512x1` with a DeepSeek4 graph assert.
- DeepSeek4 still forces fp16 KV cache; requested `q4_0` KV is ignored.
- Flash Attention remained auto-disabled.
- MTP/NextN metadata exists, but no speculative decoding implementation was
  active in this llama.cpp path.
- `parallel=4`, `concurrency=4`, `batch-size=4096` crashed during graph
  initialization and should not be treated as a performance result.
- `parallel=2`, `concurrency=2`, `batch-size=2048` crashed with the same
  DeepSeek4 graph assert, so the current branch is unsafe for `--parallel > 1`.
- `parallel=1`, `concurrency=2` stayed stable but serialized through one slot
  (`n_slots = 1`) and did not improve aggregate decode throughput. The warm
  `1024x256` aggregate completion rate was `4.588 tok/s`.
- Source inspection found that DeepSeek4 is hard-capped to one sequence in
  `llama-context.cpp`, KV cache types are forced to fp16, tensor split still has
  an unsupported-architecture path, and DeepSeek4 MTP is not exposed as a
  speculative server implementation.
- Upstream `ggml-org/llama.cpp` `master` at
  `1acee6bf8939948f9bcbf4b14034e4b475f06069` has generic `draft-mtp` plumbing,
  but no `LLM_ARCH_DEEPSEEK4` and no `src/models/deepseek4.cpp`, so it cannot
  directly replace the cchuter V4 branch for the current GGUF deployment.
- The viable llama.cpp MTP path is a merge/prototype branch: keep cchuter's
  DeepSeek4 model/CUDA implementation and port upstream `draft-mtp` interfaces,
  then validate with backend ops, smoke tests, deterministic target-vs-MTP
  checks, online matrix rows, and draft/accept/reject counters.
- The MTP prototype-start artifact defines the guarded branch shape
  (`LLAMA_CPP_DEEPSEEK4_MTP_EXPERIMENTAL=1` plus `draft-mtp`), required metrics,
  validation commands, and non-goals. It is intentionally not benchmarkable
  until a real llama.cpp worktree produces a compilable diff.
- Native vLLM support is present in `vllm/vllm-openai:deepseekv4-cu130`:
  vLLM's parser recognizes `deepseek_v4`, `DeepseekV4ForCausalLM`, 1M context,
  and `num_nextn_predict_layers=1`; source inspection also found
  `deepseek_mtp` / `DeepSeekV4MTPModel` plumbing.
- The native vLLM A100 dummy-load path failed before serving requests with
  `Unsupported architecture` in DeepGEMM hyperconnection/MHC:
  `/workspace/.deps/deepgemm-src/csrc/apis/hyperconnection.hpp:56`, reached via
  `torch.ops.vllm.mhc_pre`. Full native-weight vLLM benchmarking should be
  deferred until an Ampere-safe MHC path exists.
- Native SGLang support is present in `lmsysorg/sglang:latest`: SGLang `0.5.12`
  recognizes DeepSeek V4, exposes `deepseek-v4` reasoning and `deepseekv4` tool
  parsers, has `dsv4` attention backends, includes `deepseek_v4.py` and
  `deepseek_v4_nextn.py`, and exposes speculative/MTP flags.
- The native SGLang A100 dummy path reached server readiness, selected
  `fp8_e4m3` KV cache, loaded dummy fp8 weights with Marlin fallback at about
  `44.23 GB` per GPU, and initialized DeepSeek V4 KV pools.
- The same SGLang run failed on the first warmup/forward path because the DSV4
  JIT top-k kernel uses cluster launch features unsupported by A100:
  `__cluster_dims__ is not supported for this GPU architecture` and
  `cooperative_groups::this_cluster` is unavailable for `sm_80`. Full SGLang
  native benchmarking and MTP testing should be deferred until an
  Ampere-compatible DSV4 top-k metadata path exists.
- llama.cpp tensor split remains an implementation project, not a flag sweep:
  DeepSeek4 is not enabled for generic tensor split, row/manual split already
  failed on CUDA split-buffer `RESHAPE`, and every DeepSeek4 tensor/custom op
  needs split-role auditing before another A100 run.
- Real DeepSeek4 KV quantization remains unsupported in the current llama.cpp
  branch. The correct path is to split cache kinds, prove effective dtypes in
  runtime logs, and validate temperature-0 correctness before judging speed.
- TensorRT-LLM is not a good immediate A100 target for DeepSeek V4 Flash. Current
  public support evidence points to DeepSeek R1/V3/V3.2 and Blackwell-oriented
  DeepSeek V4 vLLM/SGLang releases; no direct TensorRT-LLM DeepSeek V4 Flash
  A100 path was found.

## New Artifacts

- `REPORT_llama_cpp_v4_q4_phase0_online_matrix_a100_20260522_114343_PDT.md`
- `REPORT_llama_cpp_v4_q4_bench_harness_a100_20260522_125816_PDT.md`
- `REPORT_llama_cpp_v4_q4_phase1_concurrency_a100_20260522_120149_PDT.md`
- `REPORT_llama_cpp_v4_q4_source_inspect_a100_20260522_120149_PDT.md`
- `REPORT_llama_cpp_v4_q4_latest_branch_probe_a100_20260522_120802_PDT.md`
- `REPORT_llama_cpp_v4_q4_mtp_design_a100_20260522_121407_PDT.md`
- `REPORT_llama_cpp_v4_q4_mtp_prototype_a100_20260522_125816_PDT.md`
- `PROTOTYPE_llama_cpp_deepseek4_mtp_source_patch_20260522_125816_PDT.md`
- `REPORT_vllm_native_v4_flash_a100_20260522_122700_PDT.md`
- `REPORT_sglang_v4_flash_a100_20260522_125109_PDT.md`
- `REPORT_llama_cpp_v4_tensor_split_design_20260522_125516_PDT.md`
- `REPORT_llama_cpp_v4_kv_quant_design_20260522_125516_PDT.md`
- `REPORT_trtllm_v4_flash_a100_20260522_125516_PDT.md`
- `FINAL_ROADMAP_DEEPSEEK_V4_FLASH_A100_20260522.md`
- `benchmark_openai_concurrent.py`
- `llama_cpp_upstream_master_source_probe_modal.py`
- `llama_cpp_v4_q4_upstream_master_source_probe_modal.py`
- `llama_cpp_v4_q4_peer512_parallel2_a100_modal.py`
- `llama_cpp_v4_q4_peer512_parallel4_a100_modal.py`
- `llama_cpp_v4_q4_peer512_concurrency2_a100_modal.py`
- `deepseek_v4_flash_vllm_native_modal.py`
- `deepseek_v4_flash_vllm_native_dummy_a100_modal.py`
- `deepseek_v4_flash_sglang_native_modal.py`

## Next Step

The roadmap execution leaves one practical code path: a focused llama.cpp
DeepSeek4 MTP prototype branch using the design report. Blind concurrency,
tensor split, KV quant, vLLM native, SGLang native, and TensorRT-LLM full
benchmarks should stop until their source/kernel blockers are addressed.

## Completion Audit

- Required context read: `README.md` and
  `docs/benchmarks/BENCHMARK_REPORT_20260521_234412_PDT.md` were inspected and
  used as the baseline evidence.
- Roadmap updated: benchmark methodology now requires TTFT, prefill, decode,
  latency, aggregate throughput, raw rows, feature activation evidence, and
  explicit stop/continue decisions.
- Phase 0 measurement: implemented harness changes and produced online/offline
  evidence in the Phase 0 report.
- Phase 1 llama.cpp sweep: executed parallel/concurrency probes and recorded
  crash/serialization evidence.
- Phase 2 MTP: source-inspected current/latest branches, produced an
  implementation design, and created a guarded source-prototype scaffold. A
  working benchmarkable MTP implementation was not claimed because current
  source lacks the required DeepSeek4 speculative serving path.
- Phase 3 vLLM: executed source/config/remote-model/A100 dummy probes and
  stopped at the A100 MHC unsupported-architecture blocker.
- Phase 4 SGLang: executed source/config/remote-model/A100 dummy-forward probes
  and stopped at the A100 DSV4 top-k unsupported-kernel blocker.
- Phase 5 TensorRT-LLM: checked current public support evidence and deferred GPU
  work until direct DeepSeek V4 Flash support appears.
- Phase 6 tensor split: produced a source-level design/prototype gate and
  deferred GPU work until split-buffer backend ops exist.
- Phase 7 KV quant: produced a source-level design/prototype gate and deferred
  benchmarking until effective per-cache dtype changes can be proven.
- Phase 8 kernel fusion/graph reduction: deferred until a supported path can be
  profiled; current native paths fail before useful A100 profiling.
- Cleanup: Modal container checks after GPU runs showed no active containers.
