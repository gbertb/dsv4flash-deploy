# DeepSeek V4 Flash Q4 llama.cpp Source Inspection Report

Date: 2026-05-22 12:01:49 PDT  
Source image: `cchuter/llama.cpp`, branch `feat/v4-port-cuda`, build `b1-781e978`  
Related benchmark reports:

- `REPORT_llama_cpp_v4_q4_phase0_online_matrix_a100_20260522_114343_PDT.md`
- `REPORT_llama_cpp_v4_q4_phase1_concurrency_a100_20260522_120149_PDT.md`

## Scope

This report executes the source-inspection part of the roadmap after Phase 1
showed that `--parallel > 1` crashes and `parallel=1/concurrency=2` serializes
through one slot. The inspection used a new `source-inspect` Modal action in
`llama_cpp_v4_q4_a100_modal.py` to read `/opt/llama.cpp` inside the built image.

Command:

```bash
rtk modal run llama_cpp_v4_q4_peer512_a100_modal.py --action source-inspect
```

## Findings

### 1. DeepSeek4 Is Hard-Capped To One Sequence

`src/llama-context.cpp` contains:

```cpp
const uint32_t n_seqs = model.arch == LLM_ARCH_DEEPSEEK4 ? 1 : cparams.n_seq_max;
```

This matches runtime evidence from the concurrency run:

```text
llama_context: n_seq_max = 1
srv load_model: initializing slots, n_slots = 1
```

Conclusion: plain client concurrency cannot improve aggregate throughput in the
current llama.cpp path because the server has one DeepSeek4 sequence/slot.

### 2. `--parallel > 1` Hits DeepSeek4 Graph/Cache Assumptions

The benchmark crash occurred here in `src/models/deepseek4.cpp`:

```cpp
const int64_t n_comp_visible = (last_pos + 1) / compress_ratio;
const int64_t n_comp_cache = mctx_dsv4->get_dsv4_n_comp(il);
GGML_ASSERT(n_comp_visible <= n_comp_cache);
```

This assert fired for both:

- `parallel=4`, `concurrency=4`, `batch-size=4096`
- `parallel=2`, `concurrency=2`, `batch-size=2048`

Conclusion: the issue is not only the larger batch. The current DeepSeek4 graph
and compressed-cache path is not safe for multi-slot `--parallel` serving.

### 3. DeepSeek4 KV Cache Quantization Is Explicitly Disabled

`src/llama-context.cpp` logs:

```text
DeepSeek4: forcing fp16 KV cache (--cache-type-k|v are ignored for V4 because compressed/indexer K caches require fp16)
```

`src/llama-model.cpp` also pins the cache types:

```cpp
ggml_type v4_type_k = GGML_TYPE_F16;
ggml_type v4_type_v = GGML_TYPE_F16;
```

The source comments say Q8-style cache pinning can silently corrupt decode.

Conclusion: `--cache-type-k q4_0` is only a compatibility flag in the current
wrapper. It is not an active memory or speed optimization for DeepSeek4.

### 4. Tensor Split Still Needs Source Work

The inspection found the generic tensor-split failure path:

```cpp
throw std::runtime_error(std::string("LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '") + llm_arch_name(arch) + "'");
```

Runtime history already showed row/manual split failing on a CUDA split-buffer
`RESHAPE` operation. Combined with the DeepSeek4 custom compressed/indexer cache
and graph assertions above, tensor split should be treated as an implementation
project, not a flag sweep.

### 5. NextN/MTP Metadata Is Loaded But Not Wired Into Serving

DeepSeek4 metadata loading includes:

```cpp
ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
```

Runtime logs show:

```text
nextn_predict_layers = 1
speculative decoding will use checkpoints
no implementations specified for speculative decoding
```

The source tree had no DeepSeek4-specific speculative type such as
`deepseek_mtp` or `deepseek4-mtp` in the inspected `common/speculative.cpp`
path. Existing speculative types include generic draft/eagle/ngram-style paths,
not a DeepSeek4 MTP path.

Conclusion: MTP remains the highest-upside llama.cpp code project, but it is not
available as a server flag in this branch.

## Decision

Stop spending A100 time on blind llama.cpp concurrency flags for this branch.
The next useful work items are:

1. Implement or find a DeepSeek4 MTP speculative path in llama.cpp.
2. Inspect newer cchuter/upstream commits before writing that code.
3. Try a native vLLM or SGLang probe only if it can load on A100 without
   Hopper-only kernel blockers.
4. Treat tensor split and KV quantization as design/prototype work, not simple
   runtime configuration.

## Cleanup

The `source-inspect` action used no GPU allocation. After the surrounding Modal
runs, `rtk modal container list --json` returned `[]`.

