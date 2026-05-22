# DeepSeek V4 Flash Q4 llama.cpp MTP Integration Design

Date: 2026-05-22 12:14:07 PDT  
Roadmap item: `REPORT_llama_cpp_v4_q4_mtp_a100_<timestamp>.md` / Phase 2 design  
Hardware target for validation: Modal `A100-80GB:4`  
Current serving baseline: `llama_cpp_v4_q4_peer512_a100_modal.py`

## Scope

This report turns the Phase 2 MTP/NextN roadmap item into an implementation
plan. It is based on:

- `REPORT_llama_cpp_v4_q4_source_inspect_a100_20260522_120149_PDT.md`
- `REPORT_llama_cpp_v4_q4_latest_branch_probe_a100_20260522_120802_PDT.md`
- Phase 0/1 online matrix and concurrency results

The goal is to improve single-stream decode while keeping the only verified
4x A100 path: cchuter DeepSeek4 GGUF + teamblobfish `Q4_K_M-XL`.

## Current State

### cchuter `feat/v4-port-cuda`

Pros:

- Loads `teamblobfish/DeepSeek-V4-Flash-GGUF`, `Q4_K_M-XL`.
- Runs DeepSeek4 CUDA custom ops on A100.
- Produces coherent chat output and prior best `~12.995 tok/s` single-stream
  decode.
- Loads `deepseek4.nextn_predict_layers = 1`.

Blockers:

- No DeepSeek4 MTP speculative server implementation is active.
- Runtime logs say:

```text
speculative decoding will use checkpoints
no implementations specified for speculative decoding
```

- `--parallel > 1` is unsafe for this branch and hits:

```cpp
GGML_ASSERT(n_comp_visible <= n_comp_cache);
```

- DeepSeek4 is hard-capped to one sequence:

```cpp
const uint32_t n_seqs = model.arch == LLM_ARCH_DEEPSEEK4 ? 1 : cparams.n_seq_max;
```

### upstream `ggml-org/llama.cpp` `master`

Commit probed:

```text
1acee6bf8939948f9bcbf4b14034e4b475f06069
```

Pros:

- Has generic MTP plumbing:

```cpp
{"draft-mtp", COMMON_SPECULATIVE_TYPE_DRAFT_MTP}
```

- Server/common code references `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`.
- Has `kv_only_nextn` for MTP-head architectures:

```cpp
bool kv_only_nextn = false; // if true, only the last nextn_predict_layers blocks have a KV cache (MTP head arches)
```

Blocker:

- No `LLM_ARCH_DEEPSEEK4`.
- No `src/models/deepseek4.cpp`.
- Cannot load the current DeepSeek4 GGUF path directly.

## Design Direction

Do not attempt to use upstream `master` as the serving branch. Instead, create a
new llama.cpp integration branch based on cchuter `feat/v4-port-cuda` and port
the minimum upstream `draft-mtp` infrastructure needed to run the existing
DeepSeek4 NextN head.

Working branch concept:

```text
cchuter/feat/v4-port-cuda
  + upstream common speculative draft-mtp plumbing
  + DeepSeek4 model wiring for nextn_predict_layers
  + DeepSeek4-specific speculative metrics
```

## Implementation Plan

### 1. Port Generic `draft-mtp` Interfaces

Source side to compare/cherry-pick from upstream:

- `common/common.h`
- `common/speculative.cpp`
- `common/speculative.h`
- `common/arg.cpp`
- `tools/server/server-context.cpp`
- any changed server settings structs that carry speculative config

Expected user-facing flag shape should reuse upstream where possible:

```bash
--spec-type draft-mtp
--draft-max N
```

Do not introduce `deepseek4-mtp` as a separate public flag unless the generic
upstream path cannot represent the DeepSeek4 head. A generic flag keeps future
upstream merging easier.

### 2. Preserve cchuter DeepSeek4 Model And CUDA Paths

Keep these cchuter-specific pieces as authoritative:

- `src/models/deepseek4.cpp`
- DeepSeek4 custom CUDA ops
- DeepSeek4 hybrid ISWA/recurrent memory path
- fp16 KV forcing behavior
- layer split behavior

Do not change tensor split, KV quantization, or `--parallel` in the same branch.
Those are separate risk domains.

### 3. Wire DeepSeek4 NextN/MTP Head

The model already loads:

```cpp
ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.nextn_predict_layers, false);
```

The initial correctness gate should assert:

```text
hparams.nextn_predict_layers == 1
```

If the GGUF stores the MTP head as trailing layers, mirror upstream MTP-head
handling:

- keep main model layers as the verification model
- expose the final MTP layer as the draft head
- ensure only the MTP layer has draft KV where required
- prevent the MTP layer from being counted as a normal verification layer

Key thing to avoid: accidentally running the MTP head as part of every target
verification step. That would add latency without producing drafts.

### 4. DeepSeek4 Draft Step

The draft step should be single-token first:

1. Run target model normally and accept target token `t`.
2. Use the previous hidden state plus embedding/projection for `t` as MTP input.
3. Run the DeepSeek4 MTP layer to draft one token.
4. Verify the drafted token through the target model.
5. Accept if logits/sampling rule matches the target path; otherwise fall back
   to target token.

Start with `num_speculative_tokens=1`. Do not implement multi-token trees until
single-token draft acceptance and timing are correct.

### 5. Required Metrics

Add server-side counters and include them in the benchmark report:

```text
spec_draft_tokens_total
spec_accepted_tokens_total
spec_rejected_tokens_total
spec_acceptance_rate
spec_fallback_count
spec_draft_eval_ms
spec_verify_eval_ms
target_eval_ms
generated_tok_s_with_spec
generated_tok_s_without_spec
```

The benchmark harness can already record online TTFT/decode; these counters are
needed to explain why a result improves or regresses.

## Validation Gates

### Build Gate

Build targets:

```bash
llama-server
llama-cli
llama-bench
test-backend-ops
```

### Backend Ops Gate

Run:

```bash
rtk modal run <mtp-wrapper>.py --action backend-ops
```

Pass condition:

- all DeepSeek4 custom op groups pass as before

### Smoke Gate

Run:

```bash
rtk modal run <mtp-wrapper>.py --action cli-smoke
```

Pass condition:

- arithmetic smoke still returns `323`
- no chat-template spillover
- no looped garbage

### Deterministic Correctness Gate

Run target-only and MTP at temperature `0` for the same prompts:

- arithmetic
- sky/science
- short factual prompt
- repeated-fact context prompt

Pass condition:

- MTP output must match target-only output at temperature `0`, or the report
  must explain exactly why the sampling path makes equality impossible.

### Benchmark Gate

Run the Phase 0 online matrix:

```bash
LLAMA_CPP_ONLINE_MATRIX=128x256,1024x256 \
rtk modal run <mtp-wrapper>.py --action online-matrix
```

Then run:

```bash
LLAMA_CPP_ONLINE_MATRIX=128x512,1024x512 \
rtk modal run <mtp-wrapper>.py --action online-matrix
```

Pass condition:

- warm online decode beats the old `~12.995 tok/s` sky-prompt decode baseline,
  or at minimum beats the Phase 0 warm matrix decode range of `9.44-9.87 tok/s`
  while showing nonzero accepted drafts
- acceptance rate is high enough to justify the added draft work
- TTFT does not regress materially

## Abort Criteria

Stop the implementation path if any of these happen:

- MTP head cannot be addressed separately from the target model layers.
- Draft acceptance is near zero on deterministic prompts.
- Draft step increases graph splits enough that decode regresses.
- It requires enabling `--parallel > 1` before single-stream correctness works.
- It requires changing DeepSeek4 KV dtype or tensor split in the same patch.
- It creates output drift at temperature `0` that cannot be explained by the
  sampler acceptance rule.

## Expected Outcome

If single-token MTP drafts are accepted at a high rate, a realistic target is
`1.3-1.8x` single-stream decode improvement. On the current baseline that means:

```text
~13 tok/s -> ~17-23 tok/s
```

This estimate is not a claim of achieved speed. It is the threshold that makes
the implementation worth continuing after correctness passes.

## Next Artifact

The next report should be:

```text
REPORT_llama_cpp_v4_q4_mtp_prototype_a100_<timestamp>.md
```

It should include:

- commit/diff summary
- exact MTP flags
- backend-op result
- smoke result
- target-only vs MTP correctness rows
- online matrix rows
- draft/accept/reject counters

