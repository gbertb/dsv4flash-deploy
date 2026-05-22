# DeepSeek V4 Flash llama.cpp KV Quantization Design

Date: 2026-05-22 12:55:16 PDT  
Roadmap item: `REPORT_llama_cpp_v4_kv_quant_design_<timestamp>.md`  
Target path: cchuter `llama.cpp`, branch `feat/v4-port-cuda`

## Scope

Define the implementation and validation work required for real DeepSeek4 KV
cache quantization in the current llama.cpp GGUF path.

This is a design/prototype gate. The current repo does not have working
DeepSeek4 KV quantization in llama.cpp.

## Current Evidence

The current server accepts flags such as:

```bash
--cache-type-k q4_0
```

but the source inspection found that this is ignored for DeepSeek4:

```text
DeepSeek4: forcing fp16 KV cache (--cache-type-k|v are ignored for V4 because compressed/indexer K caches require fp16)
```

The model code pins cache types to fp16:

```cpp
ggml_type v4_type_k = GGML_TYPE_F16;
ggml_type v4_type_v = GGML_TYPE_F16;
```

The source comments warn that Q8-style cache pinning can silently corrupt
decode. That means KV quantization is a correctness project, not a memory flag.

## Why It Matters

Short-context single-stream decode may not improve much from lower precision KV
alone. The value is more likely in:

- longer contexts
- higher concurrency
- larger prefix caches
- stacking with MTP after correctness is proven

The Phase 0 online matrix already shows that we need to split prefill, TTFT,
decode, and aggregate throughput. KV quant should be judged mainly on long
context and concurrency rows, not only short-prompt decode.

## Required Implementation Work

### 1. Split Cache Kinds

DeepSeek4 uses multiple cache concepts that should not be forced into one dtype:

- SWA/local-attention K/V
- compressed attention K/V
- indexer K cache
- recurrent/state-like compressed metadata

The implementation should separate storage by cache kind instead of sharing one
global K/V dtype and view model.

### 2. Define Allowed Dtypes Per Cache Kind

Initial conservative matrix:

| Cache kind | Initial dtype | Candidate lower dtype | Notes |
| --- | --- | --- | --- |
| SWA K/V | fp16 | q8/fp8 | easiest first target |
| compressed K/V | fp16 | fp8 only after tests | kernel-boundary dequant likely needed |
| indexer K | fp16 | keep fp16 initially | source already warns corruption risk |
| recurrent/state pools | fp16 | keep fp16 initially | do not quantize before deterministic tests |

Start with one cache kind at a time. Avoid a broad `q4_0` switch for every
DeepSeek4 cache.

### 3. Dequantize At Kernel Boundaries

Avoid exposing quantized storage to existing kernels that assume fp16 views.
Preferred approach:

- store lower precision for the chosen cache kind
- dequantize into scratch or fused kernel inputs at the boundary
- keep indexer and metadata math fp16 until proven otherwise

Only fuse dequantization after correctness and profiling show the memory traffic
is worthwhile.

### 4. Add Explicit Runtime Evidence

Every run must log:

- requested cache dtype
- effective dtype per DeepSeek4 cache kind
- bytes per token by cache kind
- total KV memory reserved
- peak VRAM per GPU
- whether any requested dtype was ignored

The report should reject runs where a flag is accepted but silently ignored.

## Validation Gates

Correctness:

- arithmetic prompt returns `323`
- sky/science prompt remains coherent
- repeated-fact long context returns the right fact
- temperature `0` output matches fp16 baseline for selected prompts, or every
  divergence is explained and bounded
- no chat-template spillover

Performance:

- `512x1`, `2048x1`, `8192x1`, `32768x1` prefill-heavy rows
- `128x256`, `128x512`, `1024x256` decode/mixed rows
- `16384x128` and `32768x128` long-context continuation rows
- concurrency 1/2/4 if memory allows

Required metrics:

- TTFT
- online prefill estimate
- steady decode tok/s
- aggregate completion tok/s
- latency p50/p90/p99
- peak VRAM per GPU

## Abort Criteria

Stop a KV quant branch if:

- indexer/cache dtype changes cause temperature-0 drift
- the implementation only logs the requested dtype, not the effective dtype
- short-context decode regresses without a long-context/concurrency benefit
- the patch also changes tensor split or MTP, making attribution unclear

## Decision

Do not treat `--cache-type-k q4_0` as an active optimization for DeepSeek4 in
current reports. The next KV artifact should be a prototype branch that splits
cache kinds and proves effective dtype changes with explicit runtime logs.

