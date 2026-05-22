# llama.cpp DeepSeek V4 Flash Q4_K_M-XL GGUF Report

Timestamp: 2026-05-21 21:28:11 PDT

## Objective

Validate the recommended GGUF path from `../../notes/GGUF_quantization_llama.cpp.md` for
DeepSeek V4 Flash on Modal `A100-80GB:4`:

- llama.cpp fork: `https://github.com/cchuter/llama.cpp.git`
- branch: `feat/v4-port-cuda`
- GGUF repo: `teamblobfish/DeepSeek-V4-Flash-GGUF`
- quant: `Q4_K_M-XL`
- CUDA arch: `80`
- scheduler split define: `GGML_SCHED_MAX_SPLIT_INPUTS=128`
- runtime: `llama-server`, `--split-mode layer`, `-ngl 999`, `--jinja`

The new script used for this run is:

```text
llama_cpp_v4_q4_a100_modal.py
```

Existing scripts were not overwritten.

## Prior GGUF Approaches That Did Not Work

- `antirez/ds4` q4-imatrix on DS4 CUDA failed prefill on Modal A100 and H100
  with `ds4: prompt processing failed: cuda prefill failed`.
- The same DS4 q4-imatrix target failed on the DS4 CPU path with
  `ds4: expected IQ2_XXS expert tensors`, indicating q4 expert tensors were not
  supported by that CPU route.
- The earlier vLLM GGUF path targeted `Preyazz/DeepSeek-V4-Flash-Q8_0-GGUF`
  with vLLM GGUF plugin patching. That path was exploratory and not the
  recommended `teamblobfish` plus `cchuter` llama.cpp route from the guide.

## Backend Op Validation

Command:

```bash
rtk modal run llama_cpp_v4_q4_a100_modal.py --action backend-ops
```

Result:

- Modal allocated 4x `NVIDIA A100-SXM4-80GB`, compute capability 8.0.
- `DSV4_ROPE_TAIL`, `DSV4_HC_SPLIT_SINKHORN`,
  `DSV4_HC_WEIGHTED_SUM`, `DSV4_HC_EXPAND`, and
  `DSV4_FP8_KV_QUANTIZE` all passed.
- Each CUDA backend reported `19/19 tests passed`.
- Overall result: `5/5 backends passed` with CPU skipped.

## Model Volume

Command:

```bash
rtk modal run llama_cpp_v4_q4_a100_modal.py --action inspect
```

The Q4 shards were already present:

```text
Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf  45.66 GiB
Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00002-of-00004.gguf  46.56 GiB
Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00003-of-00004.gguf  46.55 GiB
Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00004-of-00004.gguf  24.09 GiB
```

llama.cpp identified the model as:

```text
file type = Q4_K - Medium
file size = 162.86 GiB (4.92 BPW)
model params = 284.33 B
```

All 44 layers were offloaded across the four A100 GPUs.

## Benchmark

Command:

```bash
rtk modal run llama_cpp_v4_q4_a100_modal.py --action benchmark
```

Server command:

```text
/opt/llama.cpp/build/bin/llama-server -m /models/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf --host 0.0.0.0 --port 8000 -c 32768 -ngl 999 --split-mode layer --parallel 1 --jinja --cache-type-k q4_0
```

Important runtime note:

- llama.cpp logged: `DeepSeek4: forcing fp16 KV cache (--cache-type-k|v are
  ignored for V4 because compressed/indexer K caches require fp16...)`.
- Therefore the `--cache-type-k q4_0` optimization is not active for DeepSeek4
  in this fork.
- Flash Attention auto-disabled because the graph assigned its tensor to CPU.
- Pipeline parallelism was enabled.

### Warmup

Query:

```text
What is 17*19? Answer briefly.
```

Settings:

```json
{"temperature": 0, "max_tokens": 16, "stream": false}
```

Result:

- Elapsed wall time: 162.066 seconds.
- This includes model-load wait and 9 retries while `llama-server` returned
  `503 Loading model`.
- Server timing after load:
  - prompt: 14 tokens, 1.312 s, 10.67 tok/s
  - generation: 16 tokens, 1.297 s, 12.33 tok/s
- Response content field was empty because the model spent the 16-token cap in
  `reasoning_content`: `We need to compute 17*19. This is a simple
  multiplication.`
- This warmed the loaded server, but the answer did not reach `323` due to the
  very small warmup token cap and default thinking behavior.

### Sky Prompt, Thinking Off

Query:

```text
In a short summary, explain why the sky is blue in scientific terms
```

Settings:

```json
{
  "temperature": 0.2,
  "max_tokens": 256,
  "stream": false,
  "reasoning_effort": "none",
  "chat_template_kwargs": {"enable_thinking": false}
}
```

Result:

- Elapsed wall time: 11.535 seconds.
- Thinking controls were accepted.
- Completion: 128 tokens.
- Prompt speed: 38.76 tok/s.
- Generation speed: 12.25 tok/s.
- Response was coherent and concise. It explained that sunlight contains all
  colors, atmospheric nitrogen and oxygen cause Rayleigh scattering, shorter
  blue/violet wavelengths scatter more strongly than red wavelengths, and human
  vision makes the sky appear blue.

### Sky Prompt, Thinking On

Query:

```text
In a short summary, explain why the sky is blue in scientific terms
```

Settings:

```json
{
  "temperature": 0.2,
  "max_tokens": 256,
  "stream": false,
  "reasoning_effort": "medium",
  "chat_template_kwargs": {"enable_thinking": true}
}
```

Result:

- Elapsed wall time: 14.051 seconds.
- Thinking controls were accepted.
- Completion: 163 tokens.
- Prompt speed: 58.93 tok/s.
- Generation speed: 12.40 tok/s.
- Response was coherent and concise. It included separate
  `reasoning_content`, then answered that Rayleigh scattering by atmospheric gas
  molecules scatters shorter blue/violet wavelengths more effectively than red
  wavelengths; our lower violet sensitivity and some upper-atmosphere
  absorption make the perceived sky blue.

## Assessment

This GGUF quant is working on the target 4x A100 Modal shape. Compared with the
previous official/Base INT4 paths, it gives cleaner chat responses and avoids
the unrelated template spillover seen in the Base checkpoint.

Performance after warm load is about 12.3 to 12.4 generated tokens per second
for the tested sky prompt. The main obvious runtime optimization from the guide,
`--cache-type-k q4_0`, is currently ignored by this DeepSeek4 llama.cpp path
because compressed/indexer K caches require fp16. Flash Attention also disabled
automatically, so there is no safe flag-only performance improvement confirmed
from this run.

The Q4_K_M-XL result is satisfactory for the requested smoke benchmark:

- backend ops passed on all four A100s,
- Q4 shards loaded,
- all layers offloaded,
- warm sky responses were scientifically correct,
- thinking off/on controls were accepted,
- no runaway Modal containers were left after cleanup.

## Cleanup

After the benchmark:

- the lingering benchmark container was explicitly stopped,
- `rtk modal container list --json` returned `[]`,
- the Q4/A100 experiment app was `stopped` with `Tasks: 0`.

## Next Experiment

The next GGUF experiment from the guide is the lighter `Q2_K-XL` quant from the
same `teamblobfish/DeepSeek-V4-Flash-GGUF` repo on the same `cchuter` llama.cpp
fork and 4x A100 runtime. It should establish the quality/speed tradeoff below
the now-working recommended Q4 baseline.
