# DeepSeek V4 Flash GGUF Q4 Fast-Context Experiment

Date: 2026-05-21 21:59 PDT

## Summary

We skipped further Q2 testing because quality and throughput were both worse than the Q4 baseline. Instead, this experiment reused the working `Q4_K_M-XL` GGUF path and tried a lower-context, lower-overhead server profile to see whether the previous Q4 run could be made faster without changing quant quality.

Result: quality stayed good and end-to-end latency improved for these short prompts, but decode throughput only moved slightly. The speedup mostly came from a smaller context and shorter generated answers, not a material improvement in tokens/sec.

## New Script

Created:

```text
llama_cpp_v4_q4_fastctx_a100_modal.py
```

This is a new wrapper script and does not overwrite the previous Q4 or Q2 scripts. It imports the existing Q4 Modal implementation and changes only environment defaults:

```text
LLAMA_CPP_APP_NAME=deepseek-v4-flash-llama-cpp-q4-fastctx-a100
LLAMA_CPP_MODEL_QUANT=Q4_K_M-XL
LLAMA_CPP_GPU=A100-80GB:4
LLAMA_CPP_CUDA_ARCH=80
LLAMA_CPP_CTX=4096
LLAMA_CPP_EXTRA_SERVER_ARGS=--flash-attn off --cache-ram 0 --no-warmup
```

## Runtime

Command run:

```bash
rtk modal run llama_cpp_v4_q4_fastctx_a100_modal.py --action benchmark
```

Effective `llama-server` command:

```bash
/opt/llama.cpp/build/bin/llama-server \
  -m /models/Q4_K_M-XL/DeepSeek-V4-Flash-Q4_K_M-XL-00001-of-00004.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  -c 4096 \
  -ngl 999 \
  --split-mode layer \
  --parallel 1 \
  --jinja \
  --cache-type-k q4_0 \
  --flash-attn off \
  --cache-ram 0 \
  --no-warmup
```

Note: llama.cpp accepted `--cache-type-k q4_0`, but logged that DeepSeek V4 forces fp16 KV cache, so this flag is ignored for V4.

## Baseline vs Fast-Context

| Test | Q4 baseline | Q4 fastctx | Change |
| --- | ---: | ---: | ---: |
| Warmup wall time | 162.066 s | 132.566 s | 18.2% faster |
| Warmup generation | 12.33 tok/s | 12.55 tok/s | 1.8% faster |
| Thinking off wall time | 11.535 s | 10.136 s | 12.1% faster |
| Thinking off generation | 12.25 tok/s | 12.45 tok/s | 1.6% faster |
| Thinking on wall time | 14.051 s | 9.196 s | 34.6% faster |
| Thinking on generation | 12.40 tok/s | 12.54 tok/s | 1.1% faster |

The wall-time numbers improved, but the generated token counts also changed:

| Test | Q4 baseline completion tokens | Q4 fastctx completion tokens |
| --- | ---: | ---: |
| Thinking off | 128 | 113 |
| Thinking on | 163 | 104 |

So the meaningful decode speed remained roughly flat at about 12.4 to 12.5 tokens/sec.

## Fast-Context Results

### Warmup

Prompt:

```text
What is 17*19? Answer briefly.
```

Result:

```text
elapsed: 132.566 s
prompt: 14 tokens, 1.318 s, 10.62 tok/s
generation: 16 tokens, 1.275 s, 12.55 tok/s
```

This timing includes cold model load and repeated `503 Loading model` retries.

### Sky Prompt, Thinking Off

Prompt:

```text
In a short summary, explain why the sky is blue in scientific terms
```

Settings:

```json
{
  "temperature": 0.2,
  "max_tokens": 256,
  "reasoning_effort": "none",
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

Result:

```text
elapsed: 10.136 s
completion tokens: 113
prompt: 40.91 tok/s
generation: 12.45 tok/s
```

Quality: good. The answer correctly explained Rayleigh scattering and shorter blue wavelengths.

### Sky Prompt, Thinking On

Settings:

```json
{
  "temperature": 0.2,
  "max_tokens": 256,
  "reasoning_effort": "medium",
  "chat_template_kwargs": {
    "enable_thinking": true
  }
}
```

Result:

```text
elapsed: 9.196 s
completion tokens: 104
prompt: 66.33 tok/s
generation: 12.54 tok/s
```

Quality: good. The response was concise and scientifically correct, with separate `reasoning_content`.

## Interpretation

The fast-context profile is better for short prompt tests because it reduces context allocation and startup overhead:

```text
context: 32768 -> 4096
prompt cache: disabled with --cache-ram 0
warmup: disabled with --no-warmup
flash attention: explicitly off
```

However, llama.cpp had already disabled Flash Attention for this model path, and V4 ignores the quantized KV cache setting. That leaves little flag-only room to improve decode throughput on the current llama.cpp kernel path.

## Recommendation

Use the Q4 fast-context script for short interactive tests:

```bash
rtk modal run llama_cpp_v4_q4_fastctx_a100_modal.py --action benchmark
```

For this workload, keep `Q4_K_M-XL` and constrain generation length. Q2 is not worth pursuing here: it was lower quality and much slower in practice.

For a real tokens/sec improvement, the likely next experiments are:

1. Keep Q4 quality but test different GPU partitioning only if the llama.cpp V4 kernels support it well.
2. Compare a newer llama.cpp V4 branch/commit if available.
3. Try serving settings that reduce output tokens for the application path, because per-token decode is currently the limiting factor.

## Cleanup

After the run, Modal cleanup was verified:

```text
modal container list: []
deepseek-v4-flash-llama-cpp-q4-fastctx-a100: stopped, 0 tasks
```
