# TensorRT-LLM DeepSeek V4 Flash A100 Support Check

Date: 2026-05-22 12:55:16 PDT  
Roadmap item: `REPORT_trtllm_v4_flash_a100_<timestamp>.md`  
Target hardware: Modal `A100-80GB:4`

## Scope

Check whether TensorRT-LLM is ready for a DeepSeek V4 Flash A100 benchmark in
this repo. This is a support-gate report, not a GPU benchmark.

## Sources Checked

- TensorRT-LLM supported models:
  https://nvidia.github.io/TensorRT-LLM/latest/models/supported-models.html
- TensorRT-LLM reference support matrix:
  https://nvidia.github.io/TensorRT-LLM/reference/support-matrix.html
- TensorRT-LLM quantization support:
  https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html
- NVIDIA Dynamo support matrix:
  https://docs.nvidia.com/dynamo/dev/resources/support-matrix
- TensorRT-LLM DeepSeek R1 deployment guide:
  https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-deepseek-r1-on-trtllm.html

## Findings

TensorRT-LLM is not currently the right next A100 benchmark target for this
repo.

Evidence:

- Current TensorRT-LLM public docs discuss broad support for DeepSeek R1/V3 and
  newer DeepSeek V3.2 material, but I did not find direct DeepSeek V4 Flash
  support in the TensorRT-LLM supported-models pages.
- The TensorRT-LLM reference support matrix lists many TensorRT-backend LLM
  model families. It does not list DeepSeek V4 Flash as a validated TensorRT
  backend target in the inspected public page.
- TensorRT-LLM quantization docs show Ampere support for FP8 KV cache, W4A16
  AWQ, and W4A16 GPTQ classes, but not native FP4/MXFP4 execution on Ampere.
  DeepSeek V4 Flash's current native ecosystem work is centered on FP4/FP8 and
  hybrid attention paths that already failed or required fallbacks in vLLM and
  SGLang A100 probes.
- NVIDIA Dynamo's DeepSeek V4 development release is explicitly scoped to
  Blackwell and vLLM/SGLang containers only. The same page says TensorRT-LLM is
  not part of those DeepSeek V4 dev releases.
- The available TensorRT-LLM DeepSeek deployment guide is for DeepSeek R1 on
  Blackwell/Hopper, not DeepSeek V4 Flash on A100.

## Relationship To Local Evidence

Local native-engine probes already found A100 blockers before full checkpoint
benchmarking:

- vLLM native dummy-load failed in DeepGEMM hyperconnection/MHC with an
  unsupported architecture path.
- SGLang native dummy startup reached readiness but failed the first forward
  path because the DSV4 top-k JIT kernel uses cluster launch features not
  supported on `sm_80`.

Those failures are the same risk class TensorRT-LLM would need to overcome:
DeepSeek V4 Flash is not just a standard MoE model. It needs correct hybrid
attention, compressed/indexer cache behavior, MTP/NextN handling, and A100-safe
kernel fallbacks.

## Decision

Do not spend A100 time on a TensorRT-LLM DeepSeek V4 Flash benchmark yet.

Continue TensorRT-LLM only after one of these is true:

- NVIDIA publishes a direct DeepSeek V4 Flash TensorRT-LLM deployment guide.
- TensorRT-LLM supported-models docs list DeepSeek V4 Flash or
  `DeepseekV4ForCausalLM`.
- A released NGC container includes a DeepSeek V4 Flash example or config.
- A source probe finds a model implementation plus A100-compatible hybrid
  attention and MoE fallback paths.

## Future Probe Shape

When direct support appears, the first artifact should still be a source/config
probe, not a full 149 GiB benchmark:

1. Inspect model registry for `DeepseekV4ForCausalLM` or equivalent.
2. Confirm supported quantization on Ampere.
3. Build a minimal TP=4 short-context engine.
4. Run one dummy/smoke request without MTP.
5. Add MTP only after target-only correctness.
6. Run the roadmap benchmark matrix with TTFT, prefill, decode, aggregate
   throughput, latency percentiles, quality rows, and feature activation logs.

