# DeepSeek V4 Flash Q4 llama.cpp A100 Optimization Report

Date: 2026-05-21 22:33 PDT

## Short Answer

The successful Q4 llama.cpp runs are not using tensor parallelism. They are using `--split-mode layer`, which is llama.cpp's pipeline-parallel multi-GPU mode. I created a tensor-parallel probe script, but I did not run it because the cchuter DeepSeek V4 llama.cpp branch explicitly rejects `--split-mode tensor` for `LLM_ARCH_DEEPSEEK4`.

The best measured Q4 configuration so far is `llama_cpp_v4_q4_fastpipe_a100_modal.py`: about 12.88 generated tokens/sec on the short sky prompts, up from the prior roughly 12.45-12.54 tok/sec Q4 fast-context baseline.

## New Scripts

These were added without replacing the previous scripts:

- `llama_cpp_v4_q4_fastpipe_a100_modal.py`
  - Q4_K_M-XL, 4x A100-80GB, `-c 4096`, `--split-mode layer`
  - Adds `CUDA_SCALE_LAUNCH_QUEUES=4x`, `GGML_CUDA_P2P=1`
  - Adds `--cache-ram 0 --no-warmup --batch-size 2048 --ubatch-size 512 --poll 100 --poll-batch 1`
- `llama_cpp_v4_q4_spec_ngram_a100_modal.py`
  - Same base as fastpipe
  - Adds `--spec-type ngram-mod` and ngram tuning flags
- `llama_cpp_v4_q4_cublas_a100_modal.py`
  - Same base as fastpipe
  - Builds with `-DGGML_CUDA_FORCE_CUBLAS=ON`
- `llama_cpp_v4_q4_tensor_probe_a100_modal.py`
  - Prepared as a tensor-parallel probe with `--split-mode tensor`, f16 KV, and flash attention
  - Not executed because source inspection shows this branch will fail for DeepSeek4 tensor split.

I also extended `llama_cpp_v4_q4_a100_modal.py` so wrappers can set `LLAMA_CPP_CMAKE_EXTRA_ARGS`, `LLAMA_CPP_SPLIT_MODE`, CUDA queue scaling, CUDA P2P, and cublas compute env vars without changing older wrapper behavior.

## Results

| Script | Main difference | Prompt tok/s | Generated tok/s | Notes |
| --- | --- | ---: | ---: | --- |
| `llama_cpp_v4_q4_fastctx_a100_modal.py` | Prior fast-context baseline, `-c 4096`, no warmup, no cache RAM | 12.55 warmup, 12.45-12.54 sky decode | 12.45-12.54 | Previous best baseline |
| `llama_cpp_v4_q4_fastpipe_a100_modal.py` | Adds launch queue scaling, P2P, batch/ubatch/poll tuning | 40.73 off, 66.24 on | 12.8779 off, 12.8854 on | Best measured decode |
| `llama_cpp_v4_q4_spec_ngram_a100_modal.py` | Adds `ngram-mod` speculative decode | 33.92 off, 39.15 on | 10.616 off, 10.989 on | Slower; generated zero drafts |
| `llama_cpp_v4_q4_cublas_a100_modal.py` | Forces cuBLAS CUDA path at build time | 35.13 off, 31.85 on | 12.678 off, 12.698 on | Did not beat fastpipe |

Fastpipe did move the Q4 run above 12.5 tok/sec, but only modestly. The remaining bottleneck appears to be llama.cpp's current DeepSeek4 execution path on A100s, not a simple server flag.

## Tensor Parallelism

llama.cpp has two relevant multi-GPU split modes:

- `layer`: pipeline parallelism. This is the default and the mode used by all successful Q4 runs here.
- `tensor`: tensor parallelism. llama.cpp docs describe this as experimental and intended to reduce generation latency, while pipeline mode is more about throughput.

For this specific branch and model architecture, tensor parallelism is blocked:

- `src/llama-arch.cpp:932` in the checked-out cchuter branch has `llm_arch_supports_sm_tensor()`.
- `src/llama-arch.cpp:951` lists supported tensor-split architectures and does not include `LLM_ARCH_DEEPSEEK4`.
- `src/llama-model.cpp:301` throws if `LLAMA_SPLIT_MODE_TENSOR` is requested for an unsupported architecture.

So the direct answer is: no, the working Q4 deployments are not doing tensor parallelism. They are layer/pipeline split across 4 GPUs.

## KV Cache Quantization

The previous runtime example used `--cache-type-k q4_0`. In this DeepSeek4 branch, that flag is accepted but ignored for V4.

Source inspection shows the branch forcibly pins DeepSeek4 KV cache to f16:

- `src/llama-context.cpp:3293` in the checked-out cchuter branch starts the DeepSeek4 KV override.
- `src/llama-context.cpp:3298` forces both K and V cache types to `GGML_TYPE_F16`.
- `src/llama-model.cpp:1986` repeats the same protection during memory planning.

The comments explain that V4 has standard SWA K, compressed-attention K, and indexer K caches that must share dtype because the implementation concatenates views. It also notes that V4 K activations are already FP8-quantized internally, and `q8_0` block scaling can silently corrupt decode. That is why `--cache-type-k q4_0` is not an effective speed/memory lever for this V4 port.

## Speculative Decoding And MTP

The official llama.cpp speculative decoding docs explain the basic idea: draft multiple tokens and verify them in batches. In this branch, server-supported speculative types are:

- `none`
- `draft`
- `eagle3`
- `ngram-simple`
- `ngram-map-k`
- `ngram-map-k4v`
- `ngram-mod`
- `ngram-cache`

There is no usable MTP/NextN speculative type exposed in this cchuter V4 branch:

- `common/speculative.cpp:818` in the checked-out cchuter branch lists the available speculative types.
- `common/speculative.cpp:878` and `:922` contain TODO comments to add MTP.
- `src/models/deepseek4.cpp:15` loads `deepseek4.nextn_predict_layers`, and our model metadata reported `deepseek4.nextn_predict_layers u32 = 1`.

That means the GGUF appears to carry NextN/MTP-related metadata, but this branch does not currently wire it into llama-server speculative decoding. I tested ngram speculative decode because it was available, but it produced zero drafts on the short sky prompt and slowed the run.

## Interpretation

The runtime flags from the example were mostly already represented in the previous Q4 path:

- Q4 model shards were already used.
- Multi-GPU was already active.
- `--split-mode layer` was already the mode.
- `--jinja` was already enabled.
- `--cache-type-k q4_0` was passed, but the branch forces f16 for DeepSeek4.

The useful new speed-oriented adjustments were:

- Lower context to 4096 for the benchmark path.
- Disable warmup and cache RAM.
- Enable `CUDA_SCALE_LAUNCH_QUEUES=4x`.
- Enable `GGML_CUDA_P2P=1`.
- Tune `--batch-size`, `--ubatch-size`, and poll settings.

Those improvements raised measured decode from roughly 12.5 tok/sec to roughly 12.88 tok/sec.

## Next Useful Experiments

The next material speed jump likely requires code support, not just runtime flags:

- Find or patch a llama.cpp branch where DeepSeek4 supports `--split-mode tensor`.
- Find or patch a branch where DeepSeek4 NextN/MTP is connected to llama-server speculative decoding.
- Build a deterministic `llama-bench` style decode benchmark for this model to compare kernels without chat-template/request overhead.
- Test H100/H200 separately if the target hardware can change; A100 lacks native FP8 tensor cores, which may matter for the V4 path.

## Sources

- llama.cpp speculative decoding docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- llama.cpp multi-GPU docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- llama.cpp CUDA build docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md
- cchuter DeepSeek V4 CUDA branch inspected locally from `https://github.com/cchuter/llama.cpp/tree/feat/v4-port-cuda`, commit `781e978`.
