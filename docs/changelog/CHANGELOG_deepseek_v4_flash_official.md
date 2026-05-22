# DeepSeek V4 Flash Official Modal Changelog

## 2026-05-21

- Added a new official-inference Modal path separate from the prior DS4/GGUF and vLLM experiments.
- Target runtime is `deepseek-ai/DeepSeek-V4-Flash` using the repository's `inference/convert.py`, `generate.py`, `model.py`, and `encoding` helpers.
- Modal storage is a persistent `/dsv4-volume` volume mount so HF downloads and MP=4 converted shards can be reused across runs.
- Serving runs `torchrun --nproc-per-node 4` on `A100-80GB:4`; rank 0 exposes a small OpenAI-compatible FastAPI endpoint and broadcasts generation work to the other ranks.
- Initial conversion mode is official MP=4 conversion with `--expert-dtype fp8`, because the V4 instruct checkpoint contains FP4 expert weights and the official V4 converter does not include a full BF16 expansion path.
- Important feasibility note: fully expanding `DeepSeek-V4-Flash` to BF16 would be approximately hundreds of GiB larger than the mixed FP4/FP8 checkpoint and is expected to exceed 4x80 GB A100 VRAM. This needs to be verified before attempting an expensive full conversion.
- Image build verified via `modal run deepseek_v4_flash_official_modal.py --action remote`.
- Build fixes needed: install Torch nightly before V4 dependencies, build `fast_hadamard_transform` from its GitHub source with `CC=gcc CXX=g++`, and include `build-essential`, `ninja-build`, and `wheel`.
- Remote model metadata verified: HF SHA `6976c7ff1b30a1b2cb7805021b8ba4684041f136`, 46 safetensor shards, approximately 148.66 GiB of safetensor data.
- First download attempts failed before transfer because Modal rejected mounting the volume on `/models` and `/mnt/models`; the mount path was moved to a unique `/dsv4-volume` path.
- `/dsv4-volume` also failed while `HF_HOME` was set before the build-time source snapshot; the HF cache env is now applied after the build snapshot so the runtime volume mount path remains empty in the image.
- Download completed into the Modal volume: `/dsv4-volume/hf` has 140 files and 148.67 GiB.
- Tensor inspection confirmed the repo uses official-inference-style tensor names, FP8 scale tensors, and packed FP4 routed expert weights.
- Official MP=4 conversion with `--expert-dtype fp8` completed successfully; `/dsv4-volume/mp4` has four `model*-mp4.safetensors` shards of about 70.04 GiB each, 280.18 GiB total.
- Server fixes added after first endpoint attempts: load `tokenizer.json` directly with `PreTrainedTokenizerFast` so Transformers does not parse the inference `config.json` as a model config, and set the CUDA default device inside request generation.
- Runtime image changed to Python 3.11 and pinned `apache-tvm-ffi<0.1.8`; this fixes TileLang's `_NestedLoopCheckVisitor` / `_inst` compiler-pass failure.
- A100 endpoint attempt now reaches the first forward pass, but fails in CUTLASS FP8 MMA with `Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN without CUTE_ARCH_MMA_F32_SM89_ENABLED`, followed by CUDA XID 43. This confirms the official FP8 path is not usable on 4x A100.
- A brief H100 test was started and then explicitly stopped after the deployment constraint was restated: target must remain strictly 4x A100.
- Added an A100 compatibility patch in `dsv4_official_server.py`: it monkey-patches official inference at runtime to replace FP8 GEMM dispatch with BF16 `F.linear` after per-block FP8 weight dequantization, and turns activation FP8/FP4 quantization calls into no-ops.
- A100 compatibility endpoint test succeeded on `A100-80GB:4`: `modal run deepseek_v4_flash_official_modal.py --action test --prompt 'What is 17*19? Answer briefly.'` returned HTTP 200 with `17 * 19 = 323.`. Generation took about 65 seconds for this short prompt, so this is a functional but slow compatibility path.
- Post-run cleanup checked and stopped the remaining Modal container; `modal container list --json` returned `[]`.
