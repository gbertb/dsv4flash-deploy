# DeepSeek V4 Flash Base INT4 Modal Changelog

## 2026-05-21

- Added a dedicated Modal path for `EnsueAI/DeepSeek-V4-Flash-Base-INT4` on
  exactly `A100-80GB:4`.
- Verified from Hugging Face metadata that the INT4 repo contains four TP=4
  safetensors shards plus tokenizer and quantization metadata, with no
  `config.json` or `config-flash-base.json` present despite the README
  mentioning config files.
- Added `dsv4_int4_server.py`, which runs the official DeepSeek V4 inference
  code but replaces packed `uint8` INT4 linears with a small dequantizing
  PyTorch module before loading each rank shard.
- The loader keeps the existing A100 compatibility behavior for FP8 tensors by
  dispatching FP8 linear layers through BF16 `F.linear` after scale dequant.
- Modal image build and remote metadata inspection succeeded with:
  `rtk modal run deepseek_v4_flash_int4_modal.py --action remote`.
- Remote metadata: SHA `76de2892eb2b27b59474fe5b7da823ae4098f22f`, four model
  shards of about 46.61 GB each, and about 173.65 GiB total including metadata
  and tokenizer files.
- Post-run cleanup checked: `modal container list --json` returned `[]`; the
  ephemeral `deepseek-v4-flash-base-int4` app was stopped.
- Next verification step: download the 4 INT4 shards into the Modal Volume and
  run an endpoint smoke test.
- Download completed with
  `rtk modal run deepseek_v4_flash_int4_modal.py --action download`; Hugging
  Face fetched all 10 files in about 16 minutes, then Modal stopped the
  ephemeral app after the local entrypoint completed.
- Post-download cleanup checked: `modal container list --json` returned `[]`;
  the download app `deepseek-v4-flash-base-int4` was stopped with 0 tasks.
- User noted the Modal plan has a 10 GPU limit; continue checking container/app
  state after every Modal run before starting the next 4x A100 test.
- Volume inspection succeeded with
  `rtk modal run deepseek_v4_flash_int4_modal.py --action inspect`; the Modal
  Volume contains 22 files, 173.65 GiB total, including all four model shards at
  43.410 GiB each.
- Post-inspection cleanup checked: `modal container list --json` returned `[]`;
  the inspection app was stopped with 0 tasks.
- First 4x A100 endpoint smoke attempt started successfully, loaded the model
  with 0 missing parameters, replaced 8,509 packed INT4 linears on rank 0, and
  started Uvicorn.
- First generation failed with HTTP 500 from a custom INT4 linear dtype mismatch:
  `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float !=
  c10::BFloat16` at `PackedInt4Linear.forward`.
- Post-failure cleanup checked: `modal container list --json` returned `[]`; the
  failed smoke-test app was stopped.
- Patched `PackedInt4Linear.forward` to cast the activation to the dequantized
  weight dtype for `F.linear`, then cast the result back to the input dtype.
- Patched 4x A100 endpoint smoke test succeeded:
  `rtk modal run deepseek_v4_flash_int4_modal.py --action test --prompt 'What is
  17*19? Answer briefly.'` returned HTTP 200.
- Smoke-test response included the correct answer `323`, though the raw decoded
  content also contained extra template-looking text after the answer:
  ``323 ... <details><summary>llama3``.
- Request timing reported by Modal: POST duration about 147.4 seconds, execution
  about 43.0 seconds.
- Post-success cleanup checked: `modal container list --json` returned `[]`; the
  successful smoke-test app was stopped. No DeepSeek containers were left
  running.
- Reduced the persistent web endpoint scaledown window from 30 minutes to 60
  seconds by default, configurable with `DSV4_INT4_SCALEDOWN_WINDOW`, to avoid
  holding four A100 GPUs longer than needed after a request.
- Persistent deployment succeeded with
  `rtk modal deploy deepseek_v4_flash_int4_modal.py`.
- Deployed OpenAI-compatible endpoint:
  `https://<modal-workspace>--deepseek-v4-flash-base-int4-serve.modal.run/v1`.
- Post-deploy cleanup checked: `modal container list --json` returned `[]`; the
  deployed app `deepseek-v4-flash-base-int4` was listed with `Tasks: 0`, so it
  was not holding any GPUs immediately after deployment.
- Production endpoint smoke test succeeded with a direct HTTP request to the
  deployed `/v1/chat/completions` endpoint; it returned HTTP 200 and included
  the correct answer `323`.
- The production response still included extra decoded text after the answer,
  consistent with using the Base checkpoint through a chat-shaped endpoint.
- The production container did not disappear after the expected scaledown wait,
  so it was stopped explicitly with `modal container stop
  ta-01KS6MMRBHS6JZS72HSCMC2M6J`.
- Final cleanup check after stopping the container: `modal container list --json`
  returned `[]`; the deployed app was listed with `Tasks: 0`.
- Warm-server timing test sent two back-to-back production endpoint requests.
  The first request, from a pending/cold-ish container state, returned HTTP 200
  in 57.208 seconds. The second immediately-following warm request returned
  HTTP 200 in 4.190 seconds. Both included the correct answer `323`.
- After the warm timing test, the active production container
  `ta-01KS6NJJY82GAESXNDY870NYP6` was stopped explicitly. Final cleanup check:
  `modal container list --json` returned `[]`; the deployed app was listed with
  `Tasks: 0`.
- Updated `DSV4_INT4_MODAL.md` with the deployed endpoint, config layout, runtime
  patch strategy, self-test commands, timing results, cleanup guidance, tunables,
  and fallback order.
- During docs work, a pending production container reappeared and kept requeueing
  after `modal container stop`; the deployed app was stopped with
  `rtk modal app stop deepseek-v4-flash-base-int4` to avoid starting another
  4x A100 allocation. Cleanup after app stop showed no running containers.

## 2026-05-21 Thinking-mode test

- Patched `dsv4_int4_server.py` to accept `thinking_mode` and
  `reasoning_effort`, use the official `encoding_dsv4.encode_messages`, and
  parse completions with `parse_message_from_completion_text`.
- Redeployed `deepseek-v4-flash-base-int4`; post-deploy check showed no running
  containers and the app at `Tasks: 0`.
- Warm-up request: prompt `What is 2+2? Answer briefly.`, `thinking_mode=chat`,
  `max_tokens=8`, HTTP 200, elapsed 151.910 seconds. Response was incomplete:
  `2 + 2 = `.
- Target prompt: `What makes the sky blue, give me a short summary`,
  `max_tokens=1000`.
- Thinking on (`thinking_mode=thinking`): HTTP 200, elapsed 380.002 seconds. The
  answer started with explicit reasoning, gave a correct Rayleigh-scattering
  summary, then continued into unrelated template/training text because it hit
  the 1000-token cap.
- Thinking off (`thinking_mode=chat`, `reasoning_effort=none`): HTTP 200,
  elapsed 377.889 seconds. The answer immediately gave a correct concise
  Rayleigh-scattering summary, then also continued into unrelated template text
  because it hit the 1000-token cap.
- Modal logs reported server-side execution times of about 379.6 seconds and
  377.5 seconds for the two target requests.
- After the test, the active production container
  `ta-01KS6PMY721HRD6CT3EQKQTE8X` was stopped explicitly. Final cleanup check:
  `modal container list --json` returned `[]`; the deployed app was listed with
  `Tasks: 0`.

## Fallback order update

- If the EnsueAI INT4 checkpoint does not run, try
  `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` next, before any GGUF fallback.
- Initial Intel model inspection: it is a safetensors AutoRound W4A16 checkpoint
  with 46 shards, about 142.46 GiB total, and includes `inference/model.py`,
  `inference/convert_w4a16.py`, and `inference/config_w4a16.json`.
- The Intel README says vLLM and SGLang are not currently supported, so the
  fallback should use Intel's included inference/conversion path rather than
  vLLM.
