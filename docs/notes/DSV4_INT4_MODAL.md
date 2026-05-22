# DeepSeek V4 Flash Base INT4 Modal Deployment

This path targets `EnsueAI/DeepSeek-V4-Flash-Base-INT4` on exactly
`A100-80GB:4`.

Status: deployed and smoke-tested on Modal.

OpenAI-compatible base URL:

```text
https://<modal-workspace>--deepseek-v4-flash-base-int4-serve.modal.run/v1
```

The checkpoint is TP=4 sharded safetensors, not GGUF. The Hugging Face repo does
not currently include the `config.json` / `config-flash-base.json` files
referenced by its model card, so this deployment downloads the official
`deepseek-ai/DeepSeek-V4-Flash` inference code and uses its
`inference/config.json`.

At runtime, `dsv4_int4_server.py` forces `expert_dtype` to `fp8` for model
construction, replaces the shipped packed INT4 linears before loading each rank
shard, then loads the EnsueAI TP=4 safetensors directly.

## Current Results

- Remote model SHA: `76de2892eb2b27b59474fe5b7da823ae4098f22f`
- Modal Volume contents after download: 22 files, 173.65 GiB total
- Model shards: 4 files, 43.410 GiB each in the mounted volume
- GPU shape tested: `A100-80GB:4`
- Loader result: 0 missing parameters; 8,509 packed INT4 linears replaced on rank
  0, matching the model metadata's quantized-per-rank count
- Dev endpoint smoke test: HTTP 200, answer contained `323`
- Production endpoint smoke test: HTTP 200, answer contained `323`
- Warm production timing test:
  - first request from a pending/cold-ish state: 57.208 seconds
  - second back-to-back warm request: 4.190 seconds

The response currently includes extra decoded text after the answer. This is
expected for now because the model is the Base checkpoint served through a simple
chat-shaped endpoint.

## Commands

Inspect remote metadata:

```bash
rtk modal run deepseek_v4_flash_int4_modal.py --action remote
```

Download the INT4 checkpoint into a Modal Volume:

```bash
rtk modal run deepseek_v4_flash_int4_modal.py --action download
```

Inspect the downloaded volume:

```bash
rtk modal run deepseek_v4_flash_int4_modal.py --action inspect
```

Inspect representative tensors:

```bash
rtk modal run deepseek_v4_flash_int4_modal.py --action tensors
```

Run an endpoint smoke test on exactly 4x A100:

```bash
rtk modal run deepseek_v4_flash_int4_modal.py --action test --prompt "What is 17*19? Answer briefly."
```

Deploy a persistent Modal app only after the smoke test succeeds:

```bash
rtk modal deploy deepseek_v4_flash_int4_modal.py
```

## Test The Deployed Endpoint

Use curl:

```bash
curl -sS https://<modal-workspace>--deepseek-v4-flash-base-int4-serve.modal.run/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "deepseek-v4-flash-base-int4",
    "messages": [
      {"role": "user", "content": "What is 17*19? Answer briefly."}
    ],
    "temperature": 0,
    "max_tokens": 8,
    "stream": false
  }'
```

Or use Python:

```python
import json
import urllib.request

url = "https://<modal-workspace>--deepseek-v4-flash-base-int4-serve.modal.run/v1/chat/completions"
payload = {
    "model": "deepseek-v4-flash-base-int4",
    "messages": [{"role": "user", "content": "What is 17*19? Answer briefly."}],
    "temperature": 0.0,
    "max_tokens": 8,
    "stream": False,
}

req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=1200) as resp:
    print(resp.status)
    print(resp.read().decode("utf-8"))
```

Run two requests back-to-back to separate cold-start behavior from warm-request
latency. In the current setup, the second request was 4.190 seconds for the
`17*19` prompt with `max_tokens=8`.

## Cleanup

Always check Modal state after tests. This app uses 4 A100s while a container is
running.

```bash
rtk modal container list --json
rtk modal app list --json
```

Stop a lingering INT4 container:

```bash
rtk modal container stop <container-id>
```

If a pending request keeps requeueing containers and you do not need the endpoint
available, stop the deployed app:

```bash
rtk modal app stop deepseek-v4-flash-base-int4
```

Redeploy afterward with:

```bash
rtk modal deploy deepseek_v4_flash_int4_modal.py
```

## Configuration

- GPU request: `A100-80GB:4`
- No alternate GPU configuration is exposed in this script.
- Tensor parallelism: `torchrun --nproc-per-node 4`
- Model repo: `EnsueAI/DeepSeek-V4-Flash-Base-INT4`
- Inference repo: `deepseek-ai/DeepSeek-V4-Flash`
- Modal app name: `deepseek-v4-flash-base-int4`
- Served model name: `deepseek-v4-flash-base-int4`
- Modal Volume: `deepseek-v4-flash-base-int4`
- Checkpoint mount path: `/dsv4-int4-volume/hf`
- Official inference source path: `/opt/deepseek-v4-flash`
- Runtime config path: `/opt/deepseek-v4-flash/inference/config.json`
- Runtime patch: force `expert_dtype="fp8"` for construction, then replace
  packed `uint8` INT4 modules before `load_model`
- Web endpoint scaledown defaults to 60 seconds after the last request so it
  releases the 4 A100s quickly. Override with `DSV4_INT4_SCALEDOWN_WINDOW` if
  needed.
- Runtime mode is FP8RT-style compatibility: packed INT4 linears are dequantized
  to BF16 inside a custom PyTorch module; native Marlin INT4 kernels are not used.
- A100 compatibility is enabled by default via `DSV4_A100_COMPAT=1`; FP8 linear
  layers are run through BF16 `F.linear` after scale dequantization because the
  official FP8 kernel path is not usable on A100.

## Tunables

These environment variables are intentionally limited to operational behavior;
they do not expose other GPU shapes.

- `DSV4_INT4_STARTUP_TIMEOUT`: Modal web startup timeout, default 14,400 seconds
- `DSV4_INT4_TIMEOUT`: function timeout, default 43,200 seconds
- `DSV4_INT4_EPHEMERAL_DISK_MIB`: ephemeral disk, default 1,048,576 MiB
- `DSV4_INT4_SCALEDOWN_WINDOW`: web scaledown window, default 60 seconds
- `DSV4_INT4_MODEL_VOLUME`: Modal Volume name, default
  `deepseek-v4-flash-base-int4`
- `DSV4_INT4_HF_SECRET`: optional Modal Secret name for a Hugging Face token
- `DSV4_A100_COMPAT`: set to `0` only to disable the A100 compatibility patch;
  the tested deployment leaves it enabled

## Fallback Order

The EnsueAI INT4 path works, so fallback was not needed. If this path regresses,
the next safetensors fallback is `Intel/DeepSeek-V4-Flash-W4A16-AutoRound` on
the same `A100-80GB:4` shape before trying any GGUF quantization.
