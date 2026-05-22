# DS4 Modal Deployment

This deploys `antirez/ds4` with the project's DeepSeek V4 Flash GGUFs. It does
not use vLLM.

The original q4-imatrix target was downloaded successfully, but it did not run
on Modal's A100/H100 CUDA path. The script now defaults to q2-imatrix, which is
the DS4 format expected by the CPU path and the smaller model documented for
96/128 GB machines.

## Commands

Download the default q2-imatrix GGUF into a Modal Volume:

```bash
rtk modal run ds4_modal.py --action download
```

To reproduce the q4-imatrix attempt explicitly:

```bash
DS4_MODEL_KIND=q4-imatrix rtk modal run ds4_modal.py --action inspect
```

Check the downloaded model:

```bash
rtk modal run ds4_modal.py --action inspect
```

Run a CLI smoke test. This defaults to DS4's CPU backend because q4-imatrix
failed CUDA prefill on Modal A100 and H100 after registering the 153 GiB
host-mapped model:

```bash
rtk modal run ds4_modal.py --action cli-smoke
```

Start a dev web endpoint and print the `/v1` base URL:

```bash
rtk modal run ds4_modal.py
```

Deploy a persistent Modal app:

```bash
rtk modal deploy ds4_modal.py
```

Run an OpenAI-compatible chat test against the dev endpoint:

```bash
rtk modal run ds4_modal.py --action test --prompt "Explain Redis streams in one paragraph."
```

## Defaults

- Model: `q2-imatrix`
- Runtime: DS4 server, `./ds4-server`
- Backend: `cpu`
- GPU: none by default unless `DS4_BACKEND=cuda`
- CUDA GPU request: `A100-80GB:4`
- CUDA arch: `sm_80`
- CPU RAM request: `327680` MiB
- Context: `32768`
- Default output tokens: `256` on CPU, `4096` on CUDA
- KV cache volume: `ds4-kv-cache`
- GGUF volume: `ds4-<model-kind>-gguf`

Override before running if needed:

```bash
DS4_MEMORY_MIB=393216 rtk modal deploy ds4_modal.py
```

CUDA is still available for experiments:

```bash
DS4_MODEL_KIND=q4-imatrix DS4_BACKEND=cuda rtk modal run ds4_modal.py --action debug
```

Force H100 instead:

```bash
DS4_MODEL_KIND=q4-imatrix DS4_BACKEND=cuda DS4_GPU=H100 DS4_CUDA_ARCH=sm_90 rtk modal run ds4_modal.py --action debug
```

In testing, q4-imatrix failed CUDA prefill on both Modal A100 (`sm_80`) and H100
(`sm_90`) with:

```text
ds4: CUDA registered 153.33 GiB model mapping for device access
ds4: prompt processing failed: cuda prefill failed
```

The q4-imatrix CPU attempt also failed with:

```text
ds4: expected IQ2_XXS expert tensors
```

That points to q4 expert tensors being unsupported by DS4's CPU path. DS4 CUDA
is not a vLLM tensor-parallel runtime. Requesting `A100-80GB:4` gives Modal four
GPUs, but DS4 may still use only one visible CUDA device unless upstream adds
tensor/model parallel behavior.
