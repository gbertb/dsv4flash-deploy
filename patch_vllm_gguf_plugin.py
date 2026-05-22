import re
from pathlib import Path


SITE_PACKAGES = Path("/usr/local/lib/python3.12/dist-packages")
PATCH_VERSION = "2026-05-20.5"

print(f"Applying DeepSeek V4 GGUF Modal patch {PATCH_VERSION}", flush=True)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise RuntimeError(f"Patch target not found in {path}: {old!r}")
    path.write_text(text.replace(old, new, 1))


def make_custom_op_registration_idempotent(
    path: Path,
    op_name: str,
    variable_name: str,
) -> None:
    text = path.read_text()

    marker = f"    {variable_name} = torch.ops.vllm.{op_name}"
    idx = text.find(marker)
    if idx == -1:
        raise RuntimeError(f"Custom op assignment not found in {path}: {marker!r}")

    tail = text[idx + len(marker) :]
    attr_except = "\nexcept AttributeError as error:\n    raise error"
    attr_idx = tail.find(attr_except)
    if attr_idx == -1:
        raise RuntimeError(f"Custom op exception block not found in {path}: {op_name}")
    if "except RuntimeError as error:" in tail[:attr_idx]:
        return

    insert = f"""
except RuntimeError as error:
    if "same name and overload name multiple times" not in str(error):
        raise
    {variable_name} = torch.ops.vllm.{op_name}"""
    path.write_text(
        text[: idx + len(marker) + attr_idx]
        + insert
        + text[idx + len(marker) + attr_idx :]
    )


def make_plugin_custom_op_registration_idempotent(
    path: Path,
    op_name: str,
    variable_name: str,
) -> None:
    replace_once(
        path,
        f"""    {variable_name} = torch.ops.vllm.{op_name}
except AttributeError as error:
    raise error""",
        f"""    {variable_name} = torch.ops.vllm.{op_name}
except RuntimeError as error:
    if "same name and overload name multiple times" not in str(error):
        raise
    {variable_name} = torch.ops.vllm.{op_name}
except AttributeError as error:
    raise error""",
    )


def skip_scalar_tid2eid_loads(path: Path) -> None:
    text = path.read_text()
    if 'name.endswith(".ffn.gate.tid2eid")' in text:
        return

    pattern = re.compile(r"^([ \t]+)param = params_dict\[name\]\n", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        print(f"WARNING: DeepSeek V4 params_dict[name] load site not found in {path}", flush=True)
        return

    def add_guard(match: re.Match[str]) -> str:
        indent = match.group(1)
        return (
            match.group(0)
            + f'{indent}if (\n'
            + f'{indent}    name.endswith(".ffn.gate.tid2eid")\n'
            + f"{indent}    and loaded_weight.ndim == 0\n"
            + f"{indent}    and param.ndim > 0\n"
            + f"{indent}):\n"
            + f"{indent}    continue\n"
        )

    path.write_text(pattern.sub(add_guard, text))


plugin_py = SITE_PACKAGES / "vllm_gguf_plugin/plugin.py"
replace_once(
    plugin_py,
    """if (
        "gguf" not in QUANTIZATION_METHODS
        or get_quantization_config("gguf") is not GGUFConfig
    ):
        register_quantization_config("gguf")(GGUFConfig)""",
    """register_quantization_config("gguf")(GGUFConfig)""",
)

quantization_init_py = (
    SITE_PACKAGES / "vllm/model_executor/layers/quantization/__init__.py"
)
replace_once(
    quantization_init_py,
    """    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    # lazy import to avoid triggering `torch.compile` too early""",
    """    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")

    if quantization in _CUSTOMIZED_METHOD_TO_QUANT_CONFIG:
        return _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization]

    # lazy import to avoid triggering `torch.compile` too early""",
)

deepseek_v4_config_py = SITE_PACKAGES / "vllm/transformers_utils/configs/deepseek_v4.py"
deepseek_v4_config_py.write_text(
    """# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    model_type = "deepseek_v4"

    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters
        super().__init__(**kwargs)
"""
)

deepseek_v4_model_py = SITE_PACKAGES / "vllm/model_executor/models/deepseek_v4.py"
replace_once(
    deepseek_v4_model_py,
    """        orig_to_new_substr={
            ".attn.compressor.": ".attn.mla_attn.compressor.",
            ".shared_experts.w2": ".shared_experts.down_proj",
        },""",
    """        orig_to_new_substr={
            ".self_attn.compressor.indexer.position_bias": (
                ".attn.indexer.compressor.position_bias"
            ),
            ".self_attn.compressor.indexer.kv_proj.": (
                ".attn.indexer.compressor.wkv."
            ),
            ".self_attn.compressor.indexer.gate_proj.": (
                ".attn.indexer.compressor.wgate."
            ),
            ".self_attn.compressor.indexer.kv_norm.": (
                ".attn.indexer.compressor.k_norm."
            ),
            ".self_attn.compressor.indexer.q_b_proj.": ".attn.indexer.wq_b.",
            ".self_attn.compressor.indexer.weights_proj.": (
                ".attn.indexer.weights_proj."
            ),
            ".self_attn.compressor.position_bias": (
                ".attn.mla_attn.compressor.position_bias"
            ),
            ".self_attn.compressor.kv_proj.": ".attn.mla_attn.compressor.wkv.",
            ".self_attn.compressor.gate_proj.": ".attn.mla_attn.compressor.wgate.",
            ".self_attn.compressor.kv_norm.": ".attn.mla_attn.compressor.k_norm.",
            ".self_attn.sinks": ".attn.attn_sink",
            ".input_layernorm.": ".attn_norm.",
            ".post_attention_layernorm.": ".ffn_norm.",
            ".self_attn.q_a_proj.": ".attn.wq_a.",
            ".self_attn.q_b_proj.": ".attn.wq_b.",
            ".self_attn.q_a_norm.": ".attn.q_norm.",
            ".self_attn.kv_proj.": ".attn.wkv.",
            ".self_attn.kv_norm.": ".attn.kv_norm.",
            ".self_attn.o_a_proj.": ".attn.wo_a.",
            ".self_attn.o_b_proj.": ".attn.wo_b.",
            ".mlp.gate.weight": ".ffn.gate.weight",
            ".mlp.gate.bias": ".ffn.gate.e_score_correction_bias",
            ".mlp.gate.tid2eid": ".ffn.gate.tid2eid",
            ".mlp.": ".ffn.",
            ".attn_hc.fn": ".hc_attn_fn",
            ".attn_hc.base": ".hc_attn_base",
            ".attn_hc.scale": ".hc_attn_scale",
            ".ffn_hc.fn": ".hc_ffn_fn",
            ".ffn_hc.base": ".hc_ffn_base",
            ".ffn_hc.scale": ".hc_ffn_scale",
            "hc_head.hc_fn": "hc_head_fn",
            "hc_head.hc_base": "hc_head_base",
            "hc_head.hc_scale": "hc_head_scale",
            ".attn.compressor.": ".attn.mla_attn.compressor.",
            ".shared_experts.w2": ".shared_experts.down_proj",
        },""",
)
replace_once(
    deepseek_v4_model_py,
    """                        if success:
                            name = name_mapped
                            break
                    loaded_params.add(name_mapped)""",
    """                        if success:
                            loaded_params.add(name_mapped)
                            break""",
)
skip_scalar_tid2eid_loads(deepseek_v4_model_py)

config_parser_py = SITE_PACKAGES / "vllm_gguf_plugin/config_parser.py"
replace_once(
    config_parser_py,
    """        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )""",
    """        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )

        if getattr(config, "model_type", None) == "deepseek_v4":
            quantization_config = getattr(config, "quantization_config", {}) or {}
            if not isinstance(quantization_config, dict):
                quantization_config = {}
            quantization_config = {
                "scale_fmt": quantization_config.get("scale_fmt", "ue8m0")
            }
            config_dict["quantization_config"] = quantization_config
            config.quantization_config = quantization_config""",
)
replace_once(
    config_parser_py,
    """        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]""",
    """        if config.model_type == "deepseek_v4":
            model_type = "DeepseekV4ForCausalLM"
        else:
            if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
                raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
            model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]""",
)

default_adapter_py = SITE_PACKAGES / "vllm_gguf_plugin/weights_adapter/default.py"
gguf_quant_config_py = SITE_PACKAGES / "vllm_gguf_plugin/quantization/config.py"
replace_once(
    gguf_quant_config_py,
    """    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None
    ) -> "QuantizationMethods | None":
        del hf_quant_cfg
        if user_quant == "gguf":
            return "gguf"
        return None""",
    """    def override_quantization_method(
        cls,
        hf_quant_cfg: dict[str, Any],
        user_quant: str | None,
        **kwargs: Any,
    ) -> "QuantizationMethods | None":
        del hf_quant_cfg, kwargs
        if user_quant == "gguf":
            return "gguf"
        return None""",
)
replace_once(
    default_adapter_py,
    """from transformers import AutoModelForCausalLM""",
    """from transformers import AutoModelForCausalLM

try:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4ForCausalLM,
    )
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config

    AutoModelForCausalLM.register(
        DeepseekV4Config,
        DeepseekV4ForCausalLM,
        exist_ok=True,
    )
except Exception:
    pass""",
)
replace_once(
    default_adapter_py,
    """except Exception:
    pass""",
    """except Exception:
    pass


def _get_dummy_transformers_config(config):
    if getattr(config, "model_type", None) != "deepseek_v4":
        return config

    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config as HFDeepseekV4Config,
    )

    if hasattr(config, "to_dict"):
        data = config.to_dict()
    else:
        data = dict(getattr(config, "__dict__", {}))

    data.setdefault("architectures", ["DeepseekV4ForCausalLM"])
    data.setdefault("max_position_embeddings", 1048576)
    data.setdefault("qk_rope_head_dim", 64)
    data.setdefault("compress_rope_theta", 160000)
    data.setdefault("compress_ratios", [0, 0] + [4, 128] * 20 + [4, 0])
    data.setdefault("num_hash_layers", 3)
    data.setdefault("rope_theta", 10000)
    data.setdefault("rope_scaling", {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 65536,
        "type": "yarn",
    })
    return HFDeepseekV4Config(**data)""",
)
replace_once(
    default_adapter_py,
    """            dummy_model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=model_config.trust_remote_code
            )""",
    """            dummy_config = _get_dummy_transformers_config(config)
            dummy_model = AutoModelForCausalLM.from_config(
                dummy_config, trust_remote_code=model_config.trust_remote_code
            )""",
)
replace_once(
    default_adapter_py,
    """        if model_type in ("deepseek_v3", "deepseek_v2"):""",
    """        if model_type in ("deepseek_v4", "deepseek_v3", "deepseek_v2"):""",
)
replace_once(
    default_adapter_py,
    """        if model_type in ("deepseek_v4", "deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            for idx in range(config.num_hidden_layers):""",
    """        if model_type in ("deepseek_v4", "deepseek_v3", "deepseek_v2"):
            is_deepseek_v4 = model_type == "deepseek_v4"
            model_type = "deepseek2"
            if is_deepseek_v4:
                gguf_to_hf_name_map.update({
                    "hc_head_fn": "model.hc_head.hc_fn",
                    "hc_head_base": "model.hc_head.hc_base",
                    "hc_head_scale": "model.hc_head.hc_scale",
                })
            for idx in range(config.num_hidden_layers):
                if is_deepseek_v4:
                    gguf_to_hf_name_map.update({
                        f"blk.{idx}.attn_sinks": (
                            f"model.layers.{idx}.self_attn.sinks"
                        ),
                        f"blk.{idx}.attn_q_a_norm.weight": (
                            f"model.layers.{idx}.self_attn.q_a_norm.weight"
                        ),
                        f"blk.{idx}.attn_kv_latent.weight": (
                            f"model.layers.{idx}.self_attn.kv_proj.weight"
                        ),
                        f"blk.{idx}.attn_kv_a_norm.weight": (
                            f"model.layers.{idx}.self_attn.kv_norm.weight"
                        ),
                        f"blk.{idx}.attn_output_a.weight": (
                            f"model.layers.{idx}.self_attn.o_a_proj.weight"
                        ),
                        f"blk.{idx}.attn_output_b.weight": (
                            f"model.layers.{idx}.self_attn.o_b_proj.weight"
                        ),
                        f"blk.{idx}.attn_compress_ape": (
                            f"model.layers.{idx}.self_attn.compressor.position_bias"
                        ),
                        f"blk.{idx}.attn_compress_kv.weight": (
                            f"model.layers.{idx}.self_attn.compressor.kv_proj.weight"
                        ),
                        f"blk.{idx}.attn_compress_gate.weight": (
                            f"model.layers.{idx}.self_attn.compressor.gate_proj.weight"
                        ),
                        f"blk.{idx}.attn_compress_norm.weight": (
                            f"model.layers.{idx}.self_attn.compressor.kv_norm.weight"
                        ),
                        f"blk.{idx}.indexer.compress_ape": (
                            f"model.layers.{idx}.self_attn.compressor.indexer.position_bias"
                        ),
                        f"blk.{idx}.indexer.compress_kv.weight": (
                            f"model.layers.{idx}.self_attn.compressor.indexer.kv_proj.weight"
                        ),
                        f"blk.{idx}.indexer.compress_gate.weight": (
                            f"model.layers.{idx}.self_attn.compressor.indexer.gate_proj.weight"
                        ),
                        f"blk.{idx}.indexer.compress_norm.weight": (
                            f"model.layers.{idx}.self_attn.compressor.indexer.kv_norm.weight"
                        ),
                        f"blk.{idx}.indexer.attn_q_b.weight": (
                            f"model.layers.{idx}.self_attn.compressor.indexer.q_b_proj.weight"
                        ),
                        f"blk.{idx}.indexer.proj.weight": (
                            f"model.layers.{idx}.self_attn.compressor.indexer.weights_proj.weight"
                        ),
                        f"blk.{idx}.ffn_gate_tid2eid": (
                            f"model.layers.{idx}.mlp.gate.tid2eid"
                        ),
                        f"blk.{idx}.hc_attn_fn": (
                            f"model.layers.{idx}.attn_hc.fn"
                        ),
                        f"blk.{idx}.hc_attn_base": (
                            f"model.layers.{idx}.attn_hc.base"
                        ),
                        f"blk.{idx}.hc_attn_scale": (
                            f"model.layers.{idx}.attn_hc.scale"
                        ),
                        f"blk.{idx}.hc_ffn_fn": (
                            f"model.layers.{idx}.ffn_hc.fn"
                        ),
                        f"blk.{idx}.hc_ffn_base": (
                            f"model.layers.{idx}.ffn_hc.base"
                        ),
                        f"blk.{idx}.hc_ffn_scale": (
                            f"model.layers.{idx}.ffn_hc.scale"
                        ),
                    })""",
)

make_custom_op_registration_idempotent(
    SITE_PACKAGES / "vllm/model_executor/layers/quantization/gguf.py",
    "_fused_mul_mat_gguf",
    "fused_mul_mat_gguf",
)
make_custom_op_registration_idempotent(
    SITE_PACKAGES / "vllm/model_executor/layers/quantization/gguf.py",
    "_fused_moe_gguf",
    "fused_moe_gguf",
)
make_custom_op_registration_idempotent(
    SITE_PACKAGES / "vllm/model_executor/layers/quantization/gguf.py",
    "_apply_gguf_embedding",
    "apply_gguf_embedding",
)
make_custom_op_registration_idempotent(
    SITE_PACKAGES / "vllm_gguf_plugin/quantization/linear.py",
    "_fused_mul_mat_gguf",
    "fused_mul_mat_gguf",
)
make_plugin_custom_op_registration_idempotent(
    SITE_PACKAGES / "vllm_gguf_plugin/quantization/fused_moe.py",
    "_fused_moe_gguf",
    "fused_moe_gguf",
)
make_plugin_custom_op_registration_idempotent(
    SITE_PACKAGES / "vllm_gguf_plugin/quantization/vocal_embeds.py",
    "_apply_gguf_embedding",
    "apply_gguf_embedding",
)
