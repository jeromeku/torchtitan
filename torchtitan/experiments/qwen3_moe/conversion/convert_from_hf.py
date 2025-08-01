import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.qwen3_moe.modeling_qwen3_moe as hf_qwen3_moe
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as HFQwen3MoeConfig

from torchtitan.testing.utils import _copy_module, ModuleMapping, check_tensor_attributes
from torchtitan.experiments.qwen3_moe.model import Qwen3MoeSparseMoeBlock, Qwen3MoeRotaryEmbedding, Qwen3MoeAttention, Qwen3MoeDecoderLayer, Qwen3MoeRMSNorm, Qwen3MoeModel

HFQwen3MoeModelForCausalLM = hf_qwen3_moe.Qwen3MoeForCausalLM
HFQwen3MoeModel = hf_qwen3_moe.Qwen3MoeModel
HFQwen3MoeDecoderLayer = hf_qwen3_moe.Qwen3MoeDecoderLayer
HFQwen3MoeAttention = hf_qwen3_moe.Qwen3MoeAttention
HFQwen3RMSNorm = hf_qwen3_moe.Qwen3MoeRMSNorm
HFQwen3RoPE = hf_qwen3_moe.Qwen3MoeRotaryEmbedding
HFQwen3MoeSparseMoeBlock = hf_qwen3_moe.Qwen3MoeSparseMoeBlock

# Mappings for copying from hf -> tt
# Paths relative to model root
_QWEN3_EMBED_MAPPING = {
    "embeddings": ModuleMapping(
        src_key="model.embed_tokens", dst_key="tok_embeddings", src_weights=["weight"]
    ),
    "unembed": ModuleMapping(src_key="lm_head", dst_key="lm_head", src_weights=["weight"]),
    # "rope": ModuleMapping(
    #     src_key="model.rotary_emb", dst_key="layers.0.attn.pos_embeddings", src_weights=None
    # ),
}

# Paths relative to model.layers.*
_QWEN3_ATTN_MAPPING = {
    "pre_attn_norm": ModuleMapping(
        src_key="input_layernorm", dst_key="attn_norm", src_weights=["weight"], dst_weights=["weight"]
    ),
    "post_attn_norm": ModuleMapping(
        src_key="post_attention_layernorm",
        dst_key="mlp_norm",
        src_weights=["weight"],
        dst_weights=["weight"],
    ),
    "attn_proj": ModuleMapping(
        src_key="self_attn",
        dst_key="self_attn",
        src_submod_keys=["q_proj", "k_proj", "v_proj", "o_proj"],
        src_weights=["weight"],
        dst_submod_keys=["q_proj", "k_proj", "v_proj", "o_proj"],
        dst_weights=["weight"],
    ),
    "attn_norm": ModuleMapping(
        src_key="self_attn",
        dst_key="self_attn",
        src_submod_keys=["q_norm", "k_norm"],
        src_weights=["weight"],
        dst_submod_keys=["q_norm", "k_norm"],
        dst_weights=["weight"],
    ),
}

_QWEN3_EXPERTS_MAPPING = {
    "router": ModuleMapping(src_key="mlp.gate", dst_key="mlp.router.gate", src_weights=["weight"]),
    # "experts": ModuleMapping(src_key="post_attention_layernorm", dst_key="mlp_norm", src_weights=["weight"], dst_weights=["scale"]),
}

@torch.no_grad()
def _copy_expert_weights(src_experts: torch.nn.ModuleList, dst_experts: torch.nn.Module):
    """
    Copy individual expert weights into contiguous gate, up, and down proj weights
    Src (Huggingface Qwen3SparseMoeBlock)
        [num_experts] Qwen3MoeMLPs:
          gate, up: [intermediate, hidden]
          down: [hidden, intermediate]
    Dst (Torchtune GroupedExperts)
        gate, up: [num_experts, hidden, intermediate]
        down: [num_experts, intermediate, hidden]
    """

    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

    for i, expert in enumerate(src_experts):
        expert: Qwen3MoeMLP

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            # [intermediate, hidden] -> [hidden, intermediate] if gate | up, shapes flipped if down_proj
            src_proj = getattr(expert, proj).weight.T.contiguous()
            # [num_experts, hidden, intermediate] -> [hidden, intermediate] if gate | up; transpose(1,2) if down_proj
            dst_proj = getattr(dst_experts, proj)[i]

            check_tensor_attributes(src_proj, dst_proj, proj)
            dst_proj.data.copy_(src_proj)


def copy_from_hf(src_model: HFQwen3MoeModelForCausalLM, dst_model: Qwen3MoeModel):
    # Embeddings and RoPE
    _copy_module(src_model, dst_model, model_mapping=_QWEN3_EMBED_MAPPING)

    # Copy weights within decoder layers
    src_decoder_layers = src_model.model.layers
    dst_decoder_layers = dst_model.layers
    for src_layer, dst_layer in zip(src_decoder_layers, dst_decoder_layers):
        src_layer: HFQwen3MoeDecoderLayer
        dst_layer: Qwen3MoeDecoderLayer
        dst_mlp: Qwen3MoeSparseMoeBlock = dst_layer.mlp
        # Attention
        _copy_module(src_layer, dst_layer, model_mapping=_QWEN3_ATTN_MAPPING)

        # MoE
        # Router
        _copy_module(src_layer, dst_layer, model_mapping=_QWEN3_EXPERTS_MAPPING)

        # Experts - requires special treatment since HF uses `num_experts` individual MLPs
        _copy_expert_weights(src_layer.mlp.experts, dst_mlp.experts)