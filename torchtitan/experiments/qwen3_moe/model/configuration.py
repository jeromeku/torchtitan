# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass, field

from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import (
    Qwen3MoeConfig as HFQwen3MoeConfig,
)

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.protocols.train_spec import BaseModelArgs
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability
from functools import lru_cache



@dataclass(kw_only=True)
class Qwen3MoeConfig(BaseModelArgs):
    # Embeddings
    vocab_size: int = 151936
    hidden_size: int
    max_seq_len: int
    rope_theta: float
    tie_word_embeddings: bool = False

    # Decoder
    num_hidden_layers: int
    intermediate_size: int
    mlp_only_layers: list[int] = None

    # Attention
    num_attention_heads: int
    num_key_value_heads: int = 4
    head_dim: int = 128
    q_proj_bias: bool = False
    k_proj_bias: bool = False
    v_proj_bias: bool = False
    attention_dropout: float = 0.0

    use_flex_attention: bool = False
    attn_mask_type: str = "causal"

    # Norms
    rms_norm_eps: float = 1e-6  # Change to match HF's Qwen3ModelConfig
    q_norm: bool = True
    k_norm: bool = True

    # MoE
    use_moe: bool = True
    num_experts: int = 128
    moe_intermediate_size: int
    num_experts_per_tok: int = 8
    
    # routing
    score_fn: str = "softmax"
    use_scatter_indices: bool = False
    norm_topk_prob: bool = True
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    
    # experts
    use_grouped_gemm: bool = False
    act_fn: str = "silu"

    @classmethod
    def from_hf(cls, hf_config: HFQwen3MoeConfig, **kwargs):

        assert getattr(hf_config, "head_dim", None) is not None, "Qwen3MoeConfig should specify head_dim"
        head_dim = hf_config.head_dim
        assert hf_config.norm_topk_prob, "Qwen3Moe normalizes topk prob"

        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            intermediate_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            num_experts=hf_config.num_experts,
            moe_intermediate_size=hf_config.moe_intermediate_size,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            mlp_only_layers=hf_config.mlp_only_layers,
            q_proj_bias=hf_config.attention_bias,
            k_proj_bias=hf_config.attention_bias,
            v_proj_bias=hf_config.attention_bias,
            attention_dropout=hf_config.attention_dropout,
            rms_norm_eps=hf_config.rms_norm_eps,
            norm_topk_prob=hf_config.norm_topk_prob,
            output_router_logits=hf_config.output_router_logits,
            router_aux_loss_coef=hf_config.router_aux_loss_coef,
            max_seq_len=hf_config.max_position_embeddings,
            rope_theta=hf_config.rope_theta,
            **kwargs,
        )

    def update_from_config(
        self, job_config: JobConfig, tokenizer: BaseTokenizer
    ) -> None:
        if self.vocab_size is not None:
            if self.vocab_size != tokenizer.get_vocab_size():
                logger.warning(f"Vocab size mismatch, model config vocab_size != tokenizer vocab_size: {self.vocab_size} != {tokenizer.get_vocab_size()}")
        else:
            self.vocab_size = tokenizer.get_vocab_size()
        
        self.max_seq_len = job_config.training.seq_len
        self.eos_id = tokenizer.eos_id
        assert self.eos_id is not None, "Tokenizer does not have eos_id, please set manually."
        logger.info(f"eos_id set to {self.eos_id}")

        if self.use_grouped_gemm and not has_cuda_capability(9, 0):
            logger.warning(
                "Failed to use grouped mm, which is only supported on SM90 or later",
            )
            self.use_grouped_gemm = False

        if job_config.activation_checkpoint.mode == "selective" and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with selective AC yet. "
                "See https://github.com/pytorch/pytorch/issues/147879"
            )

        if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
            raise ValueError(
                "FlexAttention is not compatible with CP yet. "
                "We are still working on this."
            )
    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, float]:
        print("TODO!")
        nparams = sum(p.numel() for p in model.parameters())
        flops = 1.
        return nparams, flops

@lru_cache
def _get_hf_config(model_id: str):
    config = HFQwen3MoeConfig.from_pretrained(model_id)
    return config


@lru_cache
def qwen3_moe_config(model_id: str, **kwargs):
    config = Qwen3MoeConfig.from_hf(_get_hf_config(model_id))

    for k, v in kwargs.items():
        if k in Qwen3MoeConfig.__dataclass_fields__:
            setattr(config, k, v)
        else:
            raise ValueError(f"{k} not a recognized attribute of Qwen3MoeConfig")

    return config

QWEN3_MOE_30B = "Qwen3/Qwen3-30B-A3B"

Qwen3MoeConfig_30b_A3B = Qwen3MoeConfig(
    vocab_size=151936,
    hidden_size=2048,
    tie_word_embeddings=False,
    max_seq_len=32768,
    rope_theta=10000.0,
    num_hidden_layers=24,
    intermediate_size=6144,
    mlp_only_layers=None,
    num_attention_heads=32,
    num_key_value_heads=4,
    head_dim=64,
    q_proj_bias=False,
    k_proj_bias=False,
    v_proj_bias=False,
    attention_dropout=0.0,
    rms_norm_eps=1e-06,
    q_norm=True,
    k_norm=True,
    use_moe=True,
    num_experts=128,
    moe_intermediate_size=768,
    num_experts_per_tok=8,
    score_fn='softmax',
    use_scatter_indices=False,
    norm_topk_prob=False,
    output_router_logits=False,
    router_aux_loss_coef=0.001,
    use_grouped_gemm=False,
    act_fn='silu',
)

QWEN3_MOE_235B = "Qwen/Qwen3-235B-A22B"

Qwen3MoeConfig_235B_A22B = Qwen3MoeConfig(
    vocab_size=151936,
    hidden_size=4096,
    tie_word_embeddings=False,
    max_seq_len=40960,
    rope_theta=1000000.0,
    num_hidden_layers=94,
    intermediate_size=12288,
    mlp_only_layers=None,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=64,
    q_proj_bias=False,
    k_proj_bias=False,
    v_proj_bias=False,
    attention_dropout=0.0,
    rms_norm_eps=1e-06,
    q_norm=True,
    k_norm=True,
    use_moe=True,
    num_experts=128,
    moe_intermediate_size=1536,
    num_experts_per_tok=8,
    score_fn='softmax',
    use_scatter_indices=False,
    norm_topk_prob=True,
    output_router_logits=False,
    router_aux_loss_coef=0.001,
    use_grouped_gemm=False,
    act_fn='silu',
)
