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


@dataclass(kw_only=True)
class Qwen3MoeConfig(BaseModelArgs):
    # Embeddings
    vocab_size: int
    embed_dim: int
    tie_word_embeddings: bool = False
    max_seq_len: int = 32768
    rope_base: float = 1_000_000.0

    # Decoder
    num_layers: int
    intermediate_dim: int
    mlp_only_layers: list[int] = field(default_factory=list)

    # Attention
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_proj_bias: bool = False
    k_proj_bias: bool = False
    v_proj_bias: bool = False
    attn_dropout: float = 0.0

    # Norms
    norm_eps: float = 1e-6  # Change to match HF's Qwen3ModelConfig
    q_norm: bool = True
    k_norm: bool = True

    # MoE
    num_experts: int
    expert_intermediate_dim: int
    num_experts_per_tok: int

    # routing
    score_fn: str = "softmax"
    use_scatter_indices: bool = False
    norm_topk_prob: bool = True  # NOTE: this defaults to `False` in HF Qwen3MoeConfig
    output_router_logits: bool = False
    router_aux_loss_coef: float | None = 1e-3

    # experts
    use_grouped_gemm: bool = False
    act_fn: str = "silu"

    @classmethod
    def from_hf(cls, hf_config: HFQwen3MoeConfig, **kwargs):
        head_dim = (
            hf_config.hidden_size // hf_config.num_attention_heads
        )  # See __init__ of HF Qwen3MoeAttention

        return cls(
            vocab_size=hf_config.vocab_size,
            embed_dim=hf_config.hidden_size,
            num_layers=hf_config.num_hidden_layers,
            intermediate_dim=hf_config.intermediate_size,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            num_experts=hf_config.num_experts,
            expert_intermediate_dim=hf_config.moe_intermediate_size,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            tie_word_embeddings=hf_config.tie_word_embeddings,
            mlp_only_layers=hf_config.mlp_only_layers,
            q_proj_bias=hf_config.attention_bias,
            k_proj_bias=hf_config.attention_bias,
            v_proj_bias=hf_config.attention_bias,
            attn_dropout=hf_config.attention_dropout,
            norm_eps=hf_config.rms_norm_eps,
            norm_topk_prob=hf_config.norm_topk_prob,
            output_router_logits=hf_config.output_router_logits,
            router_aux_loss_coef=hf_config.router_aux_loss_coef,
            max_seq_len=hf_config.max_position_embeddings,
            rope_base=hf_config.rope_theta,
            **kwargs,
        )

    def update_from_config(
        self, job_config: JobConfig, tokenizer: BaseTokenizer
    ) -> None:
        if self.vocab_size is not None:
            assert self.vocab_size == tokenizer.get_vocab_size(), f"Vocab size mismatch: {self.vocab_size} != {tokenizer.get_vocab_size()}"
        else:
            self.vocab_size = tokenizer.get_vocab_size()
        
        self.max_seq_len = job_config.training.seq_len
        self.eos_id = tokenizer.eos_id
        assert self.eos_id is not None, "Tokenizer does not have eos_id, please set manually."
        logger.info(f"eos_id set to {self.eos_id}")

        if self.use_grouped_mm and not has_cuda_capability(9, 0):
            logger.warning(
                "Failed to use grouped mm, which is only supported on SM90 or later",
            )
            self.use_grouped_mm = False

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
        return
    
        # Calculation taken from torchtitan llama4
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_expert = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif "moe.shared_expert" in name:
                nparams_shared_expert += p.numel()
            elif "moe.router" in name:
                nparams_moe_router += p.numel()
            elif "moe.experts" in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_expert + nparams_experts
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_expert
            + nparams_experts * self.top_k // self.num_experts
        )

        logger.info(
            f"Total parameter count: dense {nparams_dense:,}, "
            f"sparse {nparams_sparse:,}, active {nparams_dense + nparams_sparse_active:,}"
        )

        l, h, q, t = (
            self.n_layers,
            self.n_heads,
            self.dim // self.n_heads,
            seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = (
            6 * (nparams_dense - nparams_embedding + nparams_sparse_active)
            + 12 * l * h * q * t
        )

        return nparams, num_flops_per_token

# Map from Qwen3MoeConfig -> HF Qwen3MoeConfig
_MAPPING_TO_HF = {
    "vocab_size": "vocab_size",
    "embed_dim": "hidden_size",
    "num_layers": "num_hidden_layers",
    "intermediate_dim": "intermediate_size",
    "num_heads": "num_attention_heads",
    "num_kv_heads": "num_key_value_heads",
    "head_dim": None,  # Derived from hidden_size and num_attention_heads
    "num_experts": "num_experts",
    "expert_intermediate_dim": "moe_intermediate_size",
    "num_experts_per_tok": "num_experts_per_tok",
    "tie_word_embeddings": "tie_word_embeddings",
    "mlp_only_layers": "mlp_only_layers",
    "q_proj_bias": "attention_bias",
    "k_proj_bias": "attention_bias",
    "v_proj_bias": "attention_bias",
    "attn_dropout": "attention_dropout",
    "norm_eps": "rms_norm_eps",
    "q_norm": None,  # True by default, since HF uses RMSNorm by default
    "k_norm": None,  # True by default, since HF uses RMSNorm by default
    "norm_topk_prob": "norm_topk_prob",
    "output_router_logits": "output_router_logits",
    "router_aux_loss_coef": "router_aux_loss_coef",
    "max_seq_len": "max_position_embeddings",
    "rope_base": "rope_theta",  # https://github.com/huggingface/transformers/blob/02a769b05860d2390e837309c3b41e99218b6555/src/transformers/modeling_rope_utils.py#L121-L125
    "act_fn": "hidden_act",
    # No HF equivalent
    "use_grouped_gemm": None,  # HF does not use grouped_gemm
    "score_fn": None,  # implicitly softmax in HF
    "use_scatter_indices": None,
}

# Map from HF Qwen3MoeConfig -> Qwen3MoeConfig
_MAPPING_FROM_HF = {
    "vocab_size": "vocab_size",
    "hidden_size": "embed_dim",
    "num_hidden_layers": "num_layers",
    "intermediate_size": "intermediate_dim",
    "num_attention_heads": "num_heads",
    "num_key_value_heads": "num_kv_heads",
    "num_experts": "num_experts",
    "moe_intermediate_size": "expert_intermediate_dim",
    "num_experts_per_tok": "num_experts_per_tok",
    "tie_word_embeddings": "tie_word_embeddings",
    "mlp_only_layers": "mlp_only_layers",
    "attention_bias": ["q_proj_bias", "k_proj_bias", "v_proj_bias"],
    "attention_dropout": "attn_dropout",
    "rms_norm_eps": "norm_eps",
    "norm_topk_prob": "norm_topk_prob",
    "output_router_logits": "output_router_logits",
    "router_aux_loss_coef": "router_aux_loss_coef",
    "max_position_embeddings": "max_seq_len",
    "rope_theta": "rope_base",  # See https://github.com/huggingface/transformers/blob/02a769b05860d2390e837309c3b41e99218b6555/src/transformers/modeling_rope_utils.py#L121-L125
    "hidden_act": "act_fn",
    # Not used
    "sliding_window": None,
    "initializer_range": None,
    "decoder_sparse_step": None,
    "use_sliding_window": None,
    "rope_scaling": None,
    "use_cache": None,
}


