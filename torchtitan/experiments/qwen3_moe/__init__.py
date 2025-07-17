# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.models.llama3 import pipeline_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_llama
from .model.args import Qwen3MoeConfig
from .model.model import Transformer
from .optimizer import build_llama4_optimizers



llama4_configs = {
    "singlelayer": Qwen3MoeConfig(
        dim=256,
        n_layers=1,
        n_heads=16,
        rope_theta=500000,
    ),
    "30B-A30B": Qwen3MoeConfig(
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
    ),
    "235B-A22B": Qwen3MoeConfig(
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
}


register_train_spec(
    TrainSpec(
        name="llama4",
        model_cls=Transformer,
        model_args=llama4_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_llama4_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
