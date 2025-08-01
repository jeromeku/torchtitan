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

from .infra.parallelize import parallelize_qwen3
from .model.configuration import Qwen3MoeConfig
from .model.model import Qwen3MoeModel
from .optimizer import build_qwen3_optimizers
from typing import Literal
from transformers.utils.logging import disable_progress_bar
from transformers import AutoConfig
from functools import lru_cache
disable_progress_bar()

QWEN3_30B_A3B = "Qwen/Qwen3-30B-A3B"
QWEN3_235B_A2B = "Qwen/Qwen3-235B-A22B"


@lru_cache
def download_hf_config(model_id: str):
    return AutoConfig.from_pretrained(model_id)
  
qwen3_moe_configs = {
    "debug": Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=2048,
        max_seq_len=32768,
        rope_theta=10000.0,
        intermediate_size=6144,
        num_attention_heads=32,
        moe_intermediate_size=768,
        router_aux_loss_coef=None
    ),
    QWEN3_30B_A3B: Qwen3MoeConfig(
        num_hidden_layers=24,
        hidden_size=2048,
        max_seq_len=32768,
        rope_theta=10000.0,
        intermediate_size=6144,
        num_attention_heads=32,
        moe_intermediate_size=768,
    ),
    QWEN3_235B_A2B: Qwen3MoeConfig(
        num_hidden_layers=94,
        hidden_size=4096,
        max_seq_len=40960,
        rope_theta=1000000.0,
        intermediate_size=12288,
        num_attention_heads=64,
        moe_intermediate_size=1536,
    ),
}


register_train_spec(
    TrainSpec(
        name="qwen3_moe",
        model_cls=Qwen3MoeModel,
        model_args=qwen3_moe_configs,
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=None,
        build_optimizers_fn=build_qwen3_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
