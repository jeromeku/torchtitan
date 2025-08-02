import gc
import re

import os

import pytest
import torch
import transformers.models.qwen3_moe.modeling_qwen3_moe as hf_qwen3_moe
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig as HFQwen3MoeConfig

from torchtitan.experiments.qwen3_moe import Qwen3MoeModel
from torchtitan.experiments.qwen3_moe.model.configuration import QWEN3_MOE_30B
from torchtitan.experiments.qwen3_moe.model import (
    Qwen3MoeSparseMoeBlock,
    Qwen3MoeDecoderLayer,
    Qwen3MoeAttention,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding
)

from torchtitan.experiments.qwen3_moe.model.configuration import Qwen3MoeConfig
from torchtitan.experiments.qwen3_moe.conversion.convert_from_hf import copy_from_hf
from torchtitan.testing.utils import (
    TEST_TOL,
    check_tensors,
    initialize_dummy_weights,
    set_default_torch_dtype,
)
from torchtitan.tools.initialization import init_on_device

HFQwen3MoeModelForCausalLM = hf_qwen3_moe.Qwen3MoeForCausalLM
HFQwen3MoeModel = hf_qwen3_moe.Qwen3MoeModel
HFQwen3MoeDecoderLayer = hf_qwen3_moe.Qwen3MoeDecoderLayer
HFQwen3MoeAttention = hf_qwen3_moe.Qwen3MoeAttention
HFQwen3RMSNorm = hf_qwen3_moe.Qwen3MoeRMSNorm
HFQwen3RoPE = hf_qwen3_moe.Qwen3MoeRotaryEmbedding
HFQwen3MoeSparseMoeBlock = hf_qwen3_moe.Qwen3MoeSparseMoeBlock
HFMoeCausalLMOutput = hf_qwen3_moe.MoeCausalLMOutputWithPast


TEST_TOL = {torch.float32: (1e-4, 1e-4), torch.float16: (1e-3, 1e-3), torch.bfloat16: (1e-2, 1e-2)}
SKIP_FP16_LOGITS_TESTS = os.getenv("SKIP_FP16_LOGITS_TESTS", "0") == "1"

# ---------------------------------------------------------------------------------------------------#
# Fixtures for parametrizing and setting up class-level model instances


# Omit `=` in id string for fixtures to enable test filtering (pytest can't parse filters with `=`)
@pytest.fixture(scope="class", params=[QWEN3_MOE_30B], ids=lambda id: f"model_id{id}")
def model_id(request):
    return request.param

@pytest.fixture(scope="class", params=[0], ids=lambda s: f"seed{s}")
def seed(request):
    return request.param


@pytest.fixture(scope="class", params=[1], ids=lambda bs: f"bs{bs}")
def batch_size(request):
    return request.param


@pytest.fixture(scope="class", params=[128, 1024], ids=lambda s: f"seqlen{s}")
def seqlen(request):
    return request.param


@pytest.fixture(
    scope="class",
    params=[torch.float32, torch.bfloat16],
    ids=lambda dtype: str(dtype).split('.')[-1],
)
def dtype(request):
    return request.param


@pytest.fixture(scope="class", params=[1, 8], ids=lambda n: f"n_layers{n}")
def num_layers(request):
    return request.param


@pytest.fixture(scope="class")
def device_type():
    return "cuda"


# Primary model setup fixture, initializes hf reference model and torchtune test model
@pytest.fixture(scope="class")
def models(request, model_id, seed, batch_size, seqlen, num_layers, dtype, device_type):
    cls = request.cls

    cls.bs = batch_size
    cls.seqlen = seqlen
    cls.num_layers = num_layers
    cls.dtype = dtype
    cls.device_type = device_type

    torch.manual_seed(seed)

    cls.atol, cls.rtol = TEST_TOL[dtype]

    hf_cfg = HFQwen3MoeConfig.from_pretrained(QWEN3_MOE_30B)
    hf_cfg.num_hidden_layers = num_layers
    qwen3_moe_config = Qwen3MoeConfig.from_hf(hf_cfg)

    # with init_on_device("meta", include_buffers=False):
    with init_on_device(device_type, include_buffers=True):
        with set_default_torch_dtype(dtype):
            cls.hf_causal_lm: HFQwen3MoeModelForCausalLM = HFQwen3MoeModelForCausalLM(hf_cfg)  # type: ignore
            cls.qwen3_moe_causal_lm: Qwen3MoeModel = Qwen3MoeModel(qwen3_moe_config)  # type: ignore

    initialize_dummy_weights(cls.hf_causal_lm, exclude_pat=["norm.weight", "layernorm.weight"])
    copy_from_hf(cls.hf_causal_lm, cls.qwen3_moe_causal_lm)

    assert all(p.dtype == dtype for p in cls.hf_causal_lm.parameters()), (
        f"Not all params are dtype {dtype}"
    )
    assert all(p.dtype == dtype for p in cls.qwen3_moe_causal_lm.parameters()), (
        f"Not all params are dtype {dtype}"
    )

    num_params = cls.hf_causal_lm.model.num_parameters() / 1e9
    print(
        f"{num_params:.1f}B params * {dtype.itemsize} bytes = {num_params * dtype.itemsize:.1f}GB"
    )

    # Aliases for easy access in tests
    # Unwrap HF causal LM
    cls.hf_model: HFQwen3MoeModel = cls.hf_causal_lm.model  # type: ignore # noqa
    cls.qwen3_moe_model = cls.qwen3_moe_causal_lm  # type: ignore # noqa

    # # Token-IDs reused by most tests
    vocab = hf_cfg.vocab_size
    cls.input_ids = torch.randint(0, vocab, (batch_size, seqlen), device=device_type)

    yield

    del cls.hf_causal_lm, cls.qwen3_moe_causal_lm, cls.hf_model, cls.qwen3_moe_model
    gc.collect()

    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------#


# Tests each Qwen3 MoE component then runs an e2e test for each hidden layer and logits.
# Set the global flag SKIP_FP16_LOGITS_TESTS to run only float32 logits tests.
@pytest.mark.usefixtures("models")
class TestQwen3MoeModel:
    @property
    def ref_embedding(self):
        if getattr(self, "_ref_embedding", None) is None:
            self._ref_embedding = self.hf_model.embed_tokens(self.input_ids)
        return self._ref_embedding

    
    @property
    def ref_rope_embeds(self):
        if getattr(self, "_rope_embeds", None) is None:
            hidden_states = self.ref_embedding

            # Rotary embeddings
            position_ids = torch.arange(0, self.seqlen, device=self.device_type).unsqueeze(0)
            position_embeddings = self.hf_model.rotary_emb(hidden_states, position_ids)
            self._rope_embeds = position_embeddings

        return self._rope_embeds
    
    @property
    def ref_attention(self):
        if getattr(self, "_ref_attention", None) is None:
            layer_idx = 0
            ref_layer: HFQwen3MoeDecoderLayer = self.hf_model.layers[layer_idx]

            hidden_states = self.ref_embedding

            position_embeddings = self.ref_rope_embeds

            attn_out, _ = ref_layer.self_attn(
                hidden_states, position_embeddings=position_embeddings, attention_mask=None
            )

            attn_post_norm = ref_layer.post_attention_layernorm(attn_out)
            self._ref_attention = (attn_out, attn_post_norm)
        return self._ref_attention

    @property
    def ref_experts(self):
        if getattr(self, "_ref_experts", None) is None:
            ref_mlp: HFQwen3MoeSparseMoeBlock = self.hf_model.layers[0].mlp
            hidden_states = self.ref_attention[1]
            experts_out, router_logits = ref_mlp(hidden_states)
            self._ref_experts = (router_logits, experts_out)
        
        return self._ref_experts

    # Test each module: embeddings, attention, experts
    def test_embeddings(self):
        test_embedding = self.qwen3_moe_model.tok_embeddings(self.input_ids)

        check_tensors(
            self.ref_embedding, test_embedding, "embedding", atol=self.atol, rtol=self.rtol
        )

    def test_attention(self):
        hidden_states = self.ref_embedding
        
        layer_idx = 0
        self_attention_layer: Qwen3MoeDecoderLayer = self.qwen3_moe_model.layers[layer_idx]
        
        position_ids = torch.arange(0, self.seqlen, device=self.device_type).unsqueeze(0)
        rope_embeds = self.qwen3_moe_model.rope_embeds.forward(hidden_states, position_ids)
        for freq_ref, freq_test in zip(self.ref_rope_embeds, rope_embeds):
            check_tensors(freq_ref, freq_test, "rope embeddings", atol=self.atol, rtol=self.rtol)

        test_attention = self_attention_layer.self_attn.forward(hidden_states, rope_embeds)
        check_tensors(
            self.ref_attention[0], test_attention, "self-attention", atol=self.atol, rtol=self.rtol
        )

    def test_router_logits(self):
        ref_router_logits = self.ref_experts[0]

        # Attention post-norm
        hidden_states = self.ref_attention[1]
        test_mlp = self.qwen3_moe_model.layers[0].mlp
        test_router_logits = test_mlp.router(hidden_states, debug=True).router_logits

        check_tensors(
            ref_router_logits,
            test_router_logits.reshape_as(ref_router_logits),
            "router logits",
            atol=self.atol,
            rtol=self.rtol,
        )

    def test_experts(self):
        ref_experts_out = self.ref_experts[1]

        hidden_states = self.ref_attention[1]
        test_mlp: Qwen3MoeSparseMoeBlock = self.qwen3_moe_model.layers[0].mlp

        test_experts_out = test_mlp.run_experts(
            hidden_states.view(-1, hidden_states.shape[-1]),
            router_outputs=test_mlp.router(hidden_states, debug=False),
        )

        check_tensors(
            ref_experts_out,
            test_experts_out.reshape_as(ref_experts_out),
            "experts",
            atol=self.atol,
            rtol=self.rtol,
        )

    # e2e test
    def test_logits(self):
        if SKIP_FP16_LOGITS_TESTS and (
            self.dtype != torch.float32 and (self.seqlen > 128 or self.num_layers > 1)
        ):
            pytest.skip(
                f"Skipping {self.dtype} for seqlen={self.seqlen} and num_layers={self.num_layers} for now"
            )

        # HF
        ref_outputs: HFMoeCausalLMOutput = self.hf_causal_lm(
            self.input_ids, output_hidden_states=True
        )

        # TT
        out_states = list(range(len(self.qwen3_moe_model.layers) + 1))
        self.qwen3_moe_model.output_hidden_states = out_states
        *test_hidden_states, test_logits = self.qwen3_moe_model(self.input_ids)

        # final layernorm on TT side to match HF contract
        test_hidden_states[-1] = self.qwen3_moe_model.norm(test_hidden_states[-1])

        assert len(ref_outputs.hidden_states) == len(test_hidden_states)

        # for i, (ref, test) in enumerate(zip(ref_outputs.hidden_states, test_hidden_states)):
        #     check_tensors(ref, test, f"hidden_state[{i}]", atol=self.atol, rtol=self.rtol)

        # HF only upcasts logits if calculating loss
        if self.dtype != torch.float32:
            ref_outputs.logits = ref_outputs.logits.float()
        
        check_tensors(ref_outputs.logits, test_logits, "logits", atol=self.atol, rtol=self.rtol)