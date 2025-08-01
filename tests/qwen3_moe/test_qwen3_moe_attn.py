from dataclasses import fields

import pytest
import torch
import transformers.models.qwen3_moe.modeling_qwen3_moe as hf_qwen3_moe

from torchtitan.experiments.qwen3_moe.model.model import Qwen3MoeRotaryEmbedding, Qwen3MoeAttention, Qwen3MoeRMSNorm, Qwen3MoeDecoderLayer, Qwen3MoeModel

from torchtitan.experiments.qwen3_moe.model.configuration import Qwen3MoeConfig

from torchtitan.testing.utils import (
    TEST_TOL,
    check_tensors,
    initialize_model,
)

HFQwen3MoeModel = hf_qwen3_moe.Qwen3MoeModel
HFQwen3MoeAttention = hf_qwen3_moe.Qwen3MoeAttention
HFQwen3MoeDecoderLayer = hf_qwen3_moe.Qwen3MoeDecoderLayer
HFQwen3RoPE = hf_qwen3_moe.Qwen3MoeRotaryEmbedding

@pytest.fixture
def hf_attention(hf_moe_model: HFQwen3MoeModel) -> HFQwen3MoeAttention: 
    """
    HFQwen3MoeAttention
    """
    decoder_layer: HFQwen3MoeDecoderLayer = hf_moe_model.layers[0]
    return decoder_layer.self_attn


@pytest.fixture
def hf_rope(hf_moe_model: HFQwen3MoeModel) -> HFQwen3RoPE:
    """
    RoPE: position embeddings are instantiated outside of attention and passed as a
    parameter to attention forward in HF's attention implementation.
    """
    return hf_moe_model.rotary_emb


@pytest.fixture
def qwen3_moe_rope(qwen3_moe_model: Qwen3MoeModel) -> Qwen3MoeRotaryEmbedding:
    """
    RoPE: position embeddings are instantiated outside of attention and passed as a
    parameter to attention forward in HF's attention implementation.
    """
    return qwen3_moe_model.rope_embeds

@pytest.fixture
def qwen3_moe_attention(qwen3_moe_model: Qwen3MoeModel) -> Qwen3MoeAttention:
    decoder_layer: Qwen3MoeDecoderLayer = qwen3_moe_model.layers[0]
    return decoder_layer.self_attn


def check_rms_norm(
    ref_attn: HFQwen3MoeAttention,
    test_attn: Qwen3MoeAttention,
    bs: int,
    seqlen: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    atol: float,
    rtol: float,
):
    x = torch.randn(bs, seqlen, num_heads, head_dim, dtype=dtype, device=device)

    for key in ["q_norm", "k_norm"]:
        ref_norm = ref_attn.get_submodule(key)
        test_norm = test_attn.get_submodule(key)

        assert ref_norm.variance_epsilon == test_norm.eps

        expected = ref_norm(x)
        actual = test_norm(x)
        check_tensors(expected, actual, key, atol=atol, rtol=rtol)


def check_attention(
    ref_attn: HFQwen3MoeAttention,
    test_attn: Qwen3MoeAttention,
    ref_rope: HFQwen3RoPE,
    test_rope: Qwen3MoeRotaryEmbedding,
    bs: int,
    seqlen: int,
    hidden_dim: int,
    dtype: torch.dtype,
    device: str,
    atol: float,
    rtol: float,
):
    assert ref_rope.config.rope_theta == test_rope.theta
    check_tensors(ref_rope.inv_freq, test_rope.inv_freq, "RoPE inv freq", atol, rtol)

    # Copy proj weights
    with torch.no_grad():
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            ref_weight = getattr(ref_attn, proj).weight
            test_weight = getattr(test_attn, proj).weight
            assert ref_weight.shape == test_weight.shape
            test_weight.copy_(ref_weight)

    # Check attention
    x = torch.randn(bs, seqlen, hidden_dim, dtype=dtype, device=device)
    pos_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    hf_rope_embeds = ref_rope.forward(x, position_ids=pos_ids)
    qwen_rope_embeds = test_rope.forward(x, position_ids=pos_ids)

    for freq_ref, freq_test in zip(hf_rope_embeds, qwen_rope_embeds):
        check_tensors(freq_ref, freq_test, "Qwen3RotaryEmbedding", atol=atol, rtol=rtol)

    # Test
    expected_attn_out, _ = ref_attn.forward(
        x, position_embeddings=hf_rope_embeds, attention_mask=None
    )

    actual_attn_out = test_attn.forward(x, rope_freqs=qwen_rope_embeds)
    check_tensors(
        expected_attn_out, actual_attn_out, "Qwen3MoeAttention", atol=atol, rtol=rtol
    )


@pytest.mark.parametrize("seqlen", [128], ids=lambda x: f"seqlen={x}")
@pytest.mark.parametrize("bs", [1], ids=lambda x: f"bs={x}")
def test_qwen3_attention(
    hf_attention: HFQwen3MoeAttention,
    hf_rope: HFQwen3RoPE,
    qwen3_moe_rope: Qwen3MoeRotaryEmbedding,
    qwen3_moe_attention: Qwen3MoeAttention,
    bs: int,
    seqlen: int,
    dtype: torch.dtype,
    device: str | torch.device = "cuda",
):
    torch.manual_seed(0)
    print(qwen3_moe_attention)

    initialize_model(
        hf_attention,
        dtype=dtype,
        device=device,
        init_fns={"Qwen3MoeRMSNorm": {"weight": torch.nn.init.ones_}}
    )
    initialize_model(
        hf_rope,
        dtype=dtype,
        device=device,
    )
    initialize_model(
        qwen3_moe_rope,
        dtype=dtype,
        device=device,
    )
    # Local implementation of Qwen3MoeRMSNorm has a reset_parameters method that initializes weight to ones
    initialize_model(
        qwen3_moe_attention,
        dtype=dtype,
        device=device,
    )

    atol, rtol = TEST_TOL[dtype]

    hidden_dim = hf_attention.config.hidden_size
    head_dim = qwen3_moe_attention.head_dim
    num_attn_heads = qwen3_moe_attention.num_heads
    num_kv_heads = qwen3_moe_attention.num_kv_heads

    assert hf_attention.head_dim == head_dim
    assert hf_attention.config.num_attention_heads == num_attn_heads
    assert hf_attention.config.num_key_value_heads == num_kv_heads

    # Rms sanity check
    check_rms_norm(
        hf_attention,
        qwen3_moe_attention,
        bs,
        seqlen,
        num_attn_heads,
        head_dim,
        dtype,
        device,
        atol,
        rtol,
    )

    # Check attention debugging outputs: q_proj, q_proj_norm, q_rot, etc.
    check_attention(
        hf_attention,
        qwen3_moe_attention,
        ref_rope=hf_rope,
        test_rope=qwen3_moe_rope,
        bs=bs,
        seqlen=seqlen,
        hidden_dim=hidden_dim,
        dtype=dtype,
        device=device,
        atol=atol,
        rtol=rtol,
    )

    # # Check against canonical HF implementation
    # check_attention(
    #     hf_attention,
    #     qwen3_moe_attention,
    #     ref_rope=hf_rope,
    #     bs=bs,
    #     seqlen=seqlen,
    #     hidden_dim=hidden_dim,
    #     dtype=dtype,
    #     device=device,
    #     atol=atol,
    #     rtol=rtol,
    # )