from dataclasses import fields

import pytest
import torch
from transformers.models.qwen3_moe import (
    Qwen3MoeConfig as HFQwen3MoeConfig
)
from transformers.models.qwen3_moe import (
    Qwen3MoeModel as HFQwen3MoeModel,
)
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock as HFQwen3MoeSparseMoeBlock,
)

from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeMLP

from torchtitan.experiments.qwen3_moe.model.moe import (
    GroupedExperts,
    RouterOutputs,
    TokenChoiceTopKRouter,
)
from torchtitan.experiments.qwen3_moe.model.model import Qwen3MoeSparseMoeBlock

from torchtitan.experiments.qwen3_moe.model.configuration import (
    Qwen3MoeConfig,
)

# from torchtune.testing.reference.qwen3_moe.modeling import HFRouterOutputs
from torchtitan.testing.utils import (
    TEST_TOL,
    check_equal,
    check_tensors,
    get_kwargs,
    initialize_model,
    key_value_format
)

torch.manual_seed(0)

@pytest.fixture
def hf_moe_block(hf_qwen3_moe_config: Qwen3MoeConfig, device_meta: None) -> HFQwen3MoeSparseMoeBlock:
    """
    Original HF SparseMoeBlock
    """
    return HFQwen3MoeSparseMoeBlock(hf_qwen3_moe_config)



@pytest.fixture
def qwen3_moe_block(qwen3_moe_config: Qwen3MoeConfig, device_meta) -> Qwen3MoeSparseMoeBlock:
    return Qwen3MoeSparseMoeBlock(config=qwen3_moe_config)


@pytest.fixture
def qwen3_moe_experts(qwen3_moe_block: Qwen3MoeSparseMoeBlock) -> GroupedExperts:
    return qwen3_moe_block.experts


@pytest.fixture
def qwen3_moe_router(qwen3_moe_block: Qwen3MoeSparseMoeBlock) -> TokenChoiceTopKRouter:
    return qwen3_moe_block.router


# @pytest.mark.parametrize("seqlen", [128], ids=lambda x: f"seqlen={x}")
# @pytest.mark.parametrize("bs", [1], ids=lambda x: f"bs={x}")
# def test_qwen3_router(
#     hf_moe_block_tester: HFQwen3MoeSparseMoeBlockTester,
#     tt_router: TokenChoiceTopKRouter,
#     bs: int,
#     seqlen: int,
#     dtype: torch.dtype,
#     device: str = "cuda"
# ):
#     torch.manual_seed(0)

#     initialize_model(tt_router, dtype=dtype, device=device)
#     initialize_model(hf_moe_block_tester, dtype=dtype, device=device)
#     atol, rtol = TEST_TOL[dtype]

#     # Copy gate weights
#     with torch.no_grad():
#         tt_router.gate.weight.copy_(hf_moe_block_tester.gate.weight)

#     assert tt_router.gate.weight.equal(hf_moe_block_tester.gate.weight)

#     topk = tt_router.topk
#     d = tt_router.dim
#     x = torch.randn(bs, seqlen, d, device="cuda", dtype=dtype)

#     hf_router_out: HFRouterOutputs
#     _, hf_router_out = hf_moe_block_tester.forward(x)
#     router_out: RouterOutputs = tt_router.forward(x, debug=True, ref_outputs=hf_router_out)

#     # router logits
#     for field in fields(HFRouterOutputs):
#         ref = getattr(hf_router_out, field.name)
#         test = getattr(router_out, field.name)
#         if test.shape != ref.shape:
#             print(f"WARNING: {ref.shape} != {test.shape}")
#             test = test.reshape_as(ref)
#         check_tensors(ref, test, field.name, atol=atol, rtol=rtol)



@pytest.mark.parametrize("use_grouped_gemm", [False, True], ids=lambda x: f"grouped_gemm={x}")
@pytest.mark.parametrize("seqlen", [128], ids=lambda x: f"seqlen={x}")
@pytest.mark.parametrize("bs", [1], ids=lambda x: f"bs={x}")
def test_qwen3_experts(
    bs: int,
    seqlen: int,
    dtype: torch.dtype,
    use_grouped_gemm: bool,
    hf_moe_block: HFQwen3MoeSparseMoeBlock,
    qwen3_moe_block: Qwen3MoeSparseMoeBlock,
    device: str = "cuda",
):
    initialize_model(hf_moe_block, dtype=dtype, device=device)
    initialize_model(qwen3_moe_block, dtype=dtype, device=device)

    atol, rtol = TEST_TOL[dtype]

    hidden_size = qwen3_moe_block.config.hidden_size
    topk = qwen3_moe_block.config.num_experts_per_tok
    num_experts = qwen3_moe_block.config.num_experts
    moe_intermediate_size = qwen3_moe_block.config.moe_intermediate_size

    x = torch.randn(bs, seqlen, hidden_size, device="cuda", dtype=dtype)

    # Unpack, check dims
    router: TokenChoiceTopKRouter = qwen3_moe_block.router
    experts: GroupedExperts = qwen3_moe_block.experts
    E, K, N = experts.gate_proj.shape
    check_equal(num_experts, E, "Num experts")
    check_equal(moe_intermediate_size, N, "Moe intermediate dim")
    check_equal(hidden_size, K, "Moe hidden size")

    # True => _grouped_mm, False => for-loop based expert calc (reference only, way too slow to use in prod)
    # TODO: add custom grouped gemm
    if use_grouped_gemm and dtype != torch.bfloat16:
        pytest.skip("_grouped_mm only supports bfloat16")
    experts.use_grouped_mm = use_grouped_gemm

    # Copy weights
    with torch.no_grad():
        # router
        router.gate.weight.copy_(hf_moe_block.gate.weight)
        hf_moe_block.gate.weight.copy_(hf_moe_block.gate.weight)

        # experts
        for i, expert in enumerate(hf_moe_block.experts):
            expert: Qwen3MoeMLP

            for proj in ["gate_proj", "up_proj", "down_proj"]:
                check_equal(
                    getattr(experts, proj)[i].shape,
                    getattr(expert, proj).weight.T.shape,
                    f"{proj} shape",
                )
                getattr(experts, proj)[i].data.copy_(getattr(expert, proj).weight.T.contiguous())
                getattr(hf_moe_block.experts[i], proj).weight.copy_(getattr(expert, proj).weight)

    hf_expert_out, _ = hf_moe_block.forward(x)

    expert_out = qwen3_moe_block.forward(x)
    # router_out = router(x)
    # expert_out = qwen3_moe_block.run_experts(x.view(-1, hidden_size), router_out)

    # check_tensors(
    #     hftester_expert_out,
    #     expert_out,
    #     label="Expert, hf test vs Router + Experts",
    #     atol=atol,
    #     rtol=rtol,
    # )

    # expert_out = qwen3_moe_block(x)
    # check_tensors(
    #     hftester_expert_out,
    #     expert_out,
    #     label="Expert, HFQwen3SparseMoeBlock Tester vs Qwen3SparseMoeBlock",
    #     atol=atol,
    #     rtol=rtol,
    # )

    # Finally check against original HF SparseMoeBlock
    check_tensors(
        hf_expert_out,
        expert_out,
        label="Expert, HFQwen3SparseMoeBlock vs Qwen3SparseMoeBlock",
        atol=atol,
        rtol=rtol,
    )