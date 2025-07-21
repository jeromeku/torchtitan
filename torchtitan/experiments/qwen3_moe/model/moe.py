import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Replicate
from torch.nn import functional as F

from .args import Qwen3MoeConfig

class GroupedExperts(nn.Module):
    """This class implements the grouped experts layer used in Mixture of Experts. Each expert
    is a variant of the Gated Linear Units network. See more details in https://arxiv.org/pdf/2002.05202.

    Args:
        dim (int): Input dimension.
        hidden_dim (int): Hidden dimension.
        num_experts (int): Number of experts in this grouped experts layer. Default is 1.
        activation (Callable): Activation function to use. Default is F.silu.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        intermediate_dim: int,
        num_experts: int = 1,
        act_fn: str | Callable = F.silu,
        use_grouped_gemm: bool = False,
    ):
        super().__init__()
        self.dim = hidden_dim
        self.num_experts = num_experts

        self.gate_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))
        self.up_proj = nn.Parameter(torch.empty(num_experts, hidden_dim, intermediate_dim))

        self.down_proj = nn.Parameter(torch.empty(num_experts, intermediate_dim, hidden_dim))
        
        if isinstance(act_fn, str):
            self.act_fn = getattr(torch.nn.functional, act_fn)
        else:
            self.act_fn = act_fn
        self.use_grouped_mm = use_grouped_gemm
        self._experts_impl = self._forward_grouped_mm if use_grouped_gemm else self._forward_no_grouped_mm

    def reset_parameters(self) -> None:
        # Default initialization used by torch.nn.Linear
        nn.init.kaiming_uniform_(self.gate_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.down_proj, a=math.sqrt(5))
        if self.up_proj is not None:
            nn.init.kaiming_uniform_(self.up_proj, a=math.sqrt(5))

#    @expert_parallel
    def _forward_no_grouped_mm(
        self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor | None = None
    ) -> torch.Tensor:
        # a tuple of tensors indexed by experts
        # each with shape (tokens_per_expert(varying), dim)
        
        x = torch.split(
            x,
            split_size_or_sections=num_tokens_per_expert.tolist(),
            dim=0,
        )
        out_experts_splits = []
        for expert_idx, x_expert in enumerate(x):
            w1, w2, w3 = (
                self.gate_proj[expert_idx],
                self.down_proj[expert_idx],
                self.up_proj[expert_idx],
            )
            h = self.act_fn(torch.matmul(x_expert, w1))
            h = h * torch.matmul(x_expert, w3)
            h = torch.matmul(h, w2)
            # h shape (tokens_per_expert(varying), dim)
            out_experts_splits.append(h)
        out = torch.cat(out_experts_splits, dim=0)
        return out

    def _forward_grouped_mm(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor | None = None):
        # grouped mm implementation
        #
        if num_tokens_per_expert is not None:
            # https://github.com/pytorch/pytorch/pull/150374
            # NOTE: torch._grouped_mm requires bf16 dtypes
            #       and shapes to be multiple of 16
            offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
            if isinstance(x, DTensor) and not isinstance(offsets, DTensor):
                offsets = DTensor.from_local(offsets, x.device_mesh, [Replicate()], run_check=False)
            # grouped mm between a 2D tensor and a 3D tensor
            assert x.dim() == 2
        else:
            offsets = None
            # fall back to regular bmm between 3D tensors
            assert x.dim() == 3

        w1, w2, w3 = (
            self.gate_proj,
            self.down_proj,
            self.up_proj,
        )
        assert (
            x.dtype == w1.dtype == w2.dtype == w3.dtype == torch.bfloat16
        ), "torch._grouped_mm only supports bf16 dtypes"
        h = self.act_fn(torch._grouped_mm(x, w1, offs=offsets))
        h = h * torch._grouped_mm(x, w3, offs=offsets)
        out = torch._grouped_mm(h, w2, offs=offsets)
        
        return out        
    
    # TODO: force no inference mode as a hack to get around
    # "Cannot set version_counter for inference tensor"
    @torch.inference_mode(mode=False)
    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
            num_tokens_per_expert (torch.Tensor): Tensor with shape ``(num_experts,)``
                enumerating the number of tokens each expert receives

        Returns:
            torch.Tensor: tensor with shape ``(bsz * seq_len * experts_per_token, dim)``
        """        
        return self._experts_impl(x, num_tokens_per_expert)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate_proj, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.down_proj, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.up_proj, mean=0.0, std=init_std)


def permute(X: torch.Tensor, gather_indices: torch.Tensor, topk: int):
    """
    Scatters X to a new tensor with shape [total_tokens, hidden_dim] where total_tokens is num_tokens * topk,
    permuting the tokens according to sorted_token_idx.

    Helper for grouped gemm where hidden states need be ordered by expert.
    X: [num_tokens, hidden_dim]
    sorted_token_idx: [num_tokens * topk]
    topk: int

    Returns:
        [total_tokens, hidden_dim]
    """
    assert gather_indices.ndim == 1
    X = X.view(-1, X.shape[-1])
    # Shortcut for topk == 1
    if topk == 1:
        return X[gather_indices]

    return X[gather_indices // topk]


def unpermute(
    X: torch.Tensor,
    gather_indices: torch.Tensor = None,
    scatter_indices: torch.Tensor = None,
):
    assert gather_indices is not None or scatter_indices is not None, (
        "Must pass at least one of gather_indices or scatter_indices"
    )

    X = X.view(-1, X.shape[-1]) if X.ndim > 2 else X
    if scatter_indices is None:
        unpermuted = torch.empty_like(X)
        unpermuted.index_copy_(0, gather_indices, X)
    else:
        unpermuted = X[scatter_indices]

    return unpermuted.view_as(X)


@dataclass(kw_only=True)
class RouterOutputs:
    router_logits: torch.Tensor = None

    # topk outputs [M, topk]
    routing_weights: torch.Tensor = None
    selected_experts: torch.Tensor = None

    # Needed for grouped_gemm
    sorted_routing_weights: torch.Tensor = (
        None  # routing_weights sorted in expert order
    )
    gather_indices: torch.Tensor = (
        None  # indices for permuting hidden states from token -> expert order
    )
    scatter_indices: torch.Tensor = None  # indices for scattering expert -> token order
    num_tokens_per_expert: torch.Tensor = None
    # token_indices_experts_sorted: torch.Tensor = None # gather_indices // topk

class TokenChoiceTopKRouter(nn.Module):
    """This class implements Token Choice routing. In Token Choice top K routing, each token is
        routed to top K experts based on the router scores.

    Args:
        gate (nn.Module): Gate module to calculate the scores, typically nn.Linear(dim, num_experts).
        dim (int): Dimension of input tokens.
        num_experts (int): Number of experts in each moe layer.
        experts_per_token (int): Number of experts each token will be routed to in Token Choice.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_experts: int,
        topk: int,
        score_fn: str = "softmax",
        norm_topk_prob: bool = False,
        use_scatter_indices: bool = False,
    ):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.dim = dim
        self.num_experts = num_experts
        self.topk = topk
        self.score_fn = score_fn
        self.norm_topk_prob = norm_topk_prob
        self.use_scatter_indices = use_scatter_indices

    #TODO: 
    @torch.no_grad
    def calculate_permutation_indices(self, selected_experts: torch.Tensor):
        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        # token_indices_experts_sorted shape (bs*slen*top_k,)
        gather_indices = torch.argsort(selected_experts.view(-1), stable=True)

        scatter_indices = gather_indices.argsort() if self.use_scatter_indices else None

        return num_tokens_per_expert, gather_indices, scatter_indices

    def forward(self, x: torch.Tensor, debug=False, ref_outputs=None) -> RouterOutputs:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.

        Returns:
            routed_input (torch.Tensor):
                Tokens grouped together by experts indices with shape ``(bs*slen*top_k,)``.
            token_indices (torch.Tensor):
                Token indices for routed_input with shape ``(bs*slen*top_k,)``.
            num_tokens_per_expert (torch.Tensor):
                Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """

        # scores shape (bs*slen, num_experts)
        x = x.view(-1, self.dim)

        outputs = RouterOutputs()

        router_logits = self.gate(x)

        if debug:
            outputs.router_logits = router_logits

        # By default, sigmoid is performed in float32 to avoid loss explosion
        if self.score_fn == "softmax":
            routing_weights = nn.functional.softmax(
                router_logits, dim=-1, dtype=torch.float32
            )
        elif self.score_fn == "sigmoid":
            routing_weights = torch.sigmoid(router_logits.to(torch.float32))
        else:
            raise ValueError(
                f"{self.score_fn} not recognized: options are `softmax` or `sigmoid`"
            )

        routing_weights, selected_experts = torch.topk(
            routing_weights, k=self.topk, dim=-1
        )

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keep_dim=True).to(x.dtype)

        # Cast back to original dtype **after** calculating topk, should help with non-deterministic sorting 
        # due to lower precision dtypes (bf16) when testing against HF transformers
        routing_weights = routing_weights.to(x.dtype)

        num_tokens_per_expert, gather_indices, scatter_indices = (
            self.calculate_permutation_indices(selected_experts)
        )

        outputs.routing_weights = routing_weights.view(-1)
        outputs.gather_indices = gather_indices.view(-1)
        outputs.scatter_indices = (
            scatter_indices.view(-1) if self.use_scatter_indices else None
        )
        outputs.num_tokens_per_expert = num_tokens_per_expert

        if debug:
            outputs.selected_experts = selected_experts.view(-1)

        return outputs

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)
