from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.distributed.tensor import DTensor


from .moe import GroupedExperts, TokenChoiceTopKRouter, permute, unpermute, RouterOutputs
from torchtitan.experiments.kernels.moe.indices import (
    generate_permute_indices,
)

from .args import Qwen3MoeConfig
from .attention import build_attention, init_attention_mask


class Qwen3MoeRMSNorm(nn.Module):
    """
    Root Mean Square Normalization in fp32.

    See: https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

    Args:
        dim (int): embedding size
        eps (float): small value to avoid division by zero. Default: 1e-6
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor to normalize

        Returns:
            torch.Tensor: The normalized and scaled tensor having the same shape as ``x``.
        """
        # computation is in fp32
        x_fp32 = x.float()
        x_normed = (x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(
            x
        )
        return x_normed * self.scale

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

# TODO: add aux_routing loss
class Qwen3MoeSparseMoeBlock(nn.Module):
    """
    Qwen3 Moe Experts Block

    Uses token choice routing.

    Args:
        config: Qwen3MoeConfig
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()

        # Configure experts
        num_experts = config.num_experts
        hidden_dim, intermediate_dim = config.hidden_size, config.moe_intermediate_size
        act_fn = config.act_fn
        self.use_grouped_gemm = config.use_grouped_gemm
        self.experts = GroupedExperts(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim,
            num_experts=num_experts,
            act_fn=act_fn,
            use_grouped_gemm=self.use_grouped_gemm,
        )

        # Configure router
        topk = config.num_experts_per_tok
        score_fn = config.score_fn  # either softmax or sigmoid
        norm_topk_prob = config.norm_topk_prob
        use_scatter_indices = config.use_scatter_indices  # for permuting from expert -> token order

        self.router = TokenChoiceTopKRouter(
            dim=hidden_dim,
            num_experts=num_experts,
            topk=topk,
            score_fn=score_fn,
            norm_topk_prob=norm_topk_prob,
            use_scatter_indices=use_scatter_indices,
        )
        self.shared_expert = None  # Qwen3 MoE does not use shared experts
        self.topk = topk

        # define fields for auxiliary-loss-free load balancing (https://arxiv.org/abs/2408.15664)
        # NOTE: tokens_per_expert is accumulated in the model forward pass.
        #       expert_bias is updated outside the model in an optimzer step pre hook
        #       to work with gradient accumulation.
        self.load_balance_coeff = config.router_aux_loss_coef
        if self.load_balance_coeff is not None:
            assert self.load_balance_coeff > 0.0
            self.register_buffer(
                "expert_bias",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "tokens_per_expert",
                torch.zeros(num_experts, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.expert_bias = None

    def _permute_and_pad(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        experts_per_ep_rank: int,
        num_ep_ranks: int,
    ):
        ALIGN_SIZE_M = 16
        with torch.no_grad():
            (
                permuted_indices,
                num_tokens_per_expert,
                _,  # offsets,
            ) = generate_permute_indices(
                num_tokens_per_expert,
                experts_per_ep_rank,
                num_ep_ranks,
                x.shape[0] + experts_per_ep_rank * ALIGN_SIZE_M,
                ALIGN_SIZE_M,
            )

        x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
        padded_shape = x.shape
        x = x[permuted_indices, :]

        return x, padded_shape, permuted_indices, num_tokens_per_expert

    # 1. ensure all tensors are Tensor and not DTensor -- needed for expert parallel sharding
    # 2. pad align x if use_grouped_gemm -- add guard as this is expensive
    # 3. ensure inputs are ordered by expert
    # 2 + 3 are taken care of by generate_permute_indices kernel
    # 3 uses simple permute for now

    # TODO: only generate gather_indices / scatter indices in router if not using grouped gemm
    # Replace grouped_gemm with more efficient implementation, entire function is very ugly right now
    def run_experts(self, x: torch.Tensor, router_outputs: RouterOutputs):
        assert x.ndim == 2, f"Expected ndim = 2, got x with shape {x.shape}"
        M, dim = x.shape
        topk = self.topk
        w1, w2, w3 = self.experts.gate_proj, self.experts.down_proj, self.experts.up_proj

        # Needed for expert parallel sharding
        if isinstance(w1, DTensor):
            w1 = w1.to_local()
            w2 = w2.to_local()
            w3 = w3.to_local()

        if not self.use_grouped_gemm:
            x = permute(x, router_outputs.gather_indices, topk=topk)
            assert x.shape == torch.Size([M * topk, dim])
            num_tokens_per_expert = router_outputs.num_tokens_per_expert
        else:
            experts_per_ep_rank = w1.shape[0]
            num_ep_ranks = router_outputs.num_tokens_per_expert.shape[0] // experts_per_ep_rank
            x, padded_shape, permute_indices, num_tokens_per_expert = self._permute_and_pad(
                x,
                num_tokens_per_expert=router_outputs.num_tokens_per_expert,
                experts_per_ep_rank=experts_per_ep_rank,
                num_ep_ranks=num_ep_ranks,
            )

        x = self.experts(x, num_tokens_per_expert)

        # sort back to token order
        if not self.use_grouped_gemm:
            x = unpermute(
                x,
                gather_indices=router_outputs.gather_indices,
                scatter_indices=router_outputs.scatter_indices,
            )
        else:
            x = x.new_empty(padded_shape)
            x[permute_indices, :] = x
            x = x[:-1]

        # Broadcasted mul by router weights -> [M * topk] -> [M * topk, 1]
        x = x * router_outputs.routing_weights[:, None]

        # Reduce across tokens per expert
        x = x.reshape(M, topk, dim).sum(dim=1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, seqlen, dim)``.

        Returns:
            out (torch.Tensor): Output tensor with shape ``(bs, seqlen, dim)``.
        """
        bs, seqlen, dim = x.shape
        topk = self.topk
        M = bs * seqlen
        x = x.view(-1, dim)  # (M, dim)

        router_outputs: RouterOutputs = self.router(x)
        assert router_outputs.gather_indices.shape == torch.Size([M * topk]), (
            f"{router_outputs.gather_indices.shape} != {torch.Size(M * topk)}"
        )

        # tokens_per_expert will be used to update the expert bias for load balancing.
        # Prevent extra local tokens accumulation on evaluation or activation recomputation.
        if self.load_balance_coeff is not None and torch.is_grad_enabled():
            with torch.no_grad():
                self.tokens_per_expert.add_(router_outputs.num_tokens_per_expert)


        x = self.run_experts(x, router_outputs=router_outputs)

        return x.view(bs, seqlen, dim)

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device,
    ):
        self.experts.init_weights(init_std)
        self.router.init_weights(init_std)
        
        if self.load_balance_coeff is not None:
            with torch.device(buffer_device):
                self.expert_bias = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )
                self.tokens_per_expert = torch.zeros(
                    self.experts.num_experts, dtype=torch.float32
                )


# for debugging only
@dataclass(kw_only=True)
class Qwen3MoeAttentionOutputs:
    # q_proj(hidden_states)
    q_proj: torch.Tensor = None
    # rms(q_proj)
    q_proj_norm: torch.Tensor = None
    # ditto
    k_proj: torch.Tensor = None
    k_proj_norm: torch.Tensor = None
    v_proj: torch.Tensor = None

    # apply_rotary_emb(q_proj_norm)
    q_rot: torch.Tensor = None
    # ditto
    k_rot: torch.Tensor = None

    # output of attention pre out proj
    attn_out: torch.Tensor = None
    # attn_weights: torch.Tensor = None

    # final output, after out proj
    attn_out_proj: torch.Tensor = None


class Qwen3MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_seq_len
        self.theta = config.rope_theta
        self.dim = config.head_dim

        self.config = config

        inv_freq = self._compute_rope_params(self.theta, self.dim, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_rope_params(
        self,
        theta: float,
        dim: int,
        device: Optional["torch.device"] = None,
    ):
        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float)
                / dim
            )
        )
        return inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin, unsqueeze_dim=2):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(
#         batch, num_key_value_heads, n_rep, slen, head_dim
#     )
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen3MoeAttention(nn.Module):
    """
    Same as torchtune's MultiHeadAttention except for the order of normalization:
    - torchtune normalizes after position embedding
    - this implementation normalizes after projection, before embedding to align more closely with HF's
    attention implementation.

    """

    def __init__(
        self,
        model_config: Qwen3MoeConfig,
        # rope_embeddings: Qwen3RoPE,
        is_causal: bool = True,
    ) -> None:
        super().__init__()
        num_heads = model_config.num_attention_heads
        num_kv_heads = model_config.num_key_value_heads
        hidden_size = model_config.hidden_size
        head_dim = model_config.head_dim
        self.config = model_config
        # self.rope_embeds = rope_embeddings

        assert head_dim == hidden_size // num_heads, (
            f"head_dim does not match hidden_size // num_heads: {head_dim} != {hidden_size // num_heads}"
        )

        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
            )

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embed_dim ({hidden_size}) must be divisible by num_heads ({num_heads})"
            )
        assert model_config.q_norm and model_config.k_norm, "Qwen3Moe uses q_norm and k_norm"

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_repeats = num_heads // num_kv_heads
        self.hidden_size = hidden_size
        self.attn_dropout = model_config.attention_dropout
        self.head_dim = model_config.head_dim
        self.max_seq_len = model_config.max_seq_len
        self.is_causal = is_causal

        # Set layers
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.sdpa = build_attention(model_config.use_flex_attention, model_config.attn_mask_type)

        self.q_norm = Qwen3MoeRMSNorm(dim=head_dim, eps=model_config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(dim=head_dim, eps=model_config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: tuple[torch.Tensor, torch.Tensor],
        input_pos: torch.Tensor = None,
        debug: bool = False,
    ) -> torch.Tensor | Qwen3MoeAttentionOutputs:
        """ """
        cos, sin = rope_freqs
        if debug:
            outputs = Qwen3MoeAttentionOutputs()

        bs, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Use -1 instead of `n_heads` (or `n_kv_heads`) to infer the actual
        # local heads from sizes of xq, xk, and xv as TP may have sharded them
        # after the above linear ops.
        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if debug:
            outputs.q_proj = xq.clone()
            outputs.k_proj = xk.clone()
            outputs.v_proj = xv.clone()

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = xq.transpose(1, 2)  # bs, num_heads, seqlen, head_dim
        xk = xk.transpose(1, 2)  # bs, num_kv_heads, seqlen, head_dim
        xv = xv.transpose(1, 2)  # bs, num_kv_heads, seqlen, head_dim

        if debug:
            outputs.q_proj_norm = xq.clone()
            outputs.k_proj_norm = xk.clone()

        xq, xk = apply_rotary_emb(
            xq, xk, cos, sin, unsqueeze_dim=1
        )  # unsqueeze_dim should be n_heads dim

        if debug:
            outputs.q_rot = xq.clone()
            outputs.k_rot = xk.clone()

        # repeat k/v heads if n_kv_heads < n_heads
        if self.num_repeats > 1:
            xk = xk.repeat_interleave(self.num_repeats, 1)  # (bs, n_local_heads, seqlen, head_dim)
            xv = xv.repeat_interleave(self.num_repeats, 1)  # (bs, n_local_heads, seqlen, head_dim)

        output = self.sdpa(xq, xk, xv)

        output = output.transpose(1, 2).contiguous()  # (bs, seqlen, n_local_heads, head_dim)
        output = output.view(bs, seqlen, -1)

        if debug:
            outputs.attn_out = output.clone()

        output = self.o_proj(output)

        if debug:
            outputs.attn_out_proj = output
            return outputs
        else:
            return output

    def init_weights(self, init_std: float):
        for linear in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.o_proj.weight, mean=0.0, std=init_std)


class Qwen3MoeDecoderLayer(nn.Module):
    """

    Args:
        layer_id (int): Identifier for the layer.
        model_config: Qwen3MoeConfig.

    Attributes:
        n_heads (int): Number of attention heads.
        dim (int): Dimension size of the model.
        head_dim (int): Dimension size of each attention head.
        attention (Attention): Attention module.
        feed_forward (FeedForward): FeedForward module.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization for attention output.
        ffn_norm (RMSNorm): Layer normalization for feedforward output.

    """

    def __init__(self, model_config: Qwen3MoeConfig, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = model_config.num_attention_heads
        self.dim = model_config.hidden_size
        self.self_attn = Qwen3MoeAttention(model_config)
        self.use_moe = model_config.use_moe
        
        if self.use_moe:
            self.mlp = Qwen3MoeSparseMoeBlock(model_config)
        else:
            raise ValueError("MoE only for now")

        self.attn_norm = Qwen3MoeRMSNorm(self.dim, eps=model_config.rms_norm_eps)
        self.mlp_norm = Qwen3MoeRMSNorm(self.dim, eps=model_config.rms_norm_eps)
        
        #depthwise layer init
        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_freqs: torch.Tensor,
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        residual = x
        x = self.attn_norm(x)
        x = self.self_attn(x, rope_freqs=rope_freqs)
        x = x + residual

        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + residual

        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attn_norm, self.mlp_norm):
            norm.reset_parameters()
        self.self_attn.init_weights(self.weight_init_std)
        self.mlp.init_weights(self.weight_init_std, buffer_device)


class Qwen3MoeModel(nn.Module):
    def __init__(self, model_config: Qwen3MoeConfig):
        super().__init__()
        self.model_config = model_config
        self.vocab_size = model_config.vocab_size
        self.n_layers = model_config.num_hidden_layers
        #        self.eos_id = model_config.eos_id
        hidden_size = model_config.hidden_size
        vocab_size = model_config.vocab_size

        self.tok_embeddings = nn.Embedding(vocab_size, hidden_size)

        self.rope_embeds = Qwen3MoeRotaryEmbedding(model_config)
        self.layers = torch.nn.ModuleList()
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(model_config, layer_idx)
                for layer_idx in range(model_config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3MoeRMSNorm(hidden_size, eps=model_config.rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.cached_position_ids = None

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor = None,
        input_batch: torch.Tensor = None,
        output_hidden_states: bool = False,
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            input_batch (torch.Tensor): The input batch read from the dataloader.
                This will always be the input batch regardless of the pipeline stage.
                This field is required for non-first PP stages to perform document
                masking attention (to analyze the boundary of the document).

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        if self.model_config.use_flex_attention:
            raise ValueError("flex attention not yet supported")
            # init_attention_mask(
            #     input_batch if input_batch is not None else tokens, eos_id=self.eos_id
            # )

        # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens
        bs, seqlen, dim = h.shape
        position_ids = position_ids or self.cached_position_ids

        if position_ids is None or position_ids.shape[1] != seqlen:
            position_ids = torch.arange(seqlen, device=h.device).unsqueeze(0)
            self.cached_position_ids = position_ids
        assert position_ids.shape == torch.Size((1, seqlen)), (
            f"position_ids shape != (1, seqlen): {position_ids.shape} != (1, {seqlen})"
        )

        rope_freqs = self.rope_embeds.forward(h, position_ids=position_ids)
        if output_hidden_states:
            hidden_states = []

        for layer in self.layers:
            hidden_states.append(h)
            h = layer(h, rope_freqs=rope_freqs)

        h = self.norm(h) if self.norm else h

        if output_hidden_states:
            hidden_states.append(h)

        output = self.lm_head(h) if self.lm_head else h
        return hidden_states, output if output_hidden_states else output

    def init_weights(
        self,
        buffer_device: torch.device | None = None,
    ):
        """
        [Note: On ``init_weights`` vs. ``reset_parameters``]
        Modules may define ``reset_parameters`` to initialize parameter values.
        ``reset_parameters`` is meant to only initialize directly owned
        parameters/buffers, not those of their child modules, and it can be
        used to give the initial values for these tensors.
        Separately, users may want custom initialization for their modules,
        different from that in ``reset_parameters``. For this, we define
        ``init_weights``. We only call it in the constructor of this
        ``Transformer`` root module to avoid reinitializing tensors.
        """
        buffer_device = buffer_device
        
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        
        for layer in self.layers:
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        
        if self.norm is not None:
            self.norm.reset_parameters()
        
        final_out_std = self.model_config.hidden_size ** -0.5
        cutoff_factor = 3
        if self.lm_head is not None:
            nn.init.trunc_normal_(
                self.lm_head.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )
