# Attention module for Falcon Vision
# Modified for compatibility with PyTorch 2.6 & Ascend NPU (SDPA + Sink Attention fallback)

import torch
import torch.nn.functional as F
from torch import nn
import einops as E

try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        AuxRequest,
        create_block_mask,
        flex_attention,
    )
    HAS_FLEX_ATTN = True
except (ImportError, AttributeError):
    BlockMask = None
    create_block_mask = None
    flex_attention = None
    AuxRequest = None
    HAS_FLEX_ATTN = False

from .rope import apply_rotary_emb, apply_3d_rotary_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads to match query heads."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x.unsqueeze(3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class StandardSinkAttention(nn.Module):
    """
    标准兼容型 Attention：同时计算注意力输出与 LogSumExp (用于 Sink Attention)。
    支持任意 PyTorch 版本，且原生兼容 NVIDIA CUDA 与华为昇腾 NPU。
    """
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ):
        head_dim = q.size(-1)
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        lse = torch.logsumexp(scores, dim=-1)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        output = torch.matmul(probs, v)
        return output, lse


class FlexAttentionWrapper(nn.Module):
    """Wrapper for flex_attention with optional compilation and aux outputs."""

    _compiled = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask=None,
        compile: bool = True,
        return_aux: bool = False,
    ):
        fn = flex_attention
        if compile:
            if FlexAttentionWrapper._compiled is None:
                FlexAttentionWrapper._compiled = torch.compile(
                    flex_attention,
                    mode="max-autotune-no-cudagraphs",
                )
            fn = FlexAttentionWrapper._compiled

        if return_aux:
            return fn(q, k, v, block_mask=block_mask, return_aux=AuxRequest(lse=True))
        else:
            return fn(q, k, v, block_mask=block_mask)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        head_dim: int | None = None,
        use_qk_norm: bool = False,
        enable_3d_rope: bool = False,
        use_flex_attn: bool = True,
        use_sink_attn: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = head_dim or dim // n_heads
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv_heads * self.head_dim

        self.wq = nn.Linear(dim, self.q_dim, bias=False)
        self.wk = nn.Linear(dim, self.kv_dim, bias=False)
        self.wv = nn.Linear(dim, self.kv_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        self.use_qk_norm = use_qk_norm
        self.enable_3d_rope = enable_3d_rope

        self.use_flex_attn = use_flex_attn and HAS_FLEX_ATTN and (AuxRequest is not None)

        self.sink_attn = use_sink_attn
        if self.sink_attn:
            self.sinks = nn.Parameter(torch.empty(n_heads))

        if self.use_flex_attn:
            self.inner_attention = FlexAttentionWrapper()
        else:
            self.inner_attention = StandardSinkAttention()

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.wo.weight)
        if self.sink_attn:
            nn.init.trunc_normal_(self.sinks, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        freqs_cis_2d: torch.Tensor | None = None,
        pos_thw: torch.Tensor | None = None,
        attention_masks=None,
        compile: bool = True,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        if self.use_qk_norm:
            xq = F.rms_norm(xq, (xq.size(-1),))
            xk = F.rms_norm(xk, (xk.size(-1),))

        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq, xk = apply_3d_rotary_emb(xq, xk, freqs_cis, freqs_cis_2d, pos_thw)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.use_flex_attn:
            output, aux = self.inner_attention(
                xq,
                xk,
                xv,
                block_mask=attention_masks,
                compile=compile,
                return_aux=True,
            )
            lse = aux.lse
        else:
            output, lse = self.inner_attention(
                xq,
                xk,
                xv,
                attn_mask=attention_masks if isinstance(attention_masks, torch.Tensor) else None,
            )

        if self.sink_attn:
            sinks_BHL = E.rearrange(self.sinks, "h -> 1 h 1")
            sink_scale = torch.sigmoid(lse - sinks_BHL)
            output = (output * sink_scale.unsqueeze(-1)).to(output.dtype)

        output = E.rearrange(output, "b h s d -> b s (h d)").contiguous()
        return self.wo(output)


def create_attention_mask(
    mask_mod,
    B: int | None,
    H: int | None,
    Q_LEN: int,
    KV_LEN: int,
    BLOCK_SIZE: tuple[int, int] = (64, 64),
):
    """Create a BlockMask for flex_attention."""
    if create_block_mask is None:
        return None
    return create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        BLOCK_SIZE=BLOCK_SIZE,
    )
