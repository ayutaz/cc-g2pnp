"""Chunk-aware streaming multi-head self-attention for CC-G2PnP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

if TYPE_CHECKING:
    from cc_g2pnp.model.config import CC_G2PnPConfig


# ── Mask generation ──────────────────────────────────────────────


def create_chunk_mask(
    seq_len: int,
    chunk_size: int,
    past_context: int,
) -> torch.Tensor:
    """Create a chunk-aware attention mask.

    Each token can attend to:
    - All tokens within its own chunk (bidirectional).
    - Up to *past_context* tokens from previous chunks.

    Args:
        seq_len: Sequence length T.
        chunk_size: Chunk size C.
        past_context: Number of past tokens P from previous chunks.

    Returns:
        Boolean tensor of shape ``[T, T]`` where ``True`` means
        the position is allowed to attend.
    """
    idx = torch.arange(seq_len)
    chunk_starts = (idx // chunk_size) * chunk_size
    chunk_ends = torch.clamp(chunk_starts + chunk_size, max=seq_len)
    attend_starts = torch.clamp(chunk_starts - past_context, min=0)

    # col[j] in [attend_start[i], chunk_end[i])
    col = idx.unsqueeze(0)  # [1, T]
    return (col >= attend_starts.unsqueeze(1)) & (col < chunk_ends.unsqueeze(1))


def create_mla_mask(
    seq_len: int,
    chunk_size: int,
    past_context: int,
    mla_size: int,
) -> torch.Tensor:
    """Create a chunk-aware mask with Minimum Look-Ahead (MLA).

    Extends the standard chunk mask by allowing each token to attend
    *mla_size* tokens beyond the chunk boundary.  Applied only to
    layer 0 in the full model.

    Args:
        seq_len: Sequence length T.
        chunk_size: Chunk size C.
        past_context: Number of past tokens P from previous chunks.
        mla_size: Number of extra look-ahead tokens M beyond chunk end.

    Returns:
        Boolean tensor of shape ``[T, T]``.
    """
    idx = torch.arange(seq_len)
    chunk_starts = (idx // chunk_size) * chunk_size
    chunk_ends = torch.clamp(chunk_starts + chunk_size, max=seq_len)
    attend_starts = torch.clamp(chunk_starts - past_context, min=0)
    attend_ends = torch.clamp(chunk_ends + mla_size, max=seq_len)

    col = idx.unsqueeze(0)  # [1, T]
    return (col >= attend_starts.unsqueeze(1)) & (col < attend_ends.unsqueeze(1))


# ── ChunkAwareAttention ─────────────────────────────────────────


class ChunkAwareAttention(nn.Module):
    """Multi-head self-attention with relative positional bias.

    Designed for chunk-aware streaming Conformer (Noroozi et al.,
    ICASSP 2024).  A pre-computed boolean mask (from
    :func:`create_chunk_mask` or :func:`create_mla_mask`) controls
    the attention span.

    Input:  [B, T, D]
    Output: [B, T, D]
    """

    def __init__(self, config: CC_G2PnPConfig) -> None:
        super().__init__()
        d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = d_model // config.num_heads

        self.norm = nn.LayerNorm(d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        # Relative positional bias: project pos_enc through per-head key
        # to produce query-key distance-dependent bias (Shaw et al., 2018)
        self.pos_k = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.attn_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute chunk-aware multi-head self-attention.

        Args:
            x: Input tensor ``[B, T, D]``.
            pos_enc: Positional encoding ``[1, T, D]``.
            mask: Optional boolean mask ``[T, T]`` (True = attend).

        Returns:
            Output tensor ``[B, T, D]``.
        """
        x = self.norm(x)
        b, t, _ = x.shape

        # Project Q, K, V → [B, H, T, d_k]
        q = self.w_q(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product: [B, H, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Relative positional bias (Shaw et al., 2018):
        # Q * pos_K^T gives query-key distance-dependent bias
        pos_k = self.pos_k(pos_enc)  # [1, T, D]
        pos_k = pos_k.view(1, t, self.num_heads, self.d_k).transpose(1, 2)  # [1, H, T, d_k]
        pos_bias = torch.matmul(q, pos_k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, T, T]
        scores = scores + pos_bias

        # Apply chunk mask
        if mask is not None:
            # mask: [T, T] → [1, 1, T, T]
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum → [B, H, T, d_k] → [B, T, D]
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.dropout(self.w_out(out))
