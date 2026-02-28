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
        self.use_flash_attention = config.use_flash_attention
        self.chunk_size = config.chunk_size
        self.past_context = config.past_context
        self.mla_size = config.mla_size

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

        Dispatches to ``_forward_chunk_sdpa`` (chunk-level SDPA, Phase 2) when
        ``use_flash_attention=True``, otherwise falls back to ``_forward_manual``.

        Args:
            x: Input tensor ``[B, T, D]``.
            pos_enc: Positional encoding ``[1, T, D]``.
            mask: Optional boolean mask ``[T, T]`` (True = attend).

        Returns:
            Output tensor ``[B, T, D]``.
        """
        if self.use_flash_attention:
            return self._forward_chunk_sdpa(x, pos_enc, mask)
        return self._forward_manual(x, pos_enc, mask)

    def _forward_manual(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Manual matmul+softmax attention (original implementation)."""
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

    def _forward_sdpa(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """F.scaled_dot_product_attention path (FlashAttention / EFFICIENT_ATTENTION kernel).

        Numerically equivalent to ``_forward_manual``:
        - pos_bias is pre-scaled by 1/sqrt(d_k) and passed as attn_mask
        - SDPA internally scales Q@K^T by 1/sqrt(d_k)
        - Result: softmax(Q@K^T/sqrt(d_k) + pos_bias + chunk_mask) @ V
        """
        x = self.norm(x)
        b, t, _ = x.shape

        # Project Q, K, V → [B, H, T, d_k]
        q = self.w_q(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)

        # Relative positional bias (pre-scaled by 1/sqrt(d_k)): [B, H, T, T]
        pos_k = self.pos_k(pos_enc)  # [1, T, D]
        pos_k = pos_k.view(1, t, self.num_heads, self.d_k).transpose(1, 2)  # [1, H, T, d_k]
        pos_bias = torch.matmul(q, pos_k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Build float attn_mask: pos_bias + -inf where mask=False
        if mask is not None:
            inv_mask = ~mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]
            attn_mask = pos_bias.masked_fill(inv_mask, float("-inf"))
        else:
            attn_mask = pos_bias

        # SDPA: softmax(Q@K^T/sqrt(d_k) + attn_mask) @ V
        dp = self.attn_dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dp)

        # Reshape: [B, H, T, d_k] → [B, T, D]
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.dropout(self.w_out(out))

    def _forward_chunk_sdpa(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Chunk-level SDPA: O(T x (C+P+M)) memory instead of O(T^2).

        Processes each chunk independently to avoid materializing the full TxT matrix:
        - Q from chunk i (C tokens)
        - K,V from the window [chunk_start - past_context, chunk_end + mla_size)

        When *mask* is ``None`` the full sequence is used as the KV window so that
        results are numerically equivalent to ``_forward_sdpa``.

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

        # Positional key projections → [1, H, T, d_k]
        pos_k = self.pos_k(pos_enc)
        pos_k = pos_k.view(1, t, self.num_heads, self.d_k).transpose(1, 2)

        outputs: list[torch.Tensor] = []
        n_chunks = (t + self.chunk_size - 1) // self.chunk_size

        for i in range(n_chunks):
            q_start = i * self.chunk_size
            q_end = min(q_start + self.chunk_size, t)

            if mask is not None:
                # Restrict KV to past context + own chunk + MLA look-ahead
                kv_start = max(0, q_start - self.past_context)
                kv_end = min(q_end + self.mla_size, t)
            else:
                # No mask: use full sequence so output matches _forward_sdpa
                kv_start = 0
                kv_end = t

            q_chunk = q[:, :, q_start:q_end]       # [B, H, C, d_k]
            k_chunk = k[:, :, kv_start:kv_end]     # [B, H, W, d_k]
            v_chunk = v[:, :, kv_start:kv_end]     # [B, H, W, d_k]
            pk_chunk = pos_k[:, :, kv_start:kv_end]  # [1, H, W, d_k]

            # Positional bias for this window: [B, H, C, W]
            pos_bias = torch.matmul(q_chunk, pk_chunk.transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_mask = pos_bias

            if mask is not None:
                mask_slice = mask[q_start:q_end, kv_start:kv_end]  # [C, W]
                attn_mask = attn_mask.masked_fill(~mask_slice.unsqueeze(0).unsqueeze(0), float("-inf"))

            dp = self.attn_dropout.p if self.training else 0.0
            out_chunk = F.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk, attn_mask=attn_mask, dropout_p=dp)
            outputs.append(out_chunk)

        # Concatenate chunks: [B, H, T, d_k] → [B, T, D]
        out = torch.cat(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.dropout(self.w_out(out))

    def forward_streaming(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        past_context: int,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Streaming forward with KV cache.

        Args:
            x: Current chunk ``[B, C, D]`` (pre-norm applied internally).
            pos_enc: Positional encoding ``[1, cache_len+C, D]``.
            kv_cache: ``(k_cache [B, H, cache_len, d_k],
                v_cache [B, H, cache_len, d_k])``.
            past_context: Maximum number of cached frames to keep.
            mask: Optional mask ``[C, cache_len+C]``.

        Returns:
            ``(output [B, C, D], new_kv_cache)``
        """
        x_normed = self.norm(x)
        b, c, _ = x_normed.shape

        # Q from current chunk only
        q = self.w_q(x_normed).view(b, c, self.num_heads, self.d_k).transpose(1, 2)

        # K, V from current chunk
        k_new = self.w_k(x_normed).view(b, c, self.num_heads, self.d_k).transpose(1, 2)
        v_new = self.w_v(x_normed).view(b, c, self.num_heads, self.d_k).transpose(1, 2)

        # Concatenate with cache
        k_cache, v_cache = kv_cache
        k = torch.cat([k_cache, k_new], dim=2)  # [B, H, cache_len+C, d_k]
        v = torch.cat([v_cache, v_new], dim=2)  # [B, H, cache_len+C, d_k]

        # Attention scores: Q attends to all of K
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Relative positional bias
        total_len = k.size(2)
        pos_k = self.pos_k(pos_enc)  # [1, total_len, D]
        pos_k = pos_k.view(1, total_len, self.num_heads, self.d_k).transpose(1, 2)
        pos_bias = torch.matmul(q, pos_k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + pos_bias

        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, C, d_k]
        out = out.transpose(1, 2).contiguous().view(b, c, -1)

        # Update KV cache: keep last past_context frames
        new_k = k[:, :, -past_context:] if k.size(2) > past_context else k
        new_v = v[:, :, -past_context:] if v.size(2) > past_context else v

        return self.dropout(self.w_out(out)), (new_k, new_v)
