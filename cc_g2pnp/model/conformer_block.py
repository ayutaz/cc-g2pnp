"""Conformer block assembling FFN-Attention-Conv-FFN with half-step residuals."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from cc_g2pnp.model.config import CC_G2PnPConfig


class ConformerBlock(nn.Module):
    """Single Conformer block (Gulati et al., 2020).

    Architecture::

        x = x + factor * FFN1(x)
        x = x + MHSA(x, pos_enc, mask)
        x = x + Conv(x)
        x = x + factor * FFN2(x)
        x = LayerNorm(x)

    where *factor* is ``config.ff_residual_factor`` (default 0.5).
    """

    def __init__(self, config: CC_G2PnPConfig) -> None:
        super().__init__()
        from cc_g2pnp.model.attention import ChunkAwareAttention
        from cc_g2pnp.model.convolution import ConformerConvModule
        from cc_g2pnp.model.feed_forward import FeedForwardModule

        self.ffn1 = FeedForwardModule(config)
        self.attention = ChunkAwareAttention(config)
        self.conv = ConformerConvModule(config)
        self.ffn2 = FeedForwardModule(config)
        self.final_norm = nn.LayerNorm(config.d_model)

        self.ff_residual_factor = config.ff_residual_factor

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one Conformer block.

        Args:
            x: Input tensor of shape ``[B, T, D]``.
            pos_enc: Positional encoding tensor for relative attention.
            mask: Optional attention mask ``[T, T]``.

        Returns:
            Output tensor of shape ``[B, T, D]``.
        """
        x = x + self.ff_residual_factor * self.ffn1(x)
        x = x + self.attention(x, pos_enc, mask)
        x = x + self.conv(x)
        x = x + self.ff_residual_factor * self.ffn2(x)
        return self.final_norm(x)

    def forward_streaming(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        attn_cache: tuple[torch.Tensor, torch.Tensor],
        conv_cache: torch.Tensor,
        past_context: int,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Streaming forward with caches.

        Args:
            x: Input tensor ``[B, C, D]``.
            pos_enc: Positional encoding ``[1, cache_len+C, D]``.
            attn_cache: KV cache for attention.
            conv_cache: Conv cache ``[B, kernel-1, D]``.
            past_context: Maximum number of cached KV frames.
            mask: Optional attention mask ``[C, cache_len+C]``.

        Returns:
            ``(output [B, C, D], new_attn_cache, new_conv_cache)``
        """
        x = x + self.ff_residual_factor * self.ffn1(x)
        attn_out, new_attn_cache = self.attention.forward_streaming(
            x, pos_enc, attn_cache, past_context, mask,
        )
        x = x + attn_out
        conv_out, new_conv_cache = self.conv.forward_streaming(x, conv_cache)
        x = x + conv_out
        x = x + self.ff_residual_factor * self.ffn2(x)
        return self.final_norm(x), new_attn_cache, new_conv_cache
