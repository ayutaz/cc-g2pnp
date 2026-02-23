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
