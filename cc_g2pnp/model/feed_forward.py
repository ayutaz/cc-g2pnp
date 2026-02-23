"""Feed-forward module for CC-G2PnP Streaming Conformer-CTC."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from cc_g2pnp.model.config import CC_G2PnPConfig


class FeedForwardModule(nn.Module):
    """Conformer feed-forward module (Gulati et al., 2020).

    Applies LayerNorm, expansion linear, Swish, dropout,
    projection linear, and dropout.

    Input:  [B, T, D]
    Output: [B, T, D]

    The half-step residual connection is applied in ConformerBlock,
    not inside this module.
    """

    def __init__(self, config: CC_G2PnPConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.d_model)
        self.linear1 = nn.Linear(config.d_model, config.d_ff)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape ``[B, T, D]``.

        Returns:
            Output tensor of shape ``[B, T, D]``.
        """
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        return self.dropout2(x)
