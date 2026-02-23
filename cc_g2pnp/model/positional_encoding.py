"""Relative sinusoidal positional encoding for Conformer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from cc_g2pnp.model.config import CC_G2PnPConfig


class RelativePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Transformer-XL / Shaw et al. style).

    Generates a fixed (non-learned) sinusoidal table at init and slices it to
    the input sequence length at forward time.  Used by ChunkAwareAttention
    to compute relative position bias.
    """

    def __init__(self, config: CC_G2PnPConfig, max_len: int = 5000) -> None:
        super().__init__()
        d_model = config.d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # [1, max_len, D]
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(p=config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply dropout to *x* and return sliced positional encoding.

        Args:
            x: Input tensor of shape ``[B, T, D]``.

        Returns:
            ``(dropout(x), pos_enc)`` where *pos_enc* has shape ``[1, T, D]``.
        """
        pos_enc: torch.Tensor = self.pe[:, : x.size(1)]
        return self.dropout(x), pos_enc
