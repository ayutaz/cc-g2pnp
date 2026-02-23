"""Token embedding with upsampling for CC-G2PnP Streaming Conformer-CTC."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from cc_g2pnp.model.config import CC_G2PnPConfig


class TokenEmbedding(nn.Module):
    """BPE token embedding with time-axis upsampling.

    Maps BPE token IDs to dense vectors, scales by sqrt(d_model),
    and repeats each embedding along the time axis by ``upsample_factor``.

    Input:  [B, T]  (long)
    Output: [B, T * upsample_factor, d_model]  (float)
    """

    def __init__(self, config: CC_G2PnPConfig) -> None:
        super().__init__()
        self.embedding = nn.Embedding(config.bpe_vocab_size, config.d_model)
        self.scale = math.sqrt(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.upsample_factor = config.upsample_factor

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed, scale, dropout, then upsample along time axis.

        Args:
            input_ids: BPE token IDs of shape ``[B, T]``.

        Returns:
            Upsampled embeddings of shape ``[B, T * upsample_factor, d_model]``.
        """
        x = self.embedding(input_ids) * self.scale  # [B, T, D]
        x = self.dropout(x)
        return torch.repeat_interleave(x, repeats=self.upsample_factor, dim=1)
