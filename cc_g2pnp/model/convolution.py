"""Conformer convolution module with causal (left-only) padding."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

if TYPE_CHECKING:
    from cc_g2pnp.model.config import CC_G2PnPConfig


class ConformerConvModule(nn.Module):
    """Causal convolution module following Gulati et al. (2020).

    Architecture::

        LayerNorm → Pointwise Conv(D→2D) → GLU → Causal Depthwise Conv
        → BatchNorm → SiLU → Pointwise Conv(D→D) → Dropout

    All convolutions use causal (left-only) padding so that the output at
    time *t* depends only on inputs at times ≤ *t*.
    """

    def __init__(self, config: CC_G2PnPConfig) -> None:
        super().__init__()
        d = config.d_model
        k = config.conv_kernel_size
        expanded = d * config.conv_expansion_factor

        self.layer_norm = nn.LayerNorm(d)
        self.pointwise_conv1 = nn.Conv1d(d, expanded, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(d, d, kernel_size=k, groups=d, bias=False)
        self.batch_norm = nn.BatchNorm1d(d)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d, d, kernel_size=1)
        self.dropout = nn.Dropout(config.dropout)

        self._causal_padding = k - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, D)``.

        Returns
        -------
        torch.Tensor
            Output tensor of shape ``(B, T, D)``.
        """
        x = self.layer_norm(x)

        # [B, T, D] → [B, D, T] for Conv1d
        x = x.transpose(1, 2)

        x = self.pointwise_conv1(x)  # [B, 2D, T]
        x = self.glu(x)  # [B, D, T]

        # Causal padding: left-pad only
        x = F.pad(x, (self._causal_padding, 0))
        x = self.depthwise_conv(x)  # [B, D, T]

        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # [B, D, T]
        x = self.dropout(x)

        # [B, D, T] → [B, T, D]
        return x.transpose(1, 2)

    def forward_streaming(
        self, x: torch.Tensor, conv_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Streaming forward using conv_cache instead of zero padding.

        Args:
            x: Current chunk ``[B, C, D]`` where C is chunk_size frames.
            conv_cache: Previous GLU output frames ``[B, kernel-1, D]``.
                On first call, pass zeros.

        Returns:
            ``(output [B, C, D], new_conv_cache [B, kernel-1, D])``
        """
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, D, C]
        x = self.pointwise_conv1(x)  # [B, 2D, C]
        x = self.glu(x)  # [B, D, C]

        # Replace zero padding with conv_cache
        cache_t = conv_cache.transpose(1, 2)  # [B, D, kernel-1]
        x_padded = torch.cat([cache_t, x], dim=2)  # [B, D, kernel-1+C]

        # Update cache: take last kernel-1 frames from GLU output space
        new_cache = x_padded[:, :, -(self._causal_padding):].transpose(1, 2)  # [B, kernel-1, D]

        x = self.depthwise_conv(x_padded)  # [B, D, C] — no extra padding needed
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # [B, D, C]
        x = self.dropout(x)

        return x.transpose(1, 2), new_cache  # [B, C, D], [B, kernel-1, D]
