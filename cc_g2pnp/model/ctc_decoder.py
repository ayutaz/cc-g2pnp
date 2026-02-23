"""CTC output head and greedy decoder for CC-G2PnP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from cc_g2pnp.model.config import CC_G2PnPConfig


class CTCHead(nn.Module):
    """Linear projection + log-softmax for CTC output.

    Input:  [B, T, d_model]
    Output: [B, T, pnp_vocab_size]  (log probabilities)
    """

    def __init__(self, config: CC_G2PnPConfig) -> None:
        super().__init__()
        self.projection = nn.Linear(config.d_model, config.pnp_vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project encoder output to log probabilities.

        Args:
            x: Encoder output of shape ``[B, T, d_model]``.

        Returns:
            Log probabilities of shape ``[B, T, pnp_vocab_size]``.
        """
        return torch.log_softmax(self.projection(x), dim=-1)


def greedy_decode(
    log_probs: torch.Tensor,
    blank_id: int = 0,
) -> list[list[int]]:
    """CTC greedy decoding: argmax, collapse repeats, remove blanks.

    Args:
        log_probs: Log probabilities of shape ``[B, T, V]``.
        blank_id: CTC blank token ID.

    Returns:
        List of decoded token ID sequences, one per batch element.
    """
    best = log_probs.argmax(dim=-1)  # [B, T]
    results: list[list[int]] = []
    for seq in best:
        collapsed = torch.unique_consecutive(seq)
        tokens = [t.item() for t in collapsed if t.item() != blank_id]
        results.append(tokens)
    return results
