"""Conformer encoder with self-conditioned CTC for CC-G2PnP."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from cc_g2pnp.model.attention import create_chunk_mask, create_mla_mask
from cc_g2pnp.model.conformer_block import ConformerBlock
from cc_g2pnp.model.positional_encoding import RelativePositionalEncoding

if TYPE_CHECKING:
    from cc_g2pnp.model.config import CC_G2PnPConfig


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks with self-conditioned CTC.

    Layer 0 uses an MLA (Minimum Look-Ahead) mask; layers 1+ use the
    standard chunk mask.  At designated intermediate layers, a shared
    CTC projection produces auxiliary logits whose feedback is added
    back into the hidden representation (ESPnet-style self-conditioning).

    Input:  ``[B, T, D]``
    Output: dict with ``'output'`` ``[B, T, D]`` and
            ``'intermediate_logits'`` (list of ``[B, T, V]``).
    """

    def __init__(self, config: CC_G2PnPConfig) -> None:
        super().__init__()
        self.config = config

        self.pos_enc = RelativePositionalEncoding(config)
        self.layers = nn.ModuleList(
            [ConformerBlock(config) for _ in range(config.num_layers)]
        )

        # Shared CTC projection (intermediate + can be reused by final head)
        self.ctc_projection = nn.Linear(config.d_model, config.pnp_vocab_size)
        # Feedback: project CTC logits back to hidden dim
        self.ctc_to_hidden = nn.Linear(config.pnp_vocab_size, config.d_model)

        self._intermediate_ctc_layers = set(config.intermediate_ctc_layers)

    def forward(
        self,
        x: torch.Tensor,
        input_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Run the Conformer encoder stack.

        Args:
            x: Input tensor of shape ``[B, T, D]``.
            input_lengths: Optional lengths (unused; reserved for masking).

        Returns:
            Dict with keys ``'output'`` and ``'intermediate_logits'``.
        """
        seq_len = x.size(1)
        cfg = self.config

        # Pre-compute masks
        mla_mask = create_mla_mask(
            seq_len, cfg.chunk_size, cfg.past_context, cfg.mla_size,
        ).to(x.device)
        chunk_mask = create_chunk_mask(
            seq_len, cfg.chunk_size, cfg.past_context,
        ).to(x.device)

        # Positional encoding
        x, pos_enc = self.pos_enc(x)

        intermediate_logits: list[torch.Tensor] = []

        for i, layer in enumerate(self.layers):
            mask = mla_mask if i == 0 else chunk_mask
            x = layer(x, pos_enc, mask)

            if i in self._intermediate_ctc_layers:
                inter_logits = self.ctc_projection(x)
                intermediate_logits.append(inter_logits)
                x = x + self.ctc_to_hidden(inter_logits)

        return {
            "output": x,
            "intermediate_logits": intermediate_logits,
        }
