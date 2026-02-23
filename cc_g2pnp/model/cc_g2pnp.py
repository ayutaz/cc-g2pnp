"""Full CC-G2PnP model: Streaming Conformer-CTC for Japanese G2P."""

from __future__ import annotations

import torch
from torch import nn

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.ctc_decoder import CTCHead, greedy_decode
from cc_g2pnp.model.embedding import TokenEmbedding
from cc_g2pnp.model.encoder import ConformerEncoder


class CC_G2PnP(nn.Module):
    """Streaming Conformer-CTC model for grapheme-to-phoneme conversion.

    Embeds BPE tokens (CALM2), upsamples along time, encodes with a
    self-conditioned Conformer, and produces PnP label sequences via CTC.
    """

    def __init__(self, config: CC_G2PnPConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = CC_G2PnPConfig()
        self.config = config

        self.embedding = TokenEmbedding(config)
        self.encoder = ConformerEncoder(config)
        self.ctc_head = CTCHead(config)

        # Share CTC projection weights between final head and encoder intermediate
        self.ctc_head.projection = self.encoder.ctc_projection

        self.ctc_loss = nn.CTCLoss(
            blank=config.blank_id,
            reduction="mean",
            zero_infinity=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Forward pass with optional CTC loss computation.

        Args:
            input_ids: BPE token IDs ``[B, T]``.
            input_lengths: Valid lengths before padding ``[B]``.
            targets: PnP label IDs ``[B, S]`` (optional, for training).
            target_lengths: Valid target lengths ``[B]`` (optional).

        Returns:
            Dict with keys:
            - ``log_probs``: ``[B, T*upsample_factor, pnp_vocab_size]``
            - ``loss``: scalar (only when targets provided)
            - ``intermediate_losses``: list of scalars (only when targets provided)
        """
        # 1. Embedding: [B, T] -> [B, T*8, D]
        x = self.embedding(input_ids)

        # 2. Encoder: [B, T*8, D] -> {'output': [B, T*8, D], 'intermediate_logits': [...]}
        enc_out = self.encoder(x, input_lengths)

        # 3. CTC Head: [B, T*8, D] -> [B, T*8, V] (log probs)
        log_probs = self.ctc_head(enc_out["output"])

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "log_probs": log_probs,
        }

        # 4. Loss computation (training only)
        if targets is not None and target_lengths is not None:
            upsampled_lengths = input_lengths * self.config.upsample_factor

            # CTCLoss expects [T, B, V]
            log_probs_t = log_probs.transpose(0, 1)
            final_loss = self.ctc_loss(
                log_probs_t, targets, upsampled_lengths, target_lengths,
            )

            # Intermediate CTC losses
            intermediate_losses: list[torch.Tensor] = []
            for inter_logits in enc_out["intermediate_logits"]:
                inter_log_probs = torch.log_softmax(inter_logits, dim=-1)
                inter_log_probs_t = inter_log_probs.transpose(0, 1)
                inter_loss = self.ctc_loss(
                    inter_log_probs_t, targets, upsampled_lengths, target_lengths,
                )
                intermediate_losses.append(inter_loss)

            total_loss = final_loss + self.config.intermediate_ctc_weight * sum(
                intermediate_losses
            )

            result["loss"] = total_loss
            result["intermediate_losses"] = intermediate_losses

        return result

    @torch.no_grad()
    def inference(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> list[list[int]]:
        """Run forward without loss and apply greedy CTC decoding.

        Args:
            input_ids: BPE token IDs ``[B, T]``.
            input_lengths: Valid lengths before padding ``[B]``.

        Returns:
            List of decoded PnP label ID sequences, one per batch element.
        """
        output = self(input_ids, input_lengths)
        return greedy_decode(output["log_probs"], blank_id=self.config.blank_id)
