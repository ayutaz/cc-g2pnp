"""Full CC-G2PnP model: Streaming Conformer-CTC for Japanese G2P."""

from __future__ import annotations

import torch
from torch import nn

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.ctc_decoder import CTCHead, greedy_decode
from cc_g2pnp.model.embedding import TokenEmbedding
from cc_g2pnp.model.encoder import ConformerEncoder, EncoderStreamingState


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

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Enable/disable gradient checkpointing for memory-efficient training."""
        self.encoder.set_gradient_checkpointing(enabled)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
        targets: torch.Tensor | None = None,
        target_lengths: torch.Tensor | None = None,
        enable_intermediate_ctc: bool = True,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """Forward pass with optional CTC loss computation.

        Args:
            input_ids: BPE token IDs ``[B, T]``.
            input_lengths: Valid lengths before padding ``[B]``.
            targets: PnP label IDs ``[B, S]`` (optional, for training).
            target_lengths: Valid target lengths ``[B]`` (optional).
            enable_intermediate_ctc: If False, skip intermediate CTC projections
                to save compute after early training stages.

        Returns:
            Dict with keys:
            - ``log_probs``: ``[B, T*upsample_factor, pnp_vocab_size]``
            - ``loss``: scalar (only when targets provided)
            - ``intermediate_losses``: list of scalars (only when targets provided)
        """
        # 1. Embedding: [B, T] -> [B, T*8, D]
        x = self.embedding(input_ids)

        # 2. Encoder: [B, T*8, D] -> {'output': [B, T*8, D], 'intermediate_logits': [...]}
        enc_out = self.encoder(x, input_lengths, enable_intermediate_ctc=enable_intermediate_ctc)

        # 3. CTC Head: [B, T*8, D] -> [B, T*8, V] (log probs)
        log_probs = self.ctc_head(enc_out["output"])

        result: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "log_probs": log_probs,
        }

        # 4. Loss computation (training only)
        if targets is not None and target_lengths is not None:
            upsampled_lengths = input_lengths * self.config.upsample_factor

            # CTCLoss expects [T, B, V]
            log_probs_t = log_probs.transpose(0, 1).contiguous()
            final_loss = self.ctc_loss(
                log_probs_t, targets, upsampled_lengths, target_lengths,
            )

            # Batch intermediate CTC losses into a single CTC call to reduce overhead
            intermediate_losses: list[torch.Tensor] = []
            inter_logits_list = enc_out["intermediate_logits"]
            if inter_logits_list:
                n_inter = len(inter_logits_list)
                # Stack [N, B, T, V] then flatten to [N*B, T, V] for a single CTC call
                stacked = torch.stack(inter_logits_list, dim=0)
                flat = stacked.reshape(-1, stacked.size(2), stacked.size(3))
                flat_log_probs_t = torch.log_softmax(flat, dim=-1).transpose(0, 1).contiguous()
                targets_rep = targets.repeat(n_inter, 1)
                up_rep = upsampled_lengths.repeat(n_inter)
                tgt_rep = target_lengths.repeat(n_inter)
                # Single CTC call (reduction="none") -> [N*B] per-sample losses
                per_sample = torch.nn.functional.ctc_loss(
                    flat_log_probs_t, targets_rep, up_rep, tgt_rep,
                    blank=self.config.blank_id, reduction="none", zero_infinity=True,
                )
                # Split by layer, normalize by target length to match reduction="mean"
                per_layer = per_sample.view(n_inter, -1)  # [N, B]
                tgt_len_f = target_lengths.float()
                intermediate_losses = [(per_layer[i] / tgt_len_f).mean() for i in range(n_inter)]

            total_loss = final_loss + self.config.intermediate_ctc_weight * sum(
                intermediate_losses
            )

            result["loss"] = total_loss
            result["intermediate_losses"] = intermediate_losses

        return result

    @torch.inference_mode()
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

    def init_streaming_state(self, batch_size: int) -> EncoderStreamingState:
        """Create initial empty streaming state.

        Args:
            batch_size: Batch size for the streaming session.

        Returns:
            Initial encoder streaming state with zeroed caches.
        """
        device = next(self.parameters()).device
        return self.encoder.init_streaming_state(batch_size, device)

    @torch.inference_mode()
    def forward_streaming(
        self,
        chunk_frames: torch.Tensor,
        state: EncoderStreamingState,
    ) -> tuple[torch.Tensor, EncoderStreamingState]:
        """Process pre-embedded chunk frames through encoder + CTC head.

        Args:
            chunk_frames: ``[B, chunk_size + mla_size, D]``
                (already embedded and upsampled).
            state: Current encoder streaming state.

        Returns:
            ``(log_probs [B, chunk_size, V], updated_state)``
        """
        enc_out, new_state = self.encoder.forward_streaming(chunk_frames, state)
        log_probs = self.ctc_head(enc_out["output"])
        return log_probs, new_state
