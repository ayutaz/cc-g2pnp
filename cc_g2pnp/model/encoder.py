"""Conformer encoder with self-conditioned CTC for CC-G2PnP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

from cc_g2pnp.model.attention import create_chunk_mask, create_mla_mask
from cc_g2pnp.model.conformer_block import ConformerBlock
from cc_g2pnp.model.positional_encoding import RelativePositionalEncoding

if TYPE_CHECKING:
    from cc_g2pnp.model.config import CC_G2PnPConfig


@dataclass
class EncoderStreamingState:
    """Per-layer caches for streaming inference."""

    conv_caches: list[torch.Tensor] = field(default_factory=list)
    """[num_layers] each [B, kernel-1, D]."""

    kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    """[num_layers] each (k [B, H, cache_len, d_k], v [B, H, cache_len, d_k])."""

    processed_frames: int = 0
    """Total frames consumed so far.  Incremented by ``chunk_size`` per
    chunk.  Not used internally by the encoder but exposed for external
    callers (e.g. progress tracking, latency measurement)."""


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

    def init_streaming_state(
        self, batch_size: int, device: torch.device,
    ) -> EncoderStreamingState:
        """Create initial empty streaming state."""
        cfg = self.config
        d_k = cfg.d_model // cfg.num_heads

        conv_caches = [
            torch.zeros(batch_size, cfg.conv_kernel_size - 1, cfg.d_model, device=device)
            for _ in range(cfg.num_layers)
        ]
        kv_caches = [
            (
                torch.zeros(batch_size, cfg.num_heads, 0, d_k, device=device),
                torch.zeros(batch_size, cfg.num_heads, 0, d_k, device=device),
            )
            for _ in range(cfg.num_layers)
        ]
        return EncoderStreamingState(
            conv_caches=conv_caches,
            kv_caches=kv_caches,
            processed_frames=0,
        )

    def forward_streaming(
        self,
        chunk_frames: torch.Tensor,
        state: EncoderStreamingState,
    ) -> tuple[dict[str, torch.Tensor], EncoderStreamingState]:
        """Process a chunk through all layers with streaming caches.

        Layer 0 receives ``chunk_size + mla_size`` frames to support MLA
        look-ahead; after layer 0 the output is trimmed to ``chunk_size``
        frames. Layers 1+ process only ``chunk_size`` frames.

        Args:
            chunk_frames: ``[B, chunk_size + mla_size, D]`` (already
                embedded + upsampled).
            state: Current streaming state.

        Returns:
            ``({"output": [B, chunk_size, D]}, updated_state)``
        """
        cfg = self.config
        chunk_size = cfg.chunk_size

        new_conv_caches: list[torch.Tensor] = []
        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []

        x = chunk_frames

        for i, layer in enumerate(self.layers):
            # Positional encoding covers cached + current frames
            kv_len = state.kv_caches[i][0].size(2)
            current_len = x.size(1)
            total_len = kv_len + current_len

            # Use positions [0, total_len) — relative positional bias
            # depends on Q-K distance, not absolute position
            pos_enc: torch.Tensor = self.pos_enc.pe[:, :total_len]

            # No mask needed in streaming: Q attends to all cached + current
            x, new_attn_cache, new_conv_cache = layer.forward_streaming(
                x, pos_enc, state.kv_caches[i], state.conv_caches[i],
                cfg.past_context, mask=None,
            )

            new_kv_caches.append(new_attn_cache)
            new_conv_caches.append(new_conv_cache)

            # After layer 0, trim MLA look-ahead frames
            if i == 0 and x.size(1) > chunk_size:
                x = x[:, :chunk_size, :]

            # Intermediate CTC at designated layers
            if i in self._intermediate_ctc_layers:
                inter_logits = self.ctc_projection(x)
                x = x + self.ctc_to_hidden(inter_logits)

        updated_state = EncoderStreamingState(
            conv_caches=new_conv_caches,
            kv_caches=new_kv_caches,
            processed_frames=state.processed_frames + chunk_size,
        )

        return {"output": x}, updated_state
