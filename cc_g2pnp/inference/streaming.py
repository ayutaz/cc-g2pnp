"""High-level streaming inference engine for CC-G2PnP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from cc_g2pnp.model.ctc_decoder import greedy_decode

if TYPE_CHECKING:
    from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
    from cc_g2pnp.model.encoder import EncoderStreamingState


@dataclass
class StreamingState:
    """Complete state for streaming inference session."""

    encoder_state: EncoderStreamingState
    """Per-layer encoder caches."""

    frame_buffer: torch.Tensor
    """Buffered upsampled frames not yet forming a complete chunk [B, ?, D]."""


class StreamingInference:
    """High-level wrapper for chunk-by-chunk streaming inference.

    Usage::

        engine = StreamingInference(model)
        state = engine.reset(batch_size=1)

        # Feed tokens as they arrive from the LLM
        labels, state = engine.process_tokens(bpe_ids, state)
    """

    def __init__(self, model: CC_G2PnP) -> None:
        self.model = model
        self.model.eval()
        self.config = model.config

    def reset(self, batch_size: int = 1) -> StreamingState:
        """Initialize streaming state.

        Args:
            batch_size: Number of concurrent sequences.

        Returns:
            Fresh streaming state with empty caches and buffers.
        """
        device = next(self.model.parameters()).device
        encoder_state = self.model.init_streaming_state(batch_size)
        frame_buffer = torch.zeros(
            batch_size, 0, self.config.d_model, device=device,
        )
        return StreamingState(
            encoder_state=encoder_state,
            frame_buffer=frame_buffer,
        )

    @torch.inference_mode()
    def process_tokens(
        self,
        bpe_token_ids: torch.Tensor,
        state: StreamingState,
    ) -> tuple[list[list[int]], StreamingState]:
        """Feed BPE tokens and get decoded PnP labels for any complete chunks.

        Args:
            bpe_token_ids: ``[B, N]`` where N can be any number of tokens.
            state: Current streaming state.

        Returns:
            ``(decoded_labels, new_state)`` where ``decoded_labels`` is the
            greedy-decoded PnP output for all newly processed chunks.
        """
        cfg = self.config
        chunk_size = cfg.chunk_size
        mla_size = cfg.mla_size
        frames_needed = chunk_size + mla_size

        # 1. Embed + upsample: [B, N] -> [B, N*upsample_factor, D]
        new_frames = self.model.embedding(bpe_token_ids)

        # 2. Append to frame_buffer
        frame_buffer = torch.cat([state.frame_buffer, new_frames], dim=1)

        # 3. Process complete chunks
        all_log_probs: list[torch.Tensor] = []
        encoder_state = state.encoder_state

        while frame_buffer.size(1) >= frames_needed:
            # Extract chunk_size + mla_size frames for layer 0 MLA
            chunk = frame_buffer[:, :frames_needed, :]

            # Add sinusoidal PE to chunk and apply dropout.  In eval mode
            # dropout is a no-op; the second return value (raw PE table)
            # is unused because encoder.forward_streaming reads the PE
            # buffer directly for relative positional bias.
            chunk, _ = self.model.encoder.pos_enc(chunk)

            # Process through encoder + CTC head
            log_probs, encoder_state = self.model.forward_streaming(
                chunk, encoder_state,
            )
            all_log_probs.append(log_probs)

            # Advance buffer: consume only chunk_size frames (MLA frames reprocessed)
            frame_buffer = frame_buffer[:, chunk_size:, :]

        # 4. Greedy decode accumulated log_probs
        if all_log_probs:
            combined = torch.cat(all_log_probs, dim=1)  # [B, total_frames, V]
            decoded = greedy_decode(combined, blank_id=cfg.blank_id)
        else:
            batch_size = bpe_token_ids.size(0)
            decoded = [[] for _ in range(batch_size)]

        new_state = StreamingState(
            encoder_state=encoder_state,
            frame_buffer=frame_buffer,
        )
        return decoded, new_state

    @torch.inference_mode()
    def flush(
        self, state: StreamingState,
    ) -> tuple[list[list[int]], StreamingState]:
        """Process any remaining frames in the buffer.

        Pads the remaining frames to form a final chunk if needed.
        Should be called after all tokens have been fed via
        :meth:`process_tokens`.  Safe to call when the buffer is empty
        (returns empty labels) or multiple times in succession.

        Args:
            state: Current streaming state.

        Returns:
            ``(decoded_labels, new_state)`` for any remaining buffered frames.
        """
        cfg = self.config
        chunk_size = cfg.chunk_size
        mla_size = cfg.mla_size
        frames_needed = chunk_size + mla_size

        frame_buffer = state.frame_buffer
        remaining = frame_buffer.size(1)

        if remaining == 0:
            batch_size = frame_buffer.size(0)
            return [[] for _ in range(batch_size)], state

        # Pad to frames_needed
        if remaining < frames_needed:
            pad_len = frames_needed - remaining
            padding = torch.zeros(
                frame_buffer.size(0), pad_len, frame_buffer.size(2),
                device=frame_buffer.device,
            )
            frame_buffer = torch.cat([frame_buffer, padding], dim=1)

        chunk = frame_buffer[:, :frames_needed, :]
        # See process_tokens for why pos_enc is called here.
        chunk, _ = self.model.encoder.pos_enc(chunk)

        log_probs, encoder_state = self.model.forward_streaming(
            chunk, state.encoder_state,
        )

        # Only take the valid frames (min of chunk_size, remaining)
        valid_frames = min(chunk_size, remaining)
        log_probs = log_probs[:, :valid_frames, :]

        decoded = greedy_decode(log_probs, blank_id=cfg.blank_id)

        new_state = StreamingState(
            encoder_state=encoder_state,
            frame_buffer=torch.zeros(
                frame_buffer.size(0), 0, frame_buffer.size(2),
                device=frame_buffer.device,
            ),
        )
        return decoded, new_state

    @torch.inference_mode()
    def process_full(
        self,
        input_ids: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> list[list[int]]:
        """Non-streaming inference (wrapper around model.inference).

        Args:
            input_ids: BPE token IDs ``[B, T]``.
            input_lengths: Valid lengths ``[B]``.

        Returns:
            List of decoded PnP label ID sequences.
        """
        return self.model.inference(input_ids, input_lengths)
