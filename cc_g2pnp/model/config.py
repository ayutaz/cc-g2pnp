"""Model configuration for CC-G2PnP Streaming Conformer-CTC."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CC_G2PnPConfig:
    """Hyperparameters for the CC-G2PnP model.

    Paper: arXiv:2602.17157 (Shirahata & Yamamoto, LY Corporation, 2026)
    Values marked (estimated) are Conformer standard defaults not explicitly
    stated in the paper.
    """

    # ── Vocabulary ──────────────────────────────────────────────
    bpe_vocab_size: int = 65_000
    """CALM2 BPE tokenizer vocabulary size."""

    pnp_vocab_size: int = 140
    """PnP CTC target vocabulary: blank(1) + mora(134) + prosody(3) + unk + pad."""

    blank_id: int = 0
    """CTC blank token ID (must be 0 for torch.nn.CTCLoss default)."""

    # ── Model dimensions ────────────────────────────────────────
    d_model: int = 512
    """Hidden dimension (paper: 512)."""

    num_heads: int = 8
    """Number of attention heads (estimated: standard for d_model=512)."""

    d_ff: int = 2048
    """Feed-forward inner dimension = d_model * 4 (estimated: Conformer standard)."""

    num_layers: int = 8
    """Number of Conformer layers (paper: 8)."""

    # ── Token upsampling ────────────────────────────────────────
    upsample_factor: int = 8
    """BPE token upsampling coefficient (paper: 8). [B,T] → [B,T*8,D]."""

    # ── Convolution ─────────────────────────────────────────────
    conv_kernel_size: int = 31
    """Depthwise convolution kernel size (estimated: Conformer standard)."""

    conv_expansion_factor: int = 2
    """Pointwise conv expansion for GLU: d_model → d_model*2 → GLU → d_model."""

    # ── Streaming ───────────────────────────────────────────────
    chunk_size: int = 5
    """Chunk size C for streaming attention (paper: 2 or 5)."""

    past_context: int = 10
    """Number of past tokens P from previous chunks (paper: 10)."""

    mla_size: int = 1
    """Minimum Look-Ahead M tokens beyond chunk boundary (paper: 0, 1, or 2).
    Applied only to layer 0 self-attention."""

    # ── Self-conditioned CTC ────────────────────────────────────
    intermediate_ctc_layers: tuple[int, ...] = (1, 3, 5)
    """0-indexed layer indices for intermediate CTC loss.
    Paper says layers 2, 4, 6 (1-indexed) = 1, 3, 5 (0-indexed)."""

    intermediate_ctc_weight: float = 1.0 / 3.0
    """Weight for each intermediate CTC loss term.
    Total loss = final_CTC + weight * sum(intermediate_CTC_losses)."""

    # ── Attention backend ───────────────────────────────────────
    use_flash_attention: bool = False
    """Use F.scaled_dot_product_attention instead of manual matmul+softmax.
    Enables EFFICIENT_ATTENTION kernel on T4 (sm75+) for ~10-20% speedup.
    Checkpoint-compatible (no weight shape changes)."""

    # ── Regularization ──────────────────────────────────────────
    dropout: float = 0.1
    """Dropout rate (estimated: common default)."""

    # ── Feed-forward ────────────────────────────────────────────
    ff_residual_factor: float = 0.5
    """Half-step residual connection factor (Conformer standard)."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.d_model <= 0:
            msg = f"d_model must be positive, got {self.d_model}"
            raise ValueError(msg)
        if self.num_heads <= 0:
            msg = f"num_heads must be positive, got {self.num_heads}"
            raise ValueError(msg)
        if self.d_model % self.num_heads != 0:
            msg = f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
            raise ValueError(msg)
        if self.num_layers <= 0:
            msg = f"num_layers must be positive, got {self.num_layers}"
            raise ValueError(msg)
        for layer_idx in self.intermediate_ctc_layers:
            if layer_idx >= self.num_layers:
                msg = f"intermediate_ctc_layers index {layer_idx} >= num_layers ({self.num_layers})"
                raise ValueError(msg)
