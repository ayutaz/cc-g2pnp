"""Training configuration for CC-G2PnP."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Hyperparameters for CC-G2PnP training.

    Paper: arXiv:2602.17157 (Shirahata & Yamamoto, LY Corporation, 2026)
    """

    # ── Optimizer (AdamW) ────────────────────────────────────────
    learning_rate: float = 1e-4
    """Peak learning rate."""

    final_learning_rate: float = 1e-5
    """Final learning rate for exponential decay."""

    weight_decay: float = 0.01
    """AdamW weight decay coefficient."""

    betas: tuple[float, float] = (0.9, 0.98)
    """AdamW beta coefficients."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    # ── Scheduler (Warmup + ExponentialLR) ────────────────────────
    total_steps: int = 1_200_000
    """Total training steps (paper: 1.2M)."""

    warmup_steps: int = 10_000
    """Linear warmup steps."""

    # ── Data ──────────────────────────────────────────────────────
    max_tokens_per_batch: int = 8192
    """Maximum BPE tokens per batch (dynamic batching)."""

    max_input_len: int = 512
    """Maximum BPE token length per sample. Longer samples are filtered out.
    With upsample_factor=8, T4 (15GB) requires ≤128 to avoid OOM in attention."""

    dataset_subset: str = "all"
    """ReazonSpeech dataset subset name."""

    # ── Checkpoint ────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    """Directory to save checkpoints."""

    save_every_n_steps: int = 10_000
    """Save checkpoint every N steps."""

    keep_last_n: int = 5
    """Number of most recent checkpoints to keep."""

    # ── Logging ───────────────────────────────────────────────────
    log_every_n_steps: int = 100
    """Log metrics every N steps."""

    project_name: str = "cc-g2pnp"
    """Project name for experiment tracking."""

    run_name: str | None = None
    """Run name for experiment tracking (auto-generated if None)."""

    # ── AMP (Mixed Precision) ─────────────────────────────────────
    use_amp: bool = True
    """Enable automatic mixed precision training."""

    amp_dtype: str = "bfloat16"
    """AMP data type: 'float16' or 'bfloat16'."""

    # ── DDP (Multi-GPU) ───────────────────────────────────────────
    use_ddp: bool = False
    """Enable DistributedDataParallel training."""

    # ── Validation ────────────────────────────────────────────────
    val_every_n_steps: int = 5_000
    """Run validation every N steps."""

    # ── Data pipeline ────────────────────────────────────────────
    num_workers: int = 4
    """Number of DataLoader worker processes for parallel preprocessing."""

    prefetch_count: int = 4
    """Number of batches to prefetch in background (0 = no prefetch)."""

    # ── Misc ──────────────────────────────────────────────────────
    seed: int = 42
    """Random seed for reproducibility."""

    max_steps: int | None = None
    """Override total_steps for debug/small runs. None uses total_steps."""

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.learning_rate <= 0:
            msg = f"learning_rate must be positive, got {self.learning_rate}"
            raise ValueError(msg)
        if self.final_learning_rate <= 0:
            msg = f"final_learning_rate must be positive, got {self.final_learning_rate}"
            raise ValueError(msg)
        if self.final_learning_rate > self.learning_rate:
            msg = (
                f"final_learning_rate ({self.final_learning_rate}) must be "
                f"<= learning_rate ({self.learning_rate})"
            )
            raise ValueError(msg)
        if self.weight_decay < 0:
            msg = f"weight_decay must be >= 0, got {self.weight_decay}"
            raise ValueError(msg)
        if not isinstance(self.betas, tuple) or len(self.betas) != 2:
            msg = f"betas must be a tuple of 2 floats, got {self.betas}"
            raise ValueError(msg)
        if not (0 <= self.betas[0] < 1 and 0 <= self.betas[1] < 1):
            msg = f"betas values must be in [0, 1), got {self.betas}"
            raise ValueError(msg)
        if self.total_steps <= 0:
            msg = f"total_steps must be positive, got {self.total_steps}"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = f"warmup_steps must be >= 0, got {self.warmup_steps}"
            raise ValueError(msg)
        if self.warmup_steps >= self.total_steps:
            msg = (
                f"warmup_steps ({self.warmup_steps}) must be "
                f"< total_steps ({self.total_steps})"
            )
            raise ValueError(msg)
        if self.max_grad_norm <= 0:
            msg = f"max_grad_norm must be positive, got {self.max_grad_norm}"
            raise ValueError(msg)
        if self.amp_dtype not in ("float16", "bfloat16"):
            msg = f"amp_dtype must be 'float16' or 'bfloat16', got '{self.amp_dtype}'"
            raise ValueError(msg)
        if self.max_tokens_per_batch <= 0:
            msg = f"max_tokens_per_batch must be positive, got {self.max_tokens_per_batch}"
            raise ValueError(msg)
        if self.save_every_n_steps <= 0:
            msg = f"save_every_n_steps must be positive, got {self.save_every_n_steps}"
            raise ValueError(msg)
        if self.val_every_n_steps <= 0:
            msg = f"val_every_n_steps must be positive, got {self.val_every_n_steps}"
            raise ValueError(msg)
        if self.log_every_n_steps <= 0:
            msg = f"log_every_n_steps must be positive, got {self.log_every_n_steps}"
            raise ValueError(msg)
        if self.keep_last_n < 1:
            msg = f"keep_last_n must be >= 1, got {self.keep_last_n}"
            raise ValueError(msg)
        if self.seed < 0:
            msg = f"seed must be >= 0, got {self.seed}"
            raise ValueError(msg)
        if self.max_steps is not None and self.max_steps <= 0:
            msg = f"max_steps must be positive or None, got {self.max_steps}"
            raise ValueError(msg)
        if self.max_steps is not None and self.max_steps <= self.warmup_steps:
            msg = (
                f"max_steps ({self.max_steps}) must be "
                f"> warmup_steps ({self.warmup_steps})"
            )
            raise ValueError(msg)

    @property
    def effective_steps(self) -> int:
        """Return max_steps if set, otherwise total_steps."""
        if self.max_steps is not None:
            return self.max_steps
        return self.total_steps

    @property
    def scheduler_gamma(self) -> float:
        """Compute ExponentialLR gamma so that LR decays from learning_rate to final_learning_rate.

        gamma = (final_lr / lr) ** (1 / (effective_steps - warmup_steps))

        When learning_rate == final_learning_rate, returns 1.0 (no decay).
        """
        decay_steps = self.effective_steps - self.warmup_steps
        return (self.final_learning_rate / self.learning_rate) ** (1.0 / decay_steps)
