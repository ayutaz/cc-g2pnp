"""Training configuration for CC-G2PnP."""

from __future__ import annotations

from dataclasses import dataclass

_VALID_MP_CONTEXTS = {"fork", "forkserver", "spawn"}
_VALID_SCHEDULER_TYPES = {"exponential", "cosine"}


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

    scheduler_type: str = "cosine"
    """LR scheduler type: 'exponential' (ExponentialLR) or 'cosine' (CosineAnnealingLR)."""

    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps. Effective batch = max_tokens * accumulation_steps."""

    # ── Data ──────────────────────────────────────────────────────
    max_tokens_per_batch: int = 8192
    """Maximum BPE tokens per batch (dynamic batching)."""

    max_input_len: int = 64
    """Maximum BPE token length per sample. Longer samples are filtered out.
    ReazonSpeech avg=11.6, P99=45 tokens — 64 covers 99.9% of data.
    With upsample_factor=8, T=max_input_len*8; smaller values reduce O(T²) attention memory."""

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

    amp_dtype: str = "float16"
    """AMP data type: 'float16' or 'bfloat16'.
    T4 GPUs lack bfloat16 Tensor Cores — use float16 for full Tensor Core utilization."""

    # ── DDP (Multi-GPU) ───────────────────────────────────────────
    use_ddp: bool = False
    """Enable DistributedDataParallel training."""

    # ── Validation ────────────────────────────────────────────────
    val_every_n_steps: int = 5_000
    """Run validation every N steps."""

    # ── Data pipeline ────────────────────────────────────────────
    num_workers: int = 8
    """Number of DataLoader worker processes for parallel preprocessing."""

    multiprocessing_context: str = "forkserver"
    """Multiprocessing start method for DataLoader workers (fork/forkserver/spawn)."""

    prefetch_count: int = 4
    """Number of batches to prefetch in background (0 = no prefetch)."""

    # ── LMDB cache ────────────────────────────────────────────────
    lmdb_cache_dir: str | None = None
    """Path to LMDB directory with pre-computed PnP labels.
    If set, dataset reads labels from cache instead of generating on-the-fly.
    Build with: uv run python scripts/preprocess_pnp.py --output <path>"""

    local_dataset_dir: str | None = None
    """Path to local Parquet dataset directory (from scripts/download_text.py).
    If set, dataset reads from local files instead of streaming from HuggingFace."""

    # ── Checkpoint ─────────────────────────────────────────────────
    async_checkpoint: bool = True
    """Save checkpoints asynchronously in a background thread."""

    # ── Misc ──────────────────────────────────────────────────────
    seed: int = 42
    """Random seed for reproducibility."""

    max_steps: int | None = None
    """Override total_steps for debug/small runs. None uses total_steps."""

    pretrained_weights_only: bool = False
    """Load only model weights from checkpoint (reset optimizer/scheduler for transfer learning)."""

    use_torch_compile: bool = False
    """Apply torch.compile to FFN and ConvModule submodules for kernel fusion (15-25% speedup).
    Graph breaks in the encoder loop prevent whole-model compile; individual modules are compiled instead."""

    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to reduce activation memory at the cost of ~30% slower backward.
    Disable if GPU memory allows, to reduce backward cost (65% of step time with checkpointing)."""

    sort_batch_buffer_size: int = 10_000
    """Buffer size for length-sorted batching. Samples are buffered, sorted by BPE length,
    then batched — reducing padding waste from ~40-60% to ~5-15%. 0 = disabled (random order)."""

    disable_intermediate_ctc_after: int | None = None
    """Disable intermediate CTC computation after this many steps to save ~5-10% compute.
    None = always enabled (paper default). Recommended: set to warmup_steps * 10."""

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
        if self.scheduler_type not in _VALID_SCHEDULER_TYPES:
            msg = f"scheduler_type must be one of {_VALID_SCHEDULER_TYPES}, got '{self.scheduler_type}'"
            raise ValueError(msg)
        if self.gradient_accumulation_steps < 1:
            msg = f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            raise ValueError(msg)
        if self.multiprocessing_context not in _VALID_MP_CONTEXTS:
            msg = f"multiprocessing_context must be one of {_VALID_MP_CONTEXTS}, got '{self.multiprocessing_context}'"
            raise ValueError(msg)
        if self.num_workers < 0:
            msg = f"num_workers must be >= 0, got {self.num_workers}"
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
