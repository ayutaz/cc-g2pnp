"""Training logger for CC-G2PnP. Supports TensorBoard and W&B."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from cc_g2pnp.training.config import TrainingConfig

try:
    import wandb as _wandb

    _wandb_available = True
except ImportError:
    _wandb = None
    _wandb_available = False


class TrainingLogger:
    """Training metrics logger. Supports TensorBoard and W&B backends.

    Args:
        config: TrainingConfig instance with logging settings.

    Example:
        >>> with TrainingLogger(config) as logger:
        ...     logger.log_metrics({"loss": 0.5}, step=100)
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._tb_writer: SummaryWriter | None = None
        self._use_wandb = False

        # TensorBoard setup
        if config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            log_dir = str(Path(config.checkpoint_dir) / "tensorboard")
            self._tb_writer = SummaryWriter(log_dir=log_dir)

        # W&B setup
        if config.use_wandb:
            if not _wandb_available:
                warnings.warn(
                    "wandb is not installed. Disabling W&B logging. "
                    "Install with: pip install wandb",
                    stacklevel=2,
                )
            else:
                _wandb.init(project=config.project_name, name=config.run_name)
                self._use_wandb = True

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to all active backends.

        Args:
            metrics: Dictionary of metric name to value.
            step: Current training step number.
        """
        if self._tb_writer is not None:
            for key, value in metrics.items():
                self._tb_writer.add_scalar(key, value, step)

        if self._use_wandb:
            _wandb.log(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to all active backends.

        Args:
            params: Dictionary of hyperparameter name to value.
        """
        if self._tb_writer is not None:
            self._tb_writer.add_hparams(params, {})

        if self._use_wandb:
            _wandb.config.update(params)

    def close(self) -> None:
        """Close all active backends and release resources."""
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

        if self._use_wandb:
            _wandb.finish()
            self._use_wandb = False

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
