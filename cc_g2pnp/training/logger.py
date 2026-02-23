"""Training logger for CC-G2PnP. Uses W&B as the logging backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cc_g2pnp.training.config import TrainingConfig

try:
    import wandb as _wandb

    _wandb_available = True
except ImportError:
    _wandb = None
    _wandb_available = False


class TrainingLogger:
    """Training metrics logger with W&B backend.

    W&B is required. Raises ``RuntimeError`` if wandb is not installed
    or the user is not logged in.

    Args:
        config: TrainingConfig instance with logging settings.

    Example:
        >>> with TrainingLogger(config) as logger:
        ...     logger.log_metrics({"loss": 0.5}, step=100)
    """

    def __init__(self, config: TrainingConfig) -> None:
        if not _wandb_available:
            msg = (
                "wandb is required but not installed. "
                "Install with: pip install wandb"
            )
            raise RuntimeError(msg)

        if not _wandb.api.api_key:
            msg = (
                "wandb is not logged in. "
                "Run `wandb login` or set the WANDB_API_KEY environment variable."
            )
            raise RuntimeError(msg)

        _wandb.init(project=config.project_name, name=config.run_name)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to W&B.

        Args:
            metrics: Dictionary of metric name to value.
            step: Current training step number.
        """
        _wandb.log(metrics, step=step)

    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters to W&B.

        Args:
            params: Dictionary of hyperparameter name to value.
        """
        _wandb.config.update(params)

    def close(self) -> None:
        """Close W&B backend and release resources."""
        _wandb.finish()

    def __enter__(self) -> TrainingLogger:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
