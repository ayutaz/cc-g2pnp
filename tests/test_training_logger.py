"""Tests for TrainingLogger."""

from __future__ import annotations

import sys
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from cc_g2pnp.training.logger import TrainingLogger


def _make_config(**overrides):
    """Create a minimal config-like object for TrainingLogger."""
    defaults = {
        "checkpoint_dir": "checkpoints",
        "use_tensorboard": False,
        "use_wandb": False,
        "project_name": "test-project",
        "run_name": "test-run",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture()
def mock_tensorboard():
    """Mock torch.utils.tensorboard so SummaryWriter can be imported."""
    mock_writer_cls = MagicMock()
    mock_writer_instance = MagicMock()
    mock_writer_cls.return_value = mock_writer_instance

    mock_tb_module = MagicMock()
    mock_tb_module.SummaryWriter = mock_writer_cls

    with patch.dict(sys.modules, {"torch.utils.tensorboard": mock_tb_module}):
        yield mock_writer_cls, mock_writer_instance


# ── TensorBoard only ────────────────────────────────────────────


class TestTensorBoardOnly:
    """Tests with only TensorBoard enabled."""

    def test_init_creates_writer(self, mock_tensorboard):
        """SummaryWriter is created with correct log_dir."""
        mock_sw_cls, mock_writer = mock_tensorboard

        config = _make_config(use_tensorboard=True)
        logger = TrainingLogger(config)

        mock_sw_cls.assert_called_once()
        call_kwargs = mock_sw_cls.call_args
        log_dir = call_kwargs.kwargs.get("log_dir", call_kwargs.args[0] if call_kwargs.args else "")
        assert "tensorboard" in log_dir
        assert logger._tb_writer is mock_writer
        assert logger._use_wandb is False

    def test_log_metrics(self, mock_tensorboard):
        """log_metrics calls add_scalar for each metric."""
        _mock_sw_cls, mock_writer = mock_tensorboard

        config = _make_config(use_tensorboard=True)
        logger = TrainingLogger(config)

        logger.log_metrics({"loss": 0.5, "lr": 1e-4}, step=100)

        assert mock_writer.add_scalar.call_count == 2
        mock_writer.add_scalar.assert_any_call("loss", 0.5, 100)
        mock_writer.add_scalar.assert_any_call("lr", 1e-4, 100)

    def test_log_hyperparams(self, mock_tensorboard):
        """log_hyperparams calls add_hparams."""
        _mock_sw_cls, mock_writer = mock_tensorboard

        config = _make_config(use_tensorboard=True)
        logger = TrainingLogger(config)

        params = {"lr": 1e-4, "batch_size": 32}
        logger.log_hyperparams(params)

        mock_writer.add_hparams.assert_called_once_with(params, {})

    def test_close(self, mock_tensorboard):
        """close() calls writer.close()."""
        _mock_sw_cls, mock_writer = mock_tensorboard

        config = _make_config(use_tensorboard=True)
        logger = TrainingLogger(config)

        logger.close()

        mock_writer.close.assert_called_once()
        assert logger._tb_writer is None


# ── W&B only ────────────────────────────────────────────────────


class TestWandBOnly:
    """Tests with only W&B enabled."""

    def test_init_calls_wandb_init(self, monkeypatch):
        """wandb.init() is called with correct project/name."""
        mock_wandb = MagicMock()
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config(use_wandb=True)
        logger = TrainingLogger(config)

        mock_wandb.init.assert_called_once_with(project="test-project", name="test-run")
        assert logger._use_wandb is True
        assert logger._tb_writer is None

    def test_log_metrics(self, monkeypatch):
        """log_metrics calls wandb.log()."""
        mock_wandb = MagicMock()
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config(use_wandb=True)
        logger = TrainingLogger(config)

        metrics = {"loss": 0.3, "acc": 0.95}
        logger.log_metrics(metrics, step=200)

        mock_wandb.log.assert_called_once_with(metrics, step=200)

    def test_log_hyperparams(self, monkeypatch):
        """log_hyperparams calls wandb.config.update()."""
        mock_wandb = MagicMock()
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config(use_wandb=True)
        logger = TrainingLogger(config)

        params = {"lr": 1e-4}
        logger.log_hyperparams(params)

        mock_wandb.config.update.assert_called_once_with(params)

    def test_close(self, monkeypatch):
        """close() calls wandb.finish()."""
        mock_wandb = MagicMock()
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config(use_wandb=True)
        logger = TrainingLogger(config)

        logger.close()

        mock_wandb.finish.assert_called_once()
        assert logger._use_wandb is False


# ── Both disabled ───────────────────────────────────────────────


class TestBothDisabled:
    """Tests with both backends disabled."""

    def test_init_no_backends(self):
        """No backends initialized when both disabled."""
        config = _make_config()
        logger = TrainingLogger(config)

        assert logger._tb_writer is None
        assert logger._use_wandb is False

    def test_log_metrics_noop(self):
        """log_metrics does nothing when both disabled."""
        config = _make_config()
        logger = TrainingLogger(config)
        # Should not raise
        logger.log_metrics({"loss": 0.5}, step=1)

    def test_log_hyperparams_noop(self):
        """log_hyperparams does nothing when both disabled."""
        config = _make_config()
        logger = TrainingLogger(config)
        logger.log_hyperparams({"lr": 1e-4})

    def test_close_noop(self):
        """close() does nothing when both disabled."""
        config = _make_config()
        logger = TrainingLogger(config)
        logger.close()


# ── Context manager ─────────────────────────────────────────────


class TestContextManager:
    """Tests for context manager protocol."""

    def test_enter_returns_self(self):
        """__enter__ returns the logger instance."""
        config = _make_config()
        logger = TrainingLogger(config)

        with logger as ctx:
            assert ctx is logger

    def test_exit_calls_close(self, mock_tensorboard):
        """__exit__ calls close()."""
        _mock_sw_cls, mock_writer = mock_tensorboard

        config = _make_config(use_tensorboard=True)
        with TrainingLogger(config):
            pass

        mock_writer.close.assert_called_once()

    def test_exit_calls_close_on_exception(self, mock_tensorboard):
        """__exit__ calls close() even when an exception occurs."""
        _mock_sw_cls, mock_writer = mock_tensorboard

        config = _make_config(use_tensorboard=True)
        with pytest.raises(RuntimeError, match="test error"), TrainingLogger(config):
            raise RuntimeError("test error")

        mock_writer.close.assert_called_once()


# ── wandb not installed ─────────────────────────────────────────


class TestWandBNotInstalled:
    """Tests for when wandb is not installed."""

    def test_warns_and_disables(self, monkeypatch):
        """Warning issued and wandb disabled when not installed."""
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", False)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", None)

        config = _make_config(use_wandb=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            logger = TrainingLogger(config)

        assert logger._use_wandb is False
        assert len(w) == 1
        assert "wandb is not installed" in str(w[0].message)


# ── Both enabled ────────────────────────────────────────────────


class TestBothEnabled:
    """Tests with both TensorBoard and W&B enabled."""

    def test_log_metrics_both(self, mock_tensorboard, monkeypatch):
        """log_metrics calls both backends."""
        _mock_sw_cls, mock_writer = mock_tensorboard
        mock_wandb = MagicMock()
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config(use_tensorboard=True, use_wandb=True)
        logger = TrainingLogger(config)

        logger.log_metrics({"loss": 0.1}, step=50)

        mock_writer.add_scalar.assert_called_once_with("loss", 0.1, 50)
        mock_wandb.log.assert_called_once_with({"loss": 0.1}, step=50)

    def test_close_both(self, mock_tensorboard, monkeypatch):
        """close() closes both backends."""
        _mock_sw_cls, mock_writer = mock_tensorboard
        mock_wandb = MagicMock()
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config(use_tensorboard=True, use_wandb=True)
        logger = TrainingLogger(config)

        logger.close()

        mock_writer.close.assert_called_once()
        mock_wandb.finish.assert_called_once()
