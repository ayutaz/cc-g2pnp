"""Tests for TrainingLogger."""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cc_g2pnp.training.logger import TrainingLogger


def _make_config(**overrides):
    """Create a minimal config-like object for TrainingLogger."""
    defaults = {
        "use_wandb": False,
        "project_name": "test-project",
        "run_name": "test-run",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ── W&B only ────────────────────────────────────────────────────


class TestWandBOnly:
    """Tests with W&B enabled."""

    def test_init_calls_wandb_init(self, monkeypatch):
        """wandb.init() is called with correct project/name."""
        mock_wandb = MagicMock()
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config(use_wandb=True)
        logger = TrainingLogger(config)

        mock_wandb.init.assert_called_once_with(project="test-project", name="test-run")
        assert logger._use_wandb is True

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


# ── Disabled ───────────────────────────────────────────────


class TestDisabled:
    """Tests with W&B disabled."""

    def test_init_no_backend(self):
        """No backend initialized when disabled."""
        config = _make_config()
        logger = TrainingLogger(config)

        assert logger._use_wandb is False

    def test_log_metrics_noop(self):
        """log_metrics does nothing when disabled."""
        config = _make_config()
        logger = TrainingLogger(config)
        # Should not raise
        logger.log_metrics({"loss": 0.5}, step=1)

    def test_log_hyperparams_noop(self):
        """log_hyperparams does nothing when disabled."""
        config = _make_config()
        logger = TrainingLogger(config)
        logger.log_hyperparams({"lr": 1e-4})

    def test_close_noop(self):
        """close() does nothing when disabled."""
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

    def test_exit_calls_close(self, monkeypatch):
        """__exit__ calls close()."""
        mock_wandb = MagicMock()
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config(use_wandb=True)
        with TrainingLogger(config):
            pass

        mock_wandb.finish.assert_called_once()

    def test_exit_calls_close_on_exception(self, monkeypatch):
        """__exit__ calls close() even when an exception occurs."""
        mock_wandb = MagicMock()
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config(use_wandb=True)
        with pytest.raises(RuntimeError, match="test error"), TrainingLogger(config):
            raise RuntimeError("test error")

        mock_wandb.finish.assert_called_once()


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
