"""Tests for TrainingLogger."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cc_g2pnp.training.logger import TrainingLogger


def _make_config(**overrides):
    """Create a minimal config-like object for TrainingLogger."""
    defaults = {
        "project_name": "test-project",
        "run_name": "test-run",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _patch_wandb(monkeypatch):
    """Patch wandb module with a MagicMock and return it."""
    mock_wandb = MagicMock()
    mock_wandb.api.api_key = "fake-key"
    monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
    monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)
    return mock_wandb


# ── Init ─────────────────────────────────────────────────────────


class TestInit:
    """Tests for TrainingLogger initialization."""

    def test_init_calls_wandb_init(self, monkeypatch):
        """wandb.init() is called with correct project/name."""
        mock_wandb = _patch_wandb(monkeypatch)
        config = _make_config()
        TrainingLogger(config)
        mock_wandb.init.assert_called_once_with(project="test-project", name="test-run")

    def test_raises_if_wandb_not_installed(self, monkeypatch):
        """RuntimeError raised when wandb is not installed."""
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", False)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", None)

        config = _make_config()
        with pytest.raises(RuntimeError, match="wandb is required but not installed"):
            TrainingLogger(config)

    def test_raises_if_wandb_not_logged_in(self, monkeypatch):
        """RuntimeError raised when wandb API key is not set."""
        mock_wandb = MagicMock()
        mock_wandb.api.api_key = None
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config()
        with pytest.raises(RuntimeError, match="wandb is not logged in"):
            TrainingLogger(config)

    def test_raises_if_wandb_api_key_empty(self, monkeypatch):
        """RuntimeError raised when wandb API key is empty string."""
        mock_wandb = MagicMock()
        mock_wandb.api.api_key = ""
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb", mock_wandb)
        monkeypatch.setattr("cc_g2pnp.training.logger._wandb_available", True)

        config = _make_config()
        with pytest.raises(RuntimeError, match="wandb is not logged in"):
            TrainingLogger(config)


# ── Logging ──────────────────────────────────────────────────────


class TestLogging:
    """Tests for log_metrics and log_hyperparams."""

    def test_log_metrics(self, monkeypatch):
        """log_metrics calls wandb.log()."""
        mock_wandb = _patch_wandb(monkeypatch)
        logger = TrainingLogger(_make_config())

        metrics = {"loss": 0.3, "acc": 0.95}
        logger.log_metrics(metrics, step=200)

        mock_wandb.log.assert_called_once_with(metrics, step=200)

    def test_log_hyperparams(self, monkeypatch):
        """log_hyperparams calls wandb.config.update()."""
        mock_wandb = _patch_wandb(monkeypatch)
        logger = TrainingLogger(_make_config())

        params = {"lr": 1e-4}
        logger.log_hyperparams(params)

        mock_wandb.config.update.assert_called_once_with(params)


# ── Close ────────────────────────────────────────────────────────


class TestClose:
    """Tests for close()."""

    def test_close_calls_wandb_finish(self, monkeypatch):
        """close() calls wandb.finish()."""
        mock_wandb = _patch_wandb(monkeypatch)
        logger = TrainingLogger(_make_config())
        logger.close()
        mock_wandb.finish.assert_called_once()


# ── Context manager ─────────────────────────────────────────────


class TestContextManager:
    """Tests for context manager protocol."""

    def test_enter_returns_self(self, monkeypatch):
        """__enter__ returns the logger instance."""
        _patch_wandb(monkeypatch)
        logger = TrainingLogger(_make_config())
        with logger as ctx:
            assert ctx is logger

    def test_exit_calls_close(self, monkeypatch):
        """__exit__ calls close()."""
        mock_wandb = _patch_wandb(monkeypatch)
        with TrainingLogger(_make_config()):
            pass
        mock_wandb.finish.assert_called_once()

    def test_exit_calls_close_on_exception(self, monkeypatch):
        """__exit__ calls close() even when an exception occurs."""
        mock_wandb = _patch_wandb(monkeypatch)
        with pytest.raises(RuntimeError, match="test error"), TrainingLogger(_make_config()):
            raise RuntimeError("test error")
        mock_wandb.finish.assert_called_once()
