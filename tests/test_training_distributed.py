"""Tests for cc_g2pnp.training.distributed module."""

from unittest.mock import MagicMock, patch

import torch

from cc_g2pnp.training.distributed import (
    cleanup_ddp,
    get_rank,
    get_world_size,
    is_main_process,
    reduce_metrics,
    setup_ddp,
    wrap_model_ddp,
)

# ---------------------------------------------------------------------------
# Tests when dist.is_initialized() returns False (default, no DDP)
# ---------------------------------------------------------------------------


class TestWithoutDDP:
    """Tests for functions when DDP is not initialized."""

    def test_is_main_process_returns_true(self):
        assert is_main_process() is True

    def test_get_rank_returns_zero(self):
        assert get_rank() == 0

    def test_get_world_size_returns_one(self):
        assert get_world_size() == 1

    def test_reduce_metrics_returns_input(self):
        metrics = {"loss": 0.5, "accuracy": 0.9}
        device = torch.device("cpu")
        result = reduce_metrics(metrics, device)
        assert result is metrics

    def test_cleanup_ddp_no_error(self):
        cleanup_ddp()


# ---------------------------------------------------------------------------
# Tests when dist.is_initialized() returns True (mocked DDP)
# ---------------------------------------------------------------------------


class TestWithDDP:
    """Tests for functions when DDP is initialized (mocked)."""

    @patch("cc_g2pnp.training.distributed.dist")
    def test_is_main_process_rank_zero(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        assert is_main_process() is True

    @patch("cc_g2pnp.training.distributed.dist")
    def test_is_main_process_rank_nonzero(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 1
        assert is_main_process() is False

    @patch("cc_g2pnp.training.distributed.dist")
    def test_get_rank_returns_correct_rank(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 3
        assert get_rank() == 3

    @patch("cc_g2pnp.training.distributed.dist")
    def test_get_world_size_returns_correct_size(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4
        assert get_world_size() == 4

    @patch("cc_g2pnp.training.distributed.dist")
    def test_cleanup_ddp_calls_destroy(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        cleanup_ddp()
        mock_dist.destroy_process_group.assert_called_once()

    @patch("cc_g2pnp.training.distributed.dist")
    def test_reduce_metrics_calls_all_reduce(self, mock_dist):
        mock_dist.is_initialized.return_value = True
        mock_dist.ReduceOp.AVG = "avg_sentinel"

        def fake_all_reduce(tensor, op):
            tensor.fill_(0.25)

        mock_dist.all_reduce.side_effect = fake_all_reduce

        metrics = {"loss": 0.5, "accuracy": 0.9}
        result = reduce_metrics(metrics, torch.device("cpu"))

        assert mock_dist.all_reduce.call_count == 2
        assert result["loss"] == 0.25
        assert result["accuracy"] == 0.25

        for call in mock_dist.all_reduce.call_args_list:
            assert call.kwargs["op"] == "avg_sentinel"


# ---------------------------------------------------------------------------
# Tests for setup_ddp
# ---------------------------------------------------------------------------


class TestSetupDDP:
    """Tests for setup_ddp function."""

    @patch("cc_g2pnp.training.distributed.torch")
    @patch("cc_g2pnp.training.distributed.dist")
    def test_setup_ddp_sets_env_and_calls_init(self, mock_dist, mock_torch, monkeypatch):
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.delenv("MASTER_PORT", raising=False)
        mock_torch.cuda.is_available.return_value = False

        setup_ddp(rank=1, world_size=4, backend="gloo")

        import os

        assert os.environ["MASTER_ADDR"] == "localhost"
        assert os.environ["MASTER_PORT"] == "29500"
        mock_dist.init_process_group.assert_called_once_with(
            "gloo", rank=1, world_size=4
        )

    @patch("cc_g2pnp.training.distributed.torch")
    @patch("cc_g2pnp.training.distributed.dist")
    def test_setup_ddp_sets_cuda_device(self, mock_dist, mock_torch, monkeypatch):
        monkeypatch.delenv("MASTER_ADDR", raising=False)
        monkeypatch.delenv("MASTER_PORT", raising=False)
        mock_torch.cuda.is_available.return_value = True

        setup_ddp(rank=2, world_size=4)

        mock_dist.init_process_group.assert_called_once_with(
            "nccl", rank=2, world_size=4
        )
        mock_torch.cuda.set_device.assert_called_once_with(2)

    @patch("cc_g2pnp.training.distributed.torch")
    @patch("cc_g2pnp.training.distributed.dist")
    def test_setup_ddp_preserves_existing_env(self, mock_dist, mock_torch, monkeypatch):
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "12345")
        mock_torch.cuda.is_available.return_value = False

        setup_ddp(rank=0, world_size=2)

        import os

        assert os.environ["MASTER_ADDR"] == "10.0.0.1"
        assert os.environ["MASTER_PORT"] == "12345"


# ---------------------------------------------------------------------------
# Tests for wrap_model_ddp
# ---------------------------------------------------------------------------


class TestWrapModelDDP:
    """Tests for wrap_model_ddp function."""

    @patch("cc_g2pnp.training.distributed.DistributedDataParallel")
    def test_wrap_model_ddp_default_params(self, mock_ddp_cls):
        model = MagicMock(spec=torch.nn.Module)
        mock_ddp_cls.return_value = MagicMock()

        result = wrap_model_ddp(model, device_id=0)

        mock_ddp_cls.assert_called_once_with(
            model,
            device_ids=[0],
            find_unused_parameters=True,
        )
        assert result is mock_ddp_cls.return_value

    @patch("cc_g2pnp.training.distributed.DistributedDataParallel")
    def test_wrap_model_ddp_custom_params(self, mock_ddp_cls):
        model = MagicMock(spec=torch.nn.Module)
        mock_ddp_cls.return_value = MagicMock()

        wrap_model_ddp(model, device_id=3, find_unused_parameters=False)

        mock_ddp_cls.assert_called_once_with(
            model,
            device_ids=[3],
            find_unused_parameters=False,
        )
