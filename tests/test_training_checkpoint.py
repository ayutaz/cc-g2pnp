"""CheckpointManager のテスト。"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from cc_g2pnp.training.checkpoint import CheckpointManager


@dataclasses.dataclass
class _DummyConfig:
    lr: float = 1e-3
    epochs: int = 10


def _make_components():
    """テスト用のモデル・オプティマイザ・スケジューラを生成する。"""
    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    config = _DummyConfig()
    return model, optimizer, scheduler, config


class TestSave:
    """save メソッドのテスト。"""

    def test_creates_file(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        path = mgr.save(100, model, opt, sched, cfg)

        assert path.exists()

    def test_filename_format(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        path = mgr.save(10000, model, opt, sched, cfg)

        assert path.name == "step_00010000.pt"

    def test_returns_correct_path(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        path = mgr.save(42, model, opt, sched, cfg)

        assert path == tmp_path / "ckpt" / "step_00000042.pt"

    def test_saved_keys(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        path = mgr.save(1, model, opt, sched, cfg, metrics={"loss": 0.5})
        ckpt = torch.load(path, weights_only=False)

        expected_keys = {"step", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "config", "metrics"}
        assert set(ckpt.keys()) == expected_keys

    def test_step_value(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        path = mgr.save(999, model, opt, sched, cfg)
        ckpt = torch.load(path, weights_only=False)

        assert ckpt["step"] == 999

    def test_metrics_included(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()
        metrics = {"loss": 0.25, "accuracy": 0.95}

        path = mgr.save(1, model, opt, sched, cfg, metrics=metrics)
        ckpt = torch.load(path, weights_only=False)

        assert ckpt["metrics"] == metrics

    def test_metrics_none(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        path = mgr.save(1, model, opt, sched, cfg)
        ckpt = torch.load(path, weights_only=False)

        assert ckpt["metrics"] is None

    def test_config_serialized(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        path = mgr.save(1, model, opt, sched, cfg)
        ckpt = torch.load(path, weights_only=False)

        assert ckpt["config"] == {"lr": 1e-3, "epochs": 10}

    def test_ddp_model(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        inner_model = nn.Linear(4, 2)
        ddp_model = MagicMock()
        ddp_model.module = inner_model

        opt = torch.optim.SGD(inner_model.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        cfg = _DummyConfig()

        path = mgr.save(1, ddp_model, opt, sched, cfg)
        ckpt = torch.load(path, weights_only=False)

        # DDP モデルの場合、inner model の state_dict が保存されること
        for key in inner_model.state_dict():
            assert key in ckpt["model_state_dict"]

    def test_creates_directory(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        mgr = CheckpointManager(nested)
        model, opt, sched, cfg = _make_components()

        path = mgr.save(1, model, opt, sched, cfg)

        assert nested.exists()
        assert path.exists()


class TestLoad:
    """load メソッドのテスト。"""

    def test_load_existing(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        saved_path = mgr.save(50, model, opt, sched, cfg)
        ckpt = mgr.load(saved_path)

        assert ckpt["step"] == 50

    def test_load_nonexistent_raises(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")

        with pytest.raises(FileNotFoundError):
            mgr.load(tmp_path / "ckpt" / "step_99999999.pt")


class TestLoadLatest:
    """load_latest メソッドのテスト。"""

    def test_returns_none_when_empty(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")

        assert mgr.load_latest() is None

    def test_loads_latest(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", keep_last_n=10)
        model, opt, sched, cfg = _make_components()

        mgr.save(100, model, opt, sched, cfg)
        mgr.save(200, model, opt, sched, cfg)
        mgr.save(300, model, opt, sched, cfg)

        ckpt = mgr.load_latest()

        assert ckpt["step"] == 300


class TestCleanup:
    """cleanup メソッドのテスト。"""

    def test_keeps_last_n(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", keep_last_n=2)
        model, opt, sched, cfg = _make_components()

        mgr.save(100, model, opt, sched, cfg)
        mgr.save(200, model, opt, sched, cfg)
        mgr.save(300, model, opt, sched, cfg)
        mgr.save(400, model, opt, sched, cfg)

        remaining = list((tmp_path / "ckpt").glob("step_*.pt"))
        remaining_names = sorted(p.name for p in remaining)

        assert len(remaining) == 2
        assert remaining_names == ["step_00000300.pt", "step_00000400.pt"]

    def test_no_deletion_when_under_limit(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt", keep_last_n=5)
        model, opt, sched, cfg = _make_components()

        mgr.save(100, model, opt, sched, cfg)
        mgr.save(200, model, opt, sched, cfg)

        remaining = list((tmp_path / "ckpt").glob("step_*.pt"))

        assert len(remaining) == 2


class TestE2E:
    """Save -> Load -> load_state_dict の E2E テスト。"""

    def test_save_load_roundtrip(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "ckpt")

        # 元モデルを作成し、パラメータを保存
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        cfg = _DummyConfig()

        original_weight = model.weight.data.clone()
        mgr.save(1, model, optimizer, scheduler, cfg)

        # 新しいモデルに読み込み
        new_model = nn.Linear(4, 2)
        assert not torch.equal(new_model.weight.data, original_weight)

        ckpt = mgr.load_latest()
        new_model.load_state_dict(ckpt["model_state_dict"])

        assert torch.equal(new_model.weight.data, original_weight)
