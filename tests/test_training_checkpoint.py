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

        expected_keys = {"step", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "config", "metrics", "scaler_state_dict"}
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


class TestAtomicSave:
    """アトミック保存のテスト。"""

    def test_no_tmp_file_after_save(self, tmp_path):
        """保存後に .pt.tmp ファイルが残っていないことを確認。"""
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        mgr.save(1, model, opt, sched, cfg)

        tmp_files = list((tmp_path / "ckpt").glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_final_file_exists(self, tmp_path):
        """アトミック保存後に最終ファイルが存在することを確認。"""
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        path = mgr.save(1, model, opt, sched, cfg)

        assert path.exists()
        assert path.suffix == ".pt"


class TestMapLocation:
    """map_location のテスト。"""

    def test_load_returns_cpu_tensors(self, tmp_path):
        """ロードされたテンソルが CPU 上にあることを確認。"""
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        mgr.save(1, model, opt, sched, cfg)
        ckpt = mgr.load_latest()

        for value in ckpt["model_state_dict"].values():
            if isinstance(value, torch.Tensor):
                assert value.device == torch.device("cpu")


class TestCorruptedCheckpointFallback:
    """破損チェックポイントのフォールバックテスト。"""

    def test_fallback_to_previous(self, tmp_path):
        """最新が破損している場合、前のチェックポイントにフォールバック。"""
        mgr = CheckpointManager(tmp_path / "ckpt", keep_last_n=10)
        model, opt, sched, cfg = _make_components()

        mgr.save(100, model, opt, sched, cfg)
        mgr.save(200, model, opt, sched, cfg)

        # 最新ファイルを破損させる
        latest = tmp_path / "ckpt" / "step_00000200.pt"
        latest.write_bytes(b"corrupted data")

        ckpt = mgr.load_latest()
        assert ckpt is not None
        assert ckpt["step"] == 100

    def test_all_corrupted_returns_none(self, tmp_path):
        """全チェックポイントが破損している場合、None を返す。"""
        mgr = CheckpointManager(tmp_path / "ckpt", keep_last_n=10)
        model, opt, sched, cfg = _make_components()

        mgr.save(100, model, opt, sched, cfg)
        mgr.save(200, model, opt, sched, cfg)

        # 全ファイルを破損させる
        for f in (tmp_path / "ckpt").glob("step_*.pt"):
            f.write_bytes(b"corrupted")

        assert mgr.load_latest() is None


class TestInvalidFilenames:
    """不正なファイル名のテスト。"""

    def test_skips_invalid_filenames(self, tmp_path):
        """不正なファイル名がスキップされることを確認。"""
        mgr = CheckpointManager(tmp_path / "ckpt", keep_last_n=10)
        model, opt, sched, cfg = _make_components()

        mgr.save(100, model, opt, sched, cfg)

        # 不正なファイルを配置
        (tmp_path / "ckpt" / "step_abc.pt").write_bytes(b"bad")

        checkpoints = mgr._sorted_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].name == "step_00000100.pt"


class TestScalerStateDictSave:
    """GradScaler state_dict 保存のテスト。"""

    def test_scaler_state_dict_saved(self, tmp_path):
        """scaler_state_dict が保存されることを確認。"""
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()
        scaler_sd = {"scale": 65536.0, "growth_factor": 2.0}

        path = mgr.save(1, model, opt, sched, cfg, scaler_state_dict=scaler_sd)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")

        assert ckpt["scaler_state_dict"] == scaler_sd

    def test_scaler_state_dict_none_by_default(self, tmp_path):
        """scaler_state_dict がデフォルトで None であることを確認。"""
        mgr = CheckpointManager(tmp_path / "ckpt")
        model, opt, sched, cfg = _make_components()

        path = mgr.save(1, model, opt, sched, cfg)
        ckpt = torch.load(path, weights_only=False, map_location="cpu")

        assert ckpt["scaler_state_dict"] is None


class TestAsyncSave:
    """非同期チェックポイント保存のテスト。"""

    def test_async_save_creates_file(self, tmp_path):
        """非同期保存がチェックポイントファイルを作成することを確認。"""
        mgr = CheckpointManager(tmp_path / "ckpt", async_save=True)
        model, opt, sched, cfg = _make_components()

        path = mgr.save(100, model, opt, sched, cfg)
        mgr.wait_for_save()

        assert path.exists()

    def test_async_save_content(self, tmp_path):
        """非同期保存の内容が同期保存と一致することを確認。"""
        sync_mgr = CheckpointManager(tmp_path / "sync", async_save=False)
        async_mgr = CheckpointManager(tmp_path / "async", async_save=True)
        model, opt, sched, cfg = _make_components()
        metrics = {"loss": 0.42}

        sync_path = sync_mgr.save(1, model, opt, sched, cfg, metrics=metrics)
        async_path = async_mgr.save(1, model, opt, sched, cfg, metrics=metrics)
        async_mgr.wait_for_save()

        sync_ckpt = torch.load(sync_path, weights_only=False, map_location="cpu")
        async_ckpt = torch.load(async_path, weights_only=False, map_location="cpu")

        assert async_ckpt["step"] == sync_ckpt["step"]
        assert async_ckpt["metrics"] == sync_ckpt["metrics"]
        assert async_ckpt["config"] == sync_ckpt["config"]
        for key in sync_ckpt["model_state_dict"]:
            assert torch.equal(async_ckpt["model_state_dict"][key], sync_ckpt["model_state_dict"][key])

    def test_async_wait_for_save(self, tmp_path):
        """wait_for_save がバックグラウンド保存完了まで待機することを確認。"""
        mgr = CheckpointManager(tmp_path / "ckpt", async_save=True)
        model, opt, sched, cfg = _make_components()

        path = mgr.save(50, model, opt, sched, cfg)
        # wait_for_save 呼び出し後はファイルが存在しているはず
        mgr.wait_for_save()

        assert path.exists()
        assert path.stat().st_size > 0

    def test_sync_save_still_works(self, tmp_path):
        """async_save=False で従来の同期保存が機能することを確認。"""
        mgr = CheckpointManager(tmp_path / "ckpt", async_save=False)
        model, opt, sched, cfg = _make_components()

        path = mgr.save(200, model, opt, sched, cfg, metrics={"loss": 0.1})

        # 同期保存なので wait_for_save 不要でファイルが即座に存在する
        assert path.exists()
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        assert ckpt["step"] == 200
        assert ckpt["metrics"] == {"loss": 0.1}

    def test_async_save_error_handling(self, tmp_path):
        """バックグラウンドスレッドのエラーが wait_for_save で伝播することを確認。"""
        import threading as _threading

        mgr = CheckpointManager(tmp_path / "ckpt", async_save=True)

        # 直接エラーをシミュレート: _save_error をセットしてスレッド終了済みにする
        mgr._save_error = None
        mgr._save_thread = _threading.Thread(
            target=lambda: setattr(mgr, "_save_error", OSError("Simulated write error")),
            daemon=True,
        )
        mgr._save_thread.start()
        mgr._save_thread.join()

        with pytest.raises(RuntimeError, match="Async checkpoint save failed"):
            mgr.wait_for_save()


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
