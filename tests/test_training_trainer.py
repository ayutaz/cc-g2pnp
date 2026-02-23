"""Trainer クラスのテスト。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.training.config import TrainingConfig


@pytest.fixture(autouse=True)
def _mock_training_logger():
    """全テストで TrainingLogger をモックして W&B 依存を排除する。"""
    with patch("cc_g2pnp.training.trainer.TrainingLogger"):
        yield


@pytest.fixture
def small_model_config() -> CC_G2PnPConfig:
    """テスト用の小さいモデル設定。"""
    return CC_G2PnPConfig(
        d_model=64,
        num_heads=2,
        d_ff=128,
        num_layers=2,
        intermediate_ctc_layers=(1,),
    )


@pytest.fixture
def training_config(tmp_path) -> TrainingConfig:
    """テスト用のトレーニング設定。"""
    return TrainingConfig(
        max_steps=3,
        use_amp=False,
        checkpoint_dir=str(tmp_path / "ckpt"),
        log_every_n_steps=1,
        save_every_n_steps=100,
        val_every_n_steps=100,
        warmup_steps=0,
    )


def _make_dummy_batch(
    bpe_vocab_size: int = 65_000,
    pnp_vocab_size: int = 140,
    batch_size: int = 2,
    seq_len: int = 10,
    label_len: int = 40,
) -> dict[str, torch.Tensor]:
    """テスト用のダミーバッチを生成する。"""
    return {
        "input_ids": torch.randint(0, bpe_vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(1, pnp_vocab_size, (batch_size, label_len)),
        "input_lengths": torch.tensor([seq_len] * batch_size),
        "label_lengths": torch.tensor([label_len] * batch_size),
    }


class TestTrainerInit:
    """Trainer 初期化のテスト。"""

    def test_creates_instance(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)

        assert trainer is not None

    def test_model_initialized(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)

        assert trainer.model is not None

    def test_optimizer_initialized(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)

        assert trainer.optimizer is not None

    def test_scheduler_initialized(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)

        assert trainer.scheduler is not None

    def test_checkpoint_manager_initialized(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)

        assert trainer.checkpoint_manager is not None

    def test_cpu_no_amp_no_ddp(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)

        assert training_config.use_amp is False
        assert training_config.use_ddp is False
        assert trainer is not None


class TestTrainStep:
    """train_step メソッドのテスト。"""

    def test_returns_dict_with_expected_keys(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        result = trainer.train_step(batch)

        assert "loss" in result
        assert "lr" in result
        assert "grad_norm" in result

    def test_loss_is_finite_float(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        result = trainer.train_step(batch)

        assert isinstance(result["loss"], float)
        assert torch.isfinite(torch.tensor(result["loss"]))

    def test_lr_is_positive(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        result = trainer.train_step(batch)

        assert result["lr"] > 0

    def test_grad_norm_is_finite(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        result = trainer.train_step(batch)

        assert isinstance(result["grad_norm"], float)
        assert torch.isfinite(torch.tensor(result["grad_norm"]))


class TestTrainLoop:
    """train ループのテスト。"""

    def test_train_completes_without_error(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        # データイテレータとバリデーションバッチをモックする
        trainer._create_data_iterator = MagicMock(
            return_value=iter([batch] * training_config.max_steps),
        )
        trainer._create_val_batches = MagicMock(return_value=[])

        trainer.train()

    def test_train_runs_expected_steps(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        trainer._create_data_iterator = MagicMock(
            return_value=iter([batch] * training_config.max_steps),
        )
        trainer._create_val_batches = MagicMock(return_value=[])

        # train_step をスパイして呼び出し回数を確認
        original_train_step = trainer.train_step
        call_count = 0

        def counting_train_step(b):
            nonlocal call_count
            call_count += 1
            return original_train_step(b)

        trainer.train_step = counting_train_step

        trainer.train()

        assert call_count == training_config.max_steps


class TestCheckpointRestore:
    """チェックポイント復元のテスト。"""

    def test_no_checkpoint_starts_from_zero(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)

        # CheckpointManager.load_latest が None を返す → step 0 開始
        trainer.checkpoint_manager.load_latest = MagicMock(return_value=None)

        start_step = trainer._restore_checkpoint()

        assert start_step == 0

    def test_restores_from_checkpoint(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)

        # 実際の state_dict を使って復元用データを作成
        fake_checkpoint = {
            "step": 500,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
        }

        trainer.checkpoint_manager.load_latest = MagicMock(
            return_value=fake_checkpoint,
        )

        start_step = trainer._restore_checkpoint()

        assert start_step == 500


class TestCollatorKeyMapping:
    """collator 出力キーがモデル引数に正しくマッピングされることのテスト。"""

    def test_labels_passed_as_targets(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        # model.forward をモックして引数を記録
        original_forward = trainer.model.forward
        captured_kwargs: dict = {}

        def capturing_forward(**kwargs):
            captured_kwargs.update(kwargs)
            return original_forward(**kwargs)

        with patch.object(trainer.model, "forward", side_effect=capturing_forward):
            trainer.train_step(batch)

        assert "targets" in captured_kwargs
        # デバイス転送後の比較のため、同じデバイスに移動して比較
        assert torch.equal(
            captured_kwargs["targets"], batch["labels"].to(trainer.device)
        )

    def test_label_lengths_passed_as_target_lengths(
        self, small_model_config, training_config
    ):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        original_forward = trainer.model.forward
        captured_kwargs: dict = {}

        def capturing_forward(**kwargs):
            captured_kwargs.update(kwargs)
            return original_forward(**kwargs)

        with patch.object(trainer.model, "forward", side_effect=capturing_forward):
            trainer.train_step(batch)

        assert "target_lengths" in captured_kwargs
        # デバイス転送後の比較のため、同じデバイスに移動して比較
        assert torch.equal(
            captured_kwargs["target_lengths"], batch["label_lengths"].to(trainer.device)
        )
