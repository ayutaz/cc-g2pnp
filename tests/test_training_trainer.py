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

    def test_optimizer_step_returns_finite_grad_norm(self, small_model_config, training_config):
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        trainer.train_step(batch)
        grad_norm = trainer._optimizer_step()

        assert isinstance(grad_norm, float)
        assert torch.isfinite(torch.tensor(grad_norm))


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


class TestDDPBarrierPlacement:
    """DDP barrier がチェックポイント保存時に全ランクで呼ばれることのテスト。"""

    def test_barrier_called_outside_is_main_process(
        self, small_model_config, tmp_path,
    ):
        """barrier は is_main_process() ブロックの外で呼ばれる (rank!=0 でも)。"""
        from cc_g2pnp.training.trainer import Trainer

        training_config = TrainingConfig(
            max_steps=2,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=1,
            val_every_n_steps=100,
            warmup_steps=0,
            use_ddp=True,
        )

        # DDP 関連 + CUDA デバイス割り当てをパッチした状態で Trainer を構築
        with (
            patch("cc_g2pnp.training.trainer.setup_ddp"),
            patch("cc_g2pnp.training.trainer.wrap_model_ddp", side_effect=lambda m, r: m),
            patch("cc_g2pnp.training.trainer.cleanup_ddp"),
            patch("cc_g2pnp.training.trainer.is_main_process", return_value=False),
            patch("cc_g2pnp.training.trainer.reduce_metrics", side_effect=lambda m, d, **kw: m),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.distributed.barrier") as mock_barrier,
        ):
            trainer = Trainer(small_model_config, training_config, rank=1, world_size=2)
            batch = _make_dummy_batch(
                bpe_vocab_size=small_model_config.bpe_vocab_size,
                pnp_vocab_size=small_model_config.pnp_vocab_size,
            )

            trainer._create_data_iterator = MagicMock(
                return_value=iter([batch] * training_config.max_steps),
            )
            trainer._create_val_batches = MagicMock(return_value=[])

            trainer.train()

            # rank != 0 でも barrier が呼ばれることを確認
            assert mock_barrier.call_count >= 1

    def test_non_main_rank_does_not_save_checkpoint(
        self, small_model_config, tmp_path,
    ):
        """rank != 0 ではチェックポイント保存されない。"""
        from cc_g2pnp.training.trainer import Trainer

        training_config = TrainingConfig(
            max_steps=2,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=1,
            val_every_n_steps=100,
            warmup_steps=0,
            use_ddp=True,
        )

        with (
            patch("cc_g2pnp.training.trainer.setup_ddp"),
            patch("cc_g2pnp.training.trainer.wrap_model_ddp", side_effect=lambda m, r: m),
            patch("cc_g2pnp.training.trainer.cleanup_ddp"),
            patch("cc_g2pnp.training.trainer.is_main_process", return_value=False),
            patch("cc_g2pnp.training.trainer.reduce_metrics", side_effect=lambda m, d, **kw: m),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.distributed.barrier"),
        ):
            trainer = Trainer(small_model_config, training_config, rank=1, world_size=2)
            batch = _make_dummy_batch(
                bpe_vocab_size=small_model_config.bpe_vocab_size,
                pnp_vocab_size=small_model_config.pnp_vocab_size,
            )

            trainer._create_data_iterator = MagicMock(
                return_value=iter([batch] * training_config.max_steps),
            )
            trainer._create_val_batches = MagicMock(return_value=[])
            trainer._save_checkpoint = MagicMock()

            trainer.train()

            trainer._save_checkpoint.assert_not_called()


class TestDDPDataSharding:
    """DDP データシャーディングのテスト。"""

    def test_data_iterator_passes_rank_world_size(
        self, small_model_config, training_config,
    ):
        """_create_data_iterator が rank/world_size を G2PnPDataset に渡す。"""
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config, rank=0, world_size=1)
        # rank/world_size をテスト用に上書き
        trainer.rank = 2
        trainer.world_size = 4

        with patch("cc_g2pnp.training.trainer.G2PnPDataset") as MockDataset:
            mock_ds = MagicMock()
            mock_ds.__iter__ = MagicMock(return_value=iter([]))
            MockDataset.return_value = mock_ds

            with patch(
                "cc_g2pnp.training.trainer.dynamic_batch_sampler",
                return_value=iter([]),
            ):
                trainer._create_data_iterator()

        MockDataset.assert_called_once()
        call_kwargs = MockDataset.call_args
        assert call_kwargs.kwargs.get("rank") == 2 or call_kwargs[1].get("rank") == 2
        assert call_kwargs.kwargs.get("world_size") == 4 or call_kwargs[1].get("world_size") == 4


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


class TestGradientAccumulation:
    """勾配累積のテスト。"""

    def test_accum_steps_1_backward_compat(self, small_model_config, training_config):
        """gradient_accumulation_steps=1 のとき、train_step が max_steps 回呼ばれる。"""
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

        original_train_step = trainer.train_step
        original_optimizer_step = trainer._optimizer_step
        train_step_count = 0
        optimizer_step_count = 0

        def counting_train_step(b):
            nonlocal train_step_count
            train_step_count += 1
            return original_train_step(b)

        def counting_optimizer_step():
            nonlocal optimizer_step_count
            optimizer_step_count += 1
            return original_optimizer_step()

        trainer.train_step = counting_train_step
        trainer._optimizer_step = counting_optimizer_step

        trainer.train()

        assert train_step_count == training_config.max_steps
        assert optimizer_step_count == training_config.max_steps

    def test_accum_steps_2_optimizer_called_half(self, small_model_config, tmp_path):
        """gradient_accumulation_steps=2 のとき、optimizer.step は max_steps 回、
        train_step は max_steps * 2 回呼ばれる。"""
        from cc_g2pnp.training.trainer import Trainer

        accum_steps = 2
        max_steps = 4
        config = TrainingConfig(
            max_steps=max_steps,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            gradient_accumulation_steps=accum_steps,
        )

        trainer = Trainer(small_model_config, config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        # Need max_steps * accum_steps batches
        trainer._create_data_iterator = MagicMock(
            return_value=iter([batch] * (max_steps * accum_steps)),
        )
        trainer._create_val_batches = MagicMock(return_value=[])

        original_train_step = trainer.train_step
        original_optimizer_step = trainer._optimizer_step
        train_step_count = 0
        optimizer_step_count = 0

        def counting_train_step(b):
            nonlocal train_step_count
            train_step_count += 1
            return original_train_step(b)

        def counting_optimizer_step():
            nonlocal optimizer_step_count
            optimizer_step_count += 1
            return original_optimizer_step()

        trainer.train_step = counting_train_step
        trainer._optimizer_step = counting_optimizer_step

        trainer.train()

        # train_step は max_steps * accum_steps 回呼ばれる
        assert train_step_count == max_steps * accum_steps
        # optimizer step は max_steps 回 (論理ステップごとに1回)
        assert optimizer_step_count == max_steps

    def test_accum_steps_2_loss_averaged(self, small_model_config, tmp_path):
        """gradient_accumulation_steps=2 のとき、step_metrics の loss はマイクロステップの平均。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=1,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            gradient_accumulation_steps=2,
        )

        trainer = Trainer(small_model_config, config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        trainer._create_data_iterator = MagicMock(
            return_value=iter([batch] * 2),
        )
        trainer._create_val_batches = MagicMock(return_value=[])

        logged_metrics = {}

        def capture_log_metrics(metrics, step):
            logged_metrics.update(metrics)

        trainer.logger = MagicMock()
        trainer.logger.log_metrics = capture_log_metrics

        trainer.train()

        assert "train/loss" in logged_metrics
        assert isinstance(logged_metrics["train/loss"], float)
        assert logged_metrics["train/loss"] > 0


class TestPretrainedWeightsOnly:
    """pretrained_weights_only のテスト。"""

    def test_returns_step_zero_with_weights_loaded(self, small_model_config, tmp_path):
        """pretrained_weights_only=True のとき、checkpoint からモデル重みは復元するが step=0 を返す。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=3,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            pretrained_weights_only=True,
        )

        trainer = Trainer(small_model_config, config)

        # Fake checkpoint at step 500
        fake_checkpoint = {
            "step": 500,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
        }

        trainer.checkpoint_manager.load_latest = MagicMock(return_value=fake_checkpoint)

        start_step = trainer._restore_checkpoint()

        assert start_step == 0

    def test_optimizer_not_restored(self, small_model_config, tmp_path):
        """pretrained_weights_only=True のとき、optimizer state は復元されない。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=3,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            pretrained_weights_only=True,
        )

        trainer = Trainer(small_model_config, config)

        fake_checkpoint = {
            "step": 500,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
        }

        trainer.checkpoint_manager.load_latest = MagicMock(return_value=fake_checkpoint)

        # Spy on optimizer.load_state_dict
        trainer.optimizer.load_state_dict = MagicMock()
        trainer.scheduler.load_state_dict = MagicMock()

        trainer._restore_checkpoint()

        trainer.optimizer.load_state_dict.assert_not_called()
        trainer.scheduler.load_state_dict.assert_not_called()

    def test_normal_restore_loads_optimizer(self, small_model_config, tmp_path):
        """pretrained_weights_only=False (デフォルト) のとき、optimizer/scheduler は復元される。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=3,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            pretrained_weights_only=False,
        )

        trainer = Trainer(small_model_config, config)

        fake_checkpoint = {
            "step": 500,
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "scheduler_state_dict": trainer.scheduler.state_dict(),
        }

        trainer.checkpoint_manager.load_latest = MagicMock(return_value=fake_checkpoint)

        start_step = trainer._restore_checkpoint()

        assert start_step == 500


class TestPhase1Optimizations:
    """Phase 1 最適化のテスト。"""

    def test_reduce_metrics_only_at_log_steps(self, small_model_config, tmp_path):
        """DDP 時、reduce_metrics は log_every_n_steps ごとにのみ呼ばれる。"""
        from cc_g2pnp.training.trainer import Trainer

        training_config = TrainingConfig(
            max_steps=5,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=2,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            use_ddp=True,
        )

        with (
            patch("cc_g2pnp.training.trainer.setup_ddp"),
            patch("cc_g2pnp.training.trainer.wrap_model_ddp", side_effect=lambda m, r: m),
            patch("cc_g2pnp.training.trainer.cleanup_ddp"),
            patch("cc_g2pnp.training.trainer.is_main_process", return_value=True),
            patch("cc_g2pnp.training.trainer.reduce_metrics", side_effect=lambda m, d, **kw: m) as mock_reduce,
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.distributed.barrier"),
        ):
            trainer = Trainer(small_model_config, training_config, rank=0, world_size=2)
            batch = _make_dummy_batch(
                bpe_vocab_size=small_model_config.bpe_vocab_size,
                pnp_vocab_size=small_model_config.pnp_vocab_size,
            )

            trainer._create_data_iterator = MagicMock(
                return_value=iter([batch] * training_config.max_steps),
            )
            trainer._create_val_batches = MagicMock(return_value=[])

            trainer.train()

            # log_every=2, steps 0-4: step 0, 2, 4 でログ → reduce は 3 回
            # (毎ステップの 5 回ではない)
            assert mock_reduce.call_count == 3

    def test_empty_cache_called_around_validation(self, small_model_config, tmp_path):
        """CUDA 環境ではバリデーション前後に empty_cache が呼ばれる。"""
        from cc_g2pnp.training.trainer import Trainer

        training_config = TrainingConfig(
            max_steps=3,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=1,
            warmup_steps=0,
        )

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        trainer._create_data_iterator = MagicMock(
            return_value=iter([batch] * training_config.max_steps),
        )
        # バリデーションバッチを用意 (空でないリスト)
        trainer._create_val_batches = MagicMock(return_value=[batch])

        with patch("torch.cuda.empty_cache") as mock_empty_cache:
            trainer.train()

            if trainer.device.type == "cuda":
                # CUDA: val_every=1, steps 1,2 でバリデーション → 前後で 2*2=4 回
                assert mock_empty_cache.call_count == 4
            else:
                # CPU: empty_cache は呼ばれない
                mock_empty_cache.assert_not_called()


class TestTorchCompile:
    """use_torch_compile フラグのテスト。"""

    def test_compile_disabled_by_default(self, small_model_config, training_config):
        """デフォルトでは use_torch_compile=False。"""
        assert training_config.use_torch_compile is False

    def test_compile_enabled_wraps_submodules(self, small_model_config, tmp_path):
        """use_torch_compile=True のとき、ffn1/ffn2/conv がコンパイルされる。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=3,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            use_torch_compile=True,
        )

        compile_calls = []

        def fake_compile(module, mode=None):
            compile_calls.append((module, mode))
            return module  # そのまま返す

        with patch("torch.compile", side_effect=fake_compile):
            trainer = Trainer(small_model_config, config)

        num_layers = len(trainer.model.encoder.layers)
        # 各レイヤーで ffn1, ffn2, conv の 3 モジュールがコンパイルされる
        assert len(compile_calls) == num_layers * 3

    def test_compile_disabled_does_not_call_torch_compile(self, small_model_config, training_config):
        """use_torch_compile=False のとき、torch.compile は呼ばれない。"""
        from cc_g2pnp.training.trainer import Trainer

        compile_calls = []

        def fake_compile(module, mode=None):
            compile_calls.append(module)
            return module

        with patch("torch.compile", side_effect=fake_compile):
            Trainer(small_model_config, training_config)

        assert len(compile_calls) == 0

    def test_compile_train_step_works(self, small_model_config, tmp_path):
        """use_torch_compile=True でも train_step が正常動作する。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=1,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            use_torch_compile=True,
        )

        # torch.compile はそのまま返す (テスト環境でコンパイルは不要)
        with patch("torch.compile", side_effect=lambda m, **kw: m):
            trainer = Trainer(small_model_config, config)

        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )
        result = trainer.train_step(batch)
        assert "loss" in result
        assert torch.isfinite(torch.tensor(result["loss"]))


class TestGradientCheckpointing:
    """gradient_checkpointing フラグのテスト。"""

    def test_gradient_checkpointing_enabled_by_default(self, training_config):
        """デフォルトでは gradient_checkpointing=True。"""
        assert training_config.gradient_checkpointing is True

    def test_gradient_checkpointing_true_sets_encoder_flag(self, small_model_config, tmp_path):
        """gradient_checkpointing=True のとき、encoder の _gradient_checkpointing が True。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=1,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            gradient_checkpointing=True,
        )
        trainer = Trainer(small_model_config, config)
        assert trainer.model.encoder._gradient_checkpointing is True

    def test_gradient_checkpointing_false_sets_encoder_flag(self, small_model_config, tmp_path):
        """gradient_checkpointing=False のとき、encoder の _gradient_checkpointing が False。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=1,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            gradient_checkpointing=False,
        )
        trainer = Trainer(small_model_config, config)
        assert trainer.model.encoder._gradient_checkpointing is False

    def test_gradient_checkpointing_false_train_step_works(self, small_model_config, tmp_path):
        """gradient_checkpointing=False でも train_step が正常動作する。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=1,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            gradient_checkpointing=False,
        )
        trainer = Trainer(small_model_config, config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )
        result = trainer.train_step(batch)
        assert torch.isfinite(torch.tensor(result["loss"]))


class TestDisableIntermediateCTC:
    """disable_intermediate_ctc_after のテスト。"""

    def test_intermediate_ctc_enabled_before_threshold(self, small_model_config, tmp_path):
        """global_step が threshold 未満のとき、中間 CTC が有効化される。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=10,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            disable_intermediate_ctc_after=5,
        )
        trainer = Trainer(small_model_config, config)
        trainer.global_step = 3  # threshold 未満

        captured_kwargs: dict = {}

        def capturing_forward(**kwargs):
            captured_kwargs.update(kwargs)
            # 実際に forward を呼ぶためダミー結果を返す
            from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
            return CC_G2PnP.forward(trainer.model, **kwargs)

        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        with patch.object(trainer.model, "forward", side_effect=capturing_forward):
            trainer.train_step(batch)

        assert captured_kwargs.get("enable_intermediate_ctc") is True

    def test_intermediate_ctc_disabled_at_or_after_threshold(self, small_model_config, tmp_path):
        """global_step が threshold 以上のとき、中間 CTC が無効化される。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=10,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            disable_intermediate_ctc_after=5,
        )
        trainer = Trainer(small_model_config, config)
        trainer.global_step = 5  # threshold 以上

        captured_kwargs: dict = {}

        def capturing_forward(**kwargs):
            captured_kwargs.update(kwargs)
            from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
            return CC_G2PnP.forward(trainer.model, **kwargs)

        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        with patch.object(trainer.model, "forward", side_effect=capturing_forward):
            trainer.train_step(batch)

        assert captured_kwargs.get("enable_intermediate_ctc") is False

    def test_intermediate_ctc_always_enabled_when_none(self, small_model_config, tmp_path):
        """disable_intermediate_ctc_after=None のとき、常に中間 CTC が有効。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=10,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            disable_intermediate_ctc_after=None,
        )
        trainer = Trainer(small_model_config, config)
        trainer.global_step = 999999  # 非常に大きいステップでも有効

        captured_kwargs: dict = {}

        def capturing_forward(**kwargs):
            captured_kwargs.update(kwargs)
            from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
            return CC_G2PnP.forward(trainer.model, **kwargs)

        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        with patch.object(trainer.model, "forward", side_effect=capturing_forward):
            trainer.train_step(batch)

        assert captured_kwargs.get("enable_intermediate_ctc") is True

    def test_intermediate_ctc_disabled_produces_no_inter_losses(self, small_model_config, tmp_path):
        """enable_intermediate_ctc=False のとき、intermediate_losses が空リストになる。"""
        import torch

        from cc_g2pnp.model.cc_g2pnp import CC_G2PnP

        model = CC_G2PnP(small_model_config)
        model.eval()

        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        with torch.no_grad():
            result = model(
                input_ids=batch["input_ids"],
                input_lengths=batch["input_lengths"],
                targets=batch["labels"],
                target_lengths=batch["label_lengths"],
                enable_intermediate_ctc=False,
            )

        assert result["intermediate_losses"] == []

    def test_intermediate_ctc_enabled_produces_inter_losses(self, small_model_config, tmp_path):
        """enable_intermediate_ctc=True のとき、intermediate_losses にエントリがある。"""
        import torch

        from cc_g2pnp.model.cc_g2pnp import CC_G2PnP

        model = CC_G2PnP(small_model_config)
        model.eval()

        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )

        with torch.no_grad():
            result = model(
                input_ids=batch["input_ids"],
                input_lengths=batch["input_lengths"],
                targets=batch["labels"],
                target_lengths=batch["label_lengths"],
                enable_intermediate_ctc=True,
            )

        assert len(result["intermediate_losses"]) > 0


class TestSortBatchBufferSize:
    """sort_batch_buffer_size のテスト。"""

    def test_sort_batch_buffer_size_zero_uses_dynamic_sampler(
        self, small_model_config, training_config
    ):
        """sort_batch_buffer_size=0 のとき、dynamic_batch_sampler が使われる。"""
        from cc_g2pnp.training.trainer import Trainer

        assert training_config.sort_batch_buffer_size == 10_000  # デフォルト値

        config_zero = TrainingConfig(
            max_steps=1,
            use_amp=False,
            checkpoint_dir=training_config.checkpoint_dir,
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            sort_batch_buffer_size=0,
        )
        trainer = Trainer(small_model_config, config_zero)

        with (
            patch("cc_g2pnp.training.trainer.G2PnPDataset") as MockDataset,
            patch("cc_g2pnp.training.trainer.dynamic_batch_sampler", return_value=iter([])) as mock_dyn,
            patch(
                "cc_g2pnp.training.trainer.sorted_dynamic_batch_sampler", return_value=iter([])
            ) as mock_sorted,
        ):
            mock_ds = MagicMock()
            mock_ds.__iter__ = MagicMock(return_value=iter([]))
            MockDataset.return_value = mock_ds
            trainer._create_data_iterator()

        mock_dyn.assert_called_once()
        mock_sorted.assert_not_called()

    def test_sort_batch_buffer_size_positive_uses_sorted_sampler(
        self, small_model_config, tmp_path
    ):
        """sort_batch_buffer_size > 0 のとき、sorted_dynamic_batch_sampler が使われる。"""
        from cc_g2pnp.training.trainer import Trainer

        config = TrainingConfig(
            max_steps=1,
            use_amp=False,
            checkpoint_dir=str(tmp_path / "ckpt"),
            log_every_n_steps=1,
            save_every_n_steps=100,
            val_every_n_steps=100,
            warmup_steps=0,
            sort_batch_buffer_size=100,
        )
        trainer = Trainer(small_model_config, config)

        with (
            patch("cc_g2pnp.training.trainer.G2PnPDataset") as MockDataset,
            patch("cc_g2pnp.training.trainer.dynamic_batch_sampler", return_value=iter([])) as mock_dyn,
            patch(
                "cc_g2pnp.training.trainer.sorted_dynamic_batch_sampler", return_value=iter([])
            ) as mock_sorted,
        ):
            mock_ds = MagicMock()
            mock_ds.__iter__ = MagicMock(return_value=iter([]))
            MockDataset.return_value = mock_ds
            trainer._create_data_iterator()

        mock_sorted.assert_called_once()
        # sorted_dynamic_batch_sampler が使われているとき、dynamic_batch_sampler は呼ばれない
        mock_dyn.assert_not_called()

    def test_sorted_dynamic_batch_sampler_importable(self):
        """sorted_dynamic_batch_sampler が trainer から import できること。"""
        from cc_g2pnp.training.trainer import sorted_dynamic_batch_sampler

        assert callable(sorted_dynamic_batch_sampler)


class TestForeachGradClip:
    """foreach=True の gradient clipping テスト。"""

    def test_clip_grad_norm_foreach_called(self, small_model_config, training_config):
        """_optimizer_step で clip_grad_norm_ が foreach=True で呼ばれる。"""
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )
        trainer.train_step(batch)

        with patch("cc_g2pnp.training.trainer.nn.utils.clip_grad_norm_") as mock_clip:
            mock_clip.return_value = torch.tensor(1.0)
            trainer._optimizer_step()

        mock_clip.assert_called_once()
        call_kwargs = mock_clip.call_args
        # foreach=True が渡されていることを確認
        assert call_kwargs.kwargs.get("foreach") is True or call_kwargs[1].get("foreach") is True

    def test_clip_grad_norm_produces_finite_grad_norm(self, small_model_config, training_config):
        """foreach=True でも grad_norm が finite float になる。"""
        from cc_g2pnp.training.trainer import Trainer

        trainer = Trainer(small_model_config, training_config)
        batch = _make_dummy_batch(
            bpe_vocab_size=small_model_config.bpe_vocab_size,
            pnp_vocab_size=small_model_config.pnp_vocab_size,
        )
        trainer.train_step(batch)
        grad_norm = trainer._optimizer_step()

        assert isinstance(grad_norm, float)
        assert torch.isfinite(torch.tensor(grad_norm))
