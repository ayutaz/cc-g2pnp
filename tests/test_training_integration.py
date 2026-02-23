"""Phase 3 統合テスト: 全モジュールの連携を検証する。"""

from __future__ import annotations

import inspect
from unittest.mock import patch

import torch

from cc_g2pnp.data.vocabulary import PnPVocabulary
from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.training.checkpoint import CheckpointManager
from cc_g2pnp.training.config import TrainingConfig
from cc_g2pnp.training.evaluator import Evaluator
from cc_g2pnp.training.optimizer import build_optimizer, build_scheduler

# ---------------------------------------------------------------------------
# 小さいモデル設定 (テスト用)
# ---------------------------------------------------------------------------
_SMALL_MODEL_CONFIG = CC_G2PnPConfig(
    d_model=64,
    num_heads=2,
    d_ff=128,
    num_layers=2,
    intermediate_ctc_layers=(1,),
)


def _make_dummy_batch(
    batch_size: int = 2,
    seq_len: int = 10,
    target_len: int = 40,
) -> dict[str, torch.Tensor]:
    """テスト用ダミーバッチを生成する。"""
    return {
        "input_ids": torch.randint(0, 65000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(1, 139, (batch_size, target_len)),
        "input_lengths": torch.tensor([seq_len] * batch_size),
        "label_lengths": torch.tensor([target_len] * batch_size),
    }


# ---------------------------------------------------------------------------
# 1. Import テスト
# ---------------------------------------------------------------------------
class TestTrainingModuleImports:
    """training モジュールの公開 API が import できることを確認。"""

    def test_training_module_imports(self) -> None:
        """training モジュールの全 export が import できることを確認。"""
        from cc_g2pnp.training import (
            CheckpointManager,
            Evaluator,
            Trainer,
            TrainingConfig,
            TrainingLogger,
            build_optimizer,
            build_scheduler,
            cleanup_ddp,
            get_rank,
            get_world_size,
            is_main_process,
            reduce_metrics,
            setup_ddp,
            wrap_model_ddp,
        )

        assert all(
            x is not None
            for x in [
                CheckpointManager,
                Evaluator,
                Trainer,
                TrainingConfig,
                TrainingLogger,
                build_optimizer,
                build_scheduler,
                cleanup_ddp,
                get_rank,
                get_world_size,
                is_main_process,
                reduce_metrics,
                setup_ddp,
                wrap_model_ddp,
            ]
        )


# ---------------------------------------------------------------------------
# 2. Data - Model 語彙サイズ整合性テスト
# ---------------------------------------------------------------------------
class TestDataModelVocabConsistency:
    """データモジュールとモデルモジュールの語彙サイズが一致することを確認。"""

    def test_data_model_vocab_consistency(self) -> None:
        """PnPVocabulary と CC_G2PnPConfig のデフォルト vocab_size が一致。"""
        vocab = PnPVocabulary()
        config = CC_G2PnPConfig()

        assert vocab.vocab_size == config.pnp_vocab_size == 140
        assert config.bpe_vocab_size == 65000


# ---------------------------------------------------------------------------
# 3. Collator - Model インターフェーステスト
# ---------------------------------------------------------------------------
class TestCollatorToModelInterface:
    """Collator 出力が Model の forward() 引数と対応していることを確認。"""

    def test_collator_to_model_interface(self) -> None:
        """forward() の引数名にキー名が含まれていることを確認。"""
        sig = inspect.signature(CC_G2PnP.forward)
        params = list(sig.parameters.keys())

        # forward の引数: self, input_ids, input_lengths, targets, target_lengths
        assert "input_ids" in params
        assert "input_lengths" in params
        assert "targets" in params
        assert "target_lengths" in params

        # Collator -> Model のマッピング確認:
        # collator["labels"]         -> model(targets=...)
        # collator["label_lengths"]  -> model(target_lengths=...)
        # collator["input_ids"]      -> model(input_ids=...)
        # collator["input_lengths"]  -> model(input_lengths=...)


# ---------------------------------------------------------------------------
# 4. Model + Optimizer + Scheduler 統合テスト
# ---------------------------------------------------------------------------
class TestModelOptimizerSchedulerIntegration:
    """モデル -> Optimizer -> Scheduler の統合テスト。"""

    def test_model_optimizer_scheduler_integration(self) -> None:
        """build_optimizer / build_scheduler で 1 ステップ訓練できることを確認。"""
        training_config = TrainingConfig(max_steps=10, warmup_steps=2)
        model = CC_G2PnP(_SMALL_MODEL_CONFIG)
        optimizer = build_optimizer(model, training_config)
        scheduler = build_scheduler(optimizer, training_config)

        assert optimizer is not None
        assert scheduler is not None

        # 1 ステップ実行
        batch = _make_dummy_batch()
        result = model(
            batch["input_ids"],
            batch["input_lengths"],
            batch["labels"],
            batch["label_lengths"],
        )
        loss = result["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


# ---------------------------------------------------------------------------
# 5. Checkpoint 保存・復元 E2E テスト
# ---------------------------------------------------------------------------
class TestCheckpointSaveLoadE2E:
    """チェックポイント保存 -> 復元の E2E テスト。"""

    def test_checkpoint_save_load_e2e(self, tmp_path: object) -> None:
        """保存したチェックポイントが正しく復元できることを確認。"""
        training_config = TrainingConfig(max_steps=10, warmup_steps=0)
        model = CC_G2PnP(_SMALL_MODEL_CONFIG)
        optimizer = build_optimizer(model, training_config)
        scheduler = build_scheduler(optimizer, training_config)

        ckpt_manager = CheckpointManager(tmp_path, keep_last_n=2)
        ckpt_manager.save(
            step=100,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=training_config,
            metrics={"loss": 0.5},
        )

        loaded = ckpt_manager.load_latest()
        assert loaded is not None
        assert loaded["step"] == 100
        assert loaded["metrics"]["loss"] == 0.5

        # state_dict 復元
        model2 = CC_G2PnP(_SMALL_MODEL_CONFIG)
        model2.load_state_dict(loaded["model_state_dict"])


# ---------------------------------------------------------------------------
# 6. Evaluator 統合テスト
# ---------------------------------------------------------------------------
class TestEvaluatorWithRealModel:
    """Evaluator が実際のモデルで動作することを確認。"""

    def test_evaluator_with_real_model(self) -> None:
        """Evaluator.evaluate() が val_loss, val_cer を返すことを確認。"""
        model = CC_G2PnP(_SMALL_MODEL_CONFIG)
        vocab = PnPVocabulary()
        device = torch.device("cpu")

        evaluator = Evaluator(vocab, device)

        val_batches = [_make_dummy_batch()]

        metrics = evaluator.evaluate(model, val_batches)
        assert "val_loss" in metrics
        assert "val_cer" in metrics
        assert isinstance(metrics["val_loss"], float)
        assert isinstance(metrics["val_cer"], float)


# ---------------------------------------------------------------------------
# 7. Trainer E2E テスト (最小構成)
# ---------------------------------------------------------------------------
class TestTrainerE2EMinimal:
    """Trainer の最小 E2E テスト (3 ステップ)。"""

    def test_trainer_e2e_minimal(self, tmp_path: object) -> None:
        """Trainer.train() が 3 ステップ正常に動作することを確認。"""
        from cc_g2pnp.training.trainer import Trainer

        training_config = TrainingConfig(
            max_steps=3,
            use_amp=False,
            use_tensorboard=False,
            use_wandb=False,
            checkpoint_dir=str(tmp_path / "ckpt"),  # type: ignore[operator]
            save_every_n_steps=100,
            val_every_n_steps=100,
            log_every_n_steps=1,
            warmup_steps=0,
        )

        trainer = Trainer(_SMALL_MODEL_CONFIG, training_config)

        dummy_batch = _make_dummy_batch()

        def mock_data_iter():
            while True:
                yield dummy_batch

        with (
            patch.object(
                trainer, "_create_data_iterator", return_value=mock_data_iter()
            ),
            patch.object(trainer, "_create_val_batches", return_value=[]),
        ):
            trainer.train()
