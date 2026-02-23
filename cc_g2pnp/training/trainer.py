"""CC-G2PnP モデルの訓練を管理するメインクラス。"""

from __future__ import annotations

import dataclasses
import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from cc_g2pnp.data.collator import DynamicBatchCollator, dynamic_batch_sampler
from cc_g2pnp.data.dataset import G2PnPDataset
from cc_g2pnp.data.vocabulary import PnPVocabulary
from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.training.checkpoint import CheckpointManager
from cc_g2pnp.training.distributed import (
    cleanup_ddp,
    is_main_process,
    reduce_metrics,
    setup_ddp,
    wrap_model_ddp,
)
from cc_g2pnp.training.evaluator import Evaluator
from cc_g2pnp.training.logger import TrainingLogger
from cc_g2pnp.training.optimizer import build_optimizer, build_scheduler

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch.optim.lr_scheduler import LRScheduler

    from cc_g2pnp.model.config import CC_G2PnPConfig
    from cc_g2pnp.training.config import TrainingConfig

logger = logging.getLogger(__name__)

_VAL_NUM_SAMPLES_KEY = frozenset({"val_num_samples"})


class Trainer:
    """CC-G2PnP モデルの訓練を管理するメインクラス。"""

    def __init__(
        self,
        model_config: CC_G2PnPConfig,
        training_config: TrainingConfig,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """初期化。

        Args:
            model_config: モデル設定。
            training_config: 訓練設定。
            rank: DDP プロセスランク。
            world_size: DDP ワールドサイズ。
        """
        self.model_config = model_config
        self.training_config = training_config
        self.rank = rank
        self.world_size = world_size

        # 1. Seed 設定
        self._set_seed(training_config.seed)

        # 2. デバイス決定
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")

        # 3. DDP セットアップ
        if training_config.use_ddp:
            setup_ddp(rank, world_size)

        # 4. モデル作成 → デバイスに移動
        self.model: nn.Module = CC_G2PnP(model_config).to(self.device)

        # 5. DDP ラップ
        if training_config.use_ddp:
            self.model = wrap_model_ddp(self.model, rank)

        # 6. Optimizer/Scheduler 構築
        self.optimizer = build_optimizer(self.model, training_config)
        self.scheduler: LRScheduler = build_scheduler(self.optimizer, training_config)

        # 7. AMP 設定
        amp_dtype = (
            torch.bfloat16
            if training_config.amp_dtype == "bfloat16"
            else torch.float16
        )
        self.amp_dtype = amp_dtype
        # bfloat16 は GradScaler 不要
        self.scaler = torch.amp.GradScaler(
            enabled=(
                training_config.use_amp
                and amp_dtype == torch.float16
                and self.device.type == "cuda"
            ),
        )

        # 8. CheckpointManager 構築
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=training_config.checkpoint_dir,
            keep_last_n=training_config.keep_last_n,
        )

        # 9. TrainingLogger 構築 (メインプロセスのみ)
        self.logger: TrainingLogger | None = None
        if is_main_process():
            self.logger = TrainingLogger(training_config)

        # 10. Evaluator 構築
        self.vocabulary = PnPVocabulary()
        self.evaluator = Evaluator(self.vocabulary, self.device)

        # 11. Collator
        self.collator = DynamicBatchCollator(
            pad_token_id=1,
            label_pad_id=-100,
        )

        # 12. エポックカウンタ (データ再開時のシャッフルシード用)
        self._epoch = 0

        # 13. チェックポイントからの復元
        self.global_step = self._restore_checkpoint()

    def train(self) -> None:
        """メイン訓練ループ。"""
        config = self.training_config
        start_step = self.global_step
        effective_steps = config.effective_steps

        # ハイパーパラメータのログ
        if self.logger is not None:
            self.logger.log_hyperparams(dataclasses.asdict(config))

        # バリデーションバッチの事前ロード
        val_batches = self._create_val_batches()

        # データイテレータ作成
        data_iter = self._create_data_iterator()

        # 進捗バー (メインプロセスのみ)
        pbar = None
        if is_main_process():
            pbar = tqdm(
                range(start_step, effective_steps),
                initial=start_step,
                total=effective_steps,
                desc="Training",
            )

        try:
            for step in range(start_step, effective_steps):
                self.global_step = step

                # バッチ取得 (StopIteration でエポック再開)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    logger.info("Epoch ended at step %d, restarting data iterator", step)
                    self._epoch += 1
                    data_iter = self._create_data_iterator()
                    batch = next(data_iter)

                # Train step
                step_metrics = self.train_step(batch)

                # DDP メトリクス平均化
                if config.use_ddp:
                    step_metrics = reduce_metrics(step_metrics, self.device)

                # ログ (step=0 も含む: 初期 loss の記録に有用)
                if step % config.log_every_n_steps == 0 and self.logger is not None:
                    self.logger.log_metrics(
                        {
                            "train/loss": step_metrics["loss"],
                            "train/lr": step_metrics["lr"],
                            "train/grad_norm": step_metrics["grad_norm"],
                        },
                        step=step,
                    )

                # 進捗バー更新
                if pbar is not None:
                    pbar.set_postfix(
                        loss=f"{step_metrics['loss']:.4f}",
                        lr=f"{step_metrics['lr']:.2e}",
                    )
                    pbar.update(1)

                # バリデーション
                if (
                    step > 0
                    and step % config.val_every_n_steps == 0
                    and val_batches
                ):
                    val_metrics = self.evaluator.evaluate(
                        self.model, val_batches,
                    )
                    if config.use_ddp:
                        val_metrics = reduce_metrics(
                            val_metrics, self.device,
                            sum_keys=_VAL_NUM_SAMPLES_KEY,
                        )
                    if self.logger is not None:
                        self.logger.log_metrics(val_metrics, step=step)
                    if is_main_process():
                        logger.info(
                            "Step %d validation: loss=%.4f, cer=%.4f, samples=%d",
                            step,
                            val_metrics["val_loss"],
                            val_metrics["val_cer"],
                            val_metrics["val_num_samples"],
                        )

                # チェックポイント保存
                if (
                    step > 0
                    and step % config.save_every_n_steps == 0
                    and is_main_process()
                ):
                    self._save_checkpoint(step, step_metrics)
                    # DDP: 他のランクが save 完了を待つ
                    if config.use_ddp:
                        import torch.distributed as _dist

                        _dist.barrier()

            # 最終ステップ更新
            self.global_step = effective_steps

            # 最終チェックポイント保存
            if is_main_process():
                self._save_checkpoint(effective_steps)
                logger.info("Training complete at step %d", effective_steps)

        finally:
            if pbar is not None:
                pbar.close()
            if self.logger is not None:
                self.logger.close()
            if config.use_ddp:
                cleanup_ddp()

    def train_step(self, batch: dict) -> dict[str, float]:
        """1ステップの訓練を実行する。

        Args:
            batch: Collator 出力のバッチ辞書。

        Returns:
            {"loss": float, "lr": float, "grad_norm": float}
        """
        self.model.train()
        config = self.training_config

        # バッチをデバイスに転送
        input_ids = batch["input_ids"].to(self.device)
        input_lengths = batch["input_lengths"].to(self.device)
        labels = batch["labels"].to(self.device)
        label_lengths = batch["label_lengths"].to(self.device)

        # AMP context
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=config.use_amp,
        ):
            result = self.model(
                input_ids=input_ids,
                input_lengths=input_lengths,
                targets=labels,
                target_lengths=label_lengths,
            )
            loss = result["loss"]

        # Backward
        self.scaler.scale(loss).backward()

        # Gradient unscale + clipping
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), config.max_grad_norm,
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # Zero grad
        self.optimizer.zero_grad(set_to_none=True)

        return {
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm.item()
            if isinstance(grad_norm, torch.Tensor)
            else float(grad_norm),
        }

    def _save_checkpoint(
        self, step: int, metrics: dict[str, float] | None = None,
    ) -> None:
        """チェックポイントを保存する (GradScaler state_dict を含む)。"""
        self.checkpoint_manager.save(
            step=step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.training_config,
            metrics=metrics,
            scaler_state_dict=self.scaler.state_dict(),
        )

    def _create_data_iterator(self) -> Iterator:
        """データイテレータを作成する。

        G2PnPDataset -> dynamic_batch_sampler -> DynamicBatchCollator

        Returns:
            コレートされたバッチのイテレータ。
        """
        dataset = G2PnPDataset(
            subset=self.training_config.dataset_subset,
            streaming=True,
            shuffle_seed=self.training_config.seed + self._epoch,
        )
        sampler = dynamic_batch_sampler(
            dataset, max_tokens=self.training_config.max_tokens_per_batch,
        )
        return (self.collator(batch) for batch in sampler)

    def _create_val_batches(self) -> list[dict]:
        """バリデーションバッチを作成する。

        G2PnPDataset から指定数のサンプルを事前にロードしてバッチ化。
        ネットワーク接続が必要なため、利用不可の場合は空リストを返す。

        Returns:
            バリデーションバッチのリスト。
        """
        try:
            dataset = G2PnPDataset(
                subset=self.training_config.dataset_subset,
                streaming=True,
            )
            # 最大100サンプルを収集
            samples: list[dict] = []
            for sample in dataset:
                samples.append(sample)
                if len(samples) >= 100:
                    break

            if not samples:
                return []

            # dynamic_batch_sampler でバッチ化
            batches = []
            for batch_samples in dynamic_batch_sampler(
                iter(samples),
                max_tokens=self.training_config.max_tokens_per_batch,
            ):
                batches.append(self.collator(batch_samples))

            logger.info("Loaded %d validation batches (%d samples)", len(batches), len(samples))
            return batches
        except Exception:
            logger.warning("Failed to load validation data, skipping validation", exc_info=True)
            return []

    def _restore_checkpoint(self) -> int:
        """チェックポイントから復元する。

        Returns:
            復元したステップ番号 (チェックポイントなしなら 0)。
        """
        checkpoint = self.checkpoint_manager.load_latest()
        if checkpoint is None:
            logger.info("No checkpoint found, starting from step 0")
            return 0

        step = checkpoint["step"]

        # モデル state_dict 復元
        model_to_load = (
            self.model.module if hasattr(self.model, "module") else self.model
        )
        model_to_load.load_state_dict(checkpoint["model_state_dict"])

        # Optimizer/Scheduler state_dict 復元
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # GradScaler state_dict 復元
        scaler_sd = checkpoint.get("scaler_state_dict")
        if scaler_sd is not None:
            self.scaler.load_state_dict(scaler_sd)

        logger.info("Restored checkpoint at step %d", step)
        return step

    @staticmethod
    def _set_seed(seed: int) -> None:
        """再現性のためにシードを設定する。

        Args:
            seed: ランダムシード。
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
