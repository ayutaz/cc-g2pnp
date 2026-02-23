"""チェックポイントの保存・読み込み・クリーンアップを管理するモジュール。"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """チェックポイントの保存・読み込み・クリーンアップを管理する。"""

    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 5) -> None:
        """初期化。

        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ。
            keep_last_n: 保持する最新チェックポイント数。
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        config: object,
        metrics: dict[str, float] | None = None,
    ) -> Path:
        """チェックポイントを保存する。

        Args:
            step: 現在のトレーニングステップ。
            model: モデル。DDP の場合は model.module を自動取得。
            optimizer: オプティマイザ。
            scheduler: スケジューラ。
            config: TrainingConfig (dataclass)。
            metrics: メトリクス辞書。

        Returns:
            保存したファイルのパス。
        """
        path = self.checkpoint_dir / f"step_{step:08d}.pt"

        # DDP モデルの場合は module から state_dict を取得
        model_to_save = model.module if hasattr(model, "module") else model

        checkpoint = {
            "step": step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": dataclasses.asdict(config),
            "metrics": metrics,
        }

        torch.save(checkpoint, path)
        logger.info("Checkpoint saved: %s", path)

        self.cleanup()
        return path

    def load_latest(self) -> dict | None:
        """最新のチェックポイントを読み込む。

        Returns:
            チェックポイント辞書。なければ None。
        """
        checkpoints = self._sorted_checkpoints()
        if not checkpoints:
            return None
        return self.load(checkpoints[-1])

    def load(self, path: str | Path) -> dict:
        """指定パスのチェックポイントを読み込む。

        Args:
            path: チェックポイントファイルパス。

        Returns:
            チェックポイント辞書。

        Raises:
            FileNotFoundError: ファイルが存在しない場合。
        """
        path = Path(path)
        if not path.exists():
            msg = f"Checkpoint not found: {path}"
            raise FileNotFoundError(msg)
        logger.info("Loading checkpoint: %s", path)
        return torch.load(path, weights_only=False)

    def cleanup(self) -> None:
        """古いチェックポイントを削除する。

        keep_last_n 個の最新チェックポイントのみ残し、古いものを削除。
        """
        checkpoints = self._sorted_checkpoints()
        to_delete = checkpoints[: -self.keep_last_n] if self.keep_last_n > 0 else checkpoints
        for path in to_delete:
            path.unlink()
            logger.info("Deleted old checkpoint: %s", path)

    def _sorted_checkpoints(self) -> list[Path]:
        """ステップ番号でソートされたチェックポイントファイルリストを返す。"""
        checkpoints = list(self.checkpoint_dir.glob("step_*.pt"))
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[1]))
        return checkpoints
