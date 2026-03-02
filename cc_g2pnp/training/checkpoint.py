"""チェックポイントの保存・読み込み・クリーンアップを管理するモジュール。"""

from __future__ import annotations

import dataclasses
import logging
import pickle
import threading
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """チェックポイントの保存・読み込み・クリーンアップを管理する。"""

    def __init__(self, checkpoint_dir: str | Path, keep_last_n: int = 5, async_save: bool = False) -> None:
        """初期化。

        Args:
            checkpoint_dir: チェックポイント保存ディレクトリ。
            keep_last_n: 保持する最新チェックポイント数。
            async_save: True の場合、バックグラウンドスレッドで非同期保存する (デフォルト: False)。
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_last_n = keep_last_n
        self.async_save = async_save
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._save_thread: threading.Thread | None = None
        self._save_error: Exception | None = None

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        config: object,
        metrics: dict[str, float] | None = None,
        scaler_state_dict: dict | None = None,
    ) -> Path:
        """チェックポイントを保存する。

        一時ファイルに書き込み後、アトミックにリネームする。

        Args:
            step: 現在のトレーニングステップ。
            model: モデル。DDP の場合は model.module を自動取得。
            optimizer: オプティマイザ。
            scheduler: スケジューラ。
            config: TrainingConfig (dataclass)。
            metrics: メトリクス辞書。
            scaler_state_dict: GradScaler の state_dict (AMP float16 使用時)。

        Returns:
            保存したファイルのパス。
        """
        # 前回の非同期保存が完了するまで待機
        self.wait_for_save()

        path = self.checkpoint_dir / f"step_{step:08d}.pt"
        tmp_path = path.with_suffix(".pt.tmp")

        # DDP モデルの場合は module から state_dict を取得
        model_to_save = model.module if hasattr(model, "module") else model

        checkpoint = {
            "step": step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": dataclasses.asdict(config),
            "metrics": metrics,
            "scaler_state_dict": scaler_state_dict,
        }

        # モデル設定 (CC_G2PnPConfig) も保存 — from_checkpoint で復元に使用
        if hasattr(model_to_save, "config") and dataclasses.is_dataclass(
            model_to_save.config
        ):
            checkpoint["model_config"] = dataclasses.asdict(model_to_save.config)

        if self.async_save:
            # バックグラウンドスレッドで非同期保存
            self._save_error = None
            self._save_thread = threading.Thread(
                target=self._background_save,
                args=(checkpoint, tmp_path, path),
                daemon=True,
            )
            self._save_thread.start()
            logger.info("Async checkpoint save started: %s", path)
        else:
            # 同期保存 (元の動作)
            torch.save(checkpoint, tmp_path)
            tmp_path.rename(path)
            logger.info("Checkpoint saved: %s", path)

        self.cleanup()
        return path

    def _background_save(self, checkpoint: dict, tmp_path: Path, path: Path) -> None:
        """バックグラウンドスレッドでチェックポイントを保存する。"""
        try:
            torch.save(checkpoint, tmp_path)
            tmp_path.rename(path)
            logger.info("Async checkpoint saved: %s", path)
        except Exception as e:
            self._save_error = e
            logger.error("Async checkpoint save failed: %s", e)

    def wait_for_save(self) -> None:
        """バックグラウンド保存が完了するまでブロックする。

        保存中にエラーが発生した場合は RuntimeError を送出する。
        """
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()
        if self._save_error is not None:
            err = self._save_error
            self._save_error = None
            raise RuntimeError(f"Async checkpoint save failed: {err}") from err

    def close(self) -> None:
        """未完了のバックグラウンド保存を待機してリソースを解放する。"""
        self.wait_for_save()

    def load_latest(self) -> dict | None:
        """最新のチェックポイントを読み込む。

        最新ファイルが破損している場合、1つ前のチェックポイントにフォールバックする。

        Returns:
            チェックポイント辞書。なければ None。
        """
        checkpoints = self._sorted_checkpoints()
        if not checkpoints:
            return None

        # 最新から順に試行し、破損時はフォールバック
        for ckpt_path in reversed(checkpoints):
            try:
                return self.load(ckpt_path)
            except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as e:
                logger.warning(
                    "Failed to load checkpoint %s: %s. Trying previous.",
                    ckpt_path,
                    e,
                )
        logger.error("All checkpoints are corrupted")
        return None

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
        # セキュリティ: weights_only=True で pickle 任意コード実行リスクを排除する。
        # チェックポイントは dataclasses.asdict() でプレーン dict に変換済みのため
        # state_dict (テンソル + Python プリミティブ) のみで構成され、
        # weights_only=True での読み込みが可能。
        # 注意: 信頼できるソースのチェックポイントのみ読み込むこと。
        # map_location="cpu" で読み込み、呼び出し元で適切なデバイスに移動する
        return torch.load(path, weights_only=True, map_location="cpu")

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
        valid: list[tuple[int, Path]] = []
        for p in checkpoints:
            try:
                step_num = int(p.stem.split("_")[1])
                valid.append((step_num, p))
            except (ValueError, IndexError):
                logger.warning("Skipping invalid checkpoint filename: %s", p)
        valid.sort(key=lambda x: x[0])
        return [p for _, p in valid]
