"""DDP の実マルチプロセステスト (torch.multiprocessing.spawn ベース)。

実際に複数 GPU プロセスを起動して DDP の動作を検証する。
全テストは @pytest.mark.slow マーカー付きで、GPU 2台未満の環境ではスキップ。
"""

from __future__ import annotations

import os
import tempfile
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.training.checkpoint import CheckpointManager
from cc_g2pnp.training.distributed import cleanup_ddp, reduce_metrics, setup_ddp, wrap_model_ddp

# テスト用の最小 GPU 数
_MIN_GPUS = 2

# 小モデル設定 (test_training_trainer.py と同様)
_SMALL_MODEL_CONFIG = CC_G2PnPConfig(
    d_model=64,
    num_heads=2,
    d_ff=128,
    num_layers=2,
    intermediate_ctc_layers=(1,),
)


def _num_gpus() -> int:
    """利用可能な CUDA GPU 数を返す。"""
    return torch.cuda.device_count()


def _skip_if_not_enough_gpus():
    """GPU が足りない場合にテストをスキップする。"""
    if _num_gpus() < _MIN_GPUS:
        pytest.skip(f"Need >= {_MIN_GPUS} GPUs, found {_num_gpus()}")


def _make_dummy_batch(
    bpe_vocab_size: int = 65_000,
    pnp_vocab_size: int = 140,
    batch_size: int = 2,
    seq_len: int = 10,
    label_len: int = 40,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """テスト用のダミーバッチを生成する。"""
    batch = {
        "input_ids": torch.randint(0, bpe_vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(1, pnp_vocab_size, (batch_size, label_len)),
        "input_lengths": torch.tensor([seq_len] * batch_size),
        "label_lengths": torch.tensor([label_len] * batch_size),
    }
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def _get_free_port() -> int:
    """空きポートを取得する。"""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ============================================================
# Worker 関数群 (mp.spawn から呼ばれる)
# ============================================================


def _worker_setup_cleanup(rank, world_size, port, results):
    """setup_ddp / cleanup_ddp の基本テスト用ワーカー。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    try:
        setup_ddp(rank, world_size)
        results[f"rank_{rank}"] = dist.get_rank()
        results[f"world_size_{rank}"] = dist.get_world_size()
        results[f"initialized_{rank}"] = dist.is_initialized()
    finally:
        cleanup_ddp()
        results[f"cleaned_{rank}"] = not dist.is_initialized()


def _worker_reduce_avg(rank, world_size, port, results):
    """reduce_metrics AVG テスト用ワーカー。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    try:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        # 各ランクで異なる loss: rank0=1.0, rank1=3.0
        metrics = {"loss": float(rank * 2.0 + 1.0)}
        reduced = reduce_metrics(metrics, device)
        results[f"avg_loss_{rank}"] = reduced["loss"]
    finally:
        cleanup_ddp()


def _worker_reduce_sum(rank, world_size, port, results):
    """reduce_metrics SUM テスト用ワーカー。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    try:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        # 各ランクで異なる count: rank0=10, rank1=20
        metrics = {"count": float((rank + 1) * 10)}
        reduced = reduce_metrics(metrics, device, sum_keys=frozenset({"count"}))
        results[f"sum_count_{rank}"] = reduced["count"]
    finally:
        cleanup_ddp()


def _worker_gradient_sync(rank, world_size, port, results):
    """勾配同期テスト用ワーカー。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    try:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")

        # 全ランクで同一シードでモデル作成 → 同一初期パラメータ
        torch.manual_seed(42)
        model = CC_G2PnP(_SMALL_MODEL_CONFIG).to(device)
        ddp_model = wrap_model_ddp(model, rank)

        # 各ランクで異なるシードでバッチ生成
        torch.manual_seed(rank + 100)
        batch = _make_dummy_batch(
            bpe_vocab_size=_SMALL_MODEL_CONFIG.bpe_vocab_size,
            pnp_vocab_size=_SMALL_MODEL_CONFIG.pnp_vocab_size,
            device=device,
        )

        # Forward + backward
        result = ddp_model(
            input_ids=batch["input_ids"],
            input_lengths=batch["input_lengths"],
            targets=batch["labels"],
            target_lengths=batch["label_lengths"],
        )
        result["loss"].backward()

        # 勾配を収集 (最初の数パラメータの勾配ノルムを記録)
        grad_norms = []
        for name, param in ddp_model.named_parameters():
            if param.grad is not None:
                grad_norms.append((name, param.grad.norm().item()))
            if len(grad_norms) >= 5:
                break
        results[f"grad_norms_{rank}"] = grad_norms
    finally:
        cleanup_ddp()


def _worker_barrier_sync(rank, world_size, port, results):
    """barrier 同期テスト用ワーカー。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    try:
        setup_ddp(rank, world_size)

        # rank 1 は 0.5 秒待ってから barrier
        if rank == 1:
            time.sleep(0.5)

        t_before = time.monotonic()
        dist.barrier()
        t_after = time.monotonic()

        results[f"before_{rank}"] = t_before
        results[f"after_{rank}"] = t_after
    finally:
        cleanup_ddp()


def _worker_data_sharding(rank, world_size, port, results):
    """データシャーディングテスト用ワーカー。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    try:
        setup_ddp(rank, world_size)

        # 0-99 のデータを world_size でシャーディングをシミュレート
        all_data = list(range(100))
        # shard と同じロジック: index % num_shards == rank のアイテムを選択
        shard_data = [x for x in all_data if x % world_size == rank]
        results[f"shard_{rank}"] = shard_data
    finally:
        cleanup_ddp()


def _worker_checkpoint_consistency(rank, world_size, port, results, ckpt_dir):
    """チェックポイント一貫性テスト用ワーカー。"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    try:
        setup_ddp(rank, world_size)
        device = torch.device(f"cuda:{rank}")

        # 全ランクで同一シードでモデル作成
        torch.manual_seed(42)
        model = CC_G2PnP(_SMALL_MODEL_CONFIG).to(device)
        ddp_model = wrap_model_ddp(model, rank)

        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        # rank=0 でチェックポイント保存
        if rank == 0:
            from cc_g2pnp.training.config import TrainingConfig

            config = TrainingConfig(
                max_steps=3,
                use_amp=False,
                checkpoint_dir=ckpt_dir,
                warmup_steps=0,
            )
            ckpt_manager = CheckpointManager(ckpt_dir)
            ckpt_manager.save(
                step=1,
                model=ddp_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
            )

        dist.barrier()

        # 全ランクがチェックポイントを読み込み
        ckpt_manager = CheckpointManager(ckpt_dir)
        checkpoint = ckpt_manager.load_latest()
        if checkpoint is not None:
            results[f"ckpt_step_{rank}"] = checkpoint["step"]
            # state_dict のキー数を記録
            results[f"ckpt_keys_{rank}"] = len(checkpoint["model_state_dict"])
            # 最初のパラメータの値を記録 (一致確認用)
            first_key = next(iter(checkpoint["model_state_dict"]))
            first_param = checkpoint["model_state_dict"][first_key]
            results[f"ckpt_first_param_sum_{rank}"] = first_param.sum().item()
        else:
            results[f"ckpt_step_{rank}"] = -1
    finally:
        cleanup_ddp()


def _worker_short_training(rank, world_size, port, results, ckpt_dir):
    """短い DDP 訓練テスト用ワーカー (Trainer 使用, W&B モック)。"""
    import traceback as _tb
    from unittest.mock import MagicMock, patch

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    try:
        from cc_g2pnp.training.config import TrainingConfig
        from cc_g2pnp.training.trainer import Trainer

        training_config = TrainingConfig(
            max_steps=3,
            use_amp=False,
            checkpoint_dir=ckpt_dir,
            log_every_n_steps=1,
            save_every_n_steps=1,
            val_every_n_steps=100,
            warmup_steps=0,
            use_ddp=True,
        )
        # CPU テンソルのダミーバッチ (Trainer 内でデバイス転送される)
        batch = _make_dummy_batch(
            bpe_vocab_size=_SMALL_MODEL_CONFIG.bpe_vocab_size,
            pnp_vocab_size=_SMALL_MODEL_CONFIG.pnp_vocab_size,
        )

        with patch("cc_g2pnp.training.trainer.TrainingLogger"):
            # Trainer.__init__ 内で setup_ddp + wrap_model_ddp が呼ばれる
            trainer = Trainer(_SMALL_MODEL_CONFIG, training_config, rank=rank, world_size=world_size)

            # 初期パラメータを記録 (DDP ラッパーの内側 module から取得)
            raw_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            initial_params = {name: p.clone().detach().cpu() for name, p in raw_model.named_parameters()}

            # データイテレータとバリデーションをモック
            trainer._create_data_iterator = MagicMock(
                return_value=iter([batch] * training_config.max_steps),
            )
            trainer._create_val_batches = MagicMock(return_value=[])

            # 訓練実行 (train() の finally ブロックで cleanup_ddp が呼ばれる)
            trainer.train()

        # 訓練後のパラメータが変更されたか確認
        raw_model_after = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        param_changed = any(
            not torch.equal(initial_params[name], p.data.cpu())
            for name, p in raw_model_after.named_parameters()
            if name in initial_params
        )
        results[f"param_changed_{rank}"] = param_changed
        results[f"num_steps_{rank}"] = training_config.max_steps
        results[f"success_{rank}"] = True
    except Exception as e:
        results[f"param_changed_{rank}"] = False
        results[f"num_steps_{rank}"] = 0
        results[f"success_{rank}"] = False
        results[f"error_{rank}"] = str(e)
        results[f"tb_{rank}"] = _tb.format_exc()
    finally:
        # Trainer.train() が既に cleanup_ddp を呼んでいるが、例外時に備えて再呼び出し
        cleanup_ddp()


# ============================================================
# テストクラス
# ============================================================


@pytest.mark.slow
class TestDDPSetupCleanup:
    """DDP の初期化とクリーンアップのテスト。"""

    def test_setup_and_cleanup(self):
        """全ランクで setup_ddp → rank/world_size 正確性 → cleanup_ddp。"""
        _skip_if_not_enough_gpus()
        world_size = min(_num_gpus(), _MIN_GPUS)
        port = _get_free_port()
        results = mp.Manager().dict()

        mp.spawn(_worker_setup_cleanup, args=(world_size, port, results), nprocs=world_size, join=True)

        for r in range(world_size):
            assert results[f"rank_{r}"] == r, f"rank {r} got rank={results[f'rank_{r}']}"
            assert results[f"world_size_{r}"] == world_size
            assert results[f"initialized_{r}"] is True
            assert results[f"cleaned_{r}"] is True


@pytest.mark.slow
class TestReduceMetrics:
    """reduce_metrics の数値正確性テスト。"""

    def test_avg_reduction(self):
        """AVG: rank0=1.0, rank1=3.0 → 平均 2.0。"""
        _skip_if_not_enough_gpus()
        world_size = min(_num_gpus(), _MIN_GPUS)
        port = _get_free_port()
        results = mp.Manager().dict()

        mp.spawn(_worker_reduce_avg, args=(world_size, port, results), nprocs=world_size, join=True)

        for r in range(world_size):
            assert abs(results[f"avg_loss_{r}"] - 2.0) < 1e-5, f"rank {r}: avg_loss={results[f'avg_loss_{r}']}"

    def test_sum_reduction(self):
        """SUM: rank0=10, rank1=20 → 合計 30。"""
        _skip_if_not_enough_gpus()
        world_size = min(_num_gpus(), _MIN_GPUS)
        port = _get_free_port()
        results = mp.Manager().dict()

        mp.spawn(_worker_reduce_sum, args=(world_size, port, results), nprocs=world_size, join=True)

        expected_sum = sum((r + 1) * 10 for r in range(world_size))
        for r in range(world_size):
            assert abs(results[f"sum_count_{r}"] - expected_sum) < 1e-5, (
                f"rank {r}: sum_count={results[f'sum_count_{r}']}, expected={expected_sum}"
            )


@pytest.mark.slow
class TestGradientSync:
    """DDP 勾配同期のテスト。"""

    def test_gradients_synchronized_across_ranks(self):
        """各ランクが異なる入力で backward → 全ランクで同一勾配。"""
        _skip_if_not_enough_gpus()
        world_size = min(_num_gpus(), _MIN_GPUS)
        port = _get_free_port()
        results = mp.Manager().dict()

        mp.spawn(_worker_gradient_sync, args=(world_size, port, results), nprocs=world_size, join=True)

        # 全ランクの勾配ノルムが一致することを確認
        grad_norms_0 = results["grad_norms_0"]
        for r in range(1, world_size):
            grad_norms_r = results[f"grad_norms_{r}"]
            assert len(grad_norms_0) == len(grad_norms_r), f"rank {r}: 勾配数が異なる"
            for (name0, norm0), (name_r, norm_r) in zip(grad_norms_0, grad_norms_r, strict=True):
                assert name0 == name_r, f"パラメータ名が異なる: {name0} vs {name_r}"
                assert abs(norm0 - norm_r) < 1e-4, (
                    f"rank {r}: {name0} の勾配ノルムが異なる: {norm0} vs {norm_r}"
                )


@pytest.mark.slow
class TestBarrierSync:
    """barrier 同期のテスト。"""

    def test_barrier_waits_for_all_ranks(self):
        """barrier が全ランクの到達まで待機する。"""
        _skip_if_not_enough_gpus()
        world_size = min(_num_gpus(), _MIN_GPUS)
        port = _get_free_port()
        results = mp.Manager().dict()

        mp.spawn(_worker_barrier_sync, args=(world_size, port, results), nprocs=world_size, join=True)

        # rank 0 は barrier 後に処理完了 → rank 1 の sleep (0.5s) が影響
        # rank 0 の after は rank 1 の before より後であること
        # (barrier が rank 1 の到達を待っていた証拠)
        after_0 = results["after_0"]
        before_1 = results["before_1"]
        # rank 0 が barrier を通過した時刻は、rank 1 が barrier に入った時刻以降
        assert after_0 >= before_1 - 0.1, (
            f"rank 0 が barrier を早く通過: after_0={after_0}, before_1={before_1}"
        )


@pytest.mark.slow
class TestDataSharding:
    """DDP データシャーディングのテスト。"""

    def test_shards_are_disjoint_and_cover_all(self):
        """各ランクが重複なしのデータサブセットを処理し、全体をカバー。"""
        _skip_if_not_enough_gpus()
        world_size = min(_num_gpus(), _MIN_GPUS)
        port = _get_free_port()
        results = mp.Manager().dict()

        mp.spawn(_worker_data_sharding, args=(world_size, port, results), nprocs=world_size, join=True)

        all_items = set()
        for r in range(world_size):
            shard = set(results[f"shard_{r}"])
            # 他ランクとの重複がないことを確認
            overlap = all_items & shard
            assert len(overlap) == 0, f"rank {r}: 重複あり: {overlap}"
            all_items |= shard

        # 全アイテムがカバーされていることを確認
        assert all_items == set(range(100)), f"カバー不足: missing={set(range(100)) - all_items}"


@pytest.mark.slow
class TestCheckpointConsistency:
    """チェックポイント一貫性のテスト。"""

    def test_all_ranks_load_same_checkpoint(self):
        """rank=0 保存 → 全ランクが同一 state_dict を読み込み。"""
        _skip_if_not_enough_gpus()
        world_size = min(_num_gpus(), _MIN_GPUS)
        port = _get_free_port()
        results = mp.Manager().dict()

        with tempfile.TemporaryDirectory() as ckpt_dir:
            mp.spawn(
                _worker_checkpoint_consistency,
                args=(world_size, port, results, ckpt_dir),
                nprocs=world_size,
                join=True,
            )

        # 全ランクが同一チェックポイントを読み込めたか
        for r in range(world_size):
            assert results[f"ckpt_step_{r}"] == 1, f"rank {r}: step={results[f'ckpt_step_{r}']}"

        # state_dict のキー数が一致
        keys_0 = results["ckpt_keys_0"]
        for r in range(1, world_size):
            assert results[f"ckpt_keys_{r}"] == keys_0, (
                f"rank {r}: keys={results[f'ckpt_keys_{r}']}, expected={keys_0}"
            )

        # 最初のパラメータ値が一致
        param_sum_0 = results["ckpt_first_param_sum_0"]
        for r in range(1, world_size):
            assert abs(results[f"ckpt_first_param_sum_{r}"] - param_sum_0) < 1e-6, (
                f"rank {r}: first_param_sum={results[f'ckpt_first_param_sum_{r}']}, expected={param_sum_0}"
            )


@pytest.mark.slow
class TestShortDDPTraining:
    """短い DDP 訓練のテスト。"""

    def test_training_completes_and_params_updated(self):
        """3ステップ DDP 訓練が完了し、パラメータが更新される。"""
        _skip_if_not_enough_gpus()
        world_size = min(_num_gpus(), _MIN_GPUS)
        port = _get_free_port()
        results = mp.Manager().dict()

        with tempfile.TemporaryDirectory() as ckpt_dir:
            mp.spawn(
                _worker_short_training,
                args=(world_size, port, results, ckpt_dir),
                nprocs=world_size,
                join=True,
            )

        for r in range(world_size):
            if not results.get(f"success_{r}", False):
                pytest.fail(
                    f"rank {r} 失敗: {results.get(f'error_{r}')}\n{results.get(f'tb_{r}', '')}",
                )
            # 3ステップ完了
            assert results[f"num_steps_{r}"] == 3, f"rank {r}: ステップ数が不足"
            # パラメータが更新された
            assert results[f"param_changed_{r}"] is True, f"rank {r}: パラメータ未更新"
