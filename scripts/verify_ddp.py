#!/usr/bin/env python3
"""DDP 動作検証スクリプト。torchrun で実行。

使用例:
    uv run torchrun --nproc_per_node=4 scripts/verify_ddp.py
"""

from __future__ import annotations

import os
import tempfile
import time

import torch
import torch.distributed as dist

from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.training.checkpoint import CheckpointManager
from cc_g2pnp.training.config import TrainingConfig
from cc_g2pnp.training.distributed import cleanup_ddp, reduce_metrics, setup_ddp, wrap_model_ddp

# 小モデル設定
_SMALL_CONFIG = CC_G2PnPConfig(
    d_model=64,
    num_heads=2,
    d_ff=128,
    num_layers=2,
    intermediate_ctc_layers=(1,),
)


def _log(rank: int, msg: str) -> None:
    """ランク付きログ出力。"""
    print(f"[rank {rank}] {msg}", flush=True)


def _log_main(rank: int, msg: str) -> None:
    """ランク0のみログ出力。"""
    if rank == 0:
        print(f"  {msg}", flush=True)


def _make_dummy_batch(
    device: torch.device,
    bpe_vocab_size: int = 65_000,
    pnp_vocab_size: int = 140,
    batch_size: int = 2,
    seq_len: int = 10,
    label_len: int = 40,
) -> dict[str, torch.Tensor]:
    """テスト用ダミーバッチ。"""
    return {
        "input_ids": torch.randint(0, bpe_vocab_size, (batch_size, seq_len), device=device),
        "attention_mask": torch.ones(batch_size, seq_len, device=device),
        "labels": torch.randint(1, pnp_vocab_size, (batch_size, label_len), device=device),
        "input_lengths": torch.tensor([seq_len] * batch_size, device=device),
        "label_lengths": torch.tensor([label_len] * batch_size, device=device),
    }


# ============================================================
# 検証関数群
# ============================================================


def verify_init(rank: int, world_size: int) -> bool:
    """検証1: DDP 初期化確認。"""
    try:
        assert dist.is_initialized(), "DDP が初期化されていない"
        assert dist.get_rank() == rank, f"rank が一致しない: expected={rank}, got={dist.get_rank()}"
        assert dist.get_world_size() == world_size, (
            f"world_size が一致しない: expected={world_size}, got={dist.get_world_size()}"
        )
        _log(rank, f"初期化成功: rank={rank}, world_size={world_size}")
        return True
    except Exception as e:
        _log(rank, f"初期化失敗: {e}")
        return False


def verify_reduce_metrics(rank: int, device: torch.device) -> bool:
    """検証2: reduce_metrics の AVG/SUM 検証。"""
    try:
        world_size = dist.get_world_size()

        # AVG テスト: 各ランクの loss = rank + 1.0
        avg_metrics = {"loss": float(rank + 1.0)}
        reduced_avg = reduce_metrics(avg_metrics, device)
        expected_avg = sum(r + 1.0 for r in range(world_size)) / world_size
        assert abs(reduced_avg["loss"] - expected_avg) < 1e-5, (
            f"AVG 不一致: got={reduced_avg['loss']}, expected={expected_avg}"
        )

        # SUM テスト: 各ランクの count = (rank + 1) * 10
        sum_metrics = {"count": float((rank + 1) * 10)}
        reduced_sum = reduce_metrics(sum_metrics, device, sum_keys=frozenset({"count"}))
        expected_sum = sum((r + 1) * 10 for r in range(world_size))
        assert abs(reduced_sum["count"] - expected_sum) < 1e-5, (
            f"SUM 不一致: got={reduced_sum['count']}, expected={expected_sum}"
        )

        # 混合テスト: loss (AVG) + count (SUM)
        mixed = {"loss": float(rank + 1.0), "count": float((rank + 1) * 10)}
        reduced_mixed = reduce_metrics(mixed, device, sum_keys=frozenset({"count"}))
        assert abs(reduced_mixed["loss"] - expected_avg) < 1e-5
        assert abs(reduced_mixed["count"] - expected_sum) < 1e-5

        _log_main(rank, f"reduce_metrics 検証成功: AVG={reduced_avg['loss']:.4f}, SUM={reduced_sum['count']:.1f}")
        return True
    except Exception as e:
        _log(rank, f"reduce_metrics 検証失敗: {e}")
        return False


def verify_short_training(rank: int, world_size: int, device: torch.device) -> bool:
    """検証3: 小モデル 3ステップ訓練。"""
    try:
        # モデル作成 + DDP ラップ
        torch.manual_seed(42)
        model = CC_G2PnP(_SMALL_CONFIG).to(device)
        ddp_model = wrap_model_ddp(model, rank)
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-3)

        num_steps = 3
        losses = []

        for step in range(num_steps):
            torch.manual_seed(step * 100 + rank)
            batch = _make_dummy_batch(device)
            ddp_model.train()

            result = ddp_model(
                input_ids=batch["input_ids"],
                input_lengths=batch["input_lengths"],
                targets=batch["labels"],
                target_lengths=batch["label_lengths"],
            )
            loss = result["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        for i, loss_val in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss_val)), f"step {i}: loss={loss_val} is not finite"

        _log_main(rank, f"3ステップ訓練成功: losses={[f'{v:.4f}' for v in losses]}")

        # DDP モデルをクリーンアップ
        del ddp_model, optimizer
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        _log(rank, f"訓練検証失敗: {e}")
        return False


def verify_checkpoint(rank: int, world_size: int, device: torch.device, ckpt_dir: str) -> bool:
    """検証4: チェックポイント保存/読み込み。"""
    try:
        torch.manual_seed(42)
        model = CC_G2PnP(_SMALL_CONFIG).to(device)
        ddp_model = wrap_model_ddp(model, rank)
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        config = TrainingConfig(
            max_steps=3,
            use_amp=False,
            checkpoint_dir=ckpt_dir,
            warmup_steps=0,
        )

        # rank=0 のみ保存
        if rank == 0:
            ckpt_manager = CheckpointManager(ckpt_dir)
            path = ckpt_manager.save(
                step=100,
                model=ddp_model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
            )
            _log_main(rank, f"チェックポイント保存: {path}")

        dist.barrier()

        # 全ランクが読み込み
        ckpt_manager = CheckpointManager(ckpt_dir)
        checkpoint = ckpt_manager.load_latest()
        assert checkpoint is not None, "チェックポイントが読み込めない"
        assert checkpoint["step"] == 100, f"step 不一致: {checkpoint['step']}"

        # state_dict の整合性確認
        saved_keys = set(checkpoint["model_state_dict"].keys())
        model_keys = set(model.state_dict().keys())
        assert saved_keys == model_keys, f"state_dict キー不一致: diff={saved_keys ^ model_keys}"

        _log_main(rank, f"チェックポイント読み込み成功: step={checkpoint['step']}, keys={len(saved_keys)}")

        del ddp_model, optimizer, scheduler
        torch.cuda.empty_cache()
        return True
    except Exception as e:
        _log(rank, f"チェックポイント検証失敗: {e}")
        return False


def verify_barrier(rank: int) -> bool:
    """検証5: barrier 同期確認。"""
    try:
        # rank が大きいほど長く sleep → barrier が全ランクを待つことを確認
        sleep_time = rank * 0.2
        time.sleep(sleep_time)

        t_before = time.monotonic()
        dist.barrier()
        t_after = time.monotonic()

        _log(rank, f"barrier 通過: sleep={sleep_time:.1f}s, wait={t_after - t_before:.3f}s")

        # 2回目の barrier: rank 0 が遅延
        if rank == 0:
            time.sleep(0.5)
        dist.barrier()

        _log_main(rank, "barrier 同期確認成功")
        return True
    except Exception as e:
        _log(rank, f"barrier 検証失敗: {e}")
        return False


def report_memory(rank: int, device: torch.device) -> bool:
    """検証6: GPU メモリ使用量レポート。"""
    try:
        allocated = torch.cuda.memory_allocated(device) / (1024**2)
        max_allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        reserved = torch.cuda.memory_reserved(device) / (1024**2)

        _log(rank, f"メモリ: allocated={allocated:.1f}MB, max={max_allocated:.1f}MB, reserved={reserved:.1f}MB")
        return True
    except Exception as e:
        _log(rank, f"メモリレポート失敗: {e}")
        return False


def print_summary(results: dict[str, bool], rank: int) -> None:
    """検証結果のサマリーを表示。"""
    if rank != 0:
        return

    print("\n" + "=" * 60)
    print("DDP 検証サマリー")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        icon = "+" if passed else "!"
        print(f"  [{icon}] {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("結果: 全検証項目 PASS")
    else:
        print("結果: 一部検証項目 FAIL")
    print("=" * 60)


def main() -> None:
    """メインエントリポイント。torchrun から呼ばれる。"""
    # torchrun が設定する環境変数
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if rank == 0:
        print(f"\nDDP 検証開始: world_size={world_size}, GPUs={torch.cuda.device_count()}")
        print("=" * 60)

    # DDP 初期化
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    results: dict[str, bool] = {}
    ckpt_dir = None

    try:
        # チェックポイント用一時ディレクトリ (rank 0 で作成、broadcast で共有)
        if rank == 0:
            ckpt_dir = tempfile.mkdtemp(prefix="ddp_verify_")
        # ディレクトリパスを rank 0 から全ランクに broadcast
        dir_list = [ckpt_dir] if rank == 0 else [None]
        dist.broadcast_object_list(dir_list, src=0)
        ckpt_dir = dir_list[0]

        # 検証1: DDP 初期化
        if rank == 0:
            print("\n[検証1] DDP 初期化")
        results["1. DDP 初期化"] = verify_init(rank, world_size)
        dist.barrier()

        # 検証2: reduce_metrics
        if rank == 0:
            print("\n[検証2] reduce_metrics")
        results["2. reduce_metrics AVG/SUM"] = verify_reduce_metrics(rank, device)
        dist.barrier()

        # 検証3: 小モデル訓練
        if rank == 0:
            print("\n[検証3] 小モデル 3ステップ訓練")
        results["3. 小モデル 3ステップ訓練"] = verify_short_training(rank, world_size, device)
        dist.barrier()

        # 検証4: チェックポイント
        if rank == 0:
            print("\n[検証4] チェックポイント保存/読み込み")
        results["4. チェックポイント保存/読み込み"] = verify_checkpoint(rank, world_size, device, ckpt_dir)
        dist.barrier()

        # 検証5: barrier 同期
        if rank == 0:
            print("\n[検証5] barrier 同期")
        results["5. barrier 同期"] = verify_barrier(rank)
        dist.barrier()

        # 検証6: メモリレポート
        if rank == 0:
            print("\n[検証6] GPU メモリレポート")
        results["6. GPU メモリレポート"] = report_memory(rank, device)
        dist.barrier()

    finally:
        # サマリー表示
        print_summary(results, rank)

        # 一時ディレクトリ削除
        if rank == 0 and ckpt_dir is not None:
            import shutil

            shutil.rmtree(ckpt_dir, ignore_errors=True)

        cleanup_ddp()


if __name__ == "__main__":
    main()
