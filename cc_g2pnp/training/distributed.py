"""Distributed Data Parallel (DDP) utilities for CC-G2PnP training."""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """DDP を初期化する。

    Args:
        rank: 現在のプロセスのランク
        world_size: 総プロセス数
        backend: 通信バックエンド ("nccl" or "gloo")
    """
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    """DDP をクリーンアップする。"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """メインプロセス (rank 0) かどうかを返す。DDP未初期化の場合はTrue。"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """現在のランクを返す。DDP未初期化の場合は0。"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """ワールドサイズを返す。DDP未初期化の場合は1。"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_metrics(
    metrics: dict[str, float], device: torch.device
) -> dict[str, float]:
    """全プロセスからメトリクスを平均化する。

    DDP未初期化の場合はそのまま返す。

    Args:
        metrics: {"loss": 0.5, ...}
        device: テンソルを作成するデバイス

    Returns:
        平均化されたメトリクス辞書
    """
    if not dist.is_initialized():
        return metrics

    result = {}
    for key, value in metrics.items():
        tensor = torch.tensor(value, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        result[key] = tensor.item()
    return result


def wrap_model_ddp(
    model: torch.nn.Module,
    device_id: int,
    find_unused_parameters: bool = True,
) -> DistributedDataParallel:
    """モデルを DDP でラップする。

    Note:
        find_unused_parameters=True は Self-conditioned CTC で必要。
        ESPnet Issue #4031 参照。

    Args:
        model: ラップ対象モデル
        device_id: CUDA デバイスID
        find_unused_parameters: 未使用パラメータを探すかどうか

    Returns:
        DDP ラップされたモデル
    """
    return DistributedDataParallel(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters,
    )
