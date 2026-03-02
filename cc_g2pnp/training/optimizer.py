"""Optimizer and learning-rate scheduler factories for CC-G2PnP training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, LinearLR, SequentialLR

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import LRScheduler

    from cc_g2pnp.training.config import TrainingConfig


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """AdamW optimizer を構築する。

    LayerNorm の weight/bias および全 bias パラメータには weight_decay を適用しない。

    Args:
        model: 学習対象モデル。
        config: TrainingConfig (learning_rate, weight_decay, betas を使用)。

    Returns:
        AdamW optimizer (decay / no_decay の2パラメータグループ)。
    """
    # LayerNorm パラメータの id を収集
    layernorm_param_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            for param in module.parameters():
                layernorm_param_ids.add(id(param))

    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or name == "bias" or id(param) in layernorm_param_ids:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=config.betas,
        fused=torch.cuda.is_available(),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
) -> LRScheduler:
    """学習率スケジューラを構築する。

    scheduler_type == "cosine" の場合:
        warmup_steps > 0 なら LinearLR (warmup) + CosineAnnealingLR を SequentialLR で結合。
        warmup_steps == 0 なら CosineAnnealingLR のみを返す。
    scheduler_type == "exponential" の場合:
        warmup_steps > 0 なら LinearLR (warmup) + ExponentialLR を SequentialLR で結合。
        warmup_steps == 0 なら ExponentialLR のみを返す。

    Args:
        optimizer: 対象 optimizer。
        config: TrainingConfig (learning_rate, warmup_steps, scheduler_gamma,
                scheduler_type, final_learning_rate, effective_steps を使用)。

    Returns:
        学習率スケジューラ。
    """
    if config.scheduler_type == "cosine":
        decay_steps = config.effective_steps - config.warmup_steps
        cosine_lr = CosineAnnealingLR(
            optimizer,
            T_max=decay_steps,
            eta_min=config.final_learning_rate,
        )
        if config.warmup_steps == 0:
            return cosine_lr
        start_factor = max(1e-8 / config.learning_rate, 1e-10)
        warmup_lr = LinearLR(
            optimizer,
            start_factor=start_factor,
            total_iters=config.warmup_steps,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup_lr, cosine_lr],
            milestones=[config.warmup_steps],
        )

    # exponential (default)
    exponential_lr = ExponentialLR(optimizer, gamma=config.scheduler_gamma)

    if config.warmup_steps == 0:
        return exponential_lr

    start_factor = max(1e-8 / config.learning_rate, 1e-10)
    warmup_lr = LinearLR(
        optimizer,
        start_factor=start_factor,
        total_iters=config.warmup_steps,
    )

    return SequentialLR(
        optimizer,
        schedulers=[warmup_lr, exponential_lr],
        milestones=[config.warmup_steps],
    )
