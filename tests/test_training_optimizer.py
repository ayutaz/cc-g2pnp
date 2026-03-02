"""Tests for cc_g2pnp.training.optimizer (build_optimizer / build_scheduler)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, SequentialLR

from cc_g2pnp.training.optimizer import build_optimizer, build_scheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.98),
    warmup_steps: int = 10_000,
    scheduler_gamma: float = 0.999_998,
    scheduler_type: str = "exponential",
    final_learning_rate: float = 1e-5,
    effective_steps: int = 1_200_000,
) -> SimpleNamespace:
    """TrainingConfig のダミーを生成する。"""
    return SimpleNamespace(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        warmup_steps=warmup_steps,
        scheduler_gamma=scheduler_gamma,
        scheduler_type=scheduler_type,
        final_learning_rate=final_learning_rate,
        effective_steps=effective_steps,
    )


def _make_model() -> nn.Module:
    """LayerNorm + bias を含む小さなモデルを返す。"""
    model = nn.Sequential(
        nn.Linear(8, 16),       # weight + bias
        nn.LayerNorm(16),       # LayerNorm.weight + LayerNorm.bias
        nn.Linear(16, 4),      # weight + bias
    )
    return model


# ===========================================================================
# build_optimizer tests
# ===========================================================================

class TestBuildOptimizer:
    """build_optimizer のテスト。"""

    def test_returns_adamw(self) -> None:
        config = _make_config()
        model = _make_model()
        optimizer = build_optimizer(model, config)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_two_param_groups(self) -> None:
        config = _make_config()
        model = _make_model()
        optimizer = build_optimizer(model, config)
        assert len(optimizer.param_groups) == 2

    def test_decay_group_weight_decay(self) -> None:
        wd = 0.05
        config = _make_config(weight_decay=wd)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        decay_group = optimizer.param_groups[0]
        assert decay_group["weight_decay"] == wd

    def test_no_decay_group_weight_decay_zero(self) -> None:
        config = _make_config(weight_decay=0.01)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        no_decay_group = optimizer.param_groups[1]
        assert no_decay_group["weight_decay"] == 0.0

    def test_learning_rate(self) -> None:
        lr = 3e-4
        config = _make_config(learning_rate=lr)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        for pg in optimizer.param_groups:
            assert pg["lr"] == lr

    def test_betas(self) -> None:
        betas = (0.9, 0.999)
        config = _make_config(betas=betas)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        for pg in optimizer.param_groups:
            assert tuple(pg["betas"]) == betas

    def test_no_decay_contains_bias_and_layernorm(self) -> None:
        """no_decay グループに bias と LayerNorm パラメータが含まれること。"""
        config = _make_config()
        model = _make_model()
        optimizer = build_optimizer(model, config)

        no_decay_params = set(id(p) for p in optimizer.param_groups[1]["params"])

        # model[0] = Linear(8,16): bias should be no_decay
        assert id(model[0].bias) in no_decay_params
        # model[1] = LayerNorm(16): weight and bias should be no_decay
        assert id(model[1].weight) in no_decay_params
        assert id(model[1].bias) in no_decay_params
        # model[2] = Linear(16,4): bias should be no_decay
        assert id(model[2].bias) in no_decay_params

    def test_decay_contains_linear_weights(self) -> None:
        """decay グループに Linear の weight が含まれること。"""
        config = _make_config()
        model = _make_model()
        optimizer = build_optimizer(model, config)

        decay_params = set(id(p) for p in optimizer.param_groups[0]["params"])

        assert id(model[0].weight) in decay_params
        assert id(model[2].weight) in decay_params

    def test_all_trainable_params_covered(self) -> None:
        """全 trainable パラメータがいずれかのグループに含まれること。"""
        config = _make_config()
        model = _make_model()
        optimizer = build_optimizer(model, config)

        optimizer_param_ids = set()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                optimizer_param_ids.add(id(p))

        for p in model.parameters():
            if p.requires_grad:
                assert id(p) in optimizer_param_ids

    def test_frozen_params_excluded(self) -> None:
        """requires_grad=False のパラメータは optimizer に含まれないこと。"""
        config = _make_config()
        model = _make_model()
        # Freeze first Linear
        for p in model[0].parameters():
            p.requires_grad = False

        optimizer = build_optimizer(model, config)
        optimizer_param_ids = set()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                optimizer_param_ids.add(id(p))

        assert id(model[0].weight) not in optimizer_param_ids
        assert id(model[0].bias) not in optimizer_param_ids


# ===========================================================================
# build_scheduler tests
# ===========================================================================

class TestBuildScheduler:
    """build_scheduler のテスト。"""

    def test_warmup_returns_sequential(self) -> None:
        config = _make_config(warmup_steps=100)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        assert isinstance(scheduler, SequentialLR)

    def test_no_warmup_returns_exponential(self) -> None:
        config = _make_config(warmup_steps=0)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        assert isinstance(scheduler, ExponentialLR)

    def test_warmup_lr_increases(self) -> None:
        """warmup 期間中に lr が増加すること。"""
        config = _make_config(learning_rate=1e-4, warmup_steps=100, scheduler_gamma=1.0)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        lrs = []
        for _ in range(50):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # lr は単調増加
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], f"lr decreased at step {i}: {lrs[i-1]} -> {lrs[i]}"

        # 最初と最後で明らかに増加
        assert lrs[-1] > lrs[0] * 10

    def test_post_warmup_lr_decreases(self) -> None:
        """warmup 終了後に lr が減少すること。"""
        gamma = 0.99
        config = _make_config(learning_rate=1e-4, warmup_steps=10, scheduler_gamma=gamma)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        # warmup を通過
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        # warmup 直後の lr を記録
        lr_after_warmup = optimizer.param_groups[0]["lr"]

        # さらに数ステップ進めて減少を確認
        for _ in range(10):
            optimizer.step()
            scheduler.step()

        lr_later = optimizer.param_groups[0]["lr"]
        assert lr_later < lr_after_warmup

    def test_exponential_gamma_applied(self) -> None:
        """ExponentialLR の gamma が正しく適用されること。"""
        gamma = 0.99
        lr = 1e-3
        config = _make_config(learning_rate=lr, warmup_steps=0, scheduler_gamma=gamma)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = build_scheduler(optimizer, config)

        optimizer.step()
        scheduler.step()
        expected_lr = lr * gamma
        actual_lr = optimizer.param_groups[0]["lr"]
        assert math.isclose(actual_lr, expected_lr, rel_tol=1e-6)

    def test_start_factor_clamp(self) -> None:
        """learning_rate が非常に大きい場合に start_factor が 1e-10 にクランプされること。"""
        # lr=100 -> 1e-8/100 = 1e-10, ちょうどクランプ境界
        config = _make_config(learning_rate=100.0, warmup_steps=10, scheduler_gamma=1.0)
        model = nn.Linear(4, 2)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        # scheduler が正常に構築されればOK
        assert isinstance(scheduler, SequentialLR)

        # 初期 lr は start_factor * learning_rate
        initial_lr = optimizer.param_groups[0]["lr"]
        assert initial_lr == pytest.approx(1e-10 * 100.0, rel=1e-6)

    def test_e2e_simple_model(self) -> None:
        """簡単なモデルでの E2E テスト。"""
        config = _make_config(
            learning_rate=1e-3,
            warmup_steps=5,
            scheduler_gamma=0.999,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        model = nn.Linear(4, 2)
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        x = torch.randn(2, 4)
        target = torch.randn(2, 2)

        for _step in range(20):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x), target)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 学習が進み lr が初期値より小さいこと
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < config.learning_rate


# ===========================================================================
# Cosine scheduler tests
# ===========================================================================

class TestCosineScheduler:
    """CosineAnnealingLR スケジューラのテスト。"""

    def test_cosine_returns_sequential_with_warmup(self) -> None:
        """warmup_steps > 0 の場合 SequentialLR を返すこと。"""
        config = _make_config(scheduler_type="cosine", warmup_steps=100, effective_steps=1000)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        assert isinstance(scheduler, SequentialLR)

    def test_cosine_returns_cosine_without_warmup(self) -> None:
        """warmup_steps == 0 の場合 CosineAnnealingLR を返すこと。"""
        config = _make_config(scheduler_type="cosine", warmup_steps=0, effective_steps=1000)
        model = _make_model()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        assert isinstance(scheduler, CosineAnnealingLR)

    def test_cosine_lr_decreases_after_warmup(self) -> None:
        """warmup 後に cosine スケジューラで lr が減少すること。"""
        config = _make_config(
            scheduler_type="cosine",
            learning_rate=1e-3,
            warmup_steps=10,
            effective_steps=1000,
            final_learning_rate=1e-5,
        )
        model = _make_model()
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)

        # warmup 期間: lr が増加する
        warmup_lrs = []
        for _ in range(10):
            warmup_lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        for i in range(1, len(warmup_lrs)):
            assert warmup_lrs[i] >= warmup_lrs[i - 1]

        # warmup 直後の lr を記録
        lr_after_warmup = optimizer.param_groups[0]["lr"]

        # cosine decay: lr が減少する
        for _ in range(50):
            optimizer.step()
            scheduler.step()

        lr_later = optimizer.param_groups[0]["lr"]
        assert lr_later < lr_after_warmup

    def test_cosine_lr_reaches_eta_min(self) -> None:
        """cosine スケジューラで lr が final_learning_rate に到達すること。"""
        final_lr = 1e-5
        effective = 200
        warmup = 0
        config = _make_config(
            scheduler_type="cosine",
            learning_rate=1e-3,
            warmup_steps=warmup,
            effective_steps=effective,
            final_learning_rate=final_lr,
        )
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scheduler = build_scheduler(optimizer, config)

        # T_max ステップ進める
        for _ in range(effective):
            optimizer.step()
            scheduler.step()

        actual_lr = optimizer.param_groups[0]["lr"]
        assert math.isclose(actual_lr, final_lr, rel_tol=1e-6)

    def test_exponential_still_works(self) -> None:
        """scheduler_type="exponential" で既存動作が維持されること。"""
        gamma = 0.99
        lr = 1e-3
        config = _make_config(
            scheduler_type="exponential",
            learning_rate=lr,
            warmup_steps=0,
            scheduler_gamma=gamma,
        )
        model = nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = build_scheduler(optimizer, config)
        assert isinstance(scheduler, ExponentialLR)

        optimizer.step()
        scheduler.step()
        expected_lr = lr * gamma
        actual_lr = optimizer.param_groups[0]["lr"]
        assert math.isclose(actual_lr, expected_lr, rel_tol=1e-6)
