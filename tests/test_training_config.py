"""Tests for TrainingConfig."""

from __future__ import annotations

import pytest

from cc_g2pnp.training.config import TrainingConfig


class TestTrainingConfigDefaults:
    """Test default values."""

    def test_default_values(self) -> None:
        cfg = TrainingConfig()
        assert cfg.learning_rate == 1e-4
        assert cfg.final_learning_rate == 1e-5
        assert cfg.weight_decay == 0.01
        assert cfg.betas == (0.9, 0.98)
        assert cfg.max_grad_norm == 1.0
        assert cfg.total_steps == 1_200_000
        assert cfg.warmup_steps == 10_000
        assert cfg.max_tokens_per_batch == 8192
        assert cfg.dataset_subset == "all"
        assert cfg.checkpoint_dir == "checkpoints"
        assert cfg.save_every_n_steps == 10_000
        assert cfg.keep_last_n == 5
        assert cfg.log_every_n_steps == 100
        assert cfg.project_name == "cc-g2pnp"
        assert cfg.run_name is None
        assert cfg.use_amp is True
        assert cfg.amp_dtype == "bfloat16"
        assert cfg.use_ddp is False
        assert cfg.val_every_n_steps == 5_000
        assert cfg.seed == 42
        assert cfg.max_steps is None

    def test_betas_is_tuple(self) -> None:
        cfg = TrainingConfig()
        assert isinstance(cfg.betas, tuple)
        assert len(cfg.betas) == 2


class TestTrainingConfigCustom:
    """Test creation with custom values."""

    def test_custom_values(self) -> None:
        cfg = TrainingConfig(
            learning_rate=5e-4,
            final_learning_rate=1e-6,
            weight_decay=0.05,
            betas=(0.95, 0.999),
            max_grad_norm=5.0,
            total_steps=500_000,
            warmup_steps=5_000,
            max_tokens_per_batch=4096,
            dataset_subset="small",
            checkpoint_dir="/tmp/ckpt",
            save_every_n_steps=5_000,
            keep_last_n=3,
            log_every_n_steps=50,
            project_name="test-project",
            run_name="run-001",
            use_amp=False,
            amp_dtype="float16",
            use_ddp=True,
            val_every_n_steps=1_000,
            seed=123,
            max_steps=10_000,
        )
        assert cfg.learning_rate == 5e-4
        assert cfg.final_learning_rate == 1e-6
        assert cfg.weight_decay == 0.05
        assert cfg.betas == (0.95, 0.999)
        assert cfg.max_grad_norm == 5.0
        assert cfg.total_steps == 500_000
        assert cfg.warmup_steps == 5_000
        assert cfg.max_tokens_per_batch == 4096
        assert cfg.dataset_subset == "small"
        assert cfg.checkpoint_dir == "/tmp/ckpt"
        assert cfg.save_every_n_steps == 5_000
        assert cfg.keep_last_n == 3
        assert cfg.log_every_n_steps == 50
        assert cfg.project_name == "test-project"
        assert cfg.run_name == "run-001"
        assert cfg.use_amp is False
        assert cfg.amp_dtype == "float16"
        assert cfg.use_ddp is True
        assert cfg.val_every_n_steps == 1_000
        assert cfg.seed == 123
        assert cfg.max_steps == 10_000


class TestTrainingConfigValidation:
    """Test __post_init__ validation."""

    def test_learning_rate_zero(self) -> None:
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=0.0)

    def test_learning_rate_negative(self) -> None:
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainingConfig(learning_rate=-1e-4)

    def test_final_learning_rate_zero(self) -> None:
        with pytest.raises(ValueError, match="final_learning_rate must be positive"):
            TrainingConfig(final_learning_rate=0.0)

    def test_final_learning_rate_negative(self) -> None:
        with pytest.raises(ValueError, match="final_learning_rate must be positive"):
            TrainingConfig(final_learning_rate=-1e-5)

    def test_final_learning_rate_exceeds_learning_rate(self) -> None:
        with pytest.raises(ValueError, match=r"final_learning_rate.*<= learning_rate"):
            TrainingConfig(learning_rate=1e-5, final_learning_rate=1e-4)

    def test_final_learning_rate_equals_learning_rate(self) -> None:
        # Equal is valid
        cfg = TrainingConfig(learning_rate=1e-4, final_learning_rate=1e-4)
        assert cfg.final_learning_rate == cfg.learning_rate

    def test_total_steps_zero(self) -> None:
        with pytest.raises(ValueError, match="total_steps must be positive"):
            TrainingConfig(total_steps=0)

    def test_total_steps_negative(self) -> None:
        with pytest.raises(ValueError, match="total_steps must be positive"):
            TrainingConfig(total_steps=-1)

    def test_warmup_steps_negative(self) -> None:
        with pytest.raises(ValueError, match="warmup_steps must be >= 0"):
            TrainingConfig(warmup_steps=-1)

    def test_warmup_steps_equals_total_steps(self) -> None:
        with pytest.raises(ValueError, match=r"warmup_steps.*< total_steps"):
            TrainingConfig(total_steps=100, warmup_steps=100)

    def test_warmup_steps_exceeds_total_steps(self) -> None:
        with pytest.raises(ValueError, match=r"warmup_steps.*< total_steps"):
            TrainingConfig(total_steps=100, warmup_steps=200)

    def test_warmup_steps_zero_is_valid(self) -> None:
        cfg = TrainingConfig(warmup_steps=0)
        assert cfg.warmup_steps == 0

    def test_max_grad_norm_zero(self) -> None:
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            TrainingConfig(max_grad_norm=0.0)

    def test_max_grad_norm_negative(self) -> None:
        with pytest.raises(ValueError, match="max_grad_norm must be positive"):
            TrainingConfig(max_grad_norm=-1.0)

    def test_amp_dtype_invalid(self) -> None:
        with pytest.raises(ValueError, match="amp_dtype must be"):
            TrainingConfig(amp_dtype="float32")

    def test_amp_dtype_float16_valid(self) -> None:
        cfg = TrainingConfig(amp_dtype="float16")
        assert cfg.amp_dtype == "float16"

    def test_amp_dtype_bfloat16_valid(self) -> None:
        cfg = TrainingConfig(amp_dtype="bfloat16")
        assert cfg.amp_dtype == "bfloat16"

    def test_max_tokens_per_batch_zero(self) -> None:
        with pytest.raises(ValueError, match="max_tokens_per_batch must be positive"):
            TrainingConfig(max_tokens_per_batch=0)

    def test_max_tokens_per_batch_negative(self) -> None:
        with pytest.raises(ValueError, match="max_tokens_per_batch must be positive"):
            TrainingConfig(max_tokens_per_batch=-1)

    def test_save_every_n_steps_zero(self) -> None:
        with pytest.raises(ValueError, match="save_every_n_steps must be positive"):
            TrainingConfig(save_every_n_steps=0)

    def test_val_every_n_steps_zero(self) -> None:
        with pytest.raises(ValueError, match="val_every_n_steps must be positive"):
            TrainingConfig(val_every_n_steps=0)

    def test_log_every_n_steps_zero(self) -> None:
        with pytest.raises(ValueError, match="log_every_n_steps must be positive"):
            TrainingConfig(log_every_n_steps=0)

    def test_keep_last_n_zero(self) -> None:
        with pytest.raises(ValueError, match="keep_last_n must be >= 1"):
            TrainingConfig(keep_last_n=0)

    def test_keep_last_n_negative(self) -> None:
        with pytest.raises(ValueError, match="keep_last_n must be >= 1"):
            TrainingConfig(keep_last_n=-1)

    def test_max_steps_zero(self) -> None:
        with pytest.raises(ValueError, match="max_steps must be positive or None"):
            TrainingConfig(max_steps=0)

    def test_max_steps_negative(self) -> None:
        with pytest.raises(ValueError, match="max_steps must be positive or None"):
            TrainingConfig(max_steps=-1)

    def test_max_steps_none_is_valid(self) -> None:
        cfg = TrainingConfig(max_steps=None)
        assert cfg.max_steps is None

    def test_max_steps_positive_is_valid(self) -> None:
        cfg = TrainingConfig(max_steps=100, warmup_steps=10)
        assert cfg.max_steps == 100

    def test_max_steps_equals_warmup_steps(self) -> None:
        with pytest.raises(ValueError, match=r"max_steps.*> warmup_steps"):
            TrainingConfig(max_steps=100, warmup_steps=100)

    def test_max_steps_less_than_warmup_steps(self) -> None:
        with pytest.raises(ValueError, match=r"max_steps.*> warmup_steps"):
            TrainingConfig(max_steps=50, warmup_steps=10_000)

    def test_max_steps_greater_than_warmup_valid(self) -> None:
        cfg = TrainingConfig(max_steps=200, warmup_steps=100)
        assert cfg.max_steps == 200

    def test_weight_decay_negative(self) -> None:
        with pytest.raises(ValueError, match="weight_decay must be >= 0"):
            TrainingConfig(weight_decay=-0.5)

    def test_weight_decay_zero_is_valid(self) -> None:
        cfg = TrainingConfig(weight_decay=0.0)
        assert cfg.weight_decay == 0.0

    def test_betas_wrong_type(self) -> None:
        with pytest.raises(ValueError, match="betas must be a tuple"):
            TrainingConfig(betas=[0.9, 0.98])  # type: ignore[arg-type]

    def test_betas_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="betas must be a tuple"):
            TrainingConfig(betas=(0.9,))  # type: ignore[arg-type]

    def test_betas_value_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="betas values must be in"):
            TrainingConfig(betas=(1.5, 0.98))

    def test_betas_negative_value(self) -> None:
        with pytest.raises(ValueError, match="betas values must be in"):
            TrainingConfig(betas=(0.9, -0.3))

    def test_betas_one_not_allowed(self) -> None:
        with pytest.raises(ValueError, match="betas values must be in"):
            TrainingConfig(betas=(0.9, 1.0))

    def test_betas_valid_custom(self) -> None:
        cfg = TrainingConfig(betas=(0.95, 0.999))
        assert cfg.betas == (0.95, 0.999)

    def test_seed_negative(self) -> None:
        with pytest.raises(ValueError, match="seed must be >= 0"):
            TrainingConfig(seed=-1)

    def test_seed_zero_is_valid(self) -> None:
        cfg = TrainingConfig(seed=0)
        assert cfg.seed == 0


class TestTrainingConfigProperties:
    """Test computed properties."""

    def test_effective_steps_without_max_steps(self) -> None:
        cfg = TrainingConfig(total_steps=1_200_000, max_steps=None)
        assert cfg.effective_steps == 1_200_000

    def test_effective_steps_with_max_steps(self) -> None:
        cfg = TrainingConfig(total_steps=1_200_000, max_steps=50_000, warmup_steps=1_000)
        assert cfg.effective_steps == 50_000

    def test_scheduler_gamma_default(self) -> None:
        cfg = TrainingConfig()
        # gamma = (1e-5 / 1e-4) ** (1 / (1_200_000 - 10_000))
        # gamma = 0.1 ** (1 / 1_190_000)
        expected = (1e-5 / 1e-4) ** (1.0 / (1_200_000 - 10_000))
        assert cfg.scheduler_gamma == pytest.approx(expected, rel=1e-10)

    def test_scheduler_gamma_custom(self) -> None:
        cfg = TrainingConfig(
            learning_rate=1e-3,
            final_learning_rate=1e-5,
            total_steps=100_000,
            warmup_steps=1_000,
        )
        expected = (1e-5 / 1e-3) ** (1.0 / (100_000 - 1_000))
        assert cfg.scheduler_gamma == pytest.approx(expected, rel=1e-10)

    def test_scheduler_gamma_with_max_steps(self) -> None:
        cfg = TrainingConfig(
            learning_rate=1e-4,
            final_learning_rate=1e-5,
            total_steps=1_200_000,
            warmup_steps=1_000,
            max_steps=10_000,
        )
        # Uses effective_steps = max_steps = 10_000
        expected = (1e-5 / 1e-4) ** (1.0 / (10_000 - 1_000))
        assert cfg.scheduler_gamma == pytest.approx(expected, rel=1e-10)

    def test_scheduler_gamma_equal_lr(self) -> None:
        cfg = TrainingConfig(learning_rate=1e-4, final_learning_rate=1e-4)
        # gamma = 1.0 (no decay)
        assert cfg.scheduler_gamma == pytest.approx(1.0, rel=1e-10)

    def test_scheduler_gamma_less_than_one(self) -> None:
        cfg = TrainingConfig()
        assert 0.0 < cfg.scheduler_gamma < 1.0
