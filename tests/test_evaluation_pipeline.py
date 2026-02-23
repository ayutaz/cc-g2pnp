"""Tests for cc_g2pnp.evaluation.pipeline."""

from __future__ import annotations

import pytest

from cc_g2pnp.data.vocabulary import PnPVocabulary
from cc_g2pnp.evaluation.eval_data import EvalDataset, EvalSample
from cc_g2pnp.evaluation.pipeline import (
    EvalConfig,
    EvalResult,
    EvaluationPipeline,
)
from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.model.config import CC_G2PnPConfig


@pytest.fixture
def small_config():
    """Tiny model config for fast tests."""
    return CC_G2PnPConfig(
        d_model=32,
        num_heads=2,
        d_ff=64,
        num_layers=2,
        conv_kernel_size=3,
        chunk_size=5,
        mla_size=1,
        past_context=4,
        upsample_factor=2,
        intermediate_ctc_layers=(),
    )


@pytest.fixture
def small_model(small_config):
    model = CC_G2PnP(small_config)
    model.eval()
    return model


@pytest.fixture
def vocabulary():
    return PnPVocabulary()


@pytest.fixture
def pipeline(small_model, vocabulary):
    return EvaluationPipeline(small_model, vocabulary, EvalConfig(batch_size=4))


@pytest.fixture
def sample_dataset():
    """Create a small eval dataset with mock BPE IDs."""
    samples = [
        EvalSample(
            text="test1",
            domain="news",
            pnp_labels=["キョ", "*", "ー"],
            bpe_ids=list(range(5)),
        ),
        EvalSample(
            text="test2",
            domain="news",
            pnp_labels=["テ", "*", "ン", "キ"],
            bpe_ids=list(range(3)),
        ),
        EvalSample(
            text="test3",
            domain="conv",
            pnp_labels=["コ", "ン", "ニ", "チ", "ワ"],
            bpe_ids=list(range(4)),
        ),
    ]
    return EvalDataset(samples=samples, name="test")


class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig()
        assert cfg.batch_size == 32
        assert cfg.device == "cpu"
        assert cfg.use_streaming is False
        assert cfg.max_samples is None

    def test_custom(self):
        cfg = EvalConfig(
            batch_size=16, device="cuda:0", use_streaming=True, max_samples=100
        )
        assert cfg.batch_size == 16
        assert cfg.device == "cuda:0"
        assert cfg.use_streaming is True
        assert cfg.max_samples == 100


class TestEvalResult:
    def test_defaults(self):
        result = EvalResult()
        assert result.metrics == {}
        assert result.per_domain_metrics == {}
        assert result.num_samples == 0
        assert result.num_domains == 0

    def test_with_data(self):
        result = EvalResult(
            metrics={"pnp_cer": 0.05},
            per_domain_metrics={"news": {"pnp_cer": 0.03}},
            num_samples=100,
            num_domains=2,
        )
        assert result.num_samples == 100


class TestEvaluationPipeline:
    def test_creation(self, pipeline):
        assert pipeline.model is not None
        assert pipeline.vocabulary is not None

    def test_evaluate_basic(self, pipeline, sample_dataset):
        """Basic evaluate runs without error and returns results."""
        result = pipeline.evaluate(sample_dataset)
        assert isinstance(result, EvalResult)
        assert result.num_samples == 3
        assert result.num_domains == 2
        assert "pnp_cer" in result.metrics
        assert "pnp_ser" in result.metrics
        assert "normalized_pnp_cer" in result.metrics
        assert "phoneme_cer" in result.metrics
        assert len(result.per_domain_metrics) == 2
        assert "news" in result.per_domain_metrics
        assert "conv" in result.per_domain_metrics

    def test_evaluate_empty_dataset(self, pipeline):
        empty = EvalDataset(samples=[], name="empty")
        result = pipeline.evaluate(empty)
        assert result.num_samples == 0
        assert result.metrics == {}

    def test_evaluate_max_samples(self, pipeline, sample_dataset):
        """max_samples limits number of evaluated samples."""
        pipeline.config.max_samples = 1
        result = pipeline.evaluate(sample_dataset)
        assert result.num_samples == 1

    def test_evaluate_streaming(self, pipeline, sample_dataset):
        """Streaming evaluation runs without error."""
        pipeline.config.use_streaming = True
        result = pipeline.evaluate(sample_dataset)
        assert isinstance(result, EvalResult)
        assert result.num_samples == 3

    def test_format_results(self, pipeline, sample_dataset):
        result = pipeline.evaluate(sample_dataset)
        formatted = pipeline.format_results(result)
        assert "Evaluation Results" in formatted
        assert "pnp_cer" in formatted
        assert "news" in formatted

    def test_format_results_empty(self, pipeline):
        result = EvalResult()
        formatted = pipeline.format_results(result)
        assert "0 samples" in formatted

    def test_decode_ids_to_tokens(self, pipeline):
        """Internal decode removes blank and pad tokens."""
        vocab = pipeline.vocabulary
        # Create IDs including blank (0) and pad
        ids = [vocab.blank_id, 1, 2, vocab.pad_id, 3]
        result = pipeline._decode_ids_to_tokens([ids])
        assert len(result) == 1
        # blank and pad should be removed
        for token in result[0]:
            assert token not in ("<blank>", "<pad>")

    def test_per_domain_metrics_breakdown(self, pipeline, sample_dataset):
        result = pipeline.evaluate(sample_dataset)
        for _domain, metrics in result.per_domain_metrics.items():
            assert "pnp_cer" in metrics
            assert "pnp_ser" in metrics
            assert "normalized_pnp_cer" in metrics
            assert "normalized_pnp_ser" in metrics
            assert "phoneme_cer" in metrics
            assert "phoneme_ser" in metrics

    def test_from_checkpoint(self, small_config, tmp_path):
        """from_checkpoint loads model config and weights from a saved checkpoint."""
        import dataclasses

        import torch

        model = CC_G2PnP(small_config)
        # Save checkpoint with model_config key
        checkpoint = {
            "step": 100,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "config": {},
            "model_config": dataclasses.asdict(small_config),
            "metrics": None,
            "scaler_state_dict": None,
        }
        ckpt_path = tmp_path / "step_00000100.pt"
        torch.save(checkpoint, ckpt_path)

        from cc_g2pnp.evaluation.pipeline import EvalConfig

        pipe = EvaluationPipeline.from_checkpoint(ckpt_path, EvalConfig(device="cpu"))
        assert pipe.model is not None
        assert pipe.model.config.d_model == small_config.d_model
        assert pipe.model.config.num_layers == small_config.num_layers

    def test_from_checkpoint_no_model_config(self, tmp_path):
        """from_checkpoint falls back to default CC_G2PnPConfig if model_config absent."""
        import torch

        from cc_g2pnp.model.config import CC_G2PnPConfig as Cfg

        default_cfg = Cfg()
        model = CC_G2PnP(default_cfg)
        checkpoint = {
            "step": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "scheduler_state_dict": {},
            "config": {"lr": 1e-3},  # TrainingConfig fields — should be ignored
            "metrics": None,
            "scaler_state_dict": None,
        }
        ckpt_path = tmp_path / "step_00000000.pt"
        torch.save(checkpoint, ckpt_path)

        from cc_g2pnp.evaluation.pipeline import EvalConfig

        pipe = EvaluationPipeline.from_checkpoint(ckpt_path, EvalConfig(device="cpu"))
        assert pipe.model.config.d_model == default_cfg.d_model
