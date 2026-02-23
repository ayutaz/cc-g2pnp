"""Tests for cc_g2pnp.training.evaluator."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cc_g2pnp.data.vocabulary import PnPVocabulary
from cc_g2pnp.training.evaluator import Evaluator, _compute_cer

# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def vocabulary():
    return PnPVocabulary()


@pytest.fixture()
def device():
    return torch.device("cpu")


class DummyModel(nn.Module):
    """Minimal model returning fixed loss and predictions."""

    def __init__(self, loss_value: float, pred_ids: list[list[int]]) -> None:
        super().__init__()
        self.loss_value = loss_value
        self.pred_ids = pred_ids
        # Dummy parameter so PyTorch treats this as a module
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, input_lengths, targets=None, target_lengths=None):
        return {"loss": torch.tensor(self.loss_value)}

    def inference(self, input_ids, input_lengths):
        return self.pred_ids


def _make_batch(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    input_lengths: torch.Tensor,
    label_lengths: torch.Tensor,
) -> dict:
    return {
        "input_ids": input_ids,
        "labels": labels,
        "input_lengths": input_lengths,
        "label_lengths": label_lengths,
    }


# ── _compute_cer tests ───────────────────────────────────────


class TestComputeCer:
    """Unit tests for the _compute_cer helper."""

    def test_perfect_match(self, vocabulary):
        """Identical predictions and references yield CER = 0.0."""
        # Use some real token IDs (e.g., 1=ア, 2=イ, 3=ウ)
        pred_ids = [[1, 2, 3]]
        ref_tensor = torch.tensor([1, 2, 3, 0, 0])  # padded with 0s
        ref_lengths = torch.tensor([3])

        cer = _compute_cer(pred_ids, [ref_tensor], ref_lengths, vocabulary)
        assert cer == 0.0

    def test_complete_mismatch(self, vocabulary):
        """Completely different predictions yield CER > 0."""
        pred_ids = [[4, 5, 6]]  # エ, オ, カ
        ref_tensor = torch.tensor([1, 2, 3])  # ア, イ, ウ
        ref_lengths = torch.tensor([3])

        cer = _compute_cer(pred_ids, [ref_tensor], ref_lengths, vocabulary)
        assert cer > 0.0

    def test_empty_reference_skipped(self, vocabulary):
        """Samples with empty reference are skipped."""
        # Only blank tokens in reference
        pred_ids = [[1, 2]]
        ref_tensor = torch.tensor([0, 0, 0])  # all blanks
        ref_lengths = torch.tensor([3])

        cer = _compute_cer(pred_ids, [ref_tensor], ref_lengths, vocabulary)
        # No valid samples, returns 0.0
        assert cer == 0.0

    def test_blank_excluded_from_both(self, vocabulary):
        """Blank tokens (ID 0) are excluded from both pred and ref."""
        # pred has blanks mixed in, ref also has blanks
        pred_ids = [[0, 1, 0, 2, 0]]
        ref_tensor = torch.tensor([0, 1, 2, 0])
        ref_lengths = torch.tensor([4])

        cer = _compute_cer(pred_ids, [ref_tensor], ref_lengths, vocabulary)
        # After blank removal: pred=[ア,イ], ref=[ア,イ] -> CER=0
        assert cer == 0.0

    def test_multiple_samples(self, vocabulary):
        """CER is averaged across multiple samples."""
        # Sample 1: perfect match
        # Sample 2: complete mismatch
        pred_ids = [[1, 2], [4, 5]]
        ref_tensors = [torch.tensor([1, 2]), torch.tensor([1, 2])]
        ref_lengths = torch.tensor([2, 2])

        cer = _compute_cer(pred_ids, ref_tensors, ref_lengths, vocabulary)
        # Sample 1 CER=0, Sample 2 CER>0, average should be > 0
        assert cer > 0.0

    def test_empty_prediction(self, vocabulary):
        """Empty prediction against non-empty reference yields CER > 0."""
        pred_ids = [[]]
        ref_tensor = torch.tensor([1, 2, 3])
        ref_lengths = torch.tensor([3])

        cer = _compute_cer(pred_ids, [ref_tensor], ref_lengths, vocabulary)
        assert cer > 0.0


# ── Evaluator tests ──────────────────────────────────────────


class TestEvaluator:
    """Tests for the Evaluator class."""

    def test_returns_correct_keys(self, vocabulary, device):
        """evaluate() returns dict with required keys."""
        model = DummyModel(loss_value=1.0, pred_ids=[[1, 2]])
        evaluator = Evaluator(vocabulary, device)
        batch = _make_batch(
            input_ids=torch.tensor([[100, 200]]),
            labels=torch.tensor([[1, 2]]),
            input_lengths=torch.tensor([2]),
            label_lengths=torch.tensor([2]),
        )

        result = evaluator.evaluate(model, [batch])

        assert "val_loss" in result
        assert "val_cer" in result
        assert "val_num_samples" in result

    def test_val_loss_averaged(self, vocabulary, device):
        """val_loss is correctly averaged across samples."""
        model = DummyModel(loss_value=2.0, pred_ids=[[1], [1]])
        evaluator = Evaluator(vocabulary, device)

        batch1 = _make_batch(
            input_ids=torch.tensor([[100, 200], [100, 200]]),
            labels=torch.tensor([[1, 0], [1, 0]]),
            input_lengths=torch.tensor([2, 2]),
            label_lengths=torch.tensor([1, 1]),
        )

        result = evaluator.evaluate(model, [batch1])

        assert result["val_loss"] == pytest.approx(2.0)
        assert result["val_num_samples"] == 2

    def test_val_cer_perfect_match(self, vocabulary, device):
        """CER is 0 when predictions match references."""
        model = DummyModel(loss_value=0.5, pred_ids=[[1, 2, 3]])
        evaluator = Evaluator(vocabulary, device)
        batch = _make_batch(
            input_ids=torch.tensor([[100, 200, 300]]),
            labels=torch.tensor([[1, 2, 3]]),
            input_lengths=torch.tensor([3]),
            label_lengths=torch.tensor([3]),
        )

        result = evaluator.evaluate(model, [batch])
        assert result["val_cer"] == 0.0

    def test_max_batches(self, vocabulary, device):
        """max_batches limits the number of batches evaluated."""
        model = DummyModel(loss_value=1.0, pred_ids=[[1]])
        evaluator = Evaluator(vocabulary, device)

        batches = [
            _make_batch(
                input_ids=torch.tensor([[100]]),
                labels=torch.tensor([[1]]),
                input_lengths=torch.tensor([1]),
                label_lengths=torch.tensor([1]),
            )
            for _ in range(5)
        ]

        result = evaluator.evaluate(model, batches, max_batches=2)
        assert result["val_num_samples"] == 2

    def test_empty_batches(self, vocabulary, device):
        """Empty batch list returns zeros."""
        model = DummyModel(loss_value=0.0, pred_ids=[])
        evaluator = Evaluator(vocabulary, device)

        result = evaluator.evaluate(model, [])

        assert result["val_loss"] == 0.0
        assert result["val_cer"] == 0.0
        assert result["val_num_samples"] == 0

    def test_eval_train_mode_switching(self, vocabulary, device):
        """model.eval() is called at start, model.train() at end."""
        model = DummyModel(loss_value=1.0, pred_ids=[[1]])
        evaluator = Evaluator(vocabulary, device)
        batch = _make_batch(
            input_ids=torch.tensor([[100]]),
            labels=torch.tensor([[1]]),
            input_lengths=torch.tensor([1]),
            label_lengths=torch.tensor([1]),
        )

        # Initially in train mode
        model.train()
        assert model.training is True

        evaluator.evaluate(model, [batch])

        # Should be back in train mode after evaluation
        assert model.training is True

    def test_eval_mode_during_evaluation(self, vocabulary, device):
        """Model is in eval mode during forward/inference calls."""
        training_states: list[bool] = []

        class TrackingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, input_lengths, targets=None, target_lengths=None):
                training_states.append(self.training)
                return {"loss": torch.tensor(1.0)}

            def inference(self, input_ids, input_lengths):
                training_states.append(self.training)
                return [[1]]

        model = TrackingModel()
        model.train()
        evaluator = Evaluator(vocabulary, device)
        batch = _make_batch(
            input_ids=torch.tensor([[100]]),
            labels=torch.tensor([[1]]),
            input_lengths=torch.tensor([1]),
            label_lengths=torch.tensor([1]),
        )

        evaluator.evaluate(model, [batch])

        # Both forward and inference should have seen training=False
        assert all(state is False for state in training_states)

    def test_multiple_batches_aggregation(self, vocabulary, device):
        """Metrics are correctly aggregated across multiple batches."""
        # Batch 1: 1 sample, loss=2.0
        # Batch 2: 1 sample, loss=4.0
        call_count = [0]
        losses = [2.0, 4.0]

        class VaryingLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, input_lengths, targets=None, target_lengths=None):
                loss = losses[call_count[0]]
                call_count[0] += 1
                return {"loss": torch.tensor(loss)}

            def inference(self, input_ids, input_lengths):
                return [[1]]

        model = VaryingLossModel()
        evaluator = Evaluator(vocabulary, device)

        batch1 = _make_batch(
            input_ids=torch.tensor([[100]]),
            labels=torch.tensor([[1]]),
            input_lengths=torch.tensor([1]),
            label_lengths=torch.tensor([1]),
        )
        batch2 = _make_batch(
            input_ids=torch.tensor([[200]]),
            labels=torch.tensor([[1]]),
            input_lengths=torch.tensor([1]),
            label_lengths=torch.tensor([1]),
        )

        result = evaluator.evaluate(model, [batch1, batch2])

        # Weighted average: (2.0*1 + 4.0*1) / 2 = 3.0
        assert result["val_loss"] == pytest.approx(3.0)
        assert result["val_num_samples"] == 2

    def test_no_grad_context(self, vocabulary, device):
        """Evaluation runs under torch.no_grad()."""
        grad_enabled_states: list[bool] = []

        class GradCheckModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._dummy = nn.Parameter(torch.zeros(1))

            def forward(self, input_ids, input_lengths, targets=None, target_lengths=None):
                grad_enabled_states.append(torch.is_grad_enabled())
                return {"loss": torch.tensor(1.0)}

            def inference(self, input_ids, input_lengths):
                grad_enabled_states.append(torch.is_grad_enabled())
                return [[1]]

        model = GradCheckModel()
        evaluator = Evaluator(vocabulary, device)
        batch = _make_batch(
            input_ids=torch.tensor([[100]]),
            labels=torch.tensor([[1]]),
            input_lengths=torch.tensor([1]),
            label_lengths=torch.tensor([1]),
        )

        evaluator.evaluate(model, [batch])

        assert all(state is False for state in grad_enabled_states)
