"""Integration tests for the full data pipeline."""

import pytest
import torch

from cc_g2pnp.data.collator import DynamicBatchCollator, dynamic_batch_sampler
from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
from cc_g2pnp.data.tokenizer import G2PnPTokenizer
from cc_g2pnp.data.vocabulary import PnPVocabulary


@pytest.fixture(scope="module")
def tokenizer():
    return G2PnPTokenizer()


@pytest.fixture(scope="module")
def vocab():
    return PnPVocabulary()


# ── Tokenizer + PnP labeler integration ──────────────────────────

def test_tokenizer_pnp_pair(tokenizer, vocab):
    """Tokenizer and PnP labeler should produce valid paired output."""
    text = "今日はいい天気"
    bpe_ids = tokenizer.encode(text)
    pnp_tokens = generate_pnp_labels(text)
    pnp_ids = vocab.encode(pnp_tokens)

    assert len(bpe_ids) > 0
    assert len(pnp_ids) > 0
    assert all(isinstance(i, int) for i in bpe_ids)
    assert all(isinstance(i, int) for i in pnp_ids)


def test_ctc_constraint_multiple(tokenizer, vocab):
    """CTC constraint should hold for a variety of texts."""
    texts = [
        "こんにちは",
        "東京都に住んでいます",
        "音声認識の研究をしています",
        "今日は天気がいいですね",
        "人工知能",
        "自然言語処理",
    ]
    for text in texts:
        bpe_ids = tokenizer.encode(text)
        pnp_tokens = generate_pnp_labels(text)
        pnp_ids = vocab.encode(pnp_tokens)
        assert len(bpe_ids) * 8 >= len(pnp_ids), (
            f"CTC violation for {text!r}: {len(bpe_ids)}*8={len(bpe_ids)*8} < {len(pnp_ids)}"
        )


# ── Collator tests ───────────────────────────────────────────────

def test_collator_padding_shape(tokenizer, vocab):
    """Collator should pad to max length and produce correct shapes."""
    texts = ["こんにちは", "東京都に住んでいます", "天気"]
    batch = []
    for text in texts:
        bpe_ids = tokenizer.encode(text)
        pnp_ids = vocab.encode(generate_pnp_labels(text))
        batch.append({"input_ids": bpe_ids, "labels": pnp_ids})

    collator = DynamicBatchCollator()
    result = collator(batch)

    assert result["input_ids"].shape[0] == len(texts)
    assert result["labels"].shape[0] == len(texts)
    assert result["input_lengths"].shape == (len(texts),)
    assert result["label_lengths"].shape == (len(texts),)

    # Max input length should match the longest sequence
    max_input = max(len(tokenizer.encode(t)) for t in texts)
    assert result["input_ids"].shape[1] == max_input


def test_collator_lengths_match(tokenizer, vocab):
    """Recorded lengths should match the original unpadded lengths."""
    texts = ["音声", "自然言語処理は面白い"]
    batch = []
    original_input_lens = []
    original_label_lens = []
    for text in texts:
        bpe_ids = tokenizer.encode(text)
        pnp_ids = vocab.encode(generate_pnp_labels(text))
        batch.append({"input_ids": bpe_ids, "labels": pnp_ids})
        original_input_lens.append(len(bpe_ids))
        original_label_lens.append(len(pnp_ids))

    collator = DynamicBatchCollator()
    result = collator(batch)

    assert result["input_lengths"].tolist() == original_input_lens
    assert result["label_lengths"].tolist() == original_label_lens


def test_collator_dtypes():
    """Collator output tensors should be long dtype."""
    batch = [
        {"input_ids": [1, 2, 3], "labels": [10, 20]},
        {"input_ids": [4, 5], "labels": [30, 40, 50]},
    ]
    collator = DynamicBatchCollator()
    result = collator(batch)
    assert result["input_ids"].dtype == torch.long
    assert result["labels"].dtype == torch.long


# ── Dynamic batch sampler tests ──────────────────────────────────

def test_dynamic_batch_sampler_respects_budget():
    """Each batch should have total tokens ≤ max_tokens."""
    samples = [{"input_ids": list(range(i * 10 + 1))} for i in range(20)]
    max_tokens = 100

    for batch in dynamic_batch_sampler(samples, max_tokens=max_tokens):
        total = sum(len(s["input_ids"]) for s in batch)
        assert total <= max_tokens or len(batch) == 1  # single oversize sample


def test_dynamic_batch_sampler_all_samples_yielded():
    """All input samples should appear in exactly one batch."""
    samples = [{"input_ids": [0] * (i + 1)} for i in range(15)]
    max_tokens = 20

    all_samples = []
    for batch in dynamic_batch_sampler(samples, max_tokens=max_tokens):
        all_samples.extend(batch)

    assert len(all_samples) == len(samples)


# ── ReazonSpeech integration (network required) ─────────────────

@pytest.mark.network
@pytest.mark.slow
def test_dataset_iteration():
    """G2PnPDataset should yield valid samples from ReazonSpeech."""
    from cc_g2pnp.data.dataset import G2PnPDataset

    ds = G2PnPDataset(subset="all", streaming=True)
    samples = []
    for sample in ds:
        samples.append(sample)
        if len(samples) >= 5:
            break

    assert len(samples) == 5
    for sample in samples:
        assert "input_ids" in sample
        assert "labels" in sample
        assert len(sample["input_ids"]) > 0
        assert len(sample["labels"]) > 0
        # CTC constraint
        assert len(sample["input_ids"]) * 8 >= len(sample["labels"])
