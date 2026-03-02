"""Tests for CALM2 tokenizer."""

import pytest

from cc_g2pnp.data.tokenizer import G2PnPTokenizer

pytestmark = pytest.mark.network

EXPECTED_VOCAB_SIZE = 65000


def test_tokenizer_loads(calm2_tokenizer):
    """Tokenizer should load successfully."""
    assert calm2_tokenizer is not None


def test_vocab_size(calm2_tokenizer):
    """CALM2 vocab size should be 65000."""
    assert calm2_tokenizer.vocab_size == EXPECTED_VOCAB_SIZE


def test_encode_japanese(calm2_tokenizer):
    """Japanese text should encode to non-empty token IDs."""
    text = "東京都に住んでいます"
    ids = calm2_tokenizer.encode(text)
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)


def test_decode_roundtrip(calm2_tokenizer):
    """Encode then decode should recover the original text."""
    text = "音声認識の研究"
    ids = calm2_tokenizer.encode(text, add_special_tokens=False)
    decoded = calm2_tokenizer.decode(ids)
    assert decoded == text


def test_batch_encode(calm2_tokenizer):
    """Batch encoding should return matching number of sequences."""
    texts = ["こんにちは", "音声認識", "自然言語処理"]
    batch = calm2_tokenizer(texts, padding=False)
    assert len(batch["input_ids"]) == len(texts)
    for ids in batch["input_ids"]:
        assert len(ids) > 0


def test_token_ids_in_range(calm2_tokenizer):
    """All token IDs should be within [0, vocab_size)."""
    text = "日本語のテストです。これは長めの文章で、色々な文字を含みます。"
    ids = calm2_tokenizer.encode(text, add_special_tokens=False)
    for token_id in ids:
        assert 0 <= token_id < EXPECTED_VOCAB_SIZE


# ── G2PnPTokenizer wrapper tests ──


def test_g2pnp_vocab_size(g2pnp_tokenizer):
    """G2PnPTokenizer.vocab_size should return 65000."""
    assert g2pnp_tokenizer.vocab_size == EXPECTED_VOCAB_SIZE


def test_g2pnp_decode(g2pnp_tokenizer):
    """G2PnPTokenizer.decode() round-trip should recover original text."""
    text = "今日はいい天気"
    ids = g2pnp_tokenizer.encode(text)
    decoded = g2pnp_tokenizer.decode(ids)
    assert len(decoded) > 0
    assert "今日" in decoded or decoded.strip() == text.strip()


def test_g2pnp_batch_encode(g2pnp_tokenizer):
    """G2PnPTokenizer.batch_encode() should match individual encode results."""
    texts = ["今日はいい天気", "明日は雨"]
    results = g2pnp_tokenizer.batch_encode(texts)
    assert len(results) == 2
    assert all(isinstance(r, list) for r in results)
    assert all(len(r) > 0 for r in results)
    for text, batch_result in zip(texts, results, strict=True):
        single_result = g2pnp_tokenizer.encode(text)
        assert batch_result == single_result


# ── get_instance / clear_cache tests ──


def test_get_instance_returns_same_object():
    """get_instance() should return the same object for the same model_name."""
    G2PnPTokenizer.clear_cache()
    try:
        t1 = G2PnPTokenizer.get_instance()
        t2 = G2PnPTokenizer.get_instance()
        assert t1 is t2
    finally:
        G2PnPTokenizer.clear_cache()


def test_clear_cache_resets_singleton():
    """clear_cache() should cause get_instance() to create a new object."""
    G2PnPTokenizer.clear_cache()
    try:
        t1 = G2PnPTokenizer.get_instance()
        G2PnPTokenizer.clear_cache()
        t2 = G2PnPTokenizer.get_instance()
        assert t1 is not t2
    finally:
        G2PnPTokenizer.clear_cache()
