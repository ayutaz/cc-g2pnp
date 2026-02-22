"""Tests for PnP label generation from Japanese text."""

from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
from cc_g2pnp.data.vocabulary import PnPVocabulary


def test_basic_output():
    """Should produce non-empty katakana + prosody list for basic input."""
    result = generate_pnp_labels("こんにちは")
    assert len(result) > 0
    # Should contain katakana characters
    katakana = [t for t in result if t not in ("*", "/", "#")]
    assert len(katakana) > 0


def test_accent_nucleus_ame():
    """雨 (LH accent type 1) should have accent nucleus on first mora."""
    result = generate_pnp_labels("雨")
    # 雨 = ア*メ (head-high: accent on ア)
    assert "*" in result
    star_idx = result.index("*")
    assert star_idx > 0  # * comes after a mora
    assert result[star_idx - 1] == "ア"


def test_no_accent_nucleus_ame():
    """飴 (LH flat pattern) should not have accent nucleus."""
    result = generate_pnp_labels("飴")
    # 飴 = アメ (flat — no accent nucleus)
    assert "*" not in result


def test_phrase_boundary():
    """Multi-phrase text should produce / or # boundaries."""
    result = generate_pnp_labels("今日はいい天気")
    boundaries = [t for t in result if t in ("/", "#")]
    assert len(boundaries) > 0, "No phrase boundaries found"


def test_long_vowel():
    """Long vowels should be represented as ー."""
    result = generate_pnp_labels("お母さん")
    # おかあさん → オ カ * ー サ ン
    assert "ー" in result


def test_empty_text():
    """Empty or whitespace text should return empty list."""
    assert generate_pnp_labels("") == []
    assert generate_pnp_labels("   ") == []


def test_all_tokens_in_vocabulary():
    """Every generated token should be in the PnP vocabulary."""
    vocab = PnPVocabulary()
    for text in ["今日は天気がいいですね", "東京都に住んでいます", "音声認識"]:
        tokens = generate_pnp_labels(text)
        for tok in tokens:
            assert tok in vocab.token_to_id, f"Token {tok!r} not in vocabulary (text={text!r})"


def test_ctc_constraint_basic():
    """Short texts should satisfy CTC constraint with typical BPE length."""
    # A single word typically has 1-3 BPE tokens and ~2-5 PnP labels.
    # CTC constraint: len(bpe) * 8 >= len(pnp)
    from cc_g2pnp.data.tokenizer import G2PnPTokenizer

    tok = G2PnPTokenizer()
    for text in ["こんにちは", "東京", "天気"]:
        bpe_ids = tok.encode(text)
        pnp_labels = generate_pnp_labels(text)
        assert len(bpe_ids) * 8 >= len(pnp_labels), (
            f"CTC violation: {len(bpe_ids)}*8 < {len(pnp_labels)} for {text!r}"
        )


def test_intonation_phrase_boundary():
    """Sentence with pause should produce # (IP boundary)."""
    result = generate_pnp_labels("私は学生です。よろしくお願いします。")
    assert "#" in result, "No intonation phrase boundary found"
