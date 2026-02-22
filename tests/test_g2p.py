"""Tests for pyopenjtalk and fugashi G2P functionality."""

import fugashi
import pyopenjtalk


def test_g2p_romaji():
    """pyopenjtalk should produce romaji phoneme output."""
    result = pyopenjtalk.g2p("こんにちは")
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain ASCII phoneme characters
    assert all(c.isascii() for c in result)


def test_g2p_kana():
    """pyopenjtalk should produce katakana output with kana=True."""
    result = pyopenjtalk.g2p("こんにちは", kana=True)
    assert isinstance(result, str)
    assert len(result) > 0
    # Should contain katakana characters (U+30A0-U+30FF)
    assert any("\u30a0" <= c <= "\u30ff" for c in result)


def test_extract_fullcontext():
    """pyopenjtalk should extract HTS full-context labels."""
    labels = pyopenjtalk.extract_fullcontext("東京都に住んでいます")
    assert isinstance(labels, list)
    assert len(labels) > 0
    assert all(isinstance(label, str) for label in labels)


def test_fullcontext_has_accent_info():
    """Full-context labels should contain /A: section for accent info."""
    labels = pyopenjtalk.extract_fullcontext("東京都に住んでいます")
    has_accent = any("/A:" in label for label in labels)
    assert has_accent, "No /A: accent section found in full-context labels"


def test_fugashi_tokenize():
    """fugashi should perform basic morphological analysis."""
    tagger = fugashi.Tagger()
    words = tagger("東京都に住んでいます")
    assert len(words) > 0
    surfaces = [w.surface for w in words]
    # Should contain some recognizable morphemes
    assert any(s in surfaces for s in ["東京", "都", "に", "住ん"])
