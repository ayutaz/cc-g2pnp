"""Tests for cc_g2pnp.evaluation.metrics."""

from __future__ import annotations

import pytest

from cc_g2pnp.evaluation.metrics import (
    PROSODY_SYMBOLS,
    _filter_phonemes,
    _normalize_tokens,
    _tokens_to_str,
    compute_cer,
    compute_normalized_pnp_cer,
    compute_normalized_pnp_ser,
    compute_phoneme_cer,
    compute_phoneme_ser,
    compute_pnp_cer,
    compute_pnp_ser,
    compute_ser,
    evaluate_all,
)


class TestHelpers:
    def test_normalize_tokens_replaces_hash(self):
        tokens = ["\u30ad\u30e7", "*", "\u30fc", "\u30ef", "#", "\u30c6", "*", "\u30f3", "\u30ad"]
        result = _normalize_tokens(tokens)
        assert "#" not in result
        assert result[4] == "/"

    def test_normalize_tokens_preserves_others(self):
        tokens = ["\u30ad\u30e7", "*", "/", "\u30c6"]
        assert _normalize_tokens(tokens) == tokens

    def test_filter_phonemes_removes_prosody(self):
        tokens = ["\u30ad\u30e7", "*", "\u30fc", "\u30ef", "/", "\u30c6", "#", "\u30f3"]
        result = _filter_phonemes(tokens)
        assert result == ["\u30ad\u30e7", "\u30fc", "\u30ef", "\u30c6", "\u30f3"]

    def test_filter_phonemes_empty(self):
        assert _filter_phonemes([]) == []
        assert _filter_phonemes(["*", "/", "#"]) == []

    def test_tokens_to_str(self):
        assert _tokens_to_str(["\u30ad\u30e7", "*", "\u30fc"]) == "\u30ad\u30e7 * \u30fc"
        assert _tokens_to_str([]) == ""

    def test_prosody_symbols_constant(self):
        assert frozenset({"*", "/", "#"}) == PROSODY_SYMBOLS


class TestComputeCER:
    def test_perfect_prediction(self):
        refs = [["\u30ad\u30e7", "*", "\u30fc", "\u30ef"]]
        preds = [["\u30ad\u30e7", "*", "\u30fc", "\u30ef"]]
        assert compute_cer(preds, refs) == 0.0

    def test_completely_wrong(self):
        refs = [["\u30a2"]]
        preds = [["\u30a4"]]
        assert compute_cer(preds, refs) == 1.0  # 1 substitution / 1 ref char

    def test_empty_prediction(self):
        refs = [["\u30a2", "\u30a4"]]
        preds = [[]]
        assert compute_cer(preds, refs) == 1.0  # all deletions

    def test_empty_reference_skipped(self):
        refs = [[], ["\u30a2"]]
        preds = [["\u30a4"], ["\u30a2"]]
        assert compute_cer(preds, refs) == 0.0  # empty ref skipped, remaining is perfect

    def test_all_empty_references(self):
        assert compute_cer([[]], [[]]) == 0.0

    def test_multiple_sentences(self):
        refs = [["\u30a2", "\u30a4"], ["\u30a6", "\u30a8"]]
        preds = [["\u30a2", "\u30a4"], ["\u30a6", "\u30aa"]]  # first perfect, second 1 sub
        cer = compute_cer(preds, refs)
        assert 0.0 < cer < 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_cer([["\u30a2"]], [["\u30a2"], ["\u30a4"]])


class TestComputeSER:
    def test_all_correct(self):
        refs = [["\u30a2"], ["\u30a4"]]
        preds = [["\u30a2"], ["\u30a4"]]
        assert compute_ser(preds, refs) == 0.0

    def test_all_wrong(self):
        refs = [["\u30a2"], ["\u30a4"]]
        preds = [["\u30a4"], ["\u30a2"]]
        assert compute_ser(preds, refs) == 100.0

    def test_half_wrong(self):
        refs = [["\u30a2"], ["\u30a4"]]
        preds = [["\u30a2"], ["\u30a6"]]
        assert compute_ser(preds, refs) == 50.0

    def test_empty_reference_skipped(self):
        refs = [[], ["\u30a2"]]
        preds = [["anything"], ["\u30a2"]]
        assert compute_ser(preds, refs) == 0.0

    def test_no_valid_pairs(self):
        assert compute_ser([[]], [[]]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_ser([["\u30a2"]], [["\u30a2"], ["\u30a4"]])


class TestPnPMetrics:
    """Test PnP CER/SER (wrappers around compute_cer/ser)."""

    def test_pnp_cer_delegates(self):
        refs = [["\u30ad\u30e7", "*", "\u30fc"]]
        preds = [["\u30ad\u30e7", "*", "\u30fc"]]
        assert compute_pnp_cer(preds, refs) == 0.0

    def test_pnp_ser_delegates(self):
        refs = [["\u30ad\u30e7", "*", "\u30fc"]]
        preds = [["\u30ad\u30e7", "*", "\u30fc"]]
        assert compute_pnp_ser(preds, refs) == 0.0


class TestNormalizedMetrics:
    def test_hash_difference_ignored(self):
        """# and / should be treated the same after normalization."""
        refs = [["\u30ad\u30e7", "#", "\u30c6"]]
        preds = [["\u30ad\u30e7", "/", "\u30c6"]]
        # Without normalization, these differ
        assert compute_pnp_cer(preds, refs) > 0.0
        # With normalization, these should be equal
        assert compute_normalized_pnp_cer(preds, refs) == 0.0
        assert compute_normalized_pnp_ser(preds, refs) == 0.0

    def test_other_differences_remain(self):
        refs = [["\u30ad\u30e7", "#", "\u30c6"]]
        preds = [["\u30ad\u30e7", "#", "\u30ad"]]
        assert compute_normalized_pnp_cer(preds, refs) > 0.0


class TestPhonemeMetrics:
    def test_prosody_difference_ignored(self):
        """Prosody symbols should be removed before comparison."""
        refs = [["\u30ad\u30e7", "*", "\u30fc", "\u30ef"]]
        preds = [["\u30ad\u30e7", "\u30fc", "\u30ef"]]  # missing accent marker
        # With prosody, these differ
        assert compute_pnp_cer(preds, refs) > 0.0
        # Without prosody, phonemes are identical
        assert compute_phoneme_cer(preds, refs) == 0.0
        assert compute_phoneme_ser(preds, refs) == 0.0

    def test_phoneme_difference_detected(self):
        refs = [["\u30ad\u30e7", "*", "\u30fc", "\u30ef"]]
        preds = [["\u30ad\u30e7", "*", "\u30fc", "\u30ab"]]  # different phoneme
        assert compute_phoneme_cer(preds, refs) > 0.0


class TestEvaluateAll:
    def test_returns_all_keys(self):
        refs = [["\u30ad\u30e7", "*", "\u30fc"]]
        preds = [["\u30ad\u30e7", "*", "\u30fc"]]
        result = evaluate_all(preds, refs)
        expected_keys = {
            "pnp_cer", "pnp_ser",
            "normalized_pnp_cer", "normalized_pnp_ser",
            "phoneme_cer", "phoneme_ser",
        }
        assert set(result.keys()) == expected_keys

    def test_perfect_prediction_all_zero(self):
        refs = [["\u30ad\u30e7", "*", "\u30fc", "\u30ef", "/", "\u30c6", "*", "\u30f3", "\u30ad"]]
        preds = [["\u30ad\u30e7", "*", "\u30fc", "\u30ef", "/", "\u30c6", "*", "\u30f3", "\u30ad"]]
        result = evaluate_all(preds, refs)
        for key, val in result.items():
            assert val == 0.0, f"{key} should be 0.0"

    def test_all_metrics_are_floats(self):
        refs = [["\u30a2"]]
        preds = [["\u30a4"]]
        result = evaluate_all(preds, refs)
        for val in result.values():
            assert isinstance(val, float)

    def test_realistic_example(self):
        # Reference: キョ * ー ワ / イ * ー / テ * ン キ
        refs = [["\u30ad\u30e7", "*", "\u30fc", "\u30ef", "/", "\u30a4", "*", "\u30fc", "/", "\u30c6", "*", "\u30f3", "\u30ad"]]
        # Prediction: same phonemes but wrong accent
        preds = [["\u30ad\u30e7", "\u30fc", "\u30ef", "/", "\u30a4", "\u30fc", "/", "\u30c6", "\u30f3", "\u30ad"]]
        result = evaluate_all(preds, refs)
        # PnP CER > 0 (missing accent markers)
        assert result["pnp_cer"] > 0.0
        # Phoneme CER = 0 (phonemes are same)
        assert result["phoneme_cer"] == 0.0
