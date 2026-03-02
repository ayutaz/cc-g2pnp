"""Unit tests for cc_g2pnp.evaluation.metrics — 6-metric suite."""

from __future__ import annotations

from cc_g2pnp.evaluation.metrics import (
    compute_normalized_pnp_cer,
    compute_phoneme_cer,
    compute_pnp_cer,
    compute_pnp_ser,
    evaluate_all,
)

# サンプル PnP トークン列
_REF_FULL = [["キョ", "*", "ー", "ワ", "/", "テ", "*", "ン", "キ"]]
_PRED_FULL = [["キョ", "*", "ー", "ワ", "/", "テ", "*", "ン", "キ"]]

_REF_PARTIAL = [["キョ", "*", "ー", "カ"]]
_PRED_PARTIAL = [["キョ", "*", "ー", "ワ"]]  # 末尾 1 トークンが異なる


def test_pnp_cer_perfect_match():
    """完全一致の場合 CER = 0.0。"""
    assert compute_pnp_cer(_PRED_FULL, _REF_FULL) == 0.0


def test_pnp_cer_partial_match():
    """部分一致の場合 0 < CER < 1.0。"""
    cer = compute_pnp_cer(_PRED_PARTIAL, _REF_PARTIAL)
    assert 0.0 < cer < 1.0


def test_pnp_ser_perfect_match():
    """完全一致の場合 SER = 0.0。"""
    assert compute_pnp_ser(_PRED_FULL, _REF_FULL) == 0.0


def test_pnp_ser_all_wrong():
    """全文不一致の場合 SER = 100.0 (パーセント表記)。"""
    preds = [["ア"], ["イ"]]
    refs = [["イ"], ["ア"]]
    assert compute_pnp_ser(preds, refs) == 100.0


def test_normalized_pnp_cer():
    """'#' を '/' に変換後に一致する場合、正規化 CER = 0.0。"""
    preds = [["キョ", "/", "テ"]]
    refs = [["キョ", "#", "テ"]]
    # 正規化前は差異あり
    assert compute_pnp_cer(preds, refs) > 0.0
    # 正規化後は一致
    assert compute_normalized_pnp_cer(preds, refs) == 0.0


def test_phoneme_cer():
    """韻律記号除去後に音素が同じなら Phoneme CER = 0.0。"""
    preds = [["キョ", "ー", "ワ"]]  # アクセント核なし
    refs = [["キョ", "*", "ー", "ワ"]]  # アクセント核あり
    # 韻律記号込みでは差異あり
    assert compute_pnp_cer(preds, refs) > 0.0
    # 韻律記号除去後は音素が一致
    assert compute_phoneme_cer(preds, refs) == 0.0


def test_empty_predictions():
    """空のリストを渡した場合 compute_pnp_cer が 0.0 を返す。"""
    assert compute_pnp_cer([], []) == 0.0


def test_compute_all_metrics():
    """evaluate_all が 6 つのキーを持つ辞書を返す。"""
    result = evaluate_all(_PRED_FULL, _REF_FULL)
    expected_keys = {
        "pnp_cer",
        "pnp_ser",
        "normalized_pnp_cer",
        "normalized_pnp_ser",
        "phoneme_cer",
        "phoneme_ser",
    }
    assert set(result.keys()) == expected_keys
    for val in result.values():
        assert isinstance(val, float)
