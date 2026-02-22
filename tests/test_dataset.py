"""G2PnPDataset offline tests using mocks (no network access required)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from cc_g2pnp.data.dataset import G2PnPDataset


# ── Helper ───────────────────────────────────────────────────────


def _make_sample(text: str) -> dict:
    """Create a mock ReazonSpeech-style sample dict."""
    return {"name": "dummy-001", "transcription": text}


def _make_dataset(**kwargs) -> G2PnPDataset:
    """Instantiate G2PnPDataset with _load_stream pre-patched (no-op).

    The actual stream data is injected per test via the returned mock.
    We patch _load_stream at class level so __init__ runs normally,
    loading the real tokenizer and vocabulary.
    """
    with patch.object(G2PnPDataset, "_load_stream", return_value=iter([])):
        return G2PnPDataset(**kwargs)


def _collect(dataset: G2PnPDataset, stream_data: list[dict]) -> list[dict]:
    """Run the dataset iterator over *stream_data* and collect all results."""
    with patch.object(dataset, "_load_stream", return_value=iter(stream_data)):
        return list(dataset)


# ── Tests ────────────────────────────────────────────────────────


class TestNormalSample:
    """Verify that a normal text sample is yielded correctly."""

    def test_normal_sample_yielded(self):
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("今日はいい天気")])
        assert len(results) == 1

    def test_output_has_input_ids(self):
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("今日はいい天気")])
        assert "input_ids" in results[0]

    def test_output_has_labels(self):
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("今日はいい天気")])
        assert "labels" in results[0]

    def test_input_ids_are_ints(self):
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("今日はいい天気")])
        assert all(isinstance(i, int) for i in results[0]["input_ids"])

    def test_labels_are_ints(self):
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("今日はいい天気")])
        assert all(isinstance(i, int) for i in results[0]["labels"])


class TestEmptyTextSkipped:
    """Empty or whitespace-only transcriptions should be skipped."""

    @pytest.mark.parametrize("text", ["", "   ", "\t", "\n"])
    def test_empty_text_skipped(self, text):
        ds = _make_dataset()
        results = _collect(ds, [_make_sample(text)])
        assert len(results) == 0

    def test_missing_transcription_skipped(self):
        ds = _make_dataset()
        results = _collect(ds, [{"name": "x"}])  # no 'transcription' key
        assert len(results) == 0


class TestLengthFiltering:
    """Length-based filtering: min_input_len and max_input_len."""

    def test_bpe_too_short_skipped(self):
        # min_input_len=9999 should reject everything
        ds = _make_dataset(min_input_len=9999)
        results = _collect(ds, [_make_sample("こんにちは")])
        assert len(results) == 0

    def test_bpe_too_long_skipped(self):
        # max_input_len=1 should reject anything with >1 BPE token
        ds = _make_dataset(max_input_len=1)
        results = _collect(ds, [_make_sample("音声認識の研究をしています")])
        assert len(results) == 0

    def test_within_bounds_accepted(self):
        ds = _make_dataset(min_input_len=1, max_input_len=512)
        results = _collect(ds, [_make_sample("自然言語処理")])
        assert len(results) == 1


class TestCTCConstraint:
    """CTC constraint: len(bpe_ids) * 8 >= len(pnp_ids)."""

    def test_ctc_violation_skipped(self):
        # Mock tokenizer to return exactly 2 BPE tokens (passes min_input_len=2)
        # so CTC allows at most 2*8=16 PnP tokens.
        # The long text below produces >> 16 PnP tokens, triggering CTC violation.
        ds = _make_dataset()
        long_text = "東京都港区芝公園四丁目二番八号にある東京タワーは高さ三百三十三メートルです"
        with patch.object(ds._tokenizer, "encode", return_value=[100, 200]):
            results = _collect(ds, [_make_sample(long_text)])
        assert len(results) == 0

    def test_ctc_satisfied_accepted(self):
        ds = _make_dataset(min_input_len=1)
        results = _collect(ds, [_make_sample("天気")])
        assert len(results) == 1
        # Verify the CTC constraint holds
        sample = results[0]
        assert len(sample["input_ids"]) * 8 >= len(sample["labels"])


class TestPnPLabelFailure:
    """Samples where PnP label generation returns empty should be skipped."""

    def test_pnp_failure_skipped(self):
        ds = _make_dataset()
        # Patch generate_pnp_labels to return empty
        with patch(
            "cc_g2pnp.data.dataset.generate_pnp_labels", return_value=[]
        ):
            results = _collect(ds, [_make_sample("何かのテキスト")])
        assert len(results) == 0


class TestMultipleSamplesFiltering:
    """Mixed input: some samples pass, some get filtered."""

    def test_filtering_mix(self):
        ds = _make_dataset(min_input_len=1, max_input_len=512)
        samples = [
            _make_sample("今日はいい天気"),     # should pass
            _make_sample(""),                    # empty -> skip
            _make_sample("東京"),                # should pass
            _make_sample("   "),                 # whitespace -> skip
            _make_sample("人工知能"),             # should pass
        ]
        results = _collect(ds, samples)
        assert len(results) == 3

    def test_all_filtered(self):
        ds = _make_dataset(min_input_len=9999)
        samples = [
            _make_sample("短い"),
            _make_sample("テスト"),
        ]
        results = _collect(ds, samples)
        assert len(results) == 0


class TestOutputFormat:
    """Verify the structure and types of yielded samples."""

    def test_keys_present(self):
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("音声合成")])
        assert len(results) == 1
        assert set(results[0].keys()) == {"input_ids", "labels"}

    def test_values_are_lists(self):
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("音声合成")])
        assert isinstance(results[0]["input_ids"], list)
        assert isinstance(results[0]["labels"], list)

    def test_non_empty_values(self):
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("音声合成")])
        assert len(results[0]["input_ids"]) > 0
        assert len(results[0]["labels"]) > 0

    def test_no_blank_in_labels(self):
        """CTC blank (id=0) should not appear in label sequences."""
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("音声合成")])
        # PnP labels should not contain the blank token (id=0)
        assert 0 not in results[0]["labels"]

    def test_no_pad_in_labels(self):
        """Pad token should not appear in label sequences."""
        ds = _make_dataset()
        results = _collect(ds, [_make_sample("音声合成")])
        pad_id = ds._vocab.pad_id
        assert pad_id not in results[0]["labels"]
