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


class TestDDPSharding:
    """Verify DDP rank/world_size are passed to _load_stream."""

    def test_default_rank_world_size(self):
        ds = _make_dataset()
        assert ds._rank == 0
        assert ds._world_size == 1

    def test_custom_rank_world_size(self):
        ds = _make_dataset(rank=2, world_size=4)
        assert ds._rank == 2
        assert ds._world_size == 4

    def test_load_stream_called_with_rank_world_size(self):
        ds = _make_dataset(rank=1, world_size=4)
        with patch.object(ds, "_load_stream", return_value=iter([])) as mock_load:
            list(ds)
        mock_load.assert_called_once_with(rank=1, world_size=4)

    def test_single_gpu_no_shard(self):
        """world_size=1 ではシャーディングしない。"""
        ds = _make_dataset(rank=0, world_size=1)
        with patch.object(ds, "_load_stream", return_value=iter([])) as mock_load:
            list(ds)
        mock_load.assert_called_once_with(rank=0, world_size=1)


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


# ── LMDB Cache Tests ─────────────────────────────────────────────


def _make_lmdb_cache(tmp_path, entries: dict[str, list[int]]):
    """Create a temporary LMDB cache with the given text -> pnp_ids entries."""
    from cc_g2pnp.data.lmdb_cache import PnPLabelCache

    cache = PnPLabelCache(str(tmp_path / "cache"), readonly=False)
    for text, pnp_ids in entries.items():
        cache.put(text, pnp_ids)
    cache.close()
    return str(tmp_path / "cache")


class TestLmdbCacheIntegration:
    """Tests for LMDB cache integration in G2PnPDataset."""

    def test_dataset_without_lmdb(self):
        """lmdb_cache_dir=None のとき既存動作が維持される。"""
        ds = _make_dataset()
        assert ds._lmdb_cache is None
        results = _collect(ds, [_make_sample("今日はいい天気")])
        assert len(results) == 1
        assert "input_ids" in results[0]
        assert "labels" in results[0]

    def test_dataset_with_lmdb_cache(self, tmp_path):
        """キャッシュヒット時は generate_pnp_labels がバイパスされる。"""
        # 実際の PnP ID を事前に生成してキャッシュへ格納
        from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
        from cc_g2pnp.data.vocabulary import PnPVocabulary

        text = "今日はいい天気"
        vocab = PnPVocabulary()
        pnp_tokens = generate_pnp_labels(text)
        pnp_ids = vocab.encode(pnp_tokens)

        cache_dir = _make_lmdb_cache(tmp_path, {text: pnp_ids})

        ds = _make_dataset(lmdb_cache_dir=cache_dir)
        assert ds._lmdb_cache is not None

        with patch("cc_g2pnp.data.dataset.generate_pnp_labels") as mock_gen:
            results = _collect(ds, [_make_sample(text)])

        # キャッシュヒットなので generate_pnp_labels は呼ばれない
        mock_gen.assert_not_called()
        assert len(results) == 1
        assert results[0]["labels"] == pnp_ids

    def test_dataset_lmdb_cache_miss_fallback(self, tmp_path):
        """キャッシュミス時は generate_pnp_labels にフォールバックする。"""
        # キャッシュは別テキスト用エントリのみ持つ (miss を確定させる)
        cache_dir = _make_lmdb_cache(tmp_path, {"別のテキスト": [1, 2, 3]})

        ds = _make_dataset(lmdb_cache_dir=cache_dir)

        with patch("cc_g2pnp.data.dataset.generate_pnp_labels", return_value=["k", "o"]) as mock_gen:
            results = _collect(ds, [_make_sample("今日はいい天気")])

        # キャッシュミスなので generate_pnp_labels が呼ばれる
        mock_gen.assert_called_once()
        assert len(results) == 1

    def test_dataset_lmdb_cache_stats_logged(self, tmp_path, caplog):
        """イテレーション完了後に LMDB 統計がログ出力される。"""
        import logging

        from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
        from cc_g2pnp.data.vocabulary import PnPVocabulary

        text = "今日はいい天気"
        vocab = PnPVocabulary()
        pnp_ids = vocab.encode(generate_pnp_labels(text))
        cache_dir = _make_lmdb_cache(tmp_path, {text: pnp_ids})

        ds = _make_dataset(lmdb_cache_dir=cache_dir)
        with caplog.at_level(logging.INFO, logger="cc_g2pnp.data.dataset"):
            _collect(ds, [_make_sample(text)])

        assert any("LMDB cache" in record.message for record in caplog.records)


class TestBinaryLmdbCacheIntegration:
    """Tests for binary format LMDB cache integration."""

    def test_dataset_with_binary_lmdb_cache(self, tmp_path):
        """bytes 形式の LMDB キャッシュでデータセットが正常に読取できる。"""
        from cc_g2pnp.data.lmdb_cache import PnPLabelCache
        from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
        from cc_g2pnp.data.vocabulary import PnPVocabulary

        text = "今日はいい天気"
        vocab = PnPVocabulary()
        pnp_tokens = generate_pnp_labels(text)
        pnp_ids = vocab.encode(pnp_tokens)

        # PnPLabelCache.put() は新しい bytes 形式で書き込む
        cache_dir = str(tmp_path / "binary_cache")
        with PnPLabelCache(cache_dir, readonly=False) as cache:
            cache.put(text, pnp_ids)

        ds = _make_dataset(lmdb_cache_dir=cache_dir)
        with patch("cc_g2pnp.data.dataset.generate_pnp_labels") as mock_gen:
            results = _collect(ds, [_make_sample(text)])

        mock_gen.assert_not_called()
        assert len(results) == 1
        assert results[0]["labels"] == pnp_ids

    def test_dataset_with_legacy_json_lmdb_cache(self, tmp_path):
        """レガシー JSON 形式の LMDB キャッシュでも後方互換で読取できる。"""
        import json

        import lmdb

        from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
        from cc_g2pnp.data.vocabulary import PnPVocabulary

        text = "今日はいい天気"
        vocab = PnPVocabulary()
        pnp_tokens = generate_pnp_labels(text)
        pnp_ids = vocab.encode(pnp_tokens)

        # レガシー JSON 形式で直接書込
        cache_dir = str(tmp_path / "json_cache")
        env = lmdb.open(cache_dir, map_size=10 * 1024**3)
        with env.begin(write=True) as txn:
            txn.put(text.encode("utf-8"), json.dumps(pnp_ids).encode("utf-8"))
        env.close()

        ds = _make_dataset(lmdb_cache_dir=cache_dir)
        with patch("cc_g2pnp.data.dataset.generate_pnp_labels") as mock_gen:
            results = _collect(ds, [_make_sample(text)])

        mock_gen.assert_not_called()
        assert len(results) == 1
        assert results[0]["labels"] == pnp_ids
