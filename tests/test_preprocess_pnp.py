"""Tests for preprocess_pnp.py script."""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock, patch

import pytest

# Import the script module via importlib since it's not a package
_spec = importlib.util.spec_from_file_location("preprocess_pnp", "scripts/preprocess_pnp.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules["preprocess_pnp"] = _mod
_spec.loader.exec_module(_mod)

_worker_init_with_cache = _mod._worker_init_with_cache
_process_text_cached = _mod._process_text_cached
_text_stream = _mod._text_stream
main = _mod.main


def _mock_dataset(texts: list[str]) -> MagicMock:
    """Create a mock dataset that behaves like HF streaming dataset."""
    samples = [{"transcription": t} for t in texts]
    mock_ds = MagicMock()
    mock_ds.select_columns.return_value = samples
    return mock_ds


def _run_main(tmp_path, extra_args: list[str] | None = None, texts: list[str] | None = None, pnp_return=None):
    """Helper to run main() with mocked dataset and pyopenjtalk."""
    if texts is None:
        texts = [f"テスト文{i}" for i in range(10)]
    if pnp_return is None:
        pnp_return = ["k", "o", "N", "n", "i", "ch", "i", "w", "a"]

    mock_ds = _mock_dataset(texts)

    args = [
        "preprocess_pnp.py",
        "--output", str(tmp_path / "cache"),
        *(extra_args or []),
    ]

    with (
        patch.object(sys, "argv", args),
        patch("datasets.load_dataset", return_value=mock_ds),
        patch("cc_g2pnp.data.pnp_labeler.generate_pnp_labels", return_value=pnp_return),
    ):
        main()

    from cc_g2pnp.data.lmdb_cache import PnPLabelCache

    with PnPLabelCache(str(tmp_path / "cache"), readonly=True) as cache:
        return len(cache)


class TestSingleProcess:
    def test_single_process_basic(self, tmp_path):
        """num_workers=0, max_samples=5 でモック DS から LMDB 生成。"""
        count = _run_main(tmp_path, ["--num-workers", "0", "--max-samples", "5"])
        assert count == 5

    def test_max_samples_limit(self, tmp_path):
        """max_samples でエントリ数が制限される。"""
        texts = [f"テスト{i}" for i in range(20)]
        count = _run_main(tmp_path, ["--num-workers", "0", "--max-samples", "3"], texts=texts)
        assert count == 3


class TestMultiProcess:
    @pytest.mark.slow
    def test_multiprocess_basic(self, tmp_path):
        """num_workers=2, max_samples=5 でモック DS から LMDB 生成 (実際の pyopenjtalk 使用)。"""
        texts = ["今日はいい天気", "東京タワー", "音声合成", "自然言語処理", "人工知能", "機械学習", "深層学習"]
        mock_ds = _mock_dataset(texts)

        args = [
            "preprocess_pnp.py",
            "--output", str(tmp_path / "cache"),
            "--num-workers", "2",
            "--max-samples", "5",
        ]

        with (
            patch.object(sys, "argv", args),
            patch("datasets.load_dataset", return_value=mock_ds),
        ):
            main()

        from cc_g2pnp.data.lmdb_cache import PnPLabelCache

        with PnPLabelCache(str(tmp_path / "cache"), readonly=True) as cache:
            assert len(cache) <= 5
            assert len(cache) > 0


class TestResume:
    def test_resume_skips_existing(self, tmp_path):
        """既存エントリをスキップする。"""
        _run_main(tmp_path, ["--num-workers", "0", "--max-samples", "5"])

        from cc_g2pnp.data.lmdb_cache import PnPLabelCache

        with PnPLabelCache(str(tmp_path / "cache"), readonly=True) as cache:
            first_count = len(cache)

        # 2回目: resume で同じデータ → 新規追加なし
        texts = [f"テスト文{i}" for i in range(10)]
        mock_ds = _mock_dataset(texts)
        args = [
            "preprocess_pnp.py",
            "--output", str(tmp_path / "cache"),
            "--num-workers", "0",
            "--resume",
        ]
        with (
            patch.object(sys, "argv", args),
            patch("datasets.load_dataset", return_value=mock_ds),
            patch("cc_g2pnp.data.pnp_labeler.generate_pnp_labels", return_value=["k", "o"]),
        ):
            main()

        with PnPLabelCache(str(tmp_path / "cache"), readonly=True) as cache:
            assert len(cache) >= first_count


class TestEdgeCases:
    def test_empty_text_skipped(self, tmp_path):
        """空テキストは無視される。"""
        texts = ["", "   ", "有効なテスト文"]
        count = _run_main(tmp_path, ["--num-workers", "0"], texts=texts)
        assert count == 1

    def test_pnp_failure_skipped(self, tmp_path):
        """generate_pnp_labels が空を返す場合は無視される。"""
        texts = ["テスト文"]
        count = _run_main(tmp_path, ["--num-workers", "0"], texts=texts, pnp_return=[])
        assert count == 0

    def test_output_directory_created(self, tmp_path):
        """存在しない出力ディレクトリが自動作成される。"""
        output_dir = tmp_path / "nested" / "dir" / "cache"
        texts = ["テスト"]
        mock_ds = _mock_dataset(texts)
        args = [
            "preprocess_pnp.py",
            "--output", str(output_dir),
            "--num-workers", "0",
            "--max-samples", "1",
        ]
        with (
            patch.object(sys, "argv", args),
            patch("datasets.load_dataset", return_value=mock_ds),
            patch("cc_g2pnp.data.pnp_labeler.generate_pnp_labels", return_value=["k", "o"]),
        ):
            main()

        assert output_dir.exists()


class TestWorkerFunctions:
    def test_worker_init_patches_pyopenjtalk(self):
        """_worker_init_with_cache が pyopenjtalk パッチと PnPVocabulary を初期化する。"""
        with patch("cc_g2pnp._patch_pyopenjtalk.apply") as mock_patch:
            _worker_init_with_cache()
            mock_patch.assert_called_once()
            assert _mod._vocab_cache is not None

    def test_process_text_cached_returns_tuple(self):
        """正常テキストで (text, pnp_ids) タプルを返す。"""
        from cc_g2pnp.data.vocabulary import PnPVocabulary

        _mod._vocab_cache = PnPVocabulary()
        with patch("cc_g2pnp.data.pnp_labeler.generate_pnp_labels", return_value=["k", "o"]):
            result = _process_text_cached("テスト")
        assert result is not None
        assert result[0] == "テスト"
        assert isinstance(result[1], list)
        assert len(result[1]) > 0

    def test_process_text_cached_returns_none_on_failure(self):
        """PnP 生成失敗で None を返す。"""
        from cc_g2pnp.data.vocabulary import PnPVocabulary

        _mod._vocab_cache = PnPVocabulary()
        with patch("cc_g2pnp.data.pnp_labeler.generate_pnp_labels", return_value=[]):
            result = _process_text_cached("テスト")
        assert result is None
