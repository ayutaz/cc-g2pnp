"""Tests for local Parquet and TSV dataset loading."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

if TYPE_CHECKING:
    from pathlib import Path


def _create_parquet_dataset(base_dir: Path, subset: str, num_rows: int = 20) -> Path:
    """Helper: create a small Parquet dataset for testing."""
    subset_dir = base_dir / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    names = [f"sample_{i:04d}" for i in range(num_rows)]
    texts = [f"テスト文{i}" for i in range(num_rows)]

    table = pa.table(
        {"name": names, "transcription": texts},
        schema=pa.schema([("name", pa.string()), ("transcription", pa.string())]),
    )
    pq.write_table(table, subset_dir / "shard_000000.parquet")
    return base_dir


class TestLocalDatasetLoading:
    """Test G2PnPDataset with local Parquet files."""

    def test_load_local_parquet(self, tmp_path):
        """Local Parquet files are loaded and samples are yielded."""
        from cc_g2pnp.data.dataset import G2PnPDataset

        dataset_dir = _create_parquet_dataset(tmp_path, "small")

        ds = G2PnPDataset(
            subset="small",
            streaming=False,
            max_input_len=512,
            local_dataset_dir=str(dataset_dir),
        )

        samples = list(ds)
        assert len(samples) > 0
        for s in samples:
            assert "input_ids" in s
            assert "labels" in s
            assert len(s["input_ids"]) > 0
            assert len(s["labels"]) > 0

    def test_load_local_parquet_multiple_shards(self, tmp_path):
        """Multiple Parquet shards are loaded correctly."""
        from cc_g2pnp.data.dataset import G2PnPDataset

        subset_dir = tmp_path / "small"
        subset_dir.mkdir(parents=True)

        for shard_idx in range(3):
            names = [f"shard{shard_idx}_sample_{i}" for i in range(5)]
            texts = [f"テスト文{shard_idx}_{i}" for i in range(5)]
            table = pa.table(
                {"name": names, "transcription": texts},
                schema=pa.schema([("name", pa.string()), ("transcription", pa.string())]),
            )
            pq.write_table(table, subset_dir / f"shard_{shard_idx:06d}.parquet")

        ds = G2PnPDataset(
            subset="small",
            streaming=False,
            max_input_len=512,
            local_dataset_dir=str(tmp_path),
        )

        samples = list(ds)
        # All 15 samples should produce at least some output
        assert len(samples) > 0

    def test_ddp_sharding(self, tmp_path):
        """DDP sharding splits data between ranks."""
        from cc_g2pnp.data.dataset import G2PnPDataset

        dataset_dir = _create_parquet_dataset(tmp_path, "small", num_rows=40)

        ds_rank0 = G2PnPDataset(
            subset="small",
            streaming=False,
            max_input_len=512,
            rank=0,
            world_size=2,
            local_dataset_dir=str(dataset_dir),
        )
        ds_rank1 = G2PnPDataset(
            subset="small",
            streaming=False,
            max_input_len=512,
            rank=1,
            world_size=2,
            local_dataset_dir=str(dataset_dir),
        )

        samples_0 = list(ds_rank0)
        samples_1 = list(ds_rank1)

        # Both ranks should produce samples
        assert len(samples_0) > 0
        assert len(samples_1) > 0

    def test_missing_parquet_dir_raises(self, tmp_path):
        """FileNotFoundError is raised when no parquet files exist."""
        from cc_g2pnp.data.dataset import G2PnPDataset

        ds = G2PnPDataset(
            subset="small",
            streaming=False,
            max_input_len=512,
            local_dataset_dir=str(tmp_path),
        )
        with pytest.raises(FileNotFoundError, match="No parquet files found"):
            list(ds)

    def test_local_dataset_dir_none_uses_streaming(self):
        """When local_dataset_dir is None, streaming behavior is used (backward compat)."""
        from cc_g2pnp.data.dataset import G2PnPDataset

        ds = G2PnPDataset(
            subset="small",
            streaming=True,
            max_input_len=512,
            local_dataset_dir=None,
        )
        # Should not call _load_local
        assert ds._local_dataset_dir is None

    def test_shuffle_with_seed(self, tmp_path):
        """Shuffle with seed produces consistent ordering."""
        from cc_g2pnp.data.dataset import G2PnPDataset

        dataset_dir = _create_parquet_dataset(tmp_path, "small", num_rows=30)

        def _get_samples(seed):
            ds = G2PnPDataset(
                subset="small",
                streaming=False,
                max_input_len=512,
                shuffle_seed=seed,
                local_dataset_dir=str(dataset_dir),
            )
            return list(ds)

        # Same seed -> same order
        samples_a = _get_samples(42)
        samples_b = _get_samples(42)
        assert len(samples_a) > 0
        assert len(samples_a) == len(samples_b)
        for a, b in zip(samples_a, samples_b, strict=True):
            assert a["input_ids"] == b["input_ids"]
            assert a["labels"] == b["labels"]


class TestTSVLoading:
    """Test G2PnPDataset TSV auto-download mode."""

    def test_tsv_loading_with_cached_file(self, tmp_path):
        """TSV mode loads from a cached TSV file without network."""
        from cc_g2pnp.data.dataset import G2PnPDataset

        # Create a fake cached TSV
        tsv_dir = tmp_path / "tsv"
        tsv_dir.mkdir()
        tsv_file = tsv_dir / "small.tsv"
        lines = [f"sample_{i:04d}.flac\tテスト文{i}" for i in range(20)]
        tsv_file.write_text("\n".join(lines) + "\n")

        ds = G2PnPDataset(subset="small", max_input_len=512)

        # Patch cache dir to use our temp dir
        with patch("cc_g2pnp.data.dataset._TSV_BASE_URL", "http://invalid"):
            # Monkey-patch _load_tsv to use our cache dir
            original_load_tsv = ds._load_tsv

            def patched_load_tsv(rank=0, world_size=1):
                from pathlib import Path

                # Temporarily override cache dir
                old_home = Path.home

                def fake_home():
                    return tmp_path

                Path.home = staticmethod(fake_home)
                # Pre-create the expected cache structure
                cache_dir = tmp_path / ".cache" / "cc_g2pnp" / "tsv"
                cache_dir.mkdir(parents=True, exist_ok=True)
                import shutil

                shutil.copy(tsv_file, cache_dir / "small.tsv")
                try:
                    return original_load_tsv(rank, world_size)
                finally:
                    Path.home = old_home

            ds._load_tsv = patched_load_tsv

            samples = list(ds)
            assert len(samples) > 0
            for s in samples:
                assert "input_ids" in s
                assert "labels" in s

    @pytest.mark.network
    def test_tsv_download_small(self):
        """TSV download works for small subset (network required)."""
        from cc_g2pnp.data.dataset import G2PnPDataset

        ds = G2PnPDataset(subset="small", max_input_len=512)
        # Just check first few samples
        count = 0
        for s in ds:
            assert "input_ids" in s
            assert "labels" in s
            count += 1
            if count >= 5:
                break
        assert count == 5

    def test_tsv_fallback_on_failure(self):
        """When TSV download fails, falls back to HF streaming."""
        from cc_g2pnp.data.dataset import G2PnPDataset

        ds = G2PnPDataset(subset="small", max_input_len=512)

        # _load_tsv raises → should call _load_hf_streaming
        with (
            patch.object(ds, "_load_tsv", side_effect=ConnectionError("test")),
            patch.object(ds, "_load_hf_streaming", return_value=iter([])) as mock_hf,
        ):
            ds._load_stream(rank=0, world_size=1)
            mock_hf.assert_called_once_with(0, 1)


class TestTrainingConfigLocalDataset:
    """Test TrainingConfig with local_dataset_dir field."""

    def test_config_default_none(self):
        from cc_g2pnp.training.config import TrainingConfig

        config = TrainingConfig()
        assert config.local_dataset_dir is None

    def test_config_with_local_dir(self):
        from cc_g2pnp.training.config import TrainingConfig

        config = TrainingConfig(local_dataset_dir="/data/reazonspeech_text")
        assert config.local_dataset_dir == "/data/reazonspeech_text"
