"""Data pipeline: ReazonSpeech → BPE tokens + PnP labels.

Loads the ReazonSpeech transcription text (via TSV from corpus server,
local Parquet, or HF streaming fallback), tokenizes each transcription
with the CALM2 BPE tokenizer, and generates PnP label sequences.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from datasets import load_dataset
from torch.utils.data import IterableDataset

from cc_g2pnp._patch_pyopenjtalk import apply as _patch_pyopenjtalk
from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
from cc_g2pnp.data.tokenizer import G2PnPTokenizer
from cc_g2pnp.data.vocabulary import PnPVocabulary

# DataLoader の子プロセスでも sudachipy キャッシュが有効になるようにパッチ適用
_patch_pyopenjtalk()

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

_REAZON_DATASET = "reazon-research/reazonspeech"
_TSV_BASE_URL = "https://corpus.reazon-research.org/reazonspeech-v2/tsv"
_CTC_UPSAMPLE_FACTOR = 8  # Token upsampling factor (BPE → Conformer input)


class G2PnPDataset(IterableDataset):
    """Streaming dataset that yields ``(input_ids, label_ids)`` pairs.

    Each sample is a dict with:
        ``input_ids``  : list[int] — BPE token IDs (model input)
        ``labels``     : list[int] — PnP vocabulary IDs (CTC target)
    """

    def __init__(
        self,
        subset: str = "all",
        streaming: bool = True,
        *,
        max_input_len: int = 512,
        min_input_len: int = 2,
        shuffle_seed: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        lmdb_cache_dir: str | None = None,
        local_dataset_dir: str | None = None,
    ) -> None:
        super().__init__()
        self._subset = subset
        self._streaming = streaming
        self._max_input_len = max_input_len
        self._min_input_len = min_input_len
        self._shuffle_seed = shuffle_seed
        self._rank = rank
        self._world_size = world_size
        self._local_dataset_dir = local_dataset_dir

        self._tokenizer = G2PnPTokenizer.get_instance()
        self._vocab = PnPVocabulary()
        # Set by DataLoader worker_init_fn for per-process OpenJTalk instance
        self.jtalk = None

        self._lmdb_cache = None
        if lmdb_cache_dir is not None:
            from cc_g2pnp.data.lmdb_cache import PnPLabelCache

            self._lmdb_cache = PnPLabelCache(lmdb_cache_dir, readonly=True)

    def _load_stream(self, rank: int = 0, world_size: int = 1) -> Iterable[dict]:
        """Load ReazonSpeech text data.

        データ取得の優先順位:
        1. local_dataset_dir 指定時 → ローカル Parquet
        2. デフォルト → TSV 直接ダウンロード (自動キャッシュ, ~76s for 'all')
        3. TSV 取得失敗時 → HF streaming フォールバック

        Args:
            rank: DDP プロセスランク (0-indexed)。
            world_size: DDP ワールドサイズ。world_size > 1 の場合、
                データを shard して各ランクが異なるサブセットを処理する。
        """
        if self._local_dataset_dir is not None:
            return self._load_local(rank, world_size)
        try:
            return self._load_tsv(rank, world_size)
        except Exception:
            logger.warning("TSV download failed, falling back to HF streaming", exc_info=True)
            return self._load_hf_streaming(rank, world_size)

    def _load_tsv(self, rank: int = 0, world_size: int = 1) -> Iterable[dict]:
        """Download TSV from corpus.reazon-research.org and load as Arrow dataset.

        TSV にはテキスト (name + transcription) のみ含まれ、音声データなし。
        'all' サブセットで ~2GB / ~76s。~/.cache/cc_g2pnp/tsv/ に自動キャッシュ。
        """
        from pathlib import Path

        import requests

        cache_dir = Path.home() / ".cache" / "cc_g2pnp" / "tsv"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tsv_path = cache_dir / f"{self._subset}.tsv"

        if not tsv_path.exists():
            url = f"{_TSV_BASE_URL}/{self._subset}.tsv"
            logger.info("Downloading ReazonSpeech TSV: %s", url)
            resp = requests.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            tmp_path = tsv_path.with_suffix(".tmp")
            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
            tmp_path.rename(tsv_path)
            logger.info("TSV cached: %s (%.1f MB)", tsv_path, (total or downloaded) / 1e6)
        else:
            logger.info("Using cached TSV: %s", tsv_path)

        ds = load_dataset(
            "csv", data_files=str(tsv_path), split="train",
            delimiter="\t", column_names=["name", "transcription"],
        )
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        if self._shuffle_seed is not None:
            ds = ds.shuffle(seed=self._shuffle_seed)
        return ds

    def _load_hf_streaming(self, rank: int = 0, world_size: int = 1) -> Iterable[dict]:
        """Fallback: HuggingFace streaming (slow — downloads audio TARs)."""
        ds = load_dataset(
            _REAZON_DATASET,
            self._subset,
            split="train",
            streaming=True,
            trust_remote_code=True,
        ).select_columns(["name", "transcription"])
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        if self._shuffle_seed is not None:
            ds = ds.shuffle(seed=self._shuffle_seed, buffer_size=10_000)
        return ds

    def _load_local(self, rank: int = 0, world_size: int = 1) -> Iterable[dict]:
        """Load dataset from local Parquet files (created by scripts/download_text.py).

        Arrow memory-mapped 読み込みでランダムアクセス可能。
        ストリーミングバッファ制限なしの完全シャッフルが可能。
        """
        import glob as glob_mod
        from pathlib import Path

        parquet_dir = Path(self._local_dataset_dir) / self._subset
        pattern = str(parquet_dir / "*.parquet")
        data_files = sorted(glob_mod.glob(pattern))
        if not data_files:
            msg = f"No parquet files found in {parquet_dir}"
            raise FileNotFoundError(msg)

        logger.info("Loading local Parquet dataset from %s (%d files)", parquet_dir, len(data_files))
        ds = load_dataset("parquet", data_files=data_files, split="train")

        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        if self._shuffle_seed is not None:
            ds = ds.shuffle(seed=self._shuffle_seed)
        return ds

    def close(self) -> None:
        """LMDB キャッシュをクローズしてリソースを解放する。"""
        if self._lmdb_cache is not None:
            self._lmdb_cache.close()
            self._lmdb_cache = None

    def __del__(self) -> None:
        self.close()

    def __iter__(self) -> Iterator[dict]:
        total = 0
        skipped_empty = 0
        skipped_bpe = 0
        skipped_length = 0
        skipped_pnp = 0
        skipped_ctc = 0
        yielded = 0
        cache_hits = 0
        cache_misses = 0

        # DataLoader マルチワーカー対応: ワーカーごとにシャードを分割して重複を防ぐ
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            effective_rank = self._rank * worker_info.num_workers + worker_info.id
            effective_world_size = self._world_size * worker_info.num_workers
        else:
            effective_rank = self._rank
            effective_world_size = self._world_size

        for sample in self._load_stream(rank=effective_rank, world_size=effective_world_size):
            total += 1
            text = sample.get("transcription", "")
            if not text or not text.strip():
                skipped_empty += 1
                continue

            # BPE tokenize
            bpe_ids = self._tokenizer.encode(text)
            if not bpe_ids:
                skipped_bpe += 1
                continue

            # Length filter
            if len(bpe_ids) < self._min_input_len or len(bpe_ids) > self._max_input_len:
                skipped_length += 1
                continue

            # PnP label: try cache first, then generate on-the-fly
            pnp_ids = None
            if self._lmdb_cache is not None:
                pnp_ids = self._lmdb_cache.get(text)
                if pnp_ids is not None:
                    cache_hits += 1
                else:
                    cache_misses += 1

            if pnp_ids is None:
                pnp_tokens = generate_pnp_labels(text, jtalk=self.jtalk)
                if not pnp_tokens:
                    skipped_pnp += 1
                    continue
                pnp_ids = self._vocab.encode(pnp_tokens)

            # CTC constraint: input_length * upsample_factor >= target_length
            if len(bpe_ids) * _CTC_UPSAMPLE_FACTOR < len(pnp_ids):
                logger.debug(
                    "CTC constraint violation: %d * %d < %d (text=%r)",
                    len(bpe_ids), _CTC_UPSAMPLE_FACTOR, len(pnp_ids), text[:40],
                )
                skipped_ctc += 1
                continue

            yielded += 1
            if yielded % 10000 == 0:
                logger.info(
                    "Progress: yielded=%d, total=%d, "
                    "skipped(empty=%d, bpe=%d, length=%d, pnp=%d, ctc=%d)",
                    yielded, total,
                    skipped_empty, skipped_bpe, skipped_length,
                    skipped_pnp, skipped_ctc,
                )

            yield {"input_ids": bpe_ids, "labels": pnp_ids}

        logger.info(
            "Dataset complete: yielded=%d/%d (%.1f%%), "
            "skipped(empty=%d, bpe=%d, length=%d, pnp=%d, ctc=%d)",
            yielded, total, yielded / max(total, 1) * 100,
            skipped_empty, skipped_bpe, skipped_length,
            skipped_pnp, skipped_ctc,
        )

        if self._lmdb_cache is not None:
            logger.info(
                "LMDB cache: hits=%d, misses=%d, rate=%.1f%%",
                cache_hits, cache_misses,
                cache_hits / max(cache_hits + cache_misses, 1) * 100,
            )
