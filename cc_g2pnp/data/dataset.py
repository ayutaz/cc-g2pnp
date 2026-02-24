"""Streaming data pipeline: ReazonSpeech → BPE tokens + PnP labels.

Loads the ReazonSpeech dataset in streaming mode (text only, no audio
decoding), tokenizes each transcription with the CALM2 BPE tokenizer, and
generates PnP (Pronunciation & Prosody) label sequences.
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
    ) -> None:
        super().__init__()
        self._subset = subset
        self._streaming = streaming
        self._max_input_len = max_input_len
        self._min_input_len = min_input_len
        self._shuffle_seed = shuffle_seed
        self._rank = rank
        self._world_size = world_size

        self._tokenizer = G2PnPTokenizer.get_instance()
        self._vocab = PnPVocabulary()
        # Set by DataLoader worker_init_fn for per-process OpenJTalk instance
        self.jtalk = None

    def _load_stream(self, rank: int = 0, world_size: int = 1) -> Iterable[dict]:
        """Load ReazonSpeech text-only stream.

        Args:
            rank: DDP プロセスランク (0-indexed)。
            world_size: DDP ワールドサイズ。world_size > 1 の場合、
                データを shard して各ランクが異なるサブセットを処理する。
        """
        ds = load_dataset(
            _REAZON_DATASET,
            self._subset,
            split="train",
            streaming=self._streaming,
            trust_remote_code=True,
        ).select_columns(["name", "transcription"])
        if world_size > 1:
            ds = ds.shard(num_shards=world_size, index=rank)
        if self._shuffle_seed is not None and self._streaming:
            ds = ds.shuffle(seed=self._shuffle_seed, buffer_size=10_000)
        return ds

    def __iter__(self) -> Iterator[dict]:
        total = 0
        skipped_empty = 0
        skipped_bpe = 0
        skipped_length = 0
        skipped_pnp = 0
        skipped_ctc = 0
        yielded = 0

        for sample in self._load_stream(rank=self._rank, world_size=self._world_size):
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

            # PnP label generation
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
