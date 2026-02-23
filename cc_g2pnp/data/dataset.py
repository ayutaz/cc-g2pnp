"""Streaming data pipeline: ReazonSpeech → BPE tokens + PnP labels.

Loads the ReazonSpeech dataset in streaming mode (text only, no audio
decoding), tokenizes each transcription with the CALM2 BPE tokenizer, and
generates PnP (Pronunciation & Prosody) label sequences.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from datasets import load_dataset
from pyopenjtalk import OPEN_JTALK_DICT_DIR
from pyopenjtalk.openjtalk import OpenJTalk
from torch.utils.data import IterableDataset

from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
from cc_g2pnp.data.tokenizer import G2PnPTokenizer
from cc_g2pnp.data.vocabulary import PnPVocabulary

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

logger = logging.getLogger(__name__)

_REAZON_DATASET = "reazon-research/reazonspeech"
_CTC_UPSAMPLE_FACTOR = 8  # Token upsampling factor (BPE → Conformer input)
_NUM_LABELER_WORKERS = 4
_LABELER_CHUNK_SIZE = 64


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
        num_labeler_workers: int = _NUM_LABELER_WORKERS,
    ) -> None:
        super().__init__()
        self._subset = subset
        self._streaming = streaming
        self._max_input_len = max_input_len
        self._min_input_len = min_input_len
        self._shuffle_seed = shuffle_seed
        self._num_labeler_workers = num_labeler_workers

        self._tokenizer = G2PnPTokenizer.get_instance()
        self._vocab = PnPVocabulary()

    def _load_stream(self) -> Iterable[dict]:
        """Load ReazonSpeech text-only stream."""
        ds = load_dataset(
            _REAZON_DATASET,
            self._subset,
            split="train",
            streaming=self._streaming,
            trust_remote_code=True,
        ).select_columns(["name", "transcription"])
        if self._shuffle_seed is not None and self._streaming:
            ds = ds.shuffle(seed=self._shuffle_seed, buffer_size=10_000)
        return ds

    def _process_chunk(
        self,
        chunk: list[tuple[str, list[int]]],
        executor: ThreadPoolExecutor,
        process_fn: Callable[[str], list[str]],
    ) -> list[dict | None]:
        """Process a chunk of (text, bpe_ids) with parallel PnP labeling.

        Returns a list aligned with *chunk*.  Each element is either a
        valid sample dict or ``None`` (filtered out by PnP-empty / CTC).
        """
        texts = [text for text, _ in chunk]
        pnp_results = list(executor.map(process_fn, texts))

        out: list[dict | None] = []
        for (text, bpe_ids), pnp_tokens in zip(chunk, pnp_results, strict=True):
            if not pnp_tokens:
                out.append(None)
                continue

            pnp_ids = self._vocab.encode(pnp_tokens)

            if len(bpe_ids) * _CTC_UPSAMPLE_FACTOR < len(pnp_ids):
                logger.debug(
                    "CTC constraint violation: %d * %d < %d (text=%r)",
                    len(bpe_ids), _CTC_UPSAMPLE_FACTOR, len(pnp_ids), text[:40],
                )
                out.append(None)
                continue

            out.append({"input_ids": bpe_ids, "labels": pnp_ids})
        return out

    def __iter__(self) -> Iterator[dict]:
        total = 0
        skipped_empty = 0
        skipped_bpe = 0
        skipped_length = 0
        skipped_pnp_ctc = 0
        yielded = 0

        # Per-thread OpenJTalk instances to avoid global Lock contention.
        _jtalk_local = threading.local()

        def _get_jtalk() -> OpenJTalk:
            if not hasattr(_jtalk_local, "instance"):
                _jtalk_local.instance = OpenJTalk(dn_mecab=OPEN_JTALK_DICT_DIR)
            return _jtalk_local.instance

        def _process_text(text: str) -> list[str]:
            return generate_pnp_labels(text, jtalk=_get_jtalk())

        with ThreadPoolExecutor(max_workers=self._num_labeler_workers) as executor:
            chunk: list[tuple[str, list[int]]] = []

            for sample in self._load_stream():
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

                chunk.append((text, bpe_ids))

                if len(chunk) >= _LABELER_CHUNK_SIZE:
                    results = self._process_chunk(chunk, executor, _process_text)
                    for r in results:
                        if r is None:
                            skipped_pnp_ctc += 1
                            continue
                        yielded += 1
                        if yielded % 10000 == 0:
                            logger.info(
                                "Progress: yielded=%d, total=%d, "
                                "skipped(empty=%d, bpe=%d, length=%d, pnp_ctc=%d)",
                                yielded, total,
                                skipped_empty, skipped_bpe, skipped_length,
                                skipped_pnp_ctc,
                            )
                        yield r
                    chunk = []

            # Process remaining chunk
            if chunk:
                results = self._process_chunk(chunk, executor, _process_text)
                for r in results:
                    if r is None:
                        skipped_pnp_ctc += 1
                        continue
                    yielded += 1
                    if yielded % 10000 == 0:
                        logger.info(
                            "Progress: yielded=%d, total=%d, "
                            "skipped(empty=%d, bpe=%d, length=%d, pnp_ctc=%d)",
                            yielded, total,
                            skipped_empty, skipped_bpe, skipped_length,
                            skipped_pnp_ctc,
                        )
                    yield r

        logger.info(
            "Dataset complete: yielded=%d/%d (%.1f%%), "
            "skipped(empty=%d, bpe=%d, length=%d, pnp_ctc=%d)",
            yielded, total, yielded / max(total, 1) * 100,
            skipped_empty, skipped_bpe, skipped_length,
            skipped_pnp_ctc,
        )
