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

from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
from cc_g2pnp.data.tokenizer import G2PnPTokenizer
from cc_g2pnp.data.vocabulary import PnPVocabulary

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)

_REAZON_DATASET = "reazon-research/reazonspeech"
_CTC_UPSAMPLE_FACTOR = 8  # Conformer subsampling ratio


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
        min_input_len: int = 1,
    ) -> None:
        super().__init__()
        self._subset = subset
        self._streaming = streaming
        self._max_input_len = max_input_len
        self._min_input_len = min_input_len

        self._tokenizer = G2PnPTokenizer()
        self._vocab = PnPVocabulary()

    def _load_stream(self):
        """Load ReazonSpeech text-only stream."""
        return load_dataset(
            _REAZON_DATASET,
            self._subset,
            split="train",
            streaming=self._streaming,
            trust_remote_code=True,
        ).select_columns(["name", "transcription"])

    def __iter__(self) -> Iterator[dict]:
        ds = self._load_stream()
        for sample in ds:
            text = sample.get("transcription", "")
            if not text or not text.strip():
                continue

            # BPE tokenize
            bpe_ids = self._tokenizer.encode(text)
            if not bpe_ids:
                continue

            # Length filter
            if len(bpe_ids) < self._min_input_len or len(bpe_ids) > self._max_input_len:
                continue

            # PnP label generation
            pnp_tokens = generate_pnp_labels(text)
            if not pnp_tokens:
                continue

            pnp_ids = self._vocab.encode(pnp_tokens)

            # CTC constraint: input_length * upsample_factor >= target_length
            if len(bpe_ids) * _CTC_UPSAMPLE_FACTOR < len(pnp_ids):
                logger.debug(
                    "CTC constraint violation: %d * %d < %d (text=%r)",
                    len(bpe_ids), _CTC_UPSAMPLE_FACTOR, len(pnp_ids), text[:40],
                )
                continue

            yield {
                "input_ids": bpe_ids,
                "labels": pnp_ids,
            }
