"""Dynamic batching utilities for CTC training.

Provides:
- ``DynamicBatchCollator``: pads variable-length ``input_ids`` / ``labels``
  within a batch and records original lengths for CTC loss.
- ``dynamic_batch_sampler``: groups samples from an iterable dataset so that
  total BPE tokens per batch stays below ``max_tokens``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


class DynamicBatchCollator:
    """Pad a batch of ``{"input_ids": ..., "labels": ...}`` dicts.

    Returns a dict of tensors:
        ``input_ids``      : (B, T_max) — padded BPE token IDs
        ``labels``         : (B, U_max) — padded PnP label IDs
        ``input_lengths``  : (B,)       — original input lengths
        ``label_lengths``  : (B,)       — original label lengths
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_id: int = -100,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids_list = [sample["input_ids"] for sample in batch]
        labels_list = [sample["labels"] for sample in batch]

        input_lengths = torch.tensor([len(ids) for ids in input_ids_list], dtype=torch.long)
        label_lengths = torch.tensor([len(lbl) for lbl in labels_list], dtype=torch.long)

        max_input = int(input_lengths.max())
        max_label = int(label_lengths.max())

        padded_inputs = torch.full((len(batch), max_input), self.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((len(batch), max_label), self.label_pad_id, dtype=torch.long)

        for i, (ids, lbls) in enumerate(zip(input_ids_list, labels_list, strict=True)):
            padded_inputs[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            padded_labels[i, : len(lbls)] = torch.tensor(lbls, dtype=torch.long)

        return {
            "input_ids": padded_inputs,
            "labels": padded_labels,
            "input_lengths": input_lengths,
            "label_lengths": label_lengths,
        }


def dynamic_batch_sampler(
    dataset: Iterable[dict],
    max_tokens: int = 8192,
) -> Iterator[list[dict]]:
    """Group samples so that total BPE tokens per batch ≤ *max_tokens*.

    Yields lists of samples (each a dict with ``input_ids``).
    Works with streaming (iterable) datasets — never materialises the full
    dataset in memory.
    """
    bucket: list[dict] = []
    bucket_tokens = 0

    for sample in dataset:
        sample_len = len(sample["input_ids"])

        # Single sample exceeding budget → yield it alone
        if sample_len > max_tokens:
            if bucket:
                yield bucket
                bucket = []
                bucket_tokens = 0
            yield [sample]
            continue

        # Would exceed budget → flush current bucket first
        if bucket_tokens + sample_len > max_tokens:
            yield bucket
            bucket = []
            bucket_tokens = 0

        bucket.append(sample)
        bucket_tokens += sample_len

    # Flush remaining samples
    if bucket:
        yield bucket
