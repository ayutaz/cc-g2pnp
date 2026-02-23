"""Validation evaluator for CC-G2PnP model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jiwer
import torch
from torch import nn

from cc_g2pnp.model.ctc_decoder import greedy_decode

if TYPE_CHECKING:
    from collections.abc import Iterable

    from cc_g2pnp.data.vocabulary import PnPVocabulary


def _compute_cer(
    predictions: list[list[int]],
    references: list[torch.Tensor],
    ref_lengths: torch.Tensor,
    vocabulary: PnPVocabulary,
) -> float:
    """Compute PnP Character Error Rate.

    Args:
        predictions: Predicted PnP label ID sequences (one per sample).
        references: Reference label tensors (one per sample).
        ref_lengths: Valid lengths of each reference tensor.
        vocabulary: PnP vocabulary for ID-to-token conversion.

    Returns:
        Average CER over all valid samples.
    """
    id_to_token = vocabulary.id_to_token
    blank_id = vocabulary.blank_id
    total_cer = 0.0
    count = 0

    for pred_ids, ref_tensor, ref_len in zip(predictions, references, ref_lengths, strict=True):
        ref_ids = ref_tensor[: ref_len.item()].tolist()
        ref_tokens = [id_to_token[i] for i in ref_ids if i != blank_id]
        pred_tokens = [id_to_token[i] for i in pred_ids if i != blank_id]

        if not ref_tokens:
            continue

        ref_str = " ".join(ref_tokens)
        pred_str = " ".join(pred_tokens)
        total_cer += jiwer.cer(ref_str, pred_str)
        count += 1

    return total_cer / max(count, 1)


class Evaluator:
    """Run validation and compute metrics."""

    def __init__(self, vocabulary: PnPVocabulary, device: torch.device) -> None:
        """Initialize evaluator.

        Args:
            vocabulary: PnP vocabulary for ID-to-token conversion.
            device: Device for evaluation tensors.
        """
        self.vocabulary = vocabulary
        self.device = device

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        val_batches: Iterable[dict],
        max_batches: int | None = None,
    ) -> dict[str, float]:
        """Run validation over batches and compute metrics.

        Args:
            model: CC_G2PnP model (automatically switched to eval mode).
            val_batches: Iterable of collator-output batches. Each batch has
                keys ``input_ids``, ``labels``, ``input_lengths``, ``label_lengths``.
            max_batches: Maximum number of batches to evaluate (None = all).

        Returns:
            Dictionary with ``val_loss``, ``val_cer``, and ``val_num_samples``.
        """
        model.eval()

        total_loss = 0.0
        total_cer = 0.0
        total_samples = 0

        raw_model = model.module if hasattr(model, "module") else model
        blank_id = raw_model.config.blank_id

        for num_batches, batch in enumerate(val_batches):
            if max_batches is not None and num_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            input_lengths = batch["input_lengths"].to(self.device)
            labels = batch["labels"].to(self.device)
            label_lengths = batch["label_lengths"].to(self.device)

            result = model(
                input_ids=input_ids,
                input_lengths=input_lengths,
                targets=labels,
                target_lengths=label_lengths,
            )

            batch_size = input_ids.size(0)
            total_loss += result["loss"].item() * batch_size

            predictions = greedy_decode(result["log_probs"], blank_id=blank_id)
            batch_cer = _compute_cer(
                predictions, labels, label_lengths, self.vocabulary,
            )
            total_cer += batch_cer * batch_size

            total_samples += batch_size

        model.train()

        if total_samples == 0:
            return {
                "val_loss": 0.0,
                "val_cer": 0.0,
                "val_num_samples": 0,
            }

        return {
            "val_loss": total_loss / total_samples,
            "val_cer": total_cer / total_samples,
            "val_num_samples": total_samples,
        }
