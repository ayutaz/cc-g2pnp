"""End-to-end evaluation pipeline for CC-G2PnP models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from cc_g2pnp.data.vocabulary import PnPVocabulary
from cc_g2pnp.evaluation.metrics import evaluate_all

if TYPE_CHECKING:
    from pathlib import Path

    from cc_g2pnp.evaluation.eval_data import EvalDataset, EvalSample
    from cc_g2pnp.model.cc_g2pnp import CC_G2PnP

logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    batch_size: int = 32
    device: str = "cpu"
    use_streaming: bool = False
    max_samples: int | None = None


@dataclass
class EvalResult:
    """Results of evaluation."""

    metrics: dict[str, float] = field(default_factory=dict)
    """Overall metrics across all samples."""

    per_domain_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    """Per-domain breakdown of metrics."""

    num_samples: int = 0
    num_domains: int = 0


class EvaluationPipeline:
    """End-to-end evaluation pipeline for CC-G2PnP models."""

    def __init__(
        self,
        model: CC_G2PnP,
        vocabulary: PnPVocabulary,
        config: EvalConfig | None = None,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self.config = config or EvalConfig()
        self._device = torch.device(self.config.device)
        self.model.to(self._device)
        self.model.eval()

    def _decode_ids_to_tokens(self, id_sequences: list[list[int]]) -> list[list[str]]:
        """Convert predicted ID sequences to PnP token sequences.

        Filters out <blank> and <pad> tokens.
        """
        blank_id = self.vocabulary.blank_id
        pad_id = self.vocabulary.pad_id
        results = []
        for ids in id_sequences:
            tokens = self.vocabulary.decode(
                [i for i in ids if i != blank_id and i != pad_id]
            )
            # Also filter out any <blank>/<pad> string tokens
            tokens = [t for t in tokens if t not in ("<blank>", "<pad>")]
            results.append(tokens)
        return results

    def _run_batch_inference(
        self,
        samples: list[EvalSample],
    ) -> list[list[str]]:
        """Run model inference on a batch of samples.

        Pads BPE IDs to max length, runs model.inference(), decodes results.
        """
        bpe_ids_list = [s.bpe_ids for s in samples]
        max_len = max(len(ids) for ids in bpe_ids_list)

        # Pad sequences
        padded = []
        lengths = []
        for ids in bpe_ids_list:
            padded.append(ids + [0] * (max_len - len(ids)))
            lengths.append(len(ids))

        input_ids = torch.tensor(padded, dtype=torch.long, device=self._device)
        input_lengths = torch.tensor(lengths, dtype=torch.long, device=self._device)

        # Run inference
        with torch.no_grad():
            predicted_ids = self.model.inference(input_ids, input_lengths)

        return self._decode_ids_to_tokens(predicted_ids)

    def _run_streaming_inference(
        self,
        samples: list[EvalSample],
    ) -> list[list[str]]:
        """Run streaming inference on samples (one at a time)."""
        from cc_g2pnp.inference.streaming import StreamingInference

        engine = StreamingInference(self.model)
        all_predictions: list[list[str]] = []

        for sample in samples:
            state = engine.reset(batch_size=1)
            bpe_ids = torch.tensor(
                [sample.bpe_ids],
                dtype=torch.long,
                device=self._device,
            )
            labels, state = engine.process_tokens(bpe_ids, state)
            flush_labels, _state = engine.flush(state)

            # Combine labels from process_tokens and flush
            combined_ids = labels[0] + flush_labels[0]
            tokens = self._decode_ids_to_tokens([combined_ids])
            all_predictions.append(tokens[0])

        return all_predictions

    def evaluate(self, dataset: EvalDataset) -> EvalResult:
        """Run evaluation on dataset.

        1. Run model inference on all samples (batched)
        2. Decode predictions to PnP token sequences
        3. Compute all 6 metrics (overall + per-domain)
        """
        samples = list(dataset.samples)
        if self.config.max_samples is not None:
            samples = samples[: self.config.max_samples]

        if not samples:
            return EvalResult()

        # Run inference in batches
        all_predictions: list[list[str]] = []
        batch_size = self.config.batch_size

        inference_fn = (
            self._run_streaming_inference
            if self.config.use_streaming
            else self._run_batch_inference
        )

        for i in range(0, len(samples), batch_size):
            batch = samples[i : i + batch_size]
            predictions = inference_fn(batch)
            all_predictions.extend(predictions)

        # Collect references
        all_references = [s.pnp_labels for s in samples]

        # Compute overall metrics
        overall_metrics = evaluate_all(all_predictions, all_references)

        # Compute per-domain metrics
        per_domain: dict[str, dict[str, float]] = {}
        domains = sorted({s.domain for s in samples})
        for domain in domains:
            domain_indices = [i for i, s in enumerate(samples) if s.domain == domain]
            if not domain_indices:
                continue
            domain_preds = [all_predictions[i] for i in domain_indices]
            domain_refs = [all_references[i] for i in domain_indices]
            per_domain[domain] = evaluate_all(domain_preds, domain_refs)

        return EvalResult(
            metrics=overall_metrics,
            per_domain_metrics=per_domain,
            num_samples=len(samples),
            num_domains=len(domains),
        )

    @staticmethod
    def from_checkpoint(
        checkpoint_path: str | Path,
        config: EvalConfig | None = None,
    ) -> EvaluationPipeline:
        """Create pipeline from a saved checkpoint.

        Loads model config and weights from checkpoint.
        """
        from pathlib import Path as _Path

        from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
        from cc_g2pnp.model.config import CC_G2PnPConfig
        from cc_g2pnp.training.checkpoint import CheckpointManager

        path = _Path(checkpoint_path)
        cfg = config or EvalConfig()

        # Load checkpoint dict via CheckpointManager
        manager = CheckpointManager(checkpoint_dir=path.parent)
        checkpoint = manager.load(path)

        # Reconstruct model config from "model_config" key (CC_G2PnPConfig).
        # Falls back to default CC_G2PnPConfig if key is absent
        # (e.g. checkpoints saved before model_config was added).
        # Note: "config" key contains TrainingConfig, NOT model config.
        model_config_dict = checkpoint.get("model_config", {})
        model_config = (
            CC_G2PnPConfig(**model_config_dict) if model_config_dict else CC_G2PnPConfig()
        )

        model = CC_G2PnP(model_config)

        # Load model weights
        model.load_state_dict(checkpoint["model_state_dict"])

        vocab = PnPVocabulary()
        return EvaluationPipeline(model, vocab, cfg)

    def format_results(self, result: EvalResult) -> str:
        """Format results as a human-readable string table."""
        lines = []
        lines.append(
            f"Evaluation Results ({result.num_samples} samples, {result.num_domains} domains)"
        )
        lines.append("=" * 60)

        if result.metrics:
            lines.append("")
            lines.append("Overall Metrics:")
            lines.append("-" * 40)
            for key, val in result.metrics.items():
                lines.append(f"  {key:.<30s} {val:>8.4f}")

        if result.per_domain_metrics:
            for domain, metrics in sorted(result.per_domain_metrics.items()):
                lines.append("")
                lines.append(f"Domain: {domain}")
                lines.append("-" * 40)
                for key, val in metrics.items():
                    lines.append(f"  {key:.<30s} {val:>8.4f}")

        return "\n".join(lines)
