"""Evaluation metrics for CC-G2PnP PnP label prediction."""

from __future__ import annotations

import jiwer

PROSODY_SYMBOLS: frozenset[str] = frozenset({"*", "/", "#"})


def _normalize_tokens(tokens: list[str]) -> list[str]:
    """Replace '#' (intonation phrase boundary) with '/' (accent phrase boundary)."""
    return [("/" if t == "#" else t) for t in tokens]


def _filter_phonemes(tokens: list[str]) -> list[str]:
    """Remove all prosody symbols, keeping only phoneme (katakana mora) tokens."""
    return [t for t in tokens if t not in PROSODY_SYMBOLS]


def _tokens_to_str(tokens: list[str]) -> str:
    """Join tokens with space for jiwer comparison."""
    return " ".join(tokens)


def compute_cer(
    predictions: list[list[str]],
    references: list[list[str]],
) -> float:
    """Compute token-level error rate using jiwer.wer.

    Each PnP token (katakana mora or prosody symbol) is treated as an atomic
    unit.  Tokens are space-joined so that ``jiwer.wer`` computes
    Levenshtein distance at the *token* level (micro-average across all
    reference tokens when given a list of sentences).

    Note: ``jiwer.cer`` would decompose multi-character tokens (e.g. キョ,
    シャ) into individual Unicode characters, yielding incorrect granularity.

    Args:
        predictions: Predicted PnP token sequences.
        references: Reference PnP token sequences.

    Returns:
        Micro-average token error rate. Returns 0.0 if no valid pairs.
    """
    valid_preds = []
    valid_refs = []
    for pred, ref in zip(predictions, references, strict=True):
        if not ref:
            continue
        valid_preds.append(_tokens_to_str(pred) if pred else "")
        valid_refs.append(_tokens_to_str(ref))

    if not valid_refs:
        return 0.0

    return jiwer.wer(valid_refs, valid_preds)


def compute_ser(
    predictions: list[list[str]],
    references: list[list[str]],
) -> float:
    """Compute Sentence Error Rate.

    SER = fraction of sentences where prediction != reference.
    A sentence has an error if its per-sentence CER > 0.

    Args:
        predictions: Predicted PnP token sequences.
        references: Reference PnP token sequences.

    Returns:
        SER as a percentage (0-100). Returns 0.0 if no valid pairs.
    """
    error_count = 0
    total = 0
    for pred, ref in zip(predictions, references, strict=True):
        if not ref:
            continue
        total += 1
        if pred != ref:
            error_count += 1

    if total == 0:
        return 0.0

    return (error_count / total) * 100.0


def compute_pnp_cer(
    predictions: list[list[str]],
    references: list[list[str]],
) -> float:
    """PnP CER: CER on full PnP sequences (phoneme + prosody)."""
    return compute_cer(predictions, references)


def compute_pnp_ser(
    predictions: list[list[str]],
    references: list[list[str]],
) -> float:
    """PnP SER: sentence error rate on full PnP sequences."""
    return compute_ser(predictions, references)


def compute_normalized_pnp_cer(
    predictions: list[list[str]],
    references: list[list[str]],
) -> float:
    """Normalized PnP CER: replace '#' with '/' then compute CER.

    This treats intonation phrase boundaries and accent phrase boundaries
    as equivalent, measuring only the phoneme + accent nucleus accuracy.
    """
    norm_preds = [_normalize_tokens(p) for p in predictions]
    norm_refs = [_normalize_tokens(r) for r in references]
    return compute_cer(norm_preds, norm_refs)


def compute_normalized_pnp_ser(
    predictions: list[list[str]],
    references: list[list[str]],
) -> float:
    """Normalized PnP SER: replace '#' with '/' then compute SER."""
    norm_preds = [_normalize_tokens(p) for p in predictions]
    norm_refs = [_normalize_tokens(r) for r in references]
    return compute_ser(norm_preds, norm_refs)


def compute_phoneme_cer(
    predictions: list[list[str]],
    references: list[list[str]],
) -> float:
    """Phoneme CER: remove prosody symbols then compute CER."""
    phon_preds = [_filter_phonemes(p) for p in predictions]
    phon_refs = [_filter_phonemes(r) for r in references]
    return compute_cer(phon_preds, phon_refs)


def compute_phoneme_ser(
    predictions: list[list[str]],
    references: list[list[str]],
) -> float:
    """Phoneme SER: remove prosody symbols then compute SER."""
    phon_preds = [_filter_phonemes(p) for p in predictions]
    phon_refs = [_filter_phonemes(r) for r in references]
    return compute_ser(phon_preds, phon_refs)


def evaluate_all(
    predictions: list[list[str]],
    references: list[list[str]],
) -> dict[str, float]:
    """Compute all 6 evaluation metrics.

    Returns:
        Dict with keys: "pnp_cer", "pnp_ser", "normalized_pnp_cer",
        "normalized_pnp_ser", "phoneme_cer", "phoneme_ser".
    """
    return {
        "pnp_cer": compute_pnp_cer(predictions, references),
        "pnp_ser": compute_pnp_ser(predictions, references),
        "normalized_pnp_cer": compute_normalized_pnp_cer(predictions, references),
        "normalized_pnp_ser": compute_normalized_pnp_ser(predictions, references),
        "phoneme_cer": compute_phoneme_cer(predictions, references),
        "phoneme_ser": compute_phoneme_ser(predictions, references),
    }
