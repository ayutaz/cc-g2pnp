"""Tests for ReazonSpeech streaming data loading."""

import re

import pytest
from datasets import load_dataset

REAZON_DATASET = "reazon-research/reazonspeech"
REAZON_SUBSET = "tiny"


def _load_reazonspeech_text(streaming=True):
    """Load ReazonSpeech dataset selecting only text columns to avoid audio decoding."""
    return load_dataset(REAZON_DATASET, REAZON_SUBSET, split="train", streaming=streaming).select_columns(
        ["name", "transcription"]
    )


@pytest.mark.network
@pytest.mark.slow
def test_load_streaming():
    """ReazonSpeech dataset should load in streaming mode."""
    ds = _load_reazonspeech_text(streaming=True)
    assert ds is not None


@pytest.mark.network
@pytest.mark.slow
def test_iterate_samples():
    """Should be able to iterate and get samples with 'transcription' field."""
    ds = _load_reazonspeech_text(streaming=True)
    samples = []
    for i, sample in enumerate(ds):
        if i >= 5:
            break
        samples.append(sample)
    assert len(samples) == 5
    for sample in samples:
        assert "transcription" in sample


@pytest.mark.network
@pytest.mark.slow
def test_transcription_is_japanese():
    """Transcription should contain Japanese characters."""
    japanese_pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
    ds = _load_reazonspeech_text(streaming=True)
    sample = next(iter(ds))
    transcription = sample["transcription"]
    assert japanese_pattern.search(transcription), f"No Japanese characters found in: {transcription}"
