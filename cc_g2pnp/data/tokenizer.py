"""Thin wrapper around the CALM2 BPE tokenizer.

Provides a consistent interface for the CC-G2PnP data pipeline,
encapsulating ``transformers.AutoTokenizer`` for the CyberAgent CALM2 model.
"""

from __future__ import annotations

from typing import ClassVar

from transformers import AutoTokenizer

_DEFAULT_MODEL = "cyberagent/calm2-7b"


class G2PnPTokenizer:
    """CALM2 BPE tokenizer wrapper for CC-G2PnP."""

    _cache: ClassVar[dict[str, G2PnPTokenizer]] = {}

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    @classmethod
    def get_instance(cls, model_name: str = _DEFAULT_MODEL) -> G2PnPTokenizer:
        """Return a cached tokenizer instance (singleton per model_name)."""
        if model_name not in cls._cache:
            cls._cache[model_name] = cls(model_name)
        return cls._cache[model_name]

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the singleton cache (for test isolation)."""
        cls._cache.clear()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode text to BPE token IDs."""
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: list[int]) -> str:
        """Decode BPE token IDs back to text."""
        return self._tokenizer.decode(ids)

    def batch_encode(self, texts: list[str], *, add_special_tokens: bool = False) -> list[list[int]]:
        """Encode a batch of texts to BPE token ID lists."""
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]
