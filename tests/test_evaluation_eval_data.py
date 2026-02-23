"""Tests for cc_g2pnp.evaluation.eval_data."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cc_g2pnp.evaluation.eval_data import (
    BUILTIN_TEXTS,
    EvalDataGenerator,
    EvalDataset,
    EvalSample,
)


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns simple BPE IDs."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(
        side_effect=lambda text: list(range(len(text))) if text.strip() else []
    )
    return tokenizer


@pytest.fixture
def generator(mock_tokenizer):
    return EvalDataGenerator(mock_tokenizer)


class TestEvalSample:
    def test_creation(self):
        sample = EvalSample(
            text="\u3053\u3093\u306b\u3061\u306f",
            domain="general",
            pnp_labels=["\u30b3", "\u30f3", "\u30cb", "\u30c1", "\u30ef"],
            bpe_ids=[1, 2, 3],
        )
        assert sample.text == "\u3053\u3093\u306b\u3061\u306f"
        assert sample.domain == "general"
        assert len(sample.pnp_labels) == 5
        assert len(sample.bpe_ids) == 3


class TestEvalDataset:
    def test_empty_dataset(self):
        ds = EvalDataset()
        assert len(ds) == 0
        assert ds.domains == []

    def test_domains(self):
        samples = [
            EvalSample("a", "news", ["\u30a2"], [1]),
            EvalSample("b", "conv", ["\u30a4"], [2]),
            EvalSample("c", "news", ["\u30a6"], [3]),
        ]
        ds = EvalDataset(samples=samples, name="test")
        assert ds.domains == ["conv", "news"]

    def test_filter_by_domain(self):
        samples = [
            EvalSample("a", "news", ["\u30a2"], [1]),
            EvalSample("b", "conv", ["\u30a4"], [2]),
            EvalSample("c", "news", ["\u30a6"], [3]),
        ]
        ds = EvalDataset(samples=samples, name="test")
        news = ds.filter_by_domain("news")
        assert len(news) == 2
        assert all(s.domain == "news" for s in news.samples)
        assert "news" in news.name

    def test_filter_nonexistent_domain(self):
        ds = EvalDataset(samples=[EvalSample("a", "news", ["\u30a2"], [1])])
        empty = ds.filter_by_domain("nonexistent")
        assert len(empty) == 0

    def test_len(self):
        samples = [EvalSample("a", "d", ["\u30a2"], [1])] * 5
        assert len(EvalDataset(samples=samples)) == 5

    def test_name_default(self):
        ds = EvalDataset()
        assert ds.name == "unnamed"

    def test_name_custom(self):
        ds = EvalDataset(name="my_dataset")
        assert ds.name == "my_dataset"

    def test_filter_preserves_name_format(self):
        ds = EvalDataset(
            samples=[EvalSample("a", "news", ["\u30a2"], [1])],
            name="test",
        )
        filtered = ds.filter_by_domain("news")
        assert filtered.name == "test:news"


class TestEvalDataGenerator:
    def test_from_texts_basic(self, generator):
        """Basic text generation works with Japanese text."""
        ds = generator.from_texts(
            ["\u3053\u3093\u306b\u3061\u306f"], domain="test", name="basic"
        )
        assert len(ds) > 0
        assert ds.name == "basic"
        assert ds.samples[0].domain == "test"
        assert ds.samples[0].text == "\u3053\u3093\u306b\u3061\u306f"
        assert len(ds.samples[0].pnp_labels) > 0
        assert len(ds.samples[0].bpe_ids) > 0

    def test_from_texts_skips_empty(self, generator):
        """Empty and whitespace-only texts are skipped."""
        ds = generator.from_texts(["", "  ", "  \n  "], domain="test")
        assert len(ds) == 0

    def test_from_texts_empty_list(self, generator):
        ds = generator.from_texts([], domain="test")
        assert len(ds) == 0

    def test_from_texts_multiple(self, generator):
        """Multiple valid texts produce multiple samples."""
        texts = [
            "\u4eca\u65e5\u306f\u3044\u3044\u5929\u6c17\u3067\u3059\u306d\u3002",
            "\u660e\u65e5\u306f\u96e8\u3067\u3059\u3002",
        ]
        ds = generator.from_texts(texts, domain="weather", name="multi")
        assert len(ds) == 2
        assert ds.name == "multi"
        assert all(s.domain == "weather" for s in ds.samples)

    def test_from_texts_mixed_valid_invalid(self, generator):
        """Mix of valid and empty texts only produces valid samples."""
        texts = ["", "\u3053\u3093\u306b\u3061\u306f", "  ", "\u3055\u3088\u3046\u306a\u3089"]
        ds = generator.from_texts(texts, domain="test")
        assert len(ds) == 2

    def test_from_texts_default_domain_and_name(self, generator):
        ds = generator.from_texts(["\u30c6\u30b9\u30c8"])
        assert ds.name == "custom"
        assert ds.samples[0].domain == "general"

    def test_from_file(self, generator, tmp_path):
        """Read from file with one sentence per line."""
        f = tmp_path / "test.txt"
        f.write_text(
            "\u3053\u3093\u306b\u3061\u306f\n\u3055\u3088\u3046\u306a\u3089\n",
            encoding="utf-8",
        )
        ds = generator.from_file(f, domain="test", name="file_test")
        assert ds.name == "file_test"
        assert len(ds) == 2

    def test_from_file_empty(self, generator, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        ds = generator.from_file(f, domain="test")
        assert len(ds) == 0

    def test_from_file_with_blank_lines(self, generator, tmp_path):
        """Blank lines in file are skipped."""
        f = tmp_path / "blanks.txt"
        f.write_text(
            "\u3053\u3093\u306b\u3061\u306f\n\n\u3055\u3088\u3046\u306a\u3089\n\n",
            encoding="utf-8",
        )
        ds = generator.from_file(f, domain="test")
        assert len(ds) == 2

    def test_from_file_default_params(self, generator, tmp_path):
        f = tmp_path / "default.txt"
        f.write_text("\u30c6\u30b9\u30c8\n", encoding="utf-8")
        ds = generator.from_file(f)
        assert ds.name == "file"
        assert ds.samples[0].domain == "general"

    def test_from_file_path_as_string(self, generator, tmp_path):
        """Accepts str path as well as Path."""
        f = tmp_path / "str_path.txt"
        f.write_text("\u30c6\u30b9\u30c8\n", encoding="utf-8")
        ds = generator.from_file(str(f), domain="test")
        assert len(ds) == 1

    def test_builtin_dataset(self, generator):
        """Built-in dataset produces samples from all domains."""
        ds = generator.builtin_dataset()
        assert ds.name == "builtin"
        assert len(ds) > 0
        assert len(ds.domains) == 4
        assert set(ds.domains) == {"news", "conversation", "literature", "technical"}

    def test_builtin_dataset_all_domains_have_samples(self, generator):
        """Each domain in built-in dataset has at least one sample."""
        ds = generator.builtin_dataset()
        for domain in ["news", "conversation", "literature", "technical"]:
            filtered = ds.filter_by_domain(domain)
            assert len(filtered) > 0, f"No samples for domain '{domain}'"

    def test_pnp_labels_are_valid(self, generator):
        """PnP labels contain expected token types."""
        ds = generator.from_texts(
            ["\u6771\u4eac\u90fd\u306f\u65b0\u3057\u3044\u6761\u4f8b\u3092\u65bd\u884c\u3059\u308b\u3068\u767a\u8868\u3057\u305f\u3002"],
            domain="test",
        )
        assert len(ds) > 0
        labels = ds.samples[0].pnp_labels
        # Should contain katakana characters and prosody markers
        assert any(c in labels for c in ["*", "/", "#"])

    def test_bpe_ids_are_ints(self, generator):
        """BPE IDs are integers."""
        ds = generator.from_texts(["\u30c6\u30b9\u30c8"], domain="test")
        assert len(ds) > 0
        assert all(isinstance(i, int) for i in ds.samples[0].bpe_ids)

    def test_skips_text_with_empty_bpe(self):
        """Texts that produce empty BPE tokens are skipped."""
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[])
        gen = EvalDataGenerator(tokenizer)
        ds = gen.from_texts(["\u3053\u3093\u306b\u3061\u306f"], domain="test")
        assert len(ds) == 0


class TestBuiltinTexts:
    def test_has_expected_domains(self):
        assert "news" in BUILTIN_TEXTS
        assert "conversation" in BUILTIN_TEXTS
        assert "literature" in BUILTIN_TEXTS
        assert "technical" in BUILTIN_TEXTS

    def test_each_domain_has_texts(self):
        for domain, texts in BUILTIN_TEXTS.items():
            assert len(texts) >= 5, f"Domain '{domain}' has too few texts"
            for text in texts:
                assert isinstance(text, str)
                assert len(text) > 0

    def test_all_texts_are_unique(self):
        """No duplicate texts across all domains."""
        all_texts = []
        for texts in BUILTIN_TEXTS.values():
            all_texts.extend(texts)
        assert len(all_texts) == len(set(all_texts))

    def test_domain_count(self):
        assert len(BUILTIN_TEXTS) == 4

    def test_each_domain_has_ten_texts(self):
        for domain, texts in BUILTIN_TEXTS.items():
            assert len(texts) == 10, f"Domain '{domain}' has {len(texts)} texts, expected 10"


@pytest.mark.network
class TestBuiltinDataset:
    """Tests that require network (for tokenizer download)."""

    def test_builtin_dataset(self):
        from cc_g2pnp.data.tokenizer import G2PnPTokenizer

        tokenizer = G2PnPTokenizer()
        gen = EvalDataGenerator(tokenizer)
        ds = gen.builtin_dataset()
        assert len(ds) > 0
        assert len(ds.domains) >= 4
