"""Tests for TokenEmbedding module."""

import math

import torch

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.embedding import TokenEmbedding


def _make_input(batch_size: int, seq_len: int, vocab_size: int = 65_000) -> torch.Tensor:
    """Create random BPE token IDs."""
    torch.manual_seed(42)
    return torch.randint(0, vocab_size, (batch_size, seq_len))


def test_output_shape_default():
    """Output shape should be [B, T*8, 512] for default config."""
    config = CC_G2PnPConfig()
    model = TokenEmbedding(config)
    model.eval()
    x = _make_input(2, 10)
    out = model(x)
    assert out.shape == (2, 10 * 8, 512)


def test_output_dtype():
    """Output dtype should be float32."""
    config = CC_G2PnPConfig()
    model = TokenEmbedding(config)
    model.eval()
    x = _make_input(1, 5)
    out = model(x)
    assert out.dtype == torch.float32


def test_batch_size_1():
    """Batch size 1 should work correctly."""
    config = CC_G2PnPConfig()
    model = TokenEmbedding(config)
    model.eval()
    x = _make_input(1, 10)
    out = model(x)
    assert out.shape == (1, 80, 512)


def test_batch_size_4():
    """Batch size 4 should work correctly."""
    config = CC_G2PnPConfig()
    model = TokenEmbedding(config)
    model.eval()
    x = _make_input(4, 10)
    out = model(x)
    assert out.shape == (4, 80, 512)


def test_seq_len_1():
    """Sequence length 1 should work correctly."""
    config = CC_G2PnPConfig()
    model = TokenEmbedding(config)
    model.eval()
    x = _make_input(2, 1)
    out = model(x)
    assert out.shape == (2, 8, 512)


def test_seq_len_50():
    """Sequence length 50 should work correctly."""
    config = CC_G2PnPConfig()
    model = TokenEmbedding(config)
    model.eval()
    x = _make_input(2, 50)
    out = model(x)
    assert out.shape == (2, 400, 512)


def test_upsample_factor():
    """Output time dim should be exactly input T * upsample_factor."""
    config = CC_G2PnPConfig()
    model = TokenEmbedding(config)
    model.eval()
    for t in [1, 3, 7, 20]:
        x = _make_input(1, t)
        out = model(x)
        assert out.shape[1] == t * config.upsample_factor


def test_scale_factor():
    """Output magnitude should be ~sqrt(d_model) times raw embedding."""
    config = CC_G2PnPConfig()
    model = TokenEmbedding(config)
    model.eval()
    x = _make_input(1, 10)

    raw = model.embedding(x)
    scaled_out = model(x)
    # Undo repeat_interleave to compare: take every upsample_factor-th frame
    scaled_no_repeat = scaled_out[:, ::config.upsample_factor, :]

    expected_scale = math.sqrt(config.d_model)
    ratio = scaled_no_repeat.abs().mean() / raw.abs().mean()
    assert abs(ratio.item() - expected_scale) < 1.0


def test_dropout_eval_deterministic():
    """In eval mode, dropout should not drop any values."""
    config = CC_G2PnPConfig()
    model = TokenEmbedding(config)
    model.eval()
    x = _make_input(2, 10)

    out1 = model(x)
    out2 = model(x)
    assert torch.equal(out1, out2)


def test_custom_config():
    """Custom d_model=256 and upsample_factor=4 should produce correct shape."""
    config = CC_G2PnPConfig(d_model=256, upsample_factor=4)
    model = TokenEmbedding(config)
    model.eval()
    x = _make_input(2, 10, vocab_size=config.bpe_vocab_size)
    out = model(x)
    assert out.shape == (2, 10 * 4, 256)
