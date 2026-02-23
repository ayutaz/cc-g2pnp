"""Tests for RelativePositionalEncoding module."""

import torch

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.positional_encoding import RelativePositionalEncoding


def test_output_shapes():
    """x_out should be [B,T,D] and pos_enc should be [1,T,D]."""
    config = CC_G2PnPConfig()
    model = RelativePositionalEncoding(config)
    model.eval()
    x = torch.randn(2, 20, 512)
    _x_out, pos_enc = model(x)
    assert _x_out.shape == (2, 20, 512)
    assert pos_enc.shape == (1, 20, 512)


def test_pos_enc_deterministic():
    """pos_enc should be the same for the same sequence length."""
    config = CC_G2PnPConfig()
    model = RelativePositionalEncoding(config)
    model.eval()
    x = torch.randn(1, 30, 512)
    _, pos_enc1 = model(x)
    _, pos_enc2 = model(x)
    assert torch.equal(pos_enc1, pos_enc2)


def test_different_seq_lengths():
    """Different sequence lengths should produce correctly sized outputs."""
    config = CC_G2PnPConfig()
    model = RelativePositionalEncoding(config)
    model.eval()
    for t in [1, 10, 50, 200]:
        x = torch.randn(2, t, 512)
        _x_out, pos_enc = model(x)
        assert _x_out.shape == (2, t, 512)
        assert pos_enc.shape == (1, t, 512)


def test_pos_enc_bounded():
    """pos_enc values should be in [-1, 1] (sinusoidal)."""
    config = CC_G2PnPConfig()
    model = RelativePositionalEncoding(config)
    model.eval()
    x = torch.randn(1, 100, 512)
    _, pos_enc = model(x)
    assert pos_enc.min().item() >= -1.0
    assert pos_enc.max().item() <= 1.0


def test_buffer_in_state_dict():
    """pe should be registered as a buffer and appear in state_dict."""
    config = CC_G2PnPConfig()
    model = RelativePositionalEncoding(config)
    assert "pe" in model.state_dict()


def test_buffer_shape():
    """pe buffer should have shape [1, max_len, d_model]."""
    config = CC_G2PnPConfig()
    model = RelativePositionalEncoding(config, max_len=5000)
    assert model.pe.shape == (1, 5000, 512)


def test_custom_max_len():
    """Custom max_len should change the pe buffer size."""
    config = CC_G2PnPConfig()
    model = RelativePositionalEncoding(config, max_len=1000)
    assert model.pe.shape == (1, 1000, 512)

    x = torch.randn(1, 999, 512)
    _x_out, pos_enc = model(x)
    assert pos_enc.shape == (1, 999, 512)


def test_pos_enc_prefix_consistency():
    """pos_enc for shorter T should be a prefix of pos_enc for longer T."""
    config = CC_G2PnPConfig()
    model = RelativePositionalEncoding(config)
    model.eval()

    x_short = torch.randn(1, 10, 512)
    x_long = torch.randn(1, 50, 512)
    _, pos_short = model(x_short)
    _, pos_long = model(x_long)
    assert torch.equal(pos_short, pos_long[:, :10, :])
