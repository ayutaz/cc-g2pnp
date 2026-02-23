"""Tests for ConformerBlock module."""

import torch

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.conformer_block import ConformerBlock


def _small_config(**overrides) -> CC_G2PnPConfig:
    """Create a small config for fast testing."""
    defaults = dict(d_model=64, num_heads=4, d_ff=128, num_layers=4, dropout=0.0, intermediate_ctc_layers=(1,))
    defaults.update(overrides)
    return CC_G2PnPConfig(**defaults)


def _make_input(batch_size: int, seq_len: int, d_model: int = 64) -> torch.Tensor:
    """Create random input tensor [B, T, D]."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, d_model)


def _make_pos_enc(seq_len: int, d_model: int = 64) -> torch.Tensor:
    """Create random positional encoding [1, T, D]."""
    torch.manual_seed(99)
    return torch.randn(1, seq_len, d_model)


def test_output_shape():
    """Output shape should be [B, T, D] matching input shape."""
    config = _small_config()
    block = ConformerBlock(config)
    block.eval()
    x = _make_input(2, 16)
    pos_enc = _make_pos_enc(16)
    out = block(x, pos_enc)
    assert out.shape == (2, 16, 64)


def test_output_shape_various_seq_lens():
    """Output shape should match input for different sequence lengths."""
    config = _small_config()
    block = ConformerBlock(config)
    block.eval()
    for t in [1, 5, 20, 31]:
        x = _make_input(1, t)
        pos_enc = _make_pos_enc(t)
        out = block(x, pos_enc)
        assert out.shape == (1, t, 64), f"Failed for seq_len={t}"


def test_without_mask():
    """Block should work without attention mask (mask=None)."""
    config = _small_config()
    block = ConformerBlock(config)
    block.eval()
    x = _make_input(2, 10)
    pos_enc = _make_pos_enc(10)
    out = block(x, pos_enc, mask=None)
    assert out.shape == (2, 10, 64)
    assert not torch.isnan(out).any()


def test_with_mask():
    """Block should work with a boolean attention mask [T, T]."""
    config = _small_config()
    block = ConformerBlock(config)
    block.eval()
    x = _make_input(2, 10)
    pos_enc = _make_pos_enc(10)
    # Simple causal mask
    mask = torch.tril(torch.ones(10, 10, dtype=torch.bool))
    out = block(x, pos_enc, mask=mask)
    assert out.shape == (2, 10, 64)
    assert not torch.isnan(out).any()


def test_output_differs_from_input():
    """Output should be different from input (transformation applied)."""
    config = _small_config()
    block = ConformerBlock(config)
    block.eval()
    x = _make_input(2, 10)
    pos_enc = _make_pos_enc(10)
    out = block(x, pos_enc)
    assert not torch.allclose(out, x, atol=1e-6)


def test_half_step_residual_factor():
    """Block with ff_residual_factor=0.5 should differ from factor=1.0."""
    config_half = _small_config(ff_residual_factor=0.5)
    config_full = _small_config(ff_residual_factor=1.0)

    torch.manual_seed(0)
    block_half = ConformerBlock(config_half)
    block_half.eval()

    torch.manual_seed(0)
    block_full = ConformerBlock(config_full)
    block_full.eval()

    x = _make_input(1, 10)
    pos_enc = _make_pos_enc(10)

    out_half = block_half(x, pos_enc)
    out_full = block_full(x, pos_enc)

    # Outputs should differ because residual factor changes the computation
    assert not torch.allclose(out_half, out_full, atol=1e-6)


def test_submodules_present():
    """Block should contain ffn1, attention, conv, ffn2 sub-modules."""
    config = _small_config()
    block = ConformerBlock(config)
    child_names = {name for name, _ in block.named_children()}
    assert "ffn1" in child_names
    assert "attention" in child_names
    assert "conv" in child_names
    assert "ffn2" in child_names
    assert "final_norm" in child_names


def test_custom_config():
    """Block should work with a custom d_model=32, num_heads=2."""
    config = _small_config(d_model=32, num_heads=2, d_ff=64)
    block = ConformerBlock(config)
    block.eval()
    x = _make_input(2, 8, d_model=32)
    pos_enc = _make_pos_enc(8, d_model=32)
    out = block(x, pos_enc)
    assert out.shape == (2, 8, 32)


def test_eval_deterministic():
    """In eval mode, repeated forward passes should give identical results."""
    config = _small_config()
    block = ConformerBlock(config)
    block.eval()
    x = _make_input(1, 10)
    pos_enc = _make_pos_enc(10)
    out1 = block(x, pos_enc)
    out2 = block(x, pos_enc)
    assert torch.equal(out1, out2)
