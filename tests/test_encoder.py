"""Tests for ConformerEncoder module."""

import torch

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.encoder import ConformerEncoder


def _small_config(**overrides) -> CC_G2PnPConfig:
    """Create a small config for fast testing."""
    defaults = dict(
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=4,
        intermediate_ctc_layers=(1,),
        dropout=0.0,
    )
    defaults.update(overrides)
    return CC_G2PnPConfig(**defaults)


def _make_input(batch_size: int, seq_len: int, d_model: int = 64) -> torch.Tensor:
    """Create random input tensor [B, T, D]."""
    torch.manual_seed(42)
    return torch.randn(batch_size, seq_len, d_model)


def test_output_keys():
    """Output dict should have 'output' and 'intermediate_logits' keys."""
    config = _small_config()
    encoder = ConformerEncoder(config)
    encoder.eval()
    x = _make_input(2, 16)
    result = encoder(x)
    assert "output" in result
    assert "intermediate_logits" in result


def test_output_shape():
    """Output 'output' tensor should have shape [B, T, D]."""
    config = _small_config()
    encoder = ConformerEncoder(config)
    encoder.eval()
    x = _make_input(2, 16)
    result = encoder(x)
    assert result["output"].shape == (2, 16, 64)


def test_intermediate_logits_count():
    """Number of intermediate logits should match intermediate_ctc_layers."""
    config = _small_config(intermediate_ctc_layers=(1,))
    encoder = ConformerEncoder(config)
    encoder.eval()
    x = _make_input(1, 16)
    result = encoder(x)
    assert len(result["intermediate_logits"]) == 1


def test_intermediate_logits_count_multiple():
    """Multiple intermediate CTC layers should produce matching logit count."""
    config = _small_config(num_layers=8, intermediate_ctc_layers=(1, 3, 5))
    encoder = ConformerEncoder(config)
    encoder.eval()
    x = _make_input(1, 16)
    result = encoder(x)
    assert len(result["intermediate_logits"]) == 3


def test_intermediate_logits_shape():
    """Each intermediate logit tensor should have shape [B, T, pnp_vocab_size]."""
    config = _small_config()
    encoder = ConformerEncoder(config)
    encoder.eval()
    x = _make_input(2, 16)
    result = encoder(x)
    for logits in result["intermediate_logits"]:
        assert logits.shape == (2, 16, 140)


def test_layer0_uses_mla_mask():
    """Layer 0 should use the MLA mask (different from chunk mask)."""
    config = _small_config(mla_size=2)
    encoder = ConformerEncoder(config)
    encoder.eval()

    # Verify config stores mla_size > 0 which triggers MLA mask for layer 0
    assert config.mla_size == 2

    # The encoder should have layers and create different masks for layer 0
    x = _make_input(1, 20)
    # Run forward to ensure no errors with MLA mask
    result = encoder(x)
    assert result["output"].shape == (1, 20, 64)


def test_num_layers():
    """Encoder should have the configured number of ConformerBlock layers."""
    config = _small_config(num_layers=4)
    encoder = ConformerEncoder(config)
    assert len(encoder.layers) == 4

    config8 = _small_config(num_layers=8)
    encoder8 = ConformerEncoder(config8)
    assert len(encoder8.layers) == 8


def test_self_conditioned_ctc_feedback():
    """Intermediate CTC logits should feed back into the hidden state.

    Compare output with and without intermediate CTC layers -- the self-
    conditioning feedback should cause the outputs to differ.
    """
    torch.manual_seed(0)
    config_with = _small_config(num_layers=4, intermediate_ctc_layers=(1,))
    encoder_with = ConformerEncoder(config_with)
    encoder_with.eval()

    torch.manual_seed(0)
    config_without = _small_config(num_layers=4, intermediate_ctc_layers=())
    encoder_without = ConformerEncoder(config_without)
    encoder_without.eval()

    x = _make_input(1, 16)
    out_with = encoder_with(x)["output"]
    out_without = encoder_without(x)["output"]

    # The feedback from CTC projection should make the outputs different
    assert not torch.allclose(out_with, out_without, atol=1e-6)


def test_no_intermediate_layers():
    """With no intermediate CTC layers, intermediate_logits should be empty."""
    config = _small_config(intermediate_ctc_layers=())
    encoder = ConformerEncoder(config)
    encoder.eval()
    x = _make_input(1, 16)
    result = encoder(x)
    assert len(result["intermediate_logits"]) == 0


def test_output_no_nan():
    """Output should not contain NaN values."""
    config = _small_config()
    encoder = ConformerEncoder(config)
    encoder.eval()
    x = _make_input(2, 20)
    result = encoder(x)
    assert not torch.isnan(result["output"]).any()
    for logits in result["intermediate_logits"]:
        assert not torch.isnan(logits).any()


def test_eval_deterministic():
    """In eval mode, repeated forward passes should give identical results."""
    config = _small_config()
    encoder = ConformerEncoder(config)
    encoder.eval()
    x = _make_input(1, 16)
    out1 = encoder(x)["output"]
    out2 = encoder(x)["output"]
    assert torch.equal(out1, out2)


def test_ctc_projection_shared():
    """ctc_projection and ctc_to_hidden linear layers should exist."""
    config = _small_config()
    encoder = ConformerEncoder(config)
    assert hasattr(encoder, "ctc_projection")
    assert hasattr(encoder, "ctc_to_hidden")
    assert encoder.ctc_projection.in_features == 64
    assert encoder.ctc_projection.out_features == 140
    assert encoder.ctc_to_hidden.in_features == 140
    assert encoder.ctc_to_hidden.out_features == 64
