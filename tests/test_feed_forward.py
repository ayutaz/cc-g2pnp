"""Tests for FeedForwardModule."""

import torch

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.feed_forward import FeedForwardModule


def test_output_shape():
    """出力形状が入力と同じ [B, T, D] であることを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig()
    model = FeedForwardModule(config)
    model.eval()

    x = torch.randn(2, 100, config.d_model)
    out = model(x)
    assert out.shape == (2, 100, config.d_model)


def test_different_batch_sizes():
    """異なるバッチサイズで正しく動作することを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig()
    model = FeedForwardModule(config)
    model.eval()

    for batch_size in [1, 4, 16]:
        x = torch.randn(batch_size, 50, config.d_model)
        out = model(x)
        assert out.shape == (batch_size, 50, config.d_model)


def test_expansion_dimension():
    """内部の拡張次元が d_ff=2048 であることを検証."""
    config = CC_G2PnPConfig()
    model = FeedForwardModule(config)

    assert model.linear1.in_features == 512
    assert model.linear1.out_features == 2048
    assert model.linear2.in_features == 2048
    assert model.linear2.out_features == 512


def test_custom_config():
    """カスタム設定 (d_model=256, d_ff=1024) で動作することを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig(d_model=256, d_ff=1024)
    model = FeedForwardModule(config)
    model.eval()

    x = torch.randn(2, 30, 256)
    out = model(x)
    assert out.shape == (2, 30, 256)
    assert model.linear1.out_features == 1024
    assert model.linear2.in_features == 1024


def test_independent_per_timestep():
    """各タイムステップが独立に処理されることを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig()
    model = FeedForwardModule(config)
    model.eval()

    x = torch.randn(1, 100, config.d_model)

    with torch.no_grad():
        out_original = model(x.clone())

    # t=30 の入力だけ変更
    x_modified = x.clone()
    x_modified[:, 30, :] = torch.randn(config.d_model)

    with torch.no_grad():
        out_modified = model(x_modified)

    # t=30 以外の出力は変わらないはず
    torch.testing.assert_close(out_original[:, :30, :], out_modified[:, :30, :])
    torch.testing.assert_close(out_original[:, 31:, :], out_modified[:, 31:, :])
    # t=30 の出力は変わるはず
    assert not torch.allclose(out_original[:, 30, :], out_modified[:, 30, :])
