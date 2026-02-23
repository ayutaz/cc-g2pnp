"""Tests for ConformerConvModule."""

import torch

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.convolution import ConformerConvModule


def test_output_shape():
    """出力形状が入力と同じ [B, T, D] であることを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig()
    model = ConformerConvModule(config)
    model.eval()

    x = torch.randn(2, 100, config.d_model)
    out = model(x)
    assert out.shape == (2, 100, config.d_model)


def test_different_sequence_lengths():
    """異なるシーケンス長で正しく動作することを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig()
    model = ConformerConvModule(config)
    model.eval()

    for seq_len in [10, 50, 200]:
        x = torch.randn(1, seq_len, config.d_model)
        out = model(x)
        assert out.shape == (1, seq_len, config.d_model)


def test_causal_property():
    """因果性: 未来の入力変更が過去の出力に影響しないことを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig()
    model = ConformerConvModule(config)
    model.eval()

    x = torch.randn(1, 100, config.d_model)

    with torch.no_grad():
        out_original = model(x.clone())

    # t=50以降の入力を変更
    x_modified = x.clone()
    x_modified[:, 50:, :] = torch.randn_like(x_modified[:, 50:, :])

    with torch.no_grad():
        out_modified = model(x_modified)

    # t<50の出力は変わらないはず
    torch.testing.assert_close(out_original[:, :50, :], out_modified[:, :50, :])


def test_custom_kernel_size():
    """カスタムカーネルサイズ (kernel=15) で動作することを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig(conv_kernel_size=15)
    model = ConformerConvModule(config)
    model.eval()

    x = torch.randn(2, 80, config.d_model)
    out = model(x)
    assert out.shape == (2, 80, config.d_model)
    assert model._causal_padding == 14


def test_dropout_effect():
    """train vs eval モードで dropout の効果が異なることを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig(dropout=0.5)
    model = ConformerConvModule(config)

    x = torch.randn(2, 50, config.d_model)

    # eval mode: deterministic
    model.eval()
    with torch.no_grad():
        out_eval1 = model(x.clone())
        out_eval2 = model(x.clone())
    torch.testing.assert_close(out_eval1, out_eval2)

    # train mode: dropout causes variance
    model.train()
    torch.manual_seed(0)
    out_train1 = model(x.clone())
    torch.manual_seed(1)
    out_train2 = model(x.clone())
    # 高い dropout 率のため出力が異なるはず
    assert not torch.allclose(out_train1, out_train2, atol=1e-6)


def test_batch_size_one_eval():
    """batch_size=1 in eval mode should work correctly (BatchNorm uses running stats)."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig()
    model = ConformerConvModule(config)
    # First pass in train mode to accumulate running stats
    model.train()
    x_train = torch.randn(4, 50, config.d_model)
    model(x_train)
    # Then eval mode with batch_size=1
    model.eval()
    x = torch.randn(1, 30, config.d_model)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 30, config.d_model)
    assert torch.isfinite(out).all()
