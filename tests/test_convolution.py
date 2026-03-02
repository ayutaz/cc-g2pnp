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
    """因果性: 未来の入力変更が過去の出力に影響しないことを検証 (BatchNorm 使用)."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig(use_groupnorm=False)
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
    """batch_size=1 in eval mode should work correctly."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig()
    model = ConformerConvModule(config)
    # First pass in train mode to accumulate stats (for BatchNorm compatibility)
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


# ── GroupNorm テスト ─────────────────────────────────────────────────────────


def test_groupnorm_output_shape():
    """use_groupnorm=True で出力形状が [B, T, D] になることを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig(use_groupnorm=True)
    model = ConformerConvModule(config)
    model.eval()

    x = torch.randn(2, 100, config.d_model)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 100, config.d_model)
    assert torch.isfinite(out).all()


def test_groupnorm_streaming_output_shape():
    """use_groupnorm=True のストリーミング推論で出力形状が正しいことを検証."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig(use_groupnorm=True, chunk_size=5, conv_kernel_size=31)
    model = ConformerConvModule(config)
    model.eval()

    chunk_size = config.chunk_size
    cache = torch.zeros(1, config.conv_kernel_size - 1, config.d_model)
    x_chunk = torch.randn(1, chunk_size, config.d_model)

    with torch.no_grad():
        out, new_cache = model.forward_streaming(x_chunk, cache)

    assert out.shape == (1, chunk_size, config.d_model)
    assert new_cache.shape == (1, config.conv_kernel_size - 1, config.d_model)
    assert torch.isfinite(out).all()


def test_groupnorm_no_batch_stats():
    """GroupNorm の出力は train/eval モードで変わらないことを検証 (BatchNorm と異なる特性)."""
    torch.manual_seed(42)
    config = CC_G2PnPConfig(use_groupnorm=True, dropout=0.0)
    model = ConformerConvModule(config)

    x = torch.randn(2, 50, config.d_model)

    model.eval()
    with torch.no_grad():
        out_eval = model(x.clone())

    model.train()
    with torch.no_grad():
        out_train = model(x.clone())

    # GroupNorm はバッチ統計を使わないため train/eval で同じ出力になる
    torch.testing.assert_close(out_eval, out_train)


def test_groupnorm_config_flag():
    """use_groupnorm=False では BatchNorm、use_groupnorm=True では GroupNorm が使われることを検証."""
    config_bn = CC_G2PnPConfig(use_groupnorm=False)
    model_bn = ConformerConvModule(config_bn)
    assert isinstance(model_bn.norm, torch.nn.BatchNorm1d)

    config_gn = CC_G2PnPConfig(use_groupnorm=True)
    model_gn = ConformerConvModule(config_gn)
    assert isinstance(model_gn.norm, torch.nn.GroupNorm)

    # GroupNorm のグループ数を確認 (d_model=512 の場合 32 groups)
    assert model_gn.norm.num_groups == min(32, config_gn.d_model)


def test_groupnorm_parameter_count():
    """GroupNorm と BatchNorm のパラメータ数が同程度であることを検証."""
    model_bn = ConformerConvModule(CC_G2PnPConfig(use_groupnorm=False))
    model_gn = ConformerConvModule(CC_G2PnPConfig(use_groupnorm=True))

    params_bn = sum(p.numel() for p in model_bn.parameters())
    params_gn = sum(p.numel() for p in model_gn.parameters())

    # どちらも norm layer に weight/bias (d_model 要素) を持つため総数は同じ
    assert params_bn == params_gn, (
        f"BatchNorm params={params_bn}, GroupNorm params={params_gn}"
    )
