"""Tests for chunk-aware attention masks and ChunkAwareAttention module."""

import torch

from cc_g2pnp.model.attention import (
    ChunkAwareAttention,
    create_chunk_mask,
    create_mla_mask,
)
from cc_g2pnp.model.config import CC_G2PnPConfig

# ── create_chunk_mask ────────────────────────────────────────────


def test_chunk_mask_output_shape():
    """Output shape should be [seq_len, seq_len] boolean tensor."""
    mask = create_chunk_mask(seq_len=12, chunk_size=4, past_context=2)
    assert mask.shape == (12, 12)
    assert mask.dtype == torch.bool


def test_chunk_mask_within_chunk_bidirectional():
    """Token 0 can attend to all tokens within chunk 0 (tokens 0..C-1)."""
    mask = create_chunk_mask(seq_len=9, chunk_size=3, past_context=2)
    # Token 0 is in chunk 0 (tokens 0,1,2) — should attend to all of them
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is True
    assert mask[0, 2].item() is True


def test_chunk_mask_past_context():
    """Token at chunk 1 start can attend P tokens back into chunk 0."""
    mask = create_chunk_mask(seq_len=9, chunk_size=3, past_context=2)
    # Token 3 is chunk 1 start. Past context P=2 means attend_start=1.
    assert mask[3, 1].item() is True  # past context
    assert mask[3, 2].item() is True  # past context
    assert mask[3, 3].item() is True  # own chunk
    assert mask[3, 4].item() is True  # own chunk
    assert mask[3, 5].item() is True  # own chunk


def test_chunk_mask_no_future_beyond_chunk():
    """Token 0 cannot attend to token C (next chunk)."""
    mask = create_chunk_mask(seq_len=9, chunk_size=3, past_context=2)
    # Token 0 is in chunk 0 (ends at 3), so cannot see token 3+
    assert mask[0, 3].item() is False
    assert mask[0, 4].item() is False
    assert mask[0, 8].item() is False


def test_chunk_mask_partial_last_chunk():
    """Works when seq_len is not divisible by chunk_size."""
    # seq_len=7, chunk_size=3 → chunks: [0,1,2], [3,4,5], [6]
    mask = create_chunk_mask(seq_len=7, chunk_size=3, past_context=2)
    assert mask.shape == (7, 7)
    # Token 6 (chunk 2, partial): attend_start=max(0,6-2)=4, chunk_end=7
    assert mask[6, 4].item() is True
    assert mask[6, 5].item() is True
    assert mask[6, 6].item() is True
    assert mask[6, 3].item() is False  # beyond past context


def test_chunk_mask_concrete_example():
    """Full manual verification: C=3, P=2, seq_len=9.

    Chunks: [0,1,2], [3,4,5], [6,7,8]

    Expected mask (1=True, 0=False):
        Col:  0 1 2 3 4 5 6 7 8
    Row 0:    1 1 1 0 0 0 0 0 0
    Row 1:    1 1 1 0 0 0 0 0 0
    Row 2:    1 1 1 0 0 0 0 0 0
    Row 3:    0 1 1 1 1 1 0 0 0
    Row 4:    0 1 1 1 1 1 0 0 0
    Row 5:    0 1 1 1 1 1 0 0 0
    Row 6:    0 0 0 0 1 1 1 1 1
    Row 7:    0 0 0 0 1 1 1 1 1
    Row 8:    0 0 0 0 1 1 1 1 1
    """
    mask = create_chunk_mask(seq_len=9, chunk_size=3, past_context=2)
    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


# ── create_mla_mask ──────────────────────────────────────────────


def test_mla_mask_extends_lookahead():
    """MLA extends look-ahead by M tokens beyond chunk boundary."""
    mask = create_mla_mask(seq_len=9, chunk_size=3, past_context=2, mla_size=1)
    # Token 0 is in chunk 0, chunk_end=3, with MLA=1 → attend_end=4
    assert mask[0, 3].item() is True  # one beyond chunk end


def test_mla_mask_token0_sees_token_c():
    """Token 0 with MLA=1 can see token C (one beyond chunk end)."""
    mask = create_mla_mask(seq_len=9, chunk_size=3, past_context=2, mla_size=1)
    assert mask[0, 0].item() is True
    assert mask[0, 1].item() is True
    assert mask[0, 2].item() is True
    assert mask[0, 3].item() is True  # MLA allows this
    assert mask[0, 4].item() is False  # but not two beyond


def test_mla_mask_zero_equals_chunk_mask():
    """MLA=0 should produce the same mask as the regular chunk mask."""
    chunk_mask = create_chunk_mask(seq_len=9, chunk_size=3, past_context=2)
    mla_mask = create_mla_mask(seq_len=9, chunk_size=3, past_context=2, mla_size=0)
    assert torch.equal(chunk_mask, mla_mask)


def test_mla_mask_strictly_more_positions():
    """MLA mask allows strictly more positions than regular chunk mask."""
    chunk_mask = create_chunk_mask(seq_len=9, chunk_size=3, past_context=2)
    mla_mask = create_mla_mask(seq_len=9, chunk_size=3, past_context=2, mla_size=1)
    # MLA mask should be a superset: everywhere chunk_mask is True, mla_mask is also True
    assert torch.all(mla_mask | ~chunk_mask).item()
    # MLA mask should have strictly more True positions
    assert mla_mask.sum().item() > chunk_mask.sum().item()


def test_mla_mask_concrete_example():
    """Full manual verification: C=3, P=2, M=1, seq_len=9.

    Compared to chunk mask, each row gets M=1 extra token beyond chunk_end.

    Expected mask:
        Col:  0 1 2 3 4 5 6 7 8
    Row 0:    1 1 1 1 0 0 0 0 0   (chunk_end=3, +1=4)
    Row 1:    1 1 1 1 0 0 0 0 0
    Row 2:    1 1 1 1 0 0 0 0 0
    Row 3:    0 1 1 1 1 1 1 0 0   (chunk_end=6, +1=7)
    Row 4:    0 1 1 1 1 1 1 0 0
    Row 5:    0 1 1 1 1 1 1 0 0
    Row 6:    0 0 0 0 1 1 1 1 1   (chunk_end=9=seq_len, no extra)
    Row 7:    0 0 0 0 1 1 1 1 1
    Row 8:    0 0 0 0 1 1 1 1 1
    """
    mask = create_mla_mask(seq_len=9, chunk_size=3, past_context=2, mla_size=1)
    expected = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


# ── ChunkAwareAttention ─────────────────────────────────────────


def _make_config(**overrides: object) -> CC_G2PnPConfig:
    """Create a small config for testing."""
    defaults = {
        "d_model": 32,
        "num_heads": 4,
        "dropout": 0.0,
    }
    defaults.update(overrides)
    return CC_G2PnPConfig(**defaults)


def test_attention_output_shape():
    """Output shape should be [B, T, D] for input [B, T, D]."""
    torch.manual_seed(42)
    config = _make_config()
    attn = ChunkAwareAttention(config)
    attn.eval()
    x = torch.randn(2, 10, 32)
    pos_enc = torch.randn(1, 10, 32)
    out = attn(x, pos_enc)
    assert out.shape == (2, 10, 32)


def test_attention_without_mask():
    """Attention works without a mask (full self-attention)."""
    torch.manual_seed(42)
    config = _make_config()
    attn = ChunkAwareAttention(config)
    attn.eval()
    x = torch.randn(1, 6, 32)
    pos_enc = torch.randn(1, 6, 32)
    out = attn(x, pos_enc)
    assert out.shape == (1, 6, 32)
    # Output should contain finite values
    assert torch.isfinite(out).all()


def test_attention_with_mask():
    """Attention works correctly with a chunk mask applied."""
    torch.manual_seed(42)
    config = _make_config()
    attn = ChunkAwareAttention(config)
    attn.eval()
    x = torch.randn(2, 9, 32)
    pos_enc = torch.randn(1, 9, 32)
    mask = create_chunk_mask(seq_len=9, chunk_size=3, past_context=2)
    out = attn(x, pos_enc, mask=mask)
    assert out.shape == (2, 9, 32)
    assert torch.isfinite(out).all()


def test_attention_pos_enc_broadcast():
    """Positional encoding with shape [1, T, D] is accepted and broadcasts."""
    torch.manual_seed(42)
    config = _make_config()
    attn = ChunkAwareAttention(config)
    attn.eval()
    x = torch.randn(3, 8, 32)
    pos_enc = torch.randn(1, 8, 32)  # batch dim = 1, should broadcast
    out = attn(x, pos_enc)
    assert out.shape == (3, 8, 32)


def test_attention_different_batch_sizes():
    """Attention works with different batch sizes."""
    torch.manual_seed(42)
    config = _make_config()
    attn = ChunkAwareAttention(config)
    attn.eval()
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 6, 32)
        pos_enc = torch.randn(1, 6, 32)
        out = attn(x, pos_enc)
        assert out.shape == (batch_size, 6, 32)


def test_attention_parameter_count():
    """Number of parameters: 4 linear projections + pos_k + LayerNorm."""
    config = _make_config(d_model=32, num_heads=4)
    attn = ChunkAwareAttention(config)

    param_names = {name for name, _ in attn.named_parameters()}

    # 4 linear projections (weight + bias each)
    assert "w_q.weight" in param_names
    assert "w_q.bias" in param_names
    assert "w_k.weight" in param_names
    assert "w_k.bias" in param_names
    assert "w_v.weight" in param_names
    assert "w_v.bias" in param_names
    assert "w_out.weight" in param_names
    assert "w_out.bias" in param_names

    # Relative positional key projection (no bias term)
    assert "pos_k.weight" in param_names
    assert "pos_k.bias" not in param_names

    # LayerNorm
    assert "norm.weight" in param_names
    assert "norm.bias" in param_names

    # Total parameter count:
    # w_q: 32*32+32=1056, w_k: same, w_v: same, w_out: same → 4*1056=4224
    # pos_k: 32*32=1024 (no bias)
    # LayerNorm: 32+32=64
    # Total: 4224 + 1024 + 64 = 5312
    total_params = sum(p.numel() for p in attn.parameters())
    assert total_params == 5312


# ── SDPA (FlashAttention) path ───────────────────────────────────


def _make_sdpa_config(**overrides: object) -> CC_G2PnPConfig:
    """Create a small config with use_flash_attention=True for testing."""
    defaults: dict = {"d_model": 32, "num_heads": 4, "dropout": 0.0, "use_flash_attention": True}
    defaults.update(overrides)
    return CC_G2PnPConfig(**defaults)


def test_sdpa_output_shape():
    """SDPA path: output shape should be [B, T, D]."""
    torch.manual_seed(42)
    config = _make_sdpa_config()
    attn = ChunkAwareAttention(config)
    attn.eval()
    x = torch.randn(2, 10, 32)
    pos_enc = torch.randn(1, 10, 32)
    out = attn(x, pos_enc)
    assert out.shape == (2, 10, 32)


def test_sdpa_without_mask():
    """SDPA path works without a mask (full self-attention)."""
    torch.manual_seed(42)
    config = _make_sdpa_config()
    attn = ChunkAwareAttention(config)
    attn.eval()
    x = torch.randn(1, 6, 32)
    pos_enc = torch.randn(1, 6, 32)
    out = attn(x, pos_enc)
    assert out.shape == (1, 6, 32)
    assert torch.isfinite(out).all()


def test_sdpa_with_mask():
    """SDPA path works correctly with a chunk mask applied."""
    torch.manual_seed(42)
    config = _make_sdpa_config()
    attn = ChunkAwareAttention(config)
    attn.eval()
    x = torch.randn(2, 9, 32)
    pos_enc = torch.randn(1, 9, 32)
    mask = create_chunk_mask(seq_len=9, chunk_size=3, past_context=2)
    out = attn(x, pos_enc, mask=mask)
    assert out.shape == (2, 9, 32)
    assert torch.isfinite(out).all()


def test_sdpa_different_batch_sizes():
    """SDPA path works with different batch sizes."""
    torch.manual_seed(42)
    config = _make_sdpa_config()
    attn = ChunkAwareAttention(config)
    attn.eval()
    for batch_size in [1, 2, 4, 8]:
        x = torch.randn(batch_size, 6, 32)
        pos_enc = torch.randn(1, 6, 32)
        out = attn(x, pos_enc)
        assert out.shape == (batch_size, 6, 32)


def test_sdpa_numerical_equivalence():
    """SDPA and manual paths produce numerically equivalent results (no mask)."""
    torch.manual_seed(0)
    config_manual = _make_config()
    config_sdpa = _make_sdpa_config()

    manual_attn = ChunkAwareAttention(config_manual)
    sdpa_attn = ChunkAwareAttention(config_sdpa)
    # Share the same weights
    sdpa_attn.load_state_dict(manual_attn.state_dict())

    manual_attn.eval()
    sdpa_attn.eval()

    x = torch.randn(2, 8, 32)
    pos_enc = torch.randn(1, 8, 32)

    with torch.no_grad():
        manual_out = manual_attn(x, pos_enc)
        sdpa_out = sdpa_attn(x, pos_enc)

    assert torch.allclose(manual_out, sdpa_out, atol=1e-5), (
        f"Max diff: {(manual_out - sdpa_out).abs().max().item()}"
    )


def test_sdpa_numerical_equivalence_with_mask():
    """SDPA and manual paths produce numerically equivalent results with chunk mask."""
    torch.manual_seed(1)
    config_manual = _make_config()
    config_sdpa = _make_sdpa_config()

    manual_attn = ChunkAwareAttention(config_manual)
    sdpa_attn = ChunkAwareAttention(config_sdpa)
    sdpa_attn.load_state_dict(manual_attn.state_dict())

    manual_attn.eval()
    sdpa_attn.eval()

    x = torch.randn(2, 9, 32)
    pos_enc = torch.randn(1, 9, 32)
    mask = create_chunk_mask(seq_len=9, chunk_size=3, past_context=2)

    with torch.no_grad():
        manual_out = manual_attn(x, pos_enc, mask=mask)
        sdpa_out = sdpa_attn(x, pos_enc, mask=mask)

    assert torch.allclose(manual_out, sdpa_out, atol=1e-5), (
        f"Max diff: {(manual_out - sdpa_out).abs().max().item()}"
    )


def test_sdpa_config_flag_dispatch():
    """use_flash_attention=False routes to _forward_manual; True routes to _forward_sdpa."""
    torch.manual_seed(2)

    manual_attn = ChunkAwareAttention(_make_config())
    sdpa_attn = ChunkAwareAttention(_make_sdpa_config())

    assert manual_attn.use_flash_attention is False
    assert sdpa_attn.use_flash_attention is True

    manual_attn.eval()
    sdpa_attn.eval()
    sdpa_attn.load_state_dict(manual_attn.state_dict())

    x = torch.randn(1, 4, 32)
    pos_enc = torch.randn(1, 4, 32)

    with torch.no_grad():
        out_manual = manual_attn._forward_manual(x, pos_enc)
        out_sdpa = sdpa_attn._forward_sdpa(x, pos_enc)
        # forward() should dispatch to respective paths
        out_dispatch_manual = manual_attn(x, pos_enc)
        out_dispatch_sdpa = sdpa_attn(x, pos_enc)

    assert torch.allclose(out_dispatch_manual, out_manual, atol=1e-6)
    assert torch.allclose(out_dispatch_sdpa, out_sdpa, atol=1e-6)


def test_sdpa_streaming_unaffected():
    """forward_streaming() works the same regardless of use_flash_attention flag."""
    torch.manual_seed(3)

    manual_attn = ChunkAwareAttention(_make_config())
    sdpa_attn = ChunkAwareAttention(_make_sdpa_config())
    sdpa_attn.load_state_dict(manual_attn.state_dict())

    manual_attn.eval()
    sdpa_attn.eval()

    b, c, d = 1, 5, 32
    cache_len = 3
    x = torch.randn(b, c, d)
    pos_enc = torch.randn(1, cache_len + c, d)
    k_cache = torch.randn(b, 4, cache_len, d // 4)
    v_cache = torch.randn(b, 4, cache_len, d // 4)

    with torch.no_grad():
        out_manual, _ = manual_attn.forward_streaming(x, pos_enc, (k_cache, v_cache), past_context=10)
        out_sdpa, _ = sdpa_attn.forward_streaming(x, pos_enc, (k_cache, v_cache), past_context=10)

    assert out_manual.shape == out_sdpa.shape == (b, c, d)
    # Both paths use the same manual streaming logic
    assert torch.allclose(out_manual, out_sdpa, atol=1e-6)


def test_sdpa_flag_default_false():
    """CC_G2PnPConfig() のデフォルトで use_flash_attention が False であること。"""
    config = CC_G2PnPConfig()
    assert config.use_flash_attention is False
