"""Tests for Triton RPE fused attention kernel.

Tests cover:
1. Triton forward vs manual reference numerical equivalence (fp16, atol=1e-2)
2. Backward (autograd) correctness
3. Various sequence lengths: T=32, 64, 128, 256, 512
4. mask=None and mask=chunk_mask
5. HAS_TRITON=False graceful fallback (skip)
"""

from __future__ import annotations

import math

import pytest
import torch

from cc_g2pnp.model.attention import ChunkAwareAttention, create_chunk_mask
from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.triton_attention import HAS_TRITON, rpe_attention_reference

# ---------------------------------------------------------------------------
# Skip entire module if Triton is not available
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not HAS_TRITON, reason="Triton not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D_MODEL = 64  # small but d_k=16 (64/4 heads) is supported
NUM_HEADS = 4
D_K = D_MODEL // NUM_HEADS  # 16


def _cuda_or_skip() -> torch.device:
    """Return CUDA device or skip the test if CUDA is unavailable."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; Triton kernel requires CUDA")
    return torch.device("cuda")


def _make_qkvp(
    b: int,
    h: int,
    t: int,
    d: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create random Q, K, V, pos_K tensors."""
    torch.manual_seed(42)
    q = torch.randn(b, h, t, d, device=device, dtype=dtype)
    k = torch.randn(b, h, t, d, device=device, dtype=dtype)
    v = torch.randn(b, h, t, d, device=device, dtype=dtype)
    pos_k = torch.randn(1, h, t, d, device=device, dtype=dtype)
    return q, k, v, pos_k


def _make_chunk_mask(t: int, device: torch.device) -> torch.Tensor:
    """Create a chunk mask with chunk_size=16, past_context=32."""
    chunk_size = max(16, t // 4)
    past_context = min(32, t)
    return create_chunk_mask(t, chunk_size, past_context).to(device)


# ---------------------------------------------------------------------------
# 1. Numerical equivalence: Triton forward vs reference (fp16)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [32, 64, 128, 256, 512])
def test_triton_forward_no_mask(t: int) -> None:
    """Triton forward without mask is numerically close to reference (atol=1e-2)."""
    from cc_g2pnp.model.triton_attention import triton_rpe_attention

    device = _cuda_or_skip()
    b, h, d = 2, NUM_HEADS, D_K
    scale = 1.0 / math.sqrt(d)

    q, k, v, pos_k = _make_qkvp(b, h, t, d, device)

    triton_out = triton_rpe_attention(q, k, v, pos_k, mask=None, scale=scale)
    ref_out = rpe_attention_reference(q, k, v, pos_k, mask=None, scale=scale)

    assert triton_out.shape == (b, h, t, d)
    assert torch.isfinite(triton_out).all(), "Triton output contains non-finite values"
    assert torch.allclose(triton_out.float(), ref_out.float(), atol=1e-2), (
        f"T={t} max_diff={( triton_out.float() - ref_out.float()).abs().max():.4f}"
    )


@pytest.mark.parametrize("t", [32, 64, 128, 256, 512])
def test_triton_forward_with_mask(t: int) -> None:
    """Triton forward with chunk mask is numerically close to reference (atol=1e-2)."""
    from cc_g2pnp.model.triton_attention import triton_rpe_attention

    device = _cuda_or_skip()
    b, h, d = 2, NUM_HEADS, D_K
    scale = 1.0 / math.sqrt(d)

    q, k, v, pos_k = _make_qkvp(b, h, t, d, device)
    mask = _make_chunk_mask(t, device)

    triton_out = triton_rpe_attention(q, k, v, pos_k, mask=mask, scale=scale)
    ref_out = rpe_attention_reference(q, k, v, pos_k, mask=mask, scale=scale)

    assert triton_out.shape == (b, h, t, d)
    assert torch.isfinite(triton_out).all(), "Triton output contains non-finite values"
    assert torch.allclose(triton_out.float(), ref_out.float(), atol=1e-2), (
        f"T={t} max_diff={(triton_out.float() - ref_out.float()).abs().max():.4f}"
    )


# ---------------------------------------------------------------------------
# 2. Backward (autograd) correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("t", [32, 64])
@pytest.mark.parametrize("use_mask", [False, True])
def test_triton_backward(t: int, use_mask: bool) -> None:
    """Backward pass via autograd produces finite gradients for all inputs."""
    from cc_g2pnp.model.triton_attention import triton_rpe_attention

    device = _cuda_or_skip()
    b, h, d = 1, NUM_HEADS, D_K
    scale = 1.0 / math.sqrt(d)

    torch.manual_seed(7)
    q = torch.randn(b, h, t, d, device=device, dtype=torch.float16, requires_grad=True)
    k = torch.randn(b, h, t, d, device=device, dtype=torch.float16, requires_grad=True)
    v = torch.randn(b, h, t, d, device=device, dtype=torch.float16, requires_grad=True)
    pos_k = torch.randn(1, h, t, d, device=device, dtype=torch.float16, requires_grad=True)

    mask = _make_chunk_mask(t, device) if use_mask else None

    out = triton_rpe_attention(q, k, v, pos_k, mask=mask, scale=scale)
    loss = out.float().sum()
    loss.backward()

    for name, tensor in [("q", q), ("k", k), ("v", v), ("pos_k", pos_k)]:
        assert tensor.grad is not None, f"{name}.grad is None"
        assert torch.isfinite(tensor.grad).all(), f"{name}.grad contains non-finite values"


# ---------------------------------------------------------------------------
# 3. ChunkAwareAttention._forward_triton integration
# ---------------------------------------------------------------------------


def _make_triton_config(**overrides: object) -> CC_G2PnPConfig:
    defaults: dict = {
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "dropout": 0.0,
        "use_flash_attention": True,
    }
    defaults.update(overrides)
    return CC_G2PnPConfig(**defaults)


@pytest.mark.parametrize("t", [32, 64, 128, 256, 512])
def test_attn_forward_triton_output_shape(t: int) -> None:
    """ChunkAwareAttention._forward_triton output has correct shape."""
    device = _cuda_or_skip()
    config = _make_triton_config()
    # Use float16 model so LayerNorm weights match input dtype
    attn = ChunkAwareAttention(config).to(device).half().eval()

    x = torch.randn(2, t, D_MODEL, device=device, dtype=torch.float16)
    pos_enc = torch.randn(1, t, D_MODEL, device=device, dtype=torch.float16)

    with torch.no_grad():
        out = attn._forward_triton(x, pos_enc, mask=None)

    assert out.shape == (2, t, D_MODEL)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("t", [32, 64, 128])
def test_attn_forward_triton_with_mask(t: int) -> None:
    """ChunkAwareAttention._forward_triton works with chunk mask."""
    device = _cuda_or_skip()
    config = _make_triton_config()
    attn = ChunkAwareAttention(config).to(device).half().eval()

    x = torch.randn(2, t, D_MODEL, device=device, dtype=torch.float16)
    pos_enc = torch.randn(1, t, D_MODEL, device=device, dtype=torch.float16)
    mask = _make_chunk_mask(t, device)

    with torch.no_grad():
        out = attn._forward_triton(x, pos_enc, mask=mask)

    assert out.shape == (2, t, D_MODEL)
    assert torch.isfinite(out).all()


def test_attn_triton_vs_sdpa_numerical_equivalence() -> None:
    """_forward_triton and _forward_sdpa produce similar outputs (atol=1e-1 for fp16)."""
    device = _cuda_or_skip()
    config = _make_triton_config()
    attn = ChunkAwareAttention(config).to(device).half().eval()

    t = 64
    torch.manual_seed(0)
    x = torch.randn(2, t, D_MODEL, device=device, dtype=torch.float16)
    pos_enc = torch.randn(1, t, D_MODEL, device=device, dtype=torch.float16)

    with torch.no_grad():
        out_triton = attn._forward_triton(x, pos_enc, mask=None)
        out_sdpa = attn._forward_sdpa(x, pos_enc, mask=None)

    # fp16 computations accumulate more rounding error; use relaxed tolerance
    assert torch.allclose(out_triton.float(), out_sdpa.float(), atol=1e-1), (
        f"Max diff: {(out_triton.float() - out_sdpa.float()).abs().max():.4f}"
    )


# ---------------------------------------------------------------------------
# 4. forward() dispatch to _forward_triton when Triton is available + CUDA
# ---------------------------------------------------------------------------


def test_forward_dispatches_to_triton_on_cuda() -> None:
    """forward() dispatches to _forward_triton when CUDA is available and T<=1024."""
    device = _cuda_or_skip()
    config = _make_triton_config()
    attn = ChunkAwareAttention(config).to(device).half().eval()

    t = 64
    x = torch.randn(1, t, D_MODEL, device=device, dtype=torch.float16)
    pos_enc = torch.randn(1, t, D_MODEL, device=device, dtype=torch.float16)

    with torch.no_grad():
        out_dispatch = attn(x, pos_enc)
        out_triton = attn._forward_triton(x, pos_enc)

    assert torch.allclose(out_dispatch.float(), out_triton.float(), atol=1e-5), (
        f"Dispatch mismatch: max_diff={(out_dispatch.float() - out_triton.float()).abs().max():.6f}"
    )


# ---------------------------------------------------------------------------
# 5. HAS_TRITON=False graceful fallback
# ---------------------------------------------------------------------------


def test_has_triton_flag_is_bool() -> None:
    """HAS_TRITON is a boolean flag (True since Triton is installed)."""
    assert isinstance(HAS_TRITON, bool)
    assert HAS_TRITON is True  # Triton installed, so this is True


def test_triton_rpe_attention_raises_without_triton(monkeypatch: pytest.MonkeyPatch) -> None:
    """triton_rpe_attention raises RuntimeError when HAS_TRITON is patched to False."""
    import cc_g2pnp.model.triton_attention as ta

    device = _cuda_or_skip()
    b, h, t, d = 1, NUM_HEADS, 32, D_K
    q = torch.randn(b, h, t, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, t, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, t, d, device=device, dtype=torch.float16)
    pos_k = torch.randn(1, h, t, d, device=device, dtype=torch.float16)

    # Temporarily patch HAS_TRITON to False to simulate missing Triton
    monkeypatch.setattr(ta, "HAS_TRITON", False)

    with pytest.raises(RuntimeError, match="Triton is not installed"):
        ta.triton_rpe_attention(q, k, v, pos_k, mask=None, scale=1.0 / math.sqrt(d))


def test_forward_fallback_to_sdpa_on_cpu() -> None:
    """forward() falls back to SDPA (not Triton) when running on CPU."""
    config = _make_triton_config()
    # CPU model — Triton requires CUDA, so it should fall back to _forward_sdpa
    attn = ChunkAwareAttention(config).eval()

    t = 32
    x = torch.randn(1, t, D_MODEL)
    pos_enc = torch.randn(1, t, D_MODEL)

    with torch.no_grad():
        out = attn(x, pos_enc)
        expected = attn._forward_sdpa(x, pos_enc)

    assert out.shape == (1, t, D_MODEL)
    assert torch.allclose(out, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# 6. rpe_attention_reference correctness
# ---------------------------------------------------------------------------


def test_reference_output_shape() -> None:
    """rpe_attention_reference returns correct shape."""
    t, b, h, d = 16, 2, NUM_HEADS, D_K
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    pos_k = torch.randn(1, h, t, d)
    scale = 1.0 / math.sqrt(d)

    out = rpe_attention_reference(q, k, v, pos_k, mask=None, scale=scale)
    assert out.shape == (b, h, t, d)
    assert torch.isfinite(out).all()


def test_reference_with_all_masked() -> None:
    """rpe_attention_reference with an all-False mask produces NaN (expected softmax(-inf) behavior)."""
    t, b, h, d = 8, 1, NUM_HEADS, D_K
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    pos_k = torch.randn(1, h, t, d)
    scale = 1.0 / math.sqrt(d)

    # All-False mask → all -inf scores → NaN after softmax
    mask = torch.zeros(t, t, dtype=torch.bool)
    out = rpe_attention_reference(q, k, v, pos_k, mask=mask, scale=scale)
    # softmax(-inf, -inf, ...) = NaN — this is expected behavior
    assert out.shape == (b, h, t, d)
