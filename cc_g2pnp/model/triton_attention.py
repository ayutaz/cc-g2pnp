"""Triton-fused RPE attention kernel for CC-G2PnP.

Shaw et al. style relative positional encoding (RPE) fused into a single
Triton kernel to avoid materializing the dense O(T^2) pos_bias tensor.

scores[b,h,i,j] = (Q[b,h,i,:] @ K[b,h,j,:] + Q[b,h,i,:] @ pos_K[0,h,j,:]) / sqrt(d_k)

Only the forward pass is implemented in Triton; backward uses autograd.
Wrapped in torch.autograd.Function so gradient flow is automatic.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ── Triton kernel ────────────────────────────────────────────────────────────

if HAS_TRITON:

    @triton.jit
    def _rpe_attention_fwd_kernel(
        Q_ptr,  # noqa: N803
        K_ptr,  # noqa: N803
        V_ptr,  # noqa: N803
        PK_ptr,  # noqa: N803
        O_ptr,  # noqa: N803
        Mask_ptr,  # noqa: N803
        scale,
        stride_qb,
        stride_qh,
        stride_qt,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vt,
        stride_vd,
        stride_pkh,
        stride_pkt,
        stride_pkd,
        stride_ob,
        stride_oh,
        stride_ot,
        stride_od,
        stride_mt,
        T: tl.constexpr,  # noqa: N803
        D: tl.constexpr,  # noqa: N803
        HAS_MASK: tl.constexpr,  # noqa: N803
        BLOCK_M: tl.constexpr,  # noqa: N803
        BLOCK_N: tl.constexpr,  # noqa: N803
        BLOCK_D: tl.constexpr,  # noqa: N803
    ):
        """Triton kernel for RPE-fused attention forward pass.

        Each program handles one (batch, head, tile_of_queries) block.
        Online softmax (Dao et al.) allows O(1) memory in the N dimension.
        """
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        tile_m_idx = tl.program_id(2)

        # Row (query) offsets for this tile
        offs_m = tile_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        offs_d = tl.arange(0, BLOCK_D)  # [BLOCK_D]

        # Base pointers for Q (this batch/head)
        q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
        # Load Q tile: [BLOCK_M, BLOCK_D]
        q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q_mask = offs_m[:, None] < T
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)  # [BLOCK_M, BLOCK_D]

        # Base pointers for K, V, pos_K
        k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
        v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
        pk_base = PK_ptr + head_idx * stride_pkh  # pos_K has no batch dim

        # Online softmax state
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)  # running max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # running sum of exp
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)  # output accumulator

        # Iterate over N (key) dimension in BLOCK_N tiles
        for start_n in range(0, T, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)  # [BLOCK_N]

            # Load K tile: [BLOCK_D, BLOCK_N] (transposed for matmul)
            k_ptrs = k_base + offs_n[None, :] * stride_kt + offs_d[:, None] * stride_kd
            k_mask = offs_n[None, :] < T
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)  # [BLOCK_D, BLOCK_N]

            # Load pos_K tile: [BLOCK_D, BLOCK_N]
            pk_ptrs = pk_base + offs_n[None, :] * stride_pkt + offs_d[:, None] * stride_pkd
            pk = tl.load(pk_ptrs, mask=k_mask, other=0.0)  # [BLOCK_D, BLOCK_N]

            # Compute QK^T + Q*pos_K^T, then scale: [BLOCK_M, BLOCK_N]
            qk = tl.dot(q, k)  # [BLOCK_M, BLOCK_N]
            qpk = tl.dot(q, pk)  # [BLOCK_M, BLOCK_N]
            scores = (qk + qpk) * scale  # [BLOCK_M, BLOCK_N]

            # Apply chunk mask: masked positions -> -inf
            if HAS_MASK:
                mask_ptrs = Mask_ptr + offs_m[:, None] * stride_mt + offs_n[None, :]
                valid_m = offs_m[:, None] < T
                valid_n = offs_n[None, :] < T
                chunk_mask = tl.load(mask_ptrs, mask=valid_m & valid_n, other=0).to(tl.int1)
                scores = tl.where(chunk_mask, scores, float("-inf"))

            # Mask out-of-bounds columns
            scores = tl.where(offs_n[None, :] < T, scores, float("-inf"))

            # Online softmax update
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))  # [BLOCK_M]
            # Guard: when m_new == -inf (all masked), keep alpha=1 to avoid NaN
            safe_m_new = tl.where(m_new == float("-inf"), 0.0, m_new)
            safe_m_i = tl.where(m_i == float("-inf"), 0.0, m_i)
            alpha = tl.exp(safe_m_i - safe_m_new)  # correction factor
            exp_scores = tl.exp(scores - safe_m_new[:, None])  # [BLOCK_M, BLOCK_N]
            # Masked scores produce exp(-inf - safe_m_new) = 0, which is correct

            # Update accumulator
            l_i = alpha * l_i + tl.sum(exp_scores, axis=1)
            acc = alpha[:, None] * acc

            # Load V tile: [BLOCK_N, BLOCK_D]
            v_ptrs = v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=offs_n[:, None] < T, other=0.0)

            acc += tl.dot(exp_scores.to(v.dtype), v)
            m_i = m_new

        # Normalize — when l_i == 0 (fully masked rows), output 0
        acc = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)

        # Store output: [BLOCK_M, BLOCK_D]
        o_base = O_ptr + batch_idx * stride_ob + head_idx * stride_oh
        o_ptrs = o_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
        tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < T)

    def _triton_rpe_attention_forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_k: torch.Tensor,
        mask: torch.Tensor | None,
        scale: float,
    ) -> torch.Tensor:
        """Launch the Triton RPE attention kernel.

        Args:
            q:     [B, H, T, d_k] query
            k:     [B, H, T, d_k] key
            v:     [B, H, T, d_k] value
            pos_k: [1, H, T, d_k] positional key (batch dim broadcast)
            mask:  [T, T] bool tensor or None (True = attend)
            scale: 1/sqrt(d_k)

        Returns:
            Output tensor [B, H, T, d_k] in float16.
        """
        b, h, t, d = q.shape
        assert d in (8, 16, 32, 64, 128), f"d_k must be power-of-2 in [8..128], got {d}"
        assert t <= 4096, f"T must be <= 4096 for Triton kernel, got {t}"

        # Ensure contiguous float16 tensors on CUDA
        q = q.contiguous().to(torch.float16)
        k = k.contiguous().to(torch.float16)
        v = v.contiguous().to(torch.float16)
        # pos_k: [1, H, T, d_k] -> squeeze batch dim -> [H, T, d_k]
        pk = pos_k.squeeze(0).contiguous().to(torch.float16)

        out = torch.empty_like(q)

        block_m = 32
        block_n = 32
        block_d = triton.next_power_of_2(d)

        has_mask = mask is not None
        mask_cont = mask.contiguous().to(torch.int8) if has_mask else torch.empty(0, dtype=torch.int8, device=q.device)

        grid = (b, h, triton.cdiv(t, block_m))

        _rpe_attention_fwd_kernel[grid](
            q,
            k,
            v,
            pk,
            out,
            mask_cont,
            scale,
            # Q strides
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            # K strides
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            # V strides
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            # pos_K strides (no batch dim)
            pk.stride(0),
            pk.stride(1),
            pk.stride(2),
            # O strides
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            # mask strides
            mask_cont.stride(0) if has_mask else 0,
            T=t,
            D=d,
            HAS_MASK=has_mask,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )

        return out


# ── autograd.Function wrapper ────────────────────────────────────────────────


class _TritonRPEAttentionFunction(torch.autograd.Function):
    """Forward: Triton RPE fused kernel. Backward: autograd (PyTorch fallback)."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_k: torch.Tensor,
        mask: torch.Tensor | None,
        scale: float,
    ) -> torch.Tensor:
        out = _triton_rpe_attention_forward(q, k, v, pos_k, mask, scale)
        ctx.save_for_backward(q, k, v, pos_k)
        ctx.mask = mask
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, pos_k = ctx.saved_tensors
        mask = ctx.mask
        scale = ctx.scale

        # Recompute attention weights for backward (remat approach)
        # Use float32 for backward stability
        q_f = q.float()
        k_f = k.float()
        v_f = v.float()
        pos_k_f = pos_k.float()

        with torch.enable_grad():
            q_g = q_f.requires_grad_(True)
            k_g = k_f.requires_grad_(True)
            v_g = v_f.requires_grad_(True)
            pos_k_g = pos_k_f.requires_grad_(True)

            # Recompute forward in float32
            scores = (torch.matmul(q_g, k_g.transpose(-2, -1)) + torch.matmul(q_g, pos_k_g.transpose(-2, -1))) * scale
            if mask is not None:
                scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            remat_out = torch.matmul(attn, v_g)

            remat_out.backward(grad_output.float())

        return (
            q_g.grad.to(q.dtype),
            k_g.grad.to(k.dtype),
            v_g.grad.to(v.dtype),
            pos_k_g.grad.to(pos_k.dtype),
            None,  # mask has no gradient
            None,  # scale has no gradient
        )


def triton_rpe_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_k: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Triton-fused RPE attention (forward Triton kernel, backward autograd).

    Args:
        q:     [B, H, T, d_k]
        k:     [B, H, T, d_k]
        v:     [B, H, T, d_k]
        pos_k: [1, H, T, d_k]
        mask:  [T, T] bool (True = attend) or None
        scale: 1/sqrt(d_k)

    Returns:
        Output [B, H, T, d_k] in float16.
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed. Cannot use triton_rpe_attention.")
    return _TritonRPEAttentionFunction.apply(q, k, v, pos_k, mask, scale)


def rpe_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    pos_k: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Pure PyTorch reference implementation for numerical validation.

    Args:
        q:     [B, H, T, d_k]
        k:     [B, H, T, d_k]
        v:     [B, H, T, d_k]
        pos_k: [1, H, T, d_k]
        mask:  [T, T] bool or None
        scale: 1/sqrt(d_k)

    Returns:
        Output [B, H, T, d_k].
    """
    scores = (torch.matmul(q, k.transpose(-2, -1)) + torch.matmul(q, pos_k.transpose(-2, -1))) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(attn, v)


def d_k_is_supported(d_k: int) -> bool:
    """Return True if d_k is supported by the Triton kernel (power-of-2, 8..128)."""
    return d_k in (8, 16, 32, 64, 128)


def default_scale(d_k: int) -> float:
    """Return 1/sqrt(d_k)."""
    return 1.0 / math.sqrt(d_k)
