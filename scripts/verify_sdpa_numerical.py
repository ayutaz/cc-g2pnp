"""SDPA ON/OFF 数値精度検証スクリプト。

同じ入力・同じ重みで use_flash_attention=True/False の出力を比較し、
数値的な等価性を確認する。

使用例:
    uv run python scripts/verify_sdpa_numerical.py
    uv run python scripts/verify_sdpa_numerical.py --seq-len 20 --batch-size 4
"""

from __future__ import annotations

import argparse

import torch

from cc_g2pnp.model import CC_G2PnP
from cc_g2pnp.model.attention import ChunkAwareAttention, create_chunk_mask, create_mla_mask
from cc_g2pnp.model.config import CC_G2PnPConfig

# ──────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────────────────────────────


def error_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    """2 つのテンソル間の誤差統計を計算する。"""
    diff = (a - b).abs()
    denom = b.abs().clamp(min=1e-8)
    rel = (diff / denom)
    return {
        "max_abs":  diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel":  rel.max().item(),
        "mean_rel": rel.mean().item(),
    }


def print_stats(label: str, stats: dict[str, float], atol: float) -> bool:
    """統計を出力し、許容誤差内なら True を返す。"""
    ok = stats["max_abs"] <= atol
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    print(f"         max_abs={stats['max_abs']:.2e}  mean_abs={stats['mean_abs']:.2e}"
          f"  max_rel={stats['max_rel']:.2e}  mean_rel={stats['mean_rel']:.2e}"
          f"  (atol={atol:.0e})")
    return ok


# ──────────────────────────────────────────────────────────────────────────
# 1. ChunkAwareAttention 単体の検証
# ──────────────────────────────────────────────────────────────────────────


def verify_attention_module(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    chunk_size: int,
    past_context: int,
    mla_size: int,
    device: torch.device,
) -> bool:
    """ChunkAwareAttention の manual / _forward_sdpa / _forward_chunk_sdpa を比較する。"""
    print(f"\n{'─'*60}")
    print("  ChunkAwareAttention 単体検証")
    print(f"  B={batch_size}  T={seq_len}  D={d_model}  H={num_heads}"
          f"  C={chunk_size}  P={past_context}  M={mla_size}")
    print(f"{'─'*60}")

    torch.manual_seed(42)
    cfg_manual = CC_G2PnPConfig(
        d_model=d_model, num_heads=num_heads, dropout=0.0,
        chunk_size=chunk_size, past_context=past_context, mla_size=mla_size,
    )
    cfg_sdpa = CC_G2PnPConfig(
        d_model=d_model, num_heads=num_heads, dropout=0.0,
        use_flash_attention=True,
        chunk_size=chunk_size, past_context=past_context, mla_size=mla_size,
    )

    manual_attn = ChunkAwareAttention(cfg_manual).to(device).eval()
    sdpa_attn = ChunkAwareAttention(cfg_sdpa).to(device).eval()
    sdpa_attn.load_state_dict(manual_attn.state_dict())

    x = torch.randn(batch_size, seq_len, d_model, device=device)
    pos_enc = torch.randn(1, seq_len, d_model, device=device)

    chunk_mask = create_chunk_mask(seq_len, chunk_size, past_context).to(device)
    mla_mask   = create_mla_mask(seq_len, chunk_size, past_context, mla_size).to(device)

    all_pass = True
    atol_fp32 = 1e-4  # FP32 精度 (positional bias 計算の丸め誤差を考慮)

    with torch.no_grad():
        # (a) マスクなし: manual vs chunk_sdpa
        out_manual_nomask   = manual_attn._forward_manual(x, pos_enc)
        out_chunk_nomask    = sdpa_attn._forward_chunk_sdpa(x, pos_enc)
        s = error_stats(out_manual_nomask, out_chunk_nomask)
        all_pass &= print_stats("マスクなし: manual vs chunk_sdpa", s, atol_fp32)

        # (b) chunk_mask あり: manual vs chunk_sdpa
        out_manual_cmask    = manual_attn._forward_manual(x, pos_enc, mask=chunk_mask)
        out_chunk_cmask     = sdpa_attn._forward_chunk_sdpa(x, pos_enc, mask=chunk_mask)
        s = error_stats(out_manual_cmask, out_chunk_cmask)
        all_pass &= print_stats("chunk_mask: manual vs chunk_sdpa", s, atol_fp32)

        # (c) mla_mask あり: manual vs chunk_sdpa
        out_manual_mla      = manual_attn._forward_manual(x, pos_enc, mask=mla_mask)
        out_chunk_mla       = sdpa_attn._forward_chunk_sdpa(x, pos_enc, mask=mla_mask)
        s = error_stats(out_manual_mla, out_chunk_mla)
        all_pass &= print_stats("mla_mask:   manual vs chunk_sdpa", s, atol_fp32)

        # (d) _forward_sdpa vs chunk_sdpa (マスクなし: 完全等価のはず)
        out_sdpa_nomask     = sdpa_attn._forward_sdpa(x, pos_enc)
        out_chunk_nomask2   = sdpa_attn._forward_chunk_sdpa(x, pos_enc)
        s = error_stats(out_sdpa_nomask, out_chunk_nomask2)
        all_pass &= print_stats("マスクなし: _forward_sdpa vs chunk_sdpa", s, atol_fp32)

    return all_pass


# ──────────────────────────────────────────────────────────────────────────
# 2. フルモデル forward の検証
# ──────────────────────────────────────────────────────────────────────────


def verify_full_model(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> bool:
    """CC_G2PnP フルモデルで use_flash_attention ON/OFF の出力を比較する。"""
    dtype_name = {torch.float32: "FP32", torch.float16: "FP16", torch.bfloat16: "BF16"}[dtype]
    print(f"\n{'─'*60}")
    print(f"  CC_G2PnP フルモデル検証  ({dtype_name})")
    print(f"  B={batch_size}  T={seq_len}  device={device}")
    print(f"{'─'*60}")

    torch.manual_seed(0)
    cfg_manual = CC_G2PnPConfig(use_flash_attention=False)
    cfg_sdpa   = CC_G2PnPConfig(use_flash_attention=True)

    model_manual = CC_G2PnP(cfg_manual).to(device).eval()
    model_sdpa   = CC_G2PnP(cfg_sdpa).to(device).eval()
    model_sdpa.load_state_dict(model_manual.state_dict())

    if dtype != torch.float32:
        model_manual = model_manual.to(dtype)
        model_sdpa   = model_sdpa.to(dtype)

    input_ids    = torch.randint(0, cfg_manual.bpe_vocab_size, (batch_size, seq_len), device=device)
    input_lengths = torch.full((batch_size,), seq_len, device=device)

    # FP32: 1e-4, FP16/BF16: 1e-2
    atol = 1e-4 if dtype == torch.float32 else 1e-2

    all_pass = True
    with torch.no_grad():
        out_manual = model_manual(input_ids, input_lengths)
        out_sdpa   = model_sdpa(input_ids, input_lengths)

        # log_probs の比較
        lp_manual = out_manual["log_probs"]  # [B, T*8, V]
        lp_sdpa   = out_sdpa["log_probs"]

        s = error_stats(lp_manual.float(), lp_sdpa.float())
        all_pass &= print_stats(f"log_probs ({dtype_name})", s, atol)

        # encoder の中間出力を直接比較 (encoder forward を再呼び出し)
        x_m = model_manual.embedding(input_ids)
        x_s = model_sdpa.embedding(input_ids)
        enc_m = model_manual.encoder(x_m, input_lengths)
        enc_s = model_sdpa.encoder(x_s, input_lengths)
        for i, (il_m, il_s) in enumerate(
            zip(enc_m["intermediate_logits"], enc_s["intermediate_logits"], strict=True)
        ):
            s = error_stats(il_m.float(), il_s.float())
            all_pass &= print_stats(f"intermediate_logits[{i}] ({dtype_name})", s, atol)

    return all_pass


# ──────────────────────────────────────────────────────────────────────────
# 3. ストリーミング推論の検証
# ──────────────────────────────────────────────────────────────────────────


def verify_streaming(
    batch_size: int,
    device: torch.device,
) -> bool:
    """streaming forward が use_flash_attention に依存しないことを確認する。"""
    print(f"\n{'─'*60}")
    print("  ストリーミング推論検証")
    print(f"  B={batch_size}  device={device}")
    print(f"{'─'*60}")

    torch.manual_seed(7)
    cfg_manual = CC_G2PnPConfig(use_flash_attention=False)
    cfg_sdpa   = CC_G2PnPConfig(use_flash_attention=True)

    model_manual = CC_G2PnP(cfg_manual).to(device).eval()
    model_sdpa   = CC_G2PnP(cfg_sdpa).to(device).eval()
    model_sdpa.load_state_dict(model_manual.state_dict())

    chunk_size = cfg_manual.chunk_size
    mla_size   = cfg_manual.mla_size
    d_model    = cfg_manual.d_model
    n_chunks   = 4

    state_m = model_manual.init_streaming_state(batch_size)
    state_s = model_sdpa.init_streaming_state(batch_size)

    all_pass = True
    atol = 1e-5

    with torch.no_grad():
        for t in range(n_chunks):
            frames = torch.randn(batch_size, chunk_size + mla_size, d_model, device=device)
            lp_m, state_m = model_manual.forward_streaming(frames, state_m)
            lp_s, state_s = model_sdpa.forward_streaming(frames, state_s)
            s = error_stats(lp_m, lp_s)
            all_pass &= print_stats(f"chunk {t}: log_probs", s, atol)

    return all_pass


# ──────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SDPA ON/OFF 数値精度検証")
    parser.add_argument("--batch-size",   type=int, default=2)
    parser.add_argument("--seq-len",      type=int, default=10,
                        help="入力トークン長 (アップサンプル前)")
    parser.add_argument("--d-model",      type=int, default=64)
    parser.add_argument("--num-heads",    type=int, default=4)
    parser.add_argument("--chunk-size",   type=int, default=3)
    parser.add_argument("--past-context", type=int, default=2)
    parser.add_argument("--mla-size",     type=int, default=1)
    parser.add_argument("--device",       type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print("  SDPA ON/OFF 数値精度検証スクリプト")
    print(f"  device={device}")
    print(f"{'='*60}")

    results: list[bool] = []

    # 1. ChunkAwareAttention 単体
    results.append(verify_attention_module(
        batch_size=args.batch_size,
        seq_len=args.seq_len * 8,  # アップサンプル後のフレーム数をシミュレート
        d_model=args.d_model,
        num_heads=args.num_heads,
        chunk_size=args.chunk_size,
        past_context=args.past_context,
        mla_size=args.mla_size,
        device=device,
    ))

    # 2. フルモデル FP32
    results.append(verify_full_model(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device,
        dtype=torch.float32,
    ))

    # 3. フルモデル BF16 (CUDA 使用時のみ)
    if device.type == "cuda":
        results.append(verify_full_model(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
            dtype=torch.bfloat16,
        ))

    # 4. ストリーミング推論
    results.append(verify_streaming(
        batch_size=args.batch_size,
        device=device,
    ))

    # ── 最終サマリ ──
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'='*60}")
    print(f"  最終結果: {n_pass}/{n_total} PASS")
    if n_pass == n_total:
        print("  ✓ 全テスト通過: SDPA ON/OFF は数値的に等価です")
    else:
        print("  ✗ 一部テスト失敗: 上記の FAIL 項目を確認してください")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
