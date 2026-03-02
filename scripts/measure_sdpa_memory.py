"""SDPA ON/OFF GPU メモリ使用量比較スクリプト。

異なるシーケンス長・バッチサイズ・AMP 設定で
use_flash_attention=True/False のピーク GPU メモリを計測する。

使用例:
    uv run python scripts/measure_sdpa_memory.py
    uv run python scripts/measure_sdpa_memory.py --batch-size 8 --device cuda:0
"""

from __future__ import annotations

import argparse

import torch

from cc_g2pnp.model import CC_G2PnP, CC_G2PnPConfig

# ──────────────────────────────────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────────────────────────────────


def reset_and_empty(device: torch.device) -> None:
    """GPU メモリ統計をリセットしてキャッシュを解放する。"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def measure_forward_memory(
    use_flash: bool,
    batch_size: int,
    seq_len: int,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    seed: int = 0,
) -> dict[str, float]:
    """1 回の forward pass でのピーク GPU メモリを計測する。

    Returns:
        Dict with 'peak_mb' and 'allocated_mb'.
    """
    reset_and_empty(device)

    cfg = CC_G2PnPConfig(use_flash_attention=use_flash)
    model = CC_G2PnP(cfg).to(device).eval()

    torch.manual_seed(seed)
    input_ids    = torch.randint(0, cfg.bpe_vocab_size, (batch_size, seq_len), device=device)
    input_lengths = torch.full((batch_size,), seq_len, device=device)

    # ウォームアップ
    with torch.no_grad():
        if amp_dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                _ = model(input_ids, input_lengths)
        else:
            _ = model(input_ids, input_lengths)

    reset_and_empty(device)

    # 計測
    with torch.no_grad():
        if amp_dtype is not None:
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                _ = model(input_ids, input_lengths)
        else:
            _ = model(input_ids, input_lengths)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    peak_bytes      = torch.cuda.max_memory_allocated(device)
    allocated_bytes = torch.cuda.memory_allocated(device)

    del model, input_ids, input_lengths
    reset_and_empty(device)

    return {
        "peak_mb":      peak_bytes      / 1e6,
        "allocated_mb": allocated_bytes / 1e6,
    }


def measure_backward_memory(
    use_flash: bool,
    batch_size: int,
    seq_len: int,
    amp_dtype: torch.dtype | None,
    device: torch.device,
    seed: int = 0,
) -> dict[str, float]:
    """1 回の forward + backward pass でのピーク GPU メモリを計測する。"""
    reset_and_empty(device)

    cfg = CC_G2PnPConfig(use_flash_attention=use_flash)
    model = CC_G2PnP(cfg).to(device).train()

    torch.manual_seed(seed)
    input_ids    = torch.randint(0, cfg.bpe_vocab_size, (batch_size, seq_len), device=device)
    input_lengths = torch.full((batch_size,), seq_len, device=device)

    upsample_factor = cfg.upsample_factor
    target_len_max  = max(4, seq_len * upsample_factor // 2)
    target_lengths  = torch.full((batch_size,), target_len_max, device=device)
    targets         = torch.randint(1, cfg.pnp_vocab_size, (batch_size, target_len_max), device=device)

    # ウォームアップ
    if amp_dtype is not None:
        scaler  = torch.amp.GradScaler(enabled=(amp_dtype == torch.float16))
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        with amp_ctx:
            out  = model(input_ids, input_lengths, targets, target_lengths)
            loss = out["loss"]
        scaler.scale(loss).backward()
    else:
        out  = model(input_ids, input_lengths, targets, target_lengths)
        loss = out["loss"]
        loss.backward()

    model.zero_grad(set_to_none=True)
    reset_and_empty(device)

    # 計測
    if amp_dtype is not None:
        with amp_ctx:
            out  = model(input_ids, input_lengths, targets, target_lengths)
            loss = out["loss"]
        scaler.scale(loss).backward()
    else:
        out  = model(input_ids, input_lengths, targets, target_lengths)
        loss = out["loss"]
        loss.backward()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    peak_bytes      = torch.cuda.max_memory_allocated(device)
    allocated_bytes = torch.cuda.memory_allocated(device)

    del model, input_ids, input_lengths, targets, target_lengths
    reset_and_empty(device)

    return {
        "peak_mb":      peak_bytes      / 1e6,
        "allocated_mb": allocated_bytes / 1e6,
    }


# ──────────────────────────────────────────────────────────────────────────
# ベンチマーク本体
# ──────────────────────────────────────────────────────────────────────────


def run_benchmark(
    batch_size: int,
    seq_lens: list[int],
    amp_dtypes: list[tuple[str, torch.dtype | None]],
    device: torch.device,
    include_backward: bool,
) -> None:
    """全設定の組み合わせでメモリを計測し比較表を出力する。"""
    if device.type != "cuda":
        print("警告: CUDA デバイスが見つかりません。CPU では GPU メモリ計測は不可能です。")
        print("      '--device cuda:0' を指定して再実行してください。")
        return

    print(f"\n{'='*70}")
    print(f"  GPU メモリ使用量比較  (batch_size={batch_size})")
    print(f"  デバイス: {torch.cuda.get_device_name(device)}")
    print(f"{'='*70}")

    for amp_label, amp_dtype in amp_dtypes:
        dtype_name = amp_label if amp_label else "FP32"
        print(f"\n{'─'*70}")
        print(f"  AMP: {dtype_name}  |  forward {'+ backward' if include_backward else 'のみ'}")
        print(f"  {'T':>6}  {'SDPA OFF':>12}  {'SDPA ON':>12}  {'削減量':>12}  {'削減率':>8}")
        print(f"  {'':>6}  {'(peak MB)':>12}  {'(peak MB)':>12}  {'(MB)':>12}  {'(%)':>8}")
        print(f"  {'─'*56}")

        for seq_len in seq_lens:
            fn = measure_backward_memory if include_backward else measure_forward_memory
            r_off = fn(False, batch_size, seq_len, amp_dtype, device)
            r_on  = fn(True,  batch_size, seq_len, amp_dtype, device)

            peak_off  = r_off["peak_mb"]
            peak_on   = r_on["peak_mb"]
            reduction = peak_off - peak_on
            pct       = (reduction / peak_off * 100) if peak_off > 0 else 0.0
            sign      = "+" if reduction < 0 else "-"
            mark      = "✓" if reduction >= 0 else "✗"

            print(f"  {seq_len:>6}  {peak_off:>12.1f}  {peak_on:>12.1f}"
                  f"  {sign}{abs(reduction):>10.1f}  {mark}{abs(pct):>6.1f}%")

    print(f"\n{'='*70}")
    print("  ※ 削減量: SDPA OFF - SDPA ON  (正値 = SDPA ON が有利)")
    print(f"{'='*70}\n")


# ──────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SDPA ON/OFF GPU メモリ比較")
    parser.add_argument("--batch-size",      type=int, default=4)
    parser.add_argument("--seq-lens",        type=int, nargs="+",
                        default=[64, 128, 256, 512],
                        help="計測するシーケンス長のリスト")
    parser.add_argument("--device",          type=str, default="cuda:0")
    parser.add_argument("--no-backward",     action="store_true",
                        help="forward のみ計測 (backward を省略)")
    parser.add_argument("--fp32-only",       action="store_true",
                        help="FP32 のみ計測 (AMP を省略)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    amp_dtypes: list[tuple[str, torch.dtype | None]] = [("FP32", None)]
    if not args.fp32_only:
        amp_dtypes += [
            ("BF16", torch.bfloat16),
            ("FP16", torch.float16),
        ]

    run_benchmark(
        batch_size=args.batch_size,
        seq_lens=args.seq_lens,
        amp_dtypes=amp_dtypes,
        device=device,
        include_backward=not args.no_backward,
    )


if __name__ == "__main__":
    main()
