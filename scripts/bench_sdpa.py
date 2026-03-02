"""SDPA ON/OFF 学習速度ベンチマーク。

合成データを用いて実際のネットワーク・W&B なしで
use_flash_attention ON/OFF の学習ステップ速度を比較する。

使用例:
    uv run python scripts/bench_sdpa.py
    uv run python scripts/bench_sdpa.py --steps 30 --max-input-len 128
"""

from __future__ import annotations

import argparse
import time

import torch

from cc_g2pnp.model import CC_G2PnP, CC_G2PnPConfig
from cc_g2pnp.training.config import TrainingConfig
from cc_g2pnp.training.optimizer import build_optimizer, build_scheduler

# ──────────────────────────────────────────────────────────────────────────
# 合成バッチ生成
# ──────────────────────────────────────────────────────────────────────────


def make_synthetic_batch(
    batch_size: int,
    max_input_len: int,
    bpe_vocab_size: int,
    pnp_vocab_size: int,
    device: torch.device,
    seed: int = 0,
) -> dict:
    """合成バッチを作成する。"""
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)

    # 各サンプルの実長を均一に分布させる
    input_lengths = torch.randint(
        max_input_len // 2, max_input_len + 1,
        (batch_size,), generator=rng,
    )
    max_len = int(input_lengths.max().item())

    input_ids = torch.randint(0, bpe_vocab_size, (batch_size, max_len), generator=rng)

    # ターゲット長: アップサンプル後フレーム数以下に収める
    upsample_factor = 8
    min_tgt_len = 4
    max_tgt_len = min(64, int(input_lengths.min().item()) * upsample_factor)
    target_lengths = torch.randint(
        min_tgt_len, max_tgt_len + 1, (batch_size,), generator=rng
    )
    max_tgt = int(target_lengths.max().item())
    targets = torch.randint(1, pnp_vocab_size, (batch_size, max_tgt), generator=rng)

    return {
        "input_ids": input_ids.to(device),
        "input_lengths": input_lengths.to(device),
        "targets": targets.to(device),
        "target_lengths": target_lengths.to(device),
    }


# ──────────────────────────────────────────────────────────────────────────
# 1 ステップ実行
# ──────────────────────────────────────────────────────────────────────────


def run_one_step(
    model: CC_G2PnP,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_ctx,
    device: torch.device,
) -> float:
    """1 訓練ステップを実行して損失値を返す。"""
    optimizer.zero_grad(set_to_none=True)

    with amp_ctx:
        out = model(
            input_ids=batch["input_ids"],
            input_lengths=batch["input_lengths"],
            targets=batch["targets"],
            target_lengths=batch["target_lengths"],
        )
        loss: torch.Tensor = out["loss"]

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return loss.item()


# ──────────────────────────────────────────────────────────────────────────
# ベンチマーク本体
# ──────────────────────────────────────────────────────────────────────────


def benchmark(
    use_flash: bool,
    steps: int,
    warmup: int,
    batch_size: int,
    max_input_len: int,
    amp_dtype: str,
    device: torch.device,
) -> dict:
    """指定設定でベンチマークを実行し結果を返す。"""
    label = "SDPA ON (Flash)" if use_flash else "SDPA OFF (Manual)"
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  steps={steps}, warmup={steps-warmup}→{warmup}, "
          f"batch_size={batch_size}, max_input_len={max_input_len}, "
          f"amp={amp_dtype}")
    print("="*60)

    # モデル構築
    config = CC_G2PnPConfig(use_flash_attention=use_flash)
    model = CC_G2PnP(config).to(device)
    model.set_gradient_checkpointing(True)
    model.train()

    # オプティマイザ
    total_steps_cfg = max(steps * 10, 10_001)  # warmup_steps=10000 を超える必要がある
    training_config = TrainingConfig(
        total_steps=total_steps_cfg,
        warmup_steps=min(10_000, total_steps_cfg - 1),
        use_amp=True,
        amp_dtype=amp_dtype,
    )
    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    # AMP
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    scaler = torch.amp.GradScaler(enabled=(amp_dtype == "float16" and device.type == "cuda"))

    # 合成バッチを複数用意 (ステップごとに異なるシードで)
    batches = [
        make_synthetic_batch(
            batch_size=batch_size,
            max_input_len=max_input_len,
            bpe_vocab_size=config.bpe_vocab_size,
            pnp_vocab_size=config.pnp_vocab_size,
            device=device,
            seed=i,
        )
        for i in range(steps)
    ]

    step_times: list[float] = []
    losses: list[float] = []

    for step in range(steps):
        batch = batches[step]

        # CUDA タイミング
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        loss_val = run_one_step(model, batch, optimizer, scaler, amp_ctx, device)
        scheduler.step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        losses.append(loss_val)

        phase = "warmup" if step < (steps - warmup) else "bench "
        print(f"  [{phase}] step {step+1:3d}: {elapsed*1000:.1f} ms  loss={loss_val:.4f}")

        if step >= (steps - warmup):
            step_times.append(elapsed)

    # GPU メモリ
    mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

    import statistics
    mean_ms = statistics.mean(step_times) * 1000
    std_ms = statistics.stdev(step_times) * 1000 if len(step_times) > 1 else 0.0

    print(f"\n  結果 ({label}):")
    print(f"    計測ステップ数: {len(step_times)}")
    print(f"    平均ステップ時間: {mean_ms:.1f} ms")
    print(f"    標準偏差:        {std_ms:.1f} ms")
    print(f"    最大 GPU メモリ: {mem_gb:.2f} GB")

    del model, optimizer, scheduler, batches
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    return {
        "label": label,
        "use_flash": use_flash,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "mem_gb": mem_gb,
        "step_times": step_times,
    }


# ──────────────────────────────────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SDPA ON/OFF ベンチマーク")
    parser.add_argument("--steps", type=int, default=25, help="総ステップ数 (warmup含む)")
    parser.add_argument("--warmup", type=int, default=5, help="計測に含めるステップ数")
    parser.add_argument("--batch-size", type=int, default=4, help="バッチサイズ")
    parser.add_argument("--max-input-len", type=int, default=64,
                        help="最大入力長 (T4: 128 推奨, OOM 注意)")
    parser.add_argument("--amp-dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"], help="AMP データ型")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        torch.cuda.reset_peak_memory_stats(device)

    kwargs = dict(
        steps=args.steps,
        warmup=args.warmup,
        batch_size=args.batch_size,
        max_input_len=args.max_input_len,
        amp_dtype=args.amp_dtype,
        device=device,
    )

    # パターン A: SDPA OFF
    result_off = benchmark(use_flash=False, **kwargs)

    # パターン B: SDPA ON
    result_on = benchmark(use_flash=True, **kwargs)

    # ──── 比較サマリ ────
    print("\n" + "="*60)
    print("  ベンチマーク比較サマリ")
    print("="*60)
    speedup = result_off["mean_ms"] / result_on["mean_ms"]
    mem_diff = result_on["mem_gb"] - result_off["mem_gb"]
    print(f"  SDPA OFF: {result_off['mean_ms']:.1f} ± {result_off['std_ms']:.1f} ms/step")
    print(f"  SDPA ON:  {result_on['mean_ms']:.1f} ± {result_on['std_ms']:.1f} ms/step")
    print(f"  速度向上率:   {speedup:.3f}x  (SDPA ON / SDPA OFF)")
    print(f"  メモリ差分:   {mem_diff:+.2f} GB  (SDPA ON - SDPA OFF)")
    print()
    if speedup > 1.0:
        print(f"  ✓ SDPA ON は SDPA OFF より {(speedup-1)*100:.1f}% 高速")
    else:
        print(f"  ✗ SDPA ON は SDPA OFF より {(1/speedup-1)*100:.1f}% 低速")
    print("="*60)


if __name__ == "__main__":
    main()
