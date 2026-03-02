"""torch.profiler + cuda.Event で訓練ステップのボトルネックを特定。

合成データを使用してネットワーク・W&B なしで実行できる。
forward / backward / optimizer の時間割合、および
torch.profiler による GPU kernel レベルの詳細分析を提供する。

使用例:
    # 基本計測 (cuda.Event 分解 + torch.profiler)
    uv run python scripts/profile_training.py

    # Gradient Checkpointing OFF との比較
    uv run python scripts/profile_training.py --no-grad-ckpt

    # max_tokens=2048 相当のバッチサイズで計測
    uv run python scripts/profile_training.py --batch-size 32 --max-input-len 64

    # TensorBoard 可視化
    tensorboard --logdir /tmp/prof_cc_g2pnp
"""

from __future__ import annotations

import argparse
import statistics

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from cc_g2pnp.model import CC_G2PnP, CC_G2PnPConfig
from cc_g2pnp.training.config import TrainingConfig
from cc_g2pnp.training.optimizer import build_optimizer, build_scheduler

# ─────────────────────────────────────────────────────────────────────────────
# 合成バッチ生成 (bench_sdpa.py と同様の方式)
# ─────────────────────────────────────────────────────────────────────────────


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

    input_lengths = torch.randint(
        max_input_len // 2, max_input_len + 1,
        (batch_size,), generator=rng,
    )
    max_len = int(input_lengths.max().item())
    input_ids = torch.randint(0, bpe_vocab_size, (batch_size, max_len), generator=rng)

    upsample_factor = 8
    min_tgt_len = 4
    max_tgt_len = min(64, int(input_lengths.min().item()) * upsample_factor)
    target_lengths = torch.randint(
        min_tgt_len, max_tgt_len + 1, (batch_size,), generator=rng,
    )
    max_tgt = int(target_lengths.max().item())
    targets = torch.randint(1, pnp_vocab_size, (batch_size, max_tgt), generator=rng)

    return {
        "input_ids": input_ids.to(device),
        "input_lengths": input_lengths.to(device),
        "targets": targets.to(device),
        "target_lengths": target_lengths.to(device),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: cuda.Event で forward/backward/optimizer を個別計測
# ─────────────────────────────────────────────────────────────────────────────


def run_with_events(
    model: torch.nn.Module,
    batch: dict,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    amp_ctx,
    device: torch.device,
) -> dict:
    """torch.cuda.Event で forward/backward/optimizer を個別計測する。

    Returns:
        fwd_ms, bwd_ms, opt_ms, total_ms, loss の辞書
    """
    e_start = torch.cuda.Event(enable_timing=True)
    e_fwd = torch.cuda.Event(enable_timing=True)
    e_bwd = torch.cuda.Event(enable_timing=True)
    e_opt = torch.cuda.Event(enable_timing=True)

    optimizer.zero_grad(set_to_none=True)

    e_start.record()
    with amp_ctx:
        out = model(
            input_ids=batch["input_ids"],
            input_lengths=batch["input_lengths"],
            targets=batch["targets"],
            target_lengths=batch["target_lengths"],
        )
        loss = out["loss"]
    e_fwd.record()

    if scaler.is_enabled():
        scaler.scale(loss).backward()
    else:
        loss.backward()
    e_bwd.record()

    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    e_opt.record()

    torch.cuda.synchronize(device)

    return {
        "fwd_ms": e_start.elapsed_time(e_fwd),
        "bwd_ms": e_fwd.elapsed_time(e_bwd),
        "opt_ms": e_bwd.elapsed_time(e_opt),
        "total_ms": e_start.elapsed_time(e_opt),
        "loss": loss.item(),
    }


def phase1_event_timing(
    model: torch.nn.Module,
    batches: list[dict],
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    amp_ctx,
    device: torch.device,
    warmup: int,
) -> list[dict]:
    """Phase 1: cuda.Event で forward/backward/optimizer の時間分解を行う。"""
    print("\n" + "=" * 65)
    print("  Phase 1: cuda.Event — forward / backward / optimizer 分解")
    print("=" * 65)
    print(f"  {'step':>4}  {'phase':6}  {'total':>8}  {'fwd':>8}  {'bwd':>8}  {'opt':>8}  loss")
    print("  " + "-" * 63)

    results = []
    for step, batch in enumerate(batches):
        m = run_with_events(model, batch, optimizer, scaler, amp_ctx, device)
        scheduler.step()

        phase = "warmup" if step < warmup else "bench "
        print(
            f"  {step+1:4d}  {phase}  "
            f"{m['total_ms']:7.1f}ms  "
            f"{m['fwd_ms']:7.1f}ms  "
            f"{m['bwd_ms']:7.1f}ms  "
            f"{m['opt_ms']:7.1f}ms  "
            f"{m['loss']:.4f}"
        )
        if step >= warmup:
            results.append(m)

    if results:
        print("\n  --- 平均 (benchmark steps のみ) ---")
        for key in ["fwd_ms", "bwd_ms", "opt_ms", "total_ms"]:
            vals = [r[key] for r in results]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0.0
            pct = mean / statistics.mean([r["total_ms"] for r in results]) * 100
            print(f"  {key:10s}: {mean:7.1f} ± {std:5.1f} ms  ({pct:4.1f}%)")

        total_means = [r["total_ms"] for r in results]
        print(f"\n  総合スループット: {statistics.mean(total_means):.1f} ms/step")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: torch.profiler で GPU kernel レベルの詳細分析
# ─────────────────────────────────────────────────────────────────────────────


def phase2_profiler(
    model: torch.nn.Module,
    batches: list[dict],
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    amp_ctx,
    device: torch.device,
    profile_dir: str,
) -> None:
    """Phase 2: torch.profiler で top GPU kernel とメモリ使用量を計測する。

    schedule: wait=1, warmup=2, active=3
    T4 での注意:
    - with_stack=False: True にすると stack unwinding で数倍遅くなる
    - profile_memory=True: schedule で active 期間のみ記録
    - DDP 使用時は rank 0 のみ有効にすること
    """
    print("\n" + "=" * 65)
    print(f"  Phase 2: torch.profiler  (出力先: {profile_dir})")
    print("  schedule: wait=1, warmup=2, active=3")
    print("=" * 65)

    schedule = torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # T4: True だとプロファイラ自体が数倍遅くなる
    ) as prof:
        for step in range(6):  # wait=1 + warmup=2 + active=3 = 6
            batch = batches[step % len(batches)]

            optimizer.zero_grad(set_to_none=True)

            with record_function("forward"), amp_ctx:
                    out = model(
                        input_ids=batch["input_ids"],
                        input_lengths=batch["input_lengths"],
                        targets=batch["targets"],
                        target_lengths=batch["target_lengths"],
                    )
                    loss = out["loss"]

            with record_function("backward"):
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            with record_function("optimizer_step"):
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            scheduler.step()
            prof.step()

    # コンソールにトップ kernel を表示
    print("\n  --- Top 20 CUDA ops (self_cuda_time_total) ---")
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    ))

    print(f"\n  TensorBoard トレース: {profile_dir}")
    print(f"  可視化コマンド: tensorboard --logdir {profile_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: データ前処理待ち時間の簡易計測
# ─────────────────────────────────────────────────────────────────────────────


def phase3_data_loading_overhead(
    model: torch.nn.Module,
    batches: list[dict],
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    amp_ctx,
    device: torch.device,
) -> None:
    """Phase 3: H2D 転送と GPU 計算の分離でデータ前処理待ちを計測。

    既に pin_memory=True でプリフェッチ済みの場合、H2D 転送はほぼ無視できるはず。
    """
    print("\n" + "=" * 65)
    print("  Phase 3: H2D 転送時間 vs GPU 計算時間")
    print("=" * 65)

    h2d_times = []
    compute_times = []

    for step, batch in enumerate(batches[:5]):
        # H2D 転送時間を計測
        # (合成バッチはすでに device 上にあるため、ここでは CPU -> GPU を模擬)
        cpu_batch = {
            k: v.cpu() for k, v in batch.items()
        }

        e_h2d_start = torch.cuda.Event(enable_timing=True)
        e_h2d_end = torch.cuda.Event(enable_timing=True)
        e_compute_end = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize(device)
        e_h2d_start.record()

        gpu_batch = {
            k: v.to(device, non_blocking=True) for k, v in cpu_batch.items()
        }
        torch.cuda.synchronize(device)  # non_blocking の完了を待つ
        e_h2d_end.record()

        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            out = model(
                input_ids=gpu_batch["input_ids"],
                input_lengths=gpu_batch["input_lengths"],
                targets=gpu_batch["targets"],
                target_lengths=gpu_batch["target_lengths"],
            )
            loss = out["loss"]

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        e_compute_end.record()
        scheduler.step()
        torch.cuda.synchronize(device)

        h2d_ms = e_h2d_start.elapsed_time(e_h2d_end)
        compute_ms = e_h2d_end.elapsed_time(e_compute_end)
        total_ms = e_h2d_start.elapsed_time(e_compute_end)

        h2d_times.append(h2d_ms)
        compute_times.append(compute_ms)

        print(
            f"  step {step+1}: "
            f"H2D={h2d_ms:.2f}ms  compute={compute_ms:.1f}ms  "
            f"total={total_ms:.1f}ms  "
            f"H2D占有率={h2d_ms/total_ms*100:.1f}%"
        )

    if h2d_times:
        print(f"\n  平均 H2D: {statistics.mean(h2d_times):.2f} ms")
        print(f"  平均 compute: {statistics.mean(compute_times):.1f} ms")
        h2d_ratio = statistics.mean(h2d_times) / (
            statistics.mean(h2d_times) + statistics.mean(compute_times)
        ) * 100
        print(f"  H2D 占有率: {h2d_ratio:.1f}%")
        if h2d_ratio < 5:
            print("  -> H2D 転送はボトルネックではない (正常)")
        else:
            print("  -> H2D 転送がボトルネックの可能性あり! pin_memory を確認")


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="CC-G2PnP 訓練ボトルネック計測")
    parser.add_argument("--steps", type=int, default=12,
                        help="Phase 1 の総ステップ数 (warmup 含む, デフォルト: 12)")
    parser.add_argument("--warmup", type=int, default=4,
                        help="Phase 1 のウォームアップステップ数 (デフォルト: 4)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="バッチサイズ (max_tokens=2048, max_input_len=64 では約32)")
    parser.add_argument("--max-input-len", type=int, default=64,
                        help="最大入力長 (デフォルト: 64, config と合わせること)")
    parser.add_argument("--amp-dtype", default="float16",
                        choices=["float16", "bfloat16"],
                        help="AMP データ型 (T4 は float16 推奨)")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="Gradient Checkpointing を無効化して比較")
    parser.add_argument("--no-sdpa", action="store_true",
                        help="SDPA (Flash Attention) を無効化して比較")
    parser.add_argument("--profile-dir", default="/tmp/prof_cc_g2pnp",
                        help="torch.profiler トレース出力ディレクトリ")
    parser.add_argument("--skip-profiler", action="store_true",
                        help="Phase 2 の torch.profiler をスキップ (Phase 1 のみ実行)")
    parser.add_argument("--skip-data-phase", action="store_true",
                        help="Phase 3 の H2D 計測をスキップ")
    parser.add_argument("--device", default="cuda:0",
                        help="使用デバイス (デフォルト: cuda:0)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA が利用できません。CPU で実行します (非常に遅い)")

    print("=" * 65)
    print("  CC-G2PnP 訓練ボトルネック計測")
    print("=" * 65)
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            print(f"  GPU Memory: {result.stdout.strip()}")
        except Exception:
            pass

    use_sdpa = not args.no_sdpa
    use_grad_ckpt = not args.no_grad_ckpt
    print(f"  SDPA (Flash Attention): {'ON' if use_sdpa else 'OFF'}")
    print(f"  Gradient Checkpointing: {'ON' if use_grad_ckpt else 'OFF'}")
    print(f"  AMP dtype: {args.amp_dtype}")
    print(f"  batch_size={args.batch_size}, max_input_len={args.max_input_len}")

    # モデル構築
    model_config = CC_G2PnPConfig(use_flash_attention=use_sdpa)
    model = CC_G2PnP(model_config).to(device)
    if use_grad_ckpt:
        model.set_gradient_checkpointing(True)
    model.train()

    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params/1e6:.1f}M")

    # Optimizer / Scheduler
    total_steps_cfg = max(args.steps * 10, 10_001)
    tcfg = TrainingConfig(
        total_steps=total_steps_cfg,
        warmup_steps=min(10_000, total_steps_cfg - 1),
        use_amp=True,
        amp_dtype=args.amp_dtype,
    )
    optimizer = build_optimizer(model, tcfg)
    scheduler = build_scheduler(optimizer, tcfg)

    # AMP
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    amp_ctx = torch.amp.autocast(device_type=device.type, dtype=dtype)
    scaler = torch.amp.GradScaler(enabled=(args.amp_dtype == "float16" and device.type == "cuda"))

    print(f"\n  fused AdamW: {optimizer.param_groups[0].get('fused', False)}")
    print(f"  GradScaler enabled: {scaler.is_enabled()}")

    # 合成バッチ作成
    total_batches = max(args.steps, 12)
    batches = [
        make_synthetic_batch(
            args.batch_size, args.max_input_len,
            model_config.bpe_vocab_size, model_config.pnp_vocab_size,
            device, seed=i,
        )
        for i in range(total_batches)
    ]
    # バッチの実際のトークン数を表示
    sample_batch = batches[0]
    tokens = sample_batch["input_ids"].numel()
    print(f"  batch tokens: {tokens} (batch_size={args.batch_size} × max_input_len≈{tokens//args.batch_size})")

    # GPU メモリリセット
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ─── Phase 1: cuda.Event 個別計測 ───
    phase1_results = phase1_event_timing(
        model, batches[:args.steps], optimizer, scheduler,
        scaler, amp_ctx, device, warmup=args.warmup,
    )

    # GPU メモリ使用量
    if device.type == "cuda":
        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        mem_reserved_gb = torch.cuda.memory_reserved(device) / 1e9
        print(f"\n  GPU メモリ: 確保={mem_gb:.2f}GB, 予約={mem_reserved_gb:.2f}GB")

    # ─── Phase 2: torch.profiler (skip オプション) ───
    if not args.skip_profiler:
        # Phase 2 用に別モデルを作成してリセット
        model2 = CC_G2PnP(model_config).to(device)
        if use_grad_ckpt:
            model2.set_gradient_checkpointing(True)
        model2.train()
        opt2 = build_optimizer(model2, tcfg)
        sch2 = build_scheduler(opt2, tcfg)

        phase2_profiler(
            model2, batches, opt2, sch2, scaler, amp_ctx, device, args.profile_dir,
        )

        del model2, opt2, sch2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ─── Phase 3: H2D 転送時間 ───
    if not args.skip_data_phase:
        # Phase 3 用モデル
        model3 = CC_G2PnP(model_config).to(device)
        if use_grad_ckpt:
            model3.set_gradient_checkpointing(True)
        model3.train()
        opt3 = build_optimizer(model3, tcfg)
        sch3 = build_scheduler(opt3, tcfg)

        phase3_data_loading_overhead(
            model3, batches, opt3, sch3, scaler, amp_ctx, device,
        )

        del model3, opt3, sch3
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ─── 最終サマリ ───
    print("\n" + "=" * 65)
    print("  最終サマリ")
    print("=" * 65)
    if phase1_results:
        fwd_mean = statistics.mean([r["fwd_ms"] for r in phase1_results])
        bwd_mean = statistics.mean([r["bwd_ms"] for r in phase1_results])
        opt_mean = statistics.mean([r["opt_ms"] for r in phase1_results])
        total_mean = statistics.mean([r["total_ms"] for r in phase1_results])

        print(f"  forward:        {fwd_mean:7.1f} ms  ({fwd_mean/total_mean*100:4.1f}%)")
        print(f"  backward:       {bwd_mean:7.1f} ms  ({bwd_mean/total_mean*100:4.1f}%)")
        print(f"  optimizer_step: {opt_mean:7.1f} ms  ({opt_mean/total_mean*100:4.1f}%)")
        print(f"  total:          {total_mean:7.1f} ms  (100.0%)")
        print()

        if bwd_mean / total_mean > 0.60:
            print("  [!] backward が 60% 超 -> gradient checkpointing が支配的")
            print("      -> gradient_checkpointing=False + バッチサイズ削減 を検討")
        elif bwd_mean / fwd_mean > 2.5:
            print("  [!] bwd/fwd 比率が 2.5x 超 -> checkpointing の再計算コストが大きい")
        if opt_mean / total_mean > 0.15:
            print("  [!] optimizer_step が 15% 超 -> fused AdamW の有効化を確認")
        if fwd_mean / total_mean > 0.50:
            print("  [!] forward が 50% 超 -> Conformer 計算が支配的 (SDPA/compile 効果大)")

    if device.type == "cuda":
        mem_final_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"\n  最大 GPU メモリ使用量: {mem_final_gb:.2f} GB")


if __name__ == "__main__":
    main()
