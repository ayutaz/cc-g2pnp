"""CC-G2PnP 訓練スクリプト。

使用例:
    # デフォルト設定で訓練
    uv run python scripts/train.py

    # カスタム設定
    uv run python scripts/train.py --max-steps 1000 --lr 1e-4 --checkpoint-dir ./ckpt

    # DDP (マルチGPU)
    torchrun --nproc_per_node=4 scripts/train.py --ddp
"""

from __future__ import annotations

import argparse
import os

from cc_g2pnp.model import CC_G2PnPConfig
from cc_g2pnp.training import Trainer, TrainingConfig


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CC-G2PnP: Streaming G2PnP with Conformer-CTC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Optimizer
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--final-lr", type=float, default=1e-5, help="Final learning rate for exponential decay")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay coefficient")
    parser.add_argument(
        "--betas", type=float, nargs=2, default=[0.9, 0.98],
        metavar=("BETA1", "BETA2"), help="AdamW beta coefficients",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm for clipping")

    # Scheduler
    parser.add_argument("--total-steps", type=int, default=1_200_000, help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=10_000, help="Linear warmup steps")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total_steps for debug/small runs")
    parser.add_argument("--scheduler-type", type=str, default="cosine",
                        choices=["exponential", "cosine"],
                        help="LR scheduler type: exponential (ExponentialLR) or cosine (CosineAnnealingLR)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = max_tokens * accum_steps)")

    # Data
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum BPE tokens per batch")
    parser.add_argument("--max-input-len", type=int, default=64,
                        help="Maximum BPE token length per sample (ReazonSpeech P99=45, 64 covers 99.9%%)")
    parser.add_argument("--dataset-subset", type=str, default="all", help="ReazonSpeech dataset subset name")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker processes for parallel preprocessing")
    parser.add_argument("--mp-context", type=str, default="forkserver",
                        choices=["fork", "forkserver", "spawn"],
                        help="Multiprocessing start method for DataLoader workers")
    parser.add_argument("--prefetch-count", type=int, default=4, help="Number of batches to prefetch in background")

    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=10_000, help="Save checkpoint every N steps")
    parser.add_argument("--keep-last", type=int, default=5, help="Number of most recent checkpoints to keep")
    parser.add_argument("--no-async-checkpoint", action="store_false", dest="async_checkpoint",
                        help="Disable async checkpoint saving")
    parser.set_defaults(async_checkpoint=True)

    # LMDB cache
    parser.add_argument("--lmdb-cache-dir", type=str, default=None,
                        help="LMDB directory with pre-computed PnP labels")

    # Local dataset
    parser.add_argument("--local-dataset-dir", type=str, default=None,
                        help="Local Parquet dataset directory (from download_text.py)")

    # Logging
    parser.add_argument("--log-every", type=int, default=100, help="Log metrics every N steps")
    parser.add_argument("--project-name", type=str, default="cc-g2pnp", help="Project name for experiment tracking")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for experiment tracking")

    # AMP (default: enabled)
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable automatic mixed precision")
    parser.set_defaults(amp=True)
    parser.add_argument(
        "--amp-dtype", type=str, default="float16", choices=["float16", "bfloat16"],
        help="AMP data type (float16 recommended for T4; bfloat16 only for Ampere+)"
    )

    # DDP
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel training")

    # SDPA
    parser.add_argument("--use-flash-attention", action="store_true",
                        help="Enable SDPA backend (Memory-Efficient on T4, recommended with --amp-dtype float16)")

    # Validation
    parser.add_argument("--val-every", type=int, default=5_000, help="Run validation every N steps")

    # Transfer learning
    parser.add_argument("--pretrained-weights-only", action="store_true",
                        help="Load only model weights from checkpoint (reset optimizer/scheduler for transfer learning)")

    # Performance optimizations
    parser.add_argument("--use-torch-compile", action="store_true",
                        help="Apply torch.compile to FFN and ConvModule for kernel fusion (15-25%% speedup)")
    parser.add_argument("--no-gradient-checkpointing", action="store_false", dest="gradient_checkpointing",
                        help="Disable gradient checkpointing (faster backward, more memory)")
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument("--sort-batch-buffer", type=int, default=10_000,
                        help="Buffer size for length-sorted batching (0 to disable)")
    parser.add_argument("--disable-intermediate-ctc-after", type=int, default=None,
                        help="Disable intermediate CTC after N steps to save compute")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    from dotenv import load_dotenv

    from cc_g2pnp._patch_pyopenjtalk import apply as _patch_pyopenjtalk

    load_dotenv()
    # CUDA メモリフラグメンテーション低減
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _patch_pyopenjtalk()

    args = parse_args()

    if args.ddp:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        rank = 0
        world_size = 1

    model_config = CC_G2PnPConfig(use_flash_attention=args.use_flash_attention)

    training_config = TrainingConfig(
        learning_rate=args.lr,
        final_learning_rate=args.final_lr,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        max_grad_norm=args.max_grad_norm,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_tokens_per_batch=args.max_tokens,
        max_input_len=args.max_input_len,
        dataset_subset=args.dataset_subset,
        num_workers=args.num_workers,
        multiprocessing_context=args.mp_context,
        prefetch_count=args.prefetch_count,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_steps=args.save_every,
        keep_last_n=args.keep_last,
        log_every_n_steps=args.log_every,
        project_name=args.project_name,
        run_name=args.run_name,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        use_ddp=args.ddp,
        val_every_n_steps=args.val_every,
        seed=args.seed,
        lmdb_cache_dir=args.lmdb_cache_dir,
        local_dataset_dir=args.local_dataset_dir,
        async_checkpoint=args.async_checkpoint,
        scheduler_type=args.scheduler_type,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        pretrained_weights_only=args.pretrained_weights_only,
        use_torch_compile=args.use_torch_compile,
        gradient_checkpointing=args.gradient_checkpointing,
        sort_batch_buffer_size=args.sort_batch_buffer,
        disable_intermediate_ctc_after=args.disable_intermediate_ctc_after,
    )

    trainer = Trainer(model_config, training_config, rank=rank, world_size=world_size)
    trainer.train()


if __name__ == "__main__":
    main()
