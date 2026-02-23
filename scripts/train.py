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
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm for clipping")

    # Scheduler
    parser.add_argument("--total-steps", type=int, default=1_200_000, help="Total training steps")
    parser.add_argument("--warmup-steps", type=int, default=10_000, help="Linear warmup steps")
    parser.add_argument("--max-steps", type=int, default=None, help="Override total_steps for debug/small runs")

    # Data
    parser.add_argument("--max-tokens", type=int, default=8192, help="Maximum BPE tokens per batch")
    parser.add_argument("--dataset-subset", type=str, default="all", help="ReazonSpeech dataset subset name")

    # Checkpoint
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=10_000, help="Save checkpoint every N steps")
    parser.add_argument("--keep-last", type=int, default=5, help="Number of most recent checkpoints to keep")

    # Logging
    parser.add_argument("--log-every", type=int, default=100, help="Log metrics every N steps")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--no-tensorboard", action="store_false", dest="tensorboard", help="Disable TensorBoard logging"
    )
    parser.add_argument("--project-name", type=str, default="cc-g2pnp", help="Project name for experiment tracking")
    parser.add_argument("--run-name", type=str, default=None, help="Run name for experiment tracking")

    # AMP
    parser.add_argument("--amp", action="store_true", default=True, help="Enable automatic mixed precision")
    parser.add_argument("--no-amp", action="store_false", dest="amp", help="Disable automatic mixed precision")
    parser.add_argument(
        "--amp-dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="AMP data type"
    )

    # DDP
    parser.add_argument("--ddp", action="store_true", help="Enable DistributedDataParallel training")

    # Validation
    parser.add_argument("--val-every", type=int, default=5_000, help="Run validation every N steps")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    return parser.parse_args()


def main() -> None:
    """Entry point for training."""
    args = parse_args()

    if args.ddp:
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        rank = 0
        world_size = 1

    model_config = CC_G2PnPConfig()

    training_config = TrainingConfig(
        learning_rate=args.lr,
        final_learning_rate=args.final_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        total_steps=args.total_steps,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_tokens_per_batch=args.max_tokens,
        dataset_subset=args.dataset_subset,
        checkpoint_dir=args.checkpoint_dir,
        save_every_n_steps=args.save_every,
        keep_last_n=args.keep_last,
        log_every_n_steps=args.log_every,
        use_wandb=args.wandb,
        use_tensorboard=args.tensorboard,
        project_name=args.project_name,
        run_name=args.run_name,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        use_ddp=args.ddp,
        val_every_n_steps=args.val_every,
        seed=args.seed,
    )

    trainer = Trainer(model_config, training_config, rank=rank, world_size=world_size)
    trainer.train()


if __name__ == "__main__":
    main()
