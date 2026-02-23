"""CC-G2PnP 訓練スクリプト (エントリポイント: cc_g2pnp.cli)。

使用例:
    uv run python scripts/train.py
    uv run python scripts/train.py --max-steps 1000 --lr 1e-4
    torchrun --nproc_per_node=4 scripts/train.py --ddp
"""

from cc_g2pnp.cli import main

if __name__ == "__main__":
    main()
