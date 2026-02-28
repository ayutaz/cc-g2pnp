# CC-G2PnP

ストリーミング対応 Conformer-CTC ベースの日本語 Grapheme-to-Phoneme and Prosody (G2PnP) モデルの再実装。

## 概要

[CC-G2PnP](https://arxiv.org/abs/2602.17157) (Shirahata & Yamamoto, LY Corporation, 2026) の PyTorch 再実装です。BPE トークン列から音素・韻律記号列を CTC で予測するモデルで、チャンク単位のストリーミング推論に対応しています。デフォルト設定で約 84M パラメータ。

## 特徴

- **ストリーミング推論** — チャンク単位の逐次処理で低遅延な音素・韻律予測
- **DDP マルチ GPU 学習** — `torchrun` によるデータ並列分散学習
- **AMP (float16 / bfloat16)** — 混合精度学習によるメモリ効率化
- **Fused AdamW** — CUDA 利用時に fused カーネルで optimizer ステップを高速化
- **バックグラウンドプリフェッチ** — 専用スレッドによるバッチ先読みでデータ転送を並列化
- **マルチプロセス音素解析** — DataLoader ワーカーごとに独立した OpenJTalk インスタンスで真の並列化
- **FP16 推論** — 評価パイプラインで CUDA autocast による高速推論
- **SDPA (Scaled Dot-Product Attention)** — `F.scaled_dot_product_attention` による EFFICIENT_ATTENTION カーネル活用 (`use_flash_attention` フラグで切替)
- **チャンク分割 Attention** — O(T^2) → O(T×C) メモリ削減によるチャンク単位処理 (FlashAttention Phase 2)
- **PnP ラベル事前キャッシュ (LMDB)** — `scripts/preprocess_pnp.py` で事前処理し学習時 GPU 利用率を大幅改善
- **torch.compile (推論)** — 推論パイプラインへの `torch.compile` 適用 (`use_compile` フラグで切替)
- **GroupNorm オプション** — DDP 通信削減と fp16 安定性向上 (`use_groupnorm` フラグで切替)
- **非同期チェックポイント保存** — 保存スパイクを排除 (`async_checkpoint` フラグで切替)
- **長さソートバッチング** — 評価時にパディング量を削減するシーケンス長ソート
- **W&B ロギング** — 学習メトリクスの自動記録・可視化
- **6 種メトリクス評価** — PnP CER/SER、Normalized PnP CER/SER、Phoneme CER/SER

## プロジェクト構成

```
cc_g2pnp/
├── __init__.py
├── cli.py                  # 学習 CLI エントリポイント
├── _patch_pyopenjtalk.py   # pyopenjtalk 互換パッチ
├── data/                   # データパイプライン
│   ├── vocabulary.py       #   PnP 語彙 (140 トークン)
│   ├── pnp_labeler.py      #   pyopenjtalk による音素・韻律ラベル生成
│   ├── tokenizer.py        #   CALM2 BPE トークナイザ
│   ├── dataset.py          #   ReazonSpeech ストリーミングデータセット
│   ├── collator.py         #   動的バッチ collator
│   └── lmdb_cache.py       #   PnP ラベル LMDB キャッシュ
├── model/                  # モデルアーキテクチャ
│   ├── config.py           #   CC_G2PnPConfig
│   ├── cc_g2pnp.py         #   CC_G2PnP (トップレベルモデル)
│   ├── encoder.py          #   ConformerEncoder + ストリーミング状態
│   ├── conformer_block.py  #   ConformerBlock
│   ├── attention.py        #   ChunkAwareAttention (MLA マスク)
│   ├── convolution.py      #   ConformerConvModule (GLU)
│   ├── feed_forward.py     #   FeedForwardModule
│   ├── embedding.py        #   TokenEmbedding
│   ├── positional_encoding.py  # RelativePositionalEncoding
│   └── ctc_decoder.py      #   CTCHead + greedy_decode
├── training/               # 学習ループ
│   ├── config.py           #   TrainingConfig
│   ├── trainer.py          #   Trainer (AMP, GradScaler, DDP)
│   ├── optimizer.py        #   AdamW + LinearLR warmup + ExponentialLR
│   ├── checkpoint.py       #   CheckpointManager (atomic save)
│   ├── logger.py           #   TrainingLogger (W&B)
│   ├── evaluator.py        #   Evaluator (PnP CER)
│   └── distributed.py      #   DDP ユーティリティ
├── inference/              # 推論
│   ├── streaming.py        #   StreamingInference
│   └── latency.py          #   レイテンシ計測
├── evaluation/             # 評価パイプライン
│   ├── metrics.py          #   6 種メトリクス
│   ├── eval_data.py        #   EvalDataset + 組み込みテスト文
│   └── pipeline.py         #   EvaluationPipeline
└── utils/
scripts/
├── train.py                # 学習スクリプト (cli.py のエイリアス)
├── evaluate.py             # 評価スクリプト
├── preprocess_pnp.py       # PnP ラベル事前キャッシュ生成スクリプト
└── verify_ddp.py           # DDP 設定検証スクリプト
```

## セットアップ

```bash
# 依存パッケージのインストール
uv sync

# UniDic 辞書のダウンロード (fugashi 用)
uv run python -m unidic download

# W&B ログイン
wandb login
```

> **Note**: PyTorch は CUDA 12.8 インデックスから自動的にインストールされます。

## 学習

### 単一 GPU

```bash
uv run cc-g2pnp-train
```

### DDP (マルチ GPU)

```bash
torchrun --nproc_per_node=N -m cc_g2pnp.cli --ddp
```

### 主要 CLI オプション

| オプション | デフォルト | 説明 |
|---|---|---|
| `--lr` | `1e-4` | ピーク学習率 |
| `--final-lr` | `1e-5` | 指数減衰の最終学習率 |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--betas` | `0.9 0.98` | AdamW beta 係数 |
| `--max-grad-norm` | `1.0` | 勾配クリッピング最大ノルム |
| `--total-steps` | `1,200,000` | 総学習ステップ数 |
| `--warmup-steps` | `10,000` | 線形ウォームアップステップ数 |
| `--max-steps` | - | デバッグ用ステップ数上限 |
| `--max-tokens` | `8192` | バッチあたり最大 BPE トークン数 |
| `--dataset-subset` | `all` | ReazonSpeech データセットサブセット |
| `--checkpoint-dir` | `checkpoints` | チェックポイント保存先 |
| `--save-every` | `10,000` | チェックポイント保存間隔 |
| `--keep-last` | `5` | 保持するチェックポイント数 |
| `--log-every` | `100` | メトリクスログ間隔 |
| `--val-every` | `5,000` | 検証実行間隔 |
| `--project-name` | `cc-g2pnp` | W&B プロジェクト名 |
| `--run-name` | - | W&B ラン名 |
| `--no-amp` | - | AMP を無効化 |
| `--amp-dtype` | `bfloat16` | AMP データ型 (`float16` / `bfloat16`) |
| `--ddp` | - | DDP 分散学習を有効化 |
| `--seed` | `42` | ランダムシード |
| `--max-input-len` | `512` | サンプルあたり最大 BPE トークン長 (T4: `128` 推奨) |
| `--num-workers` | `4` | DataLoader ワーカー数 |
| `--prefetch-count` | `4` | バックグラウンドでプリフェッチするバッチ数 |
| `--lmdb-cache-dir` | - | PnP ラベル LMDB キャッシュディレクトリ (`scripts/preprocess_pnp.py` で事前生成) |
| `--no-async-checkpoint` | - | 非同期チェックポイント保存を無効化 |

### T4 GPU での学習

T4 (15GB, compute capability 7.5) は bfloat16 テンソルコアを持たないため、`float16` AMP を使用してください。
デフォルトの `--max-tokens 8192` では OOM になるため、以下の設定を推奨します:

```bash
torchrun --nproc_per_node=4 scripts/train.py --ddp \
    --amp-dtype float16 \
    --max-tokens 4096 --max-input-len 128
```

### ネットワーク耐性

ReazonSpeech ストリーミング中の接続エラー (`ConnectionError`, `OSError`, `TimeoutError`) に対して、
エクスポネンシャルバックオフ付きの自動リトライ (最大 10 回、10s〜300s) が組み込まれています。

## 推論

```python
import torch
from cc_g2pnp.model import CC_G2PnP, CC_G2PnPConfig
from cc_g2pnp.inference import StreamingInference

model = CC_G2PnP(CC_G2PnPConfig())
model.load_state_dict(torch.load("checkpoints/step_100000.pt")["model_state_dict"])
model.eval()

engine = StreamingInference(model)
state = engine.reset(batch_size=1)

# BPE トークン列を逐次処理
bpe_ids = torch.tensor([[101, 202, 303]])
labels, state = engine.process_tokens(bpe_ids, state)

# 残りのバッファをフラッシュ
flush_labels, _ = engine.flush(state)
```

## 評価

```python
from cc_g2pnp.evaluation import EvalConfig, EvalDataGenerator, EvaluationPipeline

# チェックポイントからパイプライン構築
pipeline = EvaluationPipeline.from_checkpoint(
    "checkpoints/step_100000.pt",
    config=EvalConfig(device="cuda"),
)

# 組み込みテスト文で評価
generator = EvalDataGenerator()
dataset = generator.builtin_dataset()
result = pipeline.evaluate(dataset)

print(pipeline.format_results(result))
```

## テスト

```bash
# 全テスト実行 (529 件)
uv run pytest

# lint チェック
uv run ruff check
```

## ドキュメント

- [CC-G2PnP 論文まとめ](docs/CC-G2PnP_論文まとめ.md)
- [技術調査](docs/CC-G2PnP_技術調査.md)
- [実装ロードマップ](docs/CC-G2PnP_実装ロードマップ.md)
- [論文再現性調査](docs/CC-G2PnP_論文再現性調査.md)
- [FlashAttention 導入調査](docs/FlashAttention_導入調査.md)
- [最適化調査レポート](docs/最適化調査レポート.md)
