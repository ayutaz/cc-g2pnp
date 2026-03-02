# CLAUDE.md

CC-G2PnP: ストリーミング対応 Conformer-CTC ベースの日本語 G2PnP モデル再実装。

## 技術スタック

- Python 3.13, uv, hatchling
- PyTorch (CUDA 12.8 index, `tool.uv.sources` で設定済み)

## コマンド

```bash
uv sync                              # 依存インストール
uv run pytest                         # テスト実行 (688 件)
uv run pytest tests/test_xxx.py       # 単一ファイルテスト
uv run pytest tests/test_xxx.py -k "test_name"  # 単一テスト
uv run ruff check                     # lint
uv run ruff check --fix               # lint 自動修正

# SDPA 有効化 (推奨: T4で3.5x高速化)
uv run python scripts/train.py --use-flash-attention

# フルスケール訓練 (T4×4 DDP, 推定4-7日)
torchrun --nproc_per_node=4 scripts/train.py --ddp --use-flash-attention --lmdb-cache-dir /data/pnp_cache

# 最大性能 (全最適化有効)
torchrun --nproc_per_node=4 scripts/train.py --ddp --use-flash-attention --use-torch-compile --lmdb-cache-dir /data/pnp_cache --sort-batch-buffer 10000
```

## コーディング規約

- line-length: 120
- ruff ルール: E, W, F, I, N, UP, B, SIM, TCH, RUF
- ignore: E501 (行長超過), RUF001 (カタカナ文字), N801 (CC_G2PnP 命名)
- tests/ では N806 (大文字変数名) を許可

## 重要な注意点

- `pyopenjtalk-plus` を使用 (`pyopenjtalk` ではない) — API 互換フォーク
- `datasets>=2.14.0,<4.0.0` にピン留め (v4.0+ で breaking change)
- `uv run python -m unidic download` で UniDic 辞書のダウンロードが必要
- W&B (wandb) は必須 — 未ログイン時に RuntimeError
- データパイプラインはネットワークエラー時に指数バックオフで自動リトライ (最大 10 回)
- PnP ラベル LMDB キャッシュ: `scripts/preprocess_pnp.py` で事前生成し `--lmdb-cache-dir` で指定すると GPU 利用率が大幅改善
- デフォルト値: `amp-dtype=float16`, `max-input-len=64`, `num-workers=8`, `use_groupnorm=True`, `scheduler-type=cosine`
- `--use-flash-attention` は明示的に指定が必要 (model config デフォルト False)

## テスト

- pytest markers: `slow`, `network`
- `uv run pytest -m "not slow and not network"` でネットワーク不要テストのみ実行
- 688 テスト (Phase 1-5 + FlashAttention SDPA + Phase 2-opt + SDPA速度修正 + P0+P1+Triton訓練高速化)

## アーキテクチャ

5 モジュール構成:

1. **data** — 語彙 (140 トークン), PnP ラベラー, CALM2 トークナイザ, ReazonSpeech データセット + ローカルデータセット対応 (`download_text.py`), collator (`sorted_dynamic_batch_sampler` 含む), LMDB キャッシュ (`lmdb_cache.py`)
2. **model** — CC_G2PnP (Conformer encoder + CTC head), 84M params, SDPA 対応 (`--use-flash-attention` で有効化, 全系列SDPA `_forward_sdpa`, T4で3.5x高速化), Triton RPE kernel (推論専用 `triton_attention.py`, 訓練時は SDPA にフォールバック), GroupNorm デフォルト (`use_groupnorm=True`)
3. **training** — Trainer, CheckpointManager (非同期保存 `async_checkpoint`, `model_config` 不一致警告), DDP (勾配同期最適化, `bucket_cap_mb=50`), AMP (fused AdamW, amp-dtype default=float16), W&B logger; sorted batching (`--sort-batch-buffer`), torch.compile (`--use-torch-compile`, FFN+ConvModule個別), foreach grad clipping, gradient checkpointing (default=True, `--no-gradient-checkpointing` で無効化), intermediate CTC skip (`--disable-intermediate-ctc-after`), gradient accumulation (`--gradient-accumulation-steps`), cosine/exponential scheduler (`--scheduler-type`); データ取得ネットワークエラー自動リトライ付き
4. **inference** — StreamingInference (Conv cache + KV cache), レイテンシ計測, Triton RPE dispatch (推論時のみ)
5. **evaluation** — 6 種メトリクス (PnP CER/SER, Normalized, Phoneme), EvaluationPipeline (FP16 autocast + 長さソートバッチング + `torch.compile` オプション `use_compile` フラグ)

## 主要 CLI オプション (抜粋)

| オプション | デフォルト | 説明 |
|---|---|---|
| `--use-flash-attention` | (off) | SDPA 有効化 (T4 で 3.5x 高速化、推奨) |
| `--use-torch-compile` | (off) | FFN+ConvModule の torch.compile |
| `--amp-dtype` | `float16` | AMP データ型 (T4 は float16 推奨) |
| `--max-input-len` | `64` | サンプルあたり最大 BPE トークン長 |
| `--num-workers` | `8` | DataLoader ワーカー数 |
| `--scheduler-type` | `cosine` | LR scheduler (`cosine` / `exponential`) |
| `--sort-batch-buffer` | `10000` | 長さソートバッチングバッファ (0 で無効) |
| `--gradient-accumulation-steps` | `1` | 勾配累積ステップ数 |
| `--lmdb-cache-dir` | - | PnP ラベル LMDB キャッシュ |
| `--local-dataset-dir` | - | ローカル Parquet データセット |
| `--no-gradient-checkpointing` | - | gradient checkpointing 無効化 |
| `--disable-intermediate-ctc-after` | - | 指定ステップ後に中間 CTC 省略 |
| `--pretrained-weights-only` | (off) | 重みのみ復元 (transfer learning) |

## エージェントチーム

本プロジェクトではエージェントチーム (実験的機能) を有効化済み。

### 推奨チーム構成例

**コードレビューチーム:**
```
3人のレビュアーでPRをレビューして:
- セキュリティ (入力バリデーション, CUDA メモリ管理)
- パフォーマンス (テンソル演算, メモリ効率)
- テストカバレッジ (エッジケース, マーカー整合性)
```

**機能実装チーム:**
```
エージェントチームで新機能を並列実装して:
- モジュール担当者ごとにファイルを分離すること
- 各チームメンバーに Sonnet を使用してコスト最適化
```

**デバッグ調査チーム:**
```
5人のエージェントで異なる仮説を並列調査して。
互いの理論に反論する科学的議論形式で。
```

### 注意事項

- tmux 未インストールのため in-process モードで動作 (`--teammate-mode in-process`)
- 同一ファイルの同時編集を避けるため、モジュール単位でタスクを分割すること
- チームメンバーはこの CLAUDE.md を自動的に読み込む
- チーム完了後は必ずリーダーがクリーンアップすること

## 言語

日本語で応答すること。
