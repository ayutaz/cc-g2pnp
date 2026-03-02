# CLAUDE.md

CC-G2PnP: ストリーミング対応 Conformer-CTC ベースの日本語 G2PnP モデル再実装。

## 技術スタック

- Python 3.13, uv, hatchling
- PyTorch (CUDA 12.8 index, `tool.uv.sources` で設定済み)

## コマンド

```bash
uv sync                              # 依存インストール
uv run pytest                         # テスト実行 (561 件)
uv run pytest tests/test_xxx.py       # 単一ファイルテスト
uv run pytest tests/test_xxx.py -k "test_name"  # 単一テスト
uv run ruff check                     # lint
uv run ruff check --fix               # lint 自動修正

# SDPA 有効化 (推奨: T4で3.5x高速化)
uv run python scripts/train.py --use-flash-attention --amp-dtype float16

# フルスケール訓練 (T4×4 DDP, 推定2-5日)
torchrun --nproc_per_node=4 scripts/train.py --ddp --use-flash-attention --amp-dtype float16 --lmdb-cache-dir /data/pnp_cache
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

## テスト

- pytest markers: `slow`, `network`
- `uv run pytest -m "not slow and not network"` でネットワーク不要テストのみ実行
- 561 テスト (Phase 1-5 + FlashAttention SDPA + Phase 2-opt + SDPA速度修正)

## アーキテクチャ

5 モジュール構成:

1. **data** — 語彙 (140 トークン), PnP ラベラー, CALM2 トークナイザ, ReazonSpeech データセット, collator, LMDB キャッシュ (`lmdb_cache.py`)
2. **model** — CC_G2PnP (Conformer encoder + CTC head), 84M params, SDPA 対応 (`use_flash_attention` フラグ, 全系列SDPA `_forward_sdpa` デフォルト, T4で3.5x高速化), GroupNorm オプション (`use_groupnorm` フラグ)
3. **training** — Trainer, CheckpointManager (非同期保存 `async_checkpoint` フラグ, `model_config` 不一致警告), DDP (勾配同期最適化), AMP (fused AdamW), W&B logger; データ取得ネットワークエラー自動リトライ付き
4. **inference** — StreamingInference (Conv cache + KV cache), レイテンシ計測
5. **evaluation** — 6 種メトリクス (PnP CER/SER, Normalized, Phoneme), EvaluationPipeline (FP16 autocast + 長さソートバッチング + `torch.compile` オプション `use_compile` フラグ)

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
