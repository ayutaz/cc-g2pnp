# CLAUDE.md

CC-G2PnP: ストリーミング対応 Conformer-CTC ベースの日本語 G2PnP モデル再実装。

## 技術スタック

- Python 3.13, uv, hatchling
- PyTorch (CUDA 12.8 index, `tool.uv.sources` で設定済み)

## コマンド

```bash
uv sync                              # 依存インストール
uv run pytest                         # テスト実行 (470 件)
uv run pytest tests/test_xxx.py       # 単一ファイルテスト
uv run pytest tests/test_xxx.py -k "test_name"  # 単一テスト
uv run ruff check                     # lint
uv run ruff check --fix               # lint 自動修正
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

## テスト

- pytest markers: `slow`, `network`
- `uv run pytest -m "not slow and not network"` でネットワーク不要テストのみ実行
- 470 テスト (Phase 1-5)

## アーキテクチャ

5 モジュール構成:

1. **data** — 語彙 (140 トークン), PnP ラベラー, CALM2 トークナイザ, ReazonSpeech データセット, collator
2. **model** — CC_G2PnP (Conformer encoder + CTC head), 84M params
3. **training** — Trainer, CheckpointManager, DDP, AMP, W&B logger
4. **inference** — StreamingInference (Conv cache + KV cache), レイテンシ計測
5. **evaluation** — 6 種メトリクス (PnP CER/SER, Normalized, Phoneme), EvaluationPipeline

## 言語

日本語で応答すること。
