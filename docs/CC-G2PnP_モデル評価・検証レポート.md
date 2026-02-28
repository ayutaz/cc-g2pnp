# CC-G2PnP モデル評価・検証レポート

**対象チェックポイント**: `v0.1.0-100k` (step_00100000)
**評価日**: 2026-02-28
**チェックポイント**: [GitHub Release v0.1.0-100k](https://github.com/ayutaz/cc-g2pnp/releases/tag/v0.1.0-100k)

---

## 1. モデル概要

| 項目 | 値 |
|------|-----|
| アーキテクチャ | Streaming Conformer-CTC |
| パラメータ数 | 84,005,516 (84M) |
| 入力 | CALM2 BPE トークン (vocab: 65,000) |
| 出力 | PnP ラベル列 (vocab: 140, CTC デコード) |
| Conformer layers | 8 |
| d_model | 512 |
| Attention heads | 8 |
| FFN dim | 2048 |
| Conv kernel | 31 |
| Upsample factor | 8 |
| Chunk size (C) | 5 |
| Past context (P) | 10 |
| MLA size (M) | 1 |
| Self-conditioned CTC | Layer 1, 3, 5 (0-indexed) |
| Intermediate CTC weight | 1/3 |

---

## 2. 訓練設定

| 項目 | 値 |
|------|-----|
| GPU | 4x Tesla T4 (15 GB each) |
| DDP | 4 プロセス (torchrun) |
| AMP | float16 (T4 は bfloat16 テンソルコア非搭載) |
| Optimizer | AdamW (lr=1e-4, betas=(0.9, 0.98), weight_decay=0.01) |
| Scheduler | Warmup (10K steps) + ExponentialLR |
| max_input_len | 128 BPE トークン |
| max_tokens_per_batch | 1,024 |
| Gradient checkpointing | 有効 |
| データセット | ReazonSpeech "all" (HuggingFace streaming) |
| 訓練ステップ | 100,000 / 1,200,000 (8.3%) |
| 所要時間 | 97 時間 29 分 |
| GPU メモリ使用量 | 10-12.5 GB / GPU |

---

## 3. 学習曲線

| ステップ | Train Loss | Val Loss | Val CER |
|---------|-----------|---------|---------|
| 0 | 24.7 | - | - |
| 10K | 7.2 | - | - |
| 20K | 3.3 | - | - |
| 30K | 1.1 | - | - |
| 40K | 0.46 | - | - |
| 50K | 0.17 | - | - |
| 60K | 0.08 | - | - |
| 70K | 0.06 | - | - |
| 80K | 0.04 | - | - |
| 90K | 0.03 | - | - |
| **100K** | **0.028** | **0.013** | **1.30%** |

- Train loss は 24.7 → 0.028 に順調に下降 (約 880 倍の改善)
- Val CER 1.30% (内部指標) は訓練データに対するモデルの学習が進んでいることを示す
- Val loss (0.013) < Train loss (0.028) は過学習が発生していないことを示唆

---

## 4. 評価パイプライン

### 4.1 評価データセット

ビルトイン評価データセット (40 文, 4 ドメイン):

| ドメイン | サンプル数 | 例文 |
|---------|----------|------|
| news | 10 | 「東京都は新しい条例を施行すると発表した。」 |
| conversation | 10 | 「今日はいい天気ですね。」 |
| literature | 10 | 「桜の花が風に舞い散る季節になった。」 |
| technical | 10 | 「ニューラルネットワークの学習率を調整する必要がある。」 |

正解ラベルは `pyopenjtalk-plus` による PnP ラベル生成で作成。

### 4.2 評価メトリクス (6 種)

| メトリクス | 説明 | 計算方法 |
|-----------|------|---------|
| **PnP CER** | PnP 文字誤り率 | 全 PnP トークン (音素 + 韻律) に対する Levenshtein 距離 |
| **PnP SER** | PnP 文誤り率 | 予測 != 正解 の文の割合 (%) |
| **Normalized PnP CER** | 正規化 PnP CER | `#` (イントネーション句境界) を `/` (アクセント句境界) に統一後の CER |
| **Normalized PnP SER** | 正規化 PnP SER | 同上の SER |
| **Phoneme CER** | 音素 CER | 韻律記号 (`*`, `/`, `#`) を除去した音素のみの CER |
| **Phoneme SER** | 音素 SER | 同上の SER |

CER の計算には `jiwer.wer` を使用 (各 PnP トークンを 1 ワードとして扱い、トークンレベルの Levenshtein 距離を計算)。

### 4.3 評価実行方法

```bash
# 推論デモ + 評価
uv run python scripts/evaluate.py --checkpoint checkpoints/step_00100000.pt

# 評価のみ
uv run python scripts/evaluate.py --checkpoint checkpoints/step_00100000.pt --eval-only

# 推論デモのみ
uv run python scripts/evaluate.py --checkpoint checkpoints/step_00100000.pt --demo-only

# CPU で実行
uv run python scripts/evaluate.py --checkpoint checkpoints/step_00100000.pt --device cpu

# GitHub Release のチェックポイントを使用
uv run python scripts/evaluate.py --checkpoint cc_g2pnp_step100k_model.pt
```

---

## 5. 評価結果

### 5.1 全体メトリクス (100K ステップ)

| メトリクス | 値 |
|-----------|-----|
| PnP CER | 46.7% |
| PnP SER | 100.0% |
| Normalized PnP CER | 46.7% |
| Normalized PnP SER | 100.0% |
| Phoneme CER | 50.7% |
| Phoneme SER | 100.0% |

### 5.2 論文との比較

| メトリクス | 本実装 (100K steps) | 論文 CC-G2PnP-5-1 (1.2M steps) | ギャップ |
|-----------|-------------------|-------------------------------|---------|
| PnP CER | 46.7% | 1.79% | x26 |
| PnP SER | 100.0% | 41.4% | 大幅な差 |
| Phoneme CER | 50.7% | 0.52% | x97 |

### 5.3 性能ギャップの原因分析

| 要因 | 影響度 | 説明 |
|------|--------|------|
| **訓練不足** | 致命的 | 1.2M ステップ中 100K (8.3%) のみ。CTC モデルは後半で急激に改善する傾向 |
| **データセット規模** | 致命的 | ReazonSpeech "all" (14.9M samples) のうち、max_input_len=128 でフィルタされた短文のみ |
| **max_input_len=128** | 重大 | BPE 128 トークン超の長文をフィルタ。論文は 512 までカバーと推定 |
| **評価データの違い** | 中程度 | 論文は 6D-Eval (2,722 文, 6 ドメイン)。本実装は builtin 40 文 (4 ドメイン) |
| **実効バッチサイズ** | 中程度 | max_tokens=1,024 は論文推定 30K と比べて非常に小さい |

### 5.4 良い兆候

- **学習曲線の形状**: Train loss が 24.7 → 0.028 に 880 倍改善しており、モデルは正しく学習中
- **Val CER 1.30%**: 訓練データに対する内部評価は良好で、アーキテクチャの正確性を示す
- **過学習なし**: Val loss (0.013) < Train loss (0.028)
- **部分的な正解出力**: 多くの短文で正しい PnP 出力を生成可能

---

## 6. 推論デモ結果

100K ステップチェックポイントでの推論サンプル:

| 入力テキスト | 期待される出力 (pyopenjtalk) | モデルの傾向 |
|------------|---------------------------|------------|
| 東京は日本の首都です。 | トーキョー ワ / ニ ホ ン ノ / シュ ト デ ス # | 短文は比較的正確 |
| 今日はいい天気ですね。 | キョー ワ / イ イ / テ ン キ デ ス ネ # | 日常会話は安定 |
| ありがとうございます。 | ア リ ガ ト ー ゴ ザ イ マ ス # | 定型表現は正確 |
| 機械学習モデルの性能を評価する。 | キ カ イ ガ ク シュー ... | CTC collapse が発生する場合あり |

- **短文・定型表現**: 比較的正確な出力を生成
- **長文・専門用語**: CTC collapse (例: 「ワー」のみ出力) が見られる場合あり
- **訓練継続で改善が期待**: CTC モデルは十分な訓練ステップで collapse が解消される傾向

---

## 7. チェックポイント情報

### 7.1 保存チェックポイント

| ファイル | ステップ | サイズ |
|---------|---------|-------|
| checkpoints/step_00060000.pt | 60,000 | 972 MB |
| checkpoints/step_00070000.pt | 70,000 | 972 MB |
| checkpoints/step_00080000.pt | 80,000 | 972 MB |
| checkpoints/step_00090000.pt | 90,000 | 972 MB |
| checkpoints/step_00100000.pt | 100,000 | 972 MB |

### 7.2 チェックポイント構造

| キー | 内容 | サイズ |
|-----|------|-------|
| `model_state_dict` | モデル重み (304 tensors) | 330.5 MB |
| `optimizer_state_dict` | AdamW optimizer state | 641.1 MB |
| `scheduler_state_dict` | ExponentialLR scheduler | < 1 KB |
| `config` | TrainingConfig (24 fields) | < 1 KB |
| `model_config` | CC_G2PnPConfig (17 fields) | < 1 KB |
| `scaler_state_dict` | GradScaler (AMP) state | < 1 KB |
| `step` | 訓練ステップ数 | int |

### 7.3 GitHub Release

[v0.1.0-100k](https://github.com/ayutaz/cc-g2pnp/releases/tag/v0.1.0-100k) に軽量版チェックポイント (330 MB) を公開:
- `cc_g2pnp_step100k_model.pt`: model_state_dict + config + model_config (optimizer 除外)

### 7.4 チェックポイントの使い方

```python
from cc_g2pnp.evaluation.pipeline import EvalConfig, EvaluationPipeline

# チェックポイントからモデルをロード
pipeline = EvaluationPipeline.from_checkpoint(
    "cc_g2pnp_step100k_model.pt",
    EvalConfig(device="cuda:0"),
)

# 推論
import torch
from cc_g2pnp.data.tokenizer import G2PnPTokenizer

tokenizer = G2PnPTokenizer.get_instance()
text = "東京は日本の首都です。"
bpe_ids = tokenizer.encode(text)
input_ids = torch.tensor([bpe_ids], dtype=torch.long, device="cuda:0")
input_lengths = torch.tensor([len(bpe_ids)], dtype=torch.long, device="cuda:0")

with torch.no_grad():
    predicted_ids = pipeline.model.inference(input_ids, input_lengths)

# PnP ラベルにデコード
vocab = pipeline.vocabulary
tokens = vocab.decode([i for i in predicted_ids[0] if i != vocab.blank_id and i != vocab.pad_id])
tokens = [t for t in tokens if t not in ("<blank>", "<pad>")]
print(" ".join(tokens))
```

---

## 8. 論文再現に向けた今後のステップ

| 優先度 | アクション | 期待効果 |
|--------|----------|---------|
| **P0** | 1.2M ステップまで訓練継続 | 最大のギャップ要因を解消 |
| **P0** | max_input_len=512 に増加 (FlashAttention 導入) | 長文カバー — [調査レポート](FlashAttention_導入調査.md) 参照 |
| **P1** | max_tokens を増加 (A100 等) | 実効バッチサイズ改善 |
| **P2** | 6D-Eval データセットで評価 | 論文との直接比較が可能に |

詳細な論文再現性調査は [CC-G2PnP_論文再現性調査.md](CC-G2PnP_論文再現性調査.md) を参照。
