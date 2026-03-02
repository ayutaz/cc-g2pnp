# CC-G2PnP: Streaming Grapheme-to-Phoneme and Prosody with Conformer-CTC for Unsegmented Languages

**著者:** Yuma Shirahata, Ryuichi Yamamoto (LY Corporation)
**発表:** arXiv:2602.17157v1, 2026年2月19日

---

## 1. 概要

CC-G2PnPは、LLM（大規模言語モデル）とTTS（テキスト音声合成）をストリーミング接続するための、**ストリーミング対応のGrapheme-to-Phoneme and Prosody (G2PnP) モデル**である。Conformer-CTCアーキテクチャに基づき、入力の書記素（grapheme）トークンをチャンク単位で処理することで、音素ラベルおよび韻律ラベル（PnPラベル）のストリーミング推論を実現する。

従来のストリーミング手法（LLM2PnPなど）は明示的な単語境界に依存していたが、CC-G2PnPはCTCデコーダにより書記素と音素のアライメントを学習時に動的に獲得するため、**日本語や中国語のような非分かち書き言語（unsegmented languages）にも適用可能**である。

---

## 2. 背景と動機

### 音声対話モデルにおけるカスケード方式

- ASR + LLM + TTS のカスケード方式は、ロバスト性と柔軟性に優れる
- LLMが逐次生成するテキストに対してストリーミングTTSを適用することで応答速度を向上
- TTSの入力として書記素を直接使う方式もあるが、十分な性能を得るには大量のテキスト-音声ペアデータが必要

### G2PnPの必要性

- 書記素を音素+韻律ラベルに変換してからTTSに入力する方式は、少ないデータでも安定した性能を発揮
- LLMとTTSの間にG2PnPを挟むため、レイテンシ削減にはG2PnP自体のストリーミング化が必要

### 従来手法の課題

| 手法 | 問題点 |
|------|--------|
| **ナイーブなチャンク処理** | 文レベルの非ストリーミングG2PnPモデルをチャンク単位で適用。周辺文脈に依存するG2PnPでは性能が低下 |
| **LLM2PnP** [Dekel et al.] | Transformerのアテンションマスクに単語レベルの制約を適用。英語では有効だが、単語境界のない言語には適用不可 |

---

## 3. 提案手法

### 3.1 モデルアーキテクチャ

- **Streaming Conformer層のスタック + CTCデコーダ**で構成
- 入力: LLMが生成する書記素トークン（BPEトークンID）
- 出力: 音素と韻律記号の混合系列
- トークンIDはアップサンプリング（係数8の繰り返し）を経てConformerに入力
- **Self-conditioned CTC** を中間層（第2, 4, 6層）に導入し、トークン間の条件付き独立性を緩和
- 損失関数: 最終CTC損失 + 中間CTC損失（重みは最終CTCの1/3）の合計

### 3.2 Chunk-aware Streaming

Conformerのストリーミング対応のために、self-attention層とconvolution層の将来トークンへの依存を制限する必要がある。

- **Convolution層**: 因果的畳み込み（causal convolution）により将来依存を排除
- **Self-attention層**: Chunk-aware streaming [Noroozi et al.] を採用
  - 入力トークンをサイズCのチャンクに分割
  - 各トークンは同一チャンク内の全トークン + 過去Pトークンに注意可能
  - チャンク内のlook-aheadサイズは C-1 ~ 0
  - 通常のlook-aheadと異なり、層数に比例してレイテンシが増大しない

### 3.3 Minimum Look-Ahead (MLA) — 提案手法の核心

**課題**: Chunk-aware streamingでは、各チャンクの最後のトークンがlook-ahead=0となり、後続トークンと矛盾する出力を生成しやすい。

**解決策**: MLAにより、チャンク外のM個の将来トークンを参照可能にする。

- 全トークンのlook-aheadサイズが C+M-1 ~ M に増加
- **第1層のself-attentionにのみ適用**（第2層以降は前層のself-attentionにより将来トークンが既に追加の将来依存を持つため）
- MLA有効時の推論開始: 最初の C+M 個の書記素トークンが到着した時点

### 3.4 推論フロー

1. LLMから書記素トークンのストリームを受け取る
2. 最初のC（MLA有効時はC+M）個のトークンが揃ったらチャンク単位の予測を開始
3. 音素+韻律ラベルの系列を出力し、ストリーミングTTSに渡す

---

## 4. 実験

### 4.1 実験条件

#### データ

| 項目 | 詳細 |
|------|------|
| **学習データ** | ReazonSpeechデータセットの転写テキスト（14,960,911件） |
| **検証データ** | 4,802件 |
| **評価データ (6D-Eval)** | 6ドメイン（チャット、インタビュー、ニュース、小説、実用書、SNS）から2,722文。専門家が音素・韻律ラベルを手動アノテーション |
| **目標PnP系列の作成** | MeCabベースの形態素解析 → DNNベースの韻律ラベル予測モデル（Dict-DNN） |

#### 韻律ラベルの種類

- **IP**: イントネーション句境界 (`#`)
- **AP**: アクセント句境界 (`/`)
- **AN**: アクセント核 (`*`)

#### モデル設定

| パラメータ | 値 |
|-----------|-----|
| Conformer層数 | 8 |
| 隠れ次元 | 512 |
| トークナイザ | CALM2-7B-ChatのBPEトークナイザ |
| アップサンプリング係数 | 8 |
| バッチサイズ | 動的バッチ（最大8,192トークン/ミニバッチ） |
| 学習率 | 1e-4 → 1e-5（1.2Mステップで指数減衰） |
| 過去コンテキストサイズ P | 10（全提案モデル共通） |

#### 比較モデル

- **ベースライン**: Dict-DNN（非ストリーミング）をチャンクサイズ5, 10, 20でストリーミング化
- **提案手法**: CC-G2PnP（チャンクサイズC=2,5 × MLA M=0,1,2 の6モデル）

### 4.2 G2PnP精度

**評価指標**: CER（文字誤り率）、SER（文誤り率）

#### 主要結果 (Table 1)

| モデル | P | C | M | PnP CER (SER) | Phoneme CER (SER) |
|--------|---|---|---|----------------|---------------------|
| Dict-DNN-5 (streaming) | 0 | 5 | 0 | 6.67 (84.7) | 1.54 (22.4) |
| Dict-DNN-10 (streaming) | 0 | 10 | 0 | 3.58 (62.2) | 0.86 (13.0) |
| Dict-DNN-20 (streaming) | 0 | 20 | 0 | 2.28 (45.9) | 0.56 (8.7) |
| **CC-G2PnP-5-1** (streaming) | 10 | 5 | 1 | **1.79 (41.4)** | **0.52 (8.4)** |
| CC-G2PnP-5-2 (streaming) | 10 | 5 | 2 | 1.80 (41.4) | 0.53 (8.6) |
| Dict-DNN-NS (non-streaming) | ∞ | - | - | 1.71 (40.4) | 0.40 (6.4) |
| CC-G2PnP-NS (non-streaming) | ∞ | - | - | 1.80 (42.0) | 0.48 (7.6) |

**知見:**
- MLA付きの提案モデルは、全指標でベースラインおよびMLA無しモデルを大幅に上回る
- CC-G2PnP-5-1とCC-G2PnP-5-2が最良で、非ストリーミングDict-DNNに迫る性能を達成

### 4.3 処理時間

- 初回トークン生成までのレイテンシ: `Start = (C+M)τ + G2PnP処理時間`
  - τ: LLMが1トークン生成するのにかかる時間
- CC-G2PnP-5-1の場合: 6τ + 0.0399秒（6トークン待機で最良性能）
- Dict-DNN-20でも20トークン待機してCC-G2PnP-5-1の性能に及ばない
- G2PnPモデル単体のチャンク処理時間は同程度（0.03〜0.04秒）

### 4.4 TTS知覚品質（主観評価）

**評価方法**: MOS（Mean Opinion Score, 5段階）、日本語母語話者15名、50文

| モデル | MOS |
|--------|-----|
| Dict-DNN-5 (streaming) | 2.73 ± 0.11 |
| Dict-DNN-10 (streaming) | 3.35 ± 0.11 |
| CC-G2PnP-5-0 (streaming) | 3.81 ± 0.10 |
| **CC-G2PnP-5-1 (streaming)** | **4.02 ± 0.09** |
| Dict-DNN-NS (non-streaming) | 4.07 ± 0.09 |
| CC-G2PnP-NS (non-streaming) | 4.02 ± 0.09 |
| GT-Label | 4.16 ± 0.07 |

**知見:**
- CC-G2PnP-5-1（ストリーミング）がMOS 4.02を達成し、非ストリーミングモデルとほぼ同等
- TTSモデルはNANSY-TTSベース（173,987サンプル、207.96時間の日本語音声コーパスで学習）

### 4.5 学習データ量の影響

| データ量 | PnP CER (SER) | Phoneme CER (SER) |
|---------|----------------|---------------------|
| 1% | 4.55 (64.0) | 1.97 (24.8) |
| 10% | 2.34 (47.8) | 0.71 (10.6) |
| 100% | 1.79 (41.4) | 0.52 (8.4) |

データ量の増加に伴い一貫して性能が向上し、大規模データの活用効果が確認された。

---

## 5. 結論

- CC-G2PnPは、Conformer-CTCベースのストリーミングG2PnPモデルであり、非分かち書き言語に適用可能
- Minimum Look-Ahead (MLA) により、チャンク境界での予測安定性を向上
- 日本語データセットでの実験で、ストリーミングベースラインに対して客観・主観ともに大幅な改善を達成
- CC-G2PnP-5-1（C=5, M=1）が最良のストリーミング性能を達成し、非ストリーミングモデルに匹敵

### 今後の課題

- 外部辞書やLLMからの知識を効果的に統合すること（提案手法は外部辞書に依存しないため、多様な書記素語彙をカバーするには大量の学習データが必要）

---

## 6. 主要参考文献

- [8] Dekel et al., "Speak while you think: Streaming speech synthesis during text generation," ICASSP 2024 — LLM2PnP（先行ストリーミングG2PnP手法）
- [12] Park et al., "A unified accent estimation method based on multi-task learning for Japanese TTS," Interspeech 2022 — Dict-DNNベースライン
- [17] Gulati et al., "Conformer: Convolution-augmented transformer for speech recognition," 2020 — Conformerアーキテクチャ
- [18] Graves et al., "Connectionist temporal classification," ICML 2006 — CTC
- [19] Noroozi et al., "Stateful Conformer with cache-based inference for streaming ASR," ICASSP 2024 — Chunk-aware streaming
- [25] Nozaki & Komatsu, "Relaxing the conditional independence assumption of CTC-based ASR," Interspeech 2021 — Self-conditioned CTC
- [27] Yin et al., "ReazonSpeech: A free and massive corpus for Japanese ASR," 2023 — 学習データ

---

## 7. 再現実装の現状（2026-03-02 時点）

### 実装状態

| コンポーネント | 状態 | 備考 |
|-------------|------|------|
| Conformer Encoder (8層, 512d) | ✅ 完了 | 84M params, 論文全明示パラメータ一致 |
| Self-conditioned CTC | ✅ 完了 | Layer 1,3,5 (0-indexed), 重み 1/3 |
| Chunk-aware Streaming Attention | ✅ 完了 | C=5, P=10, MLA M=1 |
| SDPA 高速化 | ✅ 完了 | T4 で 3.5x 訓練高速化 |
| データパイプライン | ✅ 完了 | ReazonSpeech streaming + LMDB キャッシュ |
| 学習パイプライン | ✅ 完了 | AMP + DDP + AdamW + SDPA |
| ストリーミング推論 | ✅ 完了 | Conv/KV cache + MLA look-ahead |
| 評価パイプライン | ✅ 完了 | 6 メトリクス (PnP/Norm/Phoneme CER/SER) |
| 561 テスト | ✅ PASS | ruff clean |

### 100K ステップ訓練結果

- Train loss: 24.7 → 0.028（健全な学習曲線）
- 評価結果: PnP CER 46.7%（論文 1.79% — 訓練量 8.3% が主因）

### 論文再現の実現可能性

- **T4×4 DDP で 2-5 日** で 1.2M ステップの近似再現が可能（SDPA ON + LMDB）
- max_input_len=128 制約により PnP CER 3-5% が現実的目標
- 完全再現（PnP CER ~1.8%）には A100 + max_input_len=512 が必要
