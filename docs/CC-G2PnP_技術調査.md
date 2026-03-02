# CC-G2PnP 再現実装のための技術調査ドキュメント

CC-G2PnP論文（arXiv:2602.17157, Shirahata & Yamamoto, LY Corporation, 2026年2月）の再現実装に向けた包括的な技術調査結果をまとめる。

---

## 1. 論文概要

### 1.1 目的
LLM（大規模言語モデル）が逐次生成するテキストを、ストリーミングでTTS（テキスト音声合成）に接続するための **Grapheme-to-Phoneme and Prosody（G2PnP）モデル** を提案。書記素トークンをチャンク単位で処理し、音素+韻律ラベルをストリーミング出力する。

### 1.2 主要な貢献
1. **Conformer-CTC** ベースで単語境界に依存しないG2PnP（日本語・中国語等の非分かち書き言語に適用可能）
2. **Minimum Look-Ahead (MLA)**: チャンク境界での予測安定化のため、第1層のself-attentionのみM個の将来トークンを参照
3. **Self-conditioned CTC**: 中間Conformer層（第2, 4, 6層）にCTC損失を適用し条件付き独立性を緩和

### 1.3 ベスト設定と性能

**CC-G2PnP-5-1**（C=5, M=1, P=10）がストリーミング最良:

| メトリクス | CC-G2PnP-5-1 | Dict-DNN-20 (baseline) | Dict-DNN-NS (non-streaming) |
|---|---|---|---|
| PnP CER (SER) | **1.79 (41.4)** | 2.28 (45.9) | 1.71 (40.4) |
| Norm. PnP CER (SER) | **1.28 (28.6)** | 1.72 (36.0) | 1.18 (26.4) |
| Phoneme CER (SER) | **0.52 (8.4)** | 0.56 (8.7) | 0.40 (6.4) |
| Start Latency | 6τ + 0.04s | 20τ + 0.04s | Nτ + 0.03s |
| MOS | **4.02 ± 0.09** | - | 4.07 ± 0.09 |

6トークンの待機で非ストリーミングに迫る性能を達成。

---

## 2. モデルアーキテクチャ詳細

### 2.1 全体パイプライン

```
LLM → BPEトークンID [B, T]
         ↓ Embedding(65000, 512) + Repeat Upsample(×8)
       隠れ表現 [B, T×8, 512]
         ↓ Streaming Conformer ×8層
         │   L1: Chunk-aware attention(MLA=M) + Causal Conv
         │   L2: Chunk-aware attention + Causal Conv → 中間CTC損失
         │   L3: Chunk-aware attention + Causal Conv
         │   L4: Chunk-aware attention + Causal Conv → 中間CTC損失
         │   L5: Chunk-aware attention + Causal Conv
         │   L6: Chunk-aware attention + Causal Conv → 中間CTC損失
         │   L7: Chunk-aware attention + Causal Conv
         │   L8: Chunk-aware attention + Causal Conv
         ↓
       エンコーダ出力 [B, T×8, 512]
         ↓ Linear(512, ~106) + CTC Greedy Decode
       PnPラベル系列 → Streaming TTS
```

### 2.2 Conformer層の内部構造

各Conformer層は以下のサブモジュールで構成（Gulati et al., 2020 に準拠）:

1. **Feed-Forward Module 1** (half-step residual, factor=0.5)
   - LayerNorm → Linear(512→2048) → Swish → Dropout → Linear(2048→512) → Dropout
   - 出力: `x + 0.5 * FFN1(x)`
   - FFN expansion factor = 4（512 × 4 = 2048）

2. **Multi-Head Self-Attention** (chunk-aware streaming mask)
   - 相対的正弦位置エンコーディング（Transformer-XLスタイル）※論文未記載。Conformer原論文(Gulati et al.)の標準に準拠
   - **Shaw-style PE の SDPA 対応**: `use_flash_attention=True` 時、pos_bias（位置エンコーディングによる注意重みバイアス）を `attn_mask` として `F.scaled_dot_product_attention` に渡す方式で Phase 1 実装済み。全系列 SDPA (`_forward_sdpa`) と推論時チャンク分割 SDPA (`_forward_chunk_sdpa`) の両パスで対応
   - 出力: `x + Dropout(MHSA(x))`

3. **Convolution Module** (causal depthwise conv)
   - LayerNorm → Pointwise Conv(512→1024) → GLU → Depthwise Conv1D(kernel=31) → BatchNorm/GroupNorm (`use_groupnorm` フラグで切替) → Swish → Pointwise Conv(512→512) → Dropout
   - Convolution expansion factor = 2（GLU前に2倍に拡張）
   - `use_groupnorm=True` 時: GroupNorm(32グループ, d=512) を使用。DDP の同期オーバーヘッドを削減し fp16 安定性が向上。チェックポイントは BatchNorm/GroupNorm 間で非互換

4. **Feed-Forward Module 2** (half-step residual, factor=0.5)
   - FFN1と同一構造
   - 出力: `x + 0.5 * FFN2(x)`

5. **LayerNorm** (最終正規化)

### 2.3 Chunk-aware Streaming Attention

Noroozi et al. (ICASSP 2024) "Stateful Conformer with cache-based inference" に基づく。

入力トークンをサイズCのチャンクに分割。各トークンの注意範囲:
- **同一チャンク内の全トークン**（双方向）
- **過去Pトークン**（左コンテキスト）

look-aheadサイズはチャンク内の位置に依存: C-1（先頭）〜 0（末尾）

通常のlook-aheadと異なり、**参照する将来トークン数が層数に比例して増大しない**のが利点。

**NeMoの実装**: cache-aware streaming approachとして実装済み。Activation cachingにより推論時に中間アクティベーションをキャッシュし、計算の重複を回避する。
- 対応レイヤー: Depthwise convolutions, Self-attention, Downsampling layers
- 参考: `tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb`

**SpeechBrainの実装**: Dynamic Chunk Training (DCT) として実装済み。
- 訓練時にランダムなチャンクサイズをサンプリングして複数のレイテンシ設定に対応
- `DynChunkTrainConfig` でチャンクサイズ・レイテンシを調整
- 参考: `tutorials/nn/conformer-streaming-asr.html`

### 2.4 Minimum Look-Ahead (MLA) — 論文の核心

**問題**: 各チャンクの末尾トークンはlook-ahead=0で、後続トークンと矛盾する出力を生成しやすい。

**解決**: **第1層のself-attentionのみ**、チャンク外のM個の将来トークンを参照可能にする。

- MLA適用後のlook-ahead: C+M-1（先頭）〜 M（末尾）
- 第1層のみに適用する理由: 第2層以降は前層のself-attentionにより、将来トークンが既に追加の将来依存を獲得しているため
- 推論開始に必要なトークン数: C+M個

### 2.5 Causal Convolution

Conformerの畳み込み層を因果的にすることで将来依存を排除:
- 左側のみパディング: `F.pad(x, (kernel_size-1, 0))`
- ストリーミング時はカーネルサイズ-1フレーム分の過去状態をキャッシュ

### 2.6 Self-conditioned CTC

Nozaki & Komatsu (Interspeech 2021), LINE Corporation の手法。CTCの条件付き独立性の仮定を緩和する。

**手法**:
- 中間層（第2, 4, 6層）でCTC損失を計算
- 各中間予測（語彙分布）を線形層でD次元の隠れ表現に変換し、前の層の出力に**加算**して条件付け
- 後続のエンコーダー層が中間予測の結果を考慮しながら特徴を変換可能
- 訓練時だけでなく推論時にも中間予測を使用
- 損失: `final_ctc + (1/3) × (inter_ctc_L2 + inter_ctc_L4 + inter_ctc_L6)`

**性能**: 強力な自己回帰モデル+ビームサーチと同等の性能を達成しつつ、デコード速度は30倍以上高速。

**ESPnet実装**: PR #3274 (Intermediate CTC + Stochastic depth)
- TransformerとConformerエンコーダーに対応
- 注意: マルチGPU訓練時の問題報告あり（Issue #4031）

### 2.7 Token Upsampling

BPEトークンは1つで複数の音素+韻律ラベルに対応するため、係数8で繰り返しアップサンプリング:
- 入力: `[75, 16, 72]` (3トークン)
- アップサンプリング後: `[75,75,75,75,75,75,75,75, 16,16,..., 72,72,...]` (24フレーム)
- 係数8は予備実験で決定（小さい値では1トークンに対応する全PnPラベルをカバーできない）

### 2.8 ハイパーパラメータ

| パラメータ | 値 | 備考 | 実装状態 |
|---|---|---|---|
| Conformer層数 | 8 | 論文明記 | ✅ 実装済 |
| 隠れ次元 | 512 | 論文明記 | ✅ 実装済 |
| アテンションヘッド数 | 8 | 論文未記載、Conformer標準は4だが、d_model=512では8が一般的 | ✅ 8で実装 |
| FFN expansion factor | 4 | Conformer標準 | ✅ d_ff=2048 |
| 畳み込みカーネルサイズ | 31 | 論文未記載、Conformer標準 | ✅ 31で実装 |
| Convolution expansion factor | 2 | Conformer標準（GLU前の拡張） | ✅ 実装済 |
| BPE語彙サイズ | 65,000 | CALM2-7B-Chat | ✅ 実装済 |
| PnP語彙サイズ | ~106 (論文) → **140** (実装) | 134モーラ+3韻律+blank+unk+pad | ✅ 実装済 |
| アップサンプリング係数 | 8 | 論文明記 | ✅ repeat_interleave |
| 過去コンテキスト P | 10 | 全モデル共通 | ✅ 実装済 |
| 中間CTC適用層 | 2, 4, 6 (1-indexed) | 論文明記 → 0-indexed: 1, 3, 5 | ✅ 実装済 |
| 中間CTC損失重み | 1/3 | 論文明記 | ✅ 実装済 |
| 総パラメータ数 | — | 論文未記載 | **84M** (デフォルト設定) |

---

## 3. データ・ツール調査

### 3.1 ReazonSpeechデータセット

- **概要**: 地上波テレビストリームから収集された大規模日本語音声コーパス（35,000時間以上）
- **論文での利用**: 転写テキスト部分のみ使用（音声は不要）
- **学習データ数**: 14,960,911件 / 検証データ数: 4,802件

**利用可能なsubset/config**:

| Configuration | データ量 | 音声時間 | 推定サンプル数 |
|---|---|---|---|
| `tiny` | 600MB | 8.5時間 | 約30,600 |
| `small` | 6GB | 100時間 | 約360,000 |
| `medium` | 65GB | 1,000時間 | 約360万 |
| `large` | 330GB | 5,000時間 | 約1,800万 |
| `all` | 2.3TB | 35,000時間 | 約1億2,600万 |

**データ構造** (各サンプル):
```python
{
    'name': '000/0000000000000.flac',      # ファイル識別子
    'audio': {
        'path': '/path/to/file.flac',       # FLACファイルパス
        'array': array([...], dtype=float32),# 音声波形データ
        'sampling_rate': 16000              # 16kHz固定
    },
    'transcription': '今日のニュースをお伝えします。'  # 転写テキスト
}
```

**ロード方法**:
```python
from datasets import load_dataset

# 全データロード（テキストのみ使うためstreamingが推奨）
ds = load_dataset("reazon-research/reazonspeech", "all",
                   trust_remote_code=True, streaming=True)

# 転写テキストのみ取得
for sample in ds['train']:
    text = sample['transcription']
```

**ライセンス**: CDLA-Sharing-1.0。日本の著作権法第30条の4に基づく利用のみ。
`trust_remote_code=True` が必須。

**注意**: 論文の学習データ14,960,911件は `all` の全サンプル数（約1億2,600万）より少ないため、何らかのフィルタリングが適用されている可能性がある。

### 3.2 CALM2-7B-Chat BPEトークナイザ

- **開発元**: CyberAgent
- **語彙サイズ**: 65,000（確認済み）
- **トークナイザ種類**: SentencePieceベースのBPE（GPT2スタイル）
- **ベースモデル**: Llamaアーキテクチャ（標準の32,000語彙から65,000に拡張）
- **事前学習データ**: 1.3兆トークンの日英バイリンガルデータ
- **特殊トークン**: end-of-text token = 65001

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")
tokens = tokenizer.encode("今日は良い天気", add_special_tokens=False)
# → BPEトークンID列

print(tokenizer.vocab_size)  # 65000
```

**要件**: transformers >= 4.34.1

### 3.3 形態素解析: fugashi + UniDic

論文のDict-DNNパイプラインの第1段階。推奨ツールはfugashi（CythonベースのMeCabラッパー）。

**インストール**:
```bash
# 軽量版（開発・テスト用）
pip install 'fugashi[unidic-lite]'

# 完全版（本番用）
pip install 'fugashi[unidic]'
python -m unidic download
```

**UniDic辞書の種類**:

| 辞書 | ベース | サイズ | 用途 |
|---|---|---|---|
| unidic-lite | UniDic 2.1.2 | ~250MB | 開発・テスト |
| unidic | UniDic 3.1.0 | 770MB | 本格的な処理 |
| unidic-cwj | 書き言葉版 | 609MB | 新聞・Web等 |
| **unidic-csj** | **話し言葉版** | 609MB | **ReazonSpeech（話し言葉）に最適** |

**出力フォーマット**（Named tupleで構造化アクセス）:
```python
from fugashi import Tagger

tagger = Tagger()
for word in tagger("今日は良い天気です"):
    print(word.surface)       # 表層形: 今日
    print(word.feature.pos1)  # 品詞: 名詞
    print(word.feature.kana)  # 仮名: キョウ（標準的な読み）
    print(word.feature.pron)  # 発音: キョー（長音「ー」表記）
    print(word.feature.lemma) # 語彙素: 今日
```

**プラットフォーム対応**: Linux, macOS (Intel), Windows 64bit用のwheelあり。

### 3.4 G2P変換: pyopenjtalk

Open JTalkのPythonラッパー。Ryuichi Yamamoto (r9y9) 氏が開発。

**インストール**:
```bash
pip install pyopenjtalk
```

**注意点**:
- ビルド要件: Cython, C/C++コンパイラ, CMake
- Cython 3.0との互換性問題: `--no-build-isolation` が必要な場合あり
- Windows: MSVC必要。v0.4.0以降でwheel提供改善
- v0.3.0以降: `run_marine` オプション（DNNベースの日本語アクセント推定）

**主要API**:
```python
import pyopenjtalk

# 基本G2P（ローマ字音素）
pyopenjtalk.g2p("こんにちは")
# → "k o N n i ch i w a"

# カタカナ出力
pyopenjtalk.g2p("こんにちは", kana=True)
# → "コンニチワ"

# Full-context label（韻律情報を含む詳細ラベル）
labels = pyopenjtalk.extract_fullcontext("今日は良い天気")
# → HTS形式のラベル列（アクセント型、句境界等の情報を含む）
```

**pyopenjtalkの音素セット** (約45種類):
- 母音: a, i, u, e, o（大文字A, E, I, O, Uも存在）
- 子音: k, s, t, n, h, m, y, r, w, g, z, d, b, p
- 拗音: ch, sh, ts, ky, gy, ny, hy, my, ry, py, by, dy, ty
- 特殊: N（撥音）, cl（促音）, pau（ポーズ）, sil（無音）, v, f, j

### 3.5 韻律ラベル体系

論文で使用される3種類の韻律記号:

| 記号 | 名称 | 意味 |
|---|---|---|
| `*` | AN (Accent Nucleus) | アクセント核。ピッチが高→低に下がる位置 |
| `/` | AP (Accent Phrase boundary) | アクセント句境界 |
| `#` | IP (Intonation Phrase boundary) | イントネーション句境界 |

**論文Fig.1の出力例**: `キョ * ー ワ / イ * ー / テ * ン キ ...`

### 3.6 HTS Full-context Label の仕様

pyopenjtalkの `extract_fullcontext()` が返すHTS full-context labelから韻律情報を抽出可能。

**ラベルの基本構造**:
```
p1^p2-p3+p4=p5/A:a1+a2+a3/B:b1-b2_b3/C:c1_c2+c3/D:d1+d2_d3/E:e1_e2!e3_e4-e5/F:f1_f2#f3_f4@f5_f6|f7_f8/G:g1_g2%g3_g4_g5/H:h1_h2/I:i1-i2@i3+i4&i5-i6|i7+i8/J:j1_j2/K:k1+k2-k3
```

**主要セクション**:
- **音素部分** `p1^p2-p3+p4=p5`: 前々音素^前音素-**現音素**+次音素=次々音素
- **Aセクション** `A:a1+a2+a3`: 音素レベル情報（a2=アクセント句内のモーラ位置, a3=アクセント型）
- **Fセクション**: アクセント句レベルの情報（アクセント型、モーラ位置）
- **H/Iセクション**: 呼気段落（イントネーション句）レベルの情報

**韻律記号の抽出ロジック**:
1. **アクセント核(`*`)**: Aセクションのa2（モーラ位置）がa3（アクセント型）と一致する場合
2. **アクセント句境界(`/`)**: 隣接ラベルのアクセント句位置が変化する箇所
3. **イントネーション句境界(`#`)**: 隣接ラベルのイントネーション句位置が変化、またはsil/pauの出現

**詳細仕様の参照先**: HTS-demo_NIT-ATR503-M001.tar.bz2 内の `lab_format.pdf`
- URL: http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-ATR503-M001.tar.bz2

### 3.7 ESPnetでのG2Pバリエーション

| 関数 | 出力形式 | 出力例 |
|---|---|---|
| `pyopenjtalk_g2p` | 音素列（アクセント情報なし） | `k o N n i ch i w a` |
| `pyopenjtalk_g2p_kana` | カタカナ表記 | `コンニチワ` |
| `pyopenjtalk_g2p_accent` | 音素+0/1アクセント情報 | - |
| `pyopenjtalk_g2p_prosody` | 音素+韻律記号 | `^ k o [ N n i ch i w a $` |

**pyopenjtalk_g2p_prosody の韻律記号**:
- `^`: 発話開始
- `[`: アクセント句開始
- `]`: アクセント下降
- `#`: ポーズ/区切り
- `$`: 発話終了
- `_`: 韻律境界マーカー

### 3.8 ttslearn の pp_symbols 関数

r9y9による「Pythonで学ぶ音声合成」の学習用ライブラリ。

**場所**: `ttslearn.tacotron.frontend.openjtalk`

**機能**: HTS full-context labelsからphoneme + prosody symbolシーケンスを抽出。

```python
pp_symbols(labels: HTSLabelFile, drop_unvoiced_vowels: bool = True) -> list
```

**出力例**: `"^ m i [ z u o # m a [ r e ] e sh i a k a r a ... $"`

CC-G2PnP論文のPnP系列生成の参考実装として有用。

### 3.9 Dict-DNN韻律予測モデル（ベースライン）

論文のターゲットPnP系列の生成に使用されるパイプライン:

```
テキスト → MeCab+UniDic（テキスト正規化、形態素解析）
         → DNN韻律予測（IP, AP, ANの予測）
         → 音素+韻律ラベル系列
```

- Park et al. (Interspeech 2022) に基づく
- 80,061件の手動アノテーション付き韻律ラベルで学習
- **コードは未公開** → pyopenjtalkのfull-context label解析で代替する

---

## 4. 既存フレームワーク調査

### 4.1 NVIDIA NeMo

- **リポジトリ**: https://github.com/NVIDIA-NeMo/NeMo
- **ライセンス**: Apache 2.0
- **CC-G2PnPとの関連**: Noroozi et al. (ICASSP 2024) の実装元
- **Streaming対応**: Cache-aware streaming（activation caching, stateful modules）
  - 訓練時と推論時の両方で限定的な左右コンテキストを使用
  - モデルの予測がオフラインとストリーミングで一致するよう設計
  - 対応レイヤー: Depthwise convolutions, Self-attention, Downsampling layers
- **FastConformer**: 複数のlookahead設定をサポート
- **再現実装での有用性**: chunk-aware streamingとキャッシュ管理の参考実装として最も有用
- **参考コード**:
  - `tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb`
  - `examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`

### 4.2 ESPnet

- **リポジトリ**: https://github.com/espnet/espnet
- **ライセンス**: Apache 2.0
- **CC-G2PnPとの関連**: Self-conditioned CTCの実装あり（PR #3274: Intermediate CTC + Stochastic depth）
- **Streaming対応**: Contextual Block Conformer
- **再現実装での有用性**: Self-conditioned CTCと日本語G2Pの参考実装
- **注意点**: Self-conditioned CTCのマルチGPU訓練問題あり（Issue #4031）
- **参考コード**: `espnet2/text/phoneme_tokenizer.py`（pyopenjtalkの音素変換）

### 4.3 SpeechBrain

- **リポジトリ**: https://github.com/speechbrain/speechbrain
- **ライセンス**: Apache 2.0
- **CC-G2PnPとの関連**: Dynamic Chunk Trainingの実装
  - 訓練時にランダムなチャンクサイズをサンプリングして複数のレイテンシに対応
  - `DynChunkTrainConfig` で設定管理
  - Dynamic Chunk Convolution (DCC) も実装済み
- **再現実装での有用性**: chunk-aware attentionの教育的な参考実装
- **参考コード**: `tutorials/nn/conformer-streaming-asr.html`

### 4.4 WeNet

- **リポジトリ**: https://github.com/wenet-e2e/wenet
- **ライセンス**: Apache 2.0
- **CC-G2PnPとの関連**: U2フレームワーク（streaming/non-streaming統一）
- **再現実装での有用性**: 統一アーキテクチャの設計パターンの参考

### 4.5 推奨方針 → 実装完了 (Phase 0-5 + FlashAttention + Phase 2最適化)

既存フレームワークを**参考**にしつつ、**PyTorchベースでスクラッチ実装**済み:
- NeMoからchunk-aware streamingのマスク生成パターンを参考 → `create_chunk_mask`/`create_mla_mask` (vectorized)
- ESPnetからSelf-conditioned CTCの実装を参考 → `ConformerEncoder` に中間CTC+フィードバック実装
- ESPnet Issue #4031を参考にDDP設定 → `find_unused_parameters=True`
- SpeechBrainからDynamic Chunk Trainingのパターンを参考
- 学習パイプライン: Trainer + TrainingConfig + AdamW (decay/no_decay, **fused AdamW対応 Phase 1最適化**) + ExponentialLR + CheckpointManager (**非同期保存 `async_checkpoint` フラグ Phase 2最適化**) + DDP (**find_unused_parameters=True, static_graph最適化**) + AMP + W&B (必須)
- ストリーミング推論: Conv cache + KV cache + MLA look-ahead (**torch.inference_mode()使用 Phase 0最適化**)
- 評価パイプライン: 6メトリクス (`jiwer.wer` ベース) + 4ドメインビルトインデータ + batch/streaming推論 + **torch.compile オプション (`use_compile` フラグ, +30-50% 推論高速化) Phase 2最適化** + FP16 autocast（CUDA時）+ 長さソートバッチング
- データパイプライン: **ネットワークエラーリトライロジック実装済み** (exponential backoff) + **PnP ラベル LMDB キャッシュ (`PnPLabelCache`) Phase 2最適化** → GPU利用率を大幅改善
- アテンション: `use_flash_attention=True` で **全系列 SDPA (`_forward_sdpa`)** → 単一 SDPA 呼び出しで T4 訓練 3.5x 高速化 (290ms vs 1028ms/step)。`_forward_chunk_sdpa` は参照実装として保持
- 外部フレームワークへの依存なし（PyTorch標準のみ）、論文仕様を忠実に再現
- 561テスト PASS、ruff clean (FlashAttention SDPA + Phase 2最適化テスト含む)

---

## 5. 著者・関連リソース調査

### 5.1 Ryuichi Yamamoto (r9y9)

- **GitHub**: https://github.com/r9y9
- **個人サイト**: https://r9y9.github.io/
- **所属**: LY Corporation R&D（名古屋）
- **専門**: Speech synthesis, Voice conversion, Singing voice synthesis
- **主要OSS**:

| プロジェクト | 概要 |
|---|---|
| **pyopenjtalk** | Open JTalkのPythonラッパー（本論文のG2Pで使用） |
| **ttslearn** | 「Pythonで学ぶ音声合成」の学習用ライブラリ（pp_symbols関数） |
| **NNSVS** | Neural Network-Based Singing Voice Synthesis Toolkit |
| **ESPnet2-TTS** | ESPnetのTTS拡張（統一的で再現可能なend-to-end TTSツールキット） |
| **wavenet_vocoder** | WaveNetボコーダ実装 |
| **deepvoice3_pytorch** | Deep Voice 3のPyTorch実装（108話者マルチスピーカーTTS） |

### 5.2 Yuma Shirahata

- **所属**: LY Corporation R&D
- **プロフィール**: https://research.lycorp.co.jp/en/people/112
- **経歴**: 2021年東京大学電気電子情報系修士修了、2021年LINE（現LY）入社
- **研究テーマ**: 高品質感情音声合成、大規模ラベルなし音声データの基盤モデル
- **関連論文**: Audio-conditioned phonemic and prosodic annotation (Interspeech 2024)

### 5.3 LY Corporation（旧LINE Corporation）

- **音声合成研究**: 自然な日本語音声合成の研究開発を推進、オンデバイス小型化にも取り組む
- **商用ツール**: Achoris Editor（音声合成編集ツール）
- **実用化**: LINE WORKS AiCall（AI電話応答サービス）
- **関連OSS**: open-universe（UNIVERSE/UNIVERSE++ diffusion-based speech enhancement）
- **学会貢献**: Interspeech 2021に6本の論文発表（Nozaki & KomatsuのSelf-conditioned CTC論文を含む）

### 5.4 論文コードの公開状況

- **CC-G2PnPのコードは未公開**（2026年2月時点、提出直後のため）→ **本リポジトリで再現実装中** (Phase 5+最適化完了: モデルコア84M params + 学習パイプライン + ストリーミング推論 + 評価パイプライン + Phase 2最適化, 561テスト PASS)
- **Dict-DNN韻律予測モデルも未公開**（Park et al., Interspeech 2022）→ pyopenjtalk full-context label解析で代替実装済み
- **6D-Eval評価データセットも未公開**
- 今後r9y9のGitHubリポジトリで公開される可能性あり

---

## 6. 関連技術の詳細調査

### 6.1 Self-conditioned CTC

**論文**: Nozaki & Komatsu, "Relaxing the Conditional Independence Assumption of CTC-based ASR by Conditioning on Intermediate Predictions," Interspeech 2021

**手法**:
- CTCの条件付き独立性の仮定を緩和
- 中間層でCTC損失を計算し、中間予測を線形変換してから次の層の入力に加算
- 訓練時・推論時の両方で中間予測を使用
- 強力な自己回帰モデル+ビームサーチと同等の性能を達成しつつ、デコード速度は30倍以上高速

**CC-G2PnPでの適用**:
- 第2, 4, 6層 (1-indexed) = layers 1, 3, 5 (0-indexed) に中間CTC損失を配置
- 重みは最終CTCの1/3
- 出力層（Linear）は最終層と中間層で共有 → **実装済み**: `ctc_head.projection = encoder.ctc_projection`
- 中間logitsをhidden dimにフィードバック: `x = x + ctc_to_hidden(inter_logits)`

### 6.2 LLM2PnP (Dekel et al., ICASSP 2024)

**論文**: "Speak While You Think: Streaming Speech Synthesis During Text Generation"

| 項目 | LLM2PnP | CC-G2PnP |
|---|---|---|
| アーキテクチャ | Encoder-Decoder Transformer | Conformer-CTC |
| ストリーミング方式 | word-levelアテンションマスク | Chunk-aware + MLA |
| 単語境界依存 | **あり**（英語向け） | **なし**（非分かち書き言語対応） |
| アライメント学習 | 知識蒸留 | CTCで動的学習 |
| 適用言語 | 英語 | 日本語（非分かち書き言語全般） |

### 6.3 CTC損失（PyTorch）

```python
# PyTorch標準のCTC Loss
loss_fn = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

# 入力形式
# log_probs: [T, B, C] (Time, Batch, Classes) - log softmax後
# targets: [B, S] (Batch, max target length)
# input_lengths: [B] - 各サンプルの入力長
# target_lengths: [B] - 各サンプルのターゲット長

# CTC制約: input_length >= target_length（必須）
# 推奨: input_length >= 2 * target_length - 1
```

**安定化テクニック**（※論文未記載。CTC学習の安定化のために推奨される一般的なテクニック）:
- `zero_infinity=True`: 無効なアラインメントの損失・勾配をゼロに置換
- Gradient Clipping: `max_norm=1.0` が一般的
- Learning Rate Warm-up: 小さい学習率から開始し段階的に増加
- 適切な入力/ターゲット長の管理（前処理段階で保証）

**Greedy Decoding**:
```python
# emission: [B, T, C] (log probabilities)
indices = torch.argmax(emission, dim=-1)       # 各タイムステップで最大確率のクラスを選択
indices = torch.unique_consecutive(indices)     # 連続する重複を削除
result = [i for i in indices if i != blank_id]  # blankを除去
```

### 6.4 CER/SER計算

**CER (Character Error Rate)**:
```
CER = (Substitutions + Deletions + Insertions) / Reference_Length × 100
```
- Levenshtein距離ベース、0に近いほど高性能
- 計算ライブラリ: `jiwer` (pip install jiwer)

**実装上の注意**: PnPトークンは多文字カタカナ（キョ, シャ等）を含むため、`jiwer.cer`（Unicode文字単位）ではなく `jiwer.wer`（スペース区切り単語単位）を使用する。トークン列をスペースで結合して渡す: `jiwer.wer("キョ * ー ワ", "キョ ー ワ")`

**SER (Sentence Error Rate)**:
```
SER = (1つ以上のエラーを含む文の数) / 総文数 × 100
```

**Norm. PnP**: IPとAPを同一視した評価。`#` → `/` に置換してからCER/SERを計算。

---

## 7. 日本語音素セット

### 7.1 論文の音素表記

論文Fig.1の例から、**カタカナベース**の音素表記を採用: `キョ * ー ワ / イ * ー / テ * ン キ`

pyopenjtalkのローマ字音素出力をカタカナモーラ単位に変換する必要がある。

### 7.2 ローマ字→カタカナ変換マッピング（主要）

| ローマ字 | カタカナ | 説明 |
|---|---|---|
| a, i, u, e, o | ア, イ, ウ, エ, オ | 母音 |
| ka, ki, ku, ke, ko | カ, キ, ク, ケ, コ | か行 |
| ky+a/u/o | キャ, キュ, キョ | 拗音 |
| sh+i | シ | さ行特殊 |
| ch+i | チ | た行特殊 |
| ts+u | ツ | た行特殊 |
| N | ン | 撥音 |
| cl | ッ | 促音 |
| (長母音) | ー | 長音 |

### 7.3 CTC語彙の構成（論文推定 約106クラス → 実装 140トークン）

論文では「約106クラス」と推定されていたが、pyopenjtalkが出力する全音素を網羅するカタカナモーラを列挙した結果、**134モーラ**（外来語拗音含む）が必要であることが判明した。

```
ID 0:       <blank> (CTCのblank)
ID 1-134:   カタカナモーラ（134種）
  - 母音(5): ア,イ,ウ,エ,オ
  - か行(8): カ,キ,ク,ケ,コ,キャ,キュ,キョ
  - が行(8): ガ,ギ,グ,ゲ,ゴ,ギャ,ギュ,ギョ
  - さ行(8): サ,シ,ス,セ,ソ,シャ,シュ,ショ
  - ざ行(8): ザ,ジ,ズ,ゼ,ゾ,ジャ,ジュ,ジョ
  - た行(8): タ,チ,ツ,テ,ト,チャ,チュ,チョ
  - だ行(3): ダ,デ,ド
  - な行(8): ナ,ニ,ヌ,ネ,ノ,ニャ,ニュ,ニョ
  - は行(8): ハ,ヒ,フ,ヘ,ホ,ヒャ,ヒュ,ヒョ
  - ば行(8): バ,ビ,ブ,ベ,ボ,ビャ,ビュ,ビョ
  - ぱ行(8): パ,ピ,プ,ペ,ポ,ピャ,ピュ,ピョ
  - ま行(8): マ,ミ,ム,メ,モ,ミャ,ミュ,ミョ
  - や行(3): ヤ,ユ,ヨ
  - ら行(8): ラ,リ,ル,レ,ロ,リャ,リュ,リョ
  - わ行(2): ワ,ヲ
  - 特殊(3): ン,ッ,ー
  - 外来語(17): ファ,フィ,フェ,フォ,フュ,ティ,ディ,デュ,トゥ,ドゥ,
               ウィ,ウェ,ウォ,ヴァ,ヴィ,ヴ,ヴェ,ヴォ
  - 外来語拗音(6): シェ,ジェ,チェ,ディャ,ディョ (v2追加)
  ※ pyopenjtalkは「ヴ」→「バ行」変換するためヴ系5トークンは実質未使用
ID 135:     * (アクセント核)
ID 136:     / (アクセント句境界)
ID 137:     # (イントネーション句境界)
ID 138:     <unk>
ID 139:     <pad>
合計: 140トークン
```

---

## 8. 学習・評価の詳細

### 8.1 学習設定

| パラメータ | 値 |
|---|---|
| 学習データ | ReazonSpeech転写テキスト 14,960,911件 |
| 検証データ | 4,802件 |
| Dynamic batching | 最大8,192トークン/ミニバッチ |
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01, betas=(0.9, 0.98)) ※論文未記載。Conformer学習の一般的な設定を採用 |
| 学習率スケジュール | ExponentialLR: 1e-4 → 1e-5（1.2Mステップ） + LinearLR warmup (10,000 steps) |
| 学習ステップ数 | 1,200,000 |
| Warmup | 10,000 steps (LinearLR → ExponentialLR via SequentialLR) |
| 勾配クリッピング | max_norm=1.0 ※論文未記載 |
| AMP | bfloat16 (default) / float16 |
| 中間CTC損失 | 第2, 4, 6層（重み = 最終CTCの1/3） |
| トークナイザ | CALM2-7B-Chat BPE（語彙65,000） |
| アップサンプリング係数 | 8 |

### 8.2 学習率スケジュール

ExponentialLR: γ = (final_lr / lr)^(1 / (total_steps - warmup_steps)) 自動計算

```python
# ※論文未記載のパラメータ: Optimizer種類(AdamW), weight_decay, betasはConformer学習の一般的な設定を採用
# 実装: TrainingConfig.scheduler_gamma プロパティで自動計算
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, betas=(0.9, 0.98))
# decay/no_decay 2グループ: LayerNorm・biasはweight decay除外
warmup_scheduler = LinearLR(optimizer, start_factor=1e-8/1e-4, total_iters=10000)
decay_scheduler = ExponentialLR(optimizer, gamma=0.9999981)
scheduler = SequentialLR(optimizer, [warmup_scheduler, decay_scheduler], milestones=[10000])
```

### 8.3 損失関数

```python
total_loss = ctc_loss(final_output, targets)
           + (1/3) * ctc_loss(layer2_output, targets)
           + (1/3) * ctc_loss(layer4_output, targets)
           + (1/3) * ctc_loss(layer6_output, targets)
```

### 8.4 実験結果（全11モデル）

| モデル名 | P | C | M | PnP CER (SER) | Norm. PnP CER (SER) | Phoneme CER (SER) | Start Latency |
|---|---|---|---|---|---|---|---|
| Dict-DNN-5 | 0 | 5 | 0 | 6.67 (84.7) | 5.90 (83.1) | 1.54 (22.4) | 5τ+0.03s |
| Dict-DNN-10 | 0 | 10 | 0 | 3.58 (62.2) | 2.97 (57.0) | 0.86 (13.0) | 10τ+0.03s |
| Dict-DNN-20 | 0 | 20 | 0 | 2.28 (45.9) | 1.72 (36.0) | 0.56 (8.7) | 20τ+0.04s |
| CC-G2PnP-2-0 | 10 | 2 | 0 | 2.44 (51.6) | 1.77 (43.7) | 0.82 (12.8) | 2τ+0.04s |
| CC-G2PnP-2-1 | 10 | 2 | 1 | 1.90 (43.4) | 1.36 (33.1) | 0.55 (8.6) | 3τ+0.04s |
| CC-G2PnP-2-2 | 10 | 2 | 2 | 1.85 (42.4) | 1.33 (32.3) | 0.53 (8.6) | 4τ+0.04s |
| CC-G2PnP-5-0 | 10 | 5 | 0 | 2.01 (45.3) | 1.45 (35.4) | 0.62 (9.4) | 5τ+0.04s |
| **CC-G2PnP-5-1** | **10** | **5** | **1** | **1.79 (41.4)** | **1.28 (28.6)** | **0.52 (8.4)** | **6τ+0.04s** |
| CC-G2PnP-5-2 | 10 | 5 | 2 | 1.80 (41.4) | 1.30 (29.4) | 0.53 (8.6) | 7τ+0.04s |
| Dict-DNN-NS | ∞ | - | - | 1.71 (40.4) | 1.18 (26.4) | 0.40 (6.4) | Nτ+0.03s |
| CC-G2PnP-NS | ∞ | - | - | 1.80 (42.0) | 1.33 (30.9) | 0.48 (7.6) | Nτ+0.07s |

### 8.5 評価データ: 6D-Eval

- 6ドメイン（チャット, インタビュー, ニュース, 小説, 実用書, SNS）
- 合計2,722文
- 専門家による音素+韻律ラベルの手動アノテーション
- **データセットは非公開**

### 8.6 TTS主観評価

TTSモデル: NANSY-TTSベース（音素+韻律ラベル入力に改造）
- 学習データ: 173,987サンプル、207.96時間、17話者、手動韻律アノテーション付き
- 評価: 単一女性話者、MOS 5段階、日本語母語話者15名、50文

**MOS結果（論文Table 2）**:

| モデル | MOS |
|---|---|
| Dict-DNN-5 | 2.73 ± 0.11 |
| Dict-DNN-10 | 3.35 ± 0.11 |
| CC-G2PnP-5-0 | 3.81 ± 0.10 |
| CC-G2PnP-5-1 | 4.02 ± 0.09 |
| Dict-DNN-NS | 4.07 ± 0.09 |
| CC-G2PnP-NS | 4.02 ± 0.09 |
| GT-Label | 4.16 ± 0.07 |

### 8.7 学習データ量の影響

| データ量 | PnP CER (SER) | Phoneme CER (SER) |
|---|---|---|
| 1% (149,609件) | 4.55 (64.0) | 1.97 (22.2) |
| 10% (1,496,091件) | 2.34 (47.8) | 0.71 (10.6) |
| 100% (14,960,911件) | 1.79 (41.4) | 0.52 (8.4) |

→ データ量増加に伴い一貫して性能向上。大規模データの活用効果が確認済み。

---

## 9. 再現実装のための依存ライブラリ

```
# Core
torch>=2.0.0
torchaudio>=2.0.0

# Tokenizer & Data
transformers>=4.35.0        # CALM2トークナイザ読み込み
datasets>=2.14.0,<4.0.0     # ReazonSpeechデータセット（v4.0+でtrust_remote_code廃止）

# Japanese NLP
fugashi[unidic]             # MeCab + UniDic (形態素解析)
pyopenjtalk-plus>=0.4.1     # G2P + Full-context labels（Python 3.13 Windows対応フォーク）

# Training
wandb>=0.15.0               # 学習監視（必須、未ログイン時は RuntimeError）

# Evaluation
jiwer>=3.0.0                # CER/WER計算

# Utilities
tqdm, numpy, pandas
```

---

## 10. 技術的リスクと対策

| リスク | 影響度 | 対策 |
|---|---|---|
| Dict-DNN韻律予測モデルが入手不可 | ~~高~~ **対応済** | pyopenjtalkのfull-context label解析で代替実装済み（ttslearnのpp_symbols関数ベース → CC-G2PnP記法変換） |
| 6D-Eval評価データが非公開 | ~~高~~ **対応済** | 4ドメイン×10文のビルトインデータ + EvalDataGenerator で代替実装済み。pyopenjtalk出力をground truthとした自動評価 |
| 1500万件のデータ前処理が膨大 | 中 | HuggingFace datasetsのstreaming + 前処理結果のキャッシュ。並列処理で高速化 |
| 1.2Mステップの学習時間 | 中 | AMP (bfloat16/float16) + マルチGPU (DDP) **実装済み** |
| CTC収束不安定 | ~~中~~ **対策済** | `zero_infinity=True`, gradient clipping (max_norm=1.0), LinearLR warmup (10,000 steps) **全て実装済み** |
| pyopenjtalkのインストール問題 | ~~中~~ **解決済** | `pyopenjtalk-plus>=0.4.1`（Python 3.13 Windows対応フォーク）を使用。API互換 |
| ReazonSpeechデータ量の不一致 | 低 | 論文の14,960,911件は`all`設定からフィルタリングされた可能性あり。フィルタリング条件の推定が必要 |
| CALM2トークナイザの語彙サイズ | ~~—~~ **確認済** | **65,000** であることを確認済み（論文記載の65,024とは異なる） |
| Conformerのアテンションヘッド数・カーネルサイズが未記載 | ~~低~~ **解決済** | Conformerの標準値（heads=8, kernel=31）で実装。84Mパラメータ |

---

## 11. 参考文献・URL

### 論文
- CC-G2PnP: https://arxiv.org/abs/2602.17157
- Self-conditioned CTC: https://arxiv.org/abs/2104.02724
- LLM2PnP (Speak While You Think): https://arxiv.org/abs/2309.11210
- Conformer: https://arxiv.org/abs/2005.08100
- Stateful Conformer: https://arxiv.org/abs/2312.17279
- CTC: Graves et al., ICML 2006
- Dict-DNN: Park et al., Interspeech 2022

### データ・モデル
- ReazonSpeech: https://huggingface.co/datasets/reazon-research/reazonspeech
- CALM2-7B-Chat: https://huggingface.co/cyberagent/calm2-7b-chat

### フレームワーク・ツール
- NeMo: https://github.com/NVIDIA-NeMo/NeMo
- ESPnet: https://github.com/espnet/espnet
- SpeechBrain: https://github.com/speechbrain/speechbrain
- WeNet: https://github.com/wenet-e2e/wenet
- pyopenjtalk: https://github.com/r9y9/pyopenjtalk
- fugashi: https://github.com/polm/fugashi
- ttslearn: https://github.com/r9y9/ttslearn
- nnmnkwii: https://github.com/r9y9/nnmnkwii

### HTS仕様
- HTS lab_format.pdf: http://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-ATR503-M001.tar.bz2

### 著者
- Ryuichi Yamamoto: https://github.com/r9y9 / https://r9y9.github.io/
- Yuma Shirahata: https://research.lycorp.co.jp/en/people/112
