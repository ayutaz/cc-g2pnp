# CC-G2PnP 再現実装ロードマップ

arXiv:2602.17157 (Shirahata & Yamamoto, LY Corporation, 2026) の再現実装計画。

---

## 全体像

```
Phase 0: 環境構築・データ準備        ──┐
Phase 1: データパイプライン           ──┤ 基盤  ✅ 完了
Phase 2: モデルコア実装              ──┘       ✅ 完了
Phase 3: 学習パイプライン            ── 学習   ✅ 完了
Phase 4: 推論・ストリーミング         ── 推論   ← 次
Phase 5: 評価                       ── 検証
Phase 6: アブレーション・最適化       ── 発展
```

**最終目標**: CC-G2PnP-5-1 (C=5, M=1, P=10) でPnP CER ≈ 1.79を達成

---

## Phase 0: 環境構築・データ準備

**目的**: 開発環境と学習データの確保

### タスク

| # | タスク | 詳細 | 成果物 |
|---|---|---|---|
| 0.1 | Python環境構築 | Python 3.13, CUDA対応PyTorch環境, uv 0.9.2 | `pyproject.toml` (hatchling) |
| 0.2 | 依存ライブラリのインストール | torch>=2.0, transformers>=4.35, datasets>=2.14&lt;4.0, fugashi+unidic, pyopenjtalk-plus>=0.4.1, jiwer>=3.0 | インストール確認スクリプト |
| 0.3 | CALM2トークナイザの動作確認 | `cyberagent/calm2-7b` のBPEトークナイザ（語彙65,000）をロードし基本動作を確認 | トークナイザのテストコード |
| 0.4 | ReazonSpeechデータのダウンロード | `all` config（14,960,911件）をstreaming modeで利用可能にする | データロードスクリプト |
| 0.5 | プロジェクト構成の確定 | ディレクトリ構造、設定管理（Hydra）、ログ管理（W&B or TensorBoard）のセットアップ | プロジェクトスケルトン |

### ディレクトリ構成

```
cc_g2pnp/
├── data/              # データ前処理 (Phase 1 ✅)
│   ├── vocabulary.py      # CTC語彙定義 (PnPVocabulary)
│   ├── pnp_labeler.py     # PnPラベル生成 (HTS→カタカナ+韻律)
│   ├── tokenizer.py       # CALM2 BPEトークナイザラッパー
│   ├── dataset.py         # ReazonSpeech streaming IterableDataset
│   └── collator.py        # Dynamic Batching + パディング
├── model/             # モデル定義 (Phase 2 ✅)
│   ├── __init__.py          # 13 public exports
│   ├── config.py            # CC_G2PnPConfig dataclass
│   ├── embedding.py         # TokenEmbedding (BPE → D, repeat_interleave ×8)
│   ├── positional_encoding.py # RelativePositionalEncoding (sinusoidal)
│   ├── attention.py         # ChunkAwareAttention + create_chunk_mask/mla_mask
│   ├── feed_forward.py      # FeedForwardModule (Macaron half-step)
│   ├── convolution.py       # ConformerConvModule (causal depthwise conv)
│   ├── conformer_block.py   # ConformerBlock (FFN→MHSA→Conv→FFN→LN)
│   ├── encoder.py           # ConformerEncoder (8層 + self-conditioned CTC)
│   ├── ctc_decoder.py       # CTCHead + greedy_decode
│   └── cc_g2pnp.py         # CC_G2PnP (統合モデル, 84Mパラメータ)
├── training/          # 学習ループ (Phase 3 ✅)
│   ├── __init__.py        # 15 public exports
│   ├── config.py           # TrainingConfig dataclass (24フィールド)
│   ├── optimizer.py        # build_optimizer + build_scheduler
│   ├── checkpoint.py       # CheckpointManager (save/load/cleanup)
│   ├── logger.py           # TrainingLogger (TensorBoard + W&B)
│   ├── distributed.py      # DDP utilities (setup/cleanup/reduce)
│   ├── evaluator.py        # Evaluator (PnP CER評価)
│   └── trainer.py          # Trainer (AMP/DDP/validation統合)
├── inference/         # 推論・ストリーミング (Phase 4)
│   └── streaming.py
├── evaluation/        # 評価 (Phase 5)
│   └── metrics.py
└── utils/
scripts/
└── train.py               # argparse CLIエントリポイント (Phase 3 ✅)
tests/
├── test_g2p.py            # pyopenjtalk/fugashi基本テスト (5件)
├── test_tokenizer.py      # CALM2トークナイザテスト (6件)
├── test_data_loading.py   # ReazonSpeechロードテスト (3件, network)
├── test_vocabulary.py     # CTC語彙テスト (8件)
├── test_pnp_labeler.py    # PnPラベル生成テスト (9件)
├── test_pipeline.py       # 統合テスト (7件 + 1件network)
├── test_dataset.py        # データセットテスト
├── test_collator.py       # コレータテスト
├── test_embedding.py      # TokenEmbeddingテスト
├── test_positional_encoding.py # RelativePositionalEncodingテスト
├── test_attention.py      # ChunkAwareAttention + マスク生成テスト
├── test_feed_forward.py   # FeedForwardModuleテスト
├── test_convolution.py    # ConformerConvModuleテスト
├── test_conformer_block.py # ConformerBlockテスト
├── test_ctc_decoder.py    # CTCHead + greedy_decodeテスト
├── test_encoder.py        # ConformerEncoderテスト
├── test_model.py          # CC_G2PnP統合テスト (backward, variable lengths等)
├── test_training_config.py     # TrainingConfigテスト (38件)
├── test_training_optimizer.py  # Optimizer/Schedulerテスト (17件)
├── test_training_checkpoint.py # CheckpointManagerテスト (17件)
├── test_training_logger.py     # TrainingLoggerテスト (18件)
├── test_training_distributed.py # DDP utilitiesテスト (16件)
├── test_training_evaluator.py  # Evaluatorテスト (15件)
├── test_training_trainer.py    # Trainerテスト (16件)
└── test_training_integration.py # Phase 3統合テスト (7件)
```

### 完了条件
- [x] 全ライブラリがインストールされ `import` 可能
- [x] CALM2トークナイザで日本語テキストをトークン化できる（vocab_size=65,000確認済み）
- [x] ReazonSpeechから転写テキストをstreaming取得できる（"all" subset + select_columns）
- [x] ruff エラーなし
- [x] 全14テスト PASS（G2P×5, トークナイザ×6, データ読込×3）

### 計画からの変更点
| 項目 | 計画 | 実装 | 理由 |
|------|------|------|------|
| Python | 3.10+ | **3.13** | 最新版を採用 |
| vocab_size | 65,024 | **65,000** | CALM2実測値 |
| pyopenjtalk | pyopenjtalk>=0.3 | **pyopenjtalk-plus>=0.4.1** | Python 3.13 Windows対応フォーク |
| datasets | >=2.14.0 | **>=2.14.0,<4.0.0** | v4.0+でtrust_remote_code廃止 |
| モデル名 | calm2-7b-chat | **calm2-7b** | トークナイザ共通 |

---

## Phase 1: データパイプライン

**目的**: BPEトークン列 → PnPラベル列のペアデータを生成する前処理パイプラインの構築

### タスク

| # | タスク | 詳細 | 難易度 | 状態 |
|---|---|---|---|---|
| 1.1 | BPEトークン化モジュール | CALM2トークナイザで転写テキストをトークンID列に変換 | 低 | ✅ |
| 1.2 | PnPラベル生成モジュール | pyopenjtalk full-context label → 音素+韻律ラベル系列の変換。Dict-DNN未公開のため代替実装 | **高** | ✅ |
| 1.3 | CTC語彙の定義 | 140トークン（blank + カタカナ134モーラ + 韻律記号`*`,`/`,`#` + unk + pad）の語彙マッピング | 中 | ✅ |
| 1.4 | データ前処理パイプライン | ReazonSpeech streaming → BPEトークン化 + PnPラベル生成 → CTC制約チェック | 中 | ✅ |
| 1.5 | Dynamic Batchingの実装 | 最大8,192トークン/ミニバッチのdynamic batching。可変長シーケンスのパディング処理 | 中 | ✅ |

### 実装成果物

| ファイル | 概要 |
|---------|------|
| `cc_g2pnp/data/vocabulary.py` | `PnPVocabulary` — 140トークン (blank + カタカナ134モーラ + 韻律3 + unk + pad) |
| `cc_g2pnp/data/pnp_labeler.py` | `generate_pnp_labels()` — ttslearn pp_symbolsベース → CC-G2PnP記法に変換 |
| `cc_g2pnp/data/tokenizer.py` | `G2PnPTokenizer` — CALM2 BPE薄ラッパー |
| `cc_g2pnp/data/dataset.py` | `G2PnPDataset(IterableDataset)` — ReazonSpeech streaming + CTC制約チェック |
| `cc_g2pnp/data/collator.py` | `DynamicBatchCollator` + `dynamic_batch_sampler` |
| `tests/test_vocabulary.py` | 語彙テスト (8件) |
| `tests/test_pnp_labeler.py` | PnPラベル生成テスト (9件) |
| `tests/test_pipeline.py` | 統合テスト (7件 + 1件network) |

### Dict-DNN代替戦略（タスク1.2の詳細）

論文のターゲットPnPラベルはDict-DNN（Park et al., Interspeech 2022）で生成されるが、**コードは未公開**。以下の代替手段を採用:

```
テキスト
  → pyopenjtalk.extract_fullcontext()  # HTS full-context label取得
  → _extract_pp_symbols()              # ttslearn pp_symbolsベースの韻律解析
  → _phonemes_to_mora()                # ローマ字→カタカナモーラ変換 + 長音処理
  → CC-G2PnP記法のトークン列            # *, /, # による韻律マーカー
```

**実装された抽出ロジック** (ttslearn pp_symbolsベース):
1. **音素**: HTS full-context labelのp3（中心音素）を抽出
2. **アクセント核 (`*`)**: ttslearnの`]`（pitch fall）条件 — `a1==0 && a2_next==a2+1 && a2!=f1`
3. **アクセント句境界 (`/`)**: ttslearnの`#`条件 — `a3==1 && a2_next==1`
4. **イントネーション句境界 (`#`)**: ttslearnの`_`（pause）— `pau` の出現箇所
5. **長音符 (`ー`)**: 連続する同一母音を検出。`*`を跨いで追跡するが、`/`・`#`でリセット

**出力例**:
| 入力 | 出力 |
|------|------|
| 今日はいい天気 | `キョ * ー ワ / イ ー / テ * ン キ` |
| 雨 (頭高型) | `ア * メ` |
| 飴 (平板型) | `ア メ` |
| お母さん | `オ カ * ー サ ン` |

**リスク**: pyopenjtalkのルールベース韻律とDict-DNNの学習ベース韻律には差異がある。論文の数値との完全一致は困難だが、モデルアーキテクチャの検証には十分。

### 計画からの変更点

| 項目 | 計画 | 実装 | 理由 |
|------|------|------|------|
| 語彙サイズ | ~106クラス | **140トークン (134モーラ)** | 外来語・拗音モーラを網羅（v2で`シェ`,`ジェ`,`チェ`,`ディャ`,`ディョ`,`フュ`追加） |
| CTC制約 | input_length >= target_length | **input_length * 8 >= target_length** | アップサンプリング係数8を考慮 |
| キャッシュ保存 | 前処理結果のキャッシュ | **ストリーミングのみ** | メモリ効率優先、キャッシュは必要に応じて後で追加 |

### 完了条件
- [x] 任意の日本語テキストからBPEトークン列を生成できる
- [x] 任意の日本語テキストからPnPラベル列を生成できる
- [x] CTC制約（input_length×8 >= target_length）を満たすデータ対が生成される
- [x] Dynamic batchingで効率的なミニバッチを構成できる
- [x] ruff lint エラーなし
- [x] 全68テスト PASS (非ネットワーク)、data/ カバレッジ: collator=100%, vocabulary=100%, tokenizer=100%, dataset=93%, pnp_labeler=84%

---

## Phase 2: モデルコア実装

**目的**: Streaming Conformer-CTCモデルの全コンポーネントを実装

### 依存関係

```
2.1 Embedding + Token Upsampling
     ↓
2.2 Causal Convolution Module
2.3 Chunk-aware Streaming Attention  ←並列で実装可能→  2.4 Feed-Forward Module
     ↓
2.5 Conformer層の組み立て
     ↓
2.6 Self-conditioned CTC
     ↓
2.7 全体モデルの統合
```

### タスク

| # | タスク | 詳細 | 難易度 | 状態 |
|---|---|---|---|---|
| 2.1 | Embedding + Token Upsampling | `Embedding(65000, 512)` + `sqrt(d_model)` scale + `repeat_interleave(×8)`。入力 [B, T] → [B, T×8, 512] | 低 | ✅ |
| 2.2 | Causal Convolution Module | LN→Pointwise(D→2D)→GLU→CausalDWConv(k=31)→BN→SiLU→Pointwise→Dropout | 中 | ✅ |
| 2.3 | Chunk-aware Streaming Attention | MHSA (heads=8) + Shaw et al. 相対位置バイアス (Q*pos_K^T) + chunk-aware mask (vectorized) | **高** | ✅ |
| 2.4 | Feed-Forward Module | LN→Linear(512→2048)→SiLU→Dropout→Linear(2048→512)→Dropout | 低 | ✅ |
| 2.5 | Conformer層の組み立て | x+0.5*FFN1→x+MHSA→x+Conv→x+0.5*FFN2→LN (Macaron-style) | 中 | ✅ |
| 2.6 | Minimum Look-Ahead (MLA) | Layer 0のみMLA mask適用。`create_mla_mask()` でchunk外M個参照 | **高** | ✅ |
| 2.7 | Self-conditioned CTC | Layers 1,3,5 (0-indexed) に中間CTC。logitsをhiddenにフィードバック。CTC projection重み共有 | **高** | ✅ |
| 2.8 | CTC出力層 | Linear(512, 140) + log_softmax + greedy_decode (argmax→unique_consecutive→blank除去) | 低 | ✅ |
| 2.9 | 全体モデル統合 | `CC_G2PnP` クラス: embedding→encoder→ctc_head, forward()+inference(), 84Mパラメータ | 中 | ✅ |

### 論文記載 vs 推定パラメータ

| パラメータ | 値 | 出典 |
|---|---|---|
| Conformer層数 | 8 | 論文明記 |
| 隠れ次元 d_model | 512 | 論文明記 |
| アップサンプリング係数 | 8 | 論文明記 |
| 中間CTC適用層 | 2, 4, 6 | 論文明記 |
| 中間CTC損失重み | 1/3 | 論文明記 |
| 過去コンテキスト P | 10 | 論文明記 |
| アテンションヘッド数 | 8 | **※推定** (d_model=512での一般的な値) |
| FFN expansion factor | 4 | **※推定** (Conformer標準) |
| 畳み込みカーネルサイズ | 31 | **※推定** (Conformer標準) |
| 位置エンコーディング | 相対位置(Transformer-XLスタイル) | **※推定** (Conformer原論文に準拠) |
| Dropout率 | 不明 | **※推定** (0.1が一般的) |

### 参考実装（Phase 2で参照）

| コンポーネント | 参考フレームワーク | 実装結果 |
|---|---|---|
| Chunk-aware Attention | NVIDIA NeMo (cache-aware streaming) | `create_chunk_mask`/`create_mla_mask` (vectorized) |
| Self-conditioned CTC | ESPnet (PR #3274) | `ConformerEncoder` + `ctc_to_hidden` フィードバック |
| Dynamic Chunk Training | SpeechBrain (DynChunkTrainConfig) | Phase 3で参照予定 |

### 実装成果物

| ファイル | 概要 |
|---------|------|
| `cc_g2pnp/model/config.py` | `CC_G2PnPConfig` dataclass + `__post_init__` validation |
| `cc_g2pnp/model/embedding.py` | `TokenEmbedding` — Embedding(65000,512) → sqrt(d_model) scale → dropout → repeat_interleave ×8 |
| `cc_g2pnp/model/positional_encoding.py` | `RelativePositionalEncoding` — sinusoidal PE buffer, max_len=5000 |
| `cc_g2pnp/model/attention.py` | `ChunkAwareAttention` + `create_chunk_mask` + `create_mla_mask` (vectorized) — Shaw et al. relative pos bias |
| `cc_g2pnp/model/feed_forward.py` | `FeedForwardModule` — LN→Linear(512→2048)→SiLU→Dropout→Linear(2048→512)→Dropout |
| `cc_g2pnp/model/convolution.py` | `ConformerConvModule` — LN→Pointwise(D→2D)→GLU→CausalDWConv(k=31)→BN→SiLU→Pointwise→Dropout |
| `cc_g2pnp/model/conformer_block.py` | `ConformerBlock` — x+0.5*FFN1→x+MHSA→x+Conv→x+0.5*FFN2→LN (Macaron-style) |
| `cc_g2pnp/model/encoder.py` | `ConformerEncoder` — 8層 + self-conditioned CTC at layers 1,3,5 (0-indexed) |
| `cc_g2pnp/model/ctc_decoder.py` | `CTCHead` (log_softmax) + `greedy_decode` (argmax→unique_consecutive→remove blanks) |
| `cc_g2pnp/model/cc_g2pnp.py` | `CC_G2PnP` — CTC projection共有, inference uses `self()`, zero_infinity=True |
| `cc_g2pnp/model/__init__.py` | 13 public exports |
| `tests/test_embedding.py` | TokenEmbedding テスト |
| `tests/test_positional_encoding.py` | RelativePositionalEncoding テスト |
| `tests/test_attention.py` | ChunkAwareAttention + マスク生成テスト |
| `tests/test_feed_forward.py` | FeedForwardModule テスト |
| `tests/test_convolution.py` | ConformerConvModule テスト (batch_size=1含む) |
| `tests/test_conformer_block.py` | ConformerBlock テスト |
| `tests/test_ctc_decoder.py` | CTCHead + greedy_decode テスト |
| `tests/test_encoder.py` | ConformerEncoder テスト |
| `tests/test_model.py` | CC_G2PnP 統合テスト (backward/gradient, variable lengths, CTC weight, config validation, projection sharing) |

### 計画からの変更点

| 項目 | 計画 | 実装 | 理由 |
|------|------|------|------|
| CTC出力サイズ | ~106クラス | **140トークン** | Phase 1語彙定義に準拠（134モーラ+3韻律+blank+unk+pad） |
| 位置バイアス | query-only scalar bias | **Shaw et al. Q*pos_K^T** | Conformer標準のrelative positional bias方式 |
| マスク生成 | Pythonループ | **torch.arange broadcasting** | ベクトル化で高速化 |
| CTC projection | 別々のLinear層 | **最終head/中間層で重み共有** | 論文仕様に準拠 |
| 中間CTC層 | 第2,4,6層 (1-indexed) | **layers 1,3,5 (0-indexed)** | 同一の層を0-indexedで表現 |
| パラメータ数 | 数十M（推定） | **84M** | デフォルト設定 (d_model=512, 8層, heads=8) |
| モデルクラス名 | CC_G2PnP_Model | **CC_G2PnP** | 簡潔さ |

### 完了条件
- [x] モデルのforwardパスが通り、正しい形状の出力を返す
- [x] Chunk-aware attentionマスクが正しく生成される（concrete exampleで検証済み）
- [x] MLA適用時にlook-aheadサイズが正しく増加する
- [x] 中間CTC損失が正しく計算される
- [x] パラメータ数 84M（Conformer 8層×512次元）
- [x] ruff lint エラーなし
- [x] 全183テスト PASS（非ネットワーク）— Phase 3追加後は327テスト
- [x] model/ 全11ファイル カバレッジ **100%**、プロジェクト全体 **96%** (Phase 2時点)
- [x] 5人のエキスパートレビュー完了、Critical/High issue = 0

---

## Phase 3: 学習パイプライン

**目的**: 1.2Mステップの学習を実行可能な学習ループの構築

### タスク

| # | タスク | 詳細 | 難易度 | 状態 |
|---|---|---|---|---|
| 3.1 | CTC損失関数の実装 | **Phase 2で実装済み**: `CC_G2PnP.forward()` に `CTCLoss(blank=0, zero_infinity=True)` + 中間CTC損失統合済み | — | ✅ |
| 3.2 | Optimizer・スケジューラ設定 | AdamW (decay/no_decay分離) + ExponentialLR (γ自動計算) + LinearLR warmup via SequentialLR | 低 | ✅ |
| 3.3 | 学習ループ実装 | Trainer + TrainingConfig + CheckpointManager + TrainingLogger (TensorBoard/W&B) + 勾配クリッピング | 中 | ✅ |
| 3.4 | マルチGPU対応 (DDP) | setup/cleanup/reduce_metrics/wrap_model_ddp。`find_unused_parameters=True` (ESPnet #4031) | 中 | ✅ |
| 3.5 | AMP (Mixed Precision) | bfloat16 (GradScaler不要) / float16 (GradScaler, CUDA only)。CTC損失はFP32で計算 | 低 | ✅ |
| 3.6 | スモールスケール検証 | ReazonSpeech 1%（149,609件）で学習し、基本動作を確認。目標: PnP CER ≈ 4.55 | 中 | |

### 実装成果物

| ファイル | 概要 |
|---------|------|
| `cc_g2pnp/training/config.py` | `TrainingConfig` dataclass (24フィールド) + `__post_init__` validation + `scheduler_gamma` 自動計算 |
| `cc_g2pnp/training/optimizer.py` | `build_optimizer` (AdamW, decay/no_decay分離) + `build_scheduler` (LinearLR warmup + ExponentialLR) |
| `cc_g2pnp/training/checkpoint.py` | `CheckpointManager` — save/load/load_latest/cleanup (keep_last_n), DDP対応 |
| `cc_g2pnp/training/logger.py` | `TrainingLogger` — TensorBoard (SummaryWriter) + optional W&B, context manager |
| `cc_g2pnp/training/distributed.py` | `setup_ddp`, `cleanup_ddp`, `is_main_process`, `get_rank`, `get_world_size`, `reduce_metrics`, `wrap_model_ddp` |
| `cc_g2pnp/training/evaluator.py` | `Evaluator` — PnP CER (jiwer) + collator key mapping (labels→targets) |
| `cc_g2pnp/training/trainer.py` | `Trainer` — AMP autocast, GradScaler (CUDA only), gradient clipping, validation, checkpoint restore |
| `cc_g2pnp/training/__init__.py` | 15 public exports |
| `scripts/train.py` | argparse CLIエントリポイント (--lr, --ddp, --amp, --wandb 等) |
| `tests/test_training_config.py` | TrainingConfigテスト (38件) |
| `tests/test_training_optimizer.py` | Optimizer/Schedulerテスト (17件) |
| `tests/test_training_checkpoint.py` | CheckpointManagerテスト (17件) |
| `tests/test_training_logger.py` | TrainingLoggerテスト (18件) |
| `tests/test_training_distributed.py` | DDP utilitiesテスト (16件) |
| `tests/test_training_evaluator.py` | Evaluatorテスト (15件) |
| `tests/test_training_trainer.py` | Trainerテスト (16件) |
| `tests/test_training_integration.py` | Phase 3統合テスト (7件): imports, vocab整合, collator-model interface, optimizer-scheduler, checkpoint E2E, evaluator, trainer E2E |

### 学習設定まとめ

| パラメータ | 値 | 備考 |
|---|---|---|
| 学習データ | 14,960,911件 | ReazonSpeech転写テキスト |
| バッチ構成 | 最大8,192トークン/ミニバッチ | Dynamic batching |
| Optimizer | AdamW (decay/no_decay分離) | LayerNorm・biasはweight decay除外 |
| 初期学習率 | 1e-4 | 論文記載 |
| 最終学習率 | 1e-5 | 論文記載 |
| weight_decay | 0.01 | ※論文未記載。推定 |
| betas | (0.9, 0.98) | ※論文未記載。推定 |
| 学習率スケジュール | ExponentialLR (γ自動計算) + LinearLR warmup | γ = (final_lr/lr)^(1/(steps-warmup)) |
| 総ステップ数 | 1,200,000 | 論文記載 |
| Warmup | 10,000 steps | LinearLR warmup |
| 勾配クリッピング | max_norm=1.0 | ※論文未記載。推定 |
| AMP | bfloat16 (default) / float16 | GradScalerはfloat16+CUDAのみ |
| チェックポイント | 10,000 steps毎、keep_last_n=5 | DDP対応 (model.module) |

### 計画からの変更点

| 項目 | 計画 | 実装 | 理由 |
|------|------|------|------|
| 設定管理 | Hydra | **dataclass (TrainingConfig)** | 軽量・依存なし |
| Scheduler | ExponentialLRのみ | **LinearLR warmup + ExponentialLR (SequentialLR)** | CTC学習安定化 |
| Optimizer grouping | 単一グループ | **decay/no_decay 2グループ** | LayerNorm・biasのweight decay除外 |
| AMP dtype | FP16 | **bfloat16 (default)** | GradScaler不要、数値安定性 |
| GradScaler | 常に有効 | **float16 + CUDA時のみ有効** | CPU互換性 |
| 設定管理 | Hydra | **argparse (scripts/train.py)** | 依存削減 |

### 完了条件
- [x] 学習ループが正常に動作する (Trainer E2E 3ステップテスト PASS)
- [x] チェックポイントの保存/復元が動作する (CheckpointManager E2Eテスト PASS)
- [x] マルチGPUで学習できる (DDP utilities実装済み、find_unused_parameters=True)
- [x] AMP (Mixed Precision) 対応 (bfloat16/float16、GradScaler CUDA only)
- [x] ruff lint エラーなし
- [x] 全327テスト PASS（非ネットワーク）
- [x] training/ カバレッジ: config=100%, checkpoint=100%, distributed=100%, evaluator=100%, optimizer=100%, logger=98%, trainer=76%
- [ ] 1%データで学習し、CTC損失が収束する（Phase 3.6 未着手）

---

## Phase 4: 推論・ストリーミング

**目的**: チャンク単位のストリーミング推論を実装

### タスク

| # | タスク | 詳細 | 難易度 |
|---|---|---|---|
| 4.1 | Greedy CTC Decoding | argmax → unique_consecutive → blank除去。バッチ推論対応 | 低 |
| 4.2 | ストリーミング推論エンジン | C+Mトークン到着後にチャンク単位で推論を開始。アクティベーションキャッシュ（Conv層のkernel-1フレーム + Attention層の過去Pトークン） | **高** |
| 4.3 | Start Latencyの計測 | `Start = (C+M)τ + G2PnP処理時間` の計測。τはLLMのトークン生成間隔 | 中 |
| 4.4 | 非ストリーミング推論 | 全文を一度に処理する非ストリーミングモード（Dict-DNN-NS, CC-G2PnP-NS相当） | 低 |

### キャッシュ管理の設計

```
推論ステップ t:
  Conv cache: [B, kernel_size-1, D]  ← 過去フレームの活性化
  Attn cache: [B, P, D]              ← 過去Pトークンのkey/value

  新チャンク [B, C, D] を受け取り:
    1. Conv cache と結合 → Causal Conv 実行 → cache更新
    2. Attn cache と結合 → Chunk-aware Attention 実行 → cache更新
    3. CTC decode → PnPラベル出力
```

### 完了条件
- [ ] チャンク単位で逐次的にPnPラベルを出力できる
- [ ] ストリーミング出力と非ストリーミング出力が（チャンク境界効果を除き）概ね一致する
- [ ] Start Latencyが論文値（CC-G2PnP-5-1: 6τ+0.04s）と同程度

---

## Phase 5: 評価

**目的**: 論文の評価指標を再現し、性能を検証

### タスク

| # | タスク | 詳細 | 難易度 |
|---|---|---|---|
| 5.1 | CER/SER計算モジュール | Levenshtein距離ベースのCER計算 + 文単位のSER計算。jiwer利用 | 低 |
| 5.2 | Norm. PnP CER/SER | `#`→`/` に置換してからCER/SERを計算（IP/AP同一視） | 低 |
| 5.3 | Phoneme CER/SER | 韻律記号を除去し音素のみでCER/SERを計算 | 低 |
| 5.4 | 評価データの準備 | 6D-Eval未公開のため代替データを作成。複数ドメイン（ニュース、会話、小説等）のテキストに手動または半自動で韻律アノテーション | **高** |
| 5.5 | 評価パイプライン | 全11モデル設定に対してメトリクスを一括計算するスクリプト | 中 |

### 評価メトリクス一覧

| メトリクス | 定義 | 目標値 (CC-G2PnP-5-1) |
|---|---|---|
| PnP CER | 音素+韻律ラベル系列のCER | 1.79 |
| PnP SER | 音素+韻律ラベル系列のSER | 41.4 |
| Norm. PnP CER | #→/置換後のCER | 1.28 |
| Norm. PnP SER | #→/置換後のSER | 28.6 |
| Phoneme CER | 音素のみのCER | 0.52 |
| Phoneme SER | 音素のみのSER | 8.4 |
| Start Latency | 初回出力までの待ち時間 | 6τ+0.04s |

### 6D-Eval代替案

論文の6D-Eval（2,722文、6ドメイン、専門家アノテーション）は未公開。以下の段階的アプローチで代替:

1. **Phase 5初期**: pyopenjtalkの出力をground truthとした自動評価（モデルの基本動作確認）
2. **Phase 5中期**: 複数ドメインのテキスト（ニュース、Wikipedia、小説、SNS等）を収集し、pyopenjtalkで韻律ラベルを自動生成してテストセット化
3. **Phase 5後期（任意）**: 少量（100-500文）の手動韻律アノテーションで品質を検証

### 完了条件
- [ ] 全6メトリクス（PnP CER/SER, Norm. PnP CER/SER, Phoneme CER/SER）が計算できる
- [ ] 代替評価データでモデル間の相対的な性能差が論文と一致する傾向を確認

---

## Phase 6: アブレーション・最適化

**目的**: 論文のアブレーション結果を再現し、各コンポーネントの寄与を検証

### タスク

| # | タスク | 優先度 | 詳細 |
|---|---|---|---|
| 6.1 | モデルバリエーション比較 | 高 | 最低限: CC-G2PnP-5-0, CC-G2PnP-5-1, CC-G2PnP-5-2 の3モデルでMLAの効果を検証 |
| 6.2 | チャンクサイズ比較 | 高 | C=2とC=5の比較でチャンクサイズの影響を検証 |
| 6.3 | 学習データ量の影響 | 中 | 1% / 10% / 100% の3段階で学習。目標: CERの傾向が論文と一致 |
| 6.4 | Dict-DNNベースラインとの比較 | 中 | pyopenjtalkベースのDict-DNN相当モデル（C=5, 10, 20）を実装して比較 |
| 6.5 | 非ストリーミングモデル | 低 | P=∞（全コンテキスト参照）のモデルで性能上限を確認 |
| 6.6 | TTS統合評価 | 低 | NANSY-TTS未公開のため、代替TTSモデル（VITS2等のOSS）でMOS評価を実施。または客観指標（UTMOS等）で代替 |

### 最小実装セット（論文の主張を検証するために最低限必要なモデル）

1. **CC-G2PnP-5-1** (C=5, M=1, P=10) — ベスト設定
2. **CC-G2PnP-5-0** (C=5, M=0, P=10) — MLA無しとの比較
3. **CC-G2PnP-NS** (non-streaming) — 性能上限

この3モデルで「MLAの有効性」と「ストリーミング/非ストリーミングの差」を検証可能。

### 完了条件
- [ ] MLAの有無で性能差が確認できる（M=0 vs M=1）
- [ ] チャンクサイズの影響が論文と同傾向
- [ ] 学習データ量に対するスケーリングが論文と同傾向

---

## タイムライン概要

```
Phase 0: 環境構築・データ準備        ████████ ✅ 完了
Phase 1: データパイプライン           ████████ ✅ 完了
Phase 2: モデルコア実装              ████████ ✅ 完了
Phase 3: 学習パイプライン            ████████ ✅ 完了
Phase 4: 推論・ストリーミング         ░░░░░░░░████████████░░░░  ← 次
Phase 5: 評価                       ░░░░░░░░░░░░░░░░░░██████
Phase 6: アブレーション              ░░░░░░░░░░░░░░░░░░░░░░██
```

- ~~Phase 0-2 完了済み~~ → **Phase 0-3 完了済み**
- Phase 3.6（スモールスケール検証）は Phase 4 と並行可能
- Phase 4-5 はPhase 3の学習済みモデルが前提 → **学習基盤は整備済み**
- Phase 6 はPhase 5の評価基盤が前提

---

## 技術的リスクと対策

| リスク | 影響度 | 発生Phase | 対策 |
|---|---|---|---|
| Dict-DNN韻律予測モデルが未公開 | ~~高~~ 対応済 | Phase 1 | pyopenjtalkのfull-context label解析で代替実装完了（ttslearn pp_symbolsベース → CC-G2PnP記法変換）。論文の数値との完全一致は困難だが、アーキテクチャ検証には十分 |
| 6D-Eval評価データが未公開 | **高** | Phase 5 | pyopenjtalkベースの自動評価 → 複数ドメインテキスト収集 → 少量手動アノテーション の段階的アプローチ |
| 1500万件のデータ前処理が膨大 | 中 | Phase 1 | HuggingFace datasets streamingで実装済み。キャッシュ・並列化は必要に応じて追加 |
| 1.2Mステップの学習に長時間 | 中 | Phase 3 | AMP (bfloat16/float16) + DDP + 勾配クリッピング実装済み。まず1%データで検証予定 |
| CTC収束不安定 | ~~中~~ 対策済 | Phase 3 | `zero_infinity=True`、勾配クリッピング(max_norm=1.0)、LinearLR warmup(10,000 steps)を実装済み |
| Self-conditioned CTCのマルチGPU問題 | ~~中~~ 対策済 | Phase 3 | `find_unused_parameters=True`で対処 (ESPnet Issue #4031)。DDP utilities実装済み |
| Chunk-aware Attentionのマスク生成の複雑さ | ~~中~~ 解決済 | Phase 2 | vectorized実装完了。concrete exampleテストで正当性を確認済み |
| pyopenjtalkのインストール問題 | ~~中~~ 解決済 | Phase 0 | `pyopenjtalk-plus>=0.4.1`（Python 3.13 Windows対応フォーク）を採用 |
| NANSY-TTS未公開でMOS評価困難 | 低 | Phase 6 | OSS TTS（VITS2等）で代替、または客観指標（UTMOS）で代替 |
| Conformerのヘッド数/カーネルサイズが論文未記載 | ~~低~~ 解決済 | Phase 2 | Conformer標準値（heads=8, kernel=31）で実装済み。84Mパラメータ |
| ReazonSpeechデータのフィルタリング条件不明 | 低 | Phase 1 | 全データを使用。CTC制約違反・空テキスト・極端な長さのサンプルは自動除外済み |

---

## マイルストーン

| マイルストーン | Phase | 検証基準 | 状態 |
|---|---|---|---|
| **M0: 環境Ready** | 0 | 全ライブラリのimport成功、トークナイザ動作確認 | ✅ 達成 |
| **M1: データパイプライン完成** | 1 | 任意のテキスト → (BPEトークン列, PnPラベル列) のペア生成 | ✅ 達成 |
| **M2: モデルForwardパス** | 2 | ランダム入力でforward pass完了、出力形状が正しい。183テスト、96%カバレッジ、84Mパラメータ | ✅ 達成 |
| **M3-infra: 学習基盤完成** | 3 | Trainer/Optimizer/Scheduler/Checkpoint/DDP/AMP実装完了。327テスト、94%カバレッジ | ✅ 達成 |
| **M3: 小規模学習収束** | 3 | ReazonSpeech 1%でCTC損失が単調減少 | |
| **M4: ストリーミング推論動作** | 4 | チャンク単位の逐次推論でPnPラベル出力 | |
| **M5: 評価メトリクス計測** | 5 | 全6メトリクスの計算、相対的な性能傾向が論文と一致 | |
| **M6: フルスケール学習完了** | 3+6 | 100%データ・1.2Mステップでの学習完了、PnP CER ≈ 1.79（代替評価データ上） | |
