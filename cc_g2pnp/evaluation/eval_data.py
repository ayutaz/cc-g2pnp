"""Evaluation data preparation for CC-G2PnP."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from cc_g2pnp.data.pnp_labeler import generate_pnp_labels

if TYPE_CHECKING:
    from cc_g2pnp.data.tokenizer import G2PnPTokenizer

logger = logging.getLogger(__name__)

# Built-in sample texts across multiple domains for quick evaluation.
# ~10-15 diverse sentences per domain covering different prosody patterns.
BUILTIN_TEXTS: dict[str, list[str]] = {
    "news": [
        "東京都は新しい条例を施行すると発表した。",
        "日経平均株価は前日比二百円高で取引を終えた。",
        "政府は来年度の予算案を閣議決定した。",
        "全国的に気温が上昇し真夏日となった地域もあった。",
        "新幹線の新しい路線が来年開業する予定だ。",
        "国際会議で日本の代表が演説を行った。",
        "大手企業の決算発表が相次いでいる。",
        "震度四の地震が関東地方で観測された。",
        "選挙の投票率は前回を上回る見通しだ。",
        "新型ロケットの打ち上げに成功した。",
    ],
    "conversation": [
        "今日はいい天気ですね。",
        "最近どうですか。",
        "ありがとうございます。",
        "すみません、もう一度お願いします。",
        "明日の予定はどうなっていますか。",
        "お先に失礼します。",
        "それはよかったですね。",
        "ちょっと待ってください。",
        "どこで会いましょうか。",
        "お疲れ様でした。",
    ],
    "literature": [
        "桜の花が風に舞い散る季節になった。",
        "月明かりが静かに庭を照らしていた。",
        "遠くの山々が夕焼けに染まっていく。",
        "雨上がりの空に虹がかかった。",
        "古い時計が静かに時を刻んでいた。",
        "海の向こうに夕日が沈んでいく。",
        "紅葉が山を赤く彩る頃だった。",
        "窓の外で小鳥がさえずっている。",
        "雪が静かに降り積もる夜だった。",
        "春の訪れを告げる風が吹いていた。",
    ],
    "technical": [
        "この関数は引数として文字列を受け取り整数を返す。",
        "ニューラルネットワークの学習率を調整する必要がある。",
        "データベースのインデックスを最適化することで検索速度が向上する。",
        "音声認識の精度を評価するために文字誤り率を計算する。",
        "並列処理によりプログラムの実行速度を改善できる。",
        "暗号化アルゴリズムはデータの安全性を保証する。",
        "クラウドサービスの利用が急速に拡大している。",
        "機械学習モデルの過学習を防ぐために正則化を行う。",
        "アプリケーションの応答時間を短縮することが重要だ。",
        "自然言語処理の分野では大規模言語モデルが注目されている。",
    ],
}


@dataclass
class EvalSample:
    """Single evaluation sample."""

    text: str
    """Original Japanese text."""

    domain: str
    """Domain label (e.g., 'news', 'conversation')."""

    pnp_labels: list[str]
    """Ground truth PnP token sequence from pyopenjtalk."""

    bpe_ids: list[int]
    """BPE token IDs from CALM2 tokenizer."""


@dataclass
class EvalDataset:
    """Collection of evaluation samples."""

    samples: list[EvalSample] = field(default_factory=list)
    """All evaluation samples."""

    name: str = "unnamed"
    """Dataset name."""

    @property
    def domains(self) -> list[str]:
        """Unique domain labels sorted alphabetically."""
        return sorted({s.domain for s in self.samples})

    def filter_by_domain(self, domain: str) -> EvalDataset:
        """Get samples for a specific domain."""
        filtered = [s for s in self.samples if s.domain == domain]
        return EvalDataset(samples=filtered, name=f"{self.name}:{domain}")

    def __len__(self) -> int:
        return len(self.samples)


class EvalDataGenerator:
    """Generate evaluation datasets from text using pyopenjtalk."""

    def __init__(self, tokenizer: G2PnPTokenizer) -> None:
        self._tokenizer = tokenizer

    def from_texts(
        self,
        texts: list[str],
        domain: str = "general",
        name: str = "custom",
    ) -> EvalDataset:
        """Generate eval dataset from a list of texts.

        Skips texts that produce empty PnP labels or empty BPE tokens.
        """
        samples: list[EvalSample] = []
        for text in texts:
            text = text.strip()
            if not text:
                logger.debug("Skipping empty text")
                continue

            pnp_labels = generate_pnp_labels(text)
            if not pnp_labels:
                logger.debug("Skipping text with empty PnP labels: %s", text[:50])
                continue

            bpe_ids = self._tokenizer.encode(text)
            if not bpe_ids:
                logger.debug("Skipping text with empty BPE tokens: %s", text[:50])
                continue

            samples.append(EvalSample(
                text=text,
                domain=domain,
                pnp_labels=pnp_labels,
                bpe_ids=bpe_ids,
            ))

        if not samples:
            logger.warning("No valid samples generated from %d texts", len(texts))

        return EvalDataset(samples=samples, name=name)

    def from_file(
        self,
        path: str | Path,
        domain: str = "general",
        name: str = "file",
        encoding: str = "utf-8",
    ) -> EvalDataset:
        """Load texts from a file (one sentence per line) and generate eval dataset."""
        filepath = Path(path)
        texts = filepath.read_text(encoding=encoding).strip().splitlines()
        return self.from_texts(texts, domain=domain, name=name)

    def builtin_dataset(self) -> EvalDataset:
        """Generate eval dataset from built-in sample texts across multiple domains."""
        all_samples: list[EvalSample] = []
        for domain, texts in BUILTIN_TEXTS.items():
            dataset = self.from_texts(texts, domain=domain, name=f"builtin:{domain}")
            all_samples.extend(dataset.samples)
        return EvalDataset(samples=all_samples, name="builtin")
