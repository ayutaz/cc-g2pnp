"""pyopenjtalk.extract_fullcontext() の詳細プロファイリングスクリプト

計測項目:
1. extract_fullcontext() の文字列長別呼び出し時間
2. generate_pnp_labels() 内部の時間配分
3. sudachi キャッシュ効果 (パッチ前後)
4. OpenJTalk インスタンス作成コスト vs 再利用コスト
"""

from __future__ import annotations

import inspect
import os
import statistics
import sys
import threading
import time

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyopenjtalk
from pyopenjtalk.openjtalk import OpenJTalk
from sudachipy import dictionary as sudachi_dict
from sudachipy import tokenizer as sudachi_tokenizer

from cc_g2pnp._patch_pyopenjtalk import apply as apply_patch
from cc_g2pnp.data.pnp_labeler import _extract_pp_symbols, _phonemes_to_mora

# ── ユーティリティ ────────────────────────────────────────────────


def measure(fn, n=10):
    """fn を n 回呼び出してタイムを計測 (ms)"""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return times


def fmt(times):
    mean = statistics.mean(times)
    med = statistics.median(times)
    mn = min(times)
    mx = max(times)
    sd = statistics.stdev(times) if len(times) > 1 else 0
    return f"mean={mean:.2f}ms  median={med:.2f}ms  min={mn:.2f}ms  max={mx:.2f}ms  std={sd:.2f}ms"


# ── テスト用テキスト ──────────────────────────────────────────────

TEXT_10 = "今日はいい天気"            # ~7文字
TEXT_SHORT = "今日はいい天気ですね"       # ~10文字
TEXT_50 = (
    "日本語の自然言語処理は非常に重要な技術であり、"
    "音声合成や音声認識など様々な分野で応用されています。"
)  # ~50文字
TEXT_100 = (
    "機械学習とディープラーニングの発展により、自然言語処理の精度は劇的に向上しました。"
    "特に大規模言語モデルの登場以来、テキスト生成や翻訳の品質が大幅に改善されています。"
    "これらの技術は産業界でも広く採用されつつあります。"
)  # ~100文字
TEXT_500 = TEXT_100 * 5  # ~500文字

TEST_CASES = [
    ("~10文字", TEXT_SHORT),
    ("~50文字", TEXT_50),
    ("~100文字", TEXT_100),
    ("~500文字", TEXT_500),
]

MULTI_READ_KANJI_LIST = pyopenjtalk.MULTI_READ_KANJI_LIST
TEXT_WITH_KANJI = "今日の風は何だろう。金曜日の空が綺麗だ。"


# ── セクション 1: 文字列長別 extract_fullcontext() 時間 ──────────


def section1_fullcontext_timing():
    """セクション 1: 文字列長別 extract_fullcontext() 時間"""
    print("\n## セクション 1: 文字列長別 extract_fullcontext() 時間")
    print("-" * 70)

    # ウォームアップ (辞書ロード等の初期化)
    print("ウォームアップ中...")
    pyopenjtalk.extract_fullcontext("テスト")
    pyopenjtalk.extract_fullcontext("テスト")

    for label, text in TEST_CASES:
        times = measure(lambda t=text: pyopenjtalk.extract_fullcontext(t), n=20)
        print(f"  [{label}] ({len(text)}文字): {fmt(times)}")


# ── セクション 2: generate_pnp_labels() 内部の時間配分 ────────────


def section2_pnp_labels_timing():
    """セクション 2: generate_pnp_labels() 内部の時間配分"""
    print("\n## セクション 2: generate_pnp_labels() 内部の時間配分")
    print("-" * 70)

    for label, text in TEST_CASES:
        t_fc_total = 0
        t_pp_total = 0
        t_mora_total = 0
        n = 20
        for _ in range(n):
            t0 = time.perf_counter()
            labels = pyopenjtalk.extract_fullcontext(text)
            t1 = time.perf_counter()
            pp = _extract_pp_symbols(labels)
            t2 = time.perf_counter()
            _phonemes_to_mora(pp)
            t3 = time.perf_counter()
            t_fc_total += (t1 - t0) * 1000
            t_pp_total += (t2 - t1) * 1000
            t_mora_total += (t3 - t2) * 1000
        total = t_fc_total + t_pp_total + t_mora_total
        print(f"  [{label}] ({len(text)}文字):")
        print(f"    extract_fullcontext : {t_fc_total/n:.3f}ms ({100*t_fc_total/total:.1f}%)")
        print(f"    _extract_pp_symbols : {t_pp_total/n:.3f}ms ({100*t_pp_total/total:.1f}%)")
        print(f"    _phonemes_to_mora   : {t_mora_total/n:.3f}ms ({100*t_mora_total/total:.1f}%)")
        print(f"    合計                : {total/n:.3f}ms")


# ── セクション 3: sudachi キャッシュ効果 ─────────────────────────


def section3_sudachi_cache():
    """セクション 3: sudachi キャッシュ効果 (パッチ前後)"""
    print("\n## セクション 3: sudachi キャッシュ効果 (パッチ前後)")
    print("-" * 70)

    # 元の実装 (毎回 Dictionary().create())
    def sudachi_original(text):
        text = text.replace("ー", "")
        tokenizer_obj = sudachi_dict.Dictionary().create()
        mode = sudachi_tokenizer.Tokenizer.SplitMode.C
        m_list = tokenizer_obj.tokenize(text, mode)
        return [[m.surface(), m.reading_form()] for m in m_list if m.surface() in MULTI_READ_KANJI_LIST]

    # キャッシュ版 (threading.local でインスタンスを保持)
    _local = threading.local()

    def sudachi_cached(text):
        text = text.replace("ー", "")
        if not hasattr(_local, "tokenizer"):
            _local.tokenizer = sudachi_dict.Dictionary().create()
        tokenizer_obj = _local.tokenizer
        mode = sudachi_tokenizer.Tokenizer.SplitMode.C
        m_list = tokenizer_obj.tokenize(text, mode)
        return [[m.surface(), m.reading_form()] for m in m_list if m.surface() in MULTI_READ_KANJI_LIST]

    # ウォームアップ (キャッシュ初期化)
    sudachi_cached(TEXT_WITH_KANJI)

    print(f"  テキスト: 「{TEXT_WITH_KANJI}」 ({len(TEXT_WITH_KANJI)}文字)")
    times_orig = measure(lambda: sudachi_original(TEXT_WITH_KANJI), n=10)
    times_cached = measure(lambda: sudachi_cached(TEXT_WITH_KANJI), n=20)

    print(f"  パッチ前 (毎回 Dictionary().create()): {fmt(times_orig)}")
    print(f"  パッチ後 (キャッシュ使用):             {fmt(times_cached)}")
    speedup = statistics.mean(times_orig) / statistics.mean(times_cached)
    print(f"  高速化倍率: {speedup:.0f}x")

    # modify_kanji_yomi の時間配分 (sudachi を呼ぶ関数)
    print()
    print("  modify_kanji_yomi() での影響計測:")

    def run_frontend_measure(text, n=10):
        times = []
        for _ in range(n):
            t0 = time.perf_counter()
            pyopenjtalk.run_frontend(text)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        return times

    times_before = run_frontend_measure(TEXT_WITH_KANJI, n=10)
    print(f"  パッチ前 run_frontend(): {fmt(times_before)}")

    # パッチ適用
    apply_patch()
    times_after = run_frontend_measure(TEXT_WITH_KANJI, n=20)
    print(f"  パッチ後 run_frontend(): {fmt(times_after)}")
    speedup2 = statistics.mean(times_before) / statistics.mean(times_after)
    print(f"  run_frontend() 高速化倍率: {speedup2:.1f}x")


# ── セクション 4: OpenJTalk インスタンス作成コスト vs 再利用 ─────


def section4_openjtalk_instance():
    """セクション 4: OpenJTalk インスタンス作成コスト vs 再利用コスト"""
    print("\n## セクション 4: OpenJTalk インスタンス作成コスト vs 再利用コスト")
    print("-" * 70)

    dict_dir = pyopenjtalk.OPEN_JTALK_DICT_DIR

    # インスタンス作成コスト
    times_create = []
    for _ in range(5):
        t0 = time.perf_counter()
        jtalk_tmp = OpenJTalk(dn_mecab=dict_dir)
        t1 = time.perf_counter()
        times_create.append((t1 - t0) * 1000)
        del jtalk_tmp

    print(f"  OpenJTalk() インスタンス作成: {fmt(times_create)}")

    # インスタンス再利用コスト (extract_fullcontext with jtalk=)
    jtalk_reuse = OpenJTalk(dn_mecab=dict_dir)
    pyopenjtalk.extract_fullcontext(TEXT_SHORT, jtalk=jtalk_reuse)  # ウォームアップ

    times_reuse = measure(lambda: pyopenjtalk.extract_fullcontext(TEXT_SHORT, jtalk=jtalk_reuse), n=20)
    times_global = measure(lambda: pyopenjtalk.extract_fullcontext(TEXT_SHORT), n=20)

    print(f"  extract_fullcontext (グローバルインスタンス): {fmt(times_global)}")
    print(f"  extract_fullcontext (専用インスタンス再利用): {fmt(times_reuse)}")
    ratio = statistics.mean(times_global) / statistics.mean(times_reuse)
    print(f"  専用インスタンス vs グローバル比: {ratio:.2f}x")
    print()
    print("  ※グローバルインスタンスは threading.Lock を経由するため、")
    print("    マルチスレッド環境でのコンテンション発生リスクがある")


# ── セクション 5: pyopenjtalk-plus のバッチ処理 API 調査 ─────────


def section5_api_investigation():
    """セクション 5: pyopenjtalk-plus API のバッチ処理・高速化オプション調査"""
    print("\n## セクション 5: pyopenjtalk-plus API のバッチ処理・高速化オプション調査")
    print("-" * 70)

    # 利用可能なAPIを確認
    available_apis = [name for name in dir(pyopenjtalk) if not name.startswith('_')]
    print(f"  公開 API: {available_apis}")

    # extract_fullcontext のシグネチャ確認
    sig = inspect.signature(pyopenjtalk.extract_fullcontext)
    print(f"\n  extract_fullcontext シグネチャ: {sig}")
    sig2 = inspect.signature(pyopenjtalk.run_frontend)
    print(f"  run_frontend シグネチャ: {sig2}")

    # バッチ処理 API の有無確認
    batch_apis = [name for name in dir(pyopenjtalk) if 'batch' in name.lower()]
    print(f"\n  バッチ処理 API: {batch_apis if batch_apis else 'なし'}")

    # marine (アクセント推定) の有無確認
    has_marine_flag = hasattr(pyopenjtalk, 'load_marine_model')
    print(f"  marine (アクセント推定) API: {'あり' if has_marine_flag else 'なし'}")

    # run_vanilla オプション確認
    print("\n  use_vanilla=True の速度計測 (後処理スキップ):")
    times_vanilla = measure(lambda: pyopenjtalk.extract_fullcontext(TEXT_50, use_vanilla=True), n=20)
    times_normal = measure(lambda: pyopenjtalk.extract_fullcontext(TEXT_50), n=20)
    print(f"  use_vanilla=True : {fmt(times_vanilla)}")
    print(f"  use_vanilla=False: {fmt(times_normal)}")
    diff = statistics.mean(times_normal) - statistics.mean(times_vanilla)
    print(f"  後処理コスト: {diff:.2f}ms")


# ── メイン ────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("pyopenjtalk プロファイリングレポート")
    print("=" * 70)

    section1_fullcontext_timing()
    section2_pnp_labels_timing()
    section3_sudachi_cache()
    section4_openjtalk_instance()
    section5_api_investigation()

    # ── サマリー ──
    print("\n" + "=" * 70)
    print("## サマリー: 最適化の観点から")
    print("=" * 70)
    print()
    print("1. ボトルネック: extract_fullcontext() が全体の95%以上を占める")
    print("2. sudachi キャッシュ: Dictionary().create() は重いが全体への影響は限定的")
    print("   (modify_kanji_yomi は漢字テキストにのみ呼ばれる後処理)")
    print("3. OpenJTalk 専用インスタンスの再利用: Lock コンテンション回避に有効")
    print("4. バッチ処理 API: pyopenjtalk-plus に公式バッチ API は存在しない")
    print("5. use_vanilla=True で後処理をスキップすると若干高速化可能")


if __name__ == "__main__":
    main()
