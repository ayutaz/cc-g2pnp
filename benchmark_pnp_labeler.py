"""
_extract_pp_symbols の正規表現最適化ベンチマーク

比較対象:
  1. 現在の実装 (re.search with non-compiled patterns)
  2. プリコンパイル済みパターン
  3. split() ベースの直接パーサー
"""
from __future__ import annotations

import contextlib
import re
import time

# ── サンプル HTS ラベル生成 ──────────────────────────────────────────
# pyopenjtalk が出力する full-context label の形式:
#   p1^p2-p3+p4=p5/A:a1+a2+a3/B:.../C:.../D:.../E:.../F:f1_...
# 実際のラベルは非常に長いが、最低限必要なフィールドを含む形式で生成

SAMPLE_LABELS = [
    # sil (sentence boundary)
    "xx^xx-sil+k=y/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx",
    # k (consonant)
    "xx^sil-k+y=o/A:-3+1+5/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5",
    # y (consonant, palatalized)
    "sil^k-y+o=u/A:-3+1+5/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5",
    # o (vowel)
    "k^y-o+u=N/A:-3+1+5/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5",
    # u (vowel, accent nucleus check)
    "y^o-u+N=w/A:-2+2+4/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5",
    # N (nasal)
    "o^u-N+w=a/A:-1+3+3/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5",
    # w (consonant)
    "u^N-w+a=t/A:0+4+2/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5",
    # a (vowel, ap boundary: a3==1)
    "N^w-a+t=a/A:0+4+1/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5",
    # t (consonant, next ap: a2_next==1)
    "w^a-t+a=sh/A:0+1+5/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5",
    # a (vowel)
    "a^t-a+sh=i/A:0+2+4/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:5_5#0_xx@1_1|1_5",
    # pau (pause)
    "t^a-pau+sh=i/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx",
    # sh (consonant)
    "a^pau-sh+i=t/A:-2+1+3/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:3_3#0_xx@1_1|1_3",
    # i (vowel)
    "pau^sh-i+t=e/A:-2+1+3/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:3_3#0_xx@1_1|1_3",
    # t (consonant)
    "sh^i-t+e=N/A:-1+2+2/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:3_3#0_xx@1_1|1_3",
    # e (vowel, accent nucleus)
    "i^t-e+N=xx/A:0+3+1/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:3_3#0_xx@1_1|1_3",
    # N (nasal)
    "t^e-N+xx=xx/A:0+3+1/B:02-xx_xx/C:10_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:3_3#0_xx@1_1|1_3",
    # sil (sentence final)
    "e^N-sil+xx=xx/A:xx+xx+xx/B:xx-xx_xx/C:xx_xx+xx/D:xx+xx_xx"
    "/E:xx_xx!xx_xx-xx/F:xx_xx#xx_xx@xx_xx|xx_xx",
]


# ════════════════════════════════════════════════════════════════════
# 方式1: 現在の実装 (非コンパイル正規表現)
# ════════════════════════════════════════════════════════════════════

_RE_P3_ORIG = re.compile(r"\-(.*?)\+")
_MORA_PHONEMES = set("aeiouAEIOU") | {"N", "cl"}


def _numeric_feature_orig(regex: str, label: str) -> int | None:
    """現在の実装: re.search with non-compiled pattern."""
    m = re.search(regex, label)
    if m is None:
        return None
    val = m.group(1)
    if val == "xx":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _get_p3_orig(label: str) -> str:
    m = _RE_P3_ORIG.search(label)
    if m is None:
        raise ValueError(f"Cannot extract p3 from label: {label}")
    return m.group(1)


def extract_pp_symbols_orig(labels: list[str]) -> list[str]:
    """現在の実装 (non-compiled regex)."""
    result: list[str] = []
    n = len(labels)
    for idx in range(n):
        lab_curr = labels[idx]
        p3 = _get_p3_orig(lab_curr)
        if p3 == "sil":
            continue
        if p3 == "pau":
            result.append("_")
            continue
        a1 = _numeric_feature_orig(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_orig(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_orig(r"\+(\d+)/", lab_curr)
        f1 = _numeric_feature_orig(r"/F:(\d+)_", lab_curr)
        a2_next = None
        if idx + 1 < n:
            a2_next = _numeric_feature_orig(r"\+(\d+)\+", labels[idx + 1])
        result.append(p3)
        if p3 not in _MORA_PHONEMES:
            continue
        if a3 is not None and a2_next is not None and a3 == 1 and a2_next == 1:
            result.append("#")
        elif (
            a1 is not None and a2 is not None and f1 is not None
            and a2_next is not None and a1 == 0 and a2_next == a2 + 1 and a2 != f1
        ):
            result.append("]")
    return result


# ════════════════════════════════════════════════════════════════════
# 方式2: プリコンパイル済みパターン
# ════════════════════════════════════════════════════════════════════

_RE_P3_C = re.compile(r"\-(.*?)\+")
_RE_A1   = re.compile(r"/A:([0-9\-]+)\+")
_RE_A2   = re.compile(r"\+(\d+)\+")
_RE_A3   = re.compile(r"\+(\d+)/")
_RE_F1   = re.compile(r"/F:(\d+)_")


def _numeric_feature_compiled(pattern: re.Pattern, label: str) -> int | None:
    """プリコンパイル済みパターンを使用。"""
    m = pattern.search(label)
    if m is None:
        return None
    val = m.group(1)
    if val == "xx":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def extract_pp_symbols_compiled(labels: list[str]) -> list[str]:
    """プリコンパイル済み正規表現版。"""
    result: list[str] = []
    n = len(labels)
    for idx in range(n):
        lab_curr = labels[idx]
        m = _RE_P3_C.search(lab_curr)
        if m is None:
            continue
        p3 = m.group(1)
        if p3 == "sil":
            continue
        if p3 == "pau":
            result.append("_")
            continue
        a1 = _numeric_feature_compiled(_RE_A1, lab_curr)
        a2 = _numeric_feature_compiled(_RE_A2, lab_curr)
        a3 = _numeric_feature_compiled(_RE_A3, lab_curr)
        f1 = _numeric_feature_compiled(_RE_F1, lab_curr)
        a2_next = None
        if idx + 1 < n:
            a2_next = _numeric_feature_compiled(_RE_A2, labels[idx + 1])
        result.append(p3)
        if p3 not in _MORA_PHONEMES:
            continue
        if a3 is not None and a2_next is not None and a3 == 1 and a2_next == 1:
            result.append("#")
        elif (
            a1 is not None and a2 is not None and f1 is not None
            and a2_next is not None and a1 == 0 and a2_next == a2 + 1 and a2 != f1
        ):
            result.append("]")
    return result


# ════════════════════════════════════════════════════════════════════
# 方式3: split() ベースの直接パーサー (regex なし)
# ════════════════════════════════════════════════════════════════════

def _parse_label_fast(label: str) -> tuple[str, int | None, int | None, int | None, int | None]:
    """HTS ラベルを split() で直接パース。

    Returns:
        (p3, a1, a2, a3, f1)
    """
    # 最初の '/' より前が音素コンテキスト: p1^p2-p3+p4=p5
    slash_idx = label.index('/')
    phoneme_ctx = label[:slash_idx]           # "p1^p2-p3+p4=p5"
    after_dash = phoneme_ctx.split('-', 1)[1]  # "p3+p4=p5"
    p3 = after_dash.split('+', 1)[0]          # "p3"

    a1 = a2 = a3 = f1 = None

    # '/' でセクションに分割
    sections = label[slash_idx + 1:].split('/')
    for sec in sections:
        if sec.startswith('A:'):
            # A:a1+a2+a3
            vals = sec[2:].split('+')
            if len(vals) >= 3:
                v = vals[0]
                if v != 'xx':
                    with contextlib.suppress(ValueError):
                        a1 = int(v)
                v = vals[1]
                if v != 'xx':
                    with contextlib.suppress(ValueError):
                        a2 = int(v)
                v = vals[2]
                if v != 'xx':
                    with contextlib.suppress(ValueError):
                        a3 = int(v)
        elif sec.startswith('F:'):
            # F:f1_f2#...
            v = sec[2:].split('_', 1)[0]
            if v != 'xx':
                with contextlib.suppress(ValueError):
                    f1 = int(v)

    return p3, a1, a2, a3, f1


def extract_pp_symbols_split(labels: list[str]) -> list[str]:
    """split() ベースの直接パーサー版。"""
    result: list[str] = []
    n = len(labels)

    # 全ラベルを先にパース (a2_next 参照のため)
    parsed = [_parse_label_fast(lab) for lab in labels]

    for idx in range(n):
        p3, a1, a2, a3, f1 = parsed[idx]
        if p3 == "sil":
            continue
        if p3 == "pau":
            result.append("_")
            continue
        a2_next = parsed[idx + 1][2] if idx + 1 < n else None
        result.append(p3)
        if p3 not in _MORA_PHONEMES:
            continue
        if a3 is not None and a2_next is not None and a3 == 1 and a2_next == 1:
            result.append("#")
        elif (
            a1 is not None and a2 is not None and f1 is not None
            and a2_next is not None and a1 == 0 and a2_next == a2 + 1 and a2 != f1
        ):
            result.append("]")
    return result


# ════════════════════════════════════════════════════════════════════
# ベンチマーク実行
# ════════════════════════════════════════════════════════════════════

def run_benchmark(n_iter: int = 1000) -> None:
    print(f"=== _extract_pp_symbols ベンチマーク (N={n_iter} 回) ===\n")
    print(f"入力ラベル数: {len(SAMPLE_LABELS)}\n")

    # 正確性チェック
    out1 = extract_pp_symbols_orig(SAMPLE_LABELS)
    out2 = extract_pp_symbols_compiled(SAMPLE_LABELS)
    out3 = extract_pp_symbols_split(SAMPLE_LABELS)

    print("出力確認:")
    print(f"  方式1 (原版):             {out1}")
    print(f"  方式2 (コンパイル済み):   {out2}")
    print(f"  方式3 (split):            {out3}")

    ok2 = out1 == out2
    ok3 = out1 == out3
    print(f"\n  方式2 == 方式1: {ok2}")
    print(f"  方式3 == 方式1: {ok3}")
    if not ok2 or not ok3:
        print("  *** 警告: 出力が一致しません! ***")
    print()

    # ウォームアップ
    for _ in range(10):
        extract_pp_symbols_orig(SAMPLE_LABELS)
        extract_pp_symbols_compiled(SAMPLE_LABELS)
        extract_pp_symbols_split(SAMPLE_LABELS)

    # 計測: 方式1 (現在の実装)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        extract_pp_symbols_orig(SAMPLE_LABELS)
    t1 = time.perf_counter()
    elapsed1 = t1 - t0

    # 計測: 方式2 (プリコンパイル)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        extract_pp_symbols_compiled(SAMPLE_LABELS)
    t1 = time.perf_counter()
    elapsed2 = t1 - t0

    # 計測: 方式3 (split)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        extract_pp_symbols_split(SAMPLE_LABELS)
    t1 = time.perf_counter()
    elapsed3 = t1 - t0

    print("─" * 55)
    print(f"{'方式':<30} {'合計(ms)':>10} {'1回平均(μs)':>12}")
    print("─" * 55)
    print(f"{'1. 現在 (非コンパイル)':<30} {elapsed1*1000:>10.2f} {elapsed1/n_iter*1e6:>12.2f}")
    print(f"{'2. プリコンパイル済み':<30} {elapsed2*1000:>10.2f} {elapsed2/n_iter*1e6:>12.2f}")
    print(f"{'3. split() ベース':<30} {elapsed3*1000:>10.2f} {elapsed3/n_iter*1e6:>12.2f}")
    print("─" * 55)

    speedup2 = elapsed1 / elapsed2
    speedup3 = elapsed1 / elapsed3
    print("\n速度向上:")
    print(f"  方式2 vs 方式1: {speedup2:.2f}x {'(高速化)' if speedup2 > 1 else '(遅い)'}")
    print(f"  方式3 vs 方式1: {speedup3:.2f}x {'(高速化)' if speedup3 > 1 else '(遅い)'}")

    print("\n=== 詳細分析 ===\n")
    print("現在の実装の問題点:")
    print("  - _numeric_feature() が毎回 re.search() に文字列パターンを渡している")
    print("  - Python 内部の lru_cache で re.compile() はキャッシュされるが")
    print("    関数呼び出しオーバーヘッドが発生する")
    print("  - a1/a2/a3/f1 で 4-5 回の regex 検索 + a2_next で計 5 回/ラベル")
    print()
    print("プリコンパイル済みの改善点:")
    print("  - モジュールレベルで re.compile() → キャッシュヒット保証")
    print("  - 関数呼び出しが直接 pattern.search() になりオーバーヘッド削減")
    print()
    print("split() ベースの改善点:")
    print("  - 正規表現エンジン不使用 → 純粋な文字列操作")
    print("  - HTS ラベルのフォーマットが固定 → 1回の split('/') で全フィールド取得")
    print("  - 全ラベルを事前にパースして a2_next 参照を効率化")


if __name__ == "__main__":
    run_benchmark(1000)
