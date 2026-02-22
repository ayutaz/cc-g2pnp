"""PnP (Pronunciation and Prosody) label generator.

Converts Japanese text to a sequence of katakana mora tokens with prosody
markers (accent nucleus ``*``, accent phrase boundary ``/``, intonation
phrase boundary ``#``) using pyopenjtalk HTS full-context labels.

The algorithm is based on *ttslearn*'s ``pp_symbols`` (Kurihara et al., 2021)
but reformulated to match the CC-G2PnP notation:

    今日はいい天気  →  キョ * ー ワ / イ * ー / テ * ン キ

Mapping from ttslearn symbols to CC-G2PnP:
    ttslearn ``]`` (pitch fall)       →  ``*``  (accent nucleus)
    ttslearn ``#`` (phrase boundary)  →  ``/``  (accent phrase boundary)
    ttslearn ``_`` (pause)            →  ``#``  (intonation phrase boundary)
    ttslearn ``^``, ``$``, ``?``      →  (omitted — sentence boundary markers)
    ttslearn ``[`` (pitch rise)       →  (omitted — implicit from position)
"""

from __future__ import annotations

import re

import pyopenjtalk

# ── Romaji → Katakana conversion table ───────────────────────────
# pyopenjtalk outputs individual phonemes (e.g. "k", "o").
# We combine consonant(s) + vowel into a single katakana mora.

_CV_TABLE: dict[str, dict[str, str]] = {
    "k": {"a": "カ", "i": "キ", "u": "ク", "e": "ケ", "o": "コ"},
    "s": {"a": "サ", "i": "シ", "u": "ス", "e": "セ", "o": "ソ"},
    "t": {"a": "タ", "i": "チ", "u": "ツ", "e": "テ", "o": "ト"},
    "n": {"a": "ナ", "i": "ニ", "u": "ヌ", "e": "ネ", "o": "ノ"},
    "h": {"a": "ハ", "i": "ヒ", "u": "フ", "e": "ヘ", "o": "ホ"},
    "m": {"a": "マ", "i": "ミ", "u": "ム", "e": "メ", "o": "モ"},
    "y": {"a": "ヤ", "u": "ユ", "o": "ヨ"},
    "r": {"a": "ラ", "i": "リ", "u": "ル", "e": "レ", "o": "ロ"},
    "w": {"a": "ワ", "i": "ウィ", "e": "ウェ", "o": "ウォ"},
    "g": {"a": "ガ", "i": "ギ", "u": "グ", "e": "ゲ", "o": "ゴ"},
    "z": {"a": "ザ", "i": "ジ", "u": "ズ", "e": "ゼ", "o": "ゾ"},
    "d": {"a": "ダ", "i": "ヂ", "u": "ヅ", "e": "デ", "o": "ド"},
    "b": {"a": "バ", "i": "ビ", "u": "ブ", "e": "ベ", "o": "ボ"},
    "p": {"a": "パ", "i": "ピ", "u": "プ", "e": "ペ", "o": "ポ"},
    "f": {"a": "ファ", "i": "フィ", "u": "フ", "e": "フェ", "o": "フォ"},
    "v": {"a": "ヴァ", "i": "ヴィ", "u": "ヴ", "e": "ヴェ", "o": "ヴォ"},
    # Palatalized (yōon) consonants
    "ky": {"a": "キャ", "u": "キュ", "o": "キョ"},
    "sy": {"a": "シャ", "u": "シュ", "o": "ショ"},
    "sh": {"a": "シャ", "i": "シ", "u": "シュ", "e": "シェ", "o": "ショ"},
    "ty": {"a": "チャ", "u": "チュ", "o": "チョ"},
    "ch": {"a": "チャ", "i": "チ", "u": "チュ", "e": "チェ", "o": "チョ"},
    "ny": {"a": "ニャ", "u": "ニュ", "o": "ニョ"},
    "hy": {"a": "ヒャ", "u": "ヒュ", "o": "ヒョ"},
    "my": {"a": "ミャ", "u": "ミュ", "o": "ミョ"},
    "ry": {"a": "リャ", "u": "リュ", "o": "リョ"},
    "gy": {"a": "ギャ", "u": "ギュ", "o": "ギョ"},
    "j": {"a": "ジャ", "i": "ジ", "u": "ジュ", "e": "ジェ", "o": "ジョ"},
    "by": {"a": "ビャ", "u": "ビュ", "o": "ビョ"},
    "py": {"a": "ピャ", "u": "ピュ", "o": "ピョ"},
    "dy": {"a": "ディャ", "u": "デュ", "o": "ディョ"},
    "ts": {"a": "ツァ", "i": "ツィ", "u": "ツ", "e": "ツェ", "o": "ツォ"},
    "fy": {"u": "フュ"},
}

_V_TABLE: dict[str, str] = {
    "a": "ア", "i": "イ", "u": "ウ", "e": "エ", "o": "オ",
    "A": "ア", "I": "イ", "U": "ウ", "E": "エ", "O": "オ",
}

_SPECIAL_PHONEME: dict[str, str] = {
    "N": "ン",
    "cl": "ッ",
}

_CONSONANTS = set(_CV_TABLE.keys())
_VOWELS_SET = set(_V_TABLE.keys())

# Vowel base character for long-vowel detection (lowercase)
_VOWEL_OF_KANA: dict[str, str] = {}
# Build reverse lookup: katakana → base vowel
for _c, _vmap in _CV_TABLE.items():
    for _v, _kana in _vmap.items():
        _VOWEL_OF_KANA[_kana] = _v
for _v, _kana in _V_TABLE.items():
    _VOWEL_OF_KANA[_kana] = _v.lower()
_VOWEL_OF_KANA["ン"] = ""
_VOWEL_OF_KANA["ッ"] = ""

# ── Regex helpers ────────────────────────────────────────────────

_RE_P3 = re.compile(r"\-(.*?)\+")


def _numeric_feature(regex: str, label: str) -> int | None:
    """Extract a numeric feature from an HTS label, returning None for 'xx'."""
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


def _get_p3(label: str) -> str:
    """Extract center phoneme (p3) from an HTS label."""
    m = _RE_P3.search(label)
    if m is None:
        raise ValueError(f"Cannot extract p3 from label: {label}")
    return m.group(1)


# ── Step 1: ttslearn-style prosody extraction (phoneme level) ────

_MORA_PHONEMES = set("aeiouAEIOU") | {"N", "cl"}


def _extract_pp_symbols(labels: list[str]) -> list[str]:
    """Extract phoneme sequence with ttslearn-style prosody markers.

    This implements the core of ttslearn's ``pp_symbols`` function.
    Returns a list of phoneme strings interleaved with prosody markers:
        ``]`` = accent nucleus (pitch fall)
        ``#`` = accent phrase boundary
        ``_`` = pause (IP boundary)

    Sentence boundary markers (``^``, ``$``, ``?``, ``[``) are omitted.
    """
    result: list[str] = []
    n = len(labels)

    for idx in range(n):
        lab_curr = labels[idx]
        p3 = _get_p3(lab_curr)

        # Skip sentence-initial and sentence-final silence
        if p3 == "sil":
            continue

        # Pause → IP boundary marker
        if p3 == "pau":
            result.append("_")
            continue

        # Extract features from current label
        a1 = _numeric_feature(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature(r"\+(\d+)/", lab_curr)
        f1 = _numeric_feature(r"/F:(\d+)_", lab_curr)

        # Extract a2 from next label
        a2_next = None
        if idx + 1 < n:
            a2_next = _numeric_feature(r"\+(\d+)\+", labels[idx + 1])

        # Emit phoneme
        result.append(p3)

        # Only check prosody at mora boundaries (vowels, N, cl)
        if p3 not in _MORA_PHONEMES:
            continue

        # Check accent phrase boundary:
        # a3 == 1 means last mora of accent phrase,
        # a2_next == 1 means next is first mora of new accent phrase
        if (
            a3 is not None
            and a2_next is not None
            and a3 == 1
            and a2_next == 1
        ):
            result.append("#")  # accent phrase boundary
        # Check accent nucleus (pitch fall):
        # a1 == 0 indicates this mora precedes pitch drop.
        # Condition: a1 == 0, next mora has a2 incremented, and not at end of phrase.
        elif (
            a1 is not None
            and a2 is not None
            and f1 is not None
            and a2_next is not None
            and a1 == 0
            and a2_next == a2 + 1
            and a2 != f1
        ):
            result.append("]")  # accent nucleus

    return result


# ── Step 2: Phoneme sequence → Katakana mora sequence ────────────

def _phonemes_to_mora(pp_symbols: list[str]) -> list[str]:
    """Convert phoneme+prosody sequence to katakana mora+prosody sequence.

    Groups consonant+vowel into single mora tokens. Prosody markers are
    passed through as-is. Consecutive same vowels are converted to long
    vowel mark (ー), even when separated by prosody markers.
    """
    result: list[str] = []
    i = 0
    n = len(pp_symbols)
    # Track the last emitted katakana for long-vowel detection.
    # Prosody markers do NOT reset this — long vowels span across markers.
    prev_kana: str | None = None

    while i < n:
        tok = pp_symbols[i]

        # Prosody markers: pass through and map to CC-G2PnP notation
        if tok in ("]", "#", "_"):
            if tok == "]":
                result.append("*")   # accent nucleus
                # Accent nucleus does NOT break long vowel tracking —
                # e.g. キョ * ー (kyoo with accent on first mora)
            elif tok == "#":
                result.append("/")   # accent phrase boundary
                prev_kana = None     # reset — different accent phrase
            elif tok == "_":
                result.append("#")   # intonation phrase boundary (pause)
                prev_kana = None     # reset — different IP
            i += 1
            continue

        # Special phonemes (standalone mora)
        if tok in _SPECIAL_PHONEME:
            kana = _SPECIAL_PHONEME[tok]
            result.append(kana)
            prev_kana = kana
            i += 1
            continue

        # Standalone vowel
        if tok in _VOWELS_SET:
            kana = _V_TABLE[tok]
            # Long vowel detection: if previous mora ends with same vowel
            vowel = tok.lower()
            if prev_kana is not None and _VOWEL_OF_KANA.get(prev_kana) == vowel:
                kana = "ー"
            result.append(kana)
            if kana != "ー":
                prev_kana = kana  # update base for chaining
            # If ー, keep prev_kana so triple-long vowels work
            i += 1
            continue

        # Consonant: try to combine with following vowel(s)
        if tok in _CONSONANTS:
            # Direct CV match
            if i + 1 < n and pp_symbols[i + 1] in _VOWELS_SET:
                vowel = pp_symbols[i + 1].lower()
                kana = _CV_TABLE.get(tok, {}).get(vowel)
                if kana:
                    result.append(kana)
                    prev_kana = kana
                    i += 2
                    continue

            # Two-character consonant (e.g. "k"+"y" → "ky")
            if i + 1 < n and pp_symbols[i + 1] not in _VOWELS_SET and pp_symbols[i + 1] not in ("]", "#", "_"):
                two_c = tok + pp_symbols[i + 1]
                if two_c in _CV_TABLE and i + 2 < n and pp_symbols[i + 2] in _VOWELS_SET:
                    vowel = pp_symbols[i + 2].lower()
                    kana = _CV_TABLE[two_c].get(vowel)
                    if kana:
                        result.append(kana)
                        prev_kana = kana
                        i += 3
                        continue

            # Unmatched consonant — skip
            i += 1
            continue

        # Unknown token — skip
        i += 1

    return result


# ── Public API ───────────────────────────────────────────────────

def generate_pnp_labels(text: str) -> list[str]:
    """Generate PnP label sequence from Japanese text.

    Args:
        text: Input Japanese text.

    Returns:
        List of PnP tokens (katakana mora + prosody symbols).
        Example: ``["キョ", "*", "ー", "ワ", "/", "イ", "*", "ー", "/", "テ", "*", "ン", "キ"]``
    """
    if not text or not text.strip():
        return []

    labels = pyopenjtalk.extract_fullcontext(text)
    if not labels:
        return []

    pp_symbols = _extract_pp_symbols(labels)
    return _phonemes_to_mora(pp_symbols)
