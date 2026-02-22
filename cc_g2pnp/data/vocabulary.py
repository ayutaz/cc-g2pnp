"""CTC vocabulary for PnP (Pronunciation and Prosody) label sequences.

Defines the mapping between katakana mora / prosody symbols and integer IDs
used as CTC target labels.

Layout:
    ID 0       : <blank>  (CTC blank token)
    ID 1..N    : katakana mora (ア〜ヴォ)
    ID N+1     : *  (accent nucleus)
    ID N+2     : /  (accent phrase boundary)
    ID N+3     : #  (intonation phrase boundary)
    ID N+4     : <unk>
    ID N+5     : <pad>
"""

from __future__ import annotations

# ── Katakana mora inventory ──────────────────────────────────────
# Ordered: vowels → basic consonant rows → voiced → semi-voiced
# → geminate/moraic nasal → long vowel → yōon → special/loanword

_VOWELS = ["ア", "イ", "ウ", "エ", "オ"]

_BASIC = [
    # ka-row
    "カ", "キ", "ク", "ケ", "コ",
    # sa-row
    "サ", "シ", "ス", "セ", "ソ",
    # ta-row
    "タ", "チ", "ツ", "テ", "ト",
    # na-row
    "ナ", "ニ", "ヌ", "ネ", "ノ",
    # ha-row
    "ハ", "ヒ", "フ", "ヘ", "ホ",
    # ma-row
    "マ", "ミ", "ム", "メ", "モ",
    # ya-row
    "ヤ", "ユ", "ヨ",
    # ra-row
    "ラ", "リ", "ル", "レ", "ロ",
    # wa-row
    "ワ", "ヲ",
]

_VOICED = [
    # ga-row
    "ガ", "ギ", "グ", "ゲ", "ゴ",
    # za-row
    "ザ", "ジ", "ズ", "ゼ", "ゾ",
    # da-row
    "ダ", "ヂ", "ヅ", "デ", "ド",
    # ba-row
    "バ", "ビ", "ブ", "ベ", "ボ",
]

_SEMI_VOICED = [
    "パ", "ピ", "プ", "ペ", "ポ",
]

_SPECIAL_MORA = [
    "ン",   # moraic nasal (N)
    "ッ",   # geminate (cl / Q)
    "ー",   # long vowel mark (chōon)
]

_YOON = [
    # ki-row
    "キャ", "キュ", "キョ",
    # shi-row
    "シャ", "シュ", "ショ",
    # chi-row
    "チャ", "チュ", "チョ",
    # ni-row
    "ニャ", "ニュ", "ニョ",
    # hi-row
    "ヒャ", "ヒュ", "ヒョ",
    # mi-row
    "ミャ", "ミュ", "ミョ",
    # ri-row
    "リャ", "リュ", "リョ",
    # gi-row
    "ギャ", "ギュ", "ギョ",
    # ji-row
    "ジャ", "ジュ", "ジョ",
    # bi-row
    "ビャ", "ビュ", "ビョ",
    # pi-row
    "ピャ", "ピュ", "ピョ",
]

_LOANWORD = [
    # f-series
    "ファ", "フィ", "フェ", "フォ", "フュ",
    # w-series
    "ウィ", "ウェ", "ウォ",
    # v-series
    "ヴァ", "ヴィ", "ヴ", "ヴェ", "ヴォ",
    # t-series
    "ティ", "トゥ",
    # d-series
    "ディ", "ドゥ",
    # ts-series (ツァ etc.)
    "ツァ", "ツィ", "ツェ", "ツォ",
    # sh/ch/j-series (loanword e-vowel)
    "シェ", "チェ", "ジェ",
    # dy-series
    "デュ", "ディャ", "ディョ",
    # ty-series
    "テュ",
]

PHONEMES: list[str] = (
    _VOWELS + _BASIC + _VOICED + _SEMI_VOICED + _SPECIAL_MORA + _YOON + _LOANWORD
)

PROSODY_SYMBOLS: list[str] = [
    "*",  # accent nucleus
    "/",  # accent phrase boundary
    "#",  # intonation phrase boundary
]

SPECIAL_TOKENS: list[str] = [
    "<blank>",  # CTC blank — must be ID 0
    "<unk>",
    "<pad>",
]


class PnPVocabulary:
    """Bidirectional mapping between PnP tokens and integer IDs."""

    def __init__(self) -> None:
        tokens: list[str] = []
        # ID 0: <blank>
        tokens.append("<blank>")
        # ID 1..N: katakana mora
        tokens.extend(PHONEMES)
        # prosody symbols
        tokens.extend(PROSODY_SYMBOLS)
        # remaining special tokens
        tokens.append("<unk>")
        tokens.append("<pad>")

        self._tokens = tokens
        self._token_to_id: dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self._id_to_token: dict[int, str] = {i: t for i, t in enumerate(tokens)}

    # ── properties ───────────────────────────────────────────────
    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return self._token_to_id["<unk>"]

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<pad>"]

    @property
    def token_to_id(self) -> dict[str, int]:
        return dict(self._token_to_id)

    @property
    def id_to_token(self) -> dict[int, str]:
        return dict(self._id_to_token)

    # ── encode / decode ──────────────────────────────────────────
    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a list of PnP tokens to a list of integer IDs."""
        unk = self._token_to_id["<unk>"]
        return [self._token_to_id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Convert a list of integer IDs back to PnP tokens."""
        unk_tok = "<unk>"
        return [self._id_to_token.get(i, unk_tok) for i in ids]
