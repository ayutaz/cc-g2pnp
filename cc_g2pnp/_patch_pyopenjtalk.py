"""Monkey-patch pyopenjtalk.utils.sudachi_analyze to cache the sudachipy Dictionary.

pyopenjtalk-plus の ``sudachi_analyze()`` は呼び出しのたびに
``dictionary.Dictionary().create()`` を実行しており、これに ~24ms かかる。
sudachipy tokenizer を ``threading.local()`` でキャッシュすることで
0.03ms に短縮する (約 800x 高速化)。

このパッチは ``uv sync`` で .venv を再作成しても有効。
アプリケーション起動時に ``apply()`` を呼ぶこと。
"""

from __future__ import annotations

import threading

_sudachi_local = threading.local()
_patched = False


def apply() -> None:
    """pyopenjtalk.utils.sudachi_analyze をキャッシュ版で置き換える。

    冪等 — 複数回呼んでも安全。
    """
    global _patched
    if _patched:
        return

    import pyopenjtalk.utils as _utils
    from sudachipy import dictionary
    from sudachipy import tokenizer as sudachi_tokenizer

    def _get_sudachi_tokenizer():
        if not hasattr(_sudachi_local, "tokenizer"):
            _sudachi_local.tokenizer = dictionary.Dictionary().create()
        return _sudachi_local.tokenizer

    def sudachi_analyze_cached(text: str, multi_read_kanji_list: list[str]) -> list[list[str]]:
        text = text.replace("ー", "")
        tokenizer_obj = _get_sudachi_tokenizer()
        mode = sudachi_tokenizer.Tokenizer.SplitMode.C
        m_list = tokenizer_obj.tokenize(text, mode)
        return [[m.surface(), m.reading_form()] for m in m_list if m.surface() in multi_read_kanji_list]

    _utils.sudachi_analyze = sudachi_analyze_cached
    _patched = True
