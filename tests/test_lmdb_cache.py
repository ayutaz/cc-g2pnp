"""Tests for LMDB-based PnP label cache."""

from __future__ import annotations

import lmdb
import pytest

from cc_g2pnp.data.lmdb_cache import PnPLabelCache


def test_put_and_get(tmp_path):
    """write してから read で同じ値が返る。"""
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        cache.put("こんにちは", [1, 2, 3])
        result = cache.get("こんにちは")
    assert result == [1, 2, 3]


def test_get_missing(tmp_path):
    """存在しないキーは None を返す。"""
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        result = cache.get("存在しないキー")
    assert result is None


def test_put_batch(tmp_path):
    """バッチ書き込みが正しく動作する。"""
    items = [
        ("テスト一", [10, 20, 30]),
        ("テスト二", [40, 50, 60]),
        ("テスト三", [70, 80, 90]),
    ]
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        cache.put_batch(items)
        for text, pnp_ids in items:
            assert cache.get(text) == pnp_ids


def test_len(tmp_path):
    """__len__ がエントリ数を正しく返す。"""
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        assert len(cache) == 0
        cache.put("キー一", [1])
        assert len(cache) == 1
        cache.put("キー二", [2, 3])
        assert len(cache) == 2


def test_context_manager(tmp_path):
    """with 文でキャッシュが正常に使用できる。"""
    db_path = tmp_path / "cache"
    with PnPLabelCache(db_path, readonly=False) as cache:
        cache.put("テスト", [5, 6, 7])

    # 再オープンして読み取り確認
    with PnPLabelCache(db_path, readonly=True) as cache:
        assert cache.get("テスト") == [5, 6, 7]


def test_unicode_keys(tmp_path):
    """日本語テキストをキーとして正しく扱える。"""
    texts = [
        "今日はいい天気ですね",
        "東京スカイツリー",
        "ラーメン・餃子・チャーハン",
        "アイウエオカキクケコ",
    ]
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        for i, text in enumerate(texts):
            cache.put(text, [i, i + 1])
        for i, text in enumerate(texts):
            assert cache.get(text) == [i, i + 1]


def test_readonly_mode(tmp_path):
    """読み取り専用モードでは読み取りが可能、書き込みは例外が発生する。"""
    db_path = tmp_path / "cache"

    # 書き込みモードでデータ投入
    with PnPLabelCache(db_path, readonly=False) as cache:
        cache.put("元データ", [100, 200])

    # 読み取り専用モードで読み取り確認
    with PnPLabelCache(db_path, readonly=True) as cache:
        assert cache.get("元データ") == [100, 200]

    # 読み取り専用モードで書き込みは例外
    with PnPLabelCache(db_path, readonly=True) as cache, pytest.raises(lmdb.ReadonlyError):
        cache.put("新データ", [300])
