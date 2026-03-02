"""Tests for LMDB-based PnP label cache."""

from __future__ import annotations

import json

import lmdb
import pytest

from cc_g2pnp.data.lmdb_cache import _FORMAT_VERSION_BYTES, PnPLabelCache


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


# ---------------------------------------------------------------------------
# bytes 形式 (新フォーマット) テスト
# ---------------------------------------------------------------------------


def test_binary_format_roundtrip(tmp_path):
    """put → get で bytes 形式の roundtrip が正しく動作する。"""
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        cache.put("テスト", [1, 50, 139])
        result = cache.get("テスト")
    assert result == [1, 50, 139]


def test_binary_format_batch_roundtrip(tmp_path):
    """put_batch → get で bytes 形式の roundtrip が正しく動作する。"""
    items = [
        ("一", [0, 1, 2]),
        ("二", [100, 101, 102]),
        ("三", [130, 135, 139]),
    ]
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        cache.put_batch(items)
        for text, pnp_ids in items:
            assert cache.get(text) == pnp_ids


def test_backward_compat_json_read(tmp_path):
    """レガシー JSON 形式で直接書き込んだデータを get() で正常に読み取れる。"""
    db_path = tmp_path / "cache"
    # JSON 形式で直接書き込み
    env = lmdb.open(str(db_path), readonly=False, lock=False)
    with env.begin(write=True) as txn:
        txn.put("レガシー".encode(), json.dumps([10, 20, 30]).encode("utf-8"))
    env.close()

    with PnPLabelCache(db_path, readonly=True) as cache:
        result = cache.get("レガシー")
    assert result == [10, 20, 30]


def test_mixed_format_read(tmp_path):
    """JSON + bytes 混在 DB から両形式を正常に読み取れる。"""
    db_path = tmp_path / "cache"
    # JSON 形式で直接書き込み
    env = lmdb.open(str(db_path), readonly=False, lock=False)
    with env.begin(write=True) as txn:
        txn.put("旧形式".encode(), json.dumps([5, 6, 7]).encode("utf-8"))
    env.close()

    # 新形式で書き込み
    with PnPLabelCache(db_path, readonly=False) as cache:
        cache.put("新形式", [8, 9, 10])
        assert cache.get("旧形式") == [5, 6, 7]
        assert cache.get("新形式") == [8, 9, 10]


def test_binary_format_empty_list(tmp_path):
    """空リスト [] の roundtrip が正しく動作する。"""
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        cache.put("空", [])
        result = cache.get("空")
    assert result == []


def test_binary_format_max_id(tmp_path):
    """境界値 [0, 139] の roundtrip が正しく動作する。"""
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        cache.put("境界値", [0, 139])
        result = cache.get("境界値")
    assert result == [0, 139]


def test_binary_format_all_ids(tmp_path):
    """list(range(140)) の roundtrip が正しく動作する。"""
    pnp_ids = list(range(140))
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        cache.put("全ID", pnp_ids)
        result = cache.get("全ID")
    assert result == pnp_ids


def test_binary_format_db_size_smaller(tmp_path):
    """bytes 形式の DB ファイルサイズが JSON 形式より小さい (または同等以下)。"""
    pnp_ids = list(range(140))

    # bytes 形式
    bin_path = tmp_path / "bin_cache"
    with PnPLabelCache(bin_path, readonly=False) as cache:
        for i in range(50):
            cache.put(f"テキスト{i}", pnp_ids)

    # JSON 形式 (直接書き込み)
    json_path = tmp_path / "json_cache"
    env = lmdb.open(str(json_path), readonly=False, lock=False)
    with env.begin(write=True) as txn:
        for i in range(50):
            txn.put(f"テキスト{i}".encode(), json.dumps(pnp_ids).encode("utf-8"))
    env.close()

    bin_size = sum(f.stat().st_size for f in bin_path.iterdir())
    json_size = sum(f.stat().st_size for f in json_path.iterdir())
    assert bin_size <= json_size


def test_version_prefix_present(tmp_path):
    """LMDB の生データに _FORMAT_VERSION_BYTES プレフィクスが存在することを確認。"""
    db_path = tmp_path / "cache"
    with PnPLabelCache(db_path, readonly=False) as cache:
        cache.put("確認", [1, 2, 3])

    env = lmdb.open(str(db_path), readonly=True, lock=False)
    with env.begin() as txn:
        raw = txn.get("確認".encode())
    env.close()

    assert raw is not None
    assert raw[:1] == _FORMAT_VERSION_BYTES
    assert list(raw[1:]) == [1, 2, 3]


def test_binary_format_long_key_hash(tmp_path):
    """512 バイト超の長キーと bytes 形式の組合せテスト。"""
    long_text = "あ" * 200  # UTF-8 で 600 バイト → SHA-256 ハッシュキーを使用
    pnp_ids = [10, 20, 30, 40, 50]
    with PnPLabelCache(tmp_path / "cache", readonly=False) as cache:
        cache.put(long_text, pnp_ids)
        result = cache.get(long_text)
    assert result == pnp_ids
