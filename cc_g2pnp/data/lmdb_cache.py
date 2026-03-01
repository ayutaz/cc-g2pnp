"""LMDB-based cache for pre-computed PnP labels."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import lmdb

if TYPE_CHECKING:
    from pathlib import Path

# LMDB のデフォルト最大キー長。日本語 UTF-8 は 1 文字 3 バイトのため
# 171 文字超のテキストではこの上限を超える。
_MAX_KEY_BYTES = 511


class PnPLabelCache:
    """Read/write cache for PnP label IDs, backed by LMDB."""

    def __init__(
        self,
        db_path: str | Path,
        readonly: bool = True,
        map_size: int = 10 * 1024**3,
    ) -> None:
        """Open LMDB environment.

        Args:
            db_path: Path to LMDB directory.
            readonly: Open in read-only mode (for training).
            map_size: Maximum DB size in bytes (default 10 GB).
        """
        self.env = lmdb.open(str(db_path), readonly=readonly, lock=False, map_size=map_size)

    def _make_key(self, text: str) -> bytes:
        """UTF-8 エンコードしたキーを返す。511 バイト超の場合は SHA-256 ハッシュを使用。"""
        key = text.encode("utf-8")
        if len(key) > _MAX_KEY_BYTES:
            key = hashlib.sha256(key).hexdigest().encode("ascii")
        return key

    def get(self, text: str) -> list[int] | None:
        """Look up PnP label IDs for a text string. Returns None on cache miss."""
        with self.env.begin() as txn:
            value = txn.get(self._make_key(text))
            if value is None:
                return None
            return json.loads(value)

    def put(self, text: str, pnp_ids: list[int]) -> None:
        """Store PnP label IDs for a text string."""
        with self.env.begin(write=True) as txn:
            txn.put(self._make_key(text), json.dumps(pnp_ids).encode("utf-8"))

    def put_batch(self, items: list[tuple[str, list[int]]]) -> None:
        """Store multiple items in a single transaction."""
        with self.env.begin(write=True) as txn:
            for text, pnp_ids in items:
                txn.put(self._make_key(text), json.dumps(pnp_ids).encode("utf-8"))

    def __len__(self) -> int:
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> PnPLabelCache:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
