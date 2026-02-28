"""LMDB-based cache for pre-computed PnP labels."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import lmdb

if TYPE_CHECKING:
    from pathlib import Path


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

    def get(self, text: str) -> list[int] | None:
        """Look up PnP label IDs for a text string. Returns None on cache miss."""
        with self.env.begin() as txn:
            value = txn.get(text.encode("utf-8"))
            if value is None:
                return None
            return json.loads(value)

    def put(self, text: str, pnp_ids: list[int]) -> None:
        """Store PnP label IDs for a text string."""
        with self.env.begin(write=True) as txn:
            txn.put(text.encode("utf-8"), json.dumps(pnp_ids).encode("utf-8"))

    def put_batch(self, items: list[tuple[str, list[int]]]) -> None:
        """Store multiple items in a single transaction."""
        with self.env.begin(write=True) as txn:
            for text, pnp_ids in items:
                txn.put(text.encode("utf-8"), json.dumps(pnp_ids).encode("utf-8"))

    def __len__(self) -> int:
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> PnPLabelCache:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
