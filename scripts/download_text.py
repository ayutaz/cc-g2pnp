"""Download ReazonSpeech text to local Parquet files for offline training.

Downloads TSV files directly from corpus.reazon-research.org (text only,
no audio), which is ~560x faster than streaming through HuggingFace datasets.

Usage:
    uv run python scripts/download_text.py --subset small --output /data/reazonspeech_text
    uv run python scripts/download_text.py --subset all --output /data/reazonspeech_text
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_TSV_BASE_URL = "https://corpus.reazon-research.org/reazonspeech-v2/tsv"
_VALID_SUBSETS = ("small", "medium", "large", "all")


def _flush_shard(
    output_dir: Path,
    shard_idx: int,
    buffer_names: list[str],
    buffer_texts: list[str],
) -> None:
    """Write a buffer to a Parquet shard file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("name", pa.string()), ("transcription", pa.string())])
    shard_path = output_dir / f"shard_{shard_idx:06d}.parquet"
    table = pa.table({"name": buffer_names, "transcription": buffer_texts}, schema=schema)
    pq.write_table(table, shard_path)
    logger.info("Wrote %s (%d rows)", shard_path, len(buffer_names))


def main() -> None:
    import requests
    from tqdm import tqdm

    parser = argparse.ArgumentParser(
        description="Download ReazonSpeech text to local Parquet files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--subset", type=str, required=True, choices=_VALID_SUBSETS,
        help="ReazonSpeech subset",
    )
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--shard-size", type=int, default=500_000, help="Rows per Parquet shard")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to download (debug)")
    args = parser.parse_args()

    output_dir = Path(args.output) / args.subset
    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_url = f"{_TSV_BASE_URL}/{args.subset}.tsv"
    logger.info("Downloading TSV from %s", tsv_url)

    resp = requests.get(tsv_url, stream=True, timeout=60)
    resp.raise_for_status()
    total_bytes = int(resp.headers.get("content-length", 0))

    shard_idx = 0
    total_rows = 0
    buffer_names: list[str] = []
    buffer_texts: list[str] = []
    start = time.time()

    with tqdm(total=total_bytes, desc=f"Downloading {args.subset}.tsv", unit="B", unit_scale=True) as pbar:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue

            name, text = parts
            buffer_names.append(name)
            buffer_texts.append(text)
            total_rows += 1
            pbar.update(len(raw_line) + 1)  # +1 for newline

            if len(buffer_names) >= args.shard_size:
                _flush_shard(output_dir, shard_idx, buffer_names, buffer_texts)
                shard_idx += 1
                buffer_names = []
                buffer_texts = []

            if args.max_samples and total_rows >= args.max_samples:
                break

    # Write remaining buffer
    if buffer_names:
        _flush_shard(output_dir, shard_idx, buffer_names, buffer_texts)
        shard_idx += 1

    elapsed = time.time() - start

    # Write metadata
    metadata = {
        "subset": args.subset,
        "total_rows": total_rows,
        "num_shards": shard_idx,
        "shard_size": args.shard_size,
        "elapsed_seconds": round(elapsed, 1),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info("Metadata written to %s", metadata_path)

    rate = total_rows / max(elapsed, 0.01)
    print(f"Done: {total_rows} rows in {shard_idx} shards -> {output_dir} ({elapsed:.1f}s, {rate:,.0f} samples/s)")


if __name__ == "__main__":
    main()
