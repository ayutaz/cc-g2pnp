"""Pre-compute PnP labels and store in LMDB for fast training."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path


def main() -> None:
    from cc_g2pnp._patch_pyopenjtalk import apply as _patch_pyopenjtalk

    _patch_pyopenjtalk()

    from datasets import load_dataset

    from cc_g2pnp.data.lmdb_cache import PnPLabelCache
    from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
    from cc_g2pnp.data.vocabulary import PnPVocabulary

    parser = argparse.ArgumentParser(description="Build PnP label LMDB cache")
    parser.add_argument("--output", type=str, required=True, help="LMDB output directory")
    parser.add_argument("--subset", type=str, default="all", help="ReazonSpeech subset")
    parser.add_argument("--batch-size", type=int, default=1000, help="LMDB write batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    args = parser.parse_args()

    vocab = PnPVocabulary()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "reazon-research/reazonspeech",
        args.subset,
        split="train",
        streaming=True,
        trust_remote_code=True,
    ).select_columns(["transcription"])

    cache = PnPLabelCache(args.output, readonly=False)
    batch: list[tuple[str, list[int]]] = []
    total, cached = 0, 0
    start = time.time()

    for sample in ds:
        text = sample.get("transcription", "")
        if not text or not text.strip():
            continue
        total += 1
        pnp_tokens = generate_pnp_labels(text)
        if not pnp_tokens:
            continue
        pnp_ids = vocab.encode(pnp_tokens)
        batch.append((text, pnp_ids))
        cached += 1

        if len(batch) >= args.batch_size:
            cache.put_batch(batch)
            batch = []
            if cached % 10000 == 0:
                elapsed = time.time() - start
                rate = cached / elapsed
                logging.info(f"Cached {cached}/{total} ({rate:.0f} samples/s)")

        if args.max_samples and cached >= args.max_samples:
            break

    if batch:
        cache.put_batch(batch)
    cache.close()
    print(f"Done: {cached}/{total} samples cached to {args.output}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
