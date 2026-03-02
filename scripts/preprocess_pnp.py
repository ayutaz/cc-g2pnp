"""Pre-compute PnP labels and store in LMDB for fast training."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

# scripts/ を sys.path に追加 (multiprocessing forkserver でのモジュールインポートに必要)
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

# ワーカープロセスのモジュールレベルキャッシュ (forkserver で各ワーカーに初期化される)
_vocab_cache = None


def _worker_init_with_cache() -> None:
    """ワーカープロセスの初期化: pyopenjtalk パッチ + PnPVocabulary キャッシュ。

    multiprocessing.Pool の initializer として使用する。
    """
    global _vocab_cache
    from cc_g2pnp._patch_pyopenjtalk import apply as _patch_pyopenjtalk

    _patch_pyopenjtalk()
    from cc_g2pnp.data.vocabulary import PnPVocabulary

    _vocab_cache = PnPVocabulary()


def _process_text_cached(text: str) -> tuple[str, list[int]] | None:
    """ワーカー関数: テキストから PnP ラベル ID を生成して (text, pnp_ids) を返す。

    PnP ラベルが空の場合は None を返す。
    """
    global _vocab_cache
    from cc_g2pnp.data.pnp_labeler import generate_pnp_labels

    pnp_tokens = generate_pnp_labels(text)
    if not pnp_tokens:
        return None
    pnp_ids = _vocab_cache.encode(pnp_tokens)
    return (text, pnp_ids)


def _text_stream(ds: object, cache: object, resume: bool) -> Iterator[str]:
    """データセットからテキストをストリームするジェネレータ。

    Args:
        ds: Hugging Face Dataset (streaming, select_columns 済み)
        cache: PnPLabelCache (resume=True の場合のキー存在チェックに使用)
        resume: True の場合、既にキャッシュ済みのテキストをスキップ
    """
    for sample in ds:
        text = sample.get("transcription", "")
        if not text or not text.strip():
            continue
        if resume and cache.get(text) is not None:
            continue
        yield text


def main() -> None:
    from cc_g2pnp._patch_pyopenjtalk import apply as _patch_pyopenjtalk

    _patch_pyopenjtalk()

    from datasets import load_dataset
    from tqdm import tqdm

    from cc_g2pnp.data.lmdb_cache import PnPLabelCache
    from cc_g2pnp.data.pnp_labeler import generate_pnp_labels
    from cc_g2pnp.data.vocabulary import PnPVocabulary

    parser = argparse.ArgumentParser(description="Build PnP label LMDB cache")
    parser.add_argument("--output", type=str, required=True, help="LMDB output directory")
    parser.add_argument("--subset", type=str, default="all", help="ReazonSpeech subset")
    parser.add_argument("--batch-size", type=int, default=1000, help="LMDB write batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--num-workers", type=int, default=0, help="並列ワーカー数 (0=シングルプロセス)")
    parser.add_argument("--mp-context", type=str, default="fork",
                        choices=["fork", "forkserver", "spawn"],
                        help="マルチプロセス起動方式 (CUDA不使用時は fork が最速)")
    parser.add_argument("--resume", action="store_true", help="既存キャッシュのエントリをスキップ")
    parser.add_argument("--local-dataset-dir", type=str, default=None,
                        help="ローカル Parquet ディレクトリ (download_text.py の出力)")
    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    if args.local_dataset_dir:
        import glob as glob_mod

        parquet_dir = Path(args.local_dataset_dir) / args.subset
        data_files = sorted(glob_mod.glob(str(parquet_dir / "*.parquet")))
        if not data_files:
            msg = f"No parquet files found in {parquet_dir}"
            raise FileNotFoundError(msg)
        logging.info("Loading local Parquet: %s (%d files)", parquet_dir, len(data_files))
        ds = load_dataset("parquet", data_files=data_files, split="train")
    else:
        ds = load_dataset(
            "reazon-research/reazonspeech",
            args.subset,
            split="train",
            streaming=True,
            trust_remote_code=True,  # セキュリティリスク: 信頼済みの公式データセットのみに使用すること
        ).select_columns(["transcription"])

    cache = PnPLabelCache(args.output, readonly=False)
    batch: list[tuple[str, list[int]]] = []
    total, cached = 0, 0
    start = time.time()

    text_iter = _text_stream(ds, cache, args.resume)

    with tqdm(desc="Caching PnP labels", unit="samples") as pbar:
        if args.num_workers > 0:
            import multiprocessing

            ctx = multiprocessing.get_context(args.mp_context)
            with ctx.Pool(processes=args.num_workers, initializer=_worker_init_with_cache) as pool:
                for result in pool.imap_unordered(_process_text_cached, text_iter, chunksize=128):
                    total += 1
                    pbar.update(1)
                    if result is None:
                        continue
                    batch.append(result)
                    cached += 1

                    if len(batch) >= args.batch_size:
                        cache.put_batch(batch)
                        batch = []
                        if cached % 10000 == 0:
                            elapsed = time.time() - start
                            rate = cached / elapsed
                            logging.info(f"Cached {cached}/{total} ({rate:.0f} samples/s)")

                    if args.max_samples and cached >= args.max_samples:
                        pool.terminate()
                        break
        else:
            vocab = PnPVocabulary()
            for text in text_iter:
                total += 1
                pbar.update(1)
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
