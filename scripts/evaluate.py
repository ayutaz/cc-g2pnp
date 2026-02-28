"""CC-G2PnP モデルの推論デモ + 評価パイプライン実行スクリプト。"""

from __future__ import annotations

import argparse

import torch

from cc_g2pnp._patch_pyopenjtalk import apply as _patch_pyopenjtalk
from cc_g2pnp.data.tokenizer import G2PnPTokenizer
from cc_g2pnp.evaluation.eval_data import EvalDataGenerator
from cc_g2pnp.evaluation.pipeline import EvalConfig, EvaluationPipeline


def demo_inference(pipeline: EvaluationPipeline, tokenizer: G2PnPTokenizer) -> None:
    """推論デモ: 日本語テキストから PnP ラベルを生成。"""
    demo_texts = [
        "東京は日本の首都です。",
        "今日はいい天気ですね。",
        "音声合成の研究は近年急速に進んでいる。",
        "ありがとうございます。",
        "自然言語処理は人工知能の重要な分野の一つである。",
        "桜の花が美しく咲いている。",
        "明日の天気はどうなるでしょうか。",
        "機械学習モデルの性能を評価する。",
    ]

    print("=" * 70)
    print("推論デモ: テキスト → PnP ラベル")
    print("=" * 70)

    vocab = pipeline.vocabulary
    model = pipeline.model
    device = pipeline._device

    for text in demo_texts:
        bpe_ids = tokenizer.encode(text)
        input_ids = torch.tensor([bpe_ids], dtype=torch.long, device=device)
        input_lengths = torch.tensor([len(bpe_ids)], dtype=torch.long, device=device)

        with torch.no_grad():
            predicted_ids = model.inference(input_ids, input_lengths)

        # デコード
        blank_id = vocab.blank_id
        pad_id = vocab.pad_id
        tokens = vocab.decode(
            [i for i in predicted_ids[0] if i != blank_id and i != pad_id]
        )
        tokens = [t for t in tokens if t not in ("<blank>", "<pad>")]

        pnp_str = " ".join(tokens)
        print(f"\n入力: {text}")
        print(f"PnP:  {pnp_str}")

    print()


def run_evaluation(pipeline: EvaluationPipeline, tokenizer: G2PnPTokenizer) -> None:
    """ビルトインデータセットで 6 種メトリクスを評価。"""
    print("=" * 70)
    print("評価パイプライン実行")
    print("=" * 70)

    # ビルトインデータセットを生成
    generator = EvalDataGenerator(tokenizer)
    dataset = generator.builtin_dataset()
    print(f"評価サンプル数: {len(dataset)} ({', '.join(dataset.domains)})")

    # 評価実行
    result = pipeline.evaluate(dataset)

    # 結果表示
    print(pipeline.format_results(result))


def main() -> None:
    _patch_pyopenjtalk()

    parser = argparse.ArgumentParser(description="CC-G2PnP 推論・評価")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/step_00100000.pt",
        help="チェックポイントファイルパス",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="推論デバイス",
    )
    parser.add_argument("--demo-only", action="store_true", help="推論デモのみ実行")
    parser.add_argument("--eval-only", action="store_true", help="評価のみ実行")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"チェックポイント: {args.checkpoint}")
    print(f"デバイス: {device}")
    print()

    # モデルロード
    eval_config = EvalConfig(batch_size=16, device=device)
    pipeline = EvaluationPipeline.from_checkpoint(args.checkpoint, eval_config)
    tokenizer = G2PnPTokenizer.get_instance()

    print(f"モデルロード完了 (パラメータ数: {sum(p.numel() for p in pipeline.model.parameters()):,})")
    print()

    if not args.eval_only:
        demo_inference(pipeline, tokenizer)

    if not args.demo_only:
        run_evaluation(pipeline, tokenizer)


if __name__ == "__main__":
    main()
