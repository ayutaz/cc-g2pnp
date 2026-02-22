"""Tests for DynamicBatchCollator."""

import torch

from cc_g2pnp.data.collator import DynamicBatchCollator


def test_empty_batch():
    """空バッチが渡された場合に空dictを返すことを検証."""
    collator = DynamicBatchCollator()
    result = collator([])
    assert result == {}


def test_pad_token_id_default():
    """pad_token_id のデフォルトが 1 (CALM2) であることを検証."""
    collator = DynamicBatchCollator()
    assert collator.pad_token_id == 1


def test_attention_mask_present():
    """バッチ出力に attention_mask キーが含まれることを検証."""
    batch = [
        {"input_ids": [10, 20, 30], "labels": [100, 200]},
        {"input_ids": [40, 50], "labels": [300, 400, 500]},
    ]
    collator = DynamicBatchCollator()
    result = collator(batch)
    assert "attention_mask" in result


def test_attention_mask_shape():
    """attention_mask の shape が input_ids と同じであることを検証."""
    batch = [
        {"input_ids": [10, 20, 30], "labels": [100, 200]},
        {"input_ids": [40, 50], "labels": [300, 400, 500]},
    ]
    collator = DynamicBatchCollator()
    result = collator(batch)
    assert result["attention_mask"].shape == result["input_ids"].shape


def test_attention_mask_values():
    """パディング部分が 0、非パディング部分が 1 であることを検証."""
    batch = [
        {"input_ids": [10, 20, 30], "labels": [100, 200]},
        {"input_ids": [40, 50], "labels": [300]},
    ]
    collator = DynamicBatchCollator(pad_token_id=1)
    result = collator(batch)

    # 1st sample: length 3, no padding -> all 1s
    assert result["attention_mask"][0].tolist() == [1, 1, 1]
    # 2nd sample: length 2, padded to 3 -> [1, 1, 0]
    assert result["attention_mask"][1].tolist() == [1, 1, 0]


def test_attention_mask_dtype():
    """attention_mask の dtype が torch.long であることを検証."""
    batch = [
        {"input_ids": [10, 20], "labels": [100]},
    ]
    collator = DynamicBatchCollator()
    result = collator(batch)
    assert result["attention_mask"].dtype == torch.long
