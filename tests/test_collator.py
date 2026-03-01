"""Tests for DynamicBatchCollator."""

import pytest
import torch

from cc_g2pnp.data.collator import DynamicBatchCollator


@pytest.fixture()
def collator():
    """デフォルト設定の DynamicBatchCollator。"""
    return DynamicBatchCollator(pad_token_id=1, label_pad_id=-100)


def _make_sample(input_ids: list[int], labels: list[int]) -> dict:
    """テスト用サンプル dict を生成する。"""
    return {"input_ids": input_ids, "labels": labels}


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


class TestCollatorPadding:
    """input_ids のパディング動作テスト。"""

    def test_collator_padding(self, collator: DynamicBatchCollator) -> None:
        """異なる長さのサンプルが最大長にパディングされるか。"""
        batch = [
            _make_sample([2, 3, 4], [5, 6]),
            _make_sample([7, 8, 9, 10], [11, 12, 13]),
        ]
        result = collator(batch)

        # input_ids は max_len=4 にパディングされる
        assert result["input_ids"].shape == (2, 4)
        # 短いサンプルの末尾は pad_token_id=1 で埋められる
        assert result["input_ids"][0, 3].item() == 1
        # 有効な値は変わらない
        assert result["input_ids"][0, :3].tolist() == [2, 3, 4]
        assert result["input_ids"][1, :4].tolist() == [7, 8, 9, 10]

    def test_collator_attention_mask(self, collator: DynamicBatchCollator) -> None:
        """attention_mask が pad_token_id=1 に基づいて正しく生成されるか。"""
        batch = [
            _make_sample([2, 3, 4], [5, 6]),
            _make_sample([7, 8, 9, 10], [11, 12, 13]),
        ]
        result = collator(batch)

        # サンプル0: 長さ3 → 最後の1フレームはパディング
        assert result["attention_mask"][0].tolist() == [1, 1, 1, 0]
        # サンプル1: 長さ4 → 全フレームが有効
        assert result["attention_mask"][1].tolist() == [1, 1, 1, 1]

    def test_collator_input_lengths(self, collator: DynamicBatchCollator) -> None:
        """input_lengths が元の長さを正しく保持するか。"""
        batch = [
            _make_sample([2, 3, 4], [5, 6]),
            _make_sample([7, 8, 9, 10], [11, 12, 13]),
        ]
        result = collator(batch)

        # パディング後の長さではなく元の長さを保持する
        assert result["input_lengths"].tolist() == [3, 4]
        assert result["input_lengths"].dtype == torch.long

    def test_collator_label_lengths(self, collator: DynamicBatchCollator) -> None:
        """label_lengths が正しいか。"""
        batch = [
            _make_sample([2, 3, 4], [5, 6]),
            _make_sample([7, 8, 9, 10], [11, 12, 13]),
        ]
        result = collator(batch)

        assert result["label_lengths"].tolist() == [2, 3]
        assert result["label_lengths"].dtype == torch.long

    def test_collator_single_sample(self, collator: DynamicBatchCollator) -> None:
        """サンプル1個の場合、パディングなしで全フィールドが正しいか。"""
        batch = [_make_sample([2, 3, 4, 5], [6, 7])]
        result = collator(batch)

        assert result["input_ids"].shape == (1, 4)
        assert result["labels"].shape == (1, 2)
        assert result["input_lengths"].tolist() == [4]
        assert result["label_lengths"].tolist() == [2]
        # パディングなし → attention_mask 全て 1
        assert result["attention_mask"][0].tolist() == [1, 1, 1, 1]

    def test_collator_empty_batch(self, collator: DynamicBatchCollator) -> None:
        """空バッチの場合、空 dict が返るか。"""
        result = collator([])
        assert result == {}
