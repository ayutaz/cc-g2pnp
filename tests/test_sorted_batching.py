"""sorted_dynamic_batch_sampler のテスト。"""

from __future__ import annotations

import statistics

from cc_g2pnp.data.collator import dynamic_batch_sampler, sorted_dynamic_batch_sampler


def _make_sample(length: int) -> dict:
    """指定長の input_ids を持つダミーサンプルを生成する。"""
    return {"input_ids": list(range(length)), "labels": [0]}


def _collect_batches(samples: list[dict], max_tokens: int, buffer_size: int) -> list[list[dict]]:
    return list(sorted_dynamic_batch_sampler(samples, max_tokens, buffer_size))


class TestSortedDynamicBatchSamplerEmptyInput:
    """空入力のテスト。"""

    def test_empty_input_returns_empty(self):
        batches = _collect_batches([], max_tokens=100, buffer_size=10)
        assert batches == []

    def test_empty_input_buffer_size_zero(self):
        batches = _collect_batches([], max_tokens=100, buffer_size=0)
        assert batches == []


class TestSortedDynamicBatchSamplerBackwardCompat:
    """buffer_size=0 では既存の dynamic_batch_sampler と同じ動作になることを確認。"""

    def test_buffer_size_zero_same_as_dynamic_batch_sampler(self):
        import random

        random.seed(42)
        lengths = [random.randint(10, 100) for _ in range(200)]
        samples = [_make_sample(length) for length in lengths]

        expected = list(dynamic_batch_sampler(samples, max_tokens=500))
        actual = _collect_batches(samples, max_tokens=500, buffer_size=0)

        # バッチ数・各バッチのサンプル数・各バッチの input_ids が一致
        assert len(expected) == len(actual)
        for exp_batch, act_batch in zip(expected, actual, strict=True):
            assert len(exp_batch) == len(act_batch)
            for exp_s, act_s in zip(exp_batch, act_batch, strict=True):
                assert exp_s["input_ids"] == act_s["input_ids"]


class TestSortedDynamicBatchSamplerAllSamplesPreserved:
    """全サンプルが漏れなく出力されることを確認。"""

    def test_all_samples_preserved_small(self):
        samples = [_make_sample(length) for length in [10, 50, 30, 20, 40]]
        batches = _collect_batches(samples, max_tokens=200, buffer_size=3)
        all_output = [s for batch in batches for s in batch]
        # 入力と出力のサンプル数が同じ
        assert len(all_output) == len(samples)
        # input_ids の集合が一致
        input_lengths_in = sorted(len(s["input_ids"]) for s in samples)
        input_lengths_out = sorted(len(s["input_ids"]) for s in all_output)
        assert input_lengths_in == input_lengths_out

    def test_all_samples_preserved_buffer_larger_than_dataset(self):
        """buffer_size がデータ数より大きくても全件処理される。"""
        samples = [_make_sample(length) for length in range(1, 21)]
        batches = _collect_batches(samples, max_tokens=50, buffer_size=10_000)
        all_output = [s for batch in batches for s in batch]
        assert len(all_output) == len(samples)

    def test_all_samples_preserved_exact_buffer_multiple(self):
        """サンプル数が buffer_size の倍数でも全件処理される。"""
        samples = [_make_sample(length) for length in range(5, 25)]  # 20件
        batches = _collect_batches(samples, max_tokens=100, buffer_size=10)
        all_output = [s for batch in batches for s in batch]
        assert len(all_output) == len(samples)

    def test_all_samples_preserved_remainder(self):
        """buffer_size で割り切れない残りのサンプルも処理される。"""
        samples = [_make_sample(length) for length in range(5, 18)]  # 13件, buffer_size=10 で余り3件
        batches = _collect_batches(samples, max_tokens=200, buffer_size=10)
        all_output = [s for batch in batches for s in batch]
        assert len(all_output) == len(samples)


class TestSortedDynamicBatchSamplerPaddingReduction:
    """ソートによりバッチ内の長さ分散が小さくなることを確認。"""

    def _padding_waste_ratio(self, batches: list[list[dict]]) -> float:
        """全バッチのパディング無駄率を計算する(パディングトークン数 / 全トークン数)。"""
        total_slots = 0
        total_real = 0
        for batch in batches:
            max_len = max(len(s["input_ids"]) for s in batch)
            total_slots += max_len * len(batch)
            total_real += sum(len(s["input_ids"]) for s in batch)
        if total_slots == 0:
            return 0.0
        return (total_slots - total_real) / total_slots

    def test_sorted_reduces_padding_waste(self):
        """sorted_dynamic_batch_sampler がランダム順より少ないパディング無駄を生む。"""
        import random

        random.seed(123)
        # 短いサンプル (10-50) と長いサンプル (200-300) を混在させる
        lengths = [random.randint(10, 50) for _ in range(100)] + [
            random.randint(200, 300) for _ in range(100)
        ]
        random.shuffle(lengths)
        samples = [_make_sample(length) for length in lengths]

        max_tokens = 1000

        # ランダム順 (buffer_size=0)
        unsorted_batches = list(dynamic_batch_sampler(samples, max_tokens))
        unsorted_waste = self._padding_waste_ratio(unsorted_batches)

        # ソート済み
        sorted_batches = _collect_batches(samples, max_tokens, buffer_size=len(samples))
        sorted_waste = self._padding_waste_ratio(sorted_batches)

        # ソート済みの方がパディング無駄が少ない
        assert sorted_waste < unsorted_waste, (
            f"ソート済み waste={sorted_waste:.3f} がランダム waste={unsorted_waste:.3f} 以上"
        )

    def test_sorted_batch_length_variance_smaller(self):
        """ソート後のバッチ内長さ分散がランダム順より小さい。"""
        import random

        random.seed(456)
        lengths = [random.randint(10, 300) for _ in range(200)]
        samples = [_make_sample(length) for length in lengths]
        max_tokens = 800

        def avg_intra_batch_variance(batches: list[list[dict]]) -> float:
            variances = []
            for batch in batches:
                if len(batch) < 2:
                    continue
                lens = [len(s["input_ids"]) for s in batch]
                variances.append(statistics.variance(lens))
            return statistics.mean(variances) if variances else 0.0

        unsorted_batches = list(dynamic_batch_sampler(samples, max_tokens))
        sorted_batches = _collect_batches(samples, max_tokens, buffer_size=len(samples))

        unsorted_var = avg_intra_batch_variance(unsorted_batches)
        sorted_var = avg_intra_batch_variance(sorted_batches)

        assert sorted_var < unsorted_var, (
            f"ソート済み variance={sorted_var:.1f} がランダム variance={unsorted_var:.1f} 以上"
        )


class TestSortedDynamicBatchSamplerSingleSample:
    """単一サンプルのエッジケース。"""

    def test_single_sample(self):
        samples = [_make_sample(50)]
        batches = _collect_batches(samples, max_tokens=100, buffer_size=10)
        assert len(batches) == 1
        assert len(batches[0]) == 1
        assert len(batches[0][0]["input_ids"]) == 50

    def test_single_sample_exceeds_max_tokens(self):
        """max_tokens より大きい単一サンプルも yield される。"""
        samples = [_make_sample(500)]
        batches = _collect_batches(samples, max_tokens=100, buffer_size=10)
        assert len(batches) == 1
        assert len(batches[0]) == 1


class TestSortedDynamicBatchSamplerMaxTokensConstraint:
    """max_tokens 制約が守られることを確認。"""

    def test_no_batch_exceeds_max_tokens(self):
        import random

        random.seed(789)
        lengths = [random.randint(5, 150) for _ in range(300)]
        samples = [_make_sample(length) for length in lengths]
        max_tokens = 400

        batches = _collect_batches(samples, max_tokens, buffer_size=50)
        for batch in batches:
            total = sum(len(s["input_ids"]) for s in batch)
            # 単一サンプルが max_tokens を超える場合は許容
            if len(batch) > 1:
                assert total <= max_tokens, f"バッチのトークン合計 {total} が max_tokens {max_tokens} を超過"
