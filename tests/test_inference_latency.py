"""Latency measurement utilities tests."""

from __future__ import annotations

import math

import pytest
import torch

from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.model.config import CC_G2PnPConfig


@pytest.fixture
def small_config() -> CC_G2PnPConfig:
    return CC_G2PnPConfig(
        d_model=64,
        num_heads=2,
        d_ff=128,
        num_layers=2,
        intermediate_ctc_layers=(1,),
    )


@pytest.fixture
def small_model(small_config) -> CC_G2PnP:
    model = CC_G2PnP(small_config)
    model.eval()
    return model


class TestComputeTokensBeforeStart:
    """compute_tokens_before_start のテスト。"""

    def test_default_config(self, small_config):
        from cc_g2pnp.inference.latency import compute_tokens_before_start

        # chunk_size=5, mla_size=1, upsample_factor=8
        # frames_needed = 5 + 1 = 6
        # tokens_needed = ceil(6/8) = 1
        assert compute_tokens_before_start(small_config) == 1

    def test_large_chunk_size(self):
        from cc_g2pnp.inference.latency import compute_tokens_before_start

        config = CC_G2PnPConfig(
            d_model=64,
            num_heads=2,
            d_ff=128,
            num_layers=2,
            intermediate_ctc_layers=(1,),
            chunk_size=10,
            mla_size=2,
            upsample_factor=8,
        )
        # frames_needed = 10 + 2 = 12, tokens = ceil(12/8) = 2
        assert compute_tokens_before_start(config) == 2

    def test_exact_multiple(self):
        from cc_g2pnp.inference.latency import compute_tokens_before_start

        config = CC_G2PnPConfig(
            d_model=64,
            num_heads=2,
            d_ff=128,
            num_layers=2,
            intermediate_ctc_layers=(1,),
            chunk_size=7,
            mla_size=1,
            upsample_factor=8,
        )
        # frames_needed = 8, tokens = ceil(8/8) = 1
        assert compute_tokens_before_start(config) == 1

    def test_no_mla(self):
        from cc_g2pnp.inference.latency import compute_tokens_before_start

        config = CC_G2PnPConfig(
            d_model=64,
            num_heads=2,
            d_ff=128,
            num_layers=2,
            intermediate_ctc_layers=(1,),
            chunk_size=5,
            mla_size=0,
            upsample_factor=8,
        )
        # frames_needed = 5, tokens = ceil(5/8) = 1
        assert compute_tokens_before_start(config) == 1


class TestMeasureStartLatency:
    """measure_start_latency のテスト。"""

    def test_returns_result(self, small_model):
        from cc_g2pnp.inference.latency import measure_start_latency

        input_ids = torch.randint(0, 65000, (1, 5))
        result = measure_start_latency(
            small_model,
            input_ids,
            num_runs=2,
            warmup_runs=1,
        )
        assert result.tokens_before_start >= 1
        assert (
            result.frames_before_start
            == small_model.config.chunk_size + small_model.config.mla_size
        )
        assert result.processing_time_sec > 0
        assert result.wait_time_sec >= 0
        assert result.start_latency_sec == pytest.approx(
            result.wait_time_sec + result.processing_time_sec,
            abs=1e-6,
        )

    def test_custom_token_interval(self, small_model):
        from cc_g2pnp.inference.latency import measure_start_latency

        input_ids = torch.randint(0, 65000, (1, 5))
        result = measure_start_latency(
            small_model,
            input_ids,
            token_interval=0.1,
            num_runs=2,
            warmup_runs=1,
        )
        expected_wait = result.tokens_before_start * 0.1
        assert result.wait_time_sec == pytest.approx(expected_wait, abs=1e-9)


class TestMeasureChunkLatency:
    """measure_chunk_latency のテスト。"""

    def test_returns_result(self, small_model):
        from cc_g2pnp.inference.latency import measure_chunk_latency

        input_ids = torch.randint(0, 65000, (1, 10))
        input_lengths = torch.tensor([10])
        result = measure_chunk_latency(
            small_model,
            input_ids,
            input_lengths,
            num_runs=2,
            warmup_runs=1,
        )
        assert result.mean_chunk_time_sec > 0
        assert result.std_chunk_time_sec >= 0
        assert result.total_chunks > 0

    def test_chunk_count_calculation(self, small_model):
        from cc_g2pnp.inference.latency import measure_chunk_latency

        config = small_model.config
        seq_len = 5
        input_ids = torch.randint(0, 65000, (1, seq_len))
        input_lengths = torch.tensor([seq_len])
        result = measure_chunk_latency(
            small_model,
            input_ids,
            input_lengths,
            num_runs=2,
            warmup_runs=1,
        )
        expected_chunks = math.ceil(
            seq_len * config.upsample_factor / config.chunk_size
        )
        assert result.total_chunks == expected_chunks
