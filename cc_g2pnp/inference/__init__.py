"""Inference module for CC-G2PnP streaming and non-streaming inference."""

from __future__ import annotations

from cc_g2pnp.inference.latency import (
    ChunkLatencyResult,
    StartLatencyResult,
    compute_tokens_before_start,
    measure_chunk_latency,
    measure_start_latency,
)
from cc_g2pnp.inference.streaming import StreamingInference, StreamingState

__all__ = [
    "ChunkLatencyResult",
    "StartLatencyResult",
    "StreamingInference",
    "StreamingState",
    "compute_tokens_before_start",
    "measure_chunk_latency",
    "measure_start_latency",
]
