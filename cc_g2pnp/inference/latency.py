"""Latency measurement utilities for CC-G2PnP streaming inference."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from cc_g2pnp.model.cc_g2pnp import CC_G2PnP


@dataclass
class StartLatencyResult:
    """Result of start latency measurement."""

    start_latency_sec: float
    """Total start latency in seconds: wait_time + processing_time."""

    wait_time_sec: float
    """Token arrival wait time: tokens_before_start * token_interval."""

    processing_time_sec: float
    """G2PnP processing time for the first chunk (mean over runs)."""

    tokens_before_start: int
    """Number of BPE tokens needed before first output."""

    frames_before_start: int
    """Number of upsampled frames needed before first output (chunk_size + mla_size)."""


@dataclass
class ChunkLatencyResult:
    """Result of per-chunk latency measurement."""

    mean_chunk_time_sec: float
    """Mean processing time per chunk."""

    std_chunk_time_sec: float
    """Standard deviation of chunk processing times."""

    total_chunks: int
    """Total number of chunks processed."""


def compute_tokens_before_start(config) -> int:
    """Calculate how many BPE tokens must arrive before streaming can start.

    Args:
        config: CC_G2PnPConfig instance.

    Returns:
        Number of BPE tokens needed.
    """
    frames_needed = config.chunk_size + config.mla_size
    return math.ceil(frames_needed / config.upsample_factor)


def measure_start_latency(
    model: CC_G2PnP,
    input_ids: torch.Tensor,
    *,
    token_interval: float = 0.05,
    num_runs: int = 10,
    warmup_runs: int = 2,
) -> StartLatencyResult:
    """Measure the start latency for streaming inference.

    Start latency = (tokens_before_start * token_interval) + processing_time

    Args:
        model: Trained CC_G2PnP model (will be set to eval mode).
        input_ids: BPE token IDs [1, T] with enough tokens for at least one chunk.
        token_interval: LLM token generation interval tau (seconds). Default 0.05s.
        num_runs: Number of measurement runs for averaging. Default 10.
        warmup_runs: Warmup runs before measurement. Default 2.

    Returns:
        StartLatencyResult with timing details.
    """
    model.eval()
    config = model.config
    device = next(model.parameters()).device

    tokens_needed = compute_tokens_before_start(config)
    frames_needed = config.chunk_size + config.mla_size
    wait_time = tokens_needed * token_interval

    # Prepare input: take just enough tokens for the first chunk
    first_tokens = input_ids[:, :tokens_needed].to(device)

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(first_tokens, torch.tensor([tokens_needed], device=device))

    # Measure
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(first_tokens, torch.tensor([tokens_needed], device=device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    processing_time = sum(times) / len(times)

    return StartLatencyResult(
        start_latency_sec=wait_time + processing_time,
        wait_time_sec=wait_time,
        processing_time_sec=processing_time,
        tokens_before_start=tokens_needed,
        frames_before_start=frames_needed,
    )


def measure_chunk_latency(
    model: CC_G2PnP,
    input_ids: torch.Tensor,
    input_lengths: torch.Tensor,
    *,
    num_runs: int = 10,
    warmup_runs: int = 2,
) -> ChunkLatencyResult:
    """Measure per-chunk processing latency.

    Runs full inference and divides total time by number of chunks.

    Args:
        model: Trained CC_G2PnP model.
        input_ids: BPE token IDs [B, T].
        input_lengths: Valid lengths [B].
        num_runs: Number of measurement runs.
        warmup_runs: Warmup runs before measurement.

    Returns:
        ChunkLatencyResult with timing statistics.
    """
    model.eval()
    config = model.config
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    input_lengths = input_lengths.to(device)

    # Calculate number of chunks
    max_len = int(input_lengths.max().item())
    total_frames = max_len * config.upsample_factor
    total_chunks = math.ceil(total_frames / config.chunk_size)

    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_ids, input_lengths)

    # Measure
    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids, input_lengths)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_total = sum(times) / len(times)
    mean_per_chunk = mean_total / total_chunks if total_chunks > 0 else 0.0

    variance = sum((t / total_chunks - mean_per_chunk) ** 2 for t in times) / len(times)
    std_per_chunk = math.sqrt(variance)

    return ChunkLatencyResult(
        mean_chunk_time_sec=mean_per_chunk,
        std_chunk_time_sec=std_per_chunk,
        total_chunks=total_chunks,
    )
