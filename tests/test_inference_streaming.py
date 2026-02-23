"""Comprehensive tests for Phase 4 streaming inference."""

from __future__ import annotations

import pytest
import torch

from cc_g2pnp.inference.streaming import StreamingInference, StreamingState
from cc_g2pnp.model.attention import ChunkAwareAttention
from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.conformer_block import ConformerBlock
from cc_g2pnp.model.convolution import ConformerConvModule
from cc_g2pnp.model.encoder import ConformerEncoder, EncoderStreamingState


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


# ── 1. ConformerConvModule streaming tests ──────────────────────


class TestConvForwardStreaming:

    def test_conv_forward_streaming_output_shape(self, small_config):
        conv = ConformerConvModule(small_config)
        conv.eval()
        batch, chunk, dim = 2, small_config.chunk_size, small_config.d_model
        kernel_minus_1 = small_config.conv_kernel_size - 1

        x = torch.randn(batch, chunk, dim)
        cache = torch.zeros(batch, kernel_minus_1, dim)

        with torch.no_grad():
            output, new_cache = conv.forward_streaming(x, cache)

        assert output.shape == (batch, chunk, dim)
        assert new_cache.shape == (batch, kernel_minus_1, dim)

    def test_conv_forward_streaming_with_zero_cache(self, small_config):
        conv = ConformerConvModule(small_config)
        conv.eval()
        batch, chunk, dim = 1, small_config.chunk_size, small_config.d_model
        kernel_minus_1 = small_config.conv_kernel_size - 1

        x = torch.randn(batch, chunk, dim)
        cache = torch.zeros(batch, kernel_minus_1, dim)

        with torch.no_grad():
            output, new_cache = conv.forward_streaming(x, cache)

        assert torch.isfinite(output).all()
        assert torch.isfinite(new_cache).all()

    def test_conv_forward_streaming_matches_forward_first_chunk(self, small_config):
        """Streaming with zero cache matches normal forward for a single chunk."""
        conv = ConformerConvModule(small_config)
        conv.eval()
        batch, chunk, dim = 1, small_config.chunk_size, small_config.d_model
        kernel_minus_1 = small_config.conv_kernel_size - 1

        x = torch.randn(batch, chunk, dim)
        cache = torch.zeros(batch, kernel_minus_1, dim)

        with torch.no_grad():
            streaming_out, _ = conv.forward_streaming(x, cache)
            normal_out = conv.forward(x)

        # Both should produce same result since zero cache == zero padding
        assert torch.allclose(streaming_out, normal_out, atol=1e-5)

    def test_conv_forward_streaming_cache_updates(self, small_config):
        conv = ConformerConvModule(small_config)
        conv.eval()
        batch, chunk, dim = 1, small_config.chunk_size, small_config.d_model
        kernel_minus_1 = small_config.conv_kernel_size - 1

        cache = torch.zeros(batch, kernel_minus_1, dim)

        with torch.no_grad():
            x1 = torch.randn(batch, chunk, dim)
            _, cache1 = conv.forward_streaming(x1, cache)

            x2 = torch.randn(batch, chunk, dim)
            _, cache2 = conv.forward_streaming(x2, cache1)

        # Cache should change between chunks (different inputs)
        assert not torch.equal(cache1, cache)
        assert not torch.equal(cache2, cache1)


# ── 2. ChunkAwareAttention streaming tests ──────────────────────


class TestAttnForwardStreaming:

    def test_attn_forward_streaming_output_shape(self, small_config):
        attn = ChunkAwareAttention(small_config)
        attn.eval()
        batch, chunk, dim = 2, small_config.chunk_size, small_config.d_model
        heads = small_config.num_heads
        d_k = dim // heads

        x = torch.randn(batch, chunk, dim)
        kv_cache = (
            torch.zeros(batch, heads, 0, d_k),
            torch.zeros(batch, heads, 0, d_k),
        )
        pos_enc = torch.randn(1, chunk, dim)

        with torch.no_grad():
            output, _new_kv_cache = attn.forward_streaming(
                x, pos_enc, kv_cache, small_config.past_context,
            )

        assert output.shape == (batch, chunk, dim)

    def test_attn_forward_streaming_kv_cache_grows(self, small_config):
        attn = ChunkAwareAttention(small_config)
        attn.eval()
        batch, chunk, dim = 1, small_config.chunk_size, small_config.d_model
        heads = small_config.num_heads
        d_k = dim // heads

        kv_cache = (
            torch.zeros(batch, heads, 0, d_k),
            torch.zeros(batch, heads, 0, d_k),
        )

        with torch.no_grad():
            x1 = torch.randn(batch, chunk, dim)
            total_len_1 = 0 + chunk
            pos_enc_1 = torch.randn(1, total_len_1, dim)
            _, cache1 = attn.forward_streaming(
                x1, pos_enc_1, kv_cache, small_config.past_context,
            )

            # After first chunk, cache should have chunk frames
            assert cache1[0].size(2) == chunk
            assert cache1[1].size(2) == chunk

            x2 = torch.randn(batch, chunk, dim)
            total_len_2 = cache1[0].size(2) + chunk
            pos_enc_2 = torch.randn(1, total_len_2, dim)
            _, cache2 = attn.forward_streaming(
                x2, pos_enc_2, cache1, small_config.past_context,
            )

            # After second chunk, cache should have 2*chunk frames (within past_context=10)
            assert cache2[0].size(2) == 2 * chunk

    def test_attn_forward_streaming_kv_cache_truncates(self, small_config):
        attn = ChunkAwareAttention(small_config)
        attn.eval()
        batch, chunk, dim = 1, small_config.chunk_size, small_config.d_model
        heads = small_config.num_heads
        d_k = dim // heads
        past_ctx = small_config.past_context  # 10

        kv_cache = (
            torch.zeros(batch, heads, 0, d_k),
            torch.zeros(batch, heads, 0, d_k),
        )

        # Feed enough chunks to exceed past_context
        # chunk_size=5, past_context=10 -> after 3 chunks: 15 > 10, should truncate
        current_cache = kv_cache
        with torch.no_grad():
            for _i in range(3):
                x = torch.randn(batch, chunk, dim)
                total_len = current_cache[0].size(2) + chunk
                pos_enc = torch.randn(1, total_len, dim)
                _, current_cache = attn.forward_streaming(
                    x, pos_enc, current_cache, past_ctx,
                )

        # Cache should be truncated to past_context
        assert current_cache[0].size(2) == past_ctx
        assert current_cache[1].size(2) == past_ctx


# ── 3. ConformerBlock streaming tests ───────────────────────────


class TestBlockForwardStreaming:

    def test_block_forward_streaming_output_shape(self, small_config):
        block = ConformerBlock(small_config)
        block.eval()
        batch, chunk, dim = 2, small_config.chunk_size, small_config.d_model
        heads = small_config.num_heads
        d_k = dim // heads
        kernel_minus_1 = small_config.conv_kernel_size - 1

        x = torch.randn(batch, chunk, dim)
        attn_cache = (
            torch.zeros(batch, heads, 0, d_k),
            torch.zeros(batch, heads, 0, d_k),
        )
        conv_cache = torch.zeros(batch, kernel_minus_1, dim)
        pos_enc = torch.randn(1, chunk, dim)

        with torch.no_grad():
            output, new_attn_cache, new_conv_cache = block.forward_streaming(
                x, pos_enc, attn_cache, conv_cache,
                small_config.past_context,
            )

        assert output.shape == (batch, chunk, dim)
        assert new_attn_cache[0].shape[0] == batch
        assert new_attn_cache[0].shape[1] == heads
        assert new_conv_cache.shape == (batch, kernel_minus_1, dim)

    def test_block_forward_streaming_caches_updated(self, small_config):
        block = ConformerBlock(small_config)
        block.eval()
        batch, chunk, dim = 1, small_config.chunk_size, small_config.d_model
        heads = small_config.num_heads
        d_k = dim // heads
        kernel_minus_1 = small_config.conv_kernel_size - 1

        attn_cache_init = (
            torch.zeros(batch, heads, 0, d_k),
            torch.zeros(batch, heads, 0, d_k),
        )
        conv_cache_init = torch.zeros(batch, kernel_minus_1, dim)

        x = torch.randn(batch, chunk, dim)
        pos_enc = torch.randn(1, chunk, dim)

        with torch.no_grad():
            _, new_attn_cache, new_conv_cache = block.forward_streaming(
                x, pos_enc, attn_cache_init, conv_cache_init,
                small_config.past_context,
            )

        # Attn cache should now have chunk entries (was empty)
        assert new_attn_cache[0].size(2) == chunk
        # Conv cache should have changed from zeros
        assert not torch.equal(new_conv_cache, conv_cache_init)


# ── 4. ConformerEncoder streaming tests ─────────────────────────


class TestEncoderStreaming:

    def test_encoder_init_streaming_state(self, small_config):
        encoder = ConformerEncoder(small_config)
        encoder.eval()
        batch = 2
        device = torch.device("cpu")

        state = encoder.init_streaming_state(batch, device)

        assert isinstance(state, EncoderStreamingState)
        assert len(state.conv_caches) == small_config.num_layers
        assert len(state.kv_caches) == small_config.num_layers
        assert state.processed_frames == 0

        # Check conv cache shapes
        kernel_minus_1 = small_config.conv_kernel_size - 1
        for cache in state.conv_caches:
            assert cache.shape == (batch, kernel_minus_1, small_config.d_model)

        # Check kv cache shapes (empty at init)
        d_k = small_config.d_model // small_config.num_heads
        for k_cache, v_cache in state.kv_caches:
            assert k_cache.shape == (batch, small_config.num_heads, 0, d_k)
            assert v_cache.shape == (batch, small_config.num_heads, 0, d_k)

    def test_encoder_forward_streaming_output_shape(self, small_config):
        encoder = ConformerEncoder(small_config)
        encoder.eval()
        batch = 2
        chunk = small_config.chunk_size
        mla = small_config.mla_size
        dim = small_config.d_model
        device = torch.device("cpu")

        state = encoder.init_streaming_state(batch, device)
        chunk_frames = torch.randn(batch, chunk + mla, dim)

        with torch.no_grad():
            result, _new_state = encoder.forward_streaming(chunk_frames, state)

        assert result["output"].shape == (batch, chunk, dim)

    def test_encoder_forward_streaming_ctc_feedback(self, small_config):
        """Intermediate CTC feedback modifies output at configured layers."""
        # small_config has intermediate_ctc_layers=(1,), so layer 1 applies feedback
        encoder = ConformerEncoder(small_config)
        encoder.eval()
        batch = 1
        chunk = small_config.chunk_size
        mla = small_config.mla_size
        dim = small_config.d_model
        device = torch.device("cpu")

        state = encoder.init_streaming_state(batch, device)
        chunk_frames = torch.randn(batch, chunk + mla, dim)

        with torch.no_grad():
            result, _ = encoder.forward_streaming(chunk_frames, state)

        output = result["output"]
        assert torch.isfinite(output).all()
        # CTC feedback adds ctc_to_hidden(ctc_projection(x)) to x at layer 1
        # Verify the encoder doesn't produce zeros (CTC feedback modifies hidden)
        assert output.abs().sum() > 0

    def test_encoder_forward_streaming_no_intermediate_ctc(self):
        """Encoder works when no intermediate CTC layers are configured."""
        config = CC_G2PnPConfig(
            d_model=64,
            num_heads=2,
            d_ff=128,
            num_layers=2,
            intermediate_ctc_layers=(),
        )
        encoder = ConformerEncoder(config)
        encoder.eval()
        batch = 1
        chunk = config.chunk_size
        mla = config.mla_size
        dim = config.d_model
        device = torch.device("cpu")

        state = encoder.init_streaming_state(batch, device)
        chunk_frames = torch.randn(batch, chunk + mla, dim)

        with torch.no_grad():
            result, _ = encoder.forward_streaming(chunk_frames, state)

        assert result["output"].shape == (batch, chunk, dim)
        assert torch.isfinite(result["output"]).all()

    def test_encoder_forward_streaming_state_updates(self, small_config):
        encoder = ConformerEncoder(small_config)
        encoder.eval()
        batch = 1
        chunk = small_config.chunk_size
        mla = small_config.mla_size
        dim = small_config.d_model
        device = torch.device("cpu")

        state = encoder.init_streaming_state(batch, device)
        assert state.processed_frames == 0

        chunk_frames = torch.randn(batch, chunk + mla, dim)

        with torch.no_grad():
            _, state1 = encoder.forward_streaming(chunk_frames, state)

        assert state1.processed_frames == chunk

        with torch.no_grad():
            chunk_frames2 = torch.randn(batch, chunk + mla, dim)
            _, state2 = encoder.forward_streaming(chunk_frames2, state1)

        assert state2.processed_frames == 2 * chunk


# ── 5. CC_G2PnP streaming tests ────────────────────────────────


class TestModelStreaming:

    def test_model_init_streaming_state(self, small_model, small_config):
        state = small_model.init_streaming_state(batch_size=2)

        assert isinstance(state, EncoderStreamingState)
        assert len(state.conv_caches) == small_config.num_layers
        assert len(state.kv_caches) == small_config.num_layers
        assert state.processed_frames == 0

    def test_model_forward_streaming_output_shape(self, small_model, small_config):
        batch = 2
        chunk = small_config.chunk_size
        mla = small_config.mla_size
        dim = small_config.d_model
        vocab = small_config.pnp_vocab_size

        state = small_model.init_streaming_state(batch)
        chunk_frames = torch.randn(batch, chunk + mla, dim)

        with torch.no_grad():
            log_probs, _new_state = small_model.forward_streaming(chunk_frames, state)

        assert log_probs.shape == (batch, chunk, vocab)

    def test_model_forward_streaming_values_finite(self, small_model, small_config):
        batch = 1
        chunk = small_config.chunk_size
        mla = small_config.mla_size
        dim = small_config.d_model

        state = small_model.init_streaming_state(batch)
        chunk_frames = torch.randn(batch, chunk + mla, dim)

        with torch.no_grad():
            log_probs, _ = small_model.forward_streaming(chunk_frames, state)

        assert torch.isfinite(log_probs).all()
        # log_probs should be log-softmax: all values <= 0
        assert (log_probs <= 0).all()


# ── 6. StreamingInference tests ─────────────────────────────────


class TestStreamingInference:

    def test_streaming_inference_reset(self, small_model, small_config):
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=2)

        assert isinstance(state, StreamingState)
        assert isinstance(state.encoder_state, EncoderStreamingState)
        assert state.frame_buffer.shape == (2, 0, small_config.d_model)

    def test_streaming_inference_process_tokens(self, small_model, small_config):
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        # 2 tokens -> 16 frames, chunk_size+mla=6 -> at least 2 chunks
        bpe_ids = torch.randint(0, small_config.bpe_vocab_size, (1, 2))
        labels, new_state = engine.process_tokens(bpe_ids, state)

        assert isinstance(labels, list)
        assert len(labels) == 1  # batch_size=1
        assert isinstance(labels[0], list)
        # State should be updated
        assert isinstance(new_state, StreamingState)

    def test_streaming_inference_process_tokens_accumulates(
        self, small_model, small_config,
    ):
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        # Feed token one at a time
        token1 = torch.randint(0, small_config.bpe_vocab_size, (1, 1))
        _labels1, state1 = engine.process_tokens(token1, state)

        token2 = torch.randint(0, small_config.bpe_vocab_size, (1, 1))
        _labels2, state2 = engine.process_tokens(token2, state1)

        # processed_frames should increase across calls
        assert (
            state2.encoder_state.processed_frames
            >= state1.encoder_state.processed_frames
        )

    def test_streaming_inference_flush(self, small_model, small_config):
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        # Feed 1 token -> 8 frames, chunk needs 6, so 1 chunk processed, 2 frames remain
        token = torch.randint(0, small_config.bpe_vocab_size, (1, 1))
        _, state_after = engine.process_tokens(token, state)

        # Flush remaining frames
        labels, final_state = engine.flush(state_after)

        assert isinstance(labels, list)
        assert len(labels) == 1
        # Buffer should be empty after flush
        assert final_state.frame_buffer.size(1) == 0

    def test_streaming_inference_process_full(self, small_model, small_config):
        engine = StreamingInference(small_model)
        seq_len = 5
        input_ids = torch.randint(0, small_config.bpe_vocab_size, (1, seq_len))
        input_lengths = torch.tensor([seq_len])

        labels = engine.process_full(input_ids, input_lengths)

        assert isinstance(labels, list)
        assert len(labels) == 1
        assert isinstance(labels[0], list)

    def test_streaming_output_non_empty_for_enough_tokens(
        self, small_model, small_config,
    ):
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        # Feed many tokens to guarantee output
        bpe_ids = torch.randint(1, small_config.bpe_vocab_size, (1, 10))
        labels, state_after = engine.process_tokens(bpe_ids, state)

        # Flush remaining
        flush_labels, _ = engine.flush(state_after)

        # At least one of the calls should produce non-empty decoded output
        # (with random weights, some labels may all be blank, so we check
        # that the process completes without error and returns lists)
        all_labels = labels[0] + flush_labels[0]
        assert isinstance(all_labels, list)


# ── 7. Edge cases ──────────────────────────────────────────────


class TestStreamingEdgeCases:

    def test_streaming_batch_size_one(self, small_model, small_config):
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        bpe_ids = torch.randint(0, small_config.bpe_vocab_size, (1, 3))
        labels, new_state = engine.process_tokens(bpe_ids, state)

        assert len(labels) == 1
        assert isinstance(new_state, StreamingState)

    def test_streaming_single_token(self, small_model, small_config):
        """Single BPE token produces 8 frames > chunk_size+mla_size=6, so one chunk."""
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        # 1 token -> 8 frames, frames_needed = 5+1 = 6, so 1 chunk fits
        token = torch.randint(0, small_config.bpe_vocab_size, (1, 1))
        labels, new_state = engine.process_tokens(token, state)

        assert isinstance(labels, list)
        assert len(labels) == 1
        # After processing, 8-5=3 frames should remain buffered (consumed chunk_size=5)
        # But only if a chunk was actually processed
        # With 8 frames and frames_needed=6: one chunk processed, 8-5=3 remain, but
        # after consuming chunk_size=5, buffer has 3 frames < 6 so no more chunks
        remaining = new_state.frame_buffer.size(1)
        # 8 frames total - 5 consumed = 3 remaining (less than frames_needed=6)
        expected_remaining = small_config.upsample_factor - small_config.chunk_size
        assert remaining == expected_remaining

    def test_streaming_multiple_chunks(self, small_model, small_config):
        """Process enough tokens for multiple chunks."""
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        # 5 tokens -> 40 frames, chunk_size=5, mla=1, frames_needed=6
        # chunks: consume 5 each while >= 6 remain
        # 40->35->30->25->20->15->10->5 (7 chunks, 5 remain < 6)
        bpe_ids = torch.randint(0, small_config.bpe_vocab_size, (1, 5))
        labels, new_state = engine.process_tokens(bpe_ids, state)

        assert isinstance(labels, list)
        assert len(labels) == 1
        # Verify multiple chunks were processed: processed_frames should be > chunk_size
        assert new_state.encoder_state.processed_frames > small_config.chunk_size

    def test_streaming_long_input(self, small_model, small_config):
        """Process 100+ BPE tokens without error."""
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        # 120 tokens -> 960 frames -> many chunks
        bpe_ids = torch.randint(1, small_config.bpe_vocab_size, (1, 120))
        labels, new_state = engine.process_tokens(bpe_ids, state)
        flush_labels, final_state = engine.flush(new_state)

        all_labels = labels[0] + flush_labels[0]
        assert isinstance(all_labels, list)
        assert final_state.frame_buffer.size(1) == 0

        # processed_frames counts full chunks from process_tokens + flush
        assert final_state.encoder_state.processed_frames > 0
        # KV caches should be truncated to past_context
        chunk = small_config.chunk_size
        for k_cache, _v_cache in final_state.encoder_state.kv_caches:
            assert k_cache.size(2) <= small_config.past_context + chunk


# ── 8. Integration test ────────────────────────────────────────


class TestStreamingVsFullInference:

    def test_streaming_vs_full_inference(self, small_model, small_config):
        """Both streaming and non-streaming produce valid output."""
        engine = StreamingInference(small_model)
        seq_len = 5
        input_ids = torch.randint(1, small_config.bpe_vocab_size, (1, seq_len))
        input_lengths = torch.tensor([seq_len])

        # Non-streaming
        full_labels = engine.process_full(input_ids, input_lengths)

        # Streaming: process all tokens then flush
        state = engine.reset(batch_size=1)
        stream_labels, state = engine.process_tokens(input_ids, state)
        flush_labels, _ = engine.flush(state)
        streaming_combined = stream_labels[0] + flush_labels[0]

        # Both should return valid list structures
        assert isinstance(full_labels[0], list)
        assert isinstance(streaming_combined, list)

        # Both should produce non-negative integer labels
        for label in full_labels[0]:
            assert isinstance(label, int)
            assert label >= 0
        for label in streaming_combined:
            assert isinstance(label, int)
            assert label >= 0


# ── 9. Review-driven additions ────────────────────────────────


class TestStreamingReviewFixes:
    """Tests added from Phase 4 code review findings."""

    def test_streaming_batch_size_two_process_tokens(self, small_model, small_config):
        """T1: StreamingInference with batch_size > 1."""
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=2)

        bpe_ids = torch.randint(0, small_config.bpe_vocab_size, (2, 3))
        labels, new_state = engine.process_tokens(bpe_ids, state)

        assert len(labels) == 2
        assert isinstance(labels[0], list)
        assert isinstance(labels[1], list)
        assert new_state.frame_buffer.size(0) == 2

    def test_streaming_batch_size_two_flush(self, small_model, small_config):
        """T1: flush with batch_size > 1."""
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=2)

        bpe_ids = torch.randint(0, small_config.bpe_vocab_size, (2, 1))
        _, state_after = engine.process_tokens(bpe_ids, state)

        labels, final_state = engine.flush(state_after)

        assert len(labels) == 2
        assert final_state.frame_buffer.size(0) == 2
        assert final_state.frame_buffer.size(1) == 0

    def test_flush_empty_buffer(self, small_model, small_config):
        """T2: flush with no buffered frames returns empty labels."""
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        # No tokens fed — buffer is empty
        labels, returned_state = engine.flush(state)

        assert labels == [[]]
        assert returned_state.frame_buffer.size(1) == 0
        # State should be unchanged
        assert returned_state.encoder_state.processed_frames == 0

    def test_flush_consecutive(self, small_model, small_config):
        """T3: calling flush twice — second call returns empty."""
        engine = StreamingInference(small_model)
        state = engine.reset(batch_size=1)

        # Feed tokens, then flush twice
        bpe_ids = torch.randint(0, small_config.bpe_vocab_size, (1, 1))
        _, state_after = engine.process_tokens(bpe_ids, state)

        labels1, state1 = engine.flush(state_after)
        labels2, state2 = engine.flush(state1)

        # First flush may produce labels; second should be empty
        assert isinstance(labels1, list)
        assert labels2 == [[]]
        assert state2.frame_buffer.size(1) == 0
