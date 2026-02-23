"""Tests for CTCHead and greedy_decode."""

import pytest
import torch

from cc_g2pnp.model.config import CC_G2PnPConfig
from cc_g2pnp.model.ctc_decoder import CTCHead, greedy_decode


@pytest.fixture()
def small_config():
    """Config with small pnp_vocab_size for testing."""
    return CC_G2PnPConfig(
        bpe_vocab_size=100,
        pnp_vocab_size=10,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=4,
        upsample_factor=4,
        conv_kernel_size=7,
        intermediate_ctc_layers=(1,),
    )


@pytest.fixture()
def default_config():
    """Default config (pnp_vocab_size=140, d_model=512)."""
    return CC_G2PnPConfig()


class TestCTCHead:
    """Tests for CTCHead module."""

    def test_output_shape_default(self, default_config):
        """Output shape should be [B, T, 140] for input [B, T, 512]."""
        torch.manual_seed(42)
        head = CTCHead(default_config)
        head.eval()
        x = torch.randn(2, 10, 512)
        out = head(x)
        assert out.shape == (2, 10, 140)

    def test_output_shape_custom(self, small_config):
        """Output shape should match custom pnp_vocab_size."""
        torch.manual_seed(42)
        head = CTCHead(small_config)
        head.eval()
        x = torch.randn(3, 8, 64)
        out = head(x)
        assert out.shape == (3, 8, 10)

    def test_output_is_log_probabilities(self, default_config):
        """exp(output).sum(dim=-1) should be approximately 1.0."""
        torch.manual_seed(42)
        head = CTCHead(default_config)
        head.eval()
        x = torch.randn(2, 5, 512)
        out = head(x)
        probs = out.exp().sum(dim=-1)
        assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)

    def test_output_is_negative(self, default_config):
        """Log probabilities should be <= 0."""
        torch.manual_seed(42)
        head = CTCHead(default_config)
        head.eval()
        x = torch.randn(2, 5, 512)
        out = head(x)
        assert (out <= 0).all()

    def test_batch_size_1(self, small_config):
        """Batch size 1 should work."""
        torch.manual_seed(42)
        head = CTCHead(small_config)
        head.eval()
        x = torch.randn(1, 4, 64)
        out = head(x)
        assert out.shape == (1, 4, 10)

    def test_eval_deterministic(self, small_config):
        """In eval mode, outputs should be deterministic."""
        torch.manual_seed(42)
        head = CTCHead(small_config)
        head.eval()
        x = torch.randn(2, 5, 64)
        out1 = head(x)
        out2 = head(x)
        assert torch.equal(out1, out2)


class TestGreedyDecode:
    """Tests for greedy_decode function."""

    def test_all_blanks(self):
        """All-blank input should produce empty sequences."""
        # Create log_probs where blank (id=0) has highest probability
        log_probs = torch.full((1, 6, 5), -10.0)
        log_probs[:, :, 0] = 0.0  # blank has highest score
        result = greedy_decode(log_probs, blank_id=0)
        assert result == [[]]

    def test_collapse_repeats_and_remove_blanks(self):
        """[1,1,2,2,0,3] should decode to [1,2,3]."""
        # Build log_probs so argmax gives [1, 1, 2, 2, 0, 3]
        desired = [1, 1, 2, 2, 0, 3]
        log_probs = torch.full((1, 6, 5), -10.0)
        for t, v in enumerate(desired):
            log_probs[0, t, v] = 0.0
        result = greedy_decode(log_probs, blank_id=0)
        assert result == [[1, 2, 3]]

    def test_batch_processing(self):
        """Multiple sequences in a batch should decode independently."""
        # Batch of 2: seq1 = [1,1,2], seq2 = [3,3,3]
        desired_batch = [[1, 1, 2], [3, 3, 3]]
        log_probs = torch.full((2, 3, 5), -10.0)
        for b, desired in enumerate(desired_batch):
            for t, v in enumerate(desired):
                log_probs[b, t, v] = 0.0
        result = greedy_decode(log_probs, blank_id=0)
        assert result == [[1, 2], [3]]

    def test_empty_sequence(self):
        """Zero-length time dimension should produce empty result."""
        log_probs = torch.empty(1, 0, 5)
        result = greedy_decode(log_probs, blank_id=0)
        assert result == [[]]

    def test_single_token(self):
        """Single non-blank token should be decoded correctly."""
        log_probs = torch.full((1, 1, 5), -10.0)
        log_probs[0, 0, 3] = 0.0
        result = greedy_decode(log_probs, blank_id=0)
        assert result == [[3]]

    def test_alternating_blank_and_token(self):
        """[0,1,0,1] should decode to [1,1] (two separate 1s)."""
        desired = [0, 1, 0, 1]
        log_probs = torch.full((1, 4, 5), -10.0)
        for t, v in enumerate(desired):
            log_probs[0, t, v] = 0.0
        result = greedy_decode(log_probs, blank_id=0)
        assert result == [[1, 1]]

    def test_custom_blank_id(self):
        """Custom blank_id should be properly removed."""
        # blank_id=2, sequence: [0, 2, 1, 2, 3]
        desired = [0, 2, 1, 2, 3]
        log_probs = torch.full((1, 5, 5), -10.0)
        for t, v in enumerate(desired):
            log_probs[0, t, v] = 0.0
        result = greedy_decode(log_probs, blank_id=2)
        assert result == [[0, 1, 3]]
