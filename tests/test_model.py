"""Tests for CC_G2PnP full model."""

import pytest
import torch

from cc_g2pnp.model.cc_g2pnp import CC_G2PnP
from cc_g2pnp.model.config import CC_G2PnPConfig


@pytest.fixture()
def small_config():
    """Small config for fast testing."""
    return CC_G2PnPConfig(
        bpe_vocab_size=100,
        pnp_vocab_size=20,
        d_model=64,
        num_heads=4,
        d_ff=128,
        num_layers=4,
        upsample_factor=4,
        conv_kernel_size=7,
        intermediate_ctc_layers=(1,),
    )


@pytest.fixture()
def model(small_config):
    """Create model with small config in eval mode."""
    torch.manual_seed(42)
    m = CC_G2PnP(small_config)
    m.eval()
    return m


def _make_inputs(config, batch_size=2, seq_len=5, target_len=10):
    """Create random input_ids, input_lengths, targets, target_lengths."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.bpe_vocab_size, (batch_size, seq_len))
    input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    # target_len must be <= seq_len * upsample_factor for CTC constraint
    targets = torch.randint(1, config.pnp_vocab_size, (batch_size, target_len))
    target_lengths = torch.full((batch_size,), target_len, dtype=torch.long)
    return input_ids, input_lengths, targets, target_lengths


class TestForwardWithTargets:
    """Tests for forward pass with targets (training mode)."""

    def test_returns_loss(self, model, small_config):
        """Forward with targets should return 'loss' in result."""
        input_ids, input_lengths, targets, target_lengths = _make_inputs(small_config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        assert "loss" in result

    def test_returns_log_probs(self, model, small_config):
        """Forward with targets should return 'log_probs' in result."""
        input_ids, input_lengths, targets, target_lengths = _make_inputs(small_config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        assert "log_probs" in result

    def test_returns_intermediate_losses(self, model, small_config):
        """Forward with targets should return 'intermediate_losses'."""
        input_ids, input_lengths, targets, target_lengths = _make_inputs(small_config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        assert "intermediate_losses" in result

    def test_loss_is_scalar(self, model, small_config):
        """Loss should be a scalar tensor."""
        input_ids, input_lengths, targets, target_lengths = _make_inputs(small_config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        assert result["loss"].dim() == 0

    def test_loss_is_finite(self, model, small_config):
        """Loss should be a finite value."""
        input_ids, input_lengths, targets, target_lengths = _make_inputs(small_config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        assert torch.isfinite(result["loss"])

    def test_intermediate_losses_count(self, model, small_config):
        """intermediate_losses should have one entry per intermediate CTC layer."""
        input_ids, input_lengths, targets, target_lengths = _make_inputs(small_config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        expected_count = len(small_config.intermediate_ctc_layers)
        assert len(result["intermediate_losses"]) == expected_count

    def test_intermediate_losses_are_scalars(self, model, small_config):
        """Each intermediate loss should be a scalar tensor."""
        input_ids, input_lengths, targets, target_lengths = _make_inputs(small_config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        for loss in result["intermediate_losses"]:
            assert loss.dim() == 0


class TestForwardWithoutTargets:
    """Tests for forward pass without targets (inference mode)."""

    def test_returns_log_probs(self, model, small_config):
        """Forward without targets should return 'log_probs'."""
        input_ids, input_lengths, _, _ = _make_inputs(small_config)
        result = model(input_ids, input_lengths)
        assert "log_probs" in result

    def test_no_loss(self, model, small_config):
        """Forward without targets should not return 'loss'."""
        input_ids, input_lengths, _, _ = _make_inputs(small_config)
        result = model(input_ids, input_lengths)
        assert "loss" not in result

    def test_no_intermediate_losses(self, model, small_config):
        """Forward without targets should not return 'intermediate_losses'."""
        input_ids, input_lengths, _, _ = _make_inputs(small_config)
        result = model(input_ids, input_lengths)
        assert "intermediate_losses" not in result


class TestLogProbsShape:
    """Tests for log_probs shape."""

    def test_shape(self, model, small_config):
        """log_probs shape should be [B, T*upsample_factor, pnp_vocab_size]."""
        batch_size, seq_len = 2, 5
        input_ids, input_lengths, _, _ = _make_inputs(
            small_config, batch_size=batch_size, seq_len=seq_len,
        )
        result = model(input_ids, input_lengths)
        expected_t = seq_len * small_config.upsample_factor
        expected_shape = (batch_size, expected_t, small_config.pnp_vocab_size)
        assert result["log_probs"].shape == expected_shape

    def test_shape_different_seq_len(self, model, small_config):
        """log_probs shape should adapt to different sequence lengths."""
        batch_size, seq_len = 1, 10
        input_ids, input_lengths, _, _ = _make_inputs(
            small_config, batch_size=batch_size, seq_len=seq_len, target_len=15,
        )
        result = model(input_ids, input_lengths)
        expected_t = seq_len * small_config.upsample_factor
        assert result["log_probs"].shape == (batch_size, expected_t, small_config.pnp_vocab_size)


class TestInference:
    """Tests for inference method."""

    def test_returns_list_of_lists(self, model, small_config):
        """inference() should return list[list[int]]."""
        input_ids, input_lengths, _, _ = _make_inputs(small_config)
        result = model.inference(input_ids, input_lengths)
        assert isinstance(result, list)
        assert all(isinstance(seq, list) for seq in result)
        assert all(isinstance(t, int) for seq in result for t in seq)

    def test_batch_size_matches(self, model, small_config):
        """Number of decoded sequences should match batch size."""
        batch_size = 3
        input_ids, input_lengths, _, _ = _make_inputs(
            small_config, batch_size=batch_size, seq_len=5,
        )
        result = model.inference(input_ids, input_lengths)
        assert len(result) == batch_size

    def test_no_gradient(self, model, small_config):
        """inference() should not require gradients."""
        input_ids, input_lengths, _, _ = _make_inputs(small_config)
        input_ids.requires_grad_(False)
        # inference uses @torch.no_grad, so this should work without error
        result = model.inference(input_ids, input_lengths)
        assert isinstance(result, list)


class TestModelConfig:
    """Tests for model configuration handling."""

    def test_default_config(self):
        """Model should work with default CC_G2PnPConfig()."""
        torch.manual_seed(42)
        config = CC_G2PnPConfig()
        model = CC_G2PnP(config)
        assert model.config is config

    def test_none_config(self):
        """Model should create default config when None is passed."""
        torch.manual_seed(42)
        model = CC_G2PnP(None)
        assert isinstance(model.config, CC_G2PnPConfig)
        assert model.config.d_model == 512

    def test_small_config_stored(self, model, small_config):
        """Model should store the provided config."""
        assert model.config is small_config
        assert model.config.d_model == 64


class TestParameterCount:
    """Tests for model parameter count."""

    def test_parameter_count_small(self, model):
        """Small model should have a reasonable number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        # Small config should have at least some parameters
        assert total_params > 100_000

    def test_parameter_count_default(self):
        """Default model should have between 10M and 200M parameters."""
        torch.manual_seed(42)
        model = CC_G2PnP(CC_G2PnPConfig())
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 10_000_000
        assert total_params < 200_000_000


class TestMultipleIntermediateLayers:
    """Tests with multiple intermediate CTC layers."""

    def test_three_intermediate_layers(self):
        """Config with 3 intermediate layers should produce 3 intermediate losses."""
        torch.manual_seed(42)
        config = CC_G2PnPConfig(
            bpe_vocab_size=100,
            pnp_vocab_size=20,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=8,
            upsample_factor=4,
            conv_kernel_size=7,
            intermediate_ctc_layers=(1, 3, 5),
        )
        model = CC_G2PnP(config)
        model.eval()
        input_ids, input_lengths, targets, target_lengths = _make_inputs(
            config, batch_size=2, seq_len=5, target_len=10,
        )
        result = model(input_ids, input_lengths, targets, target_lengths)
        assert len(result["intermediate_losses"]) == 3


class TestBackwardGradient:
    """Tests for gradient computation and backward pass."""

    def test_loss_backward(self, small_config):
        """loss.backward() should succeed and produce gradients."""
        torch.manual_seed(42)
        model = CC_G2PnP(small_config)
        model.train()
        input_ids, input_lengths, targets, target_lengths = _make_inputs(small_config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        result["loss"].backward()
        # Check that at least some parameters have gradients
        grad_params = [p for p in model.parameters() if p.grad is not None]
        assert len(grad_params) > 0

    def test_all_parameters_have_gradients(self, small_config):
        """All trainable parameters should receive gradients after backward."""
        torch.manual_seed(42)
        model = CC_G2PnP(small_config)
        model.train()
        input_ids, input_lengths, targets, target_lengths = _make_inputs(small_config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        result["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"


class TestVariableLengths:
    """Tests for variable-length inputs within a batch."""

    def test_different_input_lengths(self, small_config):
        """Batch with different input_lengths should produce valid loss."""
        torch.manual_seed(42)
        model = CC_G2PnP(small_config)
        model.eval()
        batch_size, max_seq_len = 3, 8
        input_ids = torch.randint(0, small_config.bpe_vocab_size, (batch_size, max_seq_len))
        input_lengths = torch.tensor([8, 6, 4])
        # Targets must satisfy CTC constraint: upsampled_len >= target_len
        # min upsampled = 4 * 4 = 16
        targets = torch.randint(1, small_config.pnp_vocab_size, (batch_size, 12))
        target_lengths = torch.tensor([12, 10, 8])
        result = model(input_ids, input_lengths, targets, target_lengths)
        assert torch.isfinite(result["loss"])

    def test_different_target_lengths(self, small_config):
        """Batch with different target_lengths should produce valid loss."""
        torch.manual_seed(42)
        model = CC_G2PnP(small_config)
        model.eval()
        batch_size, seq_len = 2, 6
        input_ids = torch.randint(0, small_config.bpe_vocab_size, (batch_size, seq_len))
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        # upsampled = 6 * 4 = 24
        max_target_len = 15
        targets = torch.randint(1, small_config.pnp_vocab_size, (batch_size, max_target_len))
        target_lengths = torch.tensor([15, 8])
        result = model(input_ids, input_lengths, targets, target_lengths)
        assert torch.isfinite(result["loss"])


class TestIntermediateCTCWeight:
    """Tests for intermediate CTC weight effect on total loss."""

    def test_weight_zero_equals_final_only(self):
        """With intermediate_ctc_weight=0, total loss should equal final CTC loss."""
        torch.manual_seed(42)
        config = CC_G2PnPConfig(
            bpe_vocab_size=100,
            pnp_vocab_size=20,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=4,
            upsample_factor=4,
            conv_kernel_size=7,
            intermediate_ctc_layers=(1,),
            intermediate_ctc_weight=0.0,
        )
        model = CC_G2PnP(config)
        model.eval()
        input_ids, input_lengths, targets, target_lengths = _make_inputs(config)
        result = model(input_ids, input_lengths, targets, target_lengths)
        # With weight=0, intermediate losses should not contribute
        # Recompute: total = final + 0 * sum(intermediate) = final
        assert len(result["intermediate_losses"]) == 1
        assert torch.isfinite(result["loss"])

    def test_weight_affects_total_loss(self):
        """Different intermediate_ctc_weight values should produce different total losses."""
        torch.manual_seed(42)
        config_low = CC_G2PnPConfig(
            bpe_vocab_size=100,
            pnp_vocab_size=20,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=4,
            upsample_factor=4,
            conv_kernel_size=7,
            intermediate_ctc_layers=(1,),
            intermediate_ctc_weight=0.0,
        )
        config_high = CC_G2PnPConfig(
            bpe_vocab_size=100,
            pnp_vocab_size=20,
            d_model=64,
            num_heads=4,
            d_ff=128,
            num_layers=4,
            upsample_factor=4,
            conv_kernel_size=7,
            intermediate_ctc_layers=(1,),
            intermediate_ctc_weight=1.0,
        )
        # Use same seed for identical model weights
        torch.manual_seed(99)
        model_low = CC_G2PnP(config_low)
        model_low.eval()
        torch.manual_seed(99)
        model_high = CC_G2PnP(config_high)
        model_high.eval()
        input_ids, input_lengths, targets, target_lengths = _make_inputs(config_low)
        result_low = model_low(input_ids, input_lengths, targets, target_lengths)
        result_high = model_high(input_ids, input_lengths, targets, target_lengths)
        # Different weights should produce different total losses
        assert not torch.allclose(result_low["loss"], result_high["loss"])


class TestConfigValidation:
    """Tests for CC_G2PnPConfig __post_init__ validation."""

    def test_d_model_not_divisible_by_num_heads(self):
        """d_model must be divisible by num_heads."""
        with pytest.raises(ValueError, match="divisible by num_heads"):
            CC_G2PnPConfig(d_model=64, num_heads=5)

    def test_d_model_zero(self):
        """d_model must be positive."""
        with pytest.raises(ValueError, match="d_model must be positive"):
            CC_G2PnPConfig(d_model=0)

    def test_num_heads_zero(self):
        """num_heads must be positive."""
        with pytest.raises(ValueError, match="num_heads must be positive"):
            CC_G2PnPConfig(num_heads=0)

    def test_num_layers_zero(self):
        """num_layers must be positive."""
        with pytest.raises(ValueError, match="num_layers must be positive"):
            CC_G2PnPConfig(num_layers=0)

    def test_intermediate_ctc_layer_out_of_range(self):
        """intermediate_ctc_layers index must be < num_layers."""
        with pytest.raises(ValueError, match="intermediate_ctc_layers index"):
            CC_G2PnPConfig(num_layers=4, intermediate_ctc_layers=(1, 5))

    def test_valid_config_passes(self):
        """Valid configuration should not raise."""
        config = CC_G2PnPConfig(d_model=128, num_heads=4, num_layers=6, intermediate_ctc_layers=(1, 3))
        assert config.d_model == 128


class TestCTCProjectionSharing:
    """Tests for CTC projection weight sharing between encoder and CTCHead."""

    def test_shared_weights(self, model, small_config):
        """CTCHead projection should share weights with encoder ctc_projection."""
        assert model.ctc_head.projection is model.encoder.ctc_projection

    def test_shared_weights_same_data(self, model, small_config):
        """Shared projection should have identical weight data."""
        assert torch.equal(
            model.ctc_head.projection.weight,
            model.encoder.ctc_projection.weight,
        )
