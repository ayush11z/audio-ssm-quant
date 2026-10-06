"""Phase 3 gate (brief section 6/Phase 3): unit tests for applying
fake_quantize to a real model's nn.Linear weights and activations, against
a small toy model so these run anywhere with no GPU and no real checkpoint.
"""
import torch
import torch.nn as nn

from ssmquant.quant import ActivationFakeQuantHooks, calibrate_activation_scales, quantize_linear_weights_


def make_toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))


def test_quantize_linear_weights_touches_every_linear():
    model = make_toy_model()
    n = quantize_linear_weights_(model, {"bits": 8, "granularity": "per_channel", "symmetric": True})
    assert n == 2  # the two nn.Linear layers in the toy model


def test_quantize_linear_weights_null_bits_is_noop():
    model = make_toy_model()
    original = [m.weight.data.clone() for m in model.modules() if isinstance(m, nn.Linear)]
    n = quantize_linear_weights_(model, {"bits": None})
    assert n == 0
    after = [m.weight.data for m in model.modules() if isinstance(m, nn.Linear)]
    for o, a in zip(original, after):
        assert torch.equal(o, a)


def test_quantize_linear_weights_changes_values_but_not_shape():
    model = make_toy_model()
    before = [m.weight.data.clone() for m in model.modules() if isinstance(m, nn.Linear)]
    quantize_linear_weights_(model, {"bits": 4, "granularity": "per_channel", "symmetric": True})
    after = [m.weight.data for m in model.modules() if isinstance(m, nn.Linear)]
    for b, a in zip(before, after):
        assert a.shape == b.shape
        assert not torch.equal(a, b)  # 4-bit quantization of random weights should change values


def test_activation_hooks_null_bits_is_noop():
    model = make_toy_model()
    x = torch.randn(3, 8)
    with torch.no_grad():
        expected = model(x)
    with ActivationFakeQuantHooks(model, {"bits": None}):
        with torch.no_grad():
            out = model(x)
    assert torch.equal(out, expected)


def test_per_token_activation_quant_changes_forward_output():
    model = make_toy_model()
    x = torch.randn(2, 5, 8)  # (batch, seq, hidden) to exercise the token dim
    with torch.no_grad():
        baseline = model(x)
    with ActivationFakeQuantHooks(model, {"bits": 4, "granularity": "per_token"}):
        with torch.no_grad():
            out = model(x)
    assert out.shape == baseline.shape
    assert not torch.equal(out, baseline)


def test_per_tensor_activation_quant_requires_calibration():
    model = make_toy_model()
    x = torch.randn(3, 8)
    with ActivationFakeQuantHooks(model, {"bits": 8, "granularity": "per_tensor"}) as hooks:
        try:
            with torch.no_grad():
                model(x)
            assert False, "expected ValueError for missing calibration_scales"
        except ValueError:
            pass


def test_calibrate_then_per_tensor_activation_quant_runs():
    model = make_toy_model()
    calibration_inputs = [torch.randn(3, 8) for _ in range(5)]
    scales = calibrate_activation_scales(model, calibration_inputs, bits=8)
    linear_modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert set(scales.keys()) == set(linear_modules)
    for s in scales.values():
        assert s.item() > 0

    x = torch.randn(3, 8)
    with torch.no_grad():
        baseline = model(x)
    with ActivationFakeQuantHooks(model, {"bits": 8, "granularity": "per_tensor"}, calibration_scales=scales):
        with torch.no_grad():
            out = model(x)
    assert out.shape == baseline.shape
    assert not torch.equal(out, baseline)


def test_hooks_remove_cleanly_on_exit():
    model = make_toy_model()
    linear = next(m for m in model.modules() if isinstance(m, nn.Linear))
    n_before = len(linear._forward_pre_hooks)
    with ActivationFakeQuantHooks(model, {"bits": 8, "granularity": "per_token"}):
        assert len(linear._forward_pre_hooks) == n_before + 1
    assert len(linear._forward_pre_hooks) == n_before


def test_calibration_scale_reflects_seen_magnitudes():
    model = make_toy_model()
    small = [torch.full((1, 8), 0.1) for _ in range(3)]
    large = [torch.full((1, 8), 10.0) for _ in range(3)]
    scales_small = calibrate_activation_scales(model, small, bits=8)
    scales_large = calibrate_activation_scales(model, large, bits=8)
    first_linear = next(m for m in model.modules() if isinstance(m, nn.Linear))
    assert scales_large[first_linear] > scales_small[first_linear]
