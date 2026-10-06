"""Phase 2 gate: unit tests for src/ssmquant/quant/fake_quant.py --
correct rounding/clipping, per-channel/per-tensor/per-token scales, and an
exact 0-bit-change passthrough (brief, Phase 2).
"""
import torch

from ssmquant.quant import compute_scale, fake_quantize, fake_quantize_from_spec


def test_passthrough_is_bit_identical():
    """bits=None must return the exact same tensor, not just numerically
    close -- this is the full-precision baseline every quantized config is
    compared against, so any drift here would silently bias every gap
    measurement in later phases."""
    x = torch.randn(4, 8)
    out = fake_quantize(x, bits=None)
    assert out is x or torch.equal(out, x)


def test_known_values_per_tensor_int8():
    """Hand-computed example: max-abs=4.0, bits=8 -> qmax=127,
    scale=4/127. Values exactly on the integer grid round-trip exactly;
    values off the grid get rounded, not truncated."""
    x = torch.tensor([4.0, -4.0, 0.0, 2.0])
    scale = 4.0 / 127
    out = fake_quantize(x, bits=8, granularity="per_tensor")
    expected = torch.round(x / scale) * scale
    assert torch.allclose(out, expected, atol=1e-6)
    # the max-abs element must round-trip to (near) its original value
    assert torch.allclose(out[0], x[0], atol=scale)


def test_rounding_not_truncation():
    """Values must round to the nearest grid point, not truncate towards
    zero. Use a tensor with headroom (max-abs=10) so the small value isn't
    itself the one setting the scale, then confirm the quantized output
    matches torch.round(x/scale)*scale and NOT torch.trunc(x/scale)*scale
    for a value that isn't already on the grid."""
    x = torch.tensor([3.05, 10.0])  # max-abs=10 -> scale=10/127; 3.05/scale=38.74
    scale = 10.0 / 127
    raw = x[0] / scale
    assert torch.round(raw) != torch.trunc(raw)  # sanity: this value actually exercises rounding
    out = fake_quantize(x, bits=8, granularity="per_tensor")
    expected_rounded = torch.round(raw) * scale
    expected_truncated = torch.trunc(raw) * scale
    assert torch.allclose(out[0], expected_rounded, atol=1e-6)
    assert not torch.allclose(out[0], expected_truncated, atol=1e-6)


def test_clipping_not_wraparound():
    """A calibrated-then-reused scale can see values outside its original
    range at eval time (e.g. a louder clip than the calibration set) --
    those must clip to the representable max, never wrap around to a
    wildly wrong value the way integer overflow would."""
    calib = torch.tensor([1.0, -1.0])
    scale = compute_scale(calib, bits=8, granularity="per_tensor")
    out_of_range = torch.tensor([100.0, -100.0])
    out = fake_quantize(out_of_range, bits=8, scale=scale)
    qmax = 2**7 - 1
    expected_max = qmax * scale
    assert torch.allclose(out[0], expected_max, atol=1e-6)
    assert torch.allclose(out[1], -expected_max, atol=1e-6)
    assert out[0] < out_of_range[0]  # confirms it clipped down, didn't wrap to something larger/negative


def test_per_tensor_scale_is_scalar_shared():
    x = torch.randn(4, 8) * torch.tensor([1.0, 10.0, 1.0, 1.0]).view(4, 1)  # row 1 has much larger magnitude
    scale = compute_scale(x, bits=8, granularity="per_tensor")
    assert scale.dim() == 0
    # same scale applied everywhere means the small rows get coarsely
    # quantized relative to their own magnitude (the whole point
    # per-channel exists to avoid)
    out = fake_quantize(x, bits=8, granularity="per_tensor")
    assert out.shape == x.shape


def test_per_channel_scale_varies_per_row():
    """Per-channel (dim=0, e.g. a Linear layer's output-channel dim) must
    give each row its own scale, so a small-magnitude channel isn't
    crushed by a large-magnitude one sharing a per-tensor scale."""
    x = torch.zeros(3, 8)
    x[0] = 1.0
    x[1] = 10.0
    x[2] = 100.0
    scale = compute_scale(x, bits=8, granularity="per_channel", dim=0)
    assert scale.shape == (3, 1)
    ratios = (scale[1] / scale[0], scale[2] / scale[1])
    assert torch.allclose(ratios[0], torch.tensor(10.0), atol=1e-4)
    assert torch.allclose(ratios[1], torch.tensor(10.0), atol=1e-4)

    out = fake_quantize(x, bits=8, granularity="per_channel", dim=0)
    # each row should round-trip near its own original value, since each
    # has its own scale calibrated to its own magnitude
    for row in range(3):
        assert torch.allclose(out[row], x[row], atol=x[row].abs().max().item() / 127 + 1e-6)


def test_per_token_scale_varies_per_token_position():
    """Per-token (dim=1, e.g. activations shaped (batch, seq, hidden))
    must give each sequence position its own scale, independent of the
    hidden dim it's computed over."""
    x = torch.zeros(2, 5, 16)  # (batch, seq, hidden)
    x[:, 0, :] = 1.0
    x[:, 1, :] = 50.0
    scale = compute_scale(x, bits=8, granularity="per_token", dim=1)
    assert scale.shape == (1, 5, 1)
    assert scale[0, 1, 0] > scale[0, 0, 0]


def test_quantized_values_within_representable_range():
    """The dequantized output must never exceed qmax*scale in magnitude,
    for any bit width, regardless of input magnitude."""
    for bits in (4, 6, 8):
        x = torch.randn(100) * 1000
        out = fake_quantize(x, bits=bits, granularity="per_tensor")
        qmax = 2 ** (bits - 1) - 1
        scale = compute_scale(x, bits=bits, granularity="per_tensor")
        assert out.abs().max() <= qmax * scale + 1e-6


def test_zero_tensor_does_not_divide_by_zero():
    x = torch.zeros(4, 4)
    out = fake_quantize(x, bits=8, granularity="per_channel", dim=0)
    assert torch.all(torch.isfinite(out))
    assert torch.equal(out, x)


def test_fewer_bits_means_coarser_quantization():
    """A monotonicity sanity check: as bits decreases, the quantization
    error (relative to the unquantized tensor) should not decrease."""
    x = torch.randn(256)
    errors = {}
    for bits in (8, 6, 4):
        out = fake_quantize(x, bits=bits, granularity="per_tensor")
        errors[bits] = (out - x).abs().mean().item()
    assert errors[8] <= errors[6] <= errors[4]


def test_fake_quantize_from_spec_matches_direct_call():
    x = torch.randn(4, 8)
    spec = {"bits": 8, "granularity": "per_channel", "symmetric": True}
    out_spec = fake_quantize_from_spec(x, spec, dim=0)
    out_direct = fake_quantize(x, bits=8, granularity="per_channel", dim=0, symmetric=True)
    assert torch.equal(out_spec, out_direct)


def test_fake_quantize_from_spec_null_bits_is_passthrough():
    x = torch.randn(4, 8)
    out = fake_quantize_from_spec(x, {"bits": None})
    assert torch.equal(out, x)


def test_asymmetric_not_implemented():
    """Every configs/quant/*.yaml in this project sets symmetric: true --
    asymmetric quantization is deliberately unimplemented, not silently
    wrong, so this should raise rather than quietly do the wrong thing."""
    x = torch.randn(4)
    try:
        fake_quantize(x, bits=8, symmetric=False)
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
