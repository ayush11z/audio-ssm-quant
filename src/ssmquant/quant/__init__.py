from .fake_quant import compute_scale, fake_quantize, fake_quantize_from_spec
from .apply import ActivationFakeQuantHooks, calibrate_activation_scales, quantize_linear_weights_

__all__ = [
    "compute_scale",
    "fake_quantize",
    "fake_quantize_from_spec",
    "quantize_linear_weights_",
    "calibrate_activation_scales",
    "ActivationFakeQuantHooks",
]
