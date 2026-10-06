"""Apply Phase 2's fake_quantize to a real model's nn.Linear layers --
Phase 3's "standard quantization grid" (brief section 6/Phase 3).

Deliberately targets nn.Linear only, not the SSM-internal tensors
(Delta, A/Ab, B, C, h). That split is the point: brief hypothesis H2 asks
whether degradation is driven by quantizing "ordinary linear-layer
weights" (this module, Phase 3) vs. "SSM-specific tensors" (Phase 4's
separate ablations, which hook the reference scan validated in Phase 2).

Weight quantization is static and needs no calibration data -- a weight
tensor's own per-channel max-abs is all `compute_scale` needs, computed
once before eval and left fixed. Activation quantization is different:
- per_tensor: STATIC, calibrated once from a held-out calibration set
  (brief section 6: 256 in-distribution clips, 3 calibration seeds) and
  then reused unchanged for every eval batch -- this is what makes the
  "3 calibration seeds" meaningful: different seeds draw different
  calibration subsets, and we check how much that changes the scale.
- per_token: DYNAMIC, computed fresh per forward pass. A per-token scale
  is already as tight a fit as a single calibration run could give (every
  real token's own magnitude, every time), so there is nothing a fixed
  calibration scale would improve, and no way to "calibrate" a scale per
  token position when different inputs have different lengths anyway.
"""
import torch
import torch.nn as nn

from .fake_quant import compute_scale, fake_quantize, fake_quantize_from_spec

__all__ = [
    "quantize_linear_weights_",
    "calibrate_activation_scales",
    "ActivationFakeQuantHooks",
]


def quantize_linear_weights_(model: nn.Module, spec: dict) -> int:
    """In-place: fake-quantize every nn.Linear's .weight (per_channel
    over dim=0, the output-channel dim). spec={"bits":..,"granularity":..,
    "symmetric":..} or {"bits": None} for a no-op. Returns the number of
    Linear layers touched."""
    bits = spec.get("bits")
    if bits is None:
        return 0
    n = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            with torch.no_grad():
                module.weight.data = fake_quantize_from_spec(module.weight.data, spec, dim=0)
            n += 1
    return n


@torch.no_grad()
def calibrate_activation_scales(model: nn.Module, calibration_inputs, bits: int, forward_fn=None) -> dict:
    """Run `model` over `calibration_inputs` (an iterable of model inputs,
    e.g. 256 in-distribution clips per brief section 6), recording the
    max-abs value seen at the INPUT of every nn.Linear, and return
    {module: scale} (per-tensor symmetric scale, brief's W8A8-per-tensor
    variant -- per_token is dynamic, see module docstring, so has no
    calibration step).

    forward_fn(model, x) -> None, defaults to model(x); override for
    models whose forward signature needs more than one positional input.
    """
    forward_fn = forward_fn or (lambda m, x: m(x))
    running_amax = {}
    handles = []

    def make_hook(module):
        def hook(mod, inputs):
            x = inputs[0]
            amax = x.detach().abs().max()
            prev = running_amax.get(module)
            running_amax[module] = amax if prev is None else torch.maximum(prev, amax)
        return hook

    for module in model.modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(module)))

    try:
        for x in calibration_inputs:
            forward_fn(model, x)
    finally:
        for h in handles:
            h.remove()

    qmax = 2 ** (bits - 1) - 1
    eps = 1e-8
    return {module: amax.clamp(min=eps) / qmax for module, amax in running_amax.items()}


class ActivationFakeQuantHooks:
    """Context manager / explicit attach-detach wrapper that fake-quantizes
    every nn.Linear's input activation on the fly during forward passes.

    spec={"bits":.., "granularity": "per_tensor"|"per_token", "symmetric":..}.
    For "per_tensor", pass `calibration_scales` (from calibrate_activation_scales)
    -- required, since per_tensor is static by design (see module docstring).
    For "per_token", calibration_scales is ignored; the scale is computed
    fresh per forward call from the actual input, dim=-2 (the sequence/
    token dim for a (batch, seq, hidden) activation).
    """

    def __init__(self, model: nn.Module, spec: dict, calibration_scales: dict | None = None):
        self.model = model
        self.spec = spec
        self.calibration_scales = calibration_scales
        self.handles = []

    def _make_hook(self, module):
        bits = self.spec.get("bits")
        granularity = self.spec.get("granularity", "per_tensor")

        def hook(mod, inputs):
            x = inputs[0]
            if bits is None:
                return None
            if granularity == "per_tensor":
                if self.calibration_scales is None or module not in self.calibration_scales:
                    raise ValueError(
                        "per_tensor activation quantization requires calibration_scales "
                        "(brief section 6: calibrated from a held-out clip set, not computed on the fly)"
                    )
                scale = self.calibration_scales[module]
                q = fake_quantize(x, bits, granularity="per_tensor", symmetric=True, scale=scale)
            elif granularity == "per_token":
                # (batch, seq, hidden) -> keep the seq dim, reduce over hidden
                q = fake_quantize(x, bits, granularity="per_token", dim=-2, symmetric=True)
            else:
                raise ValueError(f"unsupported activation granularity: {granularity!r}")
            return (q,) + inputs[1:]

        return hook

    def __enter__(self):
        bits = self.spec.get("bits")
        if bits is None:
            return self
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_pre_hook(self._make_hook(module)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        return False
