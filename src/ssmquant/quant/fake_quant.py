"""Fake (simulated) quantization: quantize to an integer grid, then
immediately dequantize back to float, all in PyTorch. Brief section 6:
"All quantization is simulated (fake) quantization... Real integer kernels
and speedups are out of scope." Nothing here ever produces or stores an
actual int8/int4 tensor -- the output is always a float tensor with
quantization error baked into its values.

Only symmetric quantization is used anywhere in this project's configs
(every configs/quant/*.yaml sets symmetric: true), so that's all that's
implemented.
"""
import torch

__all__ = ["compute_scale", "fake_quantize", "fake_quantize_from_spec"]


def compute_scale(tensor: torch.Tensor, bits: int, granularity: str = "per_tensor", dim: int = 0, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric quantization scale: max-abs value in the reduction group,
    divided by qmax. Signed symmetric range is [-qmax, qmax] with
    qmax = 2**(bits-1) - 1 (e.g. [-127, 127] for int8) -- sacrificing the
    single extra negative value that an asymmetric range would keep, in
    exchange for a scale that maps exactly to zero with no bias.

    `per_channel` and `per_token` are the same operation (max-abs over
    every dim except one you keep) -- they only differ in which dim is
    kept, by convention: the output-channel dim for weights, the
    token/sequence dim for activations. `dim` is that kept dim.
    """
    qmax = 2 ** (bits - 1) - 1
    if granularity == "per_tensor":
        amax = tensor.detach().abs().max()
    elif granularity in ("per_channel", "per_token"):
        # Normalize a negative dim (e.g. -2, used for activations whose
        # rank varies by call site -- AST's are 3D (batch,seq,hidden), but
        # AuM's Mamba blocks reshape to 2D (batch*seq,hidden) before some
        # Linear calls) to its positive equivalent BEFORE building
        # reduce_dims. Comparing a negative dim against range(tensor.dim())
        # (always non-negative) would never match, silently reducing over
        # every dim instead of keeping one -- caught in Phase 3 before any
        # GPU run, see DECISIONS.md.
        norm_dim = dim if dim >= 0 else tensor.dim() + dim
        reduce_dims = [d for d in range(tensor.dim()) if d != norm_dim]
        amax = tensor.detach().abs().amax(dim=reduce_dims, keepdim=True) if reduce_dims else tensor.detach().abs()
    else:
        raise ValueError(f"unknown granularity: {granularity!r}")
    return amax.clamp(min=eps) / qmax


def fake_quantize(
    tensor: torch.Tensor,
    bits: int | None,
    granularity: str = "per_tensor",
    dim: int = 0,
    symmetric: bool = True,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Quantize-dequantize `tensor` to `bits`-bit signed integers.

    bits=None is the 0-bit-change passthrough (brief Phase 2): returns
    `tensor` completely unchanged -- not numerically close, bit-identical
    -- used for the full-precision baseline / unquantized tensors in a
    mixed config (e.g. W8A16's activations).

    If `scale` is given, it's used as-is (for calibration: compute a scale
    once from a calibration set via compute_scale, then reuse it across
    eval batches instead of recomputing per-batch).
    """
    if bits is None:
        return tensor
    if not symmetric:
        raise NotImplementedError("only symmetric quantization is used in this project")

    if scale is None:
        scale = compute_scale(tensor, bits, granularity=granularity, dim=dim)

    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax

    q = torch.round(tensor / scale)
    q = torch.clamp(q, qmin, qmax)
    return q * scale


def fake_quantize_from_spec(tensor: torch.Tensor, spec: dict, dim: int = 0) -> torch.Tensor:
    """Apply fake_quantize using a `weights:`/`activations:` block straight
    from configs/quant/*.yaml (e.g. {"bits": 8, "granularity": "per_channel",
    "symmetric": true})."""
    bits = spec.get("bits")
    if bits is None:
        return tensor
    return fake_quantize(
        tensor,
        bits,
        granularity=spec.get("granularity", "per_tensor"),
        dim=dim,
        symmetric=spec.get("symmetric", True),
    )
