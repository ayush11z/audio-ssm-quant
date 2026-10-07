"""Phase 3: a quantization-aware reimplementation of AuM's bimamba_type='v1'
Mamba block forward pass (the "Fo-Bi" variant AuM's checkpoints use).

WHY THIS EXISTS: AuM's actual forward pass (third_party/Audio-Mamba-AuM/
vim-mamba_ssm/mamba_ssm/modules/mamba_simple.py) calls in_proj/x_proj/
dt_proj/out_proj via raw `.weight` matmuls and a single fused autograd
Function (BiMambaInnerFn), never through nn.Linear.forward(). A
forward_pre_hook-based approach (src/ssmquant/quant/apply.py,
ActivationFakeQuantHooks) silently never fires on any of them -- no error,
just activations that are never quantized while looking like they are.
Caught by reading BiMambaInnerFn.forward() directly (that file is a ViM
addition, not in upstream state-spaces/mamba) before running anything on
GPU -- see DECISIONS.md.

This module reimplements that exact algorithm (verified line-by-line
against BiMambaInnerFn.forward(), including the Phase 3 gate's D_b bug
fix -- see DECISIONS.md) in plain PyTorch, with a quantization hook
inserted at the four real activation entry points (the INPUT to each of
in_proj/x_proj/dt_proj/out_proj). Weight quantization still uses
quantize_linear_weights_ as-is (it modifies .weight.data directly,
independent of how forward() later reads it, so it already worked
correctly even with the fused path -- only activation quantization needed
this). The scan itself still uses the real selective_scan_fn (the fused
kernel, validated against the reference scan in Phase 2) -- Phase 3
quantizes ordinary linear layers only; Delta/A/Ab/h are Phase 4's separate
ablation target (brief hypothesis H2).

The brief's W8A8 grid has two activation granularities with different
calibration requirements (brief section 6, mirrored in apply.py):
- per_tensor: STATIC, a scale calibrated once from a held-out clip set and
  reused across eval -- this is what the "3 calibration seeds" are for.
- per_token: DYNAMIC, computed fresh per forward call (already as tight a
  fit as calibration could give).
For AST this split is handled by apply.py's calibrate_activation_scales +
ActivationFakeQuantHooks, both forward_pre_hook-based. Since AuM's Mamba
internals bypass hooks entirely (the reason this module exists),
per_tensor calibration needs its own path here: calibrate_bimamba_v1_scales
runs the REAL (unpatched) algorithm over calibration clips and records
max-abs at the same four entry points quantized_bimamba_v1_forward will
later quantize, keyed per-module (each layer's Mamba block gets its own
scale, same as one scale per nn.Linear in apply.py). Both functions share
_bimamba_v1_forward_core so the quantized and calibration-recording paths
can't silently drift apart from each other or from the real algorithm.

Only supports bimamba_type='v1' (what AuM's released checkpoints use) with
causal_conv1d available. Not a general-purpose replacement for
Mamba.forward() -- no inference_params/caching support, no 'none'/'v2'
bimamba_type, no init_layer_scale handling beyond what AuM's base config
uses.
"""
import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from einops import rearrange
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from ssmquant.quant.fake_quant import fake_quantize

A_NONE = {"bits": None}
SLOT_NAMES = ("in_proj", "x_proj", "dt_proj", "out_proj")


def _bimamba_v1_forward_core(mamba, hidden_states, q):
    """The actual bimamba_type='v1' algorithm (verified against
    BiMambaInnerFn.forward()), parameterized by `q(x, dim, slot) -> tensor`
    applied at each of the four real activation entry points. `q` is the
    only thing that varies between the quantized and calibration-recording
    callers below.
    """
    batch, seqlen, _ = hidden_states.shape

    # ---- in_proj: input is hidden_states (batch, seqlen, d_model) ----
    # dim=-2 keeps the seqlen ("token") axis for per_token granularity.
    hs_q = q(hidden_states, dim=-2, slot="in_proj")
    xz = rearrange(
        mamba.in_proj.weight @ rearrange(hs_q, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )
    if mamba.in_proj.bias is not None:
        xz = xz + rearrange(mamba.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    A = -torch.exp(mamba.A_log.float())
    A_b = -torch.exp(mamba.A_b_log.float())
    # bimamba_type='v1' shares a single D, conv1d, x_proj, dt_proj, out_proj
    # between the forward and backward scans -- only A differs (A vs A_b).
    # D_b/conv1d_b/x_proj_b/dt_proj_b only exist for bimamba_type='v2'
    # (confirmed against BiMambaInnerFn.forward(), which passes the same D
    # to both selective_scan_cuda.fwd calls) -- using a separate mamba.D_b
    # here was the Phase 3 gate's first failure (AttributeError).

    x, z = xz.chunk(2, dim=1)  # each (batch, d_inner, seqlen)

    conv1d_out = causal_conv1d_fn(
        x=x,
        weight=rearrange(mamba.conv1d.weight, "d 1 w -> d w"),
        bias=mamba.conv1d.bias,
        activation=mamba.activation,
    )

    # ---- x_proj: input is conv1d_out (batch, d_inner, seqlen) ----
    # seqlen is the LAST dim here (channel-first layout), so dim=-1 keeps
    # the token axis -- NOT dim=-2 like the (batch, seqlen, d_model)
    # convention above. Getting this wrong per call site is exactly the
    # negative-dim bug class fixed in fake_quant.py; each activation here
    # is checked against its own actual shape, not assumed uniform.
    conv1d_out_q = q(conv1d_out, dim=-1, slot="x_proj")
    x_dbl = F.linear(rearrange(conv1d_out_q, "b d l -> (b l) d"), mamba.x_proj.weight)  # (bl, dt_rank+2*d_state)

    delta_rank = mamba.dt_proj.weight.shape[1]
    d_state = A.shape[-1]

    # ---- dt_proj: input is the dt-slice of x_dbl, (batch*seqlen, delta_rank) ----
    # 2D, time folded into dim 0 alongside batch -- dim=-2 == dim 0 here,
    # one scale per actual (batch, timestep) row, consistent with "per
    # token" everywhere else.
    dt_in = x_dbl[:, :delta_rank]
    dt_in_q = q(dt_in, dim=-2, slot="dt_proj")
    delta = rearrange(mamba.dt_proj.weight @ dt_in_q.t(), "d (b l) -> b d l", l=seqlen)

    B = x_dbl[:, delta_rank:delta_rank + d_state]
    C = x_dbl[:, delta_rank + d_state:]
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

    # ---- forward + backward scans (full precision -- Phase 4's territory) ----
    out_z_f = selective_scan_fn(
        conv1d_out, delta, A, B, C, mamba.D.float(), z=z,
        delta_bias=mamba.dt_proj.bias.float(), delta_softplus=True,
    )
    out_z_b = selective_scan_fn(
        conv1d_out.flip([-1]), delta.flip([-1]), A_b, B.flip([-1]), C.flip([-1]), mamba.D.float(),
        z=z.flip([-1]), delta_bias=mamba.dt_proj.bias.float(), delta_softplus=True,
    )
    out_z = out_z_f + out_z_b.flip([-1])  # (batch, d_inner, seqlen)

    # ---- out_proj: input is the combined scan output ----
    out_z_q = q(out_z, dim=-1, slot="out_proj")  # same channel-first layout as conv1d_out
    out = F.linear(rearrange(out_z_q, "b d l -> b l d"), mamba.out_proj.weight, mamba.out_proj.bias)

    if mamba.init_layer_scale is not None:
        out = out * mamba.gamma
    return out


def quantized_bimamba_v1_forward(mamba, hidden_states, weight_spec=None, activation_spec=None, calibration_scales=None):
    """mamba: an AuM Mamba module (bimamba_type='v1') with its Linear
    weights already quantized in-place via quantize_linear_weights_, if
    desired -- this function only handles ACTIVATION quantization; weight
    quantization must be applied by the caller beforehand (same split as
    the hook-based path, for consistency).

    hidden_states: (batch, seqlen, d_model), the block's actual input.
    activation_spec: a configs/quant/*.yaml `activations:` block, or None/
    {"bits": None} for no activation quantization (the W8A16/W4A16 case).
    calibration_scales: {slot_name: scale} for THIS module (from
    calibrate_bimamba_v1_scales), required when activation_spec's
    granularity is "per_tensor" -- per_tensor is static by design (see
    module docstring), so there is no on-the-fly fallback, matching
    ActivationFakeQuantHooks' behavior for AST.
    """
    activation_spec = activation_spec or A_NONE
    calibration_scales = calibration_scales or {}
    bits = activation_spec.get("bits")
    granularity = activation_spec.get("granularity", "per_tensor")
    symmetric = activation_spec.get("symmetric", True)

    def q(x, dim, slot):
        if bits is None:
            return x
        if granularity == "per_tensor":
            scale = calibration_scales.get(slot)
            if scale is None:
                raise ValueError(
                    "per_tensor activation quantization requires a calibrated "
                    f"scale for slot {slot!r} (brief section 6: calibrated from "
                    "a held-out clip set, not computed on the fly) -- see "
                    "calibrate_bimamba_v1_scales"
                )
            return fake_quantize(x, bits, granularity="per_tensor", symmetric=symmetric, scale=scale)
        elif granularity == "per_token":
            return fake_quantize(x, bits, granularity="per_token", dim=dim, symmetric=symmetric)
        else:
            raise ValueError(f"unsupported activation granularity: {granularity!r}")

    return _bimamba_v1_forward_core(mamba, hidden_states, q)


def calibrate_bimamba_v1_forward(mamba, hidden_states, running_amax):
    """Runs the REAL (full-precision) algorithm, recording max-abs at each
    of the four activation entry points into `running_amax` (a dict mutated
    in place, keyed by slot name) instead of quantizing. Returns the real
    output so a model can be calibrated over many clips without the
    calibration pass itself corrupting anything."""

    def record(x, dim, slot):
        amax = x.detach().abs().max()
        prev = running_amax.get(slot)
        running_amax[slot] = amax if prev is None else torch.maximum(prev, amax)
        return x

    return _bimamba_v1_forward_core(mamba, hidden_states, record)


def _iter_bimamba_v1_modules(model):
    for module in model.modules():
        if type(module).__name__ == "Mamba" and hasattr(module, "A_b_log"):
            yield module


@torch.no_grad()
def calibrate_bimamba_v1_scales(model, calibration_inputs, bits, forward_fn=None):
    """The AuM analogue of apply.py's calibrate_activation_scales: runs
    `model` over `calibration_inputs` (model-level inputs, e.g. fbanks),
    temporarily patching every bimamba_type='v1' Mamba block to record
    max-abs at its four real activation entry points (since those never
    reach a real nn.Linear.forward() call for a forward_pre_hook to
    observe -- the same bypass this module exists to work around).
    Returns {mixer_module: {slot_name: scale}}, consumed by
    patch_model_for_quantized_forward's calibration_scales argument.

    forward_fn(model, x) -> None, defaults to model(x); override for
    models whose forward signature needs more than one positional input.
    """
    forward_fn = forward_fn or (lambda m, x: m(x))
    running_amax_by_module = {}
    patched = []

    for mixer in _iter_bimamba_v1_modules(model):
        running_amax = {}
        running_amax_by_module[mixer] = running_amax
        original_forward = mixer.forward

        def make_recorder(mx, amax_dict):
            def recorder_forward(hidden_states, inference_params=None):
                assert inference_params is None, (
                    "calibrate_bimamba_v1_forward does not support cached/"
                    "incremental decoding (inference_params)"
                )
                return calibrate_bimamba_v1_forward(mx, hidden_states, amax_dict)
            return recorder_forward

        mixer.forward = make_recorder(mixer, running_amax)
        patched.append((mixer, original_forward))

    try:
        for x in calibration_inputs:
            forward_fn(model, x)
    finally:
        for mixer, original_forward in patched:
            mixer.forward = original_forward

    qmax = 2 ** (bits - 1) - 1
    eps = 1e-8
    return {
        mixer: {slot: amax.clamp(min=eps) / qmax for slot, amax in amax_dict.items()}
        for mixer, amax_dict in running_amax_by_module.items()
    }


def calibrate_bimamba_v1_scales_by_index(model, calibration_inputs, bits, forward_fn=None):
    """Same as calibrate_bimamba_v1_scales, but keyed by each Mamba
    block's position (0, 1, 2, ...) instead of module identity.

    WHY: AuM's own architecture (FlexiPatchEmbed/FlexiPosEmbed) requires a
    freshly-constructed model per input length (see scripts/
    phase1_aum_eval.py) -- so the model calibrated here and the model(s)
    quantization is later applied to are different Python objects, even
    though they're the same architecture loaded from the same checkpoint.
    A dict keyed by module identity from calibrate_bimamba_v1_scales can't
    be looked up on a different model's modules; a list keyed by position
    can be, via apply_bimamba_v1_scales_by_index, as long as both models
    enumerate their Mamba blocks in the same order (true here: same
    sequential layer stack every time)."""
    by_module = calibrate_bimamba_v1_scales(model, calibration_inputs, bits, forward_fn=forward_fn)
    modules_in_order = list(_iter_bimamba_v1_modules(model))
    return [by_module[m] for m in modules_in_order]


def apply_bimamba_v1_scales_by_index(model, scales_by_index):
    """Inverse of calibrate_bimamba_v1_scales_by_index: re-key a list of
    per-layer scale dicts onto `model`'s own Mamba modules by position,
    for use as patch_model_for_quantized_forward's calibration_scales."""
    modules_in_order = list(_iter_bimamba_v1_modules(model))
    if len(modules_in_order) != len(scales_by_index):
        raise ValueError(
            f"model has {len(modules_in_order)} bimamba_type='v1' Mamba blocks "
            f"but scales_by_index has {len(scales_by_index)} entries -- "
            "architecture mismatch between the calibration model and this one"
        )
    return dict(zip(modules_in_order, scales_by_index))


def patch_model_for_quantized_forward(model, weight_spec, activation_spec, calibration_scales=None):
    """Monkey-patch every AuM Block's mixer.forward with
    quantized_bimamba_v1_forward, bound with the given specs. Returns a
    list of (mixer_module, original_forward) so the caller can restore
    them afterward -- these are real nn.Module instances shared across the
    whole model, so patches must be undone, not left dangling.

    Weight quantization (quantize_linear_weights_) should be applied to
    `model` separately BEFORE calling this, in the same order the
    hook-based path uses -- this function only swaps the forward
    function; it never touches .weight.data itself.

    calibration_scales: {mixer_module: {slot_name: scale}}, from
    calibrate_bimamba_v1_scales -- required when activation_spec's
    granularity is "per_tensor".
    """
    calibration_scales = calibration_scales or {}
    patched = []
    for mixer in _iter_bimamba_v1_modules(model):
        original_forward = mixer.forward
        mixer_scales = calibration_scales.get(mixer)

        def make_patched(mx, scales):
            def patched_forward(hidden_states, inference_params=None):
                assert inference_params is None, (
                    "quantized_bimamba_v1_forward does not support cached/"
                    "incremental decoding (inference_params)"
                )
                return quantized_bimamba_v1_forward(mx, hidden_states, weight_spec, activation_spec, scales)
            return patched_forward

        mixer.forward = make_patched(mixer, mixer_scales)
        patched.append((mixer, original_forward))
    return patched


def unpatch_model(patched):
    for module, original_forward in patched:
        module.forward = original_forward
