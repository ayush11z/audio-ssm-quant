"""Phase 4: SSM-internal quantization ablations for AuM's bimamba_type='v1'
scan (brief section 7's "which internal tensor -- Δ, A/Ā, B, C, recurrent
state h -- drives degradation?").

WHY THIS EXISTS: Phase 3 quantized ordinary Linear-layer weights/
activations and found AuM's top-1 accuracy already at chance from 20s
onward in full precision -- there was no accuracy headroom left for
Linear-layer quantization to damage further (see DECISIONS.md's floor-
effect discussion). This module targets a different, narrower question:
quantizing the SSM's own internal tensors in isolation (Linear layers
stay full precision throughout this module), one tensor at a time, plus
per-timestep state-divergence logging -- a metric that keeps working
even where top-1 accuracy has saturated at the floor.

The fused CUDA kernel (selective_scan_fn, used everywhere in Phase 3)
can't be hooked mid-scan to quantize Δ/A/B/C/h at the point they're
used -- same bypass problem that motivated aum_quantized_mamba.py, one
level deeper. `selective_scan_ref` (plain Python, validated against the
fused kernel in Phase 2: max abs diff 3-5e-5 at both native and extended
lengths) CAN be hooked, since it's an explicit per-timestep for-loop.
`_scan_one_direction` below is that loop, generalized to optionally
fake-quantize exactly one of {delta, A, BC, state} at the point it's
used, and optionally track how far the resulting hidden-state trajectory
drifts from the unquantized one at every timestep.

Ablation targets (matching configs/quant/ssm_{delta,A,BC,state}.yaml,
pre-existing config stubs, not written as part of this module -- this
module's `target` strings are exactly those configs' `target` fields):
  - "delta": Δ (post-softplus, pre-bias-add is applied before softplus).
    Shared between the forward and backward scan (the real algorithm
    computes Δ once and only flips it for the backward direction) --
    quantized once, same quantized value flows into both.
  - "A": the raw parameter A = -exp(A_log) (and A_b for the backward
    direction) -- a static per-layer parameter, identical for every
    clip, so (unlike the other three targets) computing its scale needs
    no calibration clips at all, the same situation as Linear weight
    quantization in apply.py. Quantized independently per direction
    (A and A_b are different parameters).
  - "BC": B and C together (shared between directions, matching the
    real algorithm -- B/C are computed once and only flipped for the
    backward pass).
  - "state": the recurrent state h, quantized at every timestep inside
    the loop -- independently for the forward and backward directions'
    trajectories, since those are two separate state sequences. This is
    the slow one (brief/configs' own warning): a genuine per-timestep
    Python loop, not a single fused kernel call.

Weight quantization is deliberately absent here -- Phase 4 runs with
full-precision in_proj/x_proj/dt_proj/out_proj throughout, to isolate the
SSM-internal tensors as the only quantized thing in the model (matches
the brief's H1/H2 split: Phase 3 = ordinary Linears, Phase 4 = SSM-
specific tensors, not both at once).
"""
import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn
from einops import rearrange

from ssmquant.quant.fake_quant import fake_quantize

from .aum_quantized_mamba import _iter_bimamba_v1_modules

TARGETS = ("delta", "A", "BC", "state")


def _scan_one_direction(u, delta_raw, A, B, C, D, z, delta_bias, delta_softplus,
                         target, bits, scale, track_divergence):
    """Faithful port of selective_scan_ref's single-direction computation
    (third_party/Audio-Mamba-AuM/vim-mamba_ssm/mamba_ssm/ops/
    selective_scan_interface.py:86), generalized to optionally fake-
    quantize exactly one of {delta, A, BC, state} at the point it's used.

    u: (batch, dim, seqlen) -- conv1d_out, the scan's actual input
    delta_raw: (batch, dim, seqlen) -- PRE-softplus, pre-bias (dt_proj's
        raw output); softplus+bias are applied here, matching the real
        algorithm, so `target == "delta"` quantizes the same tensor the
        real scan actually uses.
    A: (dim, dstate) -- already -exp(A_log) for this direction
    B, C: (batch, dstate, seqlen)
    D: (dim,); z: (batch, dim, seqlen); delta_bias: (dim,)
    target: one of TARGETS, or None (no quantization -- the gate's
        baseline and Phase 4's own full-precision reference use this)
    bits: int or None (None means this call is a passthrough regardless
        of `target`, matching fake_quantize's own None convention)
    scale: meaning depends on target -- scalar for "delta"/"A", a
        (scale_B, scale_C) pair for "BC", scalar for "state" (this
        direction's own state scale). Ignored when bits is None.
    track_divergence: if True, also runs the UNQUANTIZED recurrence in
        parallel (same Δ/A/B/C inputs, before any quantization applied
        above) and records ||h_t - h_t_fullprecision|| at every
        timestep -- this is deliberately the true full-precision
        reference, not just "state quantization off", so it correctly
        measures drift for every target, not only "state".

    Returns (out_z, divergence_trace_or_None). divergence_trace is a
    list of per-timestep floats (one scan direction's worth) when
    requested, else None.
    """
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    quantizing = bits is not None

    def q(x, s, dim_label):
        return fake_quantize(x, bits, granularity="per_tensor", scale=s) if quantizing else x

    delta = delta_raw.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    A_eff = q(A, scale, "A") if target == "A" else A
    delta_eff = q(delta, scale, "delta") if target == "delta" else delta
    if target == "BC":
        B_eff = q(B, scale[0], "B")
        C_eff = q(C, scale[1], "C")
    else:
        B_eff, C_eff = B, C

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta_eff, A_eff))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta_eff, B_eff, u.float())

    if track_divergence:
        # The TRUE full-precision reference: recomputed from the original
        # (never-quantized) delta/A/B/C, independent of `target` -- this
        # is what every ablation's divergence is measured against, not
        # just a same-target-but-unquantized variant.
        deltaA_fp = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        deltaB_u_fp = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u.float())
        x_fp = A.new_zeros((batch, dim, dstate))
        divergence = []
    else:
        divergence = None

    x = A.new_zeros((batch, dim, dstate))
    ys = []
    for i in range(seqlen):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if target == "state" and quantizing:
            x = fake_quantize(x, bits, granularity="per_tensor", scale=scale)
        y = torch.einsum('bdn,bn->bd', x, C_eff[:, :, i])
        ys.append(y)
        if track_divergence:
            x_fp = deltaA_fp[:, :, i] * x_fp + deltaB_u_fp[:, :, i]
            divergence.append((x - x_fp).norm().item())

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u.float() * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    return out, divergence


def quantized_bimamba_v1_scan_forward(mamba, hidden_states, target=None, bits=None,
                                       scale=None, track_divergence=False):
    """Bidirectional wrapper: same algorithm as aum_quantized_mamba.py's
    _bimamba_v1_forward_core (in_proj -> conv1d -> x_proj -> dt_proj ->
    scan -> out_proj), but the scan runs through _scan_one_direction
    (a Python step loop) instead of the fused selective_scan_fn, so
    Δ/A/B,C/h can be quantized mid-scan. in_proj/x_proj/dt_proj/out_proj
    are never quantized here -- see module docstring.

    target: one of TARGETS, or None (full precision -- used by the
        Phase 4 validation gate to confirm this path matches the real
        fused-kernel forward before any ablation number is trusted).
    scale: {"fwd": ..., "bwd": ...} for "delta"/"A"/"state" (each value
        a scalar; "A" needs two independently-calibrated-or-computed
        scales since A and A_b are different parameters; "delta" and
        "state" also differ by direction since delta is calibrated once
        but used as a shared value -- actually shares one scale, see
        below), or {"fwd": (scale_B, scale_C), "bwd": (scale_B, scale_C)}
        for "BC" (B/C are shared between directions, so fwd/bwd carry
        the same pair -- kept in this shape for a uniform call site).
        Ignored when bits is None.

    Returns (out, divergence_or_None). divergence, when requested, is
    {"fwd": [...], "bwd": [...]} -- one list of per-timestep floats per
    scan direction.
    """
    batch, seqlen, _ = hidden_states.shape
    xz = rearrange(
        mamba.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )
    if mamba.in_proj.bias is not None:
        xz = xz + rearrange(mamba.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    A = -torch.exp(mamba.A_log.float())
    A_b = -torch.exp(mamba.A_b_log.float())

    x, z = xz.chunk(2, dim=1)
    conv1d_out = causal_conv1d_fn(
        x=x,
        weight=rearrange(mamba.conv1d.weight, "d 1 w -> d w"),
        bias=mamba.conv1d.bias,
        activation=mamba.activation,
    )

    x_dbl = F.linear(rearrange(conv1d_out, "b d l -> (b l) d"), mamba.x_proj.weight)
    delta_rank = mamba.dt_proj.weight.shape[1]
    d_state = A.shape[-1]
    delta_raw = rearrange(mamba.dt_proj.weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=seqlen)

    B = x_dbl[:, delta_rank:delta_rank + d_state]
    C = x_dbl[:, delta_rank + d_state:]
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

    scale = scale or {}
    scale_fwd = scale.get("fwd")
    scale_bwd = scale.get("bwd")

    out_z_f, div_f = _scan_one_direction(
        conv1d_out, delta_raw, A, B, C, mamba.D.float(), z,
        mamba.dt_proj.bias.float(), True,
        target=target, bits=bits, scale=scale_fwd, track_divergence=track_divergence,
    )
    out_z_b, div_b = _scan_one_direction(
        conv1d_out.flip([-1]), delta_raw.flip([-1]), A_b, B.flip([-1]), C.flip([-1]), mamba.D.float(), z.flip([-1]),
        mamba.dt_proj.bias.float(), True,
        target=target, bits=bits, scale=scale_bwd, track_divergence=track_divergence,
    )
    out_z = out_z_f + out_z_b.flip([-1])
    out = F.linear(rearrange(out_z, "b d l -> b l d"), mamba.out_proj.weight, mamba.out_proj.bias)

    if mamba.init_layer_scale is not None:
        out = out * mamba.gamma

    divergence = {"fwd": div_f, "bwd": div_b} if track_divergence else None
    return out, divergence


def calibrate_ssm_tensor_scale(model, calibration_inputs, target, bits, forward_fn=None):
    """Compute a static per-layer scale for `target` (one of TARGETS).

    For "A": no calibration data is needed at all -- A/A_b are static
    per-layer parameters, identical for every clip (the same situation
    as Linear weight quantization in apply.py's quantize_linear_weights_,
    which also needs no calibration set). Computed directly from the
    parameters themselves; `calibration_inputs` is accepted but unused
    for this target, kept only for a uniform call signature across all
    four targets.

    For "delta"/"BC"/"state": genuinely input-dependent, calibrated by
    running `calibration_inputs` through a temporarily-patched model that
    records max-abs of the real target tensor at the point it's actually
    used (same recording-forward trick as aum_quantized_mamba.py's
    calibrate_bimamba_v1_forward), then restoring the original forward.

    Returns {mixer: scale_dict} where scale_dict has the same shape
    quantized_bimamba_v1_scan_forward's `scale` argument expects.
    """
    qmax = 2 ** (bits - 1) - 1
    eps = 1e-8

    if target == "A":
        result = {}
        for mixer in _iter_bimamba_v1_modules(model):
            A = -torch.exp(mixer.A_log.float())
            A_b = -torch.exp(mixer.A_b_log.float())
            result[mixer] = {
                "fwd": A.detach().abs().max().clamp(min=eps).item() / qmax,
                "bwd": A_b.detach().abs().max().clamp(min=eps).item() / qmax,
            }
        return result

    forward_fn = forward_fn or (lambda m, x: m(x))
    running = {}
    patched = []

    for mixer in _iter_bimamba_v1_modules(model):
        state = {"fwd": None, "bwd": None}
        running[mixer] = state
        original_forward = mixer.forward

        def make_recorder(mx, st):
            def recorder_forward(hidden_states, inference_params=None):
                assert inference_params is None, (
                    "calibrate_ssm_tensor_scale does not support cached/"
                    "incremental decoding (inference_params)"
                )
                amax = _record_target_amax(mx, hidden_states, target)
                if target == "BC":
                    # amax["fwd"] is a (B_amax, C_amax) pair this clip --
                    # track each element's running max independently, not
                    # a lexicographic tuple max.
                    b_amax, c_amax = amax["fwd"]
                    if st["fwd"] is None:
                        st["fwd"] = (b_amax, c_amax)
                    else:
                        prev_b, prev_c = st["fwd"]
                        st["fwd"] = (max(prev_b, b_amax), max(prev_c, c_amax))
                else:
                    st["fwd"] = amax["fwd"] if st["fwd"] is None else max(st["fwd"], amax["fwd"])
                    st["bwd"] = amax["bwd"] if st["bwd"] is None else max(st["bwd"], amax["bwd"])
                out, _ = quantized_bimamba_v1_scan_forward(mx, hidden_states, target=None, bits=None)
                return out
            return recorder_forward

        mixer.forward = make_recorder(mixer, state)
        patched.append((mixer, original_forward))

    try:
        with torch.no_grad():
            for x in calibration_inputs:
                forward_fn(model, x)
    finally:
        for mixer, original_forward in patched:
            mixer.forward = original_forward

    def to_scale(amax):
        return max(amax, eps) / qmax

    result = {}
    for mixer, state in running.items():
        if target == "BC":
            # state["fwd"]/["bwd"] each hold a (B_amax, C_amax) pair here;
            # B/C are shared between directions in the real algorithm, so
            # fwd and bwd carry the same (scale_B, scale_C) pair.
            b_amax, c_amax = state["fwd"]
            scale_pair = (to_scale(b_amax), to_scale(c_amax))
            result[mixer] = {"fwd": scale_pair, "bwd": scale_pair}
        else:
            result[mixer] = {"fwd": to_scale(state["fwd"]), "bwd": to_scale(state["bwd"])}
    return result


@torch.no_grad()
def _record_target_amax(mamba, hidden_states, target):
    """Run one real (unquantized) forward pass of this mixer's algorithm,
    returning {"fwd": amax, "bwd": amax} (or a (B,C)-amax pair for "BC")
    for `target` -- the max-abs value the real scan actually produces at
    that tensor, for THIS clip. Shares the exact same computation path as
    quantized_bimamba_v1_scan_forward (not a separate reimplementation)
    so calibration can't silently drift from what's actually quantized
    later."""
    batch, seqlen, _ = hidden_states.shape
    xz = rearrange(
        mamba.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )
    if mamba.in_proj.bias is not None:
        xz = xz + rearrange(mamba.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    x, z = xz.chunk(2, dim=1)
    conv1d_out = causal_conv1d_fn(
        x=x,
        weight=rearrange(mamba.conv1d.weight, "d 1 w -> d w"),
        bias=mamba.conv1d.bias,
        activation=mamba.activation,
    )
    x_dbl = F.linear(rearrange(conv1d_out, "b d l -> (b l) d"), mamba.x_proj.weight)
    delta_rank = mamba.dt_proj.weight.shape[1]
    d_state = (-torch.exp(mamba.A_log.float())).shape[-1]
    delta_raw = rearrange(mamba.dt_proj.weight @ x_dbl[:, :delta_rank].t(), "d (b l) -> b d l", l=seqlen)
    B = x_dbl[:, delta_rank:delta_rank + d_state]
    C = x_dbl[:, delta_rank + d_state:]
    B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
    C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

    if target == "delta":
        delta = delta_raw.float() + mamba.dt_proj.bias.float()[..., None]
        delta = F.softplus(delta)
        amax = delta.detach().abs().max().item()
        return {"fwd": amax, "bwd": amax}
    elif target == "BC":
        return {"fwd": (B.detach().abs().max().item(), C.detach().abs().max().item()), "bwd": None}
    elif target == "state":
        A = -torch.exp(mamba.A_log.float())
        A_b = -torch.exp(mamba.A_b_log.float())
        delta = delta_raw.float() + mamba.dt_proj.bias.float()[..., None]
        delta = F.softplus(delta)
        amax_fwd = _state_trajectory_amax(conv1d_out, delta, A, B, C)
        amax_bwd = _state_trajectory_amax(
            conv1d_out.flip([-1]), delta.flip([-1]), A_b, B.flip([-1]), C.flip([-1])
        )
        return {"fwd": amax_fwd, "bwd": amax_bwd}
    else:
        raise ValueError(f"unsupported calibration target: {target!r}")


def _state_trajectory_amax(u, delta, A, B, C):
    """Max-abs of the recurrent state h over every timestep of one scan
    direction, for calibrating the "state" target's scale (the YAML's
    "quantize... at every timestep" note -- one scale, calibrated once
    from the trajectory's own extremes, reused unchanged at every step)."""
    batch, dim, seqlen = u.shape
    dstate = A.shape[1]
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u.float())
    x = A.new_zeros((batch, dim, dstate))
    amax = 0.0
    for i in range(seqlen):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        amax = max(amax, x.detach().abs().max().item())
    return amax


def calibrate_ssm_tensor_scale_by_index(model, calibration_inputs, target, bits, forward_fn=None):
    """Same as calibrate_ssm_tensor_scale, keyed by Mamba-block position
    (0, 1, 2, ...) instead of module identity -- needed because AuM's
    FlexiPatchEmbed/FlexiPosEmbed require a fresh model instance per
    input length (see aum_quantized_mamba.py's by-index helpers for the
    full reasoning; identical situation here)."""
    by_module = calibrate_ssm_tensor_scale(model, calibration_inputs, target, bits, forward_fn=forward_fn)
    modules_in_order = list(_iter_bimamba_v1_modules(model))
    return [by_module[m] for m in modules_in_order]


def apply_ssm_tensor_scale_by_index(model, scales_by_index):
    """Inverse of calibrate_ssm_tensor_scale_by_index: re-key a list of
    per-layer scale dicts onto `model`'s own Mamba modules by position."""
    modules_in_order = list(_iter_bimamba_v1_modules(model))
    if len(modules_in_order) != len(scales_by_index):
        raise ValueError(
            f"model has {len(modules_in_order)} bimamba_type='v1' Mamba blocks "
            f"but scales_by_index has {len(scales_by_index)} entries -- "
            "architecture mismatch between the calibration model and this one"
        )
    return dict(zip(modules_in_order, scales_by_index))


def patch_model_for_ssm_ablation(model, target, bits, calibration_scales=None, track_divergence=False):
    """Monkey-patch every AuM Block's mixer.forward with
    quantized_bimamba_v1_scan_forward, bound to this ablation's target/
    bits/scale. Returns a list of (mixer_module, original_forward); undo
    with aum_quantized_mamba.unpatch_model (that function is generic, no
    need to duplicate it here).

    calibration_scales: {mixer_module: scale_dict} from
    apply_ssm_tensor_scale_by_index (or calibrate_ssm_tensor_scale
    directly, if reusing the same model instance it was calibrated on).
    Required whenever bits is not None.

    track_divergence: if True, the patched forward also returns a
    divergence trace via a side-channel list passed in -- see
    `collect_divergence` below for how callers retrieve it, since
    nn.Module.forward can't change its return signature without
    breaking the rest of AudioMamba.forward's call chain.
    """
    calibration_scales = calibration_scales or {}
    divergence_log = [] if track_divergence else None
    patched = []
    for mixer in _iter_bimamba_v1_modules(model):
        original_forward = mixer.forward
        mixer_scale = calibration_scales.get(mixer)

        def make_patched(mx, sc):
            def patched_forward(hidden_states, inference_params=None):
                assert inference_params is None, (
                    "quantized_bimamba_v1_scan_forward does not support "
                    "cached/incremental decoding (inference_params)"
                )
                out, div = quantized_bimamba_v1_scan_forward(
                    mx, hidden_states, target=target, bits=bits, scale=sc,
                    track_divergence=track_divergence,
                )
                if track_divergence:
                    divergence_log.append(div)
                return out
            return patched_forward

        mixer.forward = make_patched(mixer, mixer_scale)
        patched.append((mixer, original_forward))
    return patched, divergence_log
