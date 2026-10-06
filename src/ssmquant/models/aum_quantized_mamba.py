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

This function reimplements that exact algorithm (verified line-by-line
against BiMambaInnerFn.forward()) in plain PyTorch, with fake_quantize
calls inserted at the four real activation entry points (the INPUT to each
of in_proj/x_proj/dt_proj/out_proj). Weight quantization still uses
quantize_linear_weights_ as-is (it modifies .weight.data directly,
independent of how forward() later reads it, so it already worked
correctly even with the fused path -- only activation quantization needed
this). The scan itself still uses the real selective_scan_fn (the fused
kernel, validated against the reference scan in Phase 2) -- Phase 3
quantizes ordinary linear layers only; Delta/A/Ab/h are Phase 4's separate
ablation target (brief hypothesis H2).

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

from ssmquant.quant.fake_quant import fake_quantize_from_spec

A_NONE = {"bits": None}


def quantized_bimamba_v1_forward(mamba, hidden_states, weight_spec=None, activation_spec=None):
    """mamba: an AuM Mamba module (bimamba_type='v1') with its Linear
    weights already quantized in-place via quantize_linear_weights_, if
    desired -- this function only handles ACTIVATION quantization; weight
    quantization must be applied by the caller beforehand (same split as
    the hook-based path, for consistency).

    hidden_states: (batch, seqlen, d_model), the block's actual input.
    activation_spec: a configs/quant/*.yaml `activations:` block, or None/
    {"bits": None} for no activation quantization (the W8A16/W4A16 case).
    """
    activation_spec = activation_spec or A_NONE
    batch, seqlen, _ = hidden_states.shape

    def q(x, dim):
        if activation_spec.get("bits") is None:
            return x
        return fake_quantize_from_spec(x, activation_spec, dim=dim)

    # ---- in_proj: input is hidden_states (batch, seqlen, d_model) ----
    # dim=-2 keeps the seqlen ("token") axis for per_token granularity;
    # per_tensor ignores dim entirely (see fake_quant.py).
    hs_q = q(hidden_states, dim=-2)
    xz = rearrange(
        mamba.in_proj.weight @ rearrange(hs_q, "b l d -> d (b l)"),
        "d (b l) -> b d l",
        l=seqlen,
    )
    if mamba.in_proj.bias is not None:
        xz = xz + rearrange(mamba.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

    A = -torch.exp(mamba.A_log.float())
    A_b = -torch.exp(mamba.A_b_log.float())

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
    conv1d_out_q = q(conv1d_out, dim=-1)
    x_dbl = F.linear(rearrange(conv1d_out_q, "b d l -> (b l) d"), mamba.x_proj.weight)  # (bl, dt_rank+2*d_state)

    delta_rank = mamba.dt_proj.weight.shape[1]
    d_state = A.shape[-1]

    # ---- dt_proj: input is the dt-slice of x_dbl, (batch*seqlen, delta_rank) ----
    # 2D, time folded into dim 0 alongside batch -- dim=-2 == dim 0 here,
    # one scale per actual (batch, timestep) row, consistent with "per
    # token" everywhere else.
    dt_in = x_dbl[:, :delta_rank]
    dt_in_q = q(dt_in, dim=-2)
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
        conv1d_out.flip([-1]), delta.flip([-1]), A_b, B.flip([-1]), C.flip([-1]), mamba.D_b.float(),
        z=z.flip([-1]), delta_bias=mamba.dt_proj.bias.float(), delta_softplus=True,
    )
    out_z = out_z_f + out_z_b.flip([-1])  # (batch, d_inner, seqlen)

    # ---- out_proj: input is the combined scan output ----
    out_z_q = q(out_z, dim=-1)  # same channel-first layout as conv1d_out
    out = F.linear(rearrange(out_z_q, "b d l -> b l d"), mamba.out_proj.weight, mamba.out_proj.bias)

    if mamba.init_layer_scale is not None:
        out = out * mamba.gamma
    return out


def patch_model_for_quantized_forward(model, weight_spec, activation_spec):
    """Monkey-patch every AuM Block's mixer.forward with
    quantized_bimamba_v1_forward, bound with the given specs. Returns a
    list of (mixer_module, original_forward) so the caller can restore
    them afterward -- these are real nn.Module instances shared across the
    whole model, so patches must be undone, not left dangling.

    Weight quantization (quantize_linear_weights_) should be applied to
    `model` separately BEFORE calling this, in the same order the
    hook-based path uses -- this function only swaps the forward
    function; it never touches .weight.data itself.
    """
    import functools

    patched = []
    for module in model.modules():
        if type(module).__name__ == "Mamba" and hasattr(module, "A_b_log"):
            original_forward = module.forward

            def make_patched(mixer):
                def patched_forward(hidden_states, inference_params=None):
                    assert inference_params is None, (
                        "quantized_bimamba_v1_forward does not support cached/"
                        "incremental decoding (inference_params)"
                    )
                    return quantized_bimamba_v1_forward(mixer, hidden_states, weight_spec, activation_spec)
                return patched_forward

            module.forward = make_patched(module)
            patched.append((module, original_forward))
    return patched


def unpatch_model(patched):
    for module, original_forward in patched:
        module.forward = original_forward
