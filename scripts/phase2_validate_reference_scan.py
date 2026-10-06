#!/usr/bin/env python
"""Phase 2 gate: validate mamba_ssm's pure-PyTorch reference scan
(selective_scan_ref) against its fused CUDA kernel (selective_scan_fn)
within numerical tolerance, at full precision (brief section 7 / Phase 2).

Why this matters: the fused kernel can't be hooked to read out Delta, A/Ā,
or the recurrent state h mid-scan (brief section 7), which Phase 4's
SSM-internal quantization ablations need. The reference scan CAN be
hooked, since it's a plain Python for-loop over timesteps -- but only if
it's numerically equivalent to what the model was actually trained and
evaluated with (the fused kernel). This script is that check, BEFORE any
ablation result from the reference-scan path is trusted.

Shapes/construction match AuM's actual Mamba block (checked directly in
third_party/Audio-Mamba-AuM/vim-mamba_ssm/mamba_ssm/modules/mamba_simple.py):
d_inner=1536 (768 embed_dim x expand=2), d_state=16, A=-exp(A_log),
variable B/C of shape (batch, d_state, seqlen), delta_softplus=True.

Run on a GPU instance with the AuM venv (mamba_ssm==1.1.3.post1 +
causal_conv1d installed, see scripts/phase0_setup_env.sh):
    source ~/aum_venv/bin/activate
    python3 scripts/phase2_validate_reference_scan.py
"""
import json
from pathlib import Path

import torch
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "phase2_reference_scan_validation.json"

# AuM's actual base-model Mamba block dimensions.
D_INNER = 1536
D_STATE = 16
BATCH = 2
# Test across both an in-distribution length (native Speech Commands, ~128
# mel frames) and an extended length (matching Phase 1's 20s condition),
# since Phase 4's ablations need this equivalence to hold at every length
# Phase 1 touched, not just the training length.
SEQ_LENS = {"native_128": 128, "extended_1998": 1998}


def make_inputs(seqlen, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.randn(BATCH, D_INNER, seqlen, generator=g).to(device)
    # pre-softplus delta: real values from dt_proj aren't tightly bounded,
    # but need to be in a range where softplus doesn't under/overflow.
    delta = (torch.rand(BATCH, D_INNER, seqlen, generator=g) * 4 - 2).to(device)
    A_log = torch.randn(D_INNER, D_STATE, generator=g)
    A = (-torch.exp(A_log)).to(device)
    B = torch.randn(BATCH, D_STATE, seqlen, generator=g).to(device)
    C = torch.randn(BATCH, D_STATE, seqlen, generator=g).to(device)
    D = torch.ones(D_INNER).to(device)
    z = torch.randn(BATCH, D_INNER, seqlen, generator=g).to(device)
    delta_bias = torch.randn(D_INNER, generator=g).to(device)
    return dict(u=u, delta=delta, A=A, B=B, C=C, D=D, z=z, delta_bias=delta_bias)


def compare(name, device):
    assert device == "cuda", "selective_scan_fn (the fused kernel) requires CUDA"
    results = {}
    for len_name, seqlen in SEQ_LENS.items():
        inputs = make_inputs(seqlen, device, seed=0)
        kwargs = dict(delta_softplus=True)

        out_fused = selective_scan_fn(**inputs, **kwargs)
        out_ref = selective_scan_ref(**inputs, **kwargs)

        diff = (out_fused.float() - out_ref.float()).abs()
        max_abs_diff = diff.max().item()
        mean_abs_diff = diff.mean().item()
        rel_diff = (diff / (out_ref.float().abs() + 1e-6)).mean().item()

        print(f"[{len_name}] seqlen={seqlen} max_abs_diff={max_abs_diff:.3e} "
              f"mean_abs_diff={mean_abs_diff:.3e} mean_rel_diff={rel_diff:.3e}")
        results[len_name] = {
            "seqlen": seqlen,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "mean_rel_diff": rel_diff,
        }
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("FATAL: no CUDA device -- selective_scan_fn (the fused kernel) cannot run on CPU. "
              "This script must run on a GPU instance, see docstring.")
        raise SystemExit(1)

    results = compare("validation", device)

    # Tolerance: both paths compute internally in fp32 (selective_scan_ref
    # explicitly casts u/delta/B/C to .float() regardless of input dtype;
    # the fused kernel does the same internally), so this is fp32-vs-fp32
    # agreement between two different implementations of the same math --
    # not fp16-vs-fp32, so a tight tolerance is the right bar.
    TOLERANCE = 1e-3
    all_passed = all(r["max_abs_diff"] < TOLERANCE for r in results.values())

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"tolerance": TOLERANCE, "passed": all_passed, "results": results},
        open(RESULTS_PATH, "w"),
        indent=2,
    )
    print(f"\n{'PASSED' if all_passed else 'FAILED'} (tolerance={TOLERANCE}). "
          f"Wrote {RESULTS_PATH}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
