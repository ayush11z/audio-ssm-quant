#!/usr/bin/env python
"""Phase 4 gate: validate that quantized_bimamba_v1_scan_forward (with
target=None, bits=None, i.e. no SSM-internal quantization at all)
reproduces AuM's REAL forward pass (the fused BiMambaInnerFn path) within
numerical tolerance. This must pass before trusting ANY Phase 4 ablation
result -- if the reference-scan reimplementation itself doesn't match the
real model at full precision, every ablation number built on it is wrong
regardless of what the target-specific quantization shows.

This is the Phase 4 analogue of phase3_validate_aum_custom_forward.py
(which validated the Linear-layer-quantization path), one level deeper:
here the scan itself runs through a Python per-timestep loop
(_scan_one_direction in aum_quantized_scan.py, ported from
mamba_ssm's selective_scan_ref) instead of the fused selective_scan_fn,
since Δ/A/B/C/h can't be hooked mid-scan inside the fused kernel. Phase 2
already validated selective_scan_ref against the fused kernel directly
(max abs diff 3-5e-5 at both native and extended lengths, tolerance
1e-3) -- this script validates the FULL model forward (in_proj through
out_proj, bidirectional) built on top of that same reference scan,
chaining a few more operations, hence the looser 1e-2 tolerance (matching
Phase 3's gate).

Tests at native (128) AND the longest length the real Phase 4 grid will
actually run at (15998 frames, Phase 1's 160s condition) -- not just a
middling extended length -- since the per-timestep Python loop's
numerical behavior under float accumulation is exactly the kind of thing
that could degrade differently at 16x the length Phase 2 tested.

Run on a GPU instance with the AuM venv (same as Phase 0/1/2/3). Needs
the ssmquant package importable:
    pip install -e /path/to/audio-ssm-quant
    cd third_party/Audio-Mamba-AuM
    cp ../../scripts/phase4_validate_quantized_scan.py .
    python3 phase4_validate_quantized_scan.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.append(".")  # third_party/Audio-Mamba-AuM, for `import src.models`
import src.models as models
from ssmquant.models import patch_model_for_ssm_ablation
from ssmquant.models.aum_quantized_mamba import unpatch_model

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
RESULTS_PATH = PROJECT_ROOT / "results" / "phase4_quantized_scan_validation.json"

CHECKPOINT_PATH = "exps/speechcommands/models/aum-base_audioset-spc_v2.pth"
NUM_MEL_BINS = 128
EMBED_DIM = 768
N_CLASSES = 35
TEST_LENGTHS = {"native_128": 128, "extended_15998": 15998}
TOLERANCE = 1e-2  # matches Phase 3's gate; chains Phase 2's validated
                   # reference-scan-vs-fused-kernel diff (3-5e-5) through
                   # a bidirectional combination at a much longer length


def build_model(target_length, device):
    model = models.AudioMamba(
        spectrogram_size=(NUM_MEL_BINS, target_length),
        patch_size=(16, 16),
        strides=(16, 16),
        embed_dim=EMBED_DIM,
        num_classes=N_CLASSES,
        aum_pretrain=True,
        aum_pretrain_path=CHECKPOINT_PATH,
        bimamba_type="v1",
    )
    model.to(device).eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("FATAL: needs CUDA (causal_conv1d_fn requires it; the real "
              "forward pass being compared against also needs it).")
        raise SystemExit(1)

    results = {}
    for name, target_length in TEST_LENGTHS.items():
        model = build_model(target_length, device)
        torch.manual_seed(0)
        x = torch.randn(1, NUM_MEL_BINS, target_length, device=device)

        with torch.no_grad():
            real_out = model(x)

        patched, _ = patch_model_for_ssm_ablation(model, target=None, bits=None)
        try:
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                scan_out = model(x)
            torch.cuda.synchronize()
            scan_elapsed_s = time.time() - t0
        finally:
            unpatch_model(patched)
        print(f"[{name}] refscan forward pass (one clip, no quantization): {scan_elapsed_s:.2f}s "
              f"-- this is the real per-timestep-Python-loop cost the full ablation grid will pay per clip")

        diff = (real_out.float() - scan_out.float()).abs()
        max_abs_diff = diff.max().item()
        mean_abs_diff = diff.mean().item()
        rel_diff = (diff / (real_out.float().abs() + 1e-6)).mean().item()
        passed = max_abs_diff < TOLERANCE

        print(f"[{name}] target_length={target_length} max_abs_diff={max_abs_diff:.3e} "
              f"mean_abs_diff={mean_abs_diff:.3e} mean_rel_diff={rel_diff:.3e} "
              f"{'PASS' if passed else 'FAIL'}")
        results[name] = {
            "target_length": target_length,
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "mean_rel_diff": rel_diff,
            "passed": passed,
            "refscan_forward_elapsed_s": scan_elapsed_s,
        }
        del model
        torch.cuda.empty_cache()

    all_passed = all(r["passed"] for r in results.values())
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tolerance": TOLERANCE, "passed": all_passed, "results": results}, open(RESULTS_PATH, "w"), indent=2)
    print(f"\n{'PASSED' if all_passed else 'FAILED'} overall (tolerance={TOLERANCE}). Wrote {RESULTS_PATH}")
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
