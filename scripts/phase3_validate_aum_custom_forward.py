#!/usr/bin/env python
"""Phase 3 gate: validate that quantized_bimamba_v1_forward (with
activation_spec={"bits": None}, i.e. no quantization at all) reproduces
AuM's REAL forward pass (the fused BiMambaInnerFn path) within numerical
tolerance. This must pass before trusting ANY Phase 3 AuM quantized
result -- if the custom-forward reimplementation itself doesn't match
the real model at full precision, every quantized number built on it is
wrong regardless of what the quantization grid shows.

Run on a GPU instance with the AuM venv (same as Phase 0/1/2). Needs the
ssmquant package importable -- `pip install -e .` from the repo root once,
same as it's installed locally:
    pip install -e /path/to/audio-ssm-quant
    cd third_party/Audio-Mamba-AuM
    cp ../../scripts/phase3_validate_aum_custom_forward.py .
    python3 phase3_validate_aum_custom_forward.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.append(".")  # third_party/Audio-Mamba-AuM, for `import src.models`
import src.models as models
from ssmquant.models import patch_model_for_quantized_forward, unpatch_model

HERE = Path(__file__).resolve().parent  # third_party/Audio-Mamba-AuM when copied there, per docstring
PROJECT_ROOT = HERE.parent.parent  # back to the repo root
RESULTS_PATH = PROJECT_ROOT / "results" / "phase3_aum_custom_forward_validation.json"

CHECKPOINT_PATH = "exps/speechcommands/models/aum-base_audioset-spc_v2.pth"
NUM_MEL_BINS = 128
EMBED_DIM = 768
N_CLASSES = 35
TEST_LENGTHS = {"native_128": 128, "extended_1998": 1998}
TOLERANCE = 1e-2  # looser than Phase 2's 1e-3: this chains 4 Linear layers
                   # + 2 scans + a reimplementation, not one isolated scan call


def build_model(target_length, device):
    model = models.AudioMamba(
        spectrogram_size=(NUM_MEL_BINS, target_length),
        patch_size=(16, 16),
        strides=(16, 16),
        embed_dim=EMBED_DIM,
        num_classes=N_CLASSES,
        imagenet_pretrain=False,
        imagenet_pretrain_path=None,
        aum_pretrain=True,
        aum_pretrain_path=CHECKPOINT_PATH,
        bimamba_type="v1",
    )
    model.to(device).eval()
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("FATAL: needs CUDA (causal_conv1d_fn / selective_scan_fn require it).")
        raise SystemExit(1)

    results = {}
    for name, target_length in TEST_LENGTHS.items():
        model = build_model(target_length, device)
        torch.manual_seed(0)
        x = torch.randn(1, NUM_MEL_BINS, target_length, device=device)

        with torch.no_grad():
            real_out = model(x)

        patched = patch_model_for_quantized_forward(
            model, weight_spec={"bits": None}, activation_spec={"bits": None}
        )
        try:
            with torch.no_grad():
                custom_out = model(x)
        finally:
            unpatch_model(patched)

        diff = (real_out.float() - custom_out.float()).abs()
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
