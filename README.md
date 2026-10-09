# Length-Dependent Quantization Fragility in Audio State Space Models

Does post-training quantization hurt audio Mamba encoders more as input audio
gets longer, compared to transformer audio encoders — and if so, which
internal tensors (Δ, A/Ā, B, C, recurrent state h) drive it?

Full project brief: see the plan this repo was scaffolded from (not checked
in here — ask if you need it re-pasted). Design decisions and any deviation
from that brief are logged in [`DECISIONS.md`](DECISIONS.md).

## Status

**Phase 0 is partially underway, strict dataset-level gate not yet passed
for either model — but both pipelines are now verified working
end-to-end.**

- **AST**: ran on a free Lightning.ai T4. Checkpoint loads, inference
  pipeline works, but 99.3% on the full ESC-50 set does **not** cleanly
  match the checkpoint author's reported 92.75% (train/eval leakage, no
  documented held-out fold). See `scripts/phase0_ast_esc50_eval.py`.
- **AuM / mamba_ssm**: the thing blocked since the project started —
  `mamba_ssm`'s fused kernel needs an Ampere+ GPU (8.0+), and every free GPU
  tried before this was Turing (7.5). Resolved via a free RTX A6000 (compute
  8.6) on Thunder Compute. AuM needs a bidirectional-patched `mamba_ssm`
  (not stock — see `DECISIONS.md`) plus old pinned versions
  (`torch==2.1.1+cu118`, `causal_conv1d`/`mamba_ssm==1.1.3.post1`). Ran the
  authors' own official inference checkpoint + sample data: **4/5 (80%)
  correct** with 0.91-0.99 confidence, using their exact preprocessing. Not
  a reproduction of their reported 46.78 mAP (that's over the full VGGSound
  eval set, 5 clips isn't), but strong evidence the pipeline itself is
  correct. See `scripts/phase0_aum_vggsound_inference.py`.

Both results logged honestly (`gate_passed: false`) in
`results/phase0_reproduction.jsonl` — neither claims a clean reproduction.
Closing the strict dataset-level gate for either model needs the actual
full eval set (AudioSet or VGGSound), which is a real scope decision, not
something to just do.

**Phase 1 is done — full-precision length-degradation baselines for both
models, real results.** Switched both models to Speech Commands V2 (both
have official checkpoints on this exact task — real apples-to-apples,
unlike Phase 0's mismatched pairing). Length-extended eval set built and
verified, spliced into official `_background_noise_` recordings at
start/middle/end of 20/40/80/160s clips. Ran a budget-scoped sample
(1 clip/class, 455 items) on a Thunder Compute RTX A6000:

| Length | AST (Transformer) | AuM (Mamba SSM) |
|---|---|---|
| native | 100.0% | 100.0% |
| 20s | 68.6% | 11.4% |
| 40s | 36.2% | 4.8% |
| 80s | 14.3% | 8.6% |
| 160s | 7.6% | 7.6% |

**AuM collapses almost immediately past training length (100%→11% at just
20s); AST degrades far more gradually.** This is full precision, no
quantization yet — exactly the "confound" Phase 1 exists to characterize,
and it's a real, strong signal in the direction the brief's cited prior
work predicted. See `DECISIONS.md` for the position-level breakdown and
three real bugs caught before trusting any of these numbers (most notably:
AST's actual checkpoint uses overlapping patches, stride=10, not AuM's
stride=16 convention — silently assuming they matched would have produced
wrong numbers without erroring). The full 4-clips/class design
(`results/phase1_eval_manifest.json`, 1820 items) is built and ready if
more GPU budget shows up later for tighter confidence intervals.

**Phase 2 is done.** The fake-quant (quantize-dequantize) framework is
written and unit-tested (`src/ssmquant/quant/fake_quant.py`, 13 passing
tests) — symmetric per-tensor/per-channel/per-token quantization with a
verified bit-identical full-precision passthrough. The reference-scan-vs-
fused-kernel validation (`scripts/phase2_validate_reference_scan.py`,
needed before Phase 4's SSM-internal ablations can hook Δ/A/Ā/h) ran clean:
max abs diff 3-5e-5 against a 1e-3 tolerance, at both native (128) and
extended (1998) sequence lengths, with no meaningful error growth between
them. The reference scan is safe to use for Phase 4.

## Repo layout

```
configs/            Hydra config groups: model/ task/ quant/ experiment/
src/ssmquant/        Installable package
  data/               Dataset construction: length extension, position control
  models/             AuM / AST wrappers, reference-scan hooking
  quant/              Fake-quant (quantize-dequantize) utilities
  eval/               Classification / WER evaluation loops
  analysis/           Bootstrap CIs, slope fits, figure generation
scripts/             Entry-point scripts (reproduction, grid runs, figures)
scripts/nautilus/    Kubernetes job specs for the PRP cluster
third_party/         Vendored reference repos (AuM, Mamba-HuBERT) — gitignored
tests/               Unit tests for quant utilities (Phase 2 gate)
configs/
results/             JSONL run logs (committed; raw tensors are gitignored)
figures/             Publication-style output figures (vector PDF)
cache/               Cached full-precision outputs/states — gitignored
```

## Phases (stop-and-report gates, per the brief)

- [ ] **Phase 0** — Install `mamba_ssm` + `causal-conv1d` on an A100 node.
      Load AuM + AST checkpoints. Reproduce their reported benchmark numbers
      to within ~1 point. **Blocked on Nautilus/PRP cluster access.**
- [x] **Phase 1** — Length-extended eval sets built (lengths × positions,
      Speech Commands V2). Full-precision baselines run for both models
      (budget-scoped sample): AuM collapses to near-chance by 20s past
      training length, AST degrades much more gradually.
- [x] **Phase 2** — Fake-quant framework + 13 passing unit tests
      (`src/ssmquant/quant/fake_quant.py`). Reference scan validated
      against the fused kernel: max abs diff 3-5e-5 (tolerance 1e-3) at
      both native and extended lengths — safe for Phase 4.
- [x] **Phase 3** — Standard quantization grid (W8A16, W4A16, W8A8) × both
      models × all lengths. AuM needed a custom forward reimplementation
      (`src/ssmquant/models/aum_quantized_mamba.py`) since its fused
      `BiMambaInnerFn` bypasses `nn.Linear` hooks, plus its own static
      per-tensor calibration path — gate-validated bit-exact against the
      real forward pass (max abs diff 0.0).
      **Primary result (3 clips/class, n=315 per extended length,
      `results/phase3_{ast,aum}_quantized_3pc.jsonl` + matching
      `phase1_{ast,aum}_full_precision_3pc*` baselines)**: AST shows a
      real, mostly W4A16-specific accuracy drop at 40-80s (−8 to −9pp,
      statistically significant; W8A16/W8A8 show no degradation beyond
      full precision). **AuM shows no quantization-specific signal at
      any length or bit-width** — its full-precision accuracy is already
      at chance from 20s onward, and every quantized condition stays
      within ~1 standard error of that same full-precision number.
      Brief hypothesis H1 (quantization hurts AuM more than AST) is not
      supported by top-1 accuracy — see DECISIONS.md for the full tables
      and the floor-effect discussion. An earlier 1-clip/class round
      (kept in DECISIONS.md for history) first raised this floor-effect
      concern; the 3-clip/class re-run confirmed it's a real null result,
      not a small-sample artifact. Detecting AuM-specific quantization
      sensitivity, if it exists, will need a softer metric than top-1
      accuracy (logit margin / entropy / KL divergence) — planned for
      Phase 4. Note: the two Phase 1 full-precision baseline runs' per-clip
      rows were lost to a process mistake (not copied back before the GPU
      instance was deleted); only their aggregate accuracy survives
      (`results/phase1_run_logs/`, `results/phase1_{ast,aum}_full_precision_3pc_summary.json`).
      The Phase 3 quantized grid's own per-clip rows are intact.
- [ ] **Phase 4** — SSM-internal ablations (Δ / A / B,C / h) + state
      divergence logging. Code written and gate-validated (bit-exact
      reference-scan reimplementation, confirmed on two different GPU
      generations — see DECISIONS.md), but the real ablation grid hasn't
      completed yet: lost twice to GPU-provider infrastructure failures
      (Thunder Compute account deactivated mid-run, then a DSMLP pod got
      destroyed mid-session), not a code problem. Needs a retry.
- [ ] **Phase 5 (stretch)** — Mamba-HuBERT vs. HuBERT on concatenated
      LibriSpeech (WER).
- [x] **Phase 6** — Bootstrap CIs, slope fits, figures, tables, computed
      from Phase 1/3's real data (`scripts/phase6_bootstrap_slopes.py`,
      `src/ssmquant/analysis/{bootstrap,slope_fit,figures}.py`, 15 unit
      tests). Turns the floor-effect finding into a number: AuM's length-
      degradation slope is ~15x shallower than AST's (-0.04 vs -0.61
      accuracy points per decade of length). Figures in `figures/`.
      Doesn't need a GPU — will extend to cover Phase 4 once that lands.
- [ ] **Phase 7** — Paper draft (LaTeX, 1–3 pages, arXiv-ready).

## Reproducing a figure

Nothing to reproduce yet — this section will be filled in as each phase
lands, with the exact config + seed used for each figure/table.

## Setting up on Nautilus

1. Get a Nautilus/PRP namespace and `kubectl` context (not yet configured —
   see https://nautilus.optiputer.net docs for onboarding).
2. `kubectl apply -f scripts/nautilus/phase0_pod.yaml` to launch an A100 pod.
3. Inside the pod: `bash scripts/phase0_setup_env.sh`.
4. `python scripts/phase0_reproduce.py` to run the Phase 0 gate.

## Models

- **SSM**: Audio Mamba (AuM) — https://github.com/mhamzaerol/Audio-Mamba-AuM
- **Transformer baseline**: AST —
  `MIT/ast-finetuned-audioset-10-10-0.4593` (Hugging Face)
- **Stretch (ASR)**: Mamba-based HuBERT —
  https://github.com/hckuo145/Mamba-based-HuBERT vs. standard HuBERT

Checkpoint availability has not been verified yet — that's part of the
Phase 0 gate.
