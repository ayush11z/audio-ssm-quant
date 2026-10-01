# Length-Dependent Quantization Fragility in Audio State Space Models

Does post-training quantization hurt audio Mamba encoders more as input audio
gets longer, compared to transformer audio encoders — and if so, which
internal tensors (Δ, A/Ā, B, C, recurrent state h) drive it?

Full project brief: see the plan this repo was scaffolded from (not checked
in here — ask if you need it re-pasted). Design decisions and any deviation
from that brief are logged in [`DECISIONS.md`](DECISIONS.md).

## Status

**Phase 0 (environment and reproduction) has not started.** This repo
currently contains only the project skeleton: config structure, package
layout, and a Nautilus job spec. See `DECISIONS.md` for why — short version:
this needs an A100 node (Nautilus/PRP cluster) and that access isn't
configured on the machine that scaffolded this repo yet.

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
- [ ] **Phase 1** — Build length-extended eval sets (lengths × positions).
      Full-precision baselines for both models at all lengths.
- [ ] **Phase 2** — Fake-quant framework + unit tests; validate reference
      scan against the fused kernel.
- [ ] **Phase 3** — Standard quantization grid (W8A16, W4A16, W8A8) × both
      models × all lengths × 3 calibration seeds.
- [ ] **Phase 4** — SSM-internal ablations (Δ / A / B,C / h) + state
      divergence logging.
- [ ] **Phase 5 (stretch)** — Mamba-HuBERT vs. HuBERT on concatenated
      LibriSpeech (WER).
- [ ] **Phase 6** — Bootstrap CIs, slope fits, figures, tables.
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
