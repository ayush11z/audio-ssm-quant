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

**Phase 1 is built but not yet run.** Switched both models to Speech
Commands V2 (both have official checkpoints on this exact task — real
apples-to-apples, unlike Phase 0's mismatched pairing). Length-extended eval
set built and verified: 140 native clips × (1 native + 4 lengths × 3
positions) = 1820 items (`results/phase1_eval_manifest.json`), spliced into
official `_background_noise_` recordings at start/middle/end of
20/40/80/160s clips. Both eval scripts written
(`scripts/phase1_ast_eval.py`, `scripts/phase1_aum_eval.py`) — AST
interpolates its position embeddings for long inputs (hand-rolled, see
`DECISIONS.md`), AuM uses its own built-in Flexi resizing. Caught a bug
before running anything: both checkpoints expect a fixed 128-frame native
input (zero-padded), not whatever a raw ~1s clip naturally produces — fixed
in both scripts. Needs a GPU instance to actually run (none currently
provisioned).

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
- [x] **Phase 1 (built, not run)** — Length-extended eval sets built
      (lengths × positions, Speech Commands V2). Full-precision baseline
      scripts written for both models. Needs a GPU instance to execute.
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
