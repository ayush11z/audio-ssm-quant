# Decisions log

Every design choice and deviation from the project brief, with the reason. Newest entries at the top.

## 2026-10-01 — Nebius evaluated as a compute alternative, paused

User signed up for Nebius (console.nebius.com) hoping for free GPU access.
Checked the console directly (logged-in session):

- **GPU quota**: real — 32x H100 (80GB), 32x H200, 32x L40S available in
  `eu-north1` (more in `eu-south1`), room for 5 GPU clusters. No A100s
  offered by Nebius at all; their lineup is H100/H200/L40S/RTX PRO
  6000/B200/B300/GB300. Not a downgrade from the brief's A100 target — H100
  is newer and `mamba_ssm`/`causal-conv1d` support it well.
- **Billing**: not configured, and no free-credit grant is visible anywhere
  in the console. Submitting billing details explicitly says it will charge
  the card **$25 immediately** to top up the balance; a single H100 then
  runs ~$4.63/hr (~$3,380/month if left running). This is a paid cloud, not
  a free tier.

Decision: paused Nebius. User chose to pursue the brief's original target
(Nautilus/PRP — UCSD co-founded it, likely free for this research use) or
other free options (Colab/Kaggle for small-scale Phase 0 checks) instead of
paying Nebius. If Nebius is revisited later with an actual credit code or
the user adds their own billing, the GPU quota above is already available —
no need to re-check it, just update `scripts/nautilus/phase0_pod.yaml`
analog for Nebius (VM create flow, not Kubernetes) and swap `model: h100`
in for `A100` anywhere the brief's configs assume Nautilus specifically.

## 2026-09-30 — Repo isolated from home-directory git repo

The directory this project lives in (`~/Documents/ML /SSM`) was *inside* a git
repository rooted at `/Users/ayushurs` (the whole home directory), with a
remote pointing at a classmate's GitHub repo (`vsahjwani/cse-158`). That
repo's untracked-file list included `.ssh/`, `.aws/`, shell history, tax
documents, etc. — a real risk if anyone ever ran `git add -A` from `~`.

Decision: `git init` a fresh, standalone repository scoped to this project
directory only. The home-directory repo was left completely untouched (no
files there were modified, staged, or deleted). This project's repo has no
remote yet — add one (private, under aurs@ucsd.edu's account) before pushing,
since Section 13 of the brief rules out any proprietary/private data leakage
and committed secrets are an explicit prior concern for this user.

## 2026-09-30 — Repo layout and config system

- **Hydra/OmegaConf** for config-driven experiments (brief section 11 asks
  for "YAML or Hydra"). `configs/` is split into groups (`model/`, `task/`,
  `quant/`, `experiment/`) so a run is composed as
  `model=aum task=esc50 quant=w8a16`.
- **`src/ssmquant/`** as an installable package (`pip install -e .` once a
  `pyproject.toml` is added) rather than loose scripts, so Phase 2's
  fake-quant utilities can be unit-tested (`tests/`) independently of any
  model.
- **`third_party/`** is the landing spot for vendored/cloned reference repos
  (AuM, Mamba-based-HuBERT) — gitignored, since these are large external
  codebases pulled in during Phase 0, not something this repo should vendor
  via copy. Pin exact commit hashes in `README.md`'s reproduction steps once
  cloned, so results stay reproducible even if upstream changes.
- **`results/`** holds JSONL logs (small, committed); raw tensors/arrays
  (`.npz`, `.pt`) are gitignored — too large, and brief section 11 only asks
  to log metrics, not raw caches.
- **`cache/`** (gitignored) holds cached full-precision outputs/states per
  Section 7's note that the reference scan is slow and should never be
  recomputed.

## 2026-09-30 — Compute location

Local machine (M2 Mac, no CUDA) cannot run `mamba_ssm`'s fused selective-scan
kernel or the pure-PyTorch reference scan at any real speed. All of Phase 0
onward must run on an A100 node on the Nautilus/PRP cluster, as specified in
brief section 7/10. **Blocked**: no `kubectl` context / Nautilus namespace is
configured on this machine yet. Scaffolding (configs, package skeleton,
Nautilus job spec, env file) was built locally; actual installs, checkpoint
loading, and reproduction runs are deferred until cluster access exists.
