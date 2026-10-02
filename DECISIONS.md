# Decisions log

Every design choice and deviation from the project brief, with the reason. Newest entries at the top.

## 2026-10-01 — Lightning.ai free T4: Phase 0 AST/ESC-50 run, honest result

User had free credits on lightning.ai (5.00 credits in a "default-project"
teamspace). Checked it directly: Lightning.ai genuinely offers a **free T4**
(16GB, no card required) alongside paid A100/H100/etc. (A100 does require a
verified card even on "free tier" -- declined per user's explicit "don't add
my card anywhere"). Spun up a Studio (`ssm-quant-phase0`), cloned this repo,
and ran the AST half of Phase 0 on the free T4.

**mamba_ssm is still out of scope on this GPU**: T4 is Turing (compute
capability 7.5), `mamba_ssm`'s fused kernel needs Ampere+ (8.0+), confirmed
before installing anything. AuM/the SSM side of Phase 0 is still blocked on
A100/H100 access.

**Checkpoint found**: `bioamla/ast-esc50` (mirrored at
`shreyahegde/ast-finetuned-audioset-10-10-0.450_ESC50`) is a community
fine-tune of the brief's exact checkpoint
(`MIT/ast-finetuned-audioset-10-10-0.4593`) on ESC-50, reporting **92.75%**
accuracy on its own eval split. The brief's named checkpoint
(`MIT/ast-finetuned-audioset-10-10-0.4593`) does still exist on HF, but
reproducing its 45.93 mAP exactly needs the full AudioSet eval set (~20k
YouTube-sourced clips) -- not attempted, out of scope for a quick free-GPU
session.

**Result** (`scripts/phase0_ast_esc50_eval.py`, logged in
`results/phase0_reproduction.jsonl`): ran the checkpoint over all 2000 clips
of `ashraq/esc50` (the only split that dataset ships -- no documented
held-out test fold) and got **99.3% accuracy**, not the reported 92.75%.

**This is NOT a passed Phase 0 gate.** The checkpoint author never
documented which ESC-50 fold(s) they held out during fine-tuning, and
`ashraq/esc50` has no separate test split, so evaluating on "the whole
dataset" almost certainly includes clips the model was fine-tuned on --
the 99.3% is inflated by train/eval leakage, not a clean reproduction.
Per brief section 12 ("never fabricate or estimate results"), this is
reported honestly as unresolved rather than rounded off to "~1 point, close
enough." What it DOES establish: the environment, checkpoint loading, and
AST inference pipeline all work correctly end-to-end on a free T4 (sensible
predictions, no pipeline bugs) -- genuinely useful progress, just not the
specific gate the brief asks for.

**To actually close this gate**: either get the checkpoint author's
train/test fold split, or fine-tune a fresh AST-on-ESC50 checkpoint
ourselves with a documented 4-fold-train/1-fold-test split (standard ESC-50
protocol) -- the latter is more work but fully reproducible by us.

**Engineering friction hit along the way** (fixed, noted in case it recurs):
the base conda environment's `scipy` (1.18.1) predates `numpy`'s removal of
`np.long`, breaking `transformers`' import chain as soon as any numpy >1.20
was installed; fixed by pinning `numpy==1.26.4` + `scipy==1.13.1` together.
Separately, `torchaudio` installed via plain `pip install torchaudio` built
against a different torch ABI than the already-installed `torch==2.8.0+cu128`
(`undefined symbol: torch_library_impl`); fixed by reinstalling
`torchaudio==2.8.0` from `https://download.pytorch.org/whl/cu128` explicitly.
`datasets`' newer `Audio` feature requires `torchcodec`, which in turn needs
system FFmpeg/NVIDIA NPP libraries not present on the Studio -- sidestepped
by decoding audio ourselves with `soundfile` instead of relying on
`datasets`' built-in decode-on-access.

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
