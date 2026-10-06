# Decisions log

Every design choice and deviation from the project brief, with the reason. Newest entries at the top.

## 2026-10-06 — Phase 1 RUN: full-precision length degradation, both models, real results

Ran both `scripts/phase1_ast_eval.py` and `scripts/phase1_aum_eval.py` on a
fresh Thunder Compute RTX A6000, budget-scoped to 1 clip/class
(`PHASE1_CLIPS_PER_CLASS=1`, 455 items) to fit the ~$2 credit remaining
after the earlier idle-instance mistake. Total GPU session: ~25 minutes,
instance deleted immediately after (see [[feedback_stop_idle_cloud_gpu_instances]]
-- this time done right).

**Headline result** (top-1 accuracy, `results/phase1_ast_full_precision_1pc.jsonl`,
`results/phase1_aum_full_precision_1pc.jsonl`):

| Length | AST (Transformer) | AuM (Mamba SSM) |
|---|---|---|
| native (~1s) | 100.0% (35/35) | 100.0% (35/35) |
| 20s | 68.6% (72/105) | 11.4% (12/105) |
| 40s | 36.2% (38/105) | 4.8% (5/105) |
| 80s | 14.3% (15/105) | 8.6% (9/105) |
| 160s | 7.6% (8/105) | 7.6% (8/105) |

**This is the core Phase 1 finding the brief asked for**: AuM collapses
almost immediately once extended past training length (100% -> 11% at just
20s, barely above the 2.9% chance rate for 35-way classification), while
AST degrades much more gradually (100% -> 69% -> 36% -> 14% -> 8%). This is
**full precision, no quantization applied yet** -- exactly the "confound"
Phase 1 exists to characterize, and it's a strong, real signal in the
direction the brief's cited prior work predicted (Mamba degrades past
training length more severely than transformers). Both models converge to
roughly the same near-chance accuracy by 160s, meaning there's a real floor
in both architectures, but AuM hits it almost immediately while AST takes
much longer.

**Position breakdown** (start/middle/end, n~12/cell -- small, noisy,
reported as suggestive not conclusive given the budget-constrained sample):
AST shows a fairly consistent start > middle/end pattern at 40s and 80s
(e.g. 80s: start=26% vs middle=end=9%), plausibly because the
bicubic-interpolated position embeddings are least distorted near the
sequence boundary closest to the original pretrained range. AuM shows no
clear position pattern -- all cells are low (0-17%) with no consistent
ordering, consistent with it having already lost most usable signal
regardless of where the event sits.

**Real bugs hit and fixed during this run** (all before producing any final
numbers, not after):
1. `torchaudio.load()` in the AST script hit the same torchcodec/FFmpeg-NPP
   shared library issue as Phase 0's Lightning.ai run -- fixed the same
   way, decode with `soundfile` directly instead.
2. `model.config.id2label` has **int** keys after `from_pretrained()` loads
   it (raw `config.json` on disk has string keys, but transformers
   normalizes them to int on load) -- a `str(pred_id)` lookup threw
   `KeyError`. Fixed by normalizing `id2label` to str keys once up front.
3. **The real one**: hardcoded `PATCH_SIZE=STRIDE=16` (AuM's non-overlapping
   patch convention) for AST's position-embedding grid math, but
   `MIT/ast-finetuned-speech-commands-v2` actually uses
   `frequency_stride=time_stride=10` (overlapping patches, the original AST
   paper's convention) -- a different hyperparameter choice than AuM's
   checkpoint even though they share the same label ordering. This produced
   a silent shape mismatch (`RuntimeError: shape '[1,8,8,768]' invalid for
   input of size 110592`) rather than silently wrong numbers, which is why
   it got caught before any numbers were trusted. Fixed by reading
   `patch_size`/`frequency_stride`/`time_stride` from the actual loaded
   model config instead of hardcoding them -- never assume two checkpoints
   from different authors share patch geometry just because they share a
   label scheme.

**Status**: Phase 1's full-precision baseline requirement (brief: "Run both
models at full precision across all lengths. Gate: report how much each
model degrades with length") is now satisfied, on the budget-scoped 1
clip/class sample. The full 4 clips/class design (`results/phase1_eval_manifest.json`,
1820 items, built but not run) remains available if more GPU budget shows
up later and tighter confidence intervals are wanted -- the 1/class result
already shows a large, clear effect, so this isn't blocking Phase 2.

## 2026-10-05 — Phase 1 built: length-extended Speech Commands V2, both eval scripts ready (not yet run)

**Task/dataset decision (user approved)**: switched both models to **Speech
Commands V2** for Phase 1, instead of continuing Phase 0's mismatched
AST→ESC-50 / AuM→VGGSound pairing. Both AST (`MIT/ast-finetuned-speech-commands-v2`,
98.12% reported) and AuM (their own "Speech Commands V2" checkpoint, 94.82%
reported, Base AudioSet variant) have official checkpoints on this exact
task -- real apples-to-apples, which Phase 0 didn't have. Bonus: native
clips are ~1s, so a 160s extended clip is a severe state-retention stress
test, which is exactly what brief section 5 flags as most interesting for
SSMs specifically.

**Dataset source**: the canonical `google/speech_commands` HF dataset is a
loading script, no longer supported by current `datasets` (4.5.0) -- raises
`RuntimeError: Dataset scripts are no longer supported`. A parquet mirror
(`danjacobellis/speech_commands_v2`) exists and loads fine, but uses
**alphabetical label ordering**, different from the ordering both AST and
AuM's checkpoints were actually trained with (AuM's own
`speechcommands_class_labels_indices.csv` copies AST's convention exactly:
0=backward, 1=follow, 2=five, 3=bed...). Used the **official source instead**
(`storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz`
-- the same one AuM's own `exps/speechcommands/prep_sc.py` downloads), which
also ships the official train/test/validation split lists and a
`_background_noise_` folder used as the Phase 1 background source (brief
section 5: "low-level noise or a neutral ambient recording... never clips
containing other labeled events" -- reusing the dataset authors' own
noise recordings is more faithful than synthesizing something ourselves).
First download attempt silently truncated (checked file size stabilized,
assumed done -- wrong; `curl` had been backgrounded without checking its own
exit code, same mistake as the causal_conv1d install earlier in the
project). Fixed by re-downloading with `--fail --retry` and actually
checking the exit code this time.

**Budget-driven subsampling (documented)**: stratified 4 clips/class x 35
classes = 140 native clips, x (1 native + 4 lengths x 3 positions) = 1820
total eval items. Full official test split is ~4890 clips; running the full
set x 4 lengths x 3 positions wasn't a sane use of the ~$2 Thunder Compute
credit remaining after the idle-instance mistake. 1820 items is still a
real sample, not a token few-shot check.

**AST's length confound (brief section 5's required decision)**: chose
**interpolate the positional embeddings**, not sliding-window chunking.
Checked `transformers` 4.57.1's AST source directly -- `position_embeddings`
is a plain fixed `nn.Parameter`, no built-in interpolation support (unlike
some ViT variants). Hand-rolled: bicubic-interpolate the patch-grid portion
of the position embedding along the time axis only (frequency axis is
unchanged since num_mel_bins stays 128), keeping the cls/distillation token
positions as-is. Implemented in `scripts/phase1_ast_eval.py`. AuM doesn't
need this hand-rolled -- its own codebase already has FlexiPatchEmbed/
FlexiPosEmbed (confirmed working from the Phase 0 AuM run's own log output:
"Initializing FlexiPatchEmbed... Loading position embedding!"), so
`scripts/phase1_aum_eval.py` just passes the right `spectrogram_size` per
length group at model construction time.

**Bug caught before running anything on GPU** (worth noting since it would
have silently produced wrong baseline numbers): both checkpoints were
trained/evaluated on a **fixed 128 mel-frame** native input, zero-padded --
not whatever frame count a raw ~1s clip naturally produces. Speech Commands
clips vary slightly under 1s, giving ~98 natural frames, which would have
used the WRONG (interpolated/mismatched) position embeddings for the
in-distribution baseline specifically -- the one number every length-
degradation comparison in Phase 1 is measured relative to. Caught by
computing the frame-count math by hand before running anything (no torch
needed for that check) and comparing against AuM's own `audio_length=128`
eval config. Fixed: both scripts now pad/crop only the native group to 128
frames; extended-length clips are left at their natural frame count since
they're exactly the target duration by construction already.

**Status**: dataset built and verified (`results/phase1_eval_manifest.json`,
1820 entries; spot-checked that embedded clip energy actually lands at the
intended start/middle/end position, not silence). Both checkpoints
downloaded (AST via HF Hub, AuM's Speech Commands V2 Base-AudioSet variant
via gdown). Both eval scripts written and logic-reviewed, but **NOT yet run**
-- no GPU instance is currently provisioned (deleted the idle one, ~$2
credit left). `scripts/phase0_setup_env.sh` has the exact working recipe for
re-provisioning when ready.

## 2026-10-05 — Thunder Compute instance deleted; left running idle, burned $18 of $20 credit

The RTX A6000 instance from the 2026-10-04 Phase 0 session was left running
(not stopped) across a roughly day-long gap with no active work. Thunder
Compute bills per hour regardless of utilization, and the $20 free credit
dropped to $2.07 almost entirely from idle time, not actual compute use.
Deleted the instance (`tnr delete 0 -y`) to stop further billing.

**Lesson for future sessions**: always `tnr delete <id>` (or the
equivalent stop/delete on whatever provider is in use) at the end of an
active work session, not just when "done with the project" — idle GPU time
costs the same as busy GPU time. Nothing is lost by deleting: the verified
working setup recipe lives in `scripts/phase0_setup_env.sh` and
`scripts/phase0_aum_vggsound_inference.py`, so a fresh instance just re-runs
that script rather than needing to rediscover any of it. Re-provisioning
a new Thunder Compute instance takes a few minutes; the $2.07 remaining
buys roughly 6 more hours on the A6000 at $0.35/hr.

## 2026-10-04 — mamba_ssm pipeline verified working end-to-end on Thunder Compute RTX A6000

**The big blocker since Phase 0 started is resolved**: the full `mamba_ssm`
fused-kernel pipeline now runs correctly on real hardware. This was blocked
since the project started because every free GPU found so far (T4 on
Lightning.ai and Kaggle) is Turing (compute capability 7.5), and
`mamba_ssm`'s fused kernel needs Ampere+ (8.0+).

**Compute**: user signed up for Thunder Compute's student program ($20 free
credit, US institutions only, console.thundercompute.com). Checked it
directly: the RTX A6000 (48GB, compute capability **8.6**) is unlocked with
just the free credit, no card required — only the A100/H100 require adding
a real card ("Payment required: a card or Auto-pay alone won't unlock it"),
same pattern as every other provider checked so far, so those stayed
untouched. $0.35/hr means the $20 credit covers ~57 hours.

**Connecting to it**: the console has no browser-based IDE (unlike
Lightning.ai) — it's SSH/CLI-only via their own `tnr` tool. The PyPI package
(`pip install tnr`) is explicitly deprecated and pulled in a mess of
conflicting dependencies into the *local* machine's conda environment
(numpy/pandas/pillow/rich/click version fights) — uninstalled it immediately.
The correct install is the standalone binary from
github.com/Thunder-Compute/thunder-cli/releases
(`tnr_2.1.0_darwin_arm64.tar.gz` for this Mac), placed in `~/.local/bin`.
`tnr login` needs an interactive TTY that a non-interactive shell doesn't
have; generated an API token instead (console → Settings → Authentication →
API Tokens) and ran `tnr login --token <token>`. `tnr connect 0` then both
auto-generates an SSH key and adds a `Host tnr-0` entry to `~/.ssh/config`,
after which plain `ssh tnr-0 '<command>'` works non-interactively — much
more efficient than the Lightning.ai session, which only had a browser-based
VS Code terminal.

**AuM does NOT use stock mamba_ssm.** This was the real discovery of this
session: `third_party/Audio-Mamba-AuM`'s own README requires:
- Python 3.10 (not whatever's on the box — this Ubuntu 22.04 box's default
  `python3` is 3.12; `python3.10` exists but needs `apt install
  python3.10-venv` first, which needs `apt-get update` run once first too)
- the OLD pinned `torch==2.1.1+cu118` (not whatever CUDA build matches the
  driver — the driver here reports CUDA 13.3, but PyTorch cu118 wheels still
  run fine against it; CUDA is backward compatible this way)
- the OLD pinned `causal_conv1d==1.1.3.post1` and `mamba_ssm==1.1.3.post1`
  (not latest — newer mamba_ssm's internals have diverged enough that AuM's
  `src/models/mamba_models.py` wouldn't import cleanly against them)
- **a bidirectional-processing patch** borrowed from the ViM (Vision Mamba)
  repo, shipped inside AuM's own repo at `vim-mamba_ssm/mamba_ssm/`, that
  must be copied over the installed `mamba_ssm` package in site-packages
  AFTER every `pip install mamba_ssm`. Stock mamba_ssm has no `bimamba_type`
  argument at all — AuM's forward/backward (`Fo-Bi`) and bidirectional
  (`Bi-Bi`) variants literally cannot run without this patch. The original
  Phase 0 scaffolding (`configs/model/aum.yaml`) didn't know this yet;
  `scripts/phase0_setup_env.sh` has been rewritten to do this correctly,
  replacing the earlier untested version that assumed conda and plain
  latest `mamba_ssm`/`causal-conv1d`.

**Two real build snags, both fixed**: (1) first `causal_conv1d`/`mamba_ssm`
install attempts failed on missing `wheel` — built-dep installs had gone
into a different, earlier venv by mistake, and the failure was masked
because the install was logged as `cmd > log 2>&1; echo EXIT:$?`, which
captures the `echo`'s exit code (always 0), not pip's. Fixed by checking the
log content directly, not trusting the wrapper's reported exit status, and
by using `cmd > log 2>&1 && echo SUCCESS || echo FAILED` from then on. (2)
first inference run failed in Triton's JIT compiler (`mamba_ssm`'s fused
RMSNorm needs Triton) with `Python.h: No such file or directory` — fixed
with `apt install python3.10-dev gcc`.

**Result**: `pip install mamba_ssm==1.1.3.post1` actually used a **prebuilt
wheel** from `github.com/state-spaces/mamba`'s releases matching our exact
torch/cuda/python combo (no local CUDA compilation needed for that one —
`causal_conv1d` did compile from source, ~1-2 min, no issues once `wheel`
was actually present). Ran AuM's own official inference notebook (converted
to `scripts/phase0_aum_vggsound_inference.py`) against the official
AudioSet→VGGSound checkpoint downloaded via `gdown` from the README's Google
Drive link: **4/5 (80%) correct** on the 5 sample clips the AuM authors
bundled in their own repo, with 0.91-0.99 confidence on the correct ones.
This is NOT a reproduction of their reported 46.78 mAP (that's measured
over the full VGGSound eval set, not 5 clips, and it's mAP not top-1 acc on
5 samples — different metric, wildly different N). What it does establish,
honestly: the model loads with `<All keys matched successfully>`, runs a
real bidirectional Mamba forward pass, and produces correct, confident
predictions using the exact checkpoint and preprocessing the authors
shipped. That's the pipeline-correctness bar Phase 0 needed, even though
the full dataset-level reproduction gate (within ~1 point of 46.78) is
still open — would need the actual VGGSound dataset, out of scope for this
session. Logged honestly as `gate_passed: false` in
`results/phase0_reproduction.jsonl`, same as the AST/ESC-50 entry.

**Next real step for Phase 0**: decide whether closing the strict
dataset-level reproduction gate (±1 point on a full eval set, for either
AST/AudioSet or AuM/VGGSound) is worth the engineering cost of downloading
one of those full datasets, or whether the two pipeline-correctness checks
done so far (AST on full ESC-50 modulo leakage, AuM on 5 official samples)
are sufficient grounds to move on to Phase 1 (length-extended eval sets).
This is a scope decision for the user, not something to decide unilaterally.

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
