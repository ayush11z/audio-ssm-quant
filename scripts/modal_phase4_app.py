"""Modal app for Phase 4's native-length re-run (see DECISIONS.md's
2026-10-13 entry for why: the first completed run used only 5 clips/
length, covering 2 distinct words per extended length -- this re-run
uses the full 140-clip, 35-word-balanced native-length set instead).

Unlike every other GPU provider touched in this project (Thunder
Compute, DSMLP, Lightning AI), Modal is driven directly from the local
machine via its Python SDK/CLI -- no SSH relay needed. Environment is
declared in code (the Image below) instead of shell setup scripts,
though it follows the exact same proven recipe documented in
DECISIONS.md (torch==2.1.1+cu118, causal_conv1d/mamba_ssm==1.1.3.post1
from prebuilt wheels, ViM bidirectional patch, AuM's requirements.txt
installed BEFORE any mamba_ssm import to avoid the numpy/transformers
version-ordering bug).

Data (checkpoint + Speech Commands V2) persists in a Modal Volume across
runs so it's only downloaded once. Each GPU function copies it from the
volume into the container's local disk at startup (simpler and more
robust than juggling multiple volume mount paths).

Usage (run from the repo root; each is a separate step, checked before
moving to the next -- same validate-before-scaling discipline as every
other phase):
    modal run scripts/modal_phase4_app.py::setup_data
    modal run scripts/modal_phase4_app.py::run_gate
    modal run scripts/modal_phase4_app.py::run_native_ablation
"""
import modal

app = modal.App("audio-ssm-quant-phase4")
volume = modal.Volume.from_name("audio-ssm-quant-data", create_if_missing=True)
VOL = "/vol"
REPO = "/root/audio-ssm-quant"
GPU = "L4"  # Ada, compute 8.9 -- clears AuM's Ampere+ (8.0+) requirement with margin
# Pinned so the clone step's command string changes (and Modal's image
# cache invalidates) whenever this repo gets new commits -- without this,
# Modal reuses the cached clone from whenever the image was first built,
# silently running stale code (hit exactly this: run_native_ablation's
# first attempt failed because the image predated this very script being
# pushed). Update with `git rev-parse HEAD` before any run that depends
# on a change made since the image was last built.
REPO_COMMIT = "fc66f77380c372bd717c0c4baa4ac54a2f4608c2"

image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "wget")
    .pip_install("setuptools==69.5.1", "wheel", "ninja", "packaging", "gdown", "soundfile")
    .pip_install(
        "torch==2.1.1", "torchvision==0.16.1", "torchaudio==2.1.1",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .run_commands(
        "pip install -v causal_conv1d==1.1.3.post1 --no-build-isolation",
        "pip install -v mamba_ssm==1.1.3.post1 --no-build-isolation",
        gpu=GPU,
    )
    .run_commands(
        f"git clone -q https://github.com/ayush11z/audio-ssm-quant.git {REPO} && "
        f"cd {REPO} && git checkout -q {REPO_COMMIT}",
        f"cd {REPO} && mkdir -p third_party && "
        f"git clone -q https://github.com/mhamzaerol/Audio-Mamba-AuM.git third_party/Audio-Mamba-AuM",
    )
    # AuM's requirements.txt (pins numpy<2, transformers==4.35.2) MUST
    # install before anything imports mamba_ssm -- see DECISIONS.md.
    .run_commands(f"pip install -r {REPO}/third_party/Audio-Mamba-AuM/requirements.txt")
    .run_commands(
        "SITE=$(python3 -c 'import mamba_ssm, os; print(os.path.dirname(mamba_ssm.__file__))') && "
        f"cp -r {REPO}/third_party/Audio-Mamba-AuM/vim-mamba_ssm/mamba_ssm/* $SITE/"
    )
    .run_commands(f"cd {REPO} && pip install -e .")
)

CHECKPOINT_GDRIVE_ID = "1ikkU4COOqeCNCVTn4b7LulNr9p4Efr4M"  # AuM README's "Base AudioSet", Speech Commands V2 row
DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"


@app.function(image=image, volumes={VOL: volume}, timeout=1800)
def setup_data():
    """Downloads the checkpoint + Speech Commands V2 and builds the eval
    manifest, once, into the persistent volume, storing the large/many-
    small-file pieces as single tarballs (not extracted trees) -- a
    fresh container reading ~85k individual files over the volume mount
    is slow (cold, no cache; the first run_gate attempt hit a 600s
    timeout on exactly this), copying and locally extracting one archive
    is fast regardless of cache state. Safe to re-run -- skips whatever's
    already there."""
    import os
    import subprocess

    os.makedirs(f"{VOL}/checkpoint", exist_ok=True)
    vol_ckpt = f"{VOL}/checkpoint/aum-base_audioset-spc_v2.pth"
    if not os.path.exists(vol_ckpt):
        print("Downloading AuM checkpoint...")
        subprocess.run(["gdown", CHECKPOINT_GDRIVE_ID, "-O", vol_ckpt], check=True)
        volume.commit()
    else:
        print("Checkpoint already on volume, skipping download.")

    os.makedirs(f"{VOL}/raw", exist_ok=True)
    vol_archive = f"{VOL}/raw/speech_commands_v0.02.tar.gz"
    if not os.path.exists(vol_archive):
        print("Downloading Speech Commands V2...")
        subprocess.run(["wget", "-q", "-O", vol_archive, DATASET_URL], check=True)
        volume.commit()
    else:
        print("Speech Commands V2 archive already on volume, skipping download.")

    # Extract locally (not onto the volume) to build the manifest.
    extract_dir = f"{REPO}/data/raw/SpeechCommands/speech_commands_v0.02"
    os.makedirs(extract_dir, exist_ok=True)
    subprocess.run(["tar", "-xzf", vol_archive, "-C", extract_dir], check=True)

    manifest_tarball = f"{VOL}/processed/phase1_processed.tar.gz"
    manifest_json = f"{VOL}/processed/phase1_eval_manifest.json"
    if not os.path.exists(manifest_tarball):
        print("Building the length-extended eval manifest (native + extended, 4 clips/class)...")
        subprocess.run(["python3", "scripts/phase1_build_length_extended_dataset.py"], cwd=REPO, check=True)
        os.makedirs(f"{VOL}/processed", exist_ok=True)
        subprocess.run(
            ["tar", "-czf", manifest_tarball, "-C", f"{REPO}/data/processed", "phase1_speech_commands"],
            check=True,
        )
        subprocess.run(["cp", f"{REPO}/results/phase1_eval_manifest.json", manifest_json], check=True)
        volume.commit()
    else:
        print("Eval manifest already built on volume, skipping.")

    print("Data setup complete.")


def _stage_data():
    """Shared by run_gate/run_native_ablation: copy the volume's tarballs
    into the container and extract them LOCALLY into the repo-relative
    paths the scripts expect -- single-file copies + local extraction,
    not directory-tree copies over the volume mount (see setup_data's
    docstring for why that was slow). Prints timing for each step."""
    import os
    import subprocess
    import time

    t0 = time.time()
    ckpt_dir = f"{REPO}/third_party/Audio-Mamba-AuM/exps/speechcommands/models"
    os.makedirs(ckpt_dir, exist_ok=True)
    subprocess.run(["cp", f"{VOL}/checkpoint/aum-base_audioset-spc_v2.pth", ckpt_dir], check=True)
    print(f"[_stage_data] checkpoint copied in {time.time()-t0:.1f}s")

    t0 = time.time()
    extract_dir = f"{REPO}/data/raw/SpeechCommands/speech_commands_v0.02"
    os.makedirs(extract_dir, exist_ok=True)
    subprocess.run(["tar", "-xzf", f"{VOL}/raw/speech_commands_v0.02.tar.gz", "-C", extract_dir], check=True)
    print(f"[_stage_data] SpeechCommands raw data staged in {time.time()-t0:.1f}s")

    t0 = time.time()
    os.makedirs(f"{REPO}/data/processed", exist_ok=True)
    subprocess.run(
        ["tar", "-xzf", f"{VOL}/processed/phase1_processed.tar.gz", "-C", f"{REPO}/data/processed"], check=True
    )
    os.makedirs(f"{REPO}/results", exist_ok=True)
    subprocess.run(["cp", f"{VOL}/processed/phase1_eval_manifest.json", f"{REPO}/results/"], check=True)
    print(f"[_stage_data] processed manifest + wavs staged in {time.time()-t0:.1f}s")


@app.function(image=image, volumes={VOL: volume}, gpu=GPU, timeout=1800)
def run_gate():
    """Phase 4's validation gate -- must pass before anything else is trusted."""
    import subprocess

    _stage_data()
    aum_dir = f"{REPO}/third_party/Audio-Mamba-AuM"
    subprocess.run(["cp", f"{REPO}/scripts/phase4_validate_quantized_scan.py", aum_dir], check=True)
    result = subprocess.run(["python3", "-u", "phase4_validate_quantized_scan.py"], cwd=aum_dir)
    return result.returncode


@app.function(image=image, volumes={VOL: volume}, gpu=GPU, timeout=7200)
def run_native_ablation():
    """The real native-length, class-balanced (140 clips, 35 words) Phase
    4 ablation grid. Results are copied back into the volume so they
    survive container teardown. Streams output live (see _stage_data's
    docstring for why -- a captured/buffered subprocess hid where the
    first gate attempt's 600s timeout actually went)."""
    import subprocess

    _stage_data()
    aum_dir = f"{REPO}/third_party/Audio-Mamba-AuM"
    subprocess.run(["cp", f"{REPO}/scripts/phase4_aum_ablation_native_eval.py", aum_dir], check=True)
    result = subprocess.run(["python3", "-u", "phase4_aum_ablation_native_eval.py"], cwd=aum_dir)

    import os
    os.makedirs(f"{VOL}/results_native", exist_ok=True)
    for fname in ("phase4_aum_ablation_native.jsonl", "phase4_state_divergence_native.json"):
        src = f"{REPO}/results/{fname}"
        if os.path.exists(src):
            subprocess.run(["cp", src, f"{VOL}/results_native/{fname}"], check=True)
    volume.commit()
    return result.returncode
