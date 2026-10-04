#!/usr/bin/env bash
# Run on any Ampere+ GPU node (compute capability 8.0+ required — verify
# with `nvidia-smi --query-gpu=compute_cap --format=csv` first).
#
# This is the VERIFIED WORKING recipe (ran successfully 2026-10-04 on a
# Thunder Compute RTX A6000, compute capability 8.6, Ubuntu 22.04). It
# replaces an earlier, untested version of this script that assumed conda
# and the latest mamba_ssm/causal-conv1d — both wrong. See DECISIONS.md for
# the full story of what was wrong and why.
#
# AuM does NOT use stock mamba_ssm: it needs the OLD pinned versions
# (causal_conv1d==1.1.3.post1, mamba_ssm==1.1.3.post1) PLUS a bidirectional
# patch borrowed from the ViM repo that overwrites the installed mamba_ssm
# package. This script does both, matching third_party/Audio-Mamba-AuM's
# own README exactly.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== GPU check (needs compute capability 8.0+) =="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

echo "== system packages (venv + dev headers; Triton's JIT needs Python.h) =="
sudo apt-get update -q
sudo apt-get install -y python3.10-venv python3.10-dev gcc

echo "== clone AuM into third_party/ =="
mkdir -p third_party
if [ ! -d third_party/Audio-Mamba-AuM ]; then
  git clone https://github.com/mhamzaerol/Audio-Mamba-AuM third_party/Audio-Mamba-AuM
fi
echo "Pinned commit: $(cd third_party/Audio-Mamba-AuM && git rev-parse HEAD)"

echo "== python 3.10 venv (AuM pins torch 2.1.1, needs py3.10) =="
python3.10 -m venv ~/aum_venv
source ~/aum_venv/bin/activate
pip install -q --upgrade pip wheel ninja packaging setuptools

echo "== AuM's pinned torch build (cu118) =="
pip install -q torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118

echo "== AuM's requirements.txt =="
pip install -q -r third_party/Audio-Mamba-AuM/requirements.txt

echo "== causal_conv1d + mamba_ssm, exact pinned versions =="
# System nvcc (even a newer major version, e.g. 13.0) is fine here — pip
# fetches a prebuilt wheel matching torch/cuda/python from the official
# GitHub releases when one exists, and only falls back to compiling locally
# otherwise. Needs PATH to include a CUDA toolkit's bin/ either way.
export PATH=/usr/local/cuda/bin:$PATH
MAX_JOBS=4 pip install -v causal_conv1d==1.1.3.post1 --no-build-isolation
MAX_JOBS=4 pip install -v mamba_ssm==1.1.3.post1 --no-build-isolation

echo "== apply ViM's bidirectional patch (required — stock mamba_ssm has no bimamba_type) =="
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp -rf third_party/Audio-Mamba-AuM/vim-mamba_ssm/mamba_ssm "$SITE"

echo "== verify =="
python3 -c "
import sys; sys.path.insert(0, 'third_party/Audio-Mamba-AuM')
import src.models as models
print('AuM models import OK')
"

echo "== gdown, for pulling checkpoints off the README's Google Drive links =="
pip install -q gdown

echo "Done. Next: scripts/phase0_aum_vggsound_inference.py (see its docstring for how to run it)."
