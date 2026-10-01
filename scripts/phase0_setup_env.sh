#!/usr/bin/env bash
# Run INSIDE the Nautilus/PRP A100 pod (scripts/nautilus/phase0_pod.yaml).
# Installs mamba_ssm + causal-conv1d (fused CUDA kernels — must build
# against this node's CUDA toolkit, which is why this can't run on a dev
# laptop) and clones the AuM / Mamba-HuBERT reference repos into
# third_party/, pinned to the commit used for this project's results.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== conda env =="
conda env create -f environment.yml || conda env update -f environment.yml
source activate ssmquant 2>/dev/null || conda activate ssmquant

echo "== causal-conv1d (fused kernel, needs CUDA toolkit on this node) =="
pip install causal-conv1d --no-build-isolation

echo "== mamba-ssm (fused kernel) =="
pip install mamba-ssm --no-build-isolation

echo "== clone reference repos into third_party/ =="
mkdir -p third_party
if [ ! -d third_party/Audio-Mamba-AuM ]; then
  git clone https://github.com/mhamzaerol/Audio-Mamba-AuM third_party/Audio-Mamba-AuM
fi
if [ ! -d third_party/Mamba-based-HuBERT ]; then
  git clone https://github.com/hckuo145/Mamba-based-HuBERT third_party/Mamba-based-HuBERT
fi

echo "== record pinned commits in DECISIONS.md (do this manually once cloned) =="
( cd third_party/Audio-Mamba-AuM && git rev-parse HEAD )
( cd third_party/Mamba-based-HuBERT && git rev-parse HEAD )

echo "Done. Next: python scripts/phase0_reproduce.py"
