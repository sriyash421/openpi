#!/usr/bin/env bash
# One-time environment setup, per openpi README (PyTorch) + examples/libero/README.md (without Docker).
set -euo pipefail
source /gscratch/scrubbed/sriyash/openpi_pt/env.sh
cd $OPENPI
uv sync
uv pip show transformers | grep -i version
cp -r src/openpi/models_pytorch/transformers_replace/* $S/venv/lib/python3.11/site-packages/transformers/
echo "[setup] main venv done"
uv venv --python 3.8 $S/libero_venv
source $S/libero_venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
deactivate
echo "[setup] libero venv done"
