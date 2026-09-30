#!/usr/bin/env bash
# Inside openpi.sif: uv env, released checkpoints, LIBERO-90 data + norm stats.
set -euo pipefail
S=/gscratch/scrubbed/sriyash/openpi_pt
$S/run.sh 'uv sync && echo "[setup2] venv ok"'
$S/run.sh 'uv run python -c "
from openpi.shared import download
for c in [\"pi05_base\", \"pi05_libero\"]:
    print(\"[setup2]\", download.maybe_download(f\"gs://openpi-assets/checkpoints/{c}\"))
"'
$S/run.sh 'uv run scripts/compute_norm_stats.py --config-name pi05_libero90 && echo "[setup2] norm stats ok"'
