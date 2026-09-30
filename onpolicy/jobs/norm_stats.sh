#!/usr/bin/env bash
set -euo pipefail
/gscratch/scrubbed/sriyash/openpi_pt/run.sh 'uv run scripts/compute_norm_stats.py --config-name pi05_libero90 && echo "[norm] ok"'
