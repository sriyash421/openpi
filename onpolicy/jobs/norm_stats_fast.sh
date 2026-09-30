#!/usr/bin/env bash
# scripts/compute_norm_stats.py main(), unchanged except the config's data-loader worker count (2 -> 16).
set -euo pipefail
/gscratch/scrubbed/sriyash/openpi_pt/run.sh 'uv run python -c "
import dataclasses, sys
sys.path.insert(0, \"scripts\")
import compute_norm_stats as c
import openpi.training.config as _config
orig = _config.get_config
_config.get_config = lambda name: dataclasses.replace(orig(name), num_workers=16)
c.main(\"pi05_libero90\")
" && echo "[norm-fast] ok"'
