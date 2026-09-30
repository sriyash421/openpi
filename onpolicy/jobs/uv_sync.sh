#!/usr/bin/env bash
set -euo pipefail
/gscratch/scrubbed/sriyash/openpi_pt/run.sh 'uv sync && uv run python -c "import openpi, jax; print(\"[uv] ok\", jax.__version__)"'
