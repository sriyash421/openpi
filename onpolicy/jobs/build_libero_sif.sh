#!/usr/bin/env bash
set -euo pipefail
S=/gscratch/scrubbed/sriyash/openpi_pt
export APPTAINER_CACHEDIR=$S/apptainer/cache APPTAINER_TMPDIR=$S/apptainer/tmp
apptainer build --fakeroot --force $S/apptainer/libero.sif $S/apptainer/libero.def
apptainer exec --bind /gscratch/socialrl/sriyash/openpi_pt:/app $S/apptainer/libero.sif python -c "import robosuite, mujoco, libero, openpi_client; print('libero sif ok', mujoco.__version__)"
