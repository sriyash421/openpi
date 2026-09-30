#!/usr/bin/env bash
# usage: build_sif.sh <name>   (openpi | libero)
set -euo pipefail
S=/gscratch/scrubbed/sriyash/openpi_pt
export APPTAINER_CACHEDIR=$S/apptainer/cache APPTAINER_TMPDIR=$S/apptainer/tmp/$1
mkdir -p $APPTAINER_TMPDIR
apptainer build --fakeroot --force $S/apptainer/$1.sif $S/apptainer/$1.def
if [ "$1" = libero ]; then
  apptainer exec --bind /gscratch/socialrl/sriyash/openpi_pt:/app $S/apptainer/libero.sif python -c "import robosuite, mujoco, libero, openpi_client; print('[build] libero.sif ok', mujoco.__version__)"
fi
echo "[build] $1.sif done"
