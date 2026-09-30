#!/usr/bin/env bash
# Run a command inside the openpi Ubuntu 22.04 container with all paths pinned to scrubbed.  usage: run.sh <cmd...>
S=/gscratch/scrubbed/sriyash/openpi_pt
NV=(); command -v nvidia-smi >/dev/null && NV=(--nv)
exec apptainer exec "${NV[@]}" --home $S/home --bind /gscratch:/gscratch --bind /mmfs1:/mmfs1 \
    --pwd /gscratch/socialrl/sriyash/openpi_pt $S/apptainer/openpi.sif \
    bash -c "source $S/env.sh && $*"
