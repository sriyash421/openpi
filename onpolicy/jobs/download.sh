#!/usr/bin/env bash
# Parallel downloads into the exact caches openpi reads: OPENPI_DATA_HOME (checkpoints) and HF_LEROBOT_HOME (dataset).
set -euo pipefail
S=/gscratch/scrubbed/sriyash/openpi_pt
source $S/env.sh
PY=$S/dl_venv/bin/python
for c in pi05_base pi05_libero; do
  $PY -c "
import gcsfs, os
fs = gcsfs.GCSFileSystem(token='anon')
dst = os.path.join(os.environ['OPENPI_DATA_HOME'], 'openpi-assets/checkpoints/$c')
fs.get('openpi-assets/checkpoints/$c/', dst, recursive=True)
print('[download] $c ->', dst)
" &
done
HF_HUB_ENABLE_HF_TRANSFER=1 $PY -c "
import os
from huggingface_hub import snapshot_download
d = os.path.join(os.environ['HF_LEROBOT_HOME'], 'ryanhoangt/libero_90_lerobot_v21')
snapshot_download('ryanhoangt/libero_90_lerobot_v21', repo_type='dataset', local_dir=d, max_workers=16)
print('[download] dataset ->', d)
" &
wait
du -sh $OPENPI_DATA_HOME/openpi-assets/checkpoints/* $HF_LEROBOT_HOME/ryanhoangt/*
