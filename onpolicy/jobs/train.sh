#!/usr/bin/env bash
# Default pi05_libero recipe on LIBERO-90 (JAX), 2 H200 with FSDP. usage: train.sh <exp_name> [extra tyro args...]
set -uo pipefail
S=/gscratch/scrubbed/sriyash/openpi_pt; EXP=$1; shift
EXTRA="$*"; [ -d $S/checkpoints/pi05_libero90/$EXP ] && EXTRA="$EXTRA --resume" || EXTRA="$EXTRA"
$S/run.sh "XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero90 --exp-name=$EXP --fsdp-devices=2 --keep-period=3000 --num-workers=16 $EXTRA"
echo "TRAIN_EXIT=$?"
