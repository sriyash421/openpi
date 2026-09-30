#!/usr/bin/env bash
# usage: sbatch -A socialrl -p gpu-l40 --gres=gpu:1 -c 16 --mem=64G -t 6:00:00 \
#          -o $S/onpolicy/localize_test/repr/%j.log onpolicy/jobs/localize_repr.sh extract --shard 0/3
/gscratch/scrubbed/sriyash/openpi_pt/run.sh "XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 HF_HUB_OFFLINE=1 PYTHONPATH=/gscratch/socialrl/sriyash/openpi_pt uv run --no-sync python -m onpolicy.localize_repr $*"
