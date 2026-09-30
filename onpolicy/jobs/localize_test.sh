#!/usr/bin/env bash
# usage: sbatch -A socialrl -p gpu-l40 --gres=gpu:1 -c 16 --mem=64G \
#          -t 3:00:00 -o $S/onpolicy/localize_test/%j.log onpolicy/jobs/localize_test.sh [localize_test args...]
# OPENROUTER_API_KEY must be in the submitting environment (sbatch and apptainer pass it through).
/gscratch/scrubbed/sriyash/openpi_pt/run.sh "HF_HUB_OFFLINE=1 PYTHONPATH=/gscratch/socialrl/sriyash/openpi_pt uv run --no-sync python -m onpolicy.localize_test $*"
