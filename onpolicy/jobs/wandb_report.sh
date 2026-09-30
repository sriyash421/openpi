#!/usr/bin/env bash
source /gscratch/scrubbed/sriyash/vla/.venv/bin/activate
export WANDB_DIR=/gscratch/scrubbed/sriyash/openpi_pt/wandb PYTHONDONTWRITEBYTECODE=1
cd /gscratch/socialrl/sriyash/openpi_pt && python -m onpolicy.wandb_report "$@"
