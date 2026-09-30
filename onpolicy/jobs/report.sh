#!/usr/bin/env bash
source /gscratch/scrubbed/sriyash/vla/.venv/bin/activate
export WANDB_DIR=/gscratch/scrubbed/sriyash/openpi_pt/wandb
python /gscratch/scrubbed/sriyash/openpi_pt/jobs/report.py "$@"
