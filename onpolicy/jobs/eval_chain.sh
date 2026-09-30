#!/usr/bin/env bash
export WANDB_DIR=/gscratch/scrubbed/sriyash/openpi_pt/wandb PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd /gscratch/socialrl/sriyash/openpi_pt
/gscratch/scrubbed/sriyash/openpi_pt/dl_venv/bin/python -m onpolicy.eval_chain "$@"
