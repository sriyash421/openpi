#!/usr/bin/env bash
source /gscratch/scrubbed/sriyash/vla/.venv/bin/activate
export WANDB_DIR=/gscratch/scrubbed/sriyash/openpi_pt/wandb PYTHONDONTWRITEBYTECODE=1
cd /gscratch/socialrl/sriyash/openpi_pt
python -m onpolicy.validate_judge --eval-dir /gscratch/scrubbed/sriyash/openpi_pt/eval/pi05_libero --n-per-class 60 --out /gscratch/scrubbed/sriyash/openpi_pt/onpolicy/judge_val --wandb
