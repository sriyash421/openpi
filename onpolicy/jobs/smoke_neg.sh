#!/usr/bin/env bash
# Smoke test of negative suppression + pure LoRA: 2-task rollout with negatives, then 20 train steps on 1 GPU.
set -uo pipefail
S=/gscratch/scrubbed/sriyash/openpi_pt; rm -rf $S/checkpoints/pi05_libero90_ctx_plora/smoke_neg
ls $S/onpolicy/smoke_neg/round_000/frames | head; grep -h negatives $S/onpolicy/smoke_neg/round_000/records_*.jsonl | head -c 600
CFG=pi05_libero90_ctx_plora NEG_FRAC=0.5 REFRESH=50 bash $S/jobs/train_pool.sh smoke_neg 0.8 3 --batch-size=8 \
  --num-train-steps=20 --save-interval=10 --keep-period=10 --num-workers=2 --negative-weight=0.2 --no-wandb-enabled
