#!/usr/bin/env bash
/gscratch/scrubbed/sriyash/openpi_pt/run.sh 'PYTHONPATH=/gscratch/socialrl/sriyash/openpi_pt XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_ctx --exp-name=anchor_test --data.repo-id=ctx:/gscratch/scrubbed/sriyash/openpi_pt/onpolicy/fb_demo_v1/iter_002/pairs --num-train-steps=21 --batch-size=16 --save-interval=1000 --log-interval=5 --anchor-no-context --overwrite --no-wandb-enabled'
echo "EXIT=$?"
