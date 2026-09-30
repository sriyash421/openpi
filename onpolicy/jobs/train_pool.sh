#!/usr/bin/env bash
# Continuous on-policy trainer: openpi train.py on repo_id pool:<root> (onpolicy/pool.py), never restarted per round.
# usage: [CFG=pi05_libero90_ctx NEG_FRAC=0 REFRESH=2000] train_pool.sh <exp> <instr_dropout> <recent_rounds> [train.py overrides...]
set -uo pipefail
S=/gscratch/scrubbed/sriyash/openpi_pt; EXP=$1; R=$S/onpolicy/$EXP; CFG=${CFG:-pi05_libero90_ctx}
until [ -f $S/expert_index/episodes.json ] && [ -f $R/round_000/DONE ]; do sleep 60; done
echo "{\"expert_dir\": \"$S/expert_index\", \"instr_dropout\": $2, \"recent_rounds\": $3, \"negative_fraction\": ${NEG_FRAC:-0}, \"refresh_every\": ${REFRESH:-2000}}" > $R/pool.json
RESUME=""; [ -d $S/checkpoints/$CFG/$EXP ] && RESUME="--resume" || RESUME="--overwrite"
$S/run.sh "PYTHONPATH=/gscratch/socialrl/sriyash/openpi_pt XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py $CFG --exp-name=$EXP --data.repo-id=pool:$R --num-train-steps=200000 --batch-size=64 --save-interval=200 --keep-period=1000 --num-workers=16 --anchor-no-context $RESUME ${*:4}"
echo "TRAIN_EXIT=$?"
