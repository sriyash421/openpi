#!/usr/bin/env bash
# usage: launch_eval.sh <model> <config> <ckpt_dir> [dependency]   -> LIBERO-90 in 18 jobs of 5 tasks, LIBERO-10 in 2 of 5.
S=/gscratch/scrubbed/sriyash/openpi_pt; MODEL=$1 CFG=$2 CKPT=$3; DEP=${4:+--dependency=afterok:$4}
sub() { sbatch --parsable $DEP -A ckpt-socialrl -p ckpt-all --gres=gpu:1 --constraint="l40s|l40|a40" --exclude=g3121,g3045 \
        -c 16 --mem=96G -t 6:00:00 --requeue -J ev-$MODEL-$1-$2 -o $S/logs/eval_${MODEL}_$1_$2.log $S/jobs/eval_job.sh $MODEL $CFG $CKPT $1 $2 $3; }
J=""; for t in $(seq 0 5 85); do J="$J $(sub libero_90 $t $((t + 5)))"; done
for t in 0 5; do J="$J $(sub libero_10 $t $((t + 5)))"; done
echo "$MODEL:$J" | tee -a $S/jobs/current.txt
