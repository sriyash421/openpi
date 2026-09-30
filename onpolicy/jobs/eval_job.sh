#!/usr/bin/env bash
# One GPU: stock openpi policy server (openpi.sif) + N LIBERO clients (libero.sif) over tasks [T0,T1).
# usage: eval_job.sh <model> <config> <ckpt_dir> <suite> <T0> <T1> [clients=8]
set -uo pipefail
S=/gscratch/scrubbed/sriyash/openpi_pt; O=/gscratch/socialrl/sriyash/openpi_pt
MODEL=$1 CFG=$2 CKPT=$3 SUITE=$4 T0=$5 T1=$6 NC=${7:-8}
PORT=$((8000 + ${SLURM_JOB_ID:-$$} % 1000)); OUT=$S/eval/$MODEL/$SUITE; mkdir -p $OUT
$S/run.sh "uv run scripts/serve_policy.py --port $PORT policy:checkpoint --policy.config=$CFG --policy.dir=$CKPT" > $OUT/server_${T0}_${T1}.log 2>&1 &
SERVER=$!
for _ in $(seq 1800); do grep -q "server listening\|Serving\|listening on" $OUT/server_${T0}_${T1}.log 2>/dev/null && break; kill -0 $SERVER 2>/dev/null || { tail -20 $OUT/server_${T0}_${T1}.log; exit 1; }; sleep 2; done
for c in $(seq 0 $((NC - 1))); do
  apptainer exec --nv --home $S/home_libero --env NUMBA_CACHE_DIR=$S/xdg_cache/numba,PYTHONDONTWRITEBYTECODE=1,XDG_CACHE_HOME=$S/xdg_cache --bind $O:/app --bind /gscratch:/gscratch --pwd /app/examples/libero $S/apptainer/libero.sif \
    python eval_passk.py --args.port $PORT --args.task-suite-name $SUITE --args.task-start $T0 --args.task-end $T1 \
    --args.shard $c --args.num-shards $NC --args.out-dir $OUT > $OUT/client_${T0}_${T1}_$c.log 2>&1 &
done
wait $(jobs -p | grep -v "^$SERVER$")
kill $SERVER; echo "EVAL_DONE $MODEL $SUITE $T0-$T1: $(cat $OUT/results_${T0}_${T1}_shard*.jsonl 2>/dev/null | wc -l) episodes"
