#!/usr/bin/env bash
# Same 16 clients against the stock server, then the batched one; report wall time.
T=/gscratch/scrubbed/sriyash/openpi_pt/tmp/batched_test; rm -rf $T; mkdir -p $T
for mode in stock batched; do
  if [ $mode = stock ]; then CMD="uv run scripts/serve_policy.py --port 8811 policy:checkpoint --policy.config=pi05_libero --policy.dir=/gscratch/scrubbed/sriyash/openpi_pt/openpi_data/openpi-assets/checkpoints/pi05_libero";
  else CMD="PYTHONPATH=/gscratch/socialrl/sriyash/openpi_pt uv run python -m onpolicy.batched_server --config pi05_libero --dir /gscratch/scrubbed/sriyash/openpi_pt/openpi_data/openpi-assets/checkpoints/pi05_libero --port 8811"; fi
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 /gscratch/scrubbed/sriyash/openpi_pt/run.sh "$CMD" > $T/server_$mode.log 2>&1 &
  SP=$!; until grep -q "server listening" $T/server_$mode.log; do sleep 3; kill -0 $SP || { tail $T/server_$mode.log; exit 1; }; done
  t0=$(date +%s)
  for c in $(seq 0 15); do apptainer exec --nv --home /gscratch/scrubbed/sriyash/openpi_pt/home_libero --env NUMBA_CACHE_DIR=/gscratch/scrubbed/sriyash/openpi_pt/xdg_cache/numba,PYTHONDONTWRITEBYTECODE=1 --bind /gscratch/socialrl/sriyash/openpi_pt:/app --bind /gscratch:/gscratch --pwd /app /gscratch/scrubbed/sriyash/openpi_pt/apptainer/libero.sif python onpolicy/rollout_client.py --port 8811 --tasks $c --attempts 1 --out-dir $T/$mode > $T/client_${mode}_$c.log 2>&1 & done
  wait $(jobs -p | grep -v "^$SP$"); echo "$mode: $(($(date +%s) - t0)) s, $(cat $T/$mode/records_*.jsonl | wc -l) attempts, successes $(cat $T/$mode/records_*.jsonl | grep -c '\"success\": true')"
  kill $SP; sleep 10
done
grep -h "Traceback\|Error" $T/server_batched.log | head -3
