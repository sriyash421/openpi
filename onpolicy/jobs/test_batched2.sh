#!/usr/bin/env bash
T=/gscratch/scrubbed/sriyash/openpi_pt/tmp/batched_test2; rm -rf $T; mkdir -p $T
XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 /gscratch/scrubbed/sriyash/openpi_pt/run.sh "PYTHONPATH=/gscratch/socialrl/sriyash/openpi_pt uv run python -m onpolicy.batched_server --config pi05_libero --dir /gscratch/scrubbed/sriyash/openpi_pt/openpi_data/openpi-assets/checkpoints/pi05_libero --port 8812" > $T/server.log 2>&1 &
SP=$!; for i in $(seq 200); do grep -q "server listening" $T/server.log && break; kill -0 $SP || break; sleep 3; done; echo "server up after $((i*3)) s"; tail -3 $T/server.log
t0=$(date +%s)
for c in $(seq 0 15); do apptainer exec --nv --home /gscratch/scrubbed/sriyash/openpi_pt/home_libero --env NUMBA_CACHE_DIR=/gscratch/scrubbed/sriyash/openpi_pt/xdg_cache/numba,PYTHONDONTWRITEBYTECODE=1 --bind /gscratch/socialrl/sriyash/openpi_pt:/app --bind /gscratch:/gscratch --pwd /app /gscratch/scrubbed/sriyash/openpi_pt/apptainer/libero.sif python onpolicy/rollout_client.py --port 8812 --tasks $c --attempts 1 --out-dir $T/out > $T/client_$c.log 2>&1 & done
wait $(jobs -p | grep -v "^$SP$"); echo "batched: $(($(date +%s) - t0)) s, $(cat $T/out/records_*.jsonl | wc -l) attempts, successes $(cat $T/out/records_*.jsonl | grep -c '\"success\": true')"
grep -h "Traceback\|Error" $T/server.log $T/client_0.log | head -5
kill $SP
