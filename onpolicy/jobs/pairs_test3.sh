#!/usr/bin/env bash
/gscratch/scrubbed/sriyash/openpi_pt/run.sh 'PYTHONPATH=/gscratch/socialrl/sriyash/openpi_pt uv run python -m onpolicy.pairs --records /gscratch/scrubbed/sriyash/openpi_pt/tmp/recs_test.json --out /gscratch/scrubbed/sriyash/openpi_pt/tmp/pairs_test3 --m 4 --seed 0 --instr-dropout 0.8 && PYTHONPATH=/gscratch/socialrl/sriyash/openpi_pt uv run python -c "
import json, collections
m = [json.loads(l) for l in open(\"/gscratch/scrubbed/sriyash/openpi_pt/tmp/pairs_test3/meta.jsonl\")]
print(len(m), \"pairs\", collections.Counter(x[\"source\"] for x in m), \"| feedback-only prompts:\", sum(x[\"prompt\"].startswith(\"Feedback\") for x in m))
print(m[0][\"prompt\"][:80]); print([x[\"prompt\"][:70] for x in m if x[\"has_context\"]][:2])"'
