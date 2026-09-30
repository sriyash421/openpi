#!/usr/bin/env bash
/gscratch/scrubbed/sriyash/openpi_pt/run.sh 'PYTHONPATH=/gscratch/socialrl/sriyash/openpi_pt uv run python -c "
import json, time, onpolicy.pairs as p
t = time.time()
print(p.build(json.load(open(\"/gscratch/scrubbed/sriyash/openpi_pt/onpolicy/fb_demo_v1/iter_000/train_records.json\")), \"/gscratch/scrubbed/sriyash/openpi_pt/tmp/pairs_test\", 2048, 0.3, 0))
print(\"secs\", round(time.time() - t, 1))
ds = p.ContextPairDataset(\"/gscratch/scrubbed/sriyash/openpi_pt/tmp/pairs_test\"); x = ds[0]; y = ds[len(ds) - 1]
print(len(ds), {k: getattr(v, \"shape\", v) for k, v in x.items()}, x[\"observation/image\"].mean() > 0, y[\"actions\"].std() > 0)
"'
