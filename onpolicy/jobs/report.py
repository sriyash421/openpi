"""Aggregate eval_passk.py JSONL shards -> one wandb run per model (both suites) with pass@1/pass@8, per-task table, videos.

    python report.py <model> [<model> ...]      (reads $S/eval/<model>/<suite>/results_*.jsonl)
"""
import collections
import glob
import json
import os
import random
import sys

import wandb

S = "/gscratch/scrubbed/sriyash/openpi_pt"
SUITES = ("libero_90", "libero_10")
EXPECTED = {"libero_90": 90 * 5 * 8, "libero_10": 10 * 5 * 8}


def load(model, suite):
    rows = {}
    for p in glob.glob(f"{S}/eval/{model}/{suite}/results_*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            rows[(r["task_id"], r["init"], r["rollout"])] = r  # dedupe requeued episodes
    return list(rows.values())


def summarize(rows):
    by_start = collections.defaultdict(list)
    by_task = collections.defaultdict(list)
    for r in rows:
        by_start[(r["task_id"], r["init"])].append(r["success"])
        by_task[r["task_id"]].append(r)
    pass1 = sum(r["success"] for r in rows) / len(rows)
    pass8 = sum(any(v) for v in by_start.values()) / len(by_start)
    tasks = []
    for t, rs in sorted(by_task.items()):
        starts = collections.defaultdict(list)
        for r in rs:
            starts[r["init"]].append(r["success"])
        tasks.append([t, rs[0]["task"], sum(r["success"] for r in rs) / len(rs), sum(any(v) for v in starts.values()) / len(starts), len(rs)])
    return pass1, pass8, tasks


def main():
    summary = wandb.Table(columns=["model", "suite", "episodes", "pass@1", "pass@8"])
    for model in sys.argv[1:]:
        run = wandb.init(project="openpi-libero", name=f"eval_{model}", job_type="eval", reinit=True)
        for suite in SUITES:
            rows = load(model, suite)
            if not rows:
                continue
            p1, p8, tasks = summarize(rows)
            print(f"{model} {suite}: {len(rows)}/{EXPECTED[suite]} eps  pass@1={p1:.3f}  pass@8={p8:.3f}")
            run.summary.update({f"{suite}/pass@1": p1, f"{suite}/pass@8": p8, f"{suite}/episodes": len(rows)})
            summary.add_data(model, suite, len(rows), p1, p8)
            run.log({f"{suite}/per_task": wandb.Table(columns=["task_id", "task", "pass@1", "pass@8", "episodes"], data=tasks)})
            rng = random.Random(0)
            for tag in ("success", "failure"):
                pick = [r for r in rows if r["success"] == (tag == "success")
                        and os.path.exists(os.path.join(S, "eval", model, suite, "videos", r["video"]))]
                vids = [wandb.Video(os.path.join(S, "eval", model, suite, "videos", r["video"]), caption=f"{r['task']} (init {r['init']}, r{r['rollout']})", format="mp4")
                        for r in rng.sample(pick, min(6, len(pick)))]
                if vids:
                    run.log({f"{suite}/videos_{tag}": vids})
        print(run.url)
        run.finish()
    run = wandb.init(project="openpi-libero", name="summary", job_type="summary", reinit=True)
    run.log({"summary": summary})
    print(run.url)
    run.finish()


if __name__ == "__main__":
    main()
