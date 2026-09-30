"""Step 0: judge accuracy on existing eval rollouts with known sim outcomes, plus a sheet of failure end frames.

    python -m onpolicy.validate_judge --eval-dir $S/eval/pi05_libero --n-per-class 60 --out $S/onpolicy/judge_val
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random

import imageio.v2 as imageio
import numpy as np

from onpolicy import context
from onpolicy.judge import Judge


def load_frames(path):
    """eval_passk videos are [agentview | wrist] side by side; the judge sees the agentview half."""
    frames = [f[:, : f.shape[1] // 2] for f in imageio.mimread(path, memtest=False)]
    return context.subsample(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--n-per-class", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for suite in ("libero_90", "libero_10"):
        for p in glob.glob(f"{args.eval_dir}/{suite}/results_*.jsonl"):
            rows += [dict(json.loads(l), suite=suite) for l in open(p)]
    rows = [r for r in rows if os.path.exists(f"{args.eval_dir}/{r['suite']}/videos/{r['video']}")]
    rng = random.Random(0)
    sample = []
    for ok in (True, False):
        pool = [r for r in rows if r["success"] == ok]
        sample += rng.sample(pool, min(args.n_per_class, len(pool)))

    judge = Judge()
    def run(r):
        frames = load_frames(f"{args.eval_dir}/{r['suite']}/videos/{r['video']}")
        return dict(r, judge=judge(r["task"], frames)), frames[-1]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(run, sample))

    with open(f"{args.out}/judged.jsonl", "w") as f:
        for r, _ in results:
            f.write(json.dumps(r) + "\n")
    valid = [r for r, _ in results if r["judge"]["success"] is not None]
    acc = np.mean([r["judge"]["success"] == r["success"] for r in valid])
    tpr = np.mean([r["judge"]["success"] for r in valid if r["success"]])
    tnr = np.mean([not r["judge"]["success"] for r in valid if not r["success"]])
    print(f"judge {judge.model}: n={len(valid)}/{len(results)} acc={acc:.3f} success-recall={tpr:.3f} "
          f"failure-recall={tnr:.3f} cost=${judge.cost:.3f}")
    fails = [(r, last) for r, last in results if not r["success"]][:16]
    sheet = np.concatenate([np.concatenate([last for _, last in fails[i:i + 4]], 1) for i in range(0, 16, 4)], 0)
    imageio.imwrite(f"{args.out}/failure_final_frames.png", sheet)
    for r, _ in fails[:16]:
        print(f"[fail] {r['task']!r} -> {r['judge']['feedback']!r}")

    if args.wandb:
        import wandb
        run = wandb.init(project="openpi-libero", name="judge_validation", job_type="judge")
        run.summary.update({"judge/acc": acc, "judge/success_recall": tpr, "judge/failure_recall": tnr, "judge/cost": judge.cost})
        run.log({"judge/table": wandb.Table(columns=["suite", "task", "sim_success", "judge_success", "feedback"],
                                            data=[[r["suite"], r["task"], r["success"], r["judge"]["success"], r["judge"]["feedback"]] for r in valid]),
                 "judge/failure_final_frames": wandb.Image(sheet)})
        print(run.url)
        run.finish()


if __name__ == "__main__":
    main()
