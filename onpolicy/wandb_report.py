"""Build the W&B Report: baseline pass@k tables + rollout video panels + on-policy feedback chains.

Logs all media into one run ("results_overview") with stable keys, then creates a Report whose panels point at it.
    python -m onpolicy.wandb_report --onpolicy-exp fb_demo_v1
"""
import argparse
import collections
import glob
import json
import os
import random
import shutil

import imageio.v2 as imageio
import numpy as np
import wandb
import wandb_workspaces.reports.v2 as wr

S = "/gscratch/scrubbed/sriyash/openpi_pt"
ENTITY, PROJECT = "sriyash-uw-team", "openpi-libero"
MODELS = {"pi05_base": "π0.5 base (zero-shot)", "pi05_libero": "π0.5 LIBERO (released)", "ours_3k": "ours: LIBERO-90, 3k steps"}
for prefix, label in (("ours_1k", "ours: LIBERO-90 (10k warmup, EMA .999)"), ("ours_w333", "ours: LIBERO-90 (333 warmup, EMA .99)")):
    MODELS.update({f"{prefix}_{s}": f"{label}, {s} steps" for s in sorted(
        int(os.path.basename(p).split("_")[-1]) for p in glob.glob(f"{S}/eval/{prefix}_*"))})
GIF_DIR = f"{S}/report_gifs"


def make_gif(mp4, name, frames=40, scale=2):
    """mp4 -> small looping gif (<=40 frames, half resolution) under GIF_DIR/name."""
    fr = imageio.mimread(mp4, memtest=False)
    idx = np.linspace(0, len(fr) - 1, min(frames, len(fr))).round().astype(int)
    path = f"{GIF_DIR}/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, [fr[i][::scale, ::scale] for i in idx], duration=0.15, loop=0)
    return name


SUITES = {"libero_90": "LIBERO-90 (train tasks)", "libero_10": "LIBERO-10 (held-out)"}


def load_eval(model, suite):
    rows = {}
    for p in glob.glob(f"{S}/eval/{model}/{suite}/results_*.jsonl"):
        for line in open(p):
            r = json.loads(line)
            rows[(r["task_id"], r["init"], r["rollout"])] = r
    return list(rows.values())


def passk(rows):
    starts = collections.defaultdict(list)
    for r in rows:
        starts[(r["task_id"], r["init"])].append(r["success"])
    return sum(r["success"] for r in rows) / len(rows), sum(any(v) for v in starts.values()) / len(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onpolicy-exp", default="fb_demo_v1")
    ap.add_argument("--videos", type=int, default=4)
    ap.add_argument("--chains", type=int, default=6)
    args = ap.parse_args()
    rng = random.Random(0)
    shutil.rmtree(GIF_DIR, ignore_errors=True)
    run = wandb.init(entity=ENTITY, project=PROJECT, name="results_overview", job_type="report")

    # Baselines: pass@k table + success/failure video panels.
    table, media_keys = [], []
    for model, mname in MODELS.items():
        for suite, sname in SUITES.items():
            rows = load_eval(model, suite)
            if not rows:
                continue
            p1, p8 = passk(rows)
            table.append((mname, sname, len(rows), p1, p8))
            vdir = f"{S}/eval/{model}/{suite}/videos"
            for tag in ("success", "failure"):
                pick = [r for r in rows if r["success"] == (tag == "success") and os.path.exists(f"{vdir}/{r['video']}")]
                gifs = [(make_gif(f"{vdir}/{r['video']}", f"baseline/{model}/{suite}/{tag}_{i}.gif"), r["task"])
                        for i, r in enumerate(rng.sample(pick, min(args.videos, len(pick))))]
                if gifs:
                    media_keys.append((f"{mname} · {sname} · {tag}", gifs))

    # On-policy: success by attempt per iteration, feedback chains with videos, full feedback table.
    iters = sorted(glob.glob(f"{S}/onpolicy/{args.onpolicy_exp}/iter_*"))
    curve, chain_keys, fb_rows = [], [], []
    for d in iters:
        recs = [json.loads(l) for p in glob.glob(f"{d}/records_*.jsonl") for l in open(p)]
        if len(recs) < 300:  # skip an in-progress iteration
            continue
        by_k = collections.defaultdict(list)
        for r in recs:
            by_k[r["attempt"]].append(r["success"])
        curve.append((os.path.basename(d), [sum(v) / len(v) for _, v in sorted(by_k.items())], recs))
    if curve:
        name, _, recs = curve[-1]
        by_task = collections.defaultdict(list)
        for r in recs:
            by_task[r["task_id"]].append(r)
        mixed = [t for t, rs in by_task.items() if any(x["success"] for x in rs) and not all(x["success"] for x in rs)]
        failed = [t for t, rs in by_task.items() if not any(x["success"] for x in rs)]
        for t in (mixed[: args.chains // 2] + failed[: args.chains - len(mixed[: args.chains // 2])]):
            rs = sorted(by_task[t], key=lambda x: x["attempt"])
            gifs = [(make_gif(r["video"], f"onpolicy/{name}/task{t:02d}_a{r['attempt']}.gif"),
                     f"attempt {r['attempt']} | sim {'success' if r['success'] else 'failure'} | feedback: {r['feedback']}") for r in rs]
            chain_keys.append((rs[0]["task"], gifs))
        fb_rows = [[r["task_id"], r["task"], r["attempt"], r["success"], r["judge_success"], r["feedback"]] for r in sorted(recs, key=lambda x: (x["task_id"], x["attempt"]))]
        run.log({"onpolicy/feedback_table": wandb.Table(columns=["task_id", "task", "attempt", "sim_success", "judge_success", "feedback"], data=fb_rows)})
    for f in glob.glob(f"{GIF_DIR}/**/*.gif", recursive=True):
        run.save(f, base_path=GIF_DIR, policy="now")
    base_url = f"https://api.wandb.ai/files/{ENTITY}/{PROJECT}/{run.id}"
    run.finish()

    # The report page: gifs inline (hosted as files of the results_overview run).
    md_base = "| model | suite | episodes | pass@1 | pass@8 |\n|---|---|---|---|---|\n" + "\n".join(
        f"| {m} | {s} | {n} | {p1:.3f} | {p8:.3f} |" for m, s, n, p1, p8 in table)
    md_curve = ("| iteration | " + " | ".join(f"attempt {k}" for k in range(len(curve[0][1]))) + " |\n|---|" + "---|" * len(curve[0][1]) + "\n"
                + "\n".join(f"| {n} | " + " | ".join(f"{x:.3f}" for x in c) + " |" for n, c, _ in curve)) if curve else "_no finished iteration yet_"
    blocks = [
        wr.TableOfContents(),
        wr.H1("π0.5 on LIBERO: dataset-only baselines"),
        wr.P("pass@k over the first 5 LIBERO initial states per task × 8 sampled rollouts (LIBERO-90: 3,600 episodes, LIBERO-10: 400). "
             "Ours = openpi pi05_libero recipe on LIBERO-90 (ryanhoangt/libero_90_lerobot_v21), JAX, 2×H200."),
        wr.MarkdownBlock(md_base),
        wr.H2("Rollouts (left: agentview, right: wrist)"),
        *[b for title, gifs in media_keys for b in [wr.P(title), *[wr.Image(url=f"{base_url}/{g}", caption=c) for g, c in gifs]]],
        wr.H1("On-policy with VLM feedback (in progress)"),
        wr.P(f"Experiment {args.onpolicy_exp}: pi05_libero + LoRA; each iteration runs 90 LIBERO-90 tasks × 4 chained attempts. "
             "Attempt k sees attempt k−1 (8 frames tiled in the free image slot) and Gemini-3.8-Flash feedback in the prompt; "
             "training targets are demo actions of the same task. Iteration 0 is the untrained released policy."),
        wr.MarkdownBlock("**Success by attempt index**\n\n" + md_curve),
        wr.H2("Feedback chains (4 attempts of one task, captions carry the judge's feedback)"),
        *[b for task, gifs in chain_keys for b in [wr.P(f"Task: {task}"), *[wr.Image(url=f"{base_url}/{g}", caption=c) for g, c in gifs]]],
    ]
    url_file = f"{S}/report_url.txt"  # one report, updated in place on every rebuild
    if os.path.exists(url_file):
        report = wr.Report.from_url(open(url_file).read().strip())
        report.blocks = blocks
    else:
        report = wr.Report(entity=ENTITY, project=PROJECT, title="π0.5 LIBERO: baselines + on-policy VLM feedback",
                           description="pass@k baselines, rollout videos, and on-policy feedback chains", blocks=blocks)
    report.save()
    open(url_file, "w").write(report.url)
    print("REPORT", report.url)


if __name__ == "__main__":
    main()
