"""Clean W&B report of all current π0.5-on-LIBERO results (offline baselines, on-policy VLM feedback, ablations,
failure localization). Every number is recomputed from the eval / rollout records on disk.

Media (plots, GIFs, contact sheets) are uploaded as files of one run ("results_clean") and embedded by URL, as in
onpolicy/wandb_report.py. Writes a NEW report (URL -> $S/report_clean_url.txt; updated in place on reruns).

    source /gscratch/scrubbed/sriyash/vla/.venv/bin/activate
    python -m onpolicy.wandb_report_clean            # --no-media skips GIF re-encoding
"""
import argparse
import collections
import glob
import json
import os
import shutil

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wandb  # noqa: E402
import wandb_workspaces.reports.v2 as wr  # noqa: E402

S = "/gscratch/scrubbed/sriyash/openpi_pt"
OP = f"{S}/onpolicy"
ENTITY, PROJECT, TRAIN_PROJECT = "sriyash-uw-team", "openpi-libero", "openpi"
MEDIA = f"{S}/report_clean_media"
URL_FILE = f"{S}/report_clean_url.txt"
EXCLUDED = "pick up the book and place it in the back compartment of the caddy"  # LIBERO-10 task that is also in LIBERO-90
EXP = "async_b128_draft01"
# W&B runs: trainer runs live in project "openpi", rollout loggers in "openpi-libero".
TRAIN_RUNS = {"qkj2wmcw": "offline 3k (warmup 10k, EMA .999)", "atjftuid": "offline 1k (warmup 10k, EMA .999)",
              "r4nrz47a": "offline w333 (warmup 333, EMA .99)", "zi2lyssd": "on-policy async_b128_draft01 (trainer)",
              "onlfy0d0": "on-policy fb_w333_s200_anchor (trainer)"}
ROLLOUT_RUN = "juelyp9i"
L10_STEPS = {"b128_round15": 2000, "b128_round30": 3000, "b128_round45": 5000, "b128_round60": 7000,
             "b128_round75": 9000, "b128_round90": 11000}  # dir name -> checkpoint step (read from server logs)
MODES = ["real", "real_noinstr", "shuffled", "empty"]
COLORS = {"real": "#2a78d6", "real_noinstr": "#eb6834", "shuffled": "#1baf7a", "empty": "#eda100"}
INK, MUTED = "#0b0b0b", "#52514e"
SHEETS = [("r080_task64_c80064-0_a2", "VLM right, DTW late: the policy reaches for the wrong (left) bowl at t≈12 (VLM, red); "
           "DTW only diverges at t=248 (blue)."),
          ("r091_task41_c91041-0_a0", "DTW right: fumbling grasp of the frying pan; DTW diverges at t=80, the VLM points at t≈338."),
          ("r093_task72_c93072-0_a0", "Mug never released (stall at t=222); DTW (t=209) is closer than the VLM (t≈288)."),
          ("r090_task24_c90024-0_a1", "DTW false alarm: DTW diverges at t=17 while the visible error (failed bowl grasp) is at "
           "t≈275 (VLM).")]


# ----------------------------------------------------------------------------- data
def jsonl(pattern):
    for p in sorted(glob.glob(pattern)):
        for line in open(p):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def offline_eval(model, suite):
    rows = {(r["task_id"], r["init"], r["rollout"]): r for r in jsonl(f"{S}/eval/{model}/{suite}/results_*.jsonl")}
    return list(rows.values())


def passk(rows):
    """pass@1 = mean episode success; pass@8 = fraction of (task, init) starts where any of 8 rollouts succeeds."""
    if not rows:
        return None
    starts = collections.defaultdict(list)
    for r in rows:
        starts[(r["task_id"], r["init"])].append(r["success"])
    return {"n": len(rows), "p1": sum(r["success"] for r in rows) / len(rows), "p8": sum(any(v) for v in starts.values()) / len(starts)}


def chains_of(d):
    ch = collections.defaultdict(dict)
    for r in jsonl(f"{d}/records_*.jsonl"):
        ch[r["chain"]][r["attempt"]] = r
    return ch


def chain_metrics(ch, k, keep=lambda task: True):
    """pass@k: any of the k attempts succeeds; perf@k: attempt k succeeds; P(succ | prev failed) over attempts 1..k-1."""
    full = [c for c in ch.values() if len(c) == k and keep(c[0]["task"])]
    if not full:
        return None
    succ = [[c[a]["success"] for a in range(k)] for c in full]
    after_fail = [s[a] for s in succ for a in range(1, k) if not s[a - 1]]
    return {"chains": len(full), "pass": float(np.mean([any(s) for s in succ])), "perf": float(np.mean([s[-1] for s in succ])),
            "by_attempt": [float(np.mean([s[a] for s in succ])) for a in range(k)],
            "p_after_fail": float(np.mean(after_fail)) if after_fail else float("nan"), "n_after_fail": len(after_fail)}


def compute():
    D = {"offline": {}, "rollout": [], "l10": {}, "l10_ref": {}, "l10_ctrl": {}, "l90": {}}
    for p in sorted(glob.glob(f"{S}/eval/*")):
        m = os.path.basename(p)
        if m.startswith("smoke"):
            continue
        l90, l10 = offline_eval(m, "libero_90"), offline_eval(m, "libero_10")
        D["offline"][m] = {"l90": passk(l90), "l10": passk(l10), "l10_h9": passk([r for r in l10 if r["task"] != EXCLUDED])}
    for d in sorted(glob.glob(f"{OP}/{EXP}/round_*")):
        recs = list(jsonl(f"{d}/records_*.jsonl"))
        if not os.path.exists(f"{d}/DONE") or not recs:
            continue
        by = collections.defaultdict(list)
        for r in recs:
            by[r["attempt"]].append(r["success"])
        D["rollout"].append({"round": int(d.split("_")[-1]), "policy": open(f"{d}/policy.txt").read().split()[-1],
                             "by_attempt": [float(np.mean(v)) for _, v in sorted(by.items())], "n": len(recs)})
    held = lambda t: t != EXCLUDED  # noqa: E731
    for name, dct in (("l10", L10_STEPS), ("l10_ref", {"base_step200": "w333 step 200 (on-policy init)",
                                                         "offline_old_3k": "offline 3k", "offline_w333_1000": "offline w333 step 1000",
                                                         "onpolicy_11400": "fb_w333_s200_anchor step 11400 (instr. dropout 0.25)"})):
        for sub, label in dct.items():
            for mode in MODES:
                d = f"{OP}/eval_libero10_k8/{sub}/{mode}"
                if os.path.isdir(d):
                    ch = chains_of(d)
                    D[name].setdefault(sub, {"label": label})[mode] = {"all10": chain_metrics(ch, 8), "held9": chain_metrics(ch, 8, held)}
    # Step-7000 L10 controls. textonly in b128_round60_ctrl* is invalid (judge budget cap hit); its rerun is b128_round60_text.
    # A mode counts as finished once its eval job wrote summary.json or all 80 chains are complete.
    for sub, mode in (("b128_round60_ctrl", "generic"), ("b128_round60_ctrl_preempt1", "generic"), ("b128_round60_ctrl", "null"),
                      ("b128_round60_text", "textonly")):
        d = f"{OP}/eval_libero10_k8/{sub}/{mode}"
        if os.path.isdir(d):
            ch = chains_of(d)
            m = {"records": sum(len(c) for c in ch.values()), "all10": chain_metrics(ch, 8), "held9": chain_metrics(ch, 8, held)}
            summ = f"{OP}/eval_libero10_k8/{sub}/summary.json"
            m["done"] = bool(m["all10"] and m["all10"]["chains"] == 80) or (os.path.exists(summ) and mode in json.load(open(summ)))
            D["l10_ctrl"][f"{sub}/{mode}"] = m
    # L90 step-7000 ablations. textonly: only the rerun dir (b128_7000_text) counts.
    for mode, sub in (("real", "b128_7000"), ("generic", "b128_7000_ctrl"), ("null", "b128_7000_ctrl"),
                      ("shuffled", "b128_7000_cf"), ("empty", "b128_7000_cf"), ("textonly", "b128_7000_text")):
        d = f"{OP}/eval_libero90_diag/{sub}/{mode}"
        if os.path.isdir(d):
            ch = chains_of(d)
            D["l90"][mode] = chain_metrics(ch, 3) or {}
            summ = f"{OP}/eval_libero90_diag/{sub}/summary.json"
            D["l90"][mode].update(dir=f"{sub}/{mode}", records=sum(len(c) for c in ch.values()),
                                  done=mode != "textonly" or (os.path.exists(summ) and mode in json.load(open(summ))))
    loc = f"{OP}/localize_test/demos_mp4"
    D["localize_stats"] = open(f"{loc}/stats.txt").read()
    D["localize"] = {r["id"].rsplit("_", 1)[0]: {k: r.get(k) for k in ("task", "t0_dtw", "stall", "t_vlm1", "t_vlm2", "T", "feedback", "vlm1_error", "sheet")}
                     for r in jsonl(f"{loc}/results.jsonl")}
    costs = [float(line.split()[-1]) for line in open(f"{S}/judge_ledger.txt") if line.strip()]
    D["judge"] = {"total": sum(costs), "calls": len(costs)}
    return D


# ----------------------------------------------------------------------------- media
def style(ax, title, ylabel, xlabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors=MUTED)


def save_fig(fig, name):
    os.makedirs(MEDIA, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"{MEDIA}/{name}", dpi=130, facecolor="white")
    plt.close(fig)
    return name


def plots(D):
    out = {}
    # Offline w333 curve (and the 10k-warmup run for contrast).
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, suite, title in ((axes[0], "l90", "LIBERO-90 (train tasks)"), (axes[1], "l10_h9", "LIBERO-10 held-out9")):
        for prefix, lab, c in (("ours_w333", "w333 (warmup 333, EMA .99)", "#2a78d6"), ("ours_1k", "1k (warmup 10k, EMA .999)", "#eb6834")):
            steps = sorted(int(m.split("_")[-1]) for m in D["offline"] if m.startswith(prefix + "_"))
            for key, ls in (("p8", "-"), ("p1", "--")):
                ys = [D["offline"][f"{prefix}_{s}"][suite][key] for s in steps]
                ax.plot(steps, ys, ls, color=c, lw=2, marker="o", ms=4, label=f"{lab} · {'pass@8' if key == 'p8' else 'pass@1'}")
        style(ax, title, "success", "training step")
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=7.5, frameon=False, loc="center right")
    out["offline_curve"] = save_fig(fig, "offline_curve.png")

    # Training-time rollout success by attempt.
    R = D["rollout"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    x = [r["round"] for r in R]
    for a, (lab, c) in enumerate((("attempt 0 (no context)", "#2a78d6"), ("attempt 1 (retry, feedback)", "#eb6834"), ("attempt 2 (retry, feedback)", "#1baf7a"))):
        ax.plot(x, [r["by_attempt"][a] for r in R], color=c, lw=2, label=lab)
    changes = [r["round"] for i, r in enumerate(R) if i and r["policy"] != R[i - 1]["policy"]]
    for rd in changes:
        ax.axvline(rd, color="#e6e5e1", lw=0.8, zorder=0)
    style(ax, f"{EXP}: LIBERO-90 rollout success by attempt (90 chains / round; thin lines = new policy checkpoint)", "success", "rollout round")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    out["rollout_curve"] = save_fig(fig, "rollout_curve.png")

    # Held-out L10 pass@8 / perf@8 vs step per context mode.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    subs = sorted(D["l10"], key=lambda s: L10_STEPS[s])
    ref = D["l10_ref"].get("offline_w333_1000", {}).get("empty", {}).get("held9")
    for ax, key, title in ((axes[0], "pass", "held-out9 pass@8"), (axes[1], "perf", "held-out9 perf@8 (8th attempt)")):
        for mode in MODES:
            ax.plot([L10_STEPS[s] for s in subs], [D["l10"][s][mode]["held9"][key] for s in subs], color=COLORS[mode], lw=2, marker="o", ms=5, label=mode)
        if ref:
            ax.axhline(ref[key], color=MUTED, ls="--", lw=1.2, label="offline w333-1000 (no context)")
        style(ax, title, "success", "on-policy training step")
        ax.set_ylim(0, 0.45)
    axes[0].legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    out["l10_curve"] = save_fig(fig, "l10_curve.png")

    # L90 ablation: P(success | previous attempt failed).
    order = [m for m in ("real", "generic", "textonly", "null", "shuffled", "empty") if D["l90"].get(m, {}).get("done") and "p_after_fail" in D["l90"][m]]
    labels = {"real": "real (own frames + VLM feedback)", "generic": "generic (own frames + 'previous attempt failed')",
              "null": "null (blank frames + generic text)", "textonly": "textonly (blank frames + VLM feedback)", "shuffled": "shuffled (other task's attempt)", "empty": "empty (no context)"}
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 0.5))
    vals = [D["l90"][m]["p_after_fail"] for m in order]
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, height=0.6, color="#2a78d6")
    for yi, m, v in zip(y, order, vals):
        ax.text(v + 0.01, yi, f"{v:.3f}  (n={D['l90'][m]['n_after_fail']})", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [labels[m] for m in order], fontsize=9)
    style(ax, "LIBERO-90, step 7000: P(success | previous attempt failed)", "", "")
    ax.grid(axis="x", color="#e6e5e1", lw=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    out["l90_bars"] = save_fig(fig, "l90_bars.png")
    return out


def make_gif(mp4, name, frames=48, scale=2):
    fr = imageio.mimread(mp4, memtest=False)
    idx = np.linspace(0, len(fr) - 1, min(frames, len(fr))).round().astype(int)
    path = f"{MEDIA}/{name}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimwrite(path, [fr[i][::scale, ::scale] for i in idx], duration=0.12, loop=0)
    return name


QUAL = [("LIBERO-10 held-out, step 7000", f"{OP}/eval_libero10_k8/b128_round60/real", ["12-1", "36-0"], 4),
        ("LIBERO-90, step 7000", f"{OP}/eval_libero90_diag/b128_7000/real", ["0-0", "10-1", "106-0"], 2)]


def gifs(encode):
    out = []
    for group, d, chain_ids, n in QUAL:
        ch = chains_of(d)
        for cid in chain_ids:
            c = ch[cid]
            items = []
            for a in range(n):
                r = c[a]
                name = f"qual/{os.path.basename(os.path.dirname(d))}_{cid}_a{a}.gif"
                if encode or not os.path.exists(f"{MEDIA}/{name}"):
                    make_gif(r["video"], name)
                verdict = "SUCCESS" if r["success"] else "FAIL"
                cap = f"attempt {a}: {verdict}" + ("" if r["success"] else f" — VLM feedback → next attempt: “{r['feedback']}”")
                items.append((name, cap))
            out.append((group, c[0]["task"], items))
    return out


def sheets():
    out = []
    for sid, cap in SHEETS:
        src = f"{OP}/localize_test/demos_mp4/sheets/{sid}_failure.png"
        name = f"sheets/{sid}.png"
        os.makedirs(f"{MEDIA}/sheets", exist_ok=True)
        shutil.copy(src, f"{MEDIA}/{name}")
        out.append((name, cap))
    return out


# ----------------------------------------------------------------------------- report
def f3(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def md_table(header, rows):
    return "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)


def blocks_for(D, P, G, SH, base, api):
    off = D["offline"]
    names = [("pi05_base", "π0.5 base (zero-shot)"), ("pi05_libero", "π0.5 LIBERO (released; trained on LIBERO-10 ⇒ contaminated)"),
             ("ours_3k", "ours offline 3k (warmup 10k, EMA .999)"), ("ours_1k_1000", "ours 1k, step 1000 (warmup 10k)"),
             ("ours_w333_200", "ours w333, step 200 (= on-policy init)"), ("ours_w333_1000", "ours w333, step 1000")]
    base_rows = [[lab, f3(off[m]["l90"]["p1"]), f3(off[m]["l90"]["p8"]), f3(off[m]["l10"]["p1"]), f3(off[m]["l10"]["p8"]),
                  f3(off[m]["l10_h9"]["p1"]), f3(off[m]["l10_h9"]["p8"])] for m, lab in names if m in off]
    w333 = sorted(int(m.split("_")[-1]) for m in off if m.startswith("ours_w333_"))
    w333_rows = [[s, f3(off[f"ours_w333_{s}"]["l90"]["p1"]), f3(off[f"ours_w333_{s}"]["l90"]["p8"]),
                  f3(off[f"ours_w333_{s}"]["l10_h9"]["p1"]), f3(off[f"ours_w333_{s}"]["l10_h9"]["p8"]),
                  f3(off.get(f"ours_1k_{s}", {}).get("l90", {}).get("p8") if off.get(f"ours_1k_{s}") else None)] for s in w333]

    R = D["rollout"]
    late = [r for r in R if r["round"] >= 30]
    a0 = [r["by_attempt"][0] for r in late]
    rt = [x for r in late for x in r["by_attempt"][1:]]
    early = [r for r in R if r["policy"].endswith("/200")]
    ro_rows = [[r["round"], r["policy"].split("/")[-1]] + [f3(x) for x in r["by_attempt"]] for r in R if r["round"] % 10 == 0 or r["round"] == R[-1]["round"]]

    l10 = D["l10"]
    subs = sorted(l10, key=lambda s: L10_STEPS[s])
    l10_rows = []
    for s in subs:
        row = [L10_STEPS[s]] + [f"{f3(l10[s][m]['held9']['pass'])} / {f3(l10[s][m]['held9']['perf'])}" for m in MODES]
        row.append(" / ".join(f3(l10[s][m]["all10"]["pass"]) for m in MODES))
        l10_rows.append(row)
    ref_rows = []
    for s, d in D["l10_ref"].items():
        ref_rows.append([d["label"]] + [f"{f3(d[m]['held9']['pass'])} / {f3(d[m]['held9']['perf'])}" if m in d else "–" for m in MODES]
                        + [" / ".join(f3(d[m]["all10"]["pass"]) if m in d else "–" for m in MODES)])
    ctrl = D["l10_ctrl"]

    s7 = "b128_round60"
    # Step-7000 held-out L10 rows: (label, metrics dict with held9/all10, finished?, key for progress)
    l10_7000 = [("real", l10[s7]["real"], True, None), ("real_noinstr", l10[s7]["real_noinstr"], True, None)]
    for lab, key in (("generic", "b128_round60_ctrl/generic"), ("generic (replicate, earlier preempted job)", "b128_round60_ctrl_preempt1/generic"),
                     ("null", "b128_round60_ctrl/null"), ("textonly", "b128_round60_text/textonly")):
        c = ctrl.get(key)
        l10_7000.append((lab, c, bool(c and c["done"] and c["held9"]), key))
    l10_7000 += [("shuffled", l10[s7]["shuffled"], True, None), ("empty", l10[s7]["empty"], True, None)]
    ctrl_rows = []
    for lab, c, ok, key in l10_7000:
        if ok:
            h = c["held9"]
            ctrl_rows.append([lab, f3(h["pass"]), f3(h["perf"]), f"**{f3(h['p_after_fail'])}**", h["n_after_fail"], f3(c["all10"]["pass"])])
        else:
            n = c["records"] if c else 0
            ctrl_rows.append([lab, ("rerun in progress" if "textonly" in key else "pending") + f" ({n}/640 episodes)", "–", "–", "–", "–"])
    l10pf = {lab.split(" ")[0]: c["held9"]["p_after_fail"] for lab, c, ok, _ in l10_7000 if ok and not lab.startswith("generic (")}

    L = D["l90"]
    l90_desc = {"real": "own previous attempt (frames) + VLM feedback", "generic": "own frames, feedback replaced by “The previous attempt failed.”",
                "null": "blank frames + generic text (context flag on)", "textonly": "blank frames + VLM feedback text", "shuffled": "another task's attempt + its feedback", "empty": "no context"}
    l90_rows = [[m, l90_desc[m], L[m]["chains"], " / ".join(f3(x) for x in L[m]["by_attempt"]), f"**{f3(L[m]['p_after_fail'])}**", L[m]["n_after_fail"], f"`{L[m]['dir']}`"]
                for m in ("real", "generic", "textonly", "null", "shuffled", "empty") if L.get(m, {}).get("done") and "p_after_fail" in L[m]]
    if not L.get("textonly", {}).get("done"):
        l90_rows.append(["textonly", l90_desc["textonly"], f"rerun in progress ({L.get('textonly', {}).get('records', 0)}/1080 episodes)", "–", "–", "–",
                         "`b128_7000_text/textonly`"])
    pf = {m: v["p_after_fail"] for m, v in L.items() if v.get("done") and "p_after_fail" in v}

    loc = D["localize_stats"]
    j = D["judge"]
    rrun = api.run(f"{ENTITY}/{PROJECT}/{ROLLOUT_RUN}")
    trun = api.run(f"{ENTITY}/{TRAIN_PROJECT}/zi2lyssd")
    exp_cost = sum(float(h.get("judge/cost") or 0) for h in rrun.scan_history(keys=["judge/cost"]))

    def link(rid, proj=TRAIN_PROJECT):
        return f"https://wandb.ai/{ENTITY}/{proj}/runs/{rid}"

    run_links = "\n".join(f"- [{lab}]({link(rid)}) (`{rid}`)" for rid, lab in TRAIN_RUNS.items()) + \
        f"\n- [{EXP} rollouts]({link(ROLLOUT_RUN, PROJECT)}) (`{ROLLOUT_RUN}`)"

    img = lambda name, cap="": wr.Image(url=f"{base}/{name}", caption=cap)  # noqa: E731
    B = [
        wr.TableOfContents(),
        wr.H1("TL;DR"),
        wr.MarkdownBlock(
            f"- **Offline**: fine-tuning π0.5 base on LIBERO-90 reaches L90 pass@1 {f3(off['ours_w333_1000']['l90']['p1'])} / pass@8 "
            f"{f3(off['ours_w333_1000']['l90']['p8'])} in 1k steps (warmup 333, EMA .99), but only {f3(off['ours_w333_1000']['l10_h9']['p8'])} pass@8 "
            "on the 9 held-out LIBERO-10 tasks.\n"
            f"- **On-policy VLM feedback (async_b128_draft01)**: on LIBERO-90 the model learns to use feedback — retries succeed "
            f"{min(rt):.2f}–{max(rt):.2f} vs {min(a0):.2f}–{max(a0):.2f} for first attempts (rounds ≥30).\n"
            f"- **Ablation at step 7000 (L90)**: P(success | previous failed) real {f3(pf.get('real'))} > generic {f3(pf.get('generic'))} > null "
            f"{f3(pf.get('null'))} > shuffled {f3(pf.get('shuffled'))} ≈ empty {f3(pf.get('empty'))}: the context slot, the model's own frames, "
            "and the correction text each add ≈0.16–0.24.\n"
            f"- **No transfer to held-out LIBERO-10**: held-out9 pass@8 with real feedback peaks at {max(l10[s]['real']['held9']['pass'] for s in subs):.3f} and is "
            "never better than the no-context (empty) or wrong-context (shuffled) controls or the offline w333-1000 policy "
            f"({f3(D['l10_ref']['offline_w333_1000']['empty']['held9']['pass'])}).\n"
            "- **Failure localization**: DTW-to-demo divergence detects 98% of failures (23% false alarms on successes); the VLM's error step is "
            "self-consistent (median 7 steps) but agrees with DTW only loosely (median 56 steps) ⇒ motivates retrieval-based targeted supervision."),

        wr.H1("1. Setup"),
        wr.MarkdownBlock(
            "- **Policy**: π0.5 base (openpi), fine-tuned on **LIBERO-90** (`ryanhoangt/libero_90_lerobot_v21`) with the `pi05_libero` recipe, JAX, 2×H200.\n"
            "- **Eval protocol**: 10 no-op wait steps, replan every 5 actions, horizon 400 steps (LIBERO-90) / 520 (LIBERO-10).\n"
            "  - *Offline baselines*: first 5 initial states per task × 8 sampled rollouts; **pass@1** = mean episode success, **pass@8** = any of the 8 rollouts from a start succeeds "
            "(LIBERO-90: 3,600 episodes, LIBERO-10: 400).\n"
            "  - *Chain evals (on-policy)*: a chain is k consecutive attempts from one random init, attempt i sees attempt i−1 as context; **pass@k** = any attempt "
            "succeeds, **perf@k** = the k-th attempt succeeds. LIBERO-10: 8 chains/task × 8 attempts; LIBERO-90: 4 chains/task × 3 attempts.\n"
            f"- **Held-out** = LIBERO-10. One L10 instruction (“{EXCLUDED}”) also appears in LIBERO-90 and is excluded ⇒ **held-out9** (9 tasks); **all10** reported alongside.\n"
            "- ⚠️ The released `pi05_libero` checkpoint was trained on LIBERO-10 data, so its L10 numbers are contaminated (not a held-out result)."),

        wr.H1("2. Offline baselines"),
        wr.MarkdownBlock("**pass@k (offline protocol)**\n\n" + md_table(
            ["model", "L90 pass@1", "L90 pass@8", "L10 all10 pass@1", "L10 all10 pass@8", "L10 held-out9 pass@1", "L10 held-out9 pass@8"], base_rows)),
        wr.P("The 10k-warmup run (EMA .999) lags badly at 1k steps (L90 pass@8 0.24) — warmup + slow EMA, not a data problem. "
             "With warmup 333 / EMA .99 (checkpoint every 100 steps) L90 is essentially solved by step 400–500, while held-out L10 stays ≤0.36 pass@8."),
        img(P["offline_curve"], "Offline success vs training step. Solid = pass@8, dashed = pass@1."),
        wr.MarkdownBlock("**w333 per-step curve** (last column: 10k-warmup run L90 pass@8 at the same step)\n\n" + md_table(
            ["step", "L90 pass@1", "L90 pass@8", "held-out9 pass@1", "held-out9 pass@8", "1k(10k warmup) L90 pass@8"], w333_rows)),
        wr.H2("Training loss curves"),
        wr.PanelGrid(runsets=[wr.Runset(entity=ENTITY, project=TRAIN_PROJECT, name="training runs",
                                        filters="ID in " + repr(list(TRAIN_RUNS)))],
                     panels=[wr.LinePlot(title="loss", x="Step", y=["loss"], smoothing_factor=0.8, layout=wr.Layout(x=0, y=0, w=12, h=8)),
                             wr.LinePlot(title="grad_norm", x="Step", y=["grad_norm"], smoothing_factor=0.8, layout=wr.Layout(x=12, y=0, w=12, h=8))]),
        wr.MarkdownBlock("Runs:\n" + run_links),

        wr.H1("3. On-policy method (VLM feedback)"),
        wr.MarkdownBlock(
            "```\n"
            "rollout workers (3× L40S)                         trainer (2× H200)\n"
            "  chain per task:  r0 → f0 → r1 → f1 → r2          LoRA on π0.5 (w333 step-200 init)\n"
            "  r_i  = rollout with context (r_{i-1}, f_{i-1})    samples (context, demo (s*, a*)) rows\n"
            "  f_i  = Gemini-3.8-Flash feedback on failures      publishes checkpoints every 1k steps\n"
            "         (success ⇒ “The task was completed correctly.”)\n"
            "            └──────── records + frames ───────►  pool  ◄──── expert demos (same task)\n"
            "```\n"
            "- **Context** = previous attempt: 8 uniformly sampled agentview frames tiled 2×4 into the unused right-wrist image slot, plus the VLM feedback "
            "text in the prompt. The VLM (`google/gemini-3.8-flash` via OpenRouter) is only called on failures.\n"
            "- **Targets** = expert demo (s*, a*) of the same task from the expert pool (not the policy's own actions).\n"
            "- **Instruction dropout 0.8** on context rows (forces the model to read the feedback/frames). LoRA adapters; batch 128.\n"
            "- **Anchor**: rows without context are held at the frozen base (LoRA off) by velocity matching at the same (x_t, t), DiT-style "
            "(anchor β = 0.9 + 0.1 × flow loss to the demo: “draft01”).\n"
            "- **Async**: rollouts (3× L40S) and trainer (2× H200) run concurrently; each round uses the newest published checkpoint."),

        wr.H1(f"4. Latest on-policy run: {EXP}"),
        wr.H2("Training-time rollouts (LIBERO-90)"),
        img(P["rollout_curve"]),
        wr.P(f"Rounds ≥30: attempt 0 {min(a0):.2f}–{max(a0):.2f} (mean {np.mean(a0):.2f}); retries {min(rt):.2f}–{max(rt):.2f} (mean {np.mean(rt):.2f}). "
             f"Rounds 0–{early[-1]['round'] if early else '?'} are served by the untrained init (w333 step 200, never saw context) and retries are *worse* than first attempts "
             "(≈0.1–0.19). Retry success jumps at round 11, right after the first on-policy checkpoint (step 1000) is picked up."),
        wr.MarkdownBlock("**Success by attempt (every 10th round)**\n\n" + md_table(["round", "policy ckpt", "attempt 0", "attempt 1", "attempt 2"], ro_rows)),
        wr.PanelGrid(runsets=[wr.Runset(entity=ENTITY, project=PROJECT, name="rollouts", filters=f"ID in ['{ROLLOUT_RUN}']")],
                     panels=[wr.LinePlot(title="rollout success by attempt (live)", x="iteration",
                                         y=["rollout/success_attempt0", "rollout/success_attempt1", "rollout/success_attempt2"], layout=wr.Layout(x=0, y=0, w=12, h=8)),
                             wr.LinePlot(title="judge cost per round ($)", x="iteration", y=["judge/cost"], layout=wr.Layout(x=12, y=0, w=12, h=8))]),
        wr.H2("Held-out LIBERO-10 (k = 8 chains × 8 attempts per task)"),
        wr.P("Context modes at eval: real = own previous attempt + feedback; real_noinstr = same but instruction removed; shuffled = another task's attempt "
             "and feedback; empty = no context (plain policy)."),
        img(P["l10_curve"], "Held-out9, per checkpoint step. Dashed = offline w333 step 1000 with no context."),
        wr.MarkdownBlock("**held-out9 pass@8 / perf@8** (72 chains each; last column: all10 pass@8 real / noinstr / shuffled / empty, 80 chains)\n\n" + md_table(
            ["step", "real", "real_noinstr", "shuffled", "empty", "all10 pass@8"], l10_rows)),
        wr.MarkdownBlock("**Reference / earlier checkpoints, same chain protocol**\n\n" + md_table(
            ["checkpoint", "real", "real_noinstr", "shuffled", "empty", "all10 pass@8"], ref_rows)),
        wr.P("No setting where real feedback beats the controls on held-out tasks: the policy's retries on L10 are no better than independent re-draws."),

        wr.H1("5. Transfer analysis / ablations (LIBERO-90, step 7000)"),
        wr.P("Chains of 3 attempts, 4 per task (up to 360 chains); only chains with all 3 attempts. The key statistic is P(success | previous attempt failed)."),
        wr.MarkdownBlock(md_table(["mode", "context", "chains", "success by attempt 0/1/2", "P(succ | prev failed)", "n", "dir"], l90_rows)),
        img(P["l90_bars"]),
        wr.MarkdownBlock(
            f"- **Slot on** (null − empty): +{pf['null'] - pf['empty']:.2f} — just having the context flag/format on raises retry success.\n"
            f"- **Own frames** (generic − null): +{pf['generic'] - pf['null']:.2f}.\n"
            f"- **Correction text** (real − generic): +{pf['real'] - pf['generic']:.2f}.\n"
            f"- **Wrong-task context hurts**: shuffled {f3(pf['shuffled'])} ≈ empty {f3(pf['empty'])}, far below the null slot ({f3(pf['null'])}).\n"
            "- ⇒ On training tasks the model genuinely uses frames + feedback; the same mechanism does **not** transfer to held-out LIBERO-10 (section 4).\n"
            + (f"- **Text alone** (textonly − null): {pf['textonly'] - pf['null']:+.2f}; frames + text (real) vs text alone: {pf['real'] - pf['textonly']:+.2f}.\n"
               if L.get("textonly", {}).get("done") else
               "- `textonly` (VLM feedback text, blank frames): the first run was invalid (judge budget cap hit mid-run) — **rerun in progress**.\n")),
        wr.H2("Held-out LIBERO-10 controls at step 7000"),
        wr.P("Same checkpoint, same context modes, on the 9 held-out LIBERO-10 tasks (8 chains × 8 attempts per task; 72 held-out9 chains). "
             "P(success | previous attempt failed) is computed over attempts 1–7 of held-out chains."),
        wr.MarkdownBlock(md_table(["mode", "held-out9 pass@8", "held-out9 perf@8", "P(succ | prev failed)", "n", "all10 pass@8"], ctrl_rows)),
        wr.MarkdownBlock("**L90 vs held-out L10, P(success | previous attempt failed)**\n\n" + md_table(
            ["mode", "LIBERO-90 (train tasks)", "LIBERO-10 held-out9"],
            [[m, f3(pf.get(m)), f3(l10pf.get(m))] for m in ("real", "generic", "textonly", "null", "shuffled", "empty") if m in pf or m in l10pf])
            + "\n\n" + (
            f"- **Slot on** (null − empty): L90 {pf['null'] - pf['empty']:+.2f} vs L10 {l10pf['null'] - l10pf['empty']:+.3f} — on held-out tasks switching the "
            "context slot on *hurts* (null < empty).\n"
            f"- **Own frames** (generic − null): L90 {pf['generic'] - pf['null']:+.2f} vs L10 {l10pf['generic'] - l10pf['null']:+.3f}.\n"
            f"- **Correction text** (real − generic): L90 {pf['real'] - pf['generic']:+.2f} vs L10 {l10pf['real'] - l10pf['generic']:+.3f} — the feedback adds "
            "nothing on held-out tasks (real ≈ generic).\n"
            "- Retry success after a failure on L10 is 1–7% in every mode: the learned use of feedback is specific to the training tasks."
            if all(k in l10pf for k in ("null", "generic", "real", "empty")) else "_L10 controls incomplete — contrast pending._")),
        wr.H2("Earlier ablation: instruction dropout"),
        wr.MarkdownBlock(md_table(["run", "instr. dropout", "real", "shuffled", "empty"], [
            ["fb_w333_s200_anchor, step 11400", "0.25"] + [f3(D["l10_ref"]["onpolicy_11400"][m]["held9"]["pass"]) for m in ("real", "shuffled", "empty")],
            [f"{EXP}, step 5000 (round 45)", "0.8"] + [f3(l10["b128_round45"][m]["held9"]["pass"]) for m in ("real", "shuffled", "empty")]])
            + "\n\nheld-out9 pass@8 (different runs and steps, so only indicative). With dropout 0.8 the real-vs-empty gap is smaller (−0.06 vs −0.15), but real context still does not help."),

        wr.H1("6. Failure localization: DTW vs VLM"),
        wr.MarkdownBlock(
            "60 failed + 30 successful LIBERO-90 rollouts from late rounds of the on-policy run. **DTW**: align the rollout to the nearest expert demo of the task "
            "and flag the first sustained divergence (t0). **VLM**: ask Gemini for the error frame twice (t1, t2).\n\n" + md_table(["metric", "value"], [
                ["DTW detection on failures", "59/60 = 0.98"], ["DTW false-positive rate on successes", "7/30 = 0.23"],
                ["DTW divergence step t0 (median, IQR)", "58 [25.5, 106.5]"], ["VLM self-consistency |t1 − t2| (median, IQR)", "7 [6, 32] steps"],
                ["VLM error step t1 (median)", "106"], ["DTW–VLM agreement |t0 − t1| (median, IQR)", "56 [9, 94] steps"],
                ["within 25 / within 50 steps", "0.36 / 0.44"], ["VLM earlier than DTW", "0.29"]])
            + "\n\nThe two signals are complementary: the VLM names *semantic* errors early (wrong object), DTW catches *kinematic* deviations (fumbled grasp, stalls) "
              "but also fires on harmless deviations. A representation comparison (DINOv2 patch features, π0.5 SigLIP features) for the DTW distance is in progress."),
        *[img(n, c) for n, c in SH],
        wr.P("Contact sheets: 32 frames of the failed rollout; red box = VLM error frame, blue box = DTW divergence t0."),

        wr.H1("7. Qualitative rollouts"),
        wr.P("Left: agentview, right: wrist. Captions show each attempt's outcome and the VLM feedback that the next attempt conditions on. "
             "Held-out chains that recover are rare (see section 4); these were picked as examples of recoveries, not typical behaviour."),
        *[b for group, task, items in G for b in [wr.H3(f"{group}: “{task}”"), *[img(n, c) for n, c in items]]],

        wr.H1("8. Costs & compute"),
        wr.MarkdownBlock(md_table(["item", "value"], [
            ["VLM judge spend (ledger, all experiments)", f"**${j['total']:.2f}** over {j['calls']:,} calls (${j['total'] / j['calls']:.4f}/call) — exceeded the original $500 cap (which invalidated the first textonly runs); cap now raised to $1000"],
            [f"judge spend logged by {EXP} rollouts (rounds with W&B logging)", f"${exp_cost:.2f}"],
            [f"{EXP} rollout runtime (W&B _runtime of the rollout logger; may undercount resumed segments)", f"{rrun.summary.get('_runtime', 0) / 3600:.1f} h (3× L40S)"],
            [f"{EXP} trainer (W&B)", f"{trun.summary.get('_step')} steps, _runtime {trun.summary.get('_runtime', 0) / 3600:.1f} h (2× H200)"],
            ["offline fine-tune", "2× H200 (JAX), 1k–3k steps"]])),

        wr.H1("9. Next steps"),
        wr.MarkdownBlock(
            "1. **Retrieval-based targeted supervision**: DTW-align each failed rollout to the task's demos and sample demo (s*, a*) around where it deviates, "
            "phase-weighted with soft weights p = ½·uniform + ½·deviation mass (so the whole demo stays covered).\n"
            "2. **Representation comparison** for the alignment distance (DINOv2 patches vs π0.5 SigLIP) — in progress.\n"
            "3. **Finish controls at step 7000**: `textonly` rerun on L90 and L10 (in progress), L10 `null`."),
        wr.H1("Appendix: data sources"),
        wr.MarkdownBlock(
            f"- offline evals: `{S}/eval/<model>/<suite>/results_*.jsonl`\n"
            f"- rollouts: `{OP}/{EXP}/round_*/records_*.jsonl`; checkpoints `{S}/checkpoints/pi05_libero90_ctx/{EXP}`\n"
            f"- held-out chain evals: `{OP}/eval_libero10_k8/*/<mode>/records_*.jsonl` (step read from the servers' checkpoint path; e.g. `b128_round15` = step 2000)\n"
            f"- L90 ablations: `{OP}/eval_libero90_diag/b128_7000{{,_cf,_ctrl}}/<mode>`\n"
            f"- localization: `{OP}/localize_test/demos_mp4/{{stats.txt,results.jsonl,sheets/}}`; judge ledger `{S}/judge_ledger.txt`\n"
            "- built by `onpolicy/wandb_report_clean.py` (branch onpolicy-feedback)"),
    ]
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-media", action="store_true", help="reuse existing GIFs")
    ap.add_argument("--dump", default=f"{S}/tmp/report_clean_data.json")
    args = ap.parse_args()
    D = compute()
    os.makedirs(os.path.dirname(args.dump), exist_ok=True)
    json.dump(D, open(args.dump, "w"), indent=1, default=str)
    print("data ->", args.dump, flush=True)
    P = plots(D)
    G = gifs(not args.no_media)
    SH = sheets()
    run = wandb.init(entity=ENTITY, project=PROJECT, name="results_clean", job_type="report")
    for f in glob.glob(f"{MEDIA}/**/*.*", recursive=True):
        run.save(f, base_path=MEDIA, policy="now")
    base = f"https://api.wandb.ai/files/{ENTITY}/{PROJECT}/{run.id}"
    run.finish()
    api = wandb.Api(timeout=60)
    blocks = blocks_for(D, P, G, SH, base, api)
    if os.path.exists(URL_FILE):
        report = wr.Report.from_url(open(URL_FILE).read().strip())
        report.blocks = blocks
    else:
        report = wr.Report(entity=ENTITY, project=PROJECT, title="π0.5 on LIBERO: offline baselines & on-policy VLM feedback (clean)",
                           description="All current results: offline LIBERO-90 fine-tuning, on-policy learning from VLM feedback, "
                                       "held-out LIBERO-10 transfer, ablations, failure localization.", blocks=blocks)
    report.save()
    open(URL_FILE, "w").write(report.url)
    print("REPORT", report.url)


if __name__ == "__main__":
    main()
