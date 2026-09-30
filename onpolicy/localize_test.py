"""Offline test: where does a failed rollout go wrong?  DTW divergence from expert demos vs a VLM error_frame.

Detector A (DTW): DINOv2 ViT-S/14 CLS of agentview + wrist (each L2-normalized, concatenated, renormalized), local
cost 1 - cos. Asymmetric open-end DTW: query step i maps to exactly one reference step j(i), j(0) = 0,
j(i) - j(i-1) in {0, 1, 2} (the query may dwell on a reference step or move up to 2x faster), reference end free.
tau(phase) = 95th pct of leave-one-out demo-vs-demo aligned costs in 10 phase bins. t0 = first t starting a run of
W consecutive steps with d(t) > tau(phase(t)); else "no divergence" + stall (first step at which the best
alignment reaches its final reference index).
Detector B (VLM): 32 agentview frames (128px) + instruction -> {"error_frame", "error"}, twice (second call offset
by half a sampling interval).

Steps are video-frame indices = policy steps after the 10 settle steps (record "steps" = these + 10 wait steps).
Expert demos map to rollouts by instruction; instructions shared by several LIBERO-90 scenes are excluded (the
expert index has no scene id), so only tasks with a unique instruction are sampled.

    python -m onpolicy.localize_test [--limit-fail 3 --limit-succ 1 --out .../sanity]
"""
import argparse
import base64
import collections
import concurrent.futures as cf
import faulthandler
import glob
import io
import json
import os
import random
import re
import time
import urllib.request

import imageio
import numba
import numpy as np
from PIL import Image, ImageDraw

from onpolicy.judge import _SSL, LEDGER, MODEL, URL

S = "/gscratch/scrubbed/sriyash/openpi_pt"
RUNS = f"{S}/onpolicy/async_b128_draft01"
EXPERT = f"{S}/expert_index"
DINO = "facebook/dinov2-small"
W, NBINS, PCT, NVLM = 10, 10, 95, 32

VLM_PROMPT = (
    "You are analyzing a robot arm that FAILED a household manipulation task in simulation.\n"
    "Task instruction: \"{instruction}\"\n"
    "The images are {n} frames from the failed attempt in time order, labeled Frame 0 to Frame {last} "
    "(Frame 0 = start, Frame {last} = end). The simulator confirms the task was not completed.\n"
    "Identify the FIRST frame where the attempt starts going wrong: the earliest frame at which the robot's behavior "
    "departs from what it should be doing to accomplish the instruction (e.g. reaching toward the wrong object, a "
    "missed or failed grasp, dropping the object, carrying it to the wrong place, stalling or wandering).\n"
    "Answer with JSON only: {{\"error_frame\": <integer 0-{last}>, \"error\": \"<one sentence>\"}}."
)


# ---------------------------------------------------------------- sampling
def load_records(min_round):
    recs = []
    for d in sorted(glob.glob(f"{RUNS}/round_*")):
        rnd = int(d.rsplit("_", 1)[1])
        if rnd < min_round:
            continue
        for f in sorted(glob.glob(f"{d}/records_*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                r["round"] = rnd
                r["id"] = f"r{rnd:03d}_{os.path.basename(r['video'])[:-4]}"
                recs.append(r)
    return recs


def sample(recs, episodes, n_fail, n_succ, per_task, seed):
    n_scene = collections.Counter()  # demos per instruction: ~45 per scene, >60 means shared by several scenes
    for e in episodes:
        n_scene[e["task"]] += 1
    ok = {t for t, c in n_scene.items() if c <= 60}
    recs = [r for r in recs if r["task"] in ok and os.path.exists(r["video"])]
    rng = random.Random(seed)
    rng.shuffle(recs)
    by_task = collections.defaultdict(list)
    for r in recs:
        if not r["success"]:
            by_task[r["task_id"]].append(r)
    tasks = sorted(by_task)
    rng.shuffle(tasks)
    fails = []
    for k in range(per_task):  # round-robin: one per task first, then a second, ... (spread over tasks)
        for t in tasks:
            if len(fails) < n_fail and len(by_task[t]) > k:
                fails.append(by_task[t][k])
    ftasks = {r["task_id"] for r in fails}
    succ_by = collections.defaultdict(list)
    for r in recs:
        if r["success"] and r["task_id"] in ftasks:  # successes from the same tasks: no extra demos to embed
            succ_by[r["task_id"]].append(r)
    st = sorted(succ_by)
    rng.shuffle(st)
    succs = []
    for k in range(4):
        for t in st:
            if len(succs) < n_succ and len(succ_by[t]) > k:
                succs.append(succ_by[t][k])
    return fails, succs, sorted(ok)


# ---------------------------------------------------------------- frames + embeddings
def to224(a):
    return a if a.shape[:2] == (224, 224) else np.asarray(Image.fromarray(a).resize((224, 224), Image.BILINEAR))


def video_frames(path):
    v = np.stack([f for f in imageio.get_reader(path)])  # (not list(): the reader reports len=inf)  # (T, 224, 448, 3) = [agentview | wrist]
    return v[:, :, :224], v[:, :, 224:]


def readable(path):
    try:
        imageio.get_reader(path).close()
        return True
    except Exception:
        return False


def demo_frames(ep, img_off, wr_off, pool):
    """One contiguous read per view per episode (frames of an episode are adjacent in the .bin), then PNG decode."""
    def view(binf, off):
        lo, hi = off[ep["start"]], off[ep["start"] + ep["length"]]
        with open(binf, "rb") as f:
            f.seek(lo)
            blob = f.read(hi - lo)
        cuts = off[ep["start"]:ep["start"] + ep["length"] + 1] - lo
        dec = lambda k: to224(np.asarray(Image.open(io.BytesIO(blob[cuts[k]:cuts[k + 1]])).convert("RGB")))
        return np.stack(list(pool.map(dec, range(ep["length"]))))
    return view(f"{EXPERT}/image.bin", img_off), view(f"{EXPERT}/wrist.bin", wr_off)


def mp4_roundtrip(agent, wrist):
    """Pass demo frames through the rollout client's writer (imageio mimwrite, fps 10, [agentview | wrist]) so both
    sides carry the same h264 artifacts."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        imageio.mimwrite(f"{td}/x.mp4", list(np.concatenate([agent, wrist], 2)), fps=10)
        return video_frames(f"{td}/x.mp4")


class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(DINO).to(self.dev).eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.dev).view(1, 3, 1, 1)

    def cls(self, imgs, bs=512):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(imgs), bs):
                x = self.torch.from_numpy(imgs[i:i + bs]).to(self.dev).permute(0, 3, 1, 2).float() / 255.0
                x = ((x - self.mean) / self.std).half()
                c = self.model(pixel_values=x).last_hidden_state[:, 0].float()
                out.append(self.torch.nn.functional.normalize(c, dim=-1).cpu().numpy())
        return np.concatenate(out)

    def __call__(self, agent, wrist):  # (T,224,224,3) x2 -> (T, 768), unit norm
        return (np.concatenate([self.cls(agent), self.cls(wrist)], 1) / np.sqrt(2)).astype(np.float32)


# ---------------------------------------------------------------- DTW
@numba.njit(cache=True)
def dtw_open_end(C):
    """C (n, m) local costs. Returns (mean aligned cost, j(i) path)."""
    n, m = C.shape
    D = np.full((n, m), np.inf)
    P = np.zeros((n, m), np.int8)
    D[0, 0] = C[0, 0]
    for i in range(1, n):
        for j in range(m):
            best, arg = D[i - 1, j], 0
            if j >= 1 and D[i - 1, j - 1] < best:
                best, arg = D[i - 1, j - 1], 1
            if j >= 2 and D[i - 1, j - 2] < best:
                best, arg = D[i - 1, j - 2], 2
            D[i, j] = best + C[i, j]
            P[i, j] = arg
    j = np.argmin(D[n - 1])
    total = D[n - 1, j]
    path = np.zeros(n, np.int64)
    for i in range(n - 1, -1, -1):
        path[i] = j
        j -= P[i, j]
    return total / n, path


def align(q, refs):
    """q (n, d); refs list of (m_k, d). Per step: min over refs of aligned cost, phase of that ref; plus the best
    (lowest total) ref's path phase."""
    n = len(q)
    costs = np.empty((len(refs), n))
    phases = np.empty((len(refs), n))
    totals = np.empty(len(refs))
    for k, r in enumerate(refs):
        C = (1.0 - q @ r.T).astype(np.float64)
        totals[k], path = dtw_open_end(C)
        costs[k] = C[np.arange(n), path]
        phases[k] = path / len(r)
    arg = costs.argmin(0)
    b = totals.argmin()
    return costs[arg, np.arange(n)], phases[arg, np.arange(n)], phases[b], b


def phase_bin(p):
    return np.clip((np.asarray(p) * NBINS).astype(int), 0, NBINS - 1)


def calibrate(demos):
    cs, ps = [], []
    for i, q in enumerate(demos):
        c, p, _, _ = align(q, demos[:i] + demos[i + 1:])
        cs.append(c)
        ps.append(p)
    c, b = np.concatenate(cs), phase_bin(np.concatenate(ps))
    tau = np.array([np.percentile(c[b == k], PCT) if (b == k).sum() >= 20 else np.nan for k in range(NBINS)])
    tau = np.where(np.isnan(tau), np.percentile(c, PCT), tau)
    return tau


def detect(d, phase, best_phase, tau):
    over = d > tau[phase_bin(phase)]
    run, t0 = 0, None
    for t, o in enumerate(over):
        run = run + 1 if o else 0
        if run == W:
            t0 = t - W + 1
            break
    final = best_phase[-1]
    stall = int(np.argmax(best_phase >= final - 1e-9))
    return t0, stall, over


# ---------------------------------------------------------------- VLM
class VLM:
    def __init__(self, max_cost, model=MODEL):
        self.key, self.model, self.max_cost = os.environ["OPENROUTER_API_KEY"], model, max_cost
        self.cost, self.calls = 0.0, 0

    def __call__(self, instruction, frames, retries=3):
        if self.cost >= self.max_cost:
            raise RuntimeError(f"VLM cost ${self.cost:.3f} >= ${self.max_cost}")
        n = len(frames)
        content = [{"type": "text", "text": VLM_PROMPT.format(instruction=instruction, n=n, last=n - 1)}]
        for k, f in enumerate(frames):
            buf = io.BytesIO()
            Image.fromarray(f).resize((128, 128), Image.BILINEAR).save(buf, format="PNG")
            content.append({"type": "text", "text": f"Frame {k}:"})
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0,
                           "max_tokens": 2048, "reasoning": {"effort": "low", "exclude": True}, "usage": {"include": True}}).encode()
        err, text = None, ""
        for a in range(retries + 1):
            try:
                req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120, context=_SSL) as r:
                    out = json.loads(r.read())
                cost = float(out.get("usage", {}).get("cost", 0.0) or 0.0)
                self.cost += cost
                self.calls += 1
                with open(LEDGER, "a") as f:
                    f.write(f"{self.model} {cost:.6f}\n")
                text = out["choices"][0]["message"]["content"] or ""
                try:
                    p = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                    ef = int(p["error_frame"])
                    if not 0 <= ef < n:
                        raise ValueError(f"error_frame {ef} out of range")
                    return {"frame": ef, "error": str(p.get("error", "")).strip(), "parsed": True, "raw": text}
                except Exception as e:  # a paid, unparseable answer counts against the parse rate: no retry
                    return {"frame": None, "error": "", "parsed": False, "raw": text, "exc": repr(e)}
            except Exception as e:
                err = repr(e)
                time.sleep(2 ** a)
        return {"frame": None, "error": "", "parsed": False, "raw": text, "exc": err}


def cached_vlm(vlm, path, instruction, frames):
    """Paid answers are kept on disk so a crash or requeue never pays twice (network failures are not cached)."""
    if os.path.exists(path):
        return json.load(open(path))
    out = vlm(instruction, frames)
    if out["parsed"] or out.get("raw"):
        json.dump(out, open(path, "w"))
    return out


def vlm_indices(T, n=NVLM, offset=0.0):
    return np.minimum(((np.arange(n) + offset) * T / n).astype(int), T - 1)


# ---------------------------------------------------------------- contact sheet
def sheet(frames, idx, red, blue, path, title):
    cell, cols = 128, 8
    rows = (len(idx) + cols - 1) // cols
    img = Image.new("RGB", (cols * cell, rows * (cell + 14) + 16), "white")
    dr = ImageDraw.Draw(img)
    dr.text((4, 2), title[:170], fill="black")
    for k, t in enumerate(idx):
        r, c = divmod(k, cols)
        x, y = c * cell, 16 + r * (cell + 14)
        img.paste(Image.fromarray(frames[t]).resize((cell, cell), Image.BILINEAR), (x, y))
        for kk, col in ((red, "red"), (blue, "blue")):
            if kk == k:
                w = 4 if col == "red" else 3
                o = 0 if col == "red" else 5  # blue inset so both show when they coincide
                dr.rectangle([x + o, y + o, x + cell - 1 - o, y + cell - 1 - o], outline=col, width=w)
        dr.text((x + 3, y + cell), f"#{k} t={t}", fill="black")
    img.save(path)


# ---------------------------------------------------------------- stats
def q(x):
    x = np.asarray([v for v in x if v is not None], float)
    if not len(x):
        return "n=0"
    p = np.percentile(x, [25, 50, 75])
    return f"n={len(x)} median={p[1]:.1f} IQR=[{p[0]:.1f}, {p[2]:.1f}] mean={x.mean():.1f}"


def stats(res):
    F = [r for r in res if not r["success"]]
    Sx = [r for r in res if r["success"]]
    L = []
    det = [r for r in F if r["t0_dtw"] is not None]
    L.append(f"DTW detection rate on failures: {len(det)}/{len(F)} = {len(det) / max(len(F), 1):.2f}")
    fp = [r for r in Sx if r["t0_dtw"] is not None]
    L.append(f"DTW false-positive rate on successes: {len(fp)}/{len(Sx)} = {len(fp) / max(len(Sx), 1):.2f}")
    L.append(f"t0_dtw / T (failures, detected): {q([r['t0_dtw'] / r['T'] for r in det])}")
    L.append(f"t0_dtw steps (failures, detected): {q([r['t0_dtw'] for r in det])}")
    L.append(f"stall / T (failures, not detected): {q([r['stall'] / r['T'] for r in F if r['t0_dtw'] is None])}")
    L.append(f"d/tau median over episode: failures {q([r['d_summary']['median_ratio'] for r in F])}; successes {q([r['d_summary']['median_ratio'] for r in Sx])}")
    L.append(f"frac steps over tau: failures {q([r['d_summary']['frac_over'] for r in F])}; successes {q([r['d_summary']['frac_over'] for r in Sx])}")
    p1 = [r for r in F if r["t_vlm1"] is not None]
    p2 = [r for r in F if r["t_vlm2"] is not None]
    L.append(f"VLM parse rate: call1 {len(p1)}/{len(F)}, call2 {len(p2)}/{len(F)}")
    both = [r for r in F if r["t_vlm1"] is not None and r["t_vlm2"] is not None]
    L.append(f"VLM self-consistency |t1-t2| steps: {q([abs(r['t_vlm1'] - r['t_vlm2']) for r in both])}")
    L.append(f"t_vlm1 / T: {q([r['t_vlm1'] / r['T'] for r in p1])}; t_vlm1 steps: {q([r['t_vlm1'] for r in p1])}")
    ag = [r for r in F if r["t_vlm1"] is not None and r["t0_dtw"] is not None]
    dd = np.array([abs(r["t0_dtw"] - r["t_vlm1"]) for r in ag], float)
    L.append(f"Agreement |t0_dtw - t_vlm1| (both present): {q(dd)}")
    if len(ag):
        L.append(f"  within 25: {(dd <= 25).mean():.2f}  within 50: {(dd <= 50).mean():.2f}  "
                 f"VLM earlier: {np.mean([r['t_vlm1'] < r['t0_dtw'] for r in ag]):.2f}  (n={len(ag)})")
    ag2 = [r for r in F if r["t_vlm1"] is not None]
    dd2 = np.array([abs((r["t0_dtw"] if r["t0_dtw"] is not None else r["stall"]) - r["t_vlm1"]) for r in ag2], float)
    if len(ag2):
        L.append(f"Agreement using stall when no divergence: {q(dd2)} within 25: {(dd2 <= 25).mean():.2f} within 50: {(dd2 <= 50).mean():.2f}")
    return "\n".join(L)


# ---------------------------------------------------------------- diagnostics: where does the rollout-vs-demo gap come from?
def diagnose(items, episodes, emb, pool, out, n=12):
    """(1) mp4 codec: lossless npz agentview frames vs the same mp4 frames (agentview-only cost). (2) demo PNG ->
    mp4 (same writer as the rollout client) -> embed: per-frame cost vs the PNG embedding. (3) frame-0 gap: rollout
    frame 0 vs nearest demo frame 0, against demo frame 0 vs nearest other demo frame 0."""
    import tempfile
    img_off, wr_off = np.load(f"{EXPERT}/image_off.npy"), np.load(f"{EXPERT}/wrist_off.npy")
    rep, codec, rt, f0r, f0d = {}, [], [], [], []
    for r in items[:n]:
        agent, wrist = video_frames(r["video"])
        lossless = np.load(r["frames"])["frames"]
        idx = np.linspace(0, len(agent) - 1, len(lossless)).round().astype(int)
        e1, e2 = emb.cls(lossless), emb.cls(np.ascontiguousarray(agent[idx]))
        codec.append(float(np.mean(1 - (e1 * e2).sum(1))))
    tasks = sorted({r["task"] for r in items[:n]})[:4]
    for task in tasks:
        eps = [e for e in episodes if e["task"] == task]
        z = np.load(f"{out}/cache/demo_{re.sub('[^a-z0-9]+', '_', task)}.npz")
        dem = [z[f"e{k}"] for k in range(len(eps))]
        for e, de in list(zip(eps, dem))[:3]:
            a, w = demo_frames(e, img_off, wr_off, pool)
            with tempfile.TemporaryDirectory() as td:
                imageio.mimwrite(f"{td}/x.mp4", list(np.concatenate([a, w], 2)), fps=10)
                ra, rw = video_frames(f"{td}/x.mp4")
            rt.append(float(np.mean(1 - (emb(ra, rw) * de).sum(1))))
    for r in items:
        z = np.load(f"{out}/cache/demo_{re.sub('[^a-z0-9]+', '_', r['task'])}.npz")
        d0 = np.stack([z[k][0] for k in z.files])
        e0 = np.load(f"{out}/cache/{r['id']}.npy")[0]
        f0r.append(float((1 - d0 @ e0).min()))
        c = 1 - d0 @ d0.T
        np.fill_diagonal(c, np.inf)
        f0d.append(float(np.median(c.min(1))))
    rep = {"codec_agentview_cost_npz_vs_mp4": codec, "demo_png_vs_mp4_roundtrip_cost": rt,
           "frame0_rollout_to_nearest_demo": f0r, "frame0_demo_to_nearest_other_demo_median": f0d}
    json.dump(rep, open(f"{out}/diag.json", "w"))
    for k, v in rep.items():
        print(f"{k}: {q(v)}  (x1000: median {1000 * np.median(v):.1f})")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{S}/onpolicy/localize_test")
    ap.add_argument("--n-fail", type=int, default=60)
    ap.add_argument("--n-succ", type=int, default=30)
    ap.add_argument("--limit-fail", type=int, default=None)
    ap.add_argument("--limit-succ", type=int, default=None)
    ap.add_argument("--min-round", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cost", type=float, default=5.0)
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--demo-codec", default="png", choices=["png", "mp4"], help="mp4: round-trip demos through the rollout video writer")
    ap.add_argument("--tag", default="", help="write results/stats/sheets to <out>/<tag>/ (caches stay in <out>/cache)")
    ap.add_argument("--diag", action="store_true", help="run the rollout-vs-demo gap diagnostics (needs caches) and exit")
    ap.add_argument("--precache-shard", default=None, help="k/n: only embed demos of every n-th task (reverse order), then exit")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(1200, repeat=True)  # hang diagnostics in the job log
    R = f"{a.out}/{a.tag}" if a.tag else a.out  # results dir
    os.makedirs(f"{R}/sheets", exist_ok=True)
    os.makedirs(f"{a.out}/cache", exist_ok=True)
    sfx = "" if a.demo_codec == "png" else "_mp4"
    t_start = time.time()

    episodes = json.load(open(f"{EXPERT}/episodes.json"))
    # Oversample (the round-robin order makes the first n a prefix) and drop unreadable videos (a few mp4s are corrupt).
    fails, succs, ok = sample(load_records(a.min_round), episodes, a.n_fail + 20, a.n_succ + 20, 2, a.seed)
    bad = [r["id"] for r in fails + succs if not readable(r["video"])]
    if bad:
        print(f"skipping unreadable videos: {bad}", flush=True)
    fails = [r for r in fails if r["id"] not in bad][:a.n_fail][:a.limit_fail]
    succs = [r for r in succs if r["id"] not in bad][:a.n_succ][:a.limit_succ]
    items = fails + succs
    print(f"{len(fails)} failures over {len({r['task_id'] for r in fails})} tasks, {len(succs)} successes; "
          f"{len(ok)} unique-instruction tasks eligible", flush=True)

    img_off, wr_off = np.load(f"{EXPERT}/image_off.npy"), np.load(f"{EXPERT}/wrist_off.npy")
    emb = Embedder()
    pool = cf.ThreadPoolExecutor(16)
    if a.diag:
        return diagnose([r for r in items if os.path.exists(f"{a.out}/cache/{r['id']}.npy")], episodes, emb, pool, a.out)
    demos, taus = {}, {}
    task_list = sorted({r["task"] for r in items})
    if a.precache_shard:
        k, n = map(int, a.precache_shard.split("/"))
        task_list = task_list[::-1][k::n]
    for task in task_list:
        cp = f"{a.out}/cache/demo_{re.sub('[^a-z0-9]+', '_', task)}{sfx}.npz"
        eps = [e for e in episodes if e["task"] == task]
        if os.path.exists(cp):
            z = np.load(cp)
            demos[task] = [z[f"e{k}"] for k in range(len(eps))]
        else:
            codec = mp4_roundtrip if a.demo_codec == "mp4" else (lambda x, y: (x, y))
            demos[task] = [emb(*codec(*demo_frames(e, img_off, wr_off, pool))) for e in eps]
            np.savez(cp + ".tmp.npz", **{f"e{k}": x for k, x in enumerate(demos[task])})
            os.replace(cp + ".tmp.npz", cp)  # atomic: precache shards run concurrently
        if a.precache_shard:
            print(f"[{time.time() - t_start:.0f}s] cached {task}", flush=True)
            continue
        taus[task] = calibrate(demos[task])
        print(f"[{time.time() - t_start:.0f}s] {task}: {len(eps)} demos, tau={np.round(taus[task], 3).tolist()}", flush=True)

    if a.precache_shard:
        return
    vlm = None if a.no_vlm else VLM(a.max_cost)
    results = []
    vlm_jobs = {}
    vex = cf.ThreadPoolExecutor(8)
    for r in items:
        agent, wrist = video_frames(r["video"])
        T = len(agent)
        cp = f"{a.out}/cache/{r['id']}.npy"
        if os.path.exists(cp):
            e = np.load(cp)
        else:
            e = emb(agent, wrist)
            np.save(cp, e)
        d, ph, bph, b = align(e, demos[r["task"]])
        tau = taus[r["task"]]
        t0, stall, over = detect(d, ph, bph, tau)
        ratio = d / tau[phase_bin(ph)]
        res = {"id": r["id"], "round": r["round"], "task_id": r["task_id"], "task": r["task"], "chain": r["chain"],
               "attempt": r["attempt"], "success": r["success"], "steps": r["steps"], "T": T, "video": r["video"],
               "t0_dtw": None if t0 is None else int(t0), "stall": stall, "diverged": t0 is not None,
               "d_summary": {"mean": float(d.mean()), "max": float(d.max()), "median_ratio": float(np.median(ratio)),
                             "frac_over": float(over.mean()), "final_phase_best": float(bph[-1]),
                             "d_every10": np.round(d[::10], 3).tolist(), "ratio_every10": np.round(ratio[::10], 2).tolist(),
                             "phase_every10": np.round(ph[::10], 2).tolist()},
               "feedback": r["feedback"]}
        np.savez(f"{a.out}/cache/{r['id']}_dtw{sfx}.npz", d=d, phase=ph, best_phase=bph, tau=tau)
        if not r["success"] and vlm is not None:
            i1, i2 = vlm_indices(T), vlm_indices(T, offset=0.5)
            vlm_jobs[r["id"]] = (vex.submit(cached_vlm, vlm, f"{a.out}/cache/{r['id']}_vlm1.json", r["task"], agent[i1]),
                                 vex.submit(cached_vlm, vlm, f"{a.out}/cache/{r['id']}_vlm2.json", r["task"], agent[i2]), i1, i2, agent)
        results.append(res)
        print(f"[{time.time() - t_start:.0f}s] {r['id']} succ={r['success']} T={T} t0={res['t0_dtw']} stall={stall} "
              f"frac_over={over.mean():.2f}", flush=True)

    for res in results:
        if res["id"] not in vlm_jobs:
            res.update(t_vlm1=None, t_vlm2=None)
            continue
        f1, f2, i1, i2, agent = vlm_jobs[res["id"]]
        v1, v2 = f1.result(), f2.result()
        res.update(vlm1_frame=v1["frame"], vlm2_frame=v2["frame"],
                   t_vlm1=None if v1["frame"] is None else int(i1[v1["frame"]]),
                   t_vlm2=None if v2["frame"] is None else int(i2[v2["frame"]]),
                   vlm1_error=v1["error"], vlm2_error=v2["error"],
                   vlm1_raw=None if v1["parsed"] else v1["raw"], vlm2_raw=None if v2["parsed"] else v2["raw"])
        tb = res["t0_dtw"] if res["t0_dtw"] is not None else res["stall"]
        blue = int(np.argmin(np.abs(i1 - tb)))
        res["sheet"] = f"{R}/sheets/{res['id']}.png"
        sheet(agent, i1, v1["frame"], blue, res["sheet"],
              f"{res['id']} | {res['task']} | red=VLM1 t={res['t_vlm1']}  blue={'t0_dtw' if res['t0_dtw'] is not None else 'stall(no div)'} t={tb}")

    with open(f"{R}/results.jsonl", "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    txt = stats(results)
    if vlm is not None:
        txt += f"\nVLM cost: ${vlm.cost:.4f} over {vlm.calls} calls"
    F = [r for r in results if not r["success"] and r.get("t_vlm1") is not None and r["t0_dtw"] is not None]
    F.sort(key=lambda r: -abs(r["t0_dtw"] - r["t_vlm1"]))
    top = F[:10]
    rest = [r for r in results if not r["success"] and r not in top and "sheet" in r]
    random.Random(a.seed).shuffle(rest)
    txt += "\n\nInspect (10 largest disagreements):\n" + "\n".join(
        f"  {r['id']} t0={r['t0_dtw']} vlm1={r['t_vlm1']} vlm2={r['t_vlm2']} {r['sheet']}" for r in top)
    txt += "\nInspect (10 random):\n" + "\n".join(
        f"  {r['id']} t0={r['t0_dtw']} stall={r['stall']} vlm1={r['t_vlm1']} vlm2={r['t_vlm2']} {r['sheet']}" for r in rest[:10])
    open(f"{R}/stats.txt", "w").write(txt + "\n")
    print(txt)
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
