"""Compare image representations for DTW failure localization (extends onpolicy.localize_test; same sample, same mp4
demo preprocessing, same open-end DTW + leave-one-out phase-binned calibration + W-run detector).

Representations (each view -> K unit vectors; local cost = mean over the K cells of 1 - cos; views mixed with weights):
  dino_cls      DINOv2 ViT-S/14 CLS (K=1)                         = the localize_test baseline
  dino_patch    DINOv2 ViT-S/14 patch tokens, 16x16 -> 4x4 avg pool (K=16)
  siglip_patch  pi0.5 SigLIP So400m/14 image tokens as fed to the LLM (after the 2048-d head), 16x16 -> 4x4 (K=16)
  siglip_global the same tokens mean-pooled over all 256 (K=1)
Variants: <repr>/avg (agentview 1 : wrist 1) and <repr>/wrist2 (agentview 1 : wrist 2).

Sample = the 90 rollouts of <out>/demos_mp4/results.jsonl. Demo frames are round-tripped once through the rollout
video writer and cached as mp4 under <out>/cache/demo_mp4/<task>/e<k>.mp4 (decoding the cached mp4 = the round trip).

    python -m onpolicy.localize_repr extract [--shard k/n] [--ids id1,id2]   # per task: embed, calibrate, DTW
    python -m onpolicy.localize_repr report                                    # metrics table
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import time

import imageio
import numpy as np

from onpolicy.localize_test import EXPERT, NBINS, PCT, S, demo_frames, detect, dtw_open_end, phase_bin, q, video_frames

OUT = f"{S}/onpolicy/localize_test"
REPR_DIR = f"{OUT}/repr"
SIGLIP_CKPT = f"{S}/checkpoints/pi05_libero90/libero90_1k_w333/200/params"
REPRS = ["dino_cls", "dino_patch", "siglip_patch", "siglip_global"]
WEIGHTS = {"avg": (0.5, 0.5), "wrist2": (1 / 3, 2 / 3)}
PREFIX = [25, 50, 75, 100, 150, 200, 300]
HAND = {"r080_task64_c80064-0_a2_failure": "wrong bowl ~37-62", "r091_task41_c91041-0_a0_failure": "fumble ~87",
        "r093_task72_c93072-0_a0_failure": "no release ~125-160", "r090_task24_c90024-0_a1_failure": "drawer fumble ~100"}


def slug(task):
    return re.sub("[^a-z0-9]+", "_", task)


def pool4(tok):
    """(B, 256, D) tokens on a 16x16 grid -> (B, 16, D) 4x4 average pool."""
    B, _, D = tok.shape
    return tok.reshape(B, 4, 4, 4, 4, D).mean((2, 4)).reshape(B, 16, D)


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


# ---------------------------------------------------------------- encoders: (T,224,224,3) uint8 -> {repr: (T,K,D) f16}
class Dino:
    names = ("dino_cls", "dino_patch")

    def __init__(self):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained("facebook/dinov2-small").cuda().eval().half()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def __call__(self, imgs, bs=512):
        t = self.torch
        cls, pat = [], []
        with t.no_grad():
            for i in range(0, len(imgs), bs):  # same preprocessing as localize_test.Embedder
                x = t.from_numpy(imgs[i:i + bs]).cuda().permute(0, 3, 1, 2).float() / 255.0
                h = self.model(pixel_values=((x - self.mean) / self.std).half()).last_hidden_state.float()
                cls.append(t.nn.functional.normalize(h[:, :1], dim=-1).half().cpu().numpy())
                p = h[:, 1:].reshape(-1, 4, 4, 4, 4, h.shape[-1]).mean((2, 4)).reshape(-1, 16, h.shape[-1])
                pat.append(t.nn.functional.normalize(p, dim=-1).half().cpu().numpy())
        return {"dino_cls": np.concatenate(cls), "dino_patch": np.concatenate(pat)}


class Siglip:
    """The frozen image encoder of the pi0.5 checkpoint, exactly as Pi0.embed_prefix calls it (PaliGemma.img)."""
    names = ("siglip_patch", "siglip_global")

    def __init__(self, bs=128):
        import flax.traverse_util as tu
        import jax
        import jax.numpy as jnp
        import openpi.models.siglip as _siglip
        self.jax, self.jnp, self.bs = jax, jnp, bs
        cp = f"{OUT}/cache/siglip_img_params.npz"
        if os.path.exists(cp):
            z = np.load(cp)
            flat = {tuple(k.split("/")): z[k] for k in z.files}
        else:
            flat = tu.flatten_dict(self._restore_img())
            np.savez(cp + ".tmp.npz", **{"/".join(k): v for k, v in flat.items()})
            os.replace(cp + ".tmp.npz", cp)
        params = tu.unflatten_dict(flat)
        print(f"siglip params: {sum(v.size for v in flat.values()) / 1e6:.1f}M, top keys {sorted(params)}", flush=True)
        mod = _siglip.Module(num_classes=2048, variant="So400m/14", pool_type="none", scan=True, dtype_mm="bfloat16")
        self.params = jax.device_put(params)

        @jax.jit
        def f(p, x):
            tok, _ = mod.apply({"params": p}, x.astype(jnp.float32) / 255.0 * 2.0 - 1.0, train=False)  # (B,256,2048)
            tok = tok.astype(jnp.float32)
            u = lambda v: v / jnp.maximum(jnp.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
            B, _, D = tok.shape
            grid = tok.reshape(B, 4, 4, 4, 4, D).mean((2, 4)).reshape(B, 16, D)
            return u(grid).astype(jnp.float16), u(tok.mean(1, keepdims=True)).astype(jnp.float16)
        self.f = f

    @staticmethod
    def _restore_img():
        import jax
        import orbax.checkpoint as ocp
        import flax.traverse_util as tu
        t = time.time()
        with ocp.PyTreeCheckpointer() as ck:
            meta = ck.metadata(SIGLIP_CKPT)
            sub = {"params": {"PaliGemma": {"img": meta["params"]["PaliGemma"]["img"]}}}
            ra = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), sub)
            p = ck.restore(SIGLIP_CKPT, ocp.args.PyTreeRestore(item=sub, restore_args=ra, transforms={}))
        p = tu.flatten_dict(p["params"]["PaliGemma"]["img"])
        if all(k[-1] == "value" for k in p):
            p = {k[:-1]: v for k, v in p.items()}
        print(f"restored siglip params from checkpoint in {time.time() - t:.0f}s", flush=True)
        return tu.unflatten_dict(p)

    def __call__(self, imgs):
        g, gl = [], []
        n = len(imgs)
        for i in range(0, n, self.bs):
            x = imgs[i:i + self.bs]
            m = len(x)
            if m < self.bs:  # fixed batch shape: one compile
                x = np.concatenate([x, np.zeros((self.bs - m,) + x.shape[1:], x.dtype)])
            a, b = self.f(self.params, self.jnp.asarray(x))
            g.append(np.asarray(a)[:m])
            gl.append(np.asarray(b)[:m])
        return {"siglip_patch": np.concatenate(g), "siglip_global": np.concatenate(gl)}


def embed(encoders, agent, wrist):
    """-> {repr: (agent (T,K,D), wrist (T,K,D))}"""
    out = {}
    for enc in encoders:
        ea, ew = enc(agent), enc(wrist)
        for k in enc.names:
            out[k] = (ea[k], ew[k])
    return out


def flat(pair, w):
    """(agent, wrist) per-cell unit vectors -> (T, 2KD) with dot product = wa*mean_cell_cos_a + ww*mean_cell_cos_w."""
    a, b = pair
    K = a.shape[1]
    return np.concatenate([a.reshape(len(a), -1) * np.sqrt(w[0] / K), b.reshape(len(b), -1) * np.sqrt(w[1] / K)], 1)


# ---------------------------------------------------------------- DTW on precomputed costs (same as localize_test.align/calibrate)
def align_C(Cs):
    n = Cs[0].shape[0]
    costs, phases, totals = np.empty((len(Cs), n)), np.empty((len(Cs), n)), np.empty(len(Cs))
    for k, C in enumerate(Cs):
        totals[k], path = dtw_open_end(np.ascontiguousarray(C, dtype=np.float64))
        costs[k] = C[np.arange(n), path]
        phases[k] = path / C.shape[1]
    arg, b = costs.argmin(0), totals.argmin()
    return costs[arg, np.arange(n)], phases[arg, np.arange(n)], phases[b]


def gram(A, B):
    import torch
    with torch.no_grad():
        return (torch.from_numpy(A).cuda().float() @ torch.from_numpy(B).cuda().float().T).cpu().numpy()


def run_task(task, eps, rolls, encoders, img_off, wr_off, pool, reprs):
    t = time.time()
    dd = f"{OUT}/cache/demo_mp4/{slug(task)}"
    os.makedirs(dd, exist_ok=True)
    demb = []
    for k, e in enumerate(eps):
        p = f"{dd}/e{k}.mp4"
        if not os.path.exists(p):  # the rollout client's writer: imageio mimwrite fps 10 [agentview | wrist]
            a, w = demo_frames(e, img_off, wr_off, pool)
            imageio.mimwrite(p + ".tmp.mp4", list(np.concatenate([a, w], 2)), fps=10)
            os.replace(p + ".tmp.mp4", p)
        demb.append(embed(encoders, *video_frames(p)))
    remb = [embed(encoders, *video_frames(r["video"])) for r in rolls]
    t_emb = time.time() - t
    lens = [len(x[reprs[0]][0]) for x in demb]
    cut = np.concatenate([[0], np.cumsum(lens)])
    for rp in reprs:
        for vn, w in WEIGHTS.items():
            X = np.concatenate([flat(x[rp], w) for x in demb])
            G = 1.0 - gram(X, X)
            cs, ps = [], []
            for i in range(len(eps)):
                c, p, _ = align_C([G[cut[i]:cut[i + 1], cut[k]:cut[k + 1]] for k in range(len(eps)) if k != i])
                cs.append(c)
                ps.append(p)
            c, b = np.concatenate(cs), phase_bin(np.concatenate(ps))
            tau = np.array([np.percentile(c[b == k], PCT) if (b == k).sum() >= 20 else np.nan for k in range(NBINS)])
            tau = np.where(np.isnan(tau), np.percentile(c, PCT), tau)
            od = f"{REPR_DIR}/{rp}/{vn}"
            os.makedirs(od, exist_ok=True)
            for r, x in zip(rolls, remb):
                C = 1.0 - gram(flat(x[rp], w), X)
                d, ph, bph = align_C([C[:, cut[k]:cut[k + 1]] for k in range(len(eps))])
                np.savez(f"{od}/{r['id']}.npz", d=d, phase=ph, best_phase=bph, tau=tau)
                t0, _, _ = detect(d, ph, bph, tau)
                print(f"  {rp}/{vn} {r['id']} T={len(d)} t0={t0}", flush=True)
    print(f"[task done] {task}: {len(eps)} demos ({sum(lens)} frames), {len(rolls)} rollouts, "
          f"embed {t_emb:.0f}s total {time.time() - t:.0f}s", flush=True)


# ---------------------------------------------------------------- metrics
def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def report(res):
    rows, per = [], {}
    F = [r for r in res if not r["success"]]
    Sx = [r for r in res if r["success"]]
    for rp in REPRS:
        for vn in WEIGHTS:
            od = f"{REPR_DIR}/{rp}/{vn}"
            if not all(os.path.exists(f"{od}/{r['id']}.npz") for r in res):
                print(f"skip {rp}/{vn}: missing outputs")
                continue
            ex, t0 = {}, {}
            for r in res:
                z = np.load(f"{od}/{r['id']}.npz")
                ex[r["id"]] = z["d"] / z["tau"][phase_bin(z["phase"])] - 1.0
                t = detect(z["d"], z["phase"], z["best_phase"], z["tau"])[0]
                t0[r["id"]] = None if t is None else int(t)
            au = [auroc([ex[r["id"]][:t].max() for r in F], [ex[r["id"]][:t].max() for r in Sx]) for t in PREFIX]
            det = [t0[r["id"]] for r in F if t0[r["id"]] is not None]
            fp = sum(t0[r["id"]] is not None for r in Sx)
            ag = np.array([abs(t0[r["id"]] - r["t_vlm1"]) for r in F if t0[r["id"]] is not None and r["t_vlm1"] is not None], float)
            p = np.percentile(det, [25, 50, 75]) if det else [np.nan] * 3
            name = f"{rp}/{vn}"
            per[name] = {"auroc": au, "auc_mean": float(np.mean(au)), "det": len(det) / len(F), "fp": fp / len(Sx),
                         "t0_med": p[1], "t0_iqr": [p[0], p[2]], "vlm_med": float(np.median(ag)) if len(ag) else None,
                         "vlm_w50": float((ag <= 50).mean()) if len(ag) else None, "hand": {h: t0[h] for h in HAND},
                         "full_auroc": auroc([ex[r["id"]].max() for r in F], [ex[r["id"]].max() for r in Sx])}
    L = ["Early-separation AUROC (failures vs successes; score = max_{s<=t} d/tau - 1)",
         f"{'variant':24s} " + " ".join(f"t<={t:<4d}" for t in PREFIX) + "  mean   full  det   FP    t0 med [IQR]      |t0-vlm1| med  w50"]
    for n, m in per.items():
        L.append(f"{n:24s} " + " ".join(f"{a:7.3f}" for a in m["auroc"]) +
                 f"  {m['auc_mean']:.3f}  {m['full_auroc']:.3f}  {m['det']:.2f}  {m['fp']:.2f}  "
                 f"{m['t0_med']:5.0f} [{m['t0_iqr'][0]:.0f},{m['t0_iqr'][1]:.0f}]      {m['vlm_med']:6.1f}     {m['vlm_w50']:.2f}")
    L.append("\nHand-checked t0 (None = no divergence):")
    L.append(f"{'variant':24s} " + "  ".join(f"{h[:24]:>24s}" for h in HAND))
    L.append(f"{'(true onset)':24s} " + "  ".join(f"{v:>24s}" for v in HAND.values()))
    for n, m in per.items():
        L.append(f"{n:24s} " + "  ".join(f"{str(m['hand'][h]):>24s}" for h in HAND))
    txt = "\n".join(L)
    open(f"{REPR_DIR}/table.txt", "w").write(txt + "\n")
    json.dump(per, open(f"{REPR_DIR}/metrics.json", "w"), indent=1)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "report"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ids", default=None, help="comma-separated rollout ids (sanity check): only their tasks/rollouts")
    ap.add_argument("--reprs", default=",".join(REPRS))
    a = ap.parse_args()
    res = [json.loads(l) for l in open(f"{OUT}/demos_mp4/results.jsonl")]
    if a.mode == "report":
        return report(res)
    reprs = a.reprs.split(",")
    if a.ids:
        ids = set(a.ids.split(","))
        res = [r for r in res if r["id"] in ids]
    encoders = []
    if any(r.startswith("siglip") for r in reprs):
        encoders.append(Siglip())  # before torch touches the GPU (XLA_PYTHON_CLIENT_MEM_FRACTION caps JAX's share)
    if any(r.startswith("dino") for r in reprs):
        encoders.append(Dino())
    episodes = json.load(open(f"{EXPERT}/episodes.json"))
    img_off, wr_off = np.load(f"{EXPERT}/image_off.npy"), np.load(f"{EXPERT}/wrist_off.npy")
    pool = cf.ThreadPoolExecutor(16)
    by = collections.defaultdict(list)
    for r in res:
        by[r["task"]].append(r)
    k, n = map(int, a.shard.split("/"))
    tasks = sorted(by)[k::n]
    t = time.time()
    for i, task in enumerate(tasks):
        run_task(task, [e for e in episodes if e["task"] == task], by[task], encoders, img_off, wr_off, pool, reprs)
        print(f"[{time.time() - t:.0f}s] {i + 1}/{len(tasks)} tasks", flush=True)


if __name__ == "__main__":
    main()
