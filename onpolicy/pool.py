"""On-the-fly training data: pi(a* | s*, r_i, f_i, instruction w/ dropout), sampled from two pools.

  context pool: from the most recent `recent_rounds` finished rollout rounds, every chain r0 f0 r1 f1 r2 f2 contributes
                the contexts {none (i = -1), (r0, f0), (r1, f1), (r2, f2)}; re-read from disk as new rounds land.
  expert pool:  every LIBERO-90 demo transition, by task (onpolicy/expert_index.py).

negative pool (pool.json negative_fraction > 0): failed attempts' own replan points (s_i, a_i), with context
                (r_i, f_i) of that same attempt, flagged `negative` for the suppression term in scripts/train.py.

A sample: w.p. negative_fraction a negative, else context c uniformly (kept for SAMPLES_PER_CONTEXT consecutive samples) -> (s*, a*) uniformly from the expert transitions of c's task -> prompt = instruction
(i = -1) or feedback with the instruction dropped w.p. instr_dropout. Settings come from <root>/pool.json.
"""
import collections
import glob
import io
import json
import os
import random
import time

import numpy as np
from PIL import Image

from onpolicy import context

ACTION_HORIZON = 10
REFRESH_EVERY = 2000  # samples per worker between re-reads of the context pool
SAMPLES_PER_CONTEXT = 8  # consecutive samples of a worker (so of one batch) share a context: 8 (s*, a*) per (r_i, f_i)


class ContextPoolDataset:
    def __init__(self, root):
        self.root = root
        self.cfg = json.load(open(f"{root}/pool.json"))  # expert_dir, instr_dropout, recent_rounds
        e = self.cfg["expert_dir"]
        self.img_off, self.wr_off = np.load(f"{e}/image_off.npy"), np.load(f"{e}/wrist_off.npy")
        self.state, self.actions = np.load(f"{e}/state.npy", mmap_mode="r"), np.load(f"{e}/actions.npy", mmap_mode="r")
        self.by_task = collections.defaultdict(list)
        for ep in json.load(open(f"{e}/episodes.json")):
            self.by_task[ep["task"]].append((ep["start"], ep["length"]))
        self._files, self._contexts, self._negatives, self._calls, self._tiles = None, [], [], 0, {}

    def _refresh(self):
        while True:
            rounds = sorted(os.path.dirname(p) for p in glob.glob(f"{self.root}/round_*/DONE"))[-self.cfg["recent_rounds"]:]
            chains = collections.defaultdict(list)
            for d in rounds:
                for p in glob.glob(f"{d}/records_*.jsonl"):
                    for line in open(p):
                        r = json.loads(line)
                        chains[(d, r["chain"])].append(r)
            ctxs, negs = [], []
            for recs in chains.values():
                negs += [(r["task"], r["frames"], r["feedback"], r["negatives"]) for r in recs if r.get("negatives")]
                task = recs[0]["task"]
                if task in self.by_task:  # task 51 has no demos
                    ctxs += [(task, None, None)] + [(task, r["frames"], r["feedback"]) for r in sorted(recs, key=lambda r: r["attempt"])]
            if ctxs:
                self._contexts, self._negatives, self._tiles = ctxs, negs, {}
                return
            time.sleep(30)  # trainer started before the first round finished

    def __len__(self):
        return 10_000_000  # samples are drawn on the fly; the index only seeds the draw

    def _png(self, which, j):
        f, off = self._files[which], self.img_off if which == 0 else self.wr_off
        f.seek(off[j])
        return np.asarray(Image.open(io.BytesIO(f.read(off[j + 1] - off[j]))).convert("RGB"))

    def __getitem__(self, i):
        if self._files is None:  # opened lazily: one handle per data-loader worker
            e = self.cfg["expert_dir"]
            self._files = (open(f"{e}/image.bin", "rb"), open(f"{e}/wrist.bin", "rb"))
        if self._calls % self.cfg.get("refresh_every", REFRESH_EVERY) == 0:
            self._refresh()
        rng = random.Random(hash((i, os.getpid(), self._calls)))
        if self._calls % SAMPLES_PER_CONTEXT == 0:
            self._ctx = rng.choice(self._contexts)
        self._calls += 1
        if self._negatives and rng.random() < self.cfg.get("negative_fraction", 0.0):
            return self._negative(rng)
        task, frames, feedback = self._ctx
        start, length = rng.choice(self.by_task[task])
        t = rng.randrange(length)
        if frames is None:
            ctx, has_ctx, prompt = context.empty_context(), False, task
        else:
            ctx, has_ctx = self._tile(frames), True
            prompt = context.prompt(None if rng.random() < self.cfg["instr_dropout"] else task, feedback)
        return {
            "observation/image": self._png(0, start + t),
            "observation/wrist_image": self._png(1, start + t),
            "observation/state": np.asarray(self.state[start + t]),
            "observation/context_image": ctx,
            "observation/has_context": np.bool_(has_ctx),
            "actions": np.asarray(self.actions[[start + min(t + k, length - 1) for k in range(ACTION_HORIZON)]]),
            "prompt": prompt,
            "negative": np.False_,
        }

    def _tile(self, frames):
        if frames not in self._tiles:
            self._tiles[frames] = context.tile(list(np.load(frames)["frames"]))
        return self._tiles[frames]

    def _negative(self, rng):
        task, frames, feedback, path = rng.choice(self._negatives)
        neg = np.load(path)
        j = rng.randrange(len(neg["state"]))
        return {
            "observation/image": neg["image"][j],
            "observation/wrist_image": neg["wrist"][j],
            "observation/state": neg["state"][j],
            "observation/context_image": self._tile(frames),
            "observation/has_context": np.True_,
            "actions": neg["actions"][j],
            "prompt": context.prompt(None if rng.random() < self.cfg["instr_dropout"] else task, feedback),
            "negative": np.True_,
        }
