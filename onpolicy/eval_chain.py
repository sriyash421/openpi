"""Held-out chain eval of an on-policy checkpoint: success by attempt under real / shuffled / empty context.

Reuses run.rollout (same servers, clients, judge). Each task gets `chains` independent chains (random init each); the
task order is shuffled so a 'shuffled' donor is almost always a different task.

    python -m onpolicy.eval_chain --ckpt <lora ckpt dir> --config pi05_libero_ctx --suite libero_10 --out <dir>
"""
import collections
import dataclasses
import json
import os
import pathlib
import random

import tyro

from onpolicy import run


@dataclasses.dataclass
class Args:
    ckpt: str
    out: str
    config: str = "pi05_libero_ctx"
    suite: str = "libero_10"
    chains: int = 8
    attempts: int = 8
    modes: tuple[str, ...] = ("real", "real_noinstr", "shuffled", "empty")
    num_gpus: int = 4
    exclude: tuple[str, ...] = ("pick up the book and place it in the back compartment of the caddy",)  # also in LIBERO-90 data


def metrics(recs, k):
    """pass@k: any of the chain's k attempts succeeds. perf@k: the k-th attempt succeeds. Plus success by attempt."""
    chains = collections.defaultdict(dict)
    for r in recs:
        chains[r["chain"]][r["attempt"]] = r["success"]
    by_k = collections.defaultdict(list)
    for r in recs:
        by_k[r["attempt"]].append(r["success"])
    full = [c for c in chains.values() if len(c) == k]
    return {"chains": len(full), f"pass@{k}": sum(any(c.values()) for c in full) / max(1, len(full)),
            f"perf@{k}": sum(c[k - 1] for c in full) / max(1, len(full)),
            "success_by_attempt": [round(sum(v) / len(v), 3) for _, v in sorted(by_k.items())]}


def main(a: Args):
    n_tasks = 90 if a.suite == "libero_90" else 10
    tasks = [t for t in range(n_tasks) for _ in range(a.chains)]
    random.Random(0).shuffle(tasks)
    summary = {}
    for mode in a.modes:
        d = pathlib.Path(a.out) / mode
        d.mkdir(parents=True, exist_ok=True)
        cfg = run.Config(exp="eval", suite=a.suite, tasks=",".join(map(str, tasks)), attempts=a.attempts, num_gpus=a.num_gpus,
                         clients_per_server=min(30, (len(tasks) + a.num_gpus - 1) // a.num_gpus // 2), context_mode=mode,  # >30/server: clients hang
                         base_config=a.config, base_ckpt=a.ckpt)
        recs = run.rollout(cfg, 0, d)  # iteration 0 serves base_config/base_ckpt, i.e. exactly the given checkpoint
        summary[mode] = {"all_tasks": metrics(recs, a.attempts), "held_out": metrics([r for r in recs if r["task"] not in a.exclude], a.attempts)}
        print(mode, json.dumps(summary[mode]), flush=True)
    json.dump(summary, open(pathlib.Path(a.out) / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main(tyro.cli(Args))
