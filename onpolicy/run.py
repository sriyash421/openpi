"""Async on-policy loop, rollout side: rounds of chained rollouts + VLM feedback on one GPU node (servers + clients).

Training runs separately and continuously (jobs/train_pool.sh: openpi train.py on repo_id pool:<root>, onpolicy/pool.py);
each round serves its newest checkpoint. Rounds live in <root>/<exp>/round_XXX/ with a DONE marker (requeue-safe).
Runs on the host (stdlib + wandb) and drives the containers via $S/run.sh (openpi) and apptainer (libero).

    python -m onpolicy.run --exp <exp> --num-iters 100
"""
import dataclasses
import glob
import json
import os
import pathlib
import shutil
import subprocess
import time

import tyro

from onpolicy import judge

S = "/gscratch/scrubbed/sriyash/openpi_pt"
O = "/gscratch/socialrl/sriyash/openpi_pt"
BASE_CKPT = f"{S}/openpi_data/openpi-assets/checkpoints/pi05_libero"


@dataclasses.dataclass
class Config:
    exp: str = "pilot"
    num_iters: int = 20
    num_gpus: int = 4
    servers_per_gpu: int = 1  # ckpt GPUs are exclusive-process: one CUDA context per GPU (EGL clients are fine)
    clients_per_server: int = 16
    suite: str = "libero_90"
    tasks: str = "all"  # "all" or comma-separated ids (smoke tests)
    attempts: int = 3  # r0 f0 r1 f1 r2 f2: every attempt is a context; no r3, nothing would condition on it
    judge_model: str = "google/gemini-3.8-flash"
    batched_server: bool = False  # onpolicy/batched_server.py instead of openpi's one-request-at-a-time server
    base_config: str = "pi05_libero"  # iteration 0 serves this config/checkpoint as is
    base_ckpt: str = BASE_CKPT
    train_config: str = "pi05_libero_ctx"  # LoRA config trained (and served from iteration 1)
    keep_period: int = 1000  # the trainer's --keep-period: only these checkpoints are safe to serve
    archive_period: int = 1000  # served checkpoints older than the current one are deleted unless a multiple of this
    negatives: int = 0  # rollout_client --negatives: (s_i, a_i) points saved per failed attempt
    context_mode: str = "real"  # rollout_client context: real | empty | shuffled (eval counterfactuals)
    root: str = f"{S}/onpolicy"
    wandb_project: str = "openpi-libero"


def sh(cmd, log, env=None):
    # Children must not inherit this process's wandb service socket (unreachable from inside the containers).
    base = {k: v for k, v in os.environ.items() if not k.startswith(("WANDB_SERVICE", "_WANDB"))}
    return subprocess.Popen(cmd, shell=True, stdout=open(log, "w"), stderr=subprocess.STDOUT, env={**base, **(env or {})})


def policy_for(cfg, it):
    """Round 0, and any round before the trainer's first save, serves the base checkpoint; otherwise the newest one."""
    ckpt_dir = pathlib.Path(f"{S}/checkpoints/{cfg.train_config}/{cfg.exp}")
    # Only checkpoints the trainer keeps for good (multiples of its keep_period): the latest rolling one gets deleted
    # by Orbax at the next save, possibly while the servers are still loading it.
    steps = sorted(int(p.name) for p in ckpt_dir.iterdir() if p.name.isdigit() and int(p.name) % cfg.keep_period == 0
                   and (p / "params").exists()) if ckpt_dir.exists() else []
    if it == 0 or not steps:
        return cfg.base_config, cfg.base_ckpt
    for old in steps[:-1]:  # keep_period can be small (fresh policy); keep disk bounded
        if old % cfg.archive_period:
            shutil.rmtree(ckpt_dir / str(old), ignore_errors=True)
    return cfg.train_config, str(ckpt_dir / str(steps[-1]))


def rollout(cfg, it, d):
    config, ckpt = policy_for(cfg, it)
    n_servers = cfg.num_gpus * cfg.servers_per_gpu
    ports = [9100 + (os.getpid() % 500) * 2 + i for i in range(n_servers)]
    servers = []
    for i, port in enumerate(ports):
        env = {"CUDA_VISIBLE_DEVICES": str(i // cfg.servers_per_gpu), "XLA_PYTHON_CLIENT_MEM_FRACTION": str(0.5 / cfg.servers_per_gpu)}  # headroom for the EGL clients on the same GPU
        cmd = (f"PYTHONPATH={O} uv run python -m onpolicy.batched_server --config {config} --dir {ckpt} --port {port}" if cfg.batched_server
               else f"uv run scripts/serve_policy.py --port {port} policy:checkpoint --policy.config={config} --policy.dir={ckpt}")
        servers.append(sh(f"{S}/run.sh '{cmd}'", d / f"server_{i}.log", env))
    for i in range(n_servers):
        while "server listening" not in (d / f"server_{i}.log").read_text():
            assert servers[i].poll() is None, f"server {i} died, see {d}/server_{i}.log"
            time.sleep(5)
    n_tasks = 90 if cfg.suite == "libero_90" else 10
    task_ids = list(range(n_tasks)) if cfg.tasks == "all" else [int(x) for x in cfg.tasks.split(",")]
    n_clients = min(len(task_ids), n_servers * cfg.clients_per_server)
    clients = []
    for c in range(n_clients):
        mine = ",".join(str(t) for t in task_ids[c::n_clients])
        clients.append(sh(
            f"apptainer exec --nv --home {S}/home_libero --env NUMBA_CACHE_DIR={S}/xdg_cache/numba,PYTHONDONTWRITEBYTECODE=1 "
            f"--bind {O}:/app --bind /gscratch:/gscratch --pwd /app {S}/apptainer/libero.sif python onpolicy/rollout_client.py "
            f"--port {ports[c % n_servers]} --task-suite-name {cfg.suite} --tasks {mine} --attempts {cfg.attempts} "
            f"--judge-model {cfg.judge_model} --out-dir {d} --seed {it * 1000 + c} --context-mode {cfg.context_mode} --negatives {cfg.negatives}", d / f"client_{c}.log",
            {"CUDA_VISIBLE_DEVICES": str((c % n_servers) // cfg.servers_per_gpu)}))
    for c in clients:
        c.wait()
    for s in servers:
        s.terminate()
    return [json.loads(line) for p in glob.glob(f"{d}/records_*.jsonl") for line in open(p)]


def log_iteration(run, it, records):
    by_k = {}
    for r in records:
        by_k.setdefault(r["attempt"], []).append(r["success"])
    judged = [r for r in records if r["judge_success"] is not None]
    run.log({**{f"rollout/success_attempt{k}": sum(v) / len(v) for k, v in sorted(by_k.items())},
             "rollout/success": sum(r["success"] for r in records) / len(records),
             "judge/agreement": sum(r["judge_success"] == r["success"] for r in judged) / max(1, len(judged)),
             "judge/errors": len(records) - len(judged), "judge/cost": sum(r["judge_cost"] for r in records),
             "iteration": it}, step=it)


def main(cfg: Config):
    """Rollout side of the async loop: each round serves the trainer's newest checkpoint (the base before its first
    save), runs one chain per task, and marks the round DONE so onpolicy/pool.py picks its contexts up."""
    import wandb

    root = pathlib.Path(cfg.root) / cfg.exp
    root.mkdir(parents=True, exist_ok=True)
    id_path = root / "wandb_id.txt"
    run = wandb.init(project=cfg.wandb_project, name=f"{cfg.exp}-rollouts", id=id_path.read_text() if id_path.exists() else None,
                     resume="allow", config=dataclasses.asdict(cfg))
    id_path.write_text(run.id)
    for r in range(cfg.num_iters):
        d = root / f"round_{r:03d}"
        if (d / "DONE").exists():
            continue
        if judge.spent() >= judge.BUDGET:
            print(f"judge budget reached (${judge.spent():.2f} >= ${judge.BUDGET:.0f}): stopping before round {r}")
            break
        d.mkdir(exist_ok=True)
        for p in glob.glob(f"{d}/records_*.jsonl"):  # preempted mid-round: redo it
            os.remove(p)
        records = rollout(cfg, r, d)
        if judge.spent() >= judge.BUDGET:  # clients stopped mid-round: leave it unfinished (not DONE)
            print(f"judge budget reached during round {r}: stopping")
            break
        (d / "policy.txt").write_text(" ".join(policy_for(cfg, r)))
        log_iteration(run, r, records)
        (d / "DONE").touch()
    run.finish()


if __name__ == "__main__":
    main(tyro.cli(Config))
