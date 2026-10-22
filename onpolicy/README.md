# On-policy VLM feedback for π0.5 on LIBERO-90

A π0.5 policy is fine-tuned so that, on a retry, it sees **its own previous attempt** (8 frames) and **a VLM judge's
one-sentence feedback** on that attempt. Rollouts and training run asynchronously: a rollout node keeps playing chains
of attempts with the newest checkpoint, and a trainer keeps training on the most recent rounds.

This branch (`libero90-feedback`) is the code of the run that worked: `async_b128_draft01`, evaluated at step 7000.

## Result

The eval uses all 90 LIBERO-90 tasks, 4 chains per task and 3 attempts per chain. All 3 attempts start from the same
initial state, and later attempts run even after a success. The number below is **P(success on attempt k | attempt k-1
failed)**, pooled over k = 2, 3. Only the context shown on the retry changes between rows. The first attempt never has
a context, so it is the same in every row (about 0.59).

| what the policy sees on the retry                                    | P(success \| previous failed) | n failures |
|----------------------------------------------------------------------|------------------------------:|-----------:|
| **real**: previous attempt's frames + the judge's feedback           | **0.85**                      | 176 |
| generic: real frames, feedback replaced by "The previous attempt failed." | 0.69                     | 190 |
| textonly: blank frames, real feedback                                | 0.64                          | 219 |
| null: blank frames + generic line                                    | 0.53                          | 219 |
| shuffled: frames + feedback of a *different task's* attempt          | 0.40                          | 193 |
| empty: no context at all                                             | 0.32                          | 316 |

In words: frames help and feedback text helps, together they help most, and a wrong context hurts.

A rerun with a pure-LoRA freeze (`pi05_libero90_ctx_frozen`, step 8800, on a later branch) shows the same ordering,
but weaker: real 0.74, shuffled 0.41, empty 0.40.

Read this alongside the following:
* Plain BC trained to convergence on the same data reaches 0.92 success on the first attempt (step 1000 of the base
  run below). The on-policy model starts from the *undertrained* step-200 BC checkpoint, which gets 0.46. The result
  above is about *using context on a retry*, not about beating converged BC.
* The training target ignores the feedback (see the pseudo-code). The likely reason the policy reads the context
  anyway is the 0.8 instruction dropout: most context rows hide the instruction, so the task has to be inferred from
  the previous attempt's frames and the feedback. That would explain real > textonly/generic > null > shuffled.
  It is a hypothesis; it has not been isolated by an experiment.
* In later 1-shot LIBERO-90 runs (1 demo per task, 1k steps, pure LoRA, no BC term on no-context rows) the gap
  between real and generic feedback disappeared.

## What we did, in pseudo-code

```python
# ---------------- rollout node (onpolicy/run.py -> onpolicy/rollout_client.py, onpolicy/judge.py) ----------------
for round in range(R):
    policy = newest trainer checkpoint          # round 0 / before the first save: the BC base (step 200)
    for task in LIBERO_90:                      # one chain per task per round
        s0 = random init state of task
        ctx = None
        for attempt in range(3):                # every attempt restarts from the same s0
            traj = run(policy, s0, prompt(task.instruction, ctx), context_image(ctx))
            frames = 8 frames, evenly spaced over traj
            success, feedback = VLM_judge(task.instruction, frames)   # Gemini via OpenRouter: success + one sentence
            save(record(task, attempt, env_success, feedback), frames, video)
            ctx = (frames, feedback)            # the next attempt conditions on this one
    mark round DONE

# ---------------- trainer (scripts/train.py with repo_id pool:<run root> -> onpolicy/pool.py), never restarts ------
contexts = for every chain in the 5 newest DONE rounds: [None, (frames_0, fb_0), (frames_1, fb_1), (frames_2, fb_2)]
for step in range(N):                           # batch 128
    c = uniform choice from contexts            # 8 consecutive samples share one c
    s, a = random timestep of a random expert demo of c's task      # target ignores c's content
    if c is None:                               # "first attempt" row: keep it exactly the base policy
        obs = (s, instruction, no context image)
        loss = 0.9 * ||v_theta(obs) - v_base(obs)||^2 + 0.1 * flow_matching_loss(a | obs)
    else:                                       # "retry" row
        text = feedback if rand() < 0.8 else instruction + feedback  # instruction dropout 0.8
        obs = (s, text, context_image = 2x4 tile of c.frames in the unused right-wrist camera slot)
        loss = flow_matching_loss(a | obs)
    update LoRA adapters + the non-LLM parameters the default openpi LoRA filter leaves trainable (SigLIP, projections)
```

`v_base` is the same network with every `lora_b` zeroed, so attempt-0 behavior stays the BC base's (an anchor in the
spirit of DiT). The context enters π0.5 through the third image slot (`right_wrist_0_rgb`), which LIBERO doesn't use,
and the feedback goes into the prompt, so no architecture change is needed.

### Why each piece

* **Undertrained base (step 200, 0.46 success).** The base needs to fail often enough to produce retry data, and to
  leave room for retries to help.
* **Same initial state for every attempt.** A retry is a second try at *the same* problem, so the previous attempt is
  informative about this one.
* **8 frames + one sentence.** This is cheap for the judge and fits one image slot and the prompt.
* **Anchor on no-context rows.** On-policy training should change retries, not first attempts. Holding attempt 0 at
  the base (plus a small BC term) keeps the first attempt stable, so gains on retries are attributable to the context.
* **Demo actions as targets.** No queryable expert exists, so supervision comes only from the demos (no RL reward, no
  relabeled policy actions).
* **Asynchronous rounds.** Rollouts (simulator plus judge) and training (H200s) run in parallel. The trainer reads the 5
  newest rounds, so its data stays near on-policy.

## Exact setup of the run

| | |
|---|---|
| Data | LIBERO-90, all 50 demos/task (`ryanhoangt/libero_90_lerobot_v21`) |
| BC base | `pi05_libero90` from `pi05_base`, 1001 steps, batch 256, warmup 333, EMA 0.99, **step 200** used |
| On-policy config | `pi05_libero90_ctx` (LoRA gemma_2b + gemma_300m; default openpi LoRA freeze filter) |
| Trainer | batch 128, lr 5e-5 (constant after 100 warmup), `--anchor-no-context --anchor-beta=0.9 --anchor-draft-weight=0.1`, 2x H200 |
| Pool | `instr_dropout 0.8`, `recent_rounds 5`, `negative_fraction 0` |
| Rollouts | 3 attempts per chain, one chain per task per round, 3x L40S, 30 clients per policy server |
| Judge | `google/gemini-3.8-flash` via OpenRouter, low reasoning effort, 8 frames at 256 px |
| Evaluated | step 7000 (around round 60) |
| Judge cost | roughly $0.007 per judged attempt, so on the order of $100 for this run (estimated from later runs' ledgers) |

## Reproduce

The scripts are Slurm + Apptainer and come from UW's Hyak cluster. Two paths are hardcoded and need editing:
`S=/gscratch/scrubbed/sriyash/openpi_pt` (scratch: venv, caches, data, checkpoints, rollouts) and
`/gscratch/socialrl/sriyash/openpi_pt` (this repo). They appear in `onpolicy/jobs/*.sh`, `onpolicy/run.py`,
`onpolicy/eval_chain.py` and `onpolicy/apptainer/libero.def`. The commands below assume the job scripts were copied
to `$S/jobs/` and `onpolicy/jobs/{env.sh,run.sh}` to `$S/`, which is how they were run.

```bash
# 0. clone with LIBERO
git clone -b libero90-feedback --recurse-submodules git@github.com:sriyash421/openpi.git && cd openpi
export S=/your/scratch/openpi_pt O=$PWD
mkdir -p $S/{apptainer,jobs,logs}; cp onpolicy/jobs/*.sh $S/jobs/; cp onpolicy/jobs/{env,run}.sh $S/
cp onpolicy/apptainer/*.def $S/apptainer/

# 1. containers (Ubuntu 22.04 for openpi, py3.8 for LIBERO), venv, checkpoints + LIBERO-90 data, norm stats
bash $S/jobs/build_sifs.sh
bash $S/jobs/uv_sync.sh
bash $S/jobs/download.sh            # pi05_base, pi05_libero, ryanhoangt/libero_90_lerobot_v21
bash $S/jobs/norm_stats.sh          # pi05_libero90

# 2. BC base: 1001 steps, use step 200
sbatch --gres=gpu:h200:2 -c 32 --mem=400G -t 1-00:00:00 $S/jobs/train.sh libero90_1k_w333 \
    --num-train-steps=1001 --save-interval=100 --keep-period=100 --lr-schedule.warmup-steps=333 --ema-decay=0.99

# 3. expert index for on-the-fly demo sampling ($S/expert_index)
bash $S/jobs/expert_index.sh

# 4. on-policy run: trainer + rollouts, launched together (the trainer waits for round 0)
export OPENROUTER_API_KEY=...  JUDGE_BUDGET=300   # the judge refuses to spend past the budget (ledger $S/judge_ledger.txt)
EXP=async_b128_draft01
sbatch --gres=gpu:h200:2 -c 32 --mem=300G -t 3-00:00:00 --requeue -o $S/logs/train_$EXP.log \
    $S/jobs/train_pool.sh $EXP 0.8 5 --batch-size=128 --anchor-beta=0.9 --anchor-draft-weight=0.1
sbatch --gres=gpu:3 -c 88 --mem=400G -t 3-00:00:00 --requeue -o $S/logs/onpolicy_$EXP.log \
    $S/jobs/onpolicy.sh --exp $EXP --num-iters 100 --num-gpus 3 --clients-per-server 30 \
    --base-config pi05_libero90 --base-ckpt $S/checkpoints/pi05_libero90/libero90_1k_w333/200 --train-config pi05_libero90_ctx
# stop both after step 7000 is saved ($S/checkpoints/pi05_libero90_ctx/$EXP/7000)

# 5. eval with context controls (4 GPUs each)
CK=$S/checkpoints/pi05_libero90_ctx/$EXP/7000
for M in "real shuffled empty" "generic null" "textonly"; do
  sbatch --gres=gpu:4 -c 48 --mem=180G -t 8:00:00 $S/jobs/eval_chain.sh --ckpt $CK --config pi05_libero90_ctx \
      --suite libero_90 --chains 4 --attempts 3 --modes $M --out $S/onpolicy/eval_libero90_diag/${EXP}_7000_${M// /_}
done
```

Then compute P(success | previous attempt failed) from the eval records:

```python
import collections, glob, json
for mode_dir in sorted(glob.glob(f"{S}/onpolicy/eval_libero90_diag/*/*/")):
    chains = collections.defaultdict(dict)
    for p in glob.glob(mode_dir + "records_*.jsonl"):
        for line in open(p):
            r = json.loads(line); chains[r["chain"]][r["attempt"]] = bool(r["success"])
    after_fail = [c[k] for c in chains.values() for k in (1, 2) if k in c and k - 1 in c and not c[k - 1]]
    print(mode_dir, round(sum(after_fail) / len(after_fail), 2), len(after_fail))
```

## Files

| file | what |
|---|---|
| `onpolicy/run.py` | rollout side of the async loop: one round = serve the newest checkpoint, run chains, mark DONE |
| `onpolicy/rollout_client.py` | LIBERO client: chained attempts from one init state, context modes (real/generic/textonly/null/shuffled/empty) |
| `onpolicy/judge.py` | VLM judge (OpenRouter), cost ledger and budget cap |
| `onpolicy/context.py` | frame subsampling, 2x4 tiling into the context image, prompt format |
| `onpolicy/pool.py` | trainer dataset: context from recent rounds × demo transition of the same task |
| `onpolicy/expert_index.py` | flattens the LIBERO-90 parquet into random-access arrays |
| `onpolicy/eval_chain.py` | held-out chain eval: success by attempt and pass@k per context mode |
| `onpolicy/batched_server.py` | optional batched policy server (not used in this run) |
| `scripts/train.py` | openpi trainer plus the no-context anchor loss (`anchor_*`) and optional negatives (off here) |
| `src/openpi/training/config.py` | `pi05_libero90`, `pi05_libero90_ctx`, context data config |
| `onpolicy/localize_*.py`, `onpolicy/wandb_report*.py` | offline failure-localization study and W&B reports (analysis only) |
| `onpolicy/jobs/`, `onpolicy/apptainer/` | Slurm job scripts and Apptainer recipes |
