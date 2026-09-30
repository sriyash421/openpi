"""One-time: flatten the LIBERO-90 LeRobot parquet into random-access arrays for on-the-fly expert sampling.

    <out>/image.bin, wrist.bin   concatenated PNG bytes of every frame (image_off / wrist_off: N+1 byte offsets)
    <out>/state.npy (N, 8), actions.npy (N, 7), episodes.json [{episode, task, start, length}]

Random access into parquet row groups is what made dataset loading slow; here one sample is two reads + PNG decodes.
    python -m onpolicy.expert_index
"""
import concurrent.futures as cf
import json
import os

import numpy as np
import pyarrow.parquet as pq

DEMO_ROOT = "/gscratch/scrubbed/sriyash/openpi_pt/hf/lerobot/ryanhoangt/libero_90_lerobot_v21"
OUT = "/gscratch/scrubbed/sriyash/openpi_pt/expert_index"


def read(ep):
    cols = pq.read_table(f"{DEMO_ROOT}/data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet",
                         columns=["image", "wrist_image", "state", "actions"]).to_pydict()
    return ep, [x["bytes"] for x in cols["image"]], [x["bytes"] for x in cols["wrist_image"]], cols["state"], cols["actions"]


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = [json.loads(l) for l in open(f"{DEMO_ROOT}/meta/episodes.jsonl")]
    task_of = {m["episode_index"]: m["tasks"][0] for m in meta}
    img_off, wr_off, states, actions, episodes = [0], [0], [], [], []
    with open(f"{OUT}/image.bin", "wb") as fi, open(f"{OUT}/wrist.bin", "wb") as fw, cf.ProcessPoolExecutor(32) as ex:
        for ep, imgs, wrists, st, ac in ex.map(read, sorted(task_of), chunksize=4):
            episodes.append({"episode": ep, "task": task_of[ep], "start": len(states), "length": len(ac)})
            for b in imgs:
                fi.write(b)
                img_off.append(img_off[-1] + len(b))
            for b in wrists:
                fw.write(b)
                wr_off.append(wr_off[-1] + len(b))
            states += st
            actions += ac
    np.save(f"{OUT}/image_off.npy", np.asarray(img_off, np.int64))
    np.save(f"{OUT}/wrist_off.npy", np.asarray(wr_off, np.int64))
    np.save(f"{OUT}/state.npy", np.asarray(states, np.float32))
    np.save(f"{OUT}/actions.npy", np.asarray(actions, np.float32))
    json.dump(episodes, open(f"{OUT}/episodes.json", "w"))
    print(f"{len(episodes)} episodes, {len(states)} frames -> {OUT}")


if __name__ == "__main__":
    main()
