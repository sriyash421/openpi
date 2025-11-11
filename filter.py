#!/usr/bin/env python3
# make_filtered_libero.py
import argparse, os, json, glob, math
from collections import OrderedDict

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc


def load_info(info_path):
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r))
            f.write("\n")

def parse_task_ids(arg):
    # supports: "3", "3,7,9", multiple --task-id flags, or a file path
    if isinstance(arg, list):
        vals = []
        for a in arg:
            vals += parse_task_ids(a)
        return vals
    s = str(arg)
    if os.path.isfile(s):
        with open(s, "r") as f:
            return [int(x.strip()) for x in f if x.strip()]
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    return [int(s)]

def iter_episode_parquets(data_root):
    # expects .../data/chunk-*/episode_*.parquet (zero-padded or not)
    patt1 = os.path.join(data_root, "data", "chunk-*", "episode_*.parquet")
    for p in sorted(glob.glob(patt1)):
        yield p

def read_task_index_for_episode(path):
    # read only the task_index column quickly
    tab = pq.read_table(path, columns=["task_index"])
    # Use the first row (assumes single task per episode)
    if tab.num_rows == 0:
        return None
    # Allow scalar or array; take the majority if needed
    arr = tab["task_index"]
    try:
        # fast path: all equal
        uniq = pc.unique(arr).to_pylist()
        return int(uniq[0]) if len(uniq) == 1 else int(pc.mode(arr)["mode"])
    except Exception:
        # fallback
        py = arr.to_pylist()
        return int(py[0]) if py else None

def rewrite_episode_table(path, new_ep_index):
    # load full table, and rewrite episode_index and frame_index
    table = pq.read_table(path)  # keep all columns
    cols = table.column_names

    # episode_index -> new_ep_index
    if "episode_index" in cols:
        ep_col = pa.array([new_ep_index] * table.num_rows, type=pa.int64())
        table = table.set_column(cols.index("episode_index"), "episode_index", ep_col)

    # frame_index -> 0..T-1
    if "frame_index" in cols:
        T = table.num_rows
        fr_col = pa.array(list(range(T)), type=pa.int64())
        table = table.set_column(cols.index("frame_index"), "frame_index", fr_col)

    return table

def main():
    ap = argparse.ArgumentParser(description="Filter a LIBERO/LeRobot dataset by task_id(s) and reproduce the same structure.")
    ap.add_argument("--in-root", required=True, help="Original dataset root (contains data/ and meta/).")
    ap.add_argument("--out-root", required=True, help="Destination root for the filtered dataset.")
    ap.add_argument("--task-id", action="append", required=True,
                    help="Task id(s) to keep. Can repeat, comma-separate, or pass a file path with one id per line.")
    ap.add_argument("--dry-run", action="store_true", help="Plan only; do not write any files.")
    ap.add_argument("--max-episodes", type=int, default=-1, help="If >0, limit to this many episodes (for testing).")
    args = ap.parse_args()

    in_root  = os.path.abspath(args.in_root)
    out_root = os.path.abspath(args.out_root)
    os.makedirs(out_root, exist_ok=True)

    info_path_in   = os.path.join(in_root, "meta", "info.json")
    tasks_path_in  = os.path.join(in_root, "meta", "tasks.jsonl")
    episodes_path_in = os.path.join(in_root, "meta", "episodes.jsonl")

    if not os.path.isfile(info_path_in) or not os.path.isfile(tasks_path_in):
        raise FileNotFoundError("Could not find meta/info.json or meta/tasks.jsonl under the input root.")

    info_in  = load_info(info_path_in)
    tasks_in = load_jsonl(tasks_path_in)
    # episodes.jsonl is optional for filtering (we recompute lengths), but we’ll use it if present to attach text
    episodes_in = load_jsonl(episodes_path_in) if os.path.isfile(episodes_path_in) else []

    chunk_size = int(info_in.get("chunks_size", 1000))
    fps        = int(info_in.get("fps", 10))
    codebase_version = info_in.get("codebase_version", "v2.0")
    robot_type       = info_in.get("robot_type", "panda")
    data_path_tpl    = info_in.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_path_tpl   = info_in.get("video_path", "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    features_in      = info_in.get("features", {})

    # Build task_index -> task text (if present in tasks.jsonl)
    # tasks.jsonl lines look like: {"task_index": 57, "task": "put the white bowl..."}
    id2task = {}
    for r in tasks_in:
        if "task_index" in r:
            id2task[int(r["task_index"])] = r.get("task", "")

    keep_task_ids = sorted(set(parse_task_ids(args.task_id)))
    print(f"[info] Keeping task_ids: {keep_task_ids}")

    # Scan episodes and select those whose parquet contains one of the keep_task_ids
    plan = []  # list of dicts: {src, old_ep_index, task_index, length, new_ep_index, out_path}
    total_frames = 0
    kept_task_ids_present = set()

    patt1 = os.path.join(in_root, "data", "chunk-*", "episode_*.parquet")
    parquets = sorted(glob.glob(patt1))
    import tqdm
    num_episodes_added = 0
    for ep_path in tqdm.tqdm(parquets):
        # Parse old episode_index from filename
        base = os.path.basename(ep_path)
        # expects episode_{index}.parquet
        try:
            old_ep_index = int(os.path.splitext(base)[0].split("_")[1])
        except Exception:
            # skip unexpected filenames
            continue

        ti = read_task_index_for_episode(ep_path)
        if ti is None or ti not in keep_task_ids:
            continue
        if 0 < args.max_episodes <= num_episodes_added:
            break
        num_episodes_added += 1
        # Read once to get length
        meta_tab = pq.read_table(ep_path, columns=["frame_index"])
        length = meta_tab.num_rows
        kept_task_ids_present.add(ti)

        plan.append({
            "src": ep_path,
            "old_ep_index": old_ep_index,
            "task_index": int(ti),
            "length": int(length),
        })
        total_frames += int(length)

    # Assign new contiguous episode indices
    plan.sort(key=lambda x: (x["task_index"], x["old_ep_index"]))
    for new_idx, rec in enumerate(plan):
        rec["new_ep_index"] = new_idx
        rec["episode_chunk"] = new_idx // chunk_size
        # destination path using template
        rel_out = data_path_tpl.format(episode_chunk=rec["episode_chunk"], episode_index=rec["new_ep_index"])
        rec["rel_out"] = rel_out
        rec["abs_out"] = os.path.join(out_root, rel_out)

    print(f"[info] Selected episodes: {len(plan)}; total frames: {total_frames}")

    # Write parquet files
    if not args.dry_run:
        for rec in plan:
            os.makedirs(os.path.dirname(rec["abs_out"]), exist_ok=True)
            new_table = rewrite_episode_table(rec["src"], rec["new_ep_index"])
            pq.write_table(new_table, rec["abs_out"])

    # Build meta/tasks.jsonl (only kept tasks)
    tasks_out_rows = []
    for tid in sorted(kept_task_ids_present):
        tasks_out_rows.append({"task_index": int(tid), "task": id2task.get(tid, "")})
    if not args.dry_run:
        write_jsonl(os.path.join(out_root, "meta", "tasks.jsonl"), tasks_out_rows)

    # Build meta/episodes.jsonl (reindexed, attach task text if available)
    # If original had multiple "tasks" strings, keep a single string from mapping.
    episodes_out_rows = []
    for rec in plan:
        # prefer tasks.jsonl mapping; fallback to original episodes.jsonl if it had text
        task_text = id2task.get(rec["task_index"], "")
        if not task_text and episodes_in:
            # try to find by old_ep_index
            for e in episodes_in:
                if int(e.get("episode_index", -1)) == rec["old_ep_index"]:
                    # episodes.jsonl example had "tasks": ["..."]; normalize to string list
                    t = e.get("tasks", [])
                    if isinstance(t, list) and t:
                        task_text = t[0]
                    elif isinstance(t, str):
                        task_text = t
                    break
        episodes_out_rows.append({
            "episode_index": rec["new_ep_index"],
            "tasks": [task_text] if task_text else [],
            "length": rec["length"],
            "task_index": rec["task_index"],
        })
    if not args.dry_run:
        write_jsonl(os.path.join(out_root, "meta", "episodes.jsonl"), episodes_out_rows)

    # Build meta/info.json
    info_out = OrderedDict()
    info_out["codebase_version"] = codebase_version
    info_out["robot_type"] = robot_type
    info_out["total_episodes"] = len(plan)
    info_out["total_frames"] = total_frames
    info_out["total_tasks"] = len(kept_task_ids_present)
    info_out["total_videos"] = 0  # unchanged unless you also copy videos
    info_out["total_chunks"] = math.ceil(len(plan) / chunk_size) if len(plan) > 0 else 0
    info_out["chunks_size"] = chunk_size
    info_out["fps"] = fps
    info_out["splits"] = {"train": f"0:{len(plan)}"}
    info_out["data_path"] = data_path_tpl
    info_out["video_path"] = video_path_tpl
    info_out["features"] = info_in.get("features", {})

    if not args.dry_run:
        out_info_path = os.path.join(out_root, "meta", "info.json")
        os.makedirs(os.path.dirname(out_info_path), exist_ok=True)
        with open(out_info_path, "w", encoding="utf-8") as f:
            json.dump(info_out, f, indent=2)
    print(f"[done] Wrote filtered dataset to: {out_root}")
    print(f"       episodes: {len(plan)}, tasks: {sorted(kept_task_ids_present)}, frames: {total_frames}")

if __name__ == "__main__":
    main()
