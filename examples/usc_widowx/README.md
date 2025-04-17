# USC WidowX Data Conversion to LeRobot Format

This directory contains scripts to convert raw USC WidowX expert demonstration data into the LeRobot dataset format.

The primary script is `convert_usc_data_to_lerobot.py`.

## Data Structure Assumptions

The conversion script assumes your raw data is organized as follows:

```
<raw_data_base_dir>/
├── <task_name_1>/
│  ├── traj0/
│  │  ├── obs_dict.pkl
│  │  ├── policy_out.pkl
│  │  ├── external/
│  │  │   ├── image_0000.png
│  │  │   └── ...
│  │  └── over_shoulder/
│  │      ├── image_0000.png
│  │      └── ...
│  └── traj1/
│     └── ...
└── <task_name_2>/
   └── ...
```

- Each task (e.g., `close_microwave`) has its own directory.
- Inside each task directory, there are subdirectories for each trajectory (e.g., `traj0`, `traj1`, ...).
- Each trajectory directory contains `obs_dict.pkl` (with 'state' and optional 'qvel' keys), `policy_out.pkl` (with 'actions' key), and subdirectories for each camera view (e.g., `external`, `over_shoulder`) containing PNG or JPG image sequences.

## Local Conversion Example

You can run the conversion for a single task locally using `uv run`. Make sure you have installed the project dependencies (`uv sync`).

Here is an example for the `close_microwave` task, assuming your raw data is in `/home1/jessez/retrieval_widowx_datasets` and your Hugging Face Hub username/organization is `jesbu1`:

```bash
uv run examples/usc_widowx/convert_usc_data_to_lerobot.py \
    --raw-dir "/home1/jessez/retrieval_widowx_datasets/close_microwave" \
    --repo-id "jesbu1/usc_widowx_close_microwave" \
    --task "close_microwave" \
    --mode "video" \
    --push-to-hub
```

**Arguments:**
*   `--raw-dir`: Path to the specific task's raw data directory.
*   `--repo-id`: The desired Hugging Face Hub repository ID for the converted dataset (`<org_or_user>/<dataset_name>`).
*   `--task`: A label to store in the dataset metadata (usually the task name).
*   `--mode`: Conversion mode (`video` or `image`). Defaults to `video`.
*   `--push-to-hub`: If present, uploads the dataset to the Hub after conversion.
*   `--no-push-to-hub`: Explicitly disable uploading.

## Slurm Batch Conversion

A Slurm script `convert_expert_demos.slurm` is provided to convert multiple tasks in parallel using a job array.

1.  **Configure:** Edit the `convert_expert_demos.slurm` script:
    *   Set the `RAW_DATA_BASE_DIR` variable to the parent directory containing all your task subdirectories.
    *   Set the `HF_ORG` variable to your Hugging Face username or organization.
    *   Adjust the `TASKS` array to match the task directories you want to process.
    *   Modify the `#SBATCH` directives (partition, time, memory, etc.) as needed for your cluster.
    *   Set `PUSH_TO_HUB=true` or `PUSH_TO_HUB=false`.
2.  **Submit:**
    ```bash
    sbatch examples/usc_widowx/convert_expert_demos.slurm
    ```

This will launch one Slurm job for each task listed in the `TASKS` array, saving logs to a `logs/` directory (which will be created if it doesn't exist).
