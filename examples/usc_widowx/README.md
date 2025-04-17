# USC WidowX Data Conversion to LeRobot Format

This directory contains scripts to convert raw USC WidowX expert demonstration data into the LeRobot dataset format.

The primary script is `convert_usc_data_to_lerobot.py`. It can convert a single task dataset or combine multiple task datasets into one LeRobot dataset.

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

You can run the conversion locally using `uv run`. Make sure you have installed the project dependencies (`uv sync`).

You can provide input data in two ways:

### Option 1: Using Task Directories (`--raw-dirs`)

This is useful when you want to process all trajectories within one or more task directories.

To combine multiple task datasets (e.g., `close_microwave`, `push_button`) located in `/path/to/raw/usc/data` into a single LeRobot dataset named `jesbu1/usc_widowx_combined`:

```bash
uv run examples/usc_widowx/convert_usc_data_to_lerobot.py \
    --raw-dirs "/path/to/raw/usc/data/close_microwave" "/path/to/raw/usc/data/push_button" \
    --repo-id "jesbu1/usc_widowx_combined" \
    --mode "video" \
    --push-to-hub
```

### Option 2: Using Specific Trajectory Paths (`--traj-paths`)

This is useful when you want to process only specific trajectories from potentially different tasks.

To combine specific trajectories (e.g., `traj0` from `close_microwave` and `traj5` from `push_button`) into a single dataset:

```bash
uv run examples/usc_widowx/convert_usc_data_to_lerobot.py \
    --traj-paths "/path/to/raw/usc/data/close_microwave/traj0" "/path/to/raw/usc/data/push_button/traj5" \
    --repo-id "jesbu1/usc_widowx_combined_subset" \
    --mode "video" \
    --push-to-hub
```

**Arguments:**
*   `--raw-dirs`: One or more paths to the raw data directories for the tasks you want to include.
*   `--traj-paths`: One or more paths to specific trajectory directories (e.g., `/path/to/task/traj0`).
*   **Note**: You must provide *either* `--raw-dirs` *or* `--traj-paths`, but not both.
*   `--repo-id`: The desired Hugging Face Hub repository ID for the **combined** dataset (`<org_or_user>/<combined_dataset_name>`).
*   **Task Inference**: The task label for each episode is automatically inferred from the parent directory name of the trajectory (e.g., a trajectory in `/path/to/close_microwave/traj0` will get the task label "Close microwave").
*   `--mode`: Conversion mode (`video` or `image`). Defaults to `video`.
*   `--push-to-hub`: If present, uploads the dataset to the Hub after conversion.

## Slurm Batch Conversion

A Slurm script `convert_expert_demos.slurm` is provided to convert **all tasks** found within a base directory into a **single combined** LeRobot dataset.

1.  **Configure:** Edit the `convert_expert_demos.slurm` script:
    *   Set the `RAW_DATA_BASE_DIR` variable to the parent directory containing all your task subdirectories (e.g., `/path/to/raw/usc/data`). The script will find and process all subdirectories within this path.
    *   Set the `HF_ORG` variable to your Hugging Face username or organization.
    *   Set the `COMBINED_REPO_NAME` variable to the desired name for the output dataset on the Hub (e.g., `usc_widowx_combined`).
    *   Modify the `#SBATCH` directives (partition, time, memory, etc.) as needed for your cluster.
    *   Set `PUSH_TO_HUB=true` or `PUSH_TO_HUB=false`.
2.  **Submit:**
    ```bash
    sbatch examples/usc_widowx/convert_expert_demos.slurm
    ```

This will launch a single Slurm job that processes all found task directories and creates one combined dataset (e.g., `<HF_ORG>/<COMBINED_REPO_NAME>`). Logs will be saved to a `logs/` directory (which will be created if it doesn't exist).
