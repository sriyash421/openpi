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
    *   (Optional) Set `USE_PLAY_DATA_ONLY=true` if you only want to process task subdirectories whose names contain the word "play". Defaults to `false` (process all subdirectories).
    *   Modify the `#SBATCH` directives (partition, time, memory, etc.) as needed for your cluster.
    *   Set `PUSH_TO_HUB=true` or `PUSH_TO_HUB=false`.
2.  **Submit:**
    ```bash
    sbatch examples/usc_widowx/convert_expert_demos.slurm
    ```

This will launch a single Slurm job that processes all found task directories and creates one combined dataset (e.g., `<HF_ORG>/<COMBINED_REPO_NAME>`). Logs will be saved to a `logs/` directory (which will be created if it doesn't exist).

## Training

Once you have converted the USC WidowX data into the LeRobot format and uploaded it to the Hugging Face Hub (or have it available locally), you can fine-tune a pre-trained model (e.g., pi0) on this data.

1.  **Verify Configuration:**
    *   Open `src/openpi/training/config.py`.
    *   Find the `TrainConfig` entry named `pi0_usc_widowx_expert_data` (for expert data) or `pi0_usc_widowx_combined_play_data` (for play data).
    *   Alternatively, find the corresponding `pi0-FAST` configurations: `pi0_fast_usc_widowx_expert_data` and `pi0_fast_usc_widowx_combined_play_data`.
    *   For **LoRA** fine-tuning (recommended for lower memory GPUs), look for configurations with `_lora` in the name (e.g., `pi0_lora_usc_widowx_expert_data`, `pi0_fast_lora_usc_widowx_combined_play_data`).
    *   Ensure the `repo_id` inside the `LeRobotUSCWidowXDataConfig` matches the Hugging Face Hub repository ID of your converted dataset (e.g., `"jesbu1/usc_widowx_combined"`).
    *   Ensure `local_files_only=True` if your dataset is only local, or `False` if it should be synced from the Hub.

2.  **Normalization Stats:**
    *   Training requires normalization statistics (`norm_stats.json`). The training script expects these to be located within the assets directory corresponding to the config and dataset ID.
    *   For the `pi0_usc_widowx_expert_data` config with `repo_id="jesbu1/usc_widowx_combined"`, the default expected path would be roughly `./assets/pi0_usc_widowx_expert_data/jesbu1/usc_widowx_combined/norm_stats.json` (relative to the project root, path depends on `assets_base_dir` and `asset_id` resolution).
    If the stats don't exist, create them with, e.g.:
    ```bash
    CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name=pi0_fast_usc_widowx_combined_play_data
    ```
    where `pi0_fast_usc_widowx_combined_play_data` is the config name for the combined play data but can be replaced with `pi0_usc_widowx_expert_data` for expert data.

3.  **Run Training:**
    *   Execute the training script from the root of the `openpi` repository, specifying the config name and an experiment name:
        ```bash
        # Example for expert data config
        # Full fine-tuning:
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_usc_widowx_expert_data --exp_name=my_usc_expert_finetune --overwrite 
        # LoRA fine-tuning:
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_lora_usc_widowx_expert_data --exp_name=my_usc_lora_expert_finetune --overwrite 
 
         # Example for combined play data config
        # Full fine-tuning:
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_usc_widowx_combined_play_data --exp_name=my_usc_play_finetune --overwrite 
        # LoRA fine-tuning:
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_lora_usc_widowx_combined_play_data --exp_name=my_usc_lora_play_finetune --overwrite 
 
         # --- Examples using pi0-FAST model ---
         # Expert data:
        # Full fine-tuning:
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_usc_widowx_expert_data --exp_name=my_usc_fast_expert_finetune --overwrite
        # LoRA fine-tuning:
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_lora_usc_widowx_expert_data --exp_name=my_usc_fast_lora_expert_finetune --overwrite
         
         # Combined play data:
        # Full fine-tuning:
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_usc_widowx_combined_play_data --project_name=hand-demos-openpi-training --exp_name=pi0_fast_usc_playdata_4-17 --overwrite
        # LoRA fine-tuning:
        # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_lora_usc_widowx_combined_play_data --project_name=hand-demos-openpi-training --exp_name=pi0_fast_lora_usc_playdata_4-17 --overwrite
         ```
     *   Monitor the training progress via the console output and Weights & Biases (if enabled).
     *   Checkpoints will be saved under `./checkpoints/<config_name>/<exp_name>/` (e.g., `./checkpoints/pi0_usc_widowx_expert_data/my_usc_expert_finetune/`).

## Running Inference

Once you have a trained checkpoint (either fine-tuned or a pre-trained one like `pi0_base`), you can run inference on the real WidowX robot.

1.  **Start the Policy Server:**
    *   Run the `serve_policy.py` script, specifying the training config used and the path to the checkpoint directory.
        ```bash
        # Example using a fine-tuned checkpoint
        uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_usc_widowx_expert_data --policy.dir=./checkpoints/pi0_usc_widowx_expert_data/my_usc_expert_finetune/<step_number>/

        # Example using the base pi0 model directly (ensure config matches)
        # uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_usc_widowx_expert_data --policy.dir=s3://openpi-assets/checkpoints/pi0_base/
        ```
    *   The server will load the model and wait for connections on `localhost:8000` by default.

2.  **Run the Inference Client (`main.py`):**
    *   In a separate terminal (with the `openpi` environment sourced), run the `examples/usc_widowx/main.py` script.
    *   Provide the IP address of your WidowX robot controller and the task prompt.
        ```bash
        python examples/usc_widowx/main.py \
            --policy-server-address localhost:8000 \
            --robot-ip <your_robot_ip_here> \
            --prompt "pick up the red block" \
            --cameras external over_shoulder \
            --save-dir ./trajectory_data/usc_widowx_runs
        ```
    *   **Arguments:**
        *   `--policy-server-address`: Address of the running policy server.
        *   `--robot-ip`: Network IP address of the computer running the WidowX ROS controller.
        *   `--cameras`: List of camera names the policy expects (should match the training config/data).
        *   `--prompt`: The natural language instruction for the task.
        *   `--hz`: (Optional) Control frequency (default: 10).
        *   `--save-dir`: (Optional) Directory to save recorded trajectory data (default: `./trajectory_data/usc_widowx`).

3.  **How it Works:**
    *   The `main.py` script connects to the specified policy server and the WidowX robot controller.
    *   It enters a loop:
        *   Gets the latest observation (images from specified cameras, robot state) from the robot.
        *   Formats the observation dictionary (converting images to RGB, mapping keys) to match the structure expected by the policy's input transform (`USCWidowXInputs`).
        *   Sends the formatted observation and prompt to the policy server via `policy_client.infer()`.
        *   Receives an action chunk (a sequence of future actions) from the server.
        *   Executes the *first* action in the received chunk on the robot using `widowx_client.step_action()`.
        *   Waits briefly to maintain the target control frequency (`--hz`).
    *   **Keyboard Control:**
        *   Press `r` during a rollout to stop the current attempt and reset the robot for a new episode with the same prompt.
        *   Press `q` to stop the current attempt. You will be prompted if the attempt was successful, and the trajectory data will be saved.
    *   **Data Saving:** Trajectories (raw observations, executed actions, success status, notes) are saved using `widowx_envs.utils.raw_saver.RawSaver` in the specified `--save-dir`.
