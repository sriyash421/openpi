#!/bin/bash

#SBATCH --account=biyik_1165                   # Account from launch_eval.slurm
#SBATCH --job-name=usc_lerobot_convert          # Job name
#SBATCH --output=logs/usc_convert_%A_%a.out    # Output log file
#SBATCH --error=logs/usc_convert_%A_%a.err     # Error log file
#SBATCH --nodes=1                              # Nodes
#SBATCH --time=1:00:00                         # Time limit
#SBATCH --cpus-per-task=16                     # CPUs per task
#SBATCH --mem=64G                              # Memory
#SBATCH --partition=main,debug                 # Partition set to 'main' (no GPU)
#SBATCH --array=1-5                            # Array tasks

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

# --- User Configuration ---
# Base directory containing task subdirectories (e.g., /path/to/raw/usc/data)
# The script assumes subdirectories named exactly like the tasks below exist here.
RAW_DATA_BASE_DIR="/home1/jessez/retrieval_widowx_datasets"  # <<<--- CHANGE THIS to the parent directory of your task data
# Hugging Face Hub organization name (e.g., lerobot)
HF_ORG="jesbu1"                            # <<<--- CHANGE THIS to your HF organization/username
# Set to 'true' to push the dataset to the Hub after conversion, 'false' otherwise
PUSH_TO_HUB=true
# Conversion mode ('video' or 'image') - should match default in python script unless overridden
CONVERSION_MODE="video"
# --------------------------

# List of tasks corresponding to the array index (1-based)
declare -a TASKS
TASKS[1]="close_microwave"
TASKS[2]="pick_up_green_and_slide_block_right"
TASKS[3]="push_button"
TASKS[4]="reach_green_block_distractors"
TASKS[5]="slide_pot_right"

# Get the task name for the current Slurm array task ID
CURRENT_TASK_INDEX=${SLURM_ARRAY_TASK_ID}
if [[ -z "${TASKS[$CURRENT_TASK_INDEX]+unset}" ]]; then
  echo "Error: No task defined for array index ${CURRENT_TASK_INDEX}."
  exit 1
fi
TASK_NAME=${TASKS[$CURRENT_TASK_INDEX]}

# Define task-specific paths and repo ID
TASK_RAW_DIR="${RAW_DATA_BASE_DIR}/${TASK_NAME}"
REPO_ID="${HF_ORG}/usc_widowx_${TASK_NAME}" # Construct repo ID like <org>/usc_widowx_<task_name>

echo "=========================================================="
echo "Starting Slurm Task ${SLURM_ARRAY_TASK_ID}"
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "----------------------------------------------------------"
echo "Task Name:          ${TASK_NAME}"
echo "Raw Data Directory: ${TASK_RAW_DIR}"
echo "Target LeRobot Repo ID: ${REPO_ID}"
echo "Push to Hub:        ${PUSH_TO_HUB}"
echo "Conversion Mode:    ${CONVERSION_MODE}"
echo "=========================================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# --- Activate Environment & Run Script ---
# This assumes you are submitting the job from the root of the 'openpi' workspace
# and that `uv` is available in your environment.
# `uv run` handles activating the project's virtual environment.

echo "Running Python conversion script..."

COMMAND="uv run examples/usc_widowx/convert_usc_data_to_lerobot.py \
    --raw-dir \"${TASK_RAW_DIR}\" \
    --repo-id \"${REPO_ID}\" \
    --task \"${TASK_NAME}\" \
    --mode \"${CONVERSION_MODE}\""

# Add --push-to-hub flag if PUSH_TO_HUB is true
# Tyro handles boolean flags: presence means true, absence or --no-<flag> means false.
if [ "$PUSH_TO_HUB" = true ] ; then
    COMMAND+=" --push-to-hub"
else
    COMMAND+=" --no-push-to-hub" # Explicitly disable push
fi

echo "Executing command:"
echo "${COMMAND}"
echo "----------------------------------------------------------"

eval $COMMAND

echo "=========================================================="
echo "Finished Slurm Task ${SLURM_ARRAY_TASK_ID} for task '${TASK_NAME}'"
echo "Time: $(date)"
echo "=========================================================="
