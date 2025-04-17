#!/bin/bash

#SBATCH --account=biyik_1165                   # Account from launch_eval.slurm
#SBATCH --job-name=usc_lerobot_convert_combined # Job name
#SBATCH --output=logs/usc_convert_combined_%j.out # Output log file (%j for job ID)
#SBATCH --error=logs/usc_convert_combined_%j.err  # Error log file (%j for job ID)
#SBATCH --nodes=1                              # Nodes
#SBATCH --time=1:00:00                         # Time limit
#SBATCH --cpus-per-task=16                     # CPUs per task
#SBATCH --mem=64G                              # Memory
#SBATCH --partition=main,debug                 # Partition set to 'main' (no GPU)

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error when substituting.

# --- User Configuration ---
# Base directory containing task subdirectories (e.g., /path/to/raw/usc/data)
# The script will process ALL subdirectories found here as separate tasks.
ONLY_PLAY_DATA=false
RAW_DATA_BASE_DIR="/home1/jessez/retrieval_widowx_datasets"  # <<<--- CHANGE THIS to the parent directory of your task data
# Hugging Face Hub organization name (e.g., lerobot)
HF_ORG="jesbu1"                            # <<<--- CHANGE THIS to your HF organization/username
# Define the name for the combined dataset repository on the Hub
COMBINED_REPO_NAME="usc_widowx_combined"  # <<<--- CHANGE THIS if you want a different name
if [ "$ONLY_PLAY_DATA" = true ]; then
    COMBINED_REPO_NAME="${COMBINED_REPO_NAME}_play_data"
fi
# Set to 'true' to push the dataset to the Hub after conversion, 'false' otherwise
PUSH_TO_HUB=true
# Set to 'true' to ONLY process subdirectories containing "play" in their name
# Conversion mode ('video' or 'image') - should match default in python script unless overridden
CONVERSION_MODE="video"
# --------------------------

# Find all subdirectories in the RAW_DATA_BASE_DIR - these are assumed to be the task directories
shopt -s nullglob
TASK_DIRS=("${RAW_DATA_BASE_DIR}"/*/)
shopt -u nullglob # Turn off nullglob

if [ ${#TASK_DIRS[@]} -eq 0 ]; then
    echo "Error: No task subdirectories found in ${RAW_DATA_BASE_DIR}"
    exit 1
fi

# Filter for play data directories if requested. If not play data, filter for non-play data.
if [ "$ONLY_PLAY_DATA" = true ]; then
    echo "Filtering for directories containing 'play'..."
    FILTERED_TASK_DIRS=()
    for dir in "${TASK_DIRS[@]}"; do
        task_name=$(basename "${dir%/}") # Extract directory name
        if [[ "$task_name" == *play* ]]; then
            FILTERED_TASK_DIRS+=("$dir")
        fi
    done

    if [ ${#FILTERED_TASK_DIRS[@]} -eq 0 ]; then
        echo "Error: No directories containing 'play' found in ${RAW_DATA_BASE_DIR}"
        exit 1
    fi
    # Use the filtered list from now on
    TASK_DIRS=("${FILTERED_TASK_DIRS[@]}")
    echo "Found ${#TASK_DIRS[@]} directories containing 'play'."
else
    echo "Filtering for directories not containing 'play'..."
    FILTERED_TASK_DIRS=()
    for dir in "${TASK_DIRS[@]}"; do
        task_name=$(basename "${dir%/}") # Extract directory name
        if [[ "$task_name" != *play* ]]; then
            FILTERED_TASK_DIRS+=("$dir")
        fi
    done
    echo "Processing all found directories (ONLY_PLAY_DATA=false)."
fi

# Construct the list of directories for the --raw-dirs argument
# We need to remove the trailing slash added by the glob pattern
RAW_DIRS_ARG=""
TASK_NAMES=()
for dir in "${TASK_DIRS[@]}"; do
    # Remove trailing slash for the argument and extract task name
    clean_dir=${dir%/}
    task_name=$(basename "$clean_dir")
    RAW_DIRS_ARG+=" \"${clean_dir}\"" # Add directory path in quotes
    TASK_NAMES+=("$task_name") # Store task names for logging
done

# Define the single repository ID for the combined dataset
REPO_ID="${HF_ORG}/${COMBINED_REPO_NAME}"

echo "=========================================================="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "----------------------------------------------------------"
echo "Processing tasks:"
printf " - %s\n" "${TASK_NAMES[@]}"
echo "Raw Data Base Dir:  ${RAW_DATA_BASE_DIR}"
echo "Only Play Data: ${ONLY_PLAY_DATA}"
echo "Task Dirs Found:   $(echo ${RAW_DIRS_ARG})" # Show the argument string
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
    --raw-dirs ${RAW_DIRS_ARG} \
    --repo-id \"${REPO_ID}\" \
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
echo "Finished Combined USC Data Conversion Job"
echo "Time: $(date)"
echo "=========================================================="
