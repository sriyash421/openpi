#!/bin/bash
#SBATCH --account=jessetho_1023
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=185G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err

# --- Start Relaunch Logic ---
MAX_RELAUNCHES=6

# Get the relaunch count from the environment variable set by previous submissions, default to 0
RELAUNCH_COUNT=${JOB_RELAUNCH_COUNT:-0}

if [ $RELAUNCH_COUNT -ge $MAX_RELAUNCHES ]; then
  echo "Maximum relaunch limit ($MAX_RELAUNCHES) reached. Exiting."
  exit 1
fi
# --- End Relaunch Logic ---

# --- Environment Setup ---
echo "Starting job... Relaunch attempt: $((RELAUNCH_COUNT + 1))/$MAX_RELAUNCHES"
source ~/.bashrc
module load cuda
module load glew
module load patchelf
module load git-lfs
export PATH="/apps/conda/.local/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

#EXP_NAME=pi0_libero_90_path_masked_bs164_rdp
#EXP_NAME=pi0_libero_low_mem_finetune_path_masked_mask0.01_0.12
EXP_NAME=pi0_libero_low_mem_finetune_path_masked_mask_maxep10
#EXP_NAME=pi0_libero_low_mem_finetune_vlm_path_masked_bs148
# --- End Environment Setup ---

# --- Training Command Setup ---
# Define the base training command as a variable
BASE_TRAIN_CMD="uv run scripts/train.py pi0_libero_low_mem_finetune_path_masked --exp-name=$EXP_NAME"
#BASE_TRAIN_CMD="uv run scripts/train.py pi0_libero_low_mem_finetune_vlm_path_masked --exp-name=$EXP_NAME"

# Conditionally add --resume flag based on relaunch count
if [ "$RELAUNCH_COUNT" -eq 0 ]; then
  echo "Initial run. Starting training."
  # First run: Use the base command.
  # Note: You might want to add --overwrite here if the first run should always start fresh:
  #TRAIN_CMD="$BASE_TRAIN_CMD --overwrite"
  #TRAIN_CMD="$BASE_TRAIN_CMD"
  TRAIN_CMD="$BASE_TRAIN_CMD --resume"
else
  echo "Relaunch run ($((RELAUNCH_COUNT))). Resuming training."
  # Relaunch runs: Append the --resume flag
  TRAIN_CMD="$BASE_TRAIN_CMD --resume"
fi
# --- End Training Command Setup ---

# --- Execute Training ---
echo "Executing command: $TRAIN_CMD"
# Execute the constructed command
$TRAIN_CMD

EXIT_STATUS=$?
# --- End Execute Training ---

# --- Relaunch on Failure ---
if [ $EXIT_STATUS -ne 0 ]; then
  echo "Job failed or was interrupted with exit status $EXIT_STATUS."
  # Check if we are under the relaunch limit before submitting a new job
  if [ $RELAUNCH_COUNT -lt $MAX_RELAUNCHES ]; then
      NEXT_RELAUNCH_COUNT=$((RELAUNCH_COUNT + 1))
      echo "Attempting relaunch $NEXT_RELAUNCH_COUNT of $MAX_RELAUNCHES..."
      # Resubmit the job, exporting the incremented relaunch count
      sbatch --export=ALL,JOB_RELAUNCH_COUNT=$NEXT_RELAUNCH_COUNT "$0"
  else
      echo "Maximum relaunch limit reached. Not relaunching again."
  fi
else
  echo "Job completed successfully."
fi

exit $EXIT_STATUS
# --- End Relaunch on Failure ---
