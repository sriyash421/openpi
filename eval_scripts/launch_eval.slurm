#!/bin/bash
#SBATCH --account=biyik_1165
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --gres=gpu:2

# Default parameter values
EVAL_SET=${1:-10}  # Default to libero_10 if not specified (options: 10, spatial, object, goal)
USE_PATH=${2:-0}   # Default to not using path
USE_MASK=${3:-0}   # Default to not using mask

source ~/.bashrc

cd /home1/jessez/nvidia/VILA
echo "Running VILA server"
conda run -n vila --no-capture-output /bin/bash -c "CUDA_VISIBLE_DEVICES=0 python -W ignore vila_3b_server.py --model-paths ~/.cache/huggingface/hub/models--memmelma--vila_3b_oxe_sim_path_mask/snapshots/3d08e92de1cd7517b1acbf72648ce3bbf48c19ae/checkpoint-5700" &

# Wait for the model to load
sleep 20

SERVE_CMD="CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint --port 8001"
if [ "$USE_PATH" = "1" ] && [ "$USE_MASK" = "0" ]; then
    SERVE_CMD="$SERVE_CMD --policy.config=pi0_libero_low_mem_finetune_path --policy.dir=checkpoints/pi0_libero_low_mem_finetune_path/pi0_libero_90_path_bs164_rdp/35000/"
elif [ "$USE_PATH" = "0" ] && [ "$USE_MASK" = "1" ]; then
    SERVE_CMD="$SERVE_CMD --policy.config=pi0_libero_low_mem_finetune_masked --policy.dir=checkpoints/pi0_libero_low_mem_finetune_masked/pi0_libero_90_masked_bs164_rdp/30000/"
elif [ "$USE_PATH" = "1" ] && [ "$USE_MASK" = "1" ]; then
    SERVE_CMD="$SERVE_CMD --policy.config=pi0_libero_low_mem_finetune_path_masked --policy.dir=checkpoints/pi0_libero_low_mem_finetune_path_masked/pi0_libero_90_path_masked_bs164_rdp/30000/"
fi

echo "Running serving command: $SERVE_CMD"
bash -c "$SERVE_CMD" &

sleep 20

cd /home1/jessez/nvidia/openpi

# Determine task suite based on EVAL_SET
if [ "$EVAL_SET" = "10" ]; then
  TASK_SUITE="libero_10"
elif [ "$EVAL_SET" = "spatial" ]; then
  TASK_SUITE="libero_spatial"
elif [ "$EVAL_SET" = "object" ]; then
  TASK_SUITE="libero_object"
elif [ "$EVAL_SET" = "goal" ]; then
  TASK_SUITE="libero_goal"
else
  echo "Invalid EVAL_SET: $EVAL_SET. Using default libero_10."
  TASK_SUITE="libero_10"
fi

# Build command with optional flags
EVAL_CMD="python examples/libero/main.py --args.task_suite_name=$TASK_SUITE --args.vlm_server_ip=http://0.0.0.0:8001 --args.port 8000"

# Add draw_path if USE_PATH is 1
if [ "$USE_PATH" = "1" ]; then
  EVAL_CMD="$EVAL_CMD --args.draw_path"
fi

# Add draw_mask if USE_MASK is 1
if [ "$USE_MASK" = "1" ]; then
  EVAL_CMD="$EVAL_CMD --args.draw_mask"
fi

# Execute the command
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
echo "Running: $EVAL_CMD"
$EVAL_CMD