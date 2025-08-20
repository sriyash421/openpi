#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G 
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --gres=shard:40

config=pi0_lora_bridge_1_cam_path_masked
checkpoint=checkpoints/pi0_lora_bridge_1_cam_path_masked/pi0_lora_bridge_1_cam_path_masked/19000/
vlm_freq=5

cd ~/VILA
echo "Running VILA server"
conda run -n vila --no-capture-output /bin/bash -c "CUDA_VISIBLE_DEVICES=0 python -W ignore vila_3b_server.py --model-paths ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/" &
# Wait for the model to load
sleep 30

cd ~/openpi
uv run scripts/serve_policy_vlm.py --port 8001 --vlm_img_key="observation.images.image_0" --vlm-query-frequency=$vlm_freq policy:checkpoint --policy.config=$config --policy.dir $checkpoint &

sleep 15

ssh -p 443 -R0:localhost:8001 -L4300:localhost:4300 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 1GC4MlSDYxt@pro.pinggy.io
