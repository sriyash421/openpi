#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G 
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
#SBATCH --gres=shard:24

config=pi0_lora_bridge_1_cam_arro
temporal_weight_decay=0.0
policy_port=8001
checkpoint=checkpoints/pi0_lora_bridge_1_cam_arro/27600/
arro_server_ip=tcp://edclduajln.a.pinggy.link:21379

cd ~/openpi
                
echo "Running server with ARRO"
uv run scripts/serve_policy_arro.py --port $policy_port --arro_img_key="observation.images.image_0" \
    --arro_server_ip=$arro_server_ip \
    --action-chunk-history-size=10 \
    --ensemble-window-size=5 \
    --temporal-weight-decay=$temporal_weight_decay \
    policy:checkpoint --policy.config=$config --policy.dir $checkpoint 