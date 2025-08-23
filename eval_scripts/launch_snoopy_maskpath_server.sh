#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G 
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
##SBATCH --gres=gpu:1
#SBATCH --gres=shard:30

config=pi0_lora_bridge_1_cam_path_masked
#config=pi0_bridge
#config=pi0_bridge_path_mask
checkpoint=checkpoints/pi0_lora_bridge_1_cam_path_masked/pi0_lora_bridge_1_cam_path_masked/29999/
#checkpoint=checkpoints/pi0_bridge/pi0_bridge/pi0_bridge_fft/35000/
#checkpoint=checkpoints/pi0_bridge_path_mask/pi0_bridge_path_mask/pi0_fft_bridge_path_masked/35000/
vlm_freq=5

if [[ "$config" == *"path"* ]]; then
    cd ~/VILA
    echo "Running VILA server"
    conda run -n vila --no-capture-output /bin/bash -c "python -W ignore vila_3b_server.py --model-paths ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/" &
    sleep 25
fi

cd ~/openpi
if [[ "$config" == *"path"* ]]; then                
    #uv run scripts/serve_policy_vlm.py --port 8001 --vlm_img_key="observation.images.image_0" --vlm-query-frequency=$vlm_freq policy:checkpoint --policy.config=$config --policy.dir $checkpoint &
    uv run scripts/serve_policy_autoeval.py --port 8001 \
    --vlm_img_key="observation.images.image_0" \
    --vlm-query-frequency=$vlm_freq \
    --draw-path \
    --draw-mask \
    --vlm-mask-ratio=0.08 \
    policy:checkpoint --policy.config=$config --policy.dir $checkpoint
else
    #uv run scripts/serve_policy.py --port 8001 policy:checkpoint --policy.config=$config --policy.dir $checkpoint 
    uv run scripts/serve_policy_autoeval.py --port 8001 policy:checkpoint --policy.config=$config --policy.dir $checkpoint
fi

sleep 15

ssh -p 443 -R0:localhost:8001 -L4300:localhost:4300 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 1GC4MlSDYxt@pro.pinggy.io
