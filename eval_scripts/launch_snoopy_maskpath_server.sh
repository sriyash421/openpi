#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G 
#SBATCH --output=slurm_outputs/%x_%j.out
#SBATCH --error=slurm_outputs/%x_%j.err
##SBATCH --gres=gpu:1
#SBATCH --gres=shard:30

#config=pi0_lora_bridge_1_cam_path_masked
#config=pi0_bridge
config=pi0_bridge_path_mask
temporal_weight_decay=0.5
policy_port=8001
#checkpoint=checkpoints/pi0_lora_bridge_1_cam_path_masked/pi0_lora_bridge_1_cam_path_masked/29999/
#checkpoint=checkpoints/pi0_bridge/pi0_bridge/pi0_bridge_fft/35000/
checkpoint=checkpoints/pi0_bridge_path_mask/pi0_bridge_path_mask/pi0_fft_bridge_path_masked/35000/
serve_policy_vlm_freq=3
http_vlm_freq=25
use_http_server=0

if [[ "$config" == *"path"* ]]; then
    cd ~/VILA
    echo "Running VILA server"
    conda run -n vila --no-capture-output /bin/bash -c "python -W ignore vila_3b_server.py --model-paths ~/.cache/huggingface/hub/models--memmelma--vila_3b_path_mask_fast/snapshots/12df7a04221a50e88733cd2f1132eb01257aba0d/checkpoint-11700/" &
    sleep 10
fi

cd ~/openpi
if [[ "$config" == *"path"* ]]; then                
    echo "Running autoeval server with path mask"
    if [[ "$use_http_server" == 1 ]]; then
        uv run scripts/serve_policy_autoeval.py --port $policy_port \
        --vlm_img_key="observation.images.image_0" \
        --vlm-query-frequency=$http_vlm_freq \
        --vlm-server-ip=http://0.0.0.0:8000 \
        --draw-path \
        --draw-mask \
        --vlm-mask-ratio=0.08 \
        --temporal-weight-decay=$temporal_weight_decay \
        --action-chunk-history-size=10 \
        --ensemble-window-size=5 \
        policy:checkpoint --policy.config=$config --policy.dir $checkpoint
    else
        uv run scripts/serve_policy_vlm.py --port $policy_port --vlm_img_key="observation.images.image_0" --vlm-query-frequency=$serve_policy_vlm_freq policy:checkpoint --policy.config=$config --policy.dir $checkpoint &
    fi
else
    if [[ "$use_http_server" == 1 ]]; then
    uv run scripts/serve_policy_autoeval.py --port $policy_port \
        --action-chunk-history-size=10 \
        --ensemble-window-size=5 \
        --temporal-weight-decay=$temporal_weight_decay \
        policy:checkpoint --policy.config=$config --policy.dir $checkpoint
    else
        uv run scripts/serve_policy.py --port 8001 policy:checkpoint --policy.config=$config --policy.dir $checkpoint &
    fi
fi

sleep 15

if [[ "$use_http_server" == 0 ]]; then
    ssh -p 443 -R0:localhost:${policy_port} -L4300:localhost:4300 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 1GC4MlSDYxt@pro.pinggy.io
fi
