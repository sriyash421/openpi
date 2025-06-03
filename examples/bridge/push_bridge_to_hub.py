from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset(
    "jesbu1/bridge_v2_lerobot",
   # repo_id=repo_id,
    #fps=5,
    #robot_type=robot_type,
    #features=features,
    #use_videos=True,
    #image_writer_processes=dataset_config.image_writer_processes,
    #image_writer_threads=dataset_config.image_writer_threads,
    #video_backend=dataset_config.video_backend,
)

# Consolidate the dataset, skip computing stats since we will do that later
#dataset.consolidate(run_compute_stats=True)
dataset.push_to_hub(
    tags=["widowx", "bridge-v2"],
    private=False,
    push_videos=True,
    upload_large_folder=True,
    license="apache-2.0",
)
