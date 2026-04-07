from huggingface_hub import snapshot_download

snapshot_download(repo_id="IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
                  repo_type="dataset",local_dir="./playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot")
snapshot_download(repo_id="IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
                  repo_type="dataset",local_dir="./playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot")
snapshot_download(repo_id="IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
                  repo_type="dataset",local_dir="./playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot")
snapshot_download(repo_id="IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
                  repo_type="dataset",local_dir="./playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot")
snapshot_download(repo_id="Qwen/Qwen3-VL-4B-Instruct",
                  local_dir="./playground/Pretrained_models/Qwen3-VL-4B-Instruct")
snapshot_download(repo_id="Alibaba-DAMO-Academy/RynnBrain-CoP-8B",
                  local_dir="./playground/Pretrained_models/RynnBrain-CoP-8B")