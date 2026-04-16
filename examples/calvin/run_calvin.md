# 先在一个终端中起服务

cd /home/user01/jiangnan/starVLA
conda activate starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}
CUDA_VISIBLE_DEVICES=7 python deployment/model_server/server_policy.py \
  --ckpt_path /mnt/data/jiangnan/ckpts/StarVLA-Calvin-D/checkpoints/steps_30000_pytorch_model.pt \
  --port 5694 \
  --use_bf16

# 再开一个终端跑测评 
cd /home/user01/jiangnan/starVLA
conda activate calvin310
export PYTHONPATH=$(pwd):${PYTHONPATH}
python examples/calvin/eval_files/eval_calvin.py \
  --args.pretrained-path /mnt/data/jiangnan/ckpts/StarVLA-Calvin-D/checkpoints/steps_30000_pytorch_model.pt \
  --args.unnorm-key franka \
  --args.host 127.0.0.1 \
  --args.port 5694 \
  --args.dataset_path /mnt/data/jiangnan/calvin/task_D_D \
  --args.calvin_config_path /mnt/data/jiangnan/calvin/calvin/calvin_models/conf \
  --args.eval_sequences_path /home/user01/jiangnan/starVLA/examples/calvin/eval_files/eval_sequences.json \
  --args.num_sequences 1000