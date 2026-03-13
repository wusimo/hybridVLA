#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/home/liangyuancong/anaconda3/envs/starvla/bin/python
your_ckpt="results/Checkpoints/1229_libero4in1_qwen3oft/checkpoints/steps_70000_pytorch_model.pt"
gpu_id=7
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
