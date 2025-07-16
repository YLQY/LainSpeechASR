#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Automatically detect number of gpus
if command -v nvidia-smi &> /dev/null; then
  num_gpus=$(nvidia-smi -L | wc -l)
  gpu_list=$(seq -s, 0 $((num_gpus-1)))
else
  num_gpus=-1
  gpu_list="-1"
fi
# You can also manually specify CUDA_VISIBLE_DEVICES
# if you don't want to utilize all available GPU resources.
export CUDA_VISIBLE_DEVICES="${gpu_list}"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"


stage=5 # start from 0 if you need to start from data preparation
stop_stage=5

# 微调路径
dir=exp/xhs_train_v1
train_config=exp/wenet_llm_ori/train.yaml
# 模型恢复的时候先恢复小红书的参数
# checkpoint=/root/autodl-tmp/LainSpeechASR/examples/xhs_aed_train/exp/wenet_xhs_ori/wenet_firered.pt
checkpoint=
tensorboard_dir=tensorboard

# 通用参数，多机多卡
HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=2025
train_engine=torch_ddp
deepspeed_config=conf/ds_stage2.json
deepspeed_save_states="model_only"
prefetch=10
num_workers=8


. tools/parse_options.sh || exit 1;


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  mkdir -p $dir
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

  dist_backend="nccl"
  if [ ${train_engine} == "deepspeed" ]; then
    echo "$0: using deepspeed"
  else
    echo "$0: using torch ddp"
  fi

  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wenet/bin/train_llm.py \
      --train_engine ${train_engine} \
      --config $train_config \
      --data_type 'raw' \
      --train_data data/data.list \
      --cv_data data/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --tensorboard_dir ${tensorboard_dir} \
      --ddp.dist_backend $dist_backend \
      --num_workers ${num_workers} \
      --prefetch ${prefetch} \
      --pin_memory \
      --deepspeed_config ${deepspeed_config} \
      --deepspeed.save_states ${deepspeed_save_states}
fi


# wenet微调的init模型
decode_checkpoint=/LainSpeechASR/examples/xhs_aed_train/exp/xhs_train_v1/init.pt

# xhs转wenet原始的pt
#decode_checkpoint=/root/autodl-tmp/LainSpeechASR/examples/xhs_aed_train/exp/wenet_xhs_ori/wenet_firered.pt
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

  decoding_chunk_size=
  ctc_weight=0.0
  reverse_weight=0.5
  python3 wenet/bin/recognize_llm.py --gpu 0 \
    --modes "attention" \
    --config $dir/train.yaml \
    --data_type 'raw' \
    --test_data data/data.list \
    --checkpoint $decode_checkpoint \
    --beam_size 10 \
    --batch_size 1 \
    --blank_penalty 0.0 \
    --ctc_weight $ctc_weight \
    --reverse_weight $reverse_weight \
    --result_dir $dir \
    ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
fi















