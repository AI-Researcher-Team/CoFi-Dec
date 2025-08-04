#!/bin/bash

seed=42
dataset_name="coco" # coco | aokvqa | gqa
type="random" # random | popular | adversarial

# llava
model="llava"
model_path="/root/autodl-tmp/DeGF/data/ce/model/llava-v1.5-7b"

# instructblip
# model="instructblip"
# model_path='/root/autodl-tmp/DeGF/data/ce/model/vicuna-7b-v1.1'

pope_path="/root/autodl-tmp/DeGF/data/ce/data/POPE/${dataset_name}/${dataset_name}_pope_${type}.json"
data_path="/root/autodl-tmp/DeGF/data/ce/data/coco/val2014"

# data_path="/data/ce/data/gqa/images"


neg_image_path="/root/autodl-tmp/DeGF/outputs/${experiment_subdir}/neg_images/neg_image_${batch_id}.png"


log_path="./logs"

use_ritual=False
use_vcd=False
use_m3id=False
use_diffusion=True

degf_alpha_pos=3.0
degf_alpha_neg=1.0
degf_beta=0.1

experiment_index=3

# 添加数据比例参数
data_ratio=0.1  # 使用10%的数据

#####################################
# Run single experiment
#####################################



mkdir -p /root/autodl-tmp/DeGF/logs

export CUDA_VISIBLE_DEVICES=0
torchrun --nnodes=1 --nproc_per_node=1 --master_port 1336 eval_bench/pope_eval_${model}.py \
--seed ${seed} \
--model_path ${model_path} \
--model_base ${model} \
--pope_path ${pope_path} \
--data_path ${data_path} \
--log_path ${log_path} \
--use_ritual ${use_ritual} \
--use_vcd ${use_vcd} \
--use_m3id ${use_m3id} \
--use_diffusion ${use_diffusion} \
--neg_image ${neg_image_path} \
--degf_alpha_pos ${degf_alpha_pos} \
--degf_alpha_neg ${degf_alpha_neg} \
--degf_beta ${degf_beta} \
--experiment_index ${experiment_index} \
--output_image_dir "/root/autodl-tmp/DeGF/outputs" \
--data_ratio ${data_ratio} 2>&1 | tee /root/autodl-tmp/DeGF/logs/output_$(date +"%Y%m%d_%H%M%S").log

# export CUDA_VISIBLE_DEVICES=0
# torchrun --nnodes=1 --nproc_per_node=1 --master_port 1336 eval_bench/pope_eval_${model}.py \
# --seed ${seed} \
# --model_path ${model_path} \
# --model_base ${model} \
# --pope_path ${pope_path} \
# --data_path ${data_path} \
# --log_path ${log_path} \
# --use_ritual ${use_ritual} \
# --use_vcd ${use_vcd} \
# --use_m3id ${use_m3id} \
# --use_diffusion ${use_diffusion} \
# --degf_alpha_pos ${degf_alpha_pos} \
# --degf_alpha_neg ${degf_alpha_neg} \
# --degf_beta ${degf_beta} \
# --experiment_index ${experiment_index} \
# --output_image_dir "/root/autodl-tmp/DeGF/outputs" \
# --data_ratio ${data_ratio}