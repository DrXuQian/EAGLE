#!/bin/bash
# Script to train EAGLE-3 for Qwen2.5-3B-Instruct

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate eagle

cd /home/qianxu/EAGLE/eagle/traineagle3

# Training command
deepspeed main.py \
  --deepspeed_config ds_config.json \
  --basepath Qwen2.5-3B-Instruct \
  --trainpath sample_train.jsonl \
  --testpath sample_test.jsonl \
  --savedir ./output_qwen25_3b \
  --model_type qwen

# Note: The above uses sample data. For full training, replace with:
#  --trainpath /path/to/your/full_train.jsonl \
#  --testpath /path/to/your/full_test.jsonl \
