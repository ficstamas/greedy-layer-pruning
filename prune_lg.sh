#!/bin/bash
#
# Prune greedy
#
for seed in `seq 0 1 4`; do
for model in "roberta-base" "bert-large-uncased"; do
for task in "stsb" "sst2"; do
python3 prune.py --model_name_or_path=$model \
    --config_name=$model \
    --tokenizer_name=$model \
    --task_name=$task \
    --dataset_name="glue" \
    --task_type="sequence" \
    --seed=$seed \
    --max_seq_length=256 \
    --per_device_train_batch_size=16 \
    --learning_rate=2e-5 \
    --output_dir=experiments/tmp/ \
    --logging_dir=experiments/tmp/ \
    --prune_n_layers=6 \
    --prune_method="greedy" \
    --overwrite_output_dir \
    --prune_all_but_one
done
done
done