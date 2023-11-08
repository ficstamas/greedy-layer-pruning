#!/bin/bash

source env/bin/activate

#
# Prune greedy
#
for task in "rte" "mrpc" "cola"; do
for model in "bert-base-uncased" "roberta-base" "distilbert-base-uncased"; do
for seed in `seq 0 1 4`; do
python3 prune.py --model_name_or_path=$model \
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

for task in "ner" "pos"; do
for model in "bert-base-uncased" "roberta-base" "distilbert-base-uncased"; do
for seed in `seq 0 1 4`; do
python3 prune.py --model_name_or_path=$model \
    --task_name=$task \
    --dataset_name="conll2003" \
    --task_type="token" \
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