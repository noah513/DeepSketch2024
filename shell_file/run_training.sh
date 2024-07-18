#!/bin/bash

# Check argument values
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <dataset> <cluster> [<model>]"
    exit 1
fi

# Assign argument values
dataset=$1
cluster=$2
model=${3:-"baseline"}

# Execute different scripts based on the model
if [ "$model" == "baseline" ]; then
    echo "Using baseline model"
    python3 train_baseline.py cluster_${dataset}_${cluster}/
    python3 train_hashlayer_gh.py model/model_${dataset}_4096_2048_1e-05.torchsave cluster_${dataset}_${cluster}/ 128 1 0.001 0.05
    python3 model_converter_gh.py model/model_hash_${dataset}_128_4096_2048_1_0.001.torchsave cluster_${dataset}_${cluster}/
else
    echo "Using ${model} model"
    python3 train_baseline_${model}.py cluster_${dataset}_${cluster}/
    python3 train_hashlayer_gh_${model}.py model/model_${dataset}_${model}_4096_2048_1e-05.torchsave cluster_${dataset}_${cluster}/ 128 1 0.001 0.05
    python3 model_converter_gh_${model}.py model/model_hash_${dataset}_${model}_128_4096_2048_1_0.001.torchsave cluster_${dataset}_${cluster}/
fi