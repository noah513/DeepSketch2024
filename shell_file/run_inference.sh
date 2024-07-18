#!/bin/bash

# Check argument values
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <dataset> <output file> [<model>]"
    exit 1
fi

# Assign argument values
dataset=$1
output=$2
model=${3:-"baseline"}

# Execute different scripts based on the model
if [ "$model" == "baseline" ]; then
    echo "Using baseline model"
    ./DeepSketch sensor ../04_training/model/model_hash_${dataset}_128_4096_2048_1_0.001.torchsave.pt 128 ${output}.txt
else
    echo "Using ${model} model"
    ./DeepSketch sensor ../04_training/model/model_hash_${dataset}_${model}_128_4096_2048_1_0.001.torchsave.pt 128 ${output}.txt
fi