#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 [input_file] [tag] [cluster_info]"
    exit 1
fi

input_file=$1
tag=$2
cluster_info=$3

echo $cluster_info
./create_dataset $input_file $tag < $cluster_info