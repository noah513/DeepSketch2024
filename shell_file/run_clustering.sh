#!/bin/bash

# Check argument values
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <num_thread>"
    exit 1
fi

# Assign argument values
input_file=$1
num_thread=$2

# Execute the fine program with the provided arguments
./fine_khj $input_file $num_thread