# To run a new model with this shell file, each file name must follow this format
# Python code format:
# - train_baseline_{model_name}.py
# - train_hashlayer_gh_{model_name}.py
# - model_converter_gh_{model_name}.py
# Output file format:
# - model/model_{dataset}_{model_name}_4096_2048_1e-05.torchsave
# - model/model_hash_{dataset}_{model_name}_128_4096_2048_1_0.001.torchsave

#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo "Usage: $0 <input_file_path> <num_thread> [<model_name>]"
  exit 1
fi

# Assign the input arguments to variables
input_file_path="$1"
num_thread="$2"
model_name="${3:-"baseline"}"

# Print the values to verify
echo "Input file path: $input_file_path"
echo "Number of threads: $num_thread"

# Extract the input file name without the path
input_file_name=$(basename "$input_file_path")
cluster_info_path="03_clustering/result/${input_file_name}10_fine"

# Check if the clustering result already exists
if [ -f "$cluster_info_path" ]; then
  read -p "Clustering result already exists. Do you want to use the existing result? (y/n): " use_existing
  if [ "$use_existing" == "y" ]; then
    echo "Using existing clustering result."
  else
    # Execute run_clustering script if user does not want to use the existing result
    echo "Executing run_clustering script in 03_clustering folder..."
    cd 03_clustering
    ./run_clustering.sh "../$input_file_path" $num_thread
    cd ..
    wait

    # Verify that the previous script completed successfully
    if [ $? -ne 0 ]; then
      echo "run_clustering.sh in 03_clustering folder failed."
      exit 1
    fi
  fi
else
  # Execute run_clustering script if clustering result does not exist
  echo "Executing run_clustering script in 03_clustering folder..."
  cd 03_clustering
  ./run_clustering.sh "../$input_file_path" $num_thread
  cd ..
  wait

  # Verify that the previous script completed successfully
  if [ $? -ne 0 ]; then
    echo "run_clustering.sh in 03_clustering folder failed."
    exit 1
  fi
fi

# Execute the run_create script in the 04_training/create_dataset folder
echo "Executing run_create script in 04_training/create_dataset folder..."
cd 04_training/create_dataset
./run_create.sh "../../$input_file_path" "$input_file_name" "../../$cluster_info_path"
cd ../..
wait

# Verify that the previous script completed successfully
if [ $? -ne 0 ]; then
  echo "run_create.sh in 04_training/create_dataset folder failed."
  exit 1
fi

# Get the cluster number from the create_dataset result folder name
cluster_folder=$(ls -d 04_training/cluster_${input_file_name}_* | head -n 1)
cluster_num=$(basename $cluster_folder | sed 's/.*_\([0-9]\+\)$/\1/')

# Execute the run_training script in the 04_training folder
echo "Executing run_training script in 04_training folder..."
cd 04_training
if [ "$model" == "baseline" ]; then
  ./run_training.sh "$input_file_name" "$cluster_num"
else
  ./run_training.sh "$input_file_name" "$cluster_num" "$model_name"
fi
cd ..
wait

# Verify that the previous script completed successfully
if [ $? -ne 0 ]; then
  echo "run_training.sh in 04_training folder failed."
  exit 1
fi

# Execute the run_inference script in the 05_infer folder
echo "Executing run_inference script in 05_infer folder..."
cd 05_infer
if [ "$model" == "baseline" ]; then
  ./run_inference.sh "$input_file_name" "${input_file_name}_fg_${model_name}"
else
  ./run_inference.sh "$input_file_name" "${input_file_name}_fg_${model_name}" "$model_name"
fi
cd ..
wait

# Verify that the previous script completed successfully
if [ $? -ne 0 ]; then
  echo "run_inference.sh in 05_infer folder failed."
  exit 1
fi

echo "All scripts executed successfully."
