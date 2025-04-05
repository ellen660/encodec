#!/bin/bash

# Path to the YAML file
yaml_file="no_discrim"
path="/data/scratch/ellen660/encodec/encodec/params/$yaml_file.yaml"
export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7 
run_name="6_seconds_8_codebooks"

declare -A hyperparameters
hyperparameters=(
  [".optimization.lr"]="1e-4"
  [".optimization.batch_size"]="12"
  [".model.bins"]="512"
  [".common.max_epoch"]="200"
)

#Iterate over a bunch 
lr_list=("1e-4")  
batch_size_list=("12")  

# Iterate over all combinations of hyperparameters
for lr in "${lr_list[@]}"; do
  for batch_size in "${batch_size_list[@]}"; do
    hyperparameters[".optimization.lr"]=$lr
    hyperparameters[".optimization.batch_size"]=$batch_size

    yq -yi ".model.ratios = [5, 3, 2, 2, 1]" "$path"
    yq -yi ".model.target_bandwidths = [0.08]" "$path"
    yq -yi '.exp_details.description = "6_seconds_8_codebooks"' "$path"

    # Replace parameters in the YAML file using yq
    # Shouldn't need to edit this part
    comment=""
    for param in "${!hyperparameters[@]}"; do
        new_value="${hyperparameters[$param]}"
        yq -yi "$param = $new_value" "$path"
        #split the param by '.' and get the last element
        param_name=$(echo $param | rev | cut -d'.' -f1 | rev)
        comment="$comment $param_name=$new_value"
        echo $comment
    done

    # Log directory
    curr_time=$(date +%Y%m%d)
    log_dir="/data/scratch/ellen660/encodec/encodec/ablations/$yaml_file/$run_name/$curr_time/$comment"

    # Run the Python training script with the updated parameters
    python encodec/train.py --exp_name "$yaml_file" --log_dir "$log_dir" 
    done
done