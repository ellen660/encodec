#!/bin/bash

# Path to the YAML file
yaml_file="no_discrim"
path="/data/scratch/ellen660/encodec/encodec/params/$yaml_file.yaml"
export CUDA_VISIBLE_DEVICES=0,1,4,5,6,7 

declare -A hyperparameters
hyperparameters=(
  [".optimization.lr"]="1e-4"
  [".optimization.batch_size"]="32"
  [".model.bins"]="512"
  [".common.log_every"]="1"
  [".common.max_epoch"]="10"
  [".common.debug"]=true
)

#Iterate over a bunch 
lr_list=("1e-4" "1e-2")  
batch_size_list=("16" "32")  

# Iterate over all combinations of hyperparameters
for lr in "${lr_list[@]}"; do
  for batch_size in "${batch_size_list[@]}"; do
    hyperparameters[".optimization.lr"]=$lr
    hyperparameters[".optimization.batch_size"]=$batch_size

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
    curr_minute=$(date +%H%M%S)
    log_dir="/data/scratch/ellen660/encodec/encodec/ablations/$yaml_file/$curr_time/$curr_minute/$comment"

    # Run the Python training script with the updated parameters
    python encodec/train.py --exp_name "$yaml_file" --log_dir "$log_dir" 
    done
done