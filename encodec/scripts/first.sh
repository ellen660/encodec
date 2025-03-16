#!/bin/bash

CONFIG_FILE="no_discrim.yaml"

# Hyperparameters for ablation
learning_rates=(0.01 0.001 0.0001)  # Example: varying learning rates
batch_sizes=(32 64)  # Example: varying batch sizes
betas=(0.8 0.9)  # Example: varying beta values

# Loop through different learning rates, batch sizes, and layers
for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for layer in "${layers[@]}"; do
            echo "Running ablation experiment with lr=$lr, batch_size=$bs, layers=$layer"

            # Run the training script with the ablation configuration
            python train.py --learning_rate $lr --batch_size $bs --num_layers $layer

            # You can save the model checkpoints, logs, etc., for each experiment
            # Example: Save logs or results
            # python train.py --learning_rate $lr --batch_size $bs --num_layers $layer > "logs/lr_${lr}_bs_${bs}_layers_${layer}.log"

            # Optionally, log to W&B
            # wandb.init(project="ablation_study", config={"learning_rate": lr, "batch_size": bs, "layers": layer})
        done
    done
done

echo "Ablation experiments completed."
