import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import sys

# Get the absolute path of the 'encoded' folder
encoded_path = os.path.abspath(".")  # Adjust path if necessary

# Add it to sys.path
sys.path.append(encoded_path)

from encodec.ppg.bwh_ppg import BwhPpgDataset

def init_dataset(mode="test"):
    cv = 0
    max_length = 100 * 60 * 60 * 1

    datasets = {}

    # datasets["mgh"] = BwhPpgDataset(dataset = "mgh_new", mode = mode, cv = cv, max_length = max_length)
    datasets["bwh"] = BwhPpgDataset(dataset = "bwh_new", mode = mode, cv = cv, max_length = max_length)
                    
    return datasets

def get_data_distribution(ds_names, train_datasets, save_dir=f"/data/scratch/ellen660/encodec/encodec/ppg"):
    """
    Plot distribution for dataset
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs = axs.flatten()

    for i, ds_name in enumerate(ds_names):
        train_ds = train_datasets[ds_name]  # Get the dataset for the current channel

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=10)

        # Define the histogram parameters
        bin_edges = np.linspace(-6, 6, 75)  # 50 bins from -4 to 4
        histogram = np.zeros(len(bin_edges) - 1)  # Initialize empty histogram

        # Iterate through the DataLoader
        for j, batch in enumerate(tqdm(train_loader, desc="Getting distribution")):
            if j >= 10:
                break
            x = batch["x"].numpy()  # Assuming x is in the batch and is a numpy-compatible tensor
            # Flatten and add to histogram
            histogram += np.histogram(x, bins=bin_edges)[0]

        # Normalize the histogram to get probabilities (optional)
        histogram = histogram / histogram.sum()

        axs[i].bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
        # axs[i].set_xlabel("Feature Value")
        axs[i].set_ylabel("Frequency")
        axs[i].set_title(f"Histogram of {ds_name}")
    
    #save 
    save_path = os.path.join(save_dir, f"distribution.png")  # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
    plt.close()  # Close the figure to free memory
    print(f"Finished processing")

if __name__ == "__main__":
    ds_names = ["bwh"]
    train_datasets = init_dataset(mode="train")

    #Data Distribution w/ Flipping
    get_data_distribution(ds_names, train_datasets)
