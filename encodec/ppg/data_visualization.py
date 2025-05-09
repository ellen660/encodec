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
    max_length = 100 * 60 * 60 * 1 #4 hours

    datasets = {}

    # datasets["mgh"] = BwhPpgDataset(dataset = "mgh_new", mode = mode, cv = cv, max_length = max_length)
    datasets["bwh"] = BwhPpgDataset(dataset = "bwh_new", mode = mode, cv = cv, max_length = max_length)
                    
    return datasets

def visualize_patient(ds_names, test_datasets, save_dir=f"/data/scratch/ellen660/encodec/encodec/ppg/visualization"):
    raw_files = f"/data/netmit/sleep_lab/sandbox/ppg/bwh"

    for i, ds_name in enumerate(ds_names):
        test_ds = test_datasets[ds_name]  # Get the dataset for the current channel

        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=10)

        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        axs = axs.flatten()

        # Iterate through the DataLoader
        for j, batch in enumerate(tqdm(test_loader, desc="Individual plot")):
            if j >= 9:
                break
            x = batch["x"][0].numpy().squeeze()  # Assuming x is in the batch and is a numpy-compatible tensor
            filename = batch["filename"][0]
            filepath = os.path.join(raw_files, filename)
            raw = np.load(filepath)['data'].squeeze()[:3600000]

            # --- Plot histogram ---
            bin_edges = np.linspace(-10, 10, 200)
            histogram = np.histogram(x, bins=bin_edges)[0]
            histogram = histogram / histogram.sum()

            axs[j].bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
            axs[j].set_ylabel("Frequency")
            axs[j].set_title(f"Histogram {j+1}")

            # --- Plot raw vs. processed in separate figure ---
            total_samples = len(raw)
            time_axis = np.arange(total_samples) / (100 * 3600)  # Convert to hours

            fig_line, ax_line = plt.subplots(figsize=(12, 4))
            ax_line.plot(time_axis, raw, label='Raw PPG', alpha=1.0)
            ax_line.plot(time_axis[:len(x)], x, label='Processed PPG', alpha=0.7)
            ax_line.set_title(f'Patient PPG')
            ax_line.set_xlabel('Time (hours)')
            ax_line.set_ylabel('Amplitude')
            ax_line.set_ylim(-10, 10)
            ax_line.legend()

            max_time = time_axis[-1]
            tick_locs = np.arange(0, max_time + 0.5, 0.5)
            ax_line.set_xticks(tick_locs)

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"{filename}_plot.png")
            fig_line.savefig(save_path)
            plt.close(fig_line)  # Close the figure to avoid clutter
        
        for ax in axs:
            ax.set_ylim(0, 0.12)
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"{save_dir}/{ds_name}_patient_histogram.png")
        fig.savefig(save_path)
        plt.close(fig)

def get_data_distribution(ds_names, train_datasets, save_dir=f"/data/scratch/ellen660/encodec/encodec/ppg/visualization"):
    """
    Plot distribution for dataset
    """
    fig, axs = plt.subplots(figsize=(10, 10))

    for i, ds_name in enumerate(ds_names):
        train_ds = train_datasets[ds_name]  # Get the dataset for the current channel

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=10)

        # Define the histogram parameters
        bin_edges = np.linspace(-10, 10, 100)  # 50 bins from -4 to 4
        histogram = np.zeros(len(bin_edges) - 1)  # Initialize empty histogram

        # Iterate through the DataLoader
        for j, batch in enumerate(tqdm(train_loader, desc="Getting distribution")):
            # if j >= 10:
            #     break
            x = batch["x"].numpy()  # Assuming x is in the batch and is a numpy-compatible tensor
            # Flatten and add to histogram
            histogram += np.histogram(x, bins=bin_edges)[0]

        # Normalize the histogram to get probabilities (optional)
        histogram = histogram / histogram.sum()

        axs.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
        # axs[i].set_xlabel("Feature Value")
        axs.set_ylabel("Frequency")
        axs.set_title(f"Histogram of {ds_name}")
    
    #save 
    axs.set_ylim(0, 0.12)
    save_path = os.path.join(save_dir, f"distribution.png")  # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
    plt.close()  # Close the figure to free memory
    print(f"Finished processing")

if __name__ == "__main__":
    ds_names = ["bwh"]

    test_datasets = init_dataset(mode="test")
    visualize_patient(ds_names, test_datasets)

    train_datasets = init_dataset(mode="train")

    #Data Distribution w/ Flipping
    get_data_distribution(ds_names, train_datasets)
