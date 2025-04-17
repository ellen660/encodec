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

from encodec.data.dataset import BreathingDataset
from encodec.data.bwh import BwhDataset

def init_dataset(mode="test"):
    # cv = config.dataset.cv
    cv = 0
    max_length = 10 * 60 * 60 * 4
    # max_length = config.dataset.max_length

    datasets = {}
    # selected channels
    thorax_channels = {"thorax": 1.} #Hard code for now
    abdominal_channels = {"abdominal": 1.}
    rf_channels = {"rf": 1.}
    
    # if mode == "test":
    #     mgh_dataset = "mgh_new"
    # else:
    #     mgh_dataset = "mgh_train_encodec"

    datasets["mgh"]={"thorax":(BreathingDataset(dataset = "mgh_new", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                     "abdominal":(BreathingDataset(dataset = "mgh_new", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length)),
                     "rf":(BreathingDataset(dataset = "mgh_new", mode = mode, cv = cv, channels = rf_channels, max_length = max_length))
                    }
    datasets["shhs2"] = {
                    "thorax":(BreathingDataset(dataset = "shhs2_new", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    "abdominal":(BreathingDataset(dataset = "shhs2_new", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
                    }
    datasets["shhs1"]={
                    "thorax":(BreathingDataset(dataset = "shhs1_new", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    "abdominal":(BreathingDataset(dataset = "shhs1_new", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
                    }
    datasets["mros1"]={
                    "thorax":(BreathingDataset(dataset = "mros1_new", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    "abdominal":(BreathingDataset(dataset = "mros1_new", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
                    }
    datasets["mros2"]={
                    "thorax":(BreathingDataset(dataset = "mros2_new", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    "abdominal":(BreathingDataset(dataset = "mros2_new", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
                    }
    datasets["wsc"]={
                    "thorax":(BreathingDataset(dataset = "wsc_new", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    "abdominal":(BreathingDataset(dataset = "wsc_new", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
                    }
    datasets["cfs"]={
                    "thorax":(BreathingDataset(dataset = "cfs", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    "abdominal":(BreathingDataset(dataset = "cfs", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
                    }
    datasets["bwh"]={
                    "thorax":(BwhDataset(dataset = "bwh_new", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    }
    datasets["mesa"]={
                    "thorax":(BreathingDataset(dataset = "mesa_new", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    "abdominal":(BreathingDataset(dataset = "mesa_new", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
                    }
    datasets["chat1"]={
                    "thorax":(BreathingDataset(dataset = "chat1", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    "abdominal":(BreathingDataset(dataset = "chat1", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
                    }
    datasets["nchsdb"]={
                    "thorax":(BreathingDataset(dataset = "nchsdb", mode = mode, cv = cv, channels = thorax_channels, max_length = max_length)),
                    "abdominal":(BreathingDataset(dataset = "nchsdb", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
                    }
    
    return datasets


def get_data_distribution(ds_names, channel, train_datasets, save_dir=f"/data/scratch/ellen660/encodec/encodec/visualizations/data"):
    """
    Plot distribution for dataset
    """
    # 3 by 3 plot
    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    axs = axs.flatten()

    for i, ds_name in enumerate(ds_names):
        train_ds = train_datasets[ds_name][channel]  # Get the dataset for the current channel

        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=10)

        # Define the histogram parameters
        bin_edges = np.linspace(-6, 6, 75)  # 50 bins from -4 to 4
        histogram = np.zeros(len(bin_edges) - 1)  # Initialize empty histogram

        # Iterate through the DataLoader
        for batch in tqdm(train_loader, desc="Getting distribution"):
            x = batch["x"].numpy()  # Assuming x is in the batch and is a numpy-compatible tensor
            # Flatten and add to histogram
            if x is None:
                continue
            histogram += np.histogram(x, bins=bin_edges)[0]
            
            #Flip the signal
            # x_flip = x * -1 
            # histogram += np.histogram(x_flip, bins=bin_edges)[0]

        # Normalize the histogram to get probabilities (optional)
        histogram = histogram / histogram.sum()

        axs[i].bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
        # axs[i].set_xlabel("Feature Value")
        axs[i].set_ylabel("Frequency")
        axs[i].set_title(f"Histogram of {ds_name}")
    
    # # Plot the histogram
    # plt.figure(figsize=(8, 6))
    # plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
    # plt.xlabel("Feature Value")
    # plt.ylabel("Frequency")
    # plt.title(f"Histogram of {ds_name}")
    # plt.grid(True)
    #save 
    save_path = os.path.join(save_dir, f"{channel}.png")  # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
    plt.close()  # Close the figure to free memory
    print(f"Finished processing")

if __name__ == "__main__":
    ds_names = ["shhs2", "shhs1", "mros1", "mros2", "wsc", "cfs", "bwh", "mesa"]
    train_datasets = init_dataset(mode="train")
    channel_name = "thorax"

    #Data Distribution w/ Flipping
    get_data_distribution(ds_names, channel_name, train_datasets)
