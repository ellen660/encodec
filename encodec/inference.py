import os
import sys
import torch
import torch.nn as nn

from model import EncodecModel
from data.dataset import BreathingDataset
from data.bwh import BwhDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
# Define train one step function
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np

from my_code.spectrogram_loss import BreathingSpectrogram, ReconstructionLoss, ReconstructionLosses
import torch.multiprocessing as mp

class ConfigNamespace:
    """Converts a dictionary into an object-like namespace for easy attribute access."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigNamespace(value)  # Recursively convert nested dictionaries
            setattr(self, key, value)

# Load the YAML file and convert to ConfigNamespace
def load_config(filepath, log_dir=None):
    #make directory
    with open(filepath, "r") as file:
        config_dict = yaml.safe_load(file)
    return ConfigNamespace(config_dict)

def init_model(config):
    model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, model_norm=config.model.norm, 
        audio_normalize=config.model.audio_normalize,
        segment=eval(config.model.segment), name=config.model.name,
        ratios=config.model.ratios,
        bins=config.model.bins,
        dimension=config.model.dimension,
    )
    # disc_model = MultiScaleSTFTDiscriminator(
    #     in_channels=config.model.channels,
    #     out_channels=config.model.channels,
    #     filters=config.model.filters,
    #     hop_lengths=config.model.disc_hop_lengths,
    #     win_lengths=config.model.disc_win_lengths,
    #     n_ffts=config.model.disc_n_ffts,
    # )

    # log model, disc model parameters and train mode
    # print(model)
    # print(disc_model)
    print(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")
    # print(f"disc model train mode :{disc_model.training}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total number of parameters: {total_params}")
    # total_params = sum(p.numel() for p in disc_model.parameters())
    print(f"Discriminator Total number of parameters: {total_params}")
    return model

def init_dataset(config, mode="test"):
    cv = config.dataset.cv
    max_length = config.dataset.max_length

    datasets = {}
    # selected channels
    channels = dict()
    # if config.dataset.thorax > 0:
    #     channels['thorax'] = config.dataset.thorax
    # if config.dataset.abdominal > 0:
    #     channels['abdominal'] = config.dataset.abdominal
    channels = {"thorax": 1} #Hard code for now
    mgh_channels = {"thorax": 1}
    
    if mode == "test":
        mgh_dataset = "mgh_new"
    else:
        mgh_dataset = "mgh_train_encodec"

    datasets["mgh"]=(BreathingDataset(dataset = mgh_dataset, mode = mode, cv = cv, channels = mgh_channels, max_length = max_length))
    datasets["shhs2"]=(BreathingDataset(dataset = "shhs2_new", mode = mode, cv = cv, channels = channels, max_length = max_length))
    datasets["shhs1"]=(BreathingDataset(dataset = "shhs1_new", mode = mode, cv = cv, channels = channels, max_length = max_length))
    datasets["mros1"]=(BreathingDataset(dataset = "mros1_new", mode = mode, cv = cv, channels = channels, max_length = max_length))
    datasets["mros2"]=(BreathingDataset(dataset = "mros2_new", mode = mode, cv = cv, channels = channels, max_length = max_length))
    datasets["wsc"]=(BreathingDataset(dataset = "wsc_new", mode = mode, cv = cv, channels = channels, max_length = max_length))
    datasets["cfs"]=(BreathingDataset(dataset = "cfs", mode = mode, cv = cv, channels = channels, max_length = max_length))
    datasets["bwh"]=(BwhDataset(dataset = "bwh_new", mode = mode, cv = cv, channels = channels, max_length = max_length))
    datasets["mesa"]=(BreathingDataset(dataset = "mesa_new", mode = mode, cv = cv, channels = channels, max_length = max_length))
    return datasets

def process_dataset(ds_name, test_ds, model, save_dir, compression_ratio, done):
    """
    Process a single dataset on the specified GPU.
    """
    test_ds.file_list = [f for f in test_ds.file_list if f not in done]
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
    l1, count = 0, 0
    for item in tqdm(test_loader, desc=f"Processing {ds_name}"):
        x = item["x"].to(device)
        filename = item["filename"]
        x_hat, codes, _, _ = model(x)
        # x_hat = x_hat.squeeze().cpu().detach().numpy()
        l1 += torch.nn.L1Loss(reduction='mean')(x, x_hat).item()
        count += 1

        # Save the prediction
        # np.savez(os.path.join(save_dir, "shhs2_new", "thorax", filename[0]), data=x_hat, fs=10)

        # Save the codes
        save_path = os.path.join(save_dir, ds_name, "codes", filename[0])
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, data=codes.squeeze().cpu().detach().numpy(), fs=10/compression_ratio)
    
    print(f"Finished processing {ds_name}")
    return l1 / count

def get_code_distribution(ds_name, test_ds, save_dir, bins, pivot=None):
    all_codes = []
    for filename in tqdm(test_ds.file_list):
        codes = np.load(os.path.join(save_dir, ds_name, "codes", filename))['data'] #32 by ?
        all_codes.append(codes)
    num_codebooks = all_codes[0].shape[0]
    histogram_bins = bins

    # Prepare to aggregate data for each feature
    feature_counts = np.zeros((num_codebooks, histogram_bins), dtype=int)

    # Aggregate counts for each feature
    for sample in all_codes:
        for codebook_idx in range(num_codebooks):
            feature_data = sample[codebook_idx] #T
            assert sample[codebook_idx].min() >= 0, "min 0"
            assert sample[codebook_idx].max() < bins, f"max {sample[codebook_idx].max()}"
            counts, _ = np.histogram(feature_data, bins=histogram_bins, range=(0, histogram_bins - 1))
            feature_counts[codebook_idx] += counts
    
    if pivot is None:
        #sort by highest to lowest frequency 
        pivot = {}
        for codebook_idx in range(num_codebooks):
            sorted_indices_desc = np.argsort(feature_counts[codebook_idx])[::-1] #highest to lowest
            pivot[codebook_idx] = sorted_indices_desc
            assert np.array_equal(feature_counts[codebook_idx][sorted_indices_desc],sorted(feature_counts[codebook_idx], reverse=True))
        most_common = {}
        for codebook_idx in range(num_codebooks):
            # feature_counts[codebook_idx] = feature_counts[codebook_idx][pivot[codebook_idx]]
            most_common[codebook_idx] = {i: {"idx": pivot[codebook_idx][i],
                                             "count": feature_counts[codebook_idx][pivot[codebook_idx][i]], 
                                             "frequency": feature_counts[codebook_idx][pivot[codebook_idx][i]]/feature_counts[codebook_idx].sum()}
                                            for i in range(5)} #5 most common
        return pivot, most_common
    else:
        for codebook_idx in range(num_codebooks):
            feature_counts[codebook_idx] = feature_counts[codebook_idx][pivot[codebook_idx]]
            
        # Plot histograms for each feature
        fig, axes = plt.subplots(8, 4, figsize=(20, 15))  # 8 rows, 4 columns for 32 features
        axes = axes.flatten()

        for codebook_idx in range(num_codebooks):
            axes[codebook_idx].bar(range(histogram_bins), feature_counts[codebook_idx], color='blue', alpha=0.7)
            axes[codebook_idx].set_title(f'Codebook {codebook_idx} Distribution')
            axes[codebook_idx].set_xlim(-10, histogram_bins + 10)
            axes[codebook_idx].set_xlabel('Index')
            axes[codebook_idx].grid(True)
            axes[codebook_idx].set_ylabel('Frequency')

        # Remove empty subplots if any
        for i in range(num_codebooks, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        save_path = os.path.join(save_dir, ds_name, f"{ds_name}_token_distribution.png")  # Save as PNG
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
        plt.close()  # Close the figure to free memory

    print(f"Finished processing {ds_name}")

def plot_most_frequent_signals(ds_name, pivot, model, save_dir, config, device):
    num_codebooks = int(100 * config.model.target_bandwidths[0])
    # print(f'num codebooks {num_codebooks}')
    codes = []
    for codebook_idx in range(num_codebooks):
        most_common_code = pivot[codebook_idx][0]
        codes.append(most_common_code)
    codes = torch.tensor(codes).unsqueeze(1).unsqueeze(2) #N, B, T
    codes = codes.to(device)

    # Plot histograms for each feature
    fig, axes = plt.subplots(8, 4, figsize=(20, 15))  # 8 rows, 4 columns for 32 features
    axes = axes.flatten()

    prev = None
    for n_q in range(1,num_codebooks+1):
        quantized = model.quantizer.decode(codes, n_q=n_q)
        output = model.decoder(quantized).detach().cpu().numpy().squeeze()
        if prev is not None:
            diff = output - prev 
        else:
            diff = output
        mean = np.mean(diff)
        std = np.std(diff)
        time = np.arange(0, 300)
        axes[n_q-1].plot(time, diff)
        axes[n_q-1].set_title(f"Signal n_q={n_q} - Signal n_q={n_q-1}, mean {float(f'{mean:.6f}')}, std {float(f'{std:.6f}')}")
        axes[n_q-1].set_ylim(-0.5, 0.5)
        prev = output
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, ds_name, f"{ds_name}_most_common_signals.png")  # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
    plt.close()  # Close the figure to free memory

    #plot the most common signal 
    quantized = model.quantizer.decode(codes, n_q=num_codebooks)
    output = model.decoder(quantized).detach().cpu().numpy().squeeze()
    fig, ax = plt.subplots()  # Create a single axes (not multiple)
    # Plot only the specified axes
    time = np.arange(0, 300)
    ax.plot(time, output)
    ax.set_title(f"{ds_name}_generic_signal")
    ax.set_ylim(-2, 2)
    save_path = os.path.join(save_dir, ds_name, f"{ds_name}_generic_signal.png")  # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
    plt.close()  # Close the figure to free memory

    print(f"Finished processing {ds_name}")

if __name__ == "__main__":

    # log_dir = "tensorboard/231224_l1"
    # log_dir = "tensorboard/261224_l1"
    # log_dir = "tensorboard/271224_l1"
    # save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_5s"
    
    # log_dir = "/data/netmit/wifall/breathing_tokenizer/encodec_weights/model_30s"
    # save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_30s"

    # log_dir = "/data/netmit/wifall/breathing_tokenizer/encodec_weights/model_30s_disc"
    # save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_30s_disc"

    # log_dir = "/data/netmit/wifall/breathing_tokenizer/encodec_weights/model_30s_new"
    # save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_30s_new"

    # log_dir = "/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250114/175306"
    # save_dir = "/data/scratch/ellen660/encodec/encodec/predictions/175306"

    log_dir = "/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250115/140935"
    save_dir = "/data/scratch/ellen660/encodec/encodec/predictions/140935"
    datasets = ["mgh", "shhs2", "shhs1", "mros1", "mros2", "wsc", "cfs", "bwh"]
    # datasets = ["bwh"]
    resume = False

    # Load the YAML file
    config = load_config(f'{log_dir}/config.yaml', log_dir)

    # Initialize model and discriminator
    # val_ds = BreathingDataset(dataset="shhs2_new", mode="test", cv=0, channels={"thorax": 1.0}, max_length=None)
    # val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    test_datasets = init_dataset(config, mode="test")
    os.makedirs(save_dir, exist_ok=True)
    for ds_name in test_datasets.keys():
        os.makedirs(os.path.join(save_dir, ds_name), exist_ok=True)
        os.makedirs(os.path.join(save_dir, ds_name, "codes"), exist_ok=True)

    model = init_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to device
    model = model.to(device)

    # Checkpoint path (set this to your specific checkpoint)
    checkpoint_path_model = f"{log_dir}/model.pth"
    # checkpoint_path_disc = f"{log_dir}/disc.pth"

    compression_ratio = np.prod(config.model.ratios)

    # ===================== RELOAD CHECKPOINT =====================
    print("Loading model and discriminator from checkpoint...")
    checkpoint_model = torch.load(checkpoint_path_model, map_location=device)
    # checkpoint_disc = torch.load(checkpoint_path_disc, map_location=device)

    # Load state_dict into model and discriminator
    # if config.distributed.data_parallel:
    #     model.module.load_state_dict(checkpoint_model)
    #     disc.module.load_state_dict(checkpoint_disc)
    # else:
    model.load_state_dict(checkpoint_model)
    # disc.load_state_dict(checkpoint_disc)

    print("Checkpoint loaded successfully!")

    model.eval()
    
    # for ds_name, test_ds in test_datasets.items():
    #     print(f'processing for {ds_name}')
    #     test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=10)
    #     for i, item in enumerate(tqdm(test_loader)):
    #         x = item["x"]
    #         filename = item["filename"]
    #         x = x.to(device)
    #         x_hat, codes, _, _ = model(x)
    #         x_hat = x_hat.squeeze().cpu().detach().numpy()

    #         # save the prediction in the folder
    #         # np.savez(os.path.join(save_dir, "shhs2_new", "thorax", filename[0]), data=x_hat, fs = 10)

    #         # save the codes in the folder
    #         np.savez(os.path.join(save_dir, ds_name, "codes", filename[0]), data=codes.squeeze().cpu().detach().numpy(), fs = 10/compression_ratio)

    print(f'log_dir {log_dir}')
    print(f'datasets {datasets}')

    #Code Generation
    # test_datasets = init_dataset(config, mode="test")
    # for ds_name in datasets:
    #     test_ds = test_datasets[ds_name]
    #     done = set()
    #     if resume:
    #         done = set([f for f in os.listdir(os.path.join(save_dir, ds_name, "codes")) if f.endswith('.npz')])
    #     l1 = process_dataset(ds_name, test_ds, model, save_dir, compression_ratio, done=done)
    #     print(f'l1 for {ds_name}: {l1}')

    # #Token Distribution
    # test_datasets = init_dataset(config, mode="test")
    # pivot = get_code_distribution("shhs1", test_datasets["shhs1"], save_dir, config.model.bins)
    # for i, (ds_name, test_ds) in enumerate(test_datasets.items()):
    #     get_code_distribution(ds_name, test_ds, save_dir, config.model.bins, pivot=pivot)

    #Plot most frequent bwh signals (augment of the dataset?)
    test_datasets = init_dataset(config, mode="test")
    dataframe = {}
    for ds_name in datasets:
        pivot, most_common = get_code_distribution(ds_name, test_datasets[ds_name], save_dir, config.model.bins)
        dataframe[ds_name] = most_common
        # plot_most_frequent_signals(ds_name, pivot, model, save_dir, config, device)

    import pandas as pd

    # Create a list to hold rows of data
    rows = []

    # Process each dataset
    output_file = "datasets_summary_140935.xlsx"
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        # Iterate through each codebook index to create a separate sheet
        for codebook_idx in range(32):
            rows = []
            for ds_name, dataset in dataframe.items():
                row = [ds_name]  # Start with the dataset name
                for i in range(5):  # Iterate through the top 5 entries
                    entry = dataset[codebook_idx][i]  # Get the details for the current codebook index
                    row.extend([entry["idx"], entry["count"], entry["frequency"]])
                rows.append(row)

            # Define columns for the DataFrame
            columns = ["Dataset Name"]
            for i in range(5):
                columns.extend([f"i={i}_idx", f"i={i}_count", f"i={i}_frequency"])

            # Create a DataFrame for the current codebook index
            df = pd.DataFrame(rows, columns=columns)

            # Write this DataFrame to a new sheet in the Excel file
            sheet_name = f"Codebook_{codebook_idx}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Excel file with 32 sheets saved as {output_file}")