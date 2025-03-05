import torch
import torch.nn as nn

from model import EncodecModel
from data import MergedDataset
from data.dataset import BreathingDataset
from data.bwh import BwhDataset
from my_code.losses import loss_fn_l1, loss_fn_l2, total_loss, disc_loss
from my_code.schedulers import LinearWarmupCosineAnnealingLR, WarmupScheduler
from msstftd import MultiScaleSTFTDiscriminator
# from scheduler import WarmupCosineLrScheduler
# from utils import (save_master_checkpoint, set_seed,
#                    start_dist_train)
from balancer import Balancer

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import yaml
import random
from collections import defaultdict
# Define train one step function
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np

import sys
from my_code.spectrogram_loss import BreathingSpectrogram, ReconstructionLoss, ReconstructionLosses

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

def init_logger(log_dir):
    print(f'log_dir: {log_dir}')
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def init_dataset(config, mode="test"):
    cv = config.dataset.cv
    max_length = config.dataset.max_length

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
    return model #disc_model

def set_args():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--exp_name", type=str, default="config")
    parser.add_argument("--exp_name", type=str, default="091224_l1")
    
    return parser.parse_args()

def get_data_distribution(ds_name, channel, train_ds, save_dir=f"/data/scratch/ellen660/encodec/encodec/visualizations"):
    """
    Plot distribution for dataset
    """
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

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {ds_name}")
    plt.grid(True)
    #save 
    save_path = os.path.join(save_dir, f"{ds_name}_{channel}_histogram.png")  # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
    plt.close()  # Close the figure to free memory
    print(f"Finished processing {ds_name}")


def get_patients_distribution(ds_name, channel, train_ds, save_dir=f"/data/scratch/ellen660/encodec/encodec/visualizations"):
    """
    Plot distribution for dataset
    """
    train_loader = iter(DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=10))
    for j in range(2):
        fig, axes = plt.subplots(6, 6, figsize=(20, 10))
        axes = axes.flatten()

        for i in range(36):
            item = next(train_loader)

            # Define the histogram parameters
            bin_edges = np.linspace(-4, 4, 50)  # 50 bins from -4 to 4
            histogram = np.zeros(len(bin_edges) - 1)  # Initialize empty histogram
            x = item["x"].numpy()  # Assuming x is in the batch and is a numpy-compatible tensor
            # Flatten and add to histogram
            if x is None:
                continue
            histogram = np.histogram(x, bins=bin_edges)[0]

            # Normalize the histogram to get probabilities (optional)
            histogram = histogram / histogram.sum()
            axes[i].bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
            # axes[i].set_xlabel("Feature Value")
            # axes[i].set_ylabel("Frequency")
            axes[i].set_title(f"File {item['filename'][0][:6]}")
            #set 
            axes[i].set_xlim(-6, 6)
            axes[i].grid(True)

        #save 
        save_path = os.path.join(save_dir, f"{ds_name}_{channel}_{j}_36_patients_histogram.png")  # Save as PNG
        plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
        plt.close()  # Close the figure to free memory
    print(f"Finished processing {ds_name}")

def plot_original_signals(ds_names, datasets):
    #plot the original signal for each dataset
    dataloaders = {ds_name: iter(DataLoader(datasets[ds_name], batch_size=1, shuffle=False, num_workers=4)) for ds_name in ds_names}
    dataloader = dataloaders["bwh"]
    find = {"76b9b35478467567a087eb349e4ae18e590823f59e2a47570b8d0f98201c3119.npz": 108541,
                "7066cd514c69a6f4149481fd0c05b7e4b0062c105328046a68214b8873dad8ea.npz": 114653,
                "c04da512b551b23836c7d0878151d95e871548758e72ebfa912ec77ef8632069.npz": 297372,
                "09321f0276fc017ffd0fff98947527402f14bd5cbbd00371a60dbe6a79d4a44a.npz": 175487,
                "4e39f655e59dc63a90183a756282b5cb7ccc80ac31d885acb186abb4e8b9128b.npz": 237190,
                "62b75ae63dde20c664774c901a742d161b701e0bf577b9f81776eae03961d8d3.npz": 71332,
                "cbbd41daccd9da5e6190881e2276cccfe6408ca292118b37c10f9f3d383b0fee.npz": 82833,
                "5d274bbaf979b10664a32a6aff5b367ec96377fe846f850a3e67302e20a744bf.npz": 103568
        }
    fig, axes = plt.subplots(4, 2, figsize=(20, 10))
    axes = axes.flatten()
    for i in range(8):
        item = next(dataloader)
        x = item["x"]
        start_idx = find[item['filename'][0]]
        time = np.arange(0, 300)
        axes[i].plot(time, x[0].cpu().numpy().squeeze()[start_idx:start_idx+300])
        axes[i].set_xlabel("Time")
        axes[i].set_title(f"Filename {item['filename'][0][:10]} from index {start_idx}")
        axes[i].set_ylim(-4, 4)

        #save figure 
        fig.tight_layout()
        fig.savefig(f"/data/scratch/ellen660/encodec/encodec/visualizations/new_bwh.png")
        plt.close(fig)

def testing_hierarchy(ds_name, item, model, freq_loss, device, save_dir, combine_codebooks=4):
    num_codebooks = model.quantizer.n_q
    x = item["x"]
    x = x.to(device)
    emb = model.encoder(x)
    output = model.quantizer.intermediate_results(x=emb, n_q=num_codebooks)
    quantized_stack = output['quantized_stack']
    outputs = {}
    for i in range(num_codebooks-combine_codebooks):
        #sum the quantized_stack from i to i+combine_codebooks
        quantized = torch.sum(quantized_stack[i:i+combine_codebooks], dim=0)
        x_hat = model.decoder(quantized)
        l1_loss = loss_fn_l1(x, x_hat)
        print(f'codebook {i} to {(i+combine_codebooks)}, l1 loss: {l1_loss}')
        outputs[i] = l1_loss
    return outputs


def infer(ds_name, item, model, freq_loss, device, save_dir):
    num_codebooks = model.quantizer.n_q
    x = item["x"]
    outputs = {}
    fig, axs = plt.subplots(9, 5, figsize=(20, 20))

    # plot x and the reconstructed x
    time = np.arange(0, 300)
    axs[0,0].plot(time, x[0].cpu().numpy().squeeze()[10000:10300])
    axs[0,0].set_xlabel("Time")
    axs[0,0].set_title("Original Signal 30 second")
    axs[0,0].set_ylim(-4, 4)
    axs[0,1].plot(x[0].cpu().numpy().squeeze()[10000:10100])
    axs[0,1].set_title("Original Signal 5 second")
    axs[0,1].set_ylim(-4, 4)

    x = x.to(device)
    emb = model.encoder(x)

    output = model.quantizer.intermediate_results(x=emb, n_q=num_codebooks)
    x_hat = model.decoder(output['quantized'])
    
    # breakpoint()

    # logits_real, fmap_real = disc(x)
    # logits_fake, fmap_fake = disc(x_hat)
    # disc_loss(logits_real, logits_fake)
    freq_loss_dict = freq_loss(x, x_hat)
    S_x = freq_loss_dict["S_x"]
    S_x_hat = freq_loss_dict["S_x_hat"]
    _, num_freq, _ = S_x.size()
    S_x = S_x[:, :num_freq//2, :]
    S_x_hat = S_x_hat[:, :num_freq//2, :]

    # use this to set the scale of the spectrogram
    min_spec_val = min(S_x.min(), S_x_hat.min())
    max_spec_val = max(S_x.max(), S_x_hat.max())

    time_start = 0
    time_end = x.shape[-1]

    x_time = np.arange(time_start, time_end, 1)

    # plot x and the reconstructed x
    fig1, axs1 = plt.subplots(4, 1, figsize=(20, 10), sharex=True)

    axs1[0].plot(x_time, x[0].cpu().numpy().squeeze())
    axs1[0].set_title('Original')
    axs1[0].set_ylim(-6, 6)
    axs1[1].imshow(S_x.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
    axs1[1].invert_yaxis()
    axs1[1].set_title('Original Spectrogram')

    axs1[2].plot(x_time, x_hat[0].detach().cpu().numpy().squeeze())
    axs1[2].set_title('Reconstructed')
    axs1[2].set_ylim(-6, 6)
    axs1[3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
    axs1[3].invert_yaxis()
    axs1[3].set_title('Reconstructed Spectrogram')

    fig1.tight_layout()
    fig1.savefig(f'{save_dir}/{ds_name}_{item["filename"][0][:10]}.png')
    plt.close(fig1)

    l1_losses = []
    freq_losses = []

    # for n_q in range(1, num_codebooks+1, num_codebooks//8):
    for n_q in range(1, 9):
        output = model.quantizer.intermediate_results(x=emb,n_q=(num_codebooks//8*n_q))
        out = model.decoder(output['quantized'])
        l1_loss = loss_fn_l1(x, out)
        freq_loss_dict = freq_loss(x, out)
        print(f'codebook {(num_codebooks//8*n_q)}, l1 loss: {l1_loss}')
        # print(f'out sie: {out.size()}')
        S_x = freq_loss_dict["S_x"]
        S_x_hat = freq_loss_dict["S_x_hat"]
        l1_losses.append(l1_loss.cpu().detach().numpy())
        freq_losses.append(freq_loss_dict["l1_loss"].cpu().detach().numpy())
        _, num_freq, _ = S_x.size()
        S_x = S_x[:, :num_freq//2, :]
        S_x_hat = S_x_hat[:, :num_freq//2, :]

        # axs[n_q,0].plot(out[0].detach().cpu().numpy().squeeze())
        # axs[n_q,0].set_title(f'n_q={n_q}')
        # axs[n_q,0].set_ylim(-2, 2)
        # n_q = n_q//(num_codebooks//8)
        if n_q == 1:
            axs[0,3].imshow(S_x.detach().cpu().numpy()[0], cmap='jet', aspect='auto')
            axs[0,3].invert_yaxis()
            axs[0,3].set_title('Original Spectrogram')
            axs[0,4].imshow(S_x.detach().cpu().numpy()[0, :, 10000//50: 10300//50], cmap='jet', aspect='auto')
            axs[0,4].invert_yaxis()
            axs[0,4].set_title("Spectrogram")

        time = np.arange(0, 300)
        axs[n_q,0].plot(time, out.detach().cpu().numpy().squeeze()[10000:10300])
        #plot original signal with transparent
        axs[n_q,0].plot(time, x[0].detach().cpu().numpy().squeeze()[10000:10300], alpha=0.3)
        axs[n_q,0].set_xlabel("Time")
        axs[n_q,0].set_title(f"Signal n_q={n_q*4}")
        axs[n_q,0].set_ylim(-6, 6)
        axs[n_q,1].plot(out.detach().cpu().numpy().squeeze()[10000:10100])
        #plot original signal with transparent
        axs[n_q,1].plot(x[0].detach().cpu().numpy().squeeze()[10000:10100], alpha=0.3)
        axs[n_q,1].set_title(f"Signal n_q={n_q*4}, , Loss = {l1_loss.item()}")
        axs[n_q,1].set_ylim(-6, 6)

        axs[n_q,3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto')
        axs[n_q,3].invert_yaxis()
        axs[n_q,3].set_title(f'Reconstructed Spectrogram, n_q={4*n_q}')
        axs[n_q,4].imshow(S_x_hat.detach().cpu().numpy()[0, :, 10000//50: 10300//50], cmap='jet', aspect='auto')
        axs[n_q,4].invert_yaxis()
        axs[n_q,4].set_title(f"Spectrogram n_q={n_q*4}")

        outputs[n_q] = out.detach().cpu().numpy().squeeze()

        del output, l1_loss, S_x, S_x_hat

    #plot the cumulative signal
    for j in range(1, 8):
        a = outputs[j]
        b = outputs[j+1]
        diff = b - a
        time = np.arange(0, 300)
        axs[j,2].plot(time, diff[10000:10300])
        axs[j,2].set_title(f"Signal n_q={(j+1)*4} - Signal n_q={j*4}")
        axs[j,2].set_ylim(-2, 2)

    #save figure 
    fig.tight_layout()
    # print(f'saving to /data/scratch/ellen660/encodec/encodec/visualizations/{ds_name}_visualize_{item["fiename"][0][:6]}.png')
    fig.savefig(f'{save_dir}/{ds_name}_visualize_{item["filename"][0][:10]}.png')
    plt.close(fig)

    #plot the l1_losses and freq_losses
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(l1_losses)
    axs[0].set_title("L1 Losses")
    axs[1].plot(freq_losses)
    axs[1].set_title("Frequency Losses")
    fig.tight_layout()
    fig.savefig(f'{save_dir}/{ds_name}_visualize_losses_{item["filename"][0][:10]}.png')
    plt.close(fig)


def get_zeros(ds_name, channel, test_ds, save_dir=f"/data/scratch/ellen660/encodec/encodec/visualizations"):
    """
    Plot distribution for dataset
    """
    train_loader = DataLoader(test_ds, batch_size=1, shuffle=True, num_workers=10)

    # Define the histogram parameters
    bin_edges = np.linspace(0, 1, 100)  # 50 bins from -4 to 4
    histogram = np.zeros(len(bin_edges) - 1)  # Initialize empty histogram
    count=0

    # Iterate through the DataLoader
    for batch in tqdm(train_loader, desc="Getting distribution"):
        x = batch["x"].numpy()  # Assuming x is in the batch and is a numpy-compatible tensor
        # Flatten and add to histogram
        if x is None:
            continue
        # indices = []
        # for idx in range(x.shape[-1]):
        #     all_same = np.all(x[0, 0, idx:idx+200*30] == x[0, 0, idx])
        #     if all_same:
        #         indices.append(idx)

        # zero_indices = np.array(indices)
        # normalized_indices = zero_indices / x.shape[-1]
        # histogram += np.histogram(normalized_indices, bins=bin_edges)[0]
        # breakpoint()

        # Assuming `x` is your input array of shape (batch_size, channels, T)
        window_size = 200 * 5
        T = x.shape[-1]

        # Create a sliding window view of the array
        try:
            strided_view = np.lib.stride_tricks.sliding_window_view(x[0, 0, :], window_size)
        except:
            print(f'file {batch["filename"][0]}')
            continue

        # Compare all values in each sliding window to the first value of the window
        all_same = np.all(strided_view == strided_view[:, 0][:, None], axis=1)
        all_same = [i for i, x in enumerate(all_same) if x]
        # if len(all_same)>2:
        #     val = strided_view[all_same[0], 0]
        #     val2 = strided_view[all_same[-1], 0]
        #     print(f'val {val}, val2 {val2}')
            # breakpoint()
            # assert np.all(strided_view[all_same, 0] == val)
        zero_indices = np.array(all_same)
        normalized_indices = zero_indices / (x.shape[-1])
        if len(normalized_indices)>0 and max(normalized_indices) < 0.01:
            print(f'file {batch["filename"][0]}, max index {max(normalized_indices)}')
            breakpoint()
        # if normalized_indices.size > 0 and max(normalized_indices) > 0.01:
        #     print(f'file {batch["filename"][0]}, max index {max(normalized_indices)}')
        #     count += 1
        #     #plot the x
        #     time_start = 0
        #     time_end = x.shape[-1]

        #     x_time = np.arange(time_start, time_end, 1)
        #     plt.figure(figsize=(8, 6))
        #     plt.plot(x_time, x[0, 0, :])
        #     plt.xlabel("Time")
        #     plt.title(f"Original Signal {batch['filename'][0]}")
        #     plt.grid(True)
        #     #save
        #     save_path = os.path.join(save_dir, f"bwh_bad_{batch['filename'][0][:10]}.png")  # Save as PNG
        #     plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
        #     plt.close()  # Close the figure to free memory
        # if count > 5:
        #     raise ValueError("Too many files with max index > 0.01")
        histogram += np.histogram(normalized_indices, bins=bin_edges)[0]
        # breakpoint()

    # Normalize the histogram to get probabilities (optional)
    # breakpoint()
    histogram = histogram / histogram.sum()

    # Plot the histogram
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], histogram, width=np.diff(bin_edges), edgecolor="black", align="edge")
    plt.xlabel("Index Value")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of 0 indices")
    plt.grid(True)
    #save 
    save_path = os.path.join(save_dir, f"zeros_histogram.png")  # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":

    args = set_args()

    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241216/233026'
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241218/223733'
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241219/235510' #best model L1 0.06
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241221/183951' #30 second l1 0.3
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241221/201924' #whole dataset 30 second l1 0.3
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241223/121731' #10 second 500 samples L1 0.1
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241223/121948' #15 second 500 samples L1 0.13
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241222/103721' #30 second 2048 samples L1 = 0.2
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241223/162546' #30 second model tuned L1 = 0.2 with commitment loss + bigger codebook/feature dimension
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241225/104928' #30 second model 
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241226/190339' #30 second model L1 0.1
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241229/184424' #30 second model L1 0.1
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250102/214627' #whole dataset w/ discriminator, 30 second model L1 0.1
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250104/225043' #30 second discrim good L1 0.1
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250106/161502' #whole dataset L1 0.7
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250118/135321' #multiple datasets #3
    # save_dir = f'/data/scratch/ellen660/encodec/encodec/visualizations/135321/OH'

    log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250130/174436' #multiple datasets #3
    OH = ["16376d00ed32a4dec5723859a792bd12f9d430b93bca411671d45738c10422ed.npz", "e86a4f8f2f59491028283b9759c6687183ba578744b6cedcc313cc185d2b4ab7.npz", "17ed3db05a929a7dd7b975ee55da29f24171f06cef1cac4a43caac12193804f1.npz", "0c058d54074b66ddf0ae4345eca900dea2e55366ba5b9564620af684501c8a32.npz", "d6e157d3cf2e296a4679c0a120b34e0209449fd2ae5e3db1816ef1150a9fc3c2.npz", "4bc9bda3ccf87da6933e91a1fe3ad2f651e5b0d3b6b12477004575c56735430a.npz", "0306bd6f64d947d0095f37b4f4327f666c0bdb77e91dce98d39cb7c8f7cccd2b.npz", "2a5d68e1fb9ec0233665d92158555d974b83f945d46130e29186936b86b1e4b2.npz", "6cbaf28a34f79bfa43a223850afd54aa32886879009892a7d56f8a213029f2d6.npz", "d61ad5dd2d1208c8359fd8f11cccf52b7c111c4e7e9c03da006a5274099eac06.npz", "240c0d490ac0686085abdb905c7bca1c07cc42837613710eeefebc7d22de2b11.npz", "a346cba0440cb62410ec20340559308731d63c5382ec3b4f4a30e41188da3158.npz", "83d6000b737a0a4ad57853e2274a1ebe782dc4c6552dc3e5ae2487c8697f544e.npz", "a070e6e88a779be31585692a518da4dae008ce3156919b47e32187ba576f4675.npz", "a98318d515ba1e4d7a05f416115eef75fd7bd38c22d7f74714bc3cb7d6aedf2b.npz", "269c73faf0300598190ddb6bc275f1e5e67984bf163e4e40f79c0f31f051c892.npz", "312a8060d404b2612f9f87ccbb1e7fb4ae17f1db2cbabfb9e92452fa6783c019.npz", "fdad7b68a4fc30637b31a21994e30349bf089f4f5b22c7d4822a51f5cd0e86d1.npz", "64bc0490afb8e79b65366ba47fc76013aa019032a5a848bca6b4c3ae8f2adc2c.npz", "b2ce37375d1102e20efda780e2a27e53d9ae2b534aee231748754d9ee42cfa4b.npz", "054e3b324232239441a744aea5ec7ca4dae1f97a4a1c82a169ea66a9c9784eb9.npz", "0556aec82ca70fd138012a55fc3a655bbae6f3defdc1ac1be1367fbe3d16fe94.npz", "0535f3c0572e33f8fb7052cdd7b89c87175f6eda6a8951fb13486a130263d662.npz", "11706605000be680e092cbf8e8dfed7d31466b8f7d5f0fa0c42d9b1793498bbd.npz", "74da52502817923f4d6871c70b988b25fc9d6b48281f7e4a9468b2c0ca3d4166.npz", "779a1bb664c4e0304b0679f08c368f045d08e986a6451267c63fd85c26fc04d3.npz", "dc7d4e3e505447623eed0bff6da7d7edaeb69207c4bf3170767d62c14cc719a6.npz", "dc6a9a49771af8d0ca91f358df5c33c56cc37cfc453d6b086b96cf2f2a1716fc.npz", "52f85d9d0fc9598d5da112a7096057e9285ad651d2c4de8139a4d64be923ab23.npz", "065249c6499e13ebb605b5c4fc627a47abdd86b9f246d3741965ba0fc6fc4d02.npz", "cc322f9c9ec86fc12b6693756aae1dbe9dd887bf4711a44cab322a760915ce33.npz", "fe29dfe7f9b4d96a015c59a99d973c209c72b91539db4d9f1947a88de6f1cbad.npz", "b4287482401c58b8b7a344a8156f9971a499ae79dd1ee99b123f9900ba880df0.npz", "7f92dc942fd684fe58f90dd58ed175bc4ee38cf01686b4fc94ef993b1c43f079.npz", "801e42d205723d5c64165df4e9ffa4f4b1ff3dd9cb195debea5dafbde3bbb751.npz", "daf19f0188c30caa98473558bf5e5730c6fd85c82febf33ebb1c701d91f8ec05.npz", "178ce551e9aedb15846f2c9f93ff14bd2da03ebdd8cc49d003fcbfd0a754393d.npz", "2cbea769c0c4b9ead9bbf3739b68b160935b6cd2c76b3991ad3f2070284cca42.npz", "501887b3663bef4577969a04a31a2eb06baebd92558deec68ed51407dd891f9f.npz", "1000103f3e3cd6e9ae032b8241a440a2d7d06006e7203bf97111e52f3b5ae42a.npz", "b94bbe984305fc5f44d044193fc6456d378fbc8f89d8f8a1bced14b61f269469.npz", "2412af6676460545173f99e7525b951ead5dfc8d13cf78d835c0f9123a40552b.npz", "e6a51a8e5047dcaaaeac38c0ff42b4fb6e0d19258f7b574b832f80033eef9b0b.npz", "f4eae33c31b388da7de85f5a51635b04ed7000c42e61e6a81d046b79780db523.npz", "2aa578348f7841026318aac9ce0b2d1aba0b16ee15a9289c3b302b5122c65c88.npz", "222b59b0960f0877cb89883b93b0cdd662ce35de7c0b215614c0bacb3b32f7f9.npz", "1e8b987374f9610dbac805750d068f057bba6b40eacca87b61a9d3caeb464268.npz", "64c7cc7e659f67de27b46a9d9a182a9cd6c818e65f32a938b5c73596524c9601.npz", "a8adc1ebc689d34ad306d02c6cb34404a7407f4d18f90be71354762187eba45b.npz", "e91ea1198478b3eb60c517f52f3766c5e9934ce2ecbe74c100068f9b873bcc18.npz", "0b1681fcbae699efc4a7e32e7e08b1503d38f0d5d413f81b527815f955d71138.npz", "d8b855e865dfcbc1b0495f73a076a471ec79ef3478961c89830dacd4b6fed290.npz", "e4a088cd60f9e4ae29b13e3bb75fb9a8f21be8175fdb3599a1f0f968331186ba.npz", "04d65bcf235f1215131ed9d7f98585770ab1cff54671e8e154ddeb441fd67a0f.npz", "b3485cf08876e23af85efa916c3360a33444ea95c83f6bed8ddfb61df469cedc.npz", "1253f4cff3682f4b63e1efcc98ab2cff02a5b4fbc64c1f1601f5423d06f93976.npz", "84683466b5bb5cc4e2fdb0199b8c210ec82fb5b966416949dc3e4f670a879c12.npz", "ebad04a9b9320a5c271337a7c9957c0434ca8db54dd3ee3ededc297e7b74c1f8.npz"]
    os.makedirs(save_dir, exist_ok=True)

    # Load the YAML file
    config = load_config(f'{log_dir}/config.yaml', log_dir)

    device = torch.device("cuda")
    torch.manual_seed(config.common.seed)

    # Initialize model and discriminator
    model = init_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to device
    model = model.to(device)
    # disc = disc.to(device)

    # Checkpoint path (set this to your specific checkpoint)
    checkpoint_path_model = f"{log_dir}/model.pth"
    # checkpoint_path_disc = f"{log_dir}/disc.pth"

    # ===================== RELOAD CHECKPOINT =====================
    print("Loading model and discriminator from checkpoint...")
    checkpoint_model = torch.load(checkpoint_path_model, map_location=device)
    # checkpoint_disc = torch.load(checkpoint_path_disc, map_location=device)

    # Load state_dict into model and discriminator
    model.load_state_dict(checkpoint_model)
    # disc.load_state_dict(checkpoint_disc)
    try:
        freq_loss = ReconstructionLosses(alpha=config.loss.alpha, bandwidth=config.loss.bandwidth, sampling_rate=10, n_fft=config.loss.n_fft, hop_length=config.loss.hop_length, win_length=config.loss.win_length, device=device)
    except:
        freq_loss = ReconstructionLoss(alpha=config.loss.alpha, bandwidth=config.loss.bandwidth, sampling_rate=10, n_fft=config.loss.n_fft, device=device)

    print("Checkpoint loaded successfully!")

    model.eval()
    # disc.eval()
    
    # ds_names = ["mgh", "shhs2", "shhs1", "mros1", "mros2", "wsc", "cfs", "bwh", "mesa", "mgh_rf"]
    ds_names = ["bwh"]
    train_datasets = init_dataset(config, mode="train")
    # get_zeros("bwh", "thorax", train_datasets["bwh"]["thorax"])

    #plot the original signal for each dataset
    # plot_original_signals(ds_names, train_datasets)

    # #Data Distribution w/ Flipping
    # for ds_name in ds_names:
    #     for channel, train_ds in train_datasets[ds_name].items():
    #         get_data_distribution(ds_name, channel, train_ds)

    # # #Patient Distribution
    # for ds_name in ds_names:
    #     for channel, train_ds in train_datasets[ds_name].items():
    #         get_patients_distribution(ds_name, channel, train_ds)

    outputs = {i: [] for i in range(8)}
    for ds_name in ds_names:
        train_datasets[ds_name]["thorax"].file_list = OH
        data_loader = DataLoader(train_datasets[ds_name]["thorax"], batch_size=1, shuffle=True, num_workers=4)
        for i, item in enumerate(tqdm(data_loader)):
            if item["filename"][0] in OH:
                print(f'Processing {item["filename"][0][:6]}')
                # infer(ds_name, item, model, freq_loss, device, save_dir)
                outputs_i = testing_hierarchy(ds_name, item, model, freq_loss, device, save_dir)
                #update the outputs
                for i, loss in outputs_i.items():
                    outputs[i].append(loss)
        #take average
        for i, loss in outputs.items():
            outputs[i] = np.mean(loss)
        print(f'Final outputs {outputs}')

    