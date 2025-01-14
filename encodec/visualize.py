import torch
import torch.nn as nn

from model import EncodecModel
from data import MergedDataset
from data.dataset import BreathingDataset
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

def init_dataset(config):
    train_datasets = []
    val_datasets = []
    weight_list = []
    if config.dataset.shhs2 > 0:
        train_datasets.append(BreathingDataset(dataset = "shhs2_new", mode = "train", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        val_datasets.append(BreathingDataset(dataset = "shhs2_new", mode = "val", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        weight_list.append(config.dataset.shhs2)
    if config.dataset.shhs1 > 0:
        train_datasets.append(BreathingDataset(dataset = "shhs1_new", mode = "train", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        val_datasets.append(BreathingDataset(dataset = "shhs1_new", mode = "val", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        weight_list.append(config.dataset.shhs1)
    if config.dataset.mros1 > 0:
        train_datasets.append(BreathingDataset(dataset = "mros1_new", mode = "train", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        val_datasets.append(BreathingDataset(dataset = "mros1_new", mode = "val", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        weight_list.append(config.dataset.mros1)
    if config.dataset.mros2 > 0:
        train_datasets.append(BreathingDataset(dataset = "mros2_new", mode = "train", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        val_datasets.append(BreathingDataset(dataset = "mros2_new", mode = "val", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        weight_list.append(config.dataset.mros2)
    if config.dataset.wsc > 0:
        train_datasets.append(BreathingDataset(dataset = "wsc_new", mode = "train", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        val_datasets.append(BreathingDataset(dataset = "wsc_new", mode = "val", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        weight_list.append(config.dataset.wsc)
    if config.dataset.cfs > 0:
        train_datasets.append(BreathingDataset(dataset = "cfs", mode = "train", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        val_datasets.append(BreathingDataset(dataset = "cfs", mode = "val", cv = config.dataset.cv, channel = "thorax", max_length = config.dataset.max_length))
        weight_list.append(config.dataset.cfs)

    print("Number of training datasets: ", len(train_datasets))
    # sys.exit()

    # merge the datasets
    train_dataset = MergedDataset(train_datasets, weight_list, 1, debug = config.dataset.debug)
    val_dataset = MergedDataset(val_datasets, weight_list, 0.2, debug = config.dataset.debug)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=config.dataset.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.dataset.num_workers)

    return train_loader, val_loader


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
    disc_model = MultiScaleSTFTDiscriminator(
        in_channels=config.model.channels,
        out_channels=config.model.channels,
        filters=config.model.filters,
        hop_lengths=config.model.disc_hop_lengths,
        win_lengths=config.model.disc_win_lengths,
        n_ffts=config.model.disc_n_ffts,
    )

    # log model, disc model parameters and train mode
    # print(model)
    # print(disc_model)
    print(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")
    # print(f"disc model train mode :{disc_model.training}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total number of parameters: {total_params}")
    # total_params = sum(p.numel() for p in disc_model.parameters())
    print(f"Discriminator Total number of parameters: {total_params}")
    return model, disc_model

def set_args():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--exp_name", type=str, default="config")
    parser.add_argument("--exp_name", type=str, default="091224_l1")
    
    return parser.parse_args()

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
    log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250106/161502' #whole dataset L1 0.7
    # Load the YAML file
    config = load_config(f'{log_dir}/config.yaml', log_dir)

    device = torch.device("cuda")
    torch.manual_seed(config.common.seed)

    # Initialize model and discriminator
    train_loader, val_loader = init_dataset(config)
    model, disc = init_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to device
    model = model.to(device)
    disc = disc.to(device)

    # Checkpoint path (set this to your specific checkpoint)
    checkpoint_path_model = f"{log_dir}/model.pth"
    checkpoint_path_disc = f"{log_dir}/disc.pth"

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
    disc.eval()
    
    #only first 10 samples of val_loader
    for i, item in enumerate(tqdm(val_loader)):
        x = item["x"]
        if i >=10:
            break

        # plot x and the reconstructed x
        fig, axs = plt.subplots(9, 5, figsize=(20, 20))
        # axs[0,0].plot(x[0].numpy().squeeze())
        # axs[0,0].set_title('Original')
        # axs[0,0].set_ylim(-2, 2)
        time = np.arange(0, 300)
        axs[0,0].plot(time, x[0].cpu().numpy().squeeze()[10000:10300])
        axs[0,0].set_xlabel("Time")
        axs[0,0].set_title("Original Signal 2 minute")
        axs[0,0].set_ylim(-6, 6)
        axs[0,1].plot(x[0].cpu().numpy().squeeze()[10000:10100])
        axs[0,1].set_title("Original Signal 5 second")
        axs[0,1].set_ylim(-6, 6)

        # axs[2].plot(x_hat[0].cpu().numpy().squeeze())
        # axs[2].set_title('Reconstructed')
        # axs[2].set_ylim(-4, 4)
        # axs[3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto')
        # axs[3].invert_yaxis()
        # axs[3].set_title('Reconstructed Spectrogram')

        x = x.to(device)
        emb = model.encoder(x)
        outputs = {}

        output = model.quantizer.intermediate_results(x=emb, n_q=32)
        x_hat = model.decoder(output['quantized'])

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
        fig1.savefig(f'/data/scratch/ellen660/encodec/encodec/{i}.png')
        plt.close(fig1)

        l1_losses = []
        freq_losses = []

        for n_q in range(1, 34, 4):
        # for n_q in range(1,9):
            output = model.quantizer.intermediate_results(x=emb,n_q=n_q)
            n_q = n_q//4
            out = model.decoder(output['quantized'])
            l1_loss = loss_fn_l1(x, out)
            freq_loss_dict = freq_loss(x, out)
            print(f'codebook {n_q}, l1 loss: {l1_loss}')
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
        # fig.savefig(f'/data/netmit/wifall/breathing_tokenizer/encodec/encodec/tensorboard/{config.exp_details.name}/reconstructed_{epoch}.png')
        print(f'saving to /data/scratch/ellen660/encodec/encodec/visualize_{i}.png')
        fig.savefig(f'/data/scratch/ellen660/encodec/encodec/visualize_{i}.png')
        plt.close(fig)

        #plot the l1_losses and freq_losses
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        axs[0].plot(l1_losses)
        axs[0].set_title("L1 Losses")
        axs[1].plot(freq_losses)
        axs[1].set_title("Frequency Losses")
        fig.tight_layout()
        fig.savefig(f'/data/scratch/ellen660/encodec/encodec/visualize_losses_{i}.png')
        plt.close(fig)


