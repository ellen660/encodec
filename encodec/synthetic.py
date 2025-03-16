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

def generate_synthetic_codes(config):
    bins = config.model.bins
    n_codebooks = int(100 * config.model.target_bandwidths[0])
    #random int from 1 to n_codebooks-1
    idx = torch.randint(1, n_codebooks, (1,))

    #torch tensor of size (n_codebooks) all of value 1
    indices = torch.ones(n_codebooks, dtype=torch.long)
    #replace the value at index idx+1 with 2
    indices[idx+1] = 2

    #unsqueeze to add two dimensions
    indices = indices.unsqueeze(1).unsqueeze(1)
    return indices, idx+1

if __name__ == "__main__":

    log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250203/144513' #1 then 31

    # Load the YAML file
    config = load_config(f'{log_dir}/config.yaml', log_dir)

    device = torch.device("cuda")
    torch.manual_seed(config.common.seed)

    # Initialize model and discriminator
    model = init_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to device
    model = model.to(device)

    # Checkpoint path (set this to your specific checkpoint)
    checkpoint_path_model = f"{log_dir}/model.pth"

    # ===================== RELOAD CHECKPOINT =====================
    print("Loading model and discriminator from checkpoint...")
    checkpoint_model = torch.load(checkpoint_path_model, map_location=device)

    # Load state_dict into model and discriminator
    model.load_state_dict(checkpoint_model)

    model.eval()

    # ===================== GENERATE SYNTHETIC CODES =====================
    signals = []
    fig, axs = plt.subplots(2, 5, figsize=(20, 8))
    axs = axs.flatten()
    for i in range(10):
        # Generate synthetic codes
        indices, idx = generate_synthetic_codes(config)
        indices = indices.to(device)
        # print(indices, indices.shape)
        # breakpoint()
        # Generate synthetic audio
        with torch.no_grad():
            signal = model.quantizer.decode(indices)
            signal = model.decoder(signal)
            print(signal.shape)
            #remove first two dimensions
            signal = signal.squeeze(0).squeeze(0)
            # breakpoint()

        #plot signal 
        axs[i].plot(signal.cpu().numpy())
        axs[i].set_title(f"synthetic_{idx}")
        signals.append(signal.cpu().numpy())
    #assert all signals are the same
    assert all(np.allclose(signals[0], signal) for signal in signals)


    #save plot
    plt.tight_layout()
    plt.savefig(f"/data/scratch/ellen660/encodec/encodec/visualizations/synthetic_signal.png")

