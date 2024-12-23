import os
import sys
import torch
import torch.nn as nn

from model import EncodecModel
from data.dataset import BreathingDataset
from my_code.losses import loss_fn_l1, loss_fn_l2, total_loss, disc_loss

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import yaml
# Define train one step function
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":

    # log_dir = "tensorboard/231224_l1"
    # log_dir = "tensorboard/261224_l1"
    log_dir = "tensorboard/271224_l1"
    save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions"
        
    # Load the YAML file
    config = load_config(f'{log_dir}/config.yaml', log_dir)

    device = torch.device("cuda")
    torch.manual_seed(config.common.seed)

    # Initialize model and discriminator
    val_ds = BreathingDataset(dataset="shhs2_new", mode="test", cv=0, channel="thorax", max_length=config.dataset.max_length)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config.dataset.num_workers)

    model = init_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to device
    model = model.to(device)
    # disc = disc.to(device)

    # # Data Parallel if enabled
    # if config.distributed.data_parallel:
    #     model = nn.DataParallel(model)
    #     disc = nn.DataParallel(disc)

    # Checkpoint path (set this to your specific checkpoint)
    checkpoint_path_model = f"{log_dir}/model.pth"
    # checkpoint_path_disc = f"{log_dir}/disc.pth"

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
    
    #only first 10 samples of val_loader
    for i, item in enumerate(tqdm(val_loader)):
        x = item["x"]
        filename = item["filename"]
        x = x.to(device)
        x_hat, _, _ = model(x)
        x_hat = x_hat.squeeze().cpu().detach().numpy()

        # save the prediction in the folder
        np.savez(os.path.join(save_dir, "shhs2_new", "thorax", filename[0]), data=x_hat, fs = 10)