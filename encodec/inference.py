import os
import sys
import torch
import torch.nn as nn

from model import EncodecModel
from data.dataset import BreathingDataset

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
    # log_dir = "tensorboard/271224_l1"
    # save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_5s"
    
    # log_dir = "/data/netmit/wifall/breathing_tokenizer/encodec_weights/model_30s"
    # save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_30s"

    log_dir = "/data/netmit/wifall/breathing_tokenizer/encodec_weights/model_30s_disc"
    save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_30s_disc"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, "shhs2_new"))
        os.mkdir(os.path.join(save_dir, "shhs2_new", "thorax"))
        
    # Load the YAML file
    config = load_config(f'{log_dir}/config.yaml', log_dir)

    device = torch.device("cuda")

    # Initialize model and discriminator
    val_ds = BreathingDataset(dataset="shhs2_new", mode="test", cv=0, channels={"thorax": 1.0}, max_length=None)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

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
    
    for i, item in enumerate(tqdm(val_loader)):
        x = item["x"]
        filename = item["filename"]
        x = x.to(device)
        x_hat, codes, _ = model(x)
        x_hat = x_hat.squeeze().cpu().detach().numpy()

        # save the prediction in the folder
        np.savez(os.path.join(save_dir, "shhs2_new", "thorax", filename[0]), data=x_hat, fs = 10)

        # save the codes in the folder
        np.savez(os.path.join(save_dir, "shhs2_new", "codes", filename[0]), data=codes.squeeze().cpu().detach().numpy(), fs = 10/compression_ratio)