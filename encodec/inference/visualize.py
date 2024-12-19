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

def init_logger(log_dir):
    print(f'log_dir: {log_dir}')
    writer = SummaryWriter(log_dir=log_dir)
    return writer

def init_dataset(config):
    #dataset
    dataset = BreathingDataset(debug = config.dataset.debug, max_length = config.dataset.max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
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

def set_args():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--exp_name", type=str, default="config")
    parser.add_argument("--exp_name", type=str, default="091224_l1")
    
    return parser.parse_args()

if __name__ == "__main__":

    args = set_args()

    log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20241216/233026'
        
    # Load the YAML file
    config = load_config(f'{log_dir}/config.yaml', log_dir)

    device = torch.device("cuda")
    torch.manual_seed(config.common.seed)

    # Initialize model and discriminator
    train_loader, val_loader = init_dataset(config)
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
    freq_loss = ReconstructionLosses(alpha=config.loss.alpha, bandwidth=config.loss.bandwidth, sampling_rate=10, n_fft=config.loss.n_fft, hop_length=config.loss.hop_length, win_length=config.loss.win_length, device=device)

    print("Checkpoint loaded successfully!")

    model.eval()
    
    #only first 10 samples of val_loader
    for i, (x, _) in enumerate(tqdm(val_loader)):
        if i >=10:
            break
        
        # plot x and the reconstructed x
        fig, axs = plt.subplots(9, 6, figsize=(20, 20))
        axs[0,0].plot(x[0].numpy().squeeze())
        axs[0,0].set_title('Original')
        axs[0,0].set_ylim(-2, 2)
        time = np.arange(0, 1200)
        axs[0,1].plot(time, x[0].numpy().squeeze()[10000:11200])
        axs[0,1].set_xlabel("Time")
        axs[0,1].set_title("Original Signal 2 minute")
        axs[0,1].set_ylim(-2, 2)
        axs[0,2].plot(x[0].numpy().squeeze()[10000:10100])
        axs[0,2].set_title("Original Signal 5 second")
        axs[0,2].set_ylim(-2, 2)

        # axs[2].plot(x_hat[0].cpu().numpy().squeeze())
        # axs[2].set_title('Reconstructed')
        # axs[2].set_ylim(-4, 4)
        # axs[3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto')
        # axs[3].invert_yaxis()
        # axs[3].set_title('Reconstructed Spectrogram')

        x = x.to(device)
        emb = model.encoder(x)
        outputs = {}
        for n_q in range(1, 9):
            output = model.quantizer.intermediate_results(x=emb,n_q=n_q)
            out = model.decoder(output['quantized'])
            l1_loss = loss_fn_l1(x, out)
            freq_loss_dict = freq_loss(x, out)
            print(f'loss: {l1_loss}')
            S_x = freq_loss_dict["S_x"]
            S_x_hat = freq_loss_dict["S_x_hat"]
            
            _, num_freq, _ = S_x.size()
            S_x = S_x[:, :num_freq//2, :]
            S_x_hat = S_x_hat[:, :num_freq//2, :]

            axs[n_q,0].plot(out[0].detach().cpu().numpy().squeeze())
            axs[n_q,0].set_title(f'n_q={n_q}')
            axs[n_q,0].set_ylim(-2, 2)
            
            if n_q == 1:
                axs[0,4].imshow(S_x.detach().cpu().numpy()[0], cmap='jet', aspect='auto')
                axs[0,4].invert_yaxis()
                axs[0,4].set_title('Original Spectrogram')
                axs[0,5].imshow(S_x.detach().cpu().numpy()[0, :, 10000//50: 11200//50], cmap='jet', aspect='auto')
                axs[0,5].invert_yaxis()
                axs[0,5].set_title("Spectrogram")

            time = np.arange(0, 1200)
            axs[n_q,1].plot(time, out.detach().cpu().numpy().squeeze()[10000:11200])
            #plot original signal with transparent
            axs[n_q,1].plot(time, x[0].detach().cpu().numpy().squeeze()[10000:11200], alpha=0.3)
            axs[n_q,1].set_xlabel("Time")
            axs[n_q,1].set_title(f"Signal n_q={n_q}")
            axs[n_q,1].set_ylim(-2, 2)
            axs[n_q,2].plot(out.detach().cpu().numpy().squeeze()[10000:10100])
            #plot original signal with transparent
            axs[n_q,2].plot(x[0].detach().cpu().numpy().squeeze()[10000:10100], alpha=0.3)
            axs[n_q,2].set_title(f"Signal n_q={n_q}")
            axs[n_q,2].set_ylim(-2, 2)

            axs[n_q,4].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto')
            axs[n_q,4].invert_yaxis()
            axs[n_q,4].set_title(f'Reconstructed Spectrogram, n_q={n_q}, Loss = {l1_loss.item()}')
            axs[n_q,5].imshow(S_x_hat.detach().cpu().numpy()[0, :, 10000//50: 11200//50], cmap='jet', aspect='auto')
            axs[n_q,5].invert_yaxis()
            axs[n_q,5].set_title(f"Spectrogram n_q={n_q}")

            outputs[n_q] = out.detach().cpu().numpy().squeeze()

            del output, l1_loss, S_x, S_x_hat

        #plot the cumulative signal
        for j in range(1, 8):
            a = outputs[j]
            b = outputs[j+1]
            diff = b - a
            time = np.arange(0, 1200)
            axs[j,3].plot(time, diff[10000:11200])
            axs[j,3].set_title(f"Signal n_q={j+1} - Signal n_q={j}")
            axs[j,3].set_ylim(-2, 2)

        #save figure 
        fig.tight_layout()
        # fig.savefig(f'/data/netmit/wifall/breathing_tokenizer/encodec/encodec/tensorboard/{config.exp_details.name}/reconstructed_{epoch}.png')
        print(f'saving to /data/scratch/ellen660/encodec/encodec/visualize_{i}.png')
        fig.savefig(f'/data/scratch/ellen660/encodec/encodec/visualize_{i}.png')
        plt.close(fig)
