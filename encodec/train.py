from model import EncodecModel
from data import MergedDataset
from data.dataset import BreathingDataset
from my_code.losses import loss_fn_l1, loss_fn_l2, total_loss, disc_loss
from my_code.metrics import Metrics, MetricsArgs
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
from my_code.spectrogram_loss import ReconstructionLoss, ReconstructionLosses

import socket

def train_one_step(metrics, epoch, optimizer, optimizer_disc, scheduler, disc_scheduler, model, disc, train_loader, config, writer, freq_loss):
    """train one step function

    Args:
        epoch (int): current epoch
        optimizer (_type_) : generator optimizer
        optimizer_disc (_type_): discriminator optimizer
        model (_type_): generator model
        disc_model (_type_): discriminator model
        trainloader (_type_): train dataloader
        config (_type_): hydra config file
        scheduler (_type_): adjust generate model learning rate
        disc_scheduler (_type_): adjust discriminator model learning rate
        freq_loss: freq loss on spectrogram
        warmup_scheduler (_type_): warmup learning rate
    """
    model.train()

    if config.model.train_discriminator and epoch >= config.model.train_discriminator_start_epoch:
        disc.train()

    epoch_loss = 0
    for i, item in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch")):
        x = item["x"]
        if x is None:
            continue
        # print(f'x shape: {x.shape}')
        x = x.to(device)
        x_hat, _, commit_loss = model(x)

        train_generator = (
            config.model.train_discriminator
            and epoch >= config.model.train_discriminator_start_epoch
        )

        offset_prob = 1. - float(config.model.train_discriminator_prob) if epoch - config.model.train_discriminator_start_epoch < 10 else 0.0
        train_discriminator = (
            config.model.train_discriminator
            and epoch >= config.model.train_discriminator_start_epoch
            and random.random() < float(config.model.train_discriminator_prob) + offset_prob
        )

        if train_generator and not train_discriminator:
            logits_real, fmap_real = disc(x)
            logits_fake, fmap_fake = disc(x_hat)
        else:
            logits_real, logits_fake, fmap_real, fmap_fake = None, None, None, None

        commit_loss = torch.mean(commit_loss)
        freq_loss_dict = freq_loss(x, x_hat)
        losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                x, 
                x_hat, 
                sample_rate=10,
            ) 
        
        loss_f_l1 = freq_loss_dict["l1_loss"] * config.loss.weight_freq
        loss_f_l2 = freq_loss_dict["l2_loss"] * config.loss.weight_freq
        acc = freq_loss_dict["acc"]
        loss_f = freq_loss_dict["total_loss"] * config.loss.weight_freq

        loss = losses_g['l_t'] * config.loss.weight_l1 + commit_loss * config.loss.weight_commit + loss_f 
        
        if train_generator and not train_discriminator:
            loss += losses_g['l_g'] * config.loss.weight_g + losses_g['l_feat'] * config.loss.weight_feat

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping. restrict the norm of the gradients to be less than 1
        if config.common.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        # only update discriminator with probability from paper (configure)
        if train_discriminator:
<<<<<<< HEAD
            # if random.random() < float(config.model.train_discriminator_prob):
            # print('train discriminator')
=======
>>>>>>> 0ebf6e2d1c7984951189f97b5f32d614ae5946d8
            optimizer_disc.zero_grad()

            logits_real, _ = disc(x)
            logits_fake, _ = disc(x_hat.detach()) # detach to avoid backpropagation to model
            loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss\
            loss_disc.backward() 

            if config.common.gradient_clipping:
                nn.utils.clip_grad_norm_(disc.parameters(), 0.1)

            optimizer_disc.step()
            metrics.fill_metrics({'Loss Discriminator': loss_disc.item()}, epoch*(train_loader) + i)
            # logger(writer, {'Loss Discriminator': loss_disc.item()}, 'train', epoch*len(train_loader) + i)
            epoch_loss += loss_disc.item()

            max_disc_gradient = torch.tensor(0.0).to(device)
            for param in disc.parameters():
                if param.grad is not None:
                    max_disc_gradient = max(max_disc_gradient, param.grad.abs().max().item())
            metrics.fill_metrics({'Max Discriminator Gradient': max_disc_gradient}, epoch*(train_loader) + i)
            # logger(writer, {'Max Discriminator Gradient': max_disc_gradient}, 'train', epoch*len(train_loader) + i)

        epoch_loss += loss.item()

        # add the loss to the tensorboard
        metrics.fill_metrics({
            'Loss per step': loss.item(),
            'Loss Frequency': loss_f.item(),
            'Loss L1': losses_g['l_t'].item(),
            'Loss L2': losses_g['l_t_2'].item(),
            'Loss commit_loss': commit_loss.item(),
            'Loss Frequency L1': loss_f_l1.item(),
            'Loss Frequency L2': loss_f_l2.item(),
            'Frequency Accuracy': acc.item()
        }, epoch*len(train_loader) + i)
        # logger(writer, {'Loss per step': loss.item()}, 'train', epoch*len(train_loader) + i)
        # logger(writer, {'Loss Frequency': loss_f.item()}, 'train', epoch*len(train_loader) + i)
        # logger(writer, {'Loss L1': losses_g['l_t'].item()}, 'train', epoch*len(train_loader) + i)
        # logger(writer, {'Loss L2': losses_g['l_t_2'].item()}, 'train', epoch*len(train_loader) + i)
        # logger(writer, {'Loss commit_loss': commit_loss.item()}, 'train', epoch*len(train_loader) + i)
        # logger(writer, {'Loss Frequency L1': loss_f_l1.item()}, 'train', epoch*len(train_loader) + i)
        # logger(writer, {'Loss Frequency L2': loss_f_l2.item()}, 'train', epoch*len(train_loader) + i)
        # logger(writer, {'Frequency Accuracy': acc.item()}, 'train', epoch*len(train_loader) + i)
        
        if train_generator and not train_discriminator:
            metrics.fill_metrics({
                'Loss Generator': losses_g['l_g'].item(),\
                'Loss Feature': losses_g['l_feat'].item()
            },epoch*len(train_loader) + i)
            # logger(writer, {'Loss Generator': losses_g['l_g'].item()}, 'train', epoch*len(train_loader) + i)
            # logger(writer, {'Loss Feature': losses_g['l_feat'].item()}, 'train', epoch*len(train_loader) + i)

        max_gradient = torch.tensor(0.0).to(device)
        for param in model.parameters():
            if param.grad is not None:
                max_gradient = max(max_gradient, param.grad.abs().max().item())

        # log the max gradient
        metrics.fill_metrics({
            'Max Gradient': max_gradient
        }, epoch*len(train_loader) + i)
        # logger(writer, {'Max Gradient': max_gradient}, 'train', epoch*len(train_loader) + i)

    metrics_dict = metrics.compute_and_log_metrics()
    # log the learning rate
    metrics_dict['Learning Rate'] = optimizer.param_groups[0]['lr']
    # logger(writer, {'Learning Rate': optimizer.param_groups[0]['lr']}, 'train', epoch)
    scheduler.step()  

    if train_discriminator:
        disc_scheduler.step()

    loss_per_epoch = epoch_loss/len(train_loader)
    print(f"Epoch {epoch}, training loss: {loss_per_epoch}")

    # log the metrics
    metrics_dict['Loss'] = loss_per_epoch
    # logger(writer, {'Loss': loss_per_epoch}, 'train', epoch)
    logger(writer, metrics_dict, 'train', epoch)
    metrics.clear_metrics()

@torch.no_grad()
def test(metrics, epoch, model, disc, val_loader, config, writer, freq_loss):
    model.eval()
    train_discriminator = (
        config.model.train_discriminator
        and epoch >= config.model.train_discriminator_start_epoch
    )
    if train_discriminator:
        disc.eval()
    epoch_loss = 0
    all_codes = []
    for i, item in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch")):
        x = item["x"]
        if x is None:
            continue
        x = x.to(device)
        x_hat, codes, commit_loss = model(x)

        if train_discriminator:
            logits_real, fmap_real = disc(x)
            logits_fake, fmap_fake = disc(x_hat)
        else:
            logits_real, logits_fake, fmap_real, fmap_fake = None, None, None, None

        commit_loss = torch.mean(commit_loss)
        freq_loss_dict = freq_loss(x, x_hat)
        losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                x, 
                x_hat, 
                sample_rate=10,
            )
        if train_discriminator:
            loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss

        loss_f_l1 = freq_loss_dict["l1_loss"] * config.loss.weight_freq
        loss_f_l2 = freq_loss_dict["l2_loss"] * config.loss.weight_freq
        acc = freq_loss_dict["acc"] * config.loss.weight_freq
        loss_f = freq_loss_dict["total_loss"] * config.loss.weight_freq

        loss = losses_g['l_t'] * config.loss.weight_l1 + losses_g['l_t_2'] * config.loss.weight_l2 + commit_loss * config.loss.weight_commit + loss_f 
        
        if train_discriminator:
            loss += losses_g['l_g'] * config.loss.weight_g + losses_g['l_feat'] * config.loss.weight_feat
            epoch_loss += loss_disc.item()

        epoch_loss += loss.item()

        all_codes.append(codes)
        metrics.fill_metrics({
            'Loss per step': loss.item(),
            'Loss Frequency': loss_f.item(),
            'Loss L1': losses_g['l_t'].item(),
            'Loss L2': losses_g['l_t_2'].item(),
            'Loss commit_loss': commit_loss.item(),
            'Loss Frequency L1': loss_f_l1.item(),
            'Loss Frequency L2': loss_f_l2.item(),
            'Frequency Accuracy': acc.item()
        }, epoch*len(val_loader) + i)
        # logger(writer, {'Loss per step': loss.item()}, 'val', epoch*len(train_loader) + i)
        # logger(writer, {'Loss Frequency': loss_f.item()}, 'val', epoch*len(train_loader) + i)
        # logger(writer, {'Loss L1': losses_g['l_t'].item()}, 'val', epoch*len(train_loader) + i)
        # logger(writer, {'Loss L2': losses_g['l_t_2'].item()}, 'val', epoch*len(train_loader) + i)
        # logger(writer, {'Loss commit_loss': commit_loss.item()}, 'val', epoch*len(train_loader) + i)
        # logger(writer, {'Loss Frequency L1': loss_f_l1.item()}, 'val', epoch*len(train_loader) + i)
        # logger(writer, {'Loss Frequency L2': loss_f_l2.item()}, 'val', epoch*len(train_loader) + i)
        # logger(writer, {'Frequency Accuracy': acc.item()}, 'val', epoch*len(train_loader) + i)
        if train_discriminator:
            metrics.fill_metrics({
                'Loss Generator': losses_g['l_g'].item(),
                'Loss Feature': losses_g['l_feat'].item(),
                'Loss Discriminator': loss_disc.item()
            }, epoch*len(val_loader) + i)
            # logger(writer, {'Loss Generator': losses_g['l_g'].item()}, 'val', epoch*len(val_loader) + i)
            # logger(writer, {'Loss Feature': losses_g['l_feat'].item()}, 'val', epoch*len(val_loader) + i)
            # logger(writer, {'Loss Discriminator': loss_disc.item()}, 'val', epoch*len(val_loader) + i)

        if i == 0:
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
            fig, axs = plt.subplots(4, 1, figsize=(20, 10), sharex=True)

            axs[0].plot(x_time, x[0].cpu().numpy().squeeze())
            axs[0].set_title('Original')
            axs[0].set_ylim(-4, 4)
            axs[1].imshow(S_x.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
            axs[1].invert_yaxis()
            axs[1].set_title('Original Spectrogram')

            axs[2].plot(x_time, x_hat[0].cpu().numpy().squeeze())
            axs[2].set_title('Reconstructed')
            axs[2].set_ylim(-4, 4)
            axs[3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
            axs[3].invert_yaxis()
            axs[3].set_title('Reconstructed Spectrogram')

            fig.tight_layout()
            if user_name == 'ellen660':
                fig.savefig(f'/data/scratch/ellen660/encodec/encodec/tensorboard/{config.exp_details.name}/{epoch}.png')
            elif user_name == 'chaoli':
                fig.savefig(f'/data/netmit/wifall/breathing_tokenizer/encodec/encodec/tensorboard/{config.exp_details.name}/reconstructed_{epoch}.png')
            else:
                raise Exception("User not recognized")
            plt.close(fig)

    all_codes = torch.cat(all_codes, dim=0) # B, num_codebooks, T
    all_codes = torch.permute(all_codes, (1, 0, 2))

    # flatten the last two dimensions
    all_codes = all_codes.reshape(all_codes.shape[0], -1)

    # log the distribution of codes. one distribution for each codebook
    for i in range(all_codes.shape[0]):
        writer.add_histogram(f'Codes/Codebook {i}', all_codes[i], epoch)

    # # Create a signal of 2 minutes 
    # if epoch == 0:   
    #     x = x[0].cpu().numpy().squeeze()
    #     fig, ax = plt.subplots(2, 1)
    #     time = np.arange(0, 1200)
    #     ax[0].plot(time, x[10000:11200])
    #     ax[0].set_title("Signal Graph")
    #     ax[0].set_xlabel("Time")
    #     ax[0].set_ylabel("Amplitude")
    #     ax[0].set_ylim(-4, 4)
    #     ax[1].imshow(S_x.detach().cpu().numpy()[0, :, 10000//50: 11200//50], cmap='jet', aspect='auto')
    #     ax[1].invert_yaxis()
    #     ax[1].set_title("Spectrogram")
    #     fig.tight_layout()

    #     # Add figure to TensorBoard
    #     writer.add_figure("Signal/original", fig)
    #     writer.close()
    #     plt.close(fig)

    # fig, ax = plt.subplots(2, 1)
    # x_hat = x_hat[0].cpu().numpy().squeeze()
    # time = np.arange(0, 1200)
    # ax[0].plot(time, x_hat[10000:11200])
    # ax[0].set_title("Signal Graph")
    # ax[0].set_xlabel("Time")
    # ax[0].set_ylabel("Amplitude")
    # ax[0].set_ylim(-4, 4)
    # ax[1].imshow(S_x_hat.detach().cpu().numpy()[0, :, 10000//50: 11200//50], cmap='jet', aspect='auto')
    # ax[1].invert_yaxis()
    # ax[1].set_title("Spectrogram")
    # fig.tight_layout()
    
    # writer.add_figure(f"Signal/{epoch}", fig)
    # writer.close()
    # plt.close(fig)

    loss_per_epoch = epoch_loss/len(val_loader)
    print(f"Epoch {epoch}, validation loss: {loss_per_epoch}")

    # log the metrics
    metrics_dict = metrics.compute_and_log_metrics()
    metrics_dict['Loss'] = loss_per_epoch
    logger(writer, metrics_dict, 'val', epoch)
    metrics.clear_metrics()
    # logger(writer, {'Loss': loss_per_epoch}, 'val', epoch)

#Logger for tensorboard
def logger(writer, metrics, phase, epoch_index):

    for key, value in metrics.items():

        if type(value)!= float and len(value.shape) > 0 and value.shape[0] == 2:
            value = value[1]
        elif type(value)!= float and len(value.shape) > 0 and value.shape[0] > 2:
            raise Exception("Need to handle multiclass")
            # bp()
        writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)
    writer.flush()

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
        if log_dir:
            #save yaml file to log_dir
            with open(f"{log_dir}/config.yaml", "w") as file:
                yaml.dump(config_dict, file)
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

    # merge the datasets
    train_dataset = MergedDataset(train_datasets, weight_list, 1, debug = config.dataset.debug)
    val_dataset = MergedDataset(val_datasets, weight_list, 0.2, debug = config.dataset.debug)

    train_loader = DataLoader(train_dataset, batch_size=config.dataset.batch_size, shuffle=True, num_workers=config.dataset.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.dataset.batch_size, shuffle=False, num_workers=config.dataset.num_workers)

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
    print(disc_model)
    # breakpoint()
    print(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")
    print(f"disc model train mode :{disc_model.training}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total number of parameters: {total_params}")
    total_params = sum(p.numel() for p in disc_model.parameters())
    print(f"Discriminator Total number of parameters: {total_params}")
    return model, disc_model

def set_args():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--exp_name", type=str, default="config")
    parser.add_argument("--exp_name", type=str, default="091224_l1")
    
    return parser.parse_args()

if __name__ == "__main__":

    args = set_args()
    curr_time = datetime.now().strftime("%Y%m%d")
    curr_min = datetime.now().strftime("%H%M%S")

    user_name = os.getlogin()

    if user_name == 'ellen660':
        log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/{args.exp_name}/{curr_time}/{curr_min}'
    elif user_name == 'chaoli':
        log_dir = os.path.join(f'/data/netmit/wifall/breathing_tokenizer/encodec/encodec/tensorboard', args.exp_name)
    else:
        raise Exception("User not recognized")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Load the YAML file
    config = load_config("encodec/params/%s.yaml" % args.exp_name, log_dir)
    writer = init_logger(log_dir)

    device = torch.device("cuda")
    torch.manual_seed(config.common.seed)

    metrics_args = MetricsArgs(num_datasets=1, device=device)
    metrics = Metrics(metrics_args)

    data_parallel = config.distributed.data_parallel

    train_loader, val_loader = init_dataset(config)
    # model, disc = init_model(config)
    model, disc = init_model(config)
    model = model.to(device)

    if config.model.train_discriminator:
        disc = disc.to(device)
    else:
        disc = None

    if data_parallel:
        model = nn.DataParallel(model)
        disc = nn.DataParallel(disc)
    
    optimizer = optim.AdamW(model.parameters(), lr=float(config.optimization.lr), betas=(0.8, 0.9))
    if config.model.train_discriminator:
        optimizer_disc = optim.AdamW(disc.parameters(), lr=float(config.optimization.disc_lr), betas=(0.8, 0.9))
    else:
        optimizer_disc = None

    # scheduler = WarmupCosineLrScheduler(optimizer, max_iter=config.common.max_epoch*len(trainloader), eta_ratio=0.1, warmup_iter=config.lr_scheduler.warmup_epoch*len(trainloader), warmup_ratio=1e-4)

    # cosine annealing scheduler
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.lr_scheduler.warmup_epoch, max_epochs=config.common.max_epoch)

    if config.model.train_discriminator:
        disc_scheduler = LinearWarmupCosineAnnealingLR(optimizer_disc, warmup_epochs=config.lr_scheduler.warmup_epoch, max_epochs=config.common.max_epoch)
    else:
        disc_scheduler = None

    #Reconstruction loss
    # freq_loss = ReconstructionLoss(alpha=config.loss.alpha, bandwidth=config.loss.bandwidth, sampling_rate=10, n_fft=config.loss.n_fft, device=device)
    freq_loss = ReconstructionLosses(alpha=config.loss.alpha, bandwidth=config.loss.bandwidth, sampling_rate=10, n_fft=config.loss.n_fft, hop_length=config.loss.hop_length, win_length=config.loss.win_length, device=device)

    # instantiate loss balancer
    # balancer = Balancer(config.balancer.weights)
    # if balancer:
    #     print(f'Loss balancer with weights {balancer.weights} instantiated')
    test(metrics, 0, model, disc, val_loader, config, writer, freq_loss=freq_loss)
    for epoch in tqdm(range(1, config.common.max_epoch+1), desc="Epochs", unit="epoch"):
        train_one_step(metrics, epoch, optimizer, optimizer_disc, scheduler, disc_scheduler, model, disc, train_loader, config=config, writer=writer, freq_loss=freq_loss,)
        if epoch % config.common.test_interval == 0:
            test(metrics, epoch,model,disc, val_loader,config,writer, freq_loss=freq_loss)
        # save checkpoint and epoch
        if epoch % config.checkpoint.save_every == 0:
            if config.distributed.data_parallel:
                torch.save(model.module.state_dict(), f"{log_dir}/model.pth")
                if config.model.train_discriminator:
                    torch.save(disc.module.state_dict(), f"{log_dir}/disc.pth")
            else:
                torch.save(model.state_dict(), f"{log_dir}/model.pth")
                if config.model.train_discriminator:
                    torch.save(disc.state_dict(), f"{log_dir}/disc.pth")
