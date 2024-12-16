from model import EncodecModel
from my_code.dataset import BreathingDataset
from my_code.losses import loss_fn_l1, loss_fn_l2, total_loss, disc_loss
from my_code.schedulers import LinearWarmupCosineAnnealingLR
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
from my_code.spectrogram_loss import BreathingSpectrogram, ReconstructionLoss

def train_one_step(epoch, optimizer, optimizer_disc, scheduler, disc_scheduler, model, disc, train_loader, config, writer, freq_loss):
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
    disc.train()
    epoch_loss = 0
    for i, (x, _) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch")):
        # print(f'x shape: {x.shape}')
        x = x.to(device)
        x_hat, codes, commit_loss = model(x)
        logits_real, fmap_real = disc(x)
        logits_fake, fmap_fake = disc(x_hat)

        # if model.module.quantizer.codebooks is not None:
        #     print(f'model codebooks: {model.quantizer.codebooks.keys()}')

        commit_loss = torch.mean(commit_loss)
        # loss_l1 = loss_fn_l1(x, x_hat)
        # loss_l2 = loss_fn_l2(x, x_hat)
        freq_loss_dict = freq_loss(x, x_hat)
        losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                x, 
                x_hat, 
                sample_rate=10,
            ) 
        del logits_real, logits_fake, fmap_real, fmap_fake

        loss_f_l1 = freq_loss_dict["l1_loss"] * config.loss.weight_freq
        loss_f_l2 = freq_loss_dict["l2_loss"] * config.loss.weight_freq
        acc = freq_loss_dict["acc"] * config.loss.weight_freq
        loss_f = freq_loss_dict["total_loss"] * config.loss.weight_freq
        # loss_g = losses_g['l_g'] * config.loss.weight_g
        # loss_feat = losses_g['l_feat'] * config.loss.weight_feat

        # #assert all losses have requires grad
        # assert loss_l1.requires_grad == True
        # assert loss_l2.requires_grad == True
        # assert loss_f.requires_grad == True
        # assert losses_g['l_g'].requires_grad == True
        # assert losses_g['l_feat'].requires_grad == True
        # assert losses_g['l_t'].requires_grad == True

        # loss = loss_l1 + loss_l2 + commit_loss + loss_f + loss_g + loss_feat
        loss = losses_g['l_t'] * config.loss.weight_l1 + commit_loss * config.loss.weight_commit + loss_f 
        if config.model.train_discriminator:
            losses_g['l_g'] * config.loss.weight_g + losses_g['l_feat'] * config.loss.weight_feat

        optimizer.zero_grad()
        loss.backward()

        # gradient clipping. restrict the norm of the gradients to be less than 1
        if config.common.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)

        optimizer.step()

        # only update discriminator with probability from paper (configure)
        optimizer_disc.zero_grad()
        # train_discriminator = torch.BoolTensor([config.model.train_discriminator 
        #                        and epoch >= config.lr_scheduler.warmup_epoch 
        #                        and random.random() < float(config.model.train_discriminator_prob)]).cuda()
        train_discriminator = (
            config.model.train_discriminator
            and epoch >= config.lr_scheduler.warmup_epoch
            and random.random() < float(config.model.train_discriminator_prob)
        )

        if train_discriminator:
            print(f'train discriminator')
            logits_real, _ = disc(x)
            logits_fake, _ = disc(x_hat.detach()) # detach to avoid backpropagation to model
            loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss\
            # print(f'loss device: {loss_disc.device}')
            loss_disc.backward() 
            optimizer_disc.step()
            logger(writer, {'Loss Discriminator': loss_disc.item()}, 'train', epoch*len(train_loader) + i)
            epoch_loss += loss_disc.item()
            del logits_real, logits_fake, loss_disc

        epoch_loss += loss.item()

        # add the loss to the tensorboard
        logger(writer, {'Loss per step': loss.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss Frequency': loss_f.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss L1': losses_g['l_t'].item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss L2': losses_g['l_t_2'].item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss commit_loss': commit_loss.item()}, 'train', epoch*len(train_loader) + i)
        # logger(writer, {'Loss commit_loss': torch.mean(commit_loss).item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss Frequency L1': loss_f_l1.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss Frequency L2': loss_f_l2.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Frequency Accuracy': acc.item()}, 'train', epoch*len(train_loader) + i)
        if config.model.train_discriminator:
            logger(writer, {'Loss Generator': losses_g['l_g'].item()}, 'train', epoch*len(train_loader) + i)
            logger(writer, {'Loss Feature': losses_g['l_feat'].item()}, 'train', epoch*len(train_loader) + i)

        max_gradient = torch.tensor(0.0).to(device)
        for param in model.parameters():
            if param.grad is not None:
                max_gradient = max(max_gradient, param.grad.abs().max().item())

        # log the max gradient
        logger(writer, {'Max Gradient': max_gradient}, 'train', epoch*len(train_loader) + i)

    # log the learning rate
    logger(writer, {'Learning Rate': optimizer.param_groups[0]['lr']}, 'train', epoch)
    scheduler.step()
    disc_scheduler.step()

    loss_per_epoch = epoch_loss/len(train_loader)
    print(f"Epoch {epoch}, training loss: {loss_per_epoch}")

    # log the metrics
    if epoch % config.common.log_interval == 0:
        logger(writer, {'Loss': loss_per_epoch}, 'train', epoch)

@torch.no_grad()
def test(epoch, model, disc, val_loader, config, writer, freq_loss):
    model.eval()
    disc.eval()
    epoch_loss = 0
    all_codes = []
    for i, (x, _) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch")):
        x = x.to(device)
        x_hat, codes, commit_loss = model(x)
        logits_real, fmap_real = disc(x)
        logits_fake, fmap_fake = disc(x_hat)

        commit_loss = torch.mean(commit_loss)
        # loss_l1 = loss_fn_l1(x, x_hat)
        # loss_l2 = loss_fn_l2(x, x_hat)
        freq_loss_dict = freq_loss(x, x_hat)
        losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                x, 
                x_hat, 
                sample_rate=10,
            ) 
        loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss

        loss_f_l1 = freq_loss_dict["l1_loss"] * config.loss.weight_freq
        loss_f_l2 = freq_loss_dict["l2_loss"] * config.loss.weight_freq
        acc = freq_loss_dict["acc"] * config.loss.weight_freq
        loss_f = freq_loss_dict["total_loss"] * config.loss.weight_freq
        # loss_g = losses_g['l_g'] * config.loss.weight_g
        # loss_feat = losses_g['l_feat'] * config.loss.weight_feat
        
        # loss = loss_l1 + loss_l2 + commit_loss + loss_f + loss_g + loss_feat + loss_disc
        loss = losses_g['l_t'] * config.loss.weight_l1 + losses_g['l_t_2'] * config.loss.weight_l2 + commit_loss * config.loss.weight_commit + loss_f 
        if config.model.train_discriminator:
            losses_g['l_g'] * config.loss.weight_g + losses_g['l_feat'] * config.loss.weight_feat
            epoch_loss += loss_disc.item()

        epoch_loss += loss.item()

        all_codes.append(codes)
        logger(writer, {'Loss per step': loss.item()}, 'val', epoch*len(train_loader) + i)
        logger(writer, {'Loss Frequency': loss_f.item()}, 'val', epoch*len(train_loader) + i)
        logger(writer, {'Loss L1': losses_g['l_t'].item()}, 'val', epoch*len(train_loader) + i)
        logger(writer, {'Loss L2': losses_g['l_t_2'].item()}, 'val', epoch*len(train_loader) + i)
        # logger(writer, {'Loss commit_loss': torch.mean(commit_loss).item()}, 'val', epoch*len(train_loader) + i)
        logger(writer, {'Loss commit_loss': commit_loss.item()}, 'val', epoch*len(train_loader) + i)
        logger(writer, {'Loss Frequency L1': loss_f_l1.item()}, 'val', epoch*len(train_loader) + i)
        logger(writer, {'Loss Frequency L2': loss_f_l2.item()}, 'val', epoch*len(train_loader) + i)
        logger(writer, {'Frequency Accuracy': acc.item()}, 'val', epoch*len(train_loader) + i)
        if config.model.train_discriminator:
            logger(writer, {'Loss Generator': losses_g['l_g'].item()}, 'val', epoch*len(val_loader) + i)
            logger(writer, {'Loss Feature': losses_g['l_feat'].item()}, 'val', epoch*len(val_loader) + i)
            logger(writer, {'Loss Discriminator': loss_disc.item()}, 'val', epoch*len(val_loader) + i)

        if i == 0:
            S_x = freq_loss_dict["S_x"]
            S_x_hat = freq_loss_dict["S_x_hat"]
            
            _, num_freq, _ = S_x.size()
            S_x = S_x[:, :num_freq//2, :]
            S_x_hat = S_x_hat[:, :num_freq//2, :]

            # plot x and the reconstructed x
            fig, axs = plt.subplots(4, 1, figsize=(20, 10))

            axs[0].plot(x[0].cpu().numpy().squeeze())
            axs[0].set_title('Original')
            axs[1].imshow(S_x.detach().cpu().numpy()[0], cmap='jet', aspect='auto')
            axs[1].invert_yaxis()
            axs[1].set_title('Original Spectrogram')

            axs[2].plot(x_hat[0].cpu().numpy().squeeze())
            axs[2].set_title('Reconstructed')
            axs[3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto')
            axs[3].invert_yaxis()
            axs[3].set_title('Reconstructed Spectrogram')

            fig.tight_layout()
            # fig.savefig(f'/data/netmit/wifall/breathing_tokenizer/encodec/encodec/tensorboard/{config.exp_details.name}/reconstructed_{epoch}.png')
            fig.savefig(f'/data/scratch/ellen660/encodec/encodec/tensorboard/{config.exp_details.name}/{epoch}.png')
            plt.close(fig)

    all_codes = torch.cat(all_codes, dim=0) # B, num_codebooks, T
    all_codes = torch.permute(all_codes, (1, 0, 2))

    # flatten the last two dimensions
    all_codes = all_codes.reshape(all_codes.shape[0], -1)

    # log the distribution of codes. one distribution for each codebook
    for i in range(all_codes.shape[0]):
        writer.add_histogram(f'Codes/Codebook {i}', all_codes[i], epoch)

    # Create a figure
    if epoch == 0:   
        x = x[0].cpu().numpy().squeeze()
        fig, ax = plt.subplots(2, 1)
        time = np.arange(0, 1200)
        ax[0].plot(time, x[10000:11200])
        ax[0].set_title("Signal Graph")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Amplitude")
        ax[1].imshow(S_x.detach().cpu().numpy()[0, :, 10000//50: 11200//50], cmap='jet', aspect='auto')
        ax[1].invert_yaxis()
        ax[1].set_title("Spectrogram")
        fig.tight_layout()

        # Add figure to TensorBoard
        writer.add_figure("Signal/original", fig)
        writer.close()
        plt.close(fig)

    fig, ax = plt.subplots(2, 1)
    x_hat = x_hat[0].cpu().numpy().squeeze()
    time = np.arange(0, 1200)
    ax[0].plot(time, x_hat[10000:11200])
    ax[0].set_title("Signal Graph")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Amplitude")
    ax[1].imshow(S_x_hat.detach().cpu().numpy()[0, :, 10000//50: 11200//50], cmap='jet', aspect='auto')
    ax[1].invert_yaxis()
    ax[1].set_title("Spectrogram")
    fig.tight_layout()
    
    writer.add_figure(f"Signal/{epoch}", fig)
    writer.close()
    plt.close(fig)

    loss_per_epoch = epoch_loss/len(val_loader)
    print(f"Epoch {epoch}, validation loss: {loss_per_epoch}")

    # log the metrics
    logger(writer, {'Loss': loss_per_epoch}, 'val', epoch)

#Logger for tensorboard
def logger(writer, metrics, phase, epoch_index):

    for key, value in metrics.items():
        # if key == 'confusion':
        #     fig = plot_confusion_matrix(metrics[key])
        #     writer.add_figure(f"Confusion Matrix {phase}", fig, epoch_index)
        # else:
        ## check for 2 class multiclass
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
    #dataset
    dataset = BreathingDataset(debug = config.dataset.debug, max_length = config.dataset.max_length)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
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

    # log_dir = os.path.join(f'/data/netmit/wifall/breathing_tokenizer/encodec/encodec/tensorboard', f"{config.exp_details.name}")
    log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/{args.exp_name}/{curr_time}/{curr_min}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Load the YAML file
    config = load_config("encodec/params/%s.yaml" % args.exp_name, log_dir)
    writer = init_logger(log_dir)

    device = torch.device("cuda")
    torch.manual_seed(config.common.seed)

    data_parallel = config.distributed.data_parallel

    train_loader, val_loader = init_dataset(config)
    model, disc = init_model(config)
    model = model.to(device)
    disc = disc.to(device)

    if data_parallel:
        model = nn.DataParallel(model)
        disc = nn.DataParallel(disc)
    
    optimizer = optim.AdamW(model.parameters(), lr=float(config.optimization.lr), betas=(0.8, 0.9))
    optimizer_disc = optim.AdamW(disc.parameters(), lr=float(config.optimization.disc_lr), betas=(0.8, 0.9))

    # scheduler = WarmupCosineLrScheduler(optimizer, max_iter=config.common.max_epoch*len(trainloader), eta_ratio=0.1, warmup_iter=config.lr_scheduler.warmup_epoch*len(trainloader), warmup_ratio=1e-4)

    # cosine annealing scheduler
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.lr_scheduler.warmup_epoch, max_epochs=config.common.max_epoch)
    disc_scheduler = LinearWarmupCosineAnnealingLR(optimizer_disc, warmup_epochs=config.lr_scheduler.warmup_epoch, max_epochs=config.common.max_epoch)

    #Reconstruction loss
    freq_loss = ReconstructionLoss(alpha=config.loss.alpha, bandwidth=config.loss.bandwidth, sampling_rate=10, n_fft=config.loss.n_fft, device=device)

    # instantiate loss balancer
    # balancer = Balancer(config.balancer.weights)
    # if balancer:
    #     print(f'Loss balancer with weights {balancer.weights} instantiated')
    test(0, model, disc, val_loader, config, writer, freq_loss=freq_loss)
    for epoch in tqdm(range(1, config.common.max_epoch+1), desc="Epochs", unit="epoch"):
        # train_one_step(epoch,optimizer, model, train_loader,config,scaler=None,scaler_disc=None,writer=None,balancer=None):
        train_one_step(epoch, optimizer, optimizer_disc, scheduler, disc_scheduler, model, disc, train_loader, config=config, writer=writer, freq_loss=freq_loss)
        if epoch % config.common.test_interval == 0:
            test(epoch,model,disc, val_loader,config,writer, freq_loss=freq_loss)
        # save checkpoint and epoch
        # if epoch % config.common.save_interval == 0:
        #     model_to_save = model.module if config.distributed.data_parallel else model
        #     # disc_model_to_save = disc_model.module if config.distributed.data_parallel else disc_model 
        #     if not config.distributed.data_parallel or dist.get_rank() == 0:  
        #         save_master_checkpoint(epoch, model_to_save, optimizer, scheduler, f'{config.checkpoint.save_location}epoch{epoch}_lr{config.optimization.lr}.pt')  
        #         save_master_checkpoint(epoch, disc_model_to_save, optimizer_disc, disc_scheduler, f'{config.checkpoint.save_location}epoch{epoch}_disc_lr{config.optimization.lr}.pt') 
