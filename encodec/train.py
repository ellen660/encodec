from model import EncodecModel
from my_code.dataset import BreathingDataset
from my_code.losses import loss_fn_l1, loss_fn_l2
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

import sys
from my_code.spectrogram_loss import BreathingSpectrogram, ReconstructionLoss

def train_one_step(epoch, optimizer, scheduler, model, train_loader,config,freq_loss, scaler=None,scaler_disc=None,writer=None,balancer=None):
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
        warmup_scheduler (_type_): warmup learning rate
    """
    model.train()
    epoch_loss = 0
    for i, (x, y) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch")):
        # print(f'x shape: {x.shape}')
        x = x.to(device)
        x_hat, codes, commit_loss = model(x)
        loss_l1 = loss_fn_l1(x, x_hat)
        loss_l2 = loss_fn_l2(x, x_hat)
        loss_f = freq_loss(x, x_hat)
        # print(f'loss_f: {loss_f}')
        loss = config.loss.weight_l1 * loss_l1 + config.loss.weight_l2 * loss_l2 + config.loss.weight_commit * torch.mean(commit_loss) + config.loss.weight_freq * loss_f 

        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add the loss to the tensorboard
        logger(writer, {'Loss per step': loss.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss Frequency': loss_f.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss L1': loss_l1.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss commit_loss': torch.mean(commit_loss).item()}, 'train', epoch*len(train_loader) + i)
        # logger(writer, {'Loss L2': loss_l2.item()}, 'train', epoch*len(train_loader) + i)

        # log all the losses
        logger(writer, {'L1 Loss': loss_l1.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'L2 Loss': loss_l2.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Commitment Loss': commit_loss.item()}, 'train', epoch*len(train_loader) + i)

        max_gradient = torch.tensor(0.0).to(device)
        for param in model.parameters():
            if param.grad is not None:
                max_gradient = max(max_gradient, param.grad.abs().max().item())

        # log the max gradient
        logger(writer, {'Max Gradient': max_gradient}, 'train', epoch*len(train_loader) + i)

    # log the learning rate
    logger(writer, {'Learning Rate': optimizer.param_groups[0]['lr']}, 'train', epoch)
    scheduler.step()

    # NOTE: model maps all input to the same value
    # print(x)
    # print(x_hat)
    # print(x.mean(), x_hat.mean())
    # print(x.std(), x_hat.std())
    # sys.exit()

    loss_per_epoch = epoch_loss/len(train_loader)
    print(f"Epoch {epoch}, training loss: {loss_per_epoch}")

    # log the metrics
    if epoch % config.common.log_interval == 0:
        logger(writer, {'Loss': loss_per_epoch}, 'train', epoch)

@torch.no_grad()
def test(epoch, model, val_loader, config, writer, freq_loss):
    model.eval()
    epoch_loss = 0
    all_codes = []
    for i, (x, y) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch")):
        x = x.to(device)
        x_hat, codes, commit_loss = model(x)
        loss_l1 = loss_fn_l1(x, x_hat)
        loss_l2 = loss_fn_l2(x, x_hat)
        loss_f = freq_loss(x, x_hat)
        # print(f'loss_f: {loss_f}')
        loss = config.loss.weight_l1 * loss_l1 + config.loss.weight_l2 * loss_l2 + config.loss.weight_commit * torch.mean(commit_loss) + config.loss.weight_freq * loss_f
        epoch_loss += loss.item()

        all_codes.append(codes)
        logger(writer, {'Loss per step': loss.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss Frequency': loss_f.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss L1': loss_l1.item()}, 'train', epoch*len(train_loader) + i)
        logger(writer, {'Loss commit_loss': torch.mean(commit_loss).item()}, 'train', epoch*len(train_loader) + i)

    all_codes = torch.cat(all_codes, dim=0)
    # log the distribution of codes
    writer.add_histogram('Codes', all_codes, epoch)

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
def load_config(filepath):
    #make directory
    with open(filepath, "r") as file:
        config_dict = yaml.safe_load(file)
    return ConfigNamespace(config_dict)

def init_logger(log_dir):
    # comment = f'feature_size_{feature_dim}_fc1_size_{fc1_size}_num_layers_{num_layers_vit}_num_heads_{num_heads}'
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
    # logger.info(disc_model)
    # logger.info(config)
    # logger.info(f"Encodec Model Parameters: {count_parameters(model)}")
    print(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = nn.DataParallel(model)
    #     model = model.to(device)
    return model

def set_args():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument("--exp_name", type=str, default="config")
    parser.add_argument("--exp_name", type=str, default="091224_l1")
    
    return parser.parse_args()


if __name__ == "__main__":

    args = set_args()

    # Load the YAML file
    config = load_config("encodec/params/%s.yaml" % args.exp_name)

    log_dir = os.path.join(f'/data/netmit/wifall/breathing_tokenizer/encodec/encodec/tensorboard', f"{config.exp_details.name}")
    # log_dir = f'/data/scratch/ellen660/encodec/encodec/tensorboard/{config.exp_details.name}/{config.exp_details.date}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    writer = init_logger(log_dir)

    device = torch.device("cuda")
    torch.manual_seed(config.common.seed)

    data_parallel = config.distributed.data_parallel

    train_loader, val_loader = init_dataset(config)
    model = init_model(config)
    model = model.to(device)

    if data_parallel:
        model = nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=float(config.optimization.lr), weight_decay=float(config.optimization.weight_decay))
    # scheduler = WarmupCosineLrScheduler(optimizer, max_iter=config.common.max_epoch*len(trainloader), eta_ratio=0.1, warmup_iter=config.lr_scheduler.warmup_epoch*len(trainloader), warmup_ratio=1e-4)

    # cosine annealing scheduler
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.lr_scheduler.warmup_epoch, max_epochs=config.common.max_epoch)

    #Reconstruction loss
    freq_loss = ReconstructionLoss(sampling_rate=10, n_fft=64, device=device)

    # instantiate loss balancer
    # balancer = Balancer(config.balancer.weights)
    # if balancer:
    #     print(f'Loss balancer with weights {balancer.weights} instantiated')
    # test(0, model, val_loader, config, writer, freq_loss=freq_loss)
    for epoch in tqdm(range(1, config.common.max_epoch+1), desc="Epochs", unit="epoch"):
        # train_one_step(epoch,optimizer, model, train_loader,config,scaler=None,scaler_disc=None,writer=None,balancer=None):
        train_one_step(epoch, optimizer, scheduler, model, train_loader, config=config,writer=writer, freq_loss=freq_loss)
        if epoch % config.common.test_interval == 0:
            test(epoch,model,val_loader,config,writer, freq_loss=freq_loss)
        # save checkpoint and epoch
        # if epoch % config.common.save_interval == 0:
        #     model_to_save = model.module if config.distributed.data_parallel else model
        #     # disc_model_to_save = disc_model.module if config.distributed.data_parallel else disc_model 
        #     if not config.distributed.data_parallel or dist.get_rank() == 0:  
        #         save_master_checkpoint(epoch, model_to_save, optimizer, scheduler, f'{config.checkpoint.save_location}epoch{epoch}_lr{config.optimization.lr}.pt')  
        #         save_master_checkpoint(epoch, disc_model_to_save, optimizer_disc, disc_scheduler, f'{config.checkpoint.save_location}epoch{epoch}_disc_lr{config.optimization.lr}.pt') 
