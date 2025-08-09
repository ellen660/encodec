from clean_model import EncodecModel
# from ppg import init_dataset
from baseline_data import init_dataset
from losses import total_loss, disc_loss, Metrics, MetricsArgs, LinearWarmupCosineAnnealingLR, WarmupScheduler, ReconstructionLoss
from msstftd import MultiScaleSTFTDiscriminator

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import yaml
import random
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import time
from utils import set_random_seed, print_model_details
import jsonschema
import json
from typing import Tuple, Optional


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
    if config.discrim.train_discriminator and epoch >= config.discrim.train_discriminator_start_epoch:
        disc.train()
    epoch_loss = 0
    start_data_time = time.time()

    for i, (item, ds_id) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch")):
        x = item["x"]
        data_loading_time = time.time() - start_data_time
        
        to_device_time = time.time()
        x = x.to(device)
        to_device_time = time.time() - to_device_time
        
        start_forward_time = time.time()
        x_hat, _, commit_loss, codebook_loss = model(x)

        train_generator = (
            config.discrim.train_discriminator
            and epoch >= config.discrim.train_discriminator_start_epoch
        )

        # offset_prob = 1. - float(config.model.train_discriminator_prob) if epoch - config.model.train_discriminator_start_epoch < config.model.train_discriminator_for else 0.0
        train_discriminator = (
            config.discrim.train_discriminator
            and epoch >= config.discrim.train_discriminator_start_epoch
            and random.random() < float(config.discrim.train_discriminator_prob) #+ offset_prob
        )

        if train_generator and not train_discriminator:
            logits_real, fmap_real = disc(x)
            logits_fake, fmap_fake = disc(x_hat)
        else:
            logits_real, logits_fake, fmap_real, fmap_fake = None, None, None, None

        commit_loss = torch.mean(commit_loss)
        codebook_loss = torch.mean(codebook_loss)
        freq_loss_dict = freq_loss(x, x_hat)
        losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                x, 
                x_hat, 
            ) 
        loss = losses_g['l_1'] * config.loss.weight_l1 + freq_loss_dict["total_loss"] * config.loss.weight_freq + losses_g['l_2'] * config.loss.weight_l2
        if epoch >= config.loss.commit_start_epoch:
            loss += commit_loss * config.loss.weight_commit 
        if train_generator and not train_discriminator:
            loss += losses_g['l_g'] * config.loss.weight_g + losses_g['l_feat'] * config.loss.weight_feat

        optimizer.zero_grad() #optimizer is the model only, so only updating those parameters
        loss.backward()
        # gradient clipping. restrict the norm of the gradients to be less than 1
        if config.common.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), config.common.gradient_clipping_value)
        optimizer.step()
        
        forward_time = time.time() - start_forward_time
        # tqdm.write(f"Batch {i}: Data loading time: {data_loading_time:.4f}s, To device time: {to_device_time:.4f}s, Forward pass time: {forward_time:.4f}s")
        start_data_time = time.time()

        if train_discriminator:
            logits_real, _ = disc(x)
            logits_fake, _ = disc(x_hat.detach()) # detach to avoid backpropagation to model
            loss_disc = disc_loss(logits_real, logits_fake) # compute discriminator loss

            optimizer_disc.zero_grad()
            loss_disc.backward() 
            if config.common.gradient_clipping:
                nn.utils.clip_grad_norm_(disc.parameters(), config.common.gradient_clipping_value)
            optimizer_disc.step()

            # breakpoint()
            if epoch % config.common.log_every == 0:
                metrics.fill_metrics({'Loss Discriminator': loss_disc.item()}, epoch*len(train_loader) + i)
                metrics.fill_metrics({'Logits Real': (torch.mean(logits_real[0]).item() + torch.mean(logits_real[1]).item())/2}, epoch*len(train_loader) + i)
                metrics.fill_metrics({'Logits Fake': (torch.mean(logits_fake[0]).item() + torch.mean(logits_fake[1]).item())/2}, epoch*len(train_loader) + i)
                epoch_loss += loss_disc.item()

                max_disc_gradient = torch.tensor(0.0).to(device)
                for param in disc.parameters():
                    if param.grad is not None:
                        max_disc_gradient = max(max_disc_gradient, param.grad.abs().max().item())
                metrics.fill_metrics({'Max Discriminator Gradient': max_disc_gradient}, epoch*len(train_loader) + i)

        if epoch % config.common.log_every == 0: # add the loss to the tensorboard
            epoch_loss += loss.item()
            metrics.fill_metrics({
                'Loss L1': losses_g['l_1'].item(),
                'Loss commit_loss': commit_loss.item(),
                'Loss Frequency L1': freq_loss_dict["l1_loss"].item(),
                'Frequency Accuracy': freq_loss_dict["acc"].item(),
            }, epoch*len(train_loader) + i)
            for j, d_id in enumerate(ds_id):
                dataset_id = d_id
                metrics.fill_metrics({f'Loss L1 {dataset_id}': losses_g['l_t'][j].item()}, epoch*len(train_loader) + i)
        
            if train_generator and not train_discriminator:
                metrics.fill_metrics({
                    'Loss Generator': losses_g['l_g'].item(),
                    'Loss Feature': losses_g['l_feat'].item()
                },epoch*len(train_loader) + i)

            max_gradient = torch.tensor(0.0).to(device)
            for param in model.parameters():
                if param.grad is not None:
                    max_gradient = max(max_gradient, param.grad.abs().max().item())

            # log the max gradient
            metrics.fill_metrics({
                'Max Gradient': max_gradient
            }, epoch*len(train_loader) + i)

    scheduler.step()  
    if config.discrim.train_discriminator and epoch >= config.discrim.train_discriminator_start_epoch:
        disc_scheduler.step()

    if epoch % config.common.log_every == 0:
        metrics_dict = metrics.compute_and_log_metrics()
        # log the learning rate
        metrics_dict['Learning Rate'] = optimizer.param_groups[0]['lr']
        loss_per_epoch = epoch_loss/len(train_loader)
        print(f"Epoch {epoch}, training loss: {loss_per_epoch}")

        # log the metrics
        logger(writer, metrics_dict, 'train', epoch)
        metrics.clear_metrics()
        
@torch.no_grad()
def test(metrics, epoch, model, disc, val_loader, config, writer, freq_loss, log_dir):
    model.eval()
    train_discriminator = (
        config.discrim.train_discriminator
        and epoch >= config.discrim.train_discriminator_start_epoch
    )
    if train_discriminator:
        disc.eval()
    epoch_loss = 0
    all_codes = []
    for i, (item, ds_id) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch")):
        x = item["x"]
        if x is None:
            continue
        x = x.to(device)

        x_hat, codes, commit_loss, codebook_loss = model(x)

        # if train_discriminator:
        #     logits_real, fmap_real = disc(x)
        #     logits_fake, fmap_fake = disc(x_hat)
        # else:
        #     logits_real, logits_fake, fmap_real, fmap_fake = None, None, None, None

        commit_loss = torch.mean(commit_loss)
        codebook_loss = torch.mean(codebook_loss)
        freq_loss_dict = freq_loss(x, x_hat)
        # losses_g = total_loss(
        #         fmap_real, 
        #         logits_fake, 
        #         fmap_fake, 
        #         x, 
        #         x_hat, 
        #     )
        # if train_discriminator:
        #     loss_disc = disc_loss(logits_real, logits_fake) 

        # loss = losses_g['l_1'] * config.loss.weight_l1 + freq_loss_dict["total_loss"] * config.loss.weight_freq + losses_g['l_2'] * config.loss.weight_l2
        # if epoch >= config.loss.commit_start_epoch:
        #     loss += commit_loss * config.loss.weight_commit + codebook_loss
        
        # if train_discriminator:
        #     loss += losses_g['l_g'] * config.loss.weight_g + losses_g['l_feat'] * config.loss.weight_feat
        #     epoch_loss += loss_disc.item()

        # epoch_loss += loss.item()

        all_codes.append(codes)
        # metrics.fill_metrics({
        #     'Loss Frequency': freq_loss_dict["total_loss"].item(),
        #     'Loss L1': losses_g['l_1'].item(),
        #     'Loss commit_loss': commit_loss.item(),
        #     'Loss Frequency L1': freq_loss_dict["l1_loss"].item(),
        #     'Frequency Accuracy': freq_loss_dict["acc"].item(),
        # }, epoch*len(val_loader) + i)
        # for j, d_id in enumerate(ds_id):
        #     dataset_id = d_id
        #     metrics.fill_metrics({f'Loss L1 {dataset_id}': losses_g['l_t'][j].item()}, epoch*len(val_loader) + i)
 
        # if train_discriminator:
        #     metrics.fill_metrics({
        #         'Loss Generator': losses_g['l_g'].item(),
        #         'Loss Feature': losses_g['l_feat'].item(),
        #         'Loss Discriminator': loss_disc.item(),
        #         'Logits Real': (torch.mean(logits_real[0]).item() + torch.mean(logits_real[1]).item())/2,
        #         'Logits Fake': (torch.mean(logits_fake[0]).item() + torch.mean(logits_fake[1]).item())/2
        #     }, epoch*len(val_loader) + i)

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
            axs[0].set_ylim(-6, 6)
            axs[1].imshow(S_x.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
            axs[1].invert_yaxis()
            axs[1].set_title('Original Spectrogram')

            axs[2].plot(x_time, x_hat[0].cpu().numpy().squeeze())
            axs[2].set_title('Reconstructed')
            axs[2].set_ylim(-6, 6)
            axs[3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
            axs[3].invert_yaxis()
            axs[3].set_title('Reconstructed Spectrogram')

            fig.tight_layout()
            fig.savefig(f"{log_dir}/{epoch}.png")
            plt.close(fig)

    all_codes = torch.cat(all_codes, dim=0) # B, num_codebooks, T
    all_codes = torch.permute(all_codes, (1, 0, 2))

    # flatten the last two dimensions
    all_codes = all_codes.reshape(all_codes.shape[0], -1)

    # log the distribution of codes. one distribution for each codebook
    entropies = []
    for i in range(all_codes.shape[0]):
        writer.add_histogram(f'Codes/Codebook {i}', all_codes[i], epoch)
        #calculate entropy
        _, counts = torch.unique(all_codes[i], return_counts=True)
        probabilities = counts.float() / counts.sum()
        entropy = -(probabilities * probabilities.log2()).sum()
        entropies.append(entropy.item())
    #create a graph of entropy
    fig, ax = plt.subplots()
    x_axis = np.arange(0, len(entropies))
    ax.plot(x_axis, entropies)
    ax.set_title("Entropy of Codebooks")
    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Entropy")
    ax.set_ylim(0, math.log2(config.model.bins))
    fig.tight_layout()
    writer.add_figure(f"Entropy/{epoch}", fig)
    plt.close(fig)

    # loss_per_epoch = epoch_loss/len(val_loader)
    # print(f"Epoch {epoch}, validation loss: {loss_per_epoch}")

    # log the metrics
    # metrics_dict = metrics.compute_and_log_metrics()
    # metrics_dict['Loss'] = loss_per_epoch
    # logger(writer, metrics_dict, 'val', epoch)
    # metrics.clear_metrics()

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
def load_config(filepath: str, schemapath: str | None = None):
    with open(filepath, "r") as file:
        config_dict = yaml.safe_load(file)
    if schemapath:
        with open(schemapath) as f:
            schema = json.load(f)
        jsonschema.validate(instance=config_dict, schema=schema)
        print("✅ YAML config is valid!")
    return config_dict, ConfigNamespace(config_dict)

def init_logger(log_dir, resume=False):
    print(f'log_dir: {log_dir}')
    if resume:
        # Resume logging
        writer = SummaryWriter(log_dir=log_dir, purge_step=None)  # Prevents overwriting
    else:
        writer = SummaryWriter(log_dir=log_dir)
    return writer

def init_model(config, train_discriminator: bool, save_path: str | None) -> Tuple[EncodecModel, Optional[MultiScaleSTFTDiscriminator]]:
    model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, model_norm=config.model.norm, 
        # audio_normalize=config.model.audio_normalize,
        segment=eval(config.model.segment), #name=config.model.name,
        ratios=config.model.ratios,
        bins=config.model.bins,
        dimension=config.model.dimension,
    )
    if train_discriminator:
        disc_model = MultiScaleSTFTDiscriminator(
            in_channels=config.model.channels,
            out_channels=config.model.channels,
            filters=config.discrim.filters,
            hop_lengths=config.discrim.disc_hop_lengths,
            win_lengths=config.discrim.disc_win_lengths,
            n_ffts=config.discrim.disc_n_ffts,
        )
    else:
        disc_model = None

    # log model, disc model parameters and train mode
    print(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total number of parameters: {total_params}")
    if save_path:
        print_model_details(model=model, log_path=f"{save_path}/model.txt")
    if train_discriminator:
        print(f"disc model train mode :{disc_model.training}")
        total_params = sum(p.numel() for p in disc_model.parameters())
        print(f"Discriminator Total number of parameters: {total_params}")
        if save_path:
            print_model_details(model=disc, log_path=f"{save_path}/disc.txt")
    
    return model, disc_model

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Model saved at epoch {epoch}") 

def load_checkpoint(model: EncodecModel, optimizer: optim.Optimizer | None, scheduler: Optional[LinearWarmupCosineAnnealingLR], path: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    print(f"Model loaded: from epoch {epoch}")
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Optimizer loaded")
    if scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Scheduler loaded")
    return epoch 

def save_disc(disc, disc_optimizer, disc_scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': disc.module.state_dict(),
        'optimizer_state_dict': disc_optimizer.state_dict(),
        'scheduler_state_dict': disc_scheduler.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Disc saved at epoch {epoch}")

def load_disc(disc, disc_optimizer, disc_scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    disc.load_state_dict(checkpoint['model_state_dict'])
    disc_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    disc_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch'] + 1  # Resume from next epoch
    print(f"Discriminator loaded: Resuming from epoch {epoch}")
    return epoch

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="091224_l1")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = set_args()
    user_name = os.getlogin()

    checkpoint_path = args.resume_from
    # Load the YAML file
    if os.path.exists(checkpoint_path):
        resume=True
        log_dir = checkpoint_path
        config_dict, config = load_config(filepath=f"{checkpoint_path}/config.yaml", schemapath=None)
    else:
        resume=False  
        config_dict, config = load_config(filepath=f"encodec/params/{args.exp_name}.yaml", schemapath=f"encodec/params/schema.json")
        curr_time = datetime.now().strftime("%Y%m%d")
        curr_minute = datetime.now().strftime("%H%M")
        log_dir = f"{args.log_dir}/{curr_time}_{curr_minute}"
        os.makedirs(log_dir, exist_ok=True)

    # init summarywriter and save config, set random seed, set device
    writer = init_logger(log_dir=log_dir, resume=resume)
    set_random_seed(config.common.seed)
    device = torch.device("cuda")
    if not checkpoint_path:
        #save yaml file to log_dir
        with open(f"{log_dir}/config.yaml", "w") as file:
            yaml.dump(config_dict, file)

    # init evaluation metrics logger
    metrics_args = MetricsArgs(device=device, datasets=config.dataset.datasets)
    metrics = Metrics(metrics_args)
    
    # load dataset, split into train and val
    _, train_loader = init_dataset(config=config, type="training", datasets=config.dataset.datasets, pin_memory=True, debug_training=args.debug)

    # init model params, print model details, move model to device
    model, disc = init_model(config, train_discriminator = config.discrim.train_discriminator, save_path=log_dir)
    model = model.to(device)
    if config.discrim.train_discriminator:
        disc = disc.to(device)
    
    # init optimizer, scheduler, loss 
    optimizer = optim.Adam(model.parameters(), lr=float(config.optimization.lr), betas=(config.optimization.beta1, config.optimization.beta2))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.optimization.warmup_epoch, max_epochs=config.common.max_epoch)
    freq_loss = ReconstructionLoss(alpha=config.spectrogram_loss.alpha, bandwidth=config.spectrogram_loss.bandwidth, sampling_rate=config.model.sample_rate, n_fft=config.spectrogram_loss.n_fft*config.model.sample_rate, hop_length=config.spectrogram_loss.hop_length*config.model.sample_rate, win_length=config.spectrogram_loss.win_length*config.model.sample_rate, device=device)

    if config.discrim.train_discriminator:
        optimizer_disc = optim.Adam(disc.parameters(), lr=float(config.optimization.disc_lr), betas=(config.optimization.beta1, config.optimization.beta2))
        disc_scheduler = LinearWarmupCosineAnnealingLR(optimizer_disc, warmup_epochs=config.optimization.warmup_epoch, max_epochs=config.common.max_epoch-config.discrim.train_discriminator_start_epoch)
    else:
        optimizer_disc = None
        disc_scheduler = None

    if resume and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, f"{checkpoint_path}/model.pth", device)
        if config.discrim.train_discriminator:
            load_disc(disc, optimizer_disc, disc_scheduler, f"{checkpoint_path}/disc.pth", device)
    else:
        start_epoch = 1
        
    if config.distributed.data_parallel:
        model = nn.DataParallel(model)
        disc = nn.DataParallel(disc)

    for epoch in tqdm(range(start_epoch, config.common.max_epoch+2), desc="Epochs", unit="epoch"):
        train_one_step(metrics=metrics, epoch=epoch, optimizer=optimizer, optimizer_disc=optimizer_disc, scheduler=scheduler, disc_scheduler=disc_scheduler, model=model, disc=disc, train_loader=train_loader, config=config, writer=writer, freq_loss=freq_loss)
        if epoch % config.common.test_every == 1:
            test(metrics, epoch,model,disc, train_loader, config, writer, freq_loss=freq_loss, log_dir=log_dir)
        # save checkpoint and epoch
        if epoch % config.common.save_every == 1:
            save_checkpoint(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, path=f"{log_dir}/model.pth")
            if config.discrim.train_discriminator:
                save_disc(disc, optimizer_disc, disc_scheduler, epoch, f"{log_dir}/disc.pth")

