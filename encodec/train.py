from model import EncodecModel
from my_code.dataset import BreathingDataset
from my_code.losses import loss_fn_l1, loss_fn_l2
# from msstftd import MultiScaleSTFTDiscriminator
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

def train_one_step(epoch,optimizer, model, train_loader,config,scaler=None,scaler_disc=None,writer=None,balancer=None):
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
        x_hat = model(x)
        # loss_l1 = loss_fn_l1(x, x_hat)
        loss_l2 = loss_fn_l1(x, x_hat)
        loss = loss_l2
        
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # metrics.fill_metrics(y_pred, y, loss.item())
    print(f"Epoch {epoch}, training loss: {epoch_loss}")
    if epoch % config.common.log_interval == 0:
        print('logging')
        logger(writer, loss.item(), 'train', epoch)

@torch.no_grad()
def test(epoch, model, val_loader, config, writer):
    model.eval()
    epoch_loss = 0
    for i, (x, y) in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}", unit="batch")):
        x = x.to(device)
        x_hat = model(x)
        # loss_l1 = loss_fn_l1(x, x_hat)
        loss_l2 = loss_fn_l1(x, x_hat)
        loss = loss_l2
        epoch_loss += loss.item()

    print(f"Epoch {epoch}, validation loss: {epoch_loss}")
    if epoch % config.common.log_interval == 0:
        print('logging')
        logger(writer, loss.item(), 'val', epoch)

#Logger for tensorboard
def logger(writer, loss, phase, epoch_index):
    writer.add_scalar("%s/%s"%(phase, 'loss'), loss, epoch_index)

    writer.flush()

class ConfigNamespace:
    """Converts a dictionary into an object-like namespace for easy attribute access."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigNamespace(value)  # Recursively convert nested dictionaries
            setattr(self, key, value)

# Load the YAML file and convert to ConfigNamespace
def load_config(filepath, log_dir):
    #make directory
    os.makedirs(log_dir, exist_ok=True)
    with open(filepath, "r") as file:
        config_dict = yaml.safe_load(file)
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as file:
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
    print(model)
    # breakpoint()
    # logger.info(disc_model)
    # logger.info(config)
    # logger.info(f"Encodec Model Parameters: {count_parameters(model)}")
    print(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        model = model.to(device)
    return model

if __name__ == "__main__":
    
    current_time = datetime.now().strftime('%m-%d_%H-%M')
    # log_dir = os.path.join(f'/data/netmit/wifall/breathing_tokenizer/encodec/encodec/runs', f"{current_time}")
    log_dir = f'/data/scratch/ellen660/encodec/encodec/runs/{current_time}'

    # Load the YAML file
    config = load_config("encodec/my_code/config.yaml", log_dir)
    writer = init_logger(log_dir)

    device = torch.device("cuda")
    torch.manual_seed(config.common.seed)

    train_loader, val_loader = init_dataset(config)
    model = init_model(config)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config.optimization.lr), weight_decay=float(config.optimization.weight_decay))

    # instantiate loss balancer
    # balancer = Balancer(config.balancer.weights)
    # if balancer:
    #     print(f'Loss balancer with weights {balancer.weights} instantiated')
    test(0, model, val_loader, config, writer)
    for epoch in tqdm(range(1, config.common.max_epoch+1), desc="Epochs", unit="epoch"):
        # train_one_step(epoch,optimizer, model, train_loader,config,scaler=None,scaler_disc=None,writer=None,balancer=None):
        train_one_step(epoch, optimizer, model, train_loader, config=config,writer=writer)
        if epoch % config.common.test_interval == 0:
            test(epoch,model,val_loader,config,writer)
        # save checkpoint and epoch
        # if epoch % config.common.save_interval == 0:
        #     model_to_save = model.module if config.distributed.data_parallel else model
        #     # disc_model_to_save = disc_model.module if config.distributed.data_parallel else disc_model 
        #     if not config.distributed.data_parallel or dist.get_rank() == 0:  
        #         save_master_checkpoint(epoch, model_to_save, optimizer, scheduler, f'{config.checkpoint.save_location}epoch{epoch}_lr{config.optimization.lr}.pt')  
        #         save_master_checkpoint(epoch, disc_model_to_save, optimizer_disc, disc_scheduler, f'{config.checkpoint.save_location}epoch{epoch}_disc_lr{config.optimization.lr}.pt') 
