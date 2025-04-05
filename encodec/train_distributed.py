
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import argparse 
from tqdm import tqdm
import yaml
import random
import numpy as np
import time

from clean_model import EncodecModel
from data import init_dataset
from losses import total_loss, disc_loss, Metrics, MetricsArgs, LinearWarmupCosineAnnealingLR, WarmupScheduler, ReconstructionLoss
from torch.utils.tensorboard import SummaryWriter

def setup(rank, world_size):
    """Initialize the distributed training environment."""
    if "MASTER_ADDR" in os.environ:
        master_addr = os.environ['MASTER_ADDR']
    else:
        master_addr = "localhost"
    if "MASTER_PORT" in os.environ:
        master_port = os.environ["MASTER_PORT"]
    else:
        master_port = 6008
    distributed_init_method = "tcp://%s:%s" % (master_addr, master_port)
    print(f"Distributed init method: {distributed_init_method}")

    dist.init_process_group(
        backend="nccl", 
        init_method=distributed_init_method,#"env://",
        world_size=world_size, 
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()

def init_model(config):
    model = EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal, model_norm=config.model.norm, 
        segment=eval(config.model.segment), 
        ratios=config.model.ratios,
        bins=config.model.bins,
        dimension=config.model.dimension,
    )

    print(f"model train mode :{model.training} | quantizer train mode :{model.quantizer.training} ")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total number of parameters: {total_params}")
    return model

def init_dataset_ddp(config):
    # Use a DistributedSampler
    train_dataset, val_dataset, train_mapping, val_mapping = init_dataset(config, ddp=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.optimization.batch_size, sampler=train_sampler, num_workers=config.common.num_workers, pin_memory=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.optimization.batch_size, shuffle=False, sampler=val_sampler, num_workers=config.common.num_workers, pin_memory=True)

    return train_loader, val_loader, train_mapping, val_mapping

def train_one_step(metrics, epoch, optimizer, scheduler, model, train_loader, config, writer, freq_loss, label_mapping):
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
    epoch_loss = 0.0  # Store loss on the correct GPU
    progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch") if dist.get_rank() == 0 else train_loader

    start_time = time.time()
    for i, (item, ds_id) in enumerate(progress_bar):
        if i == 0:
            print(f'time taken to load first batch: {time.time() - start_time}')
        x = item["x"].cuda()
        if dist.get_rank() == 0 and i < 5:
            print(f'filename : {item["filename"][0]}')
        if x is None:
            continue
        # print(f'x shape: {x.shape}')
        x_hat, _, commit_loss, codebook_loss = model(x)
        # print(f'x_hat shape: {x_hat.shape}')

        logits_real, logits_fake, fmap_real, fmap_fake = None, None, None, None

        commit_loss = torch.mean(commit_loss)
        codebook_loss = torch.mean(codebook_loss)
        #Reduce commit_loss and codebook_loss across all processes
        dist.all_reduce(commit_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(codebook_loss, op=dist.ReduceOp.SUM)

        freq_loss_dict = freq_loss(x, x_hat)
        losses_g = total_loss(
                fmap_real, 
                logits_fake, 
                fmap_fake, 
                x, 
                x_hat, 
                sample_rate=10,
            ) 
        loss = losses_g['l_1'] * config.loss.weight_l1 + freq_loss_dict["total_loss"] * config.loss.weight_freq + losses_g['l_2'] * config.loss.weight_l2
        if epoch >= config.loss.commit_start_epoch:
            loss += commit_loss * config.loss.weight_commit 

        optimizer.zero_grad() #optimizer is the model only, so only updating those parameters
        loss.backward()
        # gradient clipping. restrict the norm of the gradients to be less than 1
        if config.common.gradient_clipping:
            nn.utils.clip_grad_norm_(model.parameters(), config.common.gradient_clipping_value)
        optimizer.step()

        if epoch % config.common.log_every == 0: # add the loss to the tensorboard
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)  # Sum up the loss from all ranks
            loss /= dist.get_world_size()  # Divide by the world size to get the average loss
            epoch_loss += loss.item()

            if dist.get_rank() == 0:
                metrics.fill_metrics({
                    'Loss L1': losses_g['l_1'].item(),
                    'Loss commit_loss': commit_loss.item(),
                    'Loss Frequency L1': freq_loss_dict["l1_loss"].item(),
                    'Frequency Accuracy': freq_loss_dict["acc"].item(),
                }, epoch*len(train_loader) + i)
                for j, d_id in enumerate(ds_id):
                    dataset_id = d_id.item()
                    metrics.fill_metrics({f'Loss L1 {label_mapping[dataset_id]}': losses_g['l_t'][j].item()}, epoch*len(train_loader) + i)

            max_gradient = torch.tensor(0.0).cuda()
            for param in model.parameters():
                if param.grad is not None:
                    max_gradient = max(max_gradient, param.grad.abs().max().item())

            if dist.get_rank() == 0:
                # log the max gradient
                metrics.fill_metrics({
                    'Max Gradient': max_gradient
                }, epoch*len(train_loader) + i)

    scheduler.step()  

    if epoch % config.common.log_every == 0 and dist.get_rank()==0:
        metrics_dict = metrics.compute_and_log_metrics()
        # log the learning rate
        metrics_dict['Learning Rate'] = optimizer.param_groups[0]['lr']

        # Reduce loss across all processes
        # dist.all_reduce(epoch_loss, op=dist.ReduceOp.SUM)
        loss_per_epoch = epoch_loss/(len(train_loader))
        print(f"Epoch {epoch}, training loss: {loss_per_epoch}")

        # log the metrics
        logger(writer, metrics_dict, 'train', epoch)
        metrics.clear_metrics()

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

def init_logger(rank, log_dir, resume=False):
    print(f'log_dir: {log_dir}')
    if rank == 0:
        if resume:
            # Resume logging
            writer = SummaryWriter(log_dir=log_dir, purge_step=None)  # Prevents overwriting
        else:
            writer = SummaryWriter(log_dir=log_dir)
        return writer
    return None

def save_checkpoint(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, path)
    print(f"Model saved at epoch {epoch}") 

def train(rank, world_size, config, log_dir):
    set_seed(42)
    setup(rank, world_size)
    torch.cuda.set_device(rank) 
    torch.cuda.empty_cache()

    train_loader, val_loader, train_mapping, val_mapping = init_dataset_ddp(config)
    model = init_model(config)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=float(config.optimization.lr), betas=(config.optimization.beta1, config.optimization.beta2))
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=config.optimization.warmup_epoch, max_epochs=config.common.max_epoch)
    freq_loss = ReconstructionLoss(alpha=config.spectrogram_loss.alpha, bandwidth=config.spectrogram_loss.bandwidth, sampling_rate=10, n_fft=config.spectrogram_loss.n_fft, hop_length=config.spectrogram_loss.hop_length, win_length=config.spectrogram_loss.win_length, device=rank)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # wrap the model by using DDP
    model = DDP(
        model,
        device_ids=[rank],
        output_device=rank,
        broadcast_buffers=False,
        find_unused_parameters=False)

    if rank == 0:
        writer = init_logger(rank, log_dir, resume=False)
    else:
        writer = None
    metrics_args = MetricsArgs(num_datasets=1, device=rank)
    metrics = Metrics(metrics_args)

    for epoch in range(config.common.max_epoch):
        train_loader.sampler.set_epoch(epoch) 
        train_one_step(metrics, epoch, optimizer, scheduler, model, train_loader, config=config, writer=writer, freq_loss=freq_loss, label_mapping=train_mapping)
    # # test(metrics, 1, model, val_loader, config, writer, freq_loss=freq_loss, label_mapping=val_mapping)

    cleanup()

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

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ddp")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    return parser.parse_args()

def set_seed(seed):
    """set seed

    Args:
        seed (int): seed number
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def start_dist_train(train_fn, world_size, config, dist_init_method=None):  
    """start distribustion training

    Args:
        train_fn (_type_): train function
        world_size (_type_): world size
        config (_type_): config 
        dist_init_method (_type_, optional): dist init method. Defaults to None.
    """
    torch.multiprocessing.set_start_method('spawn')  
    mp.spawn(  
        train_fn,  
        args=(world_size, config, dist_init_method) if dist_init_method else (world_size, config,),  
        nprocs=world_size,  
        join=True  
    )  

if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    args = set_args()
    log_dir = f'/data/scratch/ellen660/encodec/encodec/ablations/test_dist'
    os.makedirs(log_dir, exist_ok=True)
    # Load the YAML file
    config = load_config("encodec/params/%s.yaml" % args.exp_name, log_dir)

    world_size = torch.cuda.device_count()  # Number of GPUs
    start_dist_train(train, world_size, config, log_dir)