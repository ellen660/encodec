# Configuration and setup
# Load configuration: model hyperparameters, hyperparameters, data paths

# Set seed (for reproducibility)

# Initialize logger (to track progress and metrics)
# 	setlevel(INFO)
# 	Init save path for logger
# 	Init summarywriter (tensorboard) or weights and biases
# 	Save config file
# 	Save model details/architecture
# 	Save model checkpoints

# Set device (ddp, dataparallel, one cuda)

import argparse
import json
import os
from datetime import datetime

import jsonschema
import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from losses import LinearWarmupCosineAnnealingLR, Metrics, MetricsArgs, ReconstructionLosses
from torch.distributed import destroy_process_group
from torch.utils.tensorboard import SummaryWriter

from encodec.baseline_data import init_dataset
from encodec.trainer.init_model import init_model, load_checkpoint, save_checkpoint, wrap_model
from encodec.trainer.train import train_one_step
from encodec.utils import set_random_seed


class ConfigNamespace:
    """Converts a dictionary into an object-like namespace for easy attribute access."""

    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = ConfigNamespace(value)  # Recursively convert nested dictionaries
            setattr(self, key, value)


# Load the YAML file and convert to ConfigNamespace
def load_config(filepath: str, schemapath: str | None = None):
    with open(filepath) as file:
        config_dict = yaml.safe_load(file)
    if schemapath:
        with open(schemapath) as f:
            schema = json.load(f)
        jsonschema.validate(instance=config_dict, schema=schema)
        print("✅ YAML config is valid!")
    return config_dict, ConfigNamespace(config_dict)


def init_logger(log_dir, resume=False):
    print(f"log_dir: {log_dir}")
    if resume:
        # Resume logging
        writer = SummaryWriter(log_dir=log_dir, purge_step=None)  # Prevents overwriting
    else:
        writer = SummaryWriter(log_dir=log_dir)
    return writer


def setup_device():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"World Size: {world_size}")
    print(f"Process rank {rank} is using device {local_rank}")

    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank, rank, world_size


def cleanup():
    torch.cuda.empty_cache()
    destroy_process_group()


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="091224_l1")
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    user_name = os.getlogin()

    # Load the YAML file
    if os.path.exists(args.resume_from):
        resume = True
        log_dir = args.resume_from
        config_dict, config = load_config(filepath=f"{args.resume_from}/config.yaml", schemapath=None)
    else:
        resume = False
        config_dict, config = load_config(
            filepath=f"encodec/params/{args.exp_name}.yaml",
            schemapath="encodec/params/schema.json",
        )
        curr_time = datetime.now().strftime("%Y%m%d")
        curr_minute = datetime.now().strftime("%H%M")
        log_dir = f"{args.log_dir}/{curr_time}_{curr_minute}"

    set_random_seed(config.common.seed)

    device, local_rank, rank, world_size = setup_device()

    # init summarywriter and save config, set random seed, set device
    os.makedirs(log_dir, exist_ok=True)
    writer = init_logger(log_dir=log_dir, resume=resume) if rank == 0 else None
    if rank == 0 and not resume:
        # save yaml file to log_dir
        with open(f"{log_dir}/config.yaml", "w") as file:
            yaml.dump(config_dict, file)

    # load dataset, split into train and val
    # Data preparation
    # Load dataset
    # Preprocess
    # Split into training, validation, and test sets
    # Wrap into Dataload with batch and shuffle
    # Send data to device

    _, train_loader, sampler = init_dataset(
        config=config,
        type="training",
        datasets=config.dataset.datasets,
        ddp=True,
        pin_memory=True,
        debug_training=args.debug,
    )

    # Model Initialization
    # Define architecture
    # Initialize parameters
    # Move model to device
    model, disc = init_model(
        config,
        train_discriminator=config.discrim.train_discriminator,
        save_path=log_dir,
    )
    model = model.to(device)
    model = wrap_model(model, local_rank=device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config.optimization.lr),
        betas=(config.optimization.beta1, config.optimization.beta2),
    )
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=config.optimization.warmup_epoch,
        max_epochs=config.common.max_epoch,
    )
    optimizer_disc = None
    disc_scheduler = None

    # Load checkpoint if resuming
    start_epoch = 1
    if args.resume_from:
        start_epoch = load_checkpoint(args.resume_from, model, optimizer, scheduler, local_rank=local_rank)

    hop_lengths = [int(hop_length * config.model.sample_rate) for hop_length in config.spectrogram_loss.hop_length]
    window_lengths = [int(win_length * config.model.sample_rate) for win_length in config.spectrogram_loss.win_length]
    freq_loss = ReconstructionLosses(
        alpha=config.spectrogram_loss.alpha,
        bandwidth=config.spectrogram_loss.bandwidth,
        sampling_rate=config.model.sample_rate,
        n_fft=config.spectrogram_loss.n_fft,
        hop_length=hop_lengths,
        win_length=window_lengths,
        device=device,
    )

    # init evaluation metrics logger
    metrics_args = MetricsArgs(device=device, datasets=config.dataset.datasets)
    metrics = Metrics(metrics_args)

    for epoch in range(start_epoch, config.common.max_epoch + 2):
        # Important: update sampler so DistributedSampler reshuffles globally each epoch
        if sampler is not None:
            sampler.set_epoch(epoch)

        train_one_step(
            metrics=metrics,
            epoch=epoch,
            optimizer=optimizer,
            optimizer_disc=optimizer_disc,
            scheduler=scheduler,
            disc_scheduler=disc_scheduler,
            model=model,
            disc=disc,
            train_loader=train_loader,
            config=config,
            writer=writer,
            freq_loss=freq_loss,
            device=device,
            rank=rank,
        )

        # if epoch % config.common.test_every == 1 and rank == 0:
        #     test(metrics=metrics, epoch=epoch, model=model, disc=disc, val_loader=train_loader, config=config, writer=writer, freq_loss=freq_loss, log_dir=log_dir)

        if epoch % config.common.save_every == 1 and rank == 0:
            # save_checkpoint(
            #     model=model.module if config.distributed else model,
            #     optimizer=optimizer,
            #     scheduler=scheduler,
            #     epoch=epoch,
            #     path=f"{log_dir}/model.pth",
            #     optimizer_disc=optimizer_disc,
            #     disc_scheduler=disc_scheduler
            # )

            save_checkpoint(model, optimizer, scheduler, epoch, f"{log_dir}/model.pth")

    cleanup()
