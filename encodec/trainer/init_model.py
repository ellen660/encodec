from clean_model import EncodecModel
from msstftd import MultiScaleSTFTDiscriminator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import set_random_seed, print_model_details
from typing import Tuple, Optional

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
            print_model_details(model=disc_model, log_path=f"{save_path}/disc.txt")
    
    return model, disc_model


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    """
    Save checkpoint only on rank 0 to avoid multiple writes.
    """
    if dist.get_rank() == 0:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved at epoch {epoch} by rank 0")

def load_checkpoint(path, model, optimizer=None, scheduler=None, local_rank=0):
    """
    Load checkpoint on rank 0, broadcast model weights to other ranks.
    Optionally load optimizer and scheduler state on rank 0.
    """
    map_location = {'cuda:0': f'cuda:{local_rank}'}

    if local_rank == 0:
        checkpoint = torch.load(path, map_location=map_location)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    else:
        checkpoint = None

    # Wait for rank 0 to finish loading
    dist.barrier()

    # Broadcast model parameters from rank 0 to all ranks
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # Optionally, broadcast optimizer and scheduler states if needed (usually not necessary)
    # This requires custom serialization and is usually avoided

    return checkpoint if local_rank == 0 else None


def wrap_model(model, local_rank):
    # Wrap in DDP
    return DDP(model, device_ids=[local_rank], output_device=local_rank)