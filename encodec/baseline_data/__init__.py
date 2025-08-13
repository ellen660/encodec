from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch
import sys
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch.distributed as dist
import random
import os

# Add the B directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3] / 'time_series_foundation_models/dataloaders'))
from universal_loader import BaseDataset, Object #type: ignore

def get_dist_info():
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


class UniversalWrapper(BaseDataset):  #
    def __init__(self, args: Object, type: Literal["train", "val"], compression_ratio: int, debug_training: bool):
        super().__init__(args=args, val=(type == "val"))
        self.compression_ratio = compression_ratio
        self.args = args
        self.debug_training = debug_training
        # DO NOT run heavy asserts or IO here that rely on DDP being initialized.
        # If you want a one-time check, call `assert_output_once()` explicitly after DDP init.
        
    def __len__(self):
        if self.debug_training:
            return 1024
        else:
            return super().__len__()
    
    def visualize_sample(self, save_dir: str, num_samples: int = 5):
        os.makedirs(f'{save_dir}/{self.args.mode}', exist_ok=True)
        for i in range(num_samples):
            data, label = self.__getitem__(i)
            x = data["x"]
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 2 columns

            thirty_seconds = self.fs * 30
            five_seconds = self.fs * 5

            time_30 = np.arange(0, thirty_seconds)
            signal_cpu = x[0].cpu().numpy().squeeze()

            # Plot 30 seconds signal
            axs[0].plot(time_30, signal_cpu[ : thirty_seconds])
            axs[0].set_xlabel("Time")
            axs[0].set_title("Original Signal 30 seconds")
            axs[0].set_ylim(-6, 6)  # Set y-limits here

            # Plot 5 seconds signal
            axs[1].plot(signal_cpu[ : five_seconds])
            axs[1].set_title("Original Signal 5 seconds")
            axs[0].set_ylim(-6, 6)  # Set y-limits here
            axs[1].set_ylim(-6, 6)  # Set y-limits here

            plt.tight_layout()
            fig.savefig(f'{save_dir}/{self.args.mode}/{label}_{data["filename"]}.png')
                    
    def assert_output_once(self):
        """Call this on rank 0 after DDP init to validate a sample shape/dtype."""
        rank, _ = get_dist_info()
        if rank != 0:
            return
        data, label = self.__getitem__(0)
        if self.args.seq_len != -1:
            assert data["x"].shape == (1, self.args.seq_len,), f"expected {(1, self.args.seq_len)} but got {data['x'].shape}"
        assert data["x"].dtype == torch.float32, "need torch 32"
        assert data["x"].shape[1] % self.compression_ratio == 0, f"need data length to be divisible by {self.compression_ratio}"
        print(f'labels: {label}')

    def __getitem__(self, idx):
        data, label = super().__getitem__(idx, return_object=True)
        length = data.size(0)
        new_length = (length // self.compression_ratio) * self.compression_ratio
        data = data[:new_length].unsqueeze(0)
        return {"x": data, "filename": label["filename"]}, label["dataset"]



# -------------------------
# worker_init_fn for DataLoader
# -------------------------
def make_worker_init_fn(base_seed: int):
    """Returns a worker_init_fn that seeds python/random/numpy/torch with base_seed + rank + worker_id."""
    def worker_init_fn(worker_id):
        rank, _ = get_dist_info()
        seed = base_seed + rank * 10_000 + worker_id
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        # if using torch.cuda inside workers (rare), also set torch.cuda.manual_seed_all(seed)
    return worker_init_fn

# -------------------------
# init_dataset: returns dataset, loader, sampler
# -------------------------
def init_dataset(
    config,
    type: Literal["training", "inference"],
    datasets: list[str],
    ddp: bool = False,
    pin_memory: bool = True,
    debug_training: bool = False
) -> Tuple[ConcatDataset, DataLoader, DistributedSampler | None]:
    """
    from time series univeral loader
    """
    cv = config.dataset.cv
    compression_ratio = np.prod(config.model.ratios)
    if type == "training":
        exclude_dataset = "mesa" if config.dataset.external else None
        seq_len=config.model.sample_rate * config.dataset.max_length
    elif type == "inference":
        exclude_dataset = None
        seq_len = -1
    else:
        raise ValueError
    
    args = Object(dataset=datasets, mode=config.dataset.mode, label="mit_gender", seq_len=seq_len, fold=cv, z_score=True, exclude_dataset=exclude_dataset, debug=False)
    # create train/val datasets
    train_files, val_files = set(), set()
    train_dataset = UniversalWrapper(args=args, type="train", compression_ratio=compression_ratio, debug_training=debug_training)
    val_dataset = UniversalWrapper(args=args, type="val", compression_ratio=compression_ratio, debug_training=debug_training)
    train_files.update(train_dataset.all_files)
    val_files.update(set(val_dataset.all_files))
    assert train_files.isdisjoint(val_files), "training and val sets intersect!"
    print("✅ No data leakage")
    
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    print(f'total number of samples for {type}: {len(combined_dataset)}')
    
    # sampler: use DistributedSampler for DDP (it will call set_epoch in training loop)
    if ddp:
        sampler = DistributedSampler(
            combined_dataset,
            num_replicas=dist.get_world_size() if dist.is_initialized() else None,
            rank=dist.get_rank() if dist.is_initialized() else None,
            shuffle=True
        )
    else:
        sampler = None
        
    # NOTE: If you pass sampler, set shuffle=False (PyTorch will error otherwise).
    # Build worker_init_fn seed base — use a global base seed (e.g., from config or time)
    base_seed = config.common.seed
    if type == "training":
        worker_init = make_worker_init_fn(base_seed)
    else:
        worker_init = None

    data_loader = DataLoader(
        dataset=combined_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.common.num_workers,
        pin_memory=pin_memory,
        sampler=sampler if sampler is not None else RandomSampler(combined_dataset),
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=worker_init
    )

    return combined_dataset, data_loader, sampler
    


