from torch.utils.data import DataLoader, ConcatDataset, Dataset
import torch
import sys
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt

# Add the B directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3] / 'time_series_foundation_models/dataloaders'))
from universal_loader import BaseDataset, Object #type: ignore

#debugging what is going on
#pin memory etc.
#todo: satori stuff a lot 
#is iterator faster?

class UniversalWrapper(BaseDataset):
    def __init__(self, args: Object, type: Literal["train", "val"], compression_ratio: int, debug_training: bool):
        super().__init__(
            args= args,
            val = (type=="val")
        )
        self.compression_ratio = compression_ratio
        self.args = args
        self.debug_training = debug_training
        self._assert_output()
        
    def __len__(self):
        if self.debug_training:
            return 1024
        else:
            return super().__len__()
    
    def visualize_sample(self, save_dir: str, num_samples: int = 5):
        for i in range(num_samples):
            data, label = self.__getitem__(0)
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

            plt.tight_layout()
            fig.savefig(f'{save_dir}/{self.args.mode}/{label}_{data["filename"]}.png')
                    
    def _assert_output(self):
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
    
def init_dataset(config, type: Literal["training", "inference"], datasets: list[str], ddp=False, pin_memory=True, debug_training=False) -> Tuple[Dataset, DataLoader]:
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

    train_files, val_files = set(), set()
    train_dataset = UniversalWrapper(
        type="train",
        compression_ratio=compression_ratio,
        args=args,
        debug_training=debug_training
    )
    train_files.update(train_dataset.all_files)
    
    val_dataset = UniversalWrapper(
        type="val",
        compression_ratio=compression_ratio,
        args=args,
        debug_training=debug_training
    )
    val_files.update(set(val_dataset.all_files))
    
    assert train_files.isdisjoint(val_files), "training and val sets intersect!"
    print("✅ No data leakage")
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    print(f'total number of samples for {type}: {len(combined_dataset)}')
    
    train_loader = DataLoader(
        dataset=combined_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.common.num_workers,
        pin_memory=pin_memory,
        sampler=DistributedSampler(combined_dataset, shuffle=True) \
            if ddp else \
            RandomSampler(combined_dataset),
        persistent_workers=True,
        drop_last = True
    )
    
    return combined_dataset, train_loader
