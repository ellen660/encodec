from .all_datasets import MergedDataset
from .ppg_dataset import PpgDataset
from torch.utils.data import DataLoader, ConcatDataset
import torch
import sys
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import Literal
import numpy as np

# Add the B directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3] / 'time_series_foundation_models/dataloaders'))
from universal_loader import BaseDataset, Object #type: ignore

class UniversalWrapper(BaseDataset):
    def __init__(self, args: Object, type: Literal["train", "val"], compression_ratio: int):
        super().__init__(
            args= args,
            val = (type=="val")
        )
        self.compression_ratio = compression_ratio
        self.args = args
        self._assert_output()
        
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
    
def init_inference_dataset(config, datasets: list[str]):
    """
    seq len  = -1 for inference
    return both train and val (combined = all files)
    """
    cv = config.dataset.cv
    compression_ratio = np.prod(config.model.ratios)

    train_files, val_files = set(), set()
    train_dataset = UniversalWrapper(
        type="train",
        compression_ratio=compression_ratio,
        args=Object(dataset=datasets, mode="ppg", label="mit_gender", seq_len=-1, fold=cv, z_score=True, exclude_dataset=None)
    )
    train_files.update(train_dataset.all_files)
    
    val_dataset = UniversalWrapper(
        type="val",
        compression_ratio=compression_ratio,
        args=Object(dataset=datasets, mode="ppg", label="mit_gender", seq_len=-1, fold=cv, z_score=True, exclude_dataset=None)
    )
    val_files.update(set(val_dataset.all_files))
    
    assert train_files.isdisjoint(val_files), "training and val sets intersect!"
                    
    return ConcatDataset([train_dataset, val_dataset])
    
def init_dataset(config, ddp=False):
    """
    from time series univeral loader
    """
    cv = config.dataset.cv
    max_length = config.dataset.max_length
    fs = config.model.sample_rate
    datasets = config.dataset.datasets
    exclude_dataset = "mesa" if config.dataset.external else None
    compression_ratio = np.prod(config.model.ratios)

    train_files, val_files = set(), set()
    train_dataset = UniversalWrapper(
        type="train",
        compression_ratio=compression_ratio,
        args=Object(dataset=datasets, mode="ppg", label="mit_gender", seq_len=fs * max_length, fold=cv, z_score=True, exclude_dataset=exclude_dataset)
    )
    train_files.update(train_dataset.all_files)
    
    val_dataset = UniversalWrapper(
        type="val",
        compression_ratio=compression_ratio,
        args=Object(dataset=datasets, mode="ppg", label="mit_gender", seq_len=fs * max_length, fold=cv, z_score=True, exclude_dataset=exclude_dataset)
    )
    val_files.update(set(val_dataset.all_files))
    
    assert train_files.isdisjoint(val_files), "training and val sets intersect!"
    print("Number of training samples: ", len(train_files))
    print("Number of val samples: ", len(val_files))
    print("✅ No data leakage")
    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    
    train_loader = DataLoader(
        dataset=combined_dataset,
        batch_size=config.optimization.batch_size,
        shuffle=False,
        num_workers=config.common.num_workers,
        pin_memory=True,
        sampler=DistributedSampler(combined_dataset, shuffle=True) \
            if ddp else \
            RandomSampler(combined_dataset),
        persistent_workers=True,
        drop_last = True
    )
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=config.optimization.batch_size,
    #     shuffle=False, #avoid conflict errors with sampler. sapmpler takes precedence
    #     num_workers=config.common.num_workers,
    #     pin_memory=True,
    #     sampler=DistributedSampler(val_dataset, shuffle=True) \
    #         if ddp else \
    #         RandomSampler(val_dataset),
    #     persistent_workers=True,
    #     drop_last = False
    # )
    
    return train_loader


def test_datasets(config, datasets):
    print('here', datasets)
    cv = config.dataset.cv
    max_length = config.dataset.max_length
    fs = config.model.sample_rate
    compression_ratio = np.prod(config.model.ratios)

    train_files, val_files = set(), set()
    for dataset in datasets:
        train_dataset = UniversalWrapper(
            type="train",
            compression_ratio=compression_ratio,
            args=Object(dataset=[dataset], mode="ppg", label="mit_gender", seq_len=fs * max_length, fold=cv, z_score=True, exclude_dataset=None)
        )
        train_files.update(train_dataset.all_files)
        
        val_dataset = UniversalWrapper(
            type="val",
            compression_ratio=compression_ratio,
            args=Object(dataset=[dataset], mode="ppg", label="mit_gender", seq_len=fs * max_length, fold=cv, z_score=True, exclude_dataset=None)
        )
        val_files.update(set(val_dataset.all_files))
        print(f'len {dataset} dataset: {len(train_dataset) + len(val_dataset)} split as {len(train_dataset)} and {len(val_dataset)}')
    
    assert train_files.isdisjoint(val_files), "training and val sets intersect!"
    print("Number of training samples: ", len(train_files))
    print("Number of val samples: ", len(val_files))
    print(f"Total is {len(train_files) + len(val_files)}")
    print("✅ No data leakage")

# def init_dataset(config, ddp=False):
#     cv = config.dataset.cv
#     max_length = config.dataset.max_length
#     weights = {
#         "bwh": config.dataset.bwh,
#         "mesa": config.dataset.mesa,
#         "mgh": config.dataset.mgh
#     }

#     train_datasets, val_datasets, train_weight, val_weight = [], [], [], []
#     train_files, val_files = set(), set()
#     for ds_name, weight in weights.items():
#         if weight > 0:
#             train_dataset = PpgDataset(dataset = ds_name, mode = "train", cv = cv, max_length = max_length)
#             train_datasets.append(train_dataset)
#             train_files.update(train_dataset.file_list)
            
#             val_dataset = PpgDataset(dataset = ds_name, mode = "val", cv = cv, max_length = max_length)
#             val_datasets.append(val_dataset)
#             val_files.update(set(val_dataset.file_list))
            
#             train_weight.append(float(weight))
#             val_weight.append(float(weight))
#     assert train_files.isdisjoint(val_files), "Sets intersect!"

#     #Holdout/external dataset
#     # val_datasets.append(BreathingDataset(dataset = "mesa_new", mode = "val", cv = cv, channels = channels, max_length = max_length))
#     # val_weight.append(1.)

#     print("Number of training datasets: ", len(train_datasets))
#     # merge the datasets
#     train_dataset = MergedDataset(train_datasets, train_weight, 1., config.common.debug)
#     val_dataset = MergedDataset(val_datasets, val_weight, 0.2, config.common.debug)

#     if not ddp:
#         train_loader = DataLoader(train_dataset, batch_size=config.optimization.batch_size, shuffle=True, num_workers=config.common.num_workers, drop_last=True, pin_memory=True, persistent_workers=True)
#         val_loader = DataLoader(val_dataset, batch_size=config.optimization.batch_size, shuffle=False, num_workers=config.common.num_workers, drop_last=False, pin_memory=True, persistent_workers=True)

#         print(f'Merged dataset size: {len(train_dataset)}')
#         return train_loader, val_loader, train_dataset.mapping, val_dataset.mapping
#     else:
#         return train_dataset, val_dataset, train_dataset.mapping, val_dataset.mapping