import torch
from torch.utils.data import IterableDataset
import numpy as np
import random

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


# Add the B directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3] / 'time_series_foundation_models/dataloaders'))
from iterator_dataloader import BaseIterableDataset, get_dist_info, seed_worker, is_dist_initialized #type: ignore
from universal_loader import Object #type: ignore


# ---------- Wrapper that validates output and enforces compression_ratio ----------
class UniversalIterableWrapper(IterableDataset):
    def __init__(self, args, type: Literal["train", "val"], compression_ratio: int):
        super().__init__()
        self.inner = BaseIterableDataset(args=args, val=(type == "val"))
        self.compression_ratio = compression_ratio
        self.args = args

        # NOTE: do not iterate dataset here (avoid consuming samples on import/creation)

    def assert_output_once(self, max_tries: int = 50):
        """
        Run a quick check on rank 0 only to validate output shapes & dtypes.
        This should be called after DDP init (or in single-process runs).
        """
        rank, _ = get_dist_info()
        if rank != 0:
            return  # only rank 0 runs the check

        it = iter(self)
        tries = 0
        while tries < max_tries:
            try:
                data, label = next(it)
            except StopIteration:
                raise RuntimeError("Dataset empty when asserting output.")
            except Exception:
                tries += 1
                continue

            # validations
            x = data["x"]
            if self.args.seq_len != -1:
                expected = (1, self.args.seq_len)
                assert x.shape == expected, f"expected {expected} but got {x.shape}"
            assert x.dtype == torch.float32, "need torch.float32"
            assert x.shape[1] % self.compression_ratio == 0, f"need data length divisible by {self.compression_ratio}"
            print("[UniversalIterableWrapper] assert_output OK. labels:", label)
            return
        raise RuntimeError("Failed to validate dataset output after many tries.")

    def __iter__(self):
        # delegate to inner BaseIterableDataset iterator
        # optionally limit samples for debug
        # sample_count = 0
        for data_tensor, label_tensor in self.inner:
            # compression ratio adjustment
            length = data_tensor.size(0)
            new_length = (length // self.compression_ratio) * self.compression_ratio
            if new_length == 0:
                # skip too-short signals
                print("[UniversalIterableWrapper] skipping too-short signal")
                sys.exit(1)
            data_tensor = data_tensor[:new_length].unsqueeze(0)  # (1, new_length)

            if data_tensor is None or label_tensor.get("filename", None) is None or label_tensor.get("dataset", None) is None:
                print("found bad file ")
                sys.exit(1)

            yield {"x": data_tensor, "filename": label_tensor["filename"]}, label_tensor["dataset"]

            # sample_count += 1
            # if getattr(self.args, "debug", False) and sample_count > 10000:
            #     break

class ConcatIterableDataset(IterableDataset):
    """Simple concatenation for multiple IterableDatasets."""
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __iter__(self):
        for ds in self.datasets:
            yield from iter(ds)
            
    def __len__(self):
        return sum(len(ds.inner.all_files) for ds in self.datasets)


def init_iter_dataset(
    config,
    type: Literal["training", "inference"],
    datasets: list[str],
    ddp=False,
    pin_memory=True,
    debug_training=False
) -> Tuple[IterableDataset, DataLoader]:
    """
    Initialize iterable datasets for training or inference.
    """
    cv = config.dataset.cv
    compression_ratio = int(np.prod(config.model.ratios))
    
    
    if type == "training":
        exclude_dataset = "mesa" if config.dataset.external else None
        seq_len = config.model.sample_rate * config.dataset.max_length
    elif type == "inference":
        exclude_dataset = None
        seq_len = -1
    else:
        raise ValueError(f"Invalid type: {type}")

    args = Object(
        dataset=datasets,
        mode=config.dataset.mode,
        label="mit_gender",
        seq_len=seq_len,
        fold=cv,
        z_score=True,
        exclude_dataset=exclude_dataset,
        debug=debug_training
    )

    # Create iterable datasets
    train_dataset = UniversalIterableWrapper(
        args=args,
        type="train",
        compression_ratio=compression_ratio,
    )
    val_dataset = UniversalIterableWrapper(
        args=args,
        type="val",
        compression_ratio=compression_ratio,
    )
    train_dataset.assert_output_once()

    # Make sure train & val file lists don't overlap
    assert set(train_dataset.inner.all_files).isdisjoint(val_dataset.inner.all_files), \
        "❌ Training and validation sets intersect!"
    print("✅ No data leakage")

    # Combine into one iterable if training mode
    combined_dataset = ConcatIterableDataset(train_dataset, val_dataset)
    print(f"Total number of files for {type}: "
            f"{len(train_dataset.inner.all_files) + len(val_dataset.inner.all_files)}")
    print(f'{len(combined_dataset)}')
    if is_dist_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
        base_worker_seed = config.common.seed + rank
        worker_init = lambda wid: seed_worker(wid, base_worker_seed)
    else:
        worker_init = None

    # Build DataLoader — no sampler for IterableDataset
    loader = DataLoader(
        dataset=combined_dataset,
        batch_size=config.optimization.batch_size,
        num_workers=config.common.num_workers,
        pin_memory=pin_memory,
        persistent_workers=True,
        drop_last=True,
        worker_init_fn=worker_init,
    )

    return combined_dataset, loader


    