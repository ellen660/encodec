from .all_datasets import MergedDataset
from .bwh_ppg import BwhPpgDataset
from torch.utils.data import DataLoader
import torch

def init_dataset(config, ddp=False):
    cv = config.dataset.cv
    max_length = config.dataset.max_length
    weights = {
        "bwh_new": config.dataset.bwh,
        "mgh_new": config.dataset.mgh
    }

    train_datasets, val_datasets, train_weight, val_weight = [], [], [], []
    for ds_name, weight in weights.items():
        if weight > 0:
            train_datasets.append(BwhPpgDataset(dataset = ds_name, mode = "train", cv = cv, max_length = max_length))
            val_datasets.append(BwhPpgDataset(dataset = ds_name, mode = "val", cv = cv, max_length = max_length))
            train_weight.append(float(weight))
            val_weight.append(float(weight))

    #Holdout/external dataset
    # val_datasets.append(BreathingDataset(dataset = "mesa_new", mode = "val", cv = cv, channels = channels, max_length = max_length))
    # val_weight.append(1.)

    print("Number of training datasets: ", len(train_datasets))
    # merge the datasets
    train_dataset = MergedDataset(train_datasets, train_weight, 1., config.common.debug)
    val_dataset = MergedDataset(val_datasets, val_weight, 0.2, config.common.debug)

    if not ddp:
        train_loader = DataLoader(train_dataset, batch_size=config.optimization.batch_size, shuffle=True, num_workers=config.common.num_workers)
        val_loader = DataLoader(val_dataset, batch_size=config.optimization.batch_size, shuffle=False, num_workers=config.common.num_workers)

        print(f'Merged dataset size: {len(train_dataset)}')
        return train_loader, val_loader, train_dataset.mapping, val_dataset.mapping
    else:
        return train_dataset, val_dataset, train_dataset.mapping, val_dataset.mapping