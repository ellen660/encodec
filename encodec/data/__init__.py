from .all_datasets import MergedDataset
from .dataset import BreathingDataset
from .bwh import BwhDataset
from torch.utils.data import DataLoader

def init_dataset(config):
    cv = config.dataset.cv
    max_length = config.dataset.max_length
    weights = {
        "mgh_train_encodec": config.dataset.mgh,
        "shhs2_new": config.dataset.shhs2,
        "shhs1_new": config.dataset.shhs1,
        "mros1_new": config.dataset.mros1,
        "mros2_new": config.dataset.mros2,
        "wsc_new": config.dataset.wsc,
        "cfs": config.dataset.cfs,
        "bwh_new": config.dataset.bwh
    }

    train_datasets, val_datasets, train_weight, val_weight = [], [], [], []
    channels = {'thorax': config.dataset.thorax, 'abdominal': config.dataset.abdominal}
    for ds_name, weight in weights.items():
        if weight > 0:
            if ds_name == "bwh_new":
                train_datasets.append(BwhDataset(dataset = ds_name, mode = "train", cv = cv, channels = {"thorax": 1.0}, max_length = max_length))
                val_datasets.append(BwhDataset(dataset = ds_name, mode = "val", cv = cv, channels = {"thorax": 1.0}, max_length = max_length))
            else:
                train_datasets.append(BreathingDataset(dataset = ds_name, mode = "train", cv = cv, channels = channels, max_length = max_length))
                val_datasets.append(BreathingDataset(dataset = ds_name, mode = "val", cv = cv, channels = channels, max_length = max_length))
            train_weight.append(float(weight))
            val_weight.append(float(weight))

    #Holdout/external dataset
    val_datasets.append(BreathingDataset(dataset = "mesa_new", mode = "val", cv = cv, channels = channels, max_length = max_length))
    val_weight.append(1.)

    print("Number of training datasets: ", len(train_datasets))
    # merge the datasets
    train_dataset = MergedDataset(train_datasets, train_weight, 1., config.common.debug)
    val_dataset = MergedDataset(val_datasets, val_weight, 0.2, config.common.debug)
    train_loader = DataLoader(train_dataset, batch_size=config.optimization.batch_size, shuffle=True, num_workers=config.common.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.optimization.batch_size, shuffle=False, num_workers=config.common.num_workers)
    print(f'Merged dataset size: {len(train_dataset)}')
    return train_loader, val_loader, train_dataset.mapping, val_dataset.mapping