import torch
import numpy as np
from torch.utils.data import Dataset

import sys

class MergedDataset(Dataset):
    def __init__(self, ds_list, weight_list, sfreq = 1, debug = False):
        self.ds = ds_list
        self.weight = np.array(weight_list)
        self.weight /= self.weight.sum() #TODO weigh it differently based on the dataset size
        assert self.weight[0] > 0  # the first dataset is pivot
        print(f'===> Dataset Merged: {self.weight}')
        if debug:
            size = 512
        else:
            size = 4096
        self.size = round(size * sfreq)
        self.mapping = {i: ds.dataset for i, ds in enumerate(ds_list)}
    
    def __len__(self):
        return self.size

    def __getitem__(self, item):
        ds = self.ds
        ds_id = torch.multinomial(torch.Tensor(self.weight), 1)[0].item()
        choosed_ds = ds[ds_id]
        item_id = torch.randint(0, len(choosed_ds), (1,))[0].item()
        item = choosed_ds[item_id]

        return item, ds_id