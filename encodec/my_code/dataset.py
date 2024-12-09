#data loader for shhs2 breathing dataset
#return the raw breathing, support deubgging

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

BREATHING_DIR = "/data/netmit/wifall/ADetect/data/shhs2_new/thorax"
#N = 2651

class BreathingDataset(Dataset):
    def __init__(self, root_dir=BREATHING_DIR, debug=False, max_length=10 * 60 * 60 * 4):
        self.root_dir = root_dir
        self.debug = debug
        self.max_length = max_length
        self.file_list = [f for f in os.listdir(root_dir) if f.endswith('.npz') if self.filter_files(f)] 

    def filter_files(self, f):
        filepath = os.path.join(self.root_dir, f)
        breathing, fs = np.load(filepath)['data'], np.load(filepath)['fs']
        if breathing.shape[0] < self.max_length:
            return False
        return True

    def __len__(self):
        if self.debug:
            return 2*48
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.root_dir, filename)
        breathing, fs = np.load(filepath)['data'], np.load(filepath)['fs']
        # print(f'breathing shape: {breathing.shape}, fs: {fs}')
        assert fs == 10, "Sampling rate is not 10Hz"
        breathing = breathing[:self.max_length] # 4 hours
        breathing = torch.tensor(breathing, dtype=torch.float32)
        #unsquzze dim0
        breathing = breathing.unsqueeze(0)
        return breathing, 0

def main():
    dataset = BreathingDataset()
    print(f"Dataset size is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=48, num_workers = 32, shuffle=True)

    for i, (features, labels) in enumerate(dataloader):
        print(f"Batch {i+1}:")
        print(f"Features shape: {features.shape}")

if __name__ == '__main__':
    main()
