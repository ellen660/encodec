#data loader for shhs2 breathing dataset
#return the raw breathing, support deubgging

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .preprocess import signal_crop, norm_sig, detect_motion_iterative
from scipy.ndimage import zoom

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
    
    def process_signal(self, signal, fs):
        assert fs == 10, f"fs is not 10 but {fs}"
        signal, _, _ = detect_motion_iterative(signal, fs)
        signal = signal_crop(signal)
        signal = norm_sig(signal)

        if fs != 10:
            signal = zoom(signal, 10/fs)
            fs = 8

        return signal

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.root_dir, filename)
        breathing, fs = np.load(filepath)['data'], np.load(filepath)['fs']
        # print(f'breathing shape: {breathing.shape}, fs: {fs}')
        assert fs == 10, "Sampling rate is not 10Hz"
        breathing = breathing[:self.max_length] # 4 hours
        # breathing = self.process_signal(breathing, fs)
        breathing = torch.tensor(breathing, dtype=torch.float32)

        #clip breathing -6,6

        #preserve the frequency 
        #iteralitely compute mean and std wihtin sliding window
        #destory the amplitude information in breathing belt and rf
        #so the only thing that is retained in frequency 
        #model only learns frequency information

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

