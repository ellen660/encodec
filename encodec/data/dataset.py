#data loader for shhs2 breathing dataset
#return the raw breathing, support deubgging

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .preprocess import signal_crop, norm_sig, detect_motion_iterative
from scipy.ndimage import zoom
from .fns_to_ignore import fns_to_ignore

class BreathingDataset(Dataset):
    root = "/data/netmit/wifall/ADetect/data"
    NumCv = 4
        
    def __init__(self, dataset="shhs2_new", mode = "train", cv = 0, channel = "thorax", max_length=10 * 60 * 60 * 4):

        self.dataset = dataset
        self.mode = mode
        self.cv = cv
        self.channel = channel
        self.ds_dir = os.path.join(self.root, self.dataset, self.channel)
        self.max_length = max_length

        # dataset preparation
        # file_list = sorted([f for f in os.listdir(self.ds_dir) if f.endswith('.npz') if self.filter_files(f)])
        file_list_before = sorted([f for f in os.listdir(self.ds_dir) if f.endswith('.npz')])
        len_before = len(file_list_before)
        file_list = [f for f in file_list_before if f not in fns_to_ignore]
        len_after = len(file_list)
        print(f"Filtered out {len_before - len_after} files")

        # # get the set difference
        # file_diff = set(file_list_before) - set(file_list)
        # print(f"file_diff: {file_diff}")

        train_list, val_list = self.split_train_test(file_list)

        if mode == "train":
            self.file_list = train_list
        elif mode == "val":
            self.file_list = val_list
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def split_train_test(self, file_list):
        train_files = []
        test_files = []
        for i in range(len(file_list)):
            if i % self.NumCv == self.cv:
                test_files.append(file_list[i])
            else:
                train_files.append(file_list[i])

        return train_files, test_files

    # def filter_files(self, f):
    #     filepath = os.path.join(self.ds_dir, f)
    #     breathing, fs = np.load(filepath)['data'], np.load(filepath)['fs']
    #     if breathing.shape[0] < self.max_length:
    #         return False
    #     return True

    def __len__(self):
        return len(self.file_list)
    
    def process_signal(self, signal, fs):
        assert fs == 10, f"fs is not 10 but {fs}"
        signal, _, _ = detect_motion_iterative(signal, fs)
        signal = signal_crop(signal)
        signal = norm_sig(signal)

        if fs != 10:
            signal = zoom(signal, 10/fs)
            fs = 10

        return signal

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        filepath = os.path.join(self.ds_dir, filename)
        breathing = np.load(filepath)['data'].squeeze()
        fs = np.load(filepath)['fs']
        # print(f'breathing shape: {breathing.shape}, fs: {fs}')
        assert fs == 10, "Sampling rate is not 10Hz"
        
        if self.mode == "train":
            breathing_length = breathing.shape[0] - self.max_length
            #randomly sample start index
            try:
                start_idx = np.random.randint(0, breathing_length)
            except:
                print("breathing_length is negative")
                print(f"breathing_length: {breathing_length}")
                print("filename: ", filename)
                print(f"start_idx: {start_idx}")
                sys.exit()
            breathing = breathing[start_idx:start_idx+self.max_length]
        else:
            breathing = breathing[:self.max_length]

        # breathing = breathing[:self.max_length] #4 hours
        breathing = self.process_signal(breathing, fs)
        breathing = torch.tensor(breathing, dtype=torch.float32)

        # if there is any nan or inf in the signal, return None
        if torch.isnan(breathing).any() or torch.isinf(breathing).any():
            return None, 0

        #clip breathing -6,6

        #preserve the frequency 
        #iteralitely compute mean and std wihtin sliding window
        #destory the amplitude information in breathing belt and rf
        #so the only thing that is retained in frequency 
        #model only learns frequency information

        #unsquzze dim0
        breathing = breathing.unsqueeze(0)
        return breathing, 0

# def main():
#     dataset = BreathingDataset()
#     print(f"Dataset size is {len(dataset)}")
#     dataloader = DataLoader(dataset, batch_size=48, num_workers = 32, shuffle=True)

#     for i, (features, labels) in enumerate(dataloader):
#         print(f"Batch {i+1}:")
#         print(f"Features shape: {features.shape}")

# if __name__ == '__main__':
#     main()

