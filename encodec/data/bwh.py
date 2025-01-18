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
from .fns_to_ignore_bwh import fns_to_ignore
from tqdm import tqdm

class BwhDataset(Dataset):
    root = "/data/netmit/sleep_lab/ali_2"
    processed_signal = f'{root}/bwh_encodec'
    if not os.path.exists(processed_signal):
        os.makedirs(processed_signal)
    NumCv = 4
        
    def __init__(self, dataset="bwh_new", mode = "train", cv = 0, channels = {"thorax": 1.0}, max_length=10 * 60 * 60 * 4):
        channels = {"thorax": 1.0}
        self.dataset = dataset
        self.mode = mode
        assert self.mode in ['train', 'test'], 'Only support train or test mode'
        self.cv = cv
        self.channels = channels # dictionary of channel names and their weights
        self.ds_dir = self.root
        self.max_length = max_length

        # dataset preparation (only select the intersection between all channels)
        file_list = set()
        for channel in self.channels.keys():
            file_list_before = sorted([f for f in os.listdir(os.path.join(self.ds_dir, channel)) if f.endswith('.npz')])
            file_list_after = [f for f in file_list_before if f not in fns_to_ignore]
            file_list.update(file_list_after)

        file_list = sorted(file_list)

        train_list, val_list = self.split_train_test(file_list)

        # for file in tqdm(train_list):
        #     # now randomly select a channel, sampling based on their weights
        #     selected_channel = np.random.choice(list(self.channels.keys()), p=list(self.channels.values()))
        #     filepath = os.path.join(self.ds_dir, selected_channel, file)
        #     breathing = np.load(filepath)['data'].squeeze()
        #     fs = np.load(filepath)['fs']
        #     # print(f'breathing shape: {breathing.shape}, fs: {fs}')
        #     assert fs == 200, "Sampling rate is not 200Hz"
        #     breathing_length = breathing.shape[0] - (200*60*60*4)
        #     #randomly sample start index
        #     try:
        #         start_idx = np.random.randint(0, breathing_length)
        #     except:
        #         print("breathing_length is negative")
        #         print(f"breathing_length: {breathing_length}")
        #         print("filename: ", file)
        #         print(f"start_idx: {start_idx}")
        #         sys.exit()
        #     breathing = breathing[start_idx:start_idx+200*60*60*4]

        #     # breathing = breathing[:self.max_length] #4 hours
        #     breathing = self.process_signal(breathing, fs)

        #     #save to processed_signal
        #     np.savez(f'{self.processed_signal}/{file}', data=breathing, fs=10)
            

        if mode == "train":
            self.file_list = train_list
        elif mode == "val":
            self.file_list = val_list
        elif mode == "test":
            self.file_list = train_list
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def split_train_test(self, file_list):
        train_files = []
        test_files = []
        for i in range(len(file_list)):
            # if i % self.NumCv == self.cv:
            #     test_files.append(file_list[i])
            # else:
            train_files.append(file_list[i])

        return train_files, test_files

    def __len__(self):
        return len(self.file_list)
    
    def process_signal(self, signal, fs):
        assert fs == 200, f"fs is not 200 but {fs}"
        signal, _, _ = detect_motion_iterative(signal, fs)
        signal = signal_crop(signal)
        signal = norm_sig(signal)

        if fs != 10:
            signal = zoom(signal, 10/fs)
            fs = 10

        return signal

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # now randomly select a channel, sampling based on their weights
        selected_channel = np.random.choice(list(self.channels.keys()), p=list(self.channels.values()))
        if self.mode == "train":
            filepath = os.path.join(self.processed_signal, filename)
            breathing = np.load(filepath)['data'].squeeze()
            fs = np.load(filepath)['fs']
            # print(f'breathing shape: {breathing.shape}, fs: {fs}')
            assert fs == 10, "Sampling rate is not 10Hz"
        if self.mode == "test":
            filepath = os.path.join(self.ds_dir, selected_channel, filename)
            breathing = np.load(filepath)['data'].squeeze()
            fs = np.load(filepath)['fs']
            assert fs == 200, "Sampling rate is not 200Hz"
            breathing = self.process_signal(breathing, fs)
        
        # if self.mode == "train":
        #     breathing_length = breathing.shape[0] - self.max_length
        #     #randomly sample start index
        #     try:
        #         start_idx = np.random.randint(0, breathing_length)
        #     except:
        #         print("breathing_length is negative")
        #         print(f"breathing_length: {breathing_length}")
        #         print("filename: ", filename)
        #         print(f"start_idx: {start_idx}")
        #         sys.exit()
        #     breathing = breathing[start_idx:start_idx+self.max_length]
        # elif self.mode == "val":
        #     breathing = breathing[:self.max_length]
        # elif self.mode == "test":
        #     breathing = breathing
        # else:
        #     raise ValueError(f"Invalid mode: {self.mode}")

        # breathing = breathing[:self.max_length] #4 hours
        # breathing = self.process_signal(breathing, fs)
        breathing = torch.tensor(breathing, dtype=torch.float32)
        #randomly augment by multiplying by -1
        positive_count = (breathing > 0).sum().item()
        negative_count = (breathing < 0).sum().item()
        if positive_count > negative_count:
            breathing = breathing * -1 
        # if torch.rand(1).item() < 0.5:
        #     breathing = breathing * -1

        item = {
            "x": None,
            "y": 0,
            "filename": filename,
            "selected_channel": selected_channel
        }

        # if there is any nan or inf in the signal, return None
        if torch.isnan(breathing).any() or torch.isinf(breathing).any():
            # return None, 0
            return item

        #clip breathing -6,6

        #preserve the frequency 
        #iteralitely compute mean and std wihtin sliding window
        #destory the amplitude information in breathing belt and rf
        #so the only thing that is retained in frequency 
        #model only learns frequency information

        #unsquzze dim0
        breathing = breathing.unsqueeze(0)
        item["x"] = breathing

        return item

def main():

    dataset = BwhDataset()
    # print(f"Dataset size is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, num_workers = 10, shuffle=True)

    for i, (features, labels) in enumerate(dataloader):
        breathing = features[0]

        print(f"Batch {i+1}:")
        print(f"Features shape: {features.shape}")

if __name__ == '__main__':
    main()
