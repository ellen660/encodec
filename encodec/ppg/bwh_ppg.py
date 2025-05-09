import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .fns_to_ignore_bwh import fns_to_ignore

"""
N = 4057
"""
class BwhPpgDataset(Dataset):
    root = {"bwh_new": "/data/netmit/sleep_lab/sandbox/ppg/bwh"}
    NumCv = 4
    modes = ['train', 'val', 'test']
            
    def __init__(self, dataset="bwh_new", mode = "train", cv = 0, max_length = 100 * 60 * 60 * 4):
        assert mode in ['train', 'val', 'test'], 'Only support train val or test mode'

        self.dataset = dataset
        self.mode = mode
        self.cv = cv
        self.ds_dir = self.root[dataset]
        self.max_length = max_length

        # dataset preparation (only select the intersection between all channels)
        file_list = [f for f in os.listdir(self.ds_dir) if f.endswith('.npz') and f not in fns_to_ignore]
        file_list = sorted(file_list)
        print(f'PPG BWH size : {len(file_list)}')
        # breakpoint()

        train_list, val_list = self.split_train_test(file_list)
            
        if mode == "train":
            self.file_list = train_list
        elif mode == "val":
            self.file_list = val_list
        elif mode == "test": #All the files
            self.file_list = file_list

    def split_train_test(self, file_list):
        train_files = []
        test_files = []
        for i in range(len(file_list)):
            if i % self.NumCv == self.cv:
                test_files.append(file_list[i])
            else:
                train_files.append(file_list[i])

        return train_files, test_files

    def __len__(self):
        return len(self.file_list)
    
    def process_signal(self, signal, fs):
        #divide by mean, std
        def signal_crop(signal, clip_limit=6):
            signal = np.clip(signal, -clip_limit, clip_limit)
            return signal

        def norm_sig(input_sig):
            # print(f'mean {np.mean(input_sig)} std {np.std(input_sig)}')
            return (input_sig - np.mean(input_sig)) / np.std(input_sig)
        
        signal = signal_crop(signal)
        signal = norm_sig(signal)

        return signal

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # now randomly select a channel, sampling based on their weights
        filepath = os.path.join(self.ds_dir, filename)
        ppg = np.load(filepath)['data'].squeeze()
        fs = np.load(filepath)['fs']
        assert fs == 100, f"fs is not 100 but {fs}"

        if self.mode == "train":
            ppg_length = ppg.shape[0] - self.max_length
            #randomly sample start index
            try:
                start_idx = np.random.randint(0, ppg_length + 1)
            except:
                print("breathing_length is negative")
                print(f"breathing_length: {ppg_length}")
                print("filename: ", filename)
                print(f"dataset: {self.dataset}")
                sys.exit()
            if filename == "fa63e27e501a075e678b96e5da311161ccb2a833482bc5b8c327963fe41f486f.npz":
                print(f'start_idx {start_idx}')
            ppg = ppg[start_idx:start_idx+self.max_length]
        elif self.mode == "val":
            ppg = ppg[:self.max_length]
        elif self.mode == "test":
            ppg = ppg[:3600000]

        ppg = self.process_signal(ppg, fs)
                
        ppg = torch.tensor(ppg, dtype=torch.float32)
        #randomly augment by multiplying by -1
        #flip to have everything be on same side
        positive_count = (ppg > 0).sum().item()
        negative_count = (ppg < 0).sum().item()
        if positive_count > negative_count:
            ppg = ppg * -1 

        item = {
            "x": None,
            "y": 0,
            "filename": filename,
        }

        # if there is any nan or inf in the signal, return None
        if torch.isnan(ppg).any() or torch.isinf(ppg).any() or ppg is None:
            # return None, 0
            print(f'bad file {filename}')
            sys.exit()
            return item

        #unsquzze dim0
        ppg = ppg.unsqueeze(0)
        item["x"] = ppg

        return item

def main():

    dataset = BwhPpgDataset(max_length=10 * 60 * 60 * 1)
    # print(f"Dataset size is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, num_workers = 10, shuffle=True)

    for i, (features, labels) in enumerate(dataloader):
        breathing = features[0]

        print(f"Batch {i+1}:")
        print(f"Features shape: {features.shape}")

if __name__ == '__main__':
    main()
