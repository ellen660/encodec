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
from .fns_to_ignore_less_than_4 import fns_to_ignore
from tqdm import tqdm
import ast

"""
N = 2067?
"""
class BwhDataset(Dataset):
    root = "/data/netmit/sleep_lab/ali_2"
    processed_signal = f'{root}/bwh_encodec'
    if not os.path.exists(processed_signal):
        os.makedirs(processed_signal)
    NumCv = 4
        
    def __init__(self, dataset="bwh_new", mode = "train", cv = 0, channels = {"thorax": 1.0}, max_length=10 * 60 * 60 * 4):
        assert channels == {"thorax": 1.0}, "Only support thorax channel"
        channels = {"thorax_clipped": 1.0}
        self.dataset = dataset
        self.mode = mode
        assert self.mode in ['train', 'test', 'val'], 'Only support train val or test mode'
        self.cv = cv
        self.channels = channels # dictionary of channel names and their weights
        self.ds_dir = self.root
        self.max_length = max_length
        self.max_length_200 = max_length * 20

        # dataset preparation (only select the intersection between all channels)
        file_list = set()
        for channel in self.channels.keys():
            file_list_before = sorted([f for f in os.listdir(os.path.join(self.ds_dir, channel)) if f.endswith('.npz')])
            file_list_after = [f for f in file_list_before if f not in fns_to_ignore]
            file_list.update(file_list_after)

        file_list = sorted(file_list)

        # file_list = self.filter(file_list)

        train_list, val_list = self.split_train_test(file_list)

        self.start_end = pd.read_csv(f'/data/scratch/ellen660/encodec/encodec/data/bwh_start_end_patches.csv')

        # breakpoint()
        # for file in tqdm(file_list):
        #     # now randomly select a channel, sampling based on their weights
        #     selected_channel = np.random.choice(list(self.channels.keys()), p=list(self.channels.values()))
        #     filepath = os.path.join(self.ds_dir, selected_channel, file)
        #     breathing = np.load(filepath)['data'].squeeze()
        #     # start, end, patches = self.start_end[self.start_end['file'] == file].values[0][1:]
        #     # try:
        #     #     breathing = breathing[start:end+1]
        #     # except:
        #     #     breathing = breathing[start:end]
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
        #         #ignore this iteration
        #         continue
        #     breathing = breathing[start_idx:start_idx+200*60*60*4]
        #     breathing = self.process_signal(breathing, fs)

        #     #save to processed_signal
        #     np.savez(f'{self.processed_signal}/{file}', data=breathing, fs=10, start=start_idx)
        # breakpoint()
            
        if mode == "train":
            self.file_list = train_list
        elif mode == "val":
            self.file_list = val_list
        elif mode == "test": #All the files
            self.file_list = file_list
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def filter(self, file_list):
        """
        Filters out the weird files
        """
        filtered = []
        print(f'initial length: {len(file_list)}')
        for filename in file_list:
            # print(f'filtering filename: {filename}')
            try:
                sleep_data = np.load("/data/netmit/sleep_lab/ali_2/stage_pred/" + filename)['data']
                num_zeroes = np.sum(sleep_data == 0)
                #how many hours 
                if (len(sleep_data) - num_zeroes)/(2*60) > 4:
                    filtered.append(filename)
                else:
                    # print(f'file {filename} has less than 4 hours of sleep')
                    pass
            except:
                # print(f'error with {filename}')
                pass
        print(f'remaining {len(filtered)} files')
        return filtered

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
            breathing_length = breathing.shape[0] - self.max_length
            #randomly sample start index
            try:
                start_idx = np.random.randint(0, breathing_length+1)
            except:
                print("breathing_length is negative")
                print(f"breathing_length: {breathing_length}")
                print("filename: ", filename)
                print(f"dataset: {self.dataset}")
                sys.exit()
            breathing = breathing[start_idx:start_idx+self.max_length]
            # assert fs == 10, "Sampling rate is not 10Hz"
        elif self.mode == "val":
            filepath = os.path.join(self.ds_dir, selected_channel, filename)
            breathing = np.load(filepath)['data'].squeeze()
            fs = np.load(filepath)['fs']
            assert fs == 200, "Sampling rate is not 200Hz"
            breathing = breathing[:self.max_length_200]
            breathing = self.process_signal(breathing, fs)
        elif self.mode == "test":
            filepath = os.path.join(self.ds_dir, selected_channel, filename)
            print(f'filepath: {filepath}')
            breathing = np.load(filepath)['data'].squeeze()
            fs = np.load(filepath)['fs']
            assert fs == 200, "Sampling rate is not 200Hz"
            breathing = self.process_signal(breathing, fs)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        breathing = torch.tensor(breathing, dtype=torch.float32)
        #randomly augment by multiplying by -1
        #flip to have everything be on same side
        positive_count = (breathing > 0).sum().item()
        negative_count = (breathing < 0).sum().item()
        if positive_count > negative_count:
            breathing = breathing * -1 

        item = {
            "x": None,
            "y": 0,
            "filename": filename,
            "selected_channel": selected_channel
        }

        # if there is any nan or inf in the signal, return None
        if torch.isnan(breathing).any() or torch.isinf(breathing).any() or breathing is None:
            # return None, 0
            print(f'bad file {filename}')
            sys.exit()
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

    dataset = BwhDataset(max_length=10 * 60 * 60 * 1)
    # print(f"Dataset size is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, num_workers = 10, shuffle=True)

    for i, (features, labels) in enumerate(dataloader):
        breathing = features[0]

        print(f"Batch {i+1}:")
        print(f"Features shape: {features.shape}")

if __name__ == '__main__':
    main()
