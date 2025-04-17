import pickle
from typing import cast
import numpy as np
from encodec.data import BwhDataset
from torch.utils.data import DataLoader 
import os
from tqdm import tqdm
import sys
import torch
import matplotlib.pyplot as plt

snr_labels = "/data/netmit/sleep_lab/ali_2/bwh_v10_deepsnr_labels"
save_dir = "/data/scratch/ellen660/encodec/encodec/visualizations/deepsnr_bwh"

data = BwhDataset(dataset = "bwh_new", mode = "test", cv = 0, channels = {"thorax": 1.0}, max_length = 10*60*60*4)
dataset = DataLoader(data, batch_size=1, shuffle=False, num_workers=10)
print(f'size dataset: {len(dataset)}')

for i, item in enumerate(tqdm(dataset)):
    if i >= 10:
        break
    breathing = item["x"].squeeze(0).squeeze(0).numpy()
    breathing = breathing[::2] #take every two
    filename = item["filename"][0]
    #load deepsnr
    deepsnr = np.load(os.path.join(snr_labels, filename.replace(".npz", ".npy")))

    #plot the breathing of shape (T,) of 5 Hz and deepsnr of shape (T/600) 
    fig, ax = plt.subplots(2, 1, figsize=(10, 5),sharex=True)
    ax[0].plot(np.arange(0,1,1/len(breathing)),breathing, label="Breathing")
    ax[0].set_title("Breathing Signal")
    ax[0].legend()
    ax[1].plot(np.arange(0,1,1/len(deepsnr)),deepsnr, label="DeepSNR")
    ax[1].set_title("DeepSNR Signal")
    ax[1].legend()
    #save the figure
    save_path = os.path.join(save_dir, filename.replace(".npz", ".png"))
    # plt.savefig(save_path)
    plt.show()