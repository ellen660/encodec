import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Union, Optional, Tuple, Iterable, Literal
from tqdm import tqdm
from encodec.ppg.fns_to_ignore_bwh import fns_to_ignore as bwh_fns_to_ignore
from encodec.ppg.fns_to_ignore_mesa import fns_to_ignore as mesa_fns_to_ignore

ROOT = {
    "bwh": {  # N = 4057
        "root": "/data/netmit/sleep_lab/ML4H/bwh/ppg",
        "fs": 64,
        "fns_to_ignore": [],
    },
    "mesa": {  # N = 1940
        "root": "/data/netmit/sleep_lab/ML4H/mesa/ppg",
        "fs": 64,
        "fns_to_ignore": [],
    },
    # "mgh": {  # N = 1940
    #     "root": "/data/netmit/sleep_lab/ML4H/mgh/ppg",
    #     "fs": 64,
    #     "fns_to_ignore": [],
    # },
}


# Mesa dataset
# https://sleepdata.org/datasets/mesa/pages/polysomnography-introduction.md
# Finger pulse oximetry (reflecting PPG) 256 Hz, 8000 Nonin sensor (which I am not sure what that means)
# Has a lot of flat signals. Since we are not imputing, I am very harsh and just deleted all those that have at least 1 minute of flat signal.
# 993???????/2056

class PpgDataset(Dataset):
    """
    PPG Dataset supporting bwh and mesa
    Assumes clean data
    TODO: preprocessing algorithm, flipping, testing mesa on encodec
    TODO: only frequency matters, can destory amplitude?

    Args:
        dataset: bwh or mesa
        mode: train, validation or test. Train, validation will be split according to cv,
            test takes the whole dataset
        cv: cross fold validation, must be 0,1,2, or 3
        max_length: max length for training and validation, not relevant for testing.
            default max length for testing is 10 hours
            for training and validation, returns a sample of length max_length
            randomly chosen for training
            first section for validation

    Functions:
        __len__
        __getitem__:
            Returns a dictionary with
                x: the preprocessed ppg signal
                    preprocessing: first slice the signal to max_length
                        then clip to -6, 6
                        then normalize by subtracting the individual mean and dividing by individual standard deviation
                        ! note that the normalization occurs only on a subsection of the signal and is per individual
                y: label, currently default to 0
                filename

    """

    root = ROOT
    NumCv = 4

    def __init__(
        self,
        dataset: Literal["bwh", "mesa", "mgh"] = "bwh",
        mode: Literal["train", "val", "test"] = "train",
        cv: Literal[0,1,2,3] = 0,
        max_length: int = 60 * 60 * 4,
    ):
        self.dataset = dataset
        self.mode = mode
        self.cv = cv
        self.ds_dir = self.root[dataset]["root"]
        self.fs = self.root[dataset]["fs"]
        self.max_length = self.fs * max_length
        self.max_test_length = self.fs * 60 * 60 * 10  # 10 hours

        file_list = [
            f
            for f in os.listdir(self.ds_dir)
            if f.endswith(".npz") and f not in self.root[dataset]["fns_to_ignore"]
        ]
        file_list = sorted(file_list)

        train_list, val_list = self.split_train_test(file_list)

        if mode == "train":
            self.file_list = train_list
            self._assert_output()
        elif mode == "val":
            self.file_list = val_list
            self._assert_output()
        elif mode == "test":  # All the files
            self.file_list = file_list
            
    def _assert_output(self):
        item = self.__getitem__(0)
        assert item["x"].dtype == torch.float32, "Input must be float32"  # type: ignore
        assert item["x"].shape == (1, self.max_length,), f"Expected {(1, self.max_length)} but got {item['x'].shape}" #type: ignore

    def split_train_test(self, file_list: list[str]) -> Tuple[list[str], list[str]]:
        train_files = []
        test_files = []
        for i in range(len(file_list)):
            if i % self.NumCv == self.cv:
                test_files.append(file_list[i])
            else:
                train_files.append(file_list[i])

        return train_files, test_files

    def __len__(self) -> int:
        return len(self.file_list)

    def process_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocessing algorithm for an individual PPG signal
        Very uncertain what to do

        Args:
            signal: the PPG signal

        Returns: the cropped and normalized signal
        """

        # def signal_crop(signal, clip_limit=6):
        #     signal = np.clip(signal, -clip_limit, clip_limit)
        #     return signal

        def norm_sig(input_sig):
            return (input_sig - np.mean(input_sig)) / np.std(input_sig)

        # signal = signal_crop(signal)
        signal = norm_sig(signal)

        return signal

    def __getitem__(self, idx) -> dict:
        filename = self.file_list[idx]

        # now randomly select a channel, sampling based on their weights
        filepath = os.path.join(self.ds_dir, filename)
        ppg = np.load(filepath)["data"].squeeze()
        fs = np.load(filepath)["fs"]
        assert fs == self.fs, f"fs is not {self.fs} but {fs}"
        ppg = self.process_signal(ppg)

        if self.mode == "train":
            ppg_length = ppg.shape[0] - self.max_length
            try:
                start_idx = np.random.randint(0, ppg_length + 1)
            except:
                print("breathing_length is negative")
                print(f"breathing_length: {ppg_length}")
                print("filename: ", filename)
                print(f"dataset: {self.dataset}")
                sys.exit()
            if (
                filename
                == "fa63e27e501a075e678b96e5da311161ccb2a833482bc5b8c327963fe41f486f.npz"
            ):
                print(f"random start_idx {start_idx}")
            ppg = ppg[start_idx : start_idx + self.max_length]
        elif self.mode == "val":
            ppg = ppg[: self.max_length]
        elif self.mode == "test":
            ppg = ppg[: self.max_test_length]

        ppg = torch.tensor(ppg, dtype=torch.float32)
        # randomly augment by multiplying by -1
        # flip to have everything be on same side
        positive_count = (ppg > 0).sum().item()
        negative_count = (ppg < 0).sum().item()
        if positive_count > negative_count:
            ppg = ppg * -1
        ppg = ppg.unsqueeze(0)

        item = {
            "x": ppg,
            "y": 0,
            "filename": filename,
        }

        # if there is any nan or inf in the signal, return None for debugging. Shouldn't happen
        if torch.isnan(ppg).any() or torch.isinf(ppg).any() or ppg is None:
            print(f"bad file {filename}")
            sys.exit()
            return item

        return item

def main():
    dataset = PpgDataset("mesa", "test")
    print(f"Dataset size is {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=10, shuffle=True)

    for i, item in enumerate(tqdm(dataloader)):
        ppg = item["x"]

        # print(f"Batch {i+1}:")
        # print(f"Features shape: {ppg.shape}")


if __name__ == "__main__":
    main()
