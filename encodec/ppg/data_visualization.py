import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import sys

from encodec.ppg.ppg_dataset import PpgDataset
from encodec.ppg.ppg_dataset import ROOT
from encodec.ppg.fns_to_ignore_bwh import fns_to_ignore as bwh_fns_to_ignore
from encodec.ppg.fns_to_ignore_mesa import fns_to_ignore as mesa_fns_to_ignore

class VisualizeDataset:
    """
    Visualizes a Preprocessed dataset of PPG signals
    Input ds_name: Name of the dataset to visualize (e.g. "bwh", "mesa")
    Functions:
      -visualize_individual_patients
         Plots 9 individual patients
         Inputs:
            signal: Whether to plot raw vs processed PPG signals
            histogram: Whether to plot histograms of the PPG signals
            freq: Whether to visualize the frequency spectrum of the PPG signals
      -visualize_dataset_distribution
            Plots the distribution of the dataset by plotting histograms of the PPG signals
      -print_info
         Prints class information
    """
    root = ROOT

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.fs = self.root[dataset]["fs"]
        self.test_max_length = self.fs * 60 * 60 * 10  # 10 hours
        self.test_dataset = self._init_dataset("test", self.test_max_length)[
            dataset
        ]  # note that max length doesn't actually matter for test datset
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=10
        )
        self.save_dir = (
            f"/data/scratch/ellen660/encodec/encodec/ppg/visualization/{dataset}"
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.patients = 9
        self.train_max_length = self.fs * 60 * 60 * 1  # 1 hour
        self.train_dataset = self._init_dataset("train", self.train_max_length)[dataset]
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=32, shuffle=False, num_workers=10
        )

    def print_info(self):
        """
        Prints information about the dataset
        """
        print(f"################################# Visualization #################################")
        print(f"Dataset: {self.dataset}")
        print(f"Sampling frequency: {self.fs} Hz")
        print(f"Bit depth: Unknown")
        print(f"Save directory: {self.save_dir}")
        print(f"Number of patients in entire dataset: {len(self.test_dataset)}")
        print(f"##################################################################")

    def _init_dataset(self, mode: str, max_length: int) -> dict[str, Dataset]:
        """
        Initialize the bwh PPG dataset, default cv = 0, max_length = 1 hour
        Input: train, val or test
        Output: Dictionary of datasets
        Key: dataset name, Value: BwhPpgDataset object
        """
        cv = 0
        max_length = max_length
        datasets = {"bwh":  PpgDataset(dataset="bwh", mode=mode, cv=cv, max_length=max_length),
                    "mesa": PpgDataset(dataset="mesa", mode=mode, cv=cv, max_length=max_length)
                    }
        return datasets

    def _plot_signal(self, x: np.ndarray, raw: np.ndarray, fs: int, filename: str):
        """
        Plots the raw and processed PPG signals.
        Input x: Processed PPG signal
        Input raw: Raw PPG signal
        Input filename: Name of the file to save the plot
        Input save_dir: Directory to save the plot
        Output: Saves a plot of raw vs processed PPG signals
        """
        total_samples = len(raw)
        time_axis = np.arange(total_samples) / total_samples

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_axis, raw, label="Raw PPG", alpha=1.0)
        ax.plot(time_axis[: len(x)], x, label="Processed PPG", alpha=0.7)
        ax.set_title(f"Patient PPG")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Amplitude")
        ax.set_ylim(-10, 10)
        ax.legend()

        max_time = time_axis[-1]
        tick_locs = np.arange(0, max_time + 0.5, 0.5)
        ax.set_xticks(tick_locs)

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"signal_{filename}.png")
        fig.savefig(save_path)
        plt.close(fig)

    def _plot_individual_distributions(self):
        """
        Plots individual patient data by plotting histograms and raw vs processed PPG signals.
        Output: Saves plots of histogram distribution and raw vs processed PPG signals for 9 patients
        """
        fig, axs = plt.subplots(
            int(self.patients**0.5), int(self.patients**0.5), figsize=(10, 10)
        )
        axs = axs.flatten()

        for j, batch in enumerate(self.test_dataloader):
            if j >= self.patients:
                break
            x = batch["x"][0].numpy().squeeze()
            bin_edges = np.linspace(-10, 10, 200)
            histogram = np.histogram(x, bins=bin_edges)[0]
            histogram = histogram / histogram.sum()

            axs[j].bar(
                bin_edges[:-1],
                histogram,
                width=np.diff(bin_edges),
                edgecolor="black",
                align="edge",
            )
            axs[j].set_ylabel("Frequency")
            axs[j].set_title(f"Histogram {j+1}")
        for ax in axs:
            ax.set_ylim(0, 0.12)  # Highest frequency is around 0.12
        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"preprocessed_patient_histogram.png")
        fig.savefig(save_path)
        plt.close(fig)

    def _visualize_fft(self, x: np.ndarray, fs: int, filename: str):
        """
        Visualize the fft frequency of the dataset
        Input x: Real valued 1D numpy array of the signal
        Input fs: Sampling frequency in Hz
        Output: Plot of the frequency spectrum
        """
        print(f'mean x {np.mean(x)}')
        X = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(len(x), d=1 / self.fs)
        power = np.abs(X) ** 2
        top_k = 5
        indices = np.argsort(power)[-top_k:]
        top_indices = indices[np.argsort(power[indices])[::-1]]
        top_freqs = freqs[top_indices]
        top_powers = power[top_indices]

        threshold = 1e-2  # or try 1e-3 depending on your noise level
        nonzero_indices = np.where(power > threshold)[0]
        highest_idx = nonzero_indices[-1]
        highest_freq = freqs[highest_idx]
        highest_power = power[highest_idx]

        print(f"Highest nonzero frequency: {highest_freq:.3f} Hz, Power: {highest_power:.2f}")

        # Create annotation text
        annotation = "\n".join(
            [f"Top {i}: {f:.2f} Hz ({p:.0f})" for i, (f, p) in enumerate(zip(top_freqs, top_powers), 1)]
        )
        annotation += f"\nMax nonzero: {highest_freq:.2f} Hz ({highest_power:.0f})"

        # Add the text inside the plot top right
        plt.text(
            0.98,                # x position in axis fraction (0 = left, 1 = right)
            0.98,                # y position in axis fraction (0 = bottom, 1 = top)
            annotation,         
            ha="right", va="top", fontsize=10,
            transform=plt.gca().transAxes,  # use axis-relative coordinates
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray")
        )

        plt.plot(freqs, power) 
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        # plt.xlim(0, 20)

        save_path = os.path.join(self.save_dir, f"freq_{filename}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def visualize_individual_patients(self, signal: bool, histogram: bool, freq: bool):
        """
        Visualizes individual patient data by plotting histograms and raw vs processed PPG signals.
        Input signal: Whether to plot raw vs processed PPG signals
        Input histogram: Whether to plot histograms of the PPG signals
        Input freq: Whether to visualize the frequency spectrum of the PPG signals
        Output: Saves plots of histogram distribution and raw vs processed PPG signals for 9 patients
        """
        print(f"Visualizing {self.patients} patients from {self.dataset} dataset")
        if histogram:
            self._plot_individual_distributions()

        if signal:
            for j, batch in enumerate(self.test_dataloader):
                if j >= self.patients:
                    break
                x = batch["x"][0].numpy().squeeze()
                filename = batch["filename"][0]
                filepath = os.path.join(self.root[self.dataset]["root"], filename)
                raw = np.load(filepath)["data"].squeeze()[: self.test_max_length]
                assert (
                    raw.shape[0] == x.shape[0]
                ), f"data length mismatch for {filename}: {raw.shape[0]} vs {x.shape[0]}"
                self._plot_signal(x, raw, fs=self.fs, filename=filename)

        if freq:
            for j, batch in enumerate(self.test_dataloader):
                if j >= self.patients:
                    break
                x = batch["x"][0].numpy().squeeze()
                filename = batch["filename"][0]
                self._visualize_fft(x, self.fs, filename)

    def visualize_dataset_distribution(self):
        """
        Plot distribution for overall dataset
        """
        fig, axs = plt.subplots(figsize=(10, 10))

        bin_edges = np.linspace(-10, 10, 100)
        histogram = np.zeros(len(bin_edges) - 1)

        for j, batch in enumerate(
            tqdm(self.train_dataloader, desc="Plotting dataset distribution")
        ):
            x = batch["x"].numpy()
            histogram += np.histogram(x, bins=bin_edges)[0]

        histogram = histogram / histogram.sum()

        axs.bar(
            bin_edges[:-1],
            histogram,
            width=np.diff(bin_edges),
            edgecolor="black",
            align="edge",
        )
        axs.set_ylabel("Frequency")
        axs.set_title(f"Dataset Distribution of {self.dataset}")

        axs.set_ylim(0, 0.12)
        save_path = os.path.join(self.save_dir, f"overall_preprocessed_dataset_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    # bwh_visualizer = VisualizeDataset("bwh")
    # bwh_visualizer.print_info()
    # bwh_visualizer.visualize_individual_patients(signal=True, histogram=True, freq=True)
    # bwh_visualizer.visualize_dataset_distribution()

    mesa_visualizer = VisualizeDataset("mesa") 
    mesa_visualizer.print_info()
    mesa_visualizer.visualize_individual_patients(signal=True, histogram=True, freq=True)
    # mesa_visualizer.visualize_dataset_distribution()

