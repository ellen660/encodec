import os
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from encodec.ppg.ppg_dataset import ROOT
from encodec.ppg.fns_to_ignore_bwh import fns_to_ignore as bwh_fns_to_ignore
from encodec.ppg.fns_to_ignore_mesa import fns_to_ignore as mesa_fns_to_ignore
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Union, Optional, Tuple, Iterable


def _sliding_std(data: np.ndarray, window_size: int):
    """
    Computes the sliding (moving) standard deviation over a 1D NumPy array.

    Args:
        data (np.ndarray): A 1D array of numerical data.
        window_size (int): The number of elements in each sliding window.
            Must be a positive integer less than or equal to the length of `data`.

    Returns:
        np.ndarray: A 1D array of standard deviations, one for each valid sliding window.
            The output has shape (len(data) - window_size + 1,).

    Raises:
        ValueError: If `window_size` is not a positive integer or exceeds the length of `data`.

    Example:
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> _sliding_std(x, window_size=3)
        array([0.81649658, 0.81649658, 0.81649658])
    """
    cumsum = np.cumsum(data)
    cumsum_sq = np.cumsum(data**2)

    cumsum = np.concatenate(([0], cumsum))  # Shift for correct indexing
    cumsum_sq = np.concatenate(([0], cumsum_sq))

    # Calculate mean and std for each sliding window
    window_sum = cumsum[window_size:] - cumsum[:-window_size]
    window_sq_sum = cumsum_sq[window_size:] - cumsum_sq[:-window_size]
    window_mean = window_sum / window_size
    window_var = (window_sq_sum / window_size) - (window_mean**2)

    return np.sqrt(np.maximum(window_var, 0))  # Avoid numerical errors


def _check_file(
    fn: str, fs: int, root: str, min_length: int, sliding_window: int
) -> Union[str, None]:
    """
    Helper function, checks if a file is "bad".
    A bad file is defined as:
        1. the file doesn't load
        2. the file is too short (less than 4 hours)
        3. the file has bad data (1 hour of zero values)

    Args:
        fn: filename
        fs: sampling rate
        root: datapath
        min_length: minimum length of the file. default 4 hours
        sliding_window: window length to look for flatlines. default 1 minute

    Returns:
        fn filename if bad file
        None if good file
    """
    filepath = os.path.join(root, fn)
    try:
        data = np.load(filepath)
        ppg = data["data"]
        if data["fs"] != fs:
            raise ValueError("Sampling rate mismatch")
    except:
        return fn

    if ppg.shape[0] <= min_length:
        return fn

    threshold = 1e-8  # using threshold to capture all cases
    std_values = _sliding_std(ppg, sliding_window)
    if np.any(std_values < threshold):
        return fn

    return None  # File is OK


def get_fns_to_ignore(dataset: str, num_workers: int = 10):
    """
    Given dataset (bwh or mesa), finds and records the names of the bad files.
    Uses CPU multicore processing

    Args:
        dataset: bwh or mesa
        num_workers: number of CPU cores
    """
    print(f"#################### Finding bad files for {dataset} ####################")
    fs = ROOT[dataset]["fs"]
    min_length = fs * 60 * 60 * 4  # 4 hours
    sliding_window = fs * 60 * 30 * 1  # 1 hour
    root = ROOT[dataset]["root"]
    fns_to_ignore = []
    fns = sorted(os.listdir(root))

    checker = partial(
        _check_file,
        fs=fs,
        root=root,
        min_length=min_length,
        sliding_window=sliding_window,
    )

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(checker, fns), total=len(fns)))

    fns_to_ignore = [fn for fn in results if fn is not None]
    breakpoint()

    # save_path = f"/data/scratch/ellen660/encodec/encodec/ppg/fns_to_ignore_{dataset}.py"
    # with open(save_path, "w") as f:
    #     f.write(f"fns_to_ignore = {json.dumps(fns_to_ignore)}")

    print(f"Saved {len(fns_to_ignore)} bad files to {save_path}")
    print(f"########################################")


def plot_bad_file(dataset: str, filename: str):
    """
    Plots a bad file to test whether it is really bad

    Args:
        dataset: bwh or mesa
        filename: bad file
    """
    root = ROOT[dataset]["root"]
    filepath = os.path.join(root, filename)
    data = np.load(filepath)["data"]
    print(f"data shape {data.shape}")

    plt.plot(data)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.title(f"{dataset}_{filename[:6]}")

    save_path = f"/data/scratch/ellen660/encodec/encodec/ppg/visualization/{dataset}/bad_file_{filename[:-4]}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    get_fns_to_ignore("mesa")
    # bad_file = "mesa-sleep-0010.npz"
    # _check_file(bad_file, fs=100, root = ROOT["mesa"]["root"], min_length=100*60*60*4, sliding_window=100*60*1)

    # 1 hour: 451 
    # 1 minute: 993 
    # 30 minutes: 627

    # Looks like we are gonna have to clip again