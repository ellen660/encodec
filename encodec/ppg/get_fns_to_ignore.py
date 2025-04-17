import os
import numpy as np
import json
from tqdm import tqdm

# Function to compute sliding window standard deviation
def sliding_std(data, window_size):
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

max_length = 100 * 60 * 60 * 4
fs = 100

root = "/data/netmit/sleep_lab/sandbox/ppg/bwh"

fns_to_ignore = []

fns = sorted(os.listdir(root))

for fn in tqdm(fns):
    filepath = os.path.join(root, fn)
    ppg = np.load(filepath)['data']
    fs = np.load(filepath)['fs']
    assert fs == 100
    if ppg.shape[0] <= max_length:
        fns_to_ignore.append(fn)
        print(f"ignoring {fn} because shape is {ppg.shape}")
        continue

    std_values = sliding_std(ppg, max_length)

    if np.any(std_values == 0):
        fns_to_ignore.append(fn)
        print(f"ignoring {fn} because of zero std")
        continue

        # loop through every segment of max_length and check if there are any nan or inf values
        # for i in range(0, breathing.shape[0] - max_length):
        #     breathing_segment = breathing[i:i+max_length]
            
            # breathing_segment, _, _ = detect_motion_iterative(breathing_segment, fs)
            # breathing_segment = signal_crop(breathing_segment)
            # breathing_segment = (breathing_segment - np.mean(breathing_segment)) / np.std(breathing_segment)

            # if np.std(breathing_segment) == 0:
            #     fns_to_ignore.append(fn)
            #     print(f"ignoring {fn} because of zero std")
            #     break

            # if np.any(np.isnan(breathing_segment)) or np.any(np.isinf(breathing_segment)):
            #     fns_to_ignore.append(fn)
            #     print(f"ignoring {fn} because of nan or inf")
            #     break

# save filenames to ignore to a .py file
with open("/data/scratch/ellen660/encodec/encodec/ppg/fns_to_ignore_bwh.py", "w") as f:
    f.write(f"fns_to_ignore = {json.dumps(fns_to_ignore)}")
